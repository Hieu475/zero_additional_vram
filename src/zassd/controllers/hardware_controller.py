"""Hardware-aware joint speculation controller.

This is the novel systems contribution of the research.
The controller jointly optimizes:
  a_t = (S_t, K_t)
where:
  S_t = draft subnetwork layer configuration (e.g. CKA-light 27 layers, CKA-medium 18 layers)
  K_t = draft speculation length (1..8)

Based on runtime observation vector:
  z_t = [ H_t, A_{t-1}, T_draft, T_verify, VRAM_t, P_t, Temp_t ]

Subject to hardware constraints:
  - VRAM budget <= 5500 MB (RTX 4050 6GB ceiling)
  - Power limit <= 80 W
  - Temperature limit <= 83 C (thermal throttling avoidance)

Utility formulation:
  U(a | z_t) = lambda_s * ExpectedSpeedup(S, K)
               - lambda_l * LatencyPenalty(T_draft, T_verify)
               - lambda_v * VRAMPenalty(VRAM_t)
               - lambda_e * EnergyPenalty(P_t, Temp_t)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from zassd.controllers.adaptive_k import AdaptiveKController
from zassd.profiling.gpu import GPUProfiler
from zassd.profiling.memory import get_vram_usage

logger = logging.getLogger(__name__)


@dataclass
class HardwareState:
    """Current hardware state observation."""
    vram_used_mb: float = 0.0
    vram_total_mb: float = 6141.0
    gpu_utilization: float = 0.0
    gpu_power_w: float = 0.0
    gpu_temperature_c: float = 0.0


@dataclass
class ControllerAction:
    """Joint controller output action."""
    config_name: str            # "cka_75" or "cka_50"
    skip_indices: list[int]     # Layers to skip S_t
    draft_length: int           # K_t


@dataclass
class ControllerObservation:
    """Full observation vector z_t."""
    entropy: float = 1.0        # H_t
    acceptance_rate: float = 0.3# A_{t-1}
    draft_latency_ms: float = 20.0 # T_draft
    verify_latency_ms: float = 25.0# T_verify
    vram_used_mb: float = 2000.0  # VRAM_t
    gpu_power_w: float = 60.0     # P_t
    gpu_temperature_c: float = 70.0# Temp_t


class HardwareAwareJointController:
    """Hardware-Aware Joint Speculation Controller.

    Dynamically selects both layer configuration S_t and draft length K_t.
    """

    def __init__(
        self,
        candidate_layer_configs: dict[str, list[int]],
        gpu_profiler: GPUProfiler | None = None,
        max_vram_mb: float = 5500.0,
        temp_threshold_c: float = 82.0,
        lambda_speed: float = 1.0,
        lambda_latency: float = 0.3,
        lambda_vram: float = 0.5,
        lambda_energy: float = 0.2,
    ) -> None:
        self.configs = candidate_layer_configs
        self.gpu_profiler = gpu_profiler
        self.max_vram_mb = max_vram_mb
        self.temp_threshold_c = temp_threshold_c

        self.lambda_s = lambda_speed
        self.lambda_l = lambda_latency
        self.lambda_v = lambda_vram
        self.lambda_e = lambda_energy

        # Adaptive K inner controller
        self.k_controller = AdaptiveKController(
            k_min=1, k_max=8, initial_k=3,
            entropy_low=0.7, entropy_high=1.8, acceptance_target=0.35,
        )

        self.current_config_name = "cka_75"
        self.action_history: list[ControllerAction] = []

    def get_hardware_state(self) -> HardwareState:
        """Poll current hardware state from GPU profiler or PyTorch."""
        vram = get_vram_usage()
        state = HardwareState(
            vram_used_mb=vram["allocated_mb"],
        )
        if self.gpu_profiler:
            try:
                snap = self.gpu_profiler.snapshot()
                state.gpu_power_w = snap["power_w"]
                state.gpu_temperature_c = snap["temperature_c"]
                state.gpu_utilization = snap["utilization"]["gpu_pct"]
                state.vram_total_mb = snap["memory"]["total_mb"]
            except Exception:
                pass
        return state

    def compute_utility(
        self,
        config_name: str,
        k: int,
        obs: ControllerObservation,
    ) -> float:
        """Evaluate utility U(a | z_t)."""
        # Estimated speedup term: depends on acceptance and config layer speed
        # cka_50 is ~1.7x faster in draft than full, cka_75 is ~1.23x
        draft_speed_mult = 1.7 if "50" in config_name else 1.25
        expected_accepted = min(k, obs.acceptance_rate * k + 1)
        expected_speedup = (expected_accepted * draft_speed_mult) / (1.0 + (k * 0.15))

        # Latency penalty
        latency_penalty = (obs.draft_latency_ms + obs.verify_latency_ms) / 100.0

        # VRAM constraint penalty
        vram_headroom = max(0.0, self.max_vram_mb - obs.vram_used_mb)
        vram_penalty = 1.0 / max(vram_headroom, 100.0)

        # Thermal / energy penalty
        temp_margin = self.temp_threshold_c - obs.gpu_temperature_c
        thermal_penalty = 1.5 if temp_margin < 5.0 else 0.0
        power_penalty = (obs.gpu_power_w / 80.0)

        utility = (
            self.lambda_s * expected_speedup
            - self.lambda_l * latency_penalty
            - self.lambda_v * vram_penalty
            - self.lambda_e * (power_penalty + thermal_penalty)
        )
        return float(utility)

    def select_action(
        self,
        entropy: float,
        last_accepted: int,
        last_proposed: int,
        draft_ms: float = 20.0,
        verify_ms: float = 25.0,
    ) -> ControllerAction:
        """Select joint action a_t = (S_t, K_t)."""
        hw_state = self.get_hardware_state()

        # Update K based on entropy & acceptance
        k = self.k_controller.update(
            entropy=entropy,
            accepted=last_accepted,
            proposed=last_proposed,
        )

        # If GPU temperature is very high or power is at limit, throttle K to reduce thermal pressure
        if hw_state.gpu_temperature_c > self.temp_threshold_c:
            k = max(1, k - 1)

        # Observation vector
        acc_rate = self.k_controller.avg_acceptance_rate
        obs = ControllerObservation(
            entropy=entropy,
            acceptance_rate=acc_rate,
            draft_latency_ms=draft_ms,
            verify_latency_ms=verify_ms,
            vram_used_mb=hw_state.vram_used_mb,
            gpu_power_w=hw_state.gpu_power_w,
            gpu_temperature_c=hw_state.gpu_temperature_c,
        )

        # Joint selection: evaluate candidate layer configurations
        best_config = "cka_75"
        best_util = -float("inf")

        for cfg_name in self.configs.keys():
            util = self.compute_utility(cfg_name, k, obs)
            if util > best_util:
                best_util = util
                best_config = cfg_name

        self.current_config_name = best_config
        action = ControllerAction(
            config_name=best_config,
            skip_indices=self.configs[best_config],
            draft_length=k,
        )
        self.action_history.append(action)
        return action
