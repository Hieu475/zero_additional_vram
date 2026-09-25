"""Hardware-aware joint speculation controller (Phase 8 & Phase 9).

This is the novel systems contribution of the research.
The controller jointly optimizes:
  a_t = (S_t, K_t) = argmax_{S in S, K in K} U(S, K | z_t)
where:
  S_t = draft subnetwork layer configuration (e.g. CKA-83, CKA-75, CKA-60, CKA-50)
  K_t = draft speculation length (1..8)

Based on runtime observation vector:
  z_t = [ H_t, A_{t-1}, T_draft, T_verify, VRAM_t, P_t, Temp_t ]

Subject to real hardware dynamics and constraints:
  - VRAM budget <= 5500 MB (RTX 4050 6GB ceiling)
  - Power budget <= 80 W (or limited operating budget P_budget)
  - Temperature limit <= 82 C (thermal throttling threshold)

Utility formulation:
  U(S, K | z_t) = lambda_s * Speedup(S, K)
                - lambda_l * LatencyPenalty(T_cycle)
                - lambda_v * VRAMPenalty(VRAM_headroom, K)
                - lambda_e * (PowerPenalty(P_t, Energy) + ThermalPenalty(Temp_t, S, K))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from zassd.controllers.adaptive_k import AdaptiveKController
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.profiling.action_profiler import ActionCostDatabase
from zassd.profiling.gpu import GPUProfiler
from zassd.profiling.memory import get_vram_usage

logger = logging.getLogger(__name__)


@dataclass
class HardwareState:
    """Current hardware state observation."""
    vram_used_mb: float = 2000.0
    vram_total_mb: float = 6141.0
    gpu_utilization: float = 0.0
    gpu_power_w: float = 50.0
    gpu_temperature_c: float = 55.0
    power_budget_w: float = 80.0


@dataclass
class ControllerAction:
    """Joint controller output action."""
    config_name: str            # e.g. "cka_83", "cka_75"
    skip_indices: list[int]     # Layers to skip S_t
    draft_length: int           # K_t
    predicted_utility: float = 0.0
    expected_cycle_ms: float = 0.0
    expected_energy_j_tok: float = 0.0


@dataclass
class ControllerObservation:
    """Full observation vector z_t."""
    entropy: float = 1.0        # H_t
    acceptance_rate: float = 0.5# A_{t-1}
    draft_latency_ms: float = 20.0 # T_draft
    verify_latency_ms: float = 25.0# T_verify
    vram_used_mb: float = 2000.0  # VRAM_t
    gpu_power_w: float = 50.0     # P_t
    gpu_temperature_c: float = 55.0# Temp_t
    power_budget_w: float = 80.0


class HardwareAwareJointController:
    """Hardware-Aware Joint Speculation Controller.

    Dynamically and jointly selects both layer configuration S_t and draft length K_t
    conditioned on runtime hardware state and linguistic sequence entropy.
    """

    def __init__(
        self,
        candidate_layer_configs: dict[str, list[int]],
        cost_model: Optional[MeasuredActionCostModel] = None,
        action_cost_db: Optional[ActionCostDatabase] = None,
        gpu_profiler: GPUProfiler | None = None,
        model_name: Optional[str] = None,
        max_vram_mb: float = 5500.0,
        temp_threshold_c: float = 82.0,
        power_budget_w: float = 80.0,
        lambda_speed: float = 1.0,
        lambda_latency: float = 0.2,
        lambda_vram: float = 0.5,
        lambda_energy: float = 0.3,
        k_min: int = 1,
        k_max: int = 8,
        initial_k: int = 3,
        candidate_k_values: Optional[list[int]] = None,
        hardware_override: Optional[HardwareState] = None,
    ) -> None:
        self.configs = candidate_layer_configs
        if cost_model is not None:
            self.cost_model = cost_model
        elif model_name is not None:
            self.cost_model = MeasuredActionCostModel.from_model_name(model_name)
        else:
            self.cost_model = MeasuredActionCostModel.from_files()
        self.action_cost_db = action_cost_db
        self.gpu_profiler = gpu_profiler
        self.model_name = model_name
        self.max_vram_mb = max_vram_mb
        self.temp_threshold_c = temp_threshold_c
        self.power_budget_w = power_budget_w

        self.lambda_s = lambda_speed
        self.lambda_l = lambda_latency
        self.lambda_v = lambda_vram
        self.lambda_e = lambda_energy

        self.k_min = k_min
        self.k_max = k_max
        self.candidate_k_values = candidate_k_values or [k for k in [1, 2, 3, 4] if k_min <= k <= k_max]
        self.hardware_override = hardware_override

        # Inner adaptive-K controller for tracking online acceptance momentum
        self.k_controller = AdaptiveKController(
            k_min=k_min, k_max=k_max, initial_k=initial_k,
            entropy_low=0.7, entropy_high=1.8, acceptance_target=0.35,
        )

        self.current_config_name = next(iter(candidate_layer_configs.keys())) if candidate_layer_configs else "cka_75"
        self.action_history: list[ControllerAction] = []

    def get_hardware_state(self) -> HardwareState:
        """Poll current hardware state from GPU profiler or PyTorch."""
        if self.hardware_override is not None:
            return self.hardware_override

        vram = get_vram_usage()
        state = HardwareState(
            vram_used_mb=vram.get("allocated_mb", 2000.0),
            power_budget_w=self.power_budget_w,
        )
        if self.gpu_profiler:
            try:
                snap = self.gpu_profiler.snapshot()
                state.gpu_power_w = snap.get("power_w", 50.0)
                state.gpu_temperature_c = snap.get("temperature_c", 55.0)
                state.gpu_utilization = snap.get("utilization", {}).get("gpu_pct", 0.0)
                state.vram_total_mb = snap.get("memory", {}).get("total_mb", 6141.0)
            except Exception:
                pass
        return state

    def compute_utility(
        self,
        config_name: str,
        k: int,
        obs: ControllerObservation,
    ) -> float:
        """Evaluate data-driven utility U(a | z_t) using MeasuredActionCostModel or ActionCostDatabase."""
        # 1. Action Cost DB evaluation (if provided)
        if self.action_cost_db is not None:
            cost = self.action_cost_db.get_action_cost(config_name, k)
            if cost:
                tps = cost.get("tokens_per_second")
                if tps is None and "tokens_per_step" in cost and "total_cycle_ms" in cost:
                    tps = cost["tokens_per_step"] / max(1e-3, cost["total_cycle_ms"] / 1000.0)
                if tps is not None:
                    baseline_tps = self.cost_model.baseline_tps if self.cost_model else 36.0
                    expected_speedup = tps / baseline_tps
                    cycle_ms = cost.get("total_cycle_ms", obs.draft_latency_ms + obs.verify_latency_ms)
                    energy_j_tok = cost.get("energy_j_token", 1.8)

                    return self._calculate_joint_utility(
                        speedup=expected_speedup,
                        cycle_ms=cycle_ms,
                        energy_j_tok=energy_j_tok,
                        tokens_per_step=float(cost.get("tokens_per_step", 1.8)),
                        k=k,
                        obs=obs,
                        tps=float(tps),
                        config_name=config_name,
                    )

        # 2. Parametric / Empirical Cost Model evaluation
        pred = self.cost_model.evaluate_action(
            config_name=config_name,
            k=k,
            entropy=obs.entropy,
            vram_used_mb=obs.vram_used_mb,
            gpu_power_w=obs.gpu_power_w,
            gpu_temp_c=obs.gpu_temperature_c,
            lambda_speed=self.lambda_s,
            lambda_latency=self.lambda_l,
            lambda_vram=self.lambda_v,
            lambda_energy=self.lambda_e,
            max_vram_mb=self.max_vram_mb,
            temp_threshold_c=self.temp_threshold_c,
        )

        return self._calculate_joint_utility(
            speedup=pred.expected_speedup,
            cycle_ms=pred.total_cycle_ms,
            energy_j_tok=pred.expected_energy_j_tok,
            tokens_per_step=pred.expected_tokens_per_step,
            k=k,
            obs=obs,
            tps=pred.expected_tps,
            config_name=config_name,
        )

    def _calculate_joint_utility(
        self,
        speedup: float,
        cycle_ms: float,
        energy_j_tok: float,
        tokens_per_step: float,
        k: int,
        obs: ControllerObservation,
        tps: Optional[float] = None,
        config_name: str = "cka_75",
    ) -> float:
        """Evaluate constrained utility: U = λ_s·Speedup - λ_l·LatencyPenalty - λ_v·VRAM - λ_e·(Power+Thermal).

        This formulation matches the documented objective in the module docstring and
        ensures all lambda parameters are properly weighted.
        """
        # 1. Primary Objective: Speedup relative to baseline (dimensionless, typically 0.5-1.1)
        u_speedup = self.lambda_s * speedup

        # 2. Latency Penalty: penalize long cycle times (normalize to ~30ms baseline)
        latency_penalty = max(0.0, (cycle_ms - 30.0) / 30.0)
        u_latency = -self.lambda_l * latency_penalty

        # 3. Energy Regularization: penalize high energy per token
        u_energy = -self.lambda_e * energy_j_tok

        # 4. Physical Constraint 1: VRAM Headroom Barrier (B_v)
        # Use model-aware KV cache size per draft token
        kv_mb_per_token = getattr(self.cost_model, 'kv_cache_mb_per_token', 45.0) if hasattr(self, 'cost_model') else 45.0
        candidate_vram_mb = obs.vram_used_mb + k * kv_mb_per_token
        vram_headroom = self.max_vram_mb - candidate_vram_mb
        if vram_headroom < 100.0:
            vram_barrier = 1000.0 + (100.0 - vram_headroom) * 10.0
        elif vram_headroom < 500.0:
            vram_barrier = ((500.0 - vram_headroom) / max(10.0, vram_headroom)) * 5.0
        else:
            vram_barrier = 0.0
        u_vram = -self.lambda_v * vram_barrier

        # 5. Physical Constraint 2: Power Envelope Barrier (B_p)
        p_ratio = max(0.5, obs.gpu_power_w / max(10.0, obs.power_budget_w))
        if obs.gpu_power_w >= obs.power_budget_w * 0.95:
            power_barrier = energy_j_tok * (p_ratio ** 4) * 8.0
        else:
            power_barrier = energy_j_tok * 0.1
        u_power = -self.lambda_e * power_barrier

        # 6. Physical Constraint 3: Thermal Throttling Barrier (B_T)
        temp_margin = self.temp_threshold_c - obs.gpu_temperature_c
        kept_layers = self.cost_model.resolve_kept_layers(config_name) if hasattr(self, "cost_model") else 27
        kept_ratio = kept_layers / float(self.cost_model.total_layers if hasattr(self, "cost_model") else 36)
        if temp_margin <= 0.0:
            # Overheating emergency: higher kept layers generate significantly more heat
            thermal_barrier = 40.0 + kept_ratio * 70.0 + k * 25.0
        elif temp_margin < 5.0:
            thermal_barrier = (cycle_ms / 30.0) * np.exp((5.0 - temp_margin) / 1.5) * (1.0 + kept_ratio * 2.0)
        else:
            thermal_barrier = 0.0
        u_thermal = -self.lambda_e * thermal_barrier

        return float(u_speedup + u_latency + u_energy + u_vram + u_power + u_thermal)

    def select_action(
        self,
        entropy: float,
        last_accepted: int = 1,
        last_proposed: int = 2,
        draft_ms: float = 20.0,
        verify_ms: float = 25.0,
        hardware_state: Optional[HardwareState] = None,
    ) -> ControllerAction:
        """Select joint action a_t = (S_t, K_t) = argmax U(S, K | z_t)."""
        hw_state = hardware_state or self.get_hardware_state()

        # Update momentum tracker in inner controller
        _ = self.k_controller.update(
            entropy=entropy,
            accepted=last_accepted,
            proposed=last_proposed,
        )
        acc_rate = self.k_controller.avg_acceptance_rate

        # Construct full observation vector z_t
        obs = ControllerObservation(
            entropy=entropy,
            acceptance_rate=acc_rate,
            draft_latency_ms=draft_ms,
            verify_latency_ms=verify_ms,
            vram_used_mb=hw_state.vram_used_mb,
            gpu_power_w=hw_state.gpu_power_w,
            gpu_temperature_c=hw_state.gpu_temperature_c,
            power_budget_w=hw_state.power_budget_w,
        )

        # Joint optimization over action space S x K
        best_config = self.current_config_name
        best_k = 2
        best_util = -float("inf")
        best_cycle_ms = 50.0
        best_energy = 1.8

        for cfg_name in self.configs.keys():
            for cand_k in self.candidate_k_values:
                util = self.compute_utility(cfg_name, cand_k, obs)
                if util > best_util:
                    best_util = util
                    best_config = cfg_name
                    best_k = cand_k

        # Retrieve estimated cycle time & energy for logging
        pred = self.cost_model.evaluate_action(
            config_name=best_config,
            k=best_k,
            entropy=entropy,
            vram_used_mb=hw_state.vram_used_mb,
            gpu_power_w=hw_state.gpu_power_w,
            gpu_temp_c=hw_state.gpu_temperature_c,
        )

        self.current_config_name = best_config
        action = ControllerAction(
            config_name=best_config,
            skip_indices=self.configs[best_config],
            draft_length=best_k,
            predicted_utility=best_util,
            expected_cycle_ms=pred.total_cycle_ms,
            expected_energy_j_tok=pred.expected_energy_j_tok,
        )
        self.action_history.append(action)
        return action
