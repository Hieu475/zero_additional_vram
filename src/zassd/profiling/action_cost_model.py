"""Measured Action Cost Model (Phase 8 & Phase 8-Cost-Validation).

Replaces heuristic hardware assumptions with a hybrid empirical-parametric cost model
calibrated from real hardware benchmarks (RTX 4050 Laptop GPU):
  Action: a_t = (S_i, K_j)
  Observation: z_t = (H_t, A_{t-1}, HardwareState)

Predicts:
  - T_draft(S, K): Expected draft latency in milliseconds
  - T_verify(K): Expected target verification latency in milliseconds
  - alpha(S, K, H_t): Expected draft token acceptance probability given layer retention & sequence entropy
  - E[tokens/step]: Expected accepted tokens emitted per cycle
  - TPS(S, K, H_t): Effective throughput in tokens/second
  - Speedup(S, K, H_t): Effective speedup over empirically measured baseline
  - Energy(S, K): Expected hardware energy consumption in Joules/token
  - Utility: Multi-objective objective score balancing speedup, latency, VRAM headroom, and energy

Supports:
  1. Exact lookup when candidate action (S, K) has been directly profiled.
  2. Parametric OLS regression generalization when evaluating unseen actions (Gate B validation).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


# Default mapping of canonical CKA configurations to kept layer counts (Qwen2.5-3B, 36 layers)
KNOWN_CONFIG_KEPT_LAYERS: dict[str, int] = {
    "cka_50": 18,
    "cka_60": 22,
    "cka_67": 24,
    "cka_75": 27,
    "cka_83": 30,
    "cka_90": 32,
    "static_50": 18,
    "static_75": 27,
    "random_50": 18,
    "random_75": 27,
}


@dataclass
class ModelCostProfile:
    """Model-specific cost, architectural, and performance profile."""
    model_name: str
    total_layers: int
    baseline_tps: float
    known_kept_layers: dict[str, int]
    parametric_params: dict[str, float]


QWEN_25_3B_PROFILE = ModelCostProfile(
    model_name="qwen25_3b",
    total_layers=36,
    baseline_tps=42.94,
    known_kept_layers={
        "cka_50": 18,
        "cka_60": 22,
        "cka_67": 24,
        "cka_75": 27,
        "cka_83": 30,
        "cka_90": 32,
        "static_50": 18,
        "static_75": 27,
        "random_50": 18,
        "random_75": 27,
    },
    parametric_params={
        "draft_beta_1": 0.5237,
        "draft_beta_0": 4.6415,
        "verify_gamma_1": 3.5743,
        "verify_gamma_0": 23.30,
        "acc_w_r": 1.2529,
        "acc_w_k": 0.0989,
        "acc_w_0": -0.1393,
        "cache_ms": 0.40,
        "energy_e_1": 0.0530,
        "energy_e_0": 0.3155,
    },
)

LLAMA_32_3B_PROFILE = ModelCostProfile(
    model_name="llama32_3b",
    total_layers=28,
    baseline_tps=52.58,
    known_kept_layers={
        "cka_50": 14,
        "cka_60": 17,
        "cka_67": 19,
        "cka_75": 21,
        "cka_83": 23,
        "cka_90": 25,
        "static_50": 14,
        "static_75": 21,
        "random_50": 14,
        "random_75": 21,
        "llama_cka_75": 21,
        "llama_cka_50": 14,
    },
    parametric_params={
        "draft_beta_1": 0.5120,
        "draft_beta_0": 4.1000,
        "verify_gamma_1": 3.2000,
        "verify_gamma_0": 19.04,
        "acc_w_r": 1.1500,
        "acc_w_k": 0.0950,
        "acc_w_0": -0.1100,
        "cache_ms": 0.35,
        "energy_e_1": 0.0480,
        "energy_e_0": 0.2900,
    },
)


@dataclass
class CostModelPrediction:
    """Predicted metrics for a candidate action (S, K) under observation z_t."""
    config_name: str
    k: int
    draft_ms: float
    verify_ms: float
    total_cycle_ms: float
    expected_acceptance: float
    expected_tokens_per_step: float
    expected_tps: float
    expected_speedup: float
    expected_energy_j_tok: float
    utility: float = 0.0


class MeasuredActionCostModel:
    """Data-driven cost and performance model trained on empirical benchmark runs."""

    def __init__(
        self,
        action_costs: Optional[dict[str, dict[str, Any]]] = None,
        pareto_results: Optional[dict[str, dict[str, Any]]] = None,
        baseline_tps: Optional[float] = None,
        total_layers: Optional[int] = None,
        parametric_params: Optional[dict[str, float]] = None,
        profile: Optional[ModelCostProfile] = None,
        known_kept_layers: Optional[dict[str, int]] = None,
    ) -> None:
        if profile is not None:
            self.profile = profile
            self.total_layers = profile.total_layers if total_layers is None else total_layers
            self.baseline_tps = max(1.0, profile.baseline_tps if baseline_tps is None else baseline_tps)
            self.known_kept_layers = dict(profile.known_kept_layers)
            params = dict(profile.parametric_params)
            if parametric_params:
                params.update(parametric_params)
        else:
            self.profile = QWEN_25_3B_PROFILE
            self.total_layers = total_layers if total_layers is not None else 36
            self.baseline_tps = max(1.0, baseline_tps if baseline_tps is not None else 36.0)
            self.known_kept_layers = known_kept_layers or dict(KNOWN_CONFIG_KEPT_LAYERS)
            params = parametric_params or {}

        self.action_costs = action_costs or {}
        self.pareto_results = pareto_results or {}

        # Parametric parameters empirically fitted on RTX 4050 Laptop GPU (Gate B)
        self.draft_beta_1: float = params.get("draft_beta_1", 0.5237)
        self.draft_beta_0: float = params.get("draft_beta_0", 4.6415)
        self.verify_gamma_1: float = params.get("verify_gamma_1", 3.5743)
        self.verify_gamma_0: float = params.get("verify_gamma_0", 23.30)
        self.acc_w_r: float = params.get("acc_w_r", 1.2529)
        self.acc_w_k: float = params.get("acc_w_k", 0.0989)
        self.acc_w_0: float = params.get("acc_w_0", -0.1393)
        self.default_cache_ms: float = params.get("cache_ms", 0.40)
        self.energy_e_1: float = params.get("energy_e_1", 0.0530)
        self.energy_e_0: float = params.get("energy_e_0", 0.3155)

        # Legacy aliases for backward compatibility with existing tests
        self.base_verify_ms = self.verify_gamma_0
        self.verify_delta_per_k_ms = self.verify_gamma_1

        # Online entropy modulation constants
        self.entropy_reference = 1.0  # H_0 reference entropy
        self.entropy_sensitivity = 0.25  # gamma

    @classmethod
    def from_files(
        cls,
        action_costs_path: str | Path = "results/raw/action_costs.json",
        pareto_results_path: str | Path = "experiments/07_pareto/pareto_results.json",
        k_sweep_summary_path: str | Path = "experiments/07_k_sweep_cka83/summary.json",
        validation_summary_path: str | Path = "experiments/09_cost_validation/validation_summary.json",
        baseline_tps: float = 36.0,
    ) -> MeasuredActionCostModel:
        """Load empirical data and calibrated regression parameters from disk."""
        costs: dict[str, Any] = {}
        pareto: dict[str, Any] = {}
        parametric_params: dict[str, float] = {}

        p1 = Path(action_costs_path)
        if p1.exists():
            try:
                with open(p1) as f:
                    costs = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load action costs from {p1}: {e}")

        p2 = Path(pareto_results_path)
        if p2.exists():
            try:
                with open(p2) as f:
                    pareto = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load pareto data from {p2}: {e}")

        p3 = Path(k_sweep_summary_path)
        if p3.exists():
            try:
                with open(p3) as f:
                    k_data = json.load(f)
                if "vanilla_baseline_tok_s" in k_data and k_data["vanilla_baseline_tok_s"] > 0:
                    baseline_tps = float(k_data["vanilla_baseline_tok_s"])
                if "k_comparison" in k_data:
                    cfg_k_costs = costs.setdefault("cka_83", {})
                    for k_key, k_val in k_data["k_comparison"].items():
                        cfg_k_costs[k_key] = {
                            "draft_ms": k_val.get("draft_latency_ms"),
                            "verify_ms": k_val.get("verify_latency_ms"),
                            "cache_ms": k_val.get("cache_latency_ms"),
                            "acceptance_rate": k_val.get("acceptance_rate"),
                            "tokens_per_second": k_val.get("tokens_per_second_mean"),
                            "speedup": k_val.get("speedup_vs_vanilla"),
                            "tokens_per_step": k_val.get("tokens_per_step"),
                            "total_cycle_ms": k_val.get("cycle_latency_ms"),
                        }
            except Exception as e:
                logger.warning(f"Failed to load k sweep summary from {p3}: {e}")

        p4 = Path(validation_summary_path)
        if p4.exists():
            try:
                with open(p4) as f:
                    v_data = json.load(f)
                if "parametric_model_parameters" in v_data:
                    parametric_params = v_data["parametric_model_parameters"]
                if "vanilla_baseline_tps" in v_data and v_data["vanilla_baseline_tps"] > 0:
                    baseline_tps = float(v_data["vanilla_baseline_tps"])
            except Exception as e:
                logger.warning(f"Failed to load validation summary from {p4}: {e}")

        return cls(
            action_costs=costs,
            pareto_results=pareto,
            baseline_tps=baseline_tps,
            parametric_params=parametric_params,
        )

    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        baseline_tps: Optional[float] = None,
        **kwargs: Any,
    ) -> MeasuredActionCostModel:
        """Create a model-specific cost model tailored to Qwen or Llama architecture."""
        if "llama" in model_name.lower():
            profile = LLAMA_32_3B_PROFILE
        else:
            profile = QWEN_25_3B_PROFILE
        return cls(profile=profile, baseline_tps=baseline_tps, **kwargs)

    def resolve_kept_layers(self, config_name: str, explicit_kept_layers: Optional[int] = None) -> int:
        """Resolve the number of kept layers for a configuration."""
        if explicit_kept_layers is not None:
            return explicit_kept_layers
        if hasattr(self, "known_kept_layers") and config_name in self.known_kept_layers:
            return self.known_kept_layers[config_name]
        if config_name in KNOWN_CONFIG_KEPT_LAYERS:
            return KNOWN_CONFIG_KEPT_LAYERS[config_name]
        if config_name in self.pareto_results and "layers_kept" in self.pareto_results[config_name]:
            return int(self.pareto_results[config_name]["layers_kept"])

        # Heuristic inference from config string containing percentage (e.g. cka_50, static_60, random_75)
        import re
        match = re.search(r"(\d+)", config_name)
        if match:
            pct = int(match.group(1))
            if 10 <= pct <= 100:
                return int(round(self.total_layers * (pct / 100.0)))
        return int(round(self.total_layers * 0.75))

    def get_per_token_draft_ms(self, config_name: str, kept_layers: Optional[int] = None) -> float:
        """Get draft latency per token for a layer configuration."""
        cfg_costs = self.action_costs.get(config_name, {})
        if "K1" in cfg_costs and "draft_ms" in cfg_costs["K1"]:
            return float(cfg_costs["K1"]["draft_ms"])
        if "K2" in cfg_costs and "draft_ms" in cfg_costs["K2"]:
            return float(cfg_costs["K2"]["draft_ms"]) / 2.0
        if config_name in self.pareto_results and "draft_latency_ms" in self.pareto_results[config_name]:
            return float(self.pareto_results[config_name]["draft_latency_ms"])

        l_kept = self.resolve_kept_layers(config_name, kept_layers)
        return float(max(0.1, self.draft_beta_1 * l_kept + self.draft_beta_0))

    def predict_draft_ms(self, config_name: str, k: int, kept_layers: Optional[int] = None) -> float:
        """Predict draft latency for K candidate tokens.

        Uses exact measured cost if available, otherwise evaluates the parametric
        generalization model: T_draft = K * (beta_1 * L_kept + beta_0).
        """
        k_key = f"K{k}"
        cfg_costs = self.action_costs.get(config_name, {})
        if k_key in cfg_costs and "draft_ms" in cfg_costs[k_key]:
            return float(cfg_costs[k_key]["draft_ms"])

        per_tok = self.get_per_token_draft_ms(config_name, kept_layers)
        return float(k * per_tok)

    def predict_verify_ms(self, k: int) -> float:
        """Predict batched target verification latency for K candidates."""
        return float(max(10.0, self.verify_gamma_0 + (k - 1) * self.verify_gamma_1))

    def predict_acceptance_rate(
        self,
        config_name: str,
        k: int,
        entropy: float = 1.0,
        kept_layers: Optional[int] = None,
    ) -> float:
        """Predict draft acceptance probability conditioned on layer budget and sequence entropy.

        Uses exact measurements if present, otherwise evaluates the parametric model:
            alpha = w_r * (L_kept / total_layers) - w_k * (K - 1) + w_0 + (H_0 - H) * gamma
        """
        k_key = f"K{k}"
        cfg_costs = self.action_costs.get(config_name, {})
        if k_key in cfg_costs and "acceptance_rate" in cfg_costs[k_key]:
            base_acc = float(cfg_costs[k_key]["acceptance_rate"])
        elif config_name in self.pareto_results and "acceptance_rate" in self.pareto_results[config_name]:
            base_acc = float(self.pareto_results[config_name]["acceptance_rate"])
            if k != 2:
                base_acc -= self.acc_w_k * (k - 2)
        else:
            l_kept = self.resolve_kept_layers(config_name, kept_layers)
            r_kept = l_kept / float(self.total_layers)
            base_acc = float(self.acc_w_r * r_kept - self.acc_w_k * (k - 1) + self.acc_w_0)

        # Entropy modulation: lower entropy -> higher confidence -> higher acceptance
        entropy_shift = (self.entropy_reference - entropy) * self.entropy_sensitivity
        adjusted_acc = base_acc + entropy_shift

        return float(np.clip(adjusted_acc, 0.05, 0.98))

    def predict_energy_j_token(
        self,
        config_name: str,
        k: int,
        cycle_ms: Optional[float] = None,
        tokens_per_step: Optional[float] = None,
    ) -> float:
        """Predict energy consumption in Joules per token."""
        k_key = f"K{k}"
        cfg_costs = self.action_costs.get(config_name, {})
        if k_key in cfg_costs and "energy_j_token" in cfg_costs[k_key]:
            return float(cfg_costs[k_key]["energy_j_token"])
        if config_name in self.pareto_results and "energy_j_token" in self.pareto_results[config_name]:
            return float(self.pareto_results[config_name]["energy_j_token"])

        if cycle_ms is not None and tokens_per_step is not None and tokens_per_step > 0:
            c_per_tok = cycle_ms / tokens_per_step
            return float(max(0.5, self.energy_e_1 * c_per_tok + self.energy_e_0))
        return 1.85

    def evaluate_action(
        self,
        config_name: str,
        k: int,
        entropy: float = 1.0,
        vram_used_mb: float = 2000.0,
        gpu_power_w: float = 60.0,
        gpu_temp_c: float = 65.0,
        lambda_speed: float = 1.0,
        lambda_latency: float = 0.2,
        lambda_vram: float = 0.5,
        lambda_energy: float = 0.2,
        max_vram_mb: float = 5500.0,
        temp_threshold_c: float = 82.0,
        kept_layers: Optional[int] = None,
    ) -> CostModelPrediction:
        """Evaluate complete cost model prediction and utility for candidate action (S, K)."""
        draft_ms = self.predict_draft_ms(config_name, k, kept_layers=kept_layers)
        verify_ms = self.predict_verify_ms(k)
        cycle_ms = draft_ms + verify_ms + self.default_cache_ms

        acc_rate = self.predict_acceptance_rate(config_name, k, entropy=entropy, kept_layers=kept_layers)
        tokens_per_step = 1.0 + acc_rate * k

        eff_tps = tokens_per_step / (cycle_ms / 1000.0)
        speedup = eff_tps / self.baseline_tps

        energy_j_tok = self.predict_energy_j_token(
            config_name, k, cycle_ms=cycle_ms, tokens_per_step=tokens_per_step
        )

        # Penalties
        latency_penalty = cycle_ms / 100.0
        vram_headroom = max(0.0, max_vram_mb - vram_used_mb)
        vram_penalty = 1.0 / max(vram_headroom, 100.0)

        temp_margin = temp_threshold_c - gpu_temp_c
        thermal_penalty = 1.5 if temp_margin < 5.0 else 0.0
        power_penalty = gpu_power_w / 80.0

        utility = (
            lambda_speed * speedup
            - lambda_latency * latency_penalty
            - lambda_vram * vram_penalty
            - lambda_energy * (power_penalty + thermal_penalty)
        )

        return CostModelPrediction(
            config_name=config_name,
            k=k,
            draft_ms=round(draft_ms, 2),
            verify_ms=round(verify_ms, 2),
            total_cycle_ms=round(cycle_ms, 2),
            expected_acceptance=round(acc_rate, 4),
            expected_tokens_per_step=round(tokens_per_step, 3),
            expected_tps=round(eff_tps, 2),
            expected_speedup=round(speedup, 3),
            expected_energy_j_tok=round(energy_j_tok, 4),
            utility=float(utility),
        )
