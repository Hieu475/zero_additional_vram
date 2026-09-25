"""Phase 8 — Cost Model Validation via Calibration/Holdout Experiment (Gate B).

Scientific Question:
Does the MeasuredActionCostModel generalize to unseen actions (S, K) outside
the measurements it was calibrated from?

Protocol:
- Calibration Set:
    Configurations: CKA-50 (18 kept), CKA-60 (22 kept), CKA-75 (27 kept)
    Speculation lengths: K in {1, 2, 4}
    (9 actions)
- Holdout Set (Strictly Unseen):
    Configurations: CKA-67 (24 kept), CKA-83 (30 kept), CKA-90 (32 kept)
    Speculation lengths: K in {3, 6}
    (6 actions)

Both the layer configurations AND speculation depths are disjoint between
calibration and holdout sets.

Evaluation Metrics:
- MAE_draft: Mean Absolute Error of draft latency (ms)
- MAE_verify: Mean Absolute Error of verify latency (ms)
- MAE_latency: Mean Absolute Error of total cycle latency (ms)
- MAPE_latency: Mean Absolute Percentage Error of total cycle latency (%)
- MAE_energy: Mean Absolute Error of energy consumption (J/tok)
- Spearman rho and Kendall tau rank correlation between predicted and measured utility.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr
import torch

from zassd.decoding.speculative import self_speculative_generate
from zassd.decoding.vanilla import vanilla_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.profiling.gpu import GPUProfiler
from zassd.utils.logging import setup_logging
from zassd.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Complete layer configuration definitions
CONFIG_DEFINITIONS = {
    "cka_50": {
        "kept_layers": 18,
        "skipped_layers": 18,
        "skip_indices": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22],
    },
    "cka_60": {
        "kept_layers": 22,
        "skipped_layers": 14,
        "skip_indices": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 21],
    },
    "cka_67": {
        "kept_layers": 24,
        "skipped_layers": 12,
        "skip_indices": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16],
    },
    "cka_75": {
        "kept_layers": 27,
        "skipped_layers": 9,
        "skip_indices": [3, 4, 5, 6, 7, 10, 11, 12, 13],
    },
    "cka_83": {
        "kept_layers": 30,
        "skipped_layers": 6,
        "skip_indices": [4, 5, 6, 7, 12, 13],
    },
    "cka_90": {
        "kept_layers": 32,
        "skipped_layers": 4,
        "skip_indices": [4, 5, 6, 7],
    },
}

CALIBRATION_ACTIONS = [
    ("cka_50", 1), ("cka_50", 2), ("cka_50", 4),
    ("cka_60", 1), ("cka_60", 2), ("cka_60", 4),
    ("cka_75", 1), ("cka_75", 2), ("cka_75", 4),
]

HOLDOUT_ACTIONS = [
    ("cka_67", 3), ("cka_67", 6),
    ("cka_83", 3), ("cka_83", 6),
    ("cka_90", 3), ("cka_90", 6),
]


def compute_utility(
    eff_tps: float,
    baseline_tps: float,
    cycle_ms: float,
    energy_j_tok: float,
    vram_used_mb: float = 2000.0,
    gpu_power_w: float = 60.0,
    gpu_temp_c: float = 65.0,
    lambda_speed: float = 1.0,
    lambda_latency: float = 0.2,
    lambda_vram: float = 0.5,
    lambda_energy: float = 0.2,
    max_vram_mb: float = 5500.0,
    temp_threshold_c: float = 82.0,
) -> float:
    """Compute empirical utility according to Phase 8 specification."""
    speedup = eff_tps / max(1.0, baseline_tps)
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
    return float(utility)


def benchmark_action(
    model,
    tokenizer,
    layer_mgr: LayerManager,
    config_name: str,
    k: int,
    eval_prompts: list[dict[str, Any]],
    gpu_profiler: GPUProfiler | None,
    max_new_tokens: int = 32,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Execute empirical measurement of a single candidate action (S, K)."""
    skip_indices = CONFIG_DEFINITIONS[config_name]["skip_indices"]
    kept_layers = CONFIG_DEFINITIONS[config_name]["kept_layers"]

    e_start = gpu_profiler.get_total_energy_mj() if gpu_profiler else None
    t0 = time.perf_counter()

    runs = []
    for p in eval_prompts:
        _, m = self_speculative_generate(
            model=model,
            tokenizer=tokenizer,
            layer_mgr=layer_mgr,
            skip_indices=skip_indices,
            prompt=p["prompt"],
            k=k,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            device=device,
        )
        runs.append(m)

    elapsed = time.perf_counter() - t0
    e_end = gpu_profiler.get_total_energy_mj() if gpu_profiler else None

    tot_cycles = max(1, sum(m.num_verification_cycles for m in runs))
    draft_ms = (sum(m.draft_time_s for m in runs) / tot_cycles) * 1000.0
    verify_ms = (sum(m.verify_time_s for m in runs) / tot_cycles) * 1000.0
    cache_ms = (sum(m.cache_time_s for m in runs) / tot_cycles) * 1000.0
    other_ms = (sum(m.other_time_s for m in runs) / tot_cycles) * 1000.0
    cycle_ms = draft_ms + verify_ms + cache_ms + other_ms

    tot_tokens = max(1, sum(m.total_tokens for m in runs))
    if e_start is not None and e_end is not None and e_end >= e_start:
        energy_j_tok = ((e_end - e_start) / 1000.0) / tot_tokens
    else:
        power_w = gpu_profiler.get_power_usage() if gpu_profiler else 60.0
        energy_j_tok = (power_w * elapsed) / tot_tokens

    mean_acc = float(np.mean([m.acceptance_rate for m in runs]))
    mean_tps = float(np.mean([m.tokens_per_second for m in runs]))
    mean_tokens_per_step = float(np.mean([m.tokens_per_step for m in runs]))
    mean_vram = float(np.mean([m.peak_vram_mb for m in runs]))

    return {
        "config_name": config_name,
        "k": k,
        "kept_layers": kept_layers,
        "draft_ms": draft_ms,
        "verify_ms": verify_ms,
        "cache_ms": cache_ms,
        "other_ms": other_ms,
        "cycle_ms": cycle_ms,
        "acceptance_rate": mean_acc,
        "tokens_per_step": mean_tokens_per_step,
        "tokens_per_second": mean_tps,
        "energy_j_token": energy_j_tok,
        "vram_mb": mean_vram,
    }


def fit_parametric_cost_model(calibration_records: list[dict[str, Any]], total_layers: int = 36) -> dict[str, Any]:
    """Fit parametric regression models on calibration measurements."""
    # 1. Draft Latency Model: T_draft(S, K) = K * (beta_1 * L_kept + beta_0)
    # y = draft_ms / K, x = kept_layers
    x_kept = np.array([r["kept_layers"] for r in calibration_records], dtype=float)
    k_vals = np.array([r["k"] for r in calibration_records], dtype=float)
    y_per_tok = np.array([r["draft_ms"] / r["k"] for r in calibration_records], dtype=float)

    # OLS fit: y_per_tok = beta_1 * x_kept + beta_0
    A_draft = np.vstack([x_kept, np.ones(len(x_kept))]).T
    beta_1, beta_0 = np.linalg.lstsq(A_draft, y_per_tok, rcond=None)[0]

    # 2. Verify Latency Model: T_verify(K) = gamma_1 * (K - 1) + gamma_0
    # Batched target verification forward pass
    x_k_delta = k_vals - 1.0
    y_verify = np.array([r["verify_ms"] for r in calibration_records], dtype=float)
    A_verify = np.vstack([x_k_delta, np.ones(len(x_k_delta))]).T
    gamma_1, gamma_0 = np.linalg.lstsq(A_verify, y_verify, rcond=None)[0]

    # 3. Acceptance Rate Model: alpha(S, K) = w_r * (L_kept / total_layers) - w_k * (K - 1) + w_0
    r_kept = x_kept / float(total_layers)
    y_acc = np.array([r["acceptance_rate"] for r in calibration_records], dtype=float)
    A_acc = np.vstack([r_kept, -x_k_delta, np.ones(len(r_kept))]).T
    w_r, w_k, w_0 = np.linalg.lstsq(A_acc, y_acc, rcond=None)[0]

    # 4. Cache & other latency defaults
    mean_cache = float(np.mean([r["cache_ms"] + r.get("other_ms", 0.0) for r in calibration_records]))

    # 5. Energy Model: E(S, K) = e_1 * (cycle_ms / tokens_per_step) + e_0
    y_energy = np.array([r["energy_j_token"] for r in calibration_records], dtype=float)
    cycle_per_token = np.array([r["cycle_ms"] / max(0.1, r["tokens_per_step"]) for r in calibration_records], dtype=float)
    A_energy = np.vstack([cycle_per_token, np.ones(len(cycle_per_token))]).T
    e_1, e_0 = np.linalg.lstsq(A_energy, y_energy, rcond=None)[0]

    params = {
        "draft_beta_1": float(beta_1),
        "draft_beta_0": float(beta_0),
        "verify_gamma_1": float(gamma_1),
        "verify_gamma_0": float(gamma_0),
        "acc_w_r": float(w_r),
        "acc_w_k": float(w_k),
        "acc_w_0": float(w_0),
        "cache_ms": float(mean_cache),
        "energy_e_1": float(e_1),
        "energy_e_0": float(e_0),
    }
    return params


def predict_parametric(
    params: dict[str, Any],
    config_name: str,
    k: int,
    baseline_tps: float,
    entropy: float = 1.0,
    total_layers: int = 36,
) -> dict[str, float]:
    """Predict metrics using fitted parametric regression models."""
    kept_layers = CONFIG_DEFINITIONS[config_name]["kept_layers"]
    r_kept = kept_layers / float(total_layers)

    # Draft latency
    t_tok = max(0.1, params["draft_beta_1"] * kept_layers + params["draft_beta_0"])
    pred_draft_ms = float(k * t_tok)

    # Verify latency
    pred_verify_ms = float(max(10.0, params["verify_gamma_1"] * (k - 1) + params["verify_gamma_0"]))

    # Total cycle latency
    pred_cycle_ms = pred_draft_ms + pred_verify_ms + params["cache_ms"]

    # Acceptance rate
    pred_acc = float(np.clip(
        params["acc_w_r"] * r_kept - params["acc_w_k"] * (k - 1) + params["acc_w_0"] + (1.0 - entropy) * 0.25,
        0.05,
        0.98,
    ))

    # Expected tokens per step and throughput
    tokens_per_step = 1.0 + pred_acc * k
    eff_tps = tokens_per_step / (pred_cycle_ms / 1000.0)
    speedup = eff_tps / max(1.0, baseline_tps)

    # Energy
    c_per_tok = pred_cycle_ms / max(0.1, tokens_per_step)
    pred_energy = float(max(0.5, params["energy_e_1"] * c_per_tok + params["energy_e_0"]))

    pred_utility = compute_utility(
        eff_tps=eff_tps,
        baseline_tps=baseline_tps,
        cycle_ms=pred_cycle_ms,
        energy_j_tok=pred_energy,
    )

    return {
        "draft_ms": pred_draft_ms,
        "verify_ms": pred_verify_ms,
        "cycle_ms": pred_cycle_ms,
        "acceptance_rate": pred_acc,
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": eff_tps,
        "speedup": speedup,
        "energy_j_token": pred_energy,
        "utility": pred_utility,
    }


def predict_baseline_cost_model(
    cost_model: MeasuredActionCostModel,
    config_name: str,
    k: int,
    baseline_tps: float,
    entropy: float = 1.0,
) -> dict[str, float]:
    """Predict metrics using original Phase 8 cost model (unmodified lookup)."""
    pred = cost_model.evaluate_action(
        config_name=config_name,
        k=k,
        entropy=entropy,
    )
    return {
        "draft_ms": pred.draft_ms,
        "verify_ms": pred.verify_ms,
        "cycle_ms": pred.total_cycle_ms,
        "acceptance_rate": pred.expected_acceptance,
        "tokens_per_step": pred.expected_tokens_per_step,
        "tokens_per_second": pred.expected_tps,
        "speedup": pred.expected_speedup,
        "energy_j_token": pred.expected_energy_j_tok,
        "utility": pred.utility,
    }


def evaluate_predictions(
    measured_records: list[dict[str, Any]],
    predicted_records: list[dict[str, float]],
    baseline_tps: float,
) -> dict[str, Any]:
    """Compute quantitative prediction errors and rank correlations."""
    n = len(measured_records)
    draft_errors = [abs(p["draft_ms"] - m["draft_ms"]) for p, m in zip(predicted_records, measured_records)]
    verify_errors = [abs(p["verify_ms"] - m["verify_ms"]) for p, m in zip(predicted_records, measured_records)]
    cycle_errors = [abs(p["cycle_ms"] - m["cycle_ms"]) for p, m in zip(predicted_records, measured_records)]
    mape_errors = [
        abs(p["cycle_ms"] - m["cycle_ms"]) / max(0.1, m["cycle_ms"]) * 100.0
        for p, m in zip(predicted_records, measured_records)
    ]
    energy_errors = [abs(p["energy_j_token"] - m["energy_j_token"]) for p, m in zip(predicted_records, measured_records)]

    # Compute measured utility
    measured_utilities = []
    for m in measured_records:
        u = compute_utility(
            eff_tps=m["tokens_per_second"],
            baseline_tps=baseline_tps,
            cycle_ms=m["cycle_ms"],
            energy_j_tok=m["energy_j_token"],
            vram_used_mb=m.get("vram_mb", 2000.0),
        )
        measured_utilities.append(u)

    predicted_utilities = [p["utility"] for p in predicted_records]

    # Rank correlation
    spearman_rho, spearman_p = spearmanr(predicted_utilities, measured_utilities)
    kendall_tau_val, kendall_p = kendalltau(predicted_utilities, measured_utilities)

    return {
        "mae_draft": float(np.mean(draft_errors)),
        "mae_verify": float(np.mean(verify_errors)),
        "mae_cycle": float(np.mean(cycle_errors)),
        "mape_latency_pct": float(np.mean(mape_errors)),
        "mae_energy": float(np.mean(energy_errors)),
        "spearman_rho": float(spearman_rho) if not np.isnan(spearman_rho) else 0.0,
        "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
        "kendall_tau": float(kendall_tau_val) if not np.isnan(kendall_tau_val) else 0.0,
        "kendall_p": float(kendall_p) if not np.isnan(kendall_p) else 1.0,
        "measured_utilities": measured_utilities,
        "predicted_utilities": predicted_utilities,
    }


def plot_validation_results(
    holdout_actions: list[tuple[str, int]],
    holdout_measured: list[dict[str, Any]],
    baseline_preds: list[dict[str, float]],
    parametric_preds: list[dict[str, float]],
    baseline_metrics: dict[str, Any],
    parametric_metrics: dict[str, Any],
    figures_dir: Path,
) -> None:
    """Generate publication-quality validation figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [f"{cfg} (K={k})" for cfg, k in holdout_actions]
    x = np.arange(len(labels))
    width = 0.28

    # 1. Draft Latency Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    m_draft = [m["draft_ms"] for m in holdout_measured]
    b_draft = [p["draft_ms"] for p in baseline_preds]
    p_draft = [p["draft_ms"] for p in parametric_preds]

    ax.bar(x - width, m_draft, width, label="Ground Truth (Measured)", color="#2ca02c")
    ax.bar(x, b_draft, width, label=f"Phase 8 Cost Model (MAE={baseline_metrics['mae_draft']:.1f}ms)", color="#d62728", alpha=0.8)
    ax.bar(x + width, p_draft, width, label=f"Parametric Model (MAE={parametric_metrics['mae_draft']:.1f}ms)", color="#1f77b4")
    ax.set_ylabel("Draft Latency (ms)", fontsize=12)
    ax.set_title("Cost Model Generalization: Draft Latency on Unseen Actions (Holdout)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "cost_model_validation_draft.png", dpi=300)
    plt.close()

    # 2. Total Cycle Latency Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    m_cycle = [m["cycle_ms"] for m in holdout_measured]
    b_cycle = [p["cycle_ms"] for p in baseline_preds]
    p_cycle = [p["cycle_ms"] for p in parametric_preds]

    ax.bar(x - width, m_cycle, width, label="Ground Truth (Measured)", color="#2ca02c")
    ax.bar(x, b_cycle, width, label=f"Phase 8 Cost Model (MAPE={baseline_metrics['mape_latency_pct']:.1f}%)", color="#d62728", alpha=0.8)
    ax.bar(x + width, p_cycle, width, label=f"Parametric Model (MAPE={parametric_metrics['mape_latency_pct']:.1f}%)", color="#1f77b4")
    ax.set_ylabel("Total Cycle Latency (ms)", fontsize=12)
    ax.set_title("Cost Model Generalization: Cycle Latency on Unseen Actions (Holdout)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "cost_model_validation_cycle.png", dpi=300)
    plt.close()

    # 3. Energy Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    m_energy = [m["energy_j_token"] for m in holdout_measured]
    b_energy = [p["energy_j_token"] for p in baseline_preds]
    p_energy = [p["energy_j_token"] for p in parametric_preds]

    ax.bar(x - width, m_energy, width, label="Ground Truth (Measured)", color="#2ca02c")
    ax.bar(x, b_energy, width, label=f"Phase 8 Cost Model (MAE={baseline_metrics['mae_energy']:.3f}J)", color="#d62728", alpha=0.8)
    ax.bar(x + width, p_energy, width, label=f"Parametric Model (MAE={parametric_metrics['mae_energy']:.3f}J)", color="#1f77b4")
    ax.set_ylabel("Energy (Joules / Token)", fontsize=12)
    ax.set_title("Cost Model Generalization: Energy/Token on Unseen Actions (Holdout)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "cost_model_validation_energy.png", dpi=300)
    plt.close()

    # 4. Utility Scatter & Rank Correlation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    meas_u = baseline_metrics["measured_utilities"]

    # Baseline scatter
    ax1.scatter(meas_u, baseline_metrics["predicted_utilities"], color="#d62728", s=80, edgecolors="k", zorder=3)
    ax1.plot([min(meas_u), max(meas_u)], [min(meas_u), max(meas_u)], "k--", alpha=0.6, label="Ideal (1:1)")
    ax1.set_xlabel("Measured Utility", fontsize=11)
    ax1.set_ylabel("Predicted Utility", fontsize=11)
    ax1.set_title(f"Phase 8 Cost Model\nSpearman ρ = {baseline_metrics['spearman_rho']:.3f}, Kendall τ = {baseline_metrics['kendall_tau']:.3f}", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Parametric scatter
    ax2.scatter(meas_u, parametric_metrics["predicted_utilities"], color="#1f77b4", s=80, edgecolors="k", zorder=3)
    ax2.plot([min(meas_u), max(meas_u)], [min(meas_u), max(meas_u)], "k--", alpha=0.6, label="Ideal (1:1)")
    ax2.set_xlabel("Measured Utility", fontsize=11)
    ax2.set_ylabel("Predicted Utility", fontsize=11)
    ax2.set_title(f"Upgraded Parametric Model\nSpearman ρ = {parametric_metrics['spearman_rho']:.3f}, Kendall τ = {parametric_metrics['kendall_tau']:.3f}", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.suptitle("Utility Prediction & Ranking Correlation on Unseen Actions (Gate B)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "cost_model_predicted_vs_measured_utility.png", dpi=300)
    plt.close()
    logger.info(f"Figures saved successfully to {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MeasuredActionCostModel on Calibration vs Holdout splits")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of prompts per action")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to generate per prompt")
    parser.add_argument("--output-dir", type=str, default="experiments/09_cost_validation")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--force-reprofile", action="store_true", help="Force re-running GPU profiling")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    measured_file = output_dir / "measured_actions.json"

    # Step 1: Benchmark or Load Empirical Measurements
    if measured_file.exists() and not args.force_reprofile:
        logger.info(f"Loading cached measurements from {measured_file}")
        with open(measured_file) as f:
            data = json.load(f)
        vanilla_tps = data["vanilla_tps"]
        calibration_measured = data["calibration_measured"]
        holdout_measured = data["holdout_measured"]
    else:
        logger.info("Initializing GPU and loading model for empirical benchmarking...")
        model = load_model(args.model, quantize=True, bits=4)
        tokenizer = load_tokenizer(args.model)
        adapter = ModelAdapter(model)
        layer_mgr = LayerManager(adapter)
        gpu_profiler = GPUProfiler()

        # Load prompts
        prompts_path = Path("data/benchmarks/prompts.jsonl")
        prompts = []
        with open(prompts_path) as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line.strip()))
        eval_prompts = prompts[: args.num_prompts]

        # Benchmark Vanilla Baseline
        logger.info("--- Measuring Vanilla Baseline ---")
        van_tps_list = []
        for p in eval_prompts:
            _, vm = vanilla_generate(model, tokenizer, p["prompt"], max_new_tokens=args.max_new_tokens, temperature=0.0)
            van_tps_list.append(vm.tokens_per_second)
        vanilla_tps = float(np.mean(van_tps_list))
        logger.info(f"Vanilla Target Baseline: {vanilla_tps:.2f} tok/s")

        # Benchmark Calibration Set
        logger.info("\n--- Benchmarking Calibration Actions (9 actions) ---")
        calibration_measured = []
        for cfg, k in CALIBRATION_ACTIONS:
            logger.info(f"Profiling Calibration Action: {cfg}, K={k}")
            rec = benchmark_action(
                model=model,
                tokenizer=tokenizer,
                layer_mgr=layer_mgr,
                config_name=cfg,
                k=k,
                eval_prompts=eval_prompts,
                gpu_profiler=gpu_profiler,
                max_new_tokens=args.max_new_tokens,
            )
            calibration_measured.append(rec)

        # Benchmark Holdout Set
        logger.info("\n--- Benchmarking Holdout Actions (6 actions) ---")
        holdout_measured = []
        for cfg, k in HOLDOUT_ACTIONS:
            logger.info(f"Profiling Holdout Action: {cfg}, K={k}")
            rec = benchmark_action(
                model=model,
                tokenizer=tokenizer,
                layer_mgr=layer_mgr,
                config_name=cfg,
                k=k,
                eval_prompts=eval_prompts,
                gpu_profiler=gpu_profiler,
                max_new_tokens=args.max_new_tokens,
            )
            holdout_measured.append(rec)

        # Cache measurements
        cached_data = {
            "vanilla_tps": vanilla_tps,
            "calibration_measured": calibration_measured,
            "holdout_measured": holdout_measured,
        }
        with open(measured_file, "w") as f:
            json.dump(cached_data, f, indent=2)
        logger.info(f"Measurements saved to {measured_file}")

    # Step 2: Build Calibration Database for Original Cost Model
    # The cost model is given ONLY calibration measurements!
    calib_action_costs: dict[str, dict[str, Any]] = {}
    calib_pareto: dict[str, dict[str, Any]] = {}

    for r in calibration_measured:
        cfg = r["config_name"]
        k_key = f"K{r['k']}"
        calib_action_costs.setdefault(cfg, {})[k_key] = r
        if r["k"] == 2:
            calib_pareto[cfg] = {
                "name": cfg,
                "draft_latency_ms": r["draft_ms"] / 2.0,
                "acceptance_rate": r["acceptance_rate"],
                "energy_j_token": r["energy_j_token"],
            }

    original_cost_model = MeasuredActionCostModel(
        action_costs=calib_action_costs,
        pareto_results=calib_pareto,
        baseline_tps=vanilla_tps,
    )

    # Step 3: Fit Upgraded Parametric Regression Models
    logger.info("\n--- Fitting Parametric Regression Model on Calibration Data ---")
    parametric_params = fit_parametric_cost_model(calibration_measured, total_layers=36)
    logger.info(f"Fitted Parametric Parameters: {json.dumps(parametric_params, indent=2)}")

    # Step 4: Evaluate Predictions on Unseen Holdout Set
    logger.info("\n--- Evaluating Predictions on Holdout Set (6 unseen actions) ---")
    baseline_predictions = []
    parametric_predictions = []

    for r in holdout_measured:
        cfg = r["config_name"]
        k = r["k"]

        # Baseline prediction
        bp = predict_baseline_cost_model(original_cost_model, cfg, k, vanilla_tps)
        baseline_predictions.append(bp)

        # Parametric prediction
        pp = predict_parametric(parametric_params, cfg, k, vanilla_tps)
        parametric_predictions.append(pp)

    # Compute error metrics
    baseline_eval = evaluate_predictions(holdout_measured, baseline_predictions, vanilla_tps)
    parametric_eval = evaluate_predictions(holdout_measured, parametric_predictions, vanilla_tps)

    # Step 5: Print Evaluation Table
    logger.info("\n" + "=" * 90)
    logger.info("GATE B: COST MODEL GENERALIZATION VALIDATION RESULTS")
    logger.info("=" * 90)
    header = f"{'Metric':<30} | {'Phase 8 Cost Model':<25} | {'Upgraded Parametric':<25} | {'Improvement':<10}"
    logger.info(header)
    logger.info("-" * 95)

    comp_rows = [
        ("MAE Draft Latency (ms)", baseline_eval["mae_draft"], parametric_eval["mae_draft"], True),
        ("MAE Verify Latency (ms)", baseline_eval["mae_verify"], parametric_eval["mae_verify"], True),
        ("MAE Cycle Latency (ms)", baseline_eval["mae_cycle"], parametric_eval["mae_cycle"], True),
        ("MAPE Cycle Latency (%)", baseline_eval["mape_latency_pct"], parametric_eval["mape_latency_pct"], True),
        ("MAE Energy (J/tok)", baseline_eval["mae_energy"], parametric_eval["mae_energy"], True),
        ("Spearman Rank Corr (ρ)", baseline_eval["spearman_rho"], parametric_eval["spearman_rho"], False),
        ("Kendall Rank Corr (τ)", baseline_eval["kendall_tau"], parametric_eval["kendall_tau"], False),
    ]

    for name, b_val, p_val, is_err in comp_rows:
        if is_err:
            impr = f"{(b_val - p_val) / max(1e-5, b_val) * 100:+.1f}%"
        else:
            impr = f"{p_val - b_val:+.3f}"
        logger.info(f"{name:<30} | {b_val:<25.4f} | {p_val:<25.4f} | {impr:<10}")

    logger.info("=" * 95)

    # Per-action breakdown table
    logger.info("\n--- Per-Action Holdout Detailed Breakdown ---")
    detail_header = f"{'Action':<15} | {'Draft (M/B/P)':<20} | {'Cycle (M/B/P)':<20} | {'Util (M/B/P)':<20}"
    logger.info(detail_header)
    logger.info("-" * 80)
    for i, (cfg, k) in enumerate(HOLDOUT_ACTIONS):
        m = holdout_measured[i]
        b = baseline_predictions[i]
        p = parametric_predictions[i]
        mu = baseline_eval["measured_utilities"][i]
        bu = b["utility"]
        pu = p["utility"]
        d_str = f"{m['draft_ms']:.1f} / {b['draft_ms']:.1f} / {p['draft_ms']:.1f}"
        c_str = f"{m['cycle_ms']:.1f} / {b['cycle_ms']:.1f} / {p['cycle_ms']:.1f}"
        u_str = f"{mu:.2f} / {bu:.2f} / {pu:.2f}"
        logger.info(f"{f'{cfg} (K={k})':<15} | {d_str:<20} | {c_str:<20} | {u_str:<20}")

    # Step 6: Generate Figures
    plot_validation_results(
        holdout_actions=HOLDOUT_ACTIONS,
        holdout_measured=holdout_measured,
        baseline_preds=baseline_predictions,
        parametric_preds=parametric_predictions,
        baseline_metrics=baseline_eval,
        parametric_metrics=parametric_eval,
        figures_dir=figures_dir,
    )

    # Step 7: Write summary JSON
    summary_report = {
        "gate_b_status": "PASS" if parametric_eval["spearman_rho"] >= 0.80 and parametric_eval["mape_latency_pct"] <= 15.0 else "REVIEW",
        "vanilla_baseline_tps": vanilla_tps,
        "calibration_set": {
            "actions": [f"{cfg}_K{k}" for cfg, k in CALIBRATION_ACTIONS],
            "num_actions": len(CALIBRATION_ACTIONS),
        },
        "holdout_set": {
            "actions": [f"{cfg}_K{k}" for cfg, k in HOLDOUT_ACTIONS],
            "num_actions": len(HOLDOUT_ACTIONS),
        },
        "parametric_model_parameters": parametric_params,
        "metrics_summary": {
            "baseline_model": {
                "mae_draft_ms": baseline_eval["mae_draft"],
                "mae_verify_ms": baseline_eval["mae_verify"],
                "mae_cycle_ms": baseline_eval["mae_cycle"],
                "mape_latency_pct": baseline_eval["mape_latency_pct"],
                "mae_energy_j_token": baseline_eval["mae_energy"],
                "spearman_rank_correlation": baseline_eval["spearman_rho"],
                "kendall_tau_correlation": baseline_eval["kendall_tau"],
            },
            "upgraded_parametric_model": {
                "mae_draft_ms": parametric_eval["mae_draft"],
                "mae_verify_ms": parametric_eval["mae_verify"],
                "mae_cycle_ms": parametric_eval["mae_cycle"],
                "mape_latency_pct": parametric_eval["mape_latency_pct"],
                "mae_energy_j_token": parametric_eval["mae_energy"],
                "spearman_rank_correlation": parametric_eval["spearman_rho"],
                "kendall_tau_correlation": parametric_eval["kendall_tau"],
            },
        },
    }

    summary_file = output_dir / "validation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_report, f, indent=2)
    logger.info(f"\nSummary successfully written to {summary_file}")


if __name__ == "__main__":
    main()
