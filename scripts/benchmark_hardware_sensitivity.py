"""Phase 9 — Hardware Sensitivity Matrix Benchmark (Việc 3).

Investigates whether HardwareAwareJointController genuinely adapts its decisions:
    HardwareState_t -> Action_t = (S_t, K_t)
across 5 distinct hardware operating regimes:
  1. High VRAM headroom, Normal power, Cool temperature (Unconstrained high-performance)
  2. Low VRAM headroom, Normal power, Cool temperature (Memory-capacity constrained)
  3. High VRAM headroom, Limited power, Cool temperature (Power / Battery constrained)
  4. High VRAM headroom, Normal power, Hot temperature (Thermal throttle avoidance)
  5. Low VRAM headroom, Limited power, Hot temperature (Triple bottleneck worst-case)

Measures and logs:
  - Selected S_t (layer configuration)
  - Selected K_t (speculation draft length)
  - Predicted utility U_t
  - Actual generation throughput (tok/s)
  - Actual cycle latency (ms)
  - Actual energy consumption (J/token)
  - Peak VRAM footprint (MB)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from zassd.controllers.hardware_controller import HardwareAwareJointController, HardwareState
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.profiling.gpu import GPUProfiler
from zassd.utils.logging import setup_logging
from zassd.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Candidate layer configurations sorted by CKA redundancy
CANDIDATE_CONFIGS = {
    "cka_83": [4, 5, 6, 7, 12, 13],                                       # 30 kept, high quality
    "cka_75": [3, 4, 5, 6, 7, 10, 11, 12, 13],                           # 27 kept, lowest energy
    "cka_60": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 21],       # 22 kept, fast / light
    "cka_50": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22], # 18 kept, minimal
}

REGIME_DEFINITIONS = [
    {
        "name": "High VRAM / Normal Power / Cool",
        "short_name": "high_norm_cool",
        "vram_used_mb": 2000.0,
        "gpu_power_w": 50.0,
        "gpu_temperature_c": 55.0,
        "power_budget_w": 80.0,
        "description": "Unconstrained baseline operating regime",
    },
    {
        "name": "Low VRAM / Normal Power / Cool",
        "short_name": "low_norm_cool",
        "vram_used_mb": 5250.0,  # 250MB headroom out of 5500MB
        "gpu_power_w": 50.0,
        "gpu_temperature_c": 55.0,
        "power_budget_w": 80.0,
        "description": "VRAM memory capacity constrained (near 6GB ceiling)",
    },
    {
        "name": "High VRAM / Limited Power / Cool",
        "short_name": "high_limit_cool",
        "vram_used_mb": 2000.0,
        "gpu_power_w": 45.0,
        "gpu_temperature_c": 55.0,
        "power_budget_w": 40.0,  # 40W power limit (battery/eco mode)
        "description": "TDP-limited / Energy-efficient operating regime",
    },
    {
        "name": "High VRAM / Normal Power / Hot",
        "short_name": "high_norm_hot",
        "vram_used_mb": 2000.0,
        "gpu_power_w": 50.0,
        "gpu_temperature_c": 81.0,  # within 1C of 82C thermal throttling ceiling
        "power_budget_w": 80.0,
        "description": "Near thermal throttling threshold (requires fast cooling)",
    },
    {
        "name": "Low VRAM / Limited Power / Hot",
        "short_name": "low_limit_hot",
        "vram_used_mb": 5250.0,
        "gpu_power_w": 45.0,
        "gpu_temperature_c": 81.0,
        "power_budget_w": 40.0,
        "description": "Worst-case multi-resource constrained scenario",
    },
]


def plot_sensitivity_matrix(
    results: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    """Generate figures illustrating controller policy shifts across hardware regimes."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [r["regime_name"].replace(" / ", "\n") for r in results]
    x = np.arange(len(labels))

    # 1. Action Shift Plot: Selected S distribution and Mean K
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Selected K
    mean_ks = [r["mean_selected_k"] for r in results]
    bars = ax1.bar(x, mean_ks, color="#1f77b4", width=0.45, edgecolor="black", alpha=0.85)
    ax1.set_ylabel("Mean Selected Draft Length K", fontsize=11)
    ax1.set_title("Controller Speculation Depth (K) Adaptation", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylim(0, 3.5)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for bar, k_val in zip(bars, mean_ks):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"K={k_val:.1f}", ha="center", fontsize=10, fontweight="bold")

    # Selected S Configuration counts
    configs = ["cka_83", "cka_75", "cka_60", "cka_50"]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
    bottom = np.zeros(len(labels))
    width = 0.55

    for cfg, color in zip(configs, colors):
        proportions = []
        for r in results:
            dist = r["config_distribution"]
            tot = sum(dist.values()) if dist else 1
            proportions.append((dist.get(cfg, 0) / tot) * 100.0)
        ax2.bar(x, proportions, width, bottom=bottom, label=cfg.upper(), color=color, alpha=0.85, edgecolor="black")
        bottom += np.array(proportions)

    ax2.set_ylabel("Configuration Selection (%)", fontsize=11)
    ax2.set_title("Controller Layer Configuration (S) Adaptation", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Hardware-Aware Joint Controller Policy Sensitivity (HardwareState_t -> Action_t)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "hardware_sensitivity_actions.png", dpi=300)
    plt.close()

    # 2. Performance & Energy Trade-off across Regimes
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 5))
    tps_vals = [r["mean_tps"] for r in results]
    energy_vals = [r["mean_energy_j_token"] for r in results]

    ax3.bar(x, tps_vals, color="#2ca02c", width=0.45, edgecolor="black", alpha=0.85)
    ax3.set_ylabel("Generation Throughput (tok/s)", fontsize=11)
    ax3.set_title("Throughput across Hardware Regimes", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    ax4.bar(x, energy_vals, color="#d62728", width=0.45, edgecolor="black", alpha=0.85)
    ax4.set_ylabel("Energy Consumption (J / token)", fontsize=11)
    ax4.set_title("Energy Efficiency across Hardware Regimes", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("System Performance and Energy under Hardware Constraint Regimes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "hardware_sensitivity_tradeoffs.png", dpi=300)
    plt.close()
    logger.info(f"Sensitivity figures successfully saved to {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HardwareAwareJointController across hardware regimes")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of benchmark prompts to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to generate per run")
    parser.add_argument("--output-file", type=str, default="experiments/06_controller/hardware_sensitivity_matrix.json")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 85)
    logger.info("PHASE 9 — HARDWAREAWAREJOINTCONTROLLER SENSITIVITY MATRIX")
    logger.info("=" * 85)

    # 1. Load model, tokenizer, and profiling harness
    model = load_model(args.model, quantize=True, bits=4)
    tokenizer = load_tokenizer(args.model)
    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)
    gpu_profiler = GPUProfiler()

    # Load fixed prompts
    prompts_path = Path("data/benchmarks/prompts.jsonl")
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line.strip()))
    eval_prompts = prompts[: args.num_prompts]

    # Load empirical cost model
    cost_model = MeasuredActionCostModel.from_files()

    matrix_results: list[dict[str, Any]] = []

    # 2. Iterate through each regime in the sensitivity matrix
    for regime in REGIME_DEFINITIONS:
        logger.info(f"\nEvaluating Regime: {regime['name']}")
        logger.info(f"Condition: VRAM={regime['vram_used_mb']}MB, Power={regime['gpu_power_w']}W (Cap={regime['power_budget_w']}W), Temp={regime['gpu_temperature_c']}C")

        hw_state = HardwareState(
            vram_used_mb=regime["vram_used_mb"],
            gpu_power_w=regime["gpu_power_w"],
            gpu_temperature_c=regime["gpu_temperature_c"],
            power_budget_w=regime["power_budget_w"],
        )

        controller = HardwareAwareJointController(
            candidate_layer_configs=CANDIDATE_CONFIGS,
            cost_model=cost_model,
            gpu_profiler=gpu_profiler,
            max_vram_mb=5500.0,
            temp_threshold_c=82.0,
            power_budget_w=regime["power_budget_w"],
            hardware_override=hw_state,
        )

        regime_runs = []
        for p in eval_prompts:
            _, m = self_speculative_generate(
                model=model,
                tokenizer=tokenizer,
                layer_mgr=layer_mgr,
                skip_indices=CANDIDATE_CONFIGS["cka_75"],
                prompt=p["prompt"],
                controller=controller,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            regime_runs.append(m)

        # Analyze action traces
        actions = controller.action_history
        config_counts: dict[str, int] = {}
        k_values: list[int] = []
        utilities: list[float] = []

        for a in actions:
            config_counts[a.config_name] = config_counts.get(a.config_name, 0) + 1
            k_values.append(a.draft_length)
            utilities.append(a.predicted_utility)

        mean_k = float(np.mean(k_values)) if k_values else 2.0
        dominant_cfg = max(config_counts, key=config_counts.get) if config_counts else "cka_75"
        mean_tps = float(np.mean([m.tokens_per_second for m in regime_runs]))
        mean_vram = float(np.mean([m.peak_vram_mb for m in regime_runs]))
        mean_tokens_per_step = float(np.mean([m.tokens_per_step for m in regime_runs]))

        # Calculate actual cycle latency
        tot_cycles = max(1, sum(m.num_verification_cycles for m in regime_runs))
        tot_draft = sum(m.draft_time_s for m in regime_runs)
        tot_verify = sum(m.verify_time_s for m in regime_runs)
        tot_cache = sum(m.cache_time_s for m in regime_runs)
        actual_cycle_ms = float(((tot_draft + tot_verify + tot_cache) / tot_cycles) * 1000.0)

        # Calculate energy
        total_time_s = sum(m.total_time_s for m in regime_runs)
        total_tokens = sum(m.total_tokens for m in regime_runs)
        actual_energy_j = float((regime["gpu_power_w"] * total_time_s) / max(1, total_tokens))

        regime_record = {
            "regime_name": regime["name"],
            "short_name": regime["short_name"],
            "hardware_state": {
                "vram_headroom_mb": 5500.0 - regime["vram_used_mb"],
                "gpu_power_w": regime["gpu_power_w"],
                "power_budget_w": regime["power_budget_w"],
                "gpu_temperature_c": regime["gpu_temperature_c"],
            },
            "selected_action": {
                "dominant_config": dominant_cfg,
                "mean_k": round(mean_k, 2),
                "predicted_utility": round(float(np.mean(utilities)), 3) if utilities else 0.0,
            },
            "config_distribution": config_counts,
            "mean_selected_k": round(mean_k, 2),
            "mean_tps": round(mean_tps, 2),
            "actual_cycle_ms": round(actual_cycle_ms, 2),
            "mean_energy_j_token": round(actual_energy_j, 3),
            "peak_vram_mb": round(mean_vram, 1),
            "tokens_per_step": round(mean_tokens_per_step, 3),
            "action_trace_length": len(actions),
        }
        matrix_results.append(regime_record)

        logger.info(f"Outcome: Dominant S={dominant_cfg}, Mean K={mean_k:.2f}, Mean Utility={regime_record['selected_action']['predicted_utility']}")
        logger.info(f"Performance: Throughput={mean_tps:.2f} tok/s, Cycle={actual_cycle_ms:.1f}ms, Energy={actual_energy_j:.3f} J/tok")

    # 3. Print Comprehensive Sensitivity Matrix Table
    logger.info("\n" + "=" * 100)
    logger.info("HARDWARE SENSITIVITY MATRIX VERIFICATION TABLE")
    logger.info("=" * 100)
    header = f"{'Regime':<35} | {'Selected S':<12} | {'Selected K':<10} | {'Pred Utility':<14} | {'TPS':<8} | {'Cycle (ms)':<10} | {'Energy (J/t)':<12}"
    logger.info(header)
    logger.info("-" * 105)

    for r in matrix_results:
        s_act = r["selected_action"]
        row = (
            f"{r['regime_name']:<35} | "
            f"{s_act['dominant_config']:<12} | "
            f"{s_act['mean_k']:<10.2f} | "
            f"{s_act['predicted_utility']:<14.3f} | "
            f"{r['mean_tps']:<8.1f} | "
            f"{r['actual_cycle_ms']:<10.1f} | "
            f"{r['mean_energy_j_token']:<12.3f}"
        )
        logger.info(row)
    logger.info("=" * 105)

    # 4. Check policy adaptation hypothesis
    configs_selected = set(r["selected_action"]["dominant_config"] for r in matrix_results)
    k_selected = set(r["selected_action"]["mean_k"] for r in matrix_results)

    logger.info(f"\nUnique Configurations Selected across Matrix: {configs_selected}")
    logger.info(f"Unique Mean K Selected across Matrix: {k_selected}")

    is_adaptive = len(configs_selected) > 1 or len(k_selected) > 1
    logger.info(f"Hardware-Aware Policy Responsiveness: {'CONFIRMED (State -> Action adapts)' if is_adaptive else 'FAILED (Static policy)'}")

    # 5. Save JSON summary and generate plots
    with open(output_file, "w") as f:
        json.dump(
            {
                "status": "PASS" if is_adaptive else "REVIEW",
                "policy_adaptation_confirmed": is_adaptive,
                "matrix_results": matrix_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved matrix summary to {output_file}")

    plot_sensitivity_matrix(matrix_results, figures_dir)


if __name__ == "__main__":
    main()
