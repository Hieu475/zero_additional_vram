"""Phase 10 — Competitive Baselines Benchmark Harness (Việc 4).

Unified, standardized benchmark harness evaluating 7 decoding methods
under identical experimental conditions on the local RTX 4050 Laptop GPU:
  1. Vanilla Autoregressive Baseline (full 36 layers)
  2. Static Layer Skip (even skipping, K=2)
  3. CKA Fixed Speculative (CKA-75, fixed K=2)
  4. Adaptive K Speculative (CKA-75, entropy-driven dynamic K)
  5. KnapSpec (Cha et al., ICML 2026: 0/1 Knapsack layer selection, K=2)
  6. SpecBound (Wen & Feng, ACL 2026: Confidence-bounded adaptive K)
  7. Hardware-Aware Controller (ZASSD: Joint (S_t, K_t) hardware-aware controller)

Protocol:
  - Same GPU: NVIDIA GeForce RTX 4050 Laptop GPU 6GB
  - Same Model: Qwen/Qwen2.5-3B-Instruct (4-bit NF4)
  - Same Prompts: data/benchmarks/prompts.jsonl (10 prompts)
  - Same Generation Mode: Greedy decoding (temperature=0.0)
  - Same Token Budget: max_new_tokens = 32 per prompt
  - Same Warmup: 1 warmup run prior to timed evaluation
  - Same Measurement Code: time.perf_counter(), torch.cuda.synchronize(), pynvml energy
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from zassd.baselines.knapspec import KnapSpecController
from zassd.baselines.specbound import SpecBoundController
from zassd.controllers.adaptive_k import AdaptiveKController
from zassd.controllers.hardware_controller import HardwareAwareJointController
from zassd.decoding.speculative import self_speculative_generate
from zassd.decoding.vanilla import vanilla_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.profiling.gpu import GPUProfiler
from zassd.profiling.memory import reset_vram_stats
from zassd.utils.logging import setup_logging
from zassd.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Standard layer skip definitions for Qwen2.5-3B
STATIC_75_SKIPS = [1, 5, 9, 13, 17, 21, 25, 29, 33]       # 9 layers skipped, even intervals
CKA_75_SKIPS = [3, 4, 5, 6, 7, 10, 11, 12, 13]            # 9 layers skipped, CKA-selected
CANDIDATE_CONFIGS = {
    "cka_83": [4, 5, 6, 7, 12, 13],
    "cka_75": [3, 4, 5, 6, 7, 10, 11, 12, 13],
    "cka_60": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 21],
    "cka_50": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22],
}


def compute_exact_match(reference_text: str, candidate_text: str, tokenizer) -> float:
    """Compute token-level exact match percentage between reference and candidate."""
    ref_tokens = tokenizer.encode(reference_text, add_special_tokens=False)
    cand_tokens = tokenizer.encode(candidate_text, add_special_tokens=False)

    if not ref_tokens:
        return 1.0 if not cand_tokens else 0.0

    min_len = min(len(ref_tokens), len(cand_tokens))
    if min_len == 0:
        return 0.0

    matches = sum(1 for i in range(min_len) if ref_tokens[i] == cand_tokens[i])
    return float(matches / len(ref_tokens))


def plot_baseline_comparison(results: list[dict[str, Any]], figures_dir: Path) -> None:
    """Generate publication-ready comparison bar charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    names = [r["method_name"] for r in results]
    x = np.arange(len(names))
    width = 0.55

    # 1. Throughput & Speedup
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    tps_values = [r["tokens_per_second"] for r in results]
    speedups = [r["speedup_vs_vanilla"] for r in results]

    colors = ["#7f7f7f", "#aec7e8", "#1f77b4", "#17becf", "#9467bd", "#ff7f0e", "#2ca02c"]
    bars = ax1.bar(x, tps_values, width, color=colors, edgecolor="black", alpha=0.9)
    ax1.set_ylabel("Throughput (Tokens / Second)", fontsize=12)
    ax1.set_title("Competitive Baselines Benchmark on RTX 4050 Laptop GPU (Same Harness)", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15, ha="right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)

    for bar, tps, spd in zip(bars, tps_values, speedups):
        label = f"{tps:.1f} t/s\n({spd:.2f}x)"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, label, ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(figures_dir / "competitive_baselines_throughput.png", dpi=300)
    plt.close()

    # 2. Multi-Metric Radar / Trade-off
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(13, 5))
    em_values = [r["exact_match_pct"] for r in results]
    energy_values = [r["energy_j_token"] for r in results]

    ax2.bar(x, em_values, width, color=colors, edgecolor="black", alpha=0.9)
    ax2.set_ylabel("Exact Match Rate (%)", fontsize=11)
    ax2.set_title("Exact Match Fidelity vs Vanilla", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.set_ylim(0, 110)
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3.bar(x, energy_values, width, color=colors, edgecolor="black", alpha=0.9)
    ax3.set_ylabel("Energy (Joules / Token)", fontsize=11)
    ax3.set_title("Hardware Energy Efficiency (J / tok)", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Correctness and Energy Comparison Across Competitive Baselines", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "competitive_baselines_tradeoffs.png", dpi=300)
    plt.close()
    logger.info(f"Baseline comparison plots saved to {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Competitive Baselines Benchmark Harness on RTX 4050")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-prompts", type=int, default=10, help="Number of benchmark prompts")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to generate per prompt")
    parser.add_argument("--output-dir", type=str, default="experiments/10_baselines")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 90)
    logger.info("PHASE 10 — COMPETITIVE BASELINES BENCHMARK HARNESS (RTX 4050 LAPTOP)")
    logger.info("=" * 90)

    # 1. Load Model, Tokenizer, Adapters
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

    # Warmup
    logger.info("Running warmup pass...")
    _ = vanilla_generate(model, tokenizer, eval_prompts[0]["prompt"], max_new_tokens=8, temperature=0.0)
    torch.cuda.synchronize()

    # Initialize baselines
    cost_model = MeasuredActionCostModel.from_files()
    knapspec_ctrl = KnapSpecController(total_layers=36, budget_ratio=0.75, fixed_k=2)
    specbound_ctrl = SpecBoundController(skip_indices=CKA_75_SKIPS, k_min=1, k_max=4, initial_k=2)
    hw_ctrl = HardwareAwareJointController(candidate_layer_configs=CANDIDATE_CONFIGS, cost_model=cost_model)

    methods = [
        ("Vanilla", "vanilla", None, None, 1),
        ("Static Skip", "static", STATIC_75_SKIPS, None, 2),
        ("CKA Fixed", "cka_fixed", CKA_75_SKIPS, None, 2),
        ("Adaptive K", "adaptive_k", CKA_75_SKIPS, AdaptiveKController(k_min=1, k_max=4, initial_k=2), 2),
        ("KnapSpec (ICML'26)", "knapspec", knapspec_ctrl.skip_indices, knapspec_ctrl, 2),
        ("SpecBound (ACL'26)", "specbound", CKA_75_SKIPS, specbound_ctrl, 2),
        ("HW Controller (ZASSD)", "hw_controller", CANDIDATE_CONFIGS["cka_75"], hw_ctrl, 2),
    ]

    vanilla_outputs: dict[int, str] = {}
    benchmark_results: list[dict[str, Any]] = []

    # 2. Execute Benchmark for Each Method
    for method_name, m_type, skip_idxs, ctrl, default_k in methods:
        logger.info(f"\nEvaluating: {method_name}...")
        gc.collect()
        torch.cuda.empty_cache()
        reset_vram_stats()

        e_start = gpu_profiler.get_total_energy_mj()
        t_start = time.perf_counter()

        runs = []
        texts = []
        for p in eval_prompts:
            if m_type == "vanilla":
                text, m = vanilla_generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=p["prompt"],
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
                vanilla_outputs[p["id"]] = text
            else:
                text, m = self_speculative_generate(
                    model=model,
                    tokenizer=tokenizer,
                    layer_mgr=layer_mgr,
                    skip_indices=skip_idxs,
                    prompt=p["prompt"],
                    k=default_k,
                    controller=ctrl,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
            runs.append(m)
            texts.append(text)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        e_end = gpu_profiler.get_total_energy_mj()

        # Compute Metrics
        tot_tokens = sum(m.total_tokens for m in runs)
        tps = tot_tokens / max(1e-3, elapsed)
        mean_vram = float(np.mean([m.peak_vram_mb for m in runs]))

        # Exact match vs vanilla
        em_scores = []
        for p, t in zip(eval_prompts, texts):
            ref = vanilla_outputs[p["id"]]
            em = compute_exact_match(ref, t, tokenizer)
            em_scores.append(em)
        exact_match_pct = float(np.mean(em_scores) * 100.0)

        # Acceptance rate
        if m_type == "vanilla":
            acceptance_rate = None
        else:
            acceptance_rate = float(np.mean([m.acceptance_rate for m in runs]) * 100.0)

        # Energy per token
        if e_start is not None and e_end is not None and e_end >= e_start:
            energy_j_tok = ((e_end - e_start) / 1000.0) / max(1, tot_tokens)
        else:
            p_watts = gpu_profiler.get_power_usage()
            energy_j_tok = (p_watts * elapsed) / max(1, tot_tokens)

        # Speedup calculated relative to Vanilla
        if benchmark_results:
            vanilla_tps = benchmark_results[0]["tokens_per_second"]
            speedup = tps / vanilla_tps
        else:
            speedup = 1.0

        rec = {
            "method_name": method_name,
            "method_type": m_type,
            "tokens_per_second": round(tps, 2),
            "speedup_vs_vanilla": round(speedup, 3),
            "acceptance_rate_pct": round(acceptance_rate, 1) if acceptance_rate is not None else "N/A",
            "exact_match_pct": round(exact_match_pct, 1),
            "peak_vram_mb": round(mean_vram, 1),
            "energy_j_token": round(energy_j_tok, 3),
            "total_tokens": tot_tokens,
            "elapsed_s": round(elapsed, 3),
        }
        benchmark_results.append(rec)

        logger.info(
            f"Finished {method_name}: {tps:.2f} tok/s ({speedup:.2f}x), "
            f"Acceptance={rec['acceptance_rate_pct']}, ExactMatch={exact_match_pct:.1f}%, "
            f"VRAM={mean_vram:.1f}MB, Energy={energy_j_tok:.3f} J/tok"
        )

    # 3. Print Final Publication Comparison Table
    logger.info("\n" + "=" * 105)
    logger.info("FINAL COMPETITIVE BASELINES COMPARISON TABLE (SAME HARNESS, RTX 4050 LAPTOP)")
    logger.info("=" * 105)
    header = f"{'Method':<25} | {'tok/s':<8} | {'Speedup':<8} | {'Acceptance':<12} | {'Exact Match':<12} | {'Peak VRAM':<11} | {'J/token':<8}"
    logger.info(header)
    logger.info("-" * 105)

    for r in benchmark_results:
        acc_str = f"{r['acceptance_rate_pct']}%" if r['acceptance_rate_pct'] != "N/A" else "N/A"
        row = (
            f"{r['method_name']:<25} | "
            f"{r['tokens_per_second']:<8.2f} | "
            f"{r['speedup_vs_vanilla']:<8.2f}x | "
            f"{acc_str:<12} | "
            f"{r['exact_match_pct']:<11.1f}% | "
            f"{r['peak_vram_mb']:<9.1f}MB | "
            f"{r['energy_j_token']:<8.3f}"
        )
        logger.info(row)
    logger.info("=" * 105)

    # 4. Save JSON summary and plots
    summary_file = output_dir / "competitive_baselines_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "status": "PASS",
                "hardware": "NVIDIA GeForce RTX 4050 Laptop GPU (6GB)",
                "model": args.model,
                "quantization": "4-bit (bitsandbytes NF4)",
                "num_prompts": args.num_prompts,
                "max_new_tokens": args.max_new_tokens,
                "results": benchmark_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Saved benchmark summary to {summary_file}")

    plot_baseline_comparison(benchmark_results, figures_dir)


if __name__ == "__main__":
    main()
