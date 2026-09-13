"""Phase 4 — Self-Speculative Decoding Benchmark.

Benchmarks Zero-Additional-VRAM self-speculative decoding across draft lengths:
  K in {1, 2, 4, 6, 8}

Evaluates:
  1. Acceptance Rate: total accepted draft tokens / total draft tokens
  2. Tokens per Step: E[accepted tokens + 1 per verification cycle]
  3. Speedup vs Vanilla baseline: tok/s (spec) / tok/s (vanilla)
  4. Latency breakdown: T_draft vs T_verify
  5. Peak VRAM: confirming zero additional model VRAM
  6. Greedy Consistency: exactness check (Output_spec == Output_vanilla)

Usage:
    python scripts/benchmark_speculative.py
    python scripts/benchmark_speculative.py --k-values 1 2 4 6 8
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

from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.models.layer_manager import LayerManager
from zassd.decoding.speculative import self_speculative_generate, SpeculativeMetrics
from zassd.decoding.vanilla import vanilla_generate
from zassd.profiling.memory import get_vram_usage, reset_vram_stats
from zassd.utils.config import load_config
from zassd.utils.seed import set_seed
from zassd.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def generate_speculative_figures(
    summary_results: dict[str, dict],
    figures_dir: Path,
) -> None:
    """Generate speedup and acceptance plots for Phase 4."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    k_vals = []
    speedups = []
    accept_rates = []
    tokens_per_step = []
    tps_vals = []

    for key, data in sorted(summary_results.items(), key=lambda x: int(x[0].replace("K", ""))):
        k = int(key.replace("K", ""))
        k_vals.append(k)
        speedups.append(data["speedup_vs_vanilla"])
        accept_rates.append(data["acceptance_rate"] * 100)
        tokens_per_step.append(data["tokens_per_step"])
        tps_vals.append(data["tokens_per_second_mean"])

    # Figure 1: Speedup vs Draft Length K
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(k_vals, speedups, marker="o", linewidth=2.5, color="#1f77b4", label="Self-Speculative Speedup")
    plt.axhline(y=1.0, color="gray", linestyle="--", label="Vanilla Baseline (1.0x)")
    plt.title("Inference Speedup vs. Draft Speculation Length K", fontsize=12)
    plt.xlabel("Draft Length K", fontsize=10)
    plt.ylabel("Speedup vs. Vanilla Baseline", fontsize=10)
    plt.xticks(k_vals)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    for x, y in zip(k_vals, speedups):
        plt.annotate(f"{y:.2f}x", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "speculative_speedup_vs_k.png")
    plt.close()

    # Figure 2: Acceptance Rate & Tokens per Step vs K
    fig, ax1 = plt.subplots(figsize=(7.5, 5), dpi=300)
    color1 = "#2ca02c"
    ax1.set_xlabel("Draft Length K", fontsize=10)
    ax1.set_ylabel("Acceptance Rate (%)", color=color1, fontsize=10)
    line1 = ax1.plot(k_vals, accept_rates, marker="s", color=color1, linewidth=2, label="Acceptance Rate (%)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("Tokens per Verification Cycle", color=color2, fontsize=10)
    line2 = ax2.plot(k_vals, tokens_per_step, marker="^", color=color2, linewidth=2, linestyle="-.", label="Tokens/Step")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")
    plt.title("Speculation Acceptance and Efficiency vs. Draft Length K", fontsize=12)
    plt.xticks(k_vals)
    plt.tight_layout()
    plt.savefig(figures_dir / "speculative_acceptance_vs_k.png")
    plt.close()
    logger.info("Saved Phase 4 figures to results/figures/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Speculative Decoding Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="experiments/04_self_speculative")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=str(output_dir / "speculative_benchmark.log"))
    set_seed(42)

    logger.info("=" * 70)
    logger.info("PHASE 4 — SELF-SPECULATIVE DECODING BENCHMARK")
    logger.info("=" * 70)
    logger.info(f"Target K values: {args.k_values}")
    logger.info(f"Measured runs per K: {args.runs}")

    # 1. Load Model
    model = load_model(args.model, quantize=True, bits=args.bits)
    tokenizer = load_tokenizer(args.model)
    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)
    num_layers = adapter.num_layers

    # 2. Load CKA Selected Layers (from Phase 3 cka_75)
    cka_file = Path("experiments/03_cka/benchmark_results.json")
    if cka_file.exists():
        with open(cka_file) as f:
            cka_data = json.load(f)
        skip_indices = cka_data.get("cka_75", {}).get("skipped_indices", [])
        logger.info(f"Loaded CKA-selected skip layers from Phase 3: {skip_indices} ({len(skip_indices)} skipped)")
    else:
        # Fallback: skip 9 middle redundant layers
        skip_indices = [3, 5, 7, 9, 11, 13, 16, 18, 21]
        logger.warning(f"Using default skip indices: {skip_indices}")

    # 3. Load fixed benchmark prompts
    prompts_path = Path("data/benchmarks/prompts.jsonl")
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line.strip()))
    eval_prompts = prompts[: args.runs]
    logger.info(f"Loaded {len(eval_prompts)} prompts for evaluation")

    # 4. Measure vanilla baseline on these exact prompts (for speedup & exact greedy consistency)
    logger.info("\n--- Establishing Vanilla Baseline on Prompts ---")
    vanilla_outputs = {}
    vanilla_tps_list = []
    vanilla_vram = 0.0

    for idx, p in enumerate(eval_prompts):
        text, v_metrics = vanilla_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=p["prompt"],
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        vanilla_outputs[p["id"]] = text
        vanilla_tps_list.append(v_metrics.tokens_per_second)
        vanilla_vram = max(vanilla_vram, v_metrics.peak_vram_mb)

    vanilla_mean_tps = float(np.mean(vanilla_tps_list))
    logger.info(f"Vanilla Baseline Speed: {vanilla_mean_tps:.2f} tok/s, Peak VRAM: {vanilla_vram:.0f} MB")

    # 5. Benchmark Self-Speculative across K values
    summary_results: dict[str, dict] = {}

    for k in args.k_values:
        k_key = f"K{k}"
        k_dir = output_dir / k_key
        k_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating Self-Speculative Decoding with K = {k}")
        logger.info(f"{'='*60}")

        k_runs = []
        exact_matches = 0
        total_tokens_generated = 0
        total_draft_tokens = 0
        total_accepted_tokens = 0
        total_cycles = 0

        for run_idx, p in enumerate(eval_prompts):
            output_text, s_metrics = self_speculative_generate(
                model=model,
                tokenizer=tokenizer,
                layer_mgr=layer_mgr,
                skip_indices=skip_indices,
                prompt=p["prompt"],
                k=k,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )

            # Check exact greedy consistency
            ref_text = vanilla_outputs[p["id"]]
            is_match = (output_text == ref_text)
            if is_match:
                exact_matches += 1

            total_tokens_generated += s_metrics.total_tokens
            total_draft_tokens += s_metrics.total_draft_tokens
            total_accepted_tokens += s_metrics.total_accepted_tokens
            total_cycles += s_metrics.num_verification_cycles

            speedup = (
                s_metrics.tokens_per_second / vanilla_mean_tps
                if vanilla_mean_tps > 0
                else 1.0
            )
            s_metrics.speedup_vs_vanilla = speedup

            run_record = {
                "run_idx": run_idx,
                "prompt_id": p["id"],
                "total_tokens": s_metrics.total_tokens,
                "draft_tokens": s_metrics.total_draft_tokens,
                "accepted_tokens": s_metrics.total_accepted_tokens,
                "acceptance_rate": s_metrics.acceptance_rate,
                "verification_cycles": s_metrics.num_verification_cycles,
                "tokens_per_step": s_metrics.tokens_per_step,
                "total_time_s": s_metrics.total_time_s,
                "draft_time_s": s_metrics.draft_time_s,
                "verify_time_s": s_metrics.verify_time_s,
                "tokens_per_second": s_metrics.tokens_per_second,
                "speedup_vs_vanilla": speedup,
                "peak_vram_mb": s_metrics.peak_vram_mb,
                "greedy_exact_match": is_match,
            }
            k_runs.append(run_record)

            logger.info(
                f"  Run {run_idx + 1:>2}/{len(eval_prompts)}: "
                f"{s_metrics.tokens_per_second:.1f} tok/s ({speedup:.2f}x) | "
                f"Accept: {s_metrics.acceptance_rate:.1%} ({s_metrics.total_accepted_tokens}/{s_metrics.total_draft_tokens}) | "
                f"Tok/Step: {s_metrics.tokens_per_step:.2f} | "
                f"Match: {'✓' if is_match else '✗'}"
            )

        # Aggregate metrics for this K
        tps_arr = np.array([r["tokens_per_second"] for r in k_runs])
        speedup_arr = np.array([r["speedup_vs_vanilla"] for r in k_runs])
        accept_arr = np.array([r["acceptance_rate"] for r in k_runs])
        tps_step_arr = np.array([r["tokens_per_step"] for r in k_runs])
        vram_arr = np.array([r["peak_vram_mb"] for r in k_runs])
        draft_time_arr = np.array([r["draft_time_s"] for r in k_runs])
        verify_time_arr = np.array([r["verify_time_s"] for r in k_runs])

        summary_k = {
            "k": k,
            "num_runs": len(k_runs),
            "exact_match_rate": exact_matches / len(k_runs),
            "tokens_per_second_mean": float(tps_arr.mean()),
            "tokens_per_second_std": float(tps_arr.std()),
            "speedup_vs_vanilla": float(speedup_arr.mean()),
            "speedup_vs_vanilla_std": float(speedup_arr.std()),
            "acceptance_rate": float(accept_arr.mean()),
            "acceptance_rate_std": float(accept_arr.std()),
            "tokens_per_step": float(tps_step_arr.mean()),
            "tokens_per_step_std": float(tps_step_arr.std()),
            "peak_vram_mb_mean": float(vram_arr.mean()),
            "draft_time_mean_s": float(draft_time_arr.mean()),
            "verify_time_mean_s": float(verify_time_arr.mean()),
            "draft_verify_time_ratio": float((draft_time_arr / np.maximum(verify_time_arr, 1e-6)).mean()),
        }

        with open(k_dir / "results.json", "w") as f:
            json.dump({"summary": summary_k, "per_run": k_runs}, f, indent=2)

        summary_results[k_key] = summary_k

        logger.info(f"\n--- Summary for K = {k} ---")
        logger.info(f"  Throughput:      {summary_k['tokens_per_second_mean']:.2f} ± {summary_k['tokens_per_second_std']:.2f} tok/s")
        logger.info(f"  Speedup:         {summary_k['speedup_vs_vanilla']:.2f}x vs Vanilla")
        logger.info(f"  Acceptance Rate: {summary_k['acceptance_rate']:.1%}")
        logger.info(f"  Tokens / Step:   {summary_k['tokens_per_step']:.2f} tokens/cycle")
        logger.info(f"  Greedy Match:    {summary_k['exact_match_rate']:.1%} exact consistency")
        logger.info(f"  Peak VRAM:       {summary_k['peak_vram_mb_mean']:.0f} MB")

        torch.cuda.empty_cache()
        gc.collect()

    # 6. Save Overall Summary
    final_summary = {
        "model": args.model,
        "draft_strategy": "cka_75",
        "num_layers_skipped": len(skip_indices),
        "vanilla_baseline_tok_s": vanilla_mean_tps,
        "k_comparison": summary_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    # 7. Generate Figures
    generate_speculative_figures(summary_results, figures_dir)

    # 8. Print Clean Table
    logger.info("\n" + "=" * 95)
    logger.info("PHASE 4 — SELF-SPECULATIVE DECODING RESULTS SUMMARY")
    logger.info("=" * 95)
    logger.info(
        f"{'K':>4} | {'tok/s':>10} | {'Speedup':>9} | {'Accept Rate':>12} | "
        f"{'Tokens/Step':>12} | {'Exact Match':>12} | {'Peak VRAM':>10}"
    )
    logger.info("-" * 95)
    logger.info(
        f"{'Base':>4} | {vanilla_mean_tps:>10.1f} | {'1.00x':>9} | {'N/A':>12} | "
        f"{'1.00':>12} | {'100.0%':>12} | {vanilla_vram:>9.0f}MB"
    )
    for k_key, s in summary_results.items():
        logger.info(
            f"{s['k']:>4} | "
            f"{s['tokens_per_second_mean']:>10.1f} | "
            f"{s['speedup_vs_vanilla']:>8.2f}x | "
            f"{s['acceptance_rate']:>11.1%} | "
            f"{s['tokens_per_step']:>12.2f} | "
            f"{s['exact_match_rate']:>11.1%} | "
            f"{s['peak_vram_mb_mean']:>9.0f}MB"
        )
    logger.info("=" * 95)
    logger.info(f"All Phase 4 results saved to {output_dir}")


if __name__ == "__main__":
    main()
