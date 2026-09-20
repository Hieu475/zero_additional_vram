"""Phase 6 — Correctness + Efficient Core: Self-Speculative Benchmark & Cost Decomposition.

Benchmarks Zero-Additional-VRAM self-speculative decoding across draft lengths:
  K in {1, 2, 4, 6, 8} with canonical Target KV and Ephemeral Draft KV.

Evaluates:
  1. Acceptance Rate: total accepted draft tokens / total draft tokens
  2. Tokens per Step: E[tokens emitted per verification cycle]
  3. Speedup vs Vanilla baseline: tok/s (spec) / tok/s (vanilla)
  4. Latency decomposition:
       T_total = T_draft + T_verify + T_cache + T_other
  5. Peak VRAM: confirming zero additional model VRAM
  6. Greedy Consistency: exactness check (Output_spec == Output_vanilla)

Usage:
    python scripts/benchmark_speculative.py --k-values 1 2 4 6 8 --runs 15
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

from zassd.decoding.speculative import self_speculative_generate, SpeculativeMetrics
from zassd.decoding.vanilla import vanilla_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.memory import get_vram_usage, reset_vram_stats
from zassd.utils.logging import setup_logging
from zassd.utils.seed import set_seed

logger = logging.getLogger(__name__)


def generate_speculative_figures(
    summary_results: dict[str, dict],
    vanilla_ms_per_token: float,
    figures_dir: Path,
) -> None:
    """Generate speedup, acceptance, and latency decomposition plots for Phase 6."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    k_vals = []
    speedups = []
    accept_rates = []
    tokens_per_step = []
    tps_vals = []
    draft_ms_per_tok = []
    verify_ms_per_tok = []
    cache_ms_per_tok = []
    other_ms_per_tok = []

    for key, data in sorted(summary_results.items(), key=lambda x: int(x[0].replace("K", ""))):
        k = int(key.replace("K", ""))
        k_vals.append(k)
        speedups.append(data["speedup_vs_vanilla"])
        accept_rates.append(data["acceptance_rate"] * 100)
        tokens_per_step.append(data["tokens_per_step"])
        tps_vals.append(data["tokens_per_second_mean"])

        # Per-token latency decomposition (ms)
        tot_tok = max(1, data["total_tokens_sum"])
        draft_ms_per_tok.append((data["draft_time_total_s"] / tot_tok) * 1000)
        verify_ms_per_tok.append((data["verify_time_total_s"] / tot_tok) * 1000)
        cache_ms_per_tok.append((data["cache_time_total_s"] / tot_tok) * 1000)
        other_ms_per_tok.append((data["other_time_total_s"] / tot_tok) * 1000)

    # Figure 1: Speedup vs Draft Length K
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot(k_vals, speedups, marker="o", linewidth=2.5, color="#1f77b4", label="Cache-Aware Self-Speculative")
    plt.axhline(y=1.0, color="gray", linestyle="--", label="Vanilla Baseline (1.0x)")
    plt.title("Inference Speedup vs. Draft Speculation Length K", fontsize=12, fontweight="bold")
    plt.xlabel("Draft Speculation Length K", fontsize=10)
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
    plt.title("Speculation Acceptance and Efficiency vs. Draft Length K", fontsize=12, fontweight="bold")
    plt.xticks(k_vals)
    plt.tight_layout()
    plt.savefig(figures_dir / "speculative_acceptance_vs_k.png")
    plt.close()

    # Figure 3: Latency Decomposition (Stacked Bar Chart)
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    indices = np.arange(len(k_vals) + 1)
    bar_width = 0.55

    labels = ["Vanilla"] + [f"K={k}" for k in k_vals]
    vanilla_bars = [vanilla_ms_per_token] + [0.0] * len(k_vals)
    d_bars = [0.0] + draft_ms_per_tok
    v_bars = [0.0] + verify_ms_per_tok
    c_bars = [0.0] + cache_ms_per_tok
    o_bars = [0.0] + other_ms_per_tok

    p_vanilla = ax.bar(indices, vanilla_bars, bar_width, label="Vanilla Decode", color="#7f7f7f")
    p_draft = ax.bar(indices, d_bars, bar_width, bottom=vanilla_bars, label="Draft Forward (T_draft)", color="#1f77b4")
    bottom_v = np.array(vanilla_bars) + np.array(d_bars)
    p_verify = ax.bar(indices, v_bars, bar_width, bottom=bottom_v, label="Target Verify (T_verify)", color="#ff7f0e")
    bottom_c = bottom_v + np.array(v_bars)
    p_cache = ax.bar(indices, c_bars, bar_width, bottom=bottom_c, label="Cache Ops (T_cache)", color="#2ca02c")
    bottom_o = bottom_c + np.array(c_bars)
    p_other = ax.bar(indices, o_bars, bar_width, bottom=bottom_o, label="Overhead (T_other)", color="#d62728")

    ax.set_ylabel("Latency per Token (ms)", fontsize=10)
    ax.set_title("Latency Decomposition: T_total = T_draft + T_verify + T_cache + T_other", fontsize=12, fontweight="bold")
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Annotate total ms on top of bars
    totals = [vanilla_ms_per_token] + [
        d + v + c + o for d, v, c, o in zip(draft_ms_per_tok, verify_ms_per_tok, cache_ms_per_tok, other_ms_per_tok)
    ]
    for idx, total_ms in enumerate(totals):
        ax.annotate(f"{total_ms:.1f}ms", (idx, total_ms), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(figures_dir / "speculative_latency_decomposition.png")
    plt.close()
    logger.info("Saved Phase 6 figures to results/figures/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Speculative Decoding Benchmark & Decomposition")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--config", type=str, default="cka_83", help="Layer config name (e.g. cka_83, cka_75, cka_90)")
    parser.add_argument("--skip-indices", type=int, nargs="*", default=None, help="Explicit list of layer indices to skip")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="experiments/07_k_sweep_cka83")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=str(output_dir / "speculative_benchmark.log"))
    set_seed(42)

    logger.info("=" * 75)
    logger.info("PHASE 7.7 — PARETO SPECULATIVE DECODING BENCHMARK")
    logger.info("=" * 75)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Draft lengths K: {args.k_values}")
    logger.info(f"Runs per K: {args.runs}")
    logger.info(f"Max new tokens: {args.max_new_tokens}")

    # 1. Load Model & Tokenizer
    model = load_model(args.model, quantize=True, bits=args.bits)
    tokenizer = load_tokenizer(args.model)
    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)

    # 2. Determine skip indices
    CONFIG_MAP = {
        "cka_50": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22],
        "cka_60": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 21],
        "cka_67": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16],
        "cka_75": [3, 4, 5, 6, 7, 10, 11, 12, 13],
        "cka_83": [4, 5, 6, 7, 12, 13],
        "cka_90": [4, 5, 6, 7],
    }
    if args.skip_indices is not None and len(args.skip_indices) > 0:
        skip_indices = sorted(args.skip_indices)
        config_name = "custom"
    else:
        config_name = args.config
        pareto_file = Path("experiments/07_pareto/pareto_results.json")
        if pareto_file.exists():
            with open(pareto_file) as f:
                p_data = json.load(f)
            skip_indices = p_data.get(config_name, {}).get("skip_indices", CONFIG_MAP.get(config_name, [4, 5, 6, 7, 12, 13]))
        else:
            skip_indices = CONFIG_MAP.get(config_name, [4, 5, 6, 7, 12, 13])

    logger.info(f"Using skip layers for {config_name}: {skip_indices} ({len(skip_indices)} skipped, {adapter.num_layers - len(skip_indices)} kept)")

    # 3. Load fixed benchmark prompts
    prompts_path = Path("data/benchmarks/prompts.jsonl")
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line.strip()))
    eval_prompts = prompts[: args.runs]
    logger.info(f"Loaded {len(eval_prompts)} prompts for evaluation")

    # 4. Measure Vanilla Baseline on these exact prompts
    logger.info("\n--- Establishing Vanilla Baseline on Benchmark Prompts ---")
    vanilla_outputs = {}
    vanilla_tps_list = []
    vanilla_tpot_list = []
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
        if v_metrics.tpot_ms > 0:
            vanilla_tpot_list.append(v_metrics.tpot_ms)
        vanilla_vram = max(vanilla_vram, v_metrics.peak_vram_mb)

    vanilla_mean_tps = float(np.mean(vanilla_tps_list))
    vanilla_ms_per_tok = float(np.mean(vanilla_tpot_list)) if vanilla_tpot_list else (1000.0 / vanilla_mean_tps)
    logger.info(f"Vanilla Baseline Speed: {vanilla_mean_tps:.2f} tok/s ({vanilla_ms_per_tok:.1f} ms/tok), Peak VRAM: {vanilla_vram:.0f} MB")

    # 5. Benchmark Self-Speculative Decoding across K in {1, 2, 4, 6, 8}
    summary_results: dict[str, dict] = {}

    for k in args.k_values:
        k_key = f"K{k}"
        k_dir = output_dir / k_key
        k_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'='*65}")
        logger.info(f"Evaluating Self-Speculative Decoding with K = {k}")
        logger.info(f"{'='*65}")

        k_runs = []
        exact_matches = 0

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

            ref_text = vanilla_outputs[p["id"]]
            is_match = (output_text == ref_text)
            if is_match:
                exact_matches += 1

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
                "cache_time_s": s_metrics.cache_time_s,
                "other_time_s": s_metrics.other_time_s,
                "tokens_per_second": s_metrics.tokens_per_second,
                "speedup_vs_vanilla": speedup,
                "peak_vram_mb": s_metrics.peak_vram_mb,
                "greedy_exact_match": is_match,
            }
            k_runs.append(run_record)

            logger.info(
                f"  Run {run_idx + 1:>2}/{len(eval_prompts)}: "
                f"{s_metrics.tokens_per_second:.1f} tok/s ({speedup:.2f}x) | "
                f"Accept: {s_metrics.acceptance_rate:.1%} | "
                f"Tok/Step: {s_metrics.tokens_per_step:.2f} | "
                f"Draft: {s_metrics.draft_time_s*1000:.0f}ms, Verify: {s_metrics.verify_time_s*1000:.0f}ms, Cache: {s_metrics.cache_time_s*1000:.0f}ms | "
                f"Match: {'✓' if is_match else '✗'}"
            )

        # Aggregated stats for this K
        tps_arr = np.array([r["tokens_per_second"] for r in k_runs])
        speedup_arr = np.array([r["speedup_vs_vanilla"] for r in k_runs])
        accept_arr = np.array([r["acceptance_rate"] for r in k_runs])
        tps_step_arr = np.array([r["tokens_per_step"] for r in k_runs])
        vram_arr = np.array([r["peak_vram_mb"] for r in k_runs])

        total_tok_sum = sum(r["total_tokens"] for r in k_runs)
        draft_time_sum = sum(r["draft_time_s"] for r in k_runs)
        verify_time_sum = sum(r["verify_time_s"] for r in k_runs)
        cache_time_sum = sum(r["cache_time_s"] for r in k_runs)
        other_time_sum = sum(r["other_time_s"] for r in k_runs)
        total_time_sum = sum(r["total_time_s"] for r in k_runs)

        tot_cycles = max(1, sum(r["verification_cycles"] for r in k_runs))
        ms_per_cycle = (total_time_sum / tot_cycles) * 1000
        draft_ms_cycle = (draft_time_sum / tot_cycles) * 1000
        verify_ms_cycle = (verify_time_sum / tot_cycles) * 1000
        cache_ms_cycle = (cache_time_sum / tot_cycles) * 1000
        other_ms_cycle = (other_time_sum / tot_cycles) * 1000

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
            "total_tokens_sum": total_tok_sum,
            "draft_time_total_s": draft_time_sum,
            "verify_time_total_s": verify_time_sum,
            "cache_time_total_s": cache_time_sum,
            "other_time_total_s": other_time_sum,
            "total_time_total_s": total_time_sum,
            # Per-cycle latency breakdown in ms
            "cycle_latency_ms": ms_per_cycle,
            "draft_latency_ms": draft_ms_cycle,
            "verify_latency_ms": verify_ms_cycle,
            "cache_latency_ms": cache_ms_cycle,
            "other_latency_ms": other_ms_cycle,
            # Percentage breakdown
            "draft_pct": (draft_time_sum / total_time_sum) * 100 if total_time_sum > 0 else 0,
            "verify_pct": (verify_time_sum / total_time_sum) * 100 if total_time_sum > 0 else 0,
            "cache_pct": (cache_time_sum / total_time_sum) * 100 if total_time_sum > 0 else 0,
            "other_pct": (other_time_sum / total_time_sum) * 100 if total_time_sum > 0 else 0,
        }

        with open(k_dir / "results.json", "w") as f:
            json.dump({"summary": summary_k, "per_run": k_runs}, f, indent=2)

        summary_results[k_key] = summary_k

        logger.info(f"\n--- Summary for K = {k} ---")
        logger.info(f"  Throughput:      {summary_k['tokens_per_second_mean']:.2f} ± {summary_k['tokens_per_second_std']:.2f} tok/s")
        logger.info(f"  Speedup:         {summary_k['speedup_vs_vanilla']:.2f}x vs Vanilla")
        logger.info(f"  Acceptance Rate: {summary_k['acceptance_rate']:.1%}")
        logger.info(f"  Tokens / Step:   {summary_k['tokens_per_step']:.2f} tokens/cycle")
        logger.info(f"  Greedy Match:    {summary_k['exact_match_rate']:.1%}")
        logger.info(f"  Latency / Cycle: {ms_per_cycle:.1f}ms [Draft: {draft_ms_cycle:.1f}ms ({summary_k['draft_pct']:.1f}%), Verify: {verify_ms_cycle:.1f}ms ({summary_k['verify_pct']:.1f}%), Cache: {cache_ms_cycle:.1f}ms ({summary_k['cache_pct']:.1f}%), Other: {other_ms_cycle:.1f}ms ({summary_k['other_pct']:.1f}%)]")
        logger.info(f"  Peak VRAM:       {summary_k['peak_vram_mb_mean']:.0f} MB")

        torch.cuda.empty_cache()
        gc.collect()

    # 6. Save Overall Summary
    final_summary = {
        "model": args.model,
        "draft_strategy": "cka_75",
        "num_layers_skipped": len(skip_indices),
        "vanilla_baseline_tok_s": vanilla_mean_tps,
        "vanilla_baseline_ms_tok": vanilla_ms_per_tok,
        "k_comparison": summary_results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    # 7. Generate Figures
    generate_speculative_figures(summary_results, vanilla_ms_per_tok, figures_dir)

    # 8. Print Clean Comprehensive Tables
    logger.info("\n" + "=" * 105)
    logger.info("PHASE 6 — SELF-SPECULATIVE DECODING BENCHMARK (CKA-75)")
    logger.info("=" * 105)
    logger.info(
        f"{'K':>4} | {'tok/s':>10} | {'Speedup':>9} | {'Accept Rate':>12} | "
        f"{'Tok/Step':>10} | {'Exact Match':>12} | {'Peak VRAM':>10}"
    )
    logger.info("-" * 105)
    logger.info(
        f"{'Base':>4} | {vanilla_mean_tps:>10.1f} | {'1.00x':>9} | {'N/A':>12} | "
        f"{'1.00':>10} | {'100.0%':>12} | {vanilla_vram:>9.0f}MB"
    )
    for k_key, s in summary_results.items():
        logger.info(
            f"{s['k']:>4} | "
            f"{s['tokens_per_second_mean']:>10.1f} | "
            f"{s['speedup_vs_vanilla']:>8.2f}x | "
            f"{s['acceptance_rate']:>11.1%} | "
            f"{s['tokens_per_step']:>10.2f} | "
            f"{s['exact_match_rate']:>11.1%} | "
            f"{s['peak_vram_mb_mean']:>9.0f}MB"
        )

    logger.info("\n" + "=" * 105)
    logger.info("LATENCY COST DECOMPOSITION (T_total = T_draft + T_verify + T_cache + T_other)")
    logger.info("=" * 105)
    logger.info(
        f"{'Config':>6} | {'Cycle(ms)':>10} | {'Draft(ms)':>10} | {'Verify(ms)':>11} | "
        f"{'Cache(ms)':>10} | {'Other(ms)':>10} | {'% Draft':>9} | {'% Verify':>9} | {'% Cache':>8}"
    )
    logger.info("-" * 105)
    for k_key, s in summary_results.items():
        logger.info(
            f"K={s['k']:<4} | "
            f"{s['cycle_latency_ms']:>10.1f} | "
            f"{s['draft_latency_ms']:>10.1f} | "
            f"{s['verify_latency_ms']:>11.1f} | "
            f"{s['cache_latency_ms']:>10.1f} | "
            f"{s['other_latency_ms']:>10.1f} | "
            f"{s['draft_pct']:>8.1f}% | "
            f"{s['verify_pct']:>8.1f}% | "
            f"{s['cache_pct']:>7.1f}%"
        )
    logger.info("=" * 105)
    logger.info(f"Results and figures saved to {output_dir} and {figures_dir}")


if __name__ == "__main__":
    main()
