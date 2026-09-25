"""Phase 11 — Real 4D Systems Pareto Frontier (Việc 5).

Evaluates the multi-objective trade-off space across:
    a = (S, K) in Action Space
Measuring:
    (Throughput [tok/s], Peak VRAM [MB], Energy [J/token], ExactMatch [%])

Answers Systems Optimization Queries:
  Query 1: Maximize TPS subject to VRAM <= 2500 MB and ExactMatch >= 95%
  Query 2: Minimize Energy/token subject to TPS >= 35 tok/s
  Query 3: Maximize TPS subject to Energy <= 1.60 J/token
  Query 4: Identification of the Non-Dominated 4D Pareto Frontier P*
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

from zassd.decoding.speculative import self_speculative_generate
from zassd.decoding.vanilla import vanilla_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.gpu import GPUProfiler
from zassd.profiling.memory import reset_vram_stats
from zassd.utils.logging import setup_logging
from zassd.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Complete layer configuration definitions
CONFIG_DEFINITIONS = {
    "cka_50": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22],
    "cka_60": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 21],
    "cka_67": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16],
    "cka_75": [3, 4, 5, 6, 7, 10, 11, 12, 13],
    "cka_83": [4, 5, 6, 7, 12, 13],
    "cka_90": [4, 5, 6, 7],
}

K_VALUES = [1, 2, 3, 4]


def compute_token_exact_match(ref_text: str, cand_text: str, tokenizer) -> float:
    """Compute token-level exact match percentage."""
    ref_tokens = tokenizer.encode(ref_text, add_special_tokens=False)
    cand_tokens = tokenizer.encode(cand_text, add_special_tokens=False)
    if not ref_tokens:
        return 1.0 if not cand_tokens else 0.0
    min_len = min(len(ref_tokens), len(cand_tokens))
    if min_len == 0:
        return 0.0
    matches = sum(1 for i in range(min_len) if ref_tokens[i] == cand_tokens[i])
    return float(matches / len(ref_tokens))


def is_pareto_dominated(p1: dict[str, Any], p2: dict[str, Any]) -> bool:
    """Check if point p1 is dominated by point p2 in 4D space.

    p2 dominates p1 iff:
      tps(p2) >= tps(p1)
      vram(p2) <= vram(p1)
      energy(p2) <= energy(p1)
      exact_match(p2) >= exact_match(p1)
    with at least one strict inequality.
    """
    not_worse = (
        p2["tokens_per_second"] >= p1["tokens_per_second"]
        and p2["peak_vram_mb"] <= p1["peak_vram_mb"]
        and p2["energy_j_token"] <= p1["energy_j_token"]
        and p2["exact_match_pct"] >= p1["exact_match_pct"]
    )
    strictly_better = (
        p2["tokens_per_second"] > p1["tokens_per_second"]
        or p2["peak_vram_mb"] < p1["peak_vram_mb"]
        or p2["energy_j_token"] < p1["energy_j_token"]
        or p2["exact_match_pct"] > p1["exact_match_pct"]
    )
    return bool(not_worse and strictly_better)


def compute_4d_pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Identify the non-dominated points in 4D (TPS, VRAM, Energy, ExactMatch) space."""
    frontier = []
    for i, p1 in enumerate(points):
        dominated = False
        for j, p2 in enumerate(points):
            if i != j and is_pareto_dominated(p1, p2):
                dominated = True
                break
        if not dominated:
            frontier.append(p1)
    return frontier


def plot_pareto_visualizations(
    all_points: list[dict[str, Any]],
    pareto_points: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    """Generate publication-ready 4D Pareto trade-off figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    pareto_names = set(p["action_name"] for p in pareto_points)

    # 1. Throughput vs Energy (Primary Pareto Frontier)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for p in all_points:
        is_p = p["action_name"] in pareto_names
        color = "#2ca02c" if is_p else "#aec7e8"
        size = 180 if is_p else 70
        marker = "*" if is_p else "o"
        ax.scatter(p["energy_j_token"], p["tokens_per_second"], color=color, s=size, marker=marker, edgecolors="k", zorder=4)

        if is_p:
            ax.annotate(
                f"{p['action_name']}\n({p['tokens_per_second']:.1f}t/s, {p['energy_j_token']:.2f}J)",
                (p["energy_j_token"], p["tokens_per_second"]),
                textcoords="offset points",
                xytext=(8, 4),
                fontsize=9,
                fontweight="bold",
            )

    # Connect Pareto frontier with line
    sorted_pareto = sorted(pareto_points, key=lambda x: x["energy_j_token"])
    px = [p["energy_j_token"] for p in sorted_pareto]
    py = [p["tokens_per_second"] for p in sorted_pareto]
    ax.plot(px, py, "g--", linewidth=2, alpha=0.8, label="4D Pareto Frontier Boundary")

    ax.set_xlabel("Energy Consumption (Joules / Token) [Lower is Better]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Generation Throughput (Tokens / Second) [Higher is Better]", fontsize=12, fontweight="bold")
    ax.set_title("Multi-Objective Pareto Frontier: Throughput vs. Energy Efficiency", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(figures_dir / "pareto_throughput_vs_energy.png", dpi=300)
    plt.close()

    # 2. 4D Bubble Chart: TPS vs VRAM (Bubble size = 1 / Energy, Color = ExactMatch)
    fig, ax2 = plt.subplots(figsize=(11, 6.5))
    tps = [p["tokens_per_second"] for p in all_points]
    vram = [p["peak_vram_mb"] for p in all_points]
    energies = [p["energy_j_token"] for p in all_points]
    exacts = [p["exact_match_pct"] for p in all_points]
    sizes = [max(40.0, (3.5 / e) * 60) for e in energies]

    scatter = ax2.scatter(
        vram,
        tps,
        s=sizes,
        c=exacts,
        cmap="plasma",
        edgecolors=["black" if p["action_name"] in pareto_names else "gray" for p in all_points],
        linewidths=[2.0 if p["action_name"] in pareto_names else 0.8 for p in all_points],
        alpha=0.85,
        zorder=4,
    )
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label("Exact Match Fidelity (%)", fontsize=11, fontweight="bold")

    for p in pareto_points:
        ax2.annotate(
            p["action_name"],
            (p["peak_vram_mb"], p["tokens_per_second"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            fontweight="bold",
            color="black",
        )

    ax2.set_xlabel("Peak VRAM Footprint (MB) [Budget <= 2500 MB]", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Throughput (Tokens / Second)", fontsize=12, fontweight="bold")
    ax2.set_title("4D Systems Pareto Surface (TPS x VRAM x Energy x Exactness)\n[Marker size ~ Energy Efficiency, Color ~ Exact Match]", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(figures_dir / "pareto_frontier_4d.png", dpi=300)
    plt.close()
    logger.info(f"Pareto figures successfully saved to {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct real 4D Systems Pareto Frontier on RTX 4050")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of benchmark prompts")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to generate per prompt")
    parser.add_argument("--output-dir", type=str, default="experiments/11_pareto")
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
    logger.info("PHASE 11 — 4D SYSTEMS PARETO FRONTIER SEARCH (RTX 4050 LAPTOP GPU)")
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

    # Establish Vanilla Reference Outputs
    logger.info("--- Establishing Vanilla Target Reference ---")
    vanilla_outputs: dict[int, str] = {}
    vanilla_runs = []
    t_v0 = time.perf_counter()
    e_v0 = gpu_profiler.get_total_energy_mj()

    for p in eval_prompts:
        text, m = vanilla_generate(model, tokenizer, p["prompt"], max_new_tokens=args.max_new_tokens, temperature=0.0)
        vanilla_outputs[p["id"]] = text
        vanilla_runs.append(m)

    torch.cuda.synchronize()
    t_v_elapsed = time.perf_counter() - t_v0
    e_v_end = gpu_profiler.get_total_energy_mj()

    tot_v_tokens = sum(m.total_tokens for m in vanilla_runs)
    vanilla_tps = tot_v_tokens / max(1e-3, t_v_elapsed)
    vanilla_vram = float(np.mean([m.peak_vram_mb for m in vanilla_runs]))

    if e_v0 is not None and e_v_end is not None and e_v_end >= e_v0:
        vanilla_energy = ((e_v_end - e_v0) / 1000.0) / max(1, tot_v_tokens)
    else:
        vanilla_energy = (gpu_profiler.get_power_usage() * t_v_elapsed) / max(1, tot_v_tokens)

    all_action_records: list[dict[str, Any]] = [
        {
            "action_name": "Vanilla (Full 36L)",
            "config_name": "vanilla",
            "k": 0,
            "layers_kept": 36,
            "tokens_per_second": round(vanilla_tps, 2),
            "speedup_vs_vanilla": 1.000,
            "peak_vram_mb": round(vanilla_vram, 1),
            "energy_j_token": round(vanilla_energy, 3),
            "exact_match_pct": 100.0,
            "acceptance_rate_pct": 100.0,
        }
    ]

    # 2. Benchmark the Full Grid of Candidate Actions (S, K)
    logger.info("\n--- Benchmarking Candidate Action Space (6 Configs x 4 K-values) ---")
    for cfg_name, skip_indices in CONFIG_DEFINITIONS.items():
        kept_count = 36 - len(skip_indices)
        for k in K_VALUES:
            action_label = f"{cfg_name} (K={k})"
            logger.info(f"Measuring Action: {action_label}...")

            gc.collect()
            torch.cuda.empty_cache()
            reset_vram_stats()

            e_start = gpu_profiler.get_total_energy_mj()
            t_start = time.perf_counter()

            runs = []
            texts = []
            for p in eval_prompts:
                text, m = self_speculative_generate(
                    model=model,
                    tokenizer=tokenizer,
                    layer_mgr=layer_mgr,
                    skip_indices=skip_indices,
                    prompt=p["prompt"],
                    k=k,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                )
                runs.append(m)
                texts.append(text)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start
            e_end = gpu_profiler.get_total_energy_mj()

            tot_tokens = sum(m.total_tokens for m in runs)
            tps = tot_tokens / max(1e-3, elapsed)
            mean_vram = float(np.mean([m.peak_vram_mb for m in runs]))
            mean_acc = float(np.mean([m.acceptance_rate for m in runs]) * 100.0)

            # Compute Exact Match against Vanilla
            em_scores = [compute_token_exact_match(vanilla_outputs[p["id"]], t, tokenizer) for p, t in zip(eval_prompts, texts)]
            exact_match_pct = float(np.mean(em_scores) * 100.0)

            # Energy
            if e_start is not None and e_end is not None and e_end >= e_start:
                energy_j = ((e_end - e_start) / 1000.0) / max(1, tot_tokens)
            else:
                energy_j = (gpu_profiler.get_power_usage() * elapsed) / max(1, tot_tokens)

            record = {
                "action_name": action_label,
                "config_name": cfg_name,
                "k": k,
                "layers_kept": kept_count,
                "tokens_per_second": round(tps, 2),
                "speedup_vs_vanilla": round(tps / vanilla_tps, 3),
                "peak_vram_mb": round(mean_vram, 1),
                "energy_j_token": round(energy_j, 3),
                "exact_match_pct": round(exact_match_pct, 1),
                "acceptance_rate_pct": round(mean_acc, 1),
            }
            all_action_records.append(record)

    # 3. Compute 4D Pareto Frontier
    pareto_frontier = compute_4d_pareto_frontier(all_action_records)

    # 4. Systems Constrained Optimization Queries
    logger.info("\n" + "=" * 95)
    logger.info("SYSTEMS PARETO OPTIMIZATION QUERIES")
    logger.info("=" * 95)

    # Query 1: Maximize TPS subject to VRAM <= 2500 MB and ExactMatch >= 95%
    q1_candidates = [p for p in all_action_records if p["peak_vram_mb"] <= 2500.0 and p["exact_match_pct"] >= 95.0]
    q1_best = max(q1_candidates, key=lambda x: x["tokens_per_second"]) if q1_candidates else None
    logger.info(
        f"Query 1 [max TPS s.t. VRAM <= 2500MB & ExactMatch >= 95%]:\n"
        f"  -> Optimal Action: {q1_best['action_name'] if q1_best else 'None'}\n"
        f"     TPS={q1_best['tokens_per_second']} tok/s, VRAM={q1_best['peak_vram_mb']}MB, "
        f"ExactMatch={q1_best['exact_match_pct']}%, Energy={q1_best['energy_j_token']} J/tok"
    )

    # Query 2: Minimize Energy/token subject to TPS >= 35 tok/s
    q2_candidates = [p for p in all_action_records if p["tokens_per_second"] >= 35.0]
    q2_best = min(q2_candidates, key=lambda x: x["energy_j_token"]) if q2_candidates else None
    logger.info(
        f"\nQuery 2 [min Energy/token s.t. TPS >= 35.0 tok/s]:\n"
        f"  -> Optimal Action: {q2_best['action_name'] if q2_best else 'None'}\n"
        f"     Energy={q2_best['energy_j_token']} J/tok, TPS={q2_best['tokens_per_second']} tok/s, "
        f"VRAM={q2_best['peak_vram_mb']}MB, ExactMatch={q2_best['exact_match_pct']}%"
    )

    # Query 3: Maximize TPS subject to Energy <= 1.50 J/token
    q3_candidates = [p for p in all_action_records if p["energy_j_token"] <= 1.50]
    q3_best = max(q3_candidates, key=lambda x: x["tokens_per_second"]) if q3_candidates else None
    logger.info(
        f"\nQuery 3 [max TPS s.t. Energy <= 1.50 J/token]:\n"
        f"  -> Optimal Action: {q3_best['action_name'] if q3_best else 'None'}\n"
        f"     TPS={q3_best['tokens_per_second']} tok/s, Energy={q3_best['energy_j_token']} J/tok, "
        f"ExactMatch={q3_best['exact_match_pct']}%"
    )

    # 5. Print Complete Pareto Frontier Table
    logger.info("\n" + "=" * 105)
    logger.info("OFFICIAL 4D NON-DOMINATED PARETO FRONTIER P* (RTX 4050 LAPTOP)")
    logger.info("=" * 105)
    header = f"{'Pareto Optimal Action':<25} | {'tok/s':<8} | {'Speedup':<8} | {'VRAM (MB)':<10} | {'Energy (J/t)':<12} | {'Exact Match':<11} | {'Acceptance':<10}"
    logger.info(header)
    logger.info("-" * 105)
    for p in sorted(pareto_frontier, key=lambda x: x["tokens_per_second"], reverse=True):
        row = (
            f"{p['action_name']:<25} | "
            f"{p['tokens_per_second']:<8.2f} | "
            f"{p['speedup_vs_vanilla']:<8.2f}x | "
            f"{p['peak_vram_mb']:<10.1f} | "
            f"{p['energy_j_token']:<12.3f} | "
            f"{p['exact_match_pct']:<10.1f}% | "
            f"{p['acceptance_rate_pct']:<9.1f}%"
        )
        logger.info(row)
    logger.info("=" * 105)

    # 6. Save JSON Database and Plots
    summary_data = {
        "status": "PASS",
        "num_total_actions_evaluated": len(all_action_records),
        "num_pareto_optimal_actions": len(pareto_frontier),
        "hardware": "NVIDIA GeForce RTX 4050 Laptop GPU (6GB, 80W)",
        "model": args.model,
        "systems_queries": {
            "query_1_max_tps_vram_2500mb": q1_best,
            "query_2_min_energy_tps_35": q2_best,
            "query_3_max_tps_energy_1_50j": q3_best,
        },
        "pareto_frontier": pareto_frontier,
        "all_actions": all_action_records,
    }

    summary_file = output_dir / "pareto_frontier_4d.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved full 4D Pareto database to {summary_file}")

    plot_pareto_visualizations(all_action_records, pareto_frontier, figures_dir)


if __name__ == "__main__":
    main()
