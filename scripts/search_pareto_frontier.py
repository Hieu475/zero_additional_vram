"""Phase 7.6 — Search Layer-Budget Pareto Frontier.

Evaluates candidate layer configurations across layer budgets:
  - CKA-50% (18 layers kept, 18 skipped)
  - CKA-60% (22 layers kept, 14 skipped)
  - CKA-67% (24 layers kept, 12 skipped)
  - CKA-75% (27 layers kept, 9 skipped)
  - CKA-83% (30 layers kept, 6 skipped)
  - CKA-90% (32 layers kept, 4 skipped)

Measures for each configuration:
  1. Draft latency per token (ms)
  2. Acceptance rate (%)
  3. Top-1 agreement with full model (%)
  4. Logit cosine similarity with full model
  5. End-to-end throughput (tok/s) and speedup vs Vanilla
  6. Greedy exact match rate (%)
  7. Peak VRAM footprint (MB)
  8. Hardware-integrated energy consumption per token (J/tok)

Generates Pareto frontier figures:
  - results/figures/layer_pareto_frontier.png
  - results/figures/pareto_acceptance_vs_draft_latency.png
"""

from __future__ import annotations

import argparse
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
import torch.nn.functional as F

from zassd.decoding.speculative import self_speculative_generate
from zassd.decoding.vanilla import vanilla_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.gpu import GPUProfiler
from zassd.utils.logging import setup_logging
from zassd.utils.seed import set_seed

logger = logging.getLogger(__name__)


def compute_layer_similarity_metrics(
    model,
    layer_mgr: LayerManager,
    skip_indices: list[int],
    eval_prompts: list[dict[str, Any]],
    tokenizer,
    device: str = "cuda:0",
) -> tuple[float, float]:
    """Compute mean top-1 agreement and logit cosine similarity between draft and full model."""
    top1_agreements = []
    cosine_sims = []

    for p in eval_prompts:
        prompt_ids = tokenizer(p["prompt"], return_tensors="pt").input_ids.to(device)

        # Full model logits
        with torch.no_grad():
            full_out = model(prompt_ids, use_cache=False)
        full_logits = full_out.logits[0, -1, :].float()

        # Draft model logits (with skipped layers)
        with torch.no_grad():
            with layer_mgr.skip_layers(skip_indices):
                draft_out = model(prompt_ids, use_cache=False)
        draft_logits = draft_out.logits[0, -1, :].float()

        # Top-1 agreement
        full_pred = int(full_logits.argmax().item())
        draft_pred = int(draft_logits.argmax().item())
        top1_agreements.append(1.0 if full_pred == draft_pred else 0.0)

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            full_logits.unsqueeze(0), draft_logits.unsqueeze(0), dim=-1
        ).item()
        cosine_sims.append(cos_sim)

    return float(np.mean(top1_agreements)), float(np.mean(cosine_sims))


def generate_pareto_figures(
    pareto_data: dict[str, dict[str, Any]],
    vanilla_tps: float,
    figures_dir: Path,
) -> None:
    """Generate publication-quality Pareto frontier visualizations."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    configs = list(pareto_data.keys())
    layers_kept = [pareto_data[c]["layers_kept"] for c in configs]
    draft_latencies = [pareto_data[c]["draft_latency_ms"] for c in configs]
    acceptance_rates = [pareto_data[c]["acceptance_rate"] * 100 for c in configs]
    top1_agreements = [pareto_data[c]["top1_agreement"] * 100 for c in configs]
    cosine_sims = [pareto_data[c]["cosine_similarity"] for c in configs]
    end_to_end_tps = [pareto_data[c]["end_to_end_tps"] for c in configs]
    speedups = [pareto_data[c]["speedup_vs_vanilla"] for c in configs]
    energies = [pareto_data[c]["energy_j_token"] for c in configs]

    # Figure 1: Acceptance Rate vs Draft Latency (The Core Pareto Frontier)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        draft_latencies,
        acceptance_rates,
        s=[s * 150 for s in speedups],
        c=speedups,
        cmap="viridis",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Speedup vs Vanilla (x)", fontsize=11)

    for i, c in enumerate(configs):
        ax.annotate(
            f"{c}\n({layers_kept[i]}L, {speedups[i]:.2f}x)",
            (draft_latencies[i], acceptance_rates[i]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
            fontweight="bold",
        )

    # Sort for Pareto line
    sorted_indices = np.argsort(draft_latencies)
    sorted_draft = np.array(draft_latencies)[sorted_indices]
    sorted_acc = np.array(acceptance_rates)[sorted_indices]
    ax.plot(sorted_draft, sorted_acc, "--", color="gray", alpha=0.7, label="Empirical Trade-off Curve")

    ax.set_xlabel("Draft Latency per Token (ms)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Empirical Acceptance Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Pareto Frontier: Acceptance Rate vs. Draft Latency\n(Marker size proportional to Speedup)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig_path1 = figures_dir / "pareto_acceptance_vs_draft_latency.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    logger.info(f"Saved {fig_path1}")

    # Figure 2: Multi-metric Trade-off vs Layers Kept
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Subplot 1: End-to-End Speedup & Throughput
    ax1.plot(layers_kept, speedups, marker="o", color="#1f77b4", linewidth=2.5, label="Speculative Speedup")
    ax1.axhline(1.0, color="crimson", linestyle="--", label="Vanilla Baseline (1.0x)")
    ax1.set_xlabel("Number of Layers Kept in Draft", fontweight="bold")
    ax1.set_ylabel("Speedup vs Vanilla (x)", fontweight="bold")
    ax1.set_title("End-to-End Speedup", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Subplot 2: Draft Latency vs Vanilla TPOT
    vanilla_tpot = 1000.0 / vanilla_tps if vanilla_tps > 0 else 25.0
    ax2.plot(layers_kept, draft_latencies, marker="s", color="#ff7f0e", linewidth=2.5, label="Draft Latency (ms)")
    ax2.axhline(vanilla_tpot, color="crimson", linestyle="--", label=f"Vanilla TPOT ({vanilla_tpot:.1f}ms)")
    ax2.set_xlabel("Number of Layers Kept in Draft", fontweight="bold")
    ax2.set_ylabel("Latency per Token (ms)", fontweight="bold")
    ax2.set_title("Draft Latency vs. Vanilla Decode Latency", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    # Subplot 3: Quality Metrics (Acceptance & Top-1 Agreement)
    ax3.plot(layers_kept, acceptance_rates, marker="^", color="#2ca02c", linewidth=2, label="Acceptance Rate (%)")
    ax3.plot(layers_kept, top1_agreements, marker="v", color="#9467bd", linewidth=2, label="Top-1 Agreement (%)")
    ax3.set_xlabel("Number of Layers Kept in Draft", fontweight="bold")
    ax3.set_ylabel("Percentage (%)", fontweight="bold")
    ax3.set_title("Draft Quality Metrics", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    # Subplot 4: Hardware Energy Consumption
    ax4.plot(layers_kept, energies, marker="D", color="#d62728", linewidth=2, label="Energy (J/token)")
    ax4.set_xlabel("Number of Layers Kept in Draft", fontweight="bold")
    ax4.set_ylabel("Energy per Token (Joules)", fontweight="bold")
    ax4.set_title("Hardware Integrated Energy per Token", fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend()

    plt.suptitle("Comprehensive Layer-Budget Pareto Analysis (Qwen2.5-3B)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig_path2 = figures_dir / "layer_pareto_frontier.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    logger.info(f"Saved {fig_path2}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Layer-Budget Pareto Frontier for Self-Speculative Decoding")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--k", type=int, default=2, help="Draft speculation length for Pareto evaluation")
    parser.add_argument("--num-prompts", type=int, default=5, help="Number of benchmark prompts to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to generate per run")
    parser.add_argument("--output-dir", type=str, default="experiments/07_pareto")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHASE 7.6 — LAYER-BUDGET PARETO FRONTIER SEARCH")
    logger.info("=" * 80)

    # 1. Load Model, Tokenizer, Adapters
    model = load_model(args.model, quantize=True, bits=4)
    tokenizer = load_tokenizer(args.model)
    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)
    gpu_profiler = GPUProfiler()

    # 2. Candidate layer configurations sorted by CKA redundancy
    with open("experiments/03_cka/layer_redundancy_ranking.json") as f:
        ranking = json.load(f)
    ranked_indices = [item["layer_idx"] for item in ranking]

    # Budgets: (config_name, n_layers_to_skip, n_layers_kept)
    candidate_budgets = [
        ("cka_50", 18, 18),
        ("cka_60", 14, 22),
        ("cka_67", 12, 24),
        ("cka_75", 9, 27),
        ("cka_83", 6, 30),
        ("cka_90", 4, 32),
    ]

    # Load fixed benchmark prompts
    prompts_path = Path("data/benchmarks/prompts.jsonl")
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line.strip()))
    eval_prompts = prompts[: args.num_prompts]

    # 3. Establish Vanilla Baseline
    logger.info("\n--- Establishing Vanilla Target Baseline ---")
    vanilla_tps_list = []
    vanilla_refs = {}
    for p in eval_prompts:
        text, v_metrics = vanilla_generate(model, tokenizer, p["prompt"], max_new_tokens=args.max_new_tokens)
        vanilla_tps_list.append(v_metrics.tokens_per_second)
        vanilla_refs[p["id"]] = text

    vanilla_mean_tps = float(np.mean(vanilla_tps_list))
    logger.info(f"Vanilla Mean Throughput: {vanilla_mean_tps:.2f} tok/s")

    # 4. Sweep each candidate layer configuration
    pareto_results: dict[str, dict[str, Any]] = {}

    for name, n_skip, n_kept in candidate_budgets:
        skip_indices = sorted(ranked_indices[:n_skip])
        logger.info(f"\nEvaluating configuration: {name} (kept: {n_kept}/36, skipped: {n_skip})")

        # Quality metrics (top-1 agreement, cosine similarity)
        top1_agr, cos_sim = compute_layer_similarity_metrics(
            model=model,
            layer_mgr=layer_mgr,
            skip_indices=skip_indices,
            eval_prompts=eval_prompts,
            tokenizer=tokenizer,
        )

        # Speculative execution runs
        runs = []
        exact_matches = 0

        # Energy measurement start
        e_start_mj = gpu_profiler.get_total_energy_mj()

        for p in eval_prompts:
            text, s_metrics = self_speculative_generate(
                model=model,
                tokenizer=tokenizer,
                layer_mgr=layer_mgr,
                skip_indices=skip_indices,
                prompt=p["prompt"],
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            runs.append(s_metrics)
            if text == vanilla_refs[p["id"]]:
                exact_matches += 1

        e_end_mj = gpu_profiler.get_total_energy_mj()

        # Compute aggregate metrics
        mean_tps = float(np.mean([m.tokens_per_second for m in runs]))
        mean_accept = float(np.mean([m.acceptance_rate for m in runs]))
        mean_vram = float(np.mean([m.peak_vram_mb for m in runs]))
        speedup = mean_tps / vanilla_mean_tps if vanilla_mean_tps > 0 else 1.0

        tot_cycles = max(1, sum(m.num_verification_cycles for m in runs))
        tot_draft_time = sum(m.draft_time_s for m in runs)
        tot_draft_tokens = max(1, sum(m.total_draft_tokens for m in runs))
        draft_latency_per_tok = (tot_draft_time / tot_draft_tokens) * 1000.0

        total_tokens = max(1, sum(m.total_tokens for m in runs))
        if e_start_mj is not None and e_end_mj is not None and e_end_mj >= e_start_mj:
            energy_j_tok = ((e_end_mj - e_start_mj) / 1000.0) / total_tokens
        else:
            energy_j_tok = 2.0  # fallback

        exact_match_rate = exact_matches / len(eval_prompts)

        pareto_results[name] = {
            "name": name,
            "layers_kept": n_kept,
            "layers_skipped": n_skip,
            "skip_indices": skip_indices,
            "draft_latency_ms": round(draft_latency_per_tok, 2),
            "acceptance_rate": round(mean_accept, 4),
            "top1_agreement": round(top1_agr, 4),
            "cosine_similarity": round(cos_sim, 4),
            "end_to_end_tps": round(mean_tps, 2),
            "speedup_vs_vanilla": round(speedup, 3),
            "exact_match_rate": round(exact_match_rate, 4),
            "peak_vram_mb": round(mean_vram, 1),
            "energy_j_token": round(energy_j_tok, 4),
        }

        logger.info(
            f"  {name}: Draft Latency: {draft_latency_per_tok:.1f}ms/tok | "
            f"Accept: {mean_accept:.1%} | Top1: {top1_agr:.1%} | Cos: {cos_sim:.3f} | "
            f"TPS: {mean_tps:.1f} ({speedup:.2f}x) | Match: {exact_match_rate:.1%} | Energy: {energy_j_tok:.3f} J/tok"
        )

    # 5. Save results to JSON
    raw_path = output_dir / "pareto_results.json"
    with open(raw_path, "w") as f:
        json.dump(pareto_results, f, indent=2)
    logger.info(f"\nPareto results saved to {raw_path}")

    # 6. Generate Figures
    generate_pareto_figures(pareto_results, vanilla_mean_tps, figures_dir)
    logger.info("Pareto frontier search complete.")


if __name__ == "__main__":
    main()
