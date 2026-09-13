"""Phase 3 — CKA Layer Analysis and Redundancy-Guided Layer Selection.

Performs:
1. Activation collection across all 36 transformer layers on calibration set.
2. Full 36x36 CKA similarity matrix calculation.
3. Heatmap and adjacent similarity visualization (saved to results/figures/).
4. Layer redundancy ranking and CKA-guided layer selection (75%, 50%, 25% kept).
5. Representation similarity (CKA) vs Output similarity (Cosine, KL, Top-1 agreement)
   correlation analysis (Section XI of research plan).
6. Performance & quality benchmark of CKA-selected configurations compared to
   Phase 2 static/random baselines.

Usage:
    python scripts/run_cka_analysis.py
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
import seaborn as sns
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.models.layer_manager import LayerManager
from zassd.layer_selection.cka import (
    compute_adjacent_similarity,
    compute_cka_matrix,
    linear_cka,
    rank_layers_by_redundancy,
    select_layers_cka,
)
from zassd.profiling.memory import get_vram_usage, reset_vram_stats
from zassd.utils.config import load_config
from zassd.utils.seed import set_seed
from zassd.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Activation collection
# ---------------------------------------------------------------------------

def collect_layer_activations(
    model,
    tokenizer,
    prompts: list[dict],
    device: str = "cuda:0",
    max_tokens_per_prompt: int = 64,
) -> dict[int, torch.Tensor]:
    """Collect hidden states from all transformer layers.

    Hidden states are extracted after each layer and stored on CPU (float32).
    """
    logger.info("Collecting activations across all layers...")
    layer_tensors: dict[int, list[torch.Tensor]] = {}

    with torch.no_grad():
        for idx, item in enumerate(prompts):
            text = item["prompt"]
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=128
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)
            # outputs.hidden_states is tuple: (embeddings, layer_0, layer_1, ..., layer_35)
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer
            num_layers = len(hidden_states)

            for l_idx in range(num_layers):
                # Shape: (batch, seq_len, hidden_dim) -> (seq_len, hidden_dim)
                hs = hidden_states[l_idx][0].detach().cpu().to(torch.float32)
                if hs.shape[0] > max_tokens_per_prompt:
                    hs = hs[:max_tokens_per_prompt]
                if l_idx not in layer_tensors:
                    layer_tensors[l_idx] = []
                layer_tensors[l_idx].append(hs)

            if (idx + 1) % 10 == 0 or (idx + 1) == len(prompts):
                logger.info(f"  Processed {idx + 1}/{len(prompts)} calibration prompts")

    # Concatenate tokens across all prompts for each layer
    combined: dict[int, torch.Tensor] = {}
    for l_idx, tensor_list in layer_tensors.items():
        combined[l_idx] = torch.cat(tensor_list, dim=0)

    total_tokens = combined[0].shape[0]
    hidden_dim = combined[0].shape[1]
    logger.info(
        f"Activations collected: {len(combined)} layers, "
        f"{total_tokens} tokens per layer, dim={hidden_dim}"
    )
    return combined


# ---------------------------------------------------------------------------
# 2. Visualizations
# ---------------------------------------------------------------------------

def generate_visualizations(
    cka_matrix: np.ndarray,
    adjacent_sim: np.ndarray,
    figures_dir: Path,
) -> None:
    """Generate and save publication-quality CKA visualizations."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    num_layers = cka_matrix.shape[0]

    # --- Plot 1: Full CKA Heatmap ---
    plt.figure(figsize=(10, 8), dpi=300)
    sns.heatmap(
        cka_matrix,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Linear CKA Similarity"},
        xticklabels=5,
        yticklabels=5,
    )
    plt.title(
        f"Layer-to-Layer CKA Representation Similarity Matrix (Qwen2.5-3B, {num_layers} Layers)",
        fontsize=12,
        pad=12,
    )
    plt.xlabel("Layer Index", fontsize=10)
    plt.ylabel("Layer Index", fontsize=10)
    plt.tight_layout()
    heatmap_path = figures_dir / "cka_heatmap.png"
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Saved CKA heatmap to {heatmap_path}")

    # --- Plot 2: Adjacent Layer Similarity Curve ---
    plt.figure(figsize=(9, 4.5), dpi=300)
    layer_transitions = [f"{i}→{i+1}" for i in range(num_layers - 1)]
    plt.plot(
        range(num_layers - 1),
        adjacent_sim,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="CKA(Layer i, Layer i+1)",
    )
    plt.axhline(
        y=float(np.mean(adjacent_sim)),
        color="red",
        linestyle="--",
        label=f"Mean = {np.mean(adjacent_sim):.3f}",
    )
    plt.title(
        "Adjacent Layer Representation Similarity Across Network Depth",
        fontsize=12,
    )
    plt.xlabel("Layer Transition (i → i+1)", fontsize=10)
    plt.ylabel("Linear CKA", fontsize=10)
    plt.xticks(
        range(0, num_layers - 1, 3),
        [layer_transitions[i] for i in range(0, num_layers - 1, 3)],
        rotation=45,
    )
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.ylim(min(0.5, float(np.min(adjacent_sim)) - 0.05), 1.02)
    plt.legend(loc="lower left")
    plt.tight_layout()
    adjacent_path = figures_dir / "cka_adjacent_similarity.png"
    plt.savefig(adjacent_path)
    plt.close()
    logger.info(f"Saved adjacent similarity plot to {adjacent_path}")


# ---------------------------------------------------------------------------
# 3. Correlation: CKA vs Output Metrics (Section XI)
# ---------------------------------------------------------------------------

def evaluate_single_layer_skips(
    model,
    tokenizer,
    adapter: ModelAdapter,
    layer_mgr: LayerManager,
    cka_matrix: np.ndarray,
    prompts: list[dict],
    figures_dir: Path,
    device: str = "cuda:0",
) -> dict:
    """Evaluate individual layer skips to correlate CKA with logit metrics."""
    logger.info("Evaluating single-layer skips for CKA correlation analysis...")
    num_layers = adapter.num_layers
    eval_prompts = prompts[:15]

    cka_scores = []
    cos_scores = []
    kl_scores = []
    top1_scores = []

    # Skip each middle layer individually (layers 1 to num_layers - 2)
    test_layers = list(range(1, num_layers - 1))

    for l_idx in test_layers:
        # CKA metric for layer l_idx: average adjacent similarity
        prev_sim = cka_matrix[l_idx - 1, l_idx]
        next_sim = cka_matrix[l_idx, l_idx + 1]
        cka_score = float(0.5 * (prev_sim + next_sim))
        cka_scores.append(cka_score)

        layer_cos = []
        layer_kl = []
        layer_top1 = []

        for p in eval_prompts:
            inputs = tokenizer(p["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                full_logits = model(input_ids=inputs["input_ids"]).logits[:, -1, :].float()

                with layer_mgr.skip_layers([l_idx]):
                    skip_logits = model(input_ids=inputs["input_ids"]).logits[:, -1, :].float()

            cos = F.cosine_similarity(full_logits, skip_logits, dim=-1).item()
            layer_cos.append(cos)

            full_p = F.softmax(full_logits, dim=-1)
            skip_logp = F.log_softmax(skip_logits, dim=-1)
            kl = F.kl_div(skip_logp, full_p, reduction="batchmean").item()
            layer_kl.append(kl)

            top1_match = (full_logits.argmax(dim=-1) == skip_logits.argmax(dim=-1)).float().item()
            layer_top1.append(top1_match)

        cos_scores.append(float(np.mean(layer_cos)))
        kl_scores.append(float(np.mean(layer_kl)))
        top1_scores.append(float(np.mean(layer_top1)))

    # Compute Pearson and Spearman correlations
    pearson_cos, p_cos = pearsonr(cka_scores, cos_scores)
    spearman_cos, sp_cos = spearmanr(cka_scores, cos_scores)

    pearson_kl, p_kl = pearsonr(cka_scores, [-k for k in kl_scores])
    spearman_kl, sp_kl = spearmanr(cka_scores, [-k for k in kl_scores])

    pearson_top1, p_top1 = pearsonr(cka_scores, top1_scores)
    spearman_top1, sp_top1 = spearmanr(cka_scores, top1_scores)

    correlations = {
        "cka_vs_cosine": {
            "pearson_r": float(pearson_cos),
            "pearson_p": float(p_cos),
            "spearman_r": float(spearman_cos),
            "spearman_p": float(sp_cos),
        },
        "cka_vs_neg_kl": {
            "pearson_r": float(pearson_kl),
            "pearson_p": float(p_kl),
            "spearman_r": float(spearman_kl),
            "spearman_p": float(sp_kl),
        },
        "cka_vs_top1": {
            "pearson_r": float(pearson_top1),
            "pearson_p": float(p_top1),
            "spearman_r": float(spearman_top1),
            "spearman_p": float(sp_top1),
        },
        "layer_data": [
            {
                "layer": l_idx,
                "cka": cka,
                "cosine": cos,
                "kl": kl,
                "top1": top1,
            }
            for l_idx, cka, cos, kl, top1 in zip(
                test_layers, cka_scores, cos_scores, kl_scores, top1_scores
            )
        ],
    }

    logger.info(
        f"Correlation Results:\n"
        f"  CKA vs Logit Cosine: Pearson r={pearson_cos:.3f} (p={p_cos:.3e}), Spearman r={spearman_cos:.3f}\n"
        f"  CKA vs Top-1 Match:  Pearson r={pearson_top1:.3f} (p={p_top1:.3e}), Spearman r={spearman_top1:.3f}\n"
        f"  CKA vs -KL Div:      Pearson r={pearson_kl:.3f} (p={p_kl:.3e}), Spearman r={spearman_kl:.3f}"
    )

    # --- Plot: CKA vs Output Metrics Scatter Plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    # Subplot 1: CKA vs Logit Cosine
    ax1.scatter(cka_scores, cos_scores, color="#2ca02c", alpha=0.8, edgecolors="k", s=60)
    m, b = np.polyfit(cka_scores, cos_scores, 1)
    ax1.plot(
        np.array(cka_scores),
        m * np.array(cka_scores) + b,
        color="darkgreen",
        linestyle="--",
        label=f"Fit: r={pearson_cos:.2f} (p={p_cos:.2e})",
    )
    ax1.set_title("CKA Similarity vs. Logit Cosine Similarity", fontsize=11)
    ax1.set_xlabel("Adjacent CKA Score", fontsize=10)
    ax1.set_ylabel("Logit Cosine Similarity", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Subplot 2: CKA vs Top-1 Agreement
    ax2.scatter(cka_scores, top1_scores, color="#d62728", alpha=0.8, edgecolors="k", s=60)
    m2, b2 = np.polyfit(cka_scores, top1_scores, 1)
    ax2.plot(
        np.array(cka_scores),
        m2 * np.array(cka_scores) + b2,
        color="darkred",
        linestyle="--",
        label=f"Fit: r={pearson_top1:.2f} (p={p_top1:.2e})",
    )
    ax2.set_title("CKA Similarity vs. Top-1 Token Agreement", fontsize=11)
    ax2.set_xlabel("Adjacent CKA Score", fontsize=10)
    ax2.set_ylabel("Top-1 Agreement Rate", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    scatter_path = figures_dir / "cka_vs_metrics.png"
    plt.savefig(scatter_path)
    plt.close()
    logger.info(f"Saved correlation scatter plots to {scatter_path}")

    return correlations


# ---------------------------------------------------------------------------
# 4. Benchmark CKA-selected configurations
# ---------------------------------------------------------------------------

def benchmark_cka_configurations(
    model,
    tokenizer,
    adapter: ModelAdapter,
    layer_mgr: LayerManager,
    cka_matrix: np.ndarray,
    prompts: list[dict],
    device: str = "cuda:0",
    max_new_tokens: int = 128,
    runs: int = 15,
    warmup: int = 5,
) -> dict:
    """Benchmark CKA-guided layer selection configurations."""
    from scripts.benchmark_layer_skip import measure_quality, measure_speed

    num_layers = adapter.num_layers
    ref_path = Path("experiments/01_vanilla/ctx_128/outputs.jsonl")
    reference_outputs = []
    if ref_path.exists():
        with open(ref_path) as f:
            for line in f:
                if line.strip():
                    reference_outputs.append(json.loads(line.strip()))

    # Define CKA configurations: 75% kept (skip 9), 50% kept (skip 18), 25% kept (skip 27)
    configs = {}

    for kept_ratio, skip_count, name in [
        (0.75, 9, "cka_75"),
        (0.50, 18, "cka_50"),
        (0.25, 27, "cka_25"),
    ]:
        kept, skipped = select_layers_cka(
            cka_matrix=cka_matrix,
            num_to_skip=skip_count,
            always_keep=[0, -1],
            avoid_consecutive=(kept_ratio >= 0.5),
        )
        configs[name] = {
            "name": name,
            "kept": kept,
            "skipped": skipped,
            "description": f"CKA-guided selection: {len(kept)}/{num_layers} layers kept",
        }

    # Reference full model speed
    logger.info("\n--- Measuring full model speed reference ---")
    full_speed = measure_speed(
        model, tokenizer, layer_mgr,
        skip_indices=[], prompts=prompts,
        max_new_tokens=max_new_tokens, warmup_runs=warmup, measured_runs=runs,
    )
    full_tps = full_speed["tokens_per_second"]["mean"]

    results = {}
    for name, cfg in configs.items():
        logger.info(f"\nEvaluating CKA configuration: {name}")
        logger.info(f"  Kept ({len(cfg['kept'])}): {cfg['kept']}")
        logger.info(f"  Skipped ({len(cfg['skipped'])}): {cfg['skipped']}")

        quality = measure_quality(
            model, tokenizer, adapter, layer_mgr,
            skip_indices=cfg["skipped"],
            prompts=prompts,
            reference_outputs=reference_outputs,
            max_new_tokens=max_new_tokens,
        )

        speed = measure_speed(
            model, tokenizer, layer_mgr,
            skip_indices=cfg["skipped"],
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup,
            measured_runs=runs,
        )
        speedup = speed["tokens_per_second"]["mean"] / full_tps

        results[name] = {
            "name": name,
            "layers_kept": len(cfg["kept"]),
            "layers_skipped": len(cfg["skipped"]),
            "kept_indices": cfg["kept"],
            "skipped_indices": cfg["skipped"],
            "speed": speed,
            "quality": quality,
            "speedup": speedup,
        }
        logger.info(
            f"  Result: {speed['tokens_per_second']['mean']:.1f} tok/s ({speedup:.2f}x), "
            f"Cosine={quality['avg_cosine_similarity']:.4f}, Top1={quality['avg_top1_agreement']:.1%}, "
            f"KL={quality['avg_kl_divergence']:.4f}"
        )

    return results


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CKA Layer Analysis")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output-dir", type=str, default="experiments/03_cka")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=str(output_dir / "cka_analysis.log"))
    set_seed(42)

    logger.info("=" * 70)
    logger.info("PHASE 3 — CKA LAYER ANALYSIS & REDUNDANCY INVESTIGATION")
    logger.info("=" * 70)

    # 1. Load model
    model = load_model(args.model, quantize=True, bits=args.bits)
    tokenizer = load_tokenizer(args.model)
    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)
    num_layers = adapter.num_layers
    logger.info(f"Loaded {args.model} with {num_layers} layers")

    # 2. Load calibration prompts
    prompts_path = Path("data/benchmarks/prompts.jsonl")
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line.strip()))
    logger.info(f"Loaded {len(prompts)} calibration prompts")

    # 3. Collect activations & 4. Compute CKA matrix (or load cached)
    cka_file = output_dir / "cka_matrix.npy"
    if cka_file.exists():
        logger.info(f"Loading cached CKA matrix from {cka_file}")
        cka_matrix = np.load(str(cka_file))
    else:
        activations = collect_layer_activations(model, tokenizer, prompts)
        logger.info("Computing full 36x36 CKA similarity matrix...")
        cka_matrix = compute_cka_matrix(activations, device="cpu")
        np.save(str(cka_file), cka_matrix)
        with open(output_dir / "cka_matrix.json", "w") as f:
            json.dump(cka_matrix.tolist(), f, indent=2)

    adjacent_sim = compute_adjacent_similarity(cka_matrix)

    # 5. Visualizations
    generate_visualizations(cka_matrix, adjacent_sim, figures_dir)

    # 6. Redundancy ranking
    ranked_layers = rank_layers_by_redundancy(cka_matrix, always_keep=[0, -1])
    ranking_data = [
        {"rank": r + 1, "layer_idx": idx, "redundancy_score": float(score)}
        for r, (idx, score) in enumerate(ranked_layers)
    ]
    with open(output_dir / "layer_redundancy_ranking.json", "w") as f:
        json.dump(ranking_data, f, indent=2)

    logger.info("\n--- Top 10 Most Redundant Layers ---")
    for item in ranking_data[:10]:
        logger.info(f"  Rank {item['rank']:>2}: Layer {item['layer_idx']:>2} (score={item['redundancy_score']:.4f})")

    # 7. Correlation analysis (Section XI) (or load cached)
    corr_file = output_dir / "correlation_analysis.json"
    if corr_file.exists():
        logger.info(f"Loading cached correlation analysis from {corr_file}")
        with open(corr_file) as f:
            correlations = json.load(f)
    else:
        correlations = evaluate_single_layer_skips(
            model, tokenizer, adapter, layer_mgr, cka_matrix, prompts, figures_dir
        )
        with open(corr_file, "w") as f:
            json.dump(correlations, f, indent=2)

    # 8. Benchmark CKA configurations and compare with Phase 2
    cka_benchmarks = benchmark_cka_configurations(
        model, tokenizer, adapter, layer_mgr, cka_matrix, prompts
    )
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(cka_benchmarks, f, indent=2)

    # 9. Comparative summary with Phase 2 static/random baselines
    phase2_summary_file = Path("experiments/02_layer_skip/summary.json")
    phase2_data = {}
    if phase2_summary_file.exists():
        with open(phase2_summary_file) as f:
            phase2_data = json.load(f)

    comparison_table = []
    # 75% budget
    comparison_table.append({
        "budget": "75%",
        "cka_method": "cka_75",
        "cka_speedup": cka_benchmarks["cka_75"]["speedup"],
        "cka_cosine": cka_benchmarks["cka_75"]["quality"]["avg_cosine_similarity"],
        "cka_top1": cka_benchmarks["cka_75"]["quality"]["avg_top1_agreement"],
        "static_method": "static_75",
        "static_speedup": phase2_data.get("static_75", {}).get("speedup", 0.0),
        "static_cosine": phase2_data.get("static_75", {}).get("cosine_similarity", 0.0),
        "static_top1": phase2_data.get("static_75", {}).get("top1_agreement", 0.0),
        "random_best_method": "random_75_s42",
        "random_best_cosine": phase2_data.get("random_75_s42", {}).get("cosine_similarity", 0.0),
        "random_best_top1": phase2_data.get("random_75_s42", {}).get("top1_agreement", 0.0),
    })
    # 50% budget
    comparison_table.append({
        "budget": "50%",
        "cka_method": "cka_50",
        "cka_speedup": cka_benchmarks["cka_50"]["speedup"],
        "cka_cosine": cka_benchmarks["cka_50"]["quality"]["avg_cosine_similarity"],
        "cka_top1": cka_benchmarks["cka_50"]["quality"]["avg_top1_agreement"],
        "static_method": "static_50",
        "static_speedup": phase2_data.get("static_50", {}).get("speedup", 0.0),
        "static_cosine": phase2_data.get("static_50", {}).get("cosine_similarity", 0.0),
        "static_top1": phase2_data.get("static_50", {}).get("top1_agreement", 0.0),
        "random_best_method": "contiguous_50_mid",
        "random_best_cosine": phase2_data.get("contiguous_50_mid", {}).get("cosine_similarity", 0.0),
        "random_best_top1": phase2_data.get("contiguous_50_mid", {}).get("top1_agreement", 0.0),
    })

    summary_final = {
        "num_layers": num_layers,
        "adjacent_similarity_mean": float(np.mean(adjacent_sim)),
        "adjacent_similarity_min": float(np.min(adjacent_sim)),
        "adjacent_similarity_max": float(np.max(adjacent_sim)),
        "correlations": {
            "cka_vs_cosine_pearson_r": correlations["cka_vs_cosine"]["pearson_r"],
            "cka_vs_top1_pearson_r": correlations["cka_vs_top1"]["pearson_r"],
            "cka_vs_neg_kl_pearson_r": correlations["cka_vs_neg_kl"]["pearson_r"],
        },
        "comparison_table": comparison_table,
        "cka_benchmarks": {
            name: {
                "layers_kept": val["layers_kept"],
                "speedup": val["speedup"],
                "tok_per_s": val["speed"]["tokens_per_second"]["mean"],
                "cosine": val["quality"]["avg_cosine_similarity"],
                "top1": val["quality"]["avg_top1_agreement"],
                "kl": val["quality"]["avg_kl_divergence"],
            }
            for name, val in cka_benchmarks.items()
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_final, f, indent=2)

    logger.info("\n" + "=" * 90)
    logger.info("PHASE 3 SUMMARY — CKA VS BASELINES COMPARISON")
    logger.info("=" * 90)
    for row in comparison_table:
        logger.info(
            f"Budget {row['budget']:>4} | "
            f"CKA: Cosine={row['cka_cosine']:.4f}, Top1={row['cka_top1']:.1%}, Speedup={row['cka_speedup']:.2f}x | "
            f"Static: Cosine={row['static_cosine']:.4f}, Top1={row['static_top1']:.1%} | "
            f"Best Baseline: Cosine={row['random_best_cosine']:.4f}, Top1={row['random_best_top1']:.1%}"
        )
    logger.info(f"\nAll Phase 3 results saved to {output_dir}")


if __name__ == "__main__":
    main()
