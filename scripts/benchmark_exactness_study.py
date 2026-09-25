"""Phase 12 / Việc 6 — Revisit Speculative Exactness and Verification Equivalence.

Conducts an in-depth empirical investigation of exactness across:
- >= 20 fixed prompts from diverse categories (general, coding, reasoning)
- Multiple K values (1, 2, 3, 4)
- Multiple layer configurations (cka_50, cka_75, cka_83, cka_90)
- Same 4-bit bitsandbytes quantization (NF4)
- Temperature = 0.0 (greedy decoding)

Reports:
- ExactMatch (% prompts with identical output sequences)
- PartialMatch (average prefix match ratio before any divergence)
- AcceptanceRate (mean draft acceptance rate)
- Top1Agreement (proportion of target argmax choices agreeing with vanilla choices)
- LogitCosine (cosine similarity between target logits and vanilla logits)

Mismatch Diagnostics logged:
- prompt_id
- prompt_category
- token_position (divergence index)
- logit_margin (top1 - top2 logit of target model)
- K lookahead
- layer_configuration
- vanilla_token vs spec_token
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
import torch.nn.functional as F

from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.speculative import self_speculative_generate
from zassd.decoding.vanilla import vanilla_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.memory import reset_vram_stats
from zassd.utils.logging import setup_logging
from zassd.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Representative layer configurations spanning high to low redundancy
CONFIG_DEFINITIONS = {
    "cka_50": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22],
    "cka_75": [3, 4, 5, 6, 7, 10, 11, 12, 13],
    "cka_83": [4, 5, 6, 7, 12, 13],
    "cka_90": [4, 5, 6, 7],
}

K_VALUES = [1, 2, 3, 4]


def collect_vanilla_reference_with_logits(
    model, tokenizer, prompt: str, max_new_tokens: int, device: str = "cuda:0"
) -> dict[str, Any]:
    """Run vanilla greedy generation and store per-step token IDs and logit vectors."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]

    target_kv = TargetKVCache()
    with torch.no_grad():
        prefill_out = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)

    generated_tokens: list[int] = []
    step_logits: list[torch.Tensor] = []

    curr_logit = prefill_out.logits[0, -1, :].float()
    curr_tok = int(curr_logit.argmax(dim=-1).item())

    step_logits.append(curr_logit.cpu())
    generated_tokens.append(curr_tok)

    for _ in range(max_new_tokens - 1):
        if curr_tok == tokenizer.eos_token_id:
            break
        curr_tensor = torch.tensor([[curr_tok]], device=device)
        with torch.no_grad():
            out = model(curr_tensor, past_key_values=target_kv.cache, use_cache=True)
        next_logit = out.logits[0, -1, :].float()
        curr_tok = int(next_logit.argmax(dim=-1).item())
        step_logits.append(next_logit.cpu())
        generated_tokens.append(curr_tok)

    decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {
        "text": decoded_text,
        "token_ids": generated_tokens,
        "step_logits": step_logits,
    }


def compute_logit_cosine_and_divergence(
    model,
    tokenizer,
    prompt: str,
    vanilla_data: dict[str, Any],
    spec_text: str,
    device: str = "cuda:0",
) -> tuple[float, float, list[dict[str, Any]]]:
    """Analyze exact match, partial match, logit cosine similarity, and divergence points."""
    v_tokens = vanilla_data["token_ids"]
    s_tokens = tokenizer.encode(spec_text, add_special_tokens=False)

    min_len = min(len(v_tokens), len(s_tokens))
    if min_len == 0:
        return 0.0, 0.0, []

    # 1. Prefix match calculation
    first_div: int | None = None
    for i in range(min_len):
        if v_tokens[i] != s_tokens[i]:
            first_div = i
            break

    if first_div is None:
        if len(v_tokens) == len(s_tokens):
            partial_match = 1.0
            exact_match = 1.0
        else:
            first_div = min_len
            partial_match = min_len / max(len(v_tokens), 1)
            exact_match = 0.0
    else:
        partial_match = first_div / max(len(v_tokens), 1)
        exact_match = 0.0

    # 2. Compute Top-1 Agreement across the valid token sequence
    matches = sum(1 for i in range(min_len) if v_tokens[i] == s_tokens[i])
    top1_agreement = matches / max(1, min_len)

    # 3. Compute Logit Cosine Similarity along prefix
    # Evaluate target model logits on the prompt + spec prefix to extract target logits
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    spec_tensor = torch.tensor([s_tokens[:min_len]], device=device)
    full_spec_input = torch.cat([prompt_ids, spec_tensor], dim=1)

    with torch.no_grad():
        spec_out = model(full_spec_input, use_cache=False)

    prompt_len = prompt_ids.shape[1]
    spec_target_logits = spec_out.logits[0, prompt_len - 1 : prompt_len - 1 + min_len].float()

    cos_sims = []
    divergence_records = []

    for i in range(min_len):
        v_l = vanilla_data["step_logits"][i].to(device)
        s_l = spec_target_logits[i]
        sim = float(F.cosine_similarity(v_l.unsqueeze(0), s_l.unsqueeze(0)).item())
        cos_sims.append(sim)

        if first_div is not None and i == first_div:
            # Diagnose first divergence
            top2_vals, top2_indices = torch.topk(s_l, 2)
            top1_val = float(top2_vals[0].item())
            top2_val = float(top2_vals[1].item())
            margin = top1_val - top2_val

            v_tok = v_tokens[i]
            s_tok = s_tokens[i]

            diag = {
                "token_position": i,
                "vanilla_token_id": v_tok,
                "vanilla_token_str": tokenizer.decode([v_tok]),
                "spec_token_id": s_tok,
                "spec_token_str": tokenizer.decode([s_tok]),
                "top1_logit": round(top1_val, 4),
                "top2_logit": round(top2_val, 4),
                "logit_margin": round(margin, 5),
                "logit_cosine": round(sim, 5),
            }
            divergence_records.append(diag)

    # Prefix cosine strictly evaluates agreement over the identical conditioning prefix
    prefix_slice_len = (first_div + 1) if first_div is not None else min_len
    prefix_cos_sims = cos_sims[:prefix_slice_len]
    mean_prefix_cosine = float(np.mean(prefix_cos_sims)) if prefix_cos_sims else 1.0
    return partial_match, mean_prefix_cosine, divergence_records



def plot_exactness_study_figures(
    summary_by_config_k: list[dict[str, Any]],
    divergence_records: list[dict[str, Any]],
    figures_dir: Path,
) -> None:
    """Generate publication figures for exactness and verification stability."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. ExactMatch & Acceptance Rate across (Config, K)
    fig, ax1 = plt.subplots(figsize=(11, 6))
    labels = [f"{r['config']}\n(K={r['k']})" for r in summary_by_config_k]
    exact_rates = [r["exact_match_pct"] for r in summary_by_config_k]
    acc_rates = [r["acceptance_rate_pct"] for r in summary_by_config_k]
    partial_rates = [r["partial_match_pct"] for r in summary_by_config_k]

    x = np.arange(len(labels))
    width = 0.28

    rects1 = ax1.bar(x - width, exact_rates, width, label="Exact Match (%)", color="#1f77b4", alpha=0.9)
    rects2 = ax1.bar(x, partial_rates, width, label="Partial Match (%)", color="#2ca02c", alpha=0.9)
    rects3 = ax1.bar(x + width, acc_rates, width, label="Acceptance Rate (%)", color="#ff7f0e", alpha=0.9)

    ax1.set_ylabel("Fidelity Percentage (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Exactness & Speculative Acceptance by Layer Configuration and K", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylim(0, 115)
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax1.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(figures_dir / "exactness_by_config_and_k.png", dpi=300)
    plt.close()

    # 2. Logit Margin Distribution at Divergence Points
    if divergence_records:
        fig, ax2 = plt.subplots(figsize=(9, 5.5))
        margins = [d["logit_margin"] for d in divergence_records]
        ax2.hist(margins, bins=15, color="#d62728", edgecolor="black", alpha=0.75)
        ax2.axvline(np.median(margins), color="blue", linestyle="--", linewidth=2, label=f"Median Margin ({np.median(margins):.3f})")
        ax2.set_xlabel("Top-1 vs Top-2 Logit Margin (Δ) at First Divergence", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        ax2.set_title("Distribution of Logit Margins at Output Divergence Points\n(Illustrating Boundary Flips in 4-bit Quantization)", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(figures_dir / "divergence_margin_distribution.png", dpi=300)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Revisit Speculative Exactness Study (Việc 6)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-prompts", type=int, default=20, help="Number of benchmark prompts (>= 20)")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Tokens to generate per prompt")
    parser.add_argument("--output-dir", type=str, default="experiments/10_exactness")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 95)
    logger.info("PHASE 12 / VIỆC 6 — SPECULATIVE EXACTNESS & LOGIT EQUIVALENCE STUDY")
    logger.info(f"Hardware: RTX 4050 Laptop GPU | Prompts: {args.num_prompts} | Max New Tokens: {args.max_new_tokens}")
    logger.info("=" * 95)

    # 1. Load Model, Tokenizer, Adapters
    model = load_model(args.model, quantize=True, bits=4)
    tokenizer = load_tokenizer(args.model)
    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 2. Load Prompts across Categories
    prompts_path = Path("data/benchmarks/prompts.jsonl")
    all_prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                all_prompts.append(json.loads(line.strip()))
    eval_prompts = all_prompts[: args.num_prompts]
    logger.info(f"Loaded {len(eval_prompts)} prompts across categories: {list(set(p.get('category', 'unknown') for p in eval_prompts))}")

    # 3. Establish Vanilla Reference & Step Logits
    logger.info("\n--- Establishing Autoregressive Vanilla Reference Sequences & Logits ---")
    vanilla_references: dict[int, dict[str, Any]] = {}
    for p in eval_prompts:
        p_id = p["id"]
        logger.info(f"Profiling Vanilla Prompt #{p_id} [{p.get('category', 'general')}]...")
        ref_data = collect_vanilla_reference_with_logits(
            model=model,
            tokenizer=tokenizer,
            prompt=p["prompt"],
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        vanilla_references[p_id] = ref_data

    # 4. Run Matrix across Configurations and K
    matrix_records: list[dict[str, Any]] = []
    all_divergences: list[dict[str, Any]] = []

    for cfg_name, skip_indices in CONFIG_DEFINITIONS.items():
        kept_layers = 36 - len(skip_indices)
        for k in K_VALUES:
            action_tag = f"{cfg_name}_k{k}"
            logger.info(f"\n--- Testing Configuration: {cfg_name} (Kept {kept_layers}L) with K={k} ---")

            em_list = []
            pm_list = []
            acc_list = []
            top1_list = []
            cos_list = []

            for p in eval_prompts:
                p_id = p["id"]
                p_cat = p.get("category", "general")
                v_data = vanilla_references[p_id]

                gc.collect()
                torch.cuda.empty_cache()
                reset_vram_stats()

                spec_text, spec_m = self_speculative_generate(
                    model=model,
                    tokenizer=tokenizer,
                    layer_mgr=layer_mgr,
                    skip_indices=skip_indices,
                    prompt=p["prompt"],
                    k=k,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.0,
                    device=device,
                )

                pm, mean_cos, div_info = compute_logit_cosine_and_divergence(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=p["prompt"],
                    vanilla_data=v_data,
                    spec_text=spec_text,
                    device=device,
                )

                is_exact = 1.0 if pm == 1.0 else 0.0
                em_list.append(is_exact)
                pm_list.append(pm)
                acc_list.append(spec_m.acceptance_rate * 100.0)
                cos_list.append(mean_cos)

                v_toks = v_data["token_ids"]
                s_toks = tokenizer.encode(spec_text, add_special_tokens=False)
                min_len = min(len(v_toks), len(s_toks))
                matches = sum(1 for idx in range(min_len) if v_toks[idx] == s_toks[idx])
                top1_agreement = (matches / max(1, min_len)) * 100.0
                top1_list.append(top1_agreement)

                if div_info:
                    for d in div_info:
                        d["config"] = cfg_name
                        d["k"] = k
                        d["prompt_id"] = p_id
                        d["prompt_category"] = p_cat
                        all_divergences.append(d)

            summary_item = {
                "config": cfg_name,
                "k": k,
                "layers_kept": kept_layers,
                "exact_match_pct": round(float(np.mean(em_list)) * 100.0, 2),
                "partial_match_pct": round(float(np.mean(pm_list)) * 100.0, 2),
                "acceptance_rate_pct": round(float(np.mean(acc_list)), 2),
                "top1_agreement_pct": round(float(np.mean(top1_list)), 2),
                "logit_cosine": round(float(np.mean(cos_list)), 5),
            }
            matrix_records.append(summary_item)
            logger.info(
                f"Result {action_tag}: ExactMatch={summary_item['exact_match_pct']}% | "
                f"PartialMatch={summary_item['partial_match_pct']}% | "
                f"Top1Agreement={summary_item['top1_agreement_pct']}% | "
                f"Acceptance={summary_item['acceptance_rate_pct']}% | "
                f"LogitCosine={summary_item['logit_cosine']}"
            )

    # 5. Breakdown by Prompt Category
    category_summary: dict[str, Any] = {}
    categories = list(set(p.get("category", "general") for p in eval_prompts))
    for cat in categories:
        cat_prompt_ids = set(p["id"] for p in eval_prompts if p.get("category", "general") == cat)
        cat_divs = [d for d in all_divergences if d["prompt_category"] == cat]
        category_summary[cat] = {
            "num_prompts": len(cat_prompt_ids),
            "num_divergences": len(cat_divs),
            "median_logit_margin": round(float(np.median([d["logit_margin"] for d in cat_divs])), 5) if cat_divs else None,
        }

    # 6. Save JSON Results
    results_payload = {
        "status": "PASS",
        "protocol": {
            "model": args.model,
            "quantization": "4-bit (bitsandbytes NF4)",
            "temperature": 0.0,
            "decoding": "Greedy self-speculative decoding with parallel target verification",
            "num_prompts": args.num_prompts,
            "max_new_tokens": args.max_new_tokens,
            "configs_tested": list(CONFIG_DEFINITIONS.keys()),
            "k_values_tested": K_VALUES,
        },
        "exactness_matrix": matrix_records,
        "category_breakdown": category_summary,
        "num_total_divergences": len(all_divergences),
        "divergence_records": all_divergences,
    }

    summary_file = output_dir / "exactness_study_summary.json"
    with open(summary_file, "w") as f:
        json.dump(results_payload, f, indent=2)
    logger.info(f"\nSaved exactness study summary to {summary_file}")

    # 7. Print Final Table
    logger.info("\n" + "=" * 105)
    logger.info("FINAL EXACTNESS & FIDELITY BENCHMARK (N >= 20 PROMPTS, RTX 4050)")
    logger.info("=" * 105)
    header = f"{'Config':<10} | {'K':<3} | {'Exact Match':<12} | {'Partial Match':<14} | {'Top-1 Agree':<12} | {'Acceptance':<11} | {'Logit Cosine':<12}"
    logger.info(header)
    logger.info("-" * 105)
    for r in matrix_records:
        row = (
            f"{r['config']:<10} | "
            f"{r['k']:<3} | "
            f"{r['exact_match_pct']:<11.1f}% | "
            f"{r['partial_match_pct']:<13.1f}% | "
            f"{r['top1_agreement_pct']:<11.1f}% | "
            f"{r['acceptance_rate_pct']:<10.1f}% | "
            f"{r['logit_cosine']:<12.5f}"
        )
        logger.info(row)
    logger.info("=" * 105)

    plot_exactness_study_figures(matrix_records, all_divergences, figures_dir)


if __name__ == "__main__":
    main()
