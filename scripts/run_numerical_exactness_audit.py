"""Empirical verification of 4-bit quantization numerical stability vs algorithmic exactness.

Compares vanilla single-token forward passes directly against batched verification passes
on identical contexts C to isolate and quantify:
  - Delta_logit = max_i |l_i^vanilla - l_i^verify|
  - Delta_margin = |m^vanilla - m^verify|
  - Top-1 & Top-2 agreement
  - Argmax flips stratified by true logit margin bins:
      1. margin == 0.0
      2. 0.0 < margin <= 0.125
      3. 0.125 < margin <= 0.25
      4. 0.25 < margin <= 0.50
      5. margin > 0.50
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from zassd.cache.kv_cache import TargetKVCache
from zassd.models.loader import load_model, load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_PROMPTS = [
    "Explain the concept of speculative decoding in large language models.",
    "Write a quicksort implementation in Python with type annotations.",
    "Discuss the trade-offs between model quantization and memory bandwidth.",
    "Summarize the key differences between transformer attention mechanisms.",
    "What are the primary factors affecting GPU power consumption during LLM inference?",
    "Explain how cache hierarchies work in modern computer architecture.",
    "Write a function to check if a binary tree is symmetric in C++.",
    "Describe the process of low-rank adaptation (LoRA) for fine-tuning.",
    "Compare the computational complexity of dense vs sparse attention.",
    "Why does memory-bandwidth bound regime dominate autoregressive decoding?",
    "Implement an LRU cache in Python with O(1) get and put operations.",
    "Explain the mathematics behind Centered Kernel Alignment (CKA) similarity.",
    "How does thermal throttling degrade GPU clock frequencies and memory bandwidth?",
    "Compare float16, bfloat16, and NormalFloat4 (NF4) quantization schemes.",
    "What is the theoretical speedup ceiling of speculative decoding under Amdahl's Law?",
]


def run_exactness_audit(
    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
    num_prompts: int = 15,
    eval_tokens_per_prompt: int = 15,
    device: str = "cuda",
    out_dir: str = "experiments/13_exactness_audit",
) -> dict[str, Any]:
    """Execute direct numerical audit on identical context states."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading model {model_name_or_path} for exactness audit...")
    model = load_model(model_name_or_path, quantize=True, bits=4)
    tokenizer = load_tokenizer(model_name_or_path)
    model.eval()

    audit_records: list[dict[str, Any]] = []

    # Bin definitions
    bin_labels = [
        "margin == 0.0",
        "0.0 < margin <= 0.125",
        "0.125 < margin <= 0.25",
        "0.25 < margin <= 0.50",
        "margin > 0.50",
    ]
    bins_data: dict[str, dict[str, Any]] = {
        label: {
            "total_tokens": 0,
            "argmax_flips": 0,
            "top1_matches": 0,
            "top2_matches": 0,
            "max_delta_logit": 0.0,
            "sum_delta_logit": 0.0,
            "sum_delta_margin": 0.0,
            "delta_logits": [],
        }
        for label in bin_labels
    }

    def get_bin(margin_val: float) -> str:
        if margin_val == 0.0:
            return "margin == 0.0"
        elif 0.0 < margin_val <= 0.125:
            return "0.0 < margin <= 0.125"
        elif 0.125 < margin_val <= 0.25:
            return "0.125 < margin <= 0.25"
        elif 0.25 < margin_val <= 0.50:
            return "0.25 < margin <= 0.50"
        else:
            return "margin > 0.50"

    prompts = BENCHMARK_PROMPTS[:num_prompts]

    for p_idx, prompt in enumerate(prompts):
        logger.info(f"Auditing prompt #{p_idx+1}/{len(prompts)}: '{prompt[:45]}...'")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"]

        # Prefill canonical Target KV
        target_kv = TargetKVCache()
        with torch.no_grad():
            prefill_out = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)

        curr_logit = prefill_out.logits[0, -1, :].float()
        curr_token = int(curr_logit.argmax(dim=-1).item())

        for step in range(eval_tokens_per_prompt):
            # We want to compare two paths on the EXACT same context C:
            # Path A: Vanilla single-token forward pass: input [curr_token]
            # Path B: Batched verification forward pass: input [curr_token, dummy_candidate_1, dummy_candidate_2]
            # But we must ensure target_kv is identical before both.

            # Fork a clone of the cache for Path A
            cache_a = target_kv.fork_ephemeral_draft_kv()
            token_tensor = torch.tensor([[curr_token]], device=device)

            with torch.no_grad():
                out_vanilla = model(token_tensor, past_key_values=cache_a, use_cache=True)
            v_logits = out_vanilla.logits[0, -1, :].float()

            # For Path B: simulate batched verification with candidate sequence of length 3 (K=2)
            # Candidate tokens could be arbitrary or top-k
            top2_candidates = torch.topk(v_logits, k=2).indices.tolist()
            cand_tokens = [curr_token] + top2_candidates
            cand_tensor = torch.tensor([cand_tokens], device=device)

            cache_b = target_kv.fork_ephemeral_draft_kv()
            with torch.no_grad():
                out_verify = model(cand_tensor, past_key_values=cache_b, use_cache=True)
            # Logit at index 0 corresponds to prediction for curr_token on context C
            ver_logits = out_verify.logits[0, 0, :].float()

            # Compute metrics
            abs_diff = torch.abs(v_logits - ver_logits)
            max_delta_logit = float(abs_diff.max().item())
            mean_delta_logit = float(abs_diff.mean().item())

            # Top-1 & Top-2 for vanilla
            top2_v = torch.topk(v_logits, k=2)
            v_top1_idx = int(top2_v.indices[0].item())
            v_top2_idx = int(top2_v.indices[1].item())
            v_top1_val = float(top2_v.values[0].item())
            v_top2_val = float(top2_v.values[1].item())
            v_margin = float(v_top1_val - v_top2_val)

            # Top-1 & Top-2 for verify
            top2_ver = torch.topk(ver_logits, k=2)
            ver_top1_idx = int(top2_ver.indices[0].item())
            ver_top2_idx = int(top2_ver.indices[1].item())
            ver_top1_val = float(top2_ver.values[0].item())
            ver_top2_val = float(top2_ver.values[1].item())
            ver_margin = float(ver_top1_val - ver_top2_val)

            delta_margin = abs(v_margin - ver_margin)
            top1_match = (v_top1_idx == ver_top1_idx)
            top2_match = (v_top2_idx == ver_top2_idx)
            argmax_flip = not top1_match

            # Cosine similarity
            cos_sim = float(F.cosine_similarity(v_logits.unsqueeze(0), ver_logits.unsqueeze(0)).item())

            # Update bin stats
            b_label = get_bin(v_margin)
            b = bins_data[b_label]
            b["total_tokens"] += 1
            if top1_match:
                b["top1_matches"] += 1
            if top2_match:
                b["top2_matches"] += 1
            if argmax_flip:
                b["argmax_flips"] += 1
            b["max_delta_logit"] = max(b["max_delta_logit"], max_delta_logit)
            b["sum_delta_logit"] += max_delta_logit
            b["sum_delta_margin"] += delta_margin
            b["delta_logits"].append(max_delta_logit)

            record = {
                "prompt_id": p_idx,
                "step": step,
                "v_top1_idx": v_top1_idx,
                "v_top1_token": tokenizer.decode([v_top1_idx]),
                "ver_top1_idx": ver_top1_idx,
                "ver_top1_token": tokenizer.decode([ver_top1_idx]),
                "v_margin": round(v_margin, 4),
                "ver_margin": round(ver_margin, 4),
                "delta_margin": round(delta_margin, 4),
                "max_delta_logit": round(max_delta_logit, 4),
                "mean_delta_logit": round(mean_delta_logit, 6),
                "cosine_similarity": round(cos_sim, 6),
                "top1_match": top1_match,
                "argmax_flip": argmax_flip,
                "bin": b_label,
            }
            audit_records.append(record)

            # Advance canonical target_kv with curr_token using single step
            with torch.no_grad():
                model(token_tensor, past_key_values=target_kv.cache, use_cache=True)
            curr_token = v_top1_idx
            if curr_token == tokenizer.eos_token_id:
                break

    # Summarize bin results
    summary_by_bin = {}
    for label, b in bins_data.items():
        n = b["total_tokens"]
        if n > 0:
            summary_by_bin[label] = {
                "total_tokens": n,
                "argmax_flips": b["argmax_flips"],
                "flip_rate_pct": round(b["argmax_flips"] / n * 100.0, 2),
                "agreement_rate_pct": round(b["top1_matches"] / n * 100.0, 2),
                "top2_agreement_pct": round(b["top2_matches"] / n * 100.0, 2),
                "max_delta_logit": round(b["max_delta_logit"], 4),
                "mean_max_delta_logit": round(b["sum_delta_logit"] / n, 4),
                "mean_delta_margin": round(b["sum_delta_margin"] / n, 4),
            }
        else:
            summary_by_bin[label] = {
                "total_tokens": 0,
                "argmax_flips": 0,
                "flip_rate_pct": 0.0,
                "agreement_rate_pct": 100.0,
                "max_delta_logit": 0.0,
                "mean_max_delta_logit": 0.0,
            }

    total_evaluated = len(audit_records)
    total_flips = sum(r["argmax_flip"] for r in audit_records)
    mean_cos = float(np.mean([r["cosine_similarity"] for r in audit_records]))
    overall_max_delta = float(np.max([r["max_delta_logit"] for r in audit_records]))
    overall_mean_delta = float(np.mean([r["mean_delta_logit"] for r in audit_records]))

    final_results = {
        "model": model_name_or_path,
        "total_evaluated_tokens": total_evaluated,
        "total_argmax_flips": total_flips,
        "overall_agreement_rate_pct": round((total_evaluated - total_flips) / total_evaluated * 100.0, 2),
        "mean_logit_cosine_similarity": round(mean_cos, 6),
        "overall_max_abs_logit_diff": round(overall_max_delta, 4),
        "overall_mean_abs_logit_diff": round(overall_mean_delta, 6),
        "stratification_by_margin_bins": summary_by_bin,
        "sample_divergence_records": [r for r in audit_records if r["argmax_flip"]][:15],
    }

    out_file = Path(out_dir) / "numerical_exactness_audit.json"
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Audit results successfully written to {out_file}")

    print("\n" + "=" * 80)
    print(f"NUMERICAL EXACTNESS AUDIT SUMMARY ({model_name_or_path})")
    print("=" * 80)
    print(f"Total Evaluated Tokens: {total_evaluated}")
    print(f"Mean Logit Cosine Similarity: {mean_cos:.6f}")
    print(f"Max Absolute Logit Diff (Delta_logit): {overall_max_delta:.4f}")
    print(f"Overall Agreement Rate: {final_results['overall_agreement_rate_pct']}%\n")
    print(f"{'Logit Margin Bin':<28} | {'Count':<6} | {'Flips':<6} | {'Agreement':<10} | {'Max Delta':<10}")
    print("-" * 75)
    for label, res in summary_by_bin.items():
        print(
            f"{label:<28} | {res['total_tokens']:<6} | {res['argmax_flips']:<6} | "
            f"{res['agreement_rate_pct']:>8.1f}% | {res['max_delta_logit']:>8.4f}"
        )
    print("=" * 80)

    return final_results


if __name__ == "__main__":
    run_exactness_audit()
