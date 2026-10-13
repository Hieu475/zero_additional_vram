#!/usr/bin/env python3
"""Summarize first-divergence and same-prefix KV substitution results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def classify(prompt: dict) -> dict:
    token_index = prompt.get("first_token_divergence_index")
    cycle_index = prompt.get("first_token_divergence_cycle")
    row_index = prompt.get("first_token_divergence_logit_row")
    cycle = next((c for c in prompt.get("cycles", [])
                  if c.get("cycle_index") == cycle_index), None)
    result = {
        "prompt_id": prompt["prompt_id"],
        "first_kv_divergence_cycle": prompt.get("first_kv_divergence_cycle"),
        "first_logit_divergence_cycle": prompt.get("first_logit_divergence_cycle"),
        "first_token_divergence_index": token_index,
        "first_token_divergence_cycle": cycle_index,
        "vanilla_top1_margin_at_divergence": prompt.get("vanilla_top1_margin_at_divergence"),
        "classification": "no_token_divergence_within_budget",
    }
    if token_index is None:
        return result
    if cycle is None or row_index is None or row_index >= len(cycle["canonical_replay_token_ids"]):
        result["classification"] = "unresolved_cycle_mapping"
        return result
    expected = prompt["vanilla_token_ids"][token_index]
    active = cycle["observed_batch_token_ids"][row_index]
    canonical = cycle["canonical_replay_token_ids"][row_index]
    prefix_same = cycle["prefix_matches_vanilla"]
    result.update({
        "verification_mode": cycle.get("verification_mode"),
        "same_token_prefix_as_vanilla": prefix_same,
        "logit_row": row_index,
        "vanilla_token_id": expected,
        "active_kv_argmax": active,
        "canonical_kv_argmax": canonical,
        "first_kv_different_layer": cycle.get("first_kv_different_layer"),
        "kv_max_abs_error": cycle.get("kv_max_abs_error"),
        "logits_max_abs_error": cycle.get("logits_max_abs_error"),
    })
    if not prefix_same:
        result["classification"] = "token_history_already_diverged"
    elif active != expected and canonical == expected:
        result["classification"] = "canonical_kv_substitution_restores_vanilla_argmax"
    elif active != expected and canonical != expected and cycle.get("verification_mode") == "batched":
        result["classification"] = "batch_execution_difference_persists_with_canonical_kv"
    elif active == expected:
        result["classification"] = "token_divergence_not_explained_by_this_logit_row"
    else:
        result["classification"] = "unresolved_or_interacting_effects"
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="experiments/numerical_causality")
    args = ap.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    attributions = [classify(p) for p in source["prompts"]]
    counts = {}
    for item in attributions:
        counts[item["classification"]] = counts.get(item["classification"], 0) + 1
    prompts = source["prompts"]
    summary = {
        "input": args.input,
        "n_prompts": len(prompts),
        "n_token_divergence_within_budget": sum(
            p.get("first_token_divergence_index") is not None for p in prompts),
        "sequence_exact_match_rate_within_budget": (
            sum(p.get("first_token_divergence_index") is None for p in prompts) / len(prompts)
            if prompts else None),
        "n_first_kv_divergence_observed": sum(
            p.get("first_kv_divergence_cycle") is not None for p in prompts),
        "n_first_logit_divergence_observed": sum(
            p.get("first_logit_divergence_cycle") is not None for p in prompts),
        "causal_classification_counts": counts,
        "interpretation": (
            "Canonical replay uses prompt prefill followed by token-at-a-time cache construction. "
            "At the first output divergence, the canonical cache is replayed with the same batched "
            "candidate inputs. A restored vanilla argmax supports a cache-history contribution; "
            "a remaining difference shows that canonical KV alone does not explain the flip."
        ),
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "first_divergence_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (out / "causal_attribution.json").write_text(
        json.dumps({"cases": attributions}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
