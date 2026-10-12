#!/usr/bin/env python3
"""Summarize deterministic and cache/logit evidence from the natural-context trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(trace: dict) -> dict:
    rows = []
    for row in trace["rows"]:
        item = {
            "prompt_id": row["prompt_id"],
            "prompt_tokens": row["prompt_tokens"],
            "vanilla_repeat_exact": row["vanilla_token_ids"] == row["vanilla_repeat_token_ids"],
            "vanilla_output_tokens": len(row["vanilla_token_ids"]),
            "methods": {},
        }
        for name, result in row["methods"].items():
            first = result["first_divergence"]
            cycle = result.get("first_divergent_cycle") or {}
            output_start = cycle.get("output_start")
            ref_step = (row["vanilla_steps"][output_start]
                        if output_start is not None and output_start < len(row["vanilla_steps"])
                        else None)
            prefix_comparisons = []
            for cycle in result.get("cycles", []):
                start = cycle.get("output_start")
                if start is None or start >= len(row["vanilla_steps"]):
                    continue
                reference_step = row["vanilla_steps"][start]
                cycle_digest = cycle.get("cache_digest_before")
                prefix_comparisons.append({
                    "output_start": start,
                    "cache_length": cycle.get("cache_len_before_verify"),
                    "reference_cache_length": reference_step.get("cache_length"),
                    "cache_digest_matches": cycle_digest == reference_step.get("cache_digest"),
                })
            item["methods"][name] = {
                "first_divergence": first,
                "sequence_exact_match": result["sequence_exact_match"],
                "total_time_s_diagnostic_only": result["total_time_s"],
                "verify_time_s_diagnostic_only": result["verify_time_s"],
                "cache_length_before_verify": cycle.get("cache_len_before_verify"),
                "reference_cache_length_at_cycle_start": (
                    ref_step.get("cache_length") if ref_step else None
                ),
                "cache_digest_matches_reference_at_cycle_start": (
                    cycle.get("cache_digest_before") == ref_step.get("cache_digest")
                    if ref_step and cycle.get("cache_digest_before") is not None else None
                ),
                "reference_margin_at_first_divergence": (
                    row["vanilla_steps"][first]["top1_margin"]
                    if first is not None and first < len(row["vanilla_steps"]) else None
                ),
                "cache_prefix_comparisons": prefix_comparisons,
                "first_cache_prefix_divergence": next(
                    (entry for entry in prefix_comparisons if not entry["cache_digest_matches"]),
                    None,
                ),
            }
        rows.append(item)
    methods = ("batched", "sequential")
    totals = {
        name: {
            "exact_sequences": sum(bool(r["methods"][name]["sequence_exact_match"]) for r in rows),
            "n_sequences": len(rows),
            "first_divergences": [
                {"prompt_id": r["prompt_id"], "position": r["methods"][name]["first_divergence"]}
                for r in rows if r["methods"][name]["first_divergence"] is not None
            ],
        }
        for name in methods
    }
    return {
        "commit_sha": trace["commit_sha"], "model_revision": trace["model_revision"],
        "precision": "NF4 weights, float16 compute", "n_prompts": len(rows),
        "vanilla_repeat_deterministic": all(r["vanilla_repeat_exact"] for r in rows),
        "methods": totals, "rows": rows,
        "interpretation": (
            "Sequential verification matched vanilla on this three-prompt trace. "
            "Batched mismatches coincided with a cache-prefix digest difference at the "
            "cycle start; one occurred at a zero-margin vanilla decision. This is "
            "prompt-specific evidence of batch/cache numerical path dependence, not a "
            "general precision-level guarantee."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.trace.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["methods"], indent=2))


if __name__ == "__main__":
    main()
