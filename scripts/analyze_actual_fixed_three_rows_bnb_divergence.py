#!/usr/bin/env python3
"""Locate first BNB token divergence in corrected fixed-three-row run artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel"


def main() -> None:
    records = []
    for path in sorted(BASE.glob("actual_fixed_three_rows_*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "block" not in data or "timing" not in data or "parameters" not in data:
            continue
        if data["parameters"].get("quantized_linear_strategy") != "triton_nf4_fixed_three_rows_group_scale_packed":
            continue
        block = data["block"].get("token_ids", [])
        rounds = data["timing"].get("paired_rounds", [])
        if not block or not rounds:
            continue
        bnb = rounds[0]["methods"]["original_bnb_greedy"].get("token_ids", [])
        if any(r["methods"]["original_bnb_greedy"].get("token_ids", []) != bnb for r in rounds):
            raise ValueError(f"Original BNB output changed between paired rounds: {path.name}")
        common = 0
        for left, right in zip(block, bnb):
            if left != right:
                break
            common += 1
        mismatch_count = sum(left != right for left, right in zip(block, bnb))
        budget = int(data["parameters"]["max_new_tokens"])
        records.append(
            {
                "artifact": path.name,
                "prompt_id": data["prompt_id"],
                "session": data["session"],
                "budget": budget,
                "block_tokens": len(block),
                "bnb_tokens": len(bnb),
                "full_token_match_bnb": block == bnb,
                "first_divergence_index_zero_based": None if block == bnb else common,
                "common_prefix_tokens": common,
                "mismatches_in_shared_length": mismatch_count,
                "shared_length": min(len(block), len(bnb)),
                "mismatch_fraction_in_shared_length": (
                    mismatch_count / min(len(block), len(bnb)) if min(len(block), len(bnb)) else None
                ),
                "first_divergence_tokens": (
                    {"fixed_three_row": block[common], "original_bnb": bnb[common]}
                    if block != bnb and common < min(len(block), len(bnb))
                    else None
                ),
                "b1_token_match": bool(data.get("token_match")),
                "b1_state_prefixes_match": int(data.get("shared_state_prefixes", 0)),
            }
        )

    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in records:
        groups[(record["budget"], record["prompt_id"])].append(record)
    grouped = []
    for (budget, prompt), items in sorted(groups.items()):
        first = [x["first_divergence_index_zero_based"] for x in items if x["first_divergence_index_zero_based"] is not None]
        grouped.append(
            {
                "budget": budget,
                "prompt_id": prompt,
                "sessions": [x["session"] for x in items],
                "runs": len(items),
                "bnb_full_matches": sum(x["full_token_match_bnb"] for x in items),
                "first_divergence_indices_zero_based": first,
                "divergence_index_range": [min(first), max(first)] if first else None,
                "divergence_positions_stable_across_sessions": len(first) == len(items) and len(set(first)) == 1,
            }
        )

    by_budget = {}
    for budget in sorted({r["budget"] for r in records}):
        subset = [r for r in records if r["budget"] == budget]
        divergences = [r["first_divergence_index_zero_based"] for r in subset if r["first_divergence_index_zero_based"] is not None]
        by_budget[str(budget)] = {
            "runs": len(subset),
            "full_bnb_token_matches": sum(r["full_token_match_bnb"] for r in subset),
            "diverged_runs": len(divergences),
            "first_divergence_min": min(divergences) if divergences else None,
            "first_divergence_median": sorted(divergences)[len(divergences) // 2] if divergences else None,
            "first_divergence_max": max(divergences) if divergences else None,
        }

    output = BASE / "actual_fixed_three_rows_bnb_divergence_analysis.json"
    result = {
        "classification": "token-level empirical comparison against original BNB; not a numerical proof",
        "method": "Compare corrected fixed-three-row block token IDs to original BNB greedy IDs in each paired artifact; report first mismatch and shared-length mismatch count.",
        "scope": {
            "corrected_runs": len(records),
            "budgets": sorted({r["budget"] for r in records}),
            "development_only": True,
            "held_out_used": False,
        },
        "by_budget": by_budget,
        "prompt_budget_groups": grouped,
        "per_run": records,
        "limitations": [
            "A token mismatch locates an output divergence but does not identify the responsible operator or logit margin.",
            "This analysis uses finite development generations on one GPU.",
            "B1 exactness and original BNB identity are distinct comparisons.",
            "Held-out GSM8K indices 300–349 were not used.",
        ],
    }
    output.write_text(json.dumps(result, indent=2) + "\n")

    manifest_path = BASE / "capability_gate_smoke_experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence"]["corrected_fixed_three_row_bnb_divergence_analysis"] = output.name
    manifest["source_sha256"]["scripts/analyze_actual_fixed_three_rows_bnb_divergence.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(
        json.dumps(
            {
                "summary": str(output),
                "by_budget": by_budget,
                "prompt_budget_groups": grouped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
