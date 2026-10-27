#!/usr/bin/env python3
"""Summarize how often recorded KV drift precedes a token divergence.

Reads existing multicycle traces. Reports descriptive counts only; diagnostic
cohorts selected for failures cannot estimate population failure rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_prompts(prompts: list[dict]) -> dict:
    rows = []
    for prompt in prompts:
        kv_cycle = prompt.get("first_kv_divergence_cycle")
        token_cycle = prompt.get("first_token_divergence_cycle")
        cycles = prompt.get("cycles", [])
        aligned_before_flip = [
            cycle for cycle in cycles
            if cycle.get("prefix_matches_vanilla")
            and (token_cycle is None or cycle["cycle_index"] < token_cycle)
        ]
        drift_aligned_before_flip = [
            cycle for cycle in aligned_before_flip
            if (cycle.get("first_kv_different_layer") is not None
                or cycle.get("cache_length_mismatch_count", 0) > 0)
        ]
        rows.append({
            "prompt_id": prompt["prompt_id"],
            "first_kv_divergence_cycle": kv_cycle,
            "first_token_divergence_cycle": token_cycle,
            "first_token_divergence_index": prompt.get("first_token_divergence_index"),
            "kv_before_token_flip": (kv_cycle < token_cycle
                                     if kv_cycle is not None and token_cycle is not None
                                     else None),
            "same_prefix_cycles_before_flip": len(aligned_before_flip),
            "same_prefix_cycles_with_kv_drift_before_flip": len(drift_aligned_before_flip),
        })
    flipped = [row for row in rows if row["first_token_divergence_cycle"] is not None]
    unflipped = [row for row in rows if row["first_token_divergence_cycle"] is None]
    return {
        "n_prompts": len(rows),
        "n_token_divergent": len(flipped),
        "n_with_kv_drift_before_token_flip": sum(row["kv_before_token_flip"] is True for row in flipped),
        "n_with_kv_drift_same_cycle_or_later": sum(row["kv_before_token_flip"] is False for row in flipped),
        "n_without_token_divergence_but_with_kv_drift": sum(
            row["first_kv_divergence_cycle"] is not None for row in unflipped),
        "aligned_cycles_before_flip": sum(row["same_prefix_cycles_before_flip"] for row in rows),
        "aligned_cycles_with_kv_drift_before_flip": sum(
            row["same_prefix_cycles_with_kv_drift_before_flip"] for row in rows),
        "rows": rows,
        "scope": "descriptive diagnostic trace; first-KV and first-token cycle indices are not a causal attribution",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="experiments/numerical_causality/traces/multicycle_nf4_sdpa_static.json")
    ap.add_argument("--output", default="experiments/exact_quantized_feasibility/legacy_drift_lead_summary.json")
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = summarize_prompts(data["prompts"])
    report["input"] = args.input
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
