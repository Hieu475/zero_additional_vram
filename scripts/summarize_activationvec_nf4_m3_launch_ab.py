#!/usr/bin/env python3
"""Summarize the paired default-vs-tuned NF4 M=3 launch A/B artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifacts", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    rows = [json.loads(path.read_text()) for path in args.artifacts]
    if not rows:
        raise ValueError("at least one A/B artifact is required")
    exact = all(
        all(value["tokens_match_b1_singleton"] and value["state_domains_equal"]
            and not value["state_mismatches"] for value in row["exactness_preflight"].values())
        and all(count == len(row["paired_rounds"]) for count in row["timed_token_match_rounds"].values())
        for row in rows
    )
    ratios = [row["ratio_geomeans"]["tuned_over_default_speedup"] for row in rows]
    payload = {
        "schema": "zassd-activationvec-nf4-m3-launch-ab-summary-v1",
        "classification": "selected development prompt clusters; descriptive only",
        "prompt_count": len(rows),
        "all_default_and_tuned_runs_exact_to_b1": exact,
        "paired_rounds_per_prompt": [len(row["paired_rounds"]) for row in rows],
        "tuned_over_default_speedup_geomean": math.prod(ratios) ** (1 / len(ratios)),
        "prompts": [
            {"prompt_id": row["prompt_id"],
             "tuned_over_default_speedup": row["ratio_geomeans"]["tuned_over_default_speedup"],
             "default_over_original_bnb": row["ratio_geomeans"]["default_over_bnb_speedup"],
             "tuned_over_original_bnb": row["ratio_geomeans"]["tuned_over_bnb_speedup"],
             "tuned_over_b1_singleton": row["ratio_geomeans"]["tuned_over_singleton_speedup"],
             "timed_token_matches": row["timed_token_match_rounds"],
             "artifact": str(path),
             "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for row, path in zip(rows, args.artifacts)
        ],
        "interpretation": "The shape-tuned launch is retained as exact but is not faster end to end in this two-prompt screen; preserve the default launch pending broader evidence.",
        "scope_limits": ["two selected HumanEval prompts", "one process per prompt", "single GPU",
                         "does not establish original-BNB KV identity", "GSM8K held-out indices 300-349 excluded"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
