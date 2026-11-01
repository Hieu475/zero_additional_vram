#!/usr/bin/env python3
"""Summarize same-process streaming-vs-cached NF4 activation-load A/Bs."""

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
    runs = [json.loads(path.read_text()) for path in args.artifacts]
    grouped = {}
    for run, path in zip(runs, args.artifacts):
        grouped.setdefault(run["prompt_id"], []).append((run, path))
    prompt_rows = []
    for prompt_id, entries in grouped.items():
        process_ratios = [run["ratio_geomeans"]["tuned_over_default_speedup"] for run, _ in entries]
        prompt_rows.append({
            "prompt_id": prompt_id,
            "independent_processes": len(entries),
            "cached_over_streaming_speedup_geomean": math.prod(process_ratios) ** (1 / len(process_ratios)),
            "process_speedups": process_ratios,
            "cached_over_original_bnb_geomean": math.prod(
                run["ratio_geomeans"]["tuned_over_bnb_speedup"] for run, _ in entries
            ) ** (1 / len(entries)),
            "cached_over_b1_singleton_geomean": math.prod(
                run["ratio_geomeans"]["tuned_over_singleton_speedup"] for run, _ in entries
            ) ** (1 / len(entries)),
            "all_timed_token_matches": all(
                all(count == len(run["paired_rounds"]) for count in run["timed_token_match_rounds"].values())
                for run, _ in entries
            ),
            "artifacts": [{"path": str(path),
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                          for _, path in entries],
        })
    ratios = [row["cached_over_streaming_speedup_geomean"] for row in prompt_rows]
    exact = all(
        all(item["tokens_match_b1_singleton"] and item["state_domains_equal"]
            and not item["state_mismatches"] for item in run["exactness_preflight"].values())
        and all(n == len(run["paired_rounds"]) for n in run["timed_token_match_rounds"].values())
        for run in runs
    )
    payload = {
        "schema": "zassd-activationvec-nf4-cache-policy-ab-summary-v1",
        "classification": "selected development prompt clusters; descriptive only",
        "cache_policy": {"default": "ld.global.cs.v4 activation loads",
                         "candidate": "ld.global.ca.v4 activation loads",
                         "NF4 weight loads": "ld.global.cs.v4 unchanged"},
        "prompt_count": len(prompt_rows),
        "independent_process_count": len(runs),
        "all_runs_exact_to_b1_and_timed_tokens": exact,
        "paired_rounds_per_prompt": sorted({len(run["paired_rounds"]) for run in runs}),
        "cached_over_streaming_speedup_geomean": math.prod(ratios) ** (1 / len(ratios)),
        "prompts": prompt_rows,
        "interpretation": "The cache-all activation loads preserved B1 identity, but the aggregate is near parity and effects vary by prompt. Keep it experimental; these selected clusters do not estimate population-level prompt uncertainty.",
        "processes_per_prompt": {row["prompt_id"]: row["independent_processes"] for row in prompt_rows},
        "scope_limits": ["five selected prompt clusters across HumanEval and CNN/DailyMail", "one GPU",
                         "no original-BNB KV identity claim", "GSM8K held-out indices 300-349 excluded"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
