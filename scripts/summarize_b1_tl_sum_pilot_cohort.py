#!/usr/bin/env python3
"""Summarize two-process, five-prompt B1 tl.sum development pilots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ARTIFACTS = (
    "b1_tl_sum_humaneval050_052_64_v1/rows.jsonl",
    "b1_tl_sum_humaneval050_052_64_rep2_v1/rows.jsonl",
    "b1_tl_sum_humaneval114_64_v1/rows.jsonl",
    "b1_tl_sum_humaneval114_64_rep2_v1/rows.jsonl",
    "b1_tl_sum_cnndm003_64_v1/rows.jsonl",
    "b1_tl_sum_cnndm003_64_rep2_v1/rows.jsonl",
)
ROOT = Path("experiments/exact_quantized_feasibility")


def geo_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstrap-replicates", type=int, default=20000)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    by_prompt = defaultdict(list)
    input_hashes = {}
    for relative in ARTIFACTS:
        path = ROOT / relative
        input_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                by_prompt[row["id"]].append(row)
    if len(by_prompt) != 5 or any(len(rows) != 2 for rows in by_prompt.values()):
        raise ValueError("expected five prompt clusters with exactly two process runs each")

    prompt_results = {}
    for prompt_id, rows in sorted(by_prompt.items()):
        prompt_results[prompt_id] = {
            "process_runs": len(rows),
            "block_matches_b1_all_runs": all(row["block_matches_b1"] for row in rows),
            "state_exact_all_reference_prefixes_all_runs": all(
                row["state_audit"]["state_exact_at_all_reference_prefixes"] for row in rows
            ),
            "block_matches_original_bnb_runs": sum(
                int(row["block_matches_original_bnb"]) for row in rows
            ),
            "geomean_speedup_vs_b1": geo_mean([row["block_speedup_vs_b1"] for row in rows]),
            "geomean_speedup_vs_original_bnb": geo_mean(
                [row["block_speedup_vs_original"] for row in rows]
            ),
            "process_speedups_vs_b1": [row["block_speedup_vs_b1"] for row in rows],
            "process_speedups_vs_original_bnb": [row["block_speedup_vs_original"] for row in rows],
            "acceptance_rates": [
                row["results"]["b1_block"]["accepted_draft_tokens"] / 64 for row in rows
            ],
        }

    ids = sorted(by_prompt)
    point = {
        "vs_b1": geo_mean([prompt_results[p]["geomean_speedup_vs_b1"] for p in ids]),
        "vs_original_bnb": geo_mean([
            prompt_results[p]["geomean_speedup_vs_original_bnb"] for p in ids
        ]),
    }
    rng = random.Random(20261101)
    boot = {"vs_b1": [], "vs_original_bnb": []}
    for _ in range(args.bootstrap_replicates):
        sampled_ids = rng.choices(ids, k=len(ids))
        sampled_rows = []
        for prompt_id in sampled_ids:
            sampled_rows.extend(rng.choices(by_prompt[prompt_id], k=len(by_prompt[prompt_id])))
        for key, row_key in (
            ("vs_b1", "block_speedup_vs_b1"),
            ("vs_original_bnb", "block_speedup_vs_original"),
        ):
            values = [row[row_key] for row in sampled_rows]
            boot[key].append(geo_mean(values))
    intervals = {}
    for key, values in boot.items():
        values.sort()
        intervals[key] = {
            "lower_95": values[int(0.025 * len(values))],
            "upper_95": values[int(0.975 * len(values)) - 1],
        }

    payload = {
        "schema": "zassd-b1-tl-sum-five-prompt-two-process-summary-v1",
        "classification": "diagnostic paired-process summary; selected development prompts; no held-out data",
        "strategy": "triton_nf4_m3_tl_sum",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "generated_tokens": 64,
        "k": 2,
        "physical_rows": 3,
        "prompt_clusters": len(ids),
        "process_runs": sum(len(rows) for rows in by_prompt.values()),
        "b1_token_and_state_exact_runs": sum(
            int(row["block_matches_b1"] and row["state_audit"]["state_exact_at_all_reference_prefixes"])
            for rows in by_prompt.values() for row in rows
        ),
        "total_runs": sum(len(rows) for rows in by_prompt.values()),
        "original_bnb_token_match_runs": sum(
            int(row["block_matches_original_bnb"])
            for rows in by_prompt.values() for row in rows
        ),
        "point_geomean_speedup": point,
        "hierarchical_paired_bootstrap_95": intervals,
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "seed": 20261101,
            "resampling": "prompt clusters, then two process runs within each selected prompt",
        },
        "prompts": prompt_results,
        "input_sha256": input_hashes,
        "heldout_gsm8k_used": False,
        "limitations": [
            "five selected development prompts on one RTX 4050",
            "two process runs per prompt and 64 generated tokens",
            "bootstrap intervals describe this selected cohort, not a random prompt population",
        ],
        "output": str(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "prompt_clusters", "process_runs", "b1_token_and_state_exact_runs", "total_runs",
        "original_bnb_token_match_runs", "point_geomean_speedup",
        "hierarchical_paired_bootstrap_95", "output"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
