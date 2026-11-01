#!/usr/bin/env python3
"""Summarize paired K2/R3 activation-vectorized fallback performance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def geometric_mean(values):
    values = list(values)
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean needs positive samples")
    return math.exp(statistics.mean(math.log(value) for value in values))


def paired_ratios(run, reference, candidate):
    return [
        round_row["methods"][reference]["total_s"]
        / round_row["methods"][candidate]["total_s"]
        for round_row in run["paired_rounds"]
    ]


def hierarchical_prompt_bootstrap(grouped, reference, candidate, draws, seed):
    """Resample prompt clusters, then process runs and paired timing rounds."""
    rng = random.Random(seed)
    prompt_ids = sorted(grouped)
    estimates = []
    for _ in range(draws):
        sampled_logs = []
        for prompt_id in rng.choices(prompt_ids, k=len(prompt_ids)):
            runs = grouped[prompt_id]
            for run in rng.choices(runs, k=len(runs)):
                ratios = paired_ratios(run, reference, candidate)
                sampled_logs.extend(math.log(value) for value in rng.choices(ratios, k=len(ratios)))
        estimates.append(math.exp(statistics.mean(sampled_logs)))
    estimates.sort()
    return [estimates[int(0.025 * draws)], estimates[min(draws - 1, int(0.975 * draws))]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifacts", nargs="+", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--bootstrap-draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=4050)
    args = ap.parse_args()

    runs = [json.loads(path.read_text()) for path in args.artifacts]
    grouped = {}
    for run, path in zip(runs, args.artifacts):
        grouped.setdefault(run["prompt_id"], []).append(run)
    if len(grouped) < 2:
        raise ValueError("at least two prompt clusters are required")

    comparisons = {
        "block_over_original_bnb_speedup": ("original_bnb", "fallback_default"),
        "block_over_b1_singleton_speedup": ("bnb_singleton", "fallback_default"),
    }
    prompts = []
    all_exact = True
    for prompt_id, prompt_runs in sorted(grouped.items()):
        row = {"prompt_id": prompt_id, "process_count": len(prompt_runs), "runs": []}
        block_rounds = [
            timing["methods"]["fallback_default"]
            for run in prompt_runs for timing in run["paired_rounds"]
        ]
        drafted = sum(sample["drafted"] for sample in block_rounds)
        accepted = sum(sample["accepted"] for sample in block_rounds)
        row["draft_acceptance_rate"] = accepted / drafted if drafted else None
        row["mean_verification_cycles"] = statistics.mean(sample["cycles"] for sample in block_rounds)
        row["mean_no_proposal_fallback_cycles"] = statistics.mean(
            sample["fallback_counts"]["no_proposal_cycles"] for sample in block_rounds
        )
        for metric, (reference, candidate) in comparisons.items():
            run_means = [geometric_mean(paired_ratios(run, reference, candidate)) for run in prompt_runs]
            row[metric] = geometric_mean(run_means)
        for run, path in zip(prompt_runs, [p for p, r in zip(args.artifacts, runs) if r["prompt_id"] == prompt_id]):
            exact = all(
                entry["tokens_match_b1_singleton"]
                and entry["state_domains_equal"]
                and not entry["state_mismatches"]
                for entry in run["exactness_preflight"].values()
            )
            timed_tokens = all(
                count == len(run["paired_rounds"])
                for count in run["timed_token_match_rounds"].values()
            )
            all_exact = all_exact and exact and timed_tokens
            row["runs"].append({
                "artifact": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "exact_to_b1_preflight": exact,
                "all_timed_methods_match_b1_tokens": timed_tokens,
                "paired_round_count": len(run["paired_rounds"]),
            })
        prompts.append(row)

    global_estimates = {}
    for metric, (reference, candidate) in comparisons.items():
        prompt_means = [row[metric] for row in prompts]
        global_estimates[metric] = {
            "equal_prompt_geomean": geometric_mean(prompt_means),
            "hierarchical_paired_bootstrap_95pct_ci": hierarchical_prompt_bootstrap(
                grouped, reference, candidate, args.bootstrap_draws, args.seed
            ),
            "prompt_level_geomean_samples": prompt_means,
        }

    payload = {
        "schema": "zassd-activationvec-k2r3-paired-performance-summary-v1",
        "classification": "selected development prompt clusters; paired descriptive performance estimate",
        "model": runs[0]["model"],
        "revision": runs[0]["revision"],
        "schedule": runs[0]["schedule"],
        "candidate": "default streaming .cs activation-load K2/R3 block fallback",
        "prompt_count": len(prompts),
        "process_count": len(runs),
        "all_runs_exact_to_b1_and_timed_tokens": all_exact,
        "bootstrap": {"method": "resample prompt clusters, process runs within prompt, then paired timing rounds",
                      "draws": args.bootstrap_draws, "seed": args.seed,
                      "interval_limit": "selected prompts are not a random population sample"},
        "comparisons": global_estimates,
        "prompts": prompts,
        "limitations": ["one GPU", "five selected development prompt clusters", "one or two process runs per prompt",
                        "original-BNB comparison is latency/token only, not KV identity",
                        "GSM8K held-out indices 300-349 excluded"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "prompt_count": payload["prompt_count"],
        "process_count": payload["process_count"],
        "all_exact": all_exact,
        "comparisons": global_estimates,
        "per_prompt": [{"prompt_id": row["prompt_id"],
                         "vs_original_bnb": row["block_over_original_bnb_speedup"],
                         "vs_b1_singleton": row["block_over_b1_singleton_speedup"]}
                        for row in prompts],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
