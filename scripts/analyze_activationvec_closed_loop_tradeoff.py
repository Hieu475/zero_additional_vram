#!/usr/bin/env python3
"""Summarize the development-only 64-token HumanEval activationvec screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(value) for value in values))


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        for position in range(index, end):
            ranks[indexed[position][0]] = average_rank
        index = end
    return ranks


def correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    lmean, rmean = statistics.mean(left), statistics.mean(right)
    ldev = [x - lmean for x in left]
    rdev = [x - rmean for x in right]
    denom = math.sqrt(sum(x * x for x in ldev) * sum(x * x for x in rdev))
    return sum(x * y for x, y in zip(ldev, rdev)) / denom if denom else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=(
        "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/"
        "three_row_kernel/native_bnb_profiles/activationvec_closed_loop_summary_v1.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    selected = [
        row for row in summary["runs"]
        if row["max_new_tokens"] == 64 and str(row["prompt_id"]).startswith("HumanEval/")
    ]
    grouped: dict[str, list[dict]] = {}
    for row in selected:
        grouped.setdefault(str(row["prompt_id"]), []).append(row)

    prompts = []
    for prompt_id in sorted(grouped, key=lambda value: int(value.rsplit("/", 1)[1])):
        rows = grouped[prompt_id]
        drafts = sum(row["draft_tokens"] for row in rows)
        accepted = sum(row["accepted_draft_tokens"] for row in rows)
        b1_ratios = [row["bnb_singleton_over_block_median"] for row in rows]
        bnb_ratios = [row["original_bnb_over_block_median"] for row in rows]
        prompts.append({
            "prompt_id": prompt_id,
            "sessions": len(rows),
            "paired_rounds": sum(row["paired_rounds"] for row in rows),
            "draft_tokens": drafts,
            "accepted_draft_tokens": accepted,
            "acceptance_rate": accepted / drafts if drafts else None,
            "geomean_speedup_vs_singleton": geometric_mean(b1_ratios),
            "geomean_speedup_vs_original_bnb": geometric_mean(bnb_ratios),
            "all_runs_match_singleton": all(
                row["all_token_runs_match_singleton"]
                and row["all_state_prefixes_match_singleton"]
                and row["cache_lengths_equal"] for row in rows),
            "original_bnb_token_match_rounds": sum(
                row["original_bnb_token_match_rounds"] for row in rows),
            "original_bnb_total_rounds": sum(
                row["original_bnb_total_rounds"] for row in rows),
        })

    if not prompts:
        raise RuntimeError("No 64-token HumanEval runs found in summary")
    if not all(prompt["all_runs_match_singleton"] for prompt in prompts):
        raise RuntimeError("At least one selected process failed singleton token/state exactness")

    acceptance = [prompt["acceptance_rate"] for prompt in prompts]
    bnb_speedups = [prompt["geomean_speedup_vs_original_bnb"] for prompt in prompts]
    b1_speedups = [prompt["geomean_speedup_vs_singleton"] for prompt in prompts]
    rng = random.Random(20261031)
    boot_geomeans = []
    n = len(prompts)
    for _ in range(args.bootstrap_samples):
        sample = [math.log(bnb_speedups[rng.randrange(n)]) for _ in range(n)]
        boot_geomeans.append(math.exp(statistics.mean(sample)))
    boot_geomeans.sort()
    lo = boot_geomeans[int(0.025 * len(boot_geomeans))]
    hi = boot_geomeans[min(len(boot_geomeans) - 1, int(0.975 * len(boot_geomeans)))]

    source_path = Path(__file__).resolve()
    result = {
        "schema": "zassd-activationvec-human-eval-64-tradeoff-v1",
        "classification": "descriptive development-only prompt-cluster analysis",
        "filter": {"dataset_family": "HumanEval extension", "max_new_tokens": 64},
        "prompt_count": n,
        "process_count": len(selected),
        "paired_rounds": sum(prompt["paired_rounds"] for prompt in prompts),
        "all_selected_runs_match_singleton": True,
        "original_bnb_token_match_rounds": sum(
            prompt["original_bnb_token_match_rounds"] for prompt in prompts),
        "original_bnb_total_rounds": sum(
            prompt["original_bnb_total_rounds"] for prompt in prompts),
        "prompts_faster_than_original_bnb": sum(value > 1.0 for value in bnb_speedups),
        "prompts_faster_than_singleton": sum(value > 1.0 for value in b1_speedups),
        "prompt_cluster_geomean_speedup_vs_original_bnb": geometric_mean(bnb_speedups),
        "prompt_cluster_bootstrap_95_interval_vs_original_bnb": [lo, hi],
        "prompt_cluster_geomean_speedup_vs_singleton": geometric_mean(b1_speedups),
        "spearman_acceptance_vs_speedup_original_bnb": correlation(
            rankdata(acceptance), rankdata(bnb_speedups)),
        "spearman_acceptance_vs_speedup_singleton": correlation(
            rankdata(acceptance), rankdata(b1_speedups)),
        "bootstrap": {"method": "prompt-cluster percentile bootstrap of geometric mean",
                      "samples": args.bootstrap_samples, "seed": 20261031},
        "prompts": prompts,
        "scope_limits": [
            "Prompts are the resampling units; process sessions are combined within prompt.",
            "Small fixed development cohort on one RTX 4050; interval is descriptive, not a population guarantee.",
            "Exactness is relative to BNB singleton under the fixed-row BNB-rowwise schedule.",
            "Token match to original BNB does not establish original-BNB state equivalence.",
            "GSM8K held-out indices 300-349 were not used.",
        ],
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "source": str(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "prompt_count", "process_count", "paired_rounds",
        "prompts_faster_than_original_bnb", "prompts_faster_than_singleton",
        "prompt_cluster_geomean_speedup_vs_original_bnb",
        "prompt_cluster_bootstrap_95_interval_vs_original_bnb",
        "prompt_cluster_geomean_speedup_vs_singleton",
        "spearman_acceptance_vs_speedup_original_bnb",
        "spearman_acceptance_vs_speedup_singleton")}, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
