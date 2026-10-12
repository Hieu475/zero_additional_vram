#!/usr/bin/env python3
"""Analyze paired reduced-sync measurements with prompts as bootstrap units."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def bootstrap_ci(values: list[float], rng: random.Random, samples: int = 10000) -> list[float]:
    estimates = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(draw) / len(draw))
    estimates.sort()
    return [estimates[int(.025 * samples)], estimates[int(.975 * samples)]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=271828)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.raw.read_text().splitlines() if line.strip()]
    pairs = defaultdict(dict)
    for row in rows:
        pairs[(row["prompt_id"], row["repeat"])][row["method"]] = row
    grouped = defaultdict(list)
    for (prompt_id, repeat), pair in pairs.items():
        if set(pair) != {"reference", "reduced_sync_conservative"}:
            raise ValueError(f"Incomplete paired measurement: {prompt_id}/{repeat}")
        ref = pair["reference"]["measurement"]
        red = pair["reduced_sync_conservative"]["measurement"]
        if pair["reference"]["input_token_ids_sha256"] != pair["reduced_sync_conservative"]["input_token_ids_sha256"]:
            raise ValueError(f"Input IDs differ for {prompt_id}/{repeat}")
        grouped[prompt_id].append({
            "e2e_speedup": ref["total_time_s"] / max(red["total_time_s"], 1e-12),
            "decode_speedup": red["tokens_per_second"] / max(ref["tokens_per_second"], 1e-12),
            "reference_total_s": ref["total_time_s"], "reduced_total_s": red["total_time_s"],
            "reference_draft_s": ref["draft_time_s"], "reduced_draft_s": red["draft_time_s"],
            "reference_verify_s": ref["verify_time_s"], "reduced_verify_s": red["verify_time_s"],
            "reference_cache_s": ref["cache_time_s"], "reduced_cache_s": red["cache_time_s"],
            "exact": ref["generated_token_ids"] == red["generated_token_ids"],
            "reference_tokens": ref["total_tokens"], "reduced_tokens": red["total_tokens"],
        })
    prompt_rows = {}
    for prompt_id, measurements in grouped.items():
        prompt_rows[prompt_id] = {
            key: sum(row[key] for row in measurements) / len(measurements)
            for key in measurements[0] if key not in ("exact", "reference_tokens", "reduced_tokens")
        }
        prompt_rows[prompt_id]["exact_rate"] = sum(row["exact"] for row in measurements) / len(measurements)
    rng = random.Random(args.seed)
    report = {"n_prompts": len(prompt_rows), "n_paired_runs": len(pairs), "actions": {}}
    for metric in ("e2e_speedup", "decode_speedup"):
        vals = [prompt_rows[p][metric] for p in sorted(prompt_rows)]
        report["actions"][metric] = {
            "mean_prompt_level": sum(vals) / len(vals),
            "prompt_bootstrap_95ci": bootstrap_ci(vals, rng),
            "median_prompt_level": sorted(vals)[len(vals) // 2],
        }
    report["sequence_exact_match_rate"] = sum(
        prompt_rows[p]["exact_rate"] for p in prompt_rows
    ) / max(1, len(prompt_rows))
    report["mean_stage_seconds"] = {
        stage: {
            "reference": sum(prompt_rows[p][f"reference_{stage}_s"] for p in prompt_rows) / len(prompt_rows),
            "reduced_sync": sum(prompt_rows[p][f"reduced_{stage}_s"] for p in prompt_rows) / len(prompt_rows),
        }
        for stage in ("total", "draft", "verify", "cache")
    }
    report["interpretation_note"] = (
        "Bootstrap resamples prompts; repeated runs within a prompt are averaged. "
        "Nsight synchronization counts/host-wait durations must be analyzed separately."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
