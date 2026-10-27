#!/usr/bin/env python3
"""Summarize a diagnostic paired exactness/latency feasibility run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


METHODS = ("vanilla", "pld_batched", "pld_sequential", "margin_guard")


def analyze(rows: list[dict]) -> dict:
    by_prompt: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["task"], row["id"])
        if row["method"] in by_prompt[key]:
            raise ValueError(f"duplicate method {row['method']} for {key}")
        by_prompt[key][row["method"]] = row
    for key, methods in by_prompt.items():
        if set(methods) != set(METHODS):
            raise ValueError(f"incomplete method pair for {key}: {sorted(methods)}")
        if len({method["input_token_ids_sha256"] for method in methods.values()}) != 1:
            raise ValueError(f"input token IDs differ within pair {key}")
    report: dict = {"n_prompts": len(by_prompt), "methods": {}, "by_task": {}}
    for method in METHODS:
        selected = [group[method] for group in by_prompt.values()]
        report["methods"][method] = {
            "exact_sequences": sum(row["exact_token_match"] for row in selected),
            "n_sequences": len(selected),
            "mean_paired_latency_ratio_vanilla_over_method": sum(
                row["speedup_vs_vanilla"] for row in selected
            ) / len(selected),
            "mean_paired_latency_ratio_on_exact_sequences": (
                sum(row["speedup_vs_vanilla"] for row in selected if row["exact_token_match"])
                / sum(row["exact_token_match"] for row in selected)
                if any(row["exact_token_match"] for row in selected) else None
            ),
        }
    guard = [group["margin_guard"] for group in by_prompt.values()]
    guard_exact = [row for row in guard if row["exact_token_match"]]
    report["guard_cost_screen"] = {
        "mean_replay_fraction_all": sum(
            row["free_restore_ceiling"]["replay_fraction_of_guard_time"] for row in guard
        ) / len(guard),
        "mean_free_restore_latency_ratio_all": sum(
            row["free_restore_ceiling"]["free_restore_speedup_vs_vanilla"] for row in guard
        ) / len(guard),
        "mean_free_restore_latency_ratio_exact_only": (
            sum(row["free_restore_ceiling"]["free_restore_speedup_vs_vanilla"]
                for row in guard_exact) / len(guard_exact)
            if guard_exact else None
        ),
        "exact_guard_prompts_with_free_restore_ratio_above_one": sum(
            row["free_restore_ceiling"]["free_restore_speedup_vs_vanilla"] > 1
            for row in guard_exact
        ),
        "exact_guard_prompts": len(guard_exact),
    }
    oracle_rows = []
    for key, group in by_prompt.items():
        exact_methods = [row for row in group.values() if row["exact_token_match"]]
        best = min(exact_methods, key=lambda row: row["total_s"])
        vanilla = group["vanilla"]
        oracle_rows.append({
            "task": key[0], "id": key[1], "selected_method": best["method"],
            "latency_ratio": vanilla["total_s"] / best["total_s"],
        })
    report["prompt_hindsight_oracle"] = {
        "mean_latency_ratio": sum(row["latency_ratio"] for row in oracle_rows) / len(oracle_rows),
        "selected_method_counts": {method: sum(row["selected_method"] == method for row in oracle_rows)
                                   for method in METHODS},
        "scope": "best observed exact whole-prompt method chosen with hindsight; not an executable policy or cycle-level oracle",
    }
    for task in sorted({key[0] for key in by_prompt}):
        groups = [group for (name, _), group in by_prompt.items() if name == task]
        report["by_task"][task] = {
            method: {
                "exact": sum(group[method]["exact_token_match"] for group in groups),
                "n": len(groups),
                "mean_latency_ratio": sum(
                    group[method]["speedup_vs_vanilla"] for group in groups
                ) / len(groups),
            } for method in METHODS
        }
    report["limitations"] = [
        "Previously used diagnostic prompts; no held-out claim.",
        "Single process session and one GPU; ratios are descriptive.",
        "Free restoration changes a measured cost only, not the actual decoding trajectory; it is not an exact algorithm.",
        "An inexact method's latency ratio is not an exact-decoding speedup.",
    ]
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    directory = Path(args.run_dir)
    rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines() if line]
    report = analyze(rows)
    output = directory / "analysis.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
