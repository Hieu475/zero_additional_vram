#!/usr/bin/env python3
"""Compare two locked cross-model shape-sweep sessions and retain strata."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def session_summary(rows: list[dict]) -> dict:
    completed = [r for r in rows if r.get("status") == "completed"]
    strata: dict[tuple, list[dict]] = defaultdict(list)
    for row in completed:
        strata[(row["family"], row["model"], row["task"], row["M"])].append(row)

    def summarize(items):
        ordinary = [x["ordinary_batch_vs_sequential"] for x in items]
        singleton = [x["singleton_linear_batch_vs_sequential"] for x in items]
        eq = [v for x in ordinary for v in x["argmax_equal_by_row"]]
        row_logit = [v for x in ordinary for v in x["per_row_logit_max_abs"]]
        return {
            "configurations": len(items),
            "ordinary_logit_drift_configs": sum(any(v > 0 for v in x["per_row_logit_max_abs"]) for x in ordinary),
            "ordinary_kv_drift_configs": sum(x["kv_key_max_abs"] > 0 or x["kv_value_max_abs"] > 0 for x in ordinary),
            "ordinary_argmax_matches": sum(eq), "ordinary_argmax_total": len(eq),
            "ordinary_argmax_mismatch_configs": sum(not all(x["argmax_equal_by_row"]) for x in ordinary),
            "ordinary_median_row_logit_max_abs": statistics.median(row_logit) if row_logit else None,
            "ordinary_max_row_logit_max_abs": max(row_logit, default=None),
            "ordinary_max_kv_key_abs": max((x["kv_key_max_abs"] for x in ordinary), default=None),
            "ordinary_max_kv_value_abs": max((x["kv_value_max_abs"] for x in ordinary), default=None),
            "singleton_logit_drift_configs": sum(any(v > 0 for v in x["per_row_logit_max_abs"]) for x in singleton),
            "singleton_kv_drift_configs": sum(x["kv_key_max_abs"] > 0 or x["kv_value_max_abs"] > 0 for x in singleton),
            "singleton_argmax_mismatch_configs": sum(not all(x["argmax_equal_by_row"]) for x in singleton),
        }

    by_model: dict[tuple, list[dict]] = defaultdict(list)
    by_task: dict[str, list[dict]] = defaultdict(list)
    by_m: dict[int, list[dict]] = defaultdict(list)
    for row in completed:
        by_model[(row["family"], row["model"])].append(row)
        by_task[row["task"]].append(row)
        by_m[row["M"]].append(row)
    return {
        "configuration_count": len(rows),
        "completed_configurations": len(completed),
        "failed_configurations": len(rows) - len(completed),
        "overall": summarize(completed),
        "by_model": {f"{family} | {model}": summarize(items) for (family, model), items in sorted(by_model.items())},
        "by_task": {task: summarize(items) for task, items in sorted(by_task.items())},
        "by_M": {str(m): summarize(items) for m, items in sorted(by_m.items())},
        "by_model_task_M": {
            " | ".join(map(str, key)): summarize(items)
            for key, items in sorted(strata.items())
        },
    }


def load_results(directory: Path) -> dict[str, dict]:
    return {p.name: json.loads(p.read_text()) for p in sorted((directory / "results").glob("*.json"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session1", type=Path)
    parser.add_argument("session2", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first, second = load_results(args.session1), load_results(args.session2)
    keys = sorted(set(first) & set(second))
    comparisons = {
        "matched_configurations": len(keys),
        "same_configuration_set": set(first) == set(second),
        "candidate_ids_identical": sum(first[k].get("candidate_ids") == second[k].get("candidate_ids") for k in keys),
        "ordinary_metrics_bitwise_identical": sum(first[k].get("ordinary_batch_vs_sequential") == second[k].get("ordinary_batch_vs_sequential") for k in keys),
        "singleton_linear_metrics_bitwise_identical": sum(first[k].get("singleton_linear_batch_vs_sequential") == second[k].get("singleton_linear_batch_vs_sequential") for k in keys),
    }
    large = []
    for key in keys:
        row = first[key]
        if row.get("status") == "completed" and max(row["ordinary_batch_vs_sequential"]["per_row_logit_max_abs"], default=0) > 2:
            large.append({
                "artifact": key,
                "model": row["model"], "prompt_id": row["prompt_id"], "task": row["task"], "M": row["M"],
                "candidate_ids_identical": row.get("candidate_ids") == second[key].get("candidate_ids"),
                "logit_max_abs_by_row": row["ordinary_batch_vs_sequential"]["per_row_logit_max_abs"],
                "kv_key_max_abs": row["ordinary_batch_vs_sequential"]["kv_key_max_abs"],
                "kv_value_max_abs": row["ordinary_batch_vs_sequential"]["kv_value_max_abs"],
            })
    report = {
        "scope": "expanded development cohort; descriptive single-prefix finite-precision diagnostic, not full-sequence exactness",
        "session1": session_summary(list(first.values())),
        "session2": session_summary(list(second.values())),
        "repeat_comparison": comparisons,
        "large_drift_cases_logit_max_abs_gt_2": large,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"session1": report["session1"]["overall"],
                      "session2": report["session2"]["overall"],
                      "repeat_comparison": comparisons,
                      "large_drift_cases": large}, indent=2))
    print(f"Full stratified report: {args.output}")


if __name__ == "__main__":
    main()
