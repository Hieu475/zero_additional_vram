#!/usr/bin/env python3
"""Summarize the locked cross-model NF4 shape sweep without pooling away strata."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def summarize(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "completed":
            groups[(row["family"], row["model"], row["task"], row["M"])].append(row)

    strata = []
    for (family, model, task, m), items in sorted(groups.items()):
        ordinary = [x["ordinary_batch_vs_sequential"] for x in items]
        singleton = [x["singleton_linear_batch_vs_sequential"] for x in items]
        ordinary_eq = [v for x in ordinary for v in x["argmax_equal_by_row"]]
        singleton_eq = [v for x in singleton for v in x["argmax_equal_by_row"]]
        ordinary_logits = [v for x in ordinary for v in x["per_row_logit_max_abs"]]
        singleton_logits = [v for x in singleton for v in x["per_row_logit_max_abs"]]
        strata.append({
            "family": family, "model": model, "task": task, "M": m,
            "configurations": len(items),
            "ordinary_logit_drift_configs": sum(any(v > 0 for v in x["per_row_logit_max_abs"]) for x in ordinary),
            "ordinary_kv_drift_configs": sum(x["kv_key_max_abs"] > 0 or x["kv_value_max_abs"] > 0 for x in ordinary),
            "ordinary_argmax_matches": sum(ordinary_eq), "ordinary_argmax_total": len(ordinary_eq),
            "ordinary_argmax_mismatch_configs": sum(not all(x["argmax_equal_by_row"]) for x in ordinary),
            "ordinary_median_row_logit_max_abs": statistics.median(ordinary_logits),
            "ordinary_max_row_logit_max_abs": max(ordinary_logits),
            "ordinary_max_kv_key_abs": max(x["kv_key_max_abs"] for x in ordinary),
            "ordinary_max_kv_value_abs": max(x["kv_value_max_abs"] for x in ordinary),
            "singleton_logit_drift_configs": sum(any(v > 0 for v in x["per_row_logit_max_abs"]) for x in singleton),
            "singleton_kv_drift_configs": sum(x["kv_key_max_abs"] > 0 or x["kv_value_max_abs"] > 0 for x in singleton),
            "singleton_argmax_matches": sum(singleton_eq), "singleton_argmax_total": len(singleton_eq),
            "singleton_argmax_mismatch_configs": sum(not all(x["argmax_equal_by_row"]) for x in singleton),
            "singleton_median_row_logit_max_abs": statistics.median(singleton_logits),
        })

    completed = [r for r in rows if r.get("status") == "completed"]
    failures = [r for r in rows if r.get("status") != "completed"]
    ordinary = [r["ordinary_batch_vs_sequential"] for r in completed]
    singleton = [r["singleton_linear_batch_vs_sequential"] for r in completed]

    def total_matches(values):
        eq = [v for x in values for v in x["argmax_equal_by_row"]]
        return sum(eq), len(eq)

    ordinary_matches, ordinary_total = total_matches(ordinary)
    singleton_matches, singleton_total = total_matches(singleton)
    return {
        "status": "complete" if not failures else "partial",
        "configuration_count": len(rows), "completed_configurations": len(completed),
        "failed_configurations": len(failures),
        "ordinary_logit_drift_configs": sum(any(v > 0 for v in x["per_row_logit_max_abs"]) for x in ordinary),
        "ordinary_kv_drift_configs": sum(x["kv_key_max_abs"] > 0 or x["kv_value_max_abs"] > 0 for x in ordinary),
        "ordinary_argmax_matches": ordinary_matches, "ordinary_argmax_total": ordinary_total,
        "ordinary_argmax_mismatch_configs": sum(not all(x["argmax_equal_by_row"]) for x in ordinary),
        "singleton_logit_drift_configs": sum(any(v > 0 for v in x["per_row_logit_max_abs"]) for x in singleton),
        "singleton_kv_drift_configs": sum(x["kv_key_max_abs"] > 0 or x["kv_value_max_abs"] > 0 for x in singleton),
        "singleton_argmax_matches": singleton_matches, "singleton_argmax_total": singleton_total,
        "singleton_argmax_mismatch_configs": sum(not all(x["argmax_equal_by_row"]) for x in singleton),
        "strata": strata,
        "failures": [{k: r.get(k) for k in ("model", "task", "prompt_id", "M", "error_type", "error")} for r in failures],
        "scope": "descriptive single-prefix finite-precision diagnostic; not full-sequence exactness or a universal model/runtime claim",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(path.read_text()) for path in sorted(args.results_dir.glob("*.json"))]
    report = summarize(rows)
    output = args.output or args.results_dir.parent / "analysis.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("strata", "failures")}, indent=2))
    print(f"\nPer-stratum details: {output}")


if __name__ == "__main__":
    main()
