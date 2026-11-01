#!/usr/bin/env python3
"""Decompose the recorded activationvec B1 cycle traces for 64-token HumanEval."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=(
        "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/"
        "three_row_kernel/native_bnb_profiles/activationvec_closed_loop_summary_v1.json"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text())
    records = []
    for run in summary["runs"]:
        if run["max_new_tokens"] != 64 or not str(run["prompt_id"]).startswith("HumanEval/"):
            continue
        artifact_path = Path(run["artifact"])
        artifact = json.loads(artifact_path.read_text())
        trace = artifact.get("activationvec_cycle_trace", [])
        if not trace:
            continue
        totals = {name: sum(float(cycle.get(name, 0.0)) for cycle in trace)
                  for name in ("draft_s", "verify_s", "commit_s", "cycle_s")}
        total_cycle_s = totals["cycle_s"]
        if total_cycle_s <= 0:
            continue
        draft_cycles = [cycle for cycle in trace if cycle.get("draft_tokens")]
        no_draft_cycles = len(trace) - len(draft_cycles)
        records.append({
            "prompt_id": artifact["prompt_id"],
            "max_new_tokens": 64,
            "draft_tokens": int(run["draft_tokens"]),
            "accepted_draft_tokens": int(run["accepted_draft_tokens"]),
            "acceptance_rate": float(run["acceptance_rate"]),
            "cycle_count": len(trace),
            "draft_cycle_count": len(draft_cycles),
            "no_draft_cycle_count": no_draft_cycles,
            "no_draft_cycle_fraction": no_draft_cycles / len(trace),
            "draft_s": totals["draft_s"],
            "verify_s": totals["verify_s"],
            "commit_s": totals["commit_s"],
            "cycle_s": total_cycle_s,
            "draft_fraction_of_cycle_time": totals["draft_s"] / total_cycle_s,
            "verify_fraction_of_cycle_time": totals["verify_s"] / total_cycle_s,
            "commit_fraction_of_cycle_time": totals["commit_s"] / total_cycle_s,
        })
    if not records:
        raise RuntimeError("No cycle traces found for 64-token HumanEval runs")

    def median(field: str) -> float:
        return statistics.median(float(record[field]) for record in records)

    source = Path(__file__).resolve()
    output_data = {
        "schema": "zassd-activationvec-human-eval-64-cycle-costs-v1",
        "classification": "diagnostic per-cycle timings from unpaired initial generation traces",
        "prompt_count": len({record["prompt_id"] for record in records}),
        "trace_count": len(records),
        "median_cycle_count": median("cycle_count"),
        "median_no_draft_cycles": median("no_draft_cycle_count"),
        "median_no_draft_cycle_fraction": median("no_draft_cycle_fraction"),
        "median_verify_fraction_of_cycle_time": median("verify_fraction_of_cycle_time"),
        "median_draft_fraction_of_cycle_time": median("draft_fraction_of_cycle_time"),
        "median_commit_fraction_of_cycle_time": median("commit_fraction_of_cycle_time"),
        "ranges": {
            field: [min(float(record[field]) for record in records),
                    max(float(record[field]) for record in records)]
            for field in ("verify_fraction_of_cycle_time", "draft_fraction_of_cycle_time",
                          "commit_fraction_of_cycle_time", "no_draft_cycle_fraction")
        },
        "records": records,
        "scope_limits": [
            "Cycle traces come from one initial unpaired generation per prompt; their timings are diagnostic, not paired performance estimates.",
            "verify_s includes target model execution for both proposal and no-proposal cycles; it is not proposal-only cost.",
            "Percentages omit host orchestration and other uninstrumented time outside cycle_s.",
            "Finite development cohort, one GPU; GSM8K held-out indices 300-349 were not used.",
        ],
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_data, indent=2) + "\n")
    print(json.dumps({key: output_data[key] for key in (
        "prompt_count", "trace_count", "median_cycle_count", "median_no_draft_cycles",
        "median_no_draft_cycle_fraction", "median_verify_fraction_of_cycle_time",
        "median_draft_fraction_of_cycle_time", "median_commit_fraction_of_cycle_time",
        "ranges")}, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
