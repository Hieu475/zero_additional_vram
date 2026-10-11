#!/usr/bin/env python3
"""Join raw benchmark mismatches with default and sequential diagnostic traces.

Classification requires an intervention: a mismatch in the default batched
path, a stable repeated vanilla reference, a differing active KV digest before
the mismatch, and an exact match after canonical sequential verification.
Anything missing that evidence remains unresolved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def benchmark_name(path_value: str) -> str:
    stem = Path(path_value).stem
    return stem.removesuffix("_eval")


def load_trace_index(paths: list[str]) -> tuple[dict, dict]:
    default: dict[tuple[str, str, str], dict] = {}
    sequential: dict[tuple[str, str, str], dict] = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        bench = benchmark_name(payload["benchmark"])
        method = payload["method"]
        destination = sequential if payload.get("canonical_verify") else default
        for sample in payload.get("samples", []):
            destination[(bench, str(sample["id"]), method)] = sample
    return default, sequential


def analyze(raw_path: str, trace_paths: list[str]) -> dict:
    rows = [json.loads(line) for line in Path(raw_path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    default, sequential = load_trace_index(trace_paths)
    run_root = Path(raw_path).parent
    model_id = None
    environment_path = run_root / "environment.json"
    if environment_path.exists():
        model_id = json.loads(environment_path.read_text(encoding="utf-8")).get("model_id")
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        if row.get("method") == "vanilla" or row.get("exact_token_match") is not False:
            continue
        key = (row["benchmark"], str(row["id"]), row["method"])
        grouped.setdefault(key, []).append(row)

    cases = []
    for (bench, sample_id, method), mismatch_rows in sorted(grouped.items()):
        default_trace = default.get((bench, sample_id, "prompt_lookup"))
        sequential_trace = sequential.get((bench, sample_id, "prompt_lookup"))
        first_positions = sorted({row.get("first_divergence_index") for row in mismatch_rows})
        first_pos = first_positions[0] if len(first_positions) == 1 else None
        trace_pos = default_trace.get("first_divergence") if default_trace else None
        cache_diff = default_trace.get("first_cache_difference") if default_trace else None
        vanilla_repeat_stable = (
            default_trace is not None
            and default_trace.get("vanilla_repeat_first_divergence") is None
        )
        sequential_match = (
            sequential_trace is not None
            and sequential_trace.get("first_divergence") is None
        )
        if (trace_pos is not None and vanilla_repeat_stable and cache_diff is not None
                and sequential_match and trace_pos == first_pos):
            classification = "numerical_divergence"
            evidence = "batched path diverges; vanilla repeat is stable; KV digest differs before the first mismatch; canonical sequential verification matches"
        else:
            classification = "unresolved"
            evidence = "the raw sequence mismatch lacks the complete controlled trace needed for attribution"
        cycle = default_trace.get("spec_cycle") if default_trace else None
        vanilla_step = default_trace.get("vanilla_step") if default_trace else None
        cases.append({
            "model_id": model_id,
            "benchmark": bench,
            "prompt_id": sample_id,
            "method": method,
            "repeat_count": len(mismatch_rows),
            "first_mismatch_position": first_pos,
            "first_mismatch_positions_by_run": first_positions,
            "target_logit_margin": vanilla_step.get("margin") if vanilla_step else None,
            "max_logit_error": None,
            "cache_backend": "static",
            "verification_mode": "batched",
            "kv_prefix_equal": cache_diff is None if default_trace else None,
            "first_cache_difference": cache_diff,
            "vanilla_repeat_stable": vanilla_repeat_stable,
            "canonical_sequential_match": sequential_match,
            "sequential_total_time_s": sequential_trace.get("total_time_s") if sequential_trace else None,
            "batched_total_time_s": default_trace.get("total_time_s") if default_trace else None,
            "cycle_at_divergence": cycle,
            "classification": classification,
            "evidence": evidence,
        })
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["classification"]] = counts.get(case["classification"], 0) + 1
    return {
        "raw_measurements": str(raw_path),
        "model_id": model_id,
        "trace_files": trace_paths,
        "mismatch_case_count": len(cases),
        "classification_counts": counts,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Benchmark raw_measurements.jsonl")
    parser.add_argument("--trace", nargs="+", required=True,
                        help="Default/sequential trace JSON files")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = analyze(args.raw, args.trace)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["classification_counts"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
