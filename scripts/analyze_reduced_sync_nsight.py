#!/usr/bin/env python3
"""Summarize paired Nsight CUDA API captures for reduced-sync variants."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def api_summary(path: Path) -> list[dict]:
    result = subprocess.run(
        ["nsys", "stats", "--report", "cuda_api_sum", "--format", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    start = result.stdout.find("[\n")
    end = result.stdout.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"Cannot parse nsys stats output for {path}: {result.stdout}")
    return json.loads(result.stdout[start:end + 1])


def by_name(rows: list[dict]) -> dict[str, dict]:
    return {row["Name"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    variants = ("reference", "reduced_sync_conservative")
    result = {"capture_protocol": {"trace": ["cuda", "nvtx", "osrt"],
                                   "cpu_sampling": "disabled by host environment",
                                   "repeats_per_variant": 1,
                                   "prompt_id": "1", "config": "cka_90", "k": 2,
                                   "max_new_tokens": 24}, "variants": {}}
    for variant in variants:
        profile = json.loads((root / f"{variant}.json").read_text())
        api = by_name(api_summary(root / f"{variant}.nsys-rep"))
        metrics = profile["metrics"]
        sync_rows = {name: api.get(name, {"Num Calls": 0, "Total Time (ns)": 0})
                     for name in ("cudaDeviceSynchronize", "cudaStreamSynchronize")}
        result["variants"][variant] = {
            "commit_sha": profile["commit_sha"], "gpu": profile["gpu_name"],
            "generated_token_ids": metrics["generated_token_ids"],
            "end_to_end_seconds_under_nsys": metrics["total_time_s"],
            "stage_seconds_under_nsys": {
                "prefill": metrics["prefill_time_s"], "draft": metrics["draft_time_s"],
                "verify": metrics["verify_time_s"], "cache_commit": metrics["cache_time_s"],
            },
            "synchronization_api": {
                name: {"calls": row["Num Calls"], "host_api_time_ms": row["Total Time (ns)"] / 1e6}
                for name, row in sync_rows.items()
            },
            "cuda_launch_kernel_calls": api.get("cudaLaunchKernel", {}).get("Num Calls"),
        }
    reference = result["variants"]["reference"]["synchronization_api"]
    reduced = result["variants"]["reduced_sync_conservative"]["synchronization_api"]
    result["comparison"] = {
        "tokens_identical": (result["variants"]["reference"]["generated_token_ids"]
                             == result["variants"]["reduced_sync_conservative"]["generated_token_ids"]),
        "synchronize_calls_total_reference": sum(x["calls"] for x in reference.values()),
        "synchronize_calls_total_reduced": sum(x["calls"] for x in reduced.values()),
        "synchronize_api_time_ms_reference": sum(x["host_api_time_ms"] for x in reference.values()),
        "synchronize_api_time_ms_reduced": sum(x["host_api_time_ms"] for x in reduced.values()),
        "limitation": "Single Nsight capture per variant; CPU sampling unavailable. API durations are not independent repeated-session estimates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
