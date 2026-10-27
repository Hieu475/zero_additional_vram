#!/usr/bin/env python3
"""Summarize CUDA kernels and BNB GEMM input shapes from a PyTorch trace."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def kernel_family(name: str) -> str:
    low = name.lower()
    if "gemm_4bit" in low:
        return "bnb_4bit_gemm"
    if "flash_fwd_kernel" in low or "flash_attention" in low:
        return "flash_attention"
    if "gemm" in low or "cutlass::kernel" in low:
        return "other_gemm"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    profile_path = args.trace_dir / "profile.json"
    trace_path = args.trace_dir / "trace.json"
    profile = json.loads(profile_path.read_text())
    trace = json.loads(trace_path.read_text())

    family_us: Counter[str] = Counter()
    family_count: Counter[str] = Counter()
    top_kernels: Counter[str] = Counter()
    bnb_shapes: Counter[str] = Counter()
    bnb_dispatch_us: Counter[str] = Counter()
    for event in trace.get("traceEvents", []):
        category = event.get("cat")
        name = event.get("name", "")
        if category == "kernel":
            family = kernel_family(name)
            duration = float(event.get("dur", 0.0))
            family_us[family] += duration
            family_count[family] += 1
            top_kernels[name] += duration
        elif category == "cpu_op" and name == "bitsandbytes::gemm_4bit":
            dims = event.get("args", {}).get("Input Dims", [])
            shape = str(dims[0]) if dims else "unknown"
            bnb_shapes[shape] += 1
            bnb_dispatch_us[shape] += float(event.get("dur", 0.0))

    total_kernel_us = sum(family_us.values())
    result = {
        "scope": "sum of kernel durations (not request wall time; kernels may overlap)",
        "model": profile.get("model"),
        "revision": profile.get("revision"),
        "attention_strategy": profile.get("attention_strategy"),
        "prompt_id": profile.get("prompt_id"),
        "block_total_ms": profile.get("block_total_ms"),
        "verify_ms": profile.get("verify_ms"),
        "cycles": profile.get("cycles"),
        "aggregate_kernel_ms": total_kernel_us / 1000.0,
        "kernel_families": [
            {"family": family, "calls": family_count[family],
             "aggregate_ms": family_us[family] / 1000.0,
             "share": family_us[family] / total_kernel_us if total_kernel_us else 0.0}
            for family in family_us
        ],
        "top_kernels": [
            {"kernel": name, "aggregate_ms": duration / 1000.0}
            for name, duration in top_kernels.most_common(20)
        ],
        "bnb_4bit_gemm_inputs": [
            {"input_shape": shape, "calls": count,
             "aggregate_cpu_dispatch_ms": bnb_dispatch_us[shape] / 1000.0}
            for shape, count in bnb_shapes.most_common()
        ],
    }
    output = args.trace_dir / "analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
