#!/usr/bin/env python3
"""Export compact metrics from targeted Nsight Compute kernel samples."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "experiments/backend_validation/kernel_profile"
REPORTS = {
    "vanilla_gemm": ("vanilla_4bit_gemm.ncu-rep", "gemm_4bit_simt"),
    "vanilla_dequant": ("vanilla_dequant.ncu-rep", "kDequantizeBlockwise"),
    "cka90_k2_gemm": ("cka90_4bit_gemm.ncu-rep", "gemm_4bit_simt"),
    "cka90_k2_dequant": ("cka90_dequant.ncu-rep", "kDequantizeBlockwise"),
}
METRICS = {
    "gpu__time_duration.avg": "duration_us",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct_peak",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct_peak",
    "launch__registers_per_thread": "registers_per_thread",
}


def extract(path: Path, expected_kernel: str) -> list[dict]:
    completed = subprocess.run(
        ["/usr/local/cuda/bin/ncu", "--import", str(path), "--page", "raw", "--csv"],
        capture_output=True, text=True, check=True,
    )
    rows = [r for r in csv.DictReader(io.StringIO(completed.stdout)) if r.get("Kernel Name")]
    result = []
    for row in rows:
        if expected_kernel not in row["Kernel Name"]:
            raise ValueError(f"unexpected kernel in {path.name}: {row['Kernel Name']}")
        result.append({
            "kernel_name": row["Kernel Name"],
            "grid_size": row.get("Grid Size"),
            "block_size": row.get("Block Size"),
            **{alias: float(row[key]) for key, alias in METRICS.items()},
        })
    if len(result) != 3:
        raise ValueError(f"expected 3 measured kernel samples in {path.name}, got {len(result)}")
    return result


def main() -> None:
    results = {name: extract(PROFILE_DIR / filename, kernel)
               for name, (filename, kernel) in REPORTS.items()}
    (PROFILE_DIR / "ncu_results.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "## Targeted Nsight Compute samples",
        "",
        "Nsight Compute used the `basic` set, restricted to the CUDA profiler start/stop window (`--profile-from-start off`) and the first three matching launches per run. Model loading and the four-token warmup were excluded. Each row below lists the three distinct launch samples in order; kernel timing/counters are perturbed by NCU replay and are not end-to-end performance measurements.",
        "",
        "| Workload / kernel | Duration samples (µs) | DRAM throughput samples (% peak) | SM throughput samples (% peak) | Registers/thread |",
        "|---|---|---|---|---:|",
    ]
    for name, samples in results.items():
        durations = ", ".join(f"{x['duration_us']:.2f}" for x in samples)
        dram = ", ".join(f"{x['dram_throughput_pct_peak']:.1f}" for x in samples)
        sm = ", ".join(f"{x['sm_throughput_pct_peak']:.1f}" for x in samples)
        lines.append(
            f"| {name} | {durations} | {dram} | {sm} | "
            f"{samples[0]['registers_per_thread']:.0f} / {samples[1]['registers_per_thread']:.0f} / {samples[2]['registers_per_thread']:.0f} |"
        )
    lines += [
        "",
        "The blockwise dequant sample set contains two short launches with low DRAM utilization and one larger launch near 91% of peak DRAM throughput in both vanilla and CKA90/K2. This shows strong shape dependence: a subset of dequant launches can be memory intensive, while the kernel family as a whole should not be labeled the bottleneck from presence alone. The sampled NF4 GEMMs show lower SM throughput than peak and moderate DRAM utilization. These few samples do not establish a roofline location, overall critical-path share, or method-level causal performance difference.",
        "",
        "No arithmetic-intensity claim is made: the collected basic counters provide throughput and occupancy context but not a matched FLOP/byte count for these kernels. Extending to a roofline section or source-level counters should be done only after choosing representative launch shapes from the full trace.",
        "",
        "Raw NCU reports: `vanilla_4bit_gemm.ncu-rep`, `vanilla_dequant.ncu-rep`, `cka90_4bit_gemm.ncu-rep`, and `cka90_dequant.ncu-rep`. `ncu_manifest.json` records hashes, software versions, scope, and filter protocol.",
        "",
    ]
    report = PROFILE_DIR / "report.md"
    old_report = report.read_text()
    marker = "\n## Targeted Nsight Compute samples"
    base = old_report[:old_report.find(marker)] if marker in old_report else old_report.rstrip()
    report.write_text(base.rstrip() + "\n\n" + "\n".join(lines))
    tracked = sorted([PROFILE_DIR / f for f, _ in REPORTS.values()])
    manifest = {
        "scope": "three targeted kernel invocations per method and kernel family; diagnostic only",
        "device": "NVIDIA GeForce RTX 4050 Laptop GPU, compute capability 8.9",
        "ncu_version": subprocess.run(["/usr/local/cuda/bin/ncu", "--version"], capture_output=True,
                                       text=True, check=True).stdout.strip(),
        "collection": {
            "set": "basic", "profile_from_start": "off",
            "kernel_filter": ["gemm_4bit_simt", "kDequantizeBlockwise"],
            "launch_count": 3, "window": "torch.cuda.profiler.start/stop; warmup excluded",
            "workload": "qasper_119_ctx384, 384 input tokens, 32 token budget, NF4 DQ-on, FP16, SDPA, static KV=436",
        },
        "captures_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},
        "summary_sha256": hashlib.sha256((PROFILE_DIR / "ncu_results.json").read_bytes()).hexdigest(),
    }
    (PROFILE_DIR / "ncu_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("wrote ncu_results.json, ncu_manifest.json and appended report.md")


if __name__ == "__main__":
    main()
