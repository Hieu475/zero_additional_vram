#!/usr/bin/env python3
"""Summarize exported Nsight Systems reports for backend characterization."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "experiments/backend_validation/kernel_profile"
METHODS = ("vanilla", "pld_k4", "cka90_k2")


def load(method: str, report: str) -> list[dict]:
    path = PROFILE_DIR / f"{method}_stats_{report}.json"
    return json.loads(path.read_text())


def sum_matching(rows: list[dict], fragment: str) -> tuple[int, int]:
    selected = [row for row in rows if fragment.lower() in row.get("Name", "").lower()]
    return sum(row.get("Total Time (ns)", 0) for row in selected), sum(
        row.get("Instances", row.get("Num Calls", 0)) for row in selected
    )


def main() -> None:
    summary: dict[str, dict] = {}
    for method in METHODS:
        kernels = load(method, "cuda_gpu_kern_sum")
        apis = load(method, "cuda_api_sum")
        nvtx = load(method, "nvtx_sum")
        mem = load(method, "cuda_gpu_mem_time_sum")
        total_kernel_ns = sum(row["Total Time (ns)"] for row in kernels)
        dequant_ns, dequant_n = sum_matching(kernels, "kDequantizeBlockwise")
        gemm4_ns, gemm4_n = sum_matching(kernels, "gemm_4bit")
        sync_rows = [r for r in apis if "Synchronize" in r.get("Name", "")]
        launch = next((r for r in apis if r.get("Name") == "cudaLaunchKernel"), {})
        ranges = {r.get("Range", ""): r for r in nvtx}
        transfers_ns = sum(r.get("Total Time (ns)", 0) for r in mem)
        summary[method] = {
            "total_gpu_kernel_ns": total_kernel_ns,
            "kernel_instances": sum(row.get("Instances", 0) for row in kernels),
            "bnb_4bit_gemm_ns": gemm4_ns,
            "bnb_4bit_gemm_instances": gemm4_n,
            "bnb_dequant_ns": dequant_ns,
            "bnb_dequant_instances": dequant_n,
            "bnb_dequant_share_of_kernel_time": dequant_ns / total_kernel_ns,
            "bnb_4bit_gemm_share_of_kernel_time": gemm4_ns / total_kernel_ns,
            "synchronization_api_ns": sum(r.get("Total Time (ns)", 0) for r in sync_rows),
            "synchronization_api_calls": sum(r.get("Num Calls", 0) for r in sync_rows),
            "kernel_launch_api_ns": launch.get("Total Time (ns)", 0),
            "kernel_launch_api_calls": launch.get("Num Calls", 0),
            "nvtx_ranges": {
                name: {"total_ns": row.get("Total Time (ns)"), "instances": row.get("Instances")}
                for name, row in ranges.items()
            },
            "gpu_mem_operation_total_ns": transfers_ns,
            "gpu_mem_operations": [
                {"operation": row.get("Operation"), "total_ns": row.get("Total Time (ns)"),
                 "count": row.get("Count")}
                for row in mem
            ],
            "top_kernels": [
                {"name": r.get("Name"), "total_ns": r.get("Total Time (ns)"),
                 "instances": r.get("Instances"), "share": r.get("Time (%)")}
                for r in kernels[:8]
            ],
        }
    out = PROFILE_DIR / "results.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    capture_meta = json.loads((PROFILE_DIR / "cka90_k2.json").read_text())
    tracked = sorted(
        [PROFILE_DIR / f"{method}.json" for method in METHODS]
        + [PROFILE_DIR / f"{method}.nsys-rep" for method in METHODS]
        + [out]
    )
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in tracked}
    ns = subprocess.run(["/usr/local/cuda/bin/nsys", "--version"], capture_output=True, text=True)
    manifest = {
        "scope": "single-prompt Nsight Systems diagnostic captures; not production benchmark",
        "shared_config": {k: capture_meta[k] for k in (
            "prompt_id", "prompt_token_ids_sha256", "prompt_tokens", "max_new_tokens", "kv_capacity",
            "quantization", "attention", "model_revision", "gpu", "compute_capability", "torch",
            "cuda", "transformers", "bitsandbytes", "execution_commit", "capture_range"
        )},
        "nsys_version": ns.stdout.strip() or ns.stderr.strip(),
        "capture_files_sha256": hashes,
        "exported_summary_files": sorted(str(p.relative_to(ROOT)) for p in PROFILE_DIR.glob("*_stats_*.json")),
        "analysis_script": "scripts/analyze_backend_nsight.py",
    }
    (PROFILE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Nsight Systems baseline profile",
        "",
        "These are instrumented diagnostic captures for one fixed prompt (384 input tokens, 32-token budget), NF4 double-quant, FP16 compute, SDPA, static KV capacity 436. They are not production performance measurements. Each method was captured in a separate process after warmup.",
        "",
        "GPU kernel time is the sum of reported kernel durations and may include overlap; CUDA API host durations are not additive to GPU time. Nsight Systems does not identify dequantization as causal merely because its kernel appears in the timeline.",
        "",
        "| Method | Total GPU kernel time (ms) | NF4 GEMM (ms; share) | BNB dequant (ms; share) | Synchronize API (ms; calls) | Kernel launch API (ms; calls) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, row in summary.items():
        lines.append(
            f"| {method} | {row['total_gpu_kernel_ns']/1e6:.1f} | "
            f"{row['bnb_4bit_gemm_ns']/1e6:.1f} ({row['bnb_4bit_gemm_share_of_kernel_time']:.1%}) | "
            f"{row['bnb_dequant_ns']/1e6:.1f} ({row['bnb_dequant_share_of_kernel_time']:.1%}) | "
            f"{row['synchronization_api_ns']/1e6:.1f} ({row['synchronization_api_calls']}) | "
            f"{row['kernel_launch_api_ns']/1e6:.1f} ({row['kernel_launch_api_calls']}) |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- BNB 4-bit GEMM and blockwise dequantization are measurable contributors. Their shares vary by method; the profile alone does not establish that either is the limiting critical path.",
        "- Synchronization API totals are host-side API durations, not pure GPU idle time. Pair with an uninstrumented production benchmark and a timeline/critical-path inspection before attributing slowdown.",
        "- Memory-transfer summaries are reported in results.json. Their small aggregate duration in these captures does not characterize DRAM bandwidth or arithmetic intensity.",
        "- Targeted NCU throughput samples are summarized below. They do not provide a matched FLOP/byte estimate, so no arithmetic-intensity claim is made.",
        "- Environment, capture configuration, raw capture hashes and exported summary paths are stored in manifest.json.",
        "",
    ]
    report_path = PROFILE_DIR / "report.md"
    old_report = report_path.read_text() if report_path.exists() else ""
    marker = "\n## Targeted Nsight Compute samples"
    ncu_section = old_report[old_report.find(marker):] if marker in old_report else ""
    report_path.write_text("\n".join(lines) + ncu_section)
    print(f"wrote {out.relative_to(ROOT)} and report.md")


if __name__ == "__main__":
    main()
