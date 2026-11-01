#!/usr/bin/env python3
"""Export and summarize Nsight Systems kernel/API reports for four K2 traces."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
TRACES = (
    (256, "HumanEval/123", "nsys_k2_256_he123"),
    (512, "HumanEval/123", "nsys_k2_512_he123"),
    (256, "HumanEval/114", "nsys_k2_256_he114"),
    (512, "HumanEval/114", "nsys_k2_512_he114"),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_csv(stem: str, rep: Path, report: str) -> Path:
    base = BASE / f"{stem}_stats"
    path = Path(f"{base}_{report}.csv")
    if not path.exists():
        subprocess.run(["nsys", "stats", "--report", report, "--format", "csv",
                        "--output", str(base), str(rep)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    return path


def main() -> None:
    traces = []
    for budget, prompt_id, stem in TRACES:
        rep, meta = BASE / f"{stem}.nsys-rep", BASE / f"{stem}.json"
        if not rep.exists() or not meta.exists():
            raise FileNotFoundError(f"missing Nsight capture: {stem}")
        capture = json.loads(meta.read_text())
        if capture.get("schema") != "zassd-k2-nsys-capture-v1" or not capture.get("b1_token_and_committed_kv_exact"):
            raise RuntimeError(f"invalid or non-exact capture: {stem}")

        kernel_csv = report_csv(stem, rep, "cuda_gpu_kern_sum")
        api_csv = report_csv(stem, rep, "cuda_api_sum")
        with kernel_csv.open(newline="") as f:
            kernel_rows = list(csv.DictReader(f))
        with api_csv.open(newline="") as f:
            api_rows = list(csv.DictReader(f))
        top = [{
            "time_percent": float(row["Time (%)"]),
            "total_time_ms": float(row["Total Time (ns)"]) / 1e6,
            "instances": int(row["Instances"]),
            "average_us": float(row["Avg (ns)"]) / 1e3,
            "name": row["Name"],
        } for row in kernel_rows[:12]]
        api_top = [{
            "time_percent": float(row["Time (%)"]),
            "total_time_ms": float(row["Total Time (ns)"]) / 1e6,
            "calls": int(row["Num Calls"]),
            "name": row["Name"],
        } for row in api_rows[:8]]
        total_kernel_ms = sum(float(row["Total Time (ns)"]) for row in kernel_rows) / 1e6
        traces.append({
            "prompt_id": prompt_id,
            "max_new_tokens": budget,
            "b1_exact": True,
            "reported_output_tokens": capture["metrics"]["output_tokens"],
            "cycles": capture["metrics"]["cycles"],
            "target_rows": capture["metrics"]["target_rows"],
            "accepted_draft_tokens": capture["metrics"]["accepted_draft_tokens"],
            "total_profiled_gpu_kernel_ms": total_kernel_ms,
            "top_kernels": top,
            "top_cuda_apis": api_top,
            "nsys_rep": rep.name,
            "nsys_rep_sha256": sha(rep),
            "capture_json_sha256": sha(meta),
            "kernel_csv": kernel_csv.name,
            "api_csv": api_csv.name,
        })

    result = {
        "schema": "zassd-k2-nsys-profile-summary-v1",
        "classification": "diagnostic Nsight Systems summaries; instrumented, not production latency",
        "scope": "Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, K=2/R=3, full native-BNB fallback",
        "trace_count": len(traces),
        "all_captures_b1_exact": all(x["b1_exact"] for x in traces),
        "traces": traces,
        "limits": [
            "Two selected HumanEval prompts and one GPU; kernel percentages describe these instrumented traces.",
            "CUDA API durations are host-side API timings and can overlap; do not sum them as GPU execution time.",
            "Nsight instrumentation changes runtime; use paired unprofiled runs for latency claims.",
            "GSM8K held-out indices 300-349 excluded.",
        ],
        "source": str(Path(__file__).resolve()),
        "source_sha256": sha(Path(__file__)),
    }
    out = BASE / "nsys_k2_profile_summary_v1.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), "traces": [
        {"prompt_id": t["prompt_id"], "tokens": t["max_new_tokens"],
         "total_gpu_kernel_ms": t["total_profiled_gpu_kernel_ms"],
         "top_kernels": [{"pct": k["time_percent"], "ms": k["total_time_ms"], "instances": k["instances"], "name": k["name"][:110]} for k in t["top_kernels"][:3]]}
        for t in traces]}, indent=2))


if __name__ == "__main__":
    main()
