#!/usr/bin/env python3
"""Run a small-model precision/backend matrix with batched and sequential PLD."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MATRIX = [
    ("nf4", "sdpa", "static"),
    ("nf4", "eager", "static"),
    ("nf4", "sdpa", "dynamic"),
    ("fp16", "sdpa", "static"),
    ("bf16", "sdpa", "static"),
    ("fp32", "eager", "static"),
    ("fp32", "sdpa", "static"),
]
PROMPT_IDS = [
    "qasper_119_ctx384", "qasper_109_ctx384", "gov_report_98_ctx384",
    "qmsum_2_ctx384",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Use a smaller model when the target model does not fit FP16/FP32")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--benchmark", default="experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl")
    ap.add_argument("--output-dir", default="experiments/cross_precision_validation/qwen25_05b_rtx4050")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--only", nargs="*", help="Optional precision labels, e.g. fp16 bf16")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_root = root / args.output_dir
    out_root.mkdir(parents=True, exist_ok=True)
    selected = [row for row in MATRIX if args.only is None or row[0] in args.only]
    manifest_path = out_root / "matrix_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["model"] = args.model
        manifest["revision"] = args.revision
        manifest["scope_note"] = "Small-model mechanism validation does not establish Qwen2.5-3B behavior."
    else:
        manifest = {
            "model": args.model, "revision": args.revision,
            "scope_note": "Small-model mechanism validation does not establish Qwen2.5-3B behavior.",
            "matrix": [],
        }
    entries = {row["label"]: row for row in manifest.get("matrix", [])}
    manifest["matrix"] = list(entries.values())
    trace_script = root / "scripts/trace_multicycle_numerical_divergence.py"
    analyze_script = root / "scripts/analyze_numerical_causality.py"
    for precision, attention, cache in selected:
        for verification in ("batched", "sequential"):
            label = f"{precision}_{attention}_{cache}_{verification}"
            run_dir = out_root / label
            trace_path = run_dir / "traces.json"
            command = [
                sys.executable, str(trace_script), "--model", args.model,
                "--benchmark", str(root / args.benchmark), "--ids", *PROMPT_IDS,
                "--samples", str(len(PROMPT_IDS)), "--k", str(args.k),
                "--max-new-tokens", str(args.max_new_tokens),
                "--precision", precision, "--attention-implementation", attention,
                "--backend", cache, "--verification-mode", verification,
                "--output", str(trace_path),
            ]
            if args.revision:
                command.extend(["--revision", args.revision])
            entry = entries.get(label, {"label": label})
            entry.update({"command": command, "status": "running"})
            if label not in entries:
                manifest["matrix"].append(entry)
                entries[label] = entry
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            result = subprocess.run(command, cwd=root, check=False)
            entry["returncode"] = result.returncode
            entry["status"] = "complete" if result.returncode == 0 else "failed"
            if result.returncode == 0:
                analysis = [sys.executable, str(analyze_script), "--input", str(trace_path),
                            "--output-dir", str(run_dir)]
                analyzed = subprocess.run(analysis, cwd=root, check=False)
                entry["analysis_returncode"] = analyzed.returncode
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, command)
    print(f"Saved matrix manifest: {manifest_path}")


if __name__ == "__main__":
    main()
