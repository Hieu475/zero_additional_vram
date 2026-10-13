#!/usr/bin/env python3
"""Run three independent 30-prompt reduced-sync screening sessions."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    output_root = root / "experiments/reduced_sync/repeated_screening_rtx4050"
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "screening_manifest.json"
    manifest = {
        "protocol": {
            "prompts": 30, "independent_sessions_per_config": 3,
            "max_new_tokens": 32, "k": 2, "repeats_per_process": 1,
            "configs": ["cka_90", "cka_67"],
            "methods": ["reference", "reduced_sync_conservative"],
            "profiling_instrumentation": False,
        },
        "sessions": [],
    }
    bench = root / "scripts/benchmark_reduced_sync.py"
    analyze = root / "scripts/analyze_reduced_sync.py"
    for config_index, config in enumerate(("cka_90", "cka_67")):
        for session in range(1, 4):
            out = output_root / config / f"session{session}"
            if out.exists() and any(out.iterdir()):
                raise FileExistsError(f"Refusing to overwrite existing run: {out}")
            seed = 20261020 + config_index * 10 + session
            command = [sys.executable, str(bench), "--output-dir", str(out),
                       "--limit", "30", "--repeats", "1", "--max-new-tokens", "32",
                       "--k", "2", "--config", config, "--seed", str(seed)]
            record = {"config": config, "session": session, "seed": seed,
                      "command": command, "status": "running"}
            manifest["sessions"].append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            result = subprocess.run(command, cwd=root, check=False)
            record["returncode"] = result.returncode
            record["status"] = "complete" if result.returncode == 0 else "failed"
            if result.returncode != 0:
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                raise subprocess.CalledProcessError(result.returncode, command)
            analysis = [sys.executable, str(analyze), "--raw", str(out / "raw_measurements.jsonl"),
                        "--output", str(out / "analysis.json"), "--seed", str(seed)]
            result = subprocess.run(analysis, cwd=root, check=False)
            record["analysis_returncode"] = result.returncode
            record["analysis"] = str(out / "analysis.json")
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, analysis)
    print(f"Saved screening runs under {output_root}")


if __name__ == "__main__":
    main()
