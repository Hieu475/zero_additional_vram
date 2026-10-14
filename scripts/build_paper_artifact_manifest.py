#!/usr/bin/env python3
"""Build the central provenance manifest for current ZASSD evidence bundles."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/paper_artifact_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def artifact(path: str, role: str) -> dict:
    p = ROOT / path
    return {"path": path, "role": role, "bytes": p.stat().st_size, "sha256": sha(p)}


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True,
                            capture_output=True, text=True)
    return result.stdout.strip()


def main() -> None:
    groups = {
        "WP1_reproduction_audit": {
            "claims": ["NF4 double-quant discrepancy reproduces the 8/16 versus 16/16 cohort result", "Current production/lightweight/full trace replays agree on the diagnostic prompts"],
            "scripts": ["scripts/compare_benchmark_trace_reproduction.py", "scripts/replay_benchmark_trace.py"],
            "artifacts": [
                ("experiments/reproduction_audit/reproduction_report.md", "analysis report"),
                ("experiments/reproduction_audit/benchmark_manifest.json", "benchmark protocol"),
                ("experiments/reproduction_audit/tracer_manifest.json", "tracer protocol"),
                ("experiments/reproduction_audit/prompt_comparison.json", "prompt and output comparison"),
                ("experiments/reproduction_audit/first_difference.json", "first-difference evidence"),
                ("experiments/reproduction_audit/live_replay.json", "current controlled replay"),
                ("experiments/reproduction_audit/no_double_quant_replay.json", "quantization toggle replay"),
            ],
        },
        "WP2_factorial_KV_verification": {
            "claims": ["Diagnostic factorial attribution: 15 KV-history-sensitive and 1 execution-sensitive"],
            "scripts": ["scripts/run_kv_factorial_intervention.py"],
            "artifacts": [
                ("experiments/numerical_causality/interventions/factorial_report.md", "intervention report"),
                ("experiments/numerical_causality/interventions/factorial_summary_final_v2.json", "cohort summary"),
                ("experiments/numerical_causality/interventions/factorial_kv_verification_final_v2.jsonl", "raw factorial measurements"),
            ],
        },
        "WP3_precision_backend": {
            "claims": ["30 prompts split into 16 diagnostic and 14 held-out", "Sequential verification matched vanilla 30/30 for the measured NF4 run", "Unquantized FP16/BF16 and FP32 remain unmeasured"],
            "scripts": ["scripts/run_qwen3b_precision_backend_validation.py", "scripts/summarize_qwen3b_precision_validation.py"],
            "artifacts": [
                ("experiments/numerical_causality/precision_validation_final/report.md", "precision/backend report"),
                ("experiments/numerical_causality/precision_validation_final/manifest.json", "run environment and config"),
                ("experiments/numerical_causality/precision_validation_final/sdpa_static.jsonl", "SDPA static raw data"),
                ("experiments/numerical_causality/precision_validation_final/eager_static.jsonl", "eager static raw data"),
                ("experiments/numerical_causality/precision_validation_final/sdpa_dynamic.jsonl", "SDPA dynamic raw data"),
                ("experiments/numerical_causality/precision_validation_final/sdpa_static_sequential.jsonl", "sequential verification raw data"),
            ],
        },
        "WP3_exploratory_precision_backend": {
            "claims": ["Initial 30-prompt NF4 precision/backend measurements; exploratory run superseded by precision_validation_final for reported conclusions"],
            "scripts": ["scripts/run_qwen3b_precision_backend_validation.py", "scripts/summarize_qwen3b_precision_validation.py"],
            "artifacts": [
                ("experiments/numerical_causality/precision_validation/report.md", "exploratory precision/backend report"),
                ("experiments/numerical_causality/precision_validation/manifest.json", "exploratory run environment and config"),
                ("experiments/numerical_causality/precision_validation/sdpa_static.jsonl", "exploratory SDPA static raw data"),
                ("experiments/numerical_causality/precision_validation/eager_static.jsonl", "exploratory eager static raw data"),
                ("experiments/numerical_causality/precision_validation/sdpa_dynamic.jsonl", "exploratory SDPA dynamic raw data"),
                ("experiments/numerical_causality/precision_validation/sdpa_static_sequential.jsonl", "exploratory sequential raw data"),
            ],
        },
        "WP4_paired_router": {
            "claims": ["Hybrid router is exact on the 14 held-out prompts but has no detected speed advantage over PLD K4", "The per-prompt fixed-method comparator is offline and non-deployable"],
            "scripts": ["scripts/run_paired_router_evaluation.py", "scripts/summarize_paired_router_evaluation.py"],
            "artifacts": [
                ("experiments/numerical_causality/unified_router_evaluation/report.md", "paired evaluation report"),
                ("experiments/numerical_causality/unified_router_evaluation/config.json", "frozen protocol and split"),
                ("experiments/numerical_causality/unified_router_evaluation/environment.json", "runtime environment"),
                ("experiments/numerical_causality/unified_router_evaluation/manifest.json", "run manifest with raw hash"),
                ("experiments/numerical_causality/unified_router_evaluation/raw_measurements.jsonl", "210 prompt-method paired records"),
                ("experiments/numerical_causality/unified_router_evaluation/summary.json", "bootstrap and paired analysis"),
                ("paper/tables/wp4_paired_router_table.tex", "paper held-out comparison table"),
            ],
        },
        "paper_draft": {
            "claims": ["Manuscript includes a WP4 matched-cohort result and labels router as a negative performance result"],
            "scripts": ["scripts/build_paper_artifact_manifest.py"],
            "artifacts": [
                ("paper/draft/main.tex", "manuscript source"),
                ("paper/draft/main.pdf", "compiled manuscript"),
                ("paper/draft/references.bib", "bibliography source"),
            ],
        },
    }
    for group in groups.values():
        group["source_scripts"] = [artifact(p, "execution/analysis code") for p in group.pop("scripts")]
        group["artifacts"] = [artifact(p, role) for p, role in group["artifacts"]]

    manifest = {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {"path": str(ROOT), "branch": git_value("branch", "--show-current"),
                       "head_commit": git_value("rev-parse", "HEAD"),
                       "worktree_note": "Evidence and manuscript files are present in the current worktree; execution and source-script hashes are recorded separately."},
        "global_model": {"name": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1"},
        "hardware": "NVIDIA GeForce RTX 4050 Laptop GPU, 6 GB",
        "groups": groups,
        "open_limits": ["Only the NF4 precision/backend matrix was run on RTX 4050.", "WP4 has 14 held-out prompts and one run per method/prompt; intervals do not capture session or thermal variance.", "The WP4 development subset is drawn from a diagnostic mismatch cohort.", "Offline best-per-prompt results use observed future timings and are upper-bound diagnostics."],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(relative(OUT))


if __name__ == "__main__":
    main()
