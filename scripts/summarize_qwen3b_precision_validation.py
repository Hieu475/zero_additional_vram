#!/usr/bin/env python3
"""Rebuild WP3 summary tables from the immutable per-prompt JSONL outputs."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ("sdpa_static", "eager_static", "sdpa_dynamic", "sdpa_static_sequential")


def summarize(rows: list[dict]) -> dict:
    cohorts = {}
    for cohort in ("diagnostic", "heldout"):
        group = [r for r in rows if r["cohort"] == cohort]
        cohorts[cohort] = {
            "n": len(group),
            "exact_match": sum(r["sequence_exact_match"] for r in group) / max(1, len(group)),
            "token_divergences": sum(r["first_token_divergence"] is not None for r in group),
            "kv_drift": sum(r.get("first_kv_divergence_cycle") is not None for r in group),
            "logit_drift": sum(r.get("first_logit_divergence") is not None for r in group),
        }
    attribution = Counter(r.get("wp2_factorial_attribution") or "unavailable" for r in rows)
    return {
        "n": len(rows), "cohorts": cohorts,
        "sequence_exact_match_rate": sum(r["sequence_exact_match"] for r in rows) / max(1, len(rows)),
        "token_argmax_agreement_mean": sum(r["argmax_agreement"] for r in rows) / max(1, len(rows)),
        "token_divergences": sum(r["first_token_divergence"] is not None for r in rows),
        "kv_drift": sum(r.get("first_kv_divergence_cycle") is not None for r in rows),
        "logit_drift": sum(r.get("first_logit_divergence") is not None for r in rows),
        "wp2_factorial_attribution_counts": dict(attribution),
        "mean_tps": sum(r["tokens_per_second"] for r in rows) / max(1, len(rows)),
        "mean_verification_latency_s": sum(r["verification_latency_s"] for r in rows) / max(1, len(rows)),
        "mean_peak_vram_mb": sum(r["peak_vram_mb"] for r in rows) / max(1, len(rows)),
    }


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments/numerical_causality/precision_validation_final"
    report = ["# Qwen2.5-3B NF4 precision/backend validation", ""]
    manifest_path = outdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    report += [f"Execution commit: `{manifest['execution_commit']}`. Model revision: `{manifest['model_revision']}`.",
               f"Hardware: `{manifest['hardware']}`; cohort: `{manifest['cohort_counts']}`.",
               "NF4 double quant, FP16 compute, greedy decode, 32 token limit, K=4.",
               "", "## Cohort results", "",
               "| Config | Cohort | N | Exact match | KV drift | Logit drift | Token divergence | Mean argmax agreement | TPS | VRAM MB |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for config in CONFIGS:
        path = outdir / f"{config}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        summary = summarize(rows)
        summary["config"] = config
        summary["output"] = str(path)
        (outdir / f"{config}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        for cohort, stats in summary["cohorts"].items():
            group = [r for r in rows if r["cohort"] == cohort]
            tps = sum(r["tokens_per_second"] for r in group) / max(1, len(group))
            vram = sum(r["peak_vram_mb"] for r in group) / max(1, len(group))
            argmax = sum(r["argmax_agreement"] for r in group) / max(1, len(group))
            report.append(f"| {config} | {cohort} | {stats['n']} | {stats['exact_match']:.3f} | {stats['kv_drift']} | {stats['logit_drift']} | {stats['token_divergences']} | {argmax:.4f} | {tps:.2f} | {vram:.0f} |")
        report += ["", f"WP2 intervention labels linked for the 16 diagnostic prompts in {config}: `{ {k: v for k, v in summary['wp2_factorial_attribution_counts'].items() if k != 'unavailable'} }`. Held-out prompts have no WP2 label.", ""]
    report += ["## Interpretation and limits", "",
               "KV/logit drift, argmax changes and sequence mismatch are counted separately. A numerical difference does not imply a changed token. Exact-match applies only to this 30-prompt prepared-input cohort and greedy 32-token run; it is not a lossless guarantee.",
               "The held-out rows are unselected rows from the same prepared-input artifact, not an external dataset. FP16/BF16 unquantized and FP32 controls were not run on this 6 GB GPU; they remain open.",
               "For diagnostic prompts, WP2 attribution is an intervention result from the matched SDPA/static protocol. It is linked for context and is not re-estimated independently for the other backend configurations.",
               "TPS is measured with numerical tracing enabled and includes trace snapshot overhead; treat it as diagnostic instrumentation timing, not a production speed benchmark. Verification latency measures the target verification forward pass. Max-relative error can become very large when reference logits are near zero; interpret it with MAE, max-absolute error and top-1 margin.",
               "Per-prompt token IDs, cycle-level KV/logit errors, margins, EOS positions and environment manifest are stored in the adjacent JSONL and manifest files.", ""]
    (outdir / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
