#!/usr/bin/env python3
"""Analyze matched 32-token backend pilots after a full-path warmup."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/backend_validation/paired_benchmark"
FILES = {
    "hf_bnb_nf4": DIR / "bnb_ctx384_32_fullwarm.jsonl",
    "hqq_gemlite_w4a16": DIR / "hqq_ctx384_32_fullwarm.jsonl",
}


def read_backend(path: Path) -> dict[str, dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != 3 or {r["method"] for r in records} != {"vanilla", "pld_k4", "cka90_k2"}:
        raise ValueError(f"incomplete method set in {path}")
    if any(r["max_new_tokens"] != 32 or r["prompt_tokens"] != 384 for r in records):
        raise ValueError(f"unexpected workload in {path}")
    return {r["method"]: r for r in records}


def main() -> None:
    raw = {backend: read_backend(path) for backend, path in FILES.items()}
    hashes = {methods["vanilla"]["prompt_token_ids_sha256"] for methods in raw.values()}
    if len(hashes) != 1:
        raise ValueError("prompt token IDs differ across backend runs")
    summary = {
        "scope": "one prompt, one repetition per backend and method; 32-token warmup and output; diagnostic only",
        "prompt_id": raw["hf_bnb_nf4"]["vanilla"]["prompt_id"],
        "prompt_tokens": 384,
        "prompt_token_ids_sha256": next(iter(hashes)),
        "max_new_tokens": 32,
        "backends": {},
    }
    for backend, methods in raw.items():
        vanilla_tps = methods["vanilla"]["decode_tokens_per_second"]
        summary["backends"][backend] = {}
        for method, record in methods.items():
            summary["backends"][backend][method] = {
                "decode_tokens_per_second": record["decode_tokens_per_second"],
                "internal_speedup": record["decode_tokens_per_second"] / vanilla_tps,
                "end_to_end_seconds": record["total_time_s"],
                "prefill_seconds": record["prefill_time_s"],
                "within_backend_exact_vs_vanilla": record["within_backend_exact_vs_vanilla"],
                "first_divergence_vs_vanilla": record["first_divergence_vs_vanilla"],
                "peak_allocated_mb": record["peak_allocated_mb"],
                "peak_reserved_mb": record["peak_reserved_mb"],
                "acceptance_rate": record.get("acceptance_rate"),
                "execution_order": record["execution_order"],
            }
    bnb_ids = raw["hf_bnb_nf4"]["vanilla"]["token_ids"]
    hqq_ids = raw["hqq_gemlite_w4a16"]["vanilla"]["token_ids"]
    summary["cross_backend_vanilla_exact"] = bnb_ids == hqq_ids
    summary["cross_backend_vanilla_first_difference"] = next(
        (i for i, (a, b) in enumerate(zip(bnb_ids, hqq_ids)) if a != b),
        min(len(bnb_ids), len(hqq_ids)) if len(bnb_ids) != len(hqq_ids) else None,
    )
    (DIR / "fullwarm_32_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Paired backend full-warm pilot",
        "",
        "Same prepared prompt (384 tokens and identical token-ID hash), model revision, KV capacity 436, SDPA, greedy decoding, and 32-token output budget. Each backend was loaded once in a fresh process. Vanilla, PLD K4, and CKA90/K2 each received a separate 32-token warmup before one measured repetition. This is a diagnostic pilot, not a confidence interval or a publication result.",
        "",
        "| Backend | Method | Decode TPS | Internal speedup | E2E (s) | Exact vs own vanilla | First divergence | Peak allocated MB |",
        "|---|---|---:|---:|---:|---|---:|---:|",
    ]
    for backend, methods in summary["backends"].items():
        for method, row in methods.items():
            lines.append(
                f"| {backend} | {method} | {row['decode_tokens_per_second']:.2f} | "
                f"{row['internal_speedup']:.3f}× | {row['end_to_end_seconds']:.3f} | "
                f"{row['within_backend_exact_vs_vanilla']} | {row['first_divergence_vs_vanilla']} | "
                f"{row['peak_allocated_mb']:.1f} |"
            )
    lines += [
        "",
        f"Direct vanilla outputs differ across backends at generated position {summary['cross_backend_vanilla_first_difference']}. This records cross-backend quantization fidelity; it is not speculative correctness.",
        "",
        "On this prompt, both speculative methods are slower than the vanilla baseline within each backend. HQQ+GemLite matches its own vanilla for all 32 output tokens; BNB PLD and CKA outputs both first diverge from BNB vanilla at token position 23. The candidate stack's vanilla decode throughput is lower in this single session, and its peak allocated VRAM is higher. These results do not show a backend advantage and cannot be generalized beyond this prompt/session.",
        "",
        "The earlier `hqq_smoke_paired.jsonl` used only one warmup token and showed a cold CKA verification path; exclude its latency values. The 8-token warm pilot is recorded separately in `pilot_summary.json`.",
        "",
        "Next validation: at least 3 independent sessions across the diagnostic cohort, then held-out prompts and the longer context/output grid. Keep within-backend exactness separate from cross-backend token fidelity.",
        "",
    ]
    (DIR / "fullwarm_32_report.md").write_text("\n".join(lines))
    print("wrote fullwarm_32_summary.json and fullwarm_32_report.md")


if __name__ == "__main__":
    main()
