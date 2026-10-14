#!/usr/bin/env python3
"""Summarize the matched one-prompt backend smoke benchmark."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/backend_validation/paired_benchmark"
SOURCES = {
    "hf_bnb_nf4": DIR / "bnb_smoke_paired_warm8.jsonl",
    "hqq_gemlite_w4a16": DIR / "hqq_smoke_paired_warm8.jsonl",
}


def main() -> None:
    by_backend: dict[str, dict[str, dict]] = {}
    for backend, path in SOURCES.items():
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(records) != 3 or {r["method"] for r in records} != {"vanilla", "pld_k4", "cka90_k2"}:
            raise ValueError(f"incomplete paired run: {path}")
        if len({r["prompt_token_ids_sha256"] for r in records}) != 1:
            raise ValueError(f"prompt hash mismatch within backend: {path}")
        by_backend[backend] = {r["method"]: r for r in records}
    bnb_vanilla = by_backend["hf_bnb_nf4"]["vanilla"]["token_ids"]
    hqq_vanilla = by_backend["hqq_gemlite_w4a16"]["vanilla"]["token_ids"]
    cross_div = next((i for i, (a, b) in enumerate(zip(bnb_vanilla, hqq_vanilla)) if a != b), None)
    if cross_div is None and len(bnb_vanilla) != len(hqq_vanilla):
        cross_div = min(len(bnb_vanilla), len(hqq_vanilla))
    summary = {
        "scope": "diagnostic pilot only; one prompt, one repetition, 8 output tokens, 8-token warmup",
        "prompt_id": by_backend["hf_bnb_nf4"]["vanilla"]["prompt_id"],
        "prompt_tokens": by_backend["hf_bnb_nf4"]["vanilla"]["prompt_tokens"],
        "prompt_token_ids_sha256": by_backend["hf_bnb_nf4"]["vanilla"]["prompt_token_ids_sha256"],
        "max_new_tokens": 8,
        "backends": {},
        "cross_backend_vanilla_first_token_difference": cross_div,
        "cross_backend_vanilla_exact": bnb_vanilla == hqq_vanilla,
        "excluded_cold_warmup_file": "hqq_smoke_paired.jsonl",
        "exclusion_reason": "one-token warmup did not exercise the full CKA speculative path; measured CKA latency included cold-path setup and is not a comparable warm result",
    }
    for backend, methods in by_backend.items():
        baseline_tps = methods["vanilla"]["decode_tokens_per_second"]
        summary["backends"][backend] = {
            method: {
                "decode_tokens_per_second": record["decode_tokens_per_second"],
                "end_to_end_seconds": record["total_time_s"],
                "ttft_prefill_seconds": record["prefill_time_s"],
                "internal_decode_speedup": record["decode_tokens_per_second"] / baseline_tps,
                "within_backend_exact_vs_vanilla": record["within_backend_exact_vs_vanilla"],
                "first_divergence_vs_vanilla": record["first_divergence_vs_vanilla"],
                "peak_allocated_mb": record["peak_allocated_mb"],
                "peak_reserved_mb": record["peak_reserved_mb"],
                "acceptance_rate": record.get("acceptance_rate"),
            }
            for method, record in methods.items()
        }
    out = DIR / "pilot_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Paired backend pilot",
        "",
        "This smoke uses the same prepared 384-token prompt, pinned Qwen2.5-3B revision, static KV capacity 436, SDPA, greedy decoding, and 8-token output budget on both backends. Each backend was loaded once, all three methods received an 8-token warmup, and one measured repetition used the recorded randomized method order. This is harness and compatibility evidence, not a publishable performance result or confidence interval.",
        "",
        "## Within-backend results",
        "",
        "| Backend | Method | Decode TPS | Internal speedup | E2E (s) | Exact vs own vanilla | Peak allocated (MB) |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for backend, methods in summary["backends"].items():
        for method, row in methods.items():
            lines.append(
                f"| {backend} | {method} | {row['decode_tokens_per_second']:.2f} | "
                f"{row['internal_decode_speedup']:.3f}× | {row['end_to_end_seconds']:.3f} | "
                f"{row['within_backend_exact_vs_vanilla']} | {row['peak_allocated_mb']:.1f} |"
            )
    lines += [
        "",
        f"The direct vanilla token sequences first differ across backends at generated position {cross_div}; this is cross-backend quantization fidelity, not speculative decoding correctness.",
        "",
        "In this one-prompt pilot, both PLD K4 and CKA90/K2 were slower than vanilla within each backend, while all measured speculative outputs matched that backend's vanilla output for these eight tokens. Do not generalize this result: only one prompt and one measured repetition were used, and the short output can miss later divergence and context-dependent performance.",
        "",
        "The first HQQ file (`hqq_smoke_paired.jsonl`) used only one warmup token. CKA's warmup did not exercise the full speculative path, so its large measured verification latency included cold-path work; that file is retained as exploratory evidence but excluded from this comparison.",
        "",
        "Next: use a full-path warmup (at least one complete 32-token speculative run), run 3+ sessions over the matched prompt cohort, and keep the within-backend exactness record for every prompt. Expand contexts/output budgets only after that pilot is stable.",
        "",
    ]
    (DIR / "report.md").write_text("\n".join(lines))
    print(f"wrote {out.relative_to(ROOT)} and report.md")


if __name__ == "__main__":
    main()
