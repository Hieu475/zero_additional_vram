#!/usr/bin/env python3
"""Summarize one-session paired backend smoke runs across context lengths."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/backend_validation/context_matrix"
CONTEXTS = (128, 384, 1024, 2048)
BACKENDS = ("hf_bnb_nf4", "hqq_gemlite_w4a16")
METHODS = ("vanilla", "pld_k4", "cka90_k2")
FILES = {
    (128, "hf_bnb_nf4"): "ctx128_bnb_session1.jsonl",
    (384, "hf_bnb_nf4"): "ctx384_bnb_session1.jsonl",
    (1024, "hf_bnb_nf4"): "ctx1024_bnb_session1.jsonl",
    (2048, "hf_bnb_nf4"): "ctx2048_bnb_session1.jsonl",
    (128, "hqq_gemlite_w4a16"): "ctx128_hqq_session1_retry1.jsonl",
    (384, "hqq_gemlite_w4a16"): "ctx384_hqq_session1.jsonl",
    (1024, "hqq_gemlite_w4a16"): "ctx1024_hqq_session1.jsonl",
    (2048, "hqq_gemlite_w4a16"): "ctx2048_hqq_session1_retry1.jsonl",
}


def main() -> None:
    data = {}
    statuses = {}
    for context in CONTEXTS:
        for backend in BACKENDS:
            key = (context, backend)
            path = DIR / FILES[key]
            if not path.exists():
                statuses[key] = {"status": "missing", "file": path.name, "records": 0}
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            by_prompt = {}
            for row in rows:
                by_prompt.setdefault(row["prompt_id"], {})[row["method"]] = row
            if len(by_prompt) != 5 or any(set(v) != set(METHODS) for v in by_prompt.values()):
                statuses[key] = {"status": "partial", "file": path.name, "records": len(rows)}
                continue
            if any(r.get("status") != "ok" or not r.get("warmup_status", {}).get("passed")
                   for methods in by_prompt.values() for r in methods.values()):
                statuses[key] = {"status": "failed_or_warmup_failed", "file": path.name, "records": len(rows)}
                continue
            data[key] = by_prompt
            statuses[key] = {"status": "complete", "file": path.name, "records": len(rows),
                             "prompts": len(by_prompt),
                             "capacity": rows[0]["kv_capacity"],
                             "input_tokens": rows[0]["prompt_tokens"],
                             "max_new_tokens": rows[0]["max_new_tokens"]}

    summary = {"protocol": {"contexts": list(CONTEXTS), "prompts_per_context": 5,
                             "sessions_per_cell": 1, "output_budget": 32,
                             "warmup_tokens": 32,
                             "paired_source_documents": True,
                             "inference": "greedy, static KV, SDPA, Qwen2.5-3B-Instruct pinned revision",
                             "statistics": "descriptive only; n=5 prompts, one session per cell"},
               "cell_status": {f"ctx{c}_{b}": v for (c,b),v in statuses.items()},
               "results": {}, "cross_backend_vanilla_exact": {}}
    lines = ["# Paired context-length backend smoke", "",
             "The same five LongBench source documents (one each from narrativeqa, qasper, gov_report, multi_news, qmsum) are represented at 128, 384, 1024, and 2048 input tokens. Each cell is one process/session with a 32-token full-path warmup, 32-token output budget, static capacity `context + 36`, greedy SDPA. This is a smoke run, not a performance conclusion: one session and five prompts per cell.", "",
             "| Context | Backend | Cell | Method | Decode TPS mean | Internal speedup vs vanilla | Slower fraction | Exact vs own vanilla | Peak allocated MB |", "|---:|---|---|---|---:|---:|---:|---:|---:|"]
    for context in CONTEXTS:
        for backend in BACKENDS:
            key = (context, backend)
            cell = data.get(key)
            if cell is None:
                lines.append(f"| {context} | {backend} | {statuses[key]['status']} | — | — | — | — | — | — |")
                continue
            methods_result = {}
            prompt_ids = list(cell)
            for method in METHODS:
                rows = [cell[p][method] for p in prompt_ids]
                tps = [float(r["decode_tokens_per_second"]) for r in rows]
                exact = sum(bool(r.get("within_backend_exact_vs_vanilla")) for r in rows)
                alloc = [float(r["peak_allocated_mb"]) for r in rows]
                methods_result[method] = {"decode_tps_mean": statistics.mean(tps),
                                          "exact_count": exact, "prompt_count": len(rows),
                                          "peak_allocated_mb_mean": statistics.mean(alloc),
                                          "actual_output_lengths": sorted({len(r["token_ids"]) for r in rows})}
                if method != "vanilla":
                    ratios = [cell[p][method]["decode_tokens_per_second"] /
                              cell[p]["vanilla"]["decode_tokens_per_second"] for p in prompt_ids]
                    methods_result[method]["paired_speedup_mean"] = statistics.mean(ratios)
                    methods_result[method]["slowdown_frequency"] = sum(x < 1 for x in ratios) / len(ratios)
                alloc_mean = methods_result[method]["peak_allocated_mb_mean"]
                speed = methods_result[method].get("paired_speedup_mean")
                slowdown = methods_result[method].get("slowdown_frequency")
                exact_txt = f"{exact}/{len(rows)}"
                lines.append(f"| {context} | {backend} | complete | {method} | {statistics.mean(tps):.2f} | "
                             f"{'—' if speed is None else format(speed, '.3f') + '×'} | "
                             f"{'—' if slowdown is None else format(slowdown, '.0%')} | "
                             f"{exact_txt} | {alloc_mean:.0f} |")
            summary["results"][f"ctx{context}_{backend}"] = methods_result
    for context in CONTEXTS:
        key_bnb, key_hqq = (context, BACKENDS[0]), (context, BACKENDS[1])
        if key_bnb not in data or key_hqq not in data:
            continue
        bnb, hqq = data[key_bnb], data[key_hqq]
        ids = set(bnb) & set(hqq)
        exact = sum(bnb[p]["vanilla"]["token_ids"] == hqq[p]["vanilla"]["token_ids"] for p in ids)
        summary["cross_backend_vanilla_exact"][str(context)] = {"exact": exact, "n": len(ids)}
        for prompt in ids:
            hashes = {bnb[prompt][m]["prompt_token_ids_sha256"] for m in METHODS}
            hashes |= {hqq[prompt][m]["prompt_token_ids_sha256"] for m in METHODS}
            if len(hashes) != 1:
                raise ValueError(f"cross-backend prompt token mismatch at context {context}: {prompt}")
    lines += ["", "Cross-backend vanilla exactness by context: " + "; ".join(
        f"{c}: {x['exact']}/{x['n']}" for c, x in summary["cross_backend_vanilla_exact"].items()),
        "", "Token exactness is measured against each backend's own vanilla output. Cross-backend output differences reflect stack/quantization fidelity. Do not infer lossless acceleration from speedup alone.", ""]
    (DIR / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (DIR / "smoke_report.md").write_text("\n".join(lines))
    print(f"wrote {DIR.name}/smoke_summary.json and smoke_report.md")


if __name__ == "__main__":
    main()
