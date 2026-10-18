#!/usr/bin/env python3
"""Summarize one-session paired backend smoke runs across context lengths."""
from __future__ import annotations

import json
import argparse
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/backend_validation/context_matrix"
CONTEXTS = (128, 384, 1024, 2048)
BACKENDS = ("hf_bnb_nf4", "hqq_gemlite_w4a16")
METHODS = ("vanilla", "pld_k4", "cka90_k2")
SMOKE_FILES = {
    (128, "hf_bnb_nf4"): "ctx128_bnb_session1.jsonl",
    (384, "hf_bnb_nf4"): "ctx384_bnb_session1.jsonl",
    (1024, "hf_bnb_nf4"): "ctx1024_bnb_session1.jsonl",
    (2048, "hf_bnb_nf4"): "ctx2048_bnb_session1.jsonl",
    (128, "hqq_gemlite_w4a16"): "ctx128_hqq_session1_retry1.jsonl",
    (384, "hqq_gemlite_w4a16"): "ctx384_hqq_session1.jsonl",
    (1024, "hqq_gemlite_w4a16"): "ctx1024_hqq_session1.jsonl",
    (2048, "hqq_gemlite_w4a16"): "ctx2048_hqq_session1_retry1.jsonl",
}

OUT128_FILES = {
    (128, "hf_bnb_nf4"): ["ctx128_out128_bnb_session1.jsonl", "ctx128_out128_bnb_session2.jsonl"],
    (384, "hf_bnb_nf4"): ["ctx384_out128_bnb_session1.jsonl", "ctx384_out128_bnb_session2.jsonl"],
    (1024, "hf_bnb_nf4"): ["ctx1024_out128_bnb_session1.jsonl", "ctx1024_out128_bnb_session2.jsonl"],
    (2048, "hf_bnb_nf4"): ["ctx2048_out128_bnb_session1.jsonl", "ctx2048_out128_bnb_session2.jsonl"],
    (128, "hqq_gemlite_w4a16"): ["ctx128_out128_hqq_session1.jsonl", "ctx128_out128_hqq_session2.jsonl"],
    (384, "hqq_gemlite_w4a16"): ["ctx384_out128_hqq_session1.jsonl", "ctx384_out128_hqq_session2.jsonl"],
    (1024, "hqq_gemlite_w4a16"): ["ctx1024_out128_hqq_session1.jsonl", "ctx1024_out128_hqq_session2.jsonl"],
    (2048, "hqq_gemlite_w4a16"): ["ctx2048_out128_hqq_session1_retry1.jsonl", "ctx2048_out128_hqq_session2.jsonl"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-budget", type=int, choices=(32, 128), default=32)
    args = parser.parse_args()
    output_budget = args.output_budget
    files = SMOKE_FILES if output_budget == 32 else OUT128_FILES
    stem = "smoke" if output_budget == 32 else "out128"
    sessions_per_cell = 1 if output_budget == 32 else 2
    data = {}
    statuses = {}
    for context in CONTEXTS:
        for backend in BACKENDS:
            key = (context, backend)
            paths = [DIR / name for name in files[key]] if isinstance(files[key], list) else [DIR / files[key]]
            missing = [path.name for path in paths if not path.exists()]
            if missing:
                statuses[key] = {"status": "missing", "file": missing, "records": 0}
                continue
            rows = [json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
            by_prompt = {}
            for row in rows:
                session_number = row["session_id"].split("session", 1)[-1][0]
                by_prompt.setdefault((row["prompt_id"], session_number), {})[row["method"]] = row
            if len(by_prompt) != 5 * sessions_per_cell or any(set(v) != set(METHODS) for v in by_prompt.values()):
                statuses[key] = {"status": "partial", "file": [path.name for path in paths], "records": len(rows)}
                continue
            if any(r.get("status") != "ok" or not r.get("warmup_status", {}).get("passed")
                   for methods in by_prompt.values() for r in methods.values()):
                statuses[key] = {"status": "failed_or_warmup_failed", "file": [path.name for path in paths], "records": len(rows)}
                continue
            data[key] = by_prompt
            statuses[key] = {"status": "complete", "file": [path.name for path in paths], "records": len(rows),
                             "prompts": 5, "sessions": sessions_per_cell,
                             "execution_commits": sorted({r.get("runtime", {}).get("execution_commit", "unknown") for r in rows}),
                             "dirty_worktree_sessions": sorted({r.get("session_id") for r in rows
                                                                 if r.get("runtime", {}).get("working_tree_dirty")}),
                             "capacity": rows[0]["kv_capacity"],
                             "input_tokens": rows[0]["prompt_tokens"],
                             "max_new_tokens": rows[0]["max_new_tokens"]}

    summary = {"protocol": {"contexts": list(CONTEXTS), "prompts_per_context": 5,
                             "sessions_per_cell": sessions_per_cell, "output_budget": output_budget,
                             "warmup_tokens": output_budget,
                             "paired_source_documents": True,
                             "inference": "greedy, static KV, SDPA, Qwen2.5-3B-Instruct pinned revision; capacity=context+budget+4",
                             "statistics": f"descriptive only; n=5 prompts across {sessions_per_cell} sessions per cell"},
               "cell_status": {f"ctx{c}_{b}": v for (c,b),v in statuses.items()},
               "results": {}, "cross_backend_vanilla_exact": {}}
    provenance = ("Execution provenance: session 1 used commit `7b8a398` with a clean working tree. Session 2 used commit `ad0e1d2`; runtime metadata marks its working tree dirty, so this two-session result is exploratory and does not establish clean-tree reproducibility."
                  if output_budget == 128 else
                  "Execution provenance: the recorded session used commit `7b8a398` with a clean working tree.")
    title = "Paired context-length backend smoke (32-token output)" if output_budget == 32 else "Paired context-length backend evaluation (128-token output)"
    lines = [f"# {title}", "",
             f"The same five LongBench source documents (one each from narrativeqa, qasper, gov_report, multi_news, qmsum) are represented at 128, 384, 1024, and 2048 input tokens. Each session uses a {output_budget}-token full-path warmup and output budget, static capacity `context + {output_budget + 4}`, greedy SDPA. This is exploratory, not a performance conclusion: {sessions_per_cell} sessions and five prompts per cell.", "",
             provenance,
             "",
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
    (DIR / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (DIR / f"{stem}_report.md").write_text("\n".join(lines))
    print(f"wrote {DIR.name}/{stem}_summary.json and {stem}_report.md")


if __name__ == "__main__":
    main()
