#!/usr/bin/env python3
"""Summarize the repeated held-out, 128-token paired backend cohort."""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/backend_validation/paired_benchmark"
METHODS = ("vanilla", "pld_k4", "cka90_k2")
BACKENDS = ("hf_bnb_nf4", "hqq_gemlite_w4a16")
SESSIONS = ("session1", "session2", "session3")


def mean_and_ci(values: list[float], observations: dict[tuple[str, str], float], prompts: list[str], seed: int) -> dict:
    rng = random.Random(seed)
    sessions = list(SESSIONS)
    means = []
    for _ in range(5000):
        ss = rng.choices(sessions, k=len(sessions))
        pp = rng.choices(prompts, k=len(prompts))
        sample = [observations[(s, p)] for s in ss for p in pp if (s, p) in observations]
        if sample:
            means.append(statistics.mean(sample))
    means.sort()
    return {"n": len(values), "mean": statistics.mean(values),
            "ci95_two_way_bootstrap": [means[int(.025 * len(means))], means[min(int(.975 * len(means)), len(means)-1)]]}


def main() -> None:
    data = {}
    cohort = [line.strip() for line in (DIR / "heldout_cohort.txt").read_text().splitlines() if line.strip()]
    for backend in BACKENDS:
        data[backend] = {}
        for session in SESSIONS:
            short = "bnb" if backend == "hf_bnb_nf4" else "hqq"
            path = DIR / f"heldout128_{short}_{session}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != len(cohort) * len(METHODS):
                raise ValueError(f"{path.name}: expected {len(cohort)*len(METHODS)} rows, got {len(rows)}")
            by_prompt = {}
            for row in rows:
                by_prompt.setdefault(row["prompt_id"], {})[row["method"]] = row
            if set(by_prompt) != set(cohort) or any(set(v) != set(METHODS) for v in by_prompt.values()):
                raise ValueError(f"incomplete prompt-method pairing in {path.name}")
            data[backend][session] = by_prompt
    prompt_ids = cohort
    for prompt in prompt_ids:
        hashes = {data[b][s][prompt][m]["prompt_token_ids_sha256"]
                  for b in BACKENDS for s in SESSIONS for m in METHODS}
        if len(hashes) != 1:
            raise ValueError(f"prompt token IDs differ across paired runs: {prompt}")

    summary = {"protocol": {"prompt_count": 20, "prompt_tokens": 384,
                             "output_budget": 128, "warmup_tokens": 128,
                             "kv_capacity": 520, "sessions_per_backend": len(SESSIONS),
                             "cohort": "held out from backend selection; may have appeared in earlier ZASSD studies",
                             "interval": "95% two-way bootstrap over independent sessions and fixed cohort prompts"},
               "backends": {}, "paired_backend_comparison": {}, "cross_backend_vanilla_exact": 0,
               "errors_or_oom": []}
    speedups = {}
    for backend in BACKENDS:
        summary["backends"][backend] = {}
        for method in METHODS:
            observations = {(s, p): data[backend][s][p][method]
                            for s in SESSIONS for p in prompt_ids}
            rows = list(observations.values())
            failed = [r for r in rows if r.get("status") != "ok" or not r["warmup_status"]["passed"]]
            summary["errors_or_oom"].extend({"backend": backend, "prompt_id": r["prompt_id"],
                                             "method": method, "status": r.get("status"),
                                             "error_type": r.get("error_type"), "error": r.get("error")}
                                            for r in failed)
            valid = [r for r in rows if r.get("status") == "ok" and r["warmup_status"]["passed"]]
            summary["backends"][backend][method] = {
                "valid_prompts": len(valid),
                "mean_decode_tps": mean_and_ci([r["decode_tokens_per_second"] for r in valid],
                    {(s,p):r["decode_tokens_per_second"] for (s,p),r in observations.items() if r in valid}, prompt_ids, 10),
                "mean_e2e_s": mean_and_ci([r["total_time_s"] for r in valid],
                    {(s,p):r["total_time_s"] for (s,p),r in observations.items() if r in valid}, prompt_ids, 11),
                "exact_vs_own_vanilla": sum(bool(r.get("within_backend_exact_vs_vanilla")) for r in valid),
                "exact_n": len(valid),
                "divergence_positions": {str(i): sum(r.get("first_divergence_vs_vanilla") == i for r in valid)
                                         for i in sorted({r["first_divergence_vs_vanilla"] for r in valid
                                                          if r.get("first_divergence_vs_vanilla") is not None})},
                "actual_output_lengths": {str(n): sum(len(r["token_ids"]) == n for r in valid)
                                          for n in sorted({len(r["token_ids"]) for r in valid})},
            }
        for method in ("pld_k4", "cka90_k2"):
            ratios = {}
            for s in SESSIONS:
                for p in prompt_ids:
                    method_row = data[backend][s][p][method]
                    vanilla = data[backend][s][p]["vanilla"]
                    if (method_row["status"] == "ok" and vanilla["status"] == "ok"
                            and method_row["warmup_status"]["passed"] and vanilla["warmup_status"]["passed"]):
                        ratios[(s,p)] = method_row["decode_tokens_per_second"] / vanilla["decode_tokens_per_second"]
                        speedups[(backend,method,s,p)] = ratios[(s,p)]
            vals = list(ratios.values())
            summary["backends"][backend][method]["paired_internal_speedup"] = mean_and_ci(vals, ratios, prompt_ids, 12)
            summary["backends"][backend][method]["slowdown_frequency"] = sum(x < 1 for x in vals) / len(vals)
    summary["cross_backend_vanilla_exact"] = sum(
        data["hf_bnb_nf4"][s][p]["vanilla"]["token_ids"] == data["hqq_gemlite_w4a16"][s][p]["vanilla"]["token_ids"]
        for s in SESSIONS for p in prompt_ids)
    for method in ("pld_k4", "cka90_k2"):
        deltas = {(s,p): speedups[(BACKENDS[1],method,s,p)] - speedups[(BACKENDS[0],method,s,p)]
                  for s in SESSIONS for p in prompt_ids
                  if (BACKENDS[0],method,s,p) in speedups and (BACKENDS[1],method,s,p) in speedups}
        summary["paired_backend_comparison"][method] = mean_and_ci(list(deltas.values()), deltas, prompt_ids, 13)

    (DIR / "heldout128_cohort_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Held-out 128-token backend cohort", "",
             "Three independent process sessions per backend; 20 prompts at 384 input tokens, 128 output tokens, full 128-token warmup, static KV capacity 520, greedy SDPA. The cohort was excluded from backend selection but may have appeared in earlier ZASSD studies. Input token hashes match across all paired methods and backends.", "",
             "Intervals are two-way bootstrap intervals resampling sessions and prompts. Prompt cohort is fixed and sessions=3, so treat intervals as exploratory.", "",
             "| Backend | Method | Valid / 60 | Decode TPS (two-way CI) | Paired speedup (two-way CI) | Slower fraction | Exact vs own vanilla |", "|---|---|---:|---:|---:|---:|---:|"]
    for backend in BACKENDS:
        for method in METHODS:
            r = summary["backends"][backend][method]
            tps = r["mean_decode_tps"]
            t = f"{tps['mean']:.2f} [{tps['ci95_two_way_bootstrap'][0]:.2f}, {tps['ci95_two_way_bootstrap'][1]:.2f}]"
            s = r.get("paired_internal_speedup")
            speed = "—" if s is None else f"{s['mean']:.3f} [{s['ci95_two_way_bootstrap'][0]:.3f}, {s['ci95_two_way_bootstrap'][1]:.3f}]"
            slow = "—" if s is None else f"{r['slowdown_frequency']:.1%}"
            lines.append(f"| {backend} | {method} | {r['valid_prompts']}/60 | {t} | {speed} | {slow} | {r['exact_vs_own_vanilla']}/{r['exact_n']} |")
    lines += ["", f"Cross-backend vanilla exactness: {summary['cross_backend_vanilla_exact']}/60 paired outputs. Failed/OOM or failed-warmup method runs: {len(summary['errors_or_oom'])}.", "",
              "| Method | HQQ minus BNB internal speedup (two-way CI) |", "|---|---:|"]
    for method, result in summary["paired_backend_comparison"].items():
        ci = result["ci95_two_way_bootstrap"]
        lines.append(f"| {method} | {result['mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines += ["", "The paired speedup is within-backend relative to that backend's vanilla. Raw cross-backend output equality measures quantization/backend fidelity, not speculative correctness. This cohort was held out from backend selection, but it may have appeared in earlier ZASSD studies.", ""]
    (DIR / "heldout128_cohort_report.md").write_text("\n".join(lines))
    print(f"wrote {DIR.name}/heldout128_cohort_summary.json and report")


if __name__ == "__main__":
    main()
