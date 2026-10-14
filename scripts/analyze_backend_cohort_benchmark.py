#!/usr/bin/env python3
"""Analyze the three-session paired backend benchmark on the diagnostic cohort."""
from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "experiments/backend_validation/paired_benchmark"
COHORT_FILE = DIR / "diagnostic_cohort.txt"
BACKENDS = ("hf_bnb_nf4", "hqq_gemlite_w4a16")
SESSIONS = ("session1", "session2", "session3")
METHODS = ("vanilla", "pld_k4", "cka90_k2")


def mean_ci(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "ci95": None}
    rng = random.Random(20261014)
    boot = []
    for _ in range(5000):
        boot.append(statistics.mean(rng.choices(values, k=len(values))))
    boot.sort()
    return {
        "n": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values),
        "ci95": [boot[int(.025 * len(boot))], boot[min(int(.975 * len(boot)), len(boot)-1)]],
    }


def load_records() -> tuple[dict, list[str]]:
    cohort = [line.strip() for line in COHORT_FILE.read_text().splitlines() if line.strip()]
    data: dict[tuple[str, str, str], dict[str, dict]] = {}
    for backend in BACKENDS:
        for session in SESSIONS:
            path = DIR / f"cohort_{backend.split('_')[0]}_{session}.jsonl"
            if backend == "hqq_gemlite_w4a16":
                path = DIR / f"cohort_hqq_{session}.jsonl"
            elif backend == "hf_bnb_nf4":
                path = DIR / f"cohort_bnb_{session}.jsonl"
            if not path.exists():
                raise FileNotFoundError(path)
            records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            expected = len(cohort) * len(METHODS)
            if len(records) != expected:
                raise ValueError(f"{path.name}: expected {expected} records, got {len(records)}")
            for record in records:
                key = (backend, session, record["prompt_id"])
                data.setdefault(key, {})[record["method"]] = record
            prompt_set = {r["prompt_id"] for r in records}
            if prompt_set != set(cohort):
                raise ValueError(f"cohort mismatch in {path.name}")
    for key, methods in data.items():
        if set(methods) != set(METHODS):
            raise ValueError(f"incomplete method pairing: {key}, methods={list(methods)}")
    return data, cohort


def paired_cluster_ci(observations: dict[tuple[str, str], float], prompts: list[str]) -> list[float] | None:
    """Two-way bootstrap: resample the 3 independent sessions and prompts."""
    session_ids = list(SESSIONS)
    usable = [(s, p, value) for (s, p), value in observations.items() if value is not None]
    if not usable:
        return None
    rng = random.Random(78431)
    boot = []
    for _ in range(5000):
        sampled_sessions = rng.choices(session_ids, k=len(session_ids))
        sampled_prompts = rng.choices(prompts, k=len(prompts))
        vals = [observations[(s, p)] for s in sampled_sessions for p in sampled_prompts
                if (s, p) in observations]
        if vals:
            boot.append(statistics.mean(vals))
    boot.sort()
    return [boot[int(.025 * len(boot))], boot[min(int(.975 * len(boot)), len(boot)-1)]] if boot else None


def main() -> None:
    data, cohort = load_records()
    summary = {
        "protocol": {
            "cohort_size": len(cohort), "session_count": len(SESSIONS),
            "prompt_tokens": 384, "output_budget": 32, "capacity": 436,
            "warmup_tokens": 32, "diagnostic_only": True,
            "ci_method": "two-way bootstrap over independent sessions and prompts, 5000 resamples",
        },
        "backends": {}, "paired_backend_comparison": {},
        "errors_and_oom": [], "cross_backend_vanilla": {},
    }
    speedups: dict[tuple[str, str, str, str], float] = {}
    for backend in BACKENDS:
        summary["backends"][backend] = {}
        for method in METHODS:
            rows = [data[(backend, session, prompt)][method]
                    for session in SESSIONS for prompt in cohort]
            for row in rows:
                if row.get("status") != "ok":
                    summary["errors_and_oom"].append({
                        "backend": backend, "session_id": row.get("session_id"),
                        "prompt_id": row.get("prompt_id"), "method": method,
                        "status": row.get("status"), "error_type": row.get("error_type"),
                        "error": row.get("error"),
                    })
            valid = [row for row in rows if row.get("status") == "ok"]
            tps = [float(row["decode_tokens_per_second"]) for row in valid]
            e2e = [float(row["total_time_s"]) for row in valid]
            alloc = [float(row["peak_allocated_mb"]) for row in valid]
            def paired_metric(field: str) -> dict[tuple[str, str], float]:
                return {
                    (session, prompt): float(data[(backend, session, prompt)][method][field])
                    for session in SESSIONS for prompt in cohort
                    if data[(backend, session, prompt)][method].get("status") == "ok"
                }
            exact_flags = [row.get("within_backend_exact_vs_vanilla") for row in valid
                           if row.get("within_backend_exact_vs_vanilla") is not None]
            divs = [row.get("first_divergence_vs_vanilla") for row in valid]
            summary["backends"][backend][method] = {
                "valid_runs": len(valid),
                "decode_tps": {**mean_ci(tps), "ci95_cluster": paired_cluster_ci(paired_metric("decode_tokens_per_second"), cohort)},
                "end_to_end_seconds": {**mean_ci(e2e), "ci95_cluster": paired_cluster_ci(paired_metric("total_time_s"), cohort)},
                "peak_allocated_mb": {**mean_ci(alloc), "ci95_cluster": paired_cluster_ci(paired_metric("peak_allocated_mb"), cohort)},
                "within_backend_exact_count": sum(bool(x) for x in exact_flags),
                "within_backend_exact_n": len(exact_flags),
                "exact_rate": sum(bool(x) for x in exact_flags) / len(exact_flags) if exact_flags else None,
                "first_divergence_positions": {str(pos): divs.count(pos) for pos in sorted({x for x in divs if x is not None})},
                "slowdown_rate_vs_vanilla": None,
            }
        for method in ("pld_k4", "cka90_k2"):
            paired = {}
            for session in SESSIONS:
                for prompt in cohort:
                    method_row = data[(backend, session, prompt)][method]
                    vanilla = data[(backend, session, prompt)]["vanilla"]
                    if method_row.get("status") == "ok" and vanilla.get("status") == "ok":
                        ratio = method_row["decode_tokens_per_second"] / vanilla["decode_tokens_per_second"]
                        paired[(session, prompt)] = ratio
                        speedups[(backend, method, session, prompt)] = ratio
            values = list(paired.values())
            item = summary["backends"][backend][method]
            item["paired_internal_speedup"] = {
                **mean_ci(values), "ci95_cluster": paired_cluster_ci(paired, cohort),
                "slowdown_frequency": sum(x < 1 for x in values) / len(values) if values else None,
            }
            item["slowdown_rate_vs_vanilla"] = item["paired_internal_speedup"]["slowdown_frequency"]
    for session in SESSIONS:
        for prompt in cohort:
            bnb = data[(BACKENDS[0], session, prompt)]["vanilla"]["token_ids"]
            hqq = data[(BACKENDS[1], session, prompt)]["vanilla"]["token_ids"]
            first = next((i for i, (a, b) in enumerate(zip(bnb, hqq)) if a != b), None)
            if first is None and len(bnb) != len(hqq):
                first = min(len(bnb), len(hqq))
            summary["cross_backend_vanilla"].setdefault("first_difference_positions", []).append(first)
            summary["cross_backend_vanilla"].setdefault("exact_count", 0)
            summary["cross_backend_vanilla"]["exact_count"] += bnb == hqq
    for method in ("pld_k4", "cka90_k2"):
        deltas = {}
        for session in SESSIONS:
            for prompt in cohort:
                a = speedups.get((BACKENDS[0], method, session, prompt))
                b = speedups.get((BACKENDS[1], method, session, prompt))
                if a is not None and b is not None:
                    deltas[(session, prompt)] = b - a
        summary["paired_backend_comparison"][method] = {
            "hqq_minus_bnb_internal_speedup": mean_ci(list(deltas.values())),
            "ci95_cluster": paired_cluster_ci(deltas, cohort),
        }
    out = DIR / "cohort_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Three-session paired backend benchmark",
        "",
        "Cohort: 16 diagnostic prompts at 384 tokens, 32-token generation budget, 32-token full-path warmup, capacity 436, Qwen2.5-3B pinned revision, greedy SDPA. Each backend was loaded in three separate processes, with distinct order seeds. All timings are uninstrumented generation timings; session/prompt pairs are the unit for internal speedups. This is the diagnostic mismatch cohort, not held-out evaluation.",
        "",
        "Decode TPS, E2E, and peak allocated VRAM are means across valid session/prompt runs; brackets show two-way bootstrap 95% intervals resampling independent sessions and prompts. Internal speedup is paired within the same backend, session, and prompt; its interval uses the same two-way resampling.",
        "",
        "| Backend | Method | Valid / 48 | Decode TPS (95% CI) | Internal speedup (cluster CI) | Slower than vanilla | Exact vs own vanilla | Peak allocated MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in BACKENDS:
        for method in METHODS:
            row = summary["backends"][backend][method]
            tps = row["decode_tps"]
            tps_ci = "n/a" if tps["ci95_cluster"] is None else f"[{tps['ci95_cluster'][0]:.2f}, {tps['ci95_cluster'][1]:.2f}]"
            speed = row.get("paired_internal_speedup")
            speed_txt = "—" if not speed else f"{speed['mean']:.3f} [{speed['ci95_cluster'][0]:.3f}, {speed['ci95_cluster'][1]:.3f}]"
            slowdown = "—" if not speed else f"{speed['slowdown_frequency']:.1%}"
            exact = "n/a" if row["exact_rate"] is None else f"{row['within_backend_exact_count']}/{row['within_backend_exact_n']} ({row['exact_rate']:.1%})"
            lines.append(f"| {backend} | {method} | {row['valid_runs']}/48 | {tps['mean']:.2f} {tps_ci} | {speed_txt} | {slowdown} | {exact} | {row['peak_allocated_mb']['mean']:.1f} |")
    lines += [
        "",
        f"Cross-backend vanilla exactness: {summary['cross_backend_vanilla']['exact_count']}/48 paired outputs. First-token difference positions are retained in the JSON; they measure quantization/backend fidelity, not speculative correctness.",
        f"Recorded failed/OOM method runs: {len(summary['errors_and_oom'])}.",
        "",
        "The comparison estimates backend stack performance on this one diagnostic prompt cohort. Quantization formats differ, so cross-backend raw token equality is not the correctness criterion. Within-backend exactness is reported separately. No claim of lossless acceleration follows unless every intended evaluation condition matches its own backend's vanilla output.",
        "",
        "Raw per-session JSONL, prompt cohort, and analysis code are kept beside this report. Interpret confidence intervals with care: there are only three independent sessions and prompts are a fixed diagnostic cohort.",
        "",
    ]
    (DIR / "cohort_report.md").write_text("\n".join(lines))
    print(f"wrote {out.relative_to(ROOT)} and cohort_report.md")


if __name__ == "__main__":
    main()
