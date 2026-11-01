#!/usr/bin/env python3
"""Explore whether no-proposal fallback frequency tracks its incremental speedup."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
SUMMARY_FILES = {
    "HumanEval": "activationvec_no_proposal_fallback_humaneval_256_summary_v2.json",
    "CNN/DailyMail": "activationvec_no_proposal_fallback_cnndm_256_summary_v2.json",
}
DRAWS = 20000
SEED = 74031


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in range(i, j):
            result[order[k]] = rank
        i = j
    return result


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom else float("nan")


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def bootstrap_ci(groups: list[list[dict]], *, statistic: str, seed: int) -> list[float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(DRAWS):
        rows = [rng.choice(group) for group in groups for _ in group]
        x = [row["no_proposal_cycles"] for row in rows]
        y = [row["fallback_speedup_vs_activationvec"] for row in rows]
        value = pearson(x, y) if statistic == "pearson" else spearman(x, y)
        if math.isfinite(value):
            samples.append(value)
    samples.sort()
    return [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]


def within_workload_corr(groups: list[list[dict]]) -> float:
    centered_x, centered_y = [], []
    for group in groups:
        xs = [row["no_proposal_cycles"] for row in group]
        ys = [row["fallback_speedup_vs_activationvec"] for row in group]
        mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
        centered_x.extend(value - mean_x for value in xs)
        centered_y.extend(value - mean_y for value in ys)
    return pearson(centered_x, centered_y)


def within_workload_bootstrap_ci(groups: list[list[dict]], *, seed: int) -> list[float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(DRAWS):
        resampled_groups = [
            [rng.choice(group) for _ in group] for group in groups
        ]
        value = within_workload_corr(resampled_groups)
        if math.isfinite(value):
            samples.append(value)
    samples.sort()
    return [samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]]


def analyze(rows: list[dict], groups: list[list[dict]], *, seed: int) -> dict:
    x = [row["no_proposal_cycles"] for row in rows]
    y = [row["fallback_speedup_vs_activationvec"] for row in rows]
    return {
        "prompt_clusters": len(rows),
        "pearson_r_cycles_vs_fallback_speedup": pearson(x, y),
        "pearson_prompt_stratified_bootstrap_95pct": bootstrap_ci(groups, statistic="pearson", seed=seed),
        "spearman_rho_cycles_vs_fallback_speedup": spearman(x, y),
        "spearman_prompt_stratified_bootstrap_95pct": bootstrap_ci(groups, statistic="spearman", seed=seed + 1),
    }


def main() -> None:
    rows_by_workload = {}
    summaries = {}
    process_hashes = {}
    for workload, name in SUMMARY_FILES.items():
        path = BASE / name
        summary = json.loads(path.read_text())
        summaries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = []
        for prompt in summary["prompts"]:
            sessions = prompt["sessions"]
            if len(sessions) != 2:
                raise RuntimeError(f"Expected two independent processes for {prompt['prompt_id']}")
            metric = prompt["prompt_metrics_geomean_across_processes"]
            cycles = {session["no_proposal_cycles"] for session in sessions}
            if len(cycles) != 1:
                raise RuntimeError(f"Fallback cycle count varied across replicates: {prompt['prompt_id']}")
            row = {
                "workload": workload,
                "prompt_id": prompt["prompt_id"],
                "no_proposal_cycles": cycles.pop(),
                "fallback_cycle_fraction": next(iter({session["no_proposal_cycles"] for session in sessions})) / 256,
                "fallback_speedup_vs_activationvec": metric["fallback_speedup_vs_activationvec"],
                "fallback_speedup_vs_original_bnb_latency": metric["fallback_speedup_vs_original_bnb_latency"],
            }
            rows.append(row)
            for session in sessions:
                artifact = session["artifact"]
                process_hashes[artifact] = hashlib.sha256((BASE / artifact).read_bytes()).hexdigest()
        rows_by_workload[workload] = rows

    all_rows = [row for group in rows_by_workload.values() for row in group]
    overall = analyze(all_rows, list(rows_by_workload.values()), seed=SEED)
    by_workload = {
        workload: analyze(rows, [rows], seed=SEED + i * 10)
        for i, (workload, rows) in enumerate(rows_by_workload.items())
    }
    result = {
        "schema": "zassd-activationvec-fallback-prompt-dependence-256-v1",
        "classification": "exploratory prompt-cluster association; not causal",
        "decode_length": 256,
        "overall": overall,
        "within_workload_adjusted_pearson_r": within_workload_corr(list(rows_by_workload.values())),
        "within_workload_adjusted_bootstrap_95pct": within_workload_bootstrap_ci(
            list(rows_by_workload.values()), seed=SEED + 2),
        "by_workload": by_workload,
        "prompt_rows": all_rows,
        "input_summary_sha256": summaries,
        "input_process_artifact_sha256": process_hashes,
        "heldout_used": False,
        "limits": [
            "Only twelve selected prompt clusters across two development workloads on one model and one GPU.",
            "The exploratory correlation does not establish that fallback frequency causes speedup.",
            "Prompt bootstrap is stratified by workload and does not estimate a broad task population.",
        ],
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = BASE / "activationvec_fallback_prompt_dependence_256_v1.json"
    output.write_text(json.dumps(result, indent=2) + chr(10))
    print(json.dumps({"overall": overall,
                      "within_workload_adjusted_pearson_r": result["within_workload_adjusted_pearson_r"],
                      "within_workload_adjusted_bootstrap_95pct": result["within_workload_adjusted_bootstrap_95pct"],
                      "by_workload": by_workload, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
