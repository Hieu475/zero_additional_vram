#!/usr/bin/env python3
"""Summarize replicated cool-start and heat-soaked PLD sessions."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


METHODS = ("pld_forced_miss_k4", "pld_k4", "pld_k8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap(values: list[float], seed: int, draws: int = 20000) -> list[float]:
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                   for _ in range(draws))
    return [means[int(.025 * draws)], means[int(.975 * draws)]]


def session(path: Path) -> dict[str, Any]:
    cfg = json.loads((path / "config.json").read_text())
    rows = read_jsonl(path / "raw_measurements.jsonl")
    by_prompt: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_prompt.setdefault(row["prompt_id"], {})[row["method_key"]] = row
    methods = {}
    for method in METHODS:
        ratios, exact = [], []
        for pair in by_prompt.values():
            ratio = pair[method]["measured"]["decode_tps"] / max(pair["vanilla"]["measured"]["decode_tps"], 1e-9)
            ratios.append(ratio)
            exact.append(pair[method]["measured"]["token_ids"] == pair["vanilla"]["measured"]["token_ids"])
        methods[method] = {"mean_speedup": sum(ratios) / len(ratios),
                           "prompt_bootstrap_95ci": bootstrap(ratios, 771 + len(method)),
                           "exact_matches": sum(exact), "n_prompts": len(exact)}
    telemetry = read_jsonl(path / "telemetry_samples.jsonl")
    temperatures = [x["temperature_c"] for x in telemetry if x.get("temperature_c") is not None]
    clocks = [x["sm_clock_mhz"] for x in telemetry if x.get("sm_clock_mhz") is not None]
    return {"path": str(path), "config": cfg, "methods": methods,
            "telemetry": {"samples": len(telemetry),
                          "temperature_min_c": min(temperatures) if temperatures else None,
                          "temperature_mean_c": sum(temperatures) / len(temperatures) if temperatures else None,
                          "temperature_max_c": max(temperatures) if temperatures else None,
                          "sm_clock_mean_mhz": sum(clocks) / len(clocks) if clocks else None}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cool", type=Path, nargs="+", required=True)
    ap.add_argument("--hot", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    groups = {"cool": [session(p) for p in args.cool], "hot": [session(p) for p in args.hot]}
    per_condition = {}
    for condition, records in groups.items():
        per_condition[condition] = {}
        for method in METHODS:
            vals = [s["methods"][method]["mean_speedup"] for s in records]
            mean = sum(vals) / len(vals)
            if len(vals) > 1:
                se = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) / len(vals))
                # Two-sided 95% Student-t critical value for df=2 (n=3 target protocol).
                critical = 4.303 if len(vals) == 3 else 2.0
                ci = [mean - critical * se, mean + critical * se]
            else:
                ci = None
            per_condition[condition][method] = {
                "session_mean_speedups": vals, "mean_of_session_means": mean,
                "session_level_t95ci": ci,
                "session_exact_match_counts": [s["methods"][method]["exact_matches"] for s in records],
                "session_prompt_counts": [s["methods"][method]["n_prompts"] for s in records]}
    result = {"design": "three independent sessions per condition; same held-out prompts; paired methods within prompt; randomized method order",
              "sessions": groups, "condition_summary": per_condition,
              "interpretation_note": "Session-level t intervals use independent session means (n=3, df=2) and are necessarily broad. Prompt bootstrap intervals are reported separately within each session. Thermal condition remains confounded with run occasion; telemetry is observed rather than controlled."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["condition_summary"], indent=2))


if __name__ == "__main__":
    main()
