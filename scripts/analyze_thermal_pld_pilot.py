#!/usr/bin/env python3
"""Paired cool-start versus heat-soaked analysis for the thermal PLD pilot."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap(values: list[float], seed: int = 9137, draws: int = 20000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(len(values))] for _ in values) / len(values)
                   for _ in range(draws))
    return [means[int(.025 * draws)], means[int(.975 * draws)]]


def summarize_session(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path / "raw_measurements.jsonl")
    telemetry = read_jsonl(path / "telemetry_samples.jsonl")
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(row["prompt_id"], {})[row["method_key"]] = row
    method_result = {}
    for method in ("pld_forced_miss_k4", "pld_k4", "pld_k8"):
        vals, exact = [], []
        for pair in pairs.values():
            a, v = pair[method], pair["vanilla"]
            vals.append(a["measured"]["decode_tps"] / max(v["measured"]["decode_tps"], 1e-9))
            exact.append(a["measured"]["token_ids"] == v["measured"]["token_ids"])
        method_result[method] = {"mean_prompt_speedup": sum(vals) / len(vals),
                                 "prompt_bootstrap_95ci": bootstrap(vals),
                                 "median_prompt_speedup": sorted(vals)[len(vals) // 2],
                                 "exact_match_count": sum(exact), "n_prompts": len(vals)}
    temps = [r.get("temperature_c") for r in telemetry if r.get("temperature_c") is not None]
    clocks = [r.get("sm_clock_mhz") for r in telemetry if r.get("sm_clock_mhz") is not None]
    return {"n_prompts": len(pairs), "methods": method_result,
            "telemetry": {"n_samples": len(telemetry), "temperature_min_c": min(temps) if temps else None,
                          "temperature_mean_c": sum(temps) / len(temps) if temps else None,
                          "temperature_max_c": max(temps) if temps else None,
                          "sm_clock_mean_mhz": sum(clocks) / len(clocks) if clocks else None,
                          "sm_clock_min_mhz": min(clocks) if clocks else None,
                          "sm_clock_max_mhz": max(clocks) if clocks else None}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cool", type=Path, required=True)
    parser.add_argument("--hot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cool, hot = summarize_session(args.cool), summarize_session(args.hot)
    cool_rows, hot_rows = read_jsonl(args.cool / "raw_measurements.jsonl"), read_jsonl(args.hot / "raw_measurements.jsonl")
    indexed = []
    for rows in (cool_rows, hot_rows):
        indexed.append({(r["prompt_id"], r["method_key"]): r for r in rows})
    paired = {}
    for method in ("pld_forced_miss_k4", "pld_k4", "pld_k8"):
        cool_speedups, hot_speedups, changes = [], [], []
        for pid in sorted({r["prompt_id"] for r in cool_rows} & {r["prompt_id"] for r in hot_rows}):
            c = indexed[0][(pid, method)]["measured"]["decode_tps"] / max(indexed[0][(pid, "vanilla")]["measured"]["decode_tps"], 1e-9)
            h = indexed[1][(pid, method)]["measured"]["decode_tps"] / max(indexed[1][(pid, "vanilla")]["measured"]["decode_tps"], 1e-9)
            cool_speedups.append(c)
            hot_speedups.append(h)
            changes.append(h - c)
        paired[method] = {"n_same_prompts": len(changes), "cool_mean_speedup": sum(cool_speedups) / len(changes),
                          "hot_mean_speedup": sum(hot_speedups) / len(changes),
                          "hot_minus_cool_speedup": sum(changes) / len(changes),
                          "paired_prompt_bootstrap_95ci_difference": bootstrap(changes, seed=371),
                          "fraction_prompts_hot_speedup_below_cool": sum(x < 0 for x in changes) / len(changes)}
    result = {"design": "one cool-start session versus one heat-soaked session; same 12 prompt IDs",
              "sessions": {"cool": cool, "hot": hot}, "paired_thermal_comparison": paired,
              "limitations": ["Only one session per thermal condition; CI resamples prompts, not independent sessions.",
                              "Temperature condition is confounded with session/order and time-of-day.",
                              "The hot session targeted 78 C; few or no measurements may exceed 80 C.",
                              "PLD exactness remains separate from speedup: token sequence mismatch is not a valid lossless claim."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
