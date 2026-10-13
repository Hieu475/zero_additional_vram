#!/usr/bin/env python3
"""Summarize paired PLD short-context repeats using sessions as the CI unit."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

T_CRIT_95_DF2 = 4.3026527299


def load_run(path: Path) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    rows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for line in (path / "raw_measurements.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["prompt_id"]][row["method_key"]] = row
    config = json.loads((path / "config.json").read_text())
    return dict(rows), config


def summarize(runs: list[Path]) -> dict[str, Any]:
    if len(runs) != 3:
        raise ValueError("Report session-level confidence intervals only with exactly three sessions")
    loaded = [load_run(path) for path in runs]
    common = set(loaded[0][0])
    for rows, _ in loaded[1:]:
        if set(rows) != common:
            raise ValueError("Session prompt IDs differ")
    first_ids = {prompt: rows["vanilla"]["measured"]["token_ids"]
                 for prompt, rows in loaded[0][0].items()}
    for rows, _ in loaded[1:]:
        for prompt, methods in rows.items():
            if methods["vanilla"]["measured"]["token_ids"] != first_ids[prompt]:
                raise ValueError(f"Vanilla output changed between sessions for {prompt}")

    methods = sorted(set(loaded[0][0][next(iter(common))]) - {"vanilla"})
    report: dict[str, Any] = {"n_sessions": 3, "n_prompts": len(common), "sessions": {},
                              "actions": {}, "inference": (
                                  "Session-level 95% t intervals use three session means (df=2); "
                                  "prompt repeats are not counted as independent sessions.")}
    for index, ((rows, config), path) in enumerate(zip(loaded, runs), start=1):
        temps, clocks = [], []
        for prompt_rows in rows.values():
            for row in prompt_rows.values():
                for key, dest in (("temperature_c", temps), ("sm_clock_mhz", clocks)):
                    for field in ("telemetry_before", "telemetry_after"):
                        value = row.get(field, {}).get(key)
                        if value is not None:
                            dest.append(float(value))
        report["sessions"][path.name] = {
            "order_seed": config["method_order_seed"],
            "prepared_sha256": config["source_manifest"].get("prepared_sha256"),
            "prompt_count": len(rows),
            "temperature_c_range": [min(temps), max(temps)] if temps else None,
            "sm_clock_mhz_range": [min(clocks), max(clocks)] if clocks else None,
        }

    for method in methods:
        per_session: list[dict[str, Any]] = []
        mismatch_sets: list[set[str]] = []
        divergence_maps: list[dict[str, int]] = []
        for rows, _ in loaded:
            dec, e2e, exact, hit_rates, accepted_per_cycle = [], [], [], [], []
            mismatches: set[str] = set()
            positions: dict[str, int] = {}
            for prompt, pair in rows.items():
                if method not in pair:
                    raise ValueError(f"Missing {method} for {prompt}")
                vanilla = pair["vanilla"]["measured"]
                measured = pair[method]["measured"]
                dec.append(measured["decode_tps"] / max(vanilla["decode_tps"], 1e-9))
                e2e.append(vanilla["total_ms"] / max(measured["total_ms"], 1e-9))
                same = vanilla["token_ids"] == measured["token_ids"]
                exact.append(same)
                if not same:
                    mismatches.add(prompt)
                    positions[prompt] = next(
                        (i for i, (a, b) in enumerate(zip(vanilla["token_ids"], measured["token_ids"]))
                         if a != b), min(len(vanilla["token_ids"]), len(measured["token_ids"])))
                if "lookup_hit_rate" in measured:
                    hit_rates.append(float(measured["lookup_hit_rate"]))
                    accepted_per_cycle.append(float(measured["accepted_tokens_per_cycle"]))
            per_session.append({
                "mean_decode_speedup": sum(dec) / len(dec),
                "mean_end_to_end_speedup": sum(e2e) / len(e2e),
                "exact_match_rate": sum(exact) / len(exact),
                "profitable_prompts": sum(x > 1.0 for x in dec),
                "mean_lookup_hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else None,
                "mean_accepted_tokens_per_cycle": sum(accepted_per_cycle) / len(accepted_per_cycle)
                                                    if accepted_per_cycle else None,
            })
            mismatch_sets.append(mismatches)
            divergence_maps.append(positions)
        def session_interval(key: str) -> list[float]:
            vals = [item[key] for item in per_session]
            mean = sum(vals) / 3
            sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / 2)
            half = T_CRIT_95_DF2 * sd / math.sqrt(3)
            return [mean - half, mean + half]
        all_mismatch = set.union(*mismatch_sets)
        all_match = set.intersection(*mismatch_sets)
        repeatable_positions = sum(
            len({divergence_maps[i][prompt] for i in range(3)}) == 1
            for prompt in all_match)
        report["actions"][method] = {
            "per_session": per_session,
            "mean_of_session_decode_speedups": sum(x["mean_decode_speedup"] for x in per_session) / 3,
            "session_level_t95_ci_decode_speedup": session_interval("mean_decode_speedup"),
            "mean_of_session_end_to_end_speedups": sum(x["mean_end_to_end_speedup"] for x in per_session) / 3,
            "session_level_t95_ci_end_to_end_speedup": session_interval("mean_end_to_end_speedup"),
            "session_mean_decode_speedup_range": [
                min(x["mean_decode_speedup"] for x in per_session),
                max(x["mean_decode_speedup"] for x in per_session)],
            "mismatch_prompts_in_all_sessions": len(all_match),
            "mismatch_prompts_in_any_session": len(all_mismatch),
            "mismatches_with_same_first_position_all_sessions": repeatable_positions,
        }
    fixed_k_methods = [m for m in methods if m.startswith("PLD_prompt_plus_generated_K")]
    prompt_means: dict[str, dict[str, float]] = {}
    for prompt in sorted(common):
        prompt_means[prompt] = {}
        for method in fixed_k_methods:
            values = []
            for rows, _ in loaded:
                pair = rows[prompt]
                values.append(pair[method]["measured"]["decode_tps"] /
                              max(pair["vanilla"]["measured"]["decode_tps"], 1e-9))
            prompt_means[prompt][method] = sum(values) / len(values)
    fixed_action_means = {
        method: sum(values[method] for values in prompt_means.values()) / len(prompt_means)
        for method in fixed_k_methods
    }
    best_fixed = max(fixed_action_means, key=fixed_action_means.get)
    oracle_values = {prompt: max(values.values()) for prompt, values in prompt_means.items()}
    oracle_actions = Counter(max(values, key=values.get) for values in prompt_means.values())
    oracle_mean = sum(oracle_values.values()) / len(oracle_values)
    report["fixed_action_and_prompt_oracle"] = {
        "mean_decode_speedup_by_fixed_k": fixed_action_means,
        "best_fixed_action_in_this_dataset": best_fixed,
        "best_fixed_action_mean_decode_speedup": fixed_action_means[best_fixed],
        "full_prompt_oracle_mean_decode_speedup": oracle_mean,
        "oracle_gain_over_best_fixed": oracle_mean - fixed_action_means[best_fixed],
        "oracle_action_count_by_prompt": dict(oracle_actions),
        "mean_oracle_regret_vs_k4": oracle_mean - fixed_action_means.get(
            "PLD_prompt_plus_generated_K4", 0.0),
        "definition": (
            "Diagnostic only: choose one measured K per full prompt after observing its complete "
            "trajectory. This is not an implementable runtime policy or cycle-level oracle; "
            "different K values can produce different token trajectories."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, action="append", required=True,
                        help="Provide exactly three session directories in measurement order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
