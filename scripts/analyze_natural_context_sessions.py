#!/usr/bin/env python3
"""Compare repeated natural-context runs on identical prompts and actions."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            result[(row["prompt_id"], row["method_key"])] = row
    return result


def ci_mean(values: list[float], rng: random.Random) -> list[float]:
    estimates = []
    for _ in range(10000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(sample) / len(sample))
    estimates.sort()
    return [estimates[249], estimates[9749]]


def summarize(a: dict, b: dict, seed: int) -> dict:
    prompts_a = {prompt for prompt, method in a}
    prompts_b = {prompt for prompt, method in b}
    if prompts_a != prompts_b:
        raise ValueError("Runs do not contain identical prompt IDs")
    for prompt in prompts_a:
        if a[(prompt, "vanilla")]["measured"].get("token_ids") != b[(prompt, "vanilla")]["measured"].get("token_ids"):
            raise ValueError(f"Vanilla token IDs differ between runs for {prompt}")
    methods = sorted({method for _, method in a if method != "vanilla"})
    rng = random.Random(seed)
    out = {"n_paired_prompts": len(prompts_a), "actions": {}, "by_context_bucket": {}}
    for method in methods:
        session_speedups = [[], []]
        exact = [[], []]
        diffs = []
        buckets: dict[str, list[float]] = defaultdict(list)
        for prompt in sorted(prompts_a):
            ratios = []
            for rows, idx in ((a, 0), (b, 1)):
                vanilla_tps = rows[(prompt, "vanilla")]["measured"]["decode_tps"]
                action_tps = rows[(prompt, method)]["measured"]["decode_tps"]
                ratio = action_tps / max(vanilla_tps, 1e-12)
                session_speedups[idx].append(ratio)
                exact[idx].append(
                    rows[(prompt, method)]["measured"].get("token_ids", [])
                    == rows[(prompt, "vanilla")]["measured"].get("token_ids", [])
                )
                ratios.append(ratio)
            diffs.append(ratios[1] - ratios[0])
            bucket = b[(prompt, method)]["context_bucket"]
            buckets[bucket].append(ratios[1] - ratios[0])
        out["actions"][method] = {
            "session1_mean_speedup": sum(session_speedups[0]) / len(prompts_a),
            "session1_exact_match_rate": sum(bool(x) for x in exact[0]) / len(prompts_a),
            "session2_mean_speedup": sum(session_speedups[1]) / len(prompts_a),
            "session2_exact_match_rate": sum(bool(x) for x in exact[1]) / len(prompts_a),
            "paired_mean_speedup_change_session2_minus_session1": sum(diffs) / len(diffs),
            "paired_prompt_bootstrap_95ci_change": ci_mean(diffs, rng),
            "prompt_level_speedup_correlation": _correlation(*session_speedups),
            "by_context_bucket_change": {
                bucket: {"n": len(values), "mean_change": sum(values) / len(values),
                         "prompt_bootstrap_95ci": ci_mean(values, rng)}
                for bucket, values in sorted(buckets.items())
            },
        }
    return out


def _correlation(x: list[float], y: list[float]) -> float | None:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    denom = (sum(v * v for v in dx) * sum(v * v for v in dy)) ** .5
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session1", type=Path, required=True)
    parser.add_argument("--session2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=90210)
    args = parser.parse_args()
    report = summarize(read_rows(args.session1), read_rows(args.session2), args.seed)
    report["runs"] = {"session1": str(args.session1), "session2": str(args.session2)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
