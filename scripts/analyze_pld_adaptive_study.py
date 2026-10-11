#!/usr/bin/env python3
"""Paired source-extension and loss-regime analysis for PLD study JSON."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def paired(rows_a: list[dict], rows_b: list[dict], seed: int = 2026) -> dict:
    b = {r["id"]: r for r in rows_b}
    common = [r for r in rows_a if r["id"] in b]
    ratios = np.asarray([r["speedup_vs_vanilla"] / b[r["id"]]["speedup_vs_vanilla"] for r in common])
    rng = np.random.default_rng(seed)
    boot = rng.choice(ratios, size=(10000, len(ratios)), replace=True).mean(axis=1)
    return {
        "n": len(common), "mean_speed_ratio_a_over_b": float(ratios.mean()),
        "speed_ratio_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "fraction_prompts_a_faster": float(np.mean(ratios > 1.0)),
        "accepted_tokens_delta_a_minus_b": float(np.mean([
            r["accepted_tokens"] - b[r["id"]]["accepted_tokens"] for r in common])),
        "exact_match_delta_pp_a_minus_b": float(100 * np.mean([
            int(r["exact"]) - int(b[r["id"]]["exact"]) for r in common])),
    }


def cycle_summary(rows: list[dict]) -> dict:
    cycles = [c for r in rows for c in r["cycles"]]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for c in cycles:
        by_source[c.get("match_source", "unknown")].append(c)
    out = {}
    for source, group in by_source.items():
        out[source] = {
            "cycles": len(group), "lookup_hit_fraction": float(np.mean([c["k"] > 0 for c in group])),
            "accepted_tokens_per_cycle": float(np.mean([c["accepted"] for c in group])),
            "lookup_ms_median": float(np.median([c["lookup_ms"] for c in group])),
            "verify_ms_median": float(np.median([c["verify_ms"] for c in group])),
            "crop_ms_median": float(np.median([c["crop_ms"] for c in group])),
            "decision_ms_median": float(np.median([c.get("decision_ms", 0) for c in group])),
            "explicit_sync_wait_ms_median": float(np.median([c["sync_wait_ms"] for c in group])),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="experiments/diagnostics/pld_adaptive_n10_source_matched.json")
    ap.add_argument("--output", default="experiments/diagnostics/pld_adaptive_source_analysis.json")
    args = ap.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    report = {"input": args.input, "benchmarks": {}}
    for bench, result in data["results"].items():
        rows = result["rows"]
        k = result["tuned_fixed_k_by_source"]
        prompt_fixed = rows[f"prompt_fixed_k{k['prompt']}"]
        prompt_generated_fixed = rows[f"prompt_generated_fixed_k{k['prompt+generated']}"]
        prompt_adaptive = rows["prompt_adaptive"]
        prompt_generated_adaptive = rows["prompt_generated_adaptive"]
        history_fixed = rows[f"history_fixed_k{k['history']}"]
        slow = [r for r in history_fixed if r["speedup_vs_vanilla"] < 1.0]
        fast = [r for r in history_fixed if r["speedup_vs_vanilla"] >= 1.0]

        def loss_group(group: list[dict]) -> dict:
            cycles = [c for r in group for c in r["cycles"]]
            return {
                "prompts": len(group),
                "mean_speedup": float(np.mean([r["speedup_vs_vanilla"] for r in group])) if group else None,
                "hit_fraction": float(np.mean([c["k"] > 0 for c in cycles])) if cycles else None,
                "accepted_tokens_per_output": float(np.mean([
                    r["accepted_tokens"] / max(1, r["output_tokens"]) for r in group])) if group else None,
                "lookup_ms_median": float(np.median([c["lookup_ms"] for c in cycles])) if cycles else None,
                "verify_ms_median": float(np.median([c["verify_ms"] for c in cycles])) if cycles else None,
                "crop_ms_median": float(np.median([c["crop_ms"] for c in cycles])) if cycles else None,
                "decision_ms_median": float(np.median([c.get("decision_ms", 0) for c in cycles])) if cycles else None,
                "explicit_sync_wait_ms_median": float(np.median([c["sync_wait_ms"] for c in cycles])) if cycles else None,
            }

        report["benchmarks"][bench] = {
            "tuned_k_by_source": k,
            "prompt_generated_vs_prompt_fixed": paired(prompt_generated_fixed, prompt_fixed),
            "prompt_generated_vs_prompt_adaptive": paired(prompt_generated_adaptive, prompt_adaptive, 2027),
            "prompt_fixed_cycle_costs": cycle_summary(prompt_fixed),
            "prompt_generated_fixed_cycle_costs": cycle_summary(prompt_generated_fixed),
            "history_fixed_losses_vs_wins": {"slower": loss_group(slow), "not_slower": loss_group(fast)},
        }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
