#!/usr/bin/env python3
"""Describe acceptance and latency across matched K1/R2 and K2/R3 runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


def geomean(values: list[float]) -> float | None:
    if not values or any(v <= 0 for v in values):
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/"
            "k1r2_vs_k2r3_humaneval112_114_115_summary.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/"
            "acceptance_latency_analysis.json"
        ),
    )
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    cells: dict[str, dict[str, dict[str, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for prompt, budgets in summary["prompts"].items():
        for budget, profiles in budgets.items():
            for profile, profile_data in profiles.items():
                for artifact in profile_data["artifacts"]:
                    path = Path(artifact)
                    if not path.is_absolute():
                        path = Path.cwd() / path
                    run = json.loads(path.read_text())
                    assert run["prompt_id"] == prompt
                    assert str(run["parameters"]["max_new_tokens"]) == str(budget)
                    for timing_round in run["timing"]["paired_rounds"]:
                        methods = timing_round["methods"]
                        block = methods["optimized_b1_block"]
                        singleton = methods["optimized_b1_singleton"]
                        bnb = methods["original_bnb_greedy"]
                        drafted = block.get("draft_tokens")
                        accepted = block.get("accepted_draft_tokens")
                        count = int(budget)
                        cells[prompt][budget][profile].append(
                            {
                                "source": str(path),
                                "session": run["session"],
                                "round": timing_round["round"],
                                "accepted_draft_tokens": accepted,
                                "draft_tokens": drafted,
                                "acceptance_rate": accepted / drafted
                                if drafted
                                else None,
                                "block_total_s": block["total_s"],
                                "block_prefill_s": block.get("prefill_s"),
                                "block_verify_s": block.get("verify_s"),
                                "block_ms_per_generated_token": 1000
                                * block["total_s"]
                                / count,
                                "singleton_total_s": singleton["total_s"],
                                "singleton_verify_s": singleton.get("verify_s"),
                                "singleton_ms_per_generated_token": 1000
                                * singleton["total_s"]
                                / count,
                                "bnb_total_s": bnb["total_s"],
                                "bnb_decode_s": bnb.get("decode_s"),
                                "bnb_ms_per_generated_token": 1000
                                * bnb["total_s"]
                                / count,
                                "speedup_vs_singleton": singleton["total_s"]
                                / block["total_s"],
                                "speedup_vs_bnb": bnb["total_s"]
                                / block["total_s"],
                                "bnb_token_match": block["token_ids"]
                                == bnb["token_ids"],
                                "own_singleton_exact": run["token_match"]
                                and run["all_shared_states_equal"]
                                and run["same_prefix_domain"],
                            }
                        )

    by_cell = {}
    matched_deltas = []
    for prompt, budgets in cells.items():
        by_cell[prompt] = {}
        for budget, profiles in budgets.items():
            by_cell[prompt][budget] = {}
            for profile, rows in profiles.items():
                acceptance = [r["acceptance_rate"] for r in rows if r["acceptance_rate"] is not None]
                speed_b1 = [r["speedup_vs_singleton"] for r in rows]
                speed_bnb = [r["speedup_vs_bnb"] for r in rows]
                record = {
                    "timing_rounds": len(rows),
                    "acceptance_rate": sum(
                        r["accepted_draft_tokens"] for r in rows
                    )
                    / sum(r["draft_tokens"] for r in rows)
                    if acceptance
                    else None,
                    "acceptance_rate_range_by_round": [min(acceptance), max(acceptance)]
                    if acceptance
                    else None,
                    "geomean_speedup_vs_singleton": geomean(speed_b1),
                    "geomean_speedup_vs_bnb": geomean(speed_bnb),
                    "median_block_ms_per_token": median(
                        r["block_ms_per_generated_token"] for r in rows
                    ),
                    "median_block_verify_ms_per_token": median(
                        1000 * r["block_verify_s"] / int(budget)
                        for r in rows
                        if r["block_verify_s"] is not None
                    ),
                    "median_singleton_ms_per_token": median(
                        r["singleton_ms_per_generated_token"] for r in rows
                    ),
                    "median_bnb_ms_per_token": median(
                        r["bnb_ms_per_generated_token"] for r in rows
                    ),
                    "bnb_token_match_rounds": sum(r["bnb_token_match"] for r in rows),
                    "own_singleton_exact_all_rounds": all(
                        r["own_singleton_exact"] for r in rows
                    ),
                    "rounds": rows,
                }
                by_cell[prompt][budget][profile] = record
            if "k1r2" in profiles and "k2r3" in profiles:
                left = by_cell[prompt][budget]["k1r2"]
                right = by_cell[prompt][budget]["k2r3"]
                matched_deltas.append(
                    {
                        "prompt_id": prompt,
                        "max_new_tokens": int(budget),
                        "k2r3_minus_k1r2_acceptance_percentage_points": 100
                        * (right["acceptance_rate"] - left["acceptance_rate"]),
                        "k2r3_over_k1r2_speedup_vs_bnb": right[
                            "geomean_speedup_vs_bnb"
                        ]
                        / left["geomean_speedup_vs_bnb"],
                        "k2r3_over_k1r2_speedup_vs_singleton": right[
                            "geomean_speedup_vs_singleton"
                        ]
                        / left["geomean_speedup_vs_singleton"],
                    }
                )

    result = {
        "schema": "zassd-nf4-profile-acceptance-latency-analysis-v1",
        "input_summary": str(args.summary),
        "scope": "descriptive only; selected prompts, one RTX 4050, two sessions per profile/budget; ratios are not population estimates",
        "definitions": {
            "ms_per_generated_token": "method total wall time divided by the configured generation budget",
            "block_verify_s": "runner-reported verification component; total latency remains the end-to-end comparison",
            "acceptance_rate": "total accepted draft tokens divided by total drafted tokens across timing rounds",
        },
        "by_prompt_and_budget": by_cell,
        "matched_profile_deltas": matched_deltas,
        "limitations": summary.get("limitations", []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cell_count": sum(len(b) for b in by_cell.values()), "matched_profile_cells": len(matched_deltas)}, indent=2))
    for prompt, budgets in by_cell.items():
        for budget, profiles in budgets.items():
            for profile, row in profiles.items():
                print(
                    f"{prompt} {budget} {profile}: acceptance={row['acceptance_rate']:.1%}; "
                    f"ms/token block={row['median_block_ms_per_token']:.2f} "
                    f"(verify={row['median_block_verify_ms_per_token']:.2f}), "
                    f"singleton={row['median_singleton_ms_per_token']:.2f}, "
                    f"BNB={row['median_bnb_ms_per_token']:.2f}; "
                    f"speedup BNB={row['geomean_speedup_vs_bnb']:.3f}x"
                )


if __name__ == "__main__":
    main()
