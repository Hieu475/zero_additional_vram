#!/usr/bin/env python3
"""Aggregate corrected fixed-three-row K2/R3 development runs."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel"
RUNS = [
    "actual_fixed_three_rows_humaneval112_128.json",
    "actual_fixed_three_rows_humaneval112_128_s34.json",
    "actual_fixed_three_rows_humaneval113_128_s36.json",
    "actual_fixed_three_rows_humaneval113_128_s37.json",
    "actual_fixed_three_rows_humaneval114_128.json",
    "actual_fixed_three_rows_humaneval114_128_s35.json",
    "actual_fixed_three_rows_humaneval115_128_s36.json",
    "actual_fixed_three_rows_humaneval115_128_s37.json",
]


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> None:
    runs = []
    speedups_b1: list[float] = []
    speedups_bnb: list[float] = []
    bnb_matches = 0
    accepted = drafted = 0

    for filename in RUNS:
        data = json.loads((BASE / filename).read_text())
        assert data["parameters"]["quantized_linear_strategy"] == (
            "triton_nf4_fixed_three_rows_group_scale_packed"
        )
        assert data["token_match"] and data["all_shared_states_equal"]
        assert data["shared_state_prefixes"] == 128
        rounds = []
        for result in data["timing"]["paired_rounds"]:
            methods = result["methods"]
            b1 = methods["optimized_b1_singleton"]["total_s"] / methods[
                "optimized_b1_block"
            ]["total_s"]
            bnb = methods["original_bnb_greedy"]["total_s"] / methods[
                "optimized_b1_block"
            ]["total_s"]
            matches_bnb = bool(result["block_matches_original_bnb"])
            bnb_matches += matches_bnb
            speedups_b1.append(b1)
            speedups_bnb.append(bnb)
            rounds.append(
                {
                    "round": result["round"],
                    "speedup_vs_b1": b1,
                    "speedup_vs_original_bnb": bnb,
                    "matches_bnb_tokens": matches_bnb,
                }
            )
        first_block = data["timing"]["paired_rounds"][0]["methods"][
            "optimized_b1_block"
        ]
        accepted += first_block.get("accepted_draft_tokens", 0)
        drafted += first_block.get("draft_tokens", 0)
        runs.append(
            {
                "artifact": filename,
                "prompt_id": data["prompt_id"],
                "session": data["session"],
                "token_match_b1": True,
                "state_prefixes_matched": 128,
                "paired_rounds": rounds,
            }
        )

    prompt_clusters = {}
    for prompt in [f"HumanEval/{n}" for n in range(112, 116)]:
        prompt_runs = [run for run in runs if run["prompt_id"] == prompt]
        p_b1 = [round_["speedup_vs_b1"] for run in prompt_runs for round_ in run["paired_rounds"]]
        p_bnb = [round_["speedup_vs_original_bnb"] for run in prompt_runs for round_ in run["paired_rounds"]]
        prompt_clusters[prompt] = {
            "sessions": [run["session"] for run in prompt_runs],
            "paired_timing_rounds": len(p_b1),
            "geomean_speedup_vs_b1": geomean(p_b1),
            "geomean_speedup_vs_original_bnb": geomean(p_bnb),
            "bnb_token_match_rounds": sum(
                round_["matches_bnb_tokens"]
                for run in prompt_runs
                for round_ in run["paired_rounds"]
            ),
        }

    summary = {
        "classification": "corrected fixed-three-row development benchmark; descriptive timing, not certification",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "configuration": {
            "K": 2,
            "R": 3,
            "physical_rows": 3,
            "max_new_tokens": 128,
            "strategy": "triton_nf4_fixed_three_rows_group_scale_packed",
        },
        "sample_scope": {
            "prompts": list(prompt_clusters),
            "independent_sessions": len(runs),
            "paired_timing_rounds": len(speedups_b1),
            "held_out_used": False,
        },
        "correctness_vs_b1": {
            "sessions_matched": len(runs),
            "sessions_total": len(runs),
            "state_prefixes_matched": len(runs) * 128,
            "state_prefixes_expected": len(runs) * 128,
        },
        "performance": {
            "pooled_geomean_speedup_vs_b1": geomean(speedups_b1),
            "pooled_geomean_speedup_vs_original_bnb": geomean(speedups_bnb),
            "timing_round_speedup_ranges": {
                "vs_b1": [min(speedups_b1), max(speedups_b1)],
                "vs_original_bnb": [min(speedups_bnb), max(speedups_bnb)],
            },
            "bnb_token_match_rounds": bnb_matches,
            "timing_rounds": len(speedups_b1),
            "unique_session_acceptance": {
                "accepted": accepted,
                "drafted": drafted,
                "rate": accepted / drafted,
            },
        },
        "prompt_clusters": prompt_clusters,
        "runs": runs,
        "limitations": [
            "Four fixed HumanEval prompts on one RTX 4050; timing rounds are nested within sessions and prompts.",
            "Treat prompt-level values as descriptive; no prompt-general confidence interval.",
            "Empirical equivalence is relative to experimental B1 singleton; no formal proof.",
            "Original BNB token identity remains prompt-dependent.",
            "Held-out GSM8K indices 300–349 were not used.",
        ],
    }
    output = BASE / "actual_fixed_three_rows_128_four_prompt_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")

    manifest_path = BASE / "capability_gate_smoke_experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence"]["corrected_fixed_three_row_128_four_prompt_summary"] = output.name
    manifest["evidence"]["corrected_fixed_three_row_128_four_prompt_runs"] = RUNS
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    clusters = prompt_clusters
    findings = ROOT / "experiments/exact_quantized_feasibility/FINDINGS.md"
    text = findings.read_text()
    marker = "The boundary/EOS/rollback summary is [actual_fixed_three_rows_boundary_eos_summary.json](triton_nf4_singleton_k2_r3_v1/three_row_kernel/actual_fixed_three_rows_boundary_eos_summary.json)."
    paragraph = (
        f"The corrected 128-token cohort now includes four development prompts (HumanEval/112–115), two independent sessions each, and **8/8** B1 token/state matches with **1,024/1,024 state prefixes**. Across {len(speedups_b1)} paired timing rounds, descriptive pooled geomean speedup is **{geomean(speedups_b1):.3f}x vs B1 singleton** and **{geomean(speedups_bnb):.3f}x vs original BNB**; prompt-level B1/BNB geomeans are "
        f"/112 {clusters['HumanEval/112']['geomean_speedup_vs_b1']:.3f}x/{clusters['HumanEval/112']['geomean_speedup_vs_original_bnb']:.3f}x, "
        f"/113 {clusters['HumanEval/113']['geomean_speedup_vs_b1']:.3f}x/{clusters['HumanEval/113']['geomean_speedup_vs_original_bnb']:.3f}x, "
        f"/114 {clusters['HumanEval/114']['geomean_speedup_vs_b1']:.3f}x/{clusters['HumanEval/114']['geomean_speedup_vs_original_bnb']:.3f}x, "
        f"/115 {clusters['HumanEval/115']['geomean_speedup_vs_b1']:.3f}x/{clusters['HumanEval/115']['geomean_speedup_vs_original_bnb']:.3f}x. "
        f"BNB token IDs match in **{bnb_matches}/{len(speedups_b1)}** timing rounds. The rounds are nested within only four prompts on one GPU, so these are development estimates, not workload-general claims. See [the corrected four-prompt summary](triton_nf4_singleton_k2_r3_v1/three_row_kernel/{output.name}).\n\n"
    )
    if paragraph not in text:
        text = text.replace(marker, paragraph + marker)
    findings.write_text(text)

    print(
        json.dumps(
            {
                "summary": str(output),
                "sessions": len(runs),
                "state_prefixes": len(runs) * 128,
                "b1_geomean": geomean(speedups_b1),
                "bnb_geomean": geomean(speedups_bnb),
                "bnb_matches": f"{bnb_matches}/{len(speedups_b1)}",
                "acceptance": f"{accepted}/{drafted}",
                "prompt_clusters": prompt_clusters,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
