#!/usr/bin/env python3
"""Aggregate two-session CNN/DailyMail K2/R3 fixed-three-row runs."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel"
FILES = [
    f"actual_fixed_three_rows_cnndm_{prompt}_128_s{session}.json"
    for prompt in ("003", "004", "005")
    for session in (40, 41)
]


def gm(values: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in values) / len(values))


def main() -> None:
    runs = []
    all_b1, all_bnb = [], []
    bnb_matches = 0
    accepted = drafted = 0
    for name in FILES:
        data = json.loads((BASE / name).read_text())
        assert data["parameters"]["quantized_linear_strategy"] == "triton_nf4_fixed_three_rows_group_scale_packed"
        assert data["token_match"] and data["all_shared_states_equal"]
        assert data["shared_state_prefixes"] == 128
        rounds = []
        for pair in data["timing"]["paired_rounds"]:
            methods = pair["methods"]
            block = methods["optimized_b1_block"]
            speed_b1 = methods["optimized_b1_singleton"]["total_s"] / block["total_s"]
            speed_bnb = methods["original_bnb_greedy"]["total_s"] / block["total_s"]
            matches_bnb = bool(pair["block_matches_original_bnb"])
            bnb_matches += matches_bnb
            all_b1.append(speed_b1)
            all_bnb.append(speed_bnb)
            rounds.append(
                {
                    "round": pair["round"],
                    "speedup_vs_b1": speed_b1,
                    "speedup_vs_original_bnb": speed_bnb,
                    "matches_bnb_tokens": matches_bnb,
                }
            )
        first_block = data["timing"]["paired_rounds"][0]["methods"]["optimized_b1_block"]
        accepted += first_block.get("accepted_draft_tokens", 0)
        drafted += first_block.get("draft_tokens", 0)
        runs.append(
            {
                "artifact": name,
                "prompt_id": data["prompt_id"],
                "session": data["session"],
                "state_prefixes_matched": 128,
                "paired_rounds": rounds,
            }
        )

    prompts = {}
    for prompt in ("cnndm_003", "cnndm_004", "cnndm_005"):
        prompt_runs = [run for run in runs if run["prompt_id"] == prompt]
        rounds = [pair for run in prompt_runs for pair in run["paired_rounds"]]
        prompts[prompt] = {
            "sessions": [run["session"] for run in prompt_runs],
            "timing_rounds": len(rounds),
            "geomean_speedup_vs_b1": gm([r["speedup_vs_b1"] for r in rounds]),
            "geomean_speedup_vs_original_bnb": gm(
                [r["speedup_vs_original_bnb"] for r in rounds]
            ),
            "bnb_token_match_rounds": sum(r["matches_bnb_tokens"] for r in rounds),
        }

    summary = {
        "classification": "development-only CNN/DailyMail transfer screen; two sessions per prompt; not certification",
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
            "benchmark": "data/benchmarks/cnndm_eval.jsonl",
            "prompts": list(prompts),
            "independent_sessions": len(runs),
            "paired_timing_rounds": len(all_b1),
            "held_out_used": False,
        },
        "correctness_vs_b1": {
            "sessions_matched": len(runs),
            "sessions_total": len(runs),
            "state_prefixes_matched": 128 * len(runs),
            "state_prefixes_expected": 128 * len(runs),
        },
        "performance": {
            "pooled_geomean_speedup_vs_b1": gm(all_b1),
            "pooled_geomean_speedup_vs_original_bnb": gm(all_bnb),
            "speedup_ranges": {
                "vs_b1": [min(all_b1), max(all_b1)],
                "vs_original_bnb": [min(all_bnb), max(all_bnb)],
            },
            "bnb_token_match_rounds": bnb_matches,
            "timing_rounds": len(all_b1),
            "unique_session_acceptance": {
                "accepted": accepted,
                "drafted": drafted,
                "rate": accepted / drafted,
            },
        },
        "prompt_clusters": prompts,
        "runs": runs,
        "limitations": [
            "Three CNN/DailyMail development prompts on one RTX 4050; timing rounds are nested within sessions.",
            "BNB-relative performance is prompt dependent; no prompt-general confidence interval.",
            "B1 equality is empirical relative to the experimental singleton reference; no formal proof.",
            "Held-out GSM8K indices 300–349 were not used.",
        ],
    }
    output = BASE / "actual_fixed_three_rows_cnndm_128_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")

    manifest_path = BASE / "capability_gate_smoke_experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence"]["corrected_fixed_three_row_cnndm_128_summary"] = output.name
    manifest["evidence"]["corrected_fixed_three_row_cnndm_128_runs"] = FILES
    manifest["source_sha256"]["scripts/summarize_actual_fixed_three_rows_cnndm.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    findings_path = ROOT / "experiments/exact_quantized_feasibility/FINDINGS.md"
    findings = findings_path.read_text()
    old = (
        "The corrected kernel was also screened on three CNN/DailyMail development prompts at 128 output tokens. All **3/3** runs matched B1 across **384/384 state prefixes**. Across six paired rounds, descriptive geomean speedup was **1.266x vs B1 singleton** and **1.142x vs original BNB**. Prompt-level B1/BNB geomeans were cnndm_003 1.351x/1.237x, cnndm_004 1.095x/0.994x, and cnndm_005 1.372x/1.210x. The block matched BNB tokens in **2/6** rounds. The BNB latency result is mixed by prompt; this is a three-prompt, one-session screen, not a stable cross-task claim. See [the CNN/DailyMail summary](triton_nf4_singleton_k2_r3_v1/three_row_kernel/actual_fixed_three_rows_cnndm_128_summary.json)."
    )
    paragraph = (
        f"The corrected kernel was screened on three CNN/DailyMail development prompts at 128 output tokens, with two independent sessions each. All **6/6** runs matched B1 across **768/768 state prefixes**. Across {len(all_b1)} paired rounds, descriptive geomean speedup was **{gm(all_b1):.3f}x vs B1 singleton** and **{gm(all_bnb):.3f}x vs original BNB**. Prompt-level B1/BNB geomeans were "
        f"cnndm_003 {prompts['cnndm_003']['geomean_speedup_vs_b1']:.3f}x/{prompts['cnndm_003']['geomean_speedup_vs_original_bnb']:.3f}x, "
        f"cnndm_004 {prompts['cnndm_004']['geomean_speedup_vs_b1']:.3f}x/{prompts['cnndm_004']['geomean_speedup_vs_original_bnb']:.3f}x, and "
        f"cnndm_005 {prompts['cnndm_005']['geomean_speedup_vs_b1']:.3f}x/{prompts['cnndm_005']['geomean_speedup_vs_original_bnb']:.3f}x. "
        f"The block matched BNB tokens in **{bnb_matches}/{len(all_b1)}** rounds. BNB latency was near parity but prompt dependent; this is a three-prompt, one-GPU development result, not a stable cross-task claim. See [the CNN/DailyMail summary](triton_nf4_singleton_k2_r3_v1/three_row_kernel/{output.name})."
    )
    if old in findings:
        findings = findings.replace(old, paragraph)
    elif paragraph not in findings:
        findings += "\n\n" + paragraph + "\n"
    findings_path.write_text(findings)
    print(
        json.dumps(
            {
                "summary": str(output),
                "b1_geomean": gm(all_b1),
                "bnb_geomean": gm(all_bnb),
                "bnb_token_matches": f"{bnb_matches}/{len(all_b1)}",
                "acceptance": f"{accepted}/{drafted}",
                "prompt_clusters": prompts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
