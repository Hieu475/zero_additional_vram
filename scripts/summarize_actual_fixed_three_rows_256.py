#!/usr/bin/env python3
"""Summarize one 256-token fixed-three-row K2/R3 run per development prompt."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel"
FILES = [
    f"actual_fixed_three_rows_humaneval{n}_256_s{session}.json"
    for n in range(112, 116)
    for session in (38, 39)
]


def gm(values: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in values) / len(values))


def main() -> None:
    rows = []
    b1_ratios, bnb_ratios = [], []
    matches_bnb = 0
    accepted = drafted = 0
    for name in FILES:
        data = json.loads((BASE / name).read_text())
        assert data["parameters"]["quantized_linear_strategy"] == "triton_nf4_fixed_three_rows_group_scale_packed"
        assert data["token_match"] and data["all_shared_states_equal"]
        assert data["shared_state_prefixes"] == 256
        timing_rounds = []
        for timing in data["timing"]["paired_rounds"]:
            methods = timing["methods"]
            block = methods["optimized_b1_block"]
            b1 = methods["optimized_b1_singleton"]["total_s"] / block["total_s"]
            bnb = methods["original_bnb_greedy"]["total_s"] / block["total_s"]
            matched_bnb = bool(timing["block_matches_original_bnb"])
            matches_bnb += matched_bnb
            b1_ratios.append(b1)
            bnb_ratios.append(bnb)
            timing_rounds.append(
                {
                    "round": timing["round"],
                    "speedup_vs_b1": b1,
                    "speedup_vs_original_bnb": bnb,
                    "matches_bnb_tokens": matched_bnb,
                }
            )
        # Count acceptance once per independent session, not once per timing repeat.
        first_block = data["timing"]["paired_rounds"][0]["methods"]["optimized_b1_block"]
        accepted += first_block.get("accepted_draft_tokens", 0)
        drafted += first_block.get("draft_tokens", 0)
        rows.append(
            {
                "artifact": name,
                "prompt_id": data["prompt_id"],
                "session": data["session"],
                "state_prefixes_matched": 256,
                "timing_rounds": timing_rounds,
                "accepted_draft_tokens": first_block.get("accepted_draft_tokens", 0),
                "draft_tokens": first_block.get("draft_tokens", 0),
            }
        )

    prompt_clusters = {}
    for prompt_number in range(112, 116):
        prompt = f"HumanEval/{prompt_number}"
        prompt_runs = [row for row in rows if row["prompt_id"] == prompt]
        prompt_rounds = [
            round_
            for row in prompt_runs
            for round_ in row["timing_rounds"]
        ]
        prompt_clusters[prompt] = {
            "sessions": [row["session"] for row in prompt_runs],
            "timing_rounds": len(prompt_rounds),
            "geomean_speedup_vs_b1": gm([r["speedup_vs_b1"] for r in prompt_rounds]),
            "geomean_speedup_vs_original_bnb": gm(
                [r["speedup_vs_original_bnb"] for r in prompt_rounds]
            ),
            "bnb_token_match_rounds": sum(r["matches_bnb_tokens"] for r in prompt_rounds),
        }

    summary = {
        "classification": "development-only 256-token repeat; timing rounds nested in two sessions per prompt; not certification",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "gpu": "NVIDIA GeForce RTX 4050 Laptop GPU",
        "configuration": {
            "K": 2,
            "R": 3,
            "physical_rows": 3,
            "max_new_tokens": 256,
            "strategy": "triton_nf4_fixed_three_rows_group_scale_packed",
        },
        "correctness_vs_b1": {
            "sessions_matched": len(rows),
            "sessions_total": len(rows),
            "state_prefixes_matched": 256 * len(rows),
            "state_prefixes_expected": 256 * len(rows),
        },
        "performance": {
            "geomean_speedup_vs_b1": gm(b1_ratios),
            "geomean_speedup_vs_original_bnb": gm(bnb_ratios),
            "speedup_ranges": {
                "vs_b1": [min(b1_ratios), max(b1_ratios)],
                "vs_original_bnb": [min(bnb_ratios), max(bnb_ratios)],
            },
            "bnb_token_match_timing_rounds": matches_bnb,
            "timing_rounds": len(b1_ratios),
            "prompts": 4,
            "unique_session_acceptance": {
                "accepted": accepted,
                "drafted": drafted,
                "rate": accepted / drafted,
            },
        },
        "prompt_clusters": prompt_clusters,
        "runs": rows,
        "limitations": [
            "Four fixed development prompts on one RTX 4050; timing rounds are nested within two sessions per prompt.",
            "No prompt-general confidence interval; timing repeats are not independent prompts.",
            "B1 equality is empirical relative to the experimental singleton reference; no formal proof.",
            "BNB token identity is not established across prompts.",
            "Held-out GSM8K indices 300–349 were not used.",
        ],
    }
    path = BASE / "actual_fixed_three_rows_256_four_prompt_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")

    manifest_path = BASE / "capability_gate_smoke_experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence"]["corrected_fixed_three_row_256_four_prompt_summary"] = path.name
    manifest["evidence"]["corrected_fixed_three_row_256_four_prompt_runs"] = FILES
    manifest["source_sha256"]["scripts/summarize_actual_fixed_three_rows_256.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    findings_path = ROOT / "experiments/exact_quantized_feasibility/FINDINGS.md"
    findings = findings_path.read_text()
    old_text = (
        "A longer 256-token run used one independent process and one paired timing round on each of the same four development prompts. All **4/4** runs matched B1 tokens and **1,024/1,024** state prefixes. Descriptive geomean speedup was **1.711x vs B1 singleton** and **1.540x vs original BNB**; all four timing ratios favored the block path, but this is only one timing pair per prompt. Original BNB token IDs matched on **1/4** prompts. Proposal acceptance totaled **412/661 (62.3%)**. See [the 256-token summary](triton_nf4_singleton_k2_r3_v1/three_row_kernel/actual_fixed_three_rows_256_four_prompt_summary.json). This extends finite B1 state-match evidence to a longer horizon; it does not establish B2 identity or a stable performance interval."
    )
    timing_total = len(b1_ratios)
    prompt_text = ", ".join(
        f"/{n} {prompt_clusters[f'HumanEval/{n}']['geomean_speedup_vs_b1']:.3f}x/{prompt_clusters[f'HumanEval/{n}']['geomean_speedup_vs_original_bnb']:.3f}x"
        for n in range(112, 116)
    )
    new_text = (
        f"The 256-token repeat now covers two independent processes on each of the four development prompts. All **8/8** runs matched B1 tokens and **2,048/2,048** state prefixes. Across {timing_total} paired rounds, descriptive pooled geomean speedup was **{gm(b1_ratios):.3f}x vs B1 singleton** and **{gm(bnb_ratios):.3f}x vs original BNB**; prompt-level B1/BNB estimates were {prompt_text}. Original BNB token IDs matched on **{matches_bnb}/{timing_total}** rounds. Unique-session proposal acceptance was **{accepted}/{drafted} ({100 * accepted / drafted:.1f}%)**. Timing rounds are nested within two sessions per prompt and are not independent prompt samples. See [the 256-token summary](triton_nf4_singleton_k2_r3_v1/three_row_kernel/{path.name}). This extends empirical B1 state matching to a longer horizon; it does not establish B2 identity or a prompt-general speedup."
    )
    if old_text in findings:
        findings = findings.replace(old_text, new_text)
    elif new_text not in findings:
        findings += "\n\n" + new_text + "\n"
    findings_path.write_text(findings)

    print(
        json.dumps(
            {
                "summary": str(path),
                "b1_geomean": gm(b1_ratios),
                "bnb_geomean": gm(bnb_ratios),
                "bnb_token_matches": f"{matches_bnb}/{len(b1_ratios)}",
                "acceptance": f"{accepted}/{drafted}",
                "runs": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
