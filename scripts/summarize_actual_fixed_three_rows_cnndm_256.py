#!/usr/bin/env python3
"""Summarize the two-session 256-token CNN/DailyMail transfer screen."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel"
FILES = [
    f"actual_fixed_three_rows_cnndm_{n}_256_s{session}.json"
    for n in ("003", "004", "005")
    for session in (42, 43)
]


def gm(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> None:
    rows, speed_b1, speed_bnb = [], [], []
    bnb_matches = accepted = drafted = 0
    for name in FILES:
        data = json.loads((BASE / name).read_text())
        assert data["parameters"]["quantized_linear_strategy"] == "triton_nf4_fixed_three_rows_group_scale_packed"
        assert data["token_match"] and data["all_shared_states_equal"]
        assert data["shared_state_prefixes"] == 256
        pairs = []
        for pair in data["timing"]["paired_rounds"]:
            methods = pair["methods"]
            block = methods["optimized_b1_block"]
            b1 = methods["optimized_b1_singleton"]["total_s"] / block["total_s"]
            bnb = methods["original_bnb_greedy"]["total_s"] / block["total_s"]
            matches = bool(pair["block_matches_original_bnb"])
            speed_b1.append(b1)
            speed_bnb.append(bnb)
            bnb_matches += matches
            pairs.append(
                {
                    "round": pair["round"],
                    "method_order": pair["method_order"],
                    "speedup_vs_b1": b1,
                    "speedup_vs_original_bnb": bnb,
                    "matches_bnb_tokens": matches,
                }
            )
        # Acceptance is a session-level count; timing repeats reuse this same decode trace.
        first_block = data["timing"]["paired_rounds"][0]["methods"]["optimized_b1_block"]
        block = first_block
        accepted += block.get("accepted_draft_tokens", 0)
        drafted += block.get("draft_tokens", 0)
        rows.append(
            {
                "artifact": name,
                "prompt_id": data["prompt_id"],
                "session": data["session"],
                "state_prefixes_matched": 256,
                "paired_rounds": pairs,
                "accepted_draft_tokens": block.get("accepted_draft_tokens", 0),
                "draft_tokens": block.get("draft_tokens", 0),
            }
        )

    prompts = {}
    for prompt in ("cnndm_003", "cnndm_004", "cnndm_005"):
        prompt_runs = [row for row in rows if row["prompt_id"] == prompt]
        pairs = [pair for row in prompt_runs for pair in row["paired_rounds"]]
        prompts[prompt] = {
            "sessions": [row["session"] for row in prompt_runs],
            "timing_rounds": len(pairs),
            "geomean_speedup_vs_b1": gm([p["speedup_vs_b1"] for p in pairs]),
            "geomean_speedup_vs_original_bnb": gm(
                [p["speedup_vs_original_bnb"] for p in pairs]
            ),
            "bnb_token_match_rounds": sum(p["matches_bnb_tokens"] for p in pairs),
        }

    summary = {
        "classification": "development-only CNN/DailyMail 256-token repeat; two sessions per prompt; not certification",
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
            "geomean_speedup_vs_b1": gm(speed_b1),
            "geomean_speedup_vs_original_bnb": gm(speed_bnb),
            "speedup_ranges": {
                "vs_b1": [min(speed_b1), max(speed_b1)],
                "vs_original_bnb": [min(speed_bnb), max(speed_bnb)],
            },
            "prompt_clusters": prompts,
            "bnb_token_match_timing_rounds": bnb_matches,
            "timing_rounds": len(speed_b1),
            "unique_session_acceptance": {
                "accepted": accepted,
                "drafted": drafted,
                "rate": accepted / drafted,
            },
        },
        "limitations": [
            "Three CNN/DailyMail development prompts on one GPU; timing rounds are nested within two sessions per prompt.",
            "The first timing round in cnndm_003 session 43 is an outlier with block first in the method order; performance estimates remain descriptive.",
            "Exactness is empirical relative to experimental B1 singleton; no formal proof.",
            "BNB token identity did not match in any 256-token run; no BNB equality claim.",
            "Held-out GSM8K indices 300–349 were not used.",
        ],
    }
    output = BASE / "actual_fixed_three_rows_cnndm_256_summary.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")

    manifest_path = BASE / "capability_gate_smoke_experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evidence"]["corrected_fixed_three_row_cnndm_256_summary"] = output.name
    manifest["evidence"]["corrected_fixed_three_row_cnndm_256_runs"] = FILES
    manifest["source_sha256"]["scripts/summarize_actual_fixed_three_rows_cnndm_256.py"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    findings_path = ROOT / "experiments/exact_quantized_feasibility/FINDINGS.md"
    findings = findings_path.read_text()
    old = next(
        (line for line in findings.splitlines() if line.startswith("The CNN/DailyMail screen was extended to 256 output tokens")),
        None,
    )
    p003, p004, p005 = (prompts[f"cnndm_{n}"] for n in ("003", "004", "005"))
    paragraph = (
        f"The CNN/DailyMail 256-token screen now has two independent sessions for each of three prompts. All **6/6** runs matched B1 across **1,536/1,536 state prefixes**. Across {len(speed_b1)} paired rounds, descriptive geomean speedup is **{gm(speed_b1):.3f}x vs B1 singleton** and **{gm(speed_bnb):.3f}x vs original BNB**. Prompt-level B1/BNB geomeans were cnndm_003 {p003['geomean_speedup_vs_b1']:.3f}x/{p003['geomean_speedup_vs_original_bnb']:.3f}x, cnndm_004 {p004['geomean_speedup_vs_b1']:.3f}x/{p004['geomean_speedup_vs_original_bnb']:.3f}x, and cnndm_005 {p005['geomean_speedup_vs_b1']:.3f}x/{p005['geomean_speedup_vs_original_bnb']:.3f}x. BNB token IDs matched in **{bnb_matches}/{len(speed_b1)}** rounds. The first /003 round in session 43 is an outlier (0.86x vs B1; 0.73x vs BNB) and used block-first method order; the other two rounds of session 43 were faster. The /004 block remained slower than BNB across all four paired rounds. Timing rounds are nested within two sessions per prompt, so these results are descriptive and prompt-specific. See [the 256-token CNN/DailyMail summary](triton_nf4_singleton_k2_r3_v1/three_row_kernel/{output.name})."
    )
    if old:
        findings = findings.replace(old, paragraph)
    elif paragraph not in findings:
        findings += "\n\n" + paragraph + "\n"
    findings_path.write_text(findings)
    print(
        json.dumps(
            {
                "summary": str(output),
                "b1_geomean": gm(speed_b1),
                "bnb_geomean": gm(speed_bnb),
                "bnb_matches": f"{bnb_matches}/{len(speed_b1)}",
                "runs": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
