#!/usr/bin/env python3
"""Summarize the controlled EOS/rollback/fallback branch audit artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
INPUTS = {
    "HumanEval/111": "activationvec_no_proposal_fallback_branches_humaneval111_v1.json",
    "HumanEval/114": "activationvec_no_proposal_fallback_branches_humaneval114_v1.json",
    "HumanEval/116": "activationvec_no_proposal_fallback_branches_humaneval116_v2.json",
    "HumanEval/121": "activationvec_no_proposal_fallback_branches_humaneval121_v1.json",
    "HumanEval/123": "activationvec_no_proposal_fallback_branches_humaneval123_v2.json",
}


def main() -> None:
    rows = []
    hashes = {}
    for prompt_id, name in INPUTS.items():
        path = BASE / name
        data = json.loads(path.read_text())
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        by_name = {scenario["name"]: scenario for scenario in data["scenarios"]}
        expected = {
            "accept_one_reject_eos_then_empty_proposals",
            "reject_first_and_eos_suffix_then_empty_proposals",
            "accept_terminal_eos_without_fallback",
        }
        if not data["all_scenarios_pass"] or set(by_name) != expected:
            raise RuntimeError(f"scenario set or pass flag invalid: {name}")
        if any(not x["token_match_reference"] or not x["state_exact_reference"]
               for x in by_name.values()):
            raise RuntimeError(f"token/KV mismatch: {name}")
        if any(by_name[key]["fallback_counts"]["fallback_forwards"] <= 0
               for key in ("accept_one_reject_eos_then_empty_proposals",
                           "reject_first_and_eos_suffix_then_empty_proposals")):
            raise RuntimeError(f"rollback did not reach fallback: {name}")
        terminal = by_name["accept_terminal_eos_without_fallback"]
        if not terminal["accepted_terminal_eos"] or terminal["fallback_counts"]["fallback_forwards"] != 0:
            raise RuntimeError(f"terminal EOS fallback behavior invalid: {name}")
        rows.append({
            "prompt_id": prompt_id,
            "artifact": name,
            "scenarios": {scenario_name: {
                "token_match_reference": by_name[scenario_name]["token_match_reference"],
                "state_exact_reference": by_name[scenario_name]["state_exact_reference"],
                "fallback_forwards": by_name[scenario_name]["fallback_counts"]["fallback_forwards"],
            } for scenario_name in sorted(expected)},
        })
    result = {
        "schema": "zassd-activationvec-no-proposal-fallback-branches-summary-v1",
        "classification": "controlled proposal branch audit; finite EOS/rollback/fallback evidence",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "prompt_count": len(rows),
        "scenario_count": sum(len(x["scenarios"]) for x in rows),
        "all_scenarios_pass": True,
        "rollback_cases_reach_fallback": True,
        "accepted_terminal_eos_skips_fallback": True,
        "prompts": rows,
        "input_sha256": hashes,
        "heldout_used": False,
        "limits": [
            "Controlled n-gram proposal scripts; not a natural proposal distribution.",
            "Accepted-EOS uses a test-only tokenizer EOS override.",
            "Five prompts on one GPU; GSM8K held-out indices 300-349 excluded.",
        ],
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = BASE / "activationvec_no_proposal_fallback_branches_summary_v1.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"prompt_count": result["prompt_count"],
                      "scenario_count": result["scenario_count"],
                      "all_pass": result["all_scenarios_pass"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
