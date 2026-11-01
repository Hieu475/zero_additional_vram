#!/usr/bin/env python3
"""Summarize exactness-gated activationvec/fixed-three paired development runs."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
INPUT_GROUPS = {
    "HumanEval/112": ["activationvec_fixedthree_linear_humaneval112_64_contract_paired_v1.json",
                      "activationvec_fixedthree_linear_humaneval112_64_contract_paired_rep2_v1.json"],
    "HumanEval/116": ["activationvec_fixedthree_linear_humaneval116_64_contract_paired_v2.json",
                      "activationvec_fixedthree_linear_humaneval116_64_contract_paired_rep2_v1.json"],
    "HumanEval/120": ["activationvec_fixedthree_linear_humaneval120_64_contract_paired_v1.json",
                      "activationvec_fixedthree_linear_humaneval120_64_contract_paired_rep2_v1.json"],
    "HumanEval/123": ["activationvec_fixedthree_linear_humaneval123_64_contract_paired_v2.json",
                      "activationvec_fixedthree_linear_humaneval123_64_contract_paired_rep2_v1.json"],
    "HumanEval/124": ["activationvec_fixedthree_linear_humaneval124_64_contract_paired_v1.json",
                      "activationvec_fixedthree_linear_humaneval124_64_contract_paired_rep2_v1.json"],
}


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def prompt_bootstrap(values: list[float], seed: int = 44031, draws: int = 20000) -> list[float]:
    rng = random.Random(seed)
    estimates = [geometric_mean([rng.choice(values) for _ in values]) for _ in range(draws)]
    estimates.sort()
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]]


def main() -> None:
    files = [name for group in INPUT_GROUPS.values() for name in group]
    prompts = []
    for prompt_id, group_files in INPUT_GROUPS.items():
        session_data = []
        for name in group_files:
            data = json.loads((BASE / name).read_text())
            ex = data["exactness"]
            if not (ex["activationvec_tokens_match_bnb_singleton"]
                    and ex["fixedthree_tokens_match_own_singleton"]
                    and not ex["activationvec_state_mismatches"]
                    and not ex["fixedthree_state_mismatches"]
                    and ex["activationvec_state_domain_equal_bnb"]
                    and ex["fixedthree_state_domain_equal_own"]):
                raise RuntimeError(f"exactness gate failed for {data['prompt_id']}")
            rounds = data["paired_rounds"]
            fixedthree_vs_activationvec = [
                row["methods"]["activationvec"]["total_s"] / row["methods"]["fixedthree"]["total_s"]
                for row in rounds
            ]
            fixedthree_vs_own_singleton = [
                row["methods"]["fixedthree_singleton"]["total_s"] / row["methods"]["fixedthree"]["total_s"]
                for row in rounds
            ]
            activationvec_vs_bnb = [
                row["methods"]["original_bnb_greedy"]["total_s"] / row["methods"]["activationvec"]["total_s"]
                for row in rounds
            ]
            fixedthree_vs_bnb = [
                row["methods"]["original_bnb_greedy"]["total_s"] / row["methods"]["fixedthree"]["total_s"]
                for row in rounds
            ]
            session_data.append({
                "artifact": name,
                "paired_rounds": len(rounds),
                "activationvec_acceptance": data["acceptance"]["activationvec"],
                "fixedthree_acceptance": data["acceptance"]["fixedthree"],
                "fixedthree_tokens_match_bnb_singleton": (
                    ex["fixedthree_singleton_tokens"] == ex["bnb_singleton_tokens"]
                ),
                "fixedthree_speedup_vs_own_singleton": geometric_mean(fixedthree_vs_own_singleton),
                "fixedthree_speedup_vs_activationvec_block": geometric_mean(fixedthree_vs_activationvec),
                "activationvec_speedup_vs_original_bnb_latency": geometric_mean(activationvec_vs_bnb),
                "fixedthree_speedup_vs_original_bnb_latency": geometric_mean(fixedthree_vs_bnb),
            })
        metric_keys = [key for key in session_data[0] if key.startswith("fixedthree_speedup_")
                       or key == "activationvec_speedup_vs_original_bnb_latency"]
        prompts.append({
            "prompt_id": prompt_id,
            "independent_processes": len(session_data),
            "exact_in_all_processes_to_own_reference": True,
            "fixedthree_tokens_match_bnb_in_all_processes": all(
                row["fixedthree_tokens_match_bnb_singleton"] for row in session_data),
            "prompt_metrics_geomean_across_processes": {
                key: geometric_mean([row[key] for row in session_data]) for key in metric_keys
            },
            "sessions": session_data,
        })

    across_prompt = {
        key: {
            "geomean": geometric_mean([row["prompt_metrics_geomean_across_processes"][key] for row in prompts]),
            "prompt_cluster_bootstrap_95pct": prompt_bootstrap([
                row["prompt_metrics_geomean_across_processes"][key] for row in prompts]),
            "min": min(row["prompt_metrics_geomean_across_processes"][key] for row in prompts),
            "max": max(row["prompt_metrics_geomean_across_processes"][key] for row in prompts),
        }
        for key in (
            "fixedthree_speedup_vs_own_singleton",
            "fixedthree_speedup_vs_activationvec_block",
            "activationvec_speedup_vs_original_bnb_latency",
            "fixedthree_speedup_vs_original_bnb_latency",
        )
    }
    inputs = {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in files}
    result = {
        "schema": "zassd-activationvec-fixedthree-paired-summary-v2",
        "classification": "five selected development prompt clusters, two independent processes per prompt; descriptive prompt bootstrap, not population inference",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "runtime_scope": "RTX 4050 Laptop GPU, conda env ai, K=2/R=3, 64 generated tokens",
        "exactness": "activationvec versus BNB fixed-row singleton; fixedthree versus matching custom fixed-three singleton",
        "prompt_count": len(prompts),
        "independent_processes": sum(len(group) for group in INPUT_GROUPS.values()),
        "all_runs_exact_to_own_reference": True,
        "heldout_used": False,
        "prompts": prompts,
        "across_prompt_summary": across_prompt,
        "input_sha256": inputs,
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": [
            "Five selected HumanEval prompt clusters, two independent processes and four paired rounds per prompt/process.",
            "Bootstrap resamples prompt clusters, not independent model/GPU sessions.",
            "Latency versus original BNB is reported separately; fixedthree token identity differs on some prompts.",
            "No BNB state equivalence claim; GSM8K held-out indices 300-349 excluded.",
        ],
    }
    out = BASE / "activationvec_fixedthree_paired_summary_v2.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"prompt_count": len(prompts), "exact": True,
                      "across_prompt_summary": across_prompt, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
