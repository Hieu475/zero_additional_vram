#!/usr/bin/env python3
"""Compare paired K=1/K=2 sweep summaries at 256 and 512 generated tokens."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
INPUTS = {
    256: BASE / "fallback_k_sweep_256_summary_v2.json",
    512: BASE / "fallback_k_sweep_512_summary_v3.json",
}


def gm(values: list[float]) -> float:
    return math.exp(sum(math.log(float(x)) for x in values) / len(values))


def bootstrap(values: list[float], seed: int, draws: int = 20_000) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(gm([rng.choice(values) for _ in values]) for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]]


def main() -> None:
    summaries: dict[int, dict] = {}
    for budget, path in INPUTS.items():
        data = json.loads(path.read_text())
        if data.get("prompt_count") != 6 or data.get("independent_processes") != 12:
            raise RuntimeError(f"unexpected cohort/process count: {path.name}")
        if not data.get("all_k_values_exact_to_b1_tokens_and_kv") or not data.get("timed_candidate_tokens_match_b1"):
            raise RuntimeError(f"B1 exactness gate not satisfied: {path.name}")
        if data.get("heldout_used"):
            raise RuntimeError(f"held-out data marked used: {path.name}")
        summaries[budget] = data

    by_budget = {budget: {p["prompt_id"]: p for p in summary["prompts"]} for budget, summary in summaries.items()}
    if set(by_budget[256]) != set(by_budget[512]):
        raise RuntimeError("budget summaries have different prompt cohorts")

    prompts = []
    k2k1 = {budget: [] for budget in INPUTS}
    k2bnb = {budget: [] for budget in INPUTS}
    k2k1_budget_ratio = []
    k2bnb_budget_ratio = []
    for pid in by_budget[256]:
        p256, p512 = by_budget[256][pid], by_budget[512][pid]
        ratios = {}
        for budget, prompt in ((256, p256), (512, p512)):
            ratios[budget] = {
                "k2_over_k1": float(prompt["k2_speedup_over_k1"]),
                "k2_vs_bnb": float(prompt["speedup_vs_bnb"]["k2_fallback_all"]),
                "k1_vs_bnb": float(prompt["speedup_vs_bnb"]["k1_fallback_all"]),
            }
            k2k1[budget].append(ratios[budget]["k2_over_k1"])
            k2bnb[budget].append(ratios[budget]["k2_vs_bnb"])
        budget_ratio_k = ratios[256]["k2_over_k1"] / ratios[512]["k2_over_k1"]
        budget_ratio_b = ratios[256]["k2_vs_bnb"] / ratios[512]["k2_vs_bnb"]
        k2k1_budget_ratio.append(budget_ratio_k)
        k2bnb_budget_ratio.append(budget_ratio_b)
        prompts.append({"prompt_id": pid, "budgets": {str(k): v for k, v in ratios.items()},
                        "ratio_256_over_512": {"k2_over_k1": budget_ratio_k, "k2_vs_bnb": budget_ratio_b}})

    aggregate = {}
    for budget in INPUTS:
        aggregate[str(budget)] = {
            "k2_over_k1": {
                "geomean_across_prompts": gm(k2k1[budget]),
                "prompt_cluster_bootstrap_95pct": bootstrap(k2k1[budget], 61131 + budget),
                "prompts_k2_faster": sum(x > 1 for x in k2k1[budget]),
            },
            "k2_vs_bnb": {
                "geomean_across_prompts": gm(k2bnb[budget]),
                "prompt_cluster_bootstrap_95pct": bootstrap(k2bnb[budget], 71131 + budget),
                "prompts_faster_than_bnb": sum(x > 1 for x in k2bnb[budget]),
            },
        }

    result = {
        "schema": "zassd-k-sweep-budget-effect-v1",
        "classification": "paired synthesis of two independent K=1/K=2 sweeps per prompt at 256 and 512 generated tokens",
        "scope": "Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, K=1/2, R=3, ngram3, full native-BNB empty-proposal fallback",
        "prompt_count": 6,
        "independent_processes_per_budget": 12,
        "budgets": [256, 512],
        "b1_exactness_in_both_inputs": True,
        "aggregate": aggregate,
        "budget_effect_256_over_512": {
            "k2_over_k1": {"geomean_across_prompts": gm(k2k1_budget_ratio),
                            "prompt_cluster_bootstrap_95pct": bootstrap(k2k1_budget_ratio, 81231),
                            "prompts_256_ratio_higher": sum(x > 1 for x in k2k1_budget_ratio)},
            "k2_vs_bnb": {"geomean_across_prompts": gm(k2bnb_budget_ratio),
                           "prompt_cluster_bootstrap_95pct": bootstrap(k2bnb_budget_ratio, 91231),
                           "prompts_256_ratio_higher": sum(x > 1 for x in k2bnb_budget_ratio)},
        },
        "prompts": prompts,
        "input_sha256": {str(budget): hashlib.sha256(path.read_bytes()).hexdigest() for budget, path in INPUTS.items()},
        "heldout_used": False,
        "limits": [
            "Six selected prompts on one model/GPU; bootstrap resamples prompt clusters and is descriptive.",
            "Budget contrasts share prompts but are not a randomized estimate across workloads.",
            "Original-BNB comparison is latency/token-ID only; no BNB KV-state identity claim.",
            "GSM8K held-out indices 300-349 excluded.",
        ],
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out = BASE / "fallback_k_sweep_budget_effect_summary_v1.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"aggregate": aggregate, "budget_effect_256_over_512": result["budget_effect_256_over_512"],
                      "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
