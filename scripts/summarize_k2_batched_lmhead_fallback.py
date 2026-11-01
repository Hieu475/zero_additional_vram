#!/usr/bin/env python3
"""Summarize paired K2/full-fallback rowwise vs batched LM-head ablation."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
MANIFEST = BASE / "k2_batched_lmhead_fallback_manifest.json"
COHORT = (("HumanEval/123", "HumanEval123"), ("HumanEval/114", "HumanEval114"))
BUDGETS = (256, 512)


def gm(values: list[float]) -> float:
    return math.exp(sum(math.log(float(v)) for v in values) / len(values))


def bootstrap(values: list[float], seed: int, draws: int = 20_000) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(gm([rng.choice(values) for _ in values]) for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]]


def main() -> None:
    manifest_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    strata = []
    exact_artifacts = 0
    bnb_matches = {"rowwise": 0, "batched": 0}
    inputs = {}
    for prompt_id, tag in COHORT:
        for budget in BUDGETS:
            name = f"k2_batched_lmhead_{tag}_{budget}_fallback_v1.json"
            path = BASE / name
            data = json.loads(path.read_text())
            if (data.get("schema") != "zassd-k2-batched-lmhead-fallback-v1"
                    or data.get("prompt_id") != prompt_id
                    or data.get("max_new_tokens") != budget
                    or data.get("sweep_manifest_sha256") != manifest_hash):
                raise RuntimeError(f"invalid or unpinned artifact: {name}")
            pre = data["b1_preflight"]
            if not (data["timed_b1_exact"] and all(pre["token_match"].values())
                    and pre["state_domains_equal"] and not pre["rowwise_state_mismatches"]
                    and not pre["batched_state_mismatches"]):
                raise RuntimeError(f"B1 token/KV mismatch: {name}")
            exact_artifacts += 1
            inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            r_batched = []
            r_bnb_rowwise = []
            r_bnb_batched = []
            for round_data in data["paired_rounds"]:
                methods = round_data["methods"]
                r_batched.append(methods["rowwise_full_fallback"]["total_s"] /
                                 methods["batched_lmhead_full_fallback"]["total_s"])
                r_bnb_rowwise.append(methods["original_bnb_greedy"]["total_s"] /
                                     methods["rowwise_full_fallback"]["total_s"])
                r_bnb_batched.append(methods["original_bnb_greedy"]["total_s"] /
                                     methods["batched_lmhead_full_fallback"]["total_s"])
                bnb_matches["rowwise"] += int(round_data["rowwise_matches_bnb"])
                bnb_matches["batched"] += int(round_data["batched_matches_bnb"])
            strata.append({
                "prompt_id": prompt_id,
                "max_new_tokens": budget,
                "paired_rounds": len(r_batched),
                "batched_over_rowwise_geomean": gm(r_batched),
                "rowwise_speedup_vs_bnb_geomean": gm(r_bnb_rowwise),
                "batched_speedup_vs_bnb_geomean": gm(r_bnb_batched),
                "batched_over_rowwise_round_ratios": r_batched,
                "artifact": name,
            })

    ratios = [s["batched_over_rowwise_geomean"] for s in strata]
    result = {
        "schema": "zassd-k2-batched-lmhead-fallback-summary-v1",
        "classification": "paired diagnostic ablation, two prompts by two budgets, one process and four timing rounds per stratum",
        "scope": "Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, K2/R3, ngram3, full fallback",
        "strata": strata,
        "stratum_count": len(strata),
        "timing_rounds_per_stratum": 4,
        "all_artifacts_b1_token_and_kv_exact": exact_artifacts == 4,
        "original_bnb_token_matches": bnb_matches,
        "batched_over_rowwise": {
            "geomean_across_prompt_budget_strata": gm(ratios),
            "descriptive_stratum_bootstrap_95pct": bootstrap(ratios, 58173),
            "strata_batched_faster": sum(v > 1 for v in ratios),
            "min": min(ratios),
            "max": max(ratios),
        },
        "input_sha256": inputs,
        "manifest_sha256": manifest_hash,
        "heldout_used": False,
        "limits": [
            "Four selected prompt-budget strata on one model/GPU; bootstrap is descriptive and not a population interval.",
            "Ordinary paired latency; no BNB KV-state identity claim.",
            "GSM8K held-out indices 300-349 excluded.",
        ],
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = BASE / "k2_batched_lmhead_fallback_summary_v1.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"summary": str(output), "exact_artifacts": exact_artifacts,
                      "bnb_matches": bnb_matches,
                      "batched_over_rowwise": result["batched_over_rowwise"],
                      "strata": [{k: s[k] for k in ("prompt_id", "max_new_tokens", "batched_over_rowwise_geomean",
                                                       "rowwise_speedup_vs_bnb_geomean", "batched_speedup_vs_bnb_geomean")}
                                 for s in strata]}, indent=2))


if __name__ == "__main__":
    main()
