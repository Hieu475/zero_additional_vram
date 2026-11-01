#!/usr/bin/env python3
"""Summarize exactness-gated fallback runs on CNN/DailyMail prompts."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
PROMPTS = {
    "cnndm_001": ["activationvec_no_proposal_fallback_cnndm001_64_v1.json",
                  "activationvec_no_proposal_fallback_cnndm001_64_rep2_v1.json"],
    "cnndm_002": ["activationvec_no_proposal_fallback_cnndm002_64_v1.json",
                  "activationvec_no_proposal_fallback_cnndm002_64_rep2_v1.json"],
    "cnndm_003": ["activationvec_no_proposal_fallback_cnndm003_64_v1.json",
                  "activationvec_no_proposal_fallback_cnndm003_64_rep2_v1.json"],
    "cnndm_004": ["activationvec_no_proposal_fallback_cnndm004_64_v1.json",
                  "activationvec_no_proposal_fallback_cnndm004_64_rep2_v1.json"],
    "cnndm_005": ["activationvec_no_proposal_fallback_cnndm005_64_v1.json",
                  "activationvec_no_proposal_fallback_cnndm005_64_rep2_v1.json"],
    "cnndm_006": ["activationvec_no_proposal_fallback_cnndm006_64_v1.json",
                  "activationvec_no_proposal_fallback_cnndm006_64_rep2_v1.json"],
}


def gmean(values: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in values) / len(values))


def bootstrap(values: list[float], *, seed: int, draws: int = 20000) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(gmean([rng.choice(values) for _ in values]) for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]]


def main() -> None:
    rows = []
    input_hashes = {}
    totals = {"fallback_b1": 0, "activationvec_b1": 0,
              "fallback_bnb": 0, "activationvec_bnb": 0, "rounds": 0}
    for prompt_id, files in PROMPTS.items():
        sessions = []
        for name in files:
            path = BASE / name
            data = json.loads(path.read_text())
            input_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            exact = data["exactness"]
            if not (exact["activationvec_tokens_match_singleton"]
                    and exact["fallback_tokens_match_singleton"]
                    and exact["state_domains_equal"]
                    and not exact["activationvec_state_mismatches"]
                    and not exact["fallback_state_mismatches"]):
                raise RuntimeError(f"B1 exactness failed: {name}")
            paired = data["paired_rounds"]
            fallback_b1 = sum(
                row["methods"]["native_fallback"]["token_ids"]
                == row["methods"]["bnb_singleton"]["token_ids"] for row in paired)
            activation_b1 = sum(
                row["methods"]["activationvec"]["token_ids"]
                == row["methods"]["bnb_singleton"]["token_ids"] for row in paired)
            if fallback_b1 != len(paired) or activation_b1 != len(paired):
                raise RuntimeError(f"timed B1 token mismatch: {name}")
            accounting = data["no_proposal_fallback_counts_preflight"]
            if accounting["no_proposal_cycles"] != accounting["fallback_forwards"]:
                raise RuntimeError(f"fallback accounting mismatch: {name}")
            fallback_bnb = sum(row["methods"]["native_fallback"]["token_ids"]
                               == row["methods"]["original_bnb"]["token_ids"] for row in paired)
            activation_bnb = sum(row["methods"]["activationvec"]["token_ids"]
                                 == row["methods"]["original_bnb"]["token_ids"] for row in paired)
            totals["fallback_b1"] += fallback_b1
            totals["activationvec_b1"] += activation_b1
            totals["fallback_bnb"] += fallback_bnb
            totals["activationvec_bnb"] += activation_bnb
            totals["rounds"] += len(paired)
            sessions.append({
                "artifact": name,
                "no_proposal_cycles": data["no_proposal_fallback_counts_preflight"]["no_proposal_cycles"],
                "fallback_speedup_vs_activationvec": data["ratio_medians"]["fallback_over_activationvec"],
                "fallback_speedup_vs_original_bnb_latency": data["ratio_medians"]["fallback_over_original_bnb_latency"],
                "fallback_b1_token_match_rounds": fallback_b1,
                "activationvec_b1_token_match_rounds": activation_b1,
                "fallback_bnb_token_match_rounds": fallback_bnb,
                "activationvec_bnb_token_match_rounds": activation_bnb,
                "timing_rounds": len(paired),
            })
        metrics = {}
        for key in ("fallback_speedup_vs_activationvec",
                    "fallback_speedup_vs_original_bnb_latency"):
            metrics[key] = gmean([session[key] for session in sessions])
        rows.append({"prompt_id": prompt_id, "independent_processes": len(sessions),
                     "all_processes_exact_to_b1": True,
                     "prompt_metrics_geomean_across_processes": metrics,
                     "sessions": sessions})

    across = {}
    for key in ("fallback_speedup_vs_activationvec",
                "fallback_speedup_vs_original_bnb_latency"):
        values = [row["prompt_metrics_geomean_across_processes"][key] for row in rows]
        across[key] = {"geomean_across_prompts": gmean(values),
                       "prompt_cluster_bootstrap_95pct": bootstrap(values, seed=38229),
                       "prompt_min": min(values), "prompt_max": max(values)}
    result = {
        "schema": "zassd-activationvec-no-proposal-fallback-cnndm-summary-v2",
        "classification": "six CNN/DailyMail prompt clusters, two independent processes each; descriptive prompt bootstrap",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "runtime_scope": "RTX 4050 Laptop GPU, conda env ai, K=2/R=3, 64 generated tokens, four paired rounds/process",
        "prompt_count": len(rows), "independent_processes": sum(len(x) for x in PROMPTS.values()),
        "all_runs_exact_to_b1_singleton": True,
        "fallback_b1_token_match_rounds": totals["fallback_b1"],
        "activationvec_b1_token_match_rounds": totals["activationvec_b1"],
        "fallback_original_bnb_token_match_rounds": totals["fallback_bnb"],
        "activationvec_original_bnb_token_match_rounds": totals["activationvec_bnb"],
        "paired_rounds": totals["rounds"],
        "prompts": rows, "across_prompt_summary": across,
        "input_sha256": input_hashes,
        "heldout_used": False,
        "limits": ["Bootstrap resamples six selected prompt clusters on one model and one GPU.",
                   "Bootstrap resamples prompt clusters, not sessions.",
                   "Original-BNB comparison includes token IDs; no BNB KV-state identity claim.",
                   "GSM8K held-out indices 300-349 excluded."],
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = BASE / "activationvec_no_proposal_fallback_cnndm_summary_v2.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"prompt_count": result["prompt_count"],
                      "independent_processes": result["independent_processes"],
                      "paired_rounds": result["paired_rounds"],
                      "fallback_b1_token_match_rounds": result["fallback_b1_token_match_rounds"],
                      "fallback_bnb_token_match_rounds": result["fallback_original_bnb_token_match_rounds"],
                      "across_prompt_summary": across, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
