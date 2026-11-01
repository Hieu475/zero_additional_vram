#!/usr/bin/env python3
"""Summarize exactness-gated 256-token fallback runs on HumanEval."""
from __future__ import annotations
import hashlib, json, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
PROMPTS = {
    "HumanEval/114": ["activationvec_no_proposal_fallback_humaneval114_256_v1.json", "activationvec_no_proposal_fallback_humaneval114_256_rep2_v1.json"],
    "HumanEval/112": ["activationvec_no_proposal_fallback_humaneval112_256_v1.json", "activationvec_no_proposal_fallback_humaneval112_256_rep2_v1.json"],
    "HumanEval/116": ["activationvec_no_proposal_fallback_humaneval116_256_v1.json", "activationvec_no_proposal_fallback_humaneval116_256_rep2_v1.json"],
    "HumanEval/123": ["activationvec_no_proposal_fallback_humaneval123_256_v1.json", "activationvec_no_proposal_fallback_humaneval123_256_rep2_v1.json"],
    "HumanEval/121": ["activationvec_no_proposal_fallback_humaneval121_256_v1.json", "activationvec_no_proposal_fallback_humaneval121_256_rep2_v1.json"],
    "HumanEval/124": ["activationvec_no_proposal_fallback_humaneval124_256_v1.json", "activationvec_no_proposal_fallback_humaneval124_256_rep2_v1.json"],
}

def gmean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))

def bootstrap(values, seed=62311, draws=20000):
    rng = random.Random(seed)
    estimates = sorted(gmean([rng.choice(values) for _ in values]) for _ in range(draws))
    return [estimates[int(.025 * draws)], estimates[int(.975 * draws)]]

def main():
    prompts, hashes = [], {}
    totals = {"fallback_b1": 0, "activationvec_b1": 0, "fallback_bnb": 0, "activationvec_bnb": 0, "rounds": 0}
    for prompt_id, names in PROMPTS.items():
        sessions = []
        for name in names:
            path = BASE / name
            data = json.loads(path.read_text())
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            ex = data["exactness"]
            if not (ex["activationvec_tokens_match_singleton"] and ex["fallback_tokens_match_singleton"]
                    and ex["state_domains_equal"] and not ex["activationvec_state_mismatches"]
                    and not ex["fallback_state_mismatches"]):
                raise RuntimeError(f"B1 exactness failed: {name}")
            rounds = data["paired_rounds"]
            matches = {
                "fallback_b1": sum(r["methods"]["native_fallback"]["token_ids"] == r["methods"]["bnb_singleton"]["token_ids"] for r in rounds),
                "activationvec_b1": sum(r["methods"]["activationvec"]["token_ids"] == r["methods"]["bnb_singleton"]["token_ids"] for r in rounds),
                "fallback_bnb": sum(r["methods"]["native_fallback"]["token_ids"] == r["methods"]["original_bnb"]["token_ids"] for r in rounds),
                "activationvec_bnb": sum(r["methods"]["activationvec"]["token_ids"] == r["methods"]["original_bnb"]["token_ids"] for r in rounds),
            }
            if matches["fallback_b1"] != len(rounds) or matches["activationvec_b1"] != len(rounds):
                raise RuntimeError(f"timed B1 token mismatch: {name}")
            account = data["no_proposal_fallback_counts_preflight"]
            if account["no_proposal_cycles"] != account["fallback_forwards"]:
                raise RuntimeError(f"fallback accounting mismatch: {name}")
            totals["fallback_b1"] += matches["fallback_b1"]
            totals["activationvec_b1"] += matches["activationvec_b1"]
            totals["fallback_bnb"] += matches["fallback_bnb"]
            totals["activationvec_bnb"] += matches["activationvec_bnb"]
            totals["rounds"] += len(rounds)
            sessions.append({
                "artifact": name, "no_proposal_cycles": account["no_proposal_cycles"],
                "fallback_speedup_vs_activationvec": data["ratio_medians"]["fallback_over_activationvec"],
                "fallback_speedup_vs_original_bnb_latency": data["ratio_medians"]["fallback_over_original_bnb_latency"],
                "fallback_bnb_token_match_rounds": matches["fallback_bnb"],
                "activationvec_bnb_token_match_rounds": matches["activationvec_bnb"],
                "timing_rounds": len(rounds),
            })
        metrics = {key: gmean([s[key] for s in sessions]) for key in (
            "fallback_speedup_vs_activationvec", "fallback_speedup_vs_original_bnb_latency")}
        prompts.append({"prompt_id": prompt_id, "independent_processes": len(sessions),
                        "all_processes_exact_to_b1_singleton": True,
                        "prompt_metrics_geomean_across_processes": metrics, "sessions": sessions})
    across = {}
    for key in ("fallback_speedup_vs_activationvec", "fallback_speedup_vs_original_bnb_latency"):
        values = [p["prompt_metrics_geomean_across_processes"][key] for p in prompts]
        across[key] = {"geomean_across_prompts": gmean(values), "prompt_cluster_bootstrap_95pct": bootstrap(values),
                       "prompt_min": min(values), "prompt_max": max(values)}
    result = {
        "schema": "zassd-activationvec-no-proposal-fallback-humaneval-256-summary-v2",
        "classification": "six selected HumanEval prompt clusters, two independent processes each; descriptive prompt bootstrap",
        "model": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "runtime_scope": "RTX 4050 Laptop GPU, conda env ai, K=2/R=3, 256 generated tokens, four paired rounds/process",
        "prompt_count": len(prompts), "independent_processes": sum(len(v) for v in PROMPTS.values()),
        "paired_rounds": totals["rounds"], "all_runs_exact_to_b1_singleton": True,
        "fallback_b1_token_match_rounds": totals["fallback_b1"],
        "activationvec_b1_token_match_rounds": totals["activationvec_b1"],
        "fallback_original_bnb_token_match_rounds": totals["fallback_bnb"],
        "activationvec_original_bnb_token_match_rounds": totals["activationvec_bnb"],
        "prompts": prompts, "across_prompt_summary": across, "input_sha256": hashes,
        "heldout_used": False,
        "limits": ["Bootstrap resamples six selected prompt clusters, not a task population.",
                   "Single model revision and GPU; original-BNB comparison is token IDs and latency, not KV-state identity.",
                   "GSM8K held-out indices 300-349 excluded."],
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = BASE / "activationvec_no_proposal_fallback_humaneval_256_summary_v2.json"
    output.write_text(json.dumps(result, indent=2) + chr(10))
    print(json.dumps({"prompt_count": result["prompt_count"], "independent_processes": result["independent_processes"],
                      "paired_rounds": result["paired_rounds"], "all_runs_exact": result["all_runs_exact_to_b1_singleton"],
                      "fallback_bnb_token_matches": result["fallback_original_bnb_token_match_rounds"],
                      "across_prompt_summary": across, "output": str(output)}, indent=2))

if __name__ == "__main__":
    main()
