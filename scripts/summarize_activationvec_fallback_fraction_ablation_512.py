#!/usr/bin/env python3
"""Summarize exactness-gated fallback-fraction ablation runs."""
from __future__ import annotations
import hashlib, json, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
PROMPTS = {
    "HumanEval/112": ["fallback_fraction_ablation_humaneval112_512_v1.json"],
    "HumanEval/114": ["fallback_fraction_ablation_humaneval114_512_v1.json"],
    "HumanEval/121": ["fallback_fraction_ablation_humaneval121_512_v1.json"],
    "HumanEval/123": ["fallback_fraction_ablation_humaneval123_512_v1.json",
                      "fallback_fraction_ablation_humaneval123_512_rep2_v1.json"],
    "cnndm_004": ["fallback_fraction_ablation_cnndm004_512_v1.json"],
    "cnndm_003": ["fallback_fraction_ablation_cnndm003_512_v1.json",
                  "fallback_fraction_ablation_cnndm003_512_rep2_v1.json"],
}
METRICS = {
    "half_fallback_vs_activationvec": "fallback_every_other_over_activationvec_latency",
    "full_fallback_vs_activationvec": "fallback_all_over_activationvec_latency",
    "full_fallback_vs_half_fallback": "fallback_all_over_fallback_every_other_latency",
}

def gmean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))

def bootstrap(values, seed=77341, draws=20000):
    rng = random.Random(seed)
    estimates = sorted(gmean([rng.choice(values) for _ in values]) for _ in range(draws))
    return [estimates[int(.025 * draws)], estimates[int(.975 * draws)]]

def main():
    prompts, hashes = [], {}
    for prompt_id, names in PROMPTS.items():
        sessions = []
        for name in names:
            path = BASE / name
            data = json.loads(path.read_text())
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if data.get("schema") != "zassd-fallback-fraction-ablation-v1":
                raise RuntimeError(f"unexpected schema or exactness failure: {name}")
            if not (data["exactness"]["all_modes_match_singleton_tokens"]
                    and data["exactness"]["state_domains_equal"]
                    and not any(data["exactness"]["state_mismatches"].values())
                    and all(data["timed_token_match_singleton"].values())):
                raise RuntimeError(f"B1 identity check failed: {name}")
            controls = data["preflight_controls"]
            cycles = controls["fallback_all"]["no_proposal_cycles"]
            if any(controls[mode]["no_proposal_cycles"] != cycles for mode in controls):
                raise RuntimeError(f"proposal schedules diverged: {name}")
            if controls["fallback_every_other"]["fallback_forwards"] != cycles // 2:
                raise RuntimeError(f"half-fallback accounting failed: {name}")
            sessions.append({
                "artifact": name,
                "no_proposal_cycles": cycles,
                "fallback_forwards": {
                    mode: controls[mode]["fallback_forwards"] for mode in controls
                },
                "ratio_medians": data["ratio_medians"],
                "timing_rounds": data["timing_rounds"],
            })
        prompt_metrics = {
            key: gmean([session["ratio_medians"][ratio_key] for session in sessions])
            for key, ratio_key in METRICS.items()
        }
        prompts.append({"prompt_id": prompt_id, "independent_processes": len(sessions),
                        "prompt_geomean_speedups": prompt_metrics, "sessions": sessions})
    aggregate = {}
    for key in METRICS:
        values = [prompt["prompt_geomean_speedups"][key] for prompt in prompts]
        aggregate[key] = {"geomean_across_prompts": gmean(values),
                          "prompt_cluster_bootstrap_95pct": bootstrap(values),
                          "prompt_min": min(values), "prompt_max": max(values)}
    result = {
        "schema": "zassd-fallback-fraction-ablation-summary-512-v1",
        "classification": "six selected development prompt clusters; two processes each on two clusters and one process each on four clusters; exploratory paired ablation",
        "runtime_scope": "Qwen/Qwen2.5-3B-Instruct NF4, RTX 4050 Laptop GPU, K=2/R=3, 512 tokens, four paired timing rounds/process",
        "modes": {"activationvec": "0% fallback", "fallback_every_other": "50% of empty-proposal cycles",
                  "fallback_all": "100% of empty-proposal cycles"},
        "prompt_count": len(prompts), "independent_processes": sum(len(v) for v in PROMPTS.values()),
        "all_runs_exact_to_b1_singleton": True,
        "timed_token_match_singleton": True,
        "prompts": prompts, "aggregate": aggregate, "input_sha256": hashes,
        "heldout_used": False,
        "limits": ["Only six selected prompt clusters on one model and one GPU; four clusters have a single process.",
                   "Prompt-cluster bootstrap is descriptive and does not establish population performance.",
                   "Original-BNB comparison omitted; this ablation isolates fallback schedule contribution.",
                   "GSM8K held-out indices 300-349 excluded."],
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = BASE / "fallback_fraction_ablation_512_summary_v1.json"
    output.write_text(json.dumps(result, indent=2) + chr(10))
    print(json.dumps({"prompt_count": result["prompt_count"],
                      "independent_processes": result["independent_processes"],
                      "aggregate": aggregate, "output": str(output)}, indent=2))

if __name__ == "__main__":
    main()
