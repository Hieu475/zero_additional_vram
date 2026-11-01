#!/usr/bin/env python3
"""Summarize exactness-gated no-proposal fallback A/B runs by prompt cluster."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
PROMPTS = {
    "HumanEval/112": ["activationvec_no_proposal_fallback_humaneval112_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval112_64_rep2_v1.json"],
    "HumanEval/116": ["activationvec_no_proposal_fallback_humaneval116_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval116_64_rep2_v1.json"],
    "HumanEval/120": ["activationvec_no_proposal_fallback_humaneval120_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval120_64_rep2_v1.json"],
    "HumanEval/123": ["activationvec_no_proposal_fallback_humaneval123_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval123_64_rep2_v1.json"],
    "HumanEval/124": ["activationvec_no_proposal_fallback_humaneval124_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval124_64_rep2_v1.json"],
    "HumanEval/111": ["activationvec_no_proposal_fallback_humaneval111_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval111_64_rep2_v1.json"],
    "HumanEval/113": ["activationvec_no_proposal_fallback_humaneval113_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval113_64_rep2_v1.json"],
    "HumanEval/114": ["activationvec_no_proposal_fallback_humaneval114_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval114_64_rep2_v1.json"],
    "HumanEval/117": ["activationvec_no_proposal_fallback_humaneval117_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval117_64_rep2_v1.json"],
    "HumanEval/118": ["activationvec_no_proposal_fallback_humaneval118_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval118_64_rep2_v1.json"],
    "HumanEval/121": ["activationvec_no_proposal_fallback_humaneval121_64_v1.json",
                      "activationvec_no_proposal_fallback_humaneval121_64_rep2_v1.json"],
}
METRICS = (
    "fallback_over_activationvec",
    "fallback_over_bnb_singleton",
    "fallback_over_original_bnb_latency",
    "activationvec_over_bnb_singleton",
)


def gmean(values: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in values) / len(values))


def bootstrap(values: list[float], seed: int = 61522, draws: int = 20000) -> list[float]:
    rng = random.Random(seed)
    estimates = sorted(gmean([rng.choice(values) for _ in values]) for _ in range(draws))
    return [estimates[int(0.025 * draws)], estimates[int(0.975 * draws)]]


def main() -> None:
    prompt_rows = []
    hashes = {}
    for prompt_id, artifacts in PROMPTS.items():
        sessions = []
        for artifact_name in artifacts:
            path = BASE / artifact_name
            data = json.loads(path.read_text())
            hashes[artifact_name] = hashlib.sha256(path.read_bytes()).hexdigest()
            ex = data["exactness"]
            if not (ex["activationvec_tokens_match_singleton"]
                    and ex["fallback_tokens_match_singleton"]
                    and ex["state_domains_equal"]
                    and not ex["activationvec_state_mismatches"]
                    and not ex["fallback_state_mismatches"]
                    and data["no_proposal_fallback_counts_preflight"]["no_proposal_cycles"]
                    == data["no_proposal_fallback_counts_preflight"]["fallback_forwards"]):
                raise RuntimeError(f"exactness/fallback accounting failed: {artifact_name}")
            fallback_b1_match_rounds = sum(
                row["methods"]["native_fallback"]["token_ids"]
                == row["methods"]["bnb_singleton"]["token_ids"]
                for row in data["paired_rounds"])
            activationvec_b1_match_rounds = sum(
                row["methods"]["activationvec"]["token_ids"]
                == row["methods"]["bnb_singleton"]["token_ids"]
                for row in data["paired_rounds"])
            if fallback_b1_match_rounds != len(data["paired_rounds"]):
                raise RuntimeError(f"timed fallback token mismatch: {artifact_name}")
            if activationvec_b1_match_rounds != len(data["paired_rounds"]):
                raise RuntimeError(f"timed activationvec token mismatch: {artifact_name}")
            sessions.append({
                "artifact": artifact_name,
                "no_proposal_cycles": data["no_proposal_fallback_counts_preflight"]["no_proposal_cycles"],
                "drafted": data["acceptance"]["fallback"]["drafted"],
                "accepted": data["acceptance"]["fallback"]["accepted"],
                "fallback_token_match_original_bnb_rounds": sum(
                    row["methods"]["native_fallback"]["token_ids"]
                    == row["methods"]["original_bnb"]["token_ids"]
                    for row in data["paired_rounds"]),
                "activationvec_token_match_original_bnb_rounds": sum(
                    row["methods"]["activationvec"]["token_ids"]
                    == row["methods"]["original_bnb"]["token_ids"]
                    for row in data["paired_rounds"]),
                "token_match_rounds": len(data["paired_rounds"]),
                "fallback_b1_token_match_timing_rounds": fallback_b1_match_rounds,
                "activationvec_b1_token_match_timing_rounds": activationvec_b1_match_rounds,
                "speedups": data["ratio_medians"],
            })
        metrics = {name: gmean([session["speedups"][name] for session in sessions])
                   for name in METRICS}
        prompt_rows.append({
            "prompt_id": prompt_id,
            "independent_processes": len(sessions),
            "all_processes_exact_to_b1_singleton": True,
            "no_proposal_cycles_per_process": [s["no_proposal_cycles"] for s in sessions],
            "prompt_metrics_geomean_across_processes": metrics,
            "sessions": sessions,
        })
    summary = {}
    for metric in METRICS:
        values = [row["prompt_metrics_geomean_across_processes"][metric] for row in prompt_rows]
        summary[metric] = {
            "geomean_across_prompts": gmean(values),
            "prompt_cluster_bootstrap_95pct": bootstrap(values),
            "prompt_min": min(values),
            "prompt_max": max(values),
        }
    result = {
        "schema": "zassd-activationvec-no-proposal-fallback-summary-v3",
        "classification": "eleven selected HumanEval prompt clusters, two independent processes per prompt; descriptive bootstrap",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "runtime_scope": "RTX 4050 Laptop GPU, conda env ai, K=2/R=3, 64 output tokens, 4 timing rounds/process",
        "prompt_count": len(prompt_rows),
        "independent_processes": sum(len(x) for x in PROMPTS.values()),
        "all_runs_exact_to_b1_singleton": True,
        "fallback_original_bnb_token_match_rounds": sum(
            session["fallback_token_match_original_bnb_rounds"]
            for row in prompt_rows for session in row["sessions"]),
        "activationvec_original_bnb_token_match_rounds": sum(
            session["activationvec_token_match_original_bnb_rounds"]
            for row in prompt_rows for session in row["sessions"]),
        "original_bnb_token_match_rounds_total": sum(
            session["token_match_rounds"] for row in prompt_rows for session in row["sessions"]),
        "fallback_b1_token_match_timing_rounds": sum(
            session["fallback_b1_token_match_timing_rounds"]
            for row in prompt_rows for session in row["sessions"]),
        "activationvec_b1_token_match_timing_rounds": sum(
            session["activationvec_b1_token_match_timing_rounds"]
            for row in prompt_rows for session in row["sessions"]),
        "fallback": "Use unpatched native Qwen2/BNB M=1 operators only for target transitions with an empty draft proposal list.",
        "prompts": prompt_rows,
        "across_prompt_summary": summary,
        "input_sha256": hashes,
        "heldout_used": False,
        "limits": [
            "Prompt-cluster bootstrap resamples five selected prompts, not GPU sessions or a task population.",
            "Latency versus original BNB is reported separately and does not establish original-BNB state equivalence.",
            "One GPU and one model revision; GSM8K held-out indices 300-349 excluded.",
        ],
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    out = BASE / "activationvec_no_proposal_fallback_summary_v3.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"prompt_count": result["prompt_count"],
                      "independent_processes": result["independent_processes"],
                      "all_exact": result["all_runs_exact_to_b1_singleton"],
                      "summary": summary, "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
