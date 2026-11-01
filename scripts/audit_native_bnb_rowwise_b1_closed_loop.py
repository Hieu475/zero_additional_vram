#!/usr/bin/env python3
"""Closed-loop K=2/R=3 block audit using BNB Linear4bit and native rowwise SDPA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from run_strong_baseline_comparison import run_vanilla_timed


MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
ATTENTION = "bnb_rowwise"
LINEAR = "fixed_rows"


def scheduled(model):
    return fixed_row_schedule_for_model(
        model, physical_rows=3, attention_strategy=ATTENTION,
        quantized_linear_strategy=LINEAR,
    )


def summarize_rounds(rounds: list[dict], method: str) -> dict:
    values = [r["methods"][method]["total_s"] for r in rounds]
    return {"samples_s": values,
            "median_s": statistics.median(values),
            "geomean_s": math.exp(sum(math.log(x) for x in values) / len(values))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/humaneval_ext114.jsonl")
    ap.add_argument("--prompt-id", default="HumanEval/112")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--timing-rounds", type=int, default=3)
    ap.add_argument("--session", type=int, default=0)
    ap.add_argument("--manifest", default="experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/experiment_manifest.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    source_name = "scripts/audit_native_bnb_rowwise_b1_closed_loop.py"
    if manifest.get("source_sha256", {}).get(source_name) != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("closed-loop runner is not pinned by the experiment manifest")
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)

    validator = repo / "scripts/validate_nf4_singleton_k2_r3_manifest.py"
    subprocess.run([sys.executable, str(validator), "--manifest", str(manifest_path)],
                   cwd=repo, check=True)
    rows = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in rows if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(f"prompt {args.prompt_id!r} not found in {args.benchmark}")
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    capacity = int(prompt_ids.shape[-1]) + args.max_new_tokens + 4
    if capacity > 2048:
        raise ValueError("prompt plus generation exceeds cache capacity limit")
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                  device="cuda:0", kv_cache_max_capacity=capacity,
                  physical_rows=3, attention_strategy=ATTENTION,
                  quantized_linear_strategy=LINEAR)

    ref_states, block_states = [], []
    ref_lengths, block_lengths = {}, {}

    def observe(store, lengths):
        last_prefix = {"n": -1}
        def callback(n: int, cache: TargetKVCache):
            lengths[n] = {"cache_seq_length": int(cache.get_seq_length()),
                          "layer_seq_lengths": [int(layer.get_seq_length()) for layer in cache.cache.layers]}
            start = last_prefix["n"] + 1
            for prefix in range(start, n + 1):
                store.append({"generated_prefix_tokens": prefix,
                              "sha256": diagnostic_cache_prefix_digest(
                                  cache, int(prompt_ids.shape[1]) + prefix
                              )})
            last_prefix["n"] = max(last_prefix["n"], n)
        return callback

    with scheduled(model), torch.inference_mode():
        _, ref_metrics = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **common,
            state_observer=observe(ref_states, ref_lengths),
        )
    with scheduled(model), torch.inference_mode():
        _, block_metrics = b1_fixed_row_block_generate(
            model, tokenizer, prompt_ids, **common,
            state_observer=observe(block_states, block_lengths), trace=True,
        )
    ref_map = {x["generated_prefix_tokens"]: x["sha256"] for x in ref_states}
    block_map = {x["generated_prefix_tokens"]: x["sha256"] for x in block_states}
    shared = sorted(set(ref_map) & set(block_map))
    state_mismatches = [n for n in shared if ref_map[n] != block_map[n]]
    length_frontiers = sorted(set(ref_lengths) & set(block_lengths))

    methods = ("original_bnb_greedy", "native_bnb_singleton", "native_bnb_block")
    paired = []
    for round_index in range(args.timing_rounds):
        order = list(methods[round_index % 3:] + methods[:round_index % 3])
        if args.session % 2:
            order.reverse()
        sample = {"round": round_index, "method_order": order, "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb_greedy":
                ids, prefill, decode, total = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0"
                )
                sample["methods"][method] = {"token_ids": ids, "total_s": total / 1000,
                                             "prefill_s": prefill / 1000, "decode_s": decode / 1000}
            else:
                decoder = (b1_fixed_row_generate if method.endswith("singleton")
                           else b1_fixed_row_block_generate)
                with scheduled(model), torch.inference_mode():
                    _, metrics = decoder(model, tokenizer, prompt_ids, **common)
                sample["methods"][method] = {
                    "token_ids": metrics.generated_token_ids,
                    "total_s": metrics.total_time_s,
                    "prefill_s": metrics.prefill_time_s,
                    "verify_s": getattr(metrics, "verify_time_s", None),
                    "accepted_draft_tokens": getattr(metrics, "accepted_draft_tokens", None),
                    "draft_tokens": getattr(metrics, "draft_tokens", None),
                }
        sample["block_matches_singleton"] = (
            sample["methods"]["native_bnb_block"]["token_ids"] ==
            sample["methods"]["native_bnb_singleton"]["token_ids"]
        )
        sample["block_matches_original_bnb"] = (
            sample["methods"]["native_bnb_block"]["token_ids"] ==
            sample["methods"]["original_bnb_greedy"]["token_ids"]
        )
        paired.append(sample)

    result = {
        "classification": "development-only native BNB operator schedule; finite exactness and paired latency audit",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "parameters": {"k": 2, "physical_rows": 3, "max_new_tokens": args.max_new_tokens,
                       "attention_strategy": ATTENTION, "quantized_linear_strategy": LINEAR,
                       "timing_rounds": args.timing_rounds, "session": args.session},
        "reference_and_candidate_schedule": "native BNB Linear4bit; Qwen2 bnb_rowwise native SDPA; rowwise RMSNorm and lm_head",
        "source_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                          for name in (source_name, "src/zassd/certification/qwen2_padded.py",
                                       "src/zassd/certification/fixed_row_dispatch.py",
                                       "src/zassd/decoding/b1_exact.py")},
        "exactness_audit": {
            "singleton_tokens": ref_metrics.generated_token_ids,
            "block_tokens": block_metrics.generated_token_ids,
            "token_match": ref_metrics.generated_token_ids == block_metrics.generated_token_ids,
            "shared_state_prefixes": len(shared), "state_mismatches": state_mismatches,
            "all_shared_states_equal": not state_mismatches,
            "same_prefix_domain": set(ref_map) == set(block_map),
            "logical_lengths_equal": all(ref_lengths[n] == block_lengths[n] for n in length_frontiers),
            "shared_commit_frontiers": len(length_frontiers),
            "acceptance": {"accepted": getattr(block_metrics, "accepted_draft_tokens", None),
                           "drafted": getattr(block_metrics, "draft_tokens", None)},
            "cycle_trace": block_metrics.cycle_trace,
        },
        "timing": {
            "paired_rounds": paired,
            "summaries": {method: summarize_rounds(paired, method) for method in methods},
            "block_speedup_vs_singleton": [r["methods"]["native_bnb_singleton"]["total_s"] /
                                            r["methods"]["native_bnb_block"]["total_s"] for r in paired],
            "block_speedup_vs_original_bnb": [r["methods"]["original_bnb_greedy"]["total_s"] /
                                               r["methods"]["native_bnb_block"]["total_s"] for r in paired],
        },
        "limitations": ["One development prompt/process unless repeated with new session IDs.",
                        "Finite empirical evidence, not a proof or cross-runtime guarantee.",
                        "No held-out GSM8K prompts used."],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(out), **result["exactness_audit"],
                      "timing": result["timing"]["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
