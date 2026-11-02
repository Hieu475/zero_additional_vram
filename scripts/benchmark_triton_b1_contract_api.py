#!/usr/bin/env python3
"""Paired B1/B1-block/BNB timings through the live-verified R=2 contract API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_triton_b1_closed_loop import (
    diagnostic_cache_prefix_digest,
    fixed_row_sdpa_decode,
)
from scripts._common import run_vanilla_timed
from zassd.backends.base import load_backend
from zassd.certification.triton_b1_reference_contract import (
    ATTENTION_STRATEGY,
    LINEAR_STRATEGY,
    MODEL_ID,
    MODEL_REVISION,
    verify_triton_b1_reference_contract,
)
from zassd.decoding.b1_triton_contract import (
    b1_triton_contract_block,
    b1_triton_contract_singleton,
)


def _prompt(path: Path, prompt_id: str) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = next((row for row in rows if str(row["id"]) == prompt_id), None)
    if result is None:
        raise ValueError(f"prompt {prompt_id!r} not found in {path}")
    return result


def _audit_state(model, tokenizer, prompt_ids, verified, budget: int) -> dict:
    device = "cuda:0"
    capacity = int(prompt_ids.shape[1]) + budget + 4
    reference, block = {}, {}
    reference_lengths, block_lengths = {}, {}

    def observer(store, lengths):
        def observe(n, cache):
            prefix_len = int(prompt_ids.shape[1]) + n
            store[n] = diagnostic_cache_prefix_digest(cache, prefix_len)
            lengths[n] = [int(layer.get_seq_length()) for layer in cache.cache.layers]
        return observe

    with fixed_row_sdpa_decode(model, 2, 32, ATTENTION_STRATEGY):
        _, ref_metrics = b1_triton_contract_singleton(
            model, tokenizer, prompt_ids, verified_contract=verified,
            max_new_tokens=budget, device=device, kv_cache_max_capacity=capacity,
            state_observer=observer(reference, reference_lengths),
        )
    with fixed_row_sdpa_decode(model, 2, 32, ATTENTION_STRATEGY):
        _, block_metrics = b1_triton_contract_block(
            model, tokenizer, prompt_ids, verified_contract=verified,
            max_new_tokens=budget, device=device, kv_cache_max_capacity=capacity,
            trace=True, state_observer=observer(block, block_lengths),
        )
    shared = sorted(set(reference) & set(block))
    return {
        "reference_token_ids": ref_metrics.generated_token_ids,
        "block_token_ids": block_metrics.generated_token_ids,
        "token_match": ref_metrics.generated_token_ids == block_metrics.generated_token_ids,
        "reference_prefix_count": len(reference),
        "block_prefix_count": len(block),
        "shared_prefix_count": len(shared),
        "same_prefix_domain": reference.keys() == block.keys(),
        "all_shared_kv_exact": all(reference[i] == block[i] for i in shared),
        "logical_lengths_exact": all(reference_lengths[i] == block_lengths[i] for i in shared),
        "block_cycles": block_metrics.cycle_trace,
    }


def _run_timed(method: str, model, tokenizer, prompt_ids, verified, budget: int,
               capacity: int) -> dict:
    if method == "original_bnb_greedy":
        tokens, prefill_ms, decode_ms, total_ms = run_vanilla_timed(
            model, tokenizer, prompt_ids, budget, "cuda:0"
        )
        return {"token_ids": tokens, "prefill_s": prefill_ms / 1000,
                "decode_s": decode_ms / 1000, "total_s": total_ms / 1000}
    with fixed_row_sdpa_decode(model, 2, 32, ATTENTION_STRATEGY):
        if method == "b1_contract_singleton":
            _, metrics = b1_triton_contract_singleton(
                model, tokenizer, prompt_ids, verified_contract=verified,
                max_new_tokens=budget, device="cuda:0",
                kv_cache_max_capacity=capacity,
            )
        else:
            _, metrics = b1_triton_contract_block(
                model, tokenizer, prompt_ids, verified_contract=verified,
                max_new_tokens=budget, device="cuda:0",
                kv_cache_max_capacity=capacity,
            )
    return {"token_ids": metrics.generated_token_ids,
            "prefill_s": metrics.prefill_time_s,
            "total_s": metrics.total_time_s,
            "target_forwards": metrics.target_forwards,
            "target_rows": metrics.target_rows,
            "accepted_draft_tokens": metrics.accepted_draft_tokens,
            "draft_tokens": metrics.draft_tokens}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("data/benchmarks/cnndm_eval.jsonl"))
    parser.add_argument("--prompt-id", action="append", default=None)
    parser.add_argument("--max-new-tokens", type=int, choices=(64, 128, 256, 512), default=128)
    parser.add_argument("--timing-rounds", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    prompt_ids_to_run = args.prompt_id or ["cnndm_000", "cnndm_001", "cnndm_002"]
    if args.timing_rounds < 1:
        raise ValueError("timing_rounds must be >= 1")

    loaded = load_backend("hf_bnb_nf4", MODEL_ID, MODEL_REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    verified = verify_triton_b1_reference_contract(
        model, tokenizer, args.contract, device="cuda:0"
    )
    prompt_results = []
    method_names = ["original_bnb_greedy", "b1_contract_singleton", "b1_contract_block"]
    all_ratios_b1, all_ratios_bnb = [], []
    for prompt_id in prompt_ids_to_run:
        row = _prompt(args.benchmark, prompt_id)
        prompt_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
        capacity = int(prompt_ids.shape[1]) + args.max_new_tokens + 4
        if capacity > 2048:
            raise ValueError(f"prompt {prompt_id} exceeds static cache capacity")
        audit = _audit_state(model, tokenizer, prompt_ids, verified, args.max_new_tokens)

        # Warm all paths at a short budget before timing.
        warm_capacity = int(prompt_ids.shape[1]) + 12
        for method in method_names:
            _run_timed(method, model, tokenizer, prompt_ids, verified, 8, warm_capacity)

        timing_rounds = []
        for round_index in range(args.timing_rounds):
            order = method_names[round_index % len(method_names):] + method_names[:round_index % len(method_names)]
            methods = {}
            for method in order:
                methods[method] = _run_timed(
                    method, model, tokenizer, prompt_ids, verified,
                    args.max_new_tokens, capacity,
                )
            sample = {"round": round_index, "method_order": order, "methods": methods}
            sample["block_matches_b1"] = (
                methods["b1_contract_block"]["token_ids"] ==
                methods["b1_contract_singleton"]["token_ids"]
            )
            sample["block_matches_bnb"] = (
                methods["b1_contract_block"]["token_ids"] ==
                methods["original_bnb_greedy"]["token_ids"]
            )
            ratio_b1 = (methods["b1_contract_singleton"]["total_s"] /
                        methods["b1_contract_block"]["total_s"])
            ratio_bnb = (methods["original_bnb_greedy"]["total_s"] /
                         methods["b1_contract_block"]["total_s"])
            sample["speedup_vs_b1"] = ratio_b1
            sample["speedup_vs_bnb"] = ratio_bnb
            all_ratios_b1.append(ratio_b1)
            all_ratios_bnb.append(ratio_bnb)
            timing_rounds.append(sample)
        prompt_results.append({
            "prompt_id": prompt_id,
            "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
            "input_ids_sha256": hashlib.sha256(json.dumps(prompt_ids[0].tolist()).encode()).hexdigest(),
            "audit": audit,
            "timing_rounds": timing_rounds,
        })

    result = {
        "experiment": "triton_b1_contract_api_paired_performance",
        "scope": "development-only; B1 exactness under contract; BNB comparison is separate",
        "contract_id": verified.contract_id,
        "contract_path": str(args.contract),
        "live_preflight": {"status": "pass", "model_state_sha256": verified.model_state_sha256,
                           "verified_source_count": verified.verified_source_count},
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "parameters": {
            "k": 1, "ngram_size": 3, "physical_rows": 2,
            "attention_strategy": ATTENTION_STRATEGY,
            "quantized_linear_strategy": LINEAR_STRATEGY,
            "max_new_tokens": args.max_new_tokens,
            "timing_rounds_per_prompt": args.timing_rounds,
        },
        "benchmark": str(args.benchmark),
        "paired_geomean_speedup_vs_b1": math.exp(sum(math.log(x) for x in all_ratios_b1) / len(all_ratios_b1)),
        "paired_geomean_speedup_vs_bnb": math.exp(sum(math.log(x) for x in all_ratios_bnb) / len(all_ratios_bnb)),
        "paired_speedup_range_vs_b1": [min(all_ratios_b1), max(all_ratios_b1)],
        "paired_speedup_range_vs_bnb": [min(all_ratios_bnb), max(all_ratios_bnb)],
        "bnb_token_match_rows": sum(
            int(sample["block_matches_bnb"])
            for prompt in prompt_results for sample in prompt["timing_rounds"]
        ),
        "timing_pair_count": len(all_ratios_b1),
        "prompts": prompt_results,
        "timing_note": "Paired observations on a small fixed development cohort; no confidence interval claimed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "contract_id": result["contract_id"],
        "timing_pairs": result["timing_pair_count"],
        "exact_b1": all(p["audit"]["token_match"] and p["audit"]["all_shared_kv_exact"]
                        and p["audit"]["logical_lengths_exact"] for p in prompt_results),
        "geomean_vs_b1": result["paired_geomean_speedup_vs_b1"],
        "geomean_vs_bnb": result["paired_geomean_speedup_vs_bnb"],
        "bnb_token_match_rows": result["bnb_token_match_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
