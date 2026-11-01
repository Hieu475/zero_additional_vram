#!/usr/bin/env python3
"""Same-process A/B between standard and fused K2/R3 block verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.b1_exact import b1_fixed_row_block_generate
from zassd.certification.triton_b1_reference_contract import verify_triton_b1_reference_contract
from audit_nf4_fused_gate_up_k2_r3 import (
    BASE_CONTRACT,
    MODEL,
    REVISION,
    STRATEGY,
    run_block as run_fused_block,
    run_reference,
)
from audit_nf4_singleton_k2_r3_closed_loop import singleton_kernel_schedule
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest, fixed_row_sdpa_decode
from run_strong_baseline_comparison import run_vanilla_timed


def run_standard_block(model, tokenizer, prompt_ids, common, observer):
    with singleton_kernel_schedule():
        with fixed_row_sdpa_decode(model, 3, 32, "triton_batched"), torch.inference_mode():
            _, metrics = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common, state_observer=observer
            )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/humaneval_ext114.jsonl")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--timing-rounds", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--enable-experimental-fusion", action="store_true", required=True)
    args = parser.parse_args()
    validator = Path(__file__).resolve().with_name("validate_nf4_singleton_k2_r3_manifest.py")
    checked = subprocess.run(
        [sys.executable, str(validator), "--manifest", str(args.manifest)],
        cwd=Path(__file__).resolve().parents[1], check=False,
    )
    if checked.returncode:
        raise SystemExit(checked.returncode)
    if args.output.exists():
        raise FileExistsError(args.output)
    prompt = next(
        (row for line in Path(args.benchmark).read_text().splitlines() if line.strip()
         for row in [json.loads(line)] if str(row["id"]) == args.prompt_id),
        None,
    )
    if prompt is None:
        raise ValueError(f"prompt {args.prompt_id!r} missing from {args.benchmark}")

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    identity = verify_triton_b1_reference_contract(model, tokenizer, BASE_CONTRACT, device="cuda:0")
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    common = {
        "k": 2, "ngram_size": 3, "max_new_tokens": args.max_new_tokens,
        "device": "cuda:0", "kv_cache_max_capacity": int(prompt_ids.shape[1]) + args.max_new_tokens + 4,
        "physical_rows": 3, "attention_strategy": "triton_batched",
        "quantized_linear_strategy": STRATEGY,
    }
    singleton_states, standard_states, fused_states = {}, {}, {}

    def singleton_observer_for(target):
        def observe(n, cache: TargetKVCache):
            target[n] = diagnostic_cache_prefix_digest(
                cache, int(prompt_ids.shape[1]) + n
            )
        return observe

    def block_observer_for(target):
        def observe(n, cache: TargetKVCache):
            first_missing = max(target, default=-1) + 1
            for prefix in range(first_missing, n + 1):
                target[prefix] = diagnostic_cache_prefix_digest(
                    cache, int(prompt_ids.shape[1]) + prefix
                )
        return observe

    singleton_metrics, singleton_calls = run_reference(
        model, tokenizer, prompt_ids, common, singleton_observer_for(singleton_states)
    )
    standard_metrics = run_standard_block(
        model, tokenizer, prompt_ids, common, block_observer_for(standard_states)
    )
    fused_metrics, fused_calls = run_fused_block(
        model, tokenizer, prompt_ids, common, block_observer_for(fused_states)
    )
    exact_to_singleton = (
        singleton_metrics.generated_token_ids == standard_metrics.generated_token_ids
        == fused_metrics.generated_token_ids
        and singleton_states == standard_states == fused_states
    )
    standard_equals_fused = (
        standard_metrics.generated_token_ids == fused_metrics.generated_token_ids
        and standard_states == fused_states
    )
    singleton_token_match = (
        singleton_metrics.generated_token_ids == standard_metrics.generated_token_ids
        == fused_metrics.generated_token_ids
    )
    state_prefix_mismatches = [
        prefix for prefix in sorted(set(singleton_states) | set(standard_states) | set(fused_states))
        if not (singleton_states.get(prefix) == standard_states.get(prefix) == fused_states.get(prefix))
    ]

    methods = ("original_bnb_greedy", "standard_k2_block", "fused_k2_block")
    for method in methods:
        if method == "original_bnb_greedy":
            run_vanilla_timed(model, tokenizer, prompt_ids, 8, "cuda:0")
        elif method == "standard_k2_block":
            run_standard_block(model, tokenizer, prompt_ids, {**common, "max_new_tokens": 8}, lambda *_: None)
        else:
            run_fused_block(model, tokenizer, prompt_ids, {**common, "max_new_tokens": 8}, lambda *_: None)

    rounds = []
    for index in range(args.timing_rounds):
        shift = index % len(methods)
        order = list(methods[shift:] + methods[:shift])
        if args.session % 2:
            order.reverse()
        row = {"round": index, "method_order": order, "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb_greedy":
                ids, prefill, decode, total = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0"
                )
                measured = {"token_ids": ids, "total_s": total / 1000,
                            "prefill_s": prefill / 1000, "decode_s": decode / 1000}
            else:
                fn = run_standard_block if method == "standard_k2_block" else run_fused_block
                metrics = fn(model, tokenizer, prompt_ids, common, lambda *_: None)
                if isinstance(metrics, tuple):
                    metrics = metrics[0]
                measured = {
                    "token_ids": metrics.generated_token_ids,
                    "total_s": metrics.total_time_s,
                    "prefill_s": metrics.prefill_time_s,
                    "verify_s": metrics.verify_time_s,
                    "accepted_draft_tokens": metrics.accepted_draft_tokens,
                    "draft_tokens": metrics.draft_tokens,
                }
            torch.cuda.synchronize("cuda:0")
            row["methods"][method] = measured
        rounds.append(row)

    standard_over_fused = [
        r["methods"]["standard_k2_block"]["total_s"] /
        r["methods"]["fused_k2_block"]["total_s"] for r in rounds
    ]
    fused_over_bnb = [
        r["methods"]["original_bnb_greedy"]["total_s"] /
        r["methods"]["fused_k2_block"]["total_s"] for r in rounds
    ]
    payload = {
        "schema": "zassd-fused-gate-up-paired-k2-r3-v1",
        "status": "same-process A/B diagnostic; fused implementation remains experimental",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "input_prompt_tokens": int(prompt_ids.shape[1]),
        "session": args.session,
        "base_b1_identity": {"contract_id": identity.contract_id,
                              "model_state_sha256": identity.model_state_sha256},
        "parameters": {"k": 2, "physical_rows": 3,
                       "max_new_tokens": args.max_new_tokens,
                       "attention": "triton_batched", "quantized_linear": STRATEGY,
                       "fusion": "one launch dual-output gate/up"},
        "exact_to_singleton_all_tokens_and_state": exact_to_singleton,
        "singleton_token_match": singleton_token_match,
        "standard_and_fused_blocks_equal_tokens_and_state": standard_equals_fused,
        "singleton_token_count": len(singleton_metrics.generated_token_ids),
        "shared_state_prefixes": len(set(singleton_states) & set(fused_states)),
        "state_prefix_domains_equal": set(singleton_states) == set(standard_states) == set(fused_states),
        "state_prefix_mismatches": state_prefix_mismatches[:20],
        "fused_mlp_invocations": {"singleton": singleton_calls, "block": fused_calls},
        "timing": {
            "paired_rounds": rounds,
            "geomean_speedup_fused_vs_standard_block": math.exp(
                sum(math.log(value) for value in standard_over_fused) / len(standard_over_fused)
            ),
            "geomean_speedup_fused_vs_bnb": math.exp(
                sum(math.log(value) for value in fused_over_bnb) / len(fused_over_bnb)
            ),
        },
        "limitations": [
            "Finite development prompt and one GPU; no broad performance claim.",
            "Correctness is relative to the pinned experimental singleton schedule and base model identity.",
            "No original BNB KV-state equality claim.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "exact_to_singleton": exact_to_singleton,
        "standard_equals_fused": standard_equals_fused,
        "speedup_fused_vs_standard": payload["timing"]["geomean_speedup_fused_vs_standard_block"],
        "speedup_fused_vs_bnb": payload["timing"]["geomean_speedup_fused_vs_bnb"],
    }, indent=2))


if __name__ == "__main__":
    main()
