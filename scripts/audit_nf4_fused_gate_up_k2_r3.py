#!/usr/bin/env python3
"""Closed-loop K2/R3 screen with a fused gate/up NF4 MLP prototype."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
from bitsandbytes.nn import Linear4bit

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from zassd.certification.triton_b1_reference_contract import verify_triton_b1_reference_contract
from benchmark_nf4_fused_gate_up_m3 import fused_gate_up
from audit_nf4_singleton_k2_r3_closed_loop import singleton_kernel_schedule
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest, fixed_row_sdpa_decode
from _common import run_vanilla_timed


MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
STRATEGY = "triton_nf4_group_scale_packed_rows2_n16_warp1_k256"
BASE_CONTRACT = Path(
    "experiments/exact_quantized_feasibility/b1_triton_batched_k1r2_n16_reference_contract_v5.json"
)


@contextmanager
def fused_qwen_mlp(model, physical_rows: int = 3):
    """Fuse gate/up for decode rows while leaving long prompt prefill on BNB."""
    saved = []
    counters = {"fused_calls": 0, "prefill_fallback_calls": 0}
    try:
        for transformer_layer in model.model.layers:
            mlp = transformer_layer.mlp
            original = mlp.forward

            def fused_forward(self, hidden_states, _original=original):
                if (hidden_states.ndim != 3 or hidden_states.shape[0] != 1
                        or hidden_states.shape[1] > physical_rows):
                    counters["prefill_fallback_calls"] += 1
                    return _original(hidden_states)
                counters["fused_calls"] += 1
                gate, up = fused_gate_up(
                    hidden_states, self.gate_proj, self.up_proj,
                    block_n=16, block_k=256, num_warps=1,
                )
                return self.down_proj(self.act_fn(gate) * up)

            saved.append((mlp, original))
            mlp.forward = MethodType(fused_forward, mlp)
        yield counters
    finally:
        for mlp, original in reversed(saved):
            mlp.forward = original


def run_reference(model, tokenizer, prompt_ids, common, observer):
    with singleton_kernel_schedule():
        with fixed_row_sdpa_decode(model, 3, 32, "triton_batched"), torch.inference_mode():
            with fused_qwen_mlp(model) as counters:
                _, metrics = b1_fixed_row_generate(
                    model, tokenizer, prompt_ids, **common, state_observer=observer
                )
    return metrics, counters


def run_block(model, tokenizer, prompt_ids, common, observer):
    with singleton_kernel_schedule():
        with fixed_row_sdpa_decode(model, 3, 32, "triton_batched"), torch.inference_mode():
            with fused_qwen_mlp(model) as counters:
                _, metrics = b1_fixed_row_block_generate(
                    model, tokenizer, prompt_ids, **common, state_observer=observer
                )
    return metrics, counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/humaneval_ext114.jsonl")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--timing-rounds", type=int, default=2)
    parser.add_argument("--baseline-k2-artifact", type=Path)
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
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in records if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt {args.prompt_id!r} absent from {args.benchmark}")

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    base_identity = verify_triton_b1_reference_contract(
        model, tokenizer, BASE_CONTRACT, device="cuda:0"
    )
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    common = {
        "k": 2, "ngram_size": 3, "max_new_tokens": args.max_new_tokens,
        "device": "cuda:0", "kv_cache_max_capacity": int(prompt_ids.shape[1]) + args.max_new_tokens + 4,
        "physical_rows": 3, "attention_strategy": "triton_batched",
        "quantized_linear_strategy": STRATEGY,
    }
    ref_states, block_states = {}, {}

    def ref_observer(n, cache):
        ref_states[n] = diagnostic_cache_prefix_digest(cache, int(prompt_ids.shape[1]) + n)

    def block_observer(n, cache):
        start = max(block_states, default=-1) + 1
        for prefix in range(start, n + 1):
            block_states[prefix] = diagnostic_cache_prefix_digest(
                cache, int(prompt_ids.shape[1]) + prefix
            )

    # Correctness/reference capture is run before timed methods.
    ref_metrics, ref_calls = run_reference(model, tokenizer, prompt_ids, common, ref_observer)
    block_metrics, block_calls = run_block(model, tokenizer, prompt_ids, common, block_observer)
    shared_prefixes = sorted(set(ref_states) & set(block_states))
    state_equal = ref_states == block_states
    prior_baseline_match = None
    if args.baseline_k2_artifact:
        prior = json.loads(args.baseline_k2_artifact.read_text())
        prior_baseline_match = ref_metrics.generated_token_ids == prior["reference"]["token_ids"]

    methods = ("original_bnb_greedy", "fused_b1_singleton", "fused_b1_block")
    for method in methods:
        if method == "original_bnb_greedy":
            run_vanilla_timed(model, tokenizer, prompt_ids, 8, "cuda:0")
        elif method == "fused_b1_singleton":
            run_reference(model, tokenizer, prompt_ids, {**common, "max_new_tokens": 8}, lambda *_: None)
        else:
            run_block(model, tokenizer, prompt_ids, {**common, "max_new_tokens": 8}, lambda *_: None)

    timing = []
    for round_index in range(args.timing_rounds):
        order = list(methods[round_index % len(methods):] + methods[:round_index % len(methods)])
        if args.session % 2:
            order.reverse()
        result = {"round": round_index, "method_order": order, "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb_greedy":
                ids, prefill_ms, decode_ms, total_ms = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0"
                )
                value = {"token_ids": ids, "total_s": total_ms / 1000,
                         "prefill_s": prefill_ms / 1000, "decode_s": decode_ms / 1000}
            elif method == "fused_b1_singleton":
                metrics, _ = run_reference(model, tokenizer, prompt_ids, common, lambda *_: None)
                value = {"token_ids": metrics.generated_token_ids, "total_s": metrics.total_time_s,
                         "prefill_s": metrics.prefill_time_s, "verify_s": metrics.verify_time_s,
                         "accepted_draft_tokens": metrics.accepted_draft_tokens,
                         "draft_tokens": metrics.draft_tokens}
            else:
                metrics, _ = run_block(model, tokenizer, prompt_ids, common, lambda *_: None)
                value = {"token_ids": metrics.generated_token_ids, "total_s": metrics.total_time_s,
                         "prefill_s": metrics.prefill_time_s, "verify_s": metrics.verify_time_s,
                         "accepted_draft_tokens": metrics.accepted_draft_tokens,
                         "draft_tokens": metrics.draft_tokens}
            torch.cuda.synchronize("cuda:0")
            result["methods"][method] = value
        timing.append(result)

    b1_ratios = [r["methods"]["fused_b1_singleton"]["total_s"] /
                 r["methods"]["fused_b1_block"]["total_s"] for r in timing]
    bnb_ratios = [r["methods"]["original_bnb_greedy"]["total_s"] /
                  r["methods"]["fused_b1_block"]["total_s"] for r in timing]
    payload = {
        "schema": "zassd-nf4-fused-gate-up-k2-r3-closed-loop-v1",
        "status": "experimental; fused MLP kept outside the pinned runtime",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "prompt_sha256": __import__("hashlib").sha256(prompt["prompt"].encode()).hexdigest(),
        "input_prompt_tokens": int(prompt_ids.shape[1]),
        "session": args.session,
        "experiment_manifest": str(args.manifest),
        "base_b1_reference_contract": {
            "contract_id": base_identity.contract_id,
            "model_state_sha256": base_identity.model_state_sha256,
            "verified_source_count": base_identity.verified_source_count,
        },
        "parameters": {"k": 2, "ngram_size": 3, "physical_rows": 3,
                       "max_new_tokens": args.max_new_tokens,
                       "attention_strategy": "triton_batched",
                       "quantized_linear_strategy": STRATEGY,
                       "gate_up_kernel": "one-launch dual-output Triton packed NF4 prototype"},
        "reference": {"token_ids": ref_metrics.generated_token_ids,
                      "state_prefixes": len(ref_states)},
        "block": {"token_ids": block_metrics.generated_token_ids,
                  "state_prefixes": len(block_states)},
        "token_match": ref_metrics.generated_token_ids == block_metrics.generated_token_ids,
        "state_match": state_equal,
        "shared_state_prefixes": len(shared_prefixes),
        "same_prefix_domain": set(ref_states) == set(block_states),
        "prior_k2_reference_token_match": prior_baseline_match,
        "fused_mlp_calls_reference_and_block": {
            "singleton": ref_calls, "block": block_calls,
        },
        "timing": {"paired_rounds": timing,
                   "geomean_speedup_vs_fused_singleton": math.exp(sum(math.log(x) for x in b1_ratios)/len(b1_ratios)),
                   "geomean_speedup_vs_original_bnb": math.exp(sum(math.log(x) for x in bnb_ratios)/len(bnb_ratios)),
                   "block_matches_bnb_by_round": [
                       r["methods"]["fused_b1_block"]["token_ids"] ==
                       r["methods"]["original_bnb_greedy"]["token_ids"] for r in timing]},
        "limitations": [
            "Finite development prompt and timing sessions; not a performance claim across workloads.",
            "Correctness applies to the fused prototype's singleton reference and this runtime only.",
            "The base live contract pins model/tokenizer/weights/runtime identity and does not certify fused MLP arithmetic.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "token_match": payload["token_match"],
        "state_match": payload["state_match"],
        "shared_state_prefixes": payload["shared_state_prefixes"],
        "prior_k2_reference_token_match": prior_baseline_match,
        "speedup_vs_singleton": payload["timing"]["geomean_speedup_vs_fused_singleton"],
        "speedup_vs_bnb": payload["timing"]["geomean_speedup_vs_original_bnb"],
    }, indent=2))


if __name__ == "__main__":
    main()
