#!/usr/bin/env python3
"""Paired audit of native M=1 BNB fallback on activationvec no-proposal cycles."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
from bitsandbytes.nn import Linear4bit
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RMSNorm

import zassd.certification.fixed_row_dispatch as dispatch
import zassd.decoding.b1_exact as b1_module
from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_bnb_nf4_activationvec_block_integration import triton_m3_overrides
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from benchmark_bnb_nf4_activationvec_closed_loop import (
    activationvec_schedule_hook, run_vanilla_timed,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_no_proposal_fallback_manifest.json"


def validate_manifest() -> None:
    data = json.loads(MANIFEST.read_text())
    if data.get("schema") != "zassd-activationvec-no-proposal-fallback-manifest-v1":
        raise RuntimeError("invalid no-proposal fallback manifest")
    for rel, expected in data["source_sha256"].items():
        path = ROOT / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source pin mismatch: {rel}")


@contextmanager
def no_proposal_native_singleton_hook(model, control):
    """On empty proposals, run one target row through the captured native operators."""
    original_schedule = dispatch.fixed_row_schedule_for_model
    native_modules = [
        (module, module.forward)
        for module in model.modules()
        if isinstance(module, (Linear4bit, torch.nn.Linear, Qwen2RMSNorm, Qwen2Attention))
    ]
    original_find = b1_module.find_candidate_tokens

    def tracked_find(*args, **kwargs):
        candidates = original_find(*args, **kwargs)
        control["no_proposal"] = not bool(candidates)
        if not candidates:
            control["no_proposal_cycles"] += 1
        return candidates

    @contextmanager
    def wrapped_schedule(target_model, **kwargs):
        with original_schedule(target_model, **kwargs):
            schedule_forward = target_model.forward

            def forward_with_native_fallback(self, *args, **call_kwargs):
                if not control["no_proposal"]:
                    return schedule_forward(*args, **call_kwargs)
                active = [(module, module.forward) for module, _ in native_modules]
                for module, native_forward in native_modules:
                    module.forward = native_forward
                control["fallback_forwards"] += 1
                try:
                    return schedule_forward(*args, **call_kwargs)
                finally:
                    for module, active_forward in reversed(active):
                        module.forward = active_forward
                    control["no_proposal"] = False

            target_model.forward = MethodType(forward_with_native_fallback, target_model)
            try:
                with triton_m3_overrides(target_model):
                    yield target_model
            finally:
                target_model.forward = schedule_forward

    dispatch.fixed_row_schedule_for_model = wrapped_schedule
    b1_module.find_candidate_tokens = tracked_find
    try:
        yield
    finally:
        b1_module.find_candidate_tokens = original_find
        dispatch.fixed_row_schedule_for_model = original_schedule


def observer(store: dict[int, str], prompt_len: int):
    last = {"n": -1}

    def record(n, cache):
        n = int(n)
        for prefix in range(last["n"] + 1, n + 1):
            store[prefix] = diagnostic_cache_prefix_digest(cache, prompt_len + prefix)
        last["n"] = max(last["n"], n)

    return record


def main() -> None:
    validate_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/humaneval_ext114.jsonl")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timing-rounds", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in records if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + args.max_new_tokens + 8
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                  device="cuda:0", kv_cache_max_capacity=capacity, physical_rows=3,
                  attention_strategy="bnb_rowwise", quantized_linear_strategy="fixed_rows",
                  trace=True)
    singleton_common = {key: value for key, value in common.items() if key != "trace"}
    states = {name: {} for name in ("singleton", "activationvec", "native_fallback")}
    controls = {"no_proposal": False, "no_proposal_cycles": 0, "fallback_forwards": 0}
    with torch.inference_mode():
        _, singleton = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **singleton_common,
            state_observer=observer(states["singleton"], prompt_len))
        with activationvec_schedule_hook(model):
            _, baseline = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common,
                state_observer=observer(states["activationvec"], prompt_len))
        with no_proposal_native_singleton_hook(model, controls):
            _, candidate = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common,
                state_observer=observer(states["native_fallback"], prompt_len))

    tokens = {"singleton": singleton.generated_token_ids,
              "activationvec": baseline.generated_token_ids,
              "native_fallback": candidate.generated_token_ids}
    domains_equal = all(set(value) == set(states["singleton"]) for value in states.values())
    shared = sorted(set.intersection(*(set(value) for value in states.values())))
    state_mismatches = {
        name: [prefix for prefix in shared if states[name][prefix] != states["singleton"][prefix]]
        for name in ("activationvec", "native_fallback")
    }
    exact = {
        "activationvec_tokens_match_singleton": tokens["activationvec"] == tokens["singleton"],
        "fallback_tokens_match_singleton": tokens["native_fallback"] == tokens["singleton"],
        "state_domains_equal": domains_equal,
        "activationvec_state_mismatches": state_mismatches["activationvec"],
        "fallback_state_mismatches": state_mismatches["native_fallback"],
        "shared_prefixes": shared,
    }
    if not (exact["activationvec_tokens_match_singleton"] and exact["fallback_tokens_match_singleton"]
            and domains_equal and not any(state_mismatches.values())
            and controls["no_proposal_cycles"] == controls["fallback_forwards"]):
        payload = {"schema": "zassd-activationvec-no-proposal-fallback-audit-v1",
                   "classification": "exactness gate failed; timing omitted", "prompt_id": args.prompt_id,
                   "exactness": exact, "fallback_counts": controls}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return

    methods = ("original_bnb", "bnb_singleton", "activationvec", "native_fallback")
    paired = []
    timing_counts = []
    for round_id in range(args.timing_rounds):
        order = list(methods[round_id % len(methods):] + methods[:round_id % len(methods)])
        sample = {"round": round_id, "method_order": order, "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb":
                ids, prefill, decode, total = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0")
                entry = {"token_ids": ids, "total_s": total / 1000,
                         "prefill_s": prefill / 1000, "decode_s": decode / 1000}
            elif method == "bnb_singleton":
                with torch.inference_mode():
                    _, metrics = b1_fixed_row_generate(model, tokenizer, prompt_ids, **singleton_common)
                entry = {"token_ids": metrics.generated_token_ids, "total_s": metrics.total_time_s}
            elif method == "activationvec":
                with activationvec_schedule_hook(model), torch.inference_mode():
                    _, metrics = b1_fixed_row_block_generate(model, tokenizer, prompt_ids, **common)
                entry = {"token_ids": metrics.generated_token_ids, "total_s": metrics.total_time_s}
            else:
                timed_control = {"no_proposal": False, "no_proposal_cycles": 0, "fallback_forwards": 0}
                with no_proposal_native_singleton_hook(model, timed_control), torch.inference_mode():
                    _, metrics = b1_fixed_row_block_generate(model, tokenizer, prompt_ids, **common)
                entry = {"token_ids": metrics.generated_token_ids, "total_s": metrics.total_time_s,
                         "fallback_counts": timed_control}
                timing_counts.append(timed_control)
            sample["methods"][method] = entry
        paired.append(sample)

    ratios = {
        "fallback_over_activationvec": [r["methods"]["activationvec"]["total_s"] /
                                         r["methods"]["native_fallback"]["total_s"] for r in paired],
        "fallback_over_bnb_singleton": [r["methods"]["bnb_singleton"]["total_s"] /
                                        r["methods"]["native_fallback"]["total_s"] for r in paired],
        "activationvec_over_bnb_singleton": [r["methods"]["bnb_singleton"]["total_s"] /
                                             r["methods"]["activationvec"]["total_s"] for r in paired],
        "fallback_over_original_bnb_latency": [r["methods"]["original_bnb"]["total_s"] /
                                               r["methods"]["native_fallback"]["total_s"] for r in paired],
    }
    payload = {
        "schema": "zassd-activationvec-no-proposal-fallback-audit-v1",
        "classification": "same-process paired development experiment; B1 singleton identity required before latency interpretation",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id, "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "max_new_tokens": args.max_new_tokens, "timing_rounds": args.timing_rounds,
        "schedule": {"k": 2, "ngram_size": 3, "physical_rows": 3,
                     "attention": "bnb_rowwise", "fallback": "BNB native M=1 on no-proposal cycles"},
        "exactness": exact,
        "acceptance": {"activationvec": {"drafted": baseline.draft_tokens,
                                            "accepted": baseline.accepted_draft_tokens},
                       "fallback": {"drafted": candidate.draft_tokens,
                                    "accepted": candidate.accepted_draft_tokens}},
        "no_proposal_fallback_counts_preflight": controls,
        "timing_fallback_counts": timing_counts,
        "ratio_medians": {key: statistics.median(value) for key, value in ratios.items()},
        "ratios_per_round": ratios, "paired_rounds": paired,
        "manifest": str(MANIFEST),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["Finite HumanEval development prompt, one GPU.",
                   "Original BNB comparison is latency-only; no original-BNB state proof.",
                   "GSM8K held-out indices 300-349 excluded."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompt_id": args.prompt_id, "exactness": exact,
                      "fallback_counts": controls, "ratio_medians": payload["ratio_medians"],
                      "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
