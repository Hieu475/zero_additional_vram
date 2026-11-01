#!/usr/bin/env python3
"""Paired B1-exact ablation of fallback on 0%, half, or all empty-proposal cycles."""

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


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/fallback_fraction_ablation_manifest.json"


def validate_manifest() -> None:
    data = json.loads(MANIFEST.read_text())
    if data.get("schema") != "zassd-fallback-fraction-ablation-manifest-v1":
        raise RuntimeError("invalid fallback-fraction manifest")
    for rel, expected in data["source_sha256"].items():
        path = ROOT / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source pin mismatch: {rel}")


@contextmanager
def fallback_fraction_schedule(model, control: dict, stride: int):
    """Use native BNB on every stride-th empty proposal; stride 0 means none."""
    original_schedule = dispatch.fixed_row_schedule_for_model
    native_modules = [
        (module, module.forward)
        for module in model.modules()
        if isinstance(module, (Linear4bit, torch.nn.Linear, Qwen2RMSNorm, Qwen2Attention))
    ]
    original_find = b1_module.find_candidate_tokens

    def tracked_find(*args, **kwargs):
        candidates = original_find(*args, **kwargs)
        control["use_native"] = False
        if not candidates:
            control["no_proposal_cycles"] += 1
            control["empty_cycle_index"] += 1
            control["use_native"] = stride > 0 and control["empty_cycle_index"] % stride == 0
        return candidates

    @contextmanager
    def wrapped_schedule(target_model, **kwargs):
        with original_schedule(target_model, **kwargs):
            schedule_forward = target_model.forward

            def forward_with_fraction(self, *args, **call_kwargs):
                use_native = bool(control["use_native"])
                control["use_native"] = False
                if not use_native:
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

            target_model.forward = MethodType(forward_with_fraction, target_model)
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


def cache_observer(store: dict[int, str], prompt_len: int):
    last = {"n": -1}

    def record(n, cache):
        n = int(n)
        for prefix in range(last["n"] + 1, n + 1):
            store[prefix] = diagnostic_cache_prefix_digest(cache, prompt_len + prefix)
        last["n"] = max(last["n"], n)

    return record


def run_candidate(model, tokenizer, prompt_ids, common, prompt_len, stride, states, name):
    control = {"no_proposal_cycles": 0, "fallback_forwards": 0,
               "empty_cycle_index": 0, "use_native": False}
    with fallback_fraction_schedule(model, control, stride), torch.inference_mode():
        _, metrics = b1_fixed_row_block_generate(
            model, tokenizer, prompt_ids, **common,
            state_observer=cache_observer(states[name], prompt_len))
    return metrics, control


def main() -> None:
    validate_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
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
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens, device="cuda:0",
                  kv_cache_max_capacity=capacity, physical_rows=3, attention_strategy="bnb_rowwise",
                  quantized_linear_strategy="fixed_rows", trace=True)
    singleton_common = {key: value for key, value in common.items() if key != "trace"}
    modes = {"activationvec": 0, "fallback_every_other": 2, "fallback_all": 1}
    states = {"singleton": {}}
    states.update({name: {} for name in modes})
    with torch.inference_mode():
        _, singleton = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **singleton_common,
            state_observer=cache_observer(states["singleton"], prompt_len))
        preflight = {}
        for name, stride in modes.items():
            metrics, control = run_candidate(
                model, tokenizer, prompt_ids, common, prompt_len, stride, states, name)
            preflight[name] = {"metrics": metrics, "control": control}

    singleton_tokens = singleton.generated_token_ids
    shared = sorted(set.intersection(*(set(value) for value in states.values())))
    domains_equal = all(set(value) == set(states["singleton"]) for value in states.values())
    mismatches = {
        name: [prefix for prefix in shared if states[name][prefix] != states["singleton"][prefix]]
        for name in modes
    }
    exact = {
        "all_modes_match_singleton_tokens": all(
            preflight[name]["metrics"].generated_token_ids == singleton_tokens for name in modes),
        "state_domains_equal": domains_equal,
        "state_mismatches": mismatches,
        "shared_prefix_count": len(shared),
    }
    if not (exact["all_modes_match_singleton_tokens"] and domains_equal
            and not any(mismatches.values())
            and all(v["control"]["no_proposal_cycles"] == preflight["fallback_all"]["control"]["no_proposal_cycles"]
                    for v in preflight.values())
            and preflight["fallback_every_other"]["control"]["fallback_forwards"]
            == preflight["fallback_every_other"]["control"]["no_proposal_cycles"] // 2):
        payload = {"schema": "zassd-fallback-fraction-ablation-v1",
                   "classification": "exactness gate failed; timing omitted",
                   "prompt_id": args.prompt_id, "exactness": exact,
                   "preflight_controls": {name: value["control"] for name, value in preflight.items()}}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + chr(10))
        print(json.dumps(payload, indent=2))
        return

    paired = []
    for round_id in range(args.timing_rounds):
        ordered = list(modes)[round_id % len(modes):] + list(modes)[:round_id % len(modes)]
        sample = {"round": round_id, "method_order": ordered, "methods": {}}
        for name in ordered:
            timed_states = {"timed": {}}
            control = {"no_proposal_cycles": 0, "fallback_forwards": 0,
                       "empty_cycle_index": 0, "use_native": False}
            with fallback_fraction_schedule(model, control, modes[name]), torch.inference_mode():
                _, metrics = b1_fixed_row_block_generate(
                    model, tokenizer, prompt_ids, **common,
                    state_observer=cache_observer(timed_states["timed"], prompt_len))
            sample["methods"][name] = {
                "token_ids": metrics.generated_token_ids,
                "total_s": metrics.total_time_s,
                "fallback_counts": control,
            }
        paired.append(sample)

    ratios = {}
    names = list(modes)
    for numerator in names:
        for denominator in names:
            if numerator == denominator:
                continue
            key = f"{numerator}_over_{denominator}_latency"
            ratios[key] = [
                row["methods"][denominator]["total_s"] / row["methods"][numerator]["total_s"]
                for row in paired
            ]
    timed_token_match = {
        name: all(row["methods"][name]["token_ids"] == singleton_tokens for row in paired)
        for name in modes
    }
    if not all(timed_token_match.values()):
        raise RuntimeError(f"timed token mismatch against B1 singleton: {timed_token_match}")
    payload = {
        "schema": "zassd-fallback-fraction-ablation-v1",
        "classification": "same-process paired development ablation; each schedule gated against B1 singleton tokens and state",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id, "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "max_new_tokens": args.max_new_tokens, "timing_rounds": args.timing_rounds,
        "schedule": {"k": 2, "ngram_size": 3, "physical_rows": 3,
                     "modes": {"activationvec": "native M=1 on zero empty-proposal cycles",
                               "fallback_every_other": "native M=1 on every second empty-proposal cycle",
                               "fallback_all": "native M=1 on every empty-proposal cycle"}},
        "exactness": exact,
        "acceptance": {name: {"drafted": value["metrics"].draft_tokens,
                               "accepted": value["metrics"].accepted_draft_tokens}
                       for name, value in preflight.items()},
        "preflight_controls": {name: value["control"] for name, value in preflight.items()},
        "timed_token_match_singleton": timed_token_match,
        "paired_rounds": paired,
        "ratio_medians": {key: statistics.median(value) for key, value in ratios.items()},
        "ratios_per_round": ratios,
        "manifest": str(MANIFEST),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "heldout_used": False,
        "limits": ["Selected development prompt; one model revision and GPU.",
                   "Original-BNB comparison omitted; this experiment isolates fallback schedule effects.",
                   "GSM8K held-out indices 300-349 excluded."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + chr(10))
    print(json.dumps({"prompt_id": args.prompt_id, "exactness": exact,
                      "preflight_controls": payload["preflight_controls"],
                      "ratio_medians": payload["ratio_medians"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
