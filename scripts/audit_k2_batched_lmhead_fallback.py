#!/usr/bin/env python3
"""Paired exactness/performance audit of batched LM head under K2 full fallback."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch
from types import MethodType

from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_activationvec_fallback_fraction_ablation_512 import MANIFEST as ABLATION_MANIFEST
from audit_activationvec_fallback_fraction_ablation_512 import (
    cache_observer,
    fallback_fraction_schedule,
    validate_manifest as validate_base_manifest,
)
from run_strong_baseline_comparison import run_vanilla_timed
from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles"
MANIFEST = BASE / "k2_batched_lmhead_fallback_manifest.json"
BENCHMARK = ROOT / "data/benchmarks/humaneval_ext114.jsonl"
PROMPTS = ("HumanEval/123", "HumanEval/114")
SOURCES = (
    "scripts/audit_k2_batched_lmhead_fallback.py",
    "scripts/audit_bnb_nf4_gate_reduction_orders.py",
    "scripts/audit_activationvec_fallback_fraction_ablation_512.py",
    "scripts/run_strong_baseline_comparison.py",
    "src/zassd/backends/base.py",
    "src/zassd/decoding/b1_exact.py",
    "src/zassd/decoding/ngram.py",
    "src/zassd/certification/fixed_row_dispatch.py",
    "src/zassd/certification/qwen2_padded.py",
    "src/zassd/cache/kv_cache.py",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema") != "zassd-k2-batched-lmhead-fallback-manifest-v1":
        raise RuntimeError("invalid batched-LM-head manifest")
    for rel, expected in manifest["source_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"source pin mismatch: {rel}")
    for rel, expected in manifest["benchmark_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise RuntimeError(f"benchmark pin mismatch: {rel}")
    ablation = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/fallback_fraction_ablation_512_manifest.json"
    if sha(ablation) != manifest["ablation_manifest_sha256"]:
        raise RuntimeError("base fallback ablation manifest changed")
    validate_base_manifest()


@contextmanager
def batched_lm_head(model, native_forward):
    wrapped = model.lm_head.forward

    def forward(self, hidden_states, *args, **kwargs):
        return native_forward(hidden_states, *args, **kwargs)

    model.lm_head.forward = MethodType(forward, model.lm_head)
    try:
        yield
    finally:
        model.lm_head.forward = wrapped


def run_block(model, tokenizer, prompt_ids, common, mode: str, state_store=None):
    control = {"no_proposal_cycles": 0, "fallback_forwards": 0,
               "empty_cycle_index": 0, "use_native": False}
    kwargs = {**common, "trace": True}
    observer = cache_observer(state_store, int(prompt_ids.shape[-1])) if state_store is not None else None
    lm_context = batched_lm_head(model, model._zassd_native_lm_head) if mode == "batched" else nullcontext()
    with fallback_fraction_schedule(model, control, 1), lm_context, torch.inference_mode():
        _, metrics = b1_fixed_row_block_generate(
            model, tokenizer, prompt_ids, **kwargs,
            **({"state_observer": observer} if observer is not None else {}))
    return metrics, control


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-id", choices=PROMPTS, required=True)
    ap.add_argument("--max-new-tokens", choices=(256, 512), type=int, required=True)
    ap.add_argument("--timing-rounds", type=int, default=4)
    args = ap.parse_args()
    if args.timing_rounds < 2:
        raise ValueError("at least two paired timing rounds required")
    validate_manifest()
    tag = args.prompt_id.replace("/", "")
    out = BASE / f"k2_batched_lmhead_{tag}_{args.max_new_tokens}_fallback_v1.json"
    if out.exists():
        raise FileExistsError(out)
    row = next(r for r in (json.loads(x) for x in BENCHMARK.read_text().splitlines() if x.strip())
               if str(r["id"]) == args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    native_lm_head = model.lm_head.forward
    model._zassd_native_lm_head = native_lm_head
    prompt_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + args.max_new_tokens + 8
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                  device="cuda:0", kv_cache_max_capacity=capacity, physical_rows=3,
                  attention_strategy="bnb_rowwise", quantized_linear_strategy="fixed_rows")

    ref_store, rowwise_store, batched_store = {}, {}, {}
    with torch.inference_mode():
        _, ref = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **common,
            state_observer=cache_observer(ref_store, prompt_len))
    rowwise, rowwise_control = run_block(model, tokenizer, prompt_ids, common, "rowwise", rowwise_store)
    batched, batched_control = run_block(model, tokenizer, prompt_ids, common, "batched", batched_store)
    state_domain = set(ref_store) == set(rowwise_store) == set(batched_store)
    state_diff_rowwise = [i for i in sorted(set(ref_store) & set(rowwise_store)) if ref_store[i] != rowwise_store[i]]
    state_diff_batched = [i for i in sorted(set(ref_store) & set(batched_store)) if ref_store[i] != batched_store[i]]
    token_match = {
        "rowwise": rowwise.generated_token_ids == ref.generated_token_ids,
        "batched": batched.generated_token_ids == ref.generated_token_ids,
    }
    exact = state_domain and not state_diff_rowwise and not state_diff_batched and all(token_match.values())
    if not exact:
        raise RuntimeError(f"B1 preflight failed: {token_match} {state_diff_rowwise[:8]} {state_diff_batched[:8]}")

    with torch.inference_mode():
        run_vanilla_timed(model, tokenizer, prompt_ids, 8, "cuda:0")
        run_block(model, tokenizer, prompt_ids, {**common, "max_new_tokens": 8}, "rowwise")
        run_block(model, tokenizer, prompt_ids, {**common, "max_new_tokens": 8}, "batched")

    methods = ("original_bnb_greedy", "rowwise_full_fallback", "batched_lmhead_full_fallback")
    rounds = []
    for index in range(args.timing_rounds):
        order = list(methods[index % len(methods):] + methods[:index % len(methods)])
        sample = {"round": index, "method_order": order, "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb_greedy":
                ids, prefill_ms, decode_ms, total_ms = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0")
                sample["methods"][method] = {"token_ids": ids, "total_s": total_ms / 1000,
                                             "prefill_s": prefill_ms / 1000, "decode_s": decode_ms / 1000}
            else:
                mode = "batched" if method.startswith("batched") else "rowwise"
                metrics, control = run_block(model, tokenizer, prompt_ids, common, mode)
                sample["methods"][method] = {
                    "token_ids": metrics.generated_token_ids,
                    "total_s": metrics.total_time_s,
                    "prefill_s": metrics.prefill_time_s,
                    "verify_s": metrics.verify_time_s,
                    "cycles": metrics.cycles,
                    "draft_tokens": metrics.draft_tokens,
                    "accepted_draft_tokens": metrics.accepted_draft_tokens,
                    "target_rows": metrics.target_rows,
                    "fallback_forwards": control["fallback_forwards"],
                }
        sample["rowwise_matches_b1"] = sample["methods"]["rowwise_full_fallback"]["token_ids"] == ref.generated_token_ids
        sample["batched_matches_b1"] = sample["methods"]["batched_lmhead_full_fallback"]["token_ids"] == ref.generated_token_ids
        sample["rowwise_matches_bnb"] = sample["methods"]["rowwise_full_fallback"]["token_ids"] == sample["methods"]["original_bnb_greedy"]["token_ids"]
        sample["batched_matches_bnb"] = sample["methods"]["batched_lmhead_full_fallback"]["token_ids"] == sample["methods"]["original_bnb_greedy"]["token_ids"]
        if not (sample["rowwise_matches_b1"] and sample["batched_matches_b1"]):
            raise RuntimeError(f"timed B1 mismatch at round {index}")
        rounds.append(sample)

    import math
    speedups = {
        mode: [r["methods"]["original_bnb_greedy"]["total_s"] / r["methods"][mode]["total_s"] for r in rounds]
        for mode in methods[1:]
    }
    batched_over_rowwise = [r["methods"]["rowwise_full_fallback"]["total_s"] /
                            r["methods"]["batched_lmhead_full_fallback"]["total_s"] for r in rounds]
    payload = {
        "schema": "zassd-k2-batched-lmhead-fallback-v1",
        "classification": "paired development ablation of rowwise vs batched LM head under exact K2/R3 full fallback",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id, "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
        "max_new_tokens": args.max_new_tokens, "timing_rounds": args.timing_rounds,
        "schedule": {"k": 2, "ngram_size": 3, "physical_rows": 3, "attention": "bnb_rowwise",
                     "linear": "fixed_rows", "fallback": "native BNB M=1 on every empty-proposal cycle"},
        "b1_preflight": {"token_match": token_match, "state_domains_equal": state_domain,
                         "rowwise_state_mismatches": state_diff_rowwise,
                         "batched_state_mismatches": state_diff_batched,
                         "shared_state_prefixes": len(ref_store)},
        "timed_b1_exact": True,
        "original_bnb_token_match_rounds": {
            "rowwise": sum(r["rowwise_matches_bnb"] for r in rounds),
            "batched": sum(r["batched_matches_bnb"] for r in rounds),
        },
        "paired_rounds": rounds,
        "speedup_vs_bnb_geomean": {k: math.exp(sum(math.log(x) for x in v) / len(v)) for k, v in speedups.items()},
        "batched_over_rowwise_geomean": math.exp(sum(math.log(x) for x in batched_over_rowwise) / len(batched_over_rowwise)),
        "ablation_manifest_sha256": sha(ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/fallback_fraction_ablation_512_manifest.json"),
        "sweep_manifest_sha256": sha(MANIFEST), "runner_sha256": sha(Path(__file__)),
        "heldout_used": False,
        "limits": ["Selected two-prompt development cohort on one model/GPU; paired rounds are nested within process.",
                   "Ordinary paired timing; Nsight instrumentation is not used for latency.",
                   "Original-BNB KV-state identity is not claimed.", "GSM8K held-out indices 300-349 excluded."],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(out), "prompt_id": args.prompt_id,
                      "exact": exact, "speedups_vs_bnb": payload["speedup_vs_bnb_geomean"],
                      "batched_over_rowwise": payload["batched_over_rowwise_geomean"],
                      "bnb_token_matches": payload["original_bnb_token_match_rounds"]}, indent=2))


if __name__ == "__main__":
    main()
