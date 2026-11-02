#!/usr/bin/env python3
"""Paired closed-loop K=2/R=3 benchmark for the fused NF4 M=3 linears."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_bnb_nf4_activationvec_block_integration import triton_m3_overrides
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from _common import run_vanilla_timed


ATTENTION = "bnb_rowwise"


def scheduled(model):
    return fixed_row_schedule_for_model(model, physical_rows=3, attention_strategy=ATTENTION,
                                        quantized_linear_strategy="fixed_rows")


@contextmanager
def activationvec_schedule_hook(model):
    """Wrap the decoder's fixed-row schedule so Triton replaces only M=3 linears."""
    import zassd.certification.fixed_row_dispatch as dispatch

    original_schedule = dispatch.fixed_row_schedule_for_model

    @contextmanager
    def wrapped_schedule(target_model, **kwargs):
        with original_schedule(target_model, **kwargs):
            with triton_m3_overrides(target_model):
                yield target_model

    dispatch.fixed_row_schedule_for_model = wrapped_schedule
    try:
        yield
    finally:
        dispatch.fixed_row_schedule_for_model = original_schedule


def observe_prefixes(store: dict, lengths: dict, prompt_len: int):
    last = {"n": -1}

    def callback(n, cache):
        n = int(n)
        lengths[n] = {"cache_seq_length": int(cache.get_seq_length()),
                      "layer_seq_lengths": [int(layer.get_seq_length()) for layer in cache.cache.layers]}
        for prefix in range(last["n"] + 1, n + 1):
            store[prefix] = diagnostic_cache_prefix_digest(cache, prompt_len + prefix)
        last["n"] = max(last["n"], n)

    return callback


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/cnndm_eval.jsonl")
    ap.add_argument("--prompt-id", default="cnndm_003")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--timing-rounds", type=int, default=2)
    ap.add_argument("--manifest", default="experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/experiment_manifest.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.max_new_tokens < 4 or args.timing_rounds < 1:
        raise ValueError("max_new_tokens must be >= 4 and timing_rounds must be positive")
    repo = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    source_name = "scripts/benchmark_bnb_nf4_activationvec_closed_loop.py"
    if manifest.get("source_sha256", {}).get(source_name) != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("closed-loop benchmark source is not pinned in the experiment manifest")
    validator = repo / "scripts/validate_nf4_singleton_k2_r3_manifest.py"
    subprocess.run([sys.executable, str(validator), "--manifest", str(manifest_path)], cwd=repo, check=True,
                   stdout=subprocess.DEVNULL)
    rows = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in rows if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + args.max_new_tokens + 8
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens, device="cuda:0",
                  kv_cache_max_capacity=capacity, physical_rows=3, attention_strategy=ATTENTION,
                  quantized_linear_strategy="fixed_rows", trace=True)

    singleton_hashes, block_hashes = {}, {}
    singleton_lengths, block_lengths = {}, {}
    with torch.inference_mode():
        _, singleton_metrics = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
            device="cuda:0", kv_cache_max_capacity=capacity, physical_rows=3,
            attention_strategy=ATTENTION, quantized_linear_strategy="fixed_rows",
            state_observer=observe_prefixes(singleton_hashes, singleton_lengths, prompt_len),
        )
    with activationvec_schedule_hook(model), torch.inference_mode():
        _, block_metrics = b1_fixed_row_block_generate(
            model, tokenizer, prompt_ids, **common,
            state_observer=observe_prefixes(block_hashes, block_lengths, prompt_len),
        )

    singleton_tokens = singleton_metrics.generated_token_ids
    block_tokens = block_metrics.generated_token_ids
    shared = sorted(set(singleton_hashes) & set(block_hashes))
    state_mismatches = [n for n in shared if singleton_hashes[n] != block_hashes[n]]
    length_shared = sorted(set(singleton_lengths) & set(block_lengths))
    length_mismatches = [n for n in length_shared if singleton_lengths[n] != block_lengths[n]]

    methods = ("original_bnb_greedy", "bnb_singleton", "activationvec_block")
    paired = []
    for round_idx in range(args.timing_rounds):
        order = list(methods[round_idx % len(methods):] + methods[:round_idx % len(methods)])
        sample = {"round": round_idx, "method_order": order, "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb_greedy":
                tokens, prefill_ms, decode_ms, total_ms = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0")
                sample["methods"][method] = {"token_ids": tokens, "total_s": total_ms / 1000,
                                             "prefill_s": prefill_ms / 1000, "decode_s": decode_ms / 1000}
            elif method == "bnb_singleton":
                with torch.inference_mode():
                    _, metrics = b1_fixed_row_generate(
                        model, tokenizer, prompt_ids, k=2, ngram_size=3,
                        max_new_tokens=args.max_new_tokens, device="cuda:0",
                        kv_cache_max_capacity=capacity, physical_rows=3,
                        attention_strategy=ATTENTION, quantized_linear_strategy="fixed_rows")
                sample["methods"][method] = {"token_ids": metrics.generated_token_ids,
                                             "total_s": metrics.total_time_s,
                                             "verify_s": metrics.verify_time_s,
                                             "cycles": metrics.cycles}
            else:
                with activationvec_schedule_hook(model), torch.inference_mode():
                    _, metrics = b1_fixed_row_block_generate(
                        model, tokenizer, prompt_ids, **common)
                sample["methods"][method] = {"token_ids": metrics.generated_token_ids,
                                             "total_s": metrics.total_time_s,
                                             "verify_s": metrics.verify_time_s,
                                             "cycles": metrics.cycles,
                                             "draft_tokens": metrics.draft_tokens,
                                             "accepted_draft_tokens": metrics.accepted_draft_tokens,
                                             "peak_vram_mb": metrics.peak_vram_mb}
        sample["block_matches_singleton"] = sample["methods"]["activationvec_block"]["token_ids"] == sample["methods"]["bnb_singleton"]["token_ids"]
        sample["block_matches_original_bnb"] = sample["methods"]["activationvec_block"]["token_ids"] == sample["methods"]["original_bnb_greedy"]["token_ids"]
        paired.append(sample)

    timings = {name: [r["methods"][name]["total_s"] for r in paired] for name in methods}
    result = {
        "schema": "zassd-bnb-fused-nf4-activationvec-closed-loop-v1",
        "classification": "development-only K=2/R=3 B2 closed-loop audit and paired timing",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "prompt_length": prompt_len, "max_new_tokens": args.max_new_tokens,
        "attention_strategy": ATTENTION, "physical_rows": 3, "k": 2, "ngram_size": 3,
        "singleton_tokens": singleton_tokens, "activationvec_block_tokens": block_tokens,
        "block_matches_singleton": singleton_tokens == block_tokens,
        "singleton_prefix_count": len(singleton_hashes), "block_prefix_count": len(block_hashes),
        "shared_prefixes": shared, "state_mismatches": state_mismatches,
        "state_prefixes_bitwise_equal": bool(shared) and not state_mismatches,
        "shared_length_frontiers": length_shared, "length_mismatches": length_mismatches,
        "cache_lengths_equal": bool(length_shared) and not length_mismatches,
        "singleton_cycle_trace": singleton_metrics.cycle_trace,
        "activationvec_cycle_trace": block_metrics.cycle_trace,
        "paired_rounds": paired,
        "summary": {name: {"samples_s": values, "median_s": statistics.median(values)}
                    for name, values in timings.items()},
        "paired_speedup_medians": {
            "bnb_singleton_over_activationvec_block": statistics.median(timings["bnb_singleton"][i] / timings["activationvec_block"][i] for i in range(len(paired))),
            "original_bnb_over_activationvec_block": statistics.median(timings["original_bnb_greedy"][i] / timings["activationvec_block"][i] for i in range(len(paired))),
        },
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_source": "scripts/benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2.py",
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "bitsandbytes": importlib.metadata.version("bitsandbytes"), "gpu": torch.cuda.get_device_name(0)},
        "scope_limits": ["Finite development cohort and paired rounds; no prompt-level generalization claim.",
                         "Exactness is relative to the BNB singleton schedule under bnb_rowwise attention.",
                         "GSM8K held-out indices 300-349 excluded."]
    }
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("prompt_id", "max_new_tokens", "block_matches_singleton",
                    "state_prefixes_bitwise_equal", "cache_lengths_equal", "state_mismatches",
                    "paired_speedup_medians", "summary", "output") if k != "output"}, indent=2))
    print(f"output={out}")
    if not result["block_matches_singleton"] or not result["state_prefixes_bitwise_equal"] or not result["cache_lengths_equal"]:
        raise RuntimeError("Closed-loop activationvec block diverged from BNB singleton tokens/state")


if __name__ == "__main__":
    main()
