#!/usr/bin/env python3
"""Same-process A/B of block-only and all-rows Triton activationvec schedules."""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from benchmark_bnb_nf4_activationvec_allrows_closed_loop import (
    MANIFEST, all_rows_schedule_hook, validate_variant_manifest,
)
from benchmark_bnb_nf4_activationvec_closed_loop import (
    activationvec_schedule_hook, run_vanilla_timed,
)


ROOT = Path(__file__).resolve().parents[1]


def observe(store: dict[int, str], prompt_len: int):
    last = {"n": -1}

    def callback(n, cache):
        n = int(n)
        for prefix in range(last["n"] + 1, n + 1):
            store[prefix] = diagnostic_cache_prefix_digest(cache, prompt_len + prefix)
        last["n"] = max(last["n"], n)

    return callback


def main() -> None:
    validate_variant_manifest()
    # Enforce a source pin for this paired runner before model loading.
    manifest = json.loads(MANIFEST.read_text())
    own_name = "scripts/audit_activationvec_allrows_paired_ab.py"
    own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if manifest["source_sha256"].get(own_name) != own_hash:
        raise RuntimeError("paired A/B runner is not pinned by all-rows manifest")

    import argparse
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
    if args.max_new_tokens < 4 or args.timing_rounds < 1:
        raise ValueError("require max_new_tokens >= 4 and timing_rounds >= 1")

    rows = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in rows if row["id"] == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + args.max_new_tokens + 8
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                  device="cuda:0", kv_cache_max_capacity=capacity,
                  physical_rows=3, attention_strategy="bnb_rowwise",
                  quantized_linear_strategy="fixed_rows")

    states = {name: {} for name in ("singleton", "block_only", "all_rows")}
    with torch.inference_mode():
        _, singleton_metrics = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **common,
            state_observer=observe(states["singleton"], prompt_len))
        with activationvec_schedule_hook(model):
            _, block_metrics = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common,
                state_observer=observe(states["block_only"], prompt_len))
        with all_rows_schedule_hook(model):
            _, all_rows_metrics = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common,
                state_observer=observe(states["all_rows"], prompt_len))

    tokens = {"singleton": singleton_metrics.generated_token_ids,
              "block_only": block_metrics.generated_token_ids,
              "all_rows": all_rows_metrics.generated_token_ids}
    shared_prefixes = sorted(set.intersection(*(set(value) for value in states.values())))
    state_mismatches = {
        name: [prefix for prefix in shared_prefixes
               if states[name][prefix] != states["singleton"][prefix]]
        for name in ("block_only", "all_rows")
    }

    method_names = ("original_bnb_greedy", "bnb_singleton", "block_only", "all_rows")
    paired = []
    for round_index in range(args.timing_rounds):
        order = list(method_names[round_index % len(method_names):]
                     + method_names[:round_index % len(method_names)])
        record = {"round": round_index, "method_order": order, "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb_greedy":
                ids, prefill, decode, total = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0")
                record["methods"][method] = {
                    "token_ids": ids, "total_s": total / 1000,
                    "prefill_s": prefill / 1000, "decode_s": decode / 1000}
            elif method == "bnb_singleton":
                with torch.inference_mode():
                    _, metrics = b1_fixed_row_generate(model, tokenizer, prompt_ids, **common)
                record["methods"][method] = {"token_ids": metrics.generated_token_ids,
                                             "total_s": metrics.total_time_s}
            else:
                hook = activationvec_schedule_hook if method == "block_only" else all_rows_schedule_hook
                with hook(model), torch.inference_mode():
                    _, metrics = b1_fixed_row_block_generate(model, tokenizer, prompt_ids, **common)
                record["methods"][method] = {"token_ids": metrics.generated_token_ids,
                                             "total_s": metrics.total_time_s,
                                             "verify_s": metrics.verify_time_s,
                                             "cycles": metrics.cycles,
                                             "draft_tokens": metrics.draft_tokens,
                                             "accepted_draft_tokens": metrics.accepted_draft_tokens}
            record[f"{method}_matches_singleton"] = (
                record["methods"][method]["token_ids"] == tokens["singleton"])
        paired.append(record)

    ratios = {
        "block_only_over_bnb": [r["methods"]["original_bnb_greedy"]["total_s"] /
                                 r["methods"]["block_only"]["total_s"] for r in paired],
        "all_rows_over_bnb": [r["methods"]["original_bnb_greedy"]["total_s"] /
                              r["methods"]["all_rows"]["total_s"] for r in paired],
        "all_rows_over_block_only": [r["methods"]["block_only"]["total_s"] /
                                     r["methods"]["all_rows"]["total_s"] for r in paired],
        "block_only_over_singleton": [r["methods"]["bnb_singleton"]["total_s"] /
                                      r["methods"]["block_only"]["total_s"] for r in paired],
        "all_rows_over_singleton": [r["methods"]["bnb_singleton"]["total_s"] /
                                    r["methods"]["all_rows"]["total_s"] for r in paired],
    }
    result = {
        "schema": "zassd-activationvec-allrows-paired-ab-v1",
        "classification": "development-only same-process schedule A/B",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id, "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "max_new_tokens": args.max_new_tokens, "timing_rounds": args.timing_rounds,
        "schedule": {"k": 2, "ngram_size": 3, "physical_rows": 3,
                     "attention": "bnb_rowwise", "linear": "NF4 M=3 activationvec"},
        "exactness": {
            "singleton_tokens": tokens["singleton"],
            "block_only_tokens_match_singleton": tokens["block_only"] == tokens["singleton"],
            "all_rows_tokens_match_singleton": tokens["all_rows"] == tokens["singleton"],
            "shared_state_prefixes": shared_prefixes,
            "block_only_state_mismatches": state_mismatches["block_only"],
            "all_rows_state_mismatches": state_mismatches["all_rows"],
            "all_three_state_domains_equal": all(set(v) == set(states["singleton"]) for v in states.values()),
        },
        "acceptance": {"block_only": {"accepted": block_metrics.accepted_draft_tokens,
                                       "drafted": block_metrics.draft_tokens},
                       "all_rows": {"accepted": all_rows_metrics.accepted_draft_tokens,
                                    "drafted": all_rows_metrics.draft_tokens}},
        "ratios_per_round": ratios,
        "ratio_medians": {key: statistics.median(value) for key, value in ratios.items()},
        "paired_rounds": paired,
        "variant_manifest": str(MANIFEST),
        "variant_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source": str(Path(__file__).resolve()),
        "source_sha256": own_hash,
        "scope_limits": [
            "One process per prompt; paired order mitigates but does not eliminate timing drift.",
            "Finite exactness check against the fixed-row B1 singleton, not original-BNB state equality.",
            "Development prompts only; GSM8K held-out indices 300-349 not used.",
        ],
    }
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"prompt_id": args.prompt_id, "exactness": result["exactness"],
                      "acceptance": result["acceptance"], "ratio_medians": result["ratio_medians"],
                      "output": str(output)}, indent=2))
    if not (result["exactness"]["block_only_tokens_match_singleton"]
            and result["exactness"]["all_rows_tokens_match_singleton"]
            and result["exactness"]["all_three_state_domains_equal"]
            and not result["exactness"]["block_only_state_mismatches"]
            and not result["exactness"]["all_rows_state_mismatches"]):
        raise RuntimeError("An A/B schedule diverged from the B1 singleton token/state reference")


if __name__ == "__main__":
    main()
