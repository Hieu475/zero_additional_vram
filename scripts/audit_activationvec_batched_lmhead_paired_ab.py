#!/usr/bin/env python3
"""Same-process A/B for rowwise vs batched LM-head on activationvec K2/R3."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_native_bnb_rowwise_batched_lm_head import batched_lm_head
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from benchmark_bnb_nf4_activationvec_closed_loop import (
    activationvec_schedule_hook, run_vanilla_timed,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/native_bnb_profiles/activationvec_batched_lmhead_manifest.json"


def validate_manifest() -> None:
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("schema") != "zassd-activationvec-batched-lmhead-manifest-v1":
        raise RuntimeError("bad batched LM-head manifest schema")
    for relpath, expected in manifest["source_sha256"].items():
        path = ROOT / relpath
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source pin mismatch: {relpath}")
    own = "scripts/audit_activationvec_batched_lmhead_paired_ab.py"
    if manifest["source_sha256"].get(own) != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("paired A/B runner is not pinned")


def observe(store: dict[int, str], prompt_len: int):
    last = {"n": -1}

    def callback(n, cache):
        n = int(n)
        for prefix in range(last["n"] + 1, n + 1):
            store[prefix] = diagnostic_cache_prefix_digest(cache, prompt_len + prefix)
        last["n"] = max(last["n"], n)

    return callback


@contextmanager
def batched_activationvec(model, native_lm_head_forward):
    with activationvec_schedule_hook(model):
        with batched_lm_head(model, native_lm_head_forward):
            yield model


def main() -> None:
    validate_manifest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/humaneval_ext114.jsonl")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timing-rounds", type=int, default=4)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if args.max_new_tokens < 4 or args.timing_rounds < 1:
        raise ValueError("require max_new_tokens >= 4 and timing_rounds >= 1")
    prompt_rows = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in prompt_rows if row["id"] == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    native_lm_head_forward = model.lm_head.forward
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + args.max_new_tokens + 8
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                  device="cuda:0", kv_cache_max_capacity=capacity,
                  physical_rows=3, attention_strategy="bnb_rowwise",
                  quantized_linear_strategy="fixed_rows")

    states = {name: {} for name in ("singleton", "rowwise_block", "batched_block")}
    with torch.inference_mode():
        _, singleton = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **common,
            state_observer=observe(states["singleton"], prompt_len))
        with activationvec_schedule_hook(model):
            _, rowwise = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common,
                state_observer=observe(states["rowwise_block"], prompt_len))
        with batched_activationvec(model, native_lm_head_forward):
            _, batched = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common,
                state_observer=observe(states["batched_block"], prompt_len))

    tokens = {"singleton": singleton.generated_token_ids,
              "rowwise_block": rowwise.generated_token_ids,
              "batched_block": batched.generated_token_ids}
    shared = sorted(set.intersection(*(set(value) for value in states.values())))
    mismatches = {name: [n for n in shared if states[name][n] != states["singleton"][n]]
                  for name in ("rowwise_block", "batched_block")}

    methods = ("original_bnb_greedy", "bnb_singleton", "rowwise_block", "batched_block")
    paired = []
    for round_idx in range(args.timing_rounds):
        order = list(methods[round_idx % len(methods):] + methods[:round_idx % len(methods)])
        record = {"round": round_idx, "method_order": order, "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb_greedy":
                ids, prefill, decode, total = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0")
                record["methods"][method] = {"token_ids": ids, "total_s": total / 1000,
                                             "prefill_s": prefill / 1000, "decode_s": decode / 1000}
            elif method == "bnb_singleton":
                with torch.inference_mode():
                    _, metrics = b1_fixed_row_generate(model, tokenizer, prompt_ids, **common)
                record["methods"][method] = {"token_ids": metrics.generated_token_ids,
                                             "total_s": metrics.total_time_s}
            else:
                ctx = activationvec_schedule_hook(model) if method == "rowwise_block" else batched_activationvec(model, native_lm_head_forward)
                with ctx, torch.inference_mode():
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
        "batched_over_rowwise": [r["methods"]["rowwise_block"]["total_s"] /
                                  r["methods"]["batched_block"]["total_s"] for r in paired],
        "rowwise_block_over_bnb": [r["methods"]["original_bnb_greedy"]["total_s"] /
                                    r["methods"]["rowwise_block"]["total_s"] for r in paired],
        "batched_block_over_bnb": [r["methods"]["original_bnb_greedy"]["total_s"] /
                                   r["methods"]["batched_block"]["total_s"] for r in paired],
        "rowwise_block_over_singleton": [r["methods"]["bnb_singleton"]["total_s"] /
                                         r["methods"]["rowwise_block"]["total_s"] for r in paired],
        "batched_block_over_singleton": [r["methods"]["bnb_singleton"]["total_s"] /
                                         r["methods"]["batched_block"]["total_s"] for r in paired],
    }
    result = {
        "schema": "zassd-activationvec-batched-lmhead-paired-ab-v1",
        "classification": "development-only same-process LM-head schedule A/B",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id, "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "max_new_tokens": args.max_new_tokens, "timing_rounds": args.timing_rounds,
        "schedule": {"k": 2, "ngram_size": 3, "physical_rows": 3, "attention": "bnb_rowwise"},
        "exactness": {
            "singleton_tokens": tokens["singleton"],
            "rowwise_block_tokens_match_singleton": tokens["rowwise_block"] == tokens["singleton"],
            "batched_block_tokens_match_singleton": tokens["batched_block"] == tokens["singleton"],
            "shared_state_prefixes": shared,
            "rowwise_state_mismatches": mismatches["rowwise_block"],
            "batched_state_mismatches": mismatches["batched_block"],
            "all_state_domains_equal": all(set(value) == set(states["singleton"]) for value in states.values()),
        },
        "acceptance": {"rowwise": {"accepted": rowwise.accepted_draft_tokens, "drafted": rowwise.draft_tokens},
                       "batched": {"accepted": batched.accepted_draft_tokens, "drafted": batched.draft_tokens}},
        "ratios_per_round": ratios,
        "ratio_medians": {key: statistics.median(value) for key, value in ratios.items()},
        "paired_rounds": paired,
        "manifest": str(MANIFEST),
        "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope_limits": ["One process per prompt, four timing rounds; small effects are descriptive.",
                         "Exactness is checked against B1 singleton; no original-BNB state claim.",
                         "Development prompts only; held-out GSM8K indices 300-349 unused."],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"prompt_id": args.prompt_id, "exactness": result["exactness"],
                      "acceptance": result["acceptance"], "ratio_medians": result["ratio_medians"],
                      "output": str(out)}, indent=2))
    if not (result["exactness"]["rowwise_block_tokens_match_singleton"]
            and result["exactness"]["batched_block_tokens_match_singleton"]
            and result["exactness"]["all_state_domains_equal"]
            and not result["exactness"]["rowwise_state_mismatches"]
            and not result["exactness"]["batched_state_mismatches"]):
        raise RuntimeError("batched LM-head candidate diverged from B1 singleton token/KV state")


if __name__ == "__main__":
    main()
