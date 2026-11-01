#!/usr/bin/env python3
"""Compute-only Qwen timing for the fixed-row Triton attention reference."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_triton_fixed_row_attention import fixed_row_sdpa, rowwise_lm_head
try:
    from run_kv_factorial_intervention import restore_static_cache, snapshot_cache
except ModuleNotFoundError:
    from scripts.run_kv_factorial_intervention import restore_static_cache, snapshot_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/cnndm_eval.jsonl")
    parser.add_argument("--prompt-id", default="cnndm_000")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    rows = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines()
            if line.strip()]
    prompt = next((row for row in rows if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt ID {args.prompt_id!r} absent")

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    loaded = load_backend("hf_bnb_nf4", model_name, revision, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    capacity = int(ids.shape[1]) + args.k + 8
    prefill_cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.inference_mode():
        pending = int(model(ids, past_key_values=prefill_cache.cache,
                            use_cache=True).logits[:, -1].argmax(dim=-1).item())
    prefix_snapshot = snapshot_cache(prefill_cache.cache)
    draft_cache = restore_static_cache(prefix_snapshot, capacity, "cuda:0")
    candidate_ids = [pending]
    with fixed_row_sdpa(model, args.block_k), torch.inference_mode():
        for _ in range(args.k):
            token = torch.tensor([[candidate_ids[-1]]], device="cuda:0")
            logits = model(token, past_key_values=draft_cache.cache,
                           use_cache=True).logits[:, -1]
            candidate_ids.append(int(logits.argmax(dim=-1).item()))
    token_block = torch.tensor([candidate_ids], dtype=torch.long, device="cuda:0")

    def execute(method):
        cache = restore_static_cache(prefix_snapshot, capacity, "cuda:0")
        token_ids = [torch.tensor([[token_id]], dtype=torch.long, device="cuda:0")
                     for token_id in candidate_ids]

        def forward():
            with torch.inference_mode():
                if method.endswith("singleton"):
                    for token in token_ids:
                        model(token, past_key_values=cache.cache, use_cache=True)
                else:
                    model(token_block, past_key_values=cache.cache, use_cache=True)

        if method.startswith("fixed_triton"):
            with fixed_row_sdpa(model, args.block_k), rowwise_lm_head(model):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                forward()
                end.record()
                end.synchronize()
        else:
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            forward()
            end.record()
            end.synchronize()
        return float(start.elapsed_time(end))

    methods = ("bnb_singleton", "bnb_block", "fixed_triton_singleton",
               "fixed_triton_block")
    for _ in range(args.warmups):
        for method in methods:
            execute(method)
    samples = {method: [] for method in methods}
    for round_index in range(args.rounds):
        order = methods if round_index % 2 == 0 else tuple(reversed(methods))
        for method in order:
            samples[method].append(execute(method))
    medians = {method: statistics.median(values) for method, values in samples.items()}
    payload = {
        "scope": "paired compute-only model-forward timing; KV restore outside timer; not end-to-end latency",
        "model": model_name, "revision": revision, "prompt_id": args.prompt_id,
        "prompt_tokens": int(ids.shape[1]), "k": args.k,
        "physical_rows": len(candidate_ids), "block_k": args.block_k,
        "candidate_ids": candidate_ids,
        "gpu": torch.cuda.get_device_name("cuda:0"),
        "torch": torch.__version__, "rounds": args.rounds,
        "warmups": args.warmups, "median_ms": medians, "samples_ms": samples,
        "fixed_triton_speedup_vs_singleton": (
            medians["fixed_triton_singleton"] / medians["fixed_triton_block"]),
        "bnb_speedup_vs_singleton": medians["bnb_singleton"] / medians["bnb_block"],
        "fixed_block_overhead_vs_native_block": (
            medians["fixed_triton_block"] / medians["bnb_block"]),
        "notes": [
            "fixed_triton methods use the same custom attention kernel for singleton and block, plus rowwise LM head",
            "candidate IDs are drafted under the fixed Triton singleton reference",
            "timing excludes cache restore, prefill, drafting and commit/rollback",
            "exactness for one block was separately checked by audit_triton_fixed_row_attention.py",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "samples_ms"}, indent=2))


if __name__ == "__main__":
    main()
