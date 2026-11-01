#!/usr/bin/env python3
"""Paired compute-only timing for singleton, block, and rowwise-SDPA block.

Cache restoration is deliberately outside the timed region. The rowwise mode is
a correctness fallback, not a proposed fast attention kernel.
"""

from __future__ import annotations

import argparse
import json
import statistics
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
try:
    from run_kv_factorial_intervention import restore_static_cache, snapshot_cache
except ModuleNotFoundError:
    from scripts.run_kv_factorial_intervention import restore_static_cache, snapshot_cache


@contextmanager
def rowwise_sdpa_and_lm_head(model):
    original_sdpa = F.scaled_dot_product_attention
    original_lm_forward = model.lm_head.forward
    kv_heads = int(model.config.num_key_value_heads)

    def sdpa_rowwise(query, key, value, *args, **kwargs):
        if query.shape[-2] <= 1:
            return original_sdpa(query, key, value, *args, **kwargs)
        repeated_heads = query.shape[1] // kv_heads
        compact_key = key[:, ::repeated_heads]
        compact_value = value[:, ::repeated_heads]
        row_kwargs = dict(kwargs)
        row_kwargs.pop("attn_mask", None)
        row_kwargs.pop("enable_gqa", None)
        row_kwargs["is_causal"] = False
        row_kwargs["enable_gqa"] = True
        outputs = []
        for row in range(query.shape[-2]):
            visible = key.shape[-2] - query.shape[-2] + row + 1
            outputs.append(original_sdpa(
                query[:, :, row:row + 1], compact_key[:, :, :visible],
                compact_value[:, :, :visible], *(), **row_kwargs))
        return torch.cat(outputs, dim=-2)

    def lm_head_rowwise(hidden_states, *args, **kwargs):
        if hidden_states.ndim >= 3 and hidden_states.shape[-2] > 1:
            return torch.cat([
                original_lm_forward(hidden_states[..., row:row + 1, :], *args, **kwargs)
                for row in range(hidden_states.shape[-2])
            ], dim=-2)
        return original_lm_forward(hidden_states, *args, **kwargs)

    F.scaled_dot_product_attention = sdpa_rowwise
    model.lm_head.forward = lm_head_rowwise
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original_sdpa
        model.lm_head.forward = original_lm_forward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/cnndm_eval.jsonl")
    parser.add_argument("--prompt-id", default="cnndm_000")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.k < 1 or args.rounds < 1 or args.warmups < 0:
        raise ValueError("k and rounds must be positive; warmups must be nonnegative")

    prompts = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines()
               if line.strip()]
    prompt = next((row for row in prompts if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt ID {args.prompt_id!r} absent")

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    loaded = load_backend("hf_bnb_nf4", model_name, revision, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    capacity = int(ids.shape[1]) + args.k + 8
    prompt_cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.inference_mode():
        prefill_logits = model(ids, past_key_values=prompt_cache.cache,
                               use_cache=True).logits[:, -1]
    prefix_snapshot = snapshot_cache(prompt_cache.cache)

    candidate_ids = [int(prefill_logits.argmax(dim=-1).item())]
    draft_cache = restore_static_cache(prefix_snapshot, capacity, "cuda:0")
    with torch.inference_mode():
        for _ in range(args.k):
            token = torch.tensor([[candidate_ids[-1]]], dtype=torch.long, device="cuda:0")
            logits = model(token, past_key_values=draft_cache.cache,
                           use_cache=True).logits[:, -1]
            candidate_ids.append(int(logits.argmax(dim=-1).item()))
    token_block = torch.tensor([candidate_ids], dtype=torch.long, device="cuda:0")

    def execute(method: str) -> float:
        cache = restore_static_cache(prefix_snapshot, capacity, "cuda:0")
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            if method == "singleton":
                for token_id in candidate_ids:
                    token = torch.tensor([[token_id]], dtype=torch.long, device="cuda:0")
                    model(token, past_key_values=cache.cache, use_cache=True)
            elif method == "rowwise_sdpa_block":
                with rowwise_sdpa_and_lm_head(model):
                    model(token_block, past_key_values=cache.cache, use_cache=True)
            else:
                model(token_block, past_key_values=cache.cache, use_cache=True)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))

    methods = ("singleton", "block", "rowwise_sdpa_block")
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
        "scope": "paired compute-only forward timing; KV restore outside timer; not end-to-end decode latency",
        "model": model_name, "revision": revision, "prompt_id": args.prompt_id,
        "prompt_tokens": int(ids.shape[1]), "k": args.k,
        "physical_rows": len(candidate_ids), "candidate_ids": candidate_ids,
        "gpu": torch.cuda.get_device_name("cuda:0"),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "rounds": args.rounds, "warmups": args.warmups,
        "median_ms": medians,
        "samples_ms": samples,
        "speedup_vs_singleton": {
            method: medians["singleton"] / value for method, value in medians.items()
        },
        "notes": [
            "rowwise_sdpa_block issues one singleton-shaped SDPA call per candidate row and projects lm_head rowwise",
            "timing excludes cache restore and CPU hook copies",
            "correctness was checked separately by audit_operator_shape.py; this script does not recheck outputs",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "samples_ms"}, indent=2))


if __name__ == "__main__":
    main()
