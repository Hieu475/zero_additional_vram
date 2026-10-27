#!/usr/bin/env python3
"""Locate whether 4-bit linear shape changes explain a same-prefix drift.

This diagnostic executes each token through the original 4-bit linear kernel
as a one-token call, while retaining batched attention and other operators.
It is deliberately slow and makes no exactness claim. Comparing its logits
and KV with ordinary batched and sequential verification isolates the linear
component of shape-dependent execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import types
from contextlib import contextmanager
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
try:
    from run_kv_factorial_intervention import (
        kv_layer_errors, restore_static_cache, snapshot_cache,
    )
except ModuleNotFoundError:
    from scripts.run_kv_factorial_intervention import (
        kv_layer_errors, restore_static_cache, snapshot_cache,
    )


@contextmanager
def singleton_linear_rows(model):
    """Patch quantized linears and LM head on this model instance only."""
    originals = []
    counts = {"patched_modules": 0, "split_forward_calls": 0,
              "single_row_kernel_calls": 0}
    try:
        for name, module in model.named_modules():
            if module.__class__.__name__ != "Linear4bit" and name != "lm_head":
                continue
            original = module.forward

            def wrapped(self, hidden_states, *args, _original=original, **kwargs):
                if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim < 3:
                    return _original(hidden_states, *args, **kwargs)
                sequence = hidden_states.shape[-2]
                if sequence <= 1:
                    return _original(hidden_states, *args, **kwargs)
                counts["split_forward_calls"] += 1
                counts["single_row_kernel_calls"] += sequence
                outputs = [_original(hidden_states.narrow(-2, i, 1), *args, **kwargs)
                           for i in range(sequence)]
                return torch.cat(outputs, dim=-2)

            module.forward = types.MethodType(wrapped, module)
            originals.append((module, original))
        counts["patched_modules"] = len(originals)
        if not originals:
            raise RuntimeError("no quantized linear modules or lm_head found")
        yield counts
    finally:
        for module, original in originals:
            module.forward = original


def run_variant(model, source_snapshot, candidate_ids: list[int],
                capacity: int, device: str, sequential: bool):
    cache = restore_static_cache(source_snapshot, capacity, device)
    with torch.no_grad():
        if sequential:
            chunks = []
            for token_id in candidate_ids:
                token = torch.tensor([[token_id]], device=device, dtype=torch.long)
                chunks.append(model(token, past_key_values=cache.cache,
                                    use_cache=True).logits[0, -1].float().cpu())
            logits = torch.stack(chunks)
        else:
            ids = torch.tensor([candidate_ids], device=device, dtype=torch.long)
            logits = model(ids, past_key_values=cache.cache, use_cache=True).logits[0].float().cpu()
    torch.cuda.synchronize(device)
    return logits, snapshot_cache(cache.cache)


def compare(logits: torch.Tensor, cache, reference_logits: torch.Tensor,
            reference_cache) -> dict:
    difference = (logits - reference_logits).abs()
    layer_errors = kv_layer_errors(cache, reference_cache)
    key_error = max((row.get("key_max_abs", 0.0) for row in layer_errors), default=0.0)
    value_error = max((row.get("value_max_abs", 0.0) for row in layer_errors), default=0.0)
    return {
        "per_row_logit_max_abs": difference.max(dim=-1).values.tolist(),
        "argmax_equal_by_row": (logits.argmax(-1) == reference_logits.argmax(-1)).tolist(),
        "kv_key_max_abs": key_error,
        "kv_value_max_abs": value_error,
        "first_kv_different_layer": next((row["layer"] for row in layer_errors
                                          if row.get("key_max_abs", 0.0) > 0
                                          or row.get("value_max_abs", 0.0) > 0
                                          or row.get("length_mismatch")), None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    ap.add_argument("--backend", choices=["hf_bnb_nf4"], default="hf_bnb_nf4")
    ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl")
    ap.add_argument("--prompt-id", default="gsm8k_000")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.k < 1:
        raise ValueError("k must be positive")
    rows = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line]
    prompt = next((row for row in rows if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt ID {args.prompt_id!r} is absent")
    loaded = load_backend(args.backend, args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model, loaded.tokenizer
    ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to(args.device)
    capacity = int(ids.shape[1]) + args.k + 8
    prefill = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.no_grad():
        first = model(ids, past_key_values=prefill.cache, use_cache=True).logits[0, -1]
    source_snapshot = snapshot_cache(prefill.cache)
    candidate_ids = [int(first.argmax())]
    work_cache = restore_static_cache(source_snapshot, capacity, args.device)
    with torch.no_grad():
        for _ in range(args.k):
            token = torch.tensor([[candidate_ids[-1]]], device=args.device)
            out = model(token, past_key_values=work_cache.cache, use_cache=True)
            candidate_ids.append(int(out.logits[0, -1].argmax()))
    reference_logits, reference_cache = run_variant(
        model, source_snapshot, candidate_ids, capacity, args.device, sequential=True)
    ordinary_logits, ordinary_cache = run_variant(
        model, source_snapshot, candidate_ids, capacity, args.device, sequential=False)
    with singleton_linear_rows(model) as counts:
        normalized_logits, normalized_cache = run_variant(
            model, source_snapshot, candidate_ids, capacity, args.device, sequential=False)
    if counts["split_forward_calls"] == 0:
        raise RuntimeError("linear patch did not encounter a multi-token call")
    report = {
        "scope": "single-prefix diagnostic; no exactness guarantee and no production timing",
        "model": args.model, "revision": args.revision,
        "backend": args.backend, "backend_metadata": loaded.metadata,
        "prompt_id": args.prompt_id,
        "prompt_token_ids_sha256": hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
        "prompt_length": int(ids.shape[1]), "candidate_ids": candidate_ids,
        "candidate_length": len(candidate_ids), "capacity": capacity,
        "linear_singleton_counts": counts,
        "ordinary_batch_vs_sequential": compare(ordinary_logits, ordinary_cache,
                                                 reference_logits, reference_cache),
        "singleton_linear_batch_vs_sequential": compare(normalized_logits, normalized_cache,
                                                          reference_logits, reference_cache),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
