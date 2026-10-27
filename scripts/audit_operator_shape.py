#!/usr/bin/env python3
"""Compare per-module tensors for batched and one-token verification.

This is an instrumented, same-prefix diagnostic. Hooks copy selected tensors
to CPU and change timings. A module with equal recorded *primary* input and
different output is a candidate intrinsic shape-sensitive operator, but
other unrecorded inputs or backend state can still matter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache

try:
    from run_kv_factorial_intervention import restore_static_cache, snapshot_cache
except ModuleNotFoundError:
    from scripts.run_kv_factorial_intervention import restore_static_cache, snapshot_cache


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = first_tensor(item)
            if found is not None:
                return found
    return None


@contextmanager
def capture_modules(model):
    calls = defaultdict(list)
    handles = []
    interesting = {"Linear4bit", "Qwen2RMSNorm", "LlamaRMSNorm", "Phi3RMSNorm",
                   "RMSNorm", "Qwen2Attention", "LlamaAttention", "Phi3Attention"}
    for name, module in model.named_modules():
        is_decoder_layer = re.fullmatch(r"model\.layers\.\d+", name) is not None
        is_attention_wrapper = name.endswith(".self_attn")
        if (module.__class__.__name__ not in interesting and name != "lm_head"
                and not is_decoder_layer and not is_attention_wrapper):
            continue

        def hook(_module, args, _kwargs, output, *, _name=name):
            arg = first_tensor(args)
            result = first_tensor(output)
            if arg is None or result is None or arg.ndim < 3 or result.ndim < 3:
                return
            calls[_name].append({
                "input": arg.detach().float().cpu().clone(),
                "output": result.detach().float().cpu().clone(),
            })

        handles.append(module.register_forward_hook(hook, with_kwargs=True))
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def run(model, snapshot, candidate_ids, capacity, device, sequential):
    cache = restore_static_cache(snapshot, capacity, device)
    with capture_modules(model) as calls:
        with torch.no_grad():
            if sequential:
                for token_id in candidate_ids:
                    token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                    model(token, past_key_values=cache.cache, use_cache=True)
            else:
                tokens = torch.tensor([candidate_ids], dtype=torch.long, device=device)
                model(tokens, past_key_values=cache.cache, use_cache=True)
    return calls


def compare_calls(batched, sequential, length: int) -> dict:
    rows = []
    for name, batch_calls in batched.items():
        seq_calls = sequential.get(name, [])
        if len(batch_calls) != 1 or len(seq_calls) != length:
            continue
        batch = batch_calls[0]
        row = {"module": name}
        for side in ("input", "output"):
            try:
                reference = torch.cat([entry[side] for entry in seq_calls], dim=-2)
                observed = batch[side]
                if observed.shape != reference.shape:
                    row[f"{side}_shape_mismatch"] = [list(observed.shape), list(reference.shape)]
                    continue
                delta = (observed - reference).abs()
                row[f"{side}_max_abs"] = float(delta.max())
                row[f"{side}_bitwise_equal"] = bool(torch.equal(observed, reference))
                if observed.ndim >= 3 and observed.shape[-2] == length:
                    reduce_dims = tuple(i for i in range(delta.ndim) if i != delta.ndim - 2)
                    row[f"{side}_max_abs_by_candidate_row"] = delta.amax(dim=reduce_dims).tolist()
            except (RuntimeError, ValueError) as error:
                row[f"{side}_comparison_error"] = str(error)
        rows.append(row)
    candidate = next((row for row in rows
                      if row.get("input_bitwise_equal") is True
                      and row.get("output_bitwise_equal") is False), None)
    return {
        "compared_modules": len(rows),
        "modules_with_different_outputs": sum(row.get("output_bitwise_equal") is False for row in rows),
        "modules_with_equal_primary_input_and_different_output": sum(
            row.get("input_bitwise_equal") is True and row.get("output_bitwise_equal") is False
            for row in rows),
        "first_equal_primary_input_different_output_in_module_traversal": candidate,
        "first_output_differences": [row for row in rows if row.get("output_bitwise_equal") is False][:20],
        "largest_output_differences": sorted(
            (row for row in rows if row.get("output_bitwise_equal") is False),
            key=lambda row: row.get("output_max_abs", 0.0), reverse=True,
        )[:20],
        "all_module_comparisons": rows,
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
    prompts = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line]
    prompt = next((row for row in prompts if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt ID {args.prompt_id!r} absent")
    loaded = load_backend(args.backend, args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model, loaded.tokenizer
    ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to(args.device)
    capacity = int(ids.shape[1]) + args.k + 8
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.no_grad():
        first = model(ids, past_key_values=cache.cache, use_cache=True).logits[0, -1]
    snapshot = snapshot_cache(cache.cache)
    candidate_ids = [int(first.argmax())]
    work = restore_static_cache(snapshot, capacity, args.device)
    with torch.no_grad():
        for _ in range(args.k):
            token = torch.tensor([[candidate_ids[-1]]], dtype=torch.long, device=args.device)
            result = model(token, past_key_values=work.cache, use_cache=True)
            candidate_ids.append(int(result.logits[0, -1].argmax()))
    sequential = run(model, snapshot, candidate_ids, capacity, args.device, True)
    batched = run(model, snapshot, candidate_ids, capacity, args.device, False)
    comparison = compare_calls(batched, sequential, len(candidate_ids))
    report = {
        "scope": "single-prefix diagnostic with CPU hook copies; not a production timing or causal proof",
        "model": args.model, "revision": args.revision,
        "backend": args.backend, "backend_metadata": loaded.metadata,
        "prompt_id": args.prompt_id, "prompt_length": int(ids.shape[1]),
        "prompt_token_ids_sha256": hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
        "candidate_ids": candidate_ids,
        "caveat": "Only the primary tensor input and first tensor output of each hooked module are compared; attention has additional state and arguments.",
        **comparison,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"all_module_comparisons", "first_output_differences",
                                     "largest_output_differences"}}, indent=2))


if __name__ == "__main__":
    main()
