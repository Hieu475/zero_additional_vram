#!/usr/bin/env python3
"""Integrate the synthetic fixed-row Triton SDPA in one Qwen decode block."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_operator_shape import capture_modules, compare_calls, compare_cache_snapshots
from prototype_fixed_row_sdpa_triton import fixed_row_attention_tiled
try:
    from run_kv_factorial_intervention import restore_static_cache, snapshot_cache
except ModuleNotFoundError:
    from scripts.run_kv_factorial_intervention import restore_static_cache, snapshot_cache


@contextmanager
def fixed_row_sdpa(model, block_k, trace=None):
    original = F.scaled_dot_product_attention
    kv_heads = int(model.config.num_key_value_heads)

    def dispatch(query, key, value, *args, **kwargs):
        if key.shape[1] != query.shape[1]:
            groups = query.shape[1] // key.shape[1]
            key_full = key.repeat_interleave(groups, dim=1)
            value_full = value.repeat_interleave(groups, dim=1)
        else:
            key_full, value_full = key, value
        scale = kwargs.get("scale")
        if scale is None:
            scale = query.shape[-1] ** -0.5
        result = fixed_row_attention_tiled(query, key_full, value_full,
                                           scale, block_k)
        if trace is not None and not trace:
            trace.append({
                "query": query.detach().clone(),
                "key": key_full.detach().clone(),
                "value": value_full.detach().clone(),
                "output": result.detach().clone(),
                "scale": float(scale),
            })
        return result

    F.scaled_dot_product_attention = dispatch
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


@contextmanager
def rowwise_lm_head(model):
    original = model.lm_head.forward

    def dispatch(hidden_states, *args, **kwargs):
        if hidden_states.ndim >= 3 and hidden_states.shape[-2] > 1:
            return torch.cat([
                original(hidden_states[..., row:row + 1, :], *args, **kwargs)
                for row in range(hidden_states.shape[-2])
            ], dim=-2)
        return original(hidden_states, *args, **kwargs)

    model.lm_head.forward = dispatch
    try:
        yield
    finally:
        model.lm_head.forward = original


def run(model, prefix_snapshot, candidate_ids, capacity, sequential,
        use_triton, block_k):
    cache = restore_static_cache(prefix_snapshot, capacity, "cuda:0")
    trace = []
    with capture_modules(model) as calls, torch.inference_mode():
        if use_triton:
            with fixed_row_sdpa(model, block_k, trace):
                with rowwise_lm_head(model):
                    if sequential:
                        for token_id in candidate_ids:
                            token = torch.tensor([[token_id]], device="cuda:0")
                            model(token, past_key_values=cache.cache, use_cache=True)
                    else:
                        ids = torch.tensor([candidate_ids], device="cuda:0")
                        model(ids, past_key_values=cache.cache, use_cache=True)
        elif sequential:
            for token_id in candidate_ids:
                token = torch.tensor([[token_id]], device="cuda:0")
                model(token, past_key_values=cache.cache, use_cache=True)
        else:
            ids = torch.tensor([candidate_ids], device="cuda:0")
            model(ids, past_key_values=cache.cache, use_cache=True)
    return calls, snapshot_cache(cache.cache), trace


def logits_argmax(calls):
    entries = calls.get("lm_head", [])
    if not entries:
        return []
    tensors = [entry["output"] for entry in entries]
    logits = torch.cat(tensors, dim=-2).squeeze(0)
    return logits.argmax(dim=-1).tolist()


def compare_sequential_callsets(observed, reference, length):
    rows = []
    for name, observed_calls in observed.items():
        reference_calls = reference.get(name, [])
        if len(observed_calls) != length or len(reference_calls) != length:
            continue
        row = {"module": name}
        for side in ("input", "output"):
            try:
                actual = torch.cat([entry[side] for entry in observed_calls], dim=-2)
                expected = torch.cat([entry[side] for entry in reference_calls], dim=-2)
                if actual.shape != expected.shape:
                    row[f"{side}_shape_mismatch"] = [list(actual.shape), list(expected.shape)]
                    continue
                delta = (actual - expected).abs()
                row[f"{side}_bitwise_equal"] = bool(torch.equal(actual, expected))
                row[f"{side}_max_abs"] = float(delta.max())
                if actual.ndim >= 3 and actual.shape[-2] == length:
                    dims = tuple(i for i in range(delta.ndim) if i != delta.ndim - 2)
                    row[f"{side}_max_abs_by_row"] = delta.amax(dim=dims).tolist()
            except (RuntimeError, ValueError, KeyError) as error:
                row[f"{side}_comparison_error"] = str(error)
        rows.append(row)
    mismatches = [row for row in rows if row.get("output_bitwise_equal") is False]
    return {"compared_modules": len(rows),
            "modules_with_different_outputs": len(mismatches),
            "first_output_difference": mismatches[0] if mismatches else None,
            "all_module_comparisons": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/cnndm_eval.jsonl")
    parser.add_argument("--prompt-id", default="cnndm_000")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--block-k", type=int, default=32)
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
    prompt_cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.inference_mode():
        prefill_logits = model(ids, past_key_values=prompt_cache.cache,
                               use_cache=True).logits[:, -1]
    prefix_snapshot = snapshot_cache(prompt_cache.cache)

    pending = int(prefill_logits.argmax(dim=-1).item())
    candidate_ids = [pending]
    work = restore_static_cache(prefix_snapshot, capacity, "cuda:0")
    with fixed_row_sdpa(model, args.block_k), torch.inference_mode():
        for _ in range(args.k):
            token = torch.tensor([[candidate_ids[-1]]], device="cuda:0")
            logits = model(token, past_key_values=work.cache,
                           use_cache=True).logits[:, -1]
            candidate_ids.append(int(logits.argmax(dim=-1).item()))

    bnb_singleton, bnb_cache, _ = run(
        model, prefix_snapshot, candidate_ids, capacity, True, False, args.block_k)
    fixed_singleton, fixed_cache, singleton_trace = run(
        model, prefix_snapshot, candidate_ids, capacity, True, True, args.block_k)
    fixed_block, fixed_block_cache, block_trace = run(
        model, prefix_snapshot, candidate_ids, capacity, False, True, args.block_k)

    module_match = compare_calls(fixed_block, fixed_singleton, len(candidate_ids))
    fixed_kv_match = compare_cache_snapshots(fixed_block_cache, fixed_cache)
    bnb_module_match = compare_sequential_callsets(
        fixed_singleton, bnb_singleton, len(candidate_ids))
    bnb_kv_match = compare_cache_snapshots(fixed_cache, bnb_cache)
    single_trace, block_trace0 = singleton_trace[0], block_trace[0]
    valid0 = single_trace["key"].shape[-2]
    replay_from_block_row0 = fixed_row_attention_tiled(
        block_trace0["query"][:, :, :1].contiguous(),
        block_trace0["key"][:, :, :valid0].contiguous(),
        block_trace0["value"][:, :, :valid0].contiguous(),
        block_trace0["scale"], args.block_k)
    first_sdpa_inputs_comparison = {
        "query_row0_exact": torch.equal(
            block_trace0["query"][:, :, :1], single_trace["query"]),
        "key_prefix_row0_exact": torch.equal(
            block_trace0["key"][:, :, :valid0], single_trace["key"]),
        "value_prefix_row0_exact": torch.equal(
            block_trace0["value"][:, :, :valid0], single_trace["value"]),
        "row0_output_exact": torch.equal(
            block_trace0["output"][:, :, :1], single_trace["output"]),
        "key_lengths": {"block": block_trace0["key"].shape[-2],
                        "singleton": valid0},
        "dtypes": {key: str(block_trace0[key].dtype)
                   for key in ("query", "key", "value", "output")},
        "scales": {"block": block_trace0["scale"],
                   "singleton": single_trace["scale"]},
        "max_abs_output_row0": float((block_trace0["output"][:, :, :1]
                                       - single_trace["output"]).abs().max()),
        "replay_singleton_from_block_inputs_exact": torch.equal(
        replay_from_block_row0, block_trace0["output"][:, :, :1]),
        "replay_vs_block_row0_max_abs": float((replay_from_block_row0
                                                - block_trace0["output"][:, :, :1]).abs().max()),
        "replay_matches_singleton_reference": torch.equal(
            replay_from_block_row0, single_trace["output"]),
        "replay_vs_singleton_reference_max_abs": float((replay_from_block_row0
                                                         - single_trace["output"]).abs().max()),
    }
    payload = {
        "scope": "one fixed-prefix decode block; experimental B1 Triton attention, not BNB-equivalent claim",
        "model": model_name, "revision": revision, "prompt_id": args.prompt_id,
        "prompt_tokens": int(ids.shape[1]), "k": args.k,
        "physical_rows": len(candidate_ids), "block_k": args.block_k,
        "candidate_ids": candidate_ids,
        "fixed_block_next_ids": logits_argmax(fixed_block),
        "fixed_singleton_next_ids": logits_argmax(fixed_singleton),
        "bnb_singleton_next_ids": logits_argmax(bnb_singleton),
        "fixed_block_vs_fixed_singleton": module_match,
        "fixed_block_kv_vs_fixed_singleton": fixed_kv_match,
        "fixed_singleton_vs_original_bnb_singleton": bnb_module_match,
        "fixed_singleton_kv_vs_original_bnb_singleton": bnb_kv_match,
        "first_sdpa_inputs_comparison": first_sdpa_inputs_comparison,
        "limitations": [
            "prefill cache is produced by the original BNB path; only the decode transition uses the Triton kernel",
            "one prefix and one K value; not a long closed-loop sequence",
            "no latency measurement in this hook-instrumented audit",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "prompt_id": args.prompt_id,
        "candidate_ids": candidate_ids,
        "fixed_block_next_ids": payload["fixed_block_next_ids"],
        "fixed_singleton_next_ids": payload["fixed_singleton_next_ids"],
        "fixed_module_mismatches": module_match["modules_with_different_outputs"],
        "fixed_kv_exact": fixed_kv_match["all_layers_key_value_exact"],
        "bnb_module_mismatches": bnb_module_match["modules_with_different_outputs"],
        "bnb_kv_exact": bnb_kv_match["all_layers_key_value_exact"],
        "artifact": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
