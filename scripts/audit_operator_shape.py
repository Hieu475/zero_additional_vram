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
    if isinstance(value, dict):
        for item in value.values():
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

        def hook(_module, args, kwargs, output, *, _name=name):
            # Transformers attention modules commonly receive hidden_states as
            # a keyword argument. Capturing args alone skips that operator and
            # can hide the first shape-dependent divergence before o_proj.
            arg = first_tensor(args)
            if arg is None:
                arg = first_tensor(kwargs)
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


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return None


@contextmanager
def capture_qwen2_attention_internals(num_key_value_heads, rowwise_block=False,
                                     row_parallel_padded=False):
    """Capture post-RoPE tensors and the SDPA call for diagnostic comparison."""
    from transformers.models.qwen2 import modeling_qwen2

    rotary_calls = []
    sdpa_calls = []
    original_rotary = modeling_qwen2.apply_rotary_pos_emb
    original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def rotary_wrapper(*args, **kwargs):
        result = original_rotary(*args, **kwargs)
        q_in = args[0] if len(args) > 0 else kwargs.get("q")
        k_in = args[1] if len(args) > 1 else kwargs.get("k")
        cos = args[2] if len(args) > 2 else kwargs.get("cos")
        sin = args[3] if len(args) > 3 else kwargs.get("sin")
        rotary_calls.append({
            "q_in": _cpu_copy(q_in), "k_in": _cpu_copy(k_in),
            "cos": _cpu_copy(cos), "sin": _cpu_copy(sin),
            "q_out": _cpu_copy(result[0]), "k_out": _cpu_copy(result[1]),
        })
        return result

    def sdpa_wrapper(query, key, value, *args, **kwargs):
        mask = kwargs.get("attn_mask", args[0] if args else None)
        unmodified_result = original_sdpa(query, key, value, *args, **kwargs)
        result = unmodified_result
        row_parallel_padded_output = None
        if rowwise_block and query.shape[-2] > 1:
            repeated_heads = query.shape[1] // num_key_value_heads
            compact_key = key[:, ::repeated_heads]
            compact_value = value[:, ::repeated_heads]
            row_outputs = []
            common_kwargs = dict(kwargs)
            common_kwargs.pop("attn_mask", None)
            common_kwargs.pop("enable_gqa", None)
            common_kwargs["is_causal"] = False
            common_kwargs["enable_gqa"] = True
            common_args = ()
            for row in range(query.shape[-2]):
                valid_length = key.shape[-2] - query.shape[-2] + row + 1
                row_outputs.append(original_sdpa(
                    query[:, :, row:row + 1],
                    compact_key[:, :, :valid_length],
                    compact_value[:, :, :valid_length],
                    *common_args, **common_kwargs))
            result = torch.cat(row_outputs, dim=-2)
        elif row_parallel_padded and query.shape[-2] > 1:
            if query.shape[0] != 1:
                raise ValueError("row-parallel padded diagnostic currently requires batch=1")
            repeated_heads = query.shape[1] // num_key_value_heads
            compact_key = key[:, ::repeated_heads]
            compact_value = value[:, ::repeated_heads]
            query_batch = query[0].permute(1, 0, 2).unsqueeze(2)
            key_batch = compact_key.expand(query.shape[-2], -1, -1, -1)
            value_batch = compact_value.expand(query.shape[-2], -1, -1, -1)
            kv_length = key.shape[-2]
            valid_lengths = (kv_length - query.shape[-2]
                             + torch.arange(1, query.shape[-2] + 1, device=query.device))
            positions = torch.arange(kv_length, device=query.device)
            row_mask = (positions[None, :] < valid_lengths[:, None])[:, None, None, :]
            row_parallel_padded_output = original_sdpa(
                query_batch, key_batch, value_batch,
                attn_mask=row_mask,
                dropout_p=kwargs.get("dropout_p", 0.0),
                scale=kwargs.get("scale"), is_causal=False, enable_gqa=True)
            result = (row_parallel_padded_output.squeeze(2)
                      .permute(1, 0, 2).unsqueeze(0))
        forced_math_output = None
        forced_math_error = None
        forced_math_expanded_output = None
        forced_math_expanded_error = None
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                forced_math_output = original_sdpa(query, key, value, *args, **kwargs)
        except (RuntimeError, TypeError) as error:
            forced_math_error = str(error)
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            expanded_key, expanded_value = key, value
            expanded_kwargs = dict(kwargs)
            if expanded_kwargs.pop("enable_gqa", False):
                repeats = query.shape[1] // key.shape[1]
                expanded_key = key.repeat_interleave(repeats, dim=1)
                expanded_value = value.repeat_interleave(repeats, dim=1)
            with sdpa_kernel(SDPBackend.MATH):
                forced_math_expanded_output = original_sdpa(
                    query, expanded_key, expanded_value, *args, **expanded_kwargs)
        except (RuntimeError, TypeError) as error:
            forced_math_expanded_error = str(error)
        sdpa_calls.append({
            "query": _cpu_copy(query), "key": _cpu_copy(key),
            "value": _cpu_copy(value), "mask": _cpu_copy(mask),
            "output": _cpu_copy(result),
            "unmodified_output": _cpu_copy(unmodified_result),
            "row_parallel_padded_output": _cpu_copy(row_parallel_padded_output),
            "forced_math_output": _cpu_copy(forced_math_output),
            "forced_math_expanded_output": _cpu_copy(forced_math_expanded_output),
            "forced_math_error": forced_math_error,
            "forced_math_expanded_error": forced_math_expanded_error,
            "is_causal": kwargs.get("is_causal"),
            "dropout_p": kwargs.get("dropout_p"),
            "scale": kwargs.get("scale"),
            "enable_gqa": kwargs.get("enable_gqa"),
        })
        return result

    modeling_qwen2.apply_rotary_pos_emb = rotary_wrapper
    torch.nn.functional.scaled_dot_product_attention = sdpa_wrapper
    try:
        yield {"rotary": rotary_calls, "sdpa": sdpa_calls}
    finally:
        modeling_qwen2.apply_rotary_pos_emb = original_rotary
        torch.nn.functional.scaled_dot_product_attention = original_sdpa


def _compare_tensor(block_tensor, singleton_tensor, row, sequence_axis=-2,
                    key_axis=False):
    if block_tensor is None or singleton_tensor is None:
        return {"equal": block_tensor is None and singleton_tensor is None,
                "block_none": block_tensor is None,
                "singleton_none": singleton_tensor is None}
    block = block_tensor
    singleton = singleton_tensor
    axis = sequence_axis if sequence_axis >= 0 else block.ndim + sequence_axis
    if axis < 0 or axis >= block.ndim:
        return {"error": "missing sequence dimension"}
    if key_axis:
        if sequence_axis == -1 and block.ndim >= 2 and block.shape[-2] > 1:
            block = block.narrow(-2, row, 1)
        valid_length = singleton.shape[sequence_axis]
        block = block.narrow(sequence_axis, 0, valid_length)
    elif block.shape[sequence_axis] > 1:
        block = block.narrow(sequence_axis, row, 1)
    if list(block.shape) != list(singleton.shape):
        return {"equal": False, "block_shape": list(block.shape),
                "singleton_shape": list(singleton.shape)}
    equal = torch.equal(block, singleton)
    delta = (block.float() - singleton.float()).abs()
    return {"equal": equal, "max_abs": float(delta.max()) if delta.numel() else 0.0,
            "shape": list(block.shape)}


def compare_attention_internals(block, sequential, length):
    result = {"rows": [], "rotary_call_count": len(block["rotary"]),
              "sdpa_call_count": len(block["sdpa"])}
    for stage in ("rotary", "sdpa"):
        block_calls = block[stage]
        seq_calls = sequential[stage]
        layer_count = len(block_calls)
        stage_result = []
        if layer_count == 0 or len(seq_calls) != length * layer_count:
            result[stage] = {"status": "unexpected_call_count",
                             "block_calls": layer_count,
                             "singleton_calls": len(seq_calls)}
            continue
        for layer_idx, block_call in enumerate(block_calls):
            for row in range(length):
                seq_call = seq_calls[row * layer_count + layer_idx]
                fields = ("q_in", "k_in", "cos", "sin", "q_out", "k_out") if stage == "rotary" else (
                    "query", "key", "value", "mask", "output", "forced_math_output",
                    "forced_math_expanded_output", "row_parallel_padded_output")
                comparisons = {}
                for field in fields:
                    if stage == "sdpa" and field == "row_parallel_padded_output":
                        output = block_call[field]
                        comparisons[field] = (
                            _compare_tensor(output[row:row + 1], seq_call["output"], row, -2)
                            if output is not None else {"equal": None})
                        continue
                    # Q/K/V and context use the token axis at -2. Cached K/V
                    # and attention masks compare only the prefix visible to
                    # this sequential token.
                    block_tensor = block_call[field]
                    singleton_tensor = seq_call[field]
                    if (stage == "sdpa" and field in ("key", "value")
                            and block_tensor is not None and singleton_tensor is not None
                            and block_tensor.ndim == 4 and singleton_tensor.ndim == 4
                            and block_tensor.shape[1] > singleton_tensor.shape[1]
                            and block_tensor.shape[1] % singleton_tensor.shape[1] == 0):
                        # Qwen's block path materializes repeated GQA heads;
                        # singleton SDPA can consume compact K/V with enable_gqa.
                        repeat = block_tensor.shape[1] // singleton_tensor.shape[1]
                        block_tensor = block_tensor[:, ::repeat]
                    if (stage == "sdpa" and field == "mask"
                            and block_tensor is not None and singleton_tensor is None):
                        visible = block_tensor
                        if visible.ndim >= 2 and visible.shape[-2] > 1:
                            visible = visible.narrow(-2, row, 1)
                        valid_keys = seq_call["key"].shape[-2]
                        visible = visible.narrow(-1, 0, valid_keys)
                        if visible.dtype == torch.bool:
                            equivalent = bool(visible.all())
                        else:
                            equivalent = bool(torch.equal(visible,
                                                          torch.zeros_like(visible)))
                        comparisons[field] = {
                            "equal": equivalent,
                            "equivalent_to_no_mask": equivalent,
                            "shape": list(visible.shape),
                            "max_abs": float(visible.float().abs().max()) if visible.numel() else 0.0,
                        }
                        continue
                    key_axis = stage == "sdpa" and field in ("key", "value", "mask")
                    axis = -1 if field == "mask" else -2
                    comparisons[field] = _compare_tensor(
                        block_tensor, singleton_tensor, row, axis, key_axis)
                metadata = {}
                if stage == "sdpa":
                    for field in ("is_causal", "dropout_p", "scale", "enable_gqa"):
                        metadata[field] = {"block": block_call[field],
                                           "singleton": seq_call[field]}
                    metadata["forced_math_error"] = {
                        "block": block_call["forced_math_error"],
                        "singleton": seq_call["forced_math_error"],
                    }
                    metadata["forced_math_expanded_error"] = {
                        "block": block_call["forced_math_expanded_error"],
                        "singleton": seq_call["forced_math_expanded_error"],
                    }
                stage_result.append({"layer": layer_idx, "row": row,
                                     "comparisons": comparisons,
                                     "metadata": metadata})
        result[stage] = stage_result
    return result


def run(model, snapshot, candidate_ids, capacity, device, sequential,
        rowwise_sdpa=False, rowwise_lm_head=False,
        row_parallel_padded=False):
    cache = restore_static_cache(snapshot, capacity, device)
    original_lm_head_forward = model.lm_head.forward
    def lm_head_rowwise(hidden_states, *args, **kwargs):
        if hidden_states.ndim >= 3 and hidden_states.shape[-2] > 1:
            return torch.cat([
                original_lm_head_forward(
                    hidden_states[..., row:row + 1, :], *args, **kwargs)
                for row in range(hidden_states.shape[-2])
            ], dim=-2)
        return original_lm_head_forward(hidden_states, *args, **kwargs)
    if rowwise_lm_head:
        model.lm_head.forward = lm_head_rowwise
    try:
        with capture_modules(model) as calls, capture_qwen2_attention_internals(
                model.config.num_key_value_heads, rowwise_sdpa,
                row_parallel_padded) as internals:
            with torch.no_grad():
                if sequential:
                    for token_id in candidate_ids:
                        token = torch.tensor([[token_id]], dtype=torch.long, device=device)
                        model(token, past_key_values=cache.cache, use_cache=True)
                else:
                    tokens = torch.tensor([candidate_ids], dtype=torch.long, device=device)
                    model(tokens, past_key_values=cache.cache, use_cache=True)
    finally:
        if rowwise_lm_head:
            model.lm_head.forward = original_lm_head_forward
    return calls, internals, snapshot_cache(cache.cache)


def compare_cache_snapshots(observed, reference):
    layers = []
    for index, (obs, ref) in enumerate(zip(observed, reference)):
        obs_key, obs_value, obs_length = obs
        ref_key, ref_value, ref_length = ref
        key_equal = obs_length == ref_length and torch.equal(obs_key, ref_key)
        value_equal = obs_length == ref_length and torch.equal(obs_value, ref_value)
        key_delta = (float((obs_key.float() - ref_key.float()).abs().max())
                     if obs_key.shape == ref_key.shape and obs_key.numel() else None)
        value_delta = (float((obs_value.float() - ref_value.float()).abs().max())
                       if obs_value.shape == ref_value.shape and obs_value.numel() else None)
        layers.append({"layer": index, "observed_length": obs_length,
                       "reference_length": ref_length, "key_equal": key_equal,
                       "value_equal": value_equal, "key_max_abs": key_delta,
                       "value_max_abs": value_delta})
    return {"layer_count_equal": len(observed) == len(reference),
            "all_layers_key_value_exact": len(observed) == len(reference) and all(
                row["key_equal"] and row["value_equal"] for row in layers),
            "layers": layers}


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
    sequential, sequential_internal, sequential_cache = run(
        model, snapshot, candidate_ids, capacity, args.device, True)
    batched, batched_internal, batched_cache = run(
        model, snapshot, candidate_ids, capacity, args.device, False)
    rowwise_batched, rowwise_batched_internal, rowwise_batched_cache = run(
        model, snapshot, candidate_ids, capacity, args.device, False,
        rowwise_sdpa=True, rowwise_lm_head=True)
    parallel_batched, parallel_batched_internal, parallel_batched_cache = run(
        model, snapshot, candidate_ids, capacity, args.device, False,
        rowwise_lm_head=True, row_parallel_padded=True)
    comparison = compare_calls(batched, sequential, len(candidate_ids))
    attention_internal_comparison = compare_attention_internals(
        batched_internal, sequential_internal, len(candidate_ids))
    rowwise_sdpa_comparison = compare_calls(
        rowwise_batched, sequential, len(candidate_ids))
    rowwise_sdpa_internal_comparison = compare_attention_internals(
        rowwise_batched_internal, sequential_internal, len(candidate_ids))
    batched_kv_comparison = compare_cache_snapshots(batched_cache, sequential_cache)
    rowwise_sdpa_kv_comparison = compare_cache_snapshots(
        rowwise_batched_cache, sequential_cache)
    row_parallel_padded_comparison = compare_calls(
        parallel_batched, sequential, len(candidate_ids))
    row_parallel_padded_internal_comparison = compare_attention_internals(
        parallel_batched_internal, sequential_internal, len(candidate_ids))
    row_parallel_padded_kv_comparison = compare_cache_snapshots(
        parallel_batched_cache, sequential_cache)
    report = {
        "scope": "single-prefix diagnostic with CPU hook copies; not a production timing or causal proof",
        "model": args.model, "revision": args.revision,
        "backend": args.backend, "backend_metadata": loaded.metadata,
        "prompt_id": args.prompt_id, "prompt_length": int(ids.shape[1]),
        "prompt_token_ids_sha256": hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
        "candidate_ids": candidate_ids,
        "caveat": "Module hooks compare primary tensor inputs and first tensor outputs; explicit Qwen2 instrumentation also records RoPE and SDPA tensors. This is a finite fixed-prefix diagnostic with CPU copies, not a general proof or timing result.",
        "attention_internal_comparison": attention_internal_comparison,
        "rowwise_sdpa_module_comparison": rowwise_sdpa_comparison,
        "rowwise_sdpa_internal_comparison": rowwise_sdpa_internal_comparison,
        "batched_kv_comparison": batched_kv_comparison,
        "rowwise_sdpa_kv_comparison": rowwise_sdpa_kv_comparison,
        "row_parallel_padded_module_comparison": row_parallel_padded_comparison,
        "row_parallel_padded_internal_comparison": row_parallel_padded_internal_comparison,
        "row_parallel_padded_kv_comparison": row_parallel_padded_kv_comparison,
        **comparison,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"all_module_comparisons", "first_output_differences",
                                     "largest_output_differences",
                                     "attention_internal_comparison",
                                     "rowwise_sdpa_module_comparison",
                                     "rowwise_sdpa_internal_comparison",
                                     "batched_kv_comparison",
                                     "rowwise_sdpa_kv_comparison",
                                     "row_parallel_padded_module_comparison",
                                     "row_parallel_padded_internal_comparison",
                                     "row_parallel_padded_kv_comparison"}}, indent=2))


if __name__ == "__main__":
    main()
