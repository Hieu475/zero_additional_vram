#!/usr/bin/env python3
"""Audit one K=2/R=3 teacher-forced block with native SDPA on each causal row."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
import zassd.certification.fixed_row_dispatch as fixed_row_dispatch
from audit_actual_fixed_three_rows_bnb_first_decode import fixed_three_row_linear_schedule
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest


STRATEGY = "triton_nf4_fixed_three_rows_group_scale_packed"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


@contextmanager
def capture_sdpa_calls(records: list[dict]):
    original = F.scaled_dot_product_attention

    def spy(q, k, v, *args, **kwargs):
        out = original(q, k, v, *args, **kwargs)
        records.append({name: tensor.detach().cpu().clone()
                        for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out))})
        return out

    F.scaled_dot_product_attention = spy
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


@contextmanager
def capture_first_layer_modules(model, records: dict[str, dict]):
    wanted = {
        "model.layers.0.input_layernorm",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.post_attention_layernorm",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.act_fn",
        "model.layers.0.mlp.down_proj",
    }
    handles = []
    names = {id(module): name for name, module in model.named_modules()}
    for module in model.modules():
        name = names.get(id(module))
        if name not in wanted:
            continue

        def pre(_module, inputs, _name=name):
            records.setdefault(_name, {})["input"] = inputs[0].detach().cpu().clone()

        def post(_module, _inputs, output, _name=name):
            records.setdefault(_name, {})["output"] = output.detach().cpu().clone()

        handles.extend([module.register_forward_pre_hook(pre), module.register_forward_hook(post)])
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def native_bnb_qkv_linears(model):
    restored = []
    for layer in model.model.layers:
        for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                           layer.self_attn.v_proj):
            if not isinstance(projection, Linear4bit):
                raise TypeError("expected Q/K/V projections to be bitsandbytes Linear4bit")
            restored.append((projection, projection.forward))
            projection.forward = MethodType(Linear4bit.forward, projection)
    try:
        yield
    finally:
        for module, original in reversed(restored):
            module.forward = original


@contextmanager
def native_bnb_attention_linears(model):
    """Use BNB's own arithmetic for Q/K/V and attention-output projections."""
    restored = []
    for layer in model.model.layers:
        for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj,
                           layer.self_attn.v_proj, layer.self_attn.o_proj):
            if not isinstance(projection, Linear4bit):
                raise TypeError("expected Qwen2 attention projections to be Linear4bit")
            restored.append((projection, projection.forward))
            projection.forward = MethodType(Linear4bit.forward, projection)
    try:
        yield
    finally:
        for module, original in reversed(restored):
            module.forward = original


@contextmanager
def native_bnb_all_linears(model):
    """Use the original BNB Linear4bit forward for every quantized projection."""
    restored = []
    for module in model.modules():
        if isinstance(module, Linear4bit):
            restored.append((module, module.forward))
            module.forward = MethodType(Linear4bit.forward, module)
    try:
        yield
    finally:
        for module, original in reversed(restored):
            module.forward = original


def tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    af, bf = a.float(), b.float()
    delta = af - bf
    return {"shape_a": list(a.shape), "shape_b": list(b.shape),
            "bitwise_equal": bool(torch.equal(a, b)),
            "max_abs_diff": float(delta.abs().max().item()),
            "mean_abs_diff": float(delta.abs().mean().item())}


def compare_logits(ref: torch.Tensor, got: torch.Tensor) -> dict:
    a, b = ref.float(), got.float()
    delta = a - b
    return {
        "shape_reference": list(ref.shape), "shape_candidate": list(got.shape),
        "bitwise_equal": bool(torch.equal(ref, got)),
        "argmax_reference": int(ref.argmax(dim=-1).item()),
        "argmax_candidate": int(got.argmax(dim=-1).item()),
        "max_abs_diff": float(delta.abs().max().item()),
        "mean_abs_diff": float(delta.abs().mean().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="append", required=True)
    ap.add_argument("--artifact", action="append", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads(Path(args.manifest).read_text())
    source_name = "scripts/audit_actual_fixed_three_rows_native_rowwise_sdpa_block.py"
    if manifest.get("source_sha256", {}).get(source_name) != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("native-rowwise SDPA script is not pinned in manifest")
    prompts = {}
    for filename in args.benchmark:
        prompts.update({str(row["id"]): row["prompt"] for row in
                        (json.loads(line) for line in Path(filename).read_text().splitlines() if line.strip())})
    loaded = load_backend("hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct", REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    details = []
    for filename in args.artifact:
        artifact = Path(filename)
        if not artifact.is_absolute():
            artifact = repo / artifact
        saved = json.loads(artifact.read_text())
        pid = str(saved["prompt_id"])
        expected = [int(x) for x in saved["timing"]["paired_rounds"][0]["methods"]["original_bnb_greedy"]["token_ids"][:3]]
        if len(expected) != 3:
            raise ValueError(f"artifact has fewer than three BNB tokens: {pid}")
        ids = tokenizer(prompts[pid], return_tensors="pt")["input_ids"].to("cuda:0")
        cap = int(ids.shape[-1]) + 12
        bnb_cache = TargetKVCache(backend="static", max_capacity=cap)
        block_cache = TargetKVCache(backend="static", max_capacity=cap)
        native_qkv_cache = TargetKVCache(backend="static", max_capacity=cap)
        native_attention_cache = TargetKVCache(backend="static", max_capacity=cap)
        native_all_linear_cache = TargetKVCache(backend="static", max_capacity=cap)
        with torch.inference_mode():
            bnb_logits = model(ids, past_key_values=bnb_cache.cache, use_cache=True).logits[:, -1:, :]
            model(ids, past_key_values=block_cache.cache, use_cache=True)
            model(ids, past_key_values=native_qkv_cache.cache, use_cache=True)
            model(ids, past_key_values=native_attention_cache.cache, use_cache=True)
            model(ids, past_key_values=native_all_linear_cache.cache, use_cache=True)
        forced, bnb_rows, bnb_prefix_hashes, bnb_sdpa_records = [], [], [], []
        bnb_first_layer_modules: dict[str, dict] = {}
        for index in range(3):
            token_id = int(bnb_logits.argmax(dim=-1).item())
            forced.append(token_id)
            if token_id != expected[index]:
                raise RuntimeError(f"fresh BNB prefix disagrees with saved artifact for {pid} at {index}")
            token = torch.tensor([[token_id]], device="cuda:0")
            module_context = (capture_first_layer_modules(model, bnb_first_layer_modules)
                              if index == 0 else nullcontext())
            with capture_sdpa_calls(bnb_sdpa_records), module_context, torch.inference_mode():
                bnb_logits = model(token, past_key_values=bnb_cache.cache, use_cache=True).logits[:, -1:, :]
            bnb_rows.append(bnb_logits.detach().clone())
            prompt_len = int(ids.shape[-1])
            bnb_prefix_hashes.append(diagnostic_cache_prefix_digest(
                bnb_cache, prompt_len + index + 1
            ))

        block_tokens = torch.tensor([forced], device="cuda:0")
        block_sdpa_records = []
        block_first_layer_modules: dict[str, dict] = {}
        with (fixed_three_row_linear_schedule(model),
              fixed_row_dispatch.fixed_row_schedule_for_model(
                  model, physical_rows=3, attention_strategy="bnb_rowwise",
                  quantized_linear_strategy=STRATEGY),
              capture_sdpa_calls(block_sdpa_records),
              capture_first_layer_modules(model, block_first_layer_modules),
              torch.inference_mode()):
            block_logits = model(block_tokens, past_key_values=block_cache.cache,
                                 use_cache=True).logits
        native_qkv_modules: dict[str, dict] = {}
        with (fixed_three_row_linear_schedule(model),
              fixed_row_dispatch.fixed_row_schedule_for_model(
                  model, physical_rows=3, attention_strategy="bnb_rowwise",
                  quantized_linear_strategy=STRATEGY),
              native_bnb_qkv_linears(model),
              capture_first_layer_modules(model, native_qkv_modules),
              torch.inference_mode()):
            native_qkv_logits = model(block_tokens, past_key_values=native_qkv_cache.cache,
                                      use_cache=True).logits
        native_attention_modules: dict[str, dict] = {}
        with (fixed_three_row_linear_schedule(model),
              fixed_row_dispatch.fixed_row_schedule_for_model(
                  model, physical_rows=3, attention_strategy="bnb_rowwise",
                  quantized_linear_strategy=STRATEGY),
              native_bnb_attention_linears(model),
              capture_first_layer_modules(model, native_attention_modules),
              torch.inference_mode()):
            native_attention_logits = model(block_tokens,
                                            past_key_values=native_attention_cache.cache,
                                            use_cache=True).logits
        native_all_linear_modules: dict[str, dict] = {}
        with (fixed_three_row_linear_schedule(model),
              fixed_row_dispatch.fixed_row_schedule_for_model(
                  model, physical_rows=3, attention_strategy="bnb_rowwise",
                  quantized_linear_strategy=STRATEGY),
              native_bnb_all_linears(model),
              capture_first_layer_modules(model, native_all_linear_modules),
              torch.inference_mode()):
            native_all_linear_logits = model(block_tokens,
                                             past_key_values=native_all_linear_cache.cache,
                                             use_cache=True).logits
        prefix_results = []
        prompt_len = int(ids.shape[-1])
        for index in range(3):
            candidate_hash = diagnostic_cache_prefix_digest(block_cache, prompt_len + index + 1)
            prefix_results.append({
                "consumed_tokens": index + 1,
                "logits": compare_logits(bnb_rows[index], block_logits[:, index:index + 1, :]),
                "kv_sha256_reference": bnb_prefix_hashes[index],
                "kv_sha256_candidate": candidate_hash,
                "kv_equal": bnb_prefix_hashes[index] == candidate_hash,
            })
        native_qkv_prefixes = []
        for index in range(3):
            native_qkv_hash = diagnostic_cache_prefix_digest(
                native_qkv_cache, prompt_len + index + 1
            )
            mismatching_layers = []
            for layer_index, (ref_layer, candidate_layer) in enumerate(
                zip(bnb_cache.cache.layers, native_qkv_cache.cache.layers)
            ):
                end = prompt_len + index + 1
                if (not torch.equal(ref_layer.keys[:, :, :end, :],
                                    candidate_layer.keys[:, :, :end, :]) or
                        not torch.equal(ref_layer.values[:, :, :end, :],
                                        candidate_layer.values[:, :, :end, :])):
                    mismatching_layers.append(layer_index)
            native_qkv_prefixes.append({
                "consumed_tokens": index + 1,
                "logits": compare_logits(bnb_rows[index],
                                         native_qkv_logits[:, index:index + 1, :]),
                "kv_sha256_reference": bnb_prefix_hashes[index],
                "kv_sha256_candidate": native_qkv_hash,
                "kv_equal": bnb_prefix_hashes[index] == native_qkv_hash,
                "layer0_key_equal": torch.equal(
                    bnb_cache.cache.layers[0].keys[:, :, :prompt_len + index + 1, :],
                    native_qkv_cache.cache.layers[0].keys[:, :, :prompt_len + index + 1, :],
                ),
                "layer0_value_equal": torch.equal(
                    bnb_cache.cache.layers[0].values[:, :, :prompt_len + index + 1, :],
                    native_qkv_cache.cache.layers[0].values[:, :, :prompt_len + index + 1, :],
                ),
                "first_kv_layer_mismatch": mismatching_layers[0] if mismatching_layers else None,
            })
        native_qkv_module_trace = {}
        for name, ref_values in bnb_first_layer_modules.items():
            candidate_values = native_qkv_modules[name]
            native_qkv_module_trace[name] = {
                field: tensor_metrics(ref_values[field], candidate_values[field][:, :1, ...])
                for field in ("input", "output")
            }
        native_attention_prefixes = []
        for index in range(3):
            end = prompt_len + index + 1
            mismatches = []
            for layer_index, (ref_layer, candidate_layer) in enumerate(
                zip(bnb_cache.cache.layers, native_attention_cache.cache.layers)
            ):
                if (not torch.equal(ref_layer.keys[:, :, :end, :], candidate_layer.keys[:, :, :end, :]) or
                        not torch.equal(ref_layer.values[:, :, :end, :], candidate_layer.values[:, :, :end, :])):
                    mismatches.append(layer_index)
            native_attention_prefixes.append({
                "consumed_tokens": index + 1,
                "logits": compare_logits(bnb_rows[index],
                                         native_attention_logits[:, index:index + 1, :]),
                "first_kv_layer_mismatch": mismatches[0] if mismatches else None,
                "kv_equal": not mismatches,
            })
        native_attention_module_trace = {
            name: {field: tensor_metrics(bnb_first_layer_modules[name][field],
                                         native_attention_modules[name][field][:, :1, ...])
                   for field in ("input", "output")}
            for name in bnb_first_layer_modules
        }
        native_all_linear_prefixes = []
        for index in range(3):
            end = prompt_len + index + 1
            mismatches = []
            for layer_index, (ref_layer, candidate_layer) in enumerate(
                zip(bnb_cache.cache.layers, native_all_linear_cache.cache.layers)
            ):
                if (not torch.equal(ref_layer.keys[:, :, :end, :], candidate_layer.keys[:, :, :end, :]) or
                        not torch.equal(ref_layer.values[:, :, :end, :], candidate_layer.values[:, :, :end, :])):
                    mismatches.append(layer_index)
            native_all_linear_prefixes.append({
                "consumed_tokens": index + 1,
                "logits": compare_logits(bnb_rows[index],
                                         native_all_linear_logits[:, index:index + 1, :]),
                "first_kv_layer_mismatch": mismatches[0] if mismatches else None,
                "kv_equal": not mismatches,
            })
        native_all_linear_module_trace = {
            name: {field: tensor_metrics(bnb_first_layer_modules[name][field],
                                         native_all_linear_modules[name][field][:, :1, ...])
                   for field in ("input", "output")}
            for name in bnb_first_layer_modules
        }
        layer_count = int(model.config.num_hidden_layers)
        if len(bnb_sdpa_records) != 3 * layer_count or len(block_sdpa_records) != 3 * layer_count:
            raise RuntimeError(
                f"unexpected SDPA calls for {pid}: BNB={len(bnb_sdpa_records)}, "
                f"block={len(block_sdpa_records)}, expected={3 * layer_count}"
            )
        operator_trace = []
        for row in range(3):
            for layer in range(layer_count):
                ref_call = bnb_sdpa_records[row * layer_count + layer]
                block_call = block_sdpa_records[layer * 3 + row]
                operator_trace.append({
                    "row": row, "layer": layer,
                    "q": tensor_metrics(ref_call["q"], block_call["q"]),
                    "k": tensor_metrics(ref_call["k"], block_call["k"]),
                    "v": tensor_metrics(ref_call["v"], block_call["v"]),
                    "attention_output": tensor_metrics(ref_call["out"], block_call["out"]),
                })
        first_operator_drift = next((row for row in operator_trace if not all(
            row[field]["bitwise_equal"] for field in ("q", "k", "v", "attention_output")
        )), None)
        first_qkv_drift = next((row for row in operator_trace if not all(
            row[field]["bitwise_equal"] for field in ("q", "k", "v")
        )), None)
        first_layer_module_trace = {}
        for name, ref_values in bnb_first_layer_modules.items():
            candidate_values = block_first_layer_modules[name]
            first_layer_module_trace[name] = {}
            for field in ("input", "output"):
                ref_tensor = ref_values[field]
                candidate_tensor = candidate_values[field][:, :1, ...]
                first_layer_module_trace[name][field] = tensor_metrics(ref_tensor, candidate_tensor)
        q_name = "model.layers.0.self_attn.q_proj"
        q_module = dict(model.named_modules())[q_name]
        q_inputs = block_first_layer_modules[q_name]["input"].to("cuda:0").contiguous()
        from zassd.certification.nf4_fixed_three_rows_group_scale_packed import (
            nf4_fixed_three_rows_group_scale_packed_linear,
        )
        with torch.inference_mode():
            custom_m1 = nf4_fixed_three_rows_group_scale_packed_linear(
                q_inputs[:, :1, :].contiguous(), q_module,
                block_n=16, block_k=256, num_warps=1,
            )
            custom_m3 = nf4_fixed_three_rows_group_scale_packed_linear(
                q_inputs, q_module, block_n=16, block_k=256, num_warps=1,
            )[:, :1, :]
            bnb_m1 = q_module(q_inputs[:, :1, :].contiguous())
            bnb_m3 = q_module(q_inputs)
        q_proj_schedule_matrix = {
            "input_shape": list(q_inputs.shape),
            "custom_M1_vs_custom_M3_same_row0": tensor_metrics(custom_m1, custom_m3),
            "BNB_M1_vs_BNB_M3_same_row0": tensor_metrics(bnb_m1, bnb_m3[:, :1, :]),
            "custom_M1_vs_BNB_M1": tensor_metrics(custom_m1, bnb_m1),
            "custom_M3_vs_BNB_M1": tensor_metrics(custom_m3, bnb_m1),
        }
        details.append({
            "prompt_id": pid, "expected_bnb_tokens": expected, "fresh_bnb_tokens": forced,
            "tokens_match_saved_artifact": forced == expected,
            "block_call_input_shape": list(block_tokens.shape),
            "block_logits_shape": list(block_logits.shape),
            "per_prefix": prefix_results,
            "all_logits_argmax_match": all(x["logits"]["argmax_reference"] == x["logits"]["argmax_candidate"] for x in prefix_results),
            "all_logits_bitwise_match": all(x["logits"]["bitwise_equal"] for x in prefix_results),
            "all_kv_prefixes_match": all(x["kv_equal"] for x in prefix_results),
            "native_bnb_qkv_linear_variant": {
                "per_prefix": native_qkv_prefixes,
                "all_logits_argmax_match": all(
                    x["logits"]["argmax_reference"] == x["logits"]["argmax_candidate"]
                    for x in native_qkv_prefixes
                ),
                "all_logits_bitwise_match": all(x["logits"]["bitwise_equal"]
                                                 for x in native_qkv_prefixes),
                "all_kv_prefixes_match": all(x["kv_equal"] for x in native_qkv_prefixes),
                "all_layer0_key_prefixes_match": all(x["layer0_key_equal"] for x in native_qkv_prefixes),
                "all_layer0_value_prefixes_match": all(x["layer0_value_equal"] for x in native_qkv_prefixes),
                "first_layer_module_trace_row0": native_qkv_module_trace,
                "first_mismatching_kv_layer_by_prefix": [
                    row["first_kv_layer_mismatch"] for row in native_qkv_prefixes
                ],
            },
            "native_bnb_attention_linear_variant": {
                "per_prefix": native_attention_prefixes,
                "all_logits_argmax_match": all(
                    x["logits"]["argmax_reference"] == x["logits"]["argmax_candidate"]
                    for x in native_attention_prefixes
                ),
                "all_logits_bitwise_match": all(x["logits"]["bitwise_equal"]
                                                 for x in native_attention_prefixes),
                "all_kv_prefixes_match": all(x["kv_equal"] for x in native_attention_prefixes),
                "first_mismatching_kv_layer_by_prefix": [
                    row["first_kv_layer_mismatch"] for row in native_attention_prefixes
                ],
                "first_layer_module_trace_row0": native_attention_module_trace,
            },
            "native_bnb_all_linear_variant": {
                "per_prefix": native_all_linear_prefixes,
                "all_logits_argmax_match": all(
                    x["logits"]["argmax_reference"] == x["logits"]["argmax_candidate"]
                    for x in native_all_linear_prefixes
                ),
                "all_logits_bitwise_match": all(x["logits"]["bitwise_equal"]
                                                 for x in native_all_linear_prefixes),
                "all_kv_prefixes_match": all(x["kv_equal"] for x in native_all_linear_prefixes),
                "first_mismatching_kv_layer_by_prefix": [
                    row["first_kv_layer_mismatch"] for row in native_all_linear_prefixes
                ],
                "first_layer_module_trace_row0": native_all_linear_module_trace,
            },
            "sdpa_call_count_bnb": len(bnb_sdpa_records),
            "sdpa_call_count_block": len(block_sdpa_records),
            "first_sdpa_operator_drift": first_operator_drift,
            "first_sdpa_qkv_input_drift": first_qkv_drift,
            "first_layer_module_trace_row0": first_layer_module_trace,
            "q_proj_schedule_matrix": q_proj_schedule_matrix,
            "sdpa_per_row_layer_trace": operator_trace,
        })
        print(json.dumps({k: v for k, v in details[-1].items()
                          if k != "sdpa_per_row_layer_trace"}, indent=2), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({
        "classification": "K=2/R=3 teacher-forced block vs BNB singleton; per-row/per-layer SDPA trace",
        "model": "Qwen/Qwen2.5-3B-Instruct", "revision": REVISION,
        "gpu": torch.cuda.get_device_name(0), "quantized_linear_strategy": STRATEGY,
        "attention_strategy": "Qwen2 bnb_rowwise: native GQA SDPA, singleton RMSNorm and LM head per row",
        "details": details,
        "limitations": ["Two development prompts only.",
                        "Does not exercise the speculative decoder's commit/rollback state machine.",
                        "No held-out prompts used; finite evidence is not a proof."],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
