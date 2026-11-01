#!/usr/bin/env python3
"""Compare BNB fused NF4 GEMM with dense linear on identical real activations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F
from bitsandbytes.functional import dequantize_4bit
from bitsandbytes.nn import Linear4bit

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.certification.nf4_fixed_three_rows_group_scale_packed import (
    nf4_fixed_three_rows_group_scale_packed_linear,
)


MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


@contextmanager
def original_bnb_forward(model):
    saved = []
    for module in model.modules():
        if isinstance(module, Linear4bit):
            saved.append((module, module.forward))
            module.forward = MethodType(Linear4bit.forward, module)
    try:
        yield
    finally:
        for module, forward in reversed(saved):
            module.forward = forward


def compare(a: torch.Tensor, b: torch.Tensor) -> dict:
    delta = (a.float() - b.float()).abs()
    return {"bitwise_equal": torch.equal(a, b),
            "max_abs_diff": float(delta.max().item()),
            "mean_abs_diff": float(delta.mean().item()),
            "shape": list(a.shape), "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}


def time_ms(fn, iterations: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def direct_dequant_to_compute_dtype(
    module, dtype: torch.dtype, round_code: bool = False,
    round_scale: bool = False, chunk_size: int = 1 << 20,
):
    """Reconstruct packed NF4 weights with selectable operand rounding."""
    state = module.weight.quant_state
    out_features, in_features = (int(x) for x in state.shape)
    total = out_features * in_features
    packed = module.weight.data.reshape(-1)
    result = torch.empty((total,), device=packed.device, dtype=dtype)
    for start in range(0, total, chunk_size):
        end = min(total, start + chunk_size)
        indices = torch.arange(start, end, device=packed.device, dtype=torch.long)
        byte = packed[indices // 2].to(torch.int32)
        nibble = torch.where(indices % 2 == 0, byte >> 4, byte & 0x0F).long()
        code = state.code[nibble].float()
        scale_ids = indices // int(state.blocksize)
        if state.nested:
            qscale = state.absmax[scale_ids].long()
            nested_ids = scale_ids // int(state.state2.blocksize)
            nested_code = state.state2.code[qscale].float()
            nested_scale = state.state2.absmax[nested_ids].float()
            scale = nested_code * nested_scale + state.offset.float()
        else:
            scale = state.absmax[scale_ids].float()
        if round_code:
            code = code.to(dtype).float()
        if round_scale:
            scale = scale.to(dtype).float()
        result[start:end] = (code * scale).to(dtype)
    return result.view(out_features, in_features)


class Capture:
    def __init__(self, modules: dict[str, torch.nn.Module]):
        self.modules = modules
        self.inputs = {}
        self.handles = []

    def __enter__(self):
        for name, module in self.modules.items():
            def hook(_module, inputs, name=name):
                self.inputs[name] = inputs[0].detach().clone()
            self.handles.append(module.register_forward_pre_hook(hook))
        return self.inputs

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 17, 35])
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((x for x in records if str(x["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(f"prompt {args.prompt_id!r} not found")

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    if any(i < 0 or i >= len(model.model.layers) for i in args.layers):
        raise ValueError("requested layer index out of range")

    modules = {}
    for layer_idx in args.layers:
        layer = model.model.layers[layer_idx]
        for projection in ("gate_proj", "up_proj"):
            modules[f"layers.{layer_idx}.mlp.{projection}"] = getattr(layer.mlp, projection)

    # Establish four tokens using ordinary singleton BNB, then verify the first
    # three rows as one M=3 block while capturing the exact MLP inputs.
    token_cache = TargetKVCache(backend="static", max_capacity=prompt_len + 8)
    with torch.inference_mode():
        logits = model(prompt_ids, past_key_values=token_cache.cache, use_cache=True).logits[:, -1:, :]
        forced_tokens = []
        for _ in range(4):
            token = logits.argmax(dim=-1)
            forced_tokens.append(int(token.item()))
            logits = model(token, past_key_values=token_cache.cache, use_cache=True).logits[:, -1:, :]

    block_cache = TargetKVCache(backend="static", max_capacity=prompt_len + 8)
    with torch.inference_mode():
        model(prompt_ids, past_key_values=block_cache.cache, use_cache=True)
    block_tokens = torch.tensor([forced_tokens[:3]], device="cuda:0", dtype=torch.long)
    capture = Capture(modules)
    with (fixed_row_schedule_for_model(model, physical_rows=3,
            attention_strategy="bnb_rowwise", quantized_linear_strategy="fixed_rows"),
          original_bnb_forward(model), capture, torch.inference_mode()):
        block_logits = model(block_tokens, past_key_values=block_cache.cache, use_cache=True).logits
    if len(capture.inputs) != len(modules):
        raise RuntimeError(f"captured {len(capture.inputs)} of {len(modules)} requested modules")

    torch.set_grad_enabled(False)
    results = {}
    for name, x in capture.inputs.items():
        module = modules[name]
        state = module.weight.quant_state
        compute_dtype = module.compute_dtype
        if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise RuntimeError(f"unexpected BNB compute dtype for {name}: {compute_dtype}")
        bnb_output = module(x).detach()
        triton_output = nf4_fixed_three_rows_group_scale_packed_linear(x, module).detach()

        # Public dequantization rounds to quant_state.dtype before compute dtype.
        dense_weight_stored = dequantize_4bit(module.weight.data, quant_state=state)
        dense_weight_public = dense_weight_stored.reshape(state.shape).to(compute_dtype)
        dense_weight_direct = direct_dequant_to_compute_dtype(module, compute_dtype)
        dense_input = x.to(compute_dtype)
        dense_bias = module.bias
        if dense_bias is not None:
            dense_bias = dense_bias.to(compute_dtype)
        dense_public_output = F.linear(dense_input, dense_weight_public, dense_bias).to(x.dtype).detach()
        dense_direct_output = F.linear(dense_input, dense_weight_direct, dense_bias).to(x.dtype).detach()
        rounding_variants = {}
        for variant, round_code, round_scale in (
            ("fp16_code_only", True, False),
            ("fp16_scale_only", False, True),
            ("fp16_code_and_scale", True, True),
        ):
            variant_weight = direct_dequant_to_compute_dtype(
                module, compute_dtype, round_code=round_code, round_scale=round_scale
            )
            variant_output = F.linear(dense_input, variant_weight, dense_bias).to(x.dtype)
            rounding_variants[variant] = {
                "bnb_fused_vs_dense": compare(bnb_output, variant_output),
                "triton_vs_dense": compare(triton_output, variant_output),
            }
            del variant_weight, variant_output

        functions = {
            "bnb_fused": lambda x=x, module=module: module(x),
            "dense_public_dequant": lambda a=dense_input, w=dense_weight_public, b=dense_bias, out_dtype=x.dtype: F.linear(a, w, b).to(out_dtype),
            "dense_direct_fp32_to_compute": lambda a=dense_input, w=dense_weight_direct, b=dense_bias, out_dtype=x.dtype: F.linear(a, w, b).to(out_dtype),
            "triton_nf4": lambda x=x, module=module: nf4_fixed_three_rows_group_scale_packed_linear(x, module),
        }
        samples = {key: [] for key in functions}
        for round_index in range(args.rounds):
            order = list(functions)
            if round_index % 2:
                order.reverse()
            for key in order:
                samples[key].append(time_ms(functions[key], args.iterations))
        medians = {key: statistics.median(value) for key, value in samples.items()}
        results[name] = {
            "activation_shape": list(x.shape), "activation_dtype": str(x.dtype),
            "bnb_compute_dtype": str(compute_dtype),
            "quant_state_weight_dtype": str(state.dtype),
            "public_dequant_weight_vs_direct_compute_weight": compare(dense_weight_public, dense_weight_direct),
            "bnb_fused_vs_dense_public_dequant": compare(bnb_output, dense_public_output),
            "bnb_fused_vs_dense_direct_fp32_to_compute": compare(bnb_output, dense_direct_output),
            "triton_vs_dense_direct_fp32_to_compute": compare(triton_output, dense_direct_output),
            "dequant_operand_rounding_variants": rounding_variants,
            "bnb_fused_vs_triton": compare(bnb_output, triton_output),
            "dense_dequant_vs_quant_state_dtype": str(dense_weight_stored.dtype),
            "timing": {"samples_ms": samples, "median_ms": medians,
                       "dense_public_dequant_over_bnb_ratio": medians["dense_public_dequant"] / medians["bnb_fused"],
                       "dense_direct_over_bnb_ratio": medians["dense_direct_fp32_to_compute"] / medians["bnb_fused"],
                       "triton_over_bnb_ratio": medians["triton_nf4"] / medians["bnb_fused"]},
        }
        del dense_weight_stored, dense_weight_public, dense_weight_direct, dense_input
        del dense_public_output, dense_direct_output, bnb_output, triton_output
        torch.cuda.empty_cache()

    payload = {
        "schema": "zassd-actual-bnb-nf4-gemm-vs-dense-v1",
        "classification": "finite direct operator comparison on model-generated M=3 activations; not an end-to-end benchmark or proof",
        "model": MODEL, "revision": REVISION,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "prompt_id": args.prompt_id, "prompt_tokens": prompt_len,
        "forced_bnb_tokens": forced_tokens,
        "block_predictions": [int(x) for x in block_logits.argmax(dim=-1).flatten().tolist()],
        "block_predictions_match_bnb_continuation": [int(x) for x in block_logits.argmax(dim=-1).flatten().tolist()] == forced_tokens[1:],
        "layers": args.layers,
        "activation_protocol": "Capture gate/up inputs during one M=3 block with original BNB linears and native rowwise SDPA. Compare BNB fused GEMM, dense F.linear over public BNB dequantized weights, dense F.linear over direct FP32 NF4 reconstruction rounded once to BNB compute dtype, and fixed-three-row Triton on the exact same activation tensor.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_protocol": {"iterations_per_round": args.iterations, "alternating_rounds": args.rounds,
                            "dense_weight_dequantization_excluded": True},
        "results": results,
        "limits": [
            "Only selected gate/up layers and one three-token block per prompt.",
            "Neither dense path is guaranteed to reproduce the BNB fused kernel's internal dequantization or reduction schedule.",
            "CUDA event timings are direct operator timings; they do not predict decoder latency.",
            "GSM8K held-out indices 300-349 were not used.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "prompt_id": args.prompt_id,
        "block_predictions_match_bnb": payload["block_predictions_match_bnb_continuation"],
        "results": {name: {"bnb_dense_direct_equal": value["bnb_fused_vs_dense_direct_fp32_to_compute"]["bitwise_equal"],
                           "bnb_dense_direct_max_abs": value["bnb_fused_vs_dense_direct_fp32_to_compute"]["max_abs_diff"],
                           "bnb_ms": value["timing"]["median_ms"]["bnb_fused"],
                           "dense_direct_ms": value["timing"]["median_ms"]["dense_direct_fp32_to_compute"],
                           "triton_ms": value["timing"]["median_ms"]["triton_nf4"]}
                   for name, value in results.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
