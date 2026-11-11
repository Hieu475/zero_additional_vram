#!/usr/bin/env python3
"""Screen reduction schedules against BNB on a real M=3 gate projection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
from bitsandbytes.functional import dequantize_4bit
from bitsandbytes.nn import Linear4bit
import triton
import triton.language as tl

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model


MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


@triton.jit
def _strided_lane_accum_kernel(x_ptr, w_ptr, y_ptr,
                               ROWS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                               LANES: tl.constexpr, USE_FMA: tl.constexpr,
                               ROUND_PRODUCT_FP16: tl.constexpr):
    row = tl.program_id(0)
    out = tl.program_id(1)
    lane = tl.arange(0, LANES)
    acc = tl.zeros((LANES,), dtype=tl.float32)
    for tile in range(tl.cdiv(K, LANES)):
        k = tile * LANES + lane
        x = tl.load(x_ptr + row * K + k, mask=(row < ROWS) & (k < K), other=0).to(tl.float32)
        w = tl.load(w_ptr + out * K + k, mask=(out < N) & (k < K), other=0).to(tl.float32)
        if ROUND_PRODUCT_FP16:
            product = (x * w).to(tl.float16).to(tl.float32)
            acc += product
        elif USE_FMA:
            acc = tl.fma(x, w, acc)
        else:
            acc += x * w
    total = tl.sum(acc, axis=0)
    tl.store(y_ptr + row * N + out, total.to(tl.float16), (row < ROWS) & (out < N))


@triton.jit
def _bnb_simt_lane_schedule_kernel(x_ptr, w_ptr, bias_ptr, y_ptr,
                                  ROWS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                  HAS_BIAS: tl.constexpr):
    """Mirror CUDA SIMT FP16 inner loop and its 32-lane down-shuffle tree."""
    row = tl.program_id(0)
    out = tl.program_id(1)
    lane = tl.arange(0, 32)
    acc = tl.zeros((32,), dtype=tl.float32)
    for group in range(tl.cdiv(K, 1024)):
        for pair in tl.static_range(16):
            k = group * 1024 + lane * 32 + pair * 2
            x0 = tl.load(x_ptr + row * K + k, mask=(row < ROWS) & (k < K), other=0).to(tl.float16).to(tl.float32)
            x1 = tl.load(x_ptr + row * K + k + 1, mask=(row < ROWS) & (k + 1 < K), other=0).to(tl.float16).to(tl.float32)
            w0 = tl.load(w_ptr + out * K + k, mask=(out < N) & (k < K), other=0).to(tl.float16).to(tl.float32)
            w1 = tl.load(w_ptr + out * K + k + 1, mask=(out < N) & (k + 1 < K), other=0).to(tl.float16).to(tl.float32)
            p0 = (x0 * w0).to(tl.float16).to(tl.float32)
            p1 = (x1 * w1).to(tl.float16).to(tl.float32)
            pair_sum = p0 + p1
            acc = acc + pair_sum
    for offset in tl.static_range(5):
        shift = 16 >> offset
        source_lane = (lane + shift) & 31
        other = tl.gather(acc, source_lane, axis=0)
        acc = tl.where(lane < 32 - shift, acc + other, acc)
    total = tl.sum(tl.where(lane == 0, acc, 0.0), axis=0)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + out, mask=out < N, other=0).to(tl.float32)
        total += bias
    tl.store(y_ptr + row * N + out, total.to(tl.float16), (row < ROWS) & (out < N))


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


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    delta = (reference.float() - candidate.float()).abs()
    return {"exact_elements": int(torch.count_nonzero(reference == candidate).item()),
            "total_elements": reference.numel(),
            "max_abs_diff": float(delta.max().item()),
            "mean_abs_diff": float(delta.mean().item())}


def direct_selected_weight(module, output_rows: int, compute_dtype: torch.dtype) -> torch.Tensor:
    state = module.weight.quant_state
    _, in_features = (int(x) for x in state.shape)
    rows = torch.arange(output_rows, device="cuda:0", dtype=torch.long)[:, None]
    cols = torch.arange(in_features, device="cuda:0", dtype=torch.long)[None, :]
    flat = rows * in_features + cols
    packed = module.weight.data.reshape(-1)
    byte = packed[flat // 2].to(torch.int32)
    nibble = torch.where(flat % 2 == 0, byte >> 4, byte & 0x0F).long()
    code = state.code[nibble].float()
    scale_ids = flat // int(state.blocksize)
    if state.nested:
        quantized_scale = state.absmax[scale_ids].long()
        nested_ids = scale_ids // int(state.state2.blocksize)
        nested_code = state.state2.code[quantized_scale].float()
        nested_scale = state.state2.absmax[nested_ids].float()
        scale = nested_code * nested_scale + state.offset.float()
    else:
        scale = state.absmax[scale_ids].float()
    return (code * scale).to(compute_dtype)


def bnb_half2_selected_weight(module, output_rows: int) -> torch.Tensor:
    """Reconstruct CUDA FP16 SIMT order: half LUT, half scale, half2 multiply."""
    state = module.weight.quant_state
    _, in_features = (int(x) for x in state.shape)
    rows = torch.arange(output_rows, device="cuda:0", dtype=torch.long)[:, None]
    cols = torch.arange(in_features, device="cuda:0", dtype=torch.long)[None, :]
    flat = rows * in_features + cols
    packed = module.weight.data.reshape(-1)
    byte = packed[flat // 2].to(torch.int32)
    nibble = torch.where(flat % 2 == 0, byte >> 4, byte & 0x0F).long()
    centroid_half = state.code[nibble].to(torch.float16)
    scale_ids = flat // int(state.blocksize)
    quantized_scale = state.absmax[scale_ids].long()
    nested_ids = scale_ids // int(state.state2.blocksize)
    nested_code = state.state2.code[quantized_scale].float()
    nested_scale = state.state2.absmax[nested_ids].float()
    offset = state.offset.float()
    # torch.addcmul follows the fused nested-scale expression used by the CUDA
    # source's scale_code * scale_state2 + offset expression on the target.
    scale_f = torch.addcmul(offset.expand_as(nested_scale), nested_scale, nested_code)
    scale_half = scale_f.to(torch.float16)
    return (centroid_half * scale_half).to(torch.float16)


def run_reductions(x: torch.Tensor, weight: torch.Tensor, output_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    # Both operands are rounded to the BNB compute dtype before multiplication;
    # the ordinary candidates keep the product/reduction in FP32.  The SASS
    # audit shows HMUL2 followed by HADD2.F32, so also screen products rounded
    # to FP16 before conversion and FP32 accumulation.
    products_fp32 = x.float()[:, None, :] * weight.float()[None, :, :]
    candidates = {}
    full = products_fp32.sum(dim=-1)
    candidates["one_full_k_sum_fp32"] = full.to(torch.float16).to(output_dtype)

    for tile in (16, 32, 64, 128, 256, 512, 1024):
        acc = torch.zeros(products_fp32.shape[:2], device=products_fp32.device, dtype=torch.float32)
        for start in range(0, products_fp32.shape[-1], tile):
            acc = acc + products_fp32[..., start:start + tile].sum(dim=-1)
        candidates[f"tile{tile}_sum_then_fp32_acc"] = acc.to(torch.float16).to(output_dtype)

        acc_half = torch.zeros(products_fp32.shape[:2], device=products_fp32.device, dtype=torch.float16)
        for start in range(0, products_fp32.shape[-1], tile):
            partial = products_fp32[..., start:start + tile].sum(dim=-1).to(torch.float16)
            acc_half = (acc_half + partial).to(torch.float16)
        candidates[f"tile{tile}_sum_fp16_acc"] = acc_half.to(output_dtype)

    products_fp16_rounded = products_fp32.to(torch.float16).float()
    candidates["half_product_then_full_k_sum_fp32"] = products_fp16_rounded.sum(dim=-1).to(torch.float16).to(output_dtype)
    for tile in (16, 32, 64, 128, 256, 512, 1024):
        acc = torch.zeros(products_fp16_rounded.shape[:2], device=products_fp16_rounded.device, dtype=torch.float32)
        for start in range(0, products_fp16_rounded.shape[-1], tile):
            acc = acc + products_fp16_rounded[..., start:start + tile].sum(dim=-1)
        candidates[f"half_product_then_tile{tile}_sum_fp32_acc"] = acc.to(torch.float16).to(output_dtype)

    return candidates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--output-channels", type=int, default=256)
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
    module = model.model.layers[0].mlp.gate_proj
    output_channels = min(args.output_channels, int(module.out_features))
    capture = {}

    def pre_hook(_module, inputs):
        capture["x"] = inputs[0].detach().clone()

    handle = module.register_forward_pre_hook(pre_hook)
    token_cache = TargetKVCache(backend="static", max_capacity=prompt_len + 8)
    with torch.inference_mode():
        logits = model(prompt_ids, past_key_values=token_cache.cache, use_cache=True).logits[:, -1:, :]
        forced = []
        for _ in range(4):
            token = logits.argmax(dim=-1)
            forced.append(int(token.item()))
            logits = model(token, past_key_values=token_cache.cache, use_cache=True).logits[:, -1:, :]
        block_cache = TargetKVCache(backend="static", max_capacity=prompt_len + 8)
        model(prompt_ids, past_key_values=block_cache.cache, use_cache=True)
        block_tokens = torch.tensor([forced[:3]], device="cuda:0", dtype=torch.long)
        with (fixed_row_schedule_for_model(model, physical_rows=3,
                attention_strategy="bnb_rowwise", quantized_linear_strategy="fixed_rows"),
              original_bnb_forward(model)):
            block_logits = model(block_tokens, past_key_values=block_cache.cache, use_cache=True).logits
    handle.remove()
    torch.set_grad_enabled(False)

    x = capture["x"]
    compute_dtype = module.compute_dtype
    if x.shape[1] != 3:
        raise RuntimeError(f"expected M=3 activation, got {tuple(x.shape)}")
    bnb_output = module(x)[..., :output_channels].detach()
    x_compute = x.to(compute_dtype).reshape(3, -1)
    state = module.weight.quant_state
    weight_direct = direct_selected_weight(module, output_channels, compute_dtype)
    weight_bnb_half2 = bnb_half2_selected_weight(module, output_channels)
    public_dequant = dequantize_4bit(module.weight.data, quant_state=state)
    weight_public = public_dequant.reshape(state.shape)[:output_channels].to(compute_dtype)
    bias = module.bias
    if bias is not None:
        bias = bias[:output_channels].to(compute_dtype)

    candidate_results = {}
    for label, weight in (("direct_fp32_code_scale_to_compute", weight_direct),
                          ("public_quant_state_dtype_to_compute", weight_public),
                          ("bnb_half_lut_scale_half2", weight_bnb_half2)):
        candidates = run_reductions(x_compute, weight, x.dtype)
        for name, output in candidates.items():
            if bias is not None:
                output = (output.to(torch.float16) + bias[None, :].to(torch.float16)).to(x.dtype)
            candidate_results[f"{label}/{name}"] = compare(bnb_output, output)

    for lanes in (16, 32, 64, 128):
        for use_fma in (False, True):
            out_fp16 = torch.empty((3, output_channels), device="cuda:0", dtype=torch.float16)
            _strided_lane_accum_kernel[(3, output_channels)](
                x_compute, weight_direct, out_fp16,
                3, output_channels, int(x_compute.shape[-1]), lanes, use_fma, False,
                num_warps=4, enable_fp_fusion=use_fma,
            )
            out = out_fp16.to(x.dtype)
            if bias is not None:
                out = (out_fp16 + bias[None, :].to(torch.float16)).to(x.dtype)
            candidate_results[f"strided_lane_accum/lanes{lanes}/fma{int(use_fma)}"] = compare(bnb_output, out)

    for lanes in (16, 32, 64, 128):
        out_fp16 = torch.empty((3, output_channels), device="cuda:0", dtype=torch.float16)
        _strided_lane_accum_kernel[(3, output_channels)](
            x_compute, weight_direct, out_fp16,
            3, output_channels, int(x_compute.shape[-1]), lanes, False, True,
            num_warps=4, enable_fp_fusion=False,
        )
        out = out_fp16.to(x.dtype)
        if bias is not None:
            out = (out_fp16 + bias[None, :].to(torch.float16)).to(x.dtype)
        candidate_results[f"strided_lane_accum_fp16_product/lanes{lanes}"] = compare(bnb_output, out)

    bnb_order_out = torch.empty((3, output_channels), device="cuda:0", dtype=torch.float16)
    bias_compute = bias.to(torch.float16) if bias is not None else torch.empty((1,), device="cuda:0", dtype=torch.float16)
    _bnb_simt_lane_schedule_kernel[(3, output_channels)](
        x_compute, weight_bnb_half2, bias_compute, bnb_order_out,
        3, output_channels, int(x_compute.shape[-1]), bias is not None,
        num_warps=4, enable_fp_fusion=False,
    )
    bnb_order_out = bnb_order_out.to(x.dtype)
    candidate_results["bnb_half2_weights_fp16_products_bnb_lane_order"] = compare(bnb_output, bnb_order_out)

    best = sorted(candidate_results.items(), key=lambda item: (item[1]["mean_abs_diff"], item[1]["max_abs_diff"]))[:8]
    payload = {
        "schema": "zassd-bnb-nf4-gate-reduction-order-audit-v3",
        "classification": "small output-tile numerical screen on one actual three-row activation; not a proof of BNB kernel schedule",
        "model": MODEL, "revision": REVISION,
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "prompt_id": args.prompt_id, "prompt_tokens": prompt_len,
        "layer": 0, "module": "mlp.gate_proj",
        "output_channels_compared": output_channels,
        "activation_shape": list(x.shape), "activation_dtype": str(x.dtype),
        "bnb_compute_dtype": str(compute_dtype), "quant_state_weight_dtype": str(state.dtype),
        "forced_bnb_tokens": forced,
        "block_predictions_match_bnb_continuation": [int(v) for v in block_logits.argmax(dim=-1).flatten().tolist()] == forced[1:],
        "schedule_candidates": ["FP32 products with full-K and tiled reductions", "FP16-rounded products followed by full-K and tiled FP32 reductions", "BNB CUDA FP16 LUT/scale rounding, half2 products, serial per-lane pair accumulation, and SHFL.DOWN tree", "FP16 tile accumulation"],
        "best_candidates": [{"name": name, **metrics} for name, metrics in best],
        "all_candidates": candidate_results,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": [
            "One prompt, one first-layer gate projection, and first 256 output channels.",
            "Torch reduction trees do not necessarily match the BNB SIMT kernel's lane/warp accumulation order.",
            "No timing is claimed; output-channel subset is a localization probe only.",
            "GSM8K held-out indices 300-349 were not used.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "best_candidates": payload["best_candidates"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
