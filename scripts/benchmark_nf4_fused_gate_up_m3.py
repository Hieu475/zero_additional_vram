#!/usr/bin/env python3
"""Prototype one-launch gate/up NF4 projection and compare against two launches."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows import _supported_quant_state
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear,
)


@triton.jit
def _fused_gate_up_kernel(
    x_ptr,
    gate_weight_ptr,
    gate_absmax_ptr,
    gate_code_ptr,
    gate_nested_absmax_ptr,
    gate_nested_code_ptr,
    gate_nested_offset_ptr,
    gate_bias_ptr,
    up_weight_ptr,
    up_absmax_ptr,
    up_code_ptr,
    up_nested_absmax_ptr,
    up_nested_code_ptr,
    up_nested_offset_ptr,
    up_bias_ptr,
    gate_y_ptr,
    up_y_ptr,
    ROWS,
    N: tl.constexpr,
    K: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    NESTED_BLOCK: tl.constexpr,
    HAS_NESTED: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    WEIGHT_DTYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row0 = tl.program_id(0) * 2
    row1 = row0 + 1
    output_ids = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    projection_id = tl.program_id(2)
    input_offsets = tl.arange(0, BLOCK_K)
    packed_offsets = tl.arange(0, BLOCK_K // 2)
    scale_offsets = tl.arange(0, BLOCK_K // QUANT_BLOCK)
    if projection_id == 0:
        qweight_ptr = gate_weight_ptr
        absmax_ptr = gate_absmax_ptr
        code_ptr = gate_code_ptr
        nested_absmax_ptr = gate_nested_absmax_ptr
        nested_code_ptr = gate_nested_code_ptr
        nested_offset_ptr = gate_nested_offset_ptr
        bias_ptr = gate_bias_ptr
        y_ptr = gate_y_ptr
    else:
        qweight_ptr = up_weight_ptr
        absmax_ptr = up_absmax_ptr
        code_ptr = up_code_ptr
        nested_absmax_ptr = up_nested_absmax_ptr
        nested_code_ptr = up_nested_code_ptr
        nested_offset_ptr = up_nested_offset_ptr
        bias_ptr = up_bias_ptr
        y_ptr = up_y_ptr

    acc0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for input_tile in range(tl.cdiv(K, BLOCK_K)):
        input_ids = input_tile * BLOCK_K + input_offsets
        packed_ids = input_tile * (BLOCK_K // 2) + packed_offsets
        packed_flat_ids = output_ids[:, None] * (K // 2) + packed_ids[None, :]
        valid_packed = (output_ids[:, None] < N) & (packed_ids[None, :] < (K // 2))
        packed = tl.load(qweight_ptr + packed_flat_ids, mask=valid_packed, other=0).to(tl.uint8)
        high_nibble = (packed >> 4).to(tl.int32)
        low_nibble = (packed & 0x0F).to(tl.int32)
        high_code = tl.load(code_ptr + high_nibble, mask=valid_packed, other=0).to(tl.float32)
        low_code = tl.load(code_ptr + low_nibble, mask=valid_packed, other=0).to(tl.float32)

        quant_blocks = (
            output_ids[:, None] * (K // QUANT_BLOCK)
            + input_tile * (BLOCK_K // QUANT_BLOCK)
            + scale_offsets[None, :]
        )
        valid_scales = (
            (output_ids[:, None] < N)
            & ((input_tile * (BLOCK_K // QUANT_BLOCK) + scale_offsets[None, :]) < (K // QUANT_BLOCK))
        )
        if HAS_NESTED:
            quantized_scale = tl.load(absmax_ptr + quant_blocks, mask=valid_scales, other=0).to(tl.int32)
            nested_groups = quant_blocks // NESTED_BLOCK
            nested_code = tl.load(nested_code_ptr + quantized_scale, mask=valid_scales, other=0).to(tl.float32)
            nested_scale = tl.load(nested_absmax_ptr + nested_groups, mask=valid_scales, other=0).to(tl.float32)
            nested_offset = tl.load(nested_offset_ptr).to(tl.float32)
            scale = nested_code * nested_scale + nested_offset
        else:
            scale = tl.load(absmax_ptr + quant_blocks, mask=valid_scales, other=0).to(tl.float32)

        scale_packed = tl.broadcast_to(
            scale[:, :, None], (BLOCK_N, BLOCK_K // QUANT_BLOCK, QUANT_BLOCK // 2)
        )
        scale_packed = tl.reshape(scale_packed, (BLOCK_N, BLOCK_K // 2))
        high_weight = high_code * scale_packed
        low_weight = low_code * scale_packed
        if WEIGHT_DTYPE == 0:
            high_weight = high_weight.to(tl.float16)
            low_weight = low_weight.to(tl.float16)
        elif WEIGHT_DTYPE == 1:
            high_weight = high_weight.to(tl.bfloat16)
            low_weight = low_weight.to(tl.bfloat16)
        high_weight = high_weight.to(tl.float16).to(tl.float32)
        low_weight = low_weight.to(tl.float16).to(tl.float32)
        dequantized = tl.interleave(high_weight, low_weight)

        x0 = tl.load(x_ptr + row0 * K + input_ids,
                     mask=(row0 < ROWS) & (input_ids < K), other=0).to(tl.float32)
        x1 = tl.load(x_ptr + row1 * K + input_ids,
                     mask=(row1 < ROWS) & (input_ids < K), other=0).to(tl.float32)
        acc0 += tl.sum(dequantized * x0[None, :], axis=1)
        acc1 += tl.sum(dequantized * x1[None, :], axis=1)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + output_ids, mask=output_ids < N, other=0).to(tl.float32)
        acc0 += bias
        acc1 += bias
    tl.store(y_ptr + row0 * N + output_ids, acc0.to(tl.float16),
             (row0 < ROWS) & (output_ids < N))
    tl.store(y_ptr + row1 * N + output_ids, acc1.to(tl.float16),
             (row1 < ROWS) & (output_ids < N))


def _module_parts(module, input_features: int):
    state, nested = _supported_quant_state(module, input_features)
    if nested:
        nested_absmax, nested_code, nested_offset = (
            state.state2.absmax,
            state.state2.code,
            state.offset,
        )
    else:
        nested_absmax, nested_code, nested_offset = state.absmax, state.code, state.absmax
    bias = module.bias if module.bias is not None else module.weight.data
    return state, nested, nested_absmax, nested_code, nested_offset, bias


def fused_gate_up(x: torch.Tensor, gate, up, *, block_n=16, block_k=256, num_warps=1):
    rows, in_features = map(int, x.shape[1:])
    gate_state, gate_nested, gate_nested_absmax, gate_nested_code, gate_offset, gate_bias = _module_parts(gate, in_features)
    up_state, up_nested, up_nested_absmax, up_nested_code, up_offset, up_bias = _module_parts(up, in_features)
    if (gate_state.shape != up_state.shape or gate_nested != up_nested
            or gate_state.blocksize != up_state.blocksize
            or gate.weight.data.dtype != up.weight.data.dtype
            or (gate.bias is None) != (up.bias is None)):
        raise ValueError("gate/up NF4 quantization metadata and output shape must match")
    output_features = int(gate_state.shape[0])
    y_gate = torch.empty((1, rows, output_features), device=x.device, dtype=torch.float16)
    y_up = torch.empty_like(y_gate)
    dtype_code = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[gate_state.dtype]
    quant_block = int(gate_state.blocksize)
    _fused_gate_up_kernel[(triton.cdiv(rows, 2), triton.cdiv(output_features, block_n), 2)](
        x.to(torch.float16).contiguous(),
        gate.weight.data, gate_state.absmax, gate_state.code,
        gate_nested_absmax, gate_nested_code, gate_offset, gate_bias,
        up.weight.data, up_state.absmax, up_state.code,
        up_nested_absmax, up_nested_code, up_offset, up_bias,
        y_gate, y_up, rows, output_features, in_features, quant_block,
        int(gate_state.state2.blocksize) if gate_nested else 256,
        gate_nested, gate.bias is not None, dtype_code, block_n, block_k,
        num_warps=num_warps, enable_fp_fusion=False,
    )
    return y_gate.to(x.dtype), y_up.to(x.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa",
        model_dtype=torch.float16,
    )
    layer = loaded.model.eval().model.layers[0]
    gate, up = layer.mlp.gate_proj, layer.mlp.up_proj
    torch.manual_seed(49211)
    torch.cuda.manual_seed_all(49211)
    inputs = torch.randn((1, 3, gate.in_features), device="cuda:0", dtype=torch.float16)
    correctness = {}
    with torch.inference_mode():
        fused_values = {}
        for rows in (1, 2, 3):
            x = inputs[:, :rows].contiguous()
            y_gate, y_up = fused_gate_up(x, gate, up)
            torch.cuda.synchronize()
            fused_values[rows] = (y_gate.detach().clone(), y_up.detach().clone())
            gate_singletons, up_singletons = [], []
            for i in range(rows):
                gi = nf4_fixed_rows_group_scale_packed_rows_linear(
                    x[:, i:i + 1].contiguous(), gate, physical_rows=3,
                    block_n=16, block_k=256, num_warps=1,
                )
                ui = nf4_fixed_rows_group_scale_packed_rows_linear(
                    x[:, i:i + 1].contiguous(), up, physical_rows=3,
                    block_n=16, block_k=256, num_warps=1,
                )
                gate_singletons.append(gi)
                up_singletons.append(ui)
            correctness[f"fused_gate_m{rows}_rowwise_singleton_equal"] = all(
                torch.equal(y_gate[:, i:i + 1], gate_singletons[i]) for i in range(rows)
            )
            correctness[f"fused_up_m{rows}_rowwise_singleton_equal"] = all(
                torch.equal(y_up[:, i:i + 1], up_singletons[i]) for i in range(rows)
            )
            gate_separate = nf4_fixed_rows_group_scale_packed_rows_linear(
                x, gate, physical_rows=3, block_n=16, block_k=256, num_warps=1
            )
            up_separate = nf4_fixed_rows_group_scale_packed_rows_linear(
                x, up, physical_rows=3, block_n=16, block_k=256, num_warps=1
            )
            correctness[f"fused_gate_m{rows}_equals_separate_schedule"] = torch.equal(y_gate, gate_separate)
            correctness[f"fused_up_m{rows}_equals_separate_schedule"] = torch.equal(y_up, up_separate)

        timings = {"separate_two_launches": [], "fused_one_launch": []}
        for round_id in range(args.rounds):
            order = ["separate_two_launches", "fused_one_launch"]
            if round_id % 2:
                order.reverse()
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    if name == "separate_two_launches":
                        nf4_fixed_rows_group_scale_packed_rows_linear(
                            inputs, gate, physical_rows=3,
                            block_n=16, block_k=256, num_warps=1,
                        )
                        nf4_fixed_rows_group_scale_packed_rows_linear(
                            inputs, up, physical_rows=3,
                            block_n=16, block_k=256, num_warps=1,
                        )
                    else:
                        fused_gate_up(inputs, gate, up)
                end.record()
                end.synchronize()
                timings[name].append(start.elapsed_time(end) / args.iterations)
    medians = {name: statistics.median(values) for name, values in timings.items()}
    payload = {
        "schema": "zassd-nf4-fused-gate-up-m3-prototype-benchmark-v1",
        "status": "experimental one-launch prototype; not integrated into the model runtime",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "shape": {"rows": 3, "in_features": int(gate.in_features), "out_features_each": int(gate.out_features)},
        "tile": {"block_n": 16, "block_k": 256, "num_warps": 1},
        "iterations_per_sample": args.iterations,
        "alternating_rounds": args.rounds,
        "correctness": correctness,
        "median_cuda_ms": medians,
        "fused_over_separate_latency_ratio": medians["fused_one_launch"] / medians["separate_two_launches"],
        "samples_ms": timings,
        "limitations": [
            "Single first-layer random activation benchmark on one RTX 4050.",
            "Rowwise shape invariance is checked against singleton calls of the fused kernel; integrated whole-model exactness is not tested.",
            "A fused result differing from separate projections defines a new arithmetic schedule and would require a new reference contract.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "separate_ms": medians["separate_two_launches"],
        "fused_ms": medians["fused_one_launch"],
        "fused_over_separate": payload["fused_over_separate_latency_ratio"],
        "all_correctness_checks": all(correctness.values()),
        "correctness": correctness,
    }, indent=2))


if __name__ == "__main__":
    main()
