"""Experimental grouped-scale NF4 kernel that loads each packed byte once."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None

from zassd.certification.nf4_fixed_rows import _supported_quant_state


if triton is not None:

    @triton.jit
    def _nf4_fixed_rows_group_scale_packed_kernel(
        x_ptr,
        qweight_ptr,
        absmax_ptr,
        code_ptr,
        nested_absmax_ptr,
        nested_code_ptr,
        nested_offset_ptr,
        bias_ptr,
        y_ptr,
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
        row = tl.program_id(0)
        output_ids = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        input_offsets = tl.arange(0, BLOCK_K)
        packed_offsets = tl.arange(0, BLOCK_K // 2)
        scale_offsets = tl.arange(0, BLOCK_K // QUANT_BLOCK)
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for input_tile in range(tl.cdiv(K, BLOCK_K)):
            input_ids = input_tile * BLOCK_K + input_offsets
            packed_ids = input_tile * (BLOCK_K // 2) + packed_offsets
            packed_flat_ids = output_ids[:, None] * (K // 2) + packed_ids[None, :]
            valid_packed = (output_ids[:, None] < N) & (packed_ids[None, :] < (K // 2))
            packed = tl.load(
                qweight_ptr + packed_flat_ids,
                mask=valid_packed,
                other=0,
            ).to(tl.uint8)
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
                scale[:, :, None],
                (BLOCK_N, BLOCK_K // QUANT_BLOCK, QUANT_BLOCK // 2),
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

            x_values = tl.load(x_ptr + row * K + input_ids, mask=input_ids < K, other=0).to(tl.float32)
            products = dequantized * x_values[None, :]
            accumulator += tl.sum(products, axis=1)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + output_ids, mask=output_ids < N, other=0).to(tl.float32)
            accumulator += bias
        tl.store(y_ptr + row * N + output_ids, accumulator.to(tl.float16), output_ids < N)


def nf4_fixed_rows_group_scale_packed_linear(
    input_rows: torch.Tensor,
    module,
    *,
    physical_rows: int,
    block_n: int = 8,
    block_k: int = 256,
) -> torch.Tensor:
    """Use one byte load for each pair of NF4 values; experimental inference path."""
    if triton is None:
        raise RuntimeError("Triton is required") from _TRITON_IMPORT_ERROR
    if torch.is_grad_enabled():
        raise RuntimeError("Triton NF4 prototype is inference-only")
    if input_rows.ndim != 3 or input_rows.shape[0] != 1:
        raise ValueError("expected rank-3 batch-one inputs")
    rows, input_features = int(input_rows.shape[1]), int(input_rows.shape[2])
    if rows < 1 or rows > physical_rows:
        raise ValueError("rows must be within 1..physical_rows")
    if not input_rows.is_cuda or not input_rows.is_contiguous() or input_rows.dtype not in (
        torch.float16, torch.bfloat16, torch.float32
    ):
        raise ValueError("expected contiguous CUDA floating-point inputs")
    if module.training or getattr(module, "compute_dtype", None) not in (None, torch.float16):
        raise ValueError("expected eval-mode FP16 BNB compute")

    quant_state, nested = _supported_quant_state(module, input_features)
    quant_block = int(quant_state.blocksize)
    if input_features % quant_block or block_k % quant_block:
        raise ValueError("group-scale path requires K and block_k aligned to quantization blocksize")
    if block_n not in (8, 16, 32) or block_k not in (64, 128, 256):
        raise ValueError("unsupported group-scale tile")
    output_features = int(quant_state.shape[0])
    packed_weight = module.weight.data
    if packed_weight.dtype != torch.uint8 or not packed_weight.is_cuda or not packed_weight.is_contiguous():
        raise ValueError("expected contiguous CUDA uint8 NF4 weights")
    if packed_weight.device != input_rows.device or quant_state.absmax.device != input_rows.device:
        raise ValueError("weights, scales, and inputs must share a device")
    if quant_state.code.device != input_rows.device:
        raise ValueError("NF4 codebook must be on the input device")
    if nested:
        nested_absmax, nested_code, nested_offset = (
            quant_state.state2.absmax, quant_state.state2.code, quant_state.offset
        )
    else:
        nested_absmax, nested_code, nested_offset = (
            quant_state.absmax, quant_state.code, quant_state.absmax
        )
    bias = module.bias
    if bias is not None and (bias.device != input_rows.device or bias.dtype != torch.float16):
        raise ValueError("bias must be CUDA FP16")

    x = input_rows.to(torch.float16).contiguous()
    y = torch.empty((1, rows, output_features), device=input_rows.device, dtype=torch.float16)
    dtype_code = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[quant_state.dtype]
    _nf4_fixed_rows_group_scale_packed_kernel[(rows, triton.cdiv(output_features, block_n))](
        x, packed_weight, quant_state.absmax, quant_state.code,
        nested_absmax, nested_code, nested_offset,
        bias if bias is not None else packed_weight, y,
        output_features, input_features, quant_block,
        int(quant_state.state2.blocksize) if nested else 256, nested,
        bias is not None, dtype_code, block_n, block_k,
        num_warps=4, enable_fp_fusion=False,
    )
    return y.to(input_rows.dtype)
