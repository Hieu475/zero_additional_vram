"""Experimental fixed-row NF4 kernel using FP16 tensor-core dot products."""

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
    def _nf4_fixed_rows_tensorcore_kernel(
        x_ptr,
        qweight_ptr,
        absmax_ptr,
        code_ptr,
        nested_absmax_ptr,
        nested_code_ptr,
        nested_offset_ptr,
        bias_ptr,
        y_ptr,
        ROWS,
        N: tl.constexpr,
        K: tl.constexpr,
        QUANT_BLOCK: tl.constexpr,
        NESTED_BLOCK: tl.constexpr,
        HAS_NESTED: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        WEIGHT_DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_ids = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        output_ids = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        input_offsets = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for input_tile in range(tl.cdiv(K, BLOCK_K)):
            input_ids = input_tile * BLOCK_K + input_offsets
            flat_ids = output_ids[None, :] * K + input_ids[:, None]
            valid_weight = (input_ids[:, None] < K) & (output_ids[None, :] < N)
            packed = tl.load(qweight_ptr + flat_ids // 2, mask=valid_weight, other=0).to(tl.uint8)
            nibble = tl.where(flat_ids % 2 == 0, packed >> 4, packed & 0x0F).to(tl.int32)
            code_value = tl.load(code_ptr + nibble, mask=valid_weight, other=0).to(tl.float32)

            quant_blocks = flat_ids // QUANT_BLOCK
            if HAS_NESTED:
                quantized_scale = tl.load(absmax_ptr + quant_blocks, mask=valid_weight, other=0).to(tl.int32)
                nested_groups = quant_blocks // NESTED_BLOCK
                nested_code = tl.load(nested_code_ptr + quantized_scale, mask=valid_weight, other=0).to(tl.float32)
                nested_scale = tl.load(nested_absmax_ptr + nested_groups, mask=valid_weight, other=0).to(tl.float32)
                nested_offset = tl.load(nested_offset_ptr).to(tl.float32)
                scale = nested_code * nested_scale + nested_offset
            else:
                scale = tl.load(absmax_ptr + quant_blocks, mask=valid_weight, other=0).to(tl.float32)
            dequantized = code_value * scale
            if WEIGHT_DTYPE == 0:
                dequantized = dequantized.to(tl.float16)
            elif WEIGHT_DTYPE == 1:
                dequantized = dequantized.to(tl.bfloat16)
            dequantized = dequantized.to(tl.float16)

            x_values = tl.load(
                x_ptr + row_ids[:, None] * K + input_ids[None, :],
                mask=(row_ids[:, None] < ROWS) & (input_ids[None, :] < K),
                other=0,
            ).to(tl.float16)
            accumulator = tl.dot(x_values, dequantized, accumulator)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + output_ids, mask=output_ids < N, other=0).to(tl.float32)
            accumulator += bias[None, :]
        tl.store(
            y_ptr + row_ids[:, None] * N + output_ids[None, :],
            accumulator.to(tl.float16),
            (row_ids[:, None] < ROWS) & (output_ids[None, :] < N),
        )


def nf4_fixed_rows_tensorcore_linear(
    input_rows: torch.Tensor,
    module,
    *,
    physical_rows: int,
    block_n: int = 64,
    block_k: int = 64,
) -> torch.Tensor:
    """Use one fixed physical-row tile and tensor-core FP16 dot accumulation."""
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
        raise ValueError("expected contiguous CUDA floating-point input rows")
    if module.training or getattr(module, "compute_dtype", None) not in (None, torch.float16):
        raise ValueError("expected eval-mode FP16 BNB compute")
    if block_n not in (32, 64, 128) or block_k not in (32, 64, 128):
        raise ValueError("block_n and block_k must be 32, 64, or 128")

    quant_state, nested = _supported_quant_state(module, input_features)
    if input_features % 2:
        raise ValueError("packed NF4 requires an even input feature count")
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

    block_m = 16
    x = input_rows.to(torch.float16).contiguous()
    y = torch.empty((1, rows, output_features), device=input_rows.device, dtype=torch.float16)
    weight_dtype_code = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[quant_state.dtype]
    grid = (triton.cdiv(physical_rows, block_m), triton.cdiv(output_features, block_n))
    _nf4_fixed_rows_tensorcore_kernel[grid](
        x, packed_weight, quant_state.absmax, quant_state.code,
        nested_absmax, nested_code, nested_offset,
        bias if bias is not None else packed_weight, y,
        rows, output_features, input_features, int(quant_state.blocksize),
        int(quant_state.state2.blocksize) if nested else 256, nested,
        bias is not None, weight_dtype_code, block_m, block_n, block_k,
        num_warps=4, enable_fp_fusion=False,
    )
    return y.to(input_rows.dtype)
