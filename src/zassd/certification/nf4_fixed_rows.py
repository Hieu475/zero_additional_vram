"""Experimental fixed-schedule NF4 linear for B1 research.

This prototype consumes bitsandbytes' packed NF4 weights directly. Each Triton
program owns one token row and a tile of output channels, and reduces only
along the fixed input-feature dimension. It is an experimental reference
schedule, not a bitwise match for bitsandbytes' GEMM and not a performance
claim. The caller must use this same path for both the B1 singleton reference
and block verification.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:  # pragma: no cover - depends on optional runtime
    triton = None
    tl = None
    _TRITON_IMPORT_ERROR = exc
else:
    _TRITON_IMPORT_ERROR = None


if triton is not None:

    @triton.jit
    def _nf4_fixed_rows_kernel(
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
        accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

        for input_tile in range(tl.cdiv(K, BLOCK_K)):
            input_ids = input_tile * BLOCK_K + input_offsets
            flat_ids = output_ids[:, None] * K + input_ids[None, :]
            valid = (output_ids[:, None] < N) & (input_ids[None, :] < K)

            packed = tl.load(
                qweight_ptr + flat_ids // 2, mask=valid, other=0
            ).to(tl.uint8)
            nibble = tl.where(
                flat_ids % 2 == 0,
                packed >> 4,
                packed & 0x0F,
            ).to(tl.int32)
            code_value = tl.load(code_ptr + nibble, mask=valid, other=0).to(tl.float32)

            quant_blocks = flat_ids // QUANT_BLOCK
            if HAS_NESTED:
                quantized_scale = tl.load(
                    absmax_ptr + quant_blocks, mask=valid, other=0
                ).to(tl.int32)
                nested_groups = quant_blocks // NESTED_BLOCK
                nested_code = tl.load(
                    nested_code_ptr + quantized_scale, mask=valid, other=0
                ).to(tl.float32)
                nested_scale = tl.load(
                    nested_absmax_ptr + nested_groups, mask=valid, other=0
                ).to(tl.float32)
                nested_offset = tl.load(nested_offset_ptr).to(tl.float32)
                scale = nested_code * nested_scale + nested_offset
            else:
                scale = tl.load(
                    absmax_ptr + quant_blocks, mask=valid, other=0
                ).to(tl.float32)

            # Match the dequantized weight dtype recorded by BNB, then apply
            # this prototype's explicit FP16 compute boundary.
            dequantized = code_value * scale
            if WEIGHT_DTYPE == 0:
                dequantized = dequantized.to(tl.float16)
            elif WEIGHT_DTYPE == 1:
                dequantized = dequantized.to(tl.bfloat16)
            dequantized = dequantized.to(tl.float16)
            x_values = tl.load(
                x_ptr + row * K + input_ids,
                mask=input_ids < K,
                other=0,
            ).to(tl.float32)
            products = dequantized.to(tl.float32) * x_values[None, :]
            accumulator += tl.sum(products, axis=1)

        if HAS_BIAS:
            bias = tl.load(bias_ptr + output_ids, mask=output_ids < N, other=0)
            accumulator = accumulator + bias.to(tl.float32)

        tl.store(
            y_ptr + row * N + output_ids,
            accumulator.to(tl.float16),
            output_ids < N,
        )


def _supported_quant_state(module, input_features: int) -> tuple[object, bool]:
    quant_state = getattr(module.weight, "quant_state", None)
    if quant_state is None:
        raise ValueError("Triton NF4 requires a materialized bitsandbytes quant_state")
    if quant_state.quant_type != "nf4":
        raise ValueError("Triton NF4 supports only quant_type='nf4'")
    if quant_state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Triton NF4 supports FP16/BF16/FP32 dequantized weights")
    if len(quant_state.shape) != 2:
        raise ValueError("Triton NF4 requires a two-dimensional [out, in] weight")
    if int(quant_state.shape[1]) != input_features:
        raise ValueError("input feature count does not match NF4 quant_state")
    if input_features % 2:
        raise ValueError("NF4 packed weights require an even input feature count")
    if quant_state.blocksize not in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        raise ValueError("unsupported BNB NF4 quantization blocksize")

    nested = bool(quant_state.nested)
    if nested:
        state2 = quant_state.state2
        if state2 is None or int(state2.blocksize) != 256:
            raise ValueError("nested NF4 requires the BNB 256-value scale block")
        if not isinstance(quant_state.offset, torch.Tensor) or quant_state.offset.numel() != 1:
            raise ValueError("nested NF4 offset must be a scalar tensor")
    return quant_state, nested


def nf4_fixed_rows_linear(input_rows: torch.Tensor, module, *, physical_rows: int) -> torch.Tensor:
    """Apply an NF4 Linear4bit module with a row-independent Triton schedule.

    Args:
        input_rows: contiguous tensor shaped ``[1, m, in_features]`` where
            ``1 <= m <= physical_rows``.
        module: bitsandbytes ``Linear4bit`` with NF4 weights.
        physical_rows: fixed R used by both singleton and block B1 calls.

    The prototype intentionally accepts only batch one, CUDA floating-point
    activations, FP16 compute, evaluation inference, and at most the configured
    physical row count. Unlike the BNB padded strategy, Triton launches only
    real rows; each row has an independent program and fixed-K reduction.
    Inputs are cast to FP16 for the kernel and results are cast
    back to the activation dtype, matching ``Linear4bit.forward``'s dtype
    boundary. Unsupported cases raise;
    they never fall through to BNB. No full dequantized weight matrix is
    materialized. Nested scale metadata is read directly from BNB's state.
    """
    if triton is None:
        raise RuntimeError("Triton is required for nf4_fixed_rows_linear") from _TRITON_IMPORT_ERROR
    if torch.is_grad_enabled():
        raise RuntimeError("Triton NF4 prototype is inference-only; disable gradients")
    if input_rows.ndim != 3 or input_rows.shape[0] != 1:
        raise ValueError("Triton NF4 requires a rank-3 batch-one input")
    if input_rows.shape[1] < 1 or input_rows.shape[1] > physical_rows:
        raise ValueError("input row count must be within 1..physical_rows")
    if input_rows.dtype not in (torch.float16, torch.bfloat16, torch.float32) or not input_rows.is_cuda:
        raise ValueError("Triton NF4 requires CUDA FP16/BF16/FP32 activations")
    if not input_rows.is_contiguous():
        raise ValueError("Triton NF4 requires contiguous inputs")
    if module.training:
        raise ValueError("Triton NF4 prototype requires eval mode")
    if getattr(module, "compute_dtype", None) not in (None, torch.float16):
        raise ValueError("Triton NF4 supports only FP16 Linear4bit compute")

    rows, input_features = int(input_rows.shape[1]), int(input_rows.shape[2])
    output_dtype = input_rows.dtype
    compute_rows = input_rows.to(torch.float16).contiguous()
    quant_state, nested = _supported_quant_state(module, input_features)
    output_features = int(quant_state.shape[0])
    packed_weight = module.weight.data
    if packed_weight.dtype != torch.uint8 or not packed_weight.is_cuda or not packed_weight.is_contiguous():
        raise ValueError("Triton NF4 requires contiguous CUDA uint8 packed weights")
    if packed_weight.numel() * 2 != output_features * input_features:
        raise ValueError("packed NF4 weight size does not match quant_state shape")
    if packed_weight.device != input_rows.device:
        raise ValueError("packed NF4 weights and input must be on the same CUDA device")
    if quant_state.absmax.device != input_rows.device or quant_state.code.device != input_rows.device:
        raise ValueError("NF4 quantization state must be on the input device")

    if nested:
        nested_absmax = quant_state.state2.absmax
        nested_code = quant_state.state2.code
        nested_offset = quant_state.offset
        if nested_absmax.device != input_rows.device or nested_code.device != input_rows.device:
            raise ValueError("nested NF4 state must be on the input device")
        if nested_offset.device != input_rows.device:
            raise ValueError("nested NF4 offset must be on the input device")
    else:
        # The kernel's compile-time HAS_NESTED branch prevents dereferencing
        # these placeholders. Reuse real state tensors to keep pointer args valid.
        nested_absmax = quant_state.absmax
        nested_code = quant_state.code
        nested_offset = quant_state.absmax

    bias = module.bias
    if bias is not None and (bias.dtype != torch.float16 or bias.device != input_rows.device):
        raise ValueError("Triton NF4 bias must be CUDA FP16 on the input device")
    y = torch.empty((1, rows, output_features), device=input_rows.device, dtype=torch.float16)

    block_n, block_k = 16, 128
    weight_dtype_code = {
        torch.float16: 0,
        torch.bfloat16: 1,
        torch.float32: 2,
    }[quant_state.dtype]
    _nf4_fixed_rows_kernel[(rows, triton.cdiv(output_features, block_n))](
        compute_rows,
        packed_weight,
        quant_state.absmax,
        quant_state.code,
        nested_absmax,
        nested_code,
        nested_offset,
        bias if bias is not None else packed_weight,
        y,
        output_features,
        input_features,
        int(quant_state.blocksize),
        int(quant_state.state2.blocksize) if nested else 256,
        nested,
        bias is not None,
        weight_dtype_code,
        block_n,
        block_k,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y.to(output_dtype)
