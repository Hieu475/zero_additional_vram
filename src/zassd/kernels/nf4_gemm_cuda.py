"""CUDA M-invariant NF4 GEMM (see csrc/nf4_gemm.cu). JIT-compiled on first use."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

_SRC = (Path(__file__).parent / "csrc" / "nf4_gemm.cu").read_text()
_EXT = None


def _ext():
    global _EXT
    if _EXT is None:
        _EXT = load_inline(
            name="zassd_nf4_gemm", cpp_sources="torch::Tensor nf4_gemm(torch::Tensor x, torch::Tensor w, torch::Tensor am8, torch::Tensor code, torch::Tensor am2, torch::Tensor code2, double offset, torch::Tensor bias, int64_t mp, int64_t rows, int64_t warps);",
            cuda_sources='#include <ATen/cuda/CUDAContext.h>\n' + _SRC, functions=["nf4_gemm"],
            extra_cuda_cflags=["-O3", "-arch=sm_89", "-lineinfo"], verbose=False)
    return _EXT


_PREP: dict = {}
_BIAS: dict = {}


def _bias_f32(b):
    if b is None:
        return None
    if id(b) not in _BIAS:
        _BIAS[id(b)] = (b.float().contiguous(), b)  # keep b alive so the id stays unique
    return _BIAS[id(b)][0]


def _prep(qs):
    # cached so that the (synchronising) float(offset) and dtype conversions happen once, before any CUDA-graph capture
    key = id(qs)
    if key not in _PREP:
        _PREP[key] = (qs.code.float().contiguous(), qs.state2.code.float().contiguous(), float(qs.offset), qs)
    return _PREP[key]


def nf4_gemm_cuda(x: torch.Tensor, weight: torch.Tensor, quant_state, bias: torch.Tensor | None = None,
                  mp: int = 8, rows: int = 4, warps: int = 4) -> torch.Tensor:
    M, K = x.shape
    assert M <= mp and K % 64 == 0 and quant_state.blocksize == 64 and quant_state.quant_type == "nf4"
    xp = torch.zeros((mp, K), dtype=x.dtype, device=x.device)
    xp[:M] = x
    code, code2, offset, _ = _prep(quant_state)
    st2 = quant_state.state2
    y = _ext().nf4_gemm(xp, weight.reshape(-1), quant_state.absmax, code, st2.absmax,
                        code2, offset,
                        _bias_f32(bias) if bias is not None else torch.empty(0, device=x.device, dtype=torch.float32), mp, rows, warps)
    return y[:M]
