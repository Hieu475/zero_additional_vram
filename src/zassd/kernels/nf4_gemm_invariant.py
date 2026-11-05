"""M-invariant bitsandbytes-NF4 GEMM for decode: y[M, N] = x[M, K] @ dequant(W)[N, K]^T (+ bias), M <= MP.

Packed NF4 weights (nibble-high first, blocksize 64, double-quantised absmax, 256-block second level, offset) are
dequantised in-kernel and multiplied with tensor cores. Every call runs the same kernel on an MP-row zero-padded tile with a
fixed K order, so row i depends only on x[i] and W: bitwise independent of M and of the other rows (also verified by test).
Weights are read once for all rows.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MP = 16


@triton.jit
def _nf4_kernel(x_ptr, w_ptr, am8_ptr, code_ptr, am2_ptr, code2_ptr, bias_ptr, y_ptr, offset, N,
                K: tl.constexpr, MP: tl.constexpr, BN: tl.constexpr, UNROLL: tl.constexpr, HAS_BIAS: tl.constexpr,
                DT: tl.constexpr):
    # One quantisation block (64 elements) per step: one scale per output row. Even/odd columns are handled by two dots,
    # so no interleave/reshape is needed; the summation order is fixed and independent of M.
    pid = tl.program_id(0)
    ns = pid * BN + tl.arange(0, BN)
    nmask = ns < N
    ms = tl.arange(0, MP)
    js = tl.arange(0, 32)
    acc = tl.zeros((MP, BN), dtype=tl.float32)
    for kb in range(0, K // 64):
        b = tl.load(w_ptr + ns[:, None] * (K // 2) + kb * 32 + js[None, :], mask=nmask[:, None], other=0).to(tl.int32)
        sidx = ns * (K // 64) + kb
        q = tl.load(am8_ptr + sidx, mask=nmask, other=0).to(tl.int32)
        scale = tl.load(code2_ptr + q) * tl.load(am2_ptr + (sidx // 256), mask=nmask, other=0.0) + offset
        w_hi = (tl.load(code_ptr + (b >> 4)) * scale[:, None]).to(DT)                  # elements 2j   [BN, 32]
        w_lo = (tl.load(code_ptr + (b & 15)) * scale[:, None]).to(DT)                  # elements 2j+1
        x_e = tl.load(x_ptr + ms[:, None] * K + kb * 64 + 2 * js[None, :])             # [MP, 32]
        x_o = tl.load(x_ptr + ms[:, None] * K + kb * 64 + 2 * js[None, :] + 1)
        acc = tl.dot(x_e, tl.trans(w_hi), acc)
        acc = tl.dot(x_o, tl.trans(w_lo), acc)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + ns, mask=nmask, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + ms[:, None] * N + ns[None, :], acc.to(DT), mask=nmask[None, :])


@triton.jit
def _nf4_simt_kernel(x_ptr, w_ptr, am8_ptr, code_ptr, am2_ptr, code2_ptr, bias_ptr, y_ptr, offset, N,
                     K: tl.constexpr, MP: tl.constexpr, BN: tl.constexpr, G: tl.constexpr, HAS_BIAS: tl.constexpr,
                     DT: tl.constexpr):
    # CUDA-core variant: every weight is dequantised once and used for all MP rows. acc[m, n, lane] accumulates the 32
    # strided K positions of each 64-block (even and odd columns separately), blocks in increasing order; one tl.sum over
    # the lane axis (fixed tree) finishes. The per-(m, n) arithmetic is elementwise, hence independent of MP and of the
    # other rows. G blocks are unrolled per iteration so that their loads are in flight together.
    NB: tl.constexpr = K // 64
    pid = tl.program_id(0)
    ns = pid * BN + tl.arange(0, BN)
    nmask = ns < N
    ms = tl.arange(0, MP)
    js = tl.arange(0, 32)
    acc_e = tl.zeros((MP, BN, 32), dtype=tl.float32)
    acc_o = tl.zeros((MP, BN, 32), dtype=tl.float32)
    for it in range(0, tl.cdiv(NB, G)):
        for g in tl.static_range(G):
            kb = it * G + g
            ok = kb < NB
            b = tl.load(w_ptr + ns[:, None] * (K // 2) + kb * 32 + js[None, :], mask=nmask[:, None] & ok, other=0).to(tl.int32)
            sidx = ns * NB + kb
            q = tl.load(am8_ptr + sidx, mask=nmask & ok, other=0).to(tl.int32)
            scale = tl.load(code2_ptr + q) * tl.load(am2_ptr + (sidx // 256), mask=nmask & ok, other=0.0) + offset
            w_hi = (tl.load(code_ptr + (b >> 4)) * scale[:, None]).to(DT).to(tl.float32)   # [BN, 32]
            w_lo = (tl.load(code_ptr + (b & 15)) * scale[:, None]).to(DT).to(tl.float32)
            x_e = tl.load(x_ptr + ms[:, None] * K + kb * 64 + 2 * js[None, :], mask=ok, other=0.0).to(tl.float32)   # [MP, 32]
            x_o = tl.load(x_ptr + ms[:, None] * K + kb * 64 + 2 * js[None, :] + 1, mask=ok, other=0.0).to(tl.float32)
            acc_e += x_e[:, None, :] * w_hi[None, :, :]
            acc_o += x_o[:, None, :] * w_lo[None, :, :]
    out = tl.sum(acc_e + acc_o, axis=2)                                                         # [MP, BN]
    if HAS_BIAS:
        out += tl.load(bias_ptr + ns, mask=nmask, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + ms[:, None] * N + ns[None, :], out.to(DT), mask=nmask[None, :])


def nf4_gemm_invariant(x: torch.Tensor, weight: torch.Tensor, quant_state, bias: torch.Tensor | None = None,
                       bn: int = 64, bk: int = 64, num_warps: int = 4, num_stages: int = 3, mode: str = "mma", mp: int = MP) -> torch.Tensor:
    """x: [M, K] fp16/bf16 (M <= MP); weight: bnb packed uint8 (any shape, N*K/2 bytes)."""
    M, K = x.shape
    N, K2 = quant_state.shape
    assert K == K2 and 1 <= M <= MP and K % 64 == 0 and quant_state.blocksize == 64 and quant_state.quant_type == "nf4"
    assert x.dtype in (torch.float16, torch.bfloat16)
    st2 = quant_state.state2
    P = mp if mode == "simt" else MP
    assert M <= P
    xp = torch.zeros((P, K), dtype=x.dtype, device=x.device)
    xp[:M] = x
    y = torch.empty((P, N), dtype=x.dtype, device=x.device)
    kern = _nf4_simt_kernel if mode == "simt" else _nf4_kernel
    extra = {"G": num_stages} if mode == "simt" else {"UNROLL": 1}
    kern[(triton.cdiv(N, bn),)](
        xp, weight, quant_state.absmax, quant_state.code, st2.absmax, st2.code, bias if bias is not None else xp, y,
        float(quant_state.offset), N, K=K, MP=P, BN=bn, **extra, HAS_BIAS=bias is not None,
        DT=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16, num_warps=num_warps, **({} if mode == "simt" else {"num_stages": num_stages}))
    return y[:M]
