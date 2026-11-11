"""M-invariant LM head: logits = x @ W^T for 1..MP rows, weights read once.

Invariance by construction: every call, for any number of rows M <= MP, runs the same compiled kernel on an MP-row
zero-padded activation tile. Element (m, v) is a fixed-order fp32 accumulation over K (chunks of BK via tensor-core tl.dot, in order);
the MMA computes each output element independently of the other rows, so it depends only on x[m] and W[v].
No split-K, no atomics, no autotuning. Invariance is also verified bitwise by scripts/kernels/wp1_lm_head_invariant.py.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MP = 16  # padded rows (min MMA tile); M in [1, MP]


@triton.jit
def _lm_head_kernel(x_ptr, w_ptr, y_ptr, V, K: tl.constexpr, MP: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr, OUT_DTYPE: tl.constexpr):
    pid = tl.program_id(0)
    vs = pid * BV + tl.arange(0, BV)
    ms = tl.arange(0, MP)
    acc = tl.zeros((MP, BV), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + ms[:, None] * K + ks[None, :])                                  # [MP, BK] fp16
        w = tl.load(w_ptr + vs[None, :] * K + ks[:, None], mask=(vs < V)[None, :], other=0.0)  # [BK, BV] fp16
        acc = tl.dot(x, w, acc)
    tl.store(y_ptr + ms[:, None] * V + vs[None, :], acc.to(OUT_DTYPE), mask=(vs < V)[None, :])


def lm_head_invariant(x: torch.Tensor, w: torch.Tensor, bv: int = 64, bk: int = 128, num_warps: int = 4) -> torch.Tensor:
    """x: [M, K] fp16/bf16 (M <= MP), w: [V, K] fp16/bf16 contiguous -> logits [M, V] fp16."""
    M, K = x.shape
    V = w.shape[0]
    assert 1 <= M <= MP and K % bk == 0 and x.dtype == w.dtype in (torch.float16, torch.bfloat16) and w.is_contiguous()
    xp = torch.zeros((MP, K), dtype=x.dtype, device=x.device)
    xp[:M] = x
    y = torch.empty((MP, V), dtype=x.dtype, device=x.device)
    _lm_head_kernel[(triton.cdiv(V, bv),)](xp, w, y, V, K=K, MP=MP, BV=bv, BK=bk, OUT_DTYPE=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16, num_warps=num_warps)
    return y[:M]
