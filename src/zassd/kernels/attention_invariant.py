"""M-invariant decode attention (split-KV, fixed chunking, fixed combine order).

q: [B, Hq, M, D] (M <= MP rows at absolute positions prefix..prefix+M-1), k/v: [B, Hkv, cap, D] compact KV (GQA, no
repeat_kv), `prefix` = tokens before the block. Row i attends to keys 0..prefix+i. The computation of row i uses only
absolute key positions: keys are split into fixed SPLIT-key chunks, each chunk processed in BN-key blocks by an online
softmax in fixed order, chunk partials are combined in increasing chunk order. Chunks/blocks fully masked for a row are
exact no-ops, so the row's result does not depend on M, on the other rows, or on the cache capacity.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

MP = 16
SPLIT = 256
BN = 64


@triton.jit
def _partial_kernel(q_ptr, k_ptr, v_ptr, pm_ptr, pl_ptr, pacc_ptr, prefix_arg, prefix_ptr, M, scale, S_MAX,
                    stride_kb, stride_kh, stride_kn, Hq, Hkv, D: tl.constexpr, DP: tl.constexpr, MP: tl.constexpr,
                    SPLIT: tl.constexpr, BN: tl.constexpr, DT: tl.constexpr, USE_PTR: tl.constexpr):
    if USE_PTR:
        prefix = tl.load(prefix_ptr)
    else:
        prefix = prefix_arg
    bh = tl.program_id(0)          # b * Hq + h
    s = tl.program_id(1)
    b = bh // Hq
    h = bh % Hq
    hk = h // (Hq // Hkv)
    rows = tl.arange(0, MP)
    ds = tl.arange(0, DP)
    dmask = ds < D                                           # head_dim padded to a power of two (zeros contribute exactly 0)
    q = tl.load(q_ptr + (bh * M + rows[:, None]) * D + ds[None, :], mask=(rows < M)[:, None] & dmask[None, :], other=0.0)  # [MP, D] fp16
    limit = prefix + rows                                    # row i sees keys <= prefix+i
    total = prefix + M
    kv_base = b * stride_kb + hk * stride_kh
    m_i = tl.full((MP,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((MP,), dtype=tl.float32)
    acc = tl.zeros((MP, DP), dtype=tl.float32)
    start = s * SPLIT
    for n0 in range(start, start + SPLIT, BN):
        if n0 < total:
            ns = n0 + tl.arange(0, BN)
            kmask = ns < total
            k = tl.load(k_ptr + kv_base + ns[None, :] * stride_kn + ds[:, None], mask=kmask[None, :] & dmask[:, None], other=0.0)  # [DP, BN]
            sc = tl.dot(q, k) * scale                                                                          # [MP, BN] fp32
            valid = (ns[None, :] <= limit[:, None]) & kmask[None, :]
            sc = tl.where(valid, sc, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(sc, axis=1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            p = tl.exp(sc - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(v_ptr + kv_base + ns[:, None] * stride_kn + ds[None, :], mask=kmask[:, None] & dmask[None, :], other=0.0)  # [BN, DP]
            acc = acc * alpha[:, None] + tl.dot(p.to(DT), v)
            m_i = m_new
    base = (bh * S_MAX + s) * MP
    tl.store(pm_ptr + base + rows, m_i)
    tl.store(pl_ptr + base + rows, l_i)
    tl.store(pacc_ptr + (base + rows[:, None]) * DP + ds[None, :], acc)


@triton.jit
def _combine_kernel(pm_ptr, pl_ptr, pacc_ptr, o_ptr, M, S, S_MAX, D: tl.constexpr, DP: tl.constexpr, MP: tl.constexpr, DT: tl.constexpr):
    bh = tl.program_id(0)
    rows = tl.arange(0, MP)
    ds = tl.arange(0, DP)
    m = tl.full((MP,), float("-inf"), dtype=tl.float32)
    l = tl.zeros((MP,), dtype=tl.float32)
    acc = tl.zeros((MP, DP), dtype=tl.float32)
    for s in range(0, S):
        base = (bh * S_MAX + s) * MP
        ms = tl.load(pm_ptr + base + rows)
        ls = tl.load(pl_ptr + base + rows)
        accs = tl.load(pacc_ptr + (base + rows[:, None]) * DP + ds[None, :])
        m_new = tl.maximum(m, ms)
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        a = tl.exp(m - m_safe)
        bta = tl.exp(ms - m_safe)
        l = l * a + ls * bta
        acc = acc * a[:, None] + accs * bta[:, None]
        m = m_new
    out = acc / l[:, None]
    tl.store(o_ptr + (bh * M + rows[:, None]) * D + ds[None, :], out.to(DT), mask=(rows < M)[:, None] & (ds < D)[None, :])


def attention_invariant(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, prefix: int, scale: float | None = None,
                        num_warps: int = 4, prefix_tensor: torch.Tensor | None = None) -> torch.Tensor:
    B, Hq, M, D = q.shape
    Hkv = k.shape[1]
    assert 1 <= M <= MP and q.dtype == k.dtype == v.dtype in (torch.float16, torch.bfloat16) and D in (64, 96, 128) and Hq % Hkv == 0
    q = q.contiguous()
    DP = triton.next_power_of_2(D)
    assert k.stride(3) == 1 and v.stride() == k.stride()
    scale = scale if scale is not None else 1.0 / math.sqrt(D)
    DT = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    # prefix_tensor (int32 device scalar): CUDA-graph mode, K/V are the full static cache, S covers the capacity
    S = triton.cdiv(k.shape[2], SPLIT) if prefix_tensor is not None else triton.cdiv(prefix + M, SPLIT)
    S_MAX = S
    dev = q.device
    pm = torch.empty((B * Hq, S_MAX, MP), dtype=torch.float32, device=dev)
    pl = torch.empty_like(pm)
    pacc = torch.empty((B * Hq, S_MAX, MP, DP), dtype=torch.float32, device=dev)
    _partial_kernel[(B * Hq, S)](q, k, v, pm, pl, pacc, prefix, prefix_tensor if prefix_tensor is not None else q, M, scale, S_MAX, k.stride(0), k.stride(1), k.stride(2),
                                 Hq, Hkv, D=D, DP=DP, MP=MP, SPLIT=SPLIT, BN=BN, DT=DT, USE_PTR=prefix_tensor is not None, num_warps=num_warps)
    out = torch.empty((B, Hq, M, D), dtype=q.dtype, device=dev)
    _combine_kernel[(B * Hq,)](pm, pl, pacc, out, M, S, S_MAX, D=D, DP=DP, MP=MP, DT=DT, num_warps=num_warps)
    return out
