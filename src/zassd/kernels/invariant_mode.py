"""Context manager that routes LM head, RMSNorm and decode-time attention of an HF model through the M-invariant kernels.

Use the same mode for the sequential reference (M=1) and for block verification: row results are then bitwise equal by
construction. Prefill (M > MP) keeps the stock attention path (identical in both runs).
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from .attention_invariant import MP, attention_invariant
from .lm_head_invariant import lm_head_invariant
from .rmsnorm_invariant import rmsnorm_invariant
from .nf4_gemm_cuda import nf4_gemm_cuda


def _mp_for(rows: int) -> int:
    return 1 if rows == 1 else 2 if rows == 2 else 4 if rows <= 4 else 8


def _nf4_forward(m, orig):
    def fwd(x):
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] > 8 or flat.dtype not in (torch.float16, torch.bfloat16):
            return orig(x)
        mp = _mp_for(flat.shape[0])
        y = nf4_gemm_cuda(flat, m.weight.data, m.weight.quant_state, m.bias, mp, 4, 2 if mp <= 4 else 4)
        return y.reshape(*x.shape[:-1], y.shape[-1])
    return fwd


# Set to an int32 device scalar holding the block's first absolute position when running on a static cache / CUDA graph.
PREFIX_TENSOR = None


@contextmanager
def invariant_mode(model, ops=("lm_head", "rmsnorm", "sdpa", "nf4")):
    saved, orig_sdpa = [], F.scaled_dot_product_attention
    try:
        for name, m in model.named_modules():
            if "nf4" in ops and type(m).__name__ == "Linear4bit":
                saved.append((m, m.__dict__.get("forward")))
                m.forward = _nf4_forward(m, m.forward)
            elif "lm_head" in ops and name == "lm_head":
                w = m.weight
                saved.append((m, m.__dict__.get("forward")))
                m.forward = (lambda x, _w=w: lm_head_invariant(x.reshape(-1, x.shape[-1]), _w).reshape(*x.shape[:-1], _w.shape[0])
                             if x.reshape(-1, x.shape[-1]).shape[0] <= MP else F.linear(x, _w))
            elif "rmsnorm" in ops and "rmsnorm" in type(m).__name__.lower():
                saved.append((m, m.__dict__.get("forward")))
                eps = getattr(m, "variance_epsilon", getattr(m, "eps", 1e-6))
                m.forward = lambda x, _m=m, _e=eps: rmsnorm_invariant(x, _m.weight, _e)
        if "sdpa" in ops:
            n_rep = max(1, getattr(model.config, "num_attention_heads", 1) // getattr(model.config, "num_key_value_heads", 1))

            def inv_sdpa(q, k, v, *a, **kw):
                M = q.shape[-2]
                if M > MP or q.shape[0] != 1 or q.dtype not in (torch.float16, torch.bfloat16) or q.shape[-1] not in (64, 128):
                    return orig_sdpa(q, k, v, *a, **kw)
                if n_rep > 1 and k.shape[1] == q.shape[1]:
                    k, v = k[:, ::n_rep], v[:, ::n_rep]
                if PREFIX_TENSOR is not None:
                    return attention_invariant(q, k, v, 0, scale=kw.get("scale"), prefix_tensor=PREFIX_TENSOR)
                return attention_invariant(q, k, v, k.shape[-2] - M, scale=kw.get("scale"))

            F.scaled_dot_product_attention = inv_sdpa
        yield
    finally:
        F.scaled_dot_product_attention = orig_sdpa
        for m, f in saved:
            if f is None:
                del m.__dict__["forward"]
            else:
                m.forward = f
