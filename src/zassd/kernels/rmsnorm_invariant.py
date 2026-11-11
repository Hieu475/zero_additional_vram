"""M-invariant RMSNorm: one program per row, fixed-width fp32 reduction (row result depends only on that row)."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, y_ptr, H: tl.constexpr, eps, BLOCK: tl.constexpr, DT: tl.constexpr):
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    x = tl.load(x_ptr + r * H + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    y = (x * tl.math.rsqrt(var + eps)).to(DT)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + r * H + cols, w * y, mask=mask)


def rmsnorm_invariant(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    shp = x.shape
    H = shp[-1]
    x2 = x.reshape(-1, H).contiguous()
    y = torch.empty_like(x2)
    _rmsnorm_kernel[(x2.shape[0],)](x2, weight, y, H=H, eps=eps, BLOCK=triton.next_power_of_2(H), DT=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16, num_warps=4)
    return y.reshape(shp)
