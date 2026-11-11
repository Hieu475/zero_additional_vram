#!/usr/bin/env python3
"""WP2 gate test: bitwise M-invariance, accuracy and cost of the invariant decode attention vs torch SDPA."""
import argparse, json, time
import torch, torch.nn.functional as F
from zassd.kernels.attention_invariant import attention_invariant, MP

ap = argparse.ArgumentParser()
ap.add_argument("--Hq", type=int, default=16); ap.add_argument("--Hkv", type=int, default=2); ap.add_argument("--D", type=int, default=128)
ap.add_argument("--ctx", type=int, nargs="+", default=[128, 300, 511, 512, 513, 1000, 1025, 2000, 3000])
ap.add_argument("--trials", type=int, default=10); ap.add_argument("--output", default=None)
a = ap.parse_args()
import os; DT = torch.bfloat16 if os.environ.get("DTYPE") == "bf16" else torch.float16
torch.manual_seed(0); dev = "cuda:0"; cap = max(a.ctx) + 32
res = {"Hq": a.Hq, "Hkv": a.Hkv, "D": a.D, "per_ctx": {}}
tot_bad = tot_rows = 0
def bench(fn, n=30):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
for ctx in a.ctx:
    K = torch.randn(1, a.Hkv, cap, a.D, device=dev).to(DT); V = torch.randn(1, a.Hkv, cap, a.D, device=dev).to(DT)
    bad = rows = 0; maxerr = 0.0
    for t in range(a.trials):
        Q = torch.randn(1, a.Hq, MP, a.D, device=dev).to(DT)
        # row i alone at absolute position ctx+i (prefix = ctx+i) == the M=1 step
        ref = [attention_invariant(Q[:, :, i:i + 1], K, V, ctx + i) for i in range(MP)]
        for M in range(2, 9):
            out = attention_invariant(Q[:, :, :M], K, V, ctx)
            for i in range(M):
                rows += 1; bad += int(not torch.equal(out[:, :, i], ref[i][:, :, 0]))
        # accuracy vs fp32 reference for the full 8-row block
        M = 8
        kk = K[:, :, :ctx + M].float().repeat_interleave(a.Hq // a.Hkv, 1); vv = V[:, :, :ctx + M].float().repeat_interleave(a.Hq // a.Hkv, 1)
        mask = torch.arange(ctx + M, device=dev)[None, :] <= (ctx + torch.arange(M, device=dev))[:, None]
        r32 = F.scaled_dot_product_attention(Q[:, :, :M].float(), kk, vv, attn_mask=mask)
        maxerr = max(maxerr, float((attention_invariant(Q[:, :, :M], K, V, ctx).float() - r32).abs().max()))
    tot_bad += bad; tot_rows += rows
    q1 = Q[:, :, :1]; q3 = Q[:, :, :3]
    k1 = K[:, :, :ctx + 1]; v1 = V[:, :, :ctx + 1]; k3 = K[:, :, :ctx + 3]; v3 = V[:, :, :ctx + 3]
    sd1 = bench(lambda: F.scaled_dot_product_attention(q1, k1, v1, enable_gqa=True))
    m3 = (torch.arange(ctx + 3, device=dev)[None, :] <= (ctx + torch.arange(3, device=dev))[:, None])
    sd3 = bench(lambda: F.scaled_dot_product_attention(q3, k3, v3, attn_mask=m3, enable_gqa=True))
    i1 = bench(lambda: attention_invariant(q1, K, V, ctx)); i3 = bench(lambda: attention_invariant(q3, K, V, ctx))
    res["per_ctx"][ctx] = {"mismatch_rows": bad, "rows": rows, "max_abs_err_vs_fp32": maxerr, "inv_ms_M1": i1, "sdpa_ms_M1": sd1, "inv_ms_M3": i3, "sdpa_ms_M3": sd3}
    print(ctx, json.dumps(res["per_ctx"][ctx]), flush=True)
res["total_mismatch_rows"] = tot_bad; res["total_rows"] = tot_rows
print("TOTAL invariance mismatches:", tot_bad, "of", tot_rows)
if a.output: open(a.output, "w").write(json.dumps(res, indent=1))
