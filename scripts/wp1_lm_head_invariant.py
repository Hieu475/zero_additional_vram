#!/usr/bin/env python3
"""WP1 gate test: bitwise M-invariance and cost of the invariant LM head vs cuBLAS."""
import argparse, json, time
import torch
from zassd.kernels.lm_head_invariant import lm_head_invariant, MP

ap = argparse.ArgumentParser()
ap.add_argument("--V", type=int, default=151936); ap.add_argument("--K", type=int, default=2048)
ap.add_argument("--bv", type=int, default=64); ap.add_argument("--bk", type=int, default=128); ap.add_argument("--warps", type=int, default=4)
ap.add_argument("--trials", type=int, default=200); ap.add_argument("--output", default=None)
a = ap.parse_args()
import os; DT = torch.bfloat16 if os.environ.get("DTYPE") == "bf16" else torch.float16
torch.manual_seed(0); dev = "cuda:0"
w = (torch.randn(a.V, a.K, device=dev) * 0.05).to(DT).contiguous()
f = lambda x: lm_head_invariant(x, w, a.bv, a.bk, a.warps)
# 1. bitwise invariance: row i computed alone (M=1) vs inside random blocks of every size M, at random positions
bad = 0; checked = 0
for t in range(a.trials):
    X = (torch.randn(MP, a.K, device=dev) * (1 + 3 * torch.rand(1, device=dev))).to(DT)
    ref = torch.cat([f(X[i:i + 1]) for i in range(MP)])
    for M in range(2, MP + 1):
        perm = torch.randperm(MP, device=dev)[:M]
        out = f(X[perm])
        for j, i in enumerate(perm.tolist()):
            checked += 1; bad += int(not torch.equal(out[j], ref[i]))
print(f"invariance: {bad} mismatching rows of {checked}")
# 2. agreement with cuBLAS (information only; different reduction order by design)
X = (torch.randn(MP, a.K, device=dev)).to(DT); r = f(X); c = X @ w.T
print("max |diff| vs cuBLAS:", float((r.float() - c.float()).abs().max()), "argmax agree:", float((r.argmax(-1) == c.argmax(-1)).float().mean()))
# 3. timing
def bench(fn, n=50):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
res = {"V": a.V, "K": a.K, "bv": a.bv, "bk": a.bk, "warps": a.warps, "mismatch_rows": bad, "rows_checked": checked}
for M in (1, 3, 5, 8):
    x = X[:M].contiguous()
    res[f"inv_ms_M{M}"] = bench(lambda: f(x)); res[f"cublas_ms_M{M}"] = bench(lambda: x @ w.T)
res["inv_ms_M{}x{}".format(3, "rowwise")] = bench(lambda: [f(X[i:i+1]) for i in range(3)])
res["cublas_rowwise_M3_ms"] = bench(lambda: [X[i:i+1] @ w.T for i in range(3)])
print(json.dumps(res, indent=1))
if a.output: open(a.output, "w").write(json.dumps(res, indent=1))
