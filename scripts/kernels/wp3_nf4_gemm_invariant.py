#!/usr/bin/env python3
"""WP3 gate test: NF4 GEMM invariance, accuracy vs dequantised fp32 matmul, and cost vs bitsandbytes."""
import argparse, json, time
import torch, bitsandbytes.functional as BF
from bitsandbytes.nn import Linear4bit
from zassd.kernels.nf4_gemm_invariant import nf4_gemm_invariant, MP
from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=MODEL); ap.add_argument("--revision", default=REVISION)
ap.add_argument("--bn", type=int, default=32); ap.add_argument("--bk", type=int, default=128)
ap.add_argument("--warps", type=int, default=4); ap.add_argument("--stages", type=int, default=3)
ap.add_argument("--trials", type=int, default=20); ap.add_argument("--output", default=None)
a = ap.parse_args()
loaded = load_backend("hf_bnb_nf4", a.model, a.revision, "cuda:0", "sdpa"); model = loaded.model.eval()
layer = model.model.layers[len(model.model.layers) // 2]
mods = {"q_proj": layer.self_attn.q_proj, "k_proj": layer.self_attn.k_proj, "o_proj": layer.self_attn.o_proj,
        "gate_proj": layer.mlp.gate_proj, "down_proj": layer.mlp.down_proj}
dev = "cuda:0"; res = {}
def bench(fn, n=100):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3
tot_bad = tot_rows = 0
for name, m in mods.items():
    w, qs = m.weight, m.weight.quant_state; N, K = qs.shape; DT = m.compute_dtype if hasattr(m, "compute_dtype") and m.compute_dtype else torch.float16
    DT = qs.dtype
    f = lambda x: nf4_gemm_invariant(x, w.data, qs, m.bias, a.bn, a.bk, a.warps, a.stages)
    bad = rows = 0
    for t in range(a.trials):
        X = (torch.randn(MP, K, device=dev) * 0.5).to(DT)
        ref = torch.cat([f(X[i:i + 1]) for i in range(MP)])
        for M in range(2, 9):
            perm = torch.randperm(MP, device=dev)[:M]; out = f(X[perm])
            for j, i in enumerate(perm.tolist()):
                rows += 1; bad += int(not torch.equal(out[j], ref[i]))
    X = (torch.randn(8, K, device=dev) * 0.5).to(DT)
    deq = BF.dequantize_4bit(w.data, qs).float()
    r32 = X.float() @ deq.T + (m.bias.float() if m.bias is not None else 0)
    err = float((f(X).float() - r32).abs().max()); scale = float(r32.abs().max())
    bnb_err = float((m(X[None]).float()[0] - r32).abs().max())
    d = {"N": N, "K": K, "mismatch_rows": bad, "rows": rows, "max_err_vs_fp32": err, "bnb_max_err_vs_fp32": bnb_err, "ref_scale": scale}
    for M in (1, 3, 8):
        x = X[:M].contiguous(); xb = x[None]
        d[f"inv_ms_M{M}"] = bench(lambda: f(x)); d[f"bnb_ms_M{M}"] = bench(lambda: m(xb))
    res[name] = d; tot_bad += bad; tot_rows += rows
    print(name, json.dumps(d), flush=True)
print("TOTAL invariance mismatches:", tot_bad, "of", tot_rows)
res["total_mismatch_rows"] = tot_bad; res["config"] = vars(a)
if a.output: open(a.output, "w").write(json.dumps(res, indent=1))
