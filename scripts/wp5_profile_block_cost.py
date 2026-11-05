#!/usr/bin/env python3
"""Graph-replay time of a decode step with M rows (M=1..6) under invariant_mode, plus a per-kernel-type breakdown for M=1 vs M=3."""
import contextlib, sys, time, json
import torch
from transformers import StaticCache
import zassd.kernels.invariant_mode as IM
from zassd.backends.base import load_backend
from zassd.kernels.invariant_mode import invariant_mode
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from a1_rowwise_static_block import rollback

model_name = sys.argv[1] if len(sys.argv) > 1 else MODEL
rev = sys.argv[2] if len(sys.argv) > 2 else REVISION
loaded = load_backend("hf_bnb_nf4", model_name, rev, "cuda:0", "sdpa"); model = loaded.model.eval(); dev = "cuda:0"
cache = StaticCache(config=model.config, max_cache_len=448)
pt = torch.zeros(1, dtype=torch.int32, device=dev); IM.PREFIX_TENSOR = pt
res = {}
def mk(M):
    ids = torch.ones((1, M), dtype=torch.long, device=dev); cp = torch.arange(M, device=dev) + 100
    def fn():
        h = model.model(input_ids=ids, past_key_values=cache, use_cache=True, cache_position=cp).last_hidden_state
        return model.lm_head(h).argmax(-1)
    return fn
def timeit(fn, n=100):
    rollback(cache, 100); pt.fill_(100)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): rollback(cache, 100); fn()
    torch.cuda.current_stream().wait_stream(s)
    rollback(cache, 100); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn()
    for _ in range(5): rollback(cache, 100); g.replay()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): rollback(cache, 100); g.replay()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1e3, g
with torch.inference_mode():
    with torch.no_grad():
        model(torch.ones((1, 100), dtype=torch.long, device=dev), past_key_values=cache, use_cache=True, cache_position=torch.arange(100, device=dev))
        with invariant_mode(model):
            for M in (1, 2, 3, 4, 5, 6):
                res[f"inv_M{M}_ms"] = timeit(mk(M))[0]; print(f"invariant M={M}: {res[f'inv_M{M}_ms']:.2f} ms", flush=True)
        for M in (1, 3):
            res[f"stock_M{M}_ms"] = timeit(mk(M))[0]; print(f"stock     M={M}: {res[f'stock_M{M}_ms']:.2f} ms", flush=True)
        # kernel-type breakdown with torch profiler (eager, a few steps) for M=1 and M=3 under invariant_mode
        from torch.profiler import profile, ProfilerActivity
        with invariant_mode(model):
            for M in (1, 3):
                fn = mk(M); rollback(cache, 100); fn(); torch.cuda.synchronize()
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(5): rollback(cache, 100); fn()
                    torch.cuda.synchronize()
                rows = sorted(((e.key, e.device_time_total / 5 / 1e3) for e in prof.key_averages()), key=lambda r: -r[1])[:8]
                print(f"--- top GPU kernels M={M} (ms per step)"); [print(f"{t:7.3f}  {k[:90]}") for k, t in rows]
json.dump(res, open("results/kernels/wp5_block_cost.json", "w"), indent=1)
