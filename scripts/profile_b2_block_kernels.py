#!/usr/bin/env python3
"""GPU-kernel breakdown of one B2 M=3 block forward vs one BNB M=1 forward (development only)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from benchmark_bnb_nf4_activationvec_fallback_closed_loop import SwitchableSchedule


def category(name: str) -> str:
    n = name.lower()
    if "_fused_nf4_m3" in n: return "triton_nf4_m3_linear"
    if "dequant" in n or "gemv" in n or "4bit" in n: return "bnb_4bit_linear"
    if "gemm" in n or "cutlass" in n or "sgemm" in n or "ampere" in n or "sm80" in n or "sm89" in n: return "dense_gemm(lm_head/other)"
    if "attention" in n or "sdpa" in n or "flash" in n or "softmax" in n or "efficient" in n: return "attention"
    if "rms" in n or "pow" in n or "mean" in n or "rsqrt" in n: return "rmsnorm_parts"
    if "copy" in n or "cat" in n or "pad" in n or "memcpy" in n: return "copy/cat/pad"
    return "other_elementwise"


def run(model, ids, tokens, device, capacity, mode):
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.inference_mode():
        schedule = SwitchableSchedule(model)
        schedule.__enter__()
        try:
            schedule.use("plain")
            model(ids, past_key_values=cache.cache, use_cache=True)
            schedule.use(mode)
            t = torch.tensor([tokens], device=device, dtype=ids.dtype)
            for _ in range(2):  # warm up
                c = cache.get_seq_length()
                model(t, past_key_values=cache.cache, use_cache=True); cache.crop(c)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                model(t, past_key_values=cache.cache, use_cache=True)
                torch.cuda.synchronize()
            cache.crop(c)
        finally:
            schedule.use("patched"); schedule.__exit__(None, None, None)
    cats, launches, total = defaultdict(float), defaultdict(int), 0.0
    for e in prof.key_averages():
        if e.device_type.name != "CUDA":
            continue
        cats[category(e.key)] += e.self_device_time_total / 1000
        launches[category(e.key)] += e.count
        total += e.self_device_time_total / 1000
    wall = sum(e.self_cpu_time_total for e in prof.key_averages()) / 1000
    return {"gpu_busy_ms": total, "by_category_ms": dict(cats), "launches": dict(launches)}, prof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    ids = tokenizer(rows[args.prompt_id]["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    cap = int(ids.shape[-1]) + 32
    toks = ids[0, -3:].tolist()
    res = {}
    res["bnb_m1"], _ = run(model, ids, toks[:1], "cuda:0", cap, "plain")
    res["b2_m3_block"], prof = run(model, ids, toks, "cuda:0", cap, "patched")
    top = sorted(((e.key, e.self_device_time_total / 1000, e.count) for e in prof.key_averages() if e.device_type.name == "CUDA"),
                 key=lambda x: -x[1])[:12]
    res["b2_m3_top_kernels"] = [{"name": n[:90], "ms": round(ms, 3), "count": c} for n, ms, c in top]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "zassd-b2-block-kernel-profile-v1", "prompt_id": args.prompt_id, **res}, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
