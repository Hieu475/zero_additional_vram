#!/usr/bin/env python3
"""Measured auxiliary-draft VRAM Pareto (GPU, review item 18).

The honest memory argument is NOT "every auxiliary draft OOMs" — a 0.5B NF4
draft (~0.4 GB) fits next to most targets. It is that auxiliary drafts cost
Delta_M_aux > 0 weight VRAM and shrink KV/context headroom, while ZASSD
costs exactly 0.0 MB. This script MEASURES (not estimates) peak weight
footprints by loading each draft candidate standalone on the RTX 4050.

Run: python3 scripts/measure_draft_vram_pareto.py
Output: experiments/16_hybrid_roofline/draft_vram_pareto.json
"""
import json
import sys
import time
from pathlib import Path

import torch

OUT = Path("experiments/16_hybrid_roofline/draft_vram_pareto.json")

CANDIDATES = [
    ("Qwen/Qwen2.5-0.5B-Instruct", "fp16", {}),
    ("Qwen/Qwen2.5-0.5B-Instruct", "nf4", {"load_in_4bit": True}),
    ("Qwen/Qwen2.5-1.5B-Instruct", "fp16", {}),
    ("Qwen/Qwen2.5-1.5B-Instruct", "nf4", {"load_in_4bit": True}),
]


def measure(model_id: str, quant: str):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    torch.cuda.init()  # required before memory APIs in this environment
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(0) / 1024**2
    kw = {"device_map": "cuda:0", "torch_dtype": torch.float16}
    if quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(0) / 1024**2
    dt = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    del model
    torch.cuda.empty_cache()
    return {"model": model_id, "quant": quant, "params_B": round(n_params / 1e9, 2),
            "weight_vram_mb": round(peak - base, 1), "load_s": round(dt, 1)}


def main():
    rows = []
    for mid, quant, _ in CANDIDATES:
        try:
            r = measure(mid, quant)
        except Exception as e:
            r = {"model": mid, "quant": quant, "error": str(e)[:200]}
        rows.append(r)
        print(r)
    # Honest boundary arithmetic (usable 5691 MB; 7B NF4 target ~3946.6 MB
    # as previously measured; KV/activation budget 500 MB):
    usable, target7b, kv = 5691.0, 3946.6, 500.0
    print(f"\n{'draft':32s} {'MB':>8s}  7B-NF4 total  headroom  status")
    for r in rows:
        if "error" in r:
            print(f"{r['model'].split('/')[-1]+'/'+r['quant']:32s} FAILED: {r['error'][:80]}")
            continue
        tot = target7b + r["weight_vram_mb"] + kv
        head = usable - tot
        print(f"{r['model'].split('/')[-1]+'/'+r['quant']:32s} {r['weight_vram_mb']:8.1f}  "
              f"{tot:9.1f}  {head:8.1f}  {'OOM' if head < 0 else 'fits'}")
    print(f"{'ZASSD (0.0 MB aux)':32s} {'0.0':>8s}  "
          f"{target7b + kv:9.1f}  {usable - target7b - kv:8.1f}  fits")
    json.dump({"rows": rows, "usable_mb": usable, "target7b_nf4_mb": target7b,
               "kv_budget_mb": kv,
               "claim": "Delta_M_aux > 0 always; OOM only for large drafts. "
                        "ZASSD Delta_M_weights = 0 exactly."},
              open(OUT, "w"), indent=2)
    print(f"DONE -> {OUT}")


if __name__ == "__main__":
    main()
