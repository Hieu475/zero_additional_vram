#!/usr/bin/env python3
"""Is a B2/BNB decode forward CPU-launch-bound? Compare enqueue time to total wall time (development only)."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from benchmark_bnb_nf4_activationvec_fallback_closed_loop import SwitchableSchedule


def measure(model, ids, tokens, mode, device, cap, reps=30, lazy=False):
    cache = TargetKVCache(backend="static", max_capacity=cap)
    enq, wall = [], []
    with torch.inference_mode():
        sch = SwitchableSchedule(model)
        sch.__enter__()
        try:
            sch.use("plain")
            model(ids, past_key_values=cache.cache, use_cache=True)
            sch.use(mode)
            t = torch.tensor([tokens], device=device, dtype=ids.dtype)
            base = cache.get_seq_length()
            for i in range(reps + 3):
                cache.crop(base)
                torch.cuda.synchronize()
                s = time.perf_counter()
                if lazy:
                    h = model.model(input_ids=t, past_key_values=cache.cache, use_cache=True).last_hidden_state
                    out = model.lm_head(h[:, :1, :])
                else:
                    out = model(t, past_key_values=cache.cache, use_cache=True).logits
                e = time.perf_counter()
                torch.cuda.synchronize()
                w = time.perf_counter()
                if i >= 3:
                    enq.append((e - s) * 1000); wall.append((w - s) * 1000)
        finally:
            sch.use("patched"); sch.__exit__(None, None, None)
    return {"enqueue_ms": statistics.median(enq), "wall_ms": statistics.median(wall)}


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
    model, tok = loaded.model.eval(), loaded.tokenizer
    ids = tok(rows[args.prompt_id]["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    cap = int(ids.shape[-1]) + 16
    last = ids[0, -3:].tolist()
    res = {"bnb_m1": measure(model, ids, last[:1], "plain", "cuda:0", cap),
           "b2_m3_block_eager_lm_head": measure(model, ids, last, "patched", "cuda:0", cap),
           "b2_m3_block_one_lazy_row": measure(model, ids, last, "patched", "cuda:0", cap, lazy=True)}
    for v in res.values():
        v["cpu_bound_fraction"] = v["enqueue_ms"] / v["wall_ms"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "zassd-b2-launch-bound-v1", "prompt_id": args.prompt_id, **res}, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
