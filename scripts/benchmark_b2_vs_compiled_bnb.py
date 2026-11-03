#!/usr/bin/env python3
"""Same-run, per-prompt interleaved timing: BNB eager vs compiled static-cache BNB vs B2 lazy-LM-head loop (development only)."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from benchmark_b2_lazy_lm_head import generate_lazy
from benchmark_compiled_bnb_baseline import decode
from _common import run_vanilla_timed

METHODS = ("bnb_eager", "bnb_compiled", "b2_lazy")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-cache-len", type=int, default=640)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    compiled = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
    first = tok(rows[args.prompt_ids[0]]["prompt"], return_tensors="pt")["input_ids"].to(device)
    for _ in range(3):
        decode(model, compiled, first, args.max_new_tokens, args.max_cache_len, device)
    per = []
    for pid in args.prompt_ids:
        ids = tok(rows[pid]["prompt"], return_tensors="pt")["input_ids"].to(device)
        cap = int(ids.shape[1]) + args.max_new_tokens + 8
        times = {m: [] for m in METHODS}
        toks = {}
        for r in range(args.rounds):
            for m in METHODS[r % 3:] + METHODS[:r % 3]:
                if m == "bnb_eager":
                    t, _, _, ms = run_vanilla_timed(model, tok, ids, args.max_new_tokens, device)
                    times[m].append(ms / 1000)
                elif m == "bnb_compiled":
                    t, s = decode(model, compiled, ids, args.max_new_tokens, args.max_cache_len, device)
                    times[m].append(s)
                else:
                    t, st = generate_lazy(model, tok, ids, max_new_tokens=args.max_new_tokens, device=device, capacity=cap)
                    times[m].append(st["total_s"])
                toks[m] = t
        med = {m: statistics.median(v) for m, v in times.items()}
        per.append({"prompt_id": pid, "median_s": med, "b2_lazy_over_eager": med["bnb_eager"] / med["b2_lazy"],
                    "compiled_over_eager": med["bnb_eager"] / med["bnb_compiled"],
                    "b2_lazy_over_compiled": med["bnb_compiled"] / med["b2_lazy"],
                    "b2_lazy_equals_eager": toks["b2_lazy"] == toks["bnb_eager"],
                    "compiled_equals_eager": toks["bnb_compiled"] == toks["bnb_eager"],
                    "compiled_time_samples_s": times["bnb_compiled"]})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in per[-1].items() if k not in ("median_s", "compiled_time_samples_s")}))
    gm = lambda k: statistics.geometric_mean(p[k] for p in per)
    summary = {"prompts": len(per), "geomean_b2_lazy_over_eager": gm("b2_lazy_over_eager"),
               "geomean_compiled_over_eager": gm("compiled_over_eager"), "geomean_b2_lazy_over_compiled": gm("b2_lazy_over_compiled"),
               "b2_lazy_equals_eager_prompts": sum(p["b2_lazy_equals_eager"] for p in per),
               "compiled_equals_eager_prompts": sum(p["compiled_equals_eager"] for p in per)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "zassd-b2-vs-compiled-bnb-v1", "benchmark": args.benchmark,
                                       "max_new_tokens": args.max_new_tokens, "summary": summary, "per_prompt": per}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
