#!/usr/bin/env python3
"""Speed and exactness of a torch.compile'd static-cache BNB decode vs eager BNB greedy (development only)."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import StaticCache

from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from _common import run_vanilla_timed


def decode(model, fwd, ids, n, max_len, device):
    cache = StaticCache(config=model.config, max_cache_len=max_len)
    plen = int(ids.shape[1])
    torch.cuda.synchronize()
    s = time.perf_counter()
    with torch.inference_mode():
        out = model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device=device))
        tok = out.logits[:, -1:].argmax(-1)
        toks = [tok]
        for i in range(n - 1):
            out = fwd(input_ids=tok, past_key_values=cache, use_cache=True,
                      cache_position=torch.tensor([plen + i], device=device))
            tok = out.logits[:, -1:].argmax(-1)
            toks.append(tok)
        seq = torch.cat(toks, 1)[0].tolist()
    torch.cuda.synchronize()
    return seq, time.perf_counter() - s


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
    t0 = time.perf_counter()
    for _ in range(3):
        decode(model, compiled, first, args.max_new_tokens, args.max_cache_len, device)
    warmup_s = time.perf_counter() - t0
    per = []
    for pid in args.prompt_ids:
        ids = tok(rows[pid]["prompt"], return_tensors="pt")["input_ids"].to(device)
        eager_t, comp_t = [], []
        for _ in range(args.rounds):
            ref, _, _, ms = run_vanilla_timed(model, tok, ids, args.max_new_tokens, device)
            eager_t.append(ms / 1000)
            seq, t = decode(model, compiled, ids, args.max_new_tokens, args.max_cache_len, device)
            comp_t.append(t)
        div = next((i for i, (a, b) in enumerate(zip(seq, ref)) if a != b), None)
        per.append({"prompt_id": pid, "eager_s": statistics.median(eager_t), "compiled_s": statistics.median(comp_t),
                    "compiled_speedup_vs_eager": statistics.median(eager_t) / statistics.median(comp_t),
                    "compiled_tokens_equal_eager": seq == ref, "first_divergence": div})
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in per[-1].items()}))
    summary = {"prompts": len(per), "warmup_s": warmup_s,
               "geomean_compiled_speedup_vs_eager": statistics.geometric_mean(p["compiled_speedup_vs_eager"] for p in per),
               "compiled_equal_eager_prompts": sum(p["compiled_tokens_equal_eager"] for p in per)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "zassd-compiled-bnb-baseline-v1", "benchmark": args.benchmark,
                                       "max_new_tokens": args.max_new_tokens, "summary": summary, "per_prompt": per}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
