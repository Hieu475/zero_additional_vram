#!/usr/bin/env python3
"""Feasibility probe: raw CUDA-graph capture of one M=1 (or M=R) decode step of HF+bitsandbytes with a static cache.

Captures the unmodified eager kernels (no torch.compile), replays them, and compares tokens/logits bitwise with eager
execution under the same StaticCache (development only).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import StaticCache

from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl")
    ap.add_argument("--prompt-id", default="gsm8k_000")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--max-cache-len", type=int, default=512)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    ids = tok(rows[args.prompt_id]["prompt"], return_tensors="pt")["input_ids"].to(device)
    plen = int(ids.shape[1])
    R = args.rows

    def fresh_cache():
        cache = StaticCache(config=model.config, max_cache_len=args.max_cache_len)
        with torch.inference_mode():
            out = model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device=device))
        return cache, out.logits[:, -1:].argmax(-1)

    result = {"rows": R, "prompt_tokens": plen}
    # eager reference under StaticCache
    cache, first = fresh_cache()
    tokens_in = first.repeat(1, R)
    eager_logits = []
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.inference_mode():
        for i in range(args.steps):
            pos = torch.arange(plen + i, plen + i + R, device=device)
            out = model(input_ids=tokens_in, past_key_values=cache, use_cache=True, cache_position=pos)
            eager_logits.append(out.logits[:, -1].clone())
            tokens_in = out.logits[:, -1:].argmax(-1).repeat(1, R)
    torch.cuda.synchronize(); result["eager_ms_per_step"] = (time.perf_counter() - t0) * 1000 / args.steps

    # graph capture
    cache, first = fresh_cache()
    static_ids = first.repeat(1, R).clone()
    static_pos = torch.arange(plen, plen + R, device=device)
    try:
        with torch.inference_mode():
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):  # warm-up on a throw-away cache copy region (positions overwritten later)
                    model(input_ids=static_ids, past_key_values=cache, use_cache=True, cache_position=static_pos)
            torch.cuda.current_stream().wait_stream(s)
            cache, first = fresh_cache()
            static_ids.copy_(first.repeat(1, R)); static_pos.copy_(torch.arange(plen, plen + R, device=device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = model(input_ids=static_ids, past_key_values=cache, use_cache=True, cache_position=static_pos)
            # replays must start from a fresh prefix state: rebuild the cache contents in place
            cache2, first = fresh_cache()
            for l1, l2 in zip(cache.layers, cache2.layers):
                l1.keys.copy_(l2.keys); l1.values.copy_(l2.values)
            static_ids.copy_(first.repeat(1, R)); static_pos.copy_(torch.arange(plen, plen + R, device=device))
            graph_logits = []
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for i in range(args.steps):
                graph.replay()
                graph_logits.append(static_out.logits[:, -1].clone())
                static_ids.copy_(static_out.logits[:, -1:].argmax(-1).repeat(1, R))
                static_pos.add_(R if False else 1)
            torch.cuda.synchronize(); result["graph_ms_per_step"] = (time.perf_counter() - t0) * 1000 / args.steps
        same = [bool(torch.equal(a, b)) for a, b in zip(eager_logits, graph_logits)]
        result["captured"] = True
        result["logits_bitwise_equal_steps"] = f"{sum(same)}/{len(same)}"
        result["first_unequal_step"] = next((i for i, v in enumerate(same) if not v), None)
        result["speedup_graph_over_eager"] = result["eager_ms_per_step"] / result["graph_ms_per_step"]
    except Exception as e:  # report failure as a result, not a crash
        result["captured"] = False
        result["error"] = f"{type(e).__name__}: {str(e)[:400]}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
