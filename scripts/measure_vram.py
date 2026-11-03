#!/usr/bin/env python3
"""Peak GPU memory of eager BNB greedy vs the B2 lazy-LM-head loop, one method per process (development only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION


def bnb_greedy(model, tok, ids, n, cap):
    cache = TargetKVCache(backend="static", max_capacity=cap)
    with torch.no_grad():
        out = model(ids, past_key_values=cache.cache, use_cache=True)
        token = out.logits[:, -1:, :].argmax(dim=-1)
        toks = [int(token.item())]
        while len(toks) < n and toks[-1] != tok.eos_token_id:
            out = model(token, past_key_values=cache.cache, use_cache=True)
            token = out.logits[:, -1:, :].argmax(dim=-1)
            toks.append(int(token.item()))
    return toks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=["bnb", "b2_lazy"], required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    mib = lambda b: b / 2**20
    free0, total = torch.cuda.mem_get_info("cuda:0")
    used_before_ctx = mib(total - free0)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    ids = tok(rows[args.prompt_id]["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    cap = int(ids.shape[-1]) + args.max_new_tokens + 8
    torch.cuda.synchronize()
    after_load = mib(torch.cuda.memory_allocated())
    if args.method == "b2_lazy":
        from benchmark_b2_lazy_lm_head import generate_lazy
        run = lambda: generate_lazy(model, tok, ids, max_new_tokens=args.max_new_tokens, device="cuda:0", capacity=cap)[0]
    else:
        run = lambda: bnb_greedy(model, tok, ids, args.max_new_tokens, cap)
    run()  # warm-up (kernel compilation, allocator warm)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    tokens = run()
    torch.cuda.synchronize()
    free1, _ = torch.cuda.mem_get_info("cuda:0")
    result = {"method": args.method, "prompt_id": args.prompt_id, "prompt_tokens": int(ids.shape[-1]), "capacity": cap,
              "tokens": len(tokens), "after_load_allocated_mib": after_load,
              "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
              "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
              "device_used_mib_end": mib(total - free1), "device_used_mib_before_process_cuda": used_before_ctx,
              "tokens_sha": __import__("hashlib").sha256(json.dumps(tokens).encode()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
