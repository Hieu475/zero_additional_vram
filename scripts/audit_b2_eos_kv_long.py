#!/usr/bin/env python3
"""Per-step KV/token audit of the B2 loops (eager and lazy LM head) with EOS-terminating and long generations (development only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION, original_bnb_forward
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from audit_b2_eos_termination import CHAT_QUESTIONS, SYNTHETIC
import benchmark_bnb_nf4_activationvec_fallback_closed_loop as eager_mod
import benchmark_b2_lazy_lm_head as lazy_mod


def eos_aware_reference(model, ids, steps, capacity, eos):
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    plen = int(ids.shape[-1])
    tokens, digests = [], {}
    with torch.inference_mode(), original_bnb_forward(model):
        logits = model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
        digests[0] = diagnostic_cache_prefix_digest(cache, plen)
        for n in range(steps):
            token = logits.argmax(dim=-1)
            tokens.append(int(token.item()))
            if tokens[-1] == eos or n + 1 == steps:
                break
            logits = model(token, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
            digests[n + 1] = diagnostic_cache_prefix_digest(cache, plen + n + 1)
    return tokens, digests


def audit(run, model, tok, ids, steps, cap, ref_tokens, ref_digests):
    seen, plen = {}, int(ids.shape[-1])

    def obs(cache):
        n = cache.get_seq_length() - plen
        seen[n] = diagnostic_cache_prefix_digest(cache, plen + n)

    tokens, _ = run(obs, cap)
    mismatched = [n for n, d in seen.items() if ref_digests.get(n) != d]
    return {"tokens_match": tokens == ref_tokens, "observed_prefixes": len(seen), "kv_mismatch_prefixes": mismatched,
            "kv_equal": bool(seen) and not mismatched}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--long-prompt-ids", nargs="*", default=[])
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    device, eos = "cuda:0", loaded.tokenizer.eos_token_id
    eager_mod._eos = lazy_mod._eos = eos
    chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
    prompts = ([(f"chat_gsm_{pid}", chat(rows[pid]["prompt"])) for pid in args.long_prompt_ids]
               + [(f"chat_{i}", chat(q)) for i, q in enumerate(CHAT_QUESTIONS)]
               + [(f"synthetic_{i}", s) for i, s in enumerate(SYNTHETIC)])
    per = []
    for pid, text in prompts:
        ids = tok(text, return_tensors="pt")["input_ids"].to(device)
        cap = int(ids.shape[-1]) + args.max_new_tokens + 8
        ref_tokens, ref_digests = eos_aware_reference(model, ids, args.max_new_tokens, cap, eos)
        eager = audit(lambda obs, c: eager_mod.generate(model, tok, ids, k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                      device=device, capacity=c, fallback_on_empty=True, observer=obs), model, tok, ids, args.max_new_tokens, cap, ref_tokens, ref_digests)
        lazy = audit(lambda obs, c: lazy_mod.generate_lazy(model, tok, ids, max_new_tokens=args.max_new_tokens, device=device,
                      capacity=c, observer=obs), model, tok, ids, args.max_new_tokens, cap, ref_tokens, ref_digests)
        per.append({"prompt_id": pid, "bnb_tokens": len(ref_tokens), "ended_with_eos": ref_tokens[-1] == eos, "eager": eager, "lazy": lazy})
        print(json.dumps({"prompt_id": pid, "tokens": len(ref_tokens), "eos": ref_tokens[-1] == eos, "eager_ok": eager["tokens_match"] and eager["kv_equal"],
                          "lazy_ok": lazy["tokens_match"] and lazy["kv_equal"], "prefixes": lazy["observed_prefixes"]}))
    summary = {"prompts": len(per), "ended_with_eos": sum(p["ended_with_eos"] for p in per),
               "max_tokens_seen": max(p["bnb_tokens"] for p in per),
               "prompts_ge_256_tokens": sum(p["bnb_tokens"] >= 256 for p in per),
               "all_passed": all(p[k]["tokens_match"] and p[k]["kv_equal"] for p in per for k in ("eager", "lazy"))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "zassd-b2-eos-kv-long-v1", "max_new_tokens": args.max_new_tokens, "summary": summary, "per_prompt": per}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
