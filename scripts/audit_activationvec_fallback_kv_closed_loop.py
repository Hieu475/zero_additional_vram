#!/usr/bin/env python3
"""Per-step KV-prefix and token audit of the fallback activationvec loop vs original BNB greedy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION, original_bnb_forward
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest
from benchmark_bnb_nf4_activationvec_fallback_closed_loop import generate


def bnb_reference(model, prompt_ids, steps, capacity):
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    plen = int(prompt_ids.shape[-1])
    tokens, digests = [], {}
    with torch.inference_mode(), original_bnb_forward(model):
        logits = model(prompt_ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
        digests[0] = diagnostic_cache_prefix_digest(cache, plen)
        for n in range(steps):
            token = logits.argmax(dim=-1)
            tokens.append(int(token.item()))
            if n + 1 == steps:
                break
            logits = model(token, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
            digests[n + 1] = diagnostic_cache_prefix_digest(cache, plen + n + 1)
    return tokens, digests


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    per_prompt = []
    for pid in args.prompt_ids:
        prompt_ids = tokenizer(rows[pid]["prompt"], return_tensors="pt")["input_ids"].to(device)
        plen = int(prompt_ids.shape[-1])
        capacity = plen + args.max_new_tokens + 8
        ref_tokens, ref_digests = bnb_reference(model, prompt_ids, args.max_new_tokens, capacity)
        for fallback in (True, False):
            observed = {}

            def observer(cache):
                n = cache.get_seq_length() - plen
                observed[n] = diagnostic_cache_prefix_digest(cache, plen + n)

            tokens, stats = generate(model, tokenizer, prompt_ids, k=2, ngram_size=3,
                                     max_new_tokens=args.max_new_tokens, device=device, capacity=capacity,
                                     fallback_on_empty=fallback, observer=observer)
            # Final-step KV may stop one short of the reference; compare tokens on the emitted length.
            n_tok = min(len(tokens), len(ref_tokens))
            mismatches = [n for n, d in observed.items() if n in ref_digests and ref_digests[n] != d]
            unknown = [n for n in observed if n not in ref_digests]
            per_prompt.append({
                "prompt_id": pid, "fallback_on_empty": fallback, "tokens_match": tokens[:n_tok] == ref_tokens[:n_tok]
                and len(tokens) == len(ref_tokens),
                "observed_prefixes": len(observed), "kv_mismatch_prefixes": mismatches, "unknown_prefixes": unknown,
                "kv_prefixes_equal": bool(observed) and not mismatches and not unknown,
                "fallback_cycles": stats["fallback_cycles"], "block_cycles": stats["block_cycles"],
                "accepted_draft_tokens": stats["accepted_draft_tokens"], "draft_tokens": stats["draft_tokens"]})
            print(json.dumps({k: per_prompt[-1][k] for k in ("prompt_id", "fallback_on_empty", "tokens_match",
                                                             "observed_prefixes", "kv_prefixes_equal", "block_cycles")}))
    result = {
        "schema": "zassd-activationvec-fallback-kv-closed-loop-v1",
        "classification": "development-only per-step KV/token audit; finite evidence, not a proof",
        "model": MODEL, "revision": REVISION, "benchmark": args.benchmark, "max_new_tokens": args.max_new_tokens,
        "per_prompt": per_prompt,
        "all_passed": all(p["tokens_match"] and p["kv_prefixes_equal"] for p in per_prompt),
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"torch": torch.__version__, "bitsandbytes": importlib.metadata.version("bitsandbytes"),
                    "gpu": torch.cuda.get_device_name(0)},
        "scope_limits": ["Development prompts only; GSM8K held-out indices 300-349 excluded.",
                         "KV compared at non-terminal committed prefixes; reference is original BNB greedy."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("all_passed", result["all_passed"])
    if not result["all_passed"]:
        raise RuntimeError("Fallback activationvec loop diverged from original BNB tokens/KV")


if __name__ == "__main__":
    main()
