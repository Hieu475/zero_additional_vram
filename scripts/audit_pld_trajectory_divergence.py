#!/usr/bin/env python3
"""C6: how often does naive PLD (plain block verification, no shape-invariance handling) leave the greedy trajectory?

Reference = sequential M=1 greedy decode (static cache) on the same loaded model. Naive PLD = history n-gram drafts,
block [pending]+K drafts verified by one ordinary forward pass, accept while argmax==draft, crop KV on rejection.
Exact methods would match the reference token-for-token; any mismatch here is attributable to shape-dependent numerics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.ngram import find_candidate_tokens
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION


ROWWISE_CHOICES = {"linear", "rmsnorm", "sdpa", "lm_head"}


def _rowwise_forward(orig):
    def fwd(x, *a, **kw):
        if x.ndim == 3 and x.shape[1] > 1:
            return torch.cat([orig(x[:, i:i + 1, :].contiguous(), *a, **kw) for i in range(x.shape[1])], dim=1)
        return orig(x, *a, **kw)
    return fwd


@contextmanager
def rowwise(model, ops: set[str]):
    """Run the listed operator classes one row at a time (== the M=1 call on the same inputs) during block verification."""
    saved, orig_sdpa = [], F.scaled_dot_product_attention
    try:
        for name, m in model.named_modules():
            kind = None
            if name == "lm_head":
                kind = "lm_head"
            elif "rmsnorm" in type(m).__name__.lower():
                kind = "rmsnorm"
            elif (isinstance(m, Linear4bit) or type(m).__module__.startswith("gemlite")
                  or (isinstance(m, torch.nn.Linear) and type(getattr(m, "weight", None)).__module__.startswith("torchao"))):
                kind = "linear"
            if kind in ops:
                saved.append((m, m.__dict__.get("forward")))
                m.forward = _rowwise_forward(m.forward)
        if "sdpa" in ops:
            def spy(q, k, v, *a, **kw):
                M = q.shape[-2]
                if M == 1:
                    return orig_sdpa(q, k, v, *a, **kw)
                prefix = k.shape[-2] - M
                outs = [orig_sdpa(q[:, :, i:i + 1, :].contiguous(), k[:, :, :prefix + i + 1, :].contiguous(),
                                  v[:, :, :prefix + i + 1, :].contiguous(), attn_mask=None, dropout_p=0.0,
                                  scale=kw.get("scale"), is_causal=False, enable_gqa=True) for i in range(M)]
                return torch.cat(outs, dim=2)
            F.scaled_dot_product_attention = spy
        yield
    finally:
        F.scaled_dot_product_attention = orig_sdpa
        for m, f in saved:
            if f is None:
                del m.__dict__["forward"]
            else:
                m.forward = f


def reference(model, ids, n, cap):
    cache = TargetKVCache(backend="static", max_capacity=cap)
    lg = model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
    out = []
    for _ in range(n):
        t = lg.argmax(dim=-1)
        out.append(int(t.item()))
        lg = model(t, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
    return out


def naive_pld(model, ids, n, k, ngram, cap, device, ops=frozenset()):
    cache = TargetKVCache(backend="static", max_capacity=cap)
    plen = int(ids.shape[-1])
    pending = int(model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1, :].argmax(-1).item())
    prompt = ids[0].tolist()
    gen, blocks, blocks_multi = [], 0, 0
    while len(gen) < n:
        gen.append(pending)  # pending token is committed; its KV is written by the next block's first row
        if len(gen) >= n:
            break
        cands = find_candidate_tokens(prompt + gen, ngram_size=ngram, max_candidates=k)
        inputs = [pending] + cands
        committed_kv = plen + len(gen) - 1
        with rowwise(model, set(ops)):
            logits = model(torch.tensor([inputs], device=device), past_key_values=cache.cache, use_cache=True).logits[0]
        preds = logits.argmax(dim=-1).tolist()
        blocks += 1; blocks_multi += int(len(inputs) > 1)
        acc = 0
        while acc < len(cands) and preds[acc] == cands[acc] and len(gen) + acc + 1 < n:
            acc += 1
        gen.extend(cands[:acc])
        cache.crop(committed_kv + 1 + acc)
        pending = preds[acc]
    return gen[:n], blocks, blocks_multi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl")
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--start", type=int, default=0, help="index of the first benchmark row")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ngram", type=int, default=3)
    ap.add_argument("--backend", default="hf_bnb_nf4")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--rowwise", default="none",
                    help="comma list of operator classes run row-by-row in block verification: linear,rmsnorm,sdpa,lm_head | all | none")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    ops = set() if a.rowwise == "none" else (ROWWISE_CHOICES if a.rowwise == "all" else set(a.rowwise.split(",")))
    assert ops <= ROWWISE_CHOICES, ops
    rows = [json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip()][a.start:a.start + a.n_prompts]
    loaded = load_backend(a.backend, a.model, a.revision, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    per = []
    with torch.inference_mode():
        for r in rows:
            ids = tok(r["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
            cap = int(ids.shape[-1]) + a.max_new + a.k + 8
            ref = reference(model, ids, a.max_new, cap)
            got, blocks, multi = naive_pld(model, ids, a.max_new, a.k, a.ngram, cap, "cuda:0", ops)
            first = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), None)
            per.append({"id": str(r["id"]), "diverged": first is not None, "first_divergence": first,
                        "blocks": blocks, "multi_row_blocks": multi})
            print(json.dumps(per[-1]), flush=True)
    div = [p for p in per if p["diverged"]]
    out = {"schema": "zassd-pld-trajectory-divergence-v1", "backend": a.backend, "model": a.model, "revision": a.revision,
           "start": a.start, "rowwise": sorted(ops), "k": a.k, "ngram": a.ngram, "max_new": a.max_new, "prompts": len(per),
           "diverged": len(div), "diverged_fraction": len(div) / len(per),
           "multi_row_blocks_total": sum(p["multi_row_blocks"] for p in per),
           "first_divergence_median": sorted(p["first_divergence"] for p in div)[len(div) // 2] if div else None,
           "per_prompt": per}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{a.model} {a.backend} k={a.k}: diverged {len(div)}/{len(per)}, multi-row blocks {out['multi_row_blocks_total']}")


if __name__ == "__main__":
    main()
