#!/usr/bin/env python3
"""D: block verification with a real draft model (small greedy draft, K drafted tokens per cycle) vs sequential greedy decoding.

Same exactness test as audit_pld_trajectory_divergence.py but the drafts come from a smaller model of the same family run
greedily (fp16, not quantized). Drafting precision is irrelevant to exactness (a wrong draft is simply rejected); only the
target's verification pass matters. `--rowwise` evaluates the listed operator classes of the TARGET row by row in that pass.
Reports divergence from the sequential reference and the draft acceptance statistics (HF eager harness: no speed claims).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_pld_trajectory_divergence import ROWWISE_CHOICES, reference, rowwise


def draft_tokens(draft, dcache, dtoks, seq, k, device):
    """Bring the draft cache in line with `seq` (committed tokens + pending), then greedily draft k tokens."""
    common = 0
    for x, y in zip(dtoks, seq):
        if x != y:
            break
        common += 1
    common = min(common, len(seq) - 1)  # always feed at least the last token to get its logits
    dcache.crop(common)
    del dtoks[common:]
    inp = torch.tensor([seq[common:]], device=device)
    logits = draft(inp, past_key_values=dcache.cache, use_cache=True).logits[:, -1, :]
    dtoks.extend(seq[common:])
    out = []
    for i in range(k):
        t = int(logits.argmax(-1).item()); out.append(t)
        if i < k - 1:
            logits = draft(torch.tensor([[t]], device=device), past_key_values=dcache.cache, use_cache=True).logits[:, -1, :]
            dtoks.append(t)
    return out


def sd_generate(model, draft, ids, n, k, cap, device, ops):
    tc = TargetKVCache(backend="static", max_capacity=cap)
    dc = TargetKVCache(backend="static", max_capacity=cap)
    plen = int(ids.shape[-1])
    pending = int(model(ids, past_key_values=tc.cache, use_cache=True).logits[:, -1, :].argmax(-1).item())
    prompt = ids[0].tolist()
    gen, dtoks, blocks, drafted, accepted = [], list(), 0, 0, 0
    while len(gen) < n:
        gen.append(pending)
        if len(gen) >= n:
            break
        seq = prompt + gen
        kk = min(k, n - len(gen))
        cands = draft_tokens(draft, dc, dtoks, seq, kk, device)
        committed_kv = plen + len(gen) - 1
        inputs = [pending] + cands
        with rowwise(model, set(ops)):
            logits = model(torch.tensor([inputs], device=device), past_key_values=tc.cache, use_cache=True).logits[0]
        preds = logits.argmax(dim=-1).tolist()
        blocks += 1; drafted += len(cands)
        acc = 0
        while acc < len(cands) and preds[acc] == cands[acc] and len(gen) + acc + 1 < n:
            acc += 1
        accepted += acc
        gen.extend(cands[:acc])
        tc.crop(committed_kv + 1 + acc)
        pending = preds[acc]
    return gen[:n], blocks, drafted, accepted


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--backend", default="hf_bnb_nf4")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--draft-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--draft-revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    ap.add_argument("--rowwise", default="none")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    ops = set() if a.rowwise == "none" else (ROWWISE_CHOICES if a.rowwise == "all" else set(a.rowwise.split(",")))
    assert ops <= ROWWISE_CHOICES
    rows = [json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip()][a.start:a.start + a.n_prompts]
    loaded = load_backend(a.backend, a.model, a.revision, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    draft = AutoModelForCausalLM.from_pretrained(a.draft_model, revision=a.draft_revision, dtype=torch.float16, device_map="cuda:0").eval()
    per = []
    with torch.inference_mode():
        for r in rows:
            ids = tok(r["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
            cap = int(ids.shape[-1]) + a.max_new + a.k + 8
            ref = reference(model, ids, a.max_new, cap)
            got, blocks, drafted, accepted = sd_generate(model, draft, ids, a.max_new, a.k, cap, "cuda:0", ops)
            first = next((i for i, (x, y) in enumerate(zip(ref, got)) if x != y), None)
            per.append({"id": str(r["id"]), "diverged": first is not None, "first_divergence": first, "blocks": blocks,
                        "drafted": drafted, "accepted": accepted})
            print(json.dumps(per[-1]), flush=True)
    div = sum(p["diverged"] for p in per)
    out = {"schema": "zassd-draftmodel-sd-divergence-v1", "backend": a.backend, "model": a.model, "draft_model": a.draft_model,
           "k": a.k, "rowwise": sorted(ops), "start": a.start, "max_new": a.max_new, "prompts": len(per), "diverged": div,
           "diverged_fraction": div / len(per),
           "acceptance_rate": sum(p["accepted"] for p in per) / max(1, sum(p["drafted"] for p in per)),
           "tokens_per_block": (sum(p["accepted"] for p in per) + sum(p["blocks"] for p in per)) / max(1, sum(p["blocks"] for p in per)),
           "per_prompt": per}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{a.model} + draft {a.draft_model} k={a.k} rowwise={sorted(ops)}: diverged {div}/{len(per)}, "
          f"acceptance {out['acceptance_rate']:.2f}, tokens/block {out['tokens_per_block']:.2f}")


if __name__ == "__main__":
    main()
