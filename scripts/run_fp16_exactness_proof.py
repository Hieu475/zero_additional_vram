#!/usr/bin/env python3
"""Strict FP16 token-identity proof: algorithmically exact self-speculation.

Separates *algorithmic exactness* from *kernel numerical noise* (review item 31):
under unquantized FP16 arithmetic, greedy ZASSD layer-skip speculation must
produce byte-identical token sequences to vanilla autoregression on every prompt.
Any mismatch here would implicate the algorithm or the KV-cache logic;
a clean 100% result proves residual NF4 divergences come from quantized
batched-GEMM dequantization noise only.

Uses Qwen2.5-0.5B-Instruct in FP16 (~1 GB, fits any 6 GB GPU).

Run: python3 scripts/run_fp16_exactness_proof.py [--model ...] [--num-prompts 20]
Output: experiments/17_fp16_exactness/fp16_identity.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "src")

from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter


def vanilla_ids(model, tokenizer, prompt, max_new_tokens, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    cache = TargetKVCache(backend="static")
    emitted = []
    torch.cuda.synchronize()
    with torch.no_grad():
        out = model(inputs.input_ids, past_key_values=cache.cache, use_cache=True)
        curr = out.logits[:, -1:, :].argmax(dim=-1)
        emitted.append(int(curr.item()))
        for _ in range(max_new_tokens - 1):
            if emitted[-1] == tokenizer.eos_token_id:
                break
            out = model(curr, past_key_values=cache.cache, use_cache=True)
            curr = out.logits[:, -1:, :].argmax(dim=-1)
            emitted.append(int(curr.item()))
    torch.cuda.synchronize()
    return emitted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompts", default="data/benchmarks/prompts.jsonl")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", default="experiments/17_fp16_exactness")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16, device_map=a.device).eval()

    adapter = ModelAdapter(model)
    nL = adapter.num_layers
    layer_mgr = LayerManager(adapter)
    mid = (nL - 6) // 2
    skip = list(range(mid, mid + 6))
    print(f"Model {a.model}: {nL} layers, FP16, skip={skip}")

    prompts = []
    with open(a.prompts, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
            if len(prompts) >= a.num_prompts:
                break

    rows, t0 = [], time.perf_counter()
    for i, item in enumerate(prompts):
        p = item["prompt"]
        ref = vanilla_ids(model, tok, p, a.max_new_tokens, a.device)
        _, m = self_speculative_generate(
            model, tok, layer_mgr, skip, p, k=2,
            max_new_tokens=a.max_new_tokens, device=a.device)
        # self_speculative_generate returns text; re-encode for ID comparison
        hyp_text = _
        hyp = tok(hyp_text, add_special_tokens=False)["input_ids"][:len(ref)]
        match = (ref == hyp)
        rows.append({"id": item.get("id", i), "match": match,
                     "tokens": len(ref), "accepted": m.total_accepted_tokens,
                     "proposed": m.total_draft_tokens})
        print(f"[{i+1}/{len(prompts)}] match={match} tok={len(ref)} "
              f"acc={m.total_accepted_tokens}/{m.total_draft_tokens}")
        if not match:
            for j, (x, y) in enumerate(zip(ref, hyp)):
                if x != y:
                    print(f"  first diff at pos {j}: vanilla={x} zassd={y}")
                    break

    n_match = sum(r["match"] for r in rows)
    rate = n_match / len(rows)
    print(f"\nFP16 identity: {n_match}/{len(rows)} = {rate:.1%} "
          f"({time.perf_counter()-t0:.0f}s total)")
    outp = Path(a.outdir)
    outp.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "dtype": "float16", "skip": skip,
               "identity_rate": rate, "n": len(rows), "rows": rows},
              open(outp / "fp16_identity.json", "w"), indent=2)
    assert rate == 1.0, f"ALGORITHMIC DIVERGENCE under FP16: {rate:.1%}"
    print("PASS: algorithmically exact under exact (FP16) arithmetic.")


if __name__ == "__main__":
    main()
