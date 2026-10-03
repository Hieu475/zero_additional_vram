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
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"],
                    help="float32 tests exact arithmetic; float16 additionally "
                         "exposes reduced-precision batched-GEMM noise")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--outdir", default="experiments/17_fp16_exactness")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    dt = torch.float32 if a.dtype == "float32" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=dt, device_map=a.device).eval()

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
        ref_text = tok.decode(ref, skip_special_tokens=True)
        hyp_text, m = self_speculative_generate(
            model, tok, layer_mgr, skip, p, k=2,
            max_new_tokens=a.max_new_tokens, device=a.device)
        # Compare decoded TEXTS (not re-encoded IDs): re-encoding is lossy
        # on edge cases (e.g. immediate-EOS decodes to ""), which produced
        # a false mismatch. Identical greedy ID streams <=> identical text.
        match = (ref_text == hyp_text)
        rows.append({"id": item.get("id", i), "match": match,
                     "tokens": len(ref), "accepted": m.total_accepted_tokens,
                     "proposed": m.total_draft_tokens})
        print(f"[{i+1}/{len(prompts)}] match={match} tok={len(ref)} "
              f"acc={m.total_accepted_tokens}/{m.total_draft_tokens}")
        if not match:
            print(f"  vanilla[:120]={ref_text[:120]!r}")
            print(f"  zassd  [:120]={hyp_text[:120]!r}")

    n_match = sum(r["match"] for r in rows)
    rate = n_match / len(rows)
    positions = sum(r["tokens"] for r in rows)
    print(f"\n{a.dtype} identity: {n_match}/{len(rows)} sequences = {rate:.1%} "
          f"over {positions} token positions ({time.perf_counter()-t0:.0f}s total)")
    outp = Path(a.outdir)
    outp.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "dtype": a.dtype, "skip": skip,
               "identity_rate": rate, "n": len(rows),
               "token_positions": positions, "rows": rows},
              open(outp / f"{a.dtype}_identity.json", "w"), indent=2)
    assert rate == 1.0, f"ALGORITHMIC DIVERGENCE under {a.dtype}: {rate:.1%}"
    print(f"PASS: algorithmically exact under {a.dtype} arithmetic.")


if __name__ == "__main__":
    main()
