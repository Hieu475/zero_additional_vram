#!/usr/bin/env python3
"""A1: row-wise M=3 block on HF StaticCache must equal the StaticCache M=1 decode bitwise (eager, no graph).

Block path = shared 4-bit projections over M rows (bitwise row-invariant for M<=3, see C2) + per-row RMSNorm, per-row
SDPA over the full static KV with the row's mask slice, per-row LM head. Reference = ordinary M=1 steps on StaticCache.
Teacher-forced on reference tokens; some cycles use wrong drafts for rows>0 and commit only row 0 (stale KV is later
overwritten), exercising rollback by repositioning. Compares logits per committed row and the committed KV prefix.
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import StaticCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION


@contextmanager
def rowwise_ops():
    orig_norm = Qwen2RMSNorm.forward
    orig_attn = ALL_ATTENTION_FUNCTIONS["sdpa"]

    def norm(self, x):
        if x.shape[-2] == 1:
            return orig_norm(self, x)
        return torch.cat([orig_norm(self, x[..., i:i + 1, :].contiguous()) for i in range(x.shape[-2])], dim=-2)

    def attn(module, q, k, v, mask, **kw):
        M = q.shape[2]
        if M == 1:
            return orig_attn(module, q, k, v, mask, **kw)
        outs = []
        for i in range(M):
            mi = None if mask is None else mask[:, :, i:i + 1, :].contiguous()
            o, _ = orig_attn(module, q[:, :, i:i + 1, :].contiguous(), k, v, mi, **kw)
            outs.append(o)  # (B, 1, H, D) layout returned by sdpa_attention_forward
        return torch.cat(outs, dim=1), None

    Qwen2RMSNorm.forward = norm
    ALL_ATTENTION_FUNCTIONS["sdpa"] = attn
    try:
        yield
    finally:
        Qwen2RMSNorm.forward = orig_norm
        ALL_ATTENTION_FUNCTIONS["sdpa"] = orig_attn


def rollback(cache, length: int) -> None:
    """HF StaticLayer.update writes at cumulative_length (cache_position is ignored for the write), so rollback must
    reset that device tensor in place (static address, graph-compatible)."""
    for layer in cache.layers:
        layer.cumulative_length.fill_(length)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl")
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--max-cache-len", type=int, default=768)
    ap.add_argument("--reject-every", type=int, default=3, help="every Nth cycle uses wrong drafts and commits 1 row (0=never)")
    ap.add_argument("--zero-stale", action="store_true", help="diagnostic: zero KV beyond committed prefix after each reject")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    bench = {str(r["id"]): r for r in (json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    dev, R = "cuda:0", a.rows
    vocab = int(model.config.vocab_size)
    results = []
    for pid in a.prompt_ids:
        ids = tok(bench[pid]["prompt"], return_tensors="pt")["input_ids"].to(dev)
        plen = int(ids.shape[1])

        def prefill():
            cache = StaticCache(config=model.config, max_cache_len=a.max_cache_len)
            out = model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device=dev))
            return cache, out.logits[:, -1:]

        with torch.inference_mode():
            cache, lg = prefill()
            ref_tok, ref_logits = [], []  # ref_logits[i] = logits after consuming ref_tok[i]
            for i in range(a.steps + R):
                t = lg.argmax(-1); ref_tok.append(int(t.item()))
                lg = model(t, past_key_values=cache, use_cache=True,
                           cache_position=torch.tensor([plen + i], device=dev)).logits[:, -1:]
                ref_logits.append(lg[0, 0].clone())
            ref_kv = [(l.keys[:, :, :plen + a.steps].clone(), l.values[:, :, :plen + a.steps].clone()) for l in cache.layers]
            # block path
            cache, lg = prefill()
            pos, bad_logit_rows, rows_checked, cycles, rejects = 0, 0, 0, 0, 0
            first_bad = None
            while pos + R <= a.steps:
                reject = a.reject_every > 0 and (cycles % a.reject_every == a.reject_every - 1)
                inputs = ref_tok[pos:pos + R]
                if reject:
                    inputs = [inputs[0]] + [(t + 1) % vocab for t in inputs[1:]]
                cp = torch.arange(plen + pos, plen + pos + R, device=dev)
                with rowwise_ops():
                    hidden = model.model(input_ids=torch.tensor([inputs], device=dev), past_key_values=cache,
                                         use_cache=True, cache_position=cp).last_hidden_state
                commit = 1 if reject else R
                for r in range(commit):
                    row = model.lm_head(hidden[:, r:r + 1, :])[0, 0]
                    rows_checked += 1
                    if not torch.equal(row, ref_logits[pos + r]):
                        bad_logit_rows += 1
                        first_bad = first_bad if first_bad is not None else pos + r
                pos += commit; cycles += 1; rejects += int(reject)
                rollback(cache, plen + pos)
                if reject and a.zero_stale:
                    for l in cache.layers:
                        l.keys[:, :, plen + pos:].zero_(); l.values[:, :, plen + pos:].zero_()
            kv_equal = all(torch.equal(l.keys[:, :, :plen + pos], rk[:, :, :plen + pos]) and
                           torch.equal(l.values[:, :, :plen + pos], rv[:, :, :plen + pos])
                           for l, (rk, rv) in zip(cache.layers, ref_kv))
        results.append({"prompt": pid, "prompt_tokens": plen, "cycles": cycles, "reject_cycles": rejects,
                        "rows_checked": rows_checked, "logit_rows_not_bitwise": bad_logit_rows,
                        "first_bad_step": first_bad, "committed_kv_bitwise": kv_equal})
        print(json.dumps(results[-1]), flush=True)
    summary = {"schema": "zassd-a1-rowwise-static-v1", "rows": R, "steps": a.steps, "results": results,
               "all_exact": all(r["logit_rows_not_bitwise"] == 0 and r["committed_kv_bitwise"] for r in results)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print("ALL_EXACT" if summary["all_exact"] else "NOT_EXACT")


if __name__ == "__main__":
    main()
