#!/usr/bin/env python3
"""WP5b: exact draft-model speculative decoding on CUDA graphs with the M-invariant target kernels.

Target: bnb NF4 model under invariant_mode (reference = eager M=1 StaticCache decode under invariant_mode). Draft: small
model (any numerics; only the target decides tokens). Per block: draft chain graph (K greedy steps, on-device feedback)
-> target block graph (K+1 rows) -> accept -> roll back target (and draft) caches. Reports tokens/KV exactness and
tokens/s for: eager reference, target M=1 graph (baseline graph exact), this loop.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, StaticCache

import zassd.kernels.invariant_mode as IM
from zassd.backends.base import load_backend
from zassd.kernels.invariant_mode import invariant_mode
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from a1_rowwise_static_block import rollback


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl")
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-cache-len", type=int, default=448)
    ap.add_argument("--model", default=MODEL); ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--draft", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--draft-revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    ap.add_argument("--draft-backend", default="bf16", choices=["bf16", "hf_bnb_nf4"])
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    bench = {str(r["id"]): r for r in (json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", a.model, a.revision, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    dev, K = "cuda:0", a.k
    M = K + 1
    if a.draft_backend == "bf16":
        draft = AutoModelForCausalLM.from_pretrained(a.draft, revision=a.draft_revision, dtype=torch.bfloat16).to(dev).eval()
    else:
        draft = load_backend("hf_bnb_nf4", a.draft, a.draft_revision, dev, "sdpa").model.eval()
    cache = StaticCache(config=model.config, max_cache_len=a.max_cache_len)
    dcache = StaticCache(config=draft.config, max_cache_len=a.max_cache_len)
    prefix_t = torch.zeros(1, dtype=torch.int32, device=dev); IM.PREFIX_TENSOR = prefix_t
    ids1 = torch.zeros((1, 1), dtype=torch.long, device=dev); cp1 = torch.zeros(1, dtype=torch.long, device=dev)
    idsB = torch.zeros((1, M), dtype=torch.long, device=dev); cpB = torch.arange(M, device=dev)
    d_ids = torch.zeros((1, 1), dtype=torch.long, device=dev); d_base = torch.zeros(1, dtype=torch.long, device=dev)
    stack = contextlib.ExitStack(); stack.enter_context(invariant_mode(model))

    def body1():
        h = model.model(input_ids=ids1, past_key_values=cache, use_cache=True, cache_position=cp1).last_hidden_state
        return model.lm_head(h[:, -1:, :]).argmax(dim=-1)

    def bodyB():
        h = model.model(input_ids=idsB, past_key_values=cache, use_cache=True, cache_position=cpB).last_hidden_state
        return model.lm_head(h).argmax(dim=-1)

    def draft_step(tok_in, pos_t):
        h = draft.model(input_ids=tok_in, past_key_values=dcache, use_cache=True, cache_position=pos_t).last_hidden_state
        return draft.lm_head(h[:, -1:, :]).argmax(dim=-1)

    def draft_chain():  # K greedy steps; feedback on device; returns [K] tokens
        cur, outs = d_ids, []
        for j in range(K):
            nxt = draft_step(cur, d_base + j)
            outs.append(nxt.reshape(1)); cur = nxt.reshape(1, 1)
        return torch.cat(outs)

    def draft_one():
        return draft_step(d_ids, d_base)

    def prefill_t(ids, plen):
        if cache.layers[0].is_initialized:
            rollback(cache, 0)
        return model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device=dev)).logits[:, -1:]

    def prefill_d(ids, plen):
        if dcache.layers[0].is_initialized:
            rollback(dcache, 0)
        draft(ids, past_key_values=dcache, use_cache=True, cache_position=torch.arange(plen, device=dev))

    graphs = {}
    results, tot = [], {"eager": 0.0, "g1": 0.0, "sd": 0.0, "tokens": 0, "blocks": 0, "acc": 0}
    for pid in a.prompt_ids:
        ids = tok(bench[pid]["prompt"], return_tensors="pt")["input_ids"].to(dev)
        plen = int(ids.shape[1]); n = a.max_new
        assert plen + n + M + 4 <= a.max_cache_len, (pid, plen)
        with torch.inference_mode():
            IM.PREFIX_TENSOR = prefix_t
            rc = StaticCache(config=model.config, max_cache_len=a.max_cache_len)
            lg = model(ids, past_key_values=rc, use_cache=True, cache_position=torch.arange(plen, device=dev)).logits[:, -1:]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ref = []
            for i in range(n):
                t = lg.argmax(-1); ref.append(int(t.item())); prefix_t.fill_(plen + i)
                lg = model(t, past_key_values=rc, use_cache=True, cache_position=torch.tensor([plen + i], device=dev)).logits[:, -1:]
            torch.cuda.synchronize(); t_eager = time.perf_counter() - t0
            ref_kv = [(l.keys[:, :, :plen + n - 1].clone(), l.values[:, :, :plen + n - 1].clone()) for l in rc.layers]
            del rc
            lg0 = prefill_t(ids, plen); prefill_d(ids, plen)
            if not graphs:
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        rollback(cache, plen); body1(); rollback(cache, plen); bodyB()
                        rollback(dcache, plen); draft_chain(); rollback(dcache, plen); draft_one()
                torch.cuda.current_stream().wait_stream(s)
                for name, fn in [("g1", body1), ("gB", bodyB), ("dchain", draft_chain), ("done", draft_one)]:
                    c = dcache if name.startswith("d") else cache
                    rollback(c, plen)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        graphs[name] = (g, fn())
                lg0 = prefill_t(ids, plen); prefill_d(ids, plen)
            g1, out1 = graphs["g1"]; gB, outB = graphs["gB"]; gD, outD = graphs["dchain"]; gD1, _ = graphs["done"]
            first = int(lg0.argmax(-1).item())

            torch.cuda.synchronize(); t0 = time.perf_counter()
            pend, base = first, []
            for i in range(n):
                base.append(pend)
                if i == n - 1:
                    break
                ids1.copy_(torch.tensor([[pend]], device=dev)); cp1.fill_(plen + i); prefix_t.fill_(plen + i); rollback(cache, plen + i)
                g1.replay(); pend = int(out1.item())
            torch.cuda.synchronize(); t_g1 = time.perf_counter() - t0
            lg0 = prefill_t(ids, plen); prefill_d(ids, plen)

            torch.cuda.synchronize(); t0 = time.perf_counter()
            pend, gen, blocks, acc_total = first, [], 0, 0
            while len(gen) < n:
                gen.append(pend)
                if len(gen) >= n:
                    break
                kv_base = plen + len(gen) - 1
                # draft: cache holds keys for positions < kv_base after rollback; first step consumes `pend` at kv_base
                rollback(dcache, kv_base); d_ids.fill_(pend); d_base.fill_(kv_base); gD.replay()
                idsB[0, 0] = pend; idsB[0, 1:].copy_(outD)
                rollback(cache, kv_base); cpB.copy_(torch.arange(kv_base, kv_base + M, device=dev)); prefix_t.fill_(kv_base)
                gB.replay(); blocks += 1
                preds = outB[0].tolist(); cands = outD.tolist()
                acc = 0
                while acc < K and preds[acc] == cands[acc] and len(gen) + acc + 1 < n:
                    acc += 1
                gen.extend(cands[:acc]); pend = preds[acc]; acc_total += acc
                if acc == K:  # draft never consumed its last token: advance its cache over it
                    rollback(dcache, kv_base + K); d_ids.fill_(cands[K - 1]); d_base.fill_(kv_base + K); gD1.replay()
            torch.cuda.synchronize(); t_sd = time.perf_counter() - t0
            gen = gen[:n]
            rollback(cache, plen + n - 1)
            kv_ok = all(torch.equal(l.keys[:, :, :plen + n - 1], rk) and torch.equal(l.values[:, :, :plen + n - 1], rv)
                        for l, (rk, rv) in zip(cache.layers, ref_kv))
        rec = {"prompt": pid, "prompt_tokens": plen, "tokens_equal": gen == ref, "baseline_graph_tokens_equal": base == ref,
               "committed_kv_bitwise": kv_ok, "blocks": blocks, "accepted_drafts": acc_total, "s_eager": t_eager, "s_graph_m1": t_g1, "s_sd": t_sd}
        results.append(rec); tot["eager"] += t_eager; tot["g1"] += t_g1; tot["sd"] += t_sd; tot["tokens"] += n; tot["blocks"] += blocks; tot["acc"] += acc_total
        print(json.dumps(rec), flush=True)
    summary = {"schema": "zassd-wp5b-invariant-graph-draft-sd-v1", "model": a.model, "draft": a.draft, "draft_backend": a.draft_backend, "k": K,
               "max_new": a.max_new, "max_cache_len": a.max_cache_len,
               "all_exact": all(r["tokens_equal"] and r["committed_kv_bitwise"] and r["baseline_graph_tokens_equal"] for r in results),
               "tok_per_s_eager_m1": tot["tokens"] / tot["eager"], "tok_per_s_graph_m1": tot["tokens"] / tot["g1"], "tok_per_s_sd": tot["tokens"] / tot["sd"],
               "speedup_sd_over_graph_m1": tot["g1"] / tot["sd"], "speedup_sd_over_eager": tot["eager"] / tot["sd"],
               "accepted_per_block": tot["acc"] / max(1, tot["blocks"]), "results": results}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
