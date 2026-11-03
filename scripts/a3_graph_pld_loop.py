#!/usr/bin/env python3
"""A3/A4: exact PLD (K=2) decode loop on CUDA graphs with HF StaticCache, vs the M=1 StaticCache reference.

Graphs: g1 = M=1 body + LM head + argmax (fallback and the 'baseline graph exact' decoder); g3 = row-wise M=3 body.
Row-wise LM head is lazy and eager (outside the graph). A block is used only when the n-gram lookup yields 2 drafts;
otherwise the M=1 graph runs. Checks generated tokens and committed KV bitwise vs the eager M=1 reference and times:
eager M=1 reference, M=1 graph (baseline graph exact), and this loop.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import StaticCache

from zassd.backends.base import load_backend
from zassd.decoding.ngram import find_candidate_tokens
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from a1_rowwise_static_block import rollback, rowwise_ops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl")
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--ngram", type=int, default=3)
    ap.add_argument("--max-cache-len", type=int, default=448)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    bench = {str(r["id"]): r for r in (json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    dev, M = "cuda:0", a.k + 1
    cache = StaticCache(config=model.config, max_cache_len=a.max_cache_len)
    ids1 = torch.zeros((1, 1), dtype=torch.long, device=dev); cp1 = torch.zeros(1, dtype=torch.long, device=dev)
    ids3 = torch.zeros((1, M), dtype=torch.long, device=dev); cp3 = torch.arange(M, device=dev)

    def body1():
        h = model.model(input_ids=ids1, past_key_values=cache, use_cache=True, cache_position=cp1).last_hidden_state
        return model.lm_head(h[:, -1:, :]).argmax(dim=-1)

    def body3():
        with rowwise_ops():
            return model.model(input_ids=ids3, past_key_values=cache, use_cache=True, cache_position=cp3).last_hidden_state

    def prefill(ids, plen):
        if cache.layers[0].is_initialized:
            rollback(cache, 0)
        return model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device=dev)).logits[:, -1:]

    graphs = {}
    results, tot = [], {"eager": 0.0, "g1": 0.0, "a3": 0.0, "tokens": 0}
    for pid in a.prompt_ids:
        ids = tok(bench[pid]["prompt"], return_tensors="pt")["input_ids"].to(dev)
        plen = int(ids.shape[1]); n = a.max_new
        assert plen + n + M + 4 <= a.max_cache_len, (pid, plen)
        prompt = ids[0].tolist()
        with torch.inference_mode():
            # eager M=1 reference (timed)
            rc = StaticCache(config=model.config, max_cache_len=a.max_cache_len)
            lg = model(ids, past_key_values=rc, use_cache=True, cache_position=torch.arange(plen, device=dev)).logits[:, -1:]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ref = []
            for i in range(n):
                t = lg.argmax(-1); ref.append(int(t.item()))
                lg = model(t, past_key_values=rc, use_cache=True, cache_position=torch.tensor([plen + i], device=dev)).logits[:, -1:]
            torch.cuda.synchronize(); t_eager = time.perf_counter() - t0
            ref_kv = [(l.keys[:, :, :plen + n - 1].clone(), l.values[:, :, :plen + n - 1].clone()) for l in rc.layers]
            del rc
            lg0 = prefill(ids, plen)
            if not graphs:
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        rollback(cache, plen); body1()
                        rollback(cache, plen); body3()
                torch.cuda.current_stream().wait_stream(s)
                for name, fn in (("g1", body1), ("g3", body3)):
                    rollback(cache, plen)
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        graphs[name] = (g, fn())
                lg0 = prefill(ids, plen)
            g1, out1 = graphs["g1"]; g3, out3 = graphs["g3"]
            first = int(lg0.argmax(-1).item())

            # baseline graph exact: M=1 graph for every token
            torch.cuda.synchronize(); t0 = time.perf_counter()
            pend, base = first, []
            for i in range(n):
                base.append(pend)
                if i == n - 1:
                    break
                ids1.copy_(torch.tensor([[pend]], device=dev)); cp1.fill_(plen + i); rollback(cache, plen + i)
                g1.replay(); pend = int(out1.item())
            torch.cuda.synchronize(); t_g1 = time.perf_counter() - t0
            lg0 = prefill(ids, plen)

            # A3 loop
            torch.cuda.synchronize(); t0 = time.perf_counter()
            pend, gen, blocks3, blocks1, acc_total = first, [], 0, 0, 0
            while len(gen) < n:
                gen.append(pend)
                if len(gen) >= n:
                    break
                cands = find_candidate_tokens(prompt + gen, ngram_size=a.ngram, max_candidates=a.k)
                kv_base = plen + len(gen) - 1
                rollback(cache, kv_base)
                if len(cands) == a.k:
                    ids3.copy_(torch.tensor([[pend] + cands], device=dev))
                    cp3.copy_(torch.arange(kv_base, kv_base + M, device=dev))
                    g3.replay(); blocks3 += 1
                    acc = 0
                    while True:
                        p = int(model.lm_head(out3[:, acc:acc + 1, :]).argmax(-1).item())
                        if acc < a.k and p == cands[acc] and len(gen) + acc + 1 < n:
                            acc += 1
                        else:
                            break
                    gen.extend(cands[:acc]); pend = p; acc_total += acc
                    rollback(cache, kv_base + 1 + acc)
                else:
                    ids1.copy_(torch.tensor([[pend]], device=dev)); cp1.fill_(kv_base)
                    g1.replay(); blocks1 += 1; pend = int(out1.item())
            torch.cuda.synchronize(); t_a3 = time.perf_counter() - t0
            gen = gen[:n]
            kv_ok = all(torch.equal(l.keys[:, :, :plen + n - 1], rk) and torch.equal(l.values[:, :, :plen + n - 1], rv)
                        for l, (rk, rv) in zip(cache.layers, ref_kv))
        rec = {"prompt": pid, "prompt_tokens": plen, "tokens_equal": gen == ref, "baseline_graph_tokens_equal": base == ref,
               "committed_kv_bitwise": kv_ok, "blocks_m3": blocks3, "blocks_m1": blocks1, "accepted_drafts": acc_total,
               "s_eager": t_eager, "s_graph_m1": t_g1, "s_a3": t_a3}
        results.append(rec)
        tot["eager"] += t_eager; tot["g1"] += t_g1; tot["a3"] += t_a3; tot["tokens"] += n
        print(json.dumps(rec), flush=True)
    summary = {"schema": "zassd-a3-graph-pld-v1", "k": a.k, "ngram": a.ngram, "max_new": a.max_new, "max_cache_len": a.max_cache_len,
               "all_exact": all(r["tokens_equal"] and r["committed_kv_bitwise"] and r["baseline_graph_tokens_equal"] for r in results),
               "tok_per_s_eager_m1": tot["tokens"] / tot["eager"], "tok_per_s_graph_m1": tot["tokens"] / tot["g1"],
               "tok_per_s_a3": tot["tokens"] / tot["a3"],
               "speedup_a3_over_graph_m1": tot["g1"] / tot["a3"], "speedup_a3_over_eager": tot["eager"] / tot["a3"],
               "speedup_graph_m1_over_eager": tot["eager"] / tot["g1"], "results": results}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
