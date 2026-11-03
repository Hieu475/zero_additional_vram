#!/usr/bin/env python3
"""A2: CUDA-graph capture of the row-wise M=3 decoder body on HF StaticCache; compare with the M=1 StaticCache reference.

One cache and one graph are reused across prompts (prefill rewrites the buffers in place; rollback resets the
cumulative_length device tensor in place). Per cycle: copy inputs into static tensors, replay, per-row LM head
(eager, outside the graph), compare each committed row's logits bitwise with the M=1 reference, and compare the
committed KV prefix at the end. Also reports eager-block vs graph-block body time.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import StaticCache

from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from a1_rowwise_static_block import rollback, rowwise_ops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl")
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--reject-every", type=int, default=3)
    ap.add_argument("--max-cache-len", type=int, default=768)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    bench = {str(r["id"]): r for r in (json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tok = loaded.model.eval(), loaded.tokenizer
    dev, R = "cuda:0", a.rows
    vocab = int(model.config.vocab_size)
    cache = StaticCache(config=model.config, max_cache_len=a.max_cache_len)
    static_ids = torch.zeros((1, R), dtype=torch.long, device=dev)
    static_cp = torch.arange(R, device=dev)
    graph, static_hidden = None, None
    results, t_eager, t_graph, n_timed = [], 0.0, 0.0, 0

    def body():
        with rowwise_ops():
            return model.model(input_ids=static_ids, past_key_values=cache, use_cache=True,
                               cache_position=static_cp).last_hidden_state

    def prefill(ids, plen):
        rollback(cache, 0) if cache.layers and cache.layers[0].is_initialized else None
        out = model(ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device=dev))
        return out.logits[:, -1:]

    for pid in a.prompt_ids:
        ids = tok(bench[pid]["prompt"], return_tensors="pt")["input_ids"].to(dev)
        plen = int(ids.shape[1])
        with torch.inference_mode():
            # M=1 reference on its own cache
            rc = StaticCache(config=model.config, max_cache_len=a.max_cache_len)
            lg = model(ids, past_key_values=rc, use_cache=True, cache_position=torch.arange(plen, device=dev)).logits[:, -1:]
            ref_tok, ref_logits = [], []
            for i in range(a.steps + R):
                t = lg.argmax(-1); ref_tok.append(int(t.item()))
                lg = model(t, past_key_values=rc, use_cache=True,
                           cache_position=torch.tensor([plen + i], device=dev)).logits[:, -1:]
                ref_logits.append(lg[0, 0].clone())
            ref_kv = [(l.keys[:, :, :plen + a.steps].clone(), l.values[:, :, :plen + a.steps].clone()) for l in rc.layers]
            del rc
            prefill(ids, plen)
            if graph is None:
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(3):
                        static_ids.copy_(torch.tensor([ref_tok[:R]], device=dev)); static_cp.copy_(torch.arange(plen, plen + R, device=dev))
                        rollback(cache, plen); body()
                torch.cuda.current_stream().wait_stream(s)
                rollback(cache, plen)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_hidden = body()
                prefill(ids, plen)  # restore a clean prefix state after warm-up/capture
            pos, cycles, rejects, bad, first_bad, checked = 0, 0, 0, 0, None, 0
            while pos + R <= a.steps:
                reject = a.reject_every > 0 and (cycles % a.reject_every == a.reject_every - 1)
                inputs = ref_tok[pos:pos + R]
                if reject:
                    inputs = [inputs[0]] + [(t + 1) % vocab for t in inputs[1:]]
                static_ids.copy_(torch.tensor([inputs], device=dev))
                static_cp.copy_(torch.arange(plen + pos, plen + pos + R, device=dev))
                rollback(cache, plen + pos)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                graph.replay(); torch.cuda.synchronize(); t_graph += time.perf_counter() - t0
                hidden = static_hidden.clone()
                commit = 1 if reject else R
                for r in range(commit):
                    row = model.lm_head(hidden[:, r:r + 1, :])[0, 0]
                    checked += 1
                    if not torch.equal(row, ref_logits[pos + r]):
                        bad += 1; first_bad = first_bad if first_bad is not None else pos + r
                # eager timing of the same block (state is re-written identically; result unused)
                rollback(cache, plen + pos)
                torch.cuda.synchronize(); t0 = time.perf_counter(); body(); torch.cuda.synchronize(); t_eager += time.perf_counter() - t0
                n_timed += 1
                pos += commit; cycles += 1; rejects += int(reject)
                rollback(cache, plen + pos)
            kv_ok = all(torch.equal(l.keys[:, :, :plen + pos], rk[:, :, :plen + pos]) and
                        torch.equal(l.values[:, :, :plen + pos], rv[:, :, :plen + pos])
                        for l, (rk, rv) in zip(cache.layers, ref_kv))
        results.append({"prompt": pid, "prompt_tokens": plen, "cycles": cycles, "reject_cycles": rejects, "rows_checked": checked,
                        "logit_rows_not_bitwise": bad, "first_bad_step": first_bad, "committed_kv_bitwise": kv_ok})
        print(json.dumps(results[-1]), flush=True)
    summary = {"schema": "zassd-a2-graph-block-v1", "rows": R, "steps": a.steps, "reject_every": a.reject_every,
               "body_ms_eager_per_block": 1000 * t_eager / max(n_timed, 1), "body_ms_graph_per_block": 1000 * t_graph / max(n_timed, 1),
               "all_exact": all(r["logit_rows_not_bitwise"] == 0 and r["committed_kv_bitwise"] for r in results), "results": results}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
