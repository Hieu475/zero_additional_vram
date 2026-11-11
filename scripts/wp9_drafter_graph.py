#!/usr/bin/env python3
"""WP9: exact speculative decoding with a learned drafter, fully inside one CUDA graph per step (no host sync per step).

Step graph (block M = K+1): verify [pend, d1..dK] with the M-invariant kernels -> acc = longest matching prefix -> commit (pos, KV length,
output buffer) -> run the drafter from the hidden row of the last accepted position to produce the next K drafts.
Drafts are only proposals: output is exact w.r.t. sequential M=1 greedy under the same invariant kernels (checked bitwise: tokens and KV).
Baselines: g1 async (M=1, in-graph token feedback, no per-token sync) and g1 sync (host reads each token; the earlier definition).
Reference for exactness: eager M=1 on every --ref-every-th prompt; the graph baselines are checked against it and all runs against g1.
"""
from __future__ import annotations
import argparse, copy, json, time, types, inspect, contextlib
from pathlib import Path
import torch
from transformers import StaticCache
from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
import zassd.kernels.invariant_mode as IM
from zassd.kernels.invariant_mode import invariant_mode
from zassd.kernels.drafter import Drafter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True); ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--k", type=int, default=None, help="use only the first k drafter steps (<= trained K)"); ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--max-cache-len", type=int, default=1024); ap.add_argument("--ref-every", type=int, default=10)
    ap.add_argument("--model", default=MODEL); ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    rows = [json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip()][:a.limit]
    L = load_backend("hf_bnb_nf4", a.model, a.revision, "cuda:0", "sdpa"); model, tok = L.model.eval(), L.tokenizer
    dev = "cuda:0"
    ck = torch.load(a.ckpt, map_location=dev); K = min(ck["K"], a.k or ck["K"]); M = K + 1
    vid = ck["state"]["vocab_ids"]; dr = Drafter(ck.get("d", 2048), ck.get("ff", 4096), vid, ck["state"]["head"]).to(dev); dr.load_state_dict(ck["state"]); dr = dr.to(torch.bfloat16).eval()
    emb = model.get_input_embeddings().weight
    cfg = copy.deepcopy(model.config)
    if getattr(cfg, "sliding_window", None) is not None: cfg.sliding_window = None
    cache = StaticCache(config=cfg, max_cache_len=a.max_cache_len)
    bf = torch.bfloat16
    def draft(s, cur):   # one drafter step; inputs cast to bf16 to match training (autocast bf16)
        s = dr.step(s.to(bf), emb[cur].to(bf)); return s, vid[dr.logits(s).argmax(-1)]
    n = a.max_new; SLACK = 80
    i64 = dict(dtype=torch.long, device=dev)
    pos_t = torch.zeros(1, **i64); pend_t = torch.zeros(1, **i64); gl_t = torch.zeros(1, **i64)
    drafts_t = torch.zeros(K, **i64); out_buf = torch.zeros(n + SLACK, **i64)
    accsum_t = torch.zeros(1, **i64); steps_t = torch.zeros(1, **i64)
    prefix_t = torch.zeros(1, dtype=torch.int32, device=dev); IM.PREFIX_TENSOR = prefix_t
    arM = torch.arange(M, **i64)

    def set_len(v):
        for l in cache.layers: l.cumulative_length.copy_(v.reshape(()) if torch.is_tensor(v) else v)

    def spec_step():
        ids = torch.cat([pend_t, drafts_t]).unsqueeze(0)
        prefix_t.copy_(pos_t.int())
        h = model.model(input_ids=ids, past_key_values=cache, use_cache=True, cache_position=pos_t + arM).last_hidden_state
        preds = model.lm_head(h[0]).argmax(-1)                      # [M]
        acc = (preds[:K] == drafts_t).long().cumprod(0).sum().view(1)
        newpend = preds.index_select(0, acc)
        out_buf.index_copy_(0, gl_t + arM, ids[0])
        gl_t.add_(1 + acc); pos_t.add_(1 + acc); set_len(pos_t); accsum_t.add_(acc); steps_t.add_(1)
        s = h[0].index_select(0, acc); cur = newpend; new = []
        for _ in range(K):
            s, cur = draft(s, cur); new.append(cur)
        drafts_t.copy_(torch.cat(new)); pend_t.copy_(newpend)

    def base_step():
        prefix_t.copy_(pos_t.int())
        h = model.model(input_ids=pend_t.view(1, 1), past_key_values=cache, use_cache=True, cache_position=pos_t).last_hidden_state
        nxt = model.lm_head(h[:, -1:]).argmax(-1).view(1)
        out_buf.index_copy_(0, gl_t, pend_t); gl_t.add_(1); pos_t.add_(1); set_len(pos_t); pend_t.copy_(nxt)

    def prefill(ids, plen):
        set_len(0)
        h = model.model(input_ids=ids, past_key_values=cache, use_cache=True, cache_position=torch.arange(plen, device=dev)).last_hidden_state
        return h, model.lm_head(h)[:, -1]                           # same op as model(...).logits (full rows)

    def init_state(h, lg, plen, with_drafts):
        pos_t.fill_(plen); gl_t.zero_(); accsum_t.zero_(); steps_t.zero_(); out_buf.zero_()
        first = lg.argmax(-1).view(1); pend_t.copy_(first); set_len(plen)
        if with_drafts:
            s = h[0, -1:]; cur = first; new = []
            for _ in range(K):
                s, cur = draft(s, cur); new.append(cur)
            drafts_t.copy_(torch.cat(new))

    for m in model.modules():   # LongRoPE guard (as in wp5)
        if getattr(m, "rope_type", None) == "longrope":
            m.forward = types.MethodType(inspect.unwrap(type(m).forward), m)
    eos_ids = {tok.eos_token_id}
    stack = contextlib.ExitStack(); stack.enter_context(invariant_mode(model, ("lm_head", "rmsnorm", "sdpa", "nf4")))
    graphs = {}; results = []; tot = {"g1s": 0.0, "g1a": 0.0, "spec": 0.0, "tok": 0, "acc": 0, "steps": 0}
    with torch.inference_mode():
        for pi, r in enumerate(rows):
            ids = tok(r["prompt"], return_tensors="pt")["input_ids"].to(dev); plen = int(ids.shape[1])
            assert plen + n + SLACK + 4 * M <= a.max_cache_len, (r["id"], plen)
            ref = None
            if pi % a.ref_every == 0:                                # eager M=1 reference (StaticCache, host loop)
                rc = StaticCache(config=cfg, max_cache_len=a.max_cache_len)
                lg = model(ids, past_key_values=rc, use_cache=True, cache_position=torch.arange(plen, device=dev)).logits[:, -1:]
                ref = []
                for i in range(n):
                    t = lg.argmax(-1); ref.append(int(t.item())); prefix_t.fill_(plen + i)
                    lg = model(t, past_key_values=rc, use_cache=True, cache_position=torch.tensor([plen + i], device=dev)).logits[:, -1:]
                del rc
            h, lg = prefill(ids, plen); init_state(h, lg, plen, True)
            if not graphs:
                s_ = torch.cuda.Stream(); s_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_):
                    for _ in range(3): spec_step(); init_state(h, lg, plen, True); base_step(); init_state(h, lg, plen, True)
                torch.cuda.current_stream().wait_stream(s_)
                for name, fn in (("spec", spec_step), ("base", base_step)):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g): fn()
                    graphs[name] = g
                h, lg = prefill(ids, plen); init_state(h, lg, plen, True)
            gs, gb = graphs["spec"], graphs["base"]
            # g1 async
            init_state(h, lg, plen, False); torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(n): gb.replay()
            torch.cuda.synchronize(); t_g1a = time.perf_counter() - t0
            base_tok = out_buf[:n].tolist(); base_kv = [(l.keys[:, :, :plen + n - 1].clone(), l.values[:, :, :plen + n - 1].clone()) for l in cache.layers]
            # g1 sync (host reads every token)
            h, lg = prefill(ids, plen); init_state(h, lg, plen, False); torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(n): gb.replay(); int(pend_t.item())
            torch.cuda.synchronize(); t_g1s = time.perf_counter() - t0
            # spec
            h, lg = prefill(ids, plen); init_state(h, lg, plen, True); torch.cuda.synchronize(); t0 = time.perf_counter()
            while True:
                for _ in range(4): gs.replay()
                if int(gl_t.item()) >= n: break
            torch.cuda.synchronize(); t_spec = time.perf_counter() - t0
            gen = out_buf[:n].tolist(); steps = int(steps_t.item()); accs = int(accsum_t.item())
            kv_ok = all(torch.equal(l.keys[:, :, :plen + n - 1], rk) and torch.equal(l.values[:, :, :plen + n - 1], rv) for l, (rk, rv) in zip(cache.layers, base_kv))
            rec = {"prompt": r["id"], "plen": plen, "tokens_equal_g1": gen == base_tok, "kv_bitwise_g1": kv_ok,
                   "g1_equal_eager": None if ref is None else base_tok == ref, "spec_equal_eager": None if ref is None else gen == ref,
                   "steps": steps, "mean_acc": accs / max(1, steps), "s_g1_sync": t_g1s, "s_g1_async": t_g1a, "s_spec": t_spec,
                   "eos_pos": next((i for i, t_ in enumerate(base_tok) if t_ in eos_ids), None)}
            results.append(rec); print(json.dumps(rec), flush=True)
            tot["g1s"] += t_g1s; tot["g1a"] += t_g1a; tot["spec"] += t_spec; tot["tok"] += n; tot["acc"] += accs; tot["steps"] += steps
    chk = [x for x in results if x["spec_equal_eager"] is not None]
    summ = {"schema": "zassd-wp9-v1", "k": K, "ckpt": a.ckpt, "benchmark": a.benchmark, "n_prompts": len(results),
            "all_exact_vs_g1": all(x["tokens_equal_g1"] and x["kv_bitwise_g1"] for x in results),
            "eager_checked": len(chk), "all_exact_vs_eager": all(x["spec_equal_eager"] and x["g1_equal_eager"] for x in chk),
            "tok_per_s_g1_sync": tot["tok"] / tot["g1s"], "tok_per_s_g1_async": tot["tok"] / tot["g1a"], "tok_per_s_spec": tot["tok"] / tot["spec"],
            "speedup_vs_g1_async": tot["g1a"] / tot["spec"], "speedup_vs_g1_sync": tot["g1s"] / tot["spec"],
            "mean_tokens_per_step": tot["tok"] / tot["steps"] if tot["steps"] else None, "mean_acc": tot["acc"] / max(1, tot["steps"]), "results": results}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(summ, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summ.items() if k != "results"}))

if __name__ == "__main__":
    main()
