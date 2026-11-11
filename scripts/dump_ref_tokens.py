#!/usr/bin/env python3
"""Dump the exact M=1 greedy reference sequences (invariant mode) so draft strategies can be simulated offline."""
import argparse, json, copy
from pathlib import Path
import torch
from transformers import StaticCache
from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
import zassd.kernels.invariant_mode as IM
from zassd.kernels.invariant_mode import invariant_mode

ap = argparse.ArgumentParser()
ap.add_argument("--benchmark", required=True)
ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--max-cache-len", type=int, default=1024)
ap.add_argument("--model", default=MODEL)
ap.add_argument("--revision", default=REVISION)
ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip()]
L = load_backend("hf_bnb_nf4", a.model, a.revision, "cuda:0", "sdpa")
model, tok = L.model.eval(), L.tokenizer
cfg = copy.deepcopy(model.config)
if getattr(cfg, "sliding_window", None) is not None:
    cfg.sliding_window = None
dev = "cuda:0"
pre = torch.zeros(1, dtype=torch.int32, device=dev)
IM.PREFIX_TENSOR = pre
out = []
with torch.inference_mode(), invariant_mode(model, ("lm_head", "rmsnorm", "sdpa", "nf4")):
    for r in rows:
        ids = tok(r["prompt"], return_tensors="pt")["input_ids"].to(dev)
        plen = ids.shape[1]
        c = StaticCache(config=cfg, max_cache_len=a.max_cache_len)
        lg = model(ids, past_key_values=c, use_cache=True, cache_position=torch.arange(plen, device=dev)).logits[:, -1:]
        ref = []
        for i in range(a.max_new):
            t = lg.argmax(-1)
            ref.append(int(t.item()))
            pre.fill_(plen + i)
            lg = model(t, past_key_values=c, use_cache=True, cache_position=torch.tensor([plen + i], device=dev)).logits[:, -1:]
        out.append({"id": r["id"], "prompt": ids[0].tolist(), "ref": ref})
        print(r["id"], flush=True)
a.output.write_text(json.dumps(out))
