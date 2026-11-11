#!/usr/bin/env python3
"""Predict exact-spec-decoding speedup of the trained drafter on recorded reference sequences (no kernels run).
Hidden states: teacher-forced stock forward over prompt+ref (tiny numeric difference vs in-graph invariant kernels)."""
import argparse
import json 
import glob
import os
import sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from zassd.backends.base import load_backend
from zassd.kernels.drafter import Drafter
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION

sys.path.insert(0, str(Path(__file__).parent))

import sim_draft as SD

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--kmax", type=int, default=3)
ap.add_argument("--cache", type=Path, default=None)
ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
a = ap.parse_args()
dev = "cuda:0"
docs = json.load(open(a.ref))
ck = torch.load(a.ckpt, map_location=dev)
K = ck["K"]
vid = ck["state"]["vocab_ids"]
dr = Drafter(ck.get("d", 2048), ck.get("ff", 4096), vid, ck["state"]["head"]).to(dev) 
dr.load_state_dict(ck["state"])
dr.eval()
if a.cache and a.cache.exists():
    feats = torch.load(a.cache)
else:
    L = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model = L.model.eval()
    feats = []
    with torch.inference_mode():
        for d in docs:
            ids = torch.tensor([d["prompt"] + d["ref"]], device=dev)
            feats.append(model.model(input_ids=ids).last_hidden_state[0].half().cpu())
    emb = model.get_input_embeddings().weight.detach()
    if a.cache: 
        torch.save(feats, a.cache)
    del model, L
    torch.cuda.empty_cache()
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--" + a.model_id.replace("/", "--") + "/snapshots/*"))[0]
for f in sorted(glob.glob(snap + "/*.safetensors")):
    with safe_open(f, "pt") as sf:
        if "model.embed_tokens.weight" in sf.keys(): 
            emb = sf.get_tensor("model.embed_tokens.weight").half().to(dev)
chains = {}
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):   # bf16 like training and the in-graph drafter
    for j, d in enumerate(docs):
        P, R = d["prompt"], d["ref"]
        n = len(R)
        idx = torch.arange(len(P) - 1, len(P) + n - 1)           # hidden of the input token preceding pend=R[p]
        s = feats[j][idx].to(dev).float()
        x = torch.tensor(R, device=dev)   # pend = R[p]
        cur = x
        out = []
        for k in range(K):
            s = dr.step(s, emb[cur].float())
            lg = dr.logits(s)
            cur = vid[lg.argmax(-1)]
            out.append(cur.cpu())
        chains[j] = torch.stack(out, 1).tolist()                # chains[j][p] = drafted tokens after pend=R[p]
for d in docs: 
    pass
def head_drafter(k):
    def f(hist, d):
        p = len(hist) - len(d["prompt"]) - 1
        return [chains[d["_j"]][p][:k]]
    return f
for j, d in enumerate(docs):
    d["_j"] = j
res = {}
rows = [(f"drafter K={k}", head_drafter(k)) for k in range(1, K + 1)]
rows += [("PLD K=3 (ref)", SD.pld(3)), (f"drafter K={K} + PLD K=3 tree", SD.combine(head_drafter(K), SD.pld(3)))]

class Fallback:  # PLD only when its n-gram match is >=3 tokens long, else drafter
    pass
def prefer_pld(k):
    pl, hd = SD.pld(k, 3), head_drafter(K)
    def f(hist, d):
        c = pl(hist, d)
        return c if c else hd(hist, d)
    return f
rows.append((f"PLD-first else drafter K={K}", prefer_pld(3)))
for name, dr_ in rows:
    out = []
    for ov in (0.0, 1.0, 1.5, 2.0):
        r = SD.simulate(docs, dr_, overhead=ov if "drafter" in name else 0.0)
        out.append(r["speedup"])
    res[name] = r
    print(f"{name:32s} speedup(draft cost 0/1/1.5/2 ms) {out[0]:.3f} {out[1]:.3f} {out[2]:.3f} {out[3]:.3f}  tok/step {r['tok_per_block']:.2f}", flush=True)
