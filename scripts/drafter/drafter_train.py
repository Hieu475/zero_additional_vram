#!/usr/bin/env python3
import argparse
import json
import time
import glob
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from zassd.kernels.drafter import Drafter

ap = argparse.ArgumentParser()
ap.add_argument("--data", nargs="+", required=True)
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--vocab", type=int, default=16384)
ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--bs", type=int, default=2048)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--holdout", type=int, default=150, help="sequences held out from the end of the first file")
ap.add_argument("--snapshot", default=None)
ap.add_argument("--model-id", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--ff", type=int, default=4096)
ap.add_argument("--merged", type=Path, default=Path("results/drafter/_merged_h.npy"))
ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
dev = "cuda:0"
K = a.k

snap = a.snapshot or glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--" + a.model_id.replace("/", "--") + "/snapshots/*"))[0]
emb = None
for f in sorted(glob.glob(snap + "/*.safetensors")):
    with safe_open(f, "pt") as sf:
        if "model.embed_tokens.weight" in sf.keys(): 
            emb = sf.get_tensor("model.embed_tokens.weight").half()
emb = emb.to(dev)
D = emb.shape[1]   # tied lm_head
seqs, sizes, tl = [], [], []
for p_ in a.data:
    z = np.load(p_)
    o = z["off"]
    sizes.append(int(o[-1]))
    tl.append(torch.from_numpy(z["t"].astype(np.int64)))   # only t/off are read; h stays on disk
    base = sum(sizes[:-1])
    seqs += [(base + int(o[i]), base + int(o[i + 1])) for i in range(len(o) - 1)]
merged = a.merged
total = sum(sizes)
if not merged.exists() or np.load(merged, mmap_mode="r").shape != (total, D):
    mm = np.lib.format.open_memmap(merged, mode="w+", dtype=np.float16, shape=(total, D))
    base = 0
    for p_, sz in zip(a.data, sizes):
        mm[base:base + sz] = np.load(p_)["h"]
        base += sz
        mm.flush()
    del mm
Tall = torch.cat(tl)
del tl
Hn = np.load(merged, mmap_mode="r")   # read sequentially in bounded chunks below, never gathered randomly
S = seqs
ho = [x for i, x in enumerate(S) if i % 40 == 0]
tr = [x for i, x in enumerate(S) if i % 40 != 0]   # mixed-domain holdout (~2.5%)
CHUNK = 150_000   # positions resident in RAM per chunk (~600 MB fp16)
def chunks(segs, shuffle):
    """Yield (H_chunk fp16 tensor, T_chunk, valid-start indices) for groups of whole sequences."""
    order = np.random.permutation(len(segs)) if shuffle else np.arange(len(segs))
    cur, n = [], 0
    def build(cur):
        Hc = torch.from_numpy(np.concatenate([np.asarray(Hn[s:e]) for s, e in cur]))
        Tc = torch.cat([Tall[s:e] for s, e in cur])
        st, base = [], 0
        for s, e in cur: 
            st.extend(range(base, base + (e - s) - 1 - K))
            base += e - s
        return Hc, Tc, torch.tensor(st, dtype=torch.long)
    for i in order:
        cur.append(segs[i])
        n += segs[i][1] - segs[i][0]
        if n >= CHUNK: 
            yield build(cur)
            cur, n = [], 0
    if cur: 
        yield build(cur)
# trimmed vocab: most frequent tokens in training outputs (+ pad with low ids if needed)
cnt = torch.bincount(Tall, minlength=emb.shape[0])
vocab_ids = cnt.topk(a.vocab).indices.sort().values
remap = torch.full((emb.shape[0],), -1, dtype=torch.long)
remap[vocab_ids] = torch.arange(len(vocab_ids))
print("train seqs", len(tr), "holdout", len(ho), "positions", total, "vocab coverage", float((cnt[vocab_ids].sum() / cnt.sum())), flush=True)
model = Drafter(D, a.ff, vocab_ids.to(dev), emb[vocab_ids.to(dev)].float()).to(dev)
model.head.data = model.head.data.half().float()
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
remap = remap.to(dev)
def run(Hc, Tc, idx):
    h = Hc[idx].to(dev).float()
    toks = torch.stack([Tc[idx + 1 + j] for j in range(K + 1)], 1).to(dev)   # toks[:,0]=pend, toks[:,k]=x_{k}
    s = h
    loss = 0
    correct = []
    ok_chain = torch.ones(len(idx), dtype=torch.bool, device=dev)
    for k in range(1, K + 1):
        e = emb[toks[:, k - 1]].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = model.step(s, e)
            lg = model.logits(s)
        y = remap[toks[:, k]]
        valid = y >= 0
        l = F.cross_entropy(lg.float()[valid], y[valid]) if valid.any() else 0 * lg.sum()
        loss = loss + (1.0 if k == 1 else 0.8 ** (k - 1)) * l
        hit = (lg.argmax(-1) == y) & valid
        ok_chain = ok_chain & hit
        correct.append((hit.float().mean().item(), ok_chain.float().mean().item()))
    return loss, correct
import resource
t0 = time.time()
for ep in range(a.epochs):
    for g in opt.param_groups:
        g["lr"] = a.lr * (0.5 * (1 + np.cos(np.pi * ep / a.epochs)))
    for Hc, Tc, st in chunks(tr, True):
        perm = st[torch.randperm(len(st))]
        for b in range(0, len(perm), a.bs):
            loss, _ = run(Hc, Tc, perm[b:b + a.bs])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        hl = [run(Hc, Tc, st[b:b + 4096])[1] for Hc, Tc, st in chunks(ho, False) for b in range(0, len(st), 4096)]
    per = [np.mean([x[k][0] for x in hl]) for k in range(K)]
    chain = [np.mean([x[k][1] for x in hl]) for k in range(K)]
    print(f"ep{ep} loss {loss.item():.3f} holdout top1 per-pos {np.round(per,3)} chain-accepted P(>=k) {np.round(chain,3)} {time.time()-t0:.0f}s peakRSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f}GB", flush=True)
    torch.save({"state": model.state_dict(), "K": K, "d": D, "ff": a.ff}, a.output)
