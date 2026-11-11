#!/usr/bin/env python3
"""Self-distillation data for a feature-conditioned drafter: greedy outputs of the target model on TRAIN prompts + final-norm hidden states.
Numerics need not be bit-exact here (training data only)."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION

ap = argparse.ArgumentParser()
ap.add_argument("--source", choices=["gsm8k", "cnndm", "code"], required=True)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--max-new", type=int, default=160)
ap.add_argument("--model", default=MODEL)
ap.add_argument("--revision", default=REVISION)
ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()

import datasets
if a.source == "gsm8k":
    ds = datasets.load_dataset("openai/gsm8k", "main", split="train")
    prompts = [f"Question: {r['question']}\nLet's think step by step.\nAnswer:" for r in ds]
elif a.source == "cnndm":
    ds = datasets.load_dataset("abisee/cnn_dailymail", "3.0.0", split="train[:20000]")
    prompts = [f"Article:\n{r['article'][:2000]}\n\nSummarize the key points of the above article in 2-3 sentences:\nSummary:" for r in ds]
else:
    import re
    ds = datasets.load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    prompts = []
    for r in ds:
        m = re.search(r"^def .*\):?[^\n]*:\s*$|^def [^\n]*:\s*$", r["output"], re.M)
        if not m or r["input"].strip(): continue
        doc = " ".join(r["instruction"].split())
        pr = f'{m.group(0).strip()}\n    """ {doc}\n    """\n'
        if len(pr) < 900: 
            prompts.append(pr)
    print("code prompts", len(prompts))
random.Random(0).shuffle(prompts)
prompts = prompts[a.start:a.start + a.n]
L = load_backend("hf_bnb_nf4", a.model, a.revision, "cuda:0", "sdpa")
model, tok = L.model.eval(), L.tokenizer
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
eos = {tok.eos_token_id}
ge = model.generation_config.eos_token_id
eos |= set(ge) if isinstance(ge, list) else ({ge} if ge is not None else set())
H, T, off = [], [], [0]
with torch.inference_mode():
    for b in range(0, len(prompts), a.bs):
        enc = tok(prompts[b:b + a.bs], return_tensors="pt", padding=True).to("cuda:0")
        out = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id)
        am = torch.cat([enc["attention_mask"], torch.ones_like(out[:, enc["input_ids"].shape[1]:])], 1)
        # mask out padding after EOS
        gen = out[:, enc["input_ids"].shape[1]:]
        for i in range(gen.shape[0]):
            e = [j for j, t in enumerate(gen[i].tolist()) if t in eos]
            if e:
                am[i, enc["input_ids"].shape[1] + e[0] + 1:] = 0
        hs = model.model(input_ids=out, attention_mask=am).last_hidden_state
        for i in range(out.shape[0]):
            idx = am[i].nonzero().flatten()
            n0 = int(enc["attention_mask"][i].sum())
            seq = out[i, idx]
            h = hs[i, idx]
            H.append(h[n0 - 1:].half().cpu().numpy())
            T.append(seq[n0 - 1:].cpu().numpy().astype(np.int32))  # token at index j is the INPUT whose hidden is H[j]
            off.append(off[-1] + len(H[-1]))
        print(b, off[-1], flush=True)
np.savez(a.output, h=np.concatenate(H), t=np.concatenate(T), off=np.array(off))
