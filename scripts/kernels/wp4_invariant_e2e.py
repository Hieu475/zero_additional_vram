#!/usr/bin/env python3
"""WP4: end-to-end check of the M-invariant kernels. Reference = sequential greedy under invariant_mode; candidate = naive
PLD (no row-wise fallback) under invariant_mode. Also reports agreement of the invariant reference with the stock HF
reference (information: the two references are different by definition)."""
import argparse, json, time
from pathlib import Path
import torch
from zassd.backends.base import load_backend
from zassd.kernels.invariant_mode import invariant_mode
from audit_pld_trajectory_divergence import reference, naive_pld
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION

ap = argparse.ArgumentParser()
ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl"); ap.add_argument("--start", type=int, default=0)
ap.add_argument("--n-prompts", type=int, default=20); ap.add_argument("--max-new", type=int, default=128)
ap.add_argument("--k", type=int, default=2); ap.add_argument("--ngram", type=int, default=3)
ap.add_argument("--backend", default="hf_bnb_nf4"); ap.add_argument("--model", default=MODEL); ap.add_argument("--revision", default=REVISION)
ap.add_argument("--ops", default="lm_head,rmsnorm,sdpa"); ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
rows = [json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip()][a.start:a.start + a.n_prompts]
loaded = load_backend(a.backend, a.model, a.revision, "cuda:0", "sdpa"); model, tok = loaded.model.eval(), loaded.tokenizer
ops = tuple(a.ops.split(","))
per = []
with torch.inference_mode():
    for r in rows:
        ids = tok(r["prompt"], return_tensors="pt")["input_ids"].to("cuda:0"); cap = int(ids.shape[-1]) + a.max_new + a.k + 8
        stock = reference(model, ids, a.max_new, cap)
        with invariant_mode(model, ops):
            inv_ref = reference(model, ids, a.max_new, cap)
            got = naive_pld(model, ids, a.max_new, a.k, a.ngram, cap, "cuda:0")[0]
        first = next((i for i, (x, y) in enumerate(zip(inv_ref, got)) if x != y), None)
        first_stock = next((i for i, (x, y) in enumerate(zip(stock, inv_ref)) if x != y), None)
        per.append({"id": str(r["id"]), "diverged": first is not None, "first_divergence": first, "inv_ref_differs_from_stock": first_stock is not None, "first_diff_vs_stock": first_stock})
        print(json.dumps(per[-1]), flush=True)
d = sum(p["diverged"] for p in per); s = sum(p["inv_ref_differs_from_stock"] for p in per)
out = {"schema": "zassd-invariant-e2e-v1", "model": a.model, "backend": a.backend, "ops": ops, "k": a.k, "max_new": a.max_new, "prompts": len(per),
       "diverged_vs_invariant_reference": d, "invariant_reference_differs_from_stock": s, "per_prompt": per}
a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2) + "\n")
print(f"{a.model} k={a.k}: naive PLD under invariant kernels diverged {d}/{len(per)}; invariant reference differs from stock HF reference on {s}/{len(per)}")
