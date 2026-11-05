#!/usr/bin/env python3
"""WP6: how different is the M-invariant reference from the stock HF+bnb greedy reference, and does it matter for accuracy?

For each prompt: greedy decode (EOS or --max-new tokens) with the stock path and with invariant_mode. Reports sequence identity,
first-divergence position, final-answer agreement (GSM8K), and task accuracy of both (GSM8K exact-match on prompts with gold)."""
import argparse, json, re
from pathlib import Path
import torch
from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.kernels.invariant_mode import invariant_mode
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION

ap = argparse.ArgumentParser()
ap.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl"); ap.add_argument("--start", type=int, default=0)
ap.add_argument("--n-prompts", type=int, default=50); ap.add_argument("--max-new", type=int, default=320)
ap.add_argument("--model", default=MODEL); ap.add_argument("--revision", default=REVISION)
ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
rows = [json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip()][a.start:a.start + a.n_prompts]
loaded = load_backend("hf_bnb_nf4", a.model, a.revision, "cuda:0", "sdpa"); model, tok = loaded.model.eval(), loaded.tokenizer
eos = {tok.eos_token_id} | set(getattr(model.generation_config, "eos_token_id", None) or []) if not isinstance(getattr(model.generation_config, "eos_token_id", None), int) else {tok.eos_token_id, model.generation_config.eos_token_id}

ANSWER_RE = re.compile(r"[Tt]he (?:final )?answer is[^\n]*\n")


def gen(ids, n):
    cache = TargetKVCache(backend="static", max_capacity=int(ids.shape[-1]) + n + 8)
    lg = model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
    out = []
    for _ in range(n):
        t = lg.argmax(dim=-1); v = int(t.item()); out.append(v)
        if v in eos: break
        if len(out) % 8 == 0 and ANSWER_RE.search(tok.decode(out, skip_special_tokens=True)): break
        lg = model(t, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
    return out

def final_number(text):
    m = ANSWER_RE.search(text)
    if not m: return None
    nums = re.findall(r"-?\d[\d,]*\.?\d*", m.group(0))
    return nums[-1].replace(",", "").rstrip(".") if nums else None

def gold_number(r):
    g = r.get("gold_answer")
    if g is None: return None
    return g.split("####")[-1].strip().replace(",", "")

per = []
with torch.inference_mode():
    for r in rows:
        ids = tok(r["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
        s = gen(ids, a.max_new)
        with invariant_mode(model): v = gen(ids, a.max_new)
        first = next((i for i, (x, y) in enumerate(zip(s, v)) if x != y), None)
        if first is None and len(s) != len(v): first = min(len(s), len(v))
        ts, tv = tok.decode(s, skip_special_tokens=True), tok.decode(v, skip_special_tokens=True)
        g = gold_number(r)
        rec = {"id": str(r["id"]), "identical": first is None, "first_divergence": first, "len_stock": len(s), "len_inv": len(v),
               "ans_stock": final_number(ts), "ans_inv": final_number(tv), "gold": g, "text_stock": ts[-1800:], "text_inv": tv[-1800:]}
        per.append(rec); print(json.dumps(rec), flush=True)
n = len(per); ident = sum(p["identical"] for p in per)
same_ans = sum(p["ans_stock"] == p["ans_inv"] for p in per)
withgold = [p for p in per if p["gold"] is not None]
acc = lambda k: sum(p[k] is not None and p[k] == p["gold"] for p in withgold) / max(1, len(withgold))
out = {"schema": "zassd-reference-agreement-v1", "model": a.model, "benchmark": a.benchmark, "max_new": a.max_new, "prompts": n,
       "identical_sequences": ident, "identical_fraction": ident / n, "same_final_answer": same_ans, "same_final_answer_fraction": same_ans / n,
       "with_gold": len(withgold), "accuracy_stock": acc("ans_stock"), "accuracy_invariant": acc("ans_inv"), "per_prompt": per}
a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2) + "\n")
print({k: v for k, v in out.items() if k != "per_prompt"})
