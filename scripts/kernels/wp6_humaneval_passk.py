#!/usr/bin/env python3
"""WP6b: HumanEval pass@1 (greedy) for the stock HF+bnb path vs invariant_mode, same model, same prompts, plus sequence identity.
Completions are cut at the first top-level statement after the function body; tests run in a subprocess with a timeout."""
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path
import torch
from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.kernels.invariant_mode import invariant_mode
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION

ap = argparse.ArgumentParser()
ap.add_argument("--benchmark", default="data/benchmarks/humaneval_wp5_100.jsonl"); ap.add_argument("--n-prompts", type=int, default=100)
ap.add_argument("--max-new", type=int, default=384); ap.add_argument("--model", default=MODEL); ap.add_argument("--revision", default=REVISION)
ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
rows = [json.loads(l) for l in Path(a.benchmark).read_text().splitlines() if l.strip()][:a.n_prompts]
loaded = load_backend("hf_bnb_nf4", a.model, a.revision, "cuda:0", "sdpa"); model, tok = loaded.model.eval(), loaded.tokenizer
eos = {tok.eos_token_id}
ge = getattr(model.generation_config, "eos_token_id", None)
eos |= set(ge) if isinstance(ge, list) else ({ge} if ge is not None else set())
STOP = re.compile(r"\n(?=[^\s#])")   # a new top-level line ends the function body


def cut(text):
    text = text.split("```")[0]
    m = STOP.search(text)
    return text[:m.start()] if m else text


def gen(ids, n):
    cache = TargetKVCache(backend="static", max_capacity=int(ids.shape[-1]) + n + 8)
    lg = model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
    out = []
    for _ in range(n):
        t = lg.argmax(dim=-1); v = int(t.item()); out.append(v)
        if v in eos: break
        if len(out) % 8 == 0 and STOP.search(tok.decode(out, skip_special_tokens=True)): break
        lg = model(t, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
    return out


def run_test(r, completion):
    prog = r["prompt"] + completion + "\n\n" + r["test"] + f"\n\ncheck({r['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(prog); path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", path], capture_output=True, timeout=20)
        return p.returncode == 0
    except subprocess.TimeoutExpired:
        return False


per = []
with torch.inference_mode():
    for r in rows:
        ids = tok(r["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
        s = gen(ids, a.max_new)
        with invariant_mode(model): v = gen(ids, a.max_new)
        ts, tv = cut(tok.decode(s, skip_special_tokens=True)), cut(tok.decode(v, skip_special_tokens=True))
        rec = {"id": r["id"], "identical_tokens": s == v, "identical_completion": ts == tv, "pass_stock": run_test(r, ts), "pass_inv": run_test(r, tv),
               "completion_stock": ts[:1500], "completion_inv": tv[:1500]}
        per.append(rec); print({k: rec[k] for k in ("id", "identical_tokens", "pass_stock", "pass_inv")}, flush=True)
n = len(per)
ps, pi = sum(p["pass_stock"] for p in per), sum(p["pass_inv"] for p in per)
b = sum(p["pass_stock"] and not p["pass_inv"] for p in per); c = sum((not p["pass_stock"]) and p["pass_inv"] for p in per)
out = {"schema": "zassd-humaneval-passk-v1", "model": a.model, "max_new": a.max_new, "prompts": n, "pass_stock": ps, "pass_inv": pi,
       "discordant_stock_only": b, "discordant_inv_only": c, "identical_tokens": sum(p["identical_tokens"] for p in per),
       "identical_completion": sum(p["identical_completion"] for p in per), "per_prompt": per}
a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2) + "\n")
print({k: v for k, v in out.items() if k != "per_prompt"})
