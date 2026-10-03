"""Cross-GPU calibration (run on each target GPU: 4090/A100/H100 or 3060).

Measures 5 calibration points (CPU + CUDA events, ~3 min, no full benchmark needed):
  1. vanilla TPS (Qwen2.5-3B NF4, batch-1, 64 tokens)
  2. T_draft cka_75 K=1, K=2  |  3. T_verify K=1..4
Fit: rho (verify parallel efficiency) + verify roofline MAPE on the new GPU.
Output: experiments/16_hybrid_roofline/calib_<gpu>.json

Run: python3 scripts/run_cross_gpu_calibration.py --model Qwen/Qwen2.5-3B-Instruct
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, "src")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--outdir", default="experiments/16_hybrid_roofline")
a = ap.parse_args()

t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(a.model)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb,
                                             device_map=a.device).eval()

from zassd.cache.kv_cache import TargetKVCache
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter

prompt = "Explain speculative decoding in large language models and why it accelerates inference."
ids = tok(prompt, return_tensors="pt").to(a.device)["input_ids"]

def ms(fn, iters=20):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

# 1. vanilla per-token
cache = TargetKVCache(backend="static")
with torch.no_grad():
    out = model(ids, past_key_values=cache.cache, use_cache=True)
cur = out.logits[:, -1:, :].argmax(dim=-1)
t_vanilla = ms(lambda: model(cur, past_key_values=cache.cache, use_cache=True))

# 2/3. draft (cka_75 ~ skip 9 middle layers) + verify by K
mgr = LayerManager(ModelAdapter(model))
nL = ModelAdapter(model).num_layers
skip = list(range(nL // 2 - 4, nL // 2 + 5))[:9]
res = {"t_vanilla_ms": round(t_vanilla, 2), "skip": skip}
for k in [1, 2, 4]:
    dk = TargetKVCache(backend="static")
    with torch.no_grad():
        o = model(ids, past_key_values=dk.cache, use_cache=True)
    c = o.logits[:, -1:, :].argmax(dim=-1)
    dd = dk.fork_ephemeral_draft_kv() if hasattr(dk, "fork_ephemeral_draft_kv") else dk.cache
    def draft_fn(_c=[c], _dd=[dd], _k=k):
        x = _c[0]
        with torch.no_grad(), mgr.skip_layers(skip):
            for _ in range(_k):
                oo = model(x, past_key_values=_dd[0], use_cache=True)
                x = oo.logits[:, -1:, :].argmax(dim=-1)
    res[f"t_draft_k{k}_ms"] = round(ms(draft_fn, 10), 2)
    vk = TargetKVCache(backend="static")
    with torch.no_grad():
        o2 = model(ids, past_key_values=vk.cache, use_cache=True)
    multi = torch.cat([o2.logits[:, -1:, :].argmax(dim=-1)] * (k + 1), dim=-1)
    res[f"t_verify_k{k}_ms"] = round(ms(lambda: model(multi, past_key_values=vk.cache, use_cache=True), 10), 2)

props = torch.cuda.get_device_properties(a.device)
res.update({"gpu": props.name, "bw_note": "fill the measured bandwidth into roofline KNOWN_GPUS"})
outp = Path(a.outdir) / f"calib_{props.name.replace(' ', '_')}.json"
outp.parent.mkdir(parents=True, exist_ok=True)
json.dump(res, open(outp, "w"), indent=2)
print(json.dumps(res, indent=2))
print(f"Saved -> {outp}  (total {time.perf_counter()-t0:.0f}s)")
