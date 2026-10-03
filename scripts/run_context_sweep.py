#!/usr/bin/env python3
"""Context-length sweep: Vanilla / PLD / LayerSkip / Routed vs context length.

Same repetitive document truncated to 128/512/1K/2K/4K/8K tokens, Qwen2.5-3B
NF4, greedy, 32 new tokens x 3 reps. Tests the roofline prediction that the
feasibility boundary moves with context (attention cost grows, draft/verify
gap shifts).

Run: python3 scripts/run_context_sweep.py
Output: experiments/16_hybrid_roofline/context_sweep.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from scripts.run_standard_benchmark_suite import run_vanilla as _run_vanilla_unused  # noqa (kept for provenance)
from zassd.cache.kv_cache import TargetKVCache
from zassd.baselines.prompt_lookup import prompt_lookup_generate
from zassd.cli import SKIP_STRATEGIES
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.routing.hybrid_router import HybridDraftRouter

def run_vanilla(model, tok, ctx, max_new_tokens=32, device="cuda:0"):
    """Vanilla AR with dynamic backend (static caps at 2048; dynamic used
    uniformly here for cross-length comparability)."""
    from zassd.cache.kv_cache import TargetKVCache
    inputs = tok(ctx, return_tensors="pt").to(device)
    cache = TargetKVCache(backend="dynamic")
    emitted = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(inputs.input_ids, past_key_values=cache.cache, use_cache=True)
        curr = out.logits[:, -1:, :].argmax(dim=-1)
        emitted.append(int(curr.item()))
        for _ in range(max_new_tokens - 1):
            if emitted[-1] == tok.eos_token_id:
                break
            out = model(curr, past_key_values=cache.cache, use_cache=True)
            curr = out.logits[:, -1:, :].argmax(dim=-1)
            emitted.append(int(curr.item()))
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return None, len(emitted) / dt, emitted


PARA = ("Speculative decoding accelerates autoregressive generation by drafting "
        "candidate tokens with a cheap draft model and verifying them in parallel "
        "with the target model. On memory-bound hardware the draft pass streams "
        "most of the weights, so the feasibility boundary depends on bandwidth. ")


def build_context(tok, target_tokens: int) -> str:
    ids = []
    while len(ids) < target_tokens:
        ids += tok(PARA, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:target_tokens], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--output", default="experiments/16_hybrid_roofline/context_sweep.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb, device_map=a.device).eval()
    mgr = LayerManager(ModelAdapter(model))
    skip = SKIP_STRATEGIES["empirical_6"]
    cm = MeasuredActionCostModel.for_model("qwen25_3b")

    out = {}
    if Path(a.output).exists():
        try:
            out = json.load(open(a.output))  # merge across runs
        except Exception:
            out = {}
    for L in a.lengths:
        ctx = build_context(tok, L)
        row = {}
        vt = []
        for _ in range(a.reps):
            _, tps, _ = run_vanilla(model, tok, ctx, max_new_tokens=a.max_new_tokens, device=a.device)
            vt.append(tps)
        row["vanilla"] = sum(vt) / len(vt)
        pt = []
        for _ in range(a.reps):
            _, m = prompt_lookup_generate(model, tok, ctx, k=2, max_new_tokens=a.max_new_tokens,
                                          device=a.device, kv_cache_backend="dynamic")
            pt.append(m.tokens_per_second)
        row["pld"] = sum(pt) / len(pt)
        lt = []
        for _ in range(a.reps):
            _, m = self_speculative_generate(model, tok, mgr, skip, ctx, k=1,
                                             max_new_tokens=a.max_new_tokens, device=a.device,
                                             kv_cache_backend="dynamic")
            lt.append(m.tokens_per_second)
        row["layerskip_k1"] = sum(lt) / len(lt)
        rt = []
        for _ in range(a.reps):
            router = HybridDraftRouter(cost_model=cm, pld_k=4)
            _, m = self_speculative_generate(model, tok, mgr, skip, ctx, k=2,
                                             draft_mode="routed", router=router,
                                             config_name="cka_75",
                                             max_new_tokens=a.max_new_tokens, device=a.device,
                                             kv_cache_backend="dynamic")
            rt.append(m.tokens_per_second)
        row["routed"] = sum(rt) / len(rt)
        v = row["vanilla"]
        out[str(L)] = {k: round(x, 2) for k, x in row.items()}
        out[str(L)].update({k + "_sp": round(row[k] / v, 3) for k in ("pld", "layerskip_k1", "routed")})
        print(f"L={L:5d} vanilla={v:6.1f} pld={row['pld']/v:.3f}x ls={row['layerskip_k1']/v:.3f}x routed={row['routed']/v:.3f}x")
    json.dump(out, open(Path(a.output), "w"), indent=2)
    print(f"DONE -> {a.output}")


if __name__ == "__main__":
    main()
