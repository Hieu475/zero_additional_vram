#!/usr/bin/env python3
"""Router rule ablation (GPU): full router vs 4 single-rule removals.

Configs (all else identical, Qwen2.5-3B NF4, GSM8K N=50, k=2, PLD k<=4):
  full, no-gate (enable_vanilla_skip=False), no-pref (margin 1.25->1.0),
  no-weakcap, no-horizon.
Reports mean speedup + action mix (%vanilla / %pld / %layer-skip cycles).

Run: python3 scripts/run_router_ablation.py
Output: experiments/16_hybrid_roofline/router_ablation.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from scripts.run_standard_benchmark_suite import run_vanilla
from zassd.cli import SKIP_STRATEGIES
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.routing.hybrid_router import HybridDraftRouter

CONFIGS = {
    "full": {},
    "no-gate": {"enable_vanilla_skip": False},
    "no-pref": {"prefer_pld_margin": 1.0},
    "no-weakcap": {"enable_weak_hit_cap": False},
    "no-horizon": {"enable_horizon_tuning": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--output", default="experiments/16_hybrid_roofline/router_ablation.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb, device_map=a.device).eval()
    mgr = LayerManager(ModelAdapter(model))
    skip = SKIP_STRATEGIES["empirical_6"]
    cm = MeasuredActionCostModel.for_model("qwen25_3b")

    recs = [json.loads(l) for l in open("data/benchmarks/gsm8k_eval.jsonl") if l.strip()][:50]
    v_tps = {}
    for it in recs:
        _, tps, _ = run_vanilla(model, tok, it["prompt"], max_new_tokens=a.max_new_tokens, device=a.device)
        v_tps[it["id"]] = tps
    print(f"vanilla baseline done: mean={sum(v_tps.values())/len(v_tps):.1f} tok/s")

    out = {}
    for name, kw in CONFIGS.items():
        router = HybridDraftRouter(cost_model=cm, pld_k=4, **kw)
        sps, n_v = [], 0
        for it in recs:
            _, m = self_speculative_generate(
                model, tok, mgr, skip, it["prompt"], k=2, draft_mode="routed",
                router=router, config_name="cka_75",
                max_new_tokens=a.max_new_tokens, device=a.device)
            sps.append(m.tokens_per_second / v_tps[it["id"]])
        st = router.stats.to_dict()
        tot = max(1, st["total_cycles"])
        out[name] = {
            "mean_speedup": round(sum(sps) / len(sps), 4),
            "pld_share": round(st["pld_cycles"] / tot, 3),
            "ls_share": round(st["layer_skip_cycles"] / tot, 3),
            "vanilla_share": round(st["vanilla_fallbacks"] / tot, 3),
            "alpha_pld_ema": round(st["alpha_pld_ema"], 3),
            "alpha_ls_ema": round(st["alpha_ls_ema"], 3),
        }
        print(f"[{name:10s}] speedup={out[name]['mean_speedup']:.3f} "
              f"pld={out[name]['pld_share']:.2f} ls={out[name]['ls_share']:.2f} "
              f"van={out[name]['vanilla_share']:.2f}")
    json.dump(out, open(Path(a.output), "w"), indent=2)
    print(f"DONE -> {a.output}")


if __name__ == "__main__":
    main()
