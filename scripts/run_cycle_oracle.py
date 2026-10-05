#!/usr/bin/env python3
"""Per-cycle oracle efficiency (GPU): model-based decision-quality audit.

During routed generation each cycle logs both the router's decision and the
model-based per-cycle oracle (same estimates, bare margin-1.0 argmax, no
panic rule). Reports:
  agreement rate P(router == oracle),
  model-based efficiency = mean_cyc TPS_pred(router choice) / TPS_pred(oracle),
  confusion counts, realized TPS sanity.

Run: python3 scripts/run_cycle_oracle.py
Output: experiments/16_hybrid_roofline/cycle_oracle.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from zassd.cli import SKIP_STRATEGIES
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.routing.hybrid_router import HybridDraftRouter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--output", default="experiments/16_hybrid_roofline/cycle_oracle.json")
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

    agree, ratios, conf, tps_all, ncyc = 0, [], {}, [], 0
    for it in recs:
        router = HybridDraftRouter(cost_model=cm, pld_k=4)
        _, m = self_speculative_generate(
            model, tok, mgr, skip, it["prompt"], k=2, draft_mode="routed",
            router=router, config_name="cka_75",
            max_new_tokens=a.max_new_tokens, device=a.device)
        tps_all.append(m.tokens_per_second)
        for st in m.per_iteration_stats:
            rs, osrc, ot = st.get("router_source"), st.get("oracle_source"), st.get("oracle_tps") or {}
            if not rs or not osrc or not ot:
                continue
            ncyc += 1
            agree += (rs == osrc)
            conf[(rs, osrc)] = conf.get((rs, osrc), 0) + 1
            denom = max(1e-9, ot.get(osrc, 0.0))
            num = ot.get(rs if rs in ot else "vanilla", 0.0)
            ratios.append(num / denom)
    import numpy as np
    out = {
        "cycles": ncyc,
        "agreement_rate": round(agree / max(1, ncyc), 4),
        "model_efficiency_mean": round(float(np.mean(ratios)), 4),
        "model_efficiency_std": round(float(np.std(ratios, ddof=1)), 4),
        "confusion_router_vs_oracle": {f"{k[0]}->{k[1]}": v for k, v in sorted(conf.items())},
        "realized_mean_tps": round(float(np.mean(tps_all)), 2),
        "note": "model-based: same predicted TPS for both; gap = robustness-heuristic cost only, not clairvoyance",
    }
    json.dump(out, open(Path(a.output), "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"DONE -> {a.output}")


if __name__ == "__main__":
    main()
