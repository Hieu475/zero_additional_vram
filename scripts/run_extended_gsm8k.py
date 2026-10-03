#!/usr/bin/env python3
"""Extended GSM8K run: non-overlapping test[50:200] (+150 samples, all 5 strategies).

Merges with the N50 results (test[:50]) for a combined GSM8K N=200 in
analysis. Same frozen protocol (Qwen2.5-3B NF4, greedy, 48 new tokens).

Run: python3 scripts/run_extended_gsm8k.py [--model ...]
Output: experiments/16_hybrid_roofline/standard_suite_gsm8k_ext150.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from scripts.run_standard_benchmark_suite import evaluate_benchmark_file
from zassd.cli import SKIP_STRATEGIES
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--output", default="experiments/16_hybrid_roofline/standard_suite_gsm8k_ext150.json")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    preq = "bnb-4bit" in a.model.lower() or "4bit" in a.model.lower()
    if preq:
        model = AutoModelForCausalLM.from_pretrained(a.model, device_map=a.device)
    else:
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            a.model, quantization_config=bnb, device_map=a.device)
    model.eval()
    adapter = ModelAdapter(model)
    mgr = LayerManager(adapter)
    if "empirical_6" in SKIP_STRATEGIES and adapter.num_layers == 36:
        skip = SKIP_STRATEGIES["empirical_6"]
    else:
        ms = (adapter.num_layers - 6) // 2
        skip = list(range(ms, ms + 6))

    out = evaluate_benchmark_file(
        bench_name="gsm8k_ext150", file_path=Path("data/benchmarks/gsm8k_ext150.jsonl"),
        num_samples=150, model=model, tokenizer=tok, layer_mgr=mgr,
        skip_indices=skip, max_new_tokens=a.max_new_tokens, device=a.device)
    outp = Path(a.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"model": a.model, "slice": "gsm8k test[50:200] (non-overlapping with N50 test[:50])",
               **out}, open(outp, "w"), indent=2)
    for s, v in out["summary"].items():
        print(f"{s:28s} speedup={v['mean_speedup']:.3f} acc={v['mean_acceptance']:.3f} em={v['exact_match_pct']:.1f}")
    print(f"DONE -> {outp}")


if __name__ == "__main__":
    main()
