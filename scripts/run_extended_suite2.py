#!/usr/bin/env python3
"""Extended suite part 2: HumanEval test[50:164] (full-164 when merged) +
CNN/DM test[50:200] (N200 when merged). Non-overlapping with N50 splits.

Saves incrementally per bench so a long run is never lost.
Same frozen protocol (Qwen2.5-3B NF4, greedy, 48 new tokens).

Run: python3 scripts/run_extended_suite2.py [--model ...]
Output: experiments/16_hybrid_roofline/standard_suite_ext2.json
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

BENCHES = [
    ("humaneval_ext114", Path("data/benchmarks/humaneval_ext114.jsonl"), 114),
    ("cnndm_ext150", Path("data/benchmarks/cnndm_ext150.jsonl"), 150),
    ("cnndm_ext75a", Path("data/benchmarks/cnndm_ext75a.jsonl"), 75),
    ("cnndm_ext75b", Path("data/benchmarks/cnndm_ext75b.jsonl"), 75),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--output", default="experiments/16_hybrid_roofline/standard_suite_ext2.json")
    ap.add_argument("--only", default="", help="comma list of bench names to run (default: all pending)")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb, device_map=a.device).eval()
    adapter = ModelAdapter(model)
    mgr = LayerManager(adapter)
    if "empirical_6" in SKIP_STRATEGIES and adapter.num_layers == 36:
        skip = SKIP_STRATEGIES["empirical_6"]
    else:
        ms = (adapter.num_layers - 6) // 2
        skip = list(range(ms, ms + 6))

    outp = Path(a.output)
    acc = {"model": a.model,
           "note": "non-overlapping extensions of N50 splits",
           "benchmarks": {}}
    if outp.exists():
        acc = json.load(open(outp))  # resume: skip finished benches
        print(f"Resuming, already have: {list(acc.get('benchmarks', {}))}")
    only = {s.strip() for s in a.only.split(",") if s.strip()}
    for name, path, n in BENCHES:
        if only and name not in only:
            continue
        if name in acc["benchmarks"]:
            print(f"SKIP {name} (done)")
            continue
        r = evaluate_benchmark_file(
            bench_name=name, file_path=path, num_samples=n, model=model,
            tokenizer=tok, layer_mgr=mgr, skip_indices=skip,
            max_new_tokens=a.max_new_tokens, device=a.device)
        acc["benchmarks"][name] = r
        outp.parent.mkdir(parents=True, exist_ok=True)
        json.dump(acc, open(outp, "w"), indent=1)
        for s, v in r["summary"].items():
            print(f"[{name}] {s:28s} speedup={v['mean_speedup']:.3f}")
    print(f"DONE -> {outp}")


if __name__ == "__main__":
    main()
