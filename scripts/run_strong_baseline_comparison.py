#!/usr/bin/env python3
"""Paired comparison of the strongest zero-auxiliary-weight decoding choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_standard_benchmark_suite import run_vanilla
from zassd.baselines.prompt_lookup import prompt_lookup_generate
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.routing.hybrid_router import HybridDraftRouter


BENCHMARKS = {
    "gsm8k": "data/benchmarks/gsm8k_eval.jsonl",
    "humaneval": "data/benchmarks/humaneval_eval.jsonl",
    "cnndm": "data/benchmarks/cnndm_eval.jsonl",
}
METHODS = ("pld", "pld_gate", "routed")


def bootstrap_mean_ci(values: list[float], seed: int = 2026) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(4000, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def summarize(rows: list[dict]) -> dict:
    report: dict = {}
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        speeds = [row["speedup_vs_vanilla"] for row in method_rows]
        report[method] = {
            "n": len(method_rows),
            "mean_speedup": float(np.mean(speeds)),
            "speedup_ci95": bootstrap_mean_ci(speeds),
            "exact_match_pct": float(np.mean([row["exact_token_match"] for row in method_rows]) * 100),
        }
    by_id = {(row["benchmark"], row["id"], row["method"]): row for row in rows}
    paired = []
    for row in rows:
        if row["method"] == "routed":
            pld = by_id[(row["benchmark"], row["id"], "pld")]
            paired.append(row["tps"] / pld["tps"])
    report["routed_vs_pld"] = {
        "n": len(paired),
        "mean_ratio": float(np.mean(paired)),
        "ratio_ci95": bootstrap_mean_ci(paired),
        "router_faster_prompt_pct": float(np.mean(np.asarray(paired) > 1.0) * 100),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--samples-per-bench", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="experiments/diagnostics/strong_baselines_n50.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA driver is required")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if "bnb-4bit" in args.model.lower() or "4bit" in args.model.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device)
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, device_map=args.device)
    model.eval()
    adapter = ModelAdapter(model)
    manager = LayerManager(adapter)
    skip_count = min(6, adapter.num_layers)
    skip_start = (adapter.num_layers - skip_count) // 2
    skip_indices = list(range(skip_start, skip_start + skip_count))
    model_key = "llama32_3b" if "llama" in args.model.lower() else "qwen25_3b"
    cost_model = MeasuredActionCostModel.for_model(model_key)

    rows: list[dict] = []
    for bench, filename in BENCHMARKS.items():
        with open(filename, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        for index, item in enumerate(records[:args.samples_per_bench]):
            prompt = item["prompt"]
            sample_id = item.get("id", f"{bench}_{index}")
            _, vanilla_tps, vanilla_ids = run_vanilla(
                model, tokenizer, prompt, args.max_new_tokens, args.device)
            # Rotate the non-reference methods to reduce ordering bias.
            order = METHODS[index % len(METHODS):] + METHODS[:index % len(METHODS)]
            for method in order:
                if method == "pld":
                    _, metrics = prompt_lookup_generate(
                        model, tokenizer, prompt, k=4,
                        max_new_tokens=args.max_new_tokens, device=args.device)
                else:
                    router = HybridDraftRouter(
                        cost_model=cost_model, pld_k=4,
                        enable_layer_skip=(method == "routed"))
                    _, metrics = self_speculative_generate(
                        model, tokenizer, manager, skip_indices, prompt,
                        k=2, draft_mode="routed", router=router,
                        max_new_tokens=args.max_new_tokens, device=args.device)
                rows.append({
                    "benchmark": bench, "id": sample_id, "method": method,
                    "tps": metrics.tokens_per_second,
                    "vanilla_tps": vanilla_tps,
                    "speedup_vs_vanilla": metrics.tokens_per_second / vanilla_tps,
                    "exact_token_match": metrics.generated_token_ids == vanilla_ids,
                    "output_tokens": len(metrics.generated_token_ids),
                    "pld_cycles": metrics.pld_cycles,
                    "layer_skip_cycles": metrics.layer_skip_cycles,
                })
            print(f"{bench}: {index + 1}/{min(len(records), args.samples_per_bench)}", flush=True)

    report = {"model": args.model, "max_new_tokens": args.max_new_tokens,
              "samples_per_bench": args.samples_per_bench,
              "methods": {"pld": "PLD k=4 with vanilla on lookup miss",
                          "pld_gate": "PLD k<=4 with cost-based vanilla choice; no layer skip",
                          "routed": "PLD k<=4, layer skip k=2, or vanilla"},
              "overall": summarize(rows),
              "by_benchmark": {bench: summarize([row for row in rows
                                                if row["benchmark"] == bench])
                               for bench in BENCHMARKS},
              "rows": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
