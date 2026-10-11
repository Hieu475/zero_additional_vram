#!/usr/bin/env python3
"""Tune PLD horizon on a development split, then compare adaptive policies.

Reports paired speed, exact token match, accepted tokens, lookup/verify/crop/
sync costs, and performance by source, context length, and lookup confidence.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_standard_benchmark_suite import run_vanilla
from zassd.baselines.prompt_lookup import prompt_lookup_generate

BENCHMARKS = {
    "gsm8k": "data/benchmarks/gsm8k_eval.jsonl",
    "humaneval": "data/benchmarks/humaneval_eval.jsonl",
    "cnndm": "data/benchmarks/cnndm_eval.jsonl",
}


def read_split(path: str, n: int, seed: int) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n], rows[n:2*n]


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    speed = np.asarray([x["speedup_vs_vanilla"] for x in rows])
    rng = np.random.default_rng(20261010)
    boot = rng.choice(speed, size=(4000, len(speed)), replace=True).mean(axis=1)
    cycles = [c for x in rows for c in x["cycles"]]
    prompt_buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        length = row.get("prompt_tokens", 0)
        bucket = "prompt_0_512" if length < 512 else "prompt_512_2k" if length < 2048 else "prompt_2k_plus"
        prompt_buckets[bucket].append(row)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for c in cycles:
        ctx = c.get("context_len", 0)
        grouped["context_0_512" if ctx < 512 else "context_512_2k" if ctx < 2048 else "context_2k_plus"].append(c)
        grouped[c.get("match_source", "unknown")].append(c)
        grouped["confidence_high" if c.get("match_confidence", 0) >= 0.7 else "confidence_low"].append(c)
    cost = {}
    for name, group in grouped.items():
        cost[name] = {
            "cycles": len(group),
            "lookup_ms_median": percentile([c.get("lookup_ms", 0) for c in group], 50),
            "verify_ms_median": percentile([c.get("verify_ms", 0) for c in group], 50),
            "crop_ms_median": percentile([c.get("crop_ms", 0) for c in group], 50),
            "decision_ms_median": percentile([c.get("decision_ms", 0) for c in group], 50),
            "sync_wait_ms_median": percentile([c.get("sync_wait_ms", 0) for c in group], 50),
            "accepted_per_cycle": float(np.mean([c.get("accepted", 0) for c in group])),
            "lookup_hit_rate": float(np.mean([c.get("k", 0) > 0 for c in group])),
        }
    return {
        "n": len(rows), "speedup_mean": float(speed.mean()),
        "speedup_ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
        "exact_token_match_pct": float(np.mean([x["exact"] for x in rows]) * 100),
        "accepted_tokens_per_output": float(np.mean([x["accepted_tokens"] / max(1, x["output_tokens"]) for x in rows])),
        "by_prompt_length": {
            name: {
                "n": len(group),
                "mean_speedup": float(np.mean([x["speedup_vs_vanilla"] for x in group])),
                "fraction_slower_than_vanilla": float(np.mean([x["speedup_vs_vanilla"] < 1.0 for x in group])),
            } for name, group in prompt_buckets.items()
        },
        "cost_breakdown": cost,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--samples-per-split", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-k", type=int, default=6)
    ap.add_argument("--output", default="experiments/diagnostics/pld_adaptive_study.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA driver is required")
    tok = AutoTokenizer.from_pretrained(args.model)
    if "bnb-4bit" in args.model.lower() or "4bit" in args.model.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device)
    else:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=q,
                                                    device_map=args.device)
    model.eval()
    results: dict[str, dict] = {}
    sources = ("history", "prompt", "prompt+generated")
    for bi, (bench, filename) in enumerate(BENCHMARKS.items()):
        dev, test = read_split(filename, args.samples_per_split, 9341 + bi)
        dev_runs: dict[str, dict[int, list[dict]]] = {s: defaultdict(list) for s in sources}
        verify_by_k: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in sources}
        feature_accepts: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in sources}
        vanilla_token_ms: list[float] = []
        for i, item in enumerate(dev):
            _, dev_vanilla_tps, _ = run_vanilla(model, tok, item["prompt"],
                                                 args.max_new_tokens, args.device)
            vanilla_token_ms.append(1000.0 / max(dev_vanilla_tps, 1e-6))
            k_order = list(range(1, args.max_k + 1))
            offset = i % len(k_order)
            k_order = k_order[offset:] + k_order[:offset]
            for source in sources:
                for k in k_order:
                    _, m = prompt_lookup_generate(model, tok, item["prompt"], k=k,
                        max_new_tokens=args.max_new_tokens, device=args.device,
                        candidate_source=source)
                    dev_runs[source][k].append({"tps": m.tokens_per_second,
                                                "cycles": m.per_iteration_stats})
                    for cycle in m.per_iteration_stats:
                        actual_k = cycle.get("k", 0)
                        if actual_k <= 0:
                            continue
                        verify_by_k[source][str(actual_k)].append(cycle["verify_ms"])
                        rep = min(cycle.get("match_repetitions", 0), 3)
                        order = cycle.get("match_order", 0)
                        accepted = float(cycle.get("accepted", 0))
                        feature_accepts[source][f"{order}:{rep}:{actual_k}"].append(accepted)
                        feature_accepts[source][f"{order}:*:{actual_k}"].append(accepted)
                        feature_accepts[source][f"*:*:{actual_k}"].append(accepted)
            print(f"{bench} dev {i+1}/{len(dev)}", flush=True)
        best_k = {s: max(dev_runs[s], key=lambda k: float(np.mean([r["tps"] for r in dev_runs[s][k]]))) for s in sources}
        adaptive_profile = {s: {
            "vanilla_token_ms": float(np.mean(vanilla_token_ms)),
            "verify_ms_by_k": {k: float(np.mean(v)) for k, v in verify_by_k[s].items()},
            "acceptance_by_feature_k": {k: float(np.mean(v)) for k, v in feature_accepts[s].items()},
        } for s in sources}
        eval_rows: dict[str, list[dict]] = defaultdict(list)
        for i, item in enumerate(test):
            prompt = item["prompt"]
            _, base_tps, base_ids = run_vanilla(model, tok, prompt, args.max_new_tokens, args.device)
            configs = [
                (f"history_fixed_k{best_k['history']}", best_k["history"], "fixed", "history"),
                ("history_adaptive", args.max_k, "confidence_cost", "history"),
                (f"prompt_fixed_k{best_k['prompt']}", best_k["prompt"], "fixed", "prompt"),
                (f"prompt_generated_fixed_k{best_k['prompt+generated']}", best_k["prompt+generated"], "fixed", "prompt+generated"),
                ("prompt_adaptive", args.max_k, "confidence_cost", "prompt"),
                ("prompt_generated_adaptive", args.max_k, "confidence_cost", "prompt+generated"),
            ]
            offset = i % len(configs)
            configs = configs[offset:] + configs[:offset]
            for name, k, policy, source in configs:
                _, m = prompt_lookup_generate(model, tok, prompt, k=k,
                    max_new_tokens=args.max_new_tokens, device=args.device,
                    candidate_source=source, k_policy=policy,
                    adaptive_profile=adaptive_profile[source] if policy == "confidence_cost" else None)
                eval_rows[name].append({
                    "id": item.get("id", f"{bench}_{i}"),
                    "speedup_vs_vanilla": m.tokens_per_second / base_tps,
                    "exact": m.generated_token_ids == base_ids,
                    "accepted_tokens": m.total_accepted_tokens,
                    "output_tokens": m.total_tokens,
                    "prompt_tokens": len(tok(prompt, add_special_tokens=True)["input_ids"]),
                    "cycles": m.per_iteration_stats,
                })
            print(f"{bench} test {i+1}/{len(test)}", flush=True)
        paired_to_fixed = {}
        fixed_name = f"history_fixed_k{best_k['history']}"
        fixed_map = {r["id"]: r["speedup_vs_vanilla"] for r in eval_rows[fixed_name]}
        rng = np.random.default_rng(2026 + bi)
        for method, method_rows in eval_rows.items():
            ratios = np.asarray([r["speedup_vs_vanilla"] / fixed_map[r["id"]]
                                 for r in method_rows])
            boot = rng.choice(ratios, size=(4000, len(ratios)), replace=True).mean(axis=1)
            paired_to_fixed[method] = {
                "mean_speed_ratio": float(ratios.mean()),
                "ci95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                "fraction_prompts_faster": float(np.mean(ratios > 1.0)),
            }
        results[bench] = {
            "tuned_fixed_k_by_source": best_k,
            "adaptive_profile_from_dev": adaptive_profile,
            "dev_mean_tokens_per_second_by_k": {s: {str(k): float(np.mean([r["tps"] for r in v])) for k, v in by_k.items()} for s, by_k in dev_runs.items()},
            "paired_to_tuned_fixed": paired_to_fixed,
            "test": {name: summarize(rows) for name, rows in eval_rows.items()},
            "rows": dict(eval_rows),
        }
    out = {"model": args.model, "samples_per_split": args.samples_per_split,
           "max_new_tokens": args.max_new_tokens, "max_k": args.max_k,
           "results": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({b: {"tuned_fixed_k_by_source": d["tuned_fixed_k_by_source"], "paired_to_tuned_fixed": d["paired_to_tuned_fixed"], "test": d["test"]} for b, d in results.items()}, indent=2))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
