#!/usr/bin/env python3
"""Tune an NF4 PLD margin guard on dev, then evaluate once on held-out prompts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_pld_adaptive_study import BENCHMARKS, read_split
from run_standard_benchmark_suite import run_vanilla
from zassd.baselines.prompt_lookup import prompt_lookup_generate
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter


def bootstrap_ci(values: list[float], seed: int = 20261011) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if not len(arr):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(4000, len(arr)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def run_pld(model, tok, prompt: str, k: int, limit: int, device: str) -> dict:
    _, m = prompt_lookup_generate(model, tok, prompt, k=k,
                                  max_new_tokens=limit, device=device)
    return {"tps": m.tokens_per_second, "ids": m.generated_token_ids,
            "accepted": m.total_accepted_tokens, "tokens": m.total_tokens}


def run_guard(model, tok, manager, skip_indices, prompt: str, k: int,
              limit: int, device: str, threshold: float) -> dict:
    _, m = self_speculative_generate(
        model, tok, manager, skip_indices, prompt, k=k,
        draft_mode="prompt_lookup", max_new_tokens=limit, device=device,
        margin_fallback_threshold=threshold,
    )
    return {"tps": m.tokens_per_second, "ids": m.generated_token_ids,
            "accepted": m.total_accepted_tokens, "tokens": m.total_tokens,
            "fallback_cycles": m.margin_fallback_cycles,
            "replay_time_s": m.canonical_replay_time_s}


def summarize(rows: list[dict]) -> dict:
    exact = [bool(r["exact"]) for r in rows]
    ratios = [r["speedup_vs_vanilla"] for r in rows]
    accepted = [r["accepted_per_output"] for r in rows]
    return {
        "n": len(rows),
        "exact_match_pct": float(np.mean(exact) * 100) if rows else 0.0,
        "speedup_vs_vanilla_mean": float(np.mean(ratios)) if rows else 0.0,
        "speedup_vs_vanilla_ci95": bootstrap_ci(ratios),
        "fraction_slower_than_vanilla": float(np.mean(np.asarray(ratios) < 1.0)) if rows else 0.0,
        "accepted_tokens_per_output": float(np.mean(accepted)) if rows else 0.0,
        "margin_fallback_cycles_mean": float(np.mean([r["fallback_cycles"] for r in rows])) if rows else 0.0,
        "replay_time_s_mean": float(np.mean([r["replay_time_s"] for r in rows])) if rows else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--samples-per-split", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.5, 1.0, 1.5])
    ap.add_argument("--benchmarks", nargs="+", choices=sorted(BENCHMARKS), default=sorted(BENCHMARKS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default="experiments/diagnostics/nf4_margin_guard_study.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA driver is required")

    tok = AutoTokenizer.from_pretrained(args.model)
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
    nskip = min(6, adapter.num_layers)
    start = (adapter.num_layers - nskip) // 2
    skip_indices = list(range(start, start + nskip))

    dev_rows: dict[float, list[dict]] = {t: [] for t in args.thresholds}
    test_items: list[tuple[str, dict]] = []
    for bi, bench in enumerate(args.benchmarks):
        dev, test = read_split(BENCHMARKS[bench], args.samples_per_split, 9341 + bi)
        for item in dev:
            prompt = item["prompt"]
            _, vanilla_tps, vanilla_ids = run_vanilla(
                model, tok, prompt, args.max_new_tokens, args.device)
            for threshold in args.thresholds:
                result = run_guard(model, tok, manager, skip_indices, prompt, args.k,
                                   args.max_new_tokens, args.device, threshold)
                dev_rows[threshold].append({
                    "benchmark": bench, "id": item.get("id"),
                    "exact": result["ids"] == vanilla_ids,
                    "speedup_vs_vanilla": result["tps"] / max(vanilla_tps, 1e-9),
                    "fallback_cycles": result["fallback_cycles"],
                    "replay_time_s": result["replay_time_s"],
                    "accepted_per_output": result["accepted"] / max(result["tokens"], 1),
                })
            print(f"{bench} dev {len([r for r in dev_rows[args.thresholds[0]] if r['benchmark'] == bench])}/{len(dev)}", flush=True)
        test_items.extend((bench, item) for item in test)

    dev_summary = {str(t): summarize(rows) for t, rows in dev_rows.items()}
    exact_thresholds = [t for t, rows in dev_rows.items() if rows and all(r["exact"] for r in rows)]
    if exact_thresholds:
        selected = max(exact_thresholds, key=lambda t: dev_summary[str(t)]["speedup_vs_vanilla_mean"])
        selection_rule = "fastest threshold with 100% dev exact match"
    else:
        selected = max(args.thresholds, key=lambda t: (
            dev_summary[str(t)]["exact_match_pct"],
            dev_summary[str(t)]["speedup_vs_vanilla_mean"],
        ))
        selection_rule = "highest dev exact-match rate, then speed; no threshold reached 100%"

    test_rows: list[dict] = []
    for bench, item in test_items:
        prompt = item["prompt"]
        _, vanilla_tps, vanilla_ids = run_vanilla(
            model, tok, prompt, args.max_new_tokens, args.device)
        pld = run_pld(model, tok, prompt, args.k, args.max_new_tokens, args.device)
        guard = run_guard(model, tok, manager, skip_indices, prompt, args.k,
                          args.max_new_tokens, args.device, selected)
        test_rows.append({
            "benchmark": bench, "id": item.get("id"),
            "vanilla_tps": vanilla_tps,
            "pld_tps": pld["tps"], "pld_exact": pld["ids"] == vanilla_ids,
            "pld_speedup_vs_vanilla": pld["tps"] / max(vanilla_tps, 1e-9),
            "guard_tps": guard["tps"], "guard_exact": guard["ids"] == vanilla_ids,
            "guard_speedup_vs_vanilla": guard["tps"] / max(vanilla_tps, 1e-9),
            "guard_vs_pld": guard["tps"] / max(pld["tps"], 1e-9),
            "guard_fallback_cycles": guard["fallback_cycles"],
            "guard_replay_time_s": guard["replay_time_s"],
            "guard_accepted_per_output": guard["accepted"] / max(guard["tokens"], 1),
        })
        print(f"{bench} test {sum(r['benchmark'] == bench for r in test_rows)}/{args.samples_per_split}", flush=True)

    paired_ratios = [r["guard_vs_pld"] for r in test_rows]
    report = {
        "model": args.model, "max_new_tokens": args.max_new_tokens, "k": args.k,
        "samples_per_split_per_benchmark": args.samples_per_split,
        "threshold_candidates": args.thresholds,
        "dev_split_seed": "9341 + benchmark index; shuffled once, first N is dev, next N is test",
        "dev_threshold_summary": dev_summary,
        "selected_threshold": selected, "selection_rule": selection_rule,
        "heldout_test": {
            "n": len(test_rows),
            "pld_exact_match_pct": float(np.mean([r["pld_exact"] for r in test_rows]) * 100),
            "guard_exact_match_pct": float(np.mean([r["guard_exact"] for r in test_rows]) * 100),
            "pld_speedup_vs_vanilla_mean": float(np.mean([r["pld_speedup_vs_vanilla"] for r in test_rows])),
            "guard_speedup_vs_vanilla_mean": float(np.mean([r["guard_speedup_vs_vanilla"] for r in test_rows])),
            "guard_vs_pld_mean_ratio": float(np.mean(paired_ratios)),
            "guard_vs_pld_ci95": bootstrap_ci(paired_ratios),
            "guard_faster_than_pld_pct": float(np.mean(np.asarray(paired_ratios) > 1) * 100),
            "by_benchmark": {
                b: {
                    "n": len([r for r in test_rows if r["benchmark"] == b]),
                    "pld_exact_match_pct": float(np.mean([r["pld_exact"] for r in test_rows if r["benchmark"] == b]) * 100),
                    "guard_exact_match_pct": float(np.mean([r["guard_exact"] for r in test_rows if r["benchmark"] == b]) * 100),
                    "guard_vs_pld_mean_ratio": float(np.mean([r["guard_vs_pld"] for r in test_rows if r["benchmark"] == b])),
                } for b in args.benchmarks
            },
            "rows": test_rows,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_threshold": selected,
                      "dev": dev_summary[str(selected)],
                      "test": {k: v for k, v in report["heldout_test"].items() if k != "rows"}}, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
