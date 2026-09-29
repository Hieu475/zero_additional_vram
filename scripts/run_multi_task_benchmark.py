#!/usr/bin/env python3
"""Multi-task benchmark across Coding, Reasoning, and General NLP categories.

Evaluates 4 Zero-Additional-Weight-VRAM strategies on consumer hardware:
1. Vanilla Target (Full Model)
2. Prompt Lookup Decoding (PLD, n-gram matching)
3. ZASSD Empirical Layer-Skip (k=1)
4. ZASSD Hybrid (PLD + Empirical Layer-Skip, k=2)

Saves results to experiments/final_validation/multi_task_evaluation.json.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from zassd.baselines.prompt_lookup import prompt_lookup_generate
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter
from zassd.cache.kv_cache import TargetKVCache
from zassd.cli import SKIP_STRATEGIES

console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi_task_benchmark")


def run_vanilla(model, tokenizer, prompt, max_new_tokens=48, device="cuda:0"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    cache = TargetKVCache(backend="static")
    emitted = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(inputs.input_ids, past_key_values=cache.cache, use_cache=True)
        curr = out.logits[:, -1:, :].argmax(dim=-1)
        emitted.append(int(curr.item()))
        for _ in range(max_new_tokens - 1):
            if emitted[-1] == tokenizer.eos_token_id:
                break
            out = model(curr, past_key_values=cache.cache, use_cache=True)
            curr = out.logits[:, -1:, :].argmax(dim=-1)
            emitted.append(int(curr.item()))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    duration = t1 - t0
    tps = len(emitted) / duration if duration > 0 else 0.0
    text = tokenizer.decode(emitted, skip_special_tokens=True)
    return text, tps, len(emitted)


def main():
    parser = argparse.ArgumentParser(description="Multi-Task Speculative Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="experiments/final_validation/multi_task_evaluation.json")
    args = parser.parse_args()

    console.print(f"[bold cyan]Starting Multi-Task Benchmark on {args.model}...[/bold cyan]")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    is_prequantized = "bnb-4bit" in args.model.lower() or "4bit" in args.model.lower()

    if is_prequantized:
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map=args.device,
        )
    model.eval()

    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)

    # Use empirical_6 if available, else mid_12
    if "empirical_6" in SKIP_STRATEGIES and adapter.num_layers == 36:
        skip_indices = SKIP_STRATEGIES["empirical_6"]
    else:
        mid_start = (adapter.num_layers - 6) // 2
        skip_indices = list(range(mid_start, mid_start + 6))

    # Load prompts from data/benchmarks/prompts.jsonl
    prompts_path = Path("data/benchmarks/prompts.jsonl")
    selected_prompts = []
    category_counts = {}
    with open(prompts_path) as f:
        for line in f:
            item = json.loads(line)
            cat = item.get("category", "general")
            if cat in ["general", "coding", "reasoning"]:
                category_counts[cat] = category_counts.get(cat, 0) + 1
                if category_counts[cat] <= 2:  # 2 prompts per category
                    selected_prompts.append(item)

    console.print(f"Selected {len(selected_prompts)} prompts across {len(category_counts)} categories.")

    methods = [
        "Vanilla",
        "Prompt-Lookup (PLD)",
        "ZASSD Empirical-6 (k=1)",
        "ZASSD Hybrid (k=2)",
    ]

    all_results = {m: [] for m in methods}

    for p_item in selected_prompts:
        p_id = p_item["id"]
        cat = p_item["category"]
        prompt = p_item["prompt"]
        console.print(f"\n[bold yellow]Prompt #{p_id} ({cat}):[/bold yellow] '{prompt[:60]}...'")

        # 1. Vanilla
        _, v_tps, v_toks = run_vanilla(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens, device=args.device)
        all_results["Vanilla"].append({"id": p_id, "cat": cat, "tps": v_tps, "acc": 0.0, "speedup": 1.0})
        console.print(f"  [cyan]Vanilla[/cyan]:           {v_tps:5.2f} tok/s | 1.000x")

        # 2. Prompt Lookup (PLD)
        _, m_pld = prompt_lookup_generate(model, tokenizer, prompt, k=2, max_new_tokens=args.max_new_tokens, device=args.device)
        pld_speedup = m_pld.tokens_per_second / v_tps
        all_results["Prompt-Lookup (PLD)"].append({"id": p_id, "cat": cat, "tps": m_pld.tokens_per_second, "acc": m_pld.acceptance_rate, "speedup": pld_speedup})
        console.print(f"  [cyan]Prompt-Lookup[/cyan]:     {m_pld.tokens_per_second:5.2f} tok/s | Acc: {m_pld.acceptance_rate*100:4.1f}% | {pld_speedup:.3f}x")

        # 3. ZASSD Empirical-6 (k=1)
        _, m_emp = self_speculative_generate(model, tokenizer, layer_mgr, skip_indices, prompt, k=1, max_new_tokens=args.max_new_tokens, device=args.device)
        emp_speedup = m_emp.tokens_per_second / v_tps
        all_results["ZASSD Empirical-6 (k=1)"].append({"id": p_id, "cat": cat, "tps": m_emp.tokens_per_second, "acc": m_emp.acceptance_rate, "speedup": emp_speedup})
        console.print(f"  [cyan]ZASSD Empirical[/cyan]:   {m_emp.tokens_per_second:5.2f} tok/s | Acc: {m_emp.acceptance_rate*100:4.1f}% | {emp_speedup:.3f}x")

        # 4. ZASSD Hybrid (k=2)
        _, m_hyb = self_speculative_generate(model, tokenizer, layer_mgr, skip_indices, prompt, k=2, draft_mode="hybrid", max_new_tokens=args.max_new_tokens, device=args.device)
        hyb_speedup = m_hyb.tokens_per_second / v_tps
        all_results["ZASSD Hybrid (k=2)"].append({"id": p_id, "cat": cat, "tps": m_hyb.tokens_per_second, "acc": m_hyb.acceptance_rate, "speedup": hyb_speedup, "pld_cycles": m_hyb.pld_cycles, "layer_cycles": m_hyb.layer_skip_cycles})
        console.print(f"  [cyan]ZASSD Hybrid[/cyan]:      {m_hyb.tokens_per_second:5.2f} tok/s | Acc: {m_hyb.acceptance_rate*100:4.1f}% | [bold green]{hyb_speedup:.3f}x[/bold green]")

    # Summary table
    table = Table(title=f"Multi-Task Zero-VRAM Speculative Evaluation ({args.model})")
    table.add_column("Decoding Strategy", style="bold")
    table.add_column("Mean TPS", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Acceptance Rate", justify="right")
    table.add_column("Coding Speedup", justify="right")
    table.add_column("Reasoning Speedup", justify="right")
    table.add_column("General Speedup", justify="right")

    summary_export = {}
    for m in methods:
        rows = all_results[m]
        mean_tps = float(np.mean([r["tps"] for r in rows]))
        mean_speedup = float(np.mean([r["speedup"] for r in rows]))
        mean_acc = float(np.mean([r["acc"] for r in rows]))

        cat_speedups = {}
        for c in ["coding", "reasoning", "general"]:
            c_rows = [r["speedup"] for r in rows if r["cat"] == c]
            cat_speedups[c] = float(np.mean(c_rows)) if c_rows else 1.0

        summary_export[m] = {
            "mean_tps": mean_tps,
            "mean_speedup": mean_speedup,
            "mean_acceptance_rate": mean_acc,
            "category_speedup": cat_speedups,
        }

        speedup_str = f"[bold green]{mean_speedup:.3f}x[/bold green]" if mean_speedup >= 1.0 else f"{mean_speedup:.3f}x"
        code_str = f"[bold green]{cat_speedups['coding']:.3f}x[/bold green]" if cat_speedups['coding'] >= 1.0 else f"{cat_speedups['coding']:.3f}x"

        table.add_row(
            m,
            f"{mean_tps:.2f} tok/s",
            speedup_str,
            f"{mean_acc*100:.1f}%" if m != "Vanilla" else "-",
            code_str,
            f"{cat_speedups['reasoning']:.3f}x",
            f"{cat_speedups['general']:.3f}x",
        )

    console.print("\n")
    console.print(table)

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump({"raw_results": all_results, "summary": summary_export}, f, indent=2)
    console.print(f"\n[bold green]Saved benchmark results to {out_p}[/bold green]")


if __name__ == "__main__":
    main()
