"""Empirical evaluation of the ZASSD sweet-spot action surface.

Systematically measures candidate actions:
  - Vanilla Autoregressive
  - cka_90 (K=1, K=2)
  - cka_83 (K=1, K=2)
  - cka_75 (K=1, K=2)
To identify the empirical argmax TPS on NVIDIA RTX 4050 Laptop GPU.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from zassd.decoding.speculative import self_speculative_generate
from zassd.decoding.vanilla import vanilla_generate
from zassd.layer_selection.selector import LayerSelector
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.gpu import GPUProfiler
from zassd.profiling.memory import reset_vram_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_token_metrics(ref_text: str, cand_text: str, tokenizer) -> tuple[float, float]:
    """Compute Exact Match and Partial Match ratio."""
    ref_tokens = tokenizer.encode(ref_text, add_special_tokens=False)
    cand_tokens = tokenizer.encode(cand_text, add_special_tokens=False)

    if not ref_tokens:
        return (1.0 if not cand_tokens else 0.0), (1.0 if not cand_tokens else 0.0)

    min_len = min(len(ref_tokens), len(cand_tokens))
    if min_len == 0:
        return 0.0, 0.0

    match_count = sum(1 for i in range(min_len) if ref_tokens[i] == cand_tokens[i])
    exact_match = 1.0 if ref_tokens == cand_tokens else 0.0
    partial_match = match_count / float(len(ref_tokens))
    return exact_match, partial_match


def run_sweet_spot_search(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    num_prompts: int = 5,
    max_new_tokens: int = 32,
    device: str = "cuda:0",
    out_dir: str = "experiments/13_sweet_spot",
) -> dict[str, Any]:
    """Execute sweet spot exploration across candidate (S, K) configurations."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading {model_name} for sweet spot exploration...")
    model = load_model(model_name, quantize=True, bits=4, device=device)
    tokenizer = load_tokenizer(model_name)
    model.eval()

    adapter = ModelAdapter(model)
    selector = LayerSelector(adapter)
    profiler = GPUProfiler()

    # Load prompts
    prompts_path = Path("data/benchmarks/prompts.jsonl")
    all_prompts = []
    with open(prompts_path) as f:
        for line in f:
            if line.strip():
                all_prompts.append(json.loads(line.strip()))
    eval_prompts = all_prompts[:num_prompts]

    # Candidate layer skip configs
    configs = {
        "cka_90": [4, 5, 6, 7],
        "cka_83": [4, 5, 6, 7, 12, 13],
        "cka_75": [3, 4, 5, 6, 7, 10, 11, 12, 13],
    }

    # 1. Warmup
    logger.info("Warmup pass...")
    _ = vanilla_generate(model, tokenizer, eval_prompts[0]["prompt"], max_new_tokens=8, device=device)
    torch.cuda.synchronize()

    # 2. Run Vanilla baseline
    logger.info("Evaluating Vanilla baseline...")
    vanilla_runs = []
    vanilla_texts = {}

    reset_vram_stats()
    t_v0 = time.perf_counter()
    e_v0 = profiler.get_total_energy_mj()

    for p in eval_prompts:
        txt, m = vanilla_generate(model, tokenizer, p["prompt"], max_new_tokens=max_new_tokens, temperature=0.0, device=device)
        vanilla_runs.append(m)
        vanilla_texts[p["id"]] = txt

    torch.cuda.synchronize()
    t_v_elapsed = time.perf_counter() - t_v0
    e_v_end = profiler.get_total_energy_mj()

    v_total_tokens = sum(m.total_tokens for m in vanilla_runs)
    vanilla_tps = v_total_tokens / max(1e-3, t_v_elapsed)
    v_vram = float(np.mean([m.peak_vram_mb for m in vanilla_runs]))
    if e_v0 is not None and e_v_end is not None and e_v_end >= e_v0:
        vanilla_energy = ((e_v_end - e_v0) / 1000.0) / max(1, v_total_tokens)
    else:
        vanilla_energy = (profiler.get_power_usage() * t_v_elapsed) / max(1, v_total_tokens)

    results_table: list[dict[str, Any]] = [
        {
            "config": "Vanilla",
            "k": 0,
            "tps": round(vanilla_tps, 2),
            "speedup": 1.0,
            "acceptance_pct": 100.0,
            "exact_match_pct": 100.0,
            "partial_match_pct": 100.0,
            "vram_mb": round(v_vram, 1),
            "energy_j_tok": round(vanilla_energy, 3),
            "draft_ms": 0.0,
            "verify_ms": round(float(np.mean([m.tpot_ms for m in vanilla_runs])), 2),
        }
    ]

    candidates = [
        ("cka_90", 1),
        ("cka_90", 2),
        ("cka_83", 1),
        ("cka_83", 2),
        ("cka_75", 1),
        ("cka_75", 2),
    ]

    layer_mgr = LayerManager(adapter)

    for cfg_name, k_val in candidates:
        logger.info(f"Evaluating Candidate Action: {cfg_name} (K={k_val})...")
        skip_idx = configs[cfg_name]
        runs = []
        exacts = []
        partials = []

        reset_vram_stats()
        t0 = time.perf_counter()
        e0 = profiler.get_total_energy_mj()

        for p in eval_prompts:
            txt, m = self_speculative_generate(
                model=model,
                tokenizer=tokenizer,
                layer_mgr=layer_mgr,
                prompt=p["prompt"],
                skip_indices=skip_idx,
                k=k_val,
                controller=None,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                device=device,
            )
            runs.append(m)
            em, pm = compute_token_metrics(vanilla_texts[p["id"]], txt, tokenizer)
            exacts.append(em)
            partials.append(pm)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        e_end = profiler.get_total_energy_mj()

        total_tokens = sum(m.total_tokens for m in runs)
        cand_tps = total_tokens / max(1e-3, elapsed)
        cand_vram = float(np.mean([m.peak_vram_mb for m in runs]))
        if e0 is not None and e_end is not None and e_end >= e0:
            cand_energy = ((e_end - e0) / 1000.0) / max(1, total_tokens)
        else:
            cand_energy = (profiler.get_power_usage() * elapsed) / max(1, total_tokens)

        acc_pct = float(np.mean([m.acceptance_rate * 100.0 for m in runs]))
        speedup = cand_tps / vanilla_tps

        mean_cycles = max(1, sum(m.num_verification_cycles for m in runs))
        mean_draft_ms = (sum(m.draft_time_s for m in runs) * 1000.0) / mean_cycles
        mean_verify_ms = (sum(m.verify_time_s for m in runs) * 1000.0) / mean_cycles

        item = {
            "config": cfg_name,
            "k": k_val,
            "tps": round(cand_tps, 2),
            "speedup": round(speedup, 3),
            "acceptance_pct": round(acc_pct, 1),
            "exact_match_pct": round(float(np.mean(exacts)) * 100.0, 1),
            "partial_match_pct": round(float(np.mean(partials)) * 100.0, 1),
            "vram_mb": round(cand_vram, 1),
            "energy_j_tok": round(cand_energy, 3),
            "draft_ms": round(mean_draft_ms, 2),
            "verify_ms": round(mean_verify_ms, 2),
        }
        results_table.append(item)

    out_file = Path(out_dir) / "sweet_spot_summary.json"
    with open(out_file, "w") as f:
        json.dump(
            {
                "model_name": model_name,
                "vanilla_baseline_tps": vanilla_tps,
                "action_surface_results": results_table,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 92)
    print("ZASSD SWEET-SPOT ACTION SURFACE EXPLORATION (QWEN2.5-3B, RTX 4050 6GB)")
    print("=" * 92)
    print(
        f"{'Configuration':<16} | {'K':<2} | {'TPS':<7} | {'Speedup':<8} | {'Acc (%)':<8} | "
        f"{'Exact (%)':<10} | {'VRAM (MB)':<10} | {'Energy (J/t)':<12}"
    )
    print("-" * 92)
    for r in results_table:
        print(
            f"{r['config']:<16} | {r['k']:<2} | {r['tps']:>6.2f} | {r['speedup']:>6.3f}x | "
            f"{r['acceptance_pct']:>6.1f}% | {r['exact_match_pct']:>8.1f}% | {r['vram_mb']:>9.1f} | "
            f"{r['energy_j_tok']:>10.3f}"
        )
    print("=" * 92)

    return {"summary": results_table}


if __name__ == "__main__":
    run_sweet_spot_search()
