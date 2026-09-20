"""Context-Scaling Benchmark for Self-Speculative vs Vanilla Decoding.

Evaluates how Zero-Additional-VRAM self-speculative decoding scales across
varying context lengths L in {128, 512, 1024, 2048} under 6GB VRAM budget.

Validates:
1. Zero Additional VRAM: VRAM(Self-Spec) == VRAM(Vanilla) + O(K * KV_heads)
2. Throughput scaling: Advantage of single-pass batched target verification as context grows
3. Cache latency invariance: Canonical Target KV + Ephemeral Draft KV fork overhead remains O(1)
4. Greedy exactness consistency across context lengths

Usage:
    python scripts/benchmark_speculative_context_scaling.py --context-lengths 128 512 1024 2048 --runs 5 --max-new-tokens 32
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.speculative import self_speculative_generate, SpeculativeMetrics
from zassd.decoding.vanilla import vanilla_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.memory import get_vram_usage, reset_vram_stats
from zassd.utils.logging import setup_logging
from zassd.utils.seed import set_seed

logger = logging.getLogger(__name__)


def prepare_context_prompts(
    tokenizer,
    context_length: int,
    num_prompts: int = 10,
    prompts_file: str = "data/benchmarks/prompts.jsonl",
    device: str = "cuda:0",
) -> list[dict[str, Any]]:
    """Prepare benchmark inputs padded/truncated to exact context length."""
    prompts = []
    p_file = Path(prompts_file)
    if p_file.exists():
        with open(p_file) as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line.strip()))
    else:
        prompts = [
            {"id": i, "category": "general", "prompt": f"Explain key concept number {i} in deep learning architectures."}
            for i in range(num_prompts)
        ]

    # Deterministic filler text
    filler = (
        "In artificial intelligence and deep learning, transformer models rely on multi-head self-attention "
        "mechanisms to model dependencies across variable sequence lengths. The key-value cache stores past "
        "projections to avoid quadratic recomputation during autoregressive generation. Layer-skipping architectures "
        "exploit internal representation similarity between adjacent blocks to construct speculative draft models. "
    ) * 100

    filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
    prepared = []

    for idx, p in enumerate(prompts[:num_prompts]):
        base_tokens = tokenizer.encode(p["prompt"], add_special_tokens=False)

        if len(base_tokens) >= context_length:
            input_tokens = base_tokens[:context_length]
        else:
            needed = context_length - len(base_tokens)
            offset = (idx * 150) % max(1, len(filler_tokens) - needed)
            prefix = filler_tokens[offset : offset + needed]
            input_tokens = prefix + base_tokens

        tensor_ids = torch.tensor([input_tokens], device=device)
        prepared.append({
            "id": p["id"],
            "prompt_text": p["prompt"],
            "input_ids": tensor_ids,
            "length": tensor_ids.shape[1],
        })

    return prepared


def plot_context_scaling(summary: dict[str, Any], figures_dir: Path) -> None:
    """Generate plots showing speedup, throughput, and VRAM scaling vs context length."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    ctx_lengths = [int(k.replace("ctx_", "")) for k in summary.keys()]
    ctx_lengths.sort()

    vanilla_tps = [summary[f"ctx_{c}"]["vanilla_tps"] for c in ctx_lengths]
    spec_tps = [summary[f"ctx_{c}"]["spec_tps"] for c in ctx_lengths]
    speedups = [summary[f"ctx_{c}"]["speedup"] for c in ctx_lengths]
    accept_rates = [summary[f"ctx_{c}"]["acceptance_rate"] * 100 for c in ctx_lengths]
    vanilla_vram = [summary[f"ctx_{c}"]["vanilla_vram_mb"] for c in ctx_lengths]
    spec_vram = [summary[f"ctx_{c}"]["spec_vram_mb"] for c in ctx_lengths]
    cache_latencies_ms = [summary[f"ctx_{c}"]["cache_latency_ms"] for c in ctx_lengths]

    # Figure 1: Throughput and Speedup vs Context Length
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=300)
    ax1.plot(ctx_lengths, vanilla_tps, marker="s", color="#7f7f7f", linestyle="--", linewidth=2, label="Vanilla Baseline")
    ax1.plot(ctx_lengths, spec_tps, marker="o", color="#1f77b4", linewidth=2.5, label="Self-Speculative (CKA-83%)")
    ax1.set_xlabel("Context Length (Tokens)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Throughput (tok/s)", fontsize=11, fontweight="bold", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(ctx_lengths, speedups, marker="^", color="#2ca02c", linewidth=2, linestyle="-.", label="Speedup vs Vanilla")
    ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.8)
    ax2.set_ylabel("Speedup (x)", fontsize=11, fontweight="bold", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(0.7, 1.4)

    plt.title("Scaling Behavior: Throughput & Speedup vs. Context Length", fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.savefig(figures_dir / "context_scaling_speedup.png")
    plt.close()

    # Figure 2: Peak VRAM vs Context Length
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(ctx_lengths, vanilla_vram, marker="s", color="#d62728", linestyle="--", linewidth=2, label="Vanilla Peak VRAM")
    plt.plot(ctx_lengths, spec_vram, marker="o", color="#1f77b4", linewidth=2, label="Self-Speculative Peak VRAM")
    plt.axhline(y=5500, color="black", linestyle=":", label="6GB Hardware Safety Ceiling (5500 MB)")
    plt.xlabel("Context Length (Tokens)", fontsize=10, fontweight="bold")
    plt.ylabel("Peak VRAM Allocated (MB)", fontsize=10, fontweight="bold")
    plt.title("Zero Additional VRAM Verification: Memory Footprint vs. Context Length", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(figures_dir / "context_scaling_vram.png")
    plt.close()

    # Figure 3: Ephemeral Cache Fork Overhead vs Context Length
    plt.figure(figsize=(7, 4.5), dpi=300)
    plt.plot(ctx_lengths, cache_latencies_ms, marker="D", color="#9467bd", linewidth=2)
    plt.xlabel("Context Length (Tokens)", fontsize=10, fontweight="bold")
    plt.ylabel("Cache Overhead per Cycle (ms)", fontsize=10, fontweight="bold")
    plt.title("Ephemeral Draft KV Fork Latency vs. Context Length (O(1) Invariant)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(figures_dir / "context_scaling_cache_overhead.png")
    plt.close()

    logger.info(f"Saved context scaling figures to {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Scaling Self-Speculative Benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[128, 512, 1024, 2048])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="experiments/08_context_scaling")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_file=str(out_dir / "context_scaling.log"))
    set_seed(42)

    logger.info("=" * 80)
    logger.info("PHASE 9 — CONTEXT SCALING BENCHMARK: SELF-SPECULATIVE VS VANILLA")
    logger.info("=" * 80)

    # 1. Load Model
    model = load_model(args.model, quantize=True, bits=args.bits)
    tokenizer = load_tokenizer(args.model)
    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)

    # Pareto-optimal CKA-83% skip layers: 6 skipped out of 36
    pareto_file = Path("experiments/07_pareto/pareto_results.json")
    if pareto_file.exists():
        with open(pareto_file) as f:
            p_data = json.load(f)
        skip_indices = p_data.get("cka_83", {}).get("skip_indices", [4, 5, 6, 7, 12, 13])
    else:
        skip_indices = [4, 5, 6, 7, 12, 13]

    logger.info(f"Using Pareto CKA-83% skip layers: {skip_indices} ({len(skip_indices)} skipped)")

    scaling_summary: dict[str, Any] = {}

    for ctx_len in args.context_lengths:
        logger.info(f"\n{'='*70}")
        logger.info(f"BENCHMARKING CONTEXT LENGTH L = {ctx_len}")
        logger.info(f"{'='*70}")

        prompts = prepare_context_prompts(
            tokenizer=tokenizer,
            context_length=ctx_len,
            num_prompts=args.runs,
        )

        # A. Vanilla baseline runs
        logger.info(f"Running Vanilla Baseline (L={ctx_len}, {len(prompts)} runs)...")
        vanilla_tps_list = []
        vanilla_vram_list = []
        vanilla_texts = []

        for p in prompts:
            # Decode using vanilla
            v_input_ids = p["input_ids"]
            reset_vram_stats("cuda:0")
            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = v_input_ids.clone()
                past_kv = None
                for _ in range(args.max_new_tokens):
                    if past_kv is None:
                        out = model(out_ids, use_cache=True)
                    else:
                        out = model(out_ids[:, -1:], past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    out_ids = torch.cat([out_ids, nxt], dim=-1)
                    if nxt.item() == tokenizer.eos_token_id:
                        break
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            new_tokens = out_ids.shape[1] - v_input_ids.shape[1]
            tps = new_tokens / elapsed if elapsed > 0 else 0.0
            vanilla_tps_list.append(tps)
            vanilla_vram_list.append(torch.cuda.max_memory_allocated("cuda:0") / (1024**2))
            vanilla_texts.append(tokenizer.decode(out_ids[0, v_input_ids.shape[1]:], skip_special_tokens=True))

        mean_vanilla_tps = float(np.mean(vanilla_tps_list))
        mean_vanilla_vram = float(np.mean(vanilla_vram_list))
        logger.info(f"  Vanilla L={ctx_len}: {mean_vanilla_tps:.2f} tok/s, VRAM: {mean_vanilla_vram:.0f} MB")

        # B. Self-Speculative runs
        logger.info(f"Running Self-Speculative K={args.k} (L={ctx_len}, {len(prompts)} runs)...")
        spec_tps_list = []
        spec_vram_list = []
        spec_accept_list = []
        spec_tokens_per_step = []
        spec_cache_ms_list = []
        matches = 0

        for run_idx, p in enumerate(prompts):
            text, metrics = self_speculative_generate(
                model=model,
                tokenizer=tokenizer,
                layer_mgr=layer_mgr,
                skip_indices=skip_indices,
                prompt=p["input_ids"],
                k=args.k,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )
            spec_tps_list.append(metrics.tokens_per_second)
            spec_vram_list.append(metrics.peak_vram_mb)
            spec_accept_list.append(metrics.acceptance_rate)
            spec_tokens_per_step.append(metrics.tokens_per_step)
            spec_cache_ms_list.append((metrics.cache_time_s / max(1, metrics.num_verification_cycles)) * 1000)

            if text == vanilla_texts[run_idx]:
                matches += 1

        mean_spec_tps = float(np.mean(spec_tps_list))
        mean_spec_vram = float(np.mean(spec_vram_list))
        mean_spec_acc = float(np.mean(spec_accept_list))
        mean_spec_step = float(np.mean(spec_tokens_per_step))
        mean_cache_ms = float(np.mean(spec_cache_ms_list))
        exact_rate = matches / len(prompts)
        speedup = mean_spec_tps / mean_vanilla_tps if mean_vanilla_tps > 0 else 1.0

        logger.info(
            f"  Self-Spec L={ctx_len}: {mean_spec_tps:.2f} tok/s ({speedup:.2f}x), "
            f"Accept: {mean_spec_acc:.1%}, Match: {exact_rate:.1%}, "
            f"VRAM: {mean_spec_vram:.0f} MB, Cache: {mean_cache_ms:.2f} ms/cycle"
        )

        scaling_summary[f"ctx_{ctx_len}"] = {
            "context_length": ctx_len,
            "vanilla_tps": mean_vanilla_tps,
            "vanilla_vram_mb": mean_vanilla_vram,
            "spec_tps": mean_spec_tps,
            "spec_vram_mb": mean_spec_vram,
            "speedup": speedup,
            "acceptance_rate": mean_spec_acc,
            "tokens_per_step": mean_spec_step,
            "exact_match_rate": exact_rate,
            "cache_latency_ms": mean_cache_ms,
        }

    # Save summary
    with open(out_dir / "scaling_summary.json", "w") as f:
        json.dump(scaling_summary, f, indent=2)

    # Plot results
    plot_context_scaling(scaling_summary, fig_dir)

    logger.info("\n" + "=" * 95)
    logger.info("CONTEXT SCALING BENCHMARK SUMMARY (RTX 4050 Laptop 6GB, Qwen2.5-3B NF4)")
    logger.info("=" * 95)
    logger.info(f"{'Context':>8} | {'Vanilla':>10} | {'Self-Spec':>10} | {'Speedup':>8} | {'Accept':>8} | {'Match':>8} | {'Vanilla MB':>10} | {'Spec MB':>10} | {'Cache ms':>9}")
    logger.info("-" * 95)
    for k, v in scaling_summary.items():
        logger.info(
            f"{v['context_length']:>8} | {v['vanilla_tps']:>8.2f} t/s | {v['spec_tps']:>8.2f} t/s | "
            f"{v['speedup']:>7.2f}x | {v['acceptance_rate']:>7.1%} | {v['exact_match_rate']:>7.1%} | "
            f"{v['vanilla_vram_mb']:>9.0f}MB | {v['spec_vram_mb']:>9.0f}MB | {v['cache_latency_ms']:>7.2f}ms"
        )
    logger.info("=" * 95)


if __name__ == "__main__":
    main()
