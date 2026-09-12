"""Benchmark vanilla autoregressive decoding.

This establishes the baseline performance numbers:
- tokens/s
- TTFT (time to first token)
- TPOT (time per output token)
- Peak VRAM
- GPU power/energy

Usage:
    python scripts/benchmark_vanilla.py
    python scripts/benchmark_vanilla.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from zassd.models.loader import load_model, load_tokenizer
from zassd.decoding.vanilla import vanilla_generate, GenerationMetrics
from zassd.profiling.memory import get_vram_usage, reset_vram_stats
from zassd.utils.config import load_config
from zassd.utils.seed import set_seed
from zassd.utils.logging import setup_logging

logger = logging.getLogger(__name__)

DEFAULT_PROMPTS = [
    "Explain the concept of speculative decoding in large language models.",
    "Write a Python function to compute the Fibonacci sequence.",
    "What are the advantages of quantization for model deployment?",
    "Describe the architecture of a transformer model.",
    "How does KV cache work in autoregressive generation?",
]


def run_benchmark(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    max_new_tokens: int = 128,
    warmup_runs: int = 10,
    measured_runs: int = 30,
    output_dir: str = "experiments/01_vanilla",
    quantize: bool = True,
    bits: int = 4,
) -> None:
    """Run vanilla decoding benchmark."""
    setup_logging(log_file=f"{output_dir}/benchmark.log")
    set_seed(42)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {model_name}")
    model = load_model(model_name, quantize=quantize, bits=bits)
    tokenizer = load_tokenizer(model_name)

    # Log initial VRAM
    vram_after_load = get_vram_usage()
    logger.info(f"VRAM after model load: {vram_after_load}")

    # Warmup
    logger.info(f"Running {warmup_runs} warmup iterations...")
    for i in range(warmup_runs):
        _ = vanilla_generate(
            model, tokenizer, DEFAULT_PROMPTS[0],
            max_new_tokens=32,
        )

    # Measured runs
    logger.info(f"Running {measured_runs} measured iterations...")
    all_metrics: list[dict] = []

    for run_idx in range(measured_runs):
        prompt = DEFAULT_PROMPTS[run_idx % len(DEFAULT_PROMPTS)]
        reset_vram_stats()

        output_text, metrics = vanilla_generate(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
        )

        run_result = {
            "run_idx": run_idx,
            "prompt": prompt,
            "output": output_text,
            "total_tokens": metrics.total_tokens,
            "total_time_s": metrics.total_time_s,
            "ttft_s": metrics.ttft_s,
            "tpot_ms": metrics.tpot_ms,
            "tokens_per_second": metrics.tokens_per_second,
            "peak_vram_mb": metrics.peak_vram_mb,
        }
        all_metrics.append(run_result)

        logger.info(
            f"Run {run_idx + 1}/{measured_runs}: "
            f"{metrics.tokens_per_second:.2f} tok/s, "
            f"TTFT={metrics.ttft_s*1000:.1f}ms, "
            f"TPOT={metrics.tpot_ms:.1f}ms, "
            f"VRAM={metrics.peak_vram_mb:.0f}MB"
        )

    # Save results
    results_file = output_path / "metrics.json"
    with open(results_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Compute summary
    import numpy as np
    tps = np.array([m["tokens_per_second"] for m in all_metrics])
    ttft = np.array([m["ttft_s"] for m in all_metrics]) * 1000
    tpot = np.array([m["tpot_ms"] for m in all_metrics])
    vram = np.array([m["peak_vram_mb"] for m in all_metrics])

    summary = {
        "model": model_name,
        "quantization": f"{bits}-bit" if quantize else "FP16",
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "tokens_per_second": {
            "mean": float(tps.mean()),
            "std": float(tps.std()),
            "min": float(tps.min()),
            "max": float(tps.max()),
        },
        "ttft_ms": {
            "mean": float(ttft.mean()),
            "std": float(ttft.std()),
        },
        "tpot_ms": {
            "mean": float(tpot.mean()),
            "std": float(tpot.std()),
        },
        "peak_vram_mb": {
            "mean": float(vram.mean()),
            "max": float(vram.max()),
        },
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nBenchmark Summary:")
    logger.info(f"  Tokens/s: {tps.mean():.2f} ± {tps.std():.2f}")
    logger.info(f"  TTFT:     {ttft.mean():.1f} ± {ttft.std():.1f} ms")
    logger.info(f"  TPOT:     {tpot.mean():.1f} ± {tpot.std():.1f} ms")
    logger.info(f"  Peak VRAM: {vram.max():.0f} MB")
    logger.info(f"Results saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vanilla decoding benchmark")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="experiments/01_vanilla")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
        # Override with config values
        model_name = config.get("model", {}).get("name", args.model)
        max_new_tokens = config.get("generation", {}).get("max_new_tokens", args.max_new_tokens)
        warmup = config.get("benchmark", {}).get("warmup_runs", args.warmup)
        runs = config.get("benchmark", {}).get("measured_runs", args.runs)
        quantize = config.get("model", {}).get("quantization", {}).get("enabled", True)
        bits = config.get("model", {}).get("quantization", {}).get("bits", 4)
    else:
        model_name = args.model
        max_new_tokens = args.max_new_tokens
        warmup = args.warmup
        runs = args.runs
        quantize = not args.no_quantize
        bits = args.bits

    run_benchmark(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        warmup_runs=warmup,
        measured_runs=runs,
        output_dir=args.output_dir,
        quantize=quantize,
        bits=bits,
    )


if __name__ == "__main__":
    main()
