"""Context-scaling vanilla baseline benchmark.

Benchmarks vanilla autoregressive decoding across multiple context lengths
using a fixed prompt set. Collects comprehensive metrics including:
- Performance: tokens/s, TTFT, TPOT, total latency
- Memory: peak/allocated/reserved VRAM
- Hardware: GPU utilization, clock, temperature, power
- Energy: J/token

Usage:
    python scripts/benchmark_context_scaling.py
    python scripts/benchmark_context_scaling.py --config configs/baseline.yaml
    python scripts/benchmark_context_scaling.py --context-lengths 128 512 1024
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizer

from zassd.models.loader import load_model, load_tokenizer
from zassd.profiling.gpu import GPUProfiler
from zassd.profiling.memory import get_vram_usage, reset_vram_stats
from zassd.utils.config import load_config
from zassd.utils.seed import set_seed
from zassd.utils.logging import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FullMetrics:
    """Comprehensive metrics for a single generation run."""
    # Performance
    total_tokens: int = 0
    input_tokens: int = 0
    total_time_s: float = 0.0
    ttft_s: float = 0.0
    tpot_ms: float = 0.0
    tokens_per_second: float = 0.0
    per_token_times_ms: list[float] = field(default_factory=list)
    # Memory
    peak_vram_mb: float = 0.0
    allocated_vram_mb: float = 0.0
    reserved_vram_mb: float = 0.0
    # Hardware (averaged during generation)
    gpu_utilization_pct: float = 0.0
    gpu_clock_mhz: float = 0.0
    gpu_temperature_c: float = 0.0
    gpu_power_w: float = 0.0
    # Energy
    energy_joules: float = 0.0
    joules_per_token: float = 0.0
    # Output (for greedy ground truth)
    output_text: str = ""
    prompt_text: str = ""


# ---------------------------------------------------------------------------
# Hardware sampler (background thread)
# ---------------------------------------------------------------------------

class HardwareSampler:
    """Samples GPU metrics in a background thread during generation."""

    def __init__(self, gpu_profiler: GPUProfiler, interval_s: float = 0.05):
        self.profiler = gpu_profiler
        self.interval = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self._aggregate()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            try:
                sample = {
                    "timestamp": time.perf_counter(),
                    "power_w": self.profiler.get_power_usage(),
                    "utilization": self.profiler.get_utilization(),
                    "temperature_c": self.profiler.get_temperature(),
                    "clocks": self.profiler.get_clock_speeds(),
                }
                self.samples.append(sample)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _aggregate(self) -> dict:
        if not self.samples:
            return {}

        powers = [s["power_w"] for s in self.samples]
        utils = [s["utilization"]["gpu_pct"] for s in self.samples]
        temps = [s["temperature_c"] for s in self.samples]
        clocks = [s["clocks"]["graphics_mhz"] for s in self.samples]

        # Energy via trapezoidal integration
        energy = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i]["timestamp"] - self.samples[i - 1]["timestamp"]
            avg_p = (self.samples[i]["power_w"] + self.samples[i - 1]["power_w"]) / 2
            energy += avg_p * dt

        return {
            "gpu_utilization_pct": float(np.mean(utils)),
            "gpu_clock_mhz": float(np.mean(clocks)),
            "gpu_temperature_c": float(np.mean(temps)),
            "gpu_power_w": float(np.mean(powers)),
            "energy_joules": energy,
            "num_samples": len(self.samples),
        }


# ---------------------------------------------------------------------------
# Generation with full metrics
# ---------------------------------------------------------------------------

def generate_with_full_metrics(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    gpu_profiler: GPUProfiler | None = None,
    device: str = "cuda:0",
) -> FullMetrics:
    """Generate tokens and collect comprehensive metrics."""
    metrics = FullMetrics()
    metrics.input_tokens = input_ids.shape[1]

    # Setup hardware sampler
    hw_sampler = None
    if gpu_profiler:
        hw_sampler = HardwareSampler(gpu_profiler, interval_s=0.05)

    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats(device)

    generated_ids = input_ids.clone()
    past_key_values = None

    # Start hardware sampling
    if hw_sampler:
        hw_sampler.start()

    start_time = time.perf_counter()

    with torch.no_grad():
        for step in range(max_new_tokens):
            step_start = time.perf_counter()

            if past_key_values is None:
                outputs = model(
                    input_ids=generated_ids,
                    use_cache=True,
                )
            else:
                outputs = model(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Greedy decoding (deterministic for ground truth)
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            step_time = (time.perf_counter() - step_start) * 1000
            metrics.per_token_times_ms.append(step_time)

            if step == 0:
                metrics.ttft_s = time.perf_counter() - start_time

            if next_token.item() == tokenizer.eos_token_id:
                break

    end_time = time.perf_counter()

    # Stop hardware sampling
    hw_stats = {}
    if hw_sampler:
        hw_stats = hw_sampler.stop()

    # Compute performance metrics
    new_tokens = generated_ids.shape[1] - input_ids.shape[1]
    metrics.total_tokens = new_tokens
    metrics.total_time_s = end_time - start_time
    metrics.tokens_per_second = (
        new_tokens / metrics.total_time_s if metrics.total_time_s > 0 else 0
    )
    if len(metrics.per_token_times_ms) > 1:
        metrics.tpot_ms = float(np.mean(metrics.per_token_times_ms[1:]))

    # Memory metrics
    metrics.peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    metrics.allocated_vram_mb = torch.cuda.memory_allocated(device) / (1024**2)
    metrics.reserved_vram_mb = torch.cuda.memory_reserved(device) / (1024**2)

    # Hardware metrics
    if hw_stats:
        metrics.gpu_utilization_pct = hw_stats.get("gpu_utilization_pct", 0)
        metrics.gpu_clock_mhz = hw_stats.get("gpu_clock_mhz", 0)
        metrics.gpu_temperature_c = hw_stats.get("gpu_temperature_c", 0)
        metrics.gpu_power_w = hw_stats.get("gpu_power_w", 0)
        metrics.energy_joules = hw_stats.get("energy_joules", 0)
        if new_tokens > 0:
            metrics.joules_per_token = metrics.energy_joules / new_tokens

    # Decode output for ground truth
    metrics.output_text = tokenizer.decode(
        generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    )

    return metrics


# ---------------------------------------------------------------------------
# Context preparation
# ---------------------------------------------------------------------------

def prepare_context_inputs(
    tokenizer: PreTrainedTokenizer,
    context_length: int,
    prompts_file: str = "data/benchmarks/prompts.jsonl",
    num_prompts: int = 30,
    device: str = "cuda:0",
) -> list[dict]:
    """Prepare input_ids at a specific context length.

    Uses WikiText-2 as filler text to pad prompts to the desired context
    length. Each prompt is prepended with filler to reach target length.
    """
    # Load fixed prompts
    prompts = []
    prompts_path = Path(prompts_file)
    if prompts_path.exists():
        with open(prompts_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(json.loads(line))
    else:
        logger.warning(f"Prompts file not found: {prompts_file}, using defaults")
        prompts = [
            {"id": i, "category": "general", "prompt": f"Question {i}: Explain transformers."}
            for i in range(num_prompts)
        ]

    # Load filler text from wikitext for longer contexts
    filler_text = ""
    if context_length > 128:
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            # Concatenate enough text
            texts = [t for t in ds["text"] if len(t.strip()) > 50]
            filler_text = " ".join(texts)
            logger.info(f"Loaded filler text: {len(filler_text)} chars")
        except Exception as e:
            logger.warning(f"Could not load wikitext: {e}. Using repeated text.")
            filler_text = "The transformer architecture has revolutionized natural language processing. " * 5000

    prepared = []
    for i in range(min(num_prompts, len(prompts))):
        prompt_text = prompts[i]["prompt"]
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

        if len(prompt_tokens) >= context_length:
            # Truncate prompt to context_length
            input_ids = torch.tensor([prompt_tokens[:context_length]], device=device)
        elif context_length <= 128:
            # Short context: use prompt as-is
            input_ids = tokenizer(
                prompt_text, return_tensors="pt",
                truncation=True, max_length=context_length,
            ).input_ids.to(device)
        else:
            # Pad with filler text to reach context_length
            filler_tokens = tokenizer.encode(filler_text, add_special_tokens=False)
            needed = context_length - len(prompt_tokens)
            # Take filler from a different offset for each prompt
            offset = (i * 1000) % max(1, len(filler_tokens) - needed)
            prefix_tokens = filler_tokens[offset:offset + needed]
            full_tokens = prefix_tokens + prompt_tokens
            input_ids = torch.tensor([full_tokens[:context_length]], device=device)

        prepared.append({
            "id": prompts[i]["id"],
            "category": prompts[i]["category"],
            "prompt": prompt_text,
            "input_ids": input_ids,
            "actual_length": input_ids.shape[1],
        })

    return prepared


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_context_benchmark(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context_length: int,
    max_new_tokens: int = 128,
    warmup_runs: int = 10,
    measured_runs: int = 30,
    output_dir: str = "experiments/01_vanilla",
    gpu_profiler: GPUProfiler | None = None,
    device: str = "cuda:0",
) -> dict:
    """Run benchmark for a single context length."""
    ctx_dir = Path(output_dir) / f"ctx_{context_length}"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"CONTEXT LENGTH: {context_length}")
    logger.info(f"{'='*60}")

    # Prepare inputs
    inputs = prepare_context_inputs(
        tokenizer, context_length,
        num_prompts=max(warmup_runs + measured_runs, 40),
        device=device,
    )
    logger.info(f"Prepared {len(inputs)} prompts at ~{context_length} tokens")

    # Warmup
    logger.info(f"Warmup: {warmup_runs} runs...")
    for i in range(warmup_runs):
        idx = i % len(inputs)
        _ = generate_with_full_metrics(
            model, tokenizer, inputs[idx]["input_ids"],
            max_new_tokens=32, gpu_profiler=None, device=device,
        )

    # Clear CUDA cache after warmup
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # Measured runs
    logger.info(f"Measuring: {measured_runs} runs, {max_new_tokens} new tokens each...")
    all_metrics: list[dict] = []

    for run_idx in range(measured_runs):
        idx = run_idx % len(inputs)
        reset_vram_stats(device)

        metrics = generate_with_full_metrics(
            model, tokenizer, inputs[idx]["input_ids"],
            max_new_tokens=max_new_tokens,
            gpu_profiler=gpu_profiler,
            device=device,
        )

        run_result = {
            "run_idx": run_idx,
            "prompt_id": inputs[idx]["id"],
            "category": inputs[idx]["category"],
            "context_length": inputs[idx]["actual_length"],
            "total_tokens": metrics.total_tokens,
            "total_time_s": metrics.total_time_s,
            "ttft_s": metrics.ttft_s,
            "tpot_ms": metrics.tpot_ms,
            "tokens_per_second": metrics.tokens_per_second,
            "peak_vram_mb": metrics.peak_vram_mb,
            "allocated_vram_mb": metrics.allocated_vram_mb,
            "reserved_vram_mb": metrics.reserved_vram_mb,
            "gpu_utilization_pct": metrics.gpu_utilization_pct,
            "gpu_clock_mhz": metrics.gpu_clock_mhz,
            "gpu_temperature_c": metrics.gpu_temperature_c,
            "gpu_power_w": metrics.gpu_power_w,
            "energy_joules": metrics.energy_joules,
            "joules_per_token": metrics.joules_per_token,
            "output": metrics.output_text,
        }
        all_metrics.append(run_result)

        logger.info(
            f"  Run {run_idx + 1}/{measured_runs}: "
            f"{metrics.tokens_per_second:.1f} tok/s, "
            f"TTFT={metrics.ttft_s*1000:.1f}ms, "
            f"TPOT={metrics.tpot_ms:.1f}ms, "
            f"VRAM={metrics.peak_vram_mb:.0f}MB, "
            f"Power={metrics.gpu_power_w:.1f}W, "
            f"J/tok={metrics.joules_per_token:.3f}"
        )

    # Save raw metrics
    with open(ctx_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Compute summary
    summary = _compute_summary(all_metrics, context_length, max_new_tokens)

    with open(ctx_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save ground truth outputs
    outputs = [
        {"prompt_id": m["prompt_id"], "output": m["output"]}
        for m in all_metrics
    ]
    with open(ctx_dir / "outputs.jsonl", "w") as f:
        for o in outputs:
            f.write(json.dumps(o) + "\n")

    _log_summary(summary)
    return summary


def _compute_summary(metrics: list[dict], ctx_len: int, max_new: int) -> dict:
    """Compute aggregate summary statistics."""
    def stats(values):
        arr = np.array(values)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    return {
        "context_length": ctx_len,
        "max_new_tokens": max_new,
        "num_runs": len(metrics),
        "performance": {
            "tokens_per_second": stats([m["tokens_per_second"] for m in metrics]),
            "ttft_ms": stats([m["ttft_s"] * 1000 for m in metrics]),
            "tpot_ms": stats([m["tpot_ms"] for m in metrics]),
            "total_time_s": stats([m["total_time_s"] for m in metrics]),
        },
        "memory": {
            "peak_vram_mb": stats([m["peak_vram_mb"] for m in metrics]),
            "allocated_vram_mb": stats([m["allocated_vram_mb"] for m in metrics]),
            "reserved_vram_mb": stats([m["reserved_vram_mb"] for m in metrics]),
        },
        "hardware": {
            "gpu_utilization_pct": stats([m["gpu_utilization_pct"] for m in metrics]),
            "gpu_clock_mhz": stats([m["gpu_clock_mhz"] for m in metrics]),
            "gpu_temperature_c": stats([m["gpu_temperature_c"] for m in metrics]),
            "gpu_power_w": stats([m["gpu_power_w"] for m in metrics]),
        },
        "energy": {
            "energy_joules": stats([m["energy_joules"] for m in metrics]),
            "joules_per_token": stats([m["joules_per_token"] for m in metrics]),
        },
    }


def _log_summary(summary: dict) -> None:
    """Log summary in a readable format."""
    ctx = summary["context_length"]
    perf = summary["performance"]
    mem = summary["memory"]
    hw = summary["hardware"]
    eng = summary["energy"]

    logger.info(f"\n--- Summary (ctx={ctx}) ---")
    logger.info(f"  tok/s:  {perf['tokens_per_second']['mean']:.1f} ± {perf['tokens_per_second']['std']:.1f}")
    logger.info(f"  TTFT:   {perf['ttft_ms']['mean']:.1f} ± {perf['ttft_ms']['std']:.1f} ms")
    logger.info(f"  TPOT:   {perf['tpot_ms']['mean']:.1f} ± {perf['tpot_ms']['std']:.1f} ms")
    logger.info(f"  VRAM:   peak={mem['peak_vram_mb']['mean']:.0f}MB, alloc={mem['allocated_vram_mb']['mean']:.0f}MB")
    logger.info(f"  GPU:    util={hw['gpu_utilization_pct']['mean']:.0f}%, power={hw['gpu_power_w']['mean']:.1f}W, temp={hw['gpu_temperature_c']['mean']:.0f}°C")
    logger.info(f"  Energy: {eng['joules_per_token']['mean']:.4f} J/token")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Context-scaling vanilla baseline benchmark"
    )
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=None,
                        help="Override context lengths (e.g. 128 512 1024)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/01_vanilla")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Resolve parameters (CLI overrides config)
    model_name = args.model or config.get("model", {}).get("name", "Qwen/Qwen2.5-3B-Instruct")
    context_lengths = args.context_lengths or config.get("benchmark", {}).get("context_lengths", [128, 512, 1024, 2048, 4096])
    max_new_tokens = args.max_new_tokens or config.get("generation", {}).get("max_new_tokens", 128)
    warmup_runs = args.warmup or config.get("benchmark", {}).get("warmup_runs", 10)
    measured_runs = args.runs or config.get("benchmark", {}).get("measured_runs", 30)
    quantize = not args.no_quantize and config.get("model", {}).get("quantization", {}).get("enabled", True)
    bits = args.bits or config.get("model", {}).get("quantization", {}).get("bits", 4)
    seed = config.get("benchmark", {}).get("seed", 42)

    # Setup
    setup_logging(log_file=f"{args.output_dir}/context_scaling.log")
    set_seed(seed)

    logger.info("=" * 60)
    logger.info("CONTEXT-SCALING VANILLA BASELINE")
    logger.info("=" * 60)
    logger.info(f"Model:          {model_name}")
    logger.info(f"Quantization:   {bits}-bit" if quantize else "FP16")
    logger.info(f"Context lengths: {context_lengths}")
    logger.info(f"Max new tokens: {max_new_tokens}")
    logger.info(f"Warmup/Measured: {warmup_runs}/{measured_runs}")

    # Load model (once for all context lengths)
    logger.info(f"\nLoading model: {model_name}")
    model = load_model(model_name, quantize=quantize, bits=bits)
    tokenizer = load_tokenizer(model_name)

    vram_model = get_vram_usage()
    logger.info(f"Model VRAM: {vram_model}")

    # Initialize GPU profiler
    try:
        gpu_profiler = GPUProfiler(device_index=0)
    except Exception as e:
        logger.warning(f"GPU profiler not available: {e}")
        gpu_profiler = None

    # Run benchmarks for each context length
    all_summaries = {}
    for ctx_len in context_lengths:
        summary = run_context_benchmark(
            model=model,
            tokenizer=tokenizer,
            context_length=ctx_len,
            max_new_tokens=max_new_tokens,
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            output_dir=args.output_dir,
            gpu_profiler=gpu_profiler,
        )
        all_summaries[str(ctx_len)] = summary

        # GC between context lengths
        torch.cuda.empty_cache()

    # Save combined summary
    combined_path = Path(args.output_dir) / "context_scaling_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    # Print final comparison table
    logger.info("\n" + "=" * 80)
    logger.info("CONTEXT SCALING RESULTS")
    logger.info("=" * 80)
    logger.info(f"{'Context':>8} | {'tok/s':>10} | {'TTFT(ms)':>10} | {'TPOT(ms)':>10} | {'VRAM(MB)':>10} | {'Power(W)':>10} | {'J/tok':>8}")
    logger.info("-" * 80)

    for ctx_len in context_lengths:
        s = all_summaries[str(ctx_len)]
        p = s["performance"]
        m = s["memory"]
        h = s["hardware"]
        e = s["energy"]
        logger.info(
            f"{ctx_len:>8} | "
            f"{p['tokens_per_second']['mean']:>8.1f}±{p['tokens_per_second']['std']:.1f} | "
            f"{p['ttft_ms']['mean']:>8.1f}±{p['ttft_ms']['std']:.1f} | "
            f"{p['tpot_ms']['mean']:>8.1f}±{p['tpot_ms']['std']:.1f} | "
            f"{m['peak_vram_mb']['mean']:>10.0f} | "
            f"{h['gpu_power_w']['mean']:>8.1f}±{h['gpu_power_w']['std']:.1f} | "
            f"{e['joules_per_token']['mean']:>8.4f}"
        )

    logger.info(f"\nAll results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
