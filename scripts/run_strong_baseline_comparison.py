#!/usr/bin/env python3
"""Reproducible paired benchmark for ZASSD decoding baselines.

Writes one JSONL row per method/prompt/repeat/order seed, plus run metadata and
paired statistical summaries. Existing diagnostic outputs are never overwritten
unless an explicit output directory is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
METHODS = ("vanilla", "vanilla_via_speculative_core", "pld_forced_miss",
           "pld", "pld_gate", "routed")


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
        prompt_speeds: dict[tuple[str, str], list[float]] = {}
        for row in method_rows:
            value = row["speedup_vs_vanilla"]
            if value is not None and np.isfinite(value):
                prompt_speeds.setdefault((row["benchmark"], row["id"]), []).append(value)
        prompt_means = {key: float(np.mean(values)) for key, values in prompt_speeds.items()}
        finite_speeds = list(prompt_means.values())
        report[method] = {
            "n": len(method_rows),
            "n_prompts": len(prompt_means),
            "mean_speedup": float(np.mean(finite_speeds)) if finite_speeds else None,
            "speedup_ci95": bootstrap_mean_ci(finite_speeds),
            "exact_match_pct": (float(np.mean([row["exact_token_match"] for row in method_rows]) * 100)
                                if method_rows else None),
            "speedup_by_prompt": [{"benchmark": benchmark, "id": sample_id,
                                   "speedup": speed}
                                  for (benchmark, sample_id), speed in prompt_means.items()],
        }
    grouped = {}
    for row in rows:
        grouped.setdefault((row["benchmark"], row["id"], row["repeat"], row["order_seed"]), {})[row["method"]] = row
    for method in METHODS:
        if method == "vanilla":
            continue
        prompt_ratios: dict[tuple[str, str], list[float]] = {}
        for key, values in grouped.items():
            if method in values and "vanilla" in values and values["vanilla"]["tps"] > 0:
                prompt_ratios.setdefault(key[:2], []).append(
                    values[method]["tps"] / values["vanilla"]["tps"])
        ratios = [float(np.mean(values)) for values in prompt_ratios.values()]
        report[f"{method}_vs_vanilla_paired"] = {
            "n": len(ratios), "mean_ratio": float(np.mean(ratios)) if ratios else None,
            "ratio_ci95": bootstrap_mean_ci(ratios),
            "faster_prompt_pct": float(np.mean(np.asarray(ratios) > 1.0) * 100) if ratios else None,
        }
    for control in ("vanilla_via_speculative_core", "pld_forced_miss"):
        prompt_ratios: dict[tuple[str, str], list[float]] = {}
        for key, values in grouped.items():
            if control in values and "vanilla" in values and values["vanilla"]["tps"] > 0:
                prompt_ratios.setdefault(key[:2], []).append(
                    values[control]["tps"] / values["vanilla"]["tps"])
        ratios = [float(np.mean(values)) for values in prompt_ratios.values()]
        report[f"{control}_control_ratio"] = {
            "n": len(ratios), "mean_ratio": float(np.mean(ratios)) if ratios else None,
            "ratio_ci95": bootstrap_mean_ci(ratios),
        }
    return report


def run_vanilla_timed(model, tokenizer, prompt, max_new_tokens, device):
    """Greedy vanilla path with prefill and decode wall time separated."""
    from zassd.cache.kv_cache import TargetKVCache
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    ids = inputs["input_ids"]
    cache = TargetKVCache(backend="static")
    torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model(ids, past_key_values=cache.cache, use_cache=True)
    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000
    decode_start = time.perf_counter()
    emitted = []
    token = out.logits[:, -1:, :].argmax(dim=-1)
    token_id = int(token.item())
    emitted.append(token_id)
    eos = tokenizer.eos_token_id
    while len(emitted) < max_new_tokens and token_id != eos:
        with torch.no_grad():
            out = model(token, past_key_values=cache.cache, use_cache=True)
        token = out.logits[:, -1:, :].argmax(dim=-1)
        token_id = int(token.item())
        emitted.append(token_id)
    torch.cuda.synchronize(device)
    decode_ms = (time.perf_counter() - decode_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    return emitted, prefill_ms, decode_ms, total_ms


def gpu_telemetry(device: str) -> dict:
    result = {"gpu_name": None, "gpu_clock_mhz": None,
              "power_w_mean": None, "temperature_c": None}
    try:
        result["gpu_name"] = torch.cuda.get_device_name(device)
    except Exception:
        pass
    try:
        import pynvml
        index = torch.device(device).index or 0
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        result.update({
            "gpu_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "power_w_mean": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        })
    except Exception:
        pass
    return result


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--samples-per-bench", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--order-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--warmup-prompts", type=int, default=2)
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

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    output = Path(args.output_dir or f"experiments/benchmark_validation/run_{run_stamp}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty benchmark directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    records_by_bench = {}
    for bench, filename in BENCHMARKS.items():
        with open(filename, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        records_by_bench[bench] = records[:args.samples_per_bench]

    # Warm every method on the same small set before collecting measurements.
    for bench, records in records_by_bench.items():
        for item in records[:args.warmup_prompts]:
            prompt = item["prompt"]
            run_vanilla_timed(model, tokenizer, prompt, args.max_new_tokens, args.device)
            self_speculative_generate(model, tokenizer, manager, skip_indices, prompt,
                                      k=0, draft_mode="layer_skip",
                                      max_new_tokens=args.max_new_tokens, device=args.device)
            prompt_lookup_generate(model, tokenizer, prompt, k=4, max_new_tokens=args.max_new_tokens,
                                   device=args.device, force_miss=True)
            prompt_lookup_generate(model, tokenizer, prompt, k=4, max_new_tokens=args.max_new_tokens,
                                   device=args.device)
            for mode in ("pld_gate", "routed"):
                router = HybridDraftRouter(cost_model=cost_model, pld_k=4,
                                           enable_layer_skip=(mode == "routed"))
                self_speculative_generate(model, tokenizer, manager, skip_indices, prompt,
                                          k=2, draft_mode="routed", router=router,
                                          max_new_tokens=args.max_new_tokens, device=args.device)

    for bench, records in records_by_bench.items():
        for index, item in enumerate(records):
            prompt = item["prompt"]
            sample_id = item.get("id", f"{bench}_{index}")
            for repeat in range(args.repeats):
              for order_seed in args.order_seeds:
                order = list(METHODS)
                random.Random(order_seed + index * 7919 + repeat).shuffle(order)
                measurements = {}
                for method in order:
                    torch.cuda.reset_peak_memory_stats(args.device)
                    telemetry = gpu_telemetry(args.device)
                    if method == "vanilla":
                        ids, prefill_ms, decode_ms, total_ms = run_vanilla_timed(
                            model, tokenizer, prompt, args.max_new_tokens, args.device)
                        metrics = None
                    elif method == "vanilla_via_speculative_core":
                        _, metrics = self_speculative_generate(
                            model, tokenizer, manager, skip_indices, prompt, k=0,
                            draft_mode="layer_skip", max_new_tokens=args.max_new_tokens,
                            device=args.device)
                        ids = metrics.generated_token_ids
                        total_ms = metrics.total_time_s * 1000
                        prefill_ms = metrics.prefill_time_s * 1000
                        decode_ms = max(0.0, total_ms - prefill_ms)
                    elif method == "pld_forced_miss":
                        _, metrics = prompt_lookup_generate(
                            model, tokenizer, prompt, k=4, max_new_tokens=args.max_new_tokens,
                            device=args.device, force_miss=True)
                        ids = metrics.generated_token_ids
                        total_ms = metrics.total_time_s * 1000
                        prefill_ms = metrics.prefill_time_s * 1000
                        decode_ms = max(0.0, total_ms - prefill_ms)
                    elif method == "pld":
                        _, metrics = prompt_lookup_generate(
                            model, tokenizer, prompt, k=4,
                            max_new_tokens=args.max_new_tokens, device=args.device)
                        ids = metrics.generated_token_ids
                        total_ms = metrics.total_time_s * 1000
                        prefill_ms = metrics.prefill_time_s * 1000
                        decode_ms = max(0.0, total_ms - prefill_ms)
                    else:
                        router = HybridDraftRouter(
                            cost_model=cost_model, pld_k=4,
                            enable_layer_skip=(method == "routed"))
                        _, metrics = self_speculative_generate(
                            model, tokenizer, manager, skip_indices, prompt,
                            k=2, draft_mode="routed", router=router,
                            max_new_tokens=args.max_new_tokens, device=args.device)
                        ids = metrics.generated_token_ids
                        total_ms = metrics.total_time_s * 1000
                        prefill_ms = metrics.prefill_time_s * 1000
                        decode_ms = max(0.0, total_ms - prefill_ms)
                    tps = len(ids) / (total_ms / 1000) if total_ms > 0 else 0.0
                    iteration_rows = metrics.per_iteration_stats if metrics else []
                    def phase_sum(*names):
                        return sum(float(cycle.get(name, 0.0) or 0.0)
                                   for cycle in iteration_rows for name in names)
                    measurements[method] = {"ids": ids, "tps": tps}
                    rows.append({
                        "benchmark": bench, "id": sample_id, "method": method,
                        "repeat": repeat, "order_seed": order_seed,
                        "input_tokens": len(tokenizer(prompt, add_special_tokens=True)["input_ids"]),
                        "output_tokens": len(ids), "prefill_ms": prefill_ms,
                        "decode_ms": decode_ms, "total_ms": total_ms, "tps": tps,
                        "pld_cycles": metrics.pld_cycles if metrics else 0,
                        "layer_skip_cycles": metrics.layer_skip_cycles if metrics else 0,
                        "vanilla_cycles": metrics.num_verification_cycles if metrics else len(ids),
                        "phase_ms": {
                            "draft": metrics.draft_time_s * 1000 if metrics else 0.0,
                            "verify": metrics.verify_time_s * 1000 if metrics else decode_ms,
                            "cache": metrics.cache_time_s * 1000 if metrics else 0.0,
                            "lookup": phase_sum("lookup_ms"),
                            "crop": phase_sum("crop_ms"),
                            "cpu_decision": phase_sum("decision_ms", "controller_ms"),
                            "sync_wait": phase_sum("sync_wait_ms", "sync_ms"),
                        },
                        "exact_token_match": None,
                        "peak_vram_mb": (torch.cuda.max_memory_allocated(args.device) / 1024**2),
                        **telemetry,
                    })
                vanilla_ids = measurements["vanilla"]["ids"]
                for row in rows[-len(METHODS):]:
                    row["exact_token_match"] = row["method"] == "vanilla" or row["method"] in measurements and measurements[row["method"]]["ids"] == vanilla_ids
                    row["vanilla_tps"] = measurements["vanilla"]["tps"]
                    row["speedup_vs_vanilla"] = row["tps"] / measurements["vanilla"]["tps"] if measurements["vanilla"]["tps"] else None
            print(f"{bench}: {index + 1}/{len(records)}", flush=True)

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
    report["run_config"] = vars(args)
    with (output / "raw_measurements.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    env = {"commit_sha": commit, "python": platform.python_version(),
           "platform": platform.platform(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "gpu": gpu_telemetry(args.device),
           "model": args.model, "model_revision": None,
           "dataset_sha256": {name: sha256_file(path) for name, path in BENCHMARKS.items()}}
    (output / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text("\n".join(f"{k}: {json.dumps(v)}" for k, v in vars(args).items()) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"Saved benchmark artifacts to {output}")


if __name__ == "__main__":
    main()
