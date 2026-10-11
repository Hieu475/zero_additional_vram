#!/usr/bin/env python3
"""Reproducible paired benchmark for ZASSD decoding baselines.

Writes one JSONL row per method/prompt/repeat/order seed, plus run metadata and
paired statistical summaries. Existing diagnostic outputs are never overwritten
unless an explicit output directory is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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


def tokenize_prompt(tokenizer, prompt: str) -> torch.Tensor:
    """Create the single CPU token-ID tensor shared by every method in a pair."""
    return tokenizer(prompt, return_tensors="pt")["input_ids"].detach().cpu()


def validate_complete_pairs(rows: list[dict]) -> None:
    """Reject missing/duplicate methods or differing input IDs within a pair."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["benchmark"], row["id"], row["repeat"], row["order_seed"])
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        methods = [row["method"] for row in group]
        if len(methods) != len(METHODS) or set(methods) != set(METHODS):
            raise ValueError(f"Incomplete or duplicate method pair {key}: {methods}")
        token_ids = {tuple(row["input_token_ids"]) for row in group}
        if len(token_ids) != 1:
            raise ValueError(f"Methods received different input token IDs for {key}")


def prompt_level_means(rows: list[dict], value_key: str) -> dict[tuple[str, str], float]:
    """Average repeats/order rotations within each prompt before CI estimation."""
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = row.get(value_key)
        if value is not None and np.isfinite(value):
            grouped.setdefault((row["benchmark"], row["id"]), []).append(float(value))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def paired_prompt_ratios(rows: list[dict], method: str, metric: str) -> list[float]:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["benchmark"], row["id"], row["repeat"], row["order_seed"])
        grouped.setdefault(key, {})[row["method"]] = row
    per_prompt: dict[tuple[str, str], list[float]] = {}
    for key, values in grouped.items():
        if method not in values or "vanilla" not in values:
            continue
        baseline = values["vanilla"].get(metric, 0.0)
        if baseline and np.isfinite(baseline):
            value = values[method].get(metric, 0.0)
            if np.isfinite(value):
                per_prompt.setdefault(key[:2], []).append(value / baseline)
    return [float(np.mean(values)) for values in per_prompt.values()]


def paired_method_prompt_ratios(rows: list[dict], method: str, baseline_method: str,
                                metric: str = "end_to_end_tps") -> list[float]:
    grouped: dict[tuple, dict[str, dict]] = {}
    for row in rows:
        key = (row["benchmark"], row["id"], row["repeat"], row["order_seed"])
        grouped.setdefault(key, {})[row["method"]] = row
    per_prompt: dict[tuple[str, str], list[float]] = {}
    for key, values in grouped.items():
        if method not in values or baseline_method not in values:
            continue
        baseline = values[baseline_method].get(metric, 0.0)
        value = values[method].get(metric, 0.0)
        if baseline and np.isfinite(baseline) and np.isfinite(value):
            per_prompt.setdefault(key[:2], []).append(value / baseline)
    return [float(np.mean(values)) for values in per_prompt.values()]


def ratio_summary(ratios: list[float]) -> dict:
    return {
        "n_prompts": len(ratios),
        "mean_ratio": float(np.mean(ratios)) if ratios else None,
        "median_ratio": float(np.median(ratios)) if ratios else None,
        "p10_ratio": float(np.quantile(ratios, 0.10)) if ratios else None,
        "p90_ratio": float(np.quantile(ratios, 0.90)) if ratios else None,
        "ratio_ci95": bootstrap_mean_ci(ratios),
        "faster_prompt_pct": float(np.mean(np.asarray(ratios) > 1.0) * 100) if ratios else None,
        "bootstrap_unit": "prompt (repeats/order seeds averaged within prompt)",
    }


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
        prompt_means = prompt_level_means(method_rows, "speedup_vs_vanilla")
        finite_speeds = list(prompt_means.values())
        report[method] = {
            "n": len(method_rows),
            "n_prompts": len(prompt_means),
            "mean_speedup": float(np.mean(finite_speeds)) if finite_speeds else None,
            "median_speedup": float(np.median(finite_speeds)) if finite_speeds else None,
            "p10_speedup": float(np.quantile(finite_speeds, 0.10)) if finite_speeds else None,
            "p90_speedup": float(np.quantile(finite_speeds, 0.90)) if finite_speeds else None,
            "speedup_ci95": bootstrap_mean_ci(finite_speeds),
            "bootstrap_unit": "prompt (repeats/order seeds averaged within prompt)",
            "exact_match_pct": (float(np.mean([row["exact_token_match"] for row in method_rows]) * 100)
                                if method_rows else None),
            "speedup_by_prompt": [{"benchmark": benchmark, "id": sample_id,
                                   "speedup": speed}
                                  for (benchmark, sample_id), speed in prompt_means.items()],
        }
    for method in METHODS:
        if method == "vanilla":
            continue
        report[f"{method}_vs_vanilla_paired"] = ratio_summary(
            paired_prompt_ratios(rows, method, "end_to_end_tps"))
        report[f"{method}_decode_tps_vs_vanilla_paired"] = ratio_summary(
            paired_prompt_ratios(rows, method, "decode_tps"))
    for method in ("routed", "pld_gate"):
        report[f"{method}_vs_pld_paired"] = ratio_summary(
            paired_method_prompt_ratios(rows, method, "pld"))
        report[f"{method}_decode_tps_vs_pld_paired"] = ratio_summary(
            paired_method_prompt_ratios(rows, method, "pld", "decode_tps"))
    for control in ("vanilla_via_speculative_core", "pld_forced_miss"):
        report[f"{control}_control_ratio"] = ratio_summary(
            paired_prompt_ratios(rows, control, "end_to_end_tps"))
    return report


def run_vanilla_timed(model, tokenizer, prompt, max_new_tokens, device):
    """Greedy vanilla path with prefill and decode wall time separated."""
    from zassd.cache.kv_cache import TargetKVCache
    ids = (prompt if isinstance(prompt, torch.Tensor)
           else tokenize_prompt(tokenizer, prompt)).to(device)
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
              "power_w_snapshot": None, "power_w_mean": None,
              "energy_j": None, "temperature_c": None,
              "telemetry_timestamp_unix_s": time.time()}
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
            "power_w_snapshot": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
        })
    except Exception:
        pass
    return result


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def driver_version() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


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

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    default_output = f"experiments/benchmark_validation/run_{run_stamp}_{(commit or 'unknown')[:7]}"
    output = Path(args.output_dir or default_output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty benchmark directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model_revision = (getattr(model.config, "_commit_hash", None)
                      or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"))
    env = {"commit_sha": commit, "python": platform.python_version(),
           "platform": platform.platform(), "torch_version": torch.__version__,
           "transformers_version": package_version("transformers"),
           "bitsandbytes_version": package_version("bitsandbytes"),
           "huggingface_hub_version": package_version("huggingface-hub"),
           "cuda_version": torch.version.cuda, "driver_version": driver_version(),
           "attention_implementation": getattr(model.config, "_attn_implementation", None),
           "model_id": args.model, "model_revision": model_revision,
           "tokenizer_revision": (getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
                                  or model_revision),
           "quantization": "NF4" if not ("bnb-4bit" in args.model.lower() or "4bit" in args.model.lower()) else "model-default",
           "compute_dtype": "float16" if not ("bnb-4bit" in args.model.lower() or "4bit" in args.model.lower()) else str(getattr(model.config, "torch_dtype", None)),
           "environment_variables": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
           "gpu": gpu_telemetry(args.device),
           "dataset_sha256": {name: sha256_file(path) for name, path in BENCHMARKS.items()}}
    (output / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    config = {**vars(args), "resolved_output_dir": str(output),
              "decode_tokens_definition": "generated output tokens excluding the first token selected from prefill logits",
              "ttft_definition": "prefill wall time until final prompt logits are available; excludes sampling/argmax materialization",
              "decode_seconds_definition": "post-prefill wall time through completion, including first-token selection overhead"}
    (output / "config.yaml").write_text("\n".join(f"{k}: {json.dumps(v)}" for k, v in config.items()) + "\n", encoding="utf-8")
    rows: list[dict] = []
    records_by_bench = {}
    for bench, filename in BENCHMARKS.items():
        with open(filename, encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        records_by_bench[bench] = records[:args.samples_per_bench]

    # Warm every method on the same small set before collecting measurements.
    for bench, records in records_by_bench.items():
        for item in records[:args.warmup_prompts]:
            prompt = tokenize_prompt(tokenizer, item["prompt"])
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
            prompt_ids = tokenize_prompt(tokenizer, prompt)
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
                            model, tokenizer, prompt_ids, args.max_new_tokens, args.device)
                        metrics = None
                    elif method == "vanilla_via_speculative_core":
                        _, metrics = self_speculative_generate(
                            model, tokenizer, manager, skip_indices, prompt_ids, k=0,
                            draft_mode="layer_skip", max_new_tokens=args.max_new_tokens,
                            device=args.device)
                        ids = metrics.generated_token_ids
                        total_ms = metrics.total_time_s * 1000
                        prefill_ms = metrics.prefill_time_s * 1000
                        decode_ms = max(0.0, total_ms - prefill_ms)
                    elif method == "pld_forced_miss":
                        _, metrics = prompt_lookup_generate(
                            model, tokenizer, prompt_ids, k=4, max_new_tokens=args.max_new_tokens,
                            device=args.device, force_miss=True)
                        ids = metrics.generated_token_ids
                        total_ms = metrics.total_time_s * 1000
                        prefill_ms = metrics.prefill_time_s * 1000
                        decode_ms = max(0.0, total_ms - prefill_ms)
                    elif method == "pld":
                        _, metrics = prompt_lookup_generate(
                            model, tokenizer, prompt_ids, k=4,
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
                            model, tokenizer, manager, skip_indices, prompt_ids,
                            k=2, draft_mode="routed", router=router,
                            max_new_tokens=args.max_new_tokens, device=args.device)
                        ids = metrics.generated_token_ids
                        total_ms = metrics.total_time_s * 1000
                        prefill_ms = metrics.prefill_time_s * 1000
                        decode_ms = max(0.0, total_ms - prefill_ms)
                    tps = len(ids) / (total_ms / 1000) if total_ms > 0 else 0.0
                    decode_tokens = max(0, len(ids) - 1)
                    decode_seconds = decode_ms / 1000.0
                    iteration_rows = metrics.per_iteration_stats if metrics else []
                    def phase_sum(*names):
                        return sum(float(cycle.get(name, 0.0) or 0.0)
                                   for cycle in iteration_rows for name in names)
                    measurements[method] = {"ids": ids, "tps": tps}
                    rows.append({
                        "benchmark": bench, "id": sample_id, "method": method,
                        "repeat": repeat, "order_seed": order_seed,
                        "input_tokens": int(prompt_ids.shape[1]),
                        "input_token_ids": prompt_ids[0].tolist(),
                        "output_tokens": len(ids), "prefill_ms": prefill_ms,
                        "ttft_ms": prefill_ms,
                        "decode_tokens": decode_tokens,
                        "decode_seconds": decode_seconds,
                        "decode_tps": decode_tokens / decode_seconds if decode_seconds > 0 else 0.0,
                        "decode_ms": decode_ms, "total_ms": total_ms, "tps": tps,
                        "end_to_end_tps": tps,
                        "generated_token_ids": ids,
                        "pld_cycles": metrics.pld_cycles if metrics else 0,
                        "layer_skip_cycles": metrics.layer_skip_cycles if metrics else 0,
                        "vanilla_cycles": metrics.num_verification_cycles if metrics else len(ids),
                        "pld_hit_rate": (metrics.pld_cycles / metrics.num_verification_cycles
                                         if metrics and metrics.num_verification_cycles else 0.0),
                        "accepted_tokens": metrics.total_accepted_tokens if metrics else 0,
                        "accepted_tokens_per_cycle": (
                            metrics.total_accepted_tokens / metrics.num_verification_cycles
                            if metrics and metrics.num_verification_cycles else 0.0),
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
                    row["vanilla_token_ids"] = vanilla_ids
                    candidate_ids = row["generated_token_ids"]
                    row["first_divergence_index"] = next(
                        (position for position, (left, right) in enumerate(zip(candidate_ids, vanilla_ids))
                         if left != right),
                        min(len(candidate_ids), len(vanilla_ids))
                        if len(candidate_ids) != len(vanilla_ids) else None,
                    )
                with (output / "raw_measurements.jsonl").open("a", encoding="utf-8") as stream:
                    for row in rows[-len(METHODS):]:
                        stream.write(json.dumps(row) + "\n")
            print(f"{bench}: {index + 1}/{len(records)}", flush=True)

    validate_complete_pairs(rows)
    report = {"model": args.model, "max_new_tokens": args.max_new_tokens,
              "samples_per_bench": args.samples_per_bench,
              "methods": {"pld": "PLD k=4 with vanilla on lookup miss",
                          "pld_gate": "PLD k<=4 with cost-based vanilla choice; no layer skip",
                          "routed": "PLD k<=4, layer skip k=2, or vanilla"},
              "overall": summarize(rows),
              "by_benchmark": {bench: summarize([row for row in rows
                                                if row["benchmark"] == bench])
                               for bench in BENCHMARKS}}
    report["run_config"] = vars(args)
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "statistics.json").write_text(json.dumps({
        "overall": report["overall"], "by_benchmark": report["by_benchmark"]
    }, indent=2), encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"Saved benchmark artifacts to {output}")


if __name__ == "__main__":
    main()
