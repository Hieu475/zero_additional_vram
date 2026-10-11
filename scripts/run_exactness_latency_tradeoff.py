#!/usr/bin/env python3
"""Paired correctness/latency study on prompts with observed PLD divergence."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from run_strong_baseline_comparison import (
        BENCHMARKS, bootstrap_mean_ci, driver_version, gpu_telemetry,
        package_version, run_vanilla_timed, sha256_file, tokenize_prompt,
    )
except ModuleNotFoundError:
    from scripts.run_strong_baseline_comparison import (
        BENCHMARKS, bootstrap_mean_ci, driver_version, gpu_telemetry,
        package_version, run_vanilla_timed, sha256_file, tokenize_prompt,
    )
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter


METHODS = ("vanilla", "pld_batched", "pld_sequential", "pld_kv_replay",
           "pld_margin_guard_0.05")


def mismatch_prompts(raw_path: str) -> dict[str, set[str]]:
    selected: dict[str, set[str]] = defaultdict(set)
    for line in Path(raw_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("method") in {"pld", "pld_gate", "routed"} and row.get("exact_token_match") is False:
            selected[row["benchmark"]].add(str(row["id"]))
    return selected


def bootstrap_prompt_ci(values: list[float]) -> list[float]:
    return bootstrap_mean_ci(values, seed=62026)


def summarize(rows: list[dict]) -> dict:
    per_method: dict[str, dict] = {}
    keys = {(row["benchmark"], row["id"]) for row in rows}
    by_measurement: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = (row["benchmark"], row["id"], row["repeat"], row["order_seed"])
        by_measurement[key][row["method"]] = row
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in method_rows:
            grouped[(row["benchmark"], row["id"])].append(row)
        exactness = [float(all(row["exact_token_match"] for row in grouped.get(key, [])))
                     for key in keys if grouped.get(key)]
        prompt_ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
        prompt_decode_ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
        for run_key, run in by_measurement.items():
            if method == "vanilla":
                prompt_ratios[run_key[:2]].append(1.0)
                prompt_decode_ratios[run_key[:2]].append(1.0)
                continue
            baseline = run.get("vanilla")
            candidate = run.get(method)
            if baseline and candidate and baseline["end_to_end_tps"] > 0:
                prompt_ratios[run_key[:2]].append(
                    candidate["end_to_end_tps"] / baseline["end_to_end_tps"])
                if baseline["decode_tps"] > 0:
                    prompt_decode_ratios[run_key[:2]].append(
                        candidate["decode_tps"] / baseline["decode_tps"])
        tps_ratios = [float(np.mean(values)) for values in prompt_ratios.values()]
        decode_ratios = [float(np.mean(values)) for values in prompt_decode_ratios.values()]
        per_method[method] = {
            "n_runs": len(method_rows), "n_prompts": len(exactness),
            "exact_match_pct": 100.0 * float(np.mean(exactness)) if exactness else None,
            "end_to_end_speedup_mean": float(np.mean(tps_ratios)) if tps_ratios else None,
            "end_to_end_speedup_ci95": bootstrap_prompt_ci(tps_ratios),
            "decode_tps_speedup_mean": float(np.mean(decode_ratios)) if decode_ratios else None,
            "decode_tps_speedup_ci95": bootstrap_prompt_ci(decode_ratios),
            "bootstrap_unit": "prompt (repeats/order seeds averaged within prompt)",
        }
    return {"methods": per_method, "n_mismatch_prompts": len(keys)}


def run_method(method, model, tokenizer, manager, prompt_ids, max_new_tokens, device):
    if method == "vanilla":
        ids, prefill_ms, decode_ms, total_ms = run_vanilla_timed(
            model, tokenizer, prompt_ids, max_new_tokens, device)
        return ids, prefill_ms, decode_ms, total_ms, None
    options = {
        "canonical_verify": method == "pld_sequential",
        "canonical_kv_replay": method == "pld_kv_replay",
        "margin_fallback_threshold": 0.05 if method == "pld_margin_guard_0.05" else None,
    }
    _, metrics = self_speculative_generate(
        model, tokenizer, manager, [], prompt_ids, k=4,
        draft_mode="prompt_lookup", max_new_tokens=max_new_tokens,
        device=device, **options,
    )
    total_ms = metrics.total_time_s * 1000
    prefill_ms = metrics.prefill_time_s * 1000
    return (metrics.generated_token_ids, prefill_ms,
            max(0.0, total_ms - prefill_ms), total_ms, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Full benchmark raw JSONL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--order-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--warmup-prompts", type=int, default=1)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA driver is required")

    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=quant, device_map=args.device)
    model.eval()
    adapter = ModelAdapter(model)
    manager = LayerManager(adapter)
    model_revision = getattr(model.config, "_commit_hash", None)
    env = {
        "commit_sha": commit, "model_id": args.model, "model_revision": model_revision,
        "tokenizer_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or model_revision,
        "quantization": "NF4", "compute_dtype": "float16",
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "python": platform.python_version(), "torch_version": torch.__version__,
        "transformers_version": package_version("transformers"),
        "bitsandbytes_version": package_version("bitsandbytes"),
        "cuda_version": torch.version.cuda, "driver_version": driver_version(),
        "gpu": gpu_telemetry(args.device),
        "dataset_sha256": {name: sha256_file(path) for name, path in BENCHMARKS.items()},
    }
    (output / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    config = {**vars(args), "methods": list(METHODS),
              "decode_tokens_definition": "generated output tokens excluding the first token selected from prefill logits"}
    (output / "config.yaml").write_text(
        "\n".join(f"{key}: {json.dumps(value)}" for key, value in config.items()) + "\n",
        encoding="utf-8")

    selected = mismatch_prompts(args.raw)
    records: dict[str, list[dict]] = {}
    for bench, filename in BENCHMARKS.items():
        wanted = selected.get(bench, set())
        with open(filename, encoding="utf-8") as stream:
            records[bench] = [json.loads(line) for line in stream if line.strip()
                              and str(json.loads(line).get("id")) in wanted]

    for items in records.values():
        for item in items[:args.warmup_prompts]:
            prompt_ids = tokenize_prompt(tokenizer, item["prompt"])
            for method in METHODS:
                run_method(method, model, tokenizer, manager, prompt_ids,
                           args.max_new_tokens, args.device)

    rows: list[dict] = []
    for bench, items in records.items():
        for item in items:
            prompt_ids = tokenize_prompt(tokenizer, item["prompt"])
            for repeat in range(args.repeats):
                for seed in args.order_seeds:
                    order = list(METHODS)
                    prompt_index = next(i for i, candidate in enumerate(items)
                                        if str(candidate.get("id")) == str(item.get("id")))
                    random.Random(seed + prompt_index * 7919 + repeat).shuffle(order)
                    measured = {}
                    prompt_group = []
                    for method in order:
                        torch.cuda.reset_peak_memory_stats(args.device)
                        telemetry = gpu_telemetry(args.device)
                        ids, prefill_ms, decode_ms, total_ms, metrics = run_method(
                            method, model, tokenizer, manager, prompt_ids,
                            args.max_new_tokens, args.device)
                        output_tokens = len(ids)
                        decode_tokens = max(0, output_tokens - 1)
                        decode_seconds = decode_ms / 1000
                        row = {
                            "benchmark": bench, "id": str(item.get("id")),
                            "method": method, "repeat": repeat, "order_seed": seed,
                            "input_token_ids": prompt_ids[0].tolist(),
                            "generated_token_ids": ids, "output_tokens": output_tokens,
                            "ttft_ms": prefill_ms, "decode_tokens": decode_tokens,
                            "decode_seconds": decode_seconds,
                            "decode_tps": decode_tokens / decode_seconds if decode_seconds else 0.0,
                            "total_ms": total_ms,
                            "end_to_end_tps": output_tokens / (total_ms / 1000) if total_ms else 0.0,
                            "exact_token_match": None,
                            "pld_cycles": metrics.pld_cycles if metrics else 0,
                            "draft_tokens": metrics.total_draft_tokens if metrics else 0,
                            "accepted_tokens": metrics.total_accepted_tokens if metrics else 0,
                            "verification_cycles": metrics.num_verification_cycles if metrics else output_tokens,
                            "canonical_replay_time_s": metrics.canonical_replay_time_s if metrics else 0.0,
                            "margin_fallback_cycles": metrics.margin_fallback_cycles if metrics else 0,
                            "peak_vram_mb": torch.cuda.max_memory_allocated(args.device) / 1024**2,
                            **telemetry,
                        }
                        measured[method] = ids
                        prompt_group.append(row)
                    reference = measured["vanilla"]
                    for row in prompt_group:
                        row["exact_token_match"] = row["generated_token_ids"] == reference
                        row["first_divergence_index"] = next(
                            (i for i, pair in enumerate(zip(row["generated_token_ids"], reference))
                             if pair[0] != pair[1]),
                            min(len(row["generated_token_ids"]), len(reference))
                            if len(row["generated_token_ids"]) != len(reference) else None,
                        )
                    rows.extend(prompt_group)
                    with (output / "raw_measurements.jsonl").open("a", encoding="utf-8") as stream:
                        for row in prompt_group:
                            stream.write(json.dumps(row) + "\n")
            print(f"{bench}: completed {len(items)} mismatch prompts", flush=True)

    summary = summarize(rows)
    summary["full_run_commit_sha"] = json.loads(Path(args.raw).parents[0].joinpath("environment.json").read_text()).get("commit_sha")
    (output / "correctness_latency_tradeoff.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
