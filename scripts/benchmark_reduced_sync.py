#!/usr/bin/env python3
"""Paired CUDA benchmark of reference and conservative reduced-sync decoding."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zassd.decoding.speculative import self_speculative_generate  # noqa: E402
from zassd.models.layer_manager import LayerManager  # noqa: E402
from zassd.models.model_adapter import ModelAdapter  # noqa: E402

CONFIGS = {
    "cka_60": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 21],
    "cka_67": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16],
    "cka_90": [4, 5, 6, 7],
}


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def gpu_snapshot(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {"timestamp_unix_s": time.time()}
    try:
        import pynvml
        pynvml.nvmlInit()
        index = torch.device(device).index or 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        result.update({
            "gpu_name": pynvml.nvmlDeviceGetName(handle),
            "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
            "power_w_snapshot": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
            "utilization_pct": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
            "memory_used_mb": pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2,
        })
    except Exception as error:
        result["telemetry_error"] = str(error)
    return result


def load_prompts(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No prompt rows in {path}")
    if len(rows) > limit:
        indices = [round(i * (len(rows) - 1) / (limit - 1)) for i in range(limit)] if limit > 1 else [0]
        rows = [rows[i] for i in indices]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/benchmarks/prompts.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--config", choices=tuple(CONFIGS), default="cka_90")
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    prompts_path = args.prompts if args.prompts.is_absolute() else ROOT / args.prompts
    prompts = load_prompts(prompts_path, args.limit)
    base_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.float16,
                                      bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=quantization,
        device_map=args.device, attn_implementation="sdpa").eval()
    manager = LayerManager(ModelAdapter(model))
    config = {
        "execution_commit": base_sha, "model": args.model, "model_revision": args.revision,
        "quantization": "NF4", "compute_dtype": "float16", "attention": "sdpa",
        "cache_backend": "static", "config": args.config, "skip_indices": CONFIGS[args.config],
        "k": args.k, "max_new_tokens": args.max_new_tokens, "repeats": args.repeats,
        "prompt_file": str(prompts_path), "prompt_file_sha256": hashlib.sha256(prompts_path.read_bytes()).hexdigest(),
        "prompt_ids": [str(row.get("id", row.get("prompt_id", i))) for i, row in enumerate(prompts)],
        "seed": args.seed, "methods": ["reference", "reduced_sync_conservative"],
        "measurement": "same tokenized prompt; paired method order randomized per prompt/repeat",
        "notes": "No Nsight instrumentation; each method has one warmup before timed runs.",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    environment = {
        "commit_sha": base_sha, "python": platform.python_version(), "torch": torch.__version__,
        "transformers": version("transformers"), "bitsandbytes": version("bitsandbytes"),
        "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(args.device),
        "initial_telemetry": gpu_snapshot(args.device),
    }
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    tokenized = []
    for i, row in enumerate(prompts):
        prompt = row.get("prompt", row.get("input", row.get("text")))
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt row {i} has no string prompt/input/text field")
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
        tokenized.append((row, ids, hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()))

    # Warm both paths outside the timed paired measurements.
    warm_row, warm_ids, _ = tokenized[0]
    del warm_row
    for reduced in (False, True):
        self_speculative_generate(
            model, tokenizer, manager, CONFIGS[args.config], warm_ids, k=args.k,
            max_new_tokens=min(8, args.max_new_tokens), temperature=0.0,
            device=args.device, kv_cache_backend="static",
            kv_cache_max_capacity=int(warm_ids.shape[-1]) + args.max_new_tokens + 8,
            config_name=args.config, reduced_sync=reduced)

    rng = random.Random(args.seed)
    raw = out / "raw_measurements.jsonl"
    for repeat in range(args.repeats):
        for row, ids, input_hash in tokenized:
            order = [False, True]
            rng.shuffle(order)
            for reduced in order:
                variant = "reduced_sync_conservative" if reduced else "reference"
                before = gpu_snapshot(args.device)
                _, metrics = self_speculative_generate(
                    model, tokenizer, manager, CONFIGS[args.config], ids, k=args.k,
                    max_new_tokens=args.max_new_tokens, temperature=0.0,
                    device=args.device, kv_cache_backend="static",
                    kv_cache_max_capacity=int(ids.shape[-1]) + args.max_new_tokens + 8,
                    config_name=args.config, reduced_sync=reduced)
                measurement = asdict(metrics)
                record = {
                    "prompt_id": str(row.get("id", row.get("prompt_id", ""))),
                    "repeat": repeat, "method": variant, "input_tokens": int(ids.shape[-1]),
                    "input_token_ids_sha256": input_hash, "measurement": measurement,
                    "telemetry_before": before, "telemetry_after": gpu_snapshot(args.device),
                }
                with raw.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            print(f"completed repeat={repeat} prompt={row.get('id', row.get('prompt_id', ''))}", flush=True)


if __name__ == "__main__":
    main()
