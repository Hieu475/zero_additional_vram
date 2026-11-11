"""Helpers shared by the B1 exact-quantized benchmark and audit scripts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
import time

import numpy as np
import torch

BENCHMARKS = {
    "gsm8k": "data/benchmarks/gsm8k_eval.jsonl",
    "humaneval": "data/benchmarks/humaneval_eval.jsonl",
    "cnndm": "data/benchmarks/cnndm_eval.jsonl",
}


def tokenize_prompt(tokenizer, prompt: str) -> torch.Tensor:
    """Create the single CPU token-ID tensor shared by every method in a pair."""
    return tokenizer(prompt, return_tensors="pt")["input_ids"].detach().cpu()


def bootstrap_mean_ci(values: list[float], seed: int = 2026) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(4000, len(array)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


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
