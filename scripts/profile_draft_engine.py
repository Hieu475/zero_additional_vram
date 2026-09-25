"""Micro-profiling of Draft Engine Latency on NVIDIA RTX 4050 Laptop GPU.

Decomposes the draft latency into granular, measurable hardware/software components:
1. Active Layer Execution (Transformer GEMMs + Attention over kept layers)
2. Layer Skipping Overhead (Identity function calls & context manager patching/restoration)
3. KV Update Overhead (DynamicCache.update() tensor concatenation & allocation)
4. Python Loop & Sampling Overhead (Argmax/sampling, tensor conversion, loop control)
5. CUDA Synchronization Overhead (Kernel completion fence)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from zassd.cache.kv_cache import TargetKVCache
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def profile_draft_breakdown(
    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
    config_name: str = "cka_75",
    skip_ratio: float = 0.25,
    num_warmup: int = 5,
    num_trials: int = 30,
    k_values: list[int] = [1, 2, 4],
    device: str = "cuda",
    out_dir: str = "experiments/13_draft_profiling",
) -> dict[str, Any]:
    """Execute high-precision CUDA event and host timer profiling of draft operations."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading model {model_name_or_path} for draft micro-profiling...")
    model = load_model(model_name_or_path, quantize=True, bits=4)
    tokenizer = load_tokenizer(model_name_or_path)
    model.eval()

    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)
    num_layers = adapter.num_layers

    # Calculate skipped layer indices for cka_75 (skip 25% of layers)
    num_skips = int(round(num_layers * skip_ratio))
    step = num_layers / float(num_skips)
    skip_indices = [int(i * step) for i in range(num_skips)]
    active_indices = [i for i in range(num_layers) if i not in skip_indices]

    logger.info(
        f"Model: {num_layers} layers. Config: {config_name}. "
        f"Skipping {len(skip_indices)} layers, Keeping {len(active_indices)} layers."
    )

    # Prepare prefix cache
    prompt = "Explain the mechanics of speculative decoding and memory-bandwidth bound inference."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]

    target_kv = TargetKVCache()
    with torch.no_grad():
        prefill_out = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)
    last_tok = int(prefill_out.logits[0, -1, :].argmax(dim=-1).item())

    results_by_k: dict[str, Any] = {}

    for k in k_values:
        logger.info(f"Profiling draft cycle for K={k} over {num_trials} trials...")

        # Warmup
        for _ in range(num_warmup):
            draft_cache = target_kv.fork_ephemeral_draft_kv()
            with torch.no_grad():
                with layer_mgr.skip_layers(skip_indices):
                    curr = torch.tensor([[last_tok]], device=device)
                    for _ in range(k):
                        out = model(curr, past_key_values=draft_cache, use_cache=True)
                        next_t = int(out.logits[0, -1, :].argmax(dim=-1).item())
                        curr = torch.tensor([[next_t]], device=device)
            torch.cuda.synchronize()

        # Profiling metrics accumulators
        t_patch_ms_list = []
        t_unpatch_ms_list = []
        t_total_draft_ms_list = []
        t_python_loop_ms_list = []
        t_cuda_sync_ms_list = []

        # CUDA event timing for model forwards
        forward_gpu_ms_list = []

        for _ in range(num_trials):
            draft_cache = target_kv.fork_ephemeral_draft_kv()
            curr = torch.tensor([[last_tok]], device=device)

            # 1. Measure layer patching overhead (Host CPU)
            t0 = time.perf_counter()
            layer_mgr.set_skipped_layers(skip_indices)
            t1 = time.perf_counter()
            t_patch_ms_list.append((t1 - t0) * 1000.0)

            # 2. Draft loop execution
            loop_start = time.perf_counter()
            cuda_start_event = torch.cuda.Event(enable_timing=True)
            cuda_end_event = torch.cuda.Event(enable_timing=True)

            cuda_start_event.record()
            py_overhead_accum = 0.0

            with torch.no_grad():
                for step_idx in range(k):
                    py_t0 = time.perf_counter()
                    # Model forward pass
                    out = model(curr, past_key_values=draft_cache, use_cache=True)
                    logits = out.logits[0, -1, :]
                    py_t1 = time.perf_counter()

                    # Python sampling & tensor creation overhead
                    next_t = int(logits.argmax(dim=-1).item())
                    curr = torch.tensor([[next_t]], device=device)
                    py_t2 = time.perf_counter()
                    py_overhead_accum += (py_t2 - py_t1) * 1000.0

            cuda_end_event.record()
            loop_end = time.perf_counter()

            # 3. Layer unpatching / restore overhead
            t_rest0 = time.perf_counter()
            layer_mgr.restore_layers()
            t_rest1 = time.perf_counter()
            t_unpatch_ms_list.append((t_rest1 - t_rest0) * 1000.0)

            # 4. CUDA Synchronization overhead
            sync_start = time.perf_counter()
            torch.cuda.synchronize()
            sync_end = time.perf_counter()

            forward_gpu_ms = cuda_start_event.elapsed_time(cuda_end_event)
            forward_gpu_ms_list.append(forward_gpu_ms)
            t_cuda_sync_ms_list.append((sync_end - sync_start) * 1000.0)
            t_python_loop_ms_list.append(py_overhead_accum)

            total_draft_ms = (sync_end - t0) * 1000.0
            t_total_draft_ms_list.append(total_draft_ms)

        # 5. Measure isolated KV cache update time
        kv_update_times = []
        for _ in range(num_trials):
            test_cache = DynamicCache()
            for l_idx in range(num_layers):
                test_cache.update(torch.randn(1, 4, 32, 64, device=device), torch.randn(1, 4, 32, 64, device=device), l_idx)
            torch.cuda.synchronize()

            t_kv0 = time.perf_counter()
            for step_idx in range(k):
                for l_idx in active_indices:
                    test_cache.update(torch.randn(1, 4, 1, 64, device=device), torch.randn(1, 4, 1, 64, device=device), l_idx)
            torch.cuda.synchronize()
            t_kv1 = time.perf_counter()
            kv_update_times.append((t_kv1 - t_kv0) * 1000.0)

        # Compute averages
        mean_total = float(sum(t_total_draft_ms_list) / len(t_total_draft_ms_list))
        mean_patch = float(sum(t_patch_ms_list) / len(t_patch_ms_list))
        mean_unpatch = float(sum(t_unpatch_ms_list) / len(t_unpatch_ms_list))
        mean_skip_mgmt = mean_patch + mean_unpatch
        mean_gpu_fwd = float(sum(forward_gpu_ms_list) / len(forward_gpu_ms_list))
        mean_py_loop = float(sum(t_python_loop_ms_list) / len(t_python_loop_ms_list))
        mean_sync = float(sum(t_cuda_sync_ms_list) / len(t_cuda_sync_ms_list))
        mean_kv = float(sum(kv_update_times) / len(kv_update_times))

        # Active layer transformer computation = GPU forward minus estimated KV update
        mean_active_comp = max(0.1, mean_gpu_fwd - mean_kv)

        breakdown = {
            "total_draft_latency_ms": round(mean_total, 2),
            "components": {
                "active_layer_execution_ms": round(mean_active_comp, 2),
                "kv_cache_update_ms": round(mean_kv, 2),
                "layer_skipping_management_ms": round(mean_skip_mgmt, 2),
                "python_sampling_and_control_ms": round(mean_py_loop, 2),
                "cuda_sync_and_fence_ms": round(mean_sync, 2),
            },
            "fractions_pct": {
                "active_layer_execution_pct": round(mean_active_comp / mean_total * 100.0, 1),
                "kv_cache_update_pct": round(mean_kv / mean_total * 100.0, 1),
                "layer_skipping_management_pct": round(mean_skip_mgmt / mean_total * 100.0, 1),
                "python_sampling_and_control_pct": round(mean_py_loop / mean_total * 100.0, 1),
                "cuda_sync_and_fence_pct": round(mean_sync / mean_total * 100.0, 1),
            },
        }
        results_by_k[f"K{k}"] = breakdown

    out_file = Path(out_dir) / "draft_latency_microprofiling.json"
    with open(out_file, "w") as f:
        json.dump(results_by_k, f, indent=2)
    logger.info(f"Draft micro-profiling results written to {out_file}")

    print("\n" + "=" * 78)
    print(f"DRAFT ENGINE LATENCY MICRO-PROFILING (NVIDIA RTX 4050, Config: {config_name})")
    print("=" * 78)
    for k_key, data in results_by_k.items():
        print(f"\n--- Speculation Depth: {k_key} (Total Latency: {data['total_draft_latency_ms']} ms) ---")
        print(f"{'Component':<38} | {'Time (ms)':<12} | {'Fraction (%)':<12}")
        print("-" * 68)
        comps = data["components"]
        fracs = data["fractions_pct"]
        print(f"{'Active Layer Execution (Transformer)':<38} | {comps['active_layer_execution_ms']:>10.2f} ms | {fracs['active_layer_execution_pct']:>10.1f}%")
        print(f"{'KV Cache Update (DynamicCache)':<38} | {comps['kv_cache_update_ms']:>10.2f} ms | {fracs['kv_cache_update_pct']:>10.1f}%")
        print(f"{'Layer Skip Management (Patching)':<38} | {comps['layer_skipping_management_ms']:>10.2f} ms | {fracs['layer_skipping_management_pct']:>10.1f}%")
        print(f"{'Python Sampling & Loop Control':<38} | {comps['python_sampling_and_control_ms']:>10.2f} ms | {fracs['python_sampling_and_control_pct']:>10.1f}%")
        print(f"{'CUDA Sync & Completion Fence':<38} | {comps['cuda_sync_and_fence_ms']:>10.2f} ms | {fracs['cuda_sync_and_fence_pct']:>10.1f}%")
        print("-" * 68)
    print("=" * 78)

    return results_by_k


if __name__ == "__main__":
    profile_draft_breakdown()
