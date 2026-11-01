#!/usr/bin/env python3
"""Capture a diagnostic PyTorch operator trace of the activationvec K=2/R=3 path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from benchmark_bnb_nf4_activationvec_closed_loop import activationvec_schedule_hook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/humaneval_ext114.jsonl")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated directory {output}")
    if args.max_new_tokens < 4:
        raise ValueError("max_new_tokens must be at least 4")

    prompts = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in prompts if row["id"] == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + args.max_new_tokens + 8
    decode_kwargs = {
        "k": 2, "ngram_size": 3, "max_new_tokens": args.max_new_tokens,
        "device": "cuda:0", "kv_cache_max_capacity": capacity,
        "physical_rows": 3, "attention_strategy": "bnb_rowwise",
        "quantized_linear_strategy": "fixed_rows", "trace": True,
    }

    # Warm weights, kernels, and decoder paths outside the profiler.
    with activationvec_schedule_hook(model), torch.inference_mode():
        b1_fixed_row_block_generate(model, tokenizer, prompt_ids,
                                    **{**decode_kwargs, "max_new_tokens": 8})
    torch.cuda.synchronize("cuda:0")

    output.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True, profile_memory=True,
    ) as profiler:
        with activationvec_schedule_hook(model), torch.inference_mode():
            _, metrics = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **decode_kwargs)
    torch.cuda.synchronize("cuda:0")
    trace_path = output / "trace.json"
    profiler.export_chrome_trace(str(trace_path))

    entries = []
    for event in profiler.key_averages(group_by_input_shape=True):
        entries.append({
            "key": event.key,
            "input_shapes": str(event.input_shapes),
            "calls": event.count,
            "self_cpu_time_us": float(event.self_cpu_time_total),
            "cpu_time_us": float(event.cpu_time_total),
            "self_cuda_time_us": float(getattr(
                event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))),
            "cuda_time_us": float(getattr(
                event, "device_time_total", getattr(event, "cuda_time_total", 0.0))),
            "self_cuda_memory_bytes": int(getattr(
                event, "self_device_memory_usage", getattr(event, "self_cuda_memory_usage", 0))),
        })
    top_cuda = sorted(entries, key=lambda row: row["self_cuda_time_us"], reverse=True)[:60]
    top_cpu = sorted(entries, key=lambda row: row["self_cpu_time_us"], reverse=True)[:60]
    result = {
        "schema": "zassd-activationvec-closed-loop-operator-profile-v1",
        "classification": "operator profiler; diagnostic only, not a benchmark result",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "prompt_tokens": prompt_len, "max_new_tokens": args.max_new_tokens,
        "k": 2, "ngram_size": 3, "physical_rows": 3,
        "attention_strategy": "bnb_rowwise",
        "output_tokens": len(metrics.generated_token_ids),
        "block_total_ms": metrics.total_time_s * 1000,
        "prefill_ms": metrics.prefill_time_s * 1000,
        "draft_ms": metrics.draft_time_s * 1000,
        "verify_ms": metrics.verify_time_s * 1000,
        "cycles": metrics.cycles,
        "draft_tokens": metrics.draft_tokens,
        "accepted_draft_tokens": metrics.accepted_draft_tokens,
        "target_forwards": metrics.target_forwards,
        "target_rows": metrics.target_rows,
        "peak_vram_mb": metrics.peak_vram_mb,
        "token_state_scope": "experimental B1 singleton schedule; profiler output must be checked against unprofiled benchmark artifacts",
        "runtime": {
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
            "transformers": importlib.metadata.version("transformers"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "top_self_cuda": top_cuda,
        "top_self_cpu": top_cpu,
        "trace_file": trace_path.name,
        "scope_limits": [
            "Profiler instrumentation changes timings; never use them as speedup measurements.",
            "CUDA event totals can include nested or overlapping work and must not be summed as wall time.",
            "One prompt and one process per profile; development data only.",
            "GSM8K held-out indices 300-349 not used.",
        ],
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "profile.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "prompt_id", "block_total_ms", "verify_ms", "cycles", "draft_tokens",
        "accepted_draft_tokens", "target_forwards", "target_rows", "peak_vram_mb")}, indent=2))
    print("top CUDA operators:")
    print(json.dumps(top_cuda[:20], indent=2))
    print(f"output_dir={output}")


if __name__ == "__main__":
    main()
