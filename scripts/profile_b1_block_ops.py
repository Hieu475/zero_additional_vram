#!/usr/bin/env python3
"""Capture a short operator-level PyTorch profiler trace for fixed-row B1."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
try:
    from audit_triton_b1_closed_loop import fixed_row_sdpa_decode
except ModuleNotFoundError:
    from scripts.audit_triton_b1_closed_loop import fixed_row_sdpa_decode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-strategy",
                        choices=("rowwise", "padded", "bnb_rowwise",
                                 "triton_padded", "triton_batched"), required=True)
    parser.add_argument("--method", choices=("singleton", "block"), default="block")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--physical-rows", type=int, default=5)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--quantized-linear-strategy", choices=(
        "fixed_rows", "triton_nf4", "triton_nf4_group_scale",
        "triton_nf4_group_scale_k256", "triton_nf4_group_scale_packed_k256",
        "triton_nf4_group_scale_packed_rows2_k256",
        "triton_nf4_group_scale_packed_rows2_n16_k256",
        "triton_nf4_group_scale_packed_rows2_n16_warp1_k256",
    ), default="fixed_rows")
    parser.add_argument("--model-dtype", choices=("default", "float16"), default="default")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output directory {out}")
    if args.physical_rows < args.k + 1:
        raise ValueError("physical_rows must be at least k+1")
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    row = next((item for item in records if item["id"] == args.prompt_id), None)
    if row is None:
        raise ValueError(f"prompt {args.prompt_id!r} not found in {args.benchmark}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler")

    model_dtype = torch.float16 if args.model_dtype == "float16" else None
    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, args.device, "sdpa",
                          model_dtype=model_dtype)
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].cpu()
    if prompt_ids.shape[1] + args.max_new_tokens > args.capacity:
        raise ValueError("prompt plus generation budget exceeds KV capacity")

    # Warm the same schedule before recording so one-time CUDA/kernel setup is
    # excluded from the operator trace. The warmed output is not analyzed.
    decode = b1_fixed_row_block_generate if args.method == "block" else b1_fixed_row_generate
    decode_kwargs = {
        "k": args.k, "ngram_size": args.ngram_size, "max_new_tokens": 8,
        "device": args.device, "kv_cache_max_capacity": args.capacity,
        "physical_rows": args.physical_rows,
        "attention_strategy": args.attention_strategy,
        "quantized_linear_strategy": args.quantized_linear_strategy,
    }
    patch_context = (
        fixed_row_sdpa_decode(model, args.physical_rows, 32, args.attention_strategy)
        if args.attention_strategy in {"triton_padded", "triton_batched"}
        else nullcontext()
    )
    with patch_context:
        decode(model, tokenizer, prompt_ids, **decode_kwargs)
    torch.cuda.synchronize(args.device)
    out.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True,
    ) as prof:
        measured_kwargs = dict(decode_kwargs)
        measured_kwargs["max_new_tokens"] = args.max_new_tokens
        if args.method == "block":
            measured_kwargs["trace"] = True
        patch_context = (
            fixed_row_sdpa_decode(model, args.physical_rows, 32, args.attention_strategy)
            if args.attention_strategy in {"triton_padded", "triton_batched"}
            else nullcontext()
        )
        with patch_context:
            _, metrics = decode(model, tokenizer, prompt_ids, **measured_kwargs)
    torch.cuda.synchronize(args.device)
    prof.export_chrome_trace(str(out / "trace.json"))

    entries = []
    for event in prof.key_averages(group_by_input_shape=True):
        entries.append({
            "key": event.key,
            "input_shapes": str(event.input_shapes),
            "calls": event.count,
            "self_cpu_time_us": float(event.self_cpu_time_total),
            "cpu_time_us": float(event.cpu_time_total),
            "self_cuda_time_us": float(getattr(
                event, "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0))),
            "cuda_time_us": float(getattr(
                event, "device_time_total",
                getattr(event, "cuda_time_total", 0.0))),
            "self_cpu_memory_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
            "self_cuda_memory_bytes": int(getattr(
                event, "self_device_memory_usage",
                getattr(event, "self_cuda_memory_usage", 0))),
        })
    top_cpu = sorted(entries, key=lambda x: x["self_cpu_time_us"], reverse=True)[:40]
    top_cuda = sorted(entries, key=lambda x: x["self_cuda_time_us"], reverse=True)[:40]
    report = {
        "status": "operator profile; diagnostic timing, not a benchmark result",
        "model": args.model, "revision": args.revision,
        "backend": loaded.metadata,
        "gpu": torch.cuda.get_device_name(args.device),
        "transformers": __import__("transformers").__version__,
        "attention_strategy": args.attention_strategy,
        "method": args.method,
        "quantized_linear_strategy": args.quantized_linear_strategy,
        "model_dtype_override": args.model_dtype,
        "prompt_id": args.prompt_id,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "k": args.k, "physical_rows": args.physical_rows,
        "max_new_tokens": args.max_new_tokens,
        "output_tokens": len(metrics.generated_token_ids),
        "block_total_ms": metrics.total_time_s * 1000,
        "prefill_ms": metrics.prefill_time_s * 1000,
        "draft_ms": metrics.draft_time_s * 1000,
        "verify_ms": getattr(metrics, "verify_time_s", 0.0) * 1000,
        "cycles": getattr(metrics, "cycles", 0),
        "draft_tokens": getattr(metrics, "draft_tokens", 0),
        "accepted_draft_tokens": getattr(metrics, "accepted_draft_tokens", 0),
        "target_forwards": metrics.target_forwards,
        "target_rows": metrics.target_rows,
        "capability_report": metrics.capability_report,
        "top_self_cpu": top_cpu,
        "top_self_cuda": top_cuda,
        "trace_file": "trace.json",
    }
    (out / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"top_self_cpu", "top_self_cuda", "capability_report"}}, indent=2))
    print("top cuda ops:")
    print(json.dumps(top_cuda[:15], indent=2))


if __name__ == "__main__":
    main()
