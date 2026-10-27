#!/usr/bin/env python3
"""Capture a short operator-level PyTorch profiler trace for fixed-row B1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-strategy", choices=("rowwise", "padded"), required=True)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--physical-rows", type=int, default=5)
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

    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].cpu()
    if prompt_ids.shape[1] + args.max_new_tokens > args.capacity:
        raise ValueError("prompt plus generation budget exceeds KV capacity")

    # Warm the same schedule before recording so one-time CUDA/kernel setup is
    # excluded from the operator trace. The warmed output is not analyzed.
    b1_fixed_row_block_generate(
        model, tokenizer, prompt_ids, k=args.k, max_new_tokens=8,
        device=args.device, kv_cache_max_capacity=args.capacity,
        physical_rows=args.physical_rows,
        attention_strategy=args.attention_strategy,
    )
    torch.cuda.synchronize(args.device)
    out.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU,
                  torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True,
    ) as prof:
        _, metrics = b1_fixed_row_block_generate(
            model, tokenizer, prompt_ids, k=args.k,
            max_new_tokens=args.max_new_tokens, device=args.device,
            kv_cache_max_capacity=args.capacity,
            physical_rows=args.physical_rows,
            attention_strategy=args.attention_strategy, trace=True,
        )
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
        "prompt_id": args.prompt_id,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "k": args.k, "physical_rows": args.physical_rows,
        "max_new_tokens": args.max_new_tokens,
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
