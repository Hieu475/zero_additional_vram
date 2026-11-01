#!/usr/bin/env python3
"""Paired exactness-gated A/B of tl.sum versus the baseline gather reduction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
import triton

import benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2 as kernel_base
import tune_activationvec_nf4_m3_launch as baseline
import tune_activationvec_nf4_m3_tl_sum as tl_sum


def paired_ab(reference, candidate, rounds, repeats):
    samples = {"gather": [], "tl_sum": []}
    for index in range(rounds):
        order = ("gather", "tl_sum") if index % 2 == 0 else ("tl_sum", "gather")
        for name in order:
            fn = reference if name == "gather" else candidate
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / repeats)
    ratios = [a / b for a, b in zip(samples["gather"], samples["tl_sum"])]
    return {
        "gather_median_ms": statistics.median(samples["gather"]),
        "tl_sum_median_ms": statistics.median(samples["tl_sum"]),
        "paired_gather_over_tl_sum": ratios,
        "median_speedup": statistics.median(ratios),
        "paired_rounds": rounds,
        "repeats_per_round": repeats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--rounds", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    kernel, generated = tl_sum.build_kernel()
    tl_sum.KERNEL = kernel
    records = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    prompt = next((row for row in records if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = baseline.load_backend("hf_bnb_nf4", baseline.MODEL, baseline.REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    _, forced, captured = baseline.collect_block_inputs(model, tokenizer, prompt["prompt"], "cuda:0")

    results = {}
    with torch.inference_mode():
        for name in baseline.PROJECTIONS:
            module = baseline.projection_module(model, name)
            x_raw, bnb_reference = captured[name]["x"], captured[name]["ref"]
            state, kernel_args = baseline.make_kernel_args(module)
            gather = lambda x=x_raw, m=module, s=state, ka=kernel_args: baseline.run_kernel(
                x, m, s, ka, (8, 8)
            )
            candidate = lambda x=x_raw, m=module, s=state, ka=kernel_args: tl_sum.run_kernel(
                x, m, s, ka, (8, 8)
            )
            out_gather, out_sum = gather(), candidate()
            torch.cuda.synchronize()
            n_gather = int(torch.count_nonzero(out_gather == bnb_reference).item())
            n_sum = int(torch.count_nonzero(out_sum == bnb_reference).item())
            row = {
                "input_shape": list(x_raw.shape),
                "output_shape": list(bnb_reference.shape),
                "input_sha256": hashlib.sha256(
                    x_raw.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "gather_exact_elements": n_gather,
                "tl_sum_exact_elements": n_sum,
                "total_elements": bnb_reference.numel(),
                "both_bitwise_exact_to_bnb_m3": (
                    n_gather == bnb_reference.numel() and n_sum == bnb_reference.numel()
                ),
            }
            if row["both_bitwise_exact_to_bnb_m3"]:
                row.update(paired_ab(gather, candidate, args.rounds, args.repeats))
            else:
                row["timing_status"] = "not measured: one reduction failed BNB bitwise gate"
            results[name] = row

    payload = {
        "schema": "zassd-activationvec-nf4-m3-tl-sum-ab-v1",
        "classification": "paired operator A/B; same activations and launch geometry; exactness gated",
        "model": baseline.MODEL,
        "revision": baseline.REVISION,
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "forced_tokens": forced,
        "launch": {"out_tile": 8, "num_warps": 8},
        "comparison": {"baseline": "five guarded tl.gather stages", "candidate": "tl.sum(axis=2, keep_dims=True)"},
        "results": results,
        "runtime": {
            "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "source_hashes": {
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_kernel": hashlib.sha256(Path(kernel_base.__file__).read_bytes()).hexdigest(),
            "tl_sum_driver": hashlib.sha256(Path(tl_sum.__file__).read_bytes()).hexdigest(),
            "generated_tl_sum_kernel": hashlib.sha256(generated.encode()).hexdigest(),
        },
        "limitations": [
            "one development prompt and layer-0 q/gate/down projections",
            "does not establish whole-model B1 equivalence or end-to-end speedup",
            "GSM8K held-out indices 300-349 excluded",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompt_id": args.prompt_id, "results": results,
                      "runtime": payload["runtime"], "output": str(args.output)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
