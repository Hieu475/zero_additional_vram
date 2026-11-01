#!/usr/bin/env python3
"""Paired, exactness-gated A/B of NF4 M=3 FP32 fusion on fixed activations."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
import triton

import benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2 as kernel_module
import tune_activationvec_nf4_m3_launch as base
from tune_activationvec_nf4_m3_fp_fusion import run_fp_fusion


def paired_ab(reference, candidate, rounds: int, repeats: int):
    samples = {"no_fusion": [], "fp_fusion": []}
    for index in range(rounds):
        order = ("no_fusion", "fp_fusion") if index % 2 == 0 else ("fp_fusion", "no_fusion")
        for name in order:
            fn = reference if name == "no_fusion" else candidate
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / repeats)
    ratios = [a / b for a, b in zip(samples["no_fusion"], samples["fp_fusion"])]
    return {
        "no_fusion_median_ms": statistics.median(samples["no_fusion"]),
        "fp_fusion_median_ms": statistics.median(samples["fp_fusion"]),
        "paired_no_fusion_over_fp_fusion": ratios,
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

    records = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    prompt = next((row for row in records if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = base.load_backend("hf_bnb_nf4", base.MODEL, base.REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    _, forced, captured = base.collect_block_inputs(model, tokenizer, prompt["prompt"], "cuda:0")

    results = {}
    with torch.inference_mode():
        for name in base.PROJECTIONS:
            module = base.projection_module(model, name)
            x_raw, bnb_reference = captured[name]["x"], captured[name]["ref"]
            state, kernel_args = base.make_kernel_args(module)
            no_fusion = lambda x=x_raw, m=module, s=state, ka=kernel_args: base.run_kernel(
                x, m, s, ka, (8, 8)
            )
            fp_fusion = lambda x=x_raw, m=module, s=state, ka=kernel_args: run_fp_fusion(
                x, m, s, ka, (8, 8)
            )
            out_base, out_fused = no_fusion(), fp_fusion()
            torch.cuda.synchronize()
            base_exact = int(torch.count_nonzero(out_base == bnb_reference).item())
            fused_exact = int(torch.count_nonzero(out_fused == bnb_reference).item())
            row = {
                "input_shape": list(x_raw.shape),
                "output_shape": list(bnb_reference.shape),
                "input_sha256": hashlib.sha256(
                    x_raw.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
                "no_fusion_exact_elements": base_exact,
                "fp_fusion_exact_elements": fused_exact,
                "total_elements": bnb_reference.numel(),
                "both_bitwise_exact_to_bnb_m3": (
                    base_exact == bnb_reference.numel() and fused_exact == bnb_reference.numel()
                ),
            }
            if row["both_bitwise_exact_to_bnb_m3"]:
                row.update(paired_ab(no_fusion, fp_fusion, args.rounds, args.repeats))
            else:
                row["timing_status"] = "not measured: one implementation failed BNB bitwise gate"
            results[name] = row

    payload = {
        "schema": "zassd-activationvec-nf4-m3-fp-fusion-ab-v1",
        "classification": "paired operator A/B; fixed activations, fixed geometry, exactness gated",
        "model": base.MODEL,
        "revision": base.REVISION,
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "forced_tokens": forced,
        "launch": {"out_tile": 8, "num_warps": 8},
        "comparison": {"reference": "enable_fp_fusion=False", "candidate": "enable_fp_fusion=True"},
        "results": results,
        "runtime": {
            "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "source_hashes": {
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fusion_driver": hashlib.sha256(Path(__file__).with_name("tune_activationvec_nf4_m3_fp_fusion.py").read_bytes()).hexdigest(),
            "base_kernel": hashlib.sha256(Path(kernel_module.__file__).read_bytes()).hexdigest(),
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
