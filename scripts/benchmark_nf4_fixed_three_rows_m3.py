#!/usr/bin/env python3
"""Screen a three-row packed NF4 kernel against the existing pairwise M=3 path."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
import triton

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear as pairwise_linear,
)
from zassd.certification.nf4_fixed_three_rows_group_scale_packed import (
    nf4_fixed_three_rows_group_scale_packed_linear as three_rows_linear,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa",
        model_dtype=torch.float16,
    )
    model = loaded.model.eval()
    layers = {
        "q_proj_2048x2048": model.model.layers[0].self_attn.q_proj,
        "k_proj_2048x512": model.model.layers[0].self_attn.k_proj,
        "gate_proj_2048x11008": model.model.layers[0].mlp.gate_proj,
        "down_proj_11008x2048": model.model.layers[0].mlp.down_proj,
    }
    torch.manual_seed(93751)
    torch.cuda.manual_seed_all(93751)
    report = {}
    with torch.inference_mode():
        for layer_name, layer in layers.items():
            x = torch.randn((1, 3, layer.in_features), device="cuda:0", dtype=torch.float16)
            candidates = {
                "pairwise_n16_k256_w1": lambda z: pairwise_linear(
                    z, layer, physical_rows=3, block_n=16, block_k=256, num_warps=1
                ),
                **{
                    f"three_rows_n16_k256_w{warps}": lambda z, warps=warps: three_rows_linear(
                        z, layer, block_n=16, block_k=256, num_warps=warps
                    ) for warps in (1, 2, 4)
                },
            }
            outputs = {}
            correctness = {}
            valid = {}
            rejected = {}
            for name, fn in candidates.items():
                try:
                    for _ in range(3):
                        out = fn(x)
                    torch.cuda.synchronize()
                except triton.runtime.errors.OutOfResources as exc:
                    rejected[name] = str(exc)
                    continue
                valid[name] = fn
                outputs[name] = out.detach()
                rows_equal = []
                for row in range(3):
                    singleton = fn(x[:, row:row + 1].contiguous())
                    rows_equal.append(torch.equal(out[:, row:row + 1], singleton))
                correctness[name] = {
                    "m3_output_rowwise_equal_to_own_singleton": rows_equal,
                    "all_m3_rows_equal_to_own_singleton": all(rows_equal),
                    "output_sha256": __import__("hashlib").sha256(
                        out.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest(),
                }
            base_name = "pairwise_n16_k256_w1"
            for name in outputs:
                correctness[name]["bitwise_equal_to_pairwise_reference"] = torch.equal(
                    outputs[name], outputs[base_name]
                )

            timing_samples = {name: [] for name in valid}
            order = list(valid)
            for round_id in range(args.rounds):
                order = list(valid)
                if round_id % 2:
                    order.reverse()
                for name in order:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        valid[name](x)
                    end.record()
                    end.synchronize()
                    timing_samples[name].append(start.elapsed_time(end) / args.iterations)
            medians = {name: statistics.median(samples) for name, samples in timing_samples.items()}
            baseline = medians[base_name]
            report[layer_name] = {
                "in_features": int(layer.in_features),
                "out_features": int(layer.out_features),
                "physical_rows": 3,
                "correctness": correctness,
                "resource_rejections": rejected,
                "median_cuda_ms": medians,
                "ratio_vs_pairwise": {name: value / baseline for name, value in medians.items()},
                "samples_ms": timing_samples,
            }
            fastest = min(medians, key=medians.get)
            print(json.dumps({
                "layer": layer_name,
                "pairwise_ms": baseline,
                "fastest": fastest,
                "fastest_ms": medians[fastest],
                "ratio": medians[fastest] / baseline,
                "exact_to_pairwise": correctness[fastest]["bitwise_equal_to_pairwise_reference"],
                "rowwise": correctness[fastest]["all_m3_rows_equal_to_own_singleton"],
                "rejected": rejected,
            }), flush=True)

    payload = {
        "schema": "zassd-nf4-fixed-three-rows-m3-screen-v1",
        "status": "operator-only diagnostic; no runtime dispatch or reference changed",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "schedule": "three output rows per CTA group, preserving per-row dequantization and reduction sequence",
        "iterations_per_sample": args.iterations,
        "alternating_rounds": args.rounds,
        "limits": [
            "Synthetic M=3 activations and four first-layer matrices only.",
            "Exactness to the existing pairwise schedule and rowwise singleton invariance are recorded separately.",
            "No decoder, KV-state, or end-to-end latency check was run.",
            "Finite single-GPU evidence only; held-out GSM8K indices 300-349 are excluded.",
        ],
        "results": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "layers": list(report)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
