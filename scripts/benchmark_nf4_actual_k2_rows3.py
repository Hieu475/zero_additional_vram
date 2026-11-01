#!/usr/bin/env python3
"""Benchmark exact-schedule NF4 tile candidates at K2/R3's actual M=3 shape."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    loaded = load_backend(
        "hf_bnb_nf4",
        "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "cuda:0",
        "sdpa",
        model_dtype=torch.float16,
    )
    model = loaded.model.eval()
    layers = {
        "q_proj_2048x2048": model.model.layers[0].self_attn.q_proj,
        "k_proj_2048x512": model.model.layers[0].self_attn.k_proj,
        "gate_proj_2048x11008": model.model.layers[0].mlp.gate_proj,
        "down_proj_11008x2048": model.model.layers[0].mlp.down_proj,
    }
    torch.manual_seed(18731)
    torch.cuda.manual_seed_all(18731)
    candidates = [
        (8, 256, 1),
        (8, 256, 2),
        (16, 256, 1),
        (16, 256, 2),
        (16, 256, 4),
        (32, 256, 1),
        (16, 128, 1),
    ]
    report = {}
    with torch.inference_mode():
        for layer_name, layer in layers.items():
            inputs = torch.randn(
                (1, 3, layer.in_features), device="cuda:0", dtype=torch.float16
            )

            def make_fn(block_n, block_k, warps):
                return lambda x: nf4_fixed_rows_group_scale_packed_rows_linear(
                    x,
                    layer,
                    physical_rows=3,
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=warps,
                )

            names = [f"n{n}_k{k}_w{w}" for n, k, w in candidates]
            schedules = {
                name: make_fn(*cfg) for name, cfg in zip(names, candidates, strict=True)
            }
            base_name = "n16_k256_w1"
            outputs = {}
            correctness = {}
            for name, fn in schedules.items():
                outputs[name] = {}
                for m in (1, 2, 3):
                    x = inputs[:, :m].contiguous()
                    out = fn(x).detach()
                    torch.cuda.synchronize()
                    singletons = [
                        fn(inputs[:, i : i + 1].contiguous()).detach()
                        for i in range(m)
                    ]
                    correctness[f"{name}_m{m}_rowwise_singleton_equal"] = all(
                        torch.equal(out[:, i : i + 1], singletons[i])
                        for i in range(m)
                    )
                    outputs[name][m] = out
            for name in schedules:
                if name != base_name:
                    correctness[f"{name}_m3_bitwise_equals_{base_name}"] = torch.equal(
                        outputs[name][3], outputs[base_name][3]
                    )

            samples = {name: [] for name in schedules}
            order = list(schedules)
            for round_id in range(args.rounds):
                if round_id % 2:
                    order.reverse()
                for name in order:
                    fn = schedules[name]
                    x = inputs.contiguous()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        fn(x)
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) / args.iterations)
            medians = {name: statistics.median(values) for name, values in samples.items()}
            report[layer_name] = {
                "in_features": int(layer.in_features),
                "out_features": int(layer.out_features),
                "measured_rows": 3,
                "physical_rows": 3,
                "correctness": correctness,
                "median_cuda_ms_at_m3": medians,
                "ratio_vs_n16_k256_w1": {
                    name: medians[name] / medians[base_name] for name in medians
                },
                "samples_ms": samples,
            }
            fastest = min(medians, key=medians.get)
            print(
                f"{layer_name}: baseline={medians[base_name]:.5f}ms; "
                f"fastest={fastest} {medians[fastest]:.5f}ms "
                f"({medians[fastest] / medians[base_name]:.3f}x); "
                f"baseline_rowwise_exact={correctness[f'{base_name}_m3_rowwise_singleton_equal']}"
            )

    payload = {
        "schema": "zassd-nf4-k2r3-m3-operator-tile-benchmark-v1",
        "status": "finite operator-only screen at actual K2/R3 M=3; no runtime schedule changed",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "schedule_candidates": [
            {"block_n": n, "block_k": k, "num_warps": w} for n, k, w in candidates
        ],
        "iterations_per_sample": args.iterations,
        "alternating_rounds": args.rounds,
        "reference_schedule": "N=16/K=256/num_warps=1; existing experimental K2/R3 kernel",
        "interpretation_limits": [
            "Only first-layer representative linear shapes were measured; full-model exactness and speed were not tested for alternative tiles.",
            "A candidate that is shape-invariant to its own singleton can still differ from the current reference schedule; cross-schedule bitwise equality is recorded separately.",
            "CUDA-event results are finite operator measurements on one RTX 4050.",
        ],
        "results": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "layers": list(report)}, indent=2))


if __name__ == "__main__":
    main()
