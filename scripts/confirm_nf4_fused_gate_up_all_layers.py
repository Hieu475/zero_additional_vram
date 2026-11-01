#!/usr/bin/env python3
"""Confirm fused gate/up output equality and latency across all Qwen layers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear,
)
from benchmark_nf4_fused_gate_up_m3 import fused_gate_up


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa",
        model_dtype=torch.float16,
    )
    layers = loaded.model.eval().model.layers
    results = []
    for layer_idx, transformer_layer in enumerate(layers):
        gate, up = transformer_layer.mlp.gate_proj, transformer_layer.mlp.up_proj
        torch.manual_seed(22000 + layer_idx)
        torch.cuda.manual_seed_all(22000 + layer_idx)
        x = torch.randn((1, 3, gate.in_features), device="cuda:0", dtype=torch.float16)
        correctness = {}
        with torch.inference_mode():
            for rows in (1, 2, 3):
                xr = x[:, :rows].contiguous()
                fused_gate, fused_up = fused_gate_up(xr, gate, up)
                torch.cuda.synchronize()
                sep_gate = nf4_fixed_rows_group_scale_packed_rows_linear(
                    xr, gate, physical_rows=3, block_n=16, block_k=256, num_warps=1
                )
                sep_up = nf4_fixed_rows_group_scale_packed_rows_linear(
                    xr, up, physical_rows=3, block_n=16, block_k=256, num_warps=1
                )
                correctness[f"m{rows}_gate_matches_separate"] = torch.equal(fused_gate, sep_gate)
                correctness[f"m{rows}_up_matches_separate"] = torch.equal(fused_up, sep_up)
                for row in range(rows):
                    one_gate, one_up = fused_gate_up(xr[:, row:row + 1].contiguous(), gate, up)
                    correctness[f"m{rows}_row{row}_gate_singleton_equal"] = torch.equal(
                        fused_gate[:, row:row + 1], one_gate
                    )
                    correctness[f"m{rows}_row{row}_up_singleton_equal"] = torch.equal(
                        fused_up[:, row:row + 1], one_up
                    )

            samples = {"separate_two_launches": [], "fused_one_launch": []}
            for round_idx in range(args.rounds):
                order = ["separate_two_launches", "fused_one_launch"]
                if (round_idx + layer_idx) % 2:
                    order.reverse()
                for schedule in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        if schedule == "separate_two_launches":
                            nf4_fixed_rows_group_scale_packed_rows_linear(
                                x, gate, physical_rows=3, block_n=16, block_k=256, num_warps=1
                            )
                            nf4_fixed_rows_group_scale_packed_rows_linear(
                                x, up, physical_rows=3, block_n=16, block_k=256, num_warps=1
                            )
                        else:
                            fused_gate_up(x, gate, up)
                    end.record()
                    end.synchronize()
                    samples[schedule].append(start.elapsed_time(end) / args.iterations)
        med = {name: statistics.median(values) for name, values in samples.items()}
        results.append({
            "layer": layer_idx,
            "in_features": int(gate.in_features),
            "out_features": int(gate.out_features),
            "all_equal": all(correctness.values()),
            "correctness": correctness,
            "median_cuda_ms": med,
            "fused_over_separate_ratio": med["fused_one_launch"] / med["separate_two_launches"],
            "samples_ms": samples,
        })
        print(
            f"layer {layer_idx:02d}: exact={all(correctness.values())}; "
            f"separate={med['separate_two_launches']:.5f}ms fused={med['fused_one_launch']:.5f}ms "
            f"ratio={med['fused_one_launch'] / med['separate_two_launches']:.3f}"
        )

    ratios = [item["fused_over_separate_ratio"] for item in results]
    geomean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
    payload = {
        "schema": "zassd-nf4-fused-gate-up-all-layers-confirmation-v1",
        "status": "finite all-layer operator confirmation; no runtime integration",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "layers": len(results),
        "measured_rows": 3,
        "iterations_per_sample": args.iterations,
        "alternating_rounds_per_layer": args.rounds,
        "exact_layers": sum(item["all_equal"] for item in results),
        "fused_over_separate_geomean": geomean,
        "fused_over_separate_min": min(ratios),
        "fused_over_separate_max": max(ratios),
        "results": results,
        "limitations": [
            "Synthetic random hidden states; no full decoder output or KV equivalence check.",
            "One RTX 4050 and finite paired CUDA-event rounds.",
            "The fused prototype is outside the pinned runtime and reference contract.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "exact_layers": payload["exact_layers"],
        "fused_over_separate_geomean": payload["fused_over_separate_geomean"],
        "fused_over_separate_min": payload["fused_over_separate_min"],
        "fused_over_separate_max": payload["fused_over_separate_max"],
    }, indent=2))


if __name__ == "__main__":
    main()
