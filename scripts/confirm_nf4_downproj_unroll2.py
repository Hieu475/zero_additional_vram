"""Repeated same-process A/B for the ordered unroll-2 down_proj prototype."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_unroll import (
    nf4_fixed_rows_group_scale_packed_rows_unroll_linear,
)
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    loaded = load_backend(
        "hf_bnb_nf4",
        "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "cuda:0",
        "sdpa",
        model_dtype=torch.float16,
    )
    layer = loaded.model.eval().model.layers[0].mlp.down_proj
    torch.manual_seed(90319)
    torch.cuda.manual_seed_all(90319)
    inputs = torch.randn((1, 2, layer.in_features), device="cuda:0", dtype=torch.float16)
    schedules = {
        "reference_n16_w1": lambda x: nf4_fixed_rows_group_scale_packed_rows_linear(
            x, layer, physical_rows=2, block_n=16, block_k=256, num_warps=1
        ),
        "ordered_unroll2": lambda x: nf4_fixed_rows_group_scale_packed_rows_unroll_linear(
            x,
            layer,
            physical_rows=2,
            block_n=16,
            block_k=256,
            num_warps=1,
            loop_unroll_factor=2,
        ),
    }
    results = {}
    with torch.inference_mode():
        for m in (1, 2):
            x = inputs[:, :m].contiguous()
            outputs = {}
            for name, fn in schedules.items():
                outputs[name] = fn(x).detach().clone()
                torch.cuda.synchronize()
                singleton = [fn(inputs[:, row : row + 1].contiguous()) for row in range(m)]
                results.setdefault("correctness", {})[f"{name}_m{m}_rowwise_singleton_equal"] = all(
                    torch.equal(outputs[name][:, row : row + 1], singleton[row]) for row in range(m)
                )
            results["correctness"][f"unroll2_m{m}_equals_reference"] = torch.equal(
                outputs["ordered_unroll2"], outputs["reference_n16_w1"]
            )
            samples = {name: [] for name in schedules}
            order = list(schedules)
            for round_id in range(args.rounds):
                if round_id % 2:
                    order.reverse()
                for name in order:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        schedules[name](x)
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) / args.iterations)
            medians = {name: statistics.median(values) for name, values in samples.items()}
            results[str(m)] = {
                "median_cuda_ms": medians,
                "reference_over_unroll2_latency_ratio": medians["reference_n16_w1"] / medians["ordered_unroll2"],
                "samples_ms": samples,
            }
    payload = {
        "status": "paired same-process operator confirmation; finite M=1/2, no runtime contract",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "shape": {"in_features": int(layer.in_features), "out_features": int(layer.out_features)},
        "iterations_per_sample": args.iterations,
        "alternating_rounds_per_m": args.rounds,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
