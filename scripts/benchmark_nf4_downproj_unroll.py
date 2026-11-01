"""Screen ordered K-loop unrolling for the exact experimental down_proj path."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_unroll import (
    nf4_fixed_rows_group_scale_packed_rows_unroll_linear,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=7)
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
    torch.manual_seed(90318)
    torch.cuda.manual_seed_all(90318)
    inputs = torch.randn((1, 2, layer.in_features), device="cuda:0", dtype=torch.float16)
    schedules = {
        f"unroll_{factor}": lambda x, factor=factor: nf4_fixed_rows_group_scale_packed_rows_unroll_linear(
            x,
            layer,
            physical_rows=2,
            block_n=16,
            block_k=256,
            num_warps=1,
            loop_unroll_factor=factor,
        )
        for factor in (1, 2, 4, 8)
    }
    correctness, timing = {}, {}
    outputs_by_m = {}
    with torch.inference_mode():
        for m in (1, 2):
            x = inputs[:, :m].contiguous()
            outputs = {}
            for name, fn in schedules.items():
                value = fn(x)
                torch.cuda.synchronize()
                outputs[name] = value.detach().clone()
                singles = [fn(inputs[:, r : r + 1].contiguous()) for r in range(m)]
                correctness[f"{name}_m{m}_rowwise_singleton_equal"] = all(
                    torch.equal(outputs[name][:, r : r + 1], singles[r]) for r in range(m)
                )
            outputs_by_m[str(m)] = outputs
            for name in ("unroll_2", "unroll_4", "unroll_8"):
                correctness[f"{name}_m{m}_equals_unroll_1"] = torch.equal(
                    outputs[name], outputs["unroll_1"]
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
            timing[str(m)] = {
                "median_cuda_ms": medians,
                "ratio_vs_unroll_1": {name: value / medians["unroll_1"] for name, value in medians.items()},
                "samples_ms": samples,
            }
    payload = {
        "status": "finite down_proj-only screening; each unroll factor compiles a separate ordered arithmetic schedule and has no runtime contract",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "shape": {"in_features": int(layer.in_features), "out_features": int(layer.out_features)},
        "block_n": 16,
        "block_k": 256,
        "num_warps": 1,
        "unroll_factors": [1, 2, 4, 8],
        "iterations_per_sample": args.iterations,
        "rounds_per_m": args.rounds,
        "correctness": correctness,
        "latency_by_m": timing,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
