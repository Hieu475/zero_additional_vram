"""Compare fixed-order split-K partial reductions for packed NF4 projections."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_splitk import (
    nf4_fixed_rows_group_scale_packed_rows_linear as splitk_linear,
)
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear as reference_linear,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa",
        model_dtype=torch.float16,
    )
    model = loaded.model.eval()
    layers = {
        "q_proj": model.model.layers[0].self_attn.q_proj,
        "down_proj": model.model.layers[0].mlp.down_proj,
    }
    torch.manual_seed(2030)
    torch.cuda.manual_seed_all(2030)
    results = {}
    with torch.inference_mode():
        for layer_name, layer in layers.items():
            inputs = torch.randn((1, 2, layer.in_features), device="cuda:0", dtype=torch.float16)
            schedules = {
                "reference_k256_w1": lambda x: reference_linear(
                    x, layer, physical_rows=2, block_n=16, block_k=256, num_warps=1
                ),
                **{
                    f"split{parts}_k256_w1": lambda x, p=parts: splitk_linear(
                        x, layer, physical_rows=2, block_n=16, block_k=256,
                        num_warps=1, split_parts=p,
                    )
                    for parts in (2, 4, 8)
                },
                "split8_k256_w1_reuse_workspace": lambda x: splitk_linear(
                    x, layer, physical_rows=2, block_n=16, block_k=256,
                    num_warps=1, split_parts=8, reuse_workspace=True,
                ),
                "split8_k256_w2": lambda x: splitk_linear(
                    x, layer, physical_rows=2, block_n=16, block_k=256,
                    num_warps=2, split_parts=8,
                ),
                "split8_k256_w4": lambda x: splitk_linear(
                    x, layer, physical_rows=2, block_n=16, block_k=256,
                    num_warps=4, split_parts=8,
                ),
            }
            correctness, latency = {}, {}
            for m in (1, 2):
                x = inputs[:, :m].contiguous()
                outputs = {}
                for name, fn in schedules.items():
                    for _ in range(8):
                        value = fn(x)
                    torch.cuda.synchronize()
                    outputs[name] = value.detach().clone()
                    singleton_rows = [fn(inputs[:, row:row + 1].contiguous()) for row in range(m)]
                    correctness[f"{name}_m{m}_rowwise_singleton_equal"] = all(
                        torch.equal(outputs[name][:, row:row + 1], singleton_rows[row])
                        for row in range(m)
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
                latency[str(m)] = {
                    "median_cuda_ms": medians,
                    "ratio_vs_reference": {
                        name: value / medians["reference_k256_w1"]
                        for name, value in medians.items()
                    },
                    "samples_ms": samples,
                }
            results[layer_name] = {
                "in_features": int(layer.in_features),
                "out_features": int(layer.out_features),
                "correctness": correctness,
                "latency_by_m": latency,
            }
    payload = {
        "status": "finite operator-only split-K experiment; split schedules define new B1 arithmetic references",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "block_n": 16,
        "block_k": 256,
        "num_warps": 1,
        "split_parts": [2, 4, 8],
        "iterations_per_sample": args.iterations,
        "rounds_per_m": args.rounds,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
