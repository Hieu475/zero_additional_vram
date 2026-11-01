"""Compare a true one-row NF4 kernel with two-row padded and BNB singleton calls."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear as packed_two_row,
)
from zassd.certification.nf4_fixed_row_singleton_group_scale_packed import (
    nf4_fixed_row_singleton_group_scale_packed_rows_linear as packed_one_row,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    loaded = load_backend(
        "hf_bnb_nf4",
        "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "cuda:0",
        "sdpa",
    )
    model = loaded.model.eval()
    layers = {
        "q_proj": model.model.layers[0].self_attn.q_proj,
        "down_proj": model.model.layers[0].mlp.down_proj,
    }
    torch.manual_seed(53119)
    torch.cuda.manual_seed_all(53119)
    results = {}
    with torch.inference_mode():
        for name, layer in layers.items():
            x = torch.randn((1, 1, layer.in_features), device="cuda:0", dtype=torch.float16)
            methods = {
                "bnb_singleton": lambda: layer(x),
                "packed_two_row_kernel_m1": lambda: packed_two_row(
                    x, layer, physical_rows=2, block_n=16, block_k=256, num_warps=1
                ),
                "packed_one_row_kernel": lambda: packed_one_row(
                    x, layer, block_n=16, block_k=256, num_warps=1
                ),
            }
            outputs = {}
            for method, fn in methods.items():
                outputs[method] = fn().detach().clone()
                torch.cuda.synchronize()
            correctness = {
                "one_row_equals_two_row_kernel": torch.equal(
                    outputs["packed_one_row_kernel"], outputs["packed_two_row_kernel_m1"]
                ),
                "one_row_equals_bnb_singleton": torch.equal(
                    outputs["packed_one_row_kernel"], outputs["bnb_singleton"]
                ),
                "one_row_vs_two_row_max_abs": float(
                    (outputs["packed_one_row_kernel"] - outputs["packed_two_row_kernel_m1"]).abs().max()
                ),
                "one_row_vs_bnb_max_abs": float(
                    (outputs["packed_one_row_kernel"] - outputs["bnb_singleton"]).abs().max()
                ),
            }
            samples = {method: [] for method in methods}
            order = list(methods)
            for round_index in range(args.rounds):
                if round_index % 2:
                    order.reverse()
                for method in order:
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        methods[method]()
                    end.record()
                    end.synchronize()
                    samples[method].append(start.elapsed_time(end) / args.iterations)
            medians = {method: statistics.median(values) for method, values in samples.items()}
            results[name] = {
                "shape": {"in_features": int(layer.in_features), "out_features": int(layer.out_features)},
                "correctness": correctness,
                "median_cuda_ms": medians,
                "ratio_vs_packed_one_row": {
                    method: value / medians["packed_one_row_kernel"]
                    for method, value in medians.items()
                },
                "samples_ms": samples,
            }
    payload = {
        "status": "finite M=1 operator screen; the singleton kernel is an isolated B1 reference optimization prototype",
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
        "iterations_per_sample": args.iterations,
        "alternating_rounds": args.rounds,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
