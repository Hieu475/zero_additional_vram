"""Tune row-independent group-scale NF4 tiles on the Qwen2.5-3B q_proj."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows import nf4_fixed_rows_linear
from zassd.certification.nf4_fixed_rows_group_scale import nf4_fixed_rows_group_scale_linear


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-rows", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", args.device, "sdpa",
        model_dtype=torch.float16,
    )
    model = loaded.model.eval()
    layer = model.model.layers[0].self_attn.q_proj
    base_x = torch.randn((1, args.physical_rows, layer.in_features), device=args.device, dtype=torch.float16)
    configs = [(8, 64), (8, 128), (8, 256), (16, 64), (16, 128), (16, 256), (32, 128)]
    results = []
    canonical = {(bn, bk): None for bn, bk in configs}

    with torch.inference_mode():
        for m in (1, 3, 5):
            x = base_x[:, :m, :].contiguous()
            funcs = {"original": lambda: nf4_fixed_rows_linear(x, layer, physical_rows=args.physical_rows)}
            funcs.update({
                f"n{bn}_k{bk}": lambda bn=bn, bk=bk: nf4_fixed_rows_group_scale_linear(
                    x, layer, physical_rows=args.physical_rows, block_n=bn, block_k=bk
                )
                for bn, bk in configs
            })
            retained = {}
            for name, fn in funcs.items():
                for _ in range(8):
                    value = fn()
                torch.cuda.synchronize(args.device)
                retained[name] = value.detach().clone()

            reference = retained["original"]
            shape_results = []
            for bn, bk in configs:
                name = f"n{bn}_k{bk}"
                value = retained[name]
                if m == 1:
                    canonical[(bn, bk)] = value[:, :1, :].clone()
                elif not torch.equal(value[:, :1, :], canonical[(bn, bk)]):
                    raise RuntimeError(f"{name} row output changed between M=1 and M={m}")
                shape_results.append({
                    "block_n": bn,
                    "block_k": bk,
                    "bitwise_equal_to_original": bool(torch.equal(value, reference)),
                    "row0_bitwise_invariant_across_M": True,
                })

            samples = {name: [] for name in funcs}
            for round_id in range(args.rounds):
                names = list(funcs)
                if round_id % 2:
                    names.reverse()
                for name in names:
                    torch.cuda.synchronize(args.device)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        funcs[name]()
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) / args.iterations)

            medians = {name: statistics.median(values) for name, values in samples.items()}
            results.append({
                "m": m,
                "median_cuda_ms": medians,
                "relative_to_original": {name: value / medians["original"] for name, value in medians.items()},
                "tile_checks": shape_results,
            })
            print(json.dumps(results[-1]), flush=True)

    summary = {
        "status": "finite q_proj tile tuning; each tile uses its own checked fixed-row schedule",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "backend_metadata": loaded.metadata,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "compute_capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "physical_rows": args.physical_rows,
        "iterations_per_sample": args.iterations,
        "rounds": args.rounds,
        "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
