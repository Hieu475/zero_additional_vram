"""Paired microbenchmark of BNB NF4 and the opt-in Triton NF4 row kernel."""

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
from zassd.certification.nf4_fixed_rows_reuse import nf4_fixed_rows_reuse_linear
from zassd.certification.nf4_fixed_rows_tensorcore import nf4_fixed_rows_tensorcore_linear
from zassd.certification.nf4_fixed_rows_group_scale import nf4_fixed_rows_group_scale_linear


def time_cuda(fn, iterations: int) -> tuple[float, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000 / iterations
    return start.elapsed_time(end) / iterations, wall_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-rows", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    loaded = load_backend(
        "hf_bnb_nf4", args.model, args.revision, args.device, "sdpa",
        model_dtype=torch.float16,
    )
    model = loaded.model.eval()
    layer = model.model.layers[0].self_attn.q_proj

    all_rows = []
    base_input = torch.randn(
        (1, args.physical_rows, layer.in_features),
        device=args.device,
        dtype=torch.float16,
    )
    row_invariance = {
        "triton_nf4_reuse2": None,
        "triton_nf4_reuse4": None,
        "triton_nf4_tensorcore": None,
        "triton_nf4_group_scale": None,
    }
    with torch.inference_mode():
        for m in (1, 3, 5):
            x = base_input[:, :m, :].contiguous()
            # Materialize BNB's quantization metadata before timing either path.
            _ = layer(x)
            fns = {
                "bnb_nf4": lambda x=x: layer(x),
                "triton_nf4": lambda x=x: nf4_fixed_rows_linear(x, layer, physical_rows=args.physical_rows),
                "triton_nf4_reuse2": lambda x=x: nf4_fixed_rows_reuse_linear(
                    x, layer, physical_rows=args.physical_rows, rows_per_program=2
                ),
                "triton_nf4_reuse4": lambda x=x: nf4_fixed_rows_reuse_linear(
                    x, layer, physical_rows=args.physical_rows, rows_per_program=4
                ),
                "triton_nf4_tensorcore": lambda x=x: nf4_fixed_rows_tensorcore_linear(
                    x, layer, physical_rows=args.physical_rows, block_n=32, block_k=32
                ),
                "triton_nf4_group_scale": lambda x=x: nf4_fixed_rows_group_scale_linear(
                    x, layer, physical_rows=args.physical_rows, block_n=16, block_k=128
                ),
            }
            for fn in fns.values():
                for _ in range(12):
                    fn()
            torch.cuda.synchronize(args.device)
            current_output = fns["triton_nf4"]()
            group_scale_output = fns["triton_nf4_group_scale"]()
            group_scale_matches_original = torch.equal(current_output, group_scale_output)
            for name in row_invariance:
                output_value = fns[name]()
                if m == 1:
                    row_invariance[name] = output_value[:, :1, :].detach().clone()
                elif not torch.equal(output_value[:, :1, :], row_invariance[name]):
                    raise RuntimeError(f"{name} changed its first row at M={m}")
            samples = {name: {"cuda_ms": [], "wall_ms": []} for name in fns}
            for round_id in range(args.rounds):
                methods = list(fns.items())
                if round_id % 2:
                    methods.reverse()
                for name, fn in methods:
                    cuda_ms, wall_ms = time_cuda(fn, args.iterations)
                    samples[name]["cuda_ms"].append(cuda_ms)
                    samples[name]["wall_ms"].append(wall_ms)
            all_rows.append({
                "m": m,
                "input_features": int(layer.in_features),
                "output_features": int(layer.out_features),
                "samples": samples,
                "median_cuda_ms": {name: statistics.median(value["cuda_ms"]) for name, value in samples.items()},
                "median_wall_ms": {name: statistics.median(value["wall_ms"]) for name, value in samples.items()},
                "triton_over_bnb_cuda_ratio": (
                    statistics.median(samples["triton_nf4"]["cuda_ms"])
                    / statistics.median(samples["bnb_nf4"]["cuda_ms"])
                ),
                "reuse2_over_bnb_cuda_ratio": (
                    statistics.median(samples["triton_nf4_reuse2"]["cuda_ms"])
                    / statistics.median(samples["bnb_nf4"]["cuda_ms"])
                ),
                "reuse4_over_bnb_cuda_ratio": (
                    statistics.median(samples["triton_nf4_reuse4"]["cuda_ms"])
                    / statistics.median(samples["bnb_nf4"]["cuda_ms"])
                ),
                "tensorcore_over_bnb_cuda_ratio": (
                    statistics.median(samples["triton_nf4_tensorcore"]["cuda_ms"])
                    / statistics.median(samples["bnb_nf4"]["cuda_ms"])
                ),
                "group_scale_over_bnb_cuda_ratio": (
                    statistics.median(samples["triton_nf4_group_scale"]["cuda_ms"])
                    / statistics.median(samples["bnb_nf4"]["cuda_ms"])
                ),
                "row0_bitwise_invariant_for_experimental_variants": True,
                "group_scale_bitwise_equal_to_original_kernel": group_scale_matches_original,
            })
            print(json.dumps(all_rows[-1]), flush=True)

    summary = {
        "status": "single-GPU microbenchmark; not end-to-end performance claim",
        "model": args.model,
        "revision": args.revision,
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
        "results": all_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
