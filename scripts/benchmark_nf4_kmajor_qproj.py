"""Compare K-major grouped-scale K256 with the current layout on Qwen q_proj."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale import nf4_fixed_rows_group_scale_linear
from zassd.certification.nf4_fixed_rows_group_scale_kmajor import (
    nf4_fixed_rows_group_scale_kmajor_linear,
)
from zassd.certification.nf4_fixed_rows_group_scale_packed import (
    nf4_fixed_rows_group_scale_packed_linear,
)
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows import (
    nf4_fixed_rows_group_scale_packed_rows_linear,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-rows", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--m-values", default="1,3,5",
                        help="Comma-separated row counts to probe; each must be <= physical-rows")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    m_values = [int(x) for x in args.m_values.split(",") if x.strip()]
    if not m_values or any(m < 1 or m > args.physical_rows for m in m_values):
        raise ValueError("m-values must be nonempty and within 1..physical-rows")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", args.device, "sdpa",
        model_dtype=torch.float16,
    )
    layer = loaded.model.eval().model.layers[0].self_attn.q_proj
    torch.manual_seed(4819)
    torch.cuda.manual_seed_all(4819)
    x_all = torch.randn(
        (1, args.physical_rows, layer.in_features),
        device=args.device,
        dtype=torch.float16,
    )
    methods = {
        "group_scale_n8_k256": lambda x: nf4_fixed_rows_group_scale_linear(
            x, layer, physical_rows=args.physical_rows, block_n=8, block_k=256
        ),
        "group_scale_kmajor_n8_k256": lambda x: nf4_fixed_rows_group_scale_kmajor_linear(
            x, layer, physical_rows=args.physical_rows, block_n=8, block_k=256
        ),
        "group_scale_packed_n8_k256": lambda x: nf4_fixed_rows_group_scale_packed_linear(
            x, layer, physical_rows=args.physical_rows, block_n=8, block_k=256
        ),
        "group_scale_packed_rows2_n8_k256": lambda x: nf4_fixed_rows_group_scale_packed_rows_linear(
            x, layer, physical_rows=args.physical_rows, block_n=8, block_k=256
        ),
        "group_scale_packed_rows2_n16_k256": lambda x: nf4_fixed_rows_group_scale_packed_rows_linear(
            x, layer, physical_rows=args.physical_rows, block_n=16, block_k=256
        ),
        "group_scale_packed_rows2_n32_k256": lambda x: nf4_fixed_rows_group_scale_packed_rows_linear(
            x, layer, physical_rows=args.physical_rows, block_n=32, block_k=256
        ),
    }
    canonical = {name: None for name in methods}
    results = []
    with torch.inference_mode():
        for m in m_values:
            x = x_all[:, :m, :].contiguous()
            outputs = {}
            for name, fn in methods.items():
                for _ in range(8):
                    value = fn(x)
                torch.cuda.synchronize(args.device)
                outputs[name] = value.detach().clone()
                if m == 1:
                    canonical[name] = outputs[name][:, :1, :].clone()
                elif not torch.equal(outputs[name][:, :1, :], canonical[name]):
                    raise RuntimeError(f"{name} row 0 changed across M={m}")
            comparisons = {}
            reference = outputs["group_scale_n8_k256"]
            for name, value in outputs.items():
                comparisons[name] = {
                    "bitwise_equal_to_current": bool(torch.equal(value, reference)),
                    "max_abs_difference_to_current": float((value.float() - reference.float()).abs().max().item()),
                }
            rowwise_checks = {}
            if m == 1:
                for name, fn in methods.items():
                    single = fn(x_all[:, :1, :].contiguous())
                    rowwise_checks[name] = bool(torch.equal(outputs[name][:, :1, :], single))
            elif m == 2:
                for name, fn in methods.items():
                    first = fn(x_all[:, :1, :].contiguous())
                    second = fn(x_all[:, 1:2, :].contiguous())
                    rowwise_checks[name] = bool(
                        torch.equal(outputs[name][:, :1, :], first)
                        and torch.equal(outputs[name][:, 1:2, :], second)
                    )

            samples = {name: [] for name in methods}
            for round_id in range(args.rounds):
                names = list(methods)
                if round_id % 2:
                    names.reverse()
                for name in names:
                    torch.cuda.synchronize(args.device)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        methods[name](x)
                    end.record()
                    end.synchronize()
                    samples[name].append(start.elapsed_time(end) / args.iterations)

            medians = {name: statistics.median(values) for name, values in samples.items()}
            result = {
                "m": m,
                "median_cuda_ms": medians,
                "packed_rows2_over_packed_latency_ratio": (
                    medians["group_scale_packed_rows2_n8_k256"] /
                    medians["group_scale_packed_n8_k256"]
                ),
                "packed_rows2_tile_over_packed_latency_ratio": {
                    name: medians[name] / medians["group_scale_packed_n8_k256"]
                    for name in (
                        "group_scale_packed_rows2_n8_k256",
                        "group_scale_packed_rows2_n16_k256",
                        "group_scale_packed_rows2_n32_k256",
                    )
                },
                "kmajor_over_current_latency_ratio": (
                    medians["group_scale_kmajor_n8_k256"] / medians["group_scale_n8_k256"]
                ),
                "comparisons_to_current_schedule": comparisons,
                "packed_rows2_bitwise_equal_to_packed": bool(torch.equal(
                    outputs["group_scale_packed_rows2_n8_k256"],
                    outputs["group_scale_packed_n8_k256"],
                )),
                "packed_rows2_tile_bitwise_equal_to_packed": {
                    name: bool(torch.equal(outputs[name], outputs["group_scale_packed_n8_k256"]))
                    for name in (
                        "group_scale_packed_rows2_n8_k256",
                        "group_scale_packed_rows2_n16_k256",
                        "group_scale_packed_rows2_n32_k256",
                    )
                },
                "all_rows_bitwise_equal_to_sequential_singletons": rowwise_checks,
                "row0_bitwise_equal_to_own_singleton": {name: True for name in methods},
            }
            results.append(result)
            print(json.dumps(result), flush=True)

    root = Path(__file__).resolve().parents[1]
    payload = {
        "status": "experimental q_proj layout comparison; finite microbenchmark",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "compute_capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "physical_rows": args.physical_rows,
        "iterations_per_sample": args.iterations,
        "rounds": args.rounds,
        "kernel_source_sha256": {
            "current": sha256_file(root / "src/zassd/certification/nf4_fixed_rows_group_scale.py"),
            "kmajor": sha256_file(root / "src/zassd/certification/nf4_fixed_rows_group_scale_kmajor.py"),
            "packed": sha256_file(root / "src/zassd/certification/nf4_fixed_rows_group_scale_packed.py"),
            "packed_rows2": sha256_file(root / "src/zassd/certification/nf4_fixed_rows_group_scale_packed_rows.py"),
        },
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
