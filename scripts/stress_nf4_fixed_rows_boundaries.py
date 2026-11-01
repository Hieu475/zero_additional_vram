"""Finite synthetic NF4 linear shape-invariance audit near tile/block boundaries."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F

from zassd.certification.nf4_fixed_rows import nf4_fixed_rows_linear
from zassd.certification.nf4_fixed_rows_group_scale import (
    nf4_fixed_rows_group_scale_linear,
)


CASES = [
    (30, 15), (32, 16), (34, 17),
    (62, 15), (64, 16), (66, 17),
    (126, 15), (128, 16), (130, 17),
    (254, 15), (256, 16), (258, 17),
    (510, 15), (512, 16), (514, 17),
]
VARIANT_CASES = [
    (64, 17, False, True),   # non-nested scales plus FP16 bias
    (130, 16, True, True),   # nested scales plus FP16 bias
    (256, 15, False, False), # non-nested scales and an odd output tail
]
GROUP_SCALE_CASES = [
    (64, 7), (64, 8), (64, 9),
    (128, 15), (128, 16), (128, 17),
    (256, 7), (256, 8), (256, 9),
    (512, 15), (512, 16), (512, 17),
]
GROUP_SCALE_VARIANTS = [
    (64, 9, False, True),
    (128, 15, True, True),
    (256, 17, False, False),
]


def run_linear(strategy, input_rows, layer, physical_rows, block_k):
    if strategy == "group_scale":
        return nf4_fixed_rows_group_scale_linear(
            input_rows,
            layer,
            physical_rows=physical_rows,
            block_n=8,
            block_k=block_k,
        )
    return nf4_fixed_rows_linear(
        input_rows, layer, physical_rows=physical_rows
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-rows", type=int, default=5)
    parser.add_argument("--strategy", choices=("fixed_rows", "group_scale"), default="fixed_rows")
    parser.add_argument("--block-k", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.physical_rows < 5:
        raise ValueError("this audit requires physical_rows >= 5")

    torch.manual_seed(1701)
    torch.cuda.manual_seed_all(1701)
    all_results = []
    if args.strategy == "group_scale":
        if args.physical_rows < 5:
            raise ValueError("group-scale boundary matrix requires physical_rows >= 5")
        cases = [(k, n, True, False) for k, n in GROUP_SCALE_CASES] + GROUP_SCALE_VARIANTS
    else:
        cases = [(k, n, True, False) for k, n in CASES] + VARIANT_CASES
    for input_features, output_features, nested, has_bias in cases:
        layer = bnb.nn.Linear4bit(
            input_features,
            output_features,
            bias=has_bias,
            compute_dtype=torch.float16,
            compress_statistics=nested,
            quant_type="nf4",
        )
        layer.weight.data.copy_(torch.randn_like(layer.weight.data))
        layer = layer.to(args.device).eval()
        if has_bias:
            layer.bias.data = layer.bias.data.to(device=args.device, dtype=torch.float16)
        state = layer.weight.quant_state
        if bool(state.nested) != nested or int(state.blocksize) != 64:
            raise RuntimeError("synthetic module quantization state differs from requested variant")
        row = torch.randn((1, 1, input_features), device=args.device, dtype=torch.float16)
        multi = torch.randn((1, args.physical_rows, input_features), device=args.device, dtype=torch.float16)
        multi[:, 0:1, :] = row
        repeated = row.expand(1, args.physical_rows, input_features).contiguous()

        with torch.inference_mode():
            one = run_linear(args.strategy, row, layer, args.physical_rows, args.block_k)
            repeated_out = run_linear(args.strategy, repeated, layer, args.physical_rows, args.block_k)
            per_m = {}
            for m in range(1, args.physical_rows + 1):
                per_m[m] = run_linear(
                    args.strategy,
                    multi[:, :m, :].contiguous(),
                    layer,
                    args.physical_rows,
                    args.block_k,
                )
        torch.cuda.synchronize(args.device)

        reference_weight = bnb.functional.dequantize_4bit(
            layer.weight.data, quant_state=state
        ).to(torch.float16)
        with torch.inference_mode():
            reference = F.linear(
                row.reshape(1, input_features), reference_weight, layer.bias
            ).reshape(1, 1, output_features)

        m_rows = []
        for m, result in per_m.items():
            m_rows.append({
                "m": m,
                "first_row_bitwise_equal_to_singleton": bool(torch.equal(result[:, :1, :], one)),
                "first_row_max_abs_difference": float((result[:, :1, :].float() - one.float()).abs().max().item()),
            })
        result = {
            "input_features": input_features,
            "output_features": output_features,
            "nested_quantization": nested,
            "has_fp16_bias": has_bias,
            "strategy": args.strategy,
            "block_n": 8 if args.strategy == "group_scale" else None,
            "block_k": args.block_k if args.strategy == "group_scale" else None,
            "weight_numel_mod_quant_block": (input_features * output_features) % int(state.blocksize),
            "nested_scale_count": int(state.absmax.numel()),
            "reference_max_abs_error": float((one.float() - reference.float()).abs().max().item()),
            "reference_bitwise_equal": bool(torch.equal(one, reference)),
            "equal_repeated_rows_bitwise_equal": bool(torch.equal(repeated_out, repeated_out[:, :1, :].expand_as(repeated_out))),
            "row_count_results": m_rows,
        }
        all_results.append(result)
        print(json.dumps(result), flush=True)

    summary = {
        "status": "finite synthetic shape-boundary stress; not a universal proof",
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "compute_capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "quant_type": "nf4",
        "compute_dtype": "float16",
        "variants": (
            {"nested_scale_no_bias": 12, "non_nested_or_bias": 3}
            if args.strategy == "group_scale"
            else {"nested_scale_no_bias": 15, "non_nested_or_bias": 3}
        ),
        "quant_blocksize": 64,
        "physical_rows": args.physical_rows,
        "strategy": args.strategy,
        "block_n": 8 if args.strategy == "group_scale" else None,
        "block_k": args.block_k if args.strategy == "group_scale" else None,
        "kernel_source_sha256": sha256((Path(__file__).resolve().parents[1] / (
            "src/zassd/certification/nf4_fixed_rows_group_scale.py"
            if args.strategy == "group_scale"
            else "src/zassd/certification/nf4_fixed_rows.py"
        )).read_bytes()),
        "cases": all_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
