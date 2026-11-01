"""Synthetic K/N and quantization-state stress for the experimental N16 kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import bitsandbytes as bnb
import torch
import torch.nn.functional as F

from zassd.certification.nf4_fixed_rows_group_scale_packed_rows import (
    nf4_fixed_rows_group_scale_packed_rows_linear,
)


# Cover K below/equal/above the 256 reduction tile and N tails around 16/32.
CASES = [
    (64, 15, True, False),
    (64, 16, False, True),
    (64, 17, True, True),
    (128, 15, False, False),
    (128, 16, True, False),
    (128, 17, False, True),
    (256, 15, True, False),
    (256, 16, False, True),
    (256, 17, True, True),
    (512, 31, True, False),
    (512, 32, False, True),
    (512, 33, True, True),
    (1024, 15, False, False),
    (1024, 16, True, True),
    (1024, 17, False, True),
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(20261028)
    torch.cuda.manual_seed_all(20261028)
    cases = []
    for input_features, output_features, nested, has_bias in CASES:
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
            raise RuntimeError("synthetic module quantization state differs from requested case")

        rows = torch.randn((1, 2, input_features), device=args.device, dtype=torch.float16)
        one0 = rows[:, :1, :].contiguous()
        one1 = rows[:, 1:2, :].contiguous()
        repeated = one0.expand(1, 2, input_features).contiguous()
        with torch.inference_mode():
            out0 = nf4_fixed_rows_group_scale_packed_rows_linear(
                one0, layer, physical_rows=2, block_n=16, block_k=256
            )
            out1 = nf4_fixed_rows_group_scale_packed_rows_linear(
                one1, layer, physical_rows=2, block_n=16, block_k=256
            )
            pair = nf4_fixed_rows_group_scale_packed_rows_linear(
                rows, layer, physical_rows=2, block_n=16, block_k=256
            )
            repeated_out = nf4_fixed_rows_group_scale_packed_rows_linear(
                repeated, layer, physical_rows=2, block_n=16, block_k=256
            )
            weight = bnb.functional.dequantize_4bit(
                layer.weight.data, quant_state=state
            ).to(torch.float16)
            reference0 = F.linear(one0.reshape(1, input_features), weight, layer.bias).reshape(1, 1, output_features)
        torch.cuda.synchronize(args.device)

        row0_equal = torch.equal(pair[:, :1, :], out0)
        row1_equal = torch.equal(pair[:, 1:2, :], out1)
        repeated_equal = torch.equal(
            repeated_out, repeated_out[:, :1, :].expand_as(repeated_out)
        )
        reference_error = float((out0.float() - reference0.float()).abs().max().item())
        result = {
            "input_features": input_features,
            "output_features": output_features,
            "nested_quantization": nested,
            "has_fp16_bias": has_bias,
            "block_n": 16,
            "block_k": 256,
            "row0_bitwise_equal_to_singleton": row0_equal,
            "row1_bitwise_equal_to_singleton": row1_equal,
            "repeated_rows_bitwise_equal": repeated_equal,
            "reference_max_abs_error_vs_bnb_dequant_linear": reference_error,
            "reference_bitwise_equal_vs_bnb_dequant_linear": torch.equal(out0, reference0),
        }
        cases.append(result)
        print(json.dumps(result), flush=True)

    summary = {
        "status": "finite synthetic shape/metadata stress; not a universal proof",
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "compute_capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "strategy": "triton_nf4_group_scale_packed_rows2_n16_k256",
        "physical_rows": 2,
        "kernel_source_sha256": {
            "kernel": sha256_file(
                Path(__file__).resolve().parents[1]
                / "src/zassd/certification/nf4_fixed_rows_group_scale_packed_rows.py"
            )
        },
        "case_count": len(cases),
        "all_rowwise_checks_pass": all(
            case["row0_bitwise_equal_to_singleton"]
            and case["row1_bitwise_equal_to_singleton"]
            and case["repeated_rows_bitwise_equal"]
            for case in cases
        ),
        "cases": cases,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "case_count": summary["case_count"],
        "all_rowwise_checks_pass": summary["all_rowwise_checks_pass"],
        "output": str(path),
    }))


if __name__ == "__main__":
    main()
