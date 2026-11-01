"""Screen a smaller output tile for the experimental exact down_proj kernel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows import _supported_quant_state
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    _packed_two_rows_kernel,
    nf4_fixed_rows_group_scale_packed_rows_linear,
)


def direct_n4(input_rows: torch.Tensor, module) -> torch.Tensor:
    """Launch N=4 through the existing kernel, bypassing its public tile guard."""
    if input_rows.ndim != 3 or input_rows.shape[0] != 1:
        raise ValueError("expected rank-3 batch-one inputs")
    rows, input_features = map(int, input_rows.shape[1:])
    quant_state, nested = _supported_quant_state(module, input_features)
    quant_block = int(quant_state.blocksize)
    block_k, block_n = 256, 4
    if input_features % quant_block or block_k % quant_block:
        raise ValueError("input and tile K must align to NF4 quantization blocks")
    bias = module.bias
    if bias is not None and (bias.device != input_rows.device or bias.dtype != torch.float16):
        raise ValueError("bias must be CUDA FP16")
    if nested:
        nested_absmax, nested_code, nested_offset = (
            quant_state.state2.absmax,
            quant_state.state2.code,
            quant_state.offset,
        )
    else:
        nested_absmax, nested_code, nested_offset = (
            quant_state.absmax,
            quant_state.code,
            quant_state.absmax,
        )
    x = input_rows.to(torch.float16).contiguous()
    y = torch.empty((1, rows, int(quant_state.shape[0])), device=x.device, dtype=torch.float16)
    dtype_code = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[quant_state.dtype]
    _packed_two_rows_kernel[(1, math.ceil(y.shape[-1] / block_n))](
        x,
        module.weight.data,
        quant_state.absmax,
        quant_state.code,
        nested_absmax,
        nested_code,
        nested_offset,
        bias if bias is not None else module.weight.data,
        y,
        rows,
        int(quant_state.shape[0]),
        input_features,
        quant_block,
        int(quant_state.state2.blocksize) if nested else 256,
        nested,
        bias is not None,
        dtype_code,
        block_n,
        block_k,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return y.to(input_rows.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
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
    torch.manual_seed(90317)
    torch.cuda.manual_seed_all(90317)
    inputs = torch.randn((1, 2, layer.in_features), device="cuda:0", dtype=torch.float16)
    schedules = {
        "n4_w1": lambda x: direct_n4(x, layer),
        "n8_w1": lambda x: nf4_fixed_rows_group_scale_packed_rows_linear(
            x, layer, physical_rows=2, block_n=8, block_k=256, num_warps=1
        ),
        "n16_w1": lambda x: nf4_fixed_rows_group_scale_packed_rows_linear(
            x, layer, physical_rows=2, block_n=16, block_k=256, num_warps=1
        ),
        "n16_w4": lambda x: nf4_fixed_rows_group_scale_packed_rows_linear(
            x, layer, physical_rows=2, block_n=16, block_k=256, num_warps=4
        ),
    }
    correctness, timing = {}, {}
    with torch.inference_mode():
        for m in (1, 2):
            x = inputs[:, :m].contiguous()
            outputs = {}
            for name, fn in schedules.items():
                for _ in range(8):
                    value = fn(x)
                torch.cuda.synchronize()
                outputs[name] = value.detach().clone()
                singles = [fn(inputs[:, r : r + 1].contiguous()) for r in range(m)]
                correctness[f"{name}_m{m}_rowwise_singleton_equal"] = all(
                    torch.equal(outputs[name][:, r : r + 1], singles[r]) for r in range(m)
                )
            for name in ("n4_w1", "n8_w1", "n16_w4"):
                correctness[f"{name}_m{m}_equals_n16_w1"] = torch.equal(
                    outputs[name], outputs["n16_w1"]
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
                "ratio_vs_n16_w1": {name: value / medians["n16_w1"] for name, value in medians.items()},
                "samples_ms": samples,
            }
    payload = {
        "status": "finite down_proj-only screen; N=4 directly launches the existing experimental kernel and has no runtime contract",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "shape": {"in_features": int(layer.in_features), "out_features": int(layer.out_features)},
        "block_k": 256,
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
