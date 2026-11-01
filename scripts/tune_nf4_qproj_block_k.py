"""Explore smaller reduction tiles for the current shape-invariant NF4 q_proj kernel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import torch
import triton

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows import _nf4_fixed_rows_kernel, _supported_quant_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-rows", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
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
    quant_state, nested = _supported_quant_state(layer, int(layer.in_features))
    nested_absmax = quant_state.state2.absmax if nested else quant_state.absmax
    nested_code = quant_state.state2.code if nested else quant_state.code
    nested_offset = quant_state.offset if nested else quant_state.absmax
    weight = layer.weight.data
    bias = layer.bias if layer.bias is not None else weight
    n, k = int(layer.out_features), int(layer.in_features)
    base_x = torch.randn((1, args.physical_rows, k), device=args.device, dtype=torch.float16)
    dtype_code = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}[quant_state.dtype]
    configs = [(16, 32), (16, 64), (16, 128)]
    rows_out = []
    canonical_outputs = {f"n{bn}_k{bk}": None for bn, bk in configs}

    with torch.inference_mode():
        for m in (1, 3, 5):
            x = base_x[:, :m, :].contiguous()
            outputs = {}
            samples = {f"n{bn}_k{bk}": [] for bn, bk in configs}
            for bn, bk in configs:
                y = torch.empty((1, m, n), device=args.device, dtype=torch.float16)

                def launch(bn=bn, bk=bk, y=y):
                    _nf4_fixed_rows_kernel[(m, triton.cdiv(n, bn))](
                        x, weight, quant_state.absmax, quant_state.code,
                        nested_absmax, nested_code, nested_offset, bias, y,
                        n, k, int(quant_state.blocksize),
                        int(quant_state.state2.blocksize) if nested else 256,
                        nested, layer.bias is not None, dtype_code, bn, bk,
                        num_warps=4, enable_fp_fusion=False,
                    )
                for _ in range(8):
                    launch()
                torch.cuda.synchronize(args.device)
                outputs[f"n{bn}_k{bk}"] = launch

            for round_id in range(args.rounds):
                names = list(outputs)
                if round_id % 2:
                    names.reverse()
                for name in names:
                    fn = outputs[name]
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize(args.device)
                    wall_start = time.perf_counter()
                    start.record()
                    for _ in range(args.iterations):
                        fn()
                    end.record()
                    end.synchronize()
                    samples[name].append({
                        "cuda_ms": start.elapsed_time(end) / args.iterations,
                        "wall_ms": (time.perf_counter() - wall_start) * 1000 / args.iterations,
                    })

            medians = {
                name: statistics.median(sample["cuda_ms"] for sample in values)
                for name, values in samples.items()
            }
            # Each fixed tile is compared to itself across different row counts.
            output_mismatch = []
            for name, fn in outputs.items():
                y = fn.__defaults__[-1]
                if m == 1:
                    canonical_outputs[name] = y[:, :1, :].clone()
                elif not torch.equal(y[:, :1, :], canonical_outputs[name]):
                    output_mismatch.append(name)
            if output_mismatch:
                raise RuntimeError(f"tile output changed across row counts at M={m}: {output_mismatch}")
            rows_out.append({
                "m": m,
                "samples_cuda_ms": {name: [sample["cuda_ms"] for sample in values] for name, values in samples.items()},
                "median_cuda_ms": medians,
                "relative_to_n16_k128": {name: value / medians["n16_k128"] for name, value in medians.items()},
                "each_tile_bitwise_invariant_across_M": True,
            })
            print(json.dumps(rows_out[-1]), flush=True)

    result = {
        "status": "q_proj tile sweep; same kernel arithmetic with smaller K tiles",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "backend_metadata": loaded.metadata,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "compute_capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "quant_blocksize": int(quant_state.blocksize),
        "physical_rows": args.physical_rows,
        "iterations_per_sample": args.iterations,
        "rounds": args.rounds,
        "results": rows_out,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
