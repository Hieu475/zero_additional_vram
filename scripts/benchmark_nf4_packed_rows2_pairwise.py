"""Compare interleaved and pairwise-nibble NF4 reduction dataflows."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_pairwise import (
    nf4_fixed_rows_group_scale_packed_rows_linear as pairwise_linear,
)
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear as interleaved_linear,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warps", type=int, nargs="+", default=[1])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if any(w not in (1, 2, 4) for w in args.warps):
        parser.error("--warps values must be selected from 1, 2, 4")
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
    torch.manual_seed(2027)
    torch.cuda.manual_seed_all(2027)
    results = {}
    with torch.inference_mode():
        for layer_name, layer in layers.items():
            inputs = torch.randn((1, 2, layer.in_features), device="cuda:0", dtype=torch.float16)
            schedules = {}
            for warps in args.warps:
                schedules[f"interleaved_w{warps}"] = lambda x, w=warps: interleaved_linear(
                    x, layer, physical_rows=2, block_n=16, block_k=256, num_warps=w
                )
                schedules[f"pairwise_w{warps}"] = lambda x, w=warps: pairwise_linear(
                    x, layer, physical_rows=2, block_n=16, block_k=256, num_warps=w
                )
            correctness = {}
            latency = {}
            differences = {}
            for m in (1, 2):
                x = inputs[:, :m].contiguous()
                outputs = {}
                for name, fn in schedules.items():
                    for _ in range(8):
                        value = fn(x)
                    torch.cuda.synchronize()
                    outputs[name] = value.detach().clone()
                    singleton_rows = [fn(inputs[:, i:i + 1].contiguous()) for i in range(m)]
                    correctness[f"{name}_m{m}_rowwise_singleton_equal"] = all(
                        torch.equal(outputs[name][:, i:i + 1], singleton_rows[i])
                        for i in range(m)
                    )
                differences[str(m)] = {}
                for warps in args.warps:
                    pairwise_name, interleaved_name = f"pairwise_w{warps}", f"interleaved_w{warps}"
                    delta = outputs[pairwise_name].float() - outputs[interleaved_name].float()
                    differences[str(m)][str(warps)] = {
                        "bitwise_equal_to_interleaved": bool(torch.equal(
                            outputs[pairwise_name], outputs[interleaved_name]
                        )),
                        "max_abs_difference": float(delta.abs().max().item()),
                    }
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
                latency[str(m)] = {"median_cuda_ms": medians, "samples_ms": samples}
                for warps in args.warps:
                    latency[str(m)][f"pairwise_w{warps}_over_interleaved"] = (
                        medians[f"pairwise_w{warps}"] / medians[f"interleaved_w{warps}"]
                    )
            results[layer_name] = {
                "in_features": int(layer.in_features),
                "out_features": int(layer.out_features),
                "correctness": correctness,
                "cross_schedule_difference": differences,
                "latency_by_m": latency,
            }
    payload = {
        "status": "finite operator-only dataflow experiment; pairwise reduction is a separate B1 arithmetic schedule",
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
        "num_warps_sweep": args.warps,
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
