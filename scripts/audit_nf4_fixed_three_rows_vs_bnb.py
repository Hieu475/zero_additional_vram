#!/usr/bin/env python3
"""Compare the fixed-three-row Triton NF4 kernel directly with BNB Linear4bit.

This is a finite operator-only screen using identical M=3 inputs and pinned
Qwen2.5-3B NF4 weights. It records BNB M=3 rowwise invariance, Triton-vs-BNB
output equality, and alternating CUDA-event timings. It does not imply decoder
state equivalence or end-to-end speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_three_rows_group_scale_packed import (
    nf4_fixed_three_rows_group_scale_packed_linear,
)


MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    delta = (reference.float() - candidate.float()).abs()
    return {
        "bitwise_equal": torch.equal(reference, candidate),
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "shape": list(reference.shape),
    }


def time_ms(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=78139)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.iterations < 1 or args.rounds < 1:
        raise ValueError("iterations and rounds must be positive")

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model = loaded.model.eval()
    modules = {
        "q_proj": model.model.layers[0].self_attn.q_proj,
        "o_proj": model.model.layers[0].self_attn.o_proj,
        "gate_proj": model.model.layers[0].mlp.gate_proj,
        "up_proj": model.model.layers[0].mlp.up_proj,
        "down_proj": model.model.layers[0].mlp.down_proj,
    }
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    results = {}
    with torch.inference_mode():
        for name, module in modules.items():
            x = torch.randn((1, 3, module.in_features), device="cuda:0", dtype=torch.float16)
            bnb_m3 = lambda x=x, module=module: module(x)
            triton_m3 = lambda x=x, module=module: nf4_fixed_three_rows_group_scale_packed_linear(x, module)

            out_bnb = bnb_m3().detach()
            out_triton = triton_m3().detach()
            bnb_single_rows = [
                module(x[:, row:row + 1].contiguous()).detach()
                for row in range(3)
            ]
            bnb_row_checks = [
                compare(bnb_single_rows[row], out_bnb[:, row:row + 1])
                for row in range(3)
            ]
            triton_row_checks = []
            for row in range(3):
                one_row = nf4_fixed_three_rows_group_scale_packed_linear(
                    x[:, row:row + 1].contiguous(), module
                ).detach()
                triton_row_checks.append(compare(one_row, out_triton[:, row:row + 1]))

            # Compile/warm up each path before alternating timed samples.
            for _ in range(3):
                bnb_m3()
                triton_m3()
            torch.cuda.synchronize("cuda:0")
            samples = {"bnb_m3": [], "triton_m3": []}
            for round_index in range(args.rounds):
                order = ["bnb_m3", "triton_m3"]
                if round_index % 2:
                    order.reverse()
                for key in order:
                    fn = bnb_m3 if key == "bnb_m3" else triton_m3
                    samples[key].append(time_ms(fn, args.iterations))
            medians = {key: statistics.median(values) for key, values in samples.items()}
            results[name] = {
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "physical_rows": 3,
                "same_input_output_comparison": compare(out_bnb, out_triton),
                "bnb_m3_vs_bnb_m1_rows": {
                    "all_rows_bitwise_equal": all(x["bitwise_equal"] for x in bnb_row_checks),
                    "rows": bnb_row_checks,
                },
                "triton_m3_vs_its_m1_rows": {
                    "all_rows_bitwise_equal": all(x["bitwise_equal"] for x in triton_row_checks),
                    "rows": triton_row_checks,
                },
                "timing": {
                    "iterations_per_round": args.iterations,
                    "alternating_rounds": args.rounds,
                    "samples_ms": samples,
                    "median_ms": medians,
                    "triton_over_bnb_latency_ratio": medians["triton_m3"] / medians["bnb_m3"],
                },
            }
            print(json.dumps({
                "module": name,
                "triton_bnb_bitwise_equal": results[name]["same_input_output_comparison"]["bitwise_equal"],
                "max_abs_diff": results[name]["same_input_output_comparison"]["max_abs_diff"],
                "bnb_m3_rowwise_equal": results[name]["bnb_m3_vs_bnb_m1_rows"]["all_rows_bitwise_equal"],
                "triton_m3_rowwise_equal": results[name]["triton_m3_vs_its_m1_rows"]["all_rows_bitwise_equal"],
                "bnb_ms": medians["bnb_m3"],
                "triton_ms": medians["triton_m3"],
                "triton_over_bnb": medians["triton_m3"] / medians["bnb_m3"],
            }), flush=True)

    payload = {
        "schema": "zassd-nf4-fixed-three-rows-vs-bnb-operator-audit-v1",
        "classification": "operator-only finite comparison; no runtime dispatch or reference changed",
        "model": MODEL,
        "revision": REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "seed": args.seed,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_protocol": "One synthetic M=3 FP16 activation tensor per first-layer projection; identical tensor passed to BNB and Triton.",
        "limits": [
            "One synthetic activation tensor per first-layer module; real layer activations and other layers are not covered.",
            "Operator-only latency; no decoder, KV-state, or end-to-end benchmark.",
            "Finite single-GPU evidence, not a proof across inputs, prompts, runtimes, or hardware.",
            "Held-out GSM8K indices 300-349 were not used.",
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "modules": list(results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
