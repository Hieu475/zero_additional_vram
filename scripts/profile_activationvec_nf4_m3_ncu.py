#!/usr/bin/env python3
"""Prepare one exact actual-activation M=3 NF4 launch for Nsight Compute."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
import triton

from zassd.backends.base import load_backend
import benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2 as base
from tune_activationvec_nf4_m3_launch import (
    MODEL,
    REVISION,
    collect_block_inputs,
    make_kernel_args,
    projection_module,
)


ROOT = Path(__file__).resolve().parents[1]


def launch_kernel(kernel, x_raw, module, state, kernel_args, config):
    out_tile, num_warps = config
    n, k = int(module.out_features), int(module.in_features)
    x = x_raw.to(module.compute_dtype).contiguous()
    output = torch.empty((3, n), device=x.device, dtype=torch.float16)
    kernel[(triton.cdiv(n, out_tile),)](
        x, *kernel_args, output, k, n, int(state.blocksize), module.bias is not None,
        OUT_TILE=out_tile, num_warps=num_warps, enable_fp_fusion=False,
    )
    return output.to(x_raw.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--projection", choices=("q_proj", "gate_proj", "down_proj"), default="down_proj")
    ap.add_argument("--scale-reuse-candidate", action="store_true",
                    help="profile the exact but slower logical-gather scale reuse prototype")
    ap.add_argument("--tl-sum-candidate", action="store_true",
                    help="profile the exact layout-aware tl.sum reduction candidate")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    records = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    prompt = next((row for row in records if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    _, _, captures = collect_block_inputs(model, tokenizer, prompt["prompt"], "cuda:0")
    module = projection_module(model, args.projection)
    state, kernel_args = make_kernel_args(module)
    x_raw, reference = captures[args.projection]["x"], captures[args.projection]["ref"]
    config = (8, 8)
    kernel = base._fused_nf4_m3_kernel
    kernel_name = "_fused_nf4_m3_kernel"
    kernel_source_bytes = Path(base.__file__).read_bytes()
    if args.scale_reuse_candidate and args.tl_sum_candidate:
        raise ValueError("select at most one candidate")
    if args.scale_reuse_candidate:
        candidate_path = Path(__file__).with_name("_nf4_activationvec_scale_gather_v2_kernel.py")
        spec = importlib.util.spec_from_file_location("_nf4_activationvec_scale_gather_v2_kernel", candidate_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load scale reuse candidate")
        candidate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate_module)
        kernel = candidate_module._fused_nf4_m3_scale_reuse_kernel
        kernel_name = "_fused_nf4_m3_scale_reuse_kernel"
        kernel_source_bytes = candidate_path.read_bytes()
    elif args.tl_sum_candidate:
        import tune_activationvec_nf4_m3_tl_sum as tl_sum
        kernel, generated = tl_sum.build_kernel()
        kernel_name = "_fused_nf4_m3_tl_sum_kernel"
        kernel_source_bytes = generated.encode()

    # Two launches compile and warm the exact kernel. With the matching Nsight
    # filter, --launch-skip 2 --launch-count 1 profiles only the third launch.
    candidate = None
    for _ in range(3):
        candidate = launch_kernel(kernel, x_raw, module, state, kernel_args, config)
    torch.cuda.synchronize("cuda:0")
    exact_elements = int(torch.count_nonzero(candidate == reference).item())
    payload = {
        "schema": "zassd-activationvec-nf4-m3-ncu-input-v1",
        "classification": "single actual-activation M=3 kernel profile; diagnostic only",
        "model": MODEL,
        "revision": REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "projection": args.projection,
        "kernel_name": kernel_name,
        "input_shape": list(x_raw.shape),
        "output_shape": list(reference.shape),
        "launch": {"out_tile": config[0], "num_warps": config[1], "warmup_launches": 2,
                   "profiled_matching_launch_index": 3},
        "bitwise_equal_to_native_bnb_m3": exact_elements == reference.numel(),
        "exact_elements": exact_elements,
        "total_elements": reference.numel(),
        "max_abs_diff": float((candidate.float() - reference.float()).abs().max().item()),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_source_sha256": hashlib.sha256(kernel_source_bytes).hexdigest(),
        "heldout_used": False,
        "limits": ["One prompt, one layer-0 projection and one RTX 4050 toolchain.",
                   "Nsight instrumentation is not a latency benchmark.",
                   "GSM8K held-out indices 300-349 excluded."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["bitwise_equal_to_native_bnb_m3"]:
        raise RuntimeError("profile candidate differs from native BNB M=3 output")


if __name__ == "__main__":
    main()
