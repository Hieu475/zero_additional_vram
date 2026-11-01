#!/usr/bin/env python3
"""Screen pair-broadcast reuse for duplicated nested-NF4 scale metadata loads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

import torch
import triton

import benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2 as base
import tune_activationvec_nf4_m3_launch as capture


ROOT = Path(__file__).resolve().parents[1]
PROJECTIONS = ("q_proj", "gate_proj", "down_proj")
OLD_SCALE_BLOCK = """        scale_q = tl.load(absmax8_ptr + scale_idx, mask=valid, other=0).to(tl.int32)
        scale_code = tl.load(state2code_ptr + scale_q)
        scale_state = tl.load(state2absmax_ptr + (scale_idx >> 8), mask=valid, other=0)
        scale_f = tl.fma(scale_code, scale_state, offset)
        scale_h = scale_f.to(tl.float16)"""
NEW_SCALE_BLOCK = """        even_lane = (lanes & 1) == 0
        scale_valid = valid & even_lane
        scale_q_even = tl.load(absmax8_ptr + scale_idx, mask=scale_valid, other=0).to(tl.int32)
        pair_src = tl.broadcast_to(lanes & 30, (1, OUT_TILE, 32))
        scale_q = tl.gather(scale_q_even, pair_src, axis=2)
        scale_code_even = tl.load(state2code_ptr + scale_q_even, mask=scale_valid, other=0).to(tl.float32)
        scale_code = tl.gather(scale_code_even, pair_src, axis=2)
        scale_state_even = tl.load(state2absmax_ptr + (scale_idx >> 8), mask=scale_valid, other=0).to(tl.float32)
        scale_state = tl.gather(scale_state_even, pair_src, axis=2)
        scale_f = tl.fma(scale_code, scale_state, offset)
        scale_h = scale_f.to(tl.float16)"""


def build_kernel():
    kernel_source = base._fused_nf4_m3_kernel.src
    if kernel_source.count(OLD_SCALE_BLOCK) != 1:
        raise RuntimeError("expected exactly one canonical scale block to replace")
    kernel_source = kernel_source.replace(OLD_SCALE_BLOCK, NEW_SCALE_BLOCK)
    kernel_source = kernel_source.replace(
        "def _fused_nf4_m3_kernel(", "def _fused_nf4_m3_scale_reuse_kernel(", 1
    )
    generated = (
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\n" + base._ldcs_v4.src + "\n\n"
        + "@triton.jit\n" + kernel_source + "\n"
    )
    path = Path(__file__).with_name("_nf4_activationvec_scale_gather_v2_kernel.py")
    if path.exists() and path.read_text() != generated:
        raise RuntimeError(f"generated kernel differs from existing source: {path}")
    if not path.exists():
        path.write_text(generated)
    spec = importlib.util.spec_from_file_location("_nf4_activationvec_scale_reuse_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load generated scale-reuse kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fused_nf4_m3_scale_reuse_kernel, generated, path


def launch(kernel, x_raw, module, state, kernel_args):
    n, k = int(module.out_features), int(module.in_features)
    x = x_raw.to(module.compute_dtype).contiguous()
    output = torch.empty((3, n), device=x.device, dtype=torch.float16)
    kernel[(triton.cdiv(n, 8),)](
        x, *kernel_args, output, k, n, int(state.blocksize), module.bias is not None,
        OUT_TILE=8, num_warps=8, enable_fp_fusion=False,
    )
    return output.to(x_raw.dtype)


def paired_latency(reference, candidate, rounds=12, repeats=4):
    samples = {"default": [], "scale_reuse": []}
    for index in range(rounds):
        order = ("default", "scale_reuse") if index % 2 == 0 else ("scale_reuse", "default")
        for name in order:
            fn = reference if name == "default" else candidate
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / repeats)
    ratios = [a / b for a, b in zip(samples["default"], samples["scale_reuse"])]
    return {
        "default_median_ms": statistics.median(samples["default"]),
        "scale_reuse_median_ms": statistics.median(samples["scale_reuse"]),
        "scale_reuse_speedup": statistics.median(ratios),
        "paired_speed_ratios": ratios,
        "timing_rounds": rounds,
        "repeats_per_sample": repeats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", action="append", nargs=2, metavar=("BENCHMARK", "PROMPT_ID"), required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidate_kernel, generated_source, generated_path = build_kernel()

    loaded = capture.load_backend("hf_bnb_nf4", capture.MODEL, capture.REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    cases = []
    for benchmark_name, prompt_id in args.case:
        benchmark = ROOT / benchmark_name
        records = [json.loads(line) for line in benchmark.read_text().splitlines() if line.strip()]
        prompt = next((row for row in records if str(row["id"]) == prompt_id), None)
        if prompt is None:
            raise KeyError(f"{prompt_id} not found in {benchmark_name}")
        _, forced, activations = capture.collect_block_inputs(model, tokenizer, prompt["prompt"], "cuda:0")
        projection_rows = {}
        for projection in PROJECTIONS:
            module = capture.projection_module(model, projection)
            state, kernel_args = capture.make_kernel_args(module)
            x, reference_output = activations[projection]["x"], activations[projection]["ref"]
            default = lambda x=x, module=module, state=state, kernel_args=kernel_args: launch(
                base._fused_nf4_m3_kernel, x, module, state, kernel_args
            )
            candidate = lambda x=x, module=module, state=state, kernel_args=kernel_args: launch(
                candidate_kernel, x, module, state, kernel_args
            )
            default_output, candidate_output = default(), candidate()
            torch.cuda.synchronize("cuda:0")
            default_exact = int(torch.count_nonzero(default_output == reference_output).item())
            candidate_exact = int(torch.count_nonzero(candidate_output == reference_output).item())
            row = {
                "input_shape": list(x.shape),
                "output_shape": list(reference_output.shape),
                "default_exact_elements": default_exact,
                "candidate_exact_elements": candidate_exact,
                "total_elements": reference_output.numel(),
                "default_bitwise_equal_to_bnb_m3": default_exact == reference_output.numel(),
                "candidate_bitwise_equal_to_bnb_m3": candidate_exact == reference_output.numel(),
                "candidate_max_abs_diff": float((candidate_output.float() - reference_output.float()).abs().max().item()),
            }
            if row["default_bitwise_equal_to_bnb_m3"] and row["candidate_bitwise_equal_to_bnb_m3"]:
                for _ in range(3):
                    default(); candidate()
                torch.cuda.synchronize("cuda:0")
                row.update(paired_latency(default, candidate))
            else:
                row["timing_status"] = "skipped: bitwise output gate failed"
            projection_rows[projection] = row
        cases.append({
            "prompt_id": prompt_id,
            "benchmark": benchmark_name,
            "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
            "forced_tokens": forced,
            "projections": projection_rows,
        })

    payload = {
        "schema": "zassd-activationvec-nf4-m3-scale-reuse-screen-v1",
        "classification": "operator-level pair-broadcast experiment; not integrated",
        "model": capture.MODEL,
        "revision": capture.REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "launch": {"out_tile": 8, "num_warps": 8},
        "candidate": "load nested-NF4 scale metadata on even lanes, then gather to its paired lane",
        "cases": cases,
        "generated_source": str(generated_path.relative_to(ROOT)),
        "generated_source_sha256": hashlib.sha256(generated_source.encode()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_kernel_sha256": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        "limitations": ["Operator outputs only; no full-model token/KV state-machine audit.",
                        "Two development prompts and one RTX 4050 toolchain.",
                        "GSM8K held-out indices 300-349 excluded."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    compact = [{"prompt_id": c["prompt_id"], "projections": {
        p: {"exact": v["candidate_bitwise_equal_to_bnb_m3"],
            "speedup": v.get("scale_reuse_speedup")}
        for p, v in c["projections"].items()}
    } for c in cases]
    print(json.dumps({"output": str(args.output), "cases": compact}, indent=2), flush=True)


if __name__ == "__main__":
    main()
