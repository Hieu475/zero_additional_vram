#!/usr/bin/env python3
"""Compare streaming vs cache-all activation loads for exact NF4 M=3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
import triton

import benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2 as base
import tune_activationvec_nf4_m3_launch as capture


def build_ca_kernel():
    helper_source = base._ldcs_v4.src
    kernel_source = base._fused_nf4_m3_kernel.src
    helper_source = helper_source.replace("def _ldcs_v4(", "def _ldca_v4(")
    helper_source = helper_source.replace("ld.global.cs.v4.u32", "ld.global.ca.v4.u32")
    kernel_source = kernel_source.replace("def _fused_nf4_m3_kernel(", "def _fused_nf4_m3_ca_kernel(")
    kernel_source = kernel_source.replace("_ldcs_v4(", "_ldca_v4(")
    generated_source = (
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\n" + helper_source + "\n\n@triton.jit\n" + kernel_source + "\n"
    )
    variant_path = Path(__file__).with_name("_nf4_activationvec_ca_kernel.py")
    if variant_path.exists() and variant_path.read_text() != generated_source:
        raise RuntimeError(f"generated CA kernel source differs from existing file: {variant_path}")
    if not variant_path.exists():
        variant_path.write_text(generated_source)
    spec = importlib.util.spec_from_file_location("_nf4_activationvec_ca_kernel", variant_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load generated CA kernel module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fused_nf4_m3_ca_kernel, generated_source


def launch(kernel, x_raw, module, state, kernel_args, out_tile=8, num_warps=8):
    n, k = int(module.out_features), int(module.in_features)
    x = x_raw.to(module.compute_dtype).contiguous()
    output = torch.empty((3, n), device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(n, out_tile),)
    kernel[grid](x, *kernel_args, output, k, n, int(state.blocksize),
                 module.bias is not None, OUT_TILE=out_tile, num_warps=num_warps,
                 enable_fp_fusion=False)
    return output.to(x_raw.dtype)


def paired_latency(streaming, cached, rounds, repeats):
    samples = {"streaming_cs": [], "cached_ca": []}
    for i in range(rounds):
        order = ("streaming_cs", "cached_ca") if i % 2 == 0 else ("cached_ca", "streaming_cs")
        for mode in order:
            fn = streaming if mode == "streaming_cs" else cached
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record(); end.synchronize()
            samples[mode].append(start.elapsed_time(end) / repeats)
    ratios = [cs / ca for cs, ca in zip(samples["streaming_cs"], samples["cached_ca"])]
    return {"streaming_median_ms": statistics.median(samples["streaming_cs"]),
            "cached_median_ms": statistics.median(samples["cached_ca"]),
            "cached_over_streaming_speedup": statistics.median(ratios),
            "paired_ratios": ratios, "paired_rounds": rounds,
            "repeats_per_round": repeats}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ca_kernel, generated_source = build_ca_kernel()
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in records if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = capture.load_backend("hf_bnb_nf4", capture.MODEL, capture.REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    _, forced, activations = capture.collect_block_inputs(model, tokenizer, prompt["prompt"], "cuda:0")
    results = {}
    with torch.inference_mode():
        for name in capture.PROJECTIONS:
            module = capture.projection_module(model, name)
            x, ref = activations[name]["x"], activations[name]["ref"]
            state, kernel_args = capture.make_kernel_args(module)
            cs = lambda: launch(base._fused_nf4_m3_kernel, x, module, state, kernel_args)
            ca = lambda: launch(ca_kernel, x, module, state, kernel_args)
            cs_out, ca_out = cs(), ca()
            torch.cuda.synchronize()
            cs_exact = int(torch.count_nonzero(cs_out == ref).item())
            ca_exact = int(torch.count_nonzero(ca_out == ref).item())
            row = {"input_shape": list(x.shape), "output_shape": list(ref.shape),
                   "streaming_exact_elements": cs_exact, "cached_exact_elements": ca_exact,
                   "total_elements": ref.numel(),
                   "streaming_matches_bnb_m3": cs_exact == ref.numel(),
                   "cached_matches_bnb_m3": ca_exact == ref.numel(),
                   "cached_max_abs_diff": float((ca_out.float() - ref.float()).abs().max().item())}
            if row["streaming_matches_bnb_m3"] and row["cached_matches_bnb_m3"]:
                for _ in range(4):
                    cs(); ca()
                torch.cuda.synchronize()
                row.update(paired_latency(cs, ca, args.rounds, args.repeats))
            else:
                row["timing_status"] = "not measured: bitwise operator gate failed"
            results[name] = row
    payload = {
        "schema": "zassd-activationvec-nf4-cache-policy-v1",
        "classification": "paired operator screen of activation load cache policy; no end-to-end integration claim",
        "model": capture.MODEL, "revision": capture.REVISION, "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "forced_tokens": forced, "launch": {"out_tile": 8, "num_warps": 8},
        "cache_policy": {"streaming": "ld.global.cs.v4.u32", "candidate": "ld.global.ca.v4.u32",
                         "weight_load_policy": "unchanged"},
        "results": results,
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "triton": triton.__version__,
                    "bitsandbytes": importlib.metadata.version("bitsandbytes"),
                    "gpu": torch.cuda.get_device_name(0)},
        "source_hashes": {
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "base_kernel": hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
            "generated_ca_kernel": hashlib.sha256(generated_source.encode()).hexdigest(),
        },
        "limitations": ["layer 0 projections on one development prompt", "not a B1/full-model exactness proof",
                        "GSM8K held-out indices 300-349 excluded"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompt_id": args.prompt_id, "results": results,
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
