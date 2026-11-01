#!/usr/bin/env python3
"""Audit the vectorized fused NF4 M=3 operator across every Qwen2.5-3B layer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
from pathlib import Path

import torch
import triton

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2 import (
    MODEL,
    REVISION,
    _fused_nf4_m3_kernel,
    original_bnb_forward,
)


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BENCH_LAYERS = (0, 18, 35)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--paired-rounds", type=int, default=8)
    ap.add_argument("--repeats-per-round", type=int, default=4)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((item for item in records if str(item["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    capture: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor]] = {}
    hooks = []
    modules = {}
    for layer_idx, layer in enumerate(model.model.layers):
        for proj in PROJECTIONS:
            module = getattr(layer.self_attn if proj.endswith("_proj") and proj in {"q_proj", "k_proj", "v_proj", "o_proj"} else layer.mlp, proj)
            key = (layer_idx, proj)
            modules[key] = module

            def pre_hook(_module, inputs, key=key):
                tensor = inputs[0]
                if tensor.ndim == 3 and tensor.shape[1] == 3:
                    capture.setdefault(key, [None, None])[0] = tensor.detach().reshape(-1, tensor.shape[-1]).clone()

            def post_hook(_module, _inputs, output, key=key):
                if key in capture:
                    capture[key][1] = output.detach().reshape(-1, output.shape[-1]).clone()

            hooks.append(module.register_forward_pre_hook(pre_hook))
            hooks.append(module.register_forward_hook(post_hook))

    cache = TargetKVCache(backend="static", max_capacity=int(ids.shape[-1]) + 8)
    with torch.inference_mode(), original_bnb_forward(model):
        logits = model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
        forced = []
        for _ in range(4):
            token = logits.argmax(dim=-1)
            forced.append(int(token.item()))
            logits = model(token, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
        block_cache = TargetKVCache(backend="static", max_capacity=int(ids.shape[-1]) + 8)
        model(ids, past_key_values=block_cache.cache, use_cache=True)
        model(torch.tensor([forced[:3]], device="cuda:0", dtype=torch.long), past_key_values=block_cache.cache, use_cache=True)
    for hook in hooks:
        hook.remove()

    expected = len(model.model.layers) * len(PROJECTIONS)
    if len(capture) != expected or any(x is None or ref is None for x, ref in capture.values()):
        missing = sorted(set(modules) - set(capture))
        raise RuntimeError(f"Captured {len(capture)}/{expected} M=3 linears; missing={missing[:12]}")

    rows = []
    prepared = {}
    for key, module in modules.items():
        x_raw, ref = capture[key]
        x = x_raw.to(module.compute_dtype).contiguous()
        state = module.weight.quant_state
        n, k = int(module.out_features), int(module.in_features)
        if state.quant_type != "nf4" or not state.nested or module.compute_dtype != torch.float16 or k % 64 or int(state.blocksize) != 64:
            raise RuntimeError(f"Unsupported {key}: K={k}, blocksize={state.blocksize}")
        packed = module.weight.data.reshape(-1).contiguous().view(torch.uint32)
        bias = module.bias.to(torch.float16).contiguous() if module.bias is not None else torch.empty((1,), device="cuda:0", dtype=torch.float16)
        kernel_args = (packed, state.absmax.contiguous(), state.state2.absmax.contiguous(), state.state2.code.contiguous(), state.offset.float().contiguous(), state.code.contiguous(), bias)
        output = torch.empty((3, n), device="cuda:0", dtype=torch.float16)
        grid = (triton.cdiv(n, 8),)

        def launch(x=x, kernel_args=kernel_args, output=output, k=k, n=n, grid=grid, module=module, state=state, x_raw=x_raw):
            _fused_nf4_m3_kernel[grid](x, *kernel_args, output, k, n, int(state.blocksize), module.bias is not None,
                                        OUT_TILE=8, num_warps=8, enable_fp_fusion=False)
            return output.to(x_raw.dtype)

        candidate = launch()
        exact = int(torch.count_nonzero(candidate == ref).item())
        max_diff = float((candidate.float() - ref.float()).abs().max().item())
        rows.append({"layer": key[0], "projection": key[1], "input_shape": list(x.shape), "output_shape": list(ref.shape),
                     "exact_elements": exact, "total_elements": ref.numel(), "bitwise_exact": exact == ref.numel(), "max_abs_diff": max_diff})
        prepared[key] = (module, x_raw, launch)

    torch.cuda.synchronize()
    timing_rows = []
    for layer_idx in BENCH_LAYERS:
        if layer_idx >= len(model.model.layers):
            continue
        for proj in PROJECTIONS:
            key = (layer_idx, proj)
            module, x_raw, launch = prepared[key]
            for _ in range(3):
                launch(); module(x_raw)
            torch.cuda.synchronize()
            times = {"triton": [], "bnb": []}
            ratios = []
            for round_idx in range(args.paired_rounds):
                order = ("triton", "bnb") if round_idx % 2 == 0 else ("bnb", "triton")
                per_round = {}
                for method in order:
                    fn = launch if method == "triton" else lambda module=module, x_raw=x_raw: module(x_raw)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.repeats_per_round):
                        fn()
                    end.record(); end.synchronize()
                    per_round[method] = start.elapsed_time(end) / args.repeats_per_round
                    times[method].append(per_round[method])
                ratios.append(per_round["bnb"] / per_round["triton"])
            timing_rows.append({"layer": layer_idx, "projection": proj,
                                "paired_triton_median_ms": statistics.median(times["triton"]),
                                "paired_bnb_median_ms": statistics.median(times["bnb"]),
                                "paired_median_speedup": statistics.median(ratios),
                                "paired_geomean_speedup": math.exp(sum(math.log(r) for r in ratios) / len(ratios)),
                                "paired_ratios": ratios})

    script_path = Path(__file__).resolve()
    result = {
        "schema": "zassd-bnb-fused-nf4-activationvec-all-layers-v1",
        "classification": "development-only single-operator all-layer audit; no whole-model state proof or decoder integration",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "forced_tokens": forced[:3], "layer_count": len(model.model.layers), "projection_names": list(PROJECTIONS),
        "captured_operator_count": len(rows), "exact_operator_count": sum(r["bitwise_exact"] for r in rows),
        "exact_elements": sum(r["exact_elements"] for r in rows), "total_elements": sum(r["total_elements"] for r in rows),
        "all_outputs_bitwise_exact": all(r["bitwise_exact"] for r in rows),
        "operator_rows": rows, "paired_timing_layers": list(BENCH_LAYERS), "paired_timing_rows": timing_rows,
        "paired_rounds": args.paired_rounds, "repeats_per_round": args.repeats_per_round,
        "source": str(script_path), "source_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "kernel_source": "scripts/benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2.py",
        "runtime": {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
                    "bitsandbytes": importlib.metadata.version("bitsandbytes"), "gpu": torch.cuda.get_device_name(0)},
        "scope_limits": ["M=3 output comparison against native BNB on teacher-forced rows only.",
                         "No cache/KV/token identity claim and no full decoder integration.",
                         "One prompt, one GPU, one runtime/toolchain.",
                         "GSM8K held-out indices 300-349 excluded."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"captured_operator_count": len(rows), "exact_operator_count": result["exact_operator_count"],
                      "exact_elements": result["exact_elements"], "total_elements": result["total_elements"],
                      "all_outputs_bitwise_exact": result["all_outputs_bitwise_exact"],
                      "paired_timing_rows": timing_rows, "output": str(args.output)}, indent=2))
    if not result["all_outputs_bitwise_exact"]:
        raise RuntimeError("At least one M=3 linear output differs from native BNB")


if __name__ == "__main__":
    main()
