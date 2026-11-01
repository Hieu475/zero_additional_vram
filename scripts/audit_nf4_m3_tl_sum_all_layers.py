#!/usr/bin/env python3
"""Audit tl.sum NF4 M=3 outputs against BNB across every Qwen2.5-3B layer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

import tune_activationvec_nf4_m3_launch as base
import tune_activationvec_nf4_m3_tl_sum as tl_sum
from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def module_for(layer, projection):
    return getattr(layer.self_attn, projection) if projection.endswith("_proj") and projection in {
        "q_proj", "k_proj", "v_proj", "o_proj"
    } else getattr(layer.mlp, projection)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, type=Path)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in args.benchmark.read_text().splitlines() if line.strip()]
    prompt = next((row for row in records if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)

    loaded = load_backend("hf_bnb_nf4", base.MODEL, base.REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    input_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    capacity = int(input_ids.shape[-1]) + 8
    forced_tokens = []
    with torch.inference_mode(), base.original_bnb_forward(model):
        reference_cache = TargetKVCache(backend="static", max_capacity=capacity)
        logits = model(input_ids, past_key_values=reference_cache.cache, use_cache=True).logits[:, -1:, :]
        for _ in range(3):
            token = logits.argmax(dim=-1)
            forced_tokens.append(int(token.item()))
            logits = model(token, past_key_values=reference_cache.cache, use_cache=True).logits[:, -1:, :]

        block_cache = TargetKVCache(backend="static", max_capacity=capacity)
        model(input_ids, past_key_values=block_cache.cache, use_cache=True)
        captures = {}
        handles = []
        for layer_index, layer in enumerate(model.model.layers):
            for projection in PROJECTIONS:
                module = module_for(layer, projection)
                key = f"layers.{layer_index}.{projection}"

                def pre_hook(_module, hook_inputs, capture_key=key):
                    captures.setdefault(capture_key, {})["x"] = hook_inputs[0].detach().reshape(
                        -1, hook_inputs[0].shape[-1]
                    ).clone()

                def post_hook(_module, _hook_inputs, output, capture_key=key):
                    captures.setdefault(capture_key, {})["ref"] = output.detach().reshape(
                        -1, output.shape[-1]
                    ).clone()

                handles.append(module.register_forward_pre_hook(pre_hook))
                handles.append(module.register_forward_hook(post_hook))
        block_input = torch.tensor([forced_tokens], device="cuda:0", dtype=torch.long)
        model(block_input, past_key_values=block_cache.cache, use_cache=True)
        for handle in handles:
            handle.remove()

    kernel, generated = tl_sum.build_kernel()
    tl_sum.KERNEL = kernel
    results = []
    with torch.inference_mode():
        for layer_index, layer in enumerate(model.model.layers):
            for projection in PROJECTIONS:
                key = f"layers.{layer_index}.{projection}"
                module = module_for(layer, projection)
                x, reference = captures[key]["x"], captures[key]["ref"]
                state, kernel_args = base.make_kernel_args(module)
                candidate = tl_sum.run_kernel(x, module, state, kernel_args, (8, 8))
                torch.cuda.synchronize()
                exact_elements = int(torch.count_nonzero(candidate == reference).item())
                fixed_row_counts = []
                fixed_row_max_diff = 0.0
                for row_index in range(3):
                    singleton_padded = F.pad(x[row_index:row_index + 1], (0, 0, 0, 2)).contiguous()
                    singleton_output = tl_sum.run_kernel(
                        singleton_padded, module, state, kernel_args, (8, 8)
                    )[0]
                    block_row = candidate[row_index]
                    fixed_row_counts.append(
                        int(torch.count_nonzero(block_row == singleton_output).item())
                    )
                    fixed_row_max_diff = max(
                        fixed_row_max_diff,
                        float((block_row.float() - singleton_output.float()).abs().max().item()),
                    )
                results.append({
                    "layer": layer_index,
                    "projection": projection,
                    "input_shape": list(x.shape),
                    "output_shape": list(reference.shape),
                    "exact_elements": exact_elements,
                    "total_elements": reference.numel(),
                    "bitwise_exact": exact_elements == reference.numel(),
                    "max_abs_diff": float((candidate.float() - reference.float()).abs().max().item()),
                    "fixed_row_exact_elements_per_row": fixed_row_counts,
                    "fixed_row_total_elements": reference.numel(),
                    "fixed_row_bitwise_exact": all(
                        count == reference.shape[-1] for count in fixed_row_counts
                    ),
                    "fixed_row_max_abs_diff": fixed_row_max_diff,
                })

    total_elements = sum(item["total_elements"] for item in results)
    exact_elements = sum(item["exact_elements"] for item in results)
    payload = {
        "schema": "zassd-nf4-m3-tl-sum-all-layers-audit-v1",
        "classification": "one-prompt per-operator comparison against BNB M=3 and same-kernel fixed-row singleton; no decoder integration claim",
        "model": base.MODEL,
        "revision": base.REVISION,
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "forced_tokens": forced_tokens,
        "layers": len(model.model.layers),
        "projections": list(PROJECTIONS),
        "operators_checked": len(results),
        "exact_operators": sum(int(item["bitwise_exact"]) for item in results),
        "total_operators": len(results),
        "exact_elements": exact_elements,
        "total_elements": total_elements,
        "all_operators_bitwise_exact": all(item["bitwise_exact"] for item in results),
        "fixed_row_reference_contract": (
            "For each singleton activation row, zero-pad to three physical rows, run the same tl.sum M=3 kernel, and compare its row 0 output against that row of the M=3 block call."
        ),
        "fixed_row_exact_operators": sum(int(item["fixed_row_bitwise_exact"]) for item in results),
        "fixed_row_total_operators": len(results),
        "fixed_row_exact_elements": sum(sum(item["fixed_row_exact_elements_per_row"]) for item in results),
        "fixed_row_total_elements": sum(item["fixed_row_total_elements"] for item in results),
        "all_operators_fixed_row_exact": all(item["fixed_row_bitwise_exact"] for item in results),
        "results": results,
        "runtime": {
            "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
            "bitsandbytes": importlib.metadata.version("bitsandbytes"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "source_hashes": {
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tl_sum_driver": hashlib.sha256(Path(tl_sum.__file__).read_bytes()).hexdigest(),
            "generated_kernel": hashlib.sha256(generated.encode()).hexdigest(),
        },
        "limitations": [
            "one development prompt and one GPU",
            "tests projection outputs in a BNB M=3 block, not B1 singleton equality, KV state or sequence exactness",
            "GSM8K held-out indices 300-349 excluded",
        ],
    }
    payload["output"] = str(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "prompt_id", "forced_tokens", "layers", "operators_checked", "exact_operators",
        "total_operators", "exact_elements", "total_elements", "all_operators_bitwise_exact",
        "fixed_row_exact_operators", "fixed_row_total_operators", "fixed_row_exact_elements",
        "fixed_row_total_elements", "all_operators_fixed_row_exact",
        "runtime", "output"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
