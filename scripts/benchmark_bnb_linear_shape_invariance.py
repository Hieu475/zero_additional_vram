#!/usr/bin/env python3
"""Measure BNB linear M-shape sensitivity on captured Qwen decode activations.

Captures real M=1 inputs from the pinned original BNB decode, repeats each input
row to M=2/3/5, then compares the real row's output with the M=1 result and
times each direct module call. This is an operator diagnostic, not a model
speed benchmark or a proof of all-input equivalence.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache


DEFAULT_MODULES = (
    "model.layers.0.self_attn.q_proj",
    "model.layers.0.self_attn.k_proj",
    "model.layers.0.self_attn.v_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.mlp.up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.33.self_attn.q_proj",
    "model.layers.33.mlp.gate_proj",
    "model.layers.33.mlp.down_proj",
    "model.layers.35.self_attn.q_proj",
    "model.layers.35.mlp.gate_proj",
    "model.layers.35.mlp.down_proj",
    "lm_head",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/cnndm_eval.jsonl")
    parser.add_argument("--prompt-id", default="cnndm_000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--modules", nargs="*", default=None)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)

    prompts = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines()
               if line.strip()]
    row = next((item for item in prompts if str(item["id"]) == args.prompt_id), None)
    if row is None:
        raise ValueError(f"prompt {args.prompt_id!r} not found")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rounds < 1 or args.iterations < 1:
        raise ValueError("rounds and iterations must be positive")

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    loaded = load_backend("hf_bnb_nf4", model_name, revision, args.device, "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    modules = dict(model.named_modules())
    names = args.modules or list(DEFAULT_MODULES)
    missing = [name for name in names if name not in modules]
    if missing:
        raise ValueError(f"unknown module names: {missing}")
    captured = {}
    handles = []
    for name in names:
        def pre_hook(_module, inputs, _name=name):
            tensor = inputs[0] if inputs else None
            if isinstance(tensor, torch.Tensor):
                captured[_name] = tensor.detach().clone()
        handles.append(modules[name].register_forward_pre_hook(pre_hook))

    ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].to(args.device)
    capacity = int(ids.shape[1]) + 8
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.no_grad():
        logits = model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
        pending = logits.argmax(dim=-1)
        model(pending, past_key_values=cache.cache, use_cache=True)
    for handle in handles:
        handle.remove()
    torch.cuda.synchronize(args.device)

    results = {}
    with torch.inference_mode():
        for name in names:
            if name not in captured:
                results[name] = {"status": "not_called_in_first_decode"}
                continue
            module = modules[name]
            x1 = captured[name]
            if x1.ndim < 3 or x1.shape[-2] != 1:
                results[name] = {"status": "unexpected_input_shape",
                                 "shape": list(x1.shape)}
                continue
            reference = module(x1).detach()
            by_m = {}
            for m in (1, 2, 3, 5):
                x = x1.expand(*x1.shape[:-2], m, x1.shape[-1]).contiguous()
                value = module(x).detach()
                row_diffs = []
                for row_index in range(m):
                    delta = (value[..., row_index:row_index + 1, :].float()
                             - reference.float()).abs()
                    row_diffs.append({
                        "bitwise_equal_to_m1": torch.equal(
                            value[..., row_index:row_index + 1, :], reference
                        ),
                        "max_abs_diff": float(delta.max().item()),
                    })
                samples = []
                for round_index in range(args.rounds):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.iterations):
                        module(x)
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / args.iterations)
                by_m[str(m)] = {
                    "input_shape": list(x.shape),
                    "output_shape": list(value.shape),
                    "rows_vs_singleton": row_diffs,
                    "median_cuda_ms": statistics.median(samples),
                    "samples_ms": samples,
                }
            baseline_ms = by_m["1"]["median_cuda_ms"]
            for value in by_m.values():
                value["latency_ratio_vs_m1"] = value["median_cuda_ms"] / baseline_ms
            results[name] = {
                "module_class": module.__class__.__name__,
                "captured_m1_input_shape": list(x1.shape),
                "input_dtype": str(x1.dtype),
                "m_shapes": by_m,
            }
            print(name, json.dumps({m: {
                "all_rows_exact": all(x["bitwise_equal_to_m1"] for x in v["rows_vs_singleton"]),
                "row0_max_abs": v["rows_vs_singleton"][0]["max_abs_diff"],
                "latency_ratio": v["latency_ratio_vs_m1"],
            } for m, v in by_m.items()}), flush=True)

    payload = {
        "status": "operator-only finite shape screen; direct module calls, not end-to-end latency",
        "model": model_name,
        "revision": revision,
        "prompt_id": args.prompt_id,
        "prompt_tokens": int(ids.shape[1]),
        "first_generated_token": int(pending.item()),
        "backend": loaded.metadata,
        "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": [1, 2, 3, 5],
        "rounds": args.rounds,
        "iterations_per_round": args.iterations,
        "results": results,
        "caveat": "Repeated copies of real singleton activations screen row-count dispatch; block rows with distinct activations and full-model propagation need separate audits.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
