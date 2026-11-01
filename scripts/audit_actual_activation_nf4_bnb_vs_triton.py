#!/usr/bin/env python3
"""Compare BNB and fixed-three-row Triton on model-generated M=3 activations."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path
from types import MethodType

import torch
from bitsandbytes.nn import Linear4bit

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.certification.nf4_fixed_three_rows_group_scale_packed import (
    nf4_fixed_three_rows_group_scale_packed_linear,
)


MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


@contextmanager
def native_bnb_linears(model):
    saved = []
    for module in model.modules():
        if isinstance(module, Linear4bit):
            saved.append((module, module.forward))
            module.forward = MethodType(Linear4bit.forward, module)
    try:
        yield
    finally:
        for module, forward in reversed(saved):
            module.forward = forward


class Context:
    def __init__(self, model):
        self.model = model
        self.saved = {}
        self.handles = []

    def __enter__(self):
        for index, layer in enumerate(self.model.model.layers):
            for name in ("gate_proj", "up_proj"):
                module = getattr(layer.mlp, name)
                key = f"layers.{index}.mlp.{name}"
                def hook(_module, inputs, key=key):
                    self.saved[key] = inputs[0].detach().clone()
                self.handles.append(module.register_forward_pre_hook(hook))
        return self.saved

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    delta = (a.float() - b.float()).abs()
    return {"bitwise_equal": torch.equal(a, b),
            "max_abs_diff": float(delta.max().item()),
            "mean_abs_diff": float(delta.mean().item()),
            "shape": list(a.shape), "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}


def bench(fn, iterations: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    row = next((item for item in records if str(item["id"]) == args.prompt_id), None)
    if row is None:
        raise KeyError(f"prompt {args.prompt_id!r} not found")

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + 8

    # Build a three-token BNB greedy prefix using the regular singleton path.
    token_cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.inference_mode():
        logits = model(prompt_ids, past_key_values=token_cache.cache, use_cache=True).logits[:, -1:, :]
        token_ids = []
        for _ in range(4):
            token = logits.argmax(dim=-1)
            token_ids.append(int(token.item()))
            logits = model(token, past_key_values=token_cache.cache, use_cache=True).logits[:, -1:, :]

    # Replay those same rows as one M=3 native-BNB block and capture real MLP inputs.
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.inference_mode():
        model(prompt_ids, past_key_values=cache.cache, use_cache=True)
    block_tokens = torch.tensor([token_ids[:3]], device="cuda:0", dtype=torch.long)
    capture = Context(model)
    with (fixed_row_schedule_for_model(
            model, physical_rows=3, attention_strategy="bnb_rowwise",
            quantized_linear_strategy="fixed_rows"),
          native_bnb_linears(model), capture, torch.inference_mode()):
        block_logits = model(block_tokens, past_key_values=cache.cache, use_cache=True).logits
    activations = capture.saved
    if len(activations) != 72:
        raise RuntimeError(f"expected 72 gate/up activations, captured {len(activations)}")

    results = {}
    with torch.inference_mode():
        modules = dict(model.named_modules())
        for key, x in activations.items():
            module = modules["model." + key]
            bnb_m3 = module(x).detach()
            triton_m3 = nf4_fixed_three_rows_group_scale_packed_linear(x, module).detach()
            bnb_rows = [module(x[:, r:r + 1].contiguous()).detach() for r in range(3)]
            triton_rows = [nf4_fixed_three_rows_group_scale_packed_linear(
                x[:, r:r + 1].contiguous(), module).detach() for r in range(3)]
            bnb_rowwise = [metrics(bnb_rows[r], bnb_m3[:, r:r + 1]) for r in range(3)]
            triton_rowwise = [metrics(triton_rows[r], triton_m3[:, r:r + 1]) for r in range(3)]
            functions = {
                "bnb_m3": lambda x=x, module=module: module(x),
                "triton_m3": lambda x=x, module=module: nf4_fixed_three_rows_group_scale_packed_linear(x, module),
            }
            samples = {name: [] for name in functions}
            for round_index in range(args.rounds):
                order = list(functions)
                if round_index % 2:
                    order.reverse()
                for name in order:
                    samples[name].append(bench(functions[name], args.iterations))
            medians = {name: statistics.median(values) for name, values in samples.items()}
            results[key] = {
                "input_shape": list(x.shape), "input_dtype": str(x.dtype),
                "bnb_m3_vs_triton_m3": metrics(bnb_m3, triton_m3),
                "bnb_m3_vs_bnb_m1_rows": {
                    "all_rows_bitwise_equal": all(v["bitwise_equal"] for v in bnb_rowwise),
                    "rows": bnb_rowwise,
                },
                "triton_m3_vs_triton_m1_rows": {
                    "all_rows_bitwise_equal": all(v["bitwise_equal"] for v in triton_rowwise),
                    "rows": triton_rowwise,
                },
                "timing": {"samples_ms": samples, "median_ms": medians,
                           "triton_over_bnb_ratio": medians["triton_m3"] / medians["bnb_m3"]},
            }
            del bnb_m3, triton_m3, bnb_rows, triton_rows

    layer_results = {}
    for layer in range(len(model.model.layers)):
        entries = [results[f"layers.{layer}.mlp.{name}"] for name in ("gate_proj", "up_proj")]
        layer_results[str(layer)] = {
            "modules": 2,
            "bnb_triton_bitwise_equal": all(x["bnb_m3_vs_triton_m3"]["bitwise_equal"] for x in entries),
            "bnb_rowwise_invariant": all(x["bnb_m3_vs_bnb_m1_rows"]["all_rows_bitwise_equal"] for x in entries),
            "triton_rowwise_invariant": all(x["triton_m3_vs_triton_m1_rows"]["all_rows_bitwise_equal"] for x in entries),
            "gate_triton_over_bnb": entries[0]["timing"]["triton_over_bnb_ratio"],
            "up_triton_over_bnb": entries[1]["timing"]["triton_over_bnb_ratio"],
        }
    payload = {
        "schema": "zassd-actual-activation-nf4-bnb-vs-triton-v1",
        "classification": "finite operator audit with model-generated M=3 activations; no runtime dispatch or reference changed",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "prompt_id": args.prompt_id, "prompt_tokens": prompt_len,
        "forced_bnb_tokens": token_ids,
        "block_row_argmaxes": [int(x) for x in block_logits.argmax(dim=-1).flatten().tolist()],
        "activation_protocol": "Capture gate_proj/up_proj inputs from all 36 layers during one three-row teacher-forced block using native BNB linears and rowwise native SDPA; compare BNB and Triton on the exact captured tensor.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing_protocol": {"alternating_rounds": args.rounds, "iterations_per_round": args.iterations},
        "whole_model_token_match_to_bnb_forced_rows": [int(x) for x in block_logits.argmax(dim=-1).flatten().tolist()] == token_ids[1:],
        "layer_summary": layer_results,
        "results": results,
        "limits": [
            "One prompt and one three-token teacher-forced block; timings are direct operator CUDA events, not end-to-end latency.",
            "Gate/up only; attention projections and down_proj are not included here.",
            "The full block output argmax check is a short finite screen, not whole-sequence B2 proof.",
            "GSM8K held-out indices 300-349 were not used.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    exact_modules = sum(x["bnb_m3_vs_triton_m3"]["bitwise_equal"] for x in results.values())
    print(json.dumps({"output": str(args.output), "prompt_id": args.prompt_id,
                      "modules": len(results), "bnb_triton_bitwise_equal": exact_modules,
                      "bnb_rowwise_invariant": sum(x["bnb_m3_vs_bnb_m1_rows"]["all_rows_bitwise_equal"] for x in results.values()),
                      "triton_rowwise_invariant": sum(x["triton_m3_vs_triton_m1_rows"]["all_rows_bitwise_equal"] for x in results.values()),
                      "whole_model_token_match": payload["whole_model_token_match_to_bnb_forced_rows"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
