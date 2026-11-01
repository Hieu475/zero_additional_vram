#!/usr/bin/env python3
"""Exactness-gated launch sweep for the activation-vectorized NF4 M=3 kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
import triton

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION, original_bnb_forward
from benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2 import _fused_nf4_m3_kernel


PROJECTIONS = ("q_proj", "gate_proj", "down_proj")
CONFIGS = ((4, 4), (4, 8), (8, 4), (8, 8), (16, 8), (16, 16))


def projection_module(model, name):
    if name.endswith("_proj") and name in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return getattr(model.model.layers[0].self_attn, name)
    return getattr(model.model.layers[0].mlp, name)


def collect_block_inputs(model, tokenizer, prompt, device):
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    capacity = int(ids.shape[-1]) + 8
    with torch.inference_mode(), original_bnb_forward(model):
        cache = TargetKVCache(backend="static", max_capacity=capacity)
        logits = model(ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
        forced = []
        for _ in range(3):
            token = logits.argmax(dim=-1)
            forced.append(int(token.item()))
            logits = model(token, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]

        captured = {}
        handles = []
        for name in PROJECTIONS:
            module = projection_module(model, name)
            def pre_hook(_module, args, _name=name):
                captured.setdefault(_name, {})["x"] = args[0].detach().reshape(-1, args[0].shape[-1]).clone()
            def post_hook(_module, _args, output, _name=name):
                captured.setdefault(_name, {})["ref"] = output.detach().reshape(-1, output.shape[-1]).clone()
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(post_hook))
        block_cache = TargetKVCache(backend="static", max_capacity=capacity)
        model(ids, past_key_values=block_cache.cache, use_cache=True)
        model(torch.tensor([forced], device=device, dtype=torch.long),
              past_key_values=block_cache.cache, use_cache=True)
        for handle in handles:
            handle.remove()
    return ids, forced, captured


def make_kernel_args(module):
    state = module.weight.quant_state
    if state.quant_type != "nf4" or not state.nested or module.compute_dtype != torch.float16:
        raise RuntimeError("expected nested NF4 with FP16 compute")
    if int(state.blocksize) != 64 or int(module.in_features) % 64:
        raise RuntimeError("unsupported quantization block contract")
    packed = module.weight.data.reshape(-1).contiguous().view(torch.uint32)
    bias = (module.bias.to(torch.float16).contiguous() if module.bias is not None
            else torch.empty((1,), device=packed.device, dtype=torch.float16))
    return state, (
        packed, state.absmax.contiguous(), state.state2.absmax.contiguous(),
        state.state2.code.contiguous(), state.offset.float().contiguous(),
        state.code.contiguous(), bias,
    )


def run_kernel(x_raw, module, state, kernel_args, config):
    out_tile, num_warps = config
    n, k = int(module.out_features), int(module.in_features)
    x = x_raw.to(module.compute_dtype).contiguous()
    output = torch.empty((3, n), device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(n, out_tile),)
    _fused_nf4_m3_kernel[grid](
        x, *kernel_args, output, k, n, int(state.blocksize), module.bias is not None,
        OUT_TILE=out_tile, num_warps=num_warps, enable_fp_fusion=False,
    )
    return output.to(x_raw.dtype)


def paired_latency(candidate, reference, rounds, repeats):
    samples = {"candidate": [], "bnb": []}
    for index in range(rounds):
        order = ("candidate", "bnb") if index % 2 == 0 else ("bnb", "candidate")
        for method in order:
            fn = candidate if method == "candidate" else reference
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                fn()
            end.record(); end.synchronize()
            samples[method].append(start.elapsed_time(end) / repeats)
    ratios = [b / c for c, b in zip(samples["candidate"], samples["bnb"])]
    return {
        "candidate_median_ms": statistics.median(samples["candidate"]),
        "bnb_median_ms": statistics.median(samples["bnb"]),
        "bnb_over_candidate_median": statistics.median(ratios),
        "paired_ratios": ratios,
        "paired_rounds": rounds,
        "repeats_per_round": repeats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in records if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    _, forced, captured = collect_block_inputs(model, tokenizer, prompt["prompt"], "cuda:0")
    results = {}
    with torch.inference_mode():
        for name in PROJECTIONS:
            module = projection_module(model, name)
            x_raw, ref = captured[name]["x"], captured[name]["ref"]
            state, kernel_args = make_kernel_args(module)
            bnb_fn = lambda x=x_raw, m=module: m(x)
            candidate_results = []
            for config in CONFIGS:
                try:
                    candidate_fn = lambda x=x_raw, m=module, s=state, ka=kernel_args, c=config: run_kernel(x, m, s, ka, c)
                    output = candidate_fn()
                    torch.cuda.synchronize()
                    exact_elements = int(torch.count_nonzero(output == ref).item())
                    max_abs_diff = float((output.float() - ref.float()).abs().max().item())
                    row = {"out_tile": config[0], "num_warps": config[1],
                           "exact_elements": exact_elements, "total_elements": ref.numel(),
                           "bitwise_exact": exact_elements == ref.numel(), "max_abs_diff": max_abs_diff}
                    if row["bitwise_exact"]:
                        for _ in range(3):
                            candidate_fn()
                        torch.cuda.synchronize()
                        row.update(paired_latency(candidate_fn, bnb_fn, args.rounds, args.repeats))
                    else:
                        row["latency_status"] = "not measured: exactness gate failed"
                except Exception as exc:
                    row = {"out_tile": config[0], "num_warps": config[1],
                           "launch_error": f"{type(exc).__name__}: {exc}"}
                candidate_results.append(row)
            results[name] = {
                "input_shape": list(x_raw.shape), "output_shape": list(ref.shape),
                "input_sha256": hashlib.sha256(x_raw.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                "configs": candidate_results,
            }
    payload = {
        "schema": "zassd-activationvec-nf4-m3-launch-sweep-v1",
        "classification": "single-prompt operator tuning; latency is measured only after bitwise equality to BNB M=3",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "forced_tokens": forced, "scope": "layer 0 q_proj/gate_proj/down_proj",
        "configs": [{"out_tile": n, "num_warps": w} for n, w in CONFIGS],
        "runtime": {"torch": torch.__version__, "triton": triton.__version__,
                    "cuda": torch.version.cuda,
                    "bitsandbytes": importlib.metadata.version("bitsandbytes"),
                    "gpu": torch.cuda.get_device_name(0)},
        "results": results,
        "limitations": ["one development prompt and one layer", "does not establish B1 singleton or end-to-end speedup",
                        "GSM8K held-out indices 300-349 excluded"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "prompt_id": args.prompt_id,
                      "results": results, "runtime": payload["runtime"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
