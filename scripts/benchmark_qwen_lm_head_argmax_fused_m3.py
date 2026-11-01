#!/usr/bin/env python3
"""Benchmark a Triton LM-head plus argmax kernel at fixed physical M=3."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from zassd.backends.base import load_backend


@triton.jit
def _lm_head_argmax_partial(
    x_ptr, weight_ptr, bias_ptr, partial_values_ptr, partial_ids_ptr,
    N: tl.constexpr, K: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    row_ids = tl.arange(0, 4)
    vocab_ids = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    acc = tl.zeros((4, BLOCK_N), dtype=tl.float32)
    for k_start in range(tl.cdiv(K, BLOCK_K)):
        k_ids = k_start * BLOCK_K + k_offsets
        x_tile = tl.load(
            x_ptr + row_ids[:, None] * K + k_ids[None, :],
            mask=(row_ids[:, None] < 3) & (k_ids[None, :] < K),
            other=0,
        )
        w_nk = tl.load(
            weight_ptr + vocab_ids[:, None] * K + k_ids[None, :],
            mask=(vocab_ids[:, None] < N) & (k_ids[None, :] < K),
            other=0,
        )
        acc = tl.dot(x_tile, tl.trans(w_nk), acc)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + vocab_ids, mask=vocab_ids < N, other=0).to(tl.float32)
        acc += bias[None, :]
    logits = acc.to(partial_values_ptr.dtype.element_ty).to(tl.float32)
    max_values = tl.max(logits, axis=1)
    max_ids = tl.min(
        tl.where(
            (logits == max_values[:, None]) & (vocab_ids[None, :] < N),
            tl.broadcast_to(vocab_ids[None, :], (4, BLOCK_N)),
            2147483647,
        ),
        axis=1,
    )
    tile_count = tl.cdiv(N, BLOCK_N)
    tl.store(partial_values_ptr + row_ids * tile_count + pid_n, max_values, row_ids < 3)
    tl.store(partial_ids_ptr + row_ids * tile_count + pid_n, max_ids, row_ids < 3)


@triton.jit
def _reduce_lm_head_argmax(
    partial_values_ptr, partial_ids_ptr, output_values_ptr, output_ids_ptr,
    TILE_COUNT: tl.constexpr, BLOCK_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    values = tl.load(
        partial_values_ptr + row * TILE_COUNT + tiles,
        mask=tiles < TILE_COUNT,
        other=-float("inf"),
    )
    ids = tl.load(
        partial_ids_ptr + row * TILE_COUNT + tiles,
        mask=tiles < TILE_COUNT,
        other=2147483647,
    )
    maximum = tl.max(values, axis=0)
    winning_id = tl.min(tl.where(values == maximum, ids, 2147483647), axis=0)
    tl.store(output_values_ptr + row, maximum)
    tl.store(output_ids_ptr + row, winning_id)


def triton_argmax(x, module, *, block_n, block_k, num_warps):
    if x.ndim != 3 or tuple(x.shape) != (1, 3, module.in_features):
        raise ValueError("prototype requires fixed [1,3,H] inputs")
    vocab, hidden = int(module.weight.shape[0]), int(module.weight.shape[1])
    tiles = triton.cdiv(vocab, block_n)
    partial_values = torch.empty((1, 3, tiles), device=x.device, dtype=torch.float32)
    partial_ids = torch.empty((1, 3, tiles), device=x.device, dtype=torch.int32)
    output_values = torch.empty((1, 3), device=x.device, dtype=torch.float32)
    output_ids = torch.empty((1, 3), device=x.device, dtype=torch.int64)
    bias = module.bias if module.bias is not None else module.weight
    _lm_head_argmax_partial[(tiles,)](
        x.contiguous(), module.weight, bias,
        partial_values, partial_ids, vocab, hidden, module.bias is not None,
        block_n, block_k, num_warps=num_warps, enable_fp_fusion=False,
    )
    _reduce_lm_head_argmax[(3,)](
        partial_values, partial_ids, output_values, output_ids,
        tiles, triton.next_power_of_2(tiles), num_warps=4,
    )
    return output_ids, output_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/humaneval_ext114.jsonl")
    parser.add_argument("--prompt-id", default="HumanEval/112")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prompts = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((p for p in prompts if str(p["id"]) == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt {args.prompt_id!r} absent")
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa",
    )
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    lm_head = model.lm_head
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    captured = {}

    def capture(_module, inputs):
        captured["hidden"] = inputs[0].detach()

    hook = lm_head.register_forward_pre_hook(capture)
    with torch.inference_mode():
        model(prompt_ids, use_cache=False)
    hook.remove()
    x = captured["hidden"][:, -3:, :].contiguous()
    with torch.inference_mode():
        torch_logits = F.linear(x, lm_head.weight, lm_head.bias)
        torch_ids = torch_logits.argmax(dim=-1)
        torch_values = torch_logits.max(dim=-1).values
    candidates = [(64, 64, 4), (128, 64, 4), (128, 128, 4), (256, 64, 4), (128, 64, 8)]
    correctness, valid, rejected = {}, {}, {}
    with torch.inference_mode():
        for n, k, warps in candidates:
            name = f"n{n}_k{k}_w{warps}"
            try:
                ids, values = triton_argmax(x, lm_head, block_n=n, block_k=k, num_warps=warps)
                torch.cuda.synchronize()
            except triton.runtime.errors.OutOfResources as exc:
                rejected[name] = str(exc)
                continue
            valid[name] = (n, k, warps)
            row_ids = []
            row_values = []
            for row in range(3):
                singleton_padded = torch.zeros_like(x)
                singleton_padded[:, row:row + 1, :] = x[:, row:row + 1, :]
                one_ids, one_values = triton_argmax(
                    singleton_padded, lm_head, block_n=n, block_k=k, num_warps=warps
                )
                row_ids.append(int(one_ids[0, row].item()) == int(ids[0, row].item()))
                row_values.append(torch.equal(one_values[:, row], values[:, row]))
            correctness[name] = {
                "ids_match_torch_all_rows": torch.equal(ids, torch_ids),
                "max_values_bitwise_match_torch_all_rows": torch.equal(
                    values.to(torch_values.dtype), torch_values
                ),
                "max_abs_top_value_error": float(
                    (values.float() - torch_values.float()).abs().max().item()
                ),
                "rowwise_id_invariant": row_ids,
                "rowwise_max_value_invariant": row_values,
                "output_token_ids": ids.tolist(),
                "torch_token_ids": torch_ids.tolist(),
            }

    timing_samples = {"torch_linear_plus_argmax": [], **{n: [] for n in valid}}
    with torch.inference_mode():
        for round_idx in range(args.rounds):
            order = ["torch_linear_plus_argmax", *valid]
            if round_idx % 2:
                order.reverse()
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    if name == "torch_linear_plus_argmax":
                        F.linear(x, lm_head.weight, lm_head.bias).argmax(dim=-1)
                    else:
                        n, k, warps = valid[name]
                        triton_argmax(x, lm_head, block_n=n, block_k=k, num_warps=warps)
                end.record()
                end.synchronize()
                timing_samples[name].append(start.elapsed_time(end) / args.iterations)
    medians = {name: statistics.median(samples) for name, samples in timing_samples.items()}
    baseline = medians["torch_linear_plus_argmax"]
    payload = {
        "schema": "zassd-qwen-lm-head-fused-argmax-m3-screen-v1",
        "status": "operator-only diagnostic; not integrated into the decoder",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "prompt_id": args.prompt_id,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "activation_source": "last three lm_head inputs captured during prompt prefill",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "shape": {"physical_rows": 3, "hidden": int(lm_head.in_features),
                  "vocab": int(lm_head.out_features), "dtype": str(x.dtype)},
        "iterations_per_sample": args.iterations,
        "alternating_rounds": args.rounds,
        "correctness": correctness,
        "resource_rejections": rejected,
        "timing": {name: {"median_cuda_ms": value, "ratio_vs_torch": value / baseline,
                          "samples_ms": timing_samples[name]} for name, value in medians.items()},
        "limits": [
            "Two or fewer prompt activation samples; finite argmax evidence only.",
            "This returns token IDs and top values rather than a full logits tensor; integrating requires a decoder-specific interface.",
            "End-to-end token/KV and latency were not tested.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "torch_ms": baseline,
        "candidates": {name: {"ms": medians[name], "ratio": medians[name] / baseline,
                              "ids_match": correctness[name]["ids_match_torch_all_rows"],
                              "values_match": correctness[name]["max_values_bitwise_match_torch_all_rows"],
                              "rowwise_ids": correctness[name]["rowwise_id_invariant"]}
                       for name in valid},
        "resource_rejections": rejected,
    }, indent=2))


if __name__ == "__main__":
    main()
