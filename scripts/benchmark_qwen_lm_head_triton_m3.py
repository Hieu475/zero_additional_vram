#!/usr/bin/env python3
"""Screen Triton LM-head matmuls at the fixed three-row K2/R3 decode shape."""

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
from triton.runtime.errors import OutOfResources

from zassd.backends.base import load_backend


@triton.jit
def _lm_head_mm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(tl.cdiv(K, BLOCK_K)):
        k_ids = k_start * BLOCK_K + offs_k
        a = tl.load(
            x_ptr + offs_m[:, None] * K + k_ids[None, :],
            mask=(offs_m[:, None] < M) & (k_ids[None, :] < K),
            other=0,
        )
        b_nk = tl.load(
            weight_ptr + offs_n[:, None] * K + k_ids[None, :],
            mask=(offs_n[:, None] < N) & (k_ids[None, :] < K),
            other=0,
        )
        acc = tl.dot(a, tl.trans(b_nk), acc)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(y_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def triton_lm_head(x, module, *, block_n, block_k, num_warps):
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != 3:
        raise ValueError("prototype expects fixed [1,3,H] physical rows")
    weight = module.weight
    rows, hidden = int(x.shape[1]), int(x.shape[2])
    vocab = int(weight.shape[0])
    out = torch.empty((1, rows, vocab), device=x.device, dtype=weight.dtype)
    bias = module.bias if module.bias is not None else weight
    _lm_head_mm_kernel[(triton.cdiv(vocab, block_n),)](
        x.contiguous(), weight, bias, out,
        rows, vocab, hidden, module.bias is not None,
        4, block_n, block_k,
        num_warps=num_warps,
        enable_fp_fusion=False,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/humaneval_ext114.jsonl")
    parser.add_argument("--prompt-id", default="HumanEval/112")
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in rows if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt {args.prompt_id!r} not found")
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa",
    )
    model = loaded.model.eval()
    tokenizer = loaded.tokenizer
    lm_head = model.lm_head
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    captured = {}

    def capture_hidden(_module, inputs):
        captured["hidden"] = inputs[0].detach()

    hook = lm_head.register_forward_pre_hook(capture_hidden)
    with torch.inference_mode():
        model(prompt_ids, use_cache=False)
    hook.remove()
    hidden = captured["hidden"]
    if hidden.shape[1] < 3:
        raise ValueError("prompt needs at least three tokens")
    x = hidden[:, -3:, :].contiguous()
    with torch.inference_mode():
        reference = F.linear(x, lm_head.weight, lm_head.bias)
    torch.cuda.synchronize()
    candidates = [
        (64, 32, 4), (64, 64, 4), (128, 64, 4), (128, 128, 4),
        (256, 64, 4), (256, 128, 4), (128, 64, 8),
    ]
    outputs, timings, correctness, failures = {}, {}, {}, {}
    schedules = {f"n{n}_k{k}_w{w}": (n, k, w) for n, k, w in candidates}
    valid_schedules = {}
    with torch.inference_mode():
        for name, (block_n, block_k, warps) in schedules.items():
            # Warm compiler and allocator before measuring.
            try:
                candidate = triton_lm_head(
                    x, lm_head, block_n=block_n, block_k=block_k, num_warps=warps
                )
            except OutOfResources as exc:
                failures[name] = str(exc)
                continue
            valid_schedules[name] = (block_n, block_k, warps)
            torch.cuda.synchronize()
            outputs[name] = candidate.detach()
            reference_ids = reference.argmax(dim=-1)
            candidate_ids = candidate.argmax(dim=-1)
            correctness[name] = {
                "argmax_matches_torch_by_row": (reference_ids == candidate_ids).tolist(),
                "bitwise_logits_match_torch": torch.equal(candidate, reference),
                "max_abs_logit_error": float((candidate.float() - reference.float()).abs().max().item()),
                "max_abs_error_index": int((candidate.float() - reference.float()).abs().argmax().item()),
            }
            row_equal = []
            for row in range(3):
                one_live = torch.zeros_like(x)
                one_live[:, row:row + 1, :] = x[:, row:row + 1, :]
                padded = triton_lm_head(
                    one_live, lm_head, block_n=block_n, block_k=block_k, num_warps=warps
                )
                row_equal.append(torch.equal(
                    candidate[:, row:row + 1, :], padded[:, row:row + 1, :]
                ))
            correctness[name]["rowwise_equal_with_other_rows_changed"] = row_equal

        timing_samples = {"torch_linear": [], **{name: [] for name in valid_schedules}}
        for round_id in range(args.rounds):
            order = ["torch_linear", *valid_schedules]
            if round_id % 2:
                order.reverse()
            for name in order:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    if name == "torch_linear":
                        F.linear(x, lm_head.weight, lm_head.bias)
                    else:
                        n, k, w = valid_schedules[name]
                        triton_lm_head(x, lm_head, block_n=n, block_k=k, num_warps=w)
                end.record()
                end.synchronize()
                timing_samples[name].append(start.elapsed_time(end) / args.iterations)
        medians = {name: statistics.median(values) for name, values in timing_samples.items()}
    baseline = medians["torch_linear"]
    for name, value in medians.items():
        timings[name] = {"median_cuda_ms": value, "ratio_vs_torch_linear": value / baseline,
                         "samples_ms": timing_samples[name]}
    payload = {
        "schema": "zassd-qwen-lm-head-m3-triton-screen-v1",
        "status": "finite operator-only screen on actual prompt hidden states; not integrated",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "prompt_id": args.prompt_id,
        "prompt_tokens": int(prompt_ids.shape[1]),
        "activation_source": "last three hidden rows captured at lm_head during prompt prefill",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": importlib.metadata.version("triton"),
        "shape": {"physical_rows": 3, "hidden": int(lm_head.in_features),
                  "vocab": int(lm_head.out_features), "input_dtype": str(x.dtype),
                  "weight_dtype": str(lm_head.weight.dtype)},
        "iterations_per_sample": args.iterations,
        "alternating_rounds": args.rounds,
        "correctness": correctness,
        "resource_rejections": failures,
        "timing": timings,
        "limits": [
            "One RTX 4050, one set of three prompt activation rows; no full decode comparison.",
            "Triton reduction schedule differs from PyTorch GEMM and may change argmax on other rows/prompts.",
            "Only the exact M=3 path is tested; runtime must pad singleton rows to the same physical shape.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "torch_linear_ms": baseline,
        "candidates": {name: {"ms": timings[name]["median_cuda_ms"],
                              "ratio": timings[name]["ratio_vs_torch_linear"],
                              "argmax_matches": correctness[name]["argmax_matches_torch_by_row"],
                              "rowwise_shape_invariant": correctness[name]["rowwise_equal_with_other_rows_changed"]}
                       for name in valid_schedules},
    }, indent=2))


if __name__ == "__main__":
    main()
