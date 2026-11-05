#!/usr/bin/env python3
"""Shape-invariance prototype: one Triton program per attention query/head.

This uses a fixed serial key accumulation order within each query. It is a
synthetic operator prototype, not a drop-in Qwen kernel or BNB equivalence claim.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fixed_row_attention(Q, K, V, O,
                         H: tl.constexpr, Q_LEN: tl.constexpr,
                         K_LEN: tl.constexpr, D: tl.constexpr,
                         SCALE: tl.constexpr, BLOCK_D: tl.constexpr):
    row_head = tl.program_id(0)
    row = row_head // H
    head = row_head % H
    visible = K_LEN - Q_LEN + row + 1
    d = tl.arange(0, BLOCK_D)
    q_ptr = Q + (head * Q_LEN + row) * D + d
    q = tl.load(q_ptr, d < D, other=0).to(tl.float32)
    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.full((), 0.0, tl.float32)
    running_value = tl.zeros((BLOCK_D,), tl.float32)

    for key_index in range(visible):
        k_ptr = K + (head * K_LEN + key_index) * D + d
        v_ptr = V + (head * K_LEN + key_index) * D + d
        k = tl.load(k_ptr, d < D, other=0).to(tl.float32)
        v = tl.load(v_ptr, d < D, other=0).to(tl.float32)
        score = tl.sum(q * k, axis=0) * SCALE
        next_max = tl.maximum(running_max, score)
        old_scale = tl.exp(running_max - next_max)
        new_scale = tl.exp(score - next_max)
        running_value = running_value * old_scale + v * new_scale
        running_sum = running_sum * old_scale + new_scale
        running_max = next_max

    result = running_value / running_sum
    out_ptr = O + (head * Q_LEN + row) * D + d
    tl.store(out_ptr, result, d < D)


@triton.jit
def _fixed_row_attention_tiled(Q, K, V, O,
                               H: tl.constexpr, Q_LEN: tl.constexpr,
                               K_LEN: tl.constexpr, D: tl.constexpr,
                               SCALE: tl.constexpr, BLOCK_D: tl.constexpr,
                               BLOCK_K: tl.constexpr,
                               PHYSICAL_ROWS: tl.constexpr):
    row_head = tl.program_id(0)
    row = row_head // H
    head = row_head % H
    visible = K_LEN - PHYSICAL_ROWS + row + 1
    d = tl.arange(0, BLOCK_D)
    keys = tl.arange(0, BLOCK_K)
    q_ptr = Q + (head * Q_LEN + row) * D + d
    q = tl.load(q_ptr, d < D, other=0).to(tl.float32)
    running_max = tl.full((), float("-inf"), tl.float32)
    running_sum = tl.full((), 0.0, tl.float32)
    running_value = tl.zeros((BLOCK_D,), tl.float32)

    for tile_start in range(0, visible, BLOCK_K):
        key_indices = tile_start + keys
        valid = key_indices < visible
        k_ptr = K + (head * K_LEN + key_indices[:, None]) * D + d[None, :]
        v_ptr = V + (head * K_LEN + key_indices[:, None]) * D + d[None, :]
        k = tl.load(k_ptr, valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
        v = tl.load(v_ptr, valid[:, None] & (d[None, :] < D), other=0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, float("-inf"))
        tile_max = tl.max(scores, axis=0)
        weights = tl.exp(scores - tile_max)
        tile_sum = tl.sum(weights, axis=0)
        tile_value = tl.sum(weights[:, None] * v, axis=0)

        next_max = tl.maximum(running_max, tile_max)
        old_scale = tl.exp(running_max - next_max)
        tile_scale = tl.exp(tile_max - next_max)
        running_value = running_value * old_scale + tile_value * tile_scale
        running_sum = running_sum * old_scale + tile_sum * tile_scale
        running_max = next_max

    result = running_value / running_sum
    out_ptr = O + (head * Q_LEN + row) * D + d
    tl.store(out_ptr, result, d < D)


def fixed_row_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: float) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("expected Q[B,H,Q,D], K/V[B,H,K,D]")
    if q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("prototype currently supports batch=1")
    if q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
        raise ValueError("query and key/value head dimensions must match")
    q_len, k_len, dim = q.shape[-2], k.shape[-2], q.shape[-1]
    if k_len < q_len:
        raise ValueError("KV sequence must include all query rows")
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    grid = (q_len * q.shape[1],)
    _fixed_row_attention[grid](
        q.contiguous(), k.contiguous(), v.contiguous(), output,
        q.shape[1], q_len, k_len, dim, scale,
        triton.next_power_of_2(dim), num_warps=4,
    )
    return output


def fixed_row_attention_tiled(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                              scale: float, block_k: int = 16,
                              physical_rows: int | None = None) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("expected Q[B,H,Q,D], K/V[B,H,K,D]")
    if q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError("prototype currently supports batch=1")
    if q.shape[1] != k.shape[1] or q.shape[-1] != k.shape[-1]:
        raise ValueError("query and key/value head dimensions must match")
    q_len, k_len, dim = q.shape[-2], k.shape[-2], q.shape[-1]
    physical_rows = q_len if physical_rows is None else int(physical_rows)
    if physical_rows < q_len or k_len < physical_rows:
        raise ValueError("KV sequence must include the fixed physical row count")
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    grid = (q_len * q.shape[1],)
    _fixed_row_attention_tiled[grid](
        q.contiguous(), k.contiguous(), v.contiguous(), output,
        q.shape[1], q_len, k_len, dim, scale,
        triton.next_power_of_2(dim), block_k, physical_rows, num_warps=4,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-length", type=int, default=397)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    if min(args.prefix_length, args.rows, args.heads, args.head_dim,
           args.rounds, args.block_k) <= 0:
        raise ValueError("all dimensions and rounds must be positive")
    if args.block_k & (args.block_k - 1):
        raise ValueError("block-k must be a power of two")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = "cuda:0"
    dtype = torch.bfloat16
    torch.manual_seed(20261028)
    q_block = torch.randn((1, args.heads, args.rows, args.head_dim),
                          device=device, dtype=dtype)
    k_full = torch.randn((1, args.heads, args.prefix_length + args.rows,
                          args.head_dim), device=device, dtype=dtype)
    v_full = torch.randn_like(k_full)
    scale = args.head_dim ** -0.5

    block_output = fixed_row_attention(q_block, k_full, v_full, scale)
    tiled_block_output = fixed_row_attention_tiled(
        q_block, k_full, v_full, scale, args.block_k)
    torch.cuda.synchronize()
    row_results = []
    tiled_row_results = []
    for row in range(args.rows):
        visible = args.prefix_length + row + 1
        singleton = fixed_row_attention(
            q_block[:, :, row:row + 1].contiguous(),
            k_full[:, :, :visible].contiguous(),
            v_full[:, :, :visible].contiguous(), scale)
        observed = block_output[:, :, row:row + 1]
        delta = (observed.float() - singleton.float()).abs()
        row_results.append({
            "row": row,
            "visible_kv_length": visible,
            "bitwise_equal": torch.equal(observed, singleton),
            "max_abs": float(delta.max().item()),
        })
        tiled_singleton = fixed_row_attention_tiled(
            q_block[:, :, row:row + 1].contiguous(),
            k_full[:, :, :visible].contiguous(),
            v_full[:, :, :visible].contiguous(), scale, args.block_k)
        tiled_observed = tiled_block_output[:, :, row:row + 1]
        tiled_delta = (tiled_observed.float() - tiled_singleton.float()).abs()
        tiled_row_results.append({
            "row": row, "visible_kv_length": visible,
            "bitwise_equal": torch.equal(tiled_observed, tiled_singleton),
            "max_abs": float(tiled_delta.max().item()),
        })

    positions = torch.arange(args.prefix_length + args.rows, device=device)
    valid_lengths = args.prefix_length + torch.arange(1, args.rows + 1, device=device)
    block_mask = (positions[None, :] < valid_lengths[:, None])[None, None, :, :]

    def torch_sdpa_block():
        return F.scaled_dot_product_attention(
            q_block, k_full, v_full, attn_mask=block_mask,
            dropout_p=0.0, is_causal=False, scale=scale)

    def torch_sdpa_singletons():
        return [F.scaled_dot_product_attention(
            q_block[:, :, row:row + 1],
            k_full[:, :, :args.prefix_length + row + 1],
            v_full[:, :, :args.prefix_length + row + 1],
            dropout_p=0.0, is_causal=False, scale=scale)
            for row in range(args.rows)]

    torch_block_output = torch_sdpa_block()
    torch_single_outputs = torch_sdpa_singletons()
    torch_row_results = []
    for row, singleton in enumerate(torch_single_outputs):
        observed = torch_block_output[:, :, row:row + 1]
        delta = (observed.float() - singleton.float()).abs()
        torch_row_results.append({
            "row": row,
            "bitwise_equal": torch.equal(observed, singleton),
            "max_abs": float(delta.max().item()),
        })

    def time_call(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.rounds):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        return samples

    block_samples = time_call(lambda: fixed_row_attention(q_block, k_full, v_full, scale))
    singleton_samples = time_call(lambda: [
        fixed_row_attention(
            q_block[:, :, row:row + 1].contiguous(),
            k_full[:, :, :args.prefix_length + row + 1].contiguous(),
            v_full[:, :, :args.prefix_length + row + 1].contiguous(), scale)
        for row in range(args.rows)
    ])
    tiled_block_samples = time_call(
        lambda: fixed_row_attention_tiled(q_block, k_full, v_full,
                                          scale, args.block_k))
    tiled_singleton_samples = time_call(lambda: [
        fixed_row_attention_tiled(
            q_block[:, :, row:row + 1].contiguous(),
            k_full[:, :, :args.prefix_length + row + 1].contiguous(),
            v_full[:, :, :args.prefix_length + row + 1].contiguous(),
            scale, args.block_k)
        for row in range(args.rows)
    ])
    torch_block_samples = time_call(torch_sdpa_block)
    torch_singleton_samples = time_call(torch_sdpa_singletons)
    payload = {
        "scope": "synthetic fixed-row Triton attention; shape-invariance screen only",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__, "triton": triton.__version__,
        "dtype": str(dtype), "prefix_length": args.prefix_length,
        "rows": args.rows, "heads": args.heads, "head_dim": args.head_dim,
        "reduction": "fixed per-query online-softmax; tiled variant uses a fixed key-tile reduction",
        "block_k": args.block_k,
        "rowwise_exact": all(row["bitwise_equal"] for row in row_results),
        "row_results": row_results,
        "tiled_rowwise_exact": all(row["bitwise_equal"] for row in tiled_row_results),
        "tiled_row_results": tiled_row_results,
        "torch_sdpa_row_results": torch_row_results,
        "median_block_ms": statistics.median(block_samples),
        "median_singleton_calls_ms": statistics.median(singleton_samples),
        "speedup_vs_singleton_calls": statistics.median(singleton_samples)
        / statistics.median(block_samples),
        "tiled_median_block_ms": statistics.median(tiled_block_samples),
        "tiled_median_singleton_calls_ms": statistics.median(tiled_singleton_samples),
        "tiled_speedup_vs_singleton_calls": statistics.median(tiled_singleton_samples)
        / statistics.median(tiled_block_samples),
        "torch_sdpa_median_block_ms": statistics.median(torch_block_samples),
        "torch_sdpa_median_singleton_calls_ms": statistics.median(torch_singleton_samples),
        "torch_sdpa_speedup_vs_singleton_calls": statistics.median(torch_singleton_samples)
        / statistics.median(torch_block_samples),
        "rounds": args.rounds,
        "limitations": [
            "synthetic tensors only for this timing; Qwen integration is audited separately",
            "one program per query/head with sequential key tiles; not a production-optimized kernel",
            "kernel expects expanded K/V and active cache tensors; adapter handles GQA and causal-prefix construction",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
