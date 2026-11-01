#!/usr/bin/env python3
"""Tile sweep plus interleaved paired timing for the fused NF4 M=3 kernel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import math
from pathlib import Path

import torch
import triton
import triton.language as tl

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION, original_bnb_forward


@triton.jit
def _fused_nf4_m3_kernel(
    x_ptr, packed_ptr, absmax8_ptr, state2absmax_ptr, state2code_ptr,
    offset_ptr, codebook_ptr, bias_ptr, y_ptr,
    K: tl.constexpr, N: tl.constexpr, BLOCKSIZE: tl.constexpr,
    HAS_BIAS: tl.constexpr, OUT_TILE: tl.constexpr = 4,
    USE_SHFL_LUT: tl.constexpr = False,
):
    # One warp computes one output channel. Each CTA handles a tile of N columns
    # and all three physical rows through separate scalar row accumulators.
    outs = tl.program_id(0) * OUT_TILE + tl.arange(0, OUT_TILE)[None, :, None]
    lanes = tl.arange(0, 32)[None, None, :]
    acc0 = tl.zeros((1, OUT_TILE, 32), dtype=tl.float32)
    acc1 = tl.zeros((1, OUT_TILE, 32), dtype=tl.float32)
    acc2 = tl.zeros((1, OUT_TILE, 32), dtype=tl.float32)
    offset = tl.load(offset_ptr).to(tl.float32)
    if USE_SHFL_LUT:
        physical_lane = tl.inline_asm_elementwise(
            "mov.u32 $0, %laneid; and.b32 $1, $1, 0; add.u32 $0, $0, $1;",
            constraints="=r,r", args=[lanes.to(tl.uint32)], dtype=tl.uint32,
            is_pure=True, pack=1,
        )
        lut_half = tl.load(codebook_ptr + physical_lane, mask=physical_lane < 16, other=0).to(tl.float16)
        lut_bits = tl.broadcast_to(lut_half.to(tl.uint16, bitcast=True).to(tl.uint32), (1, OUT_TILE, 32))
    block_log2: tl.constexpr = 6
    blocks_per_row: tl.constexpr = K // BLOCKSIZE

    for group in range(tl.cdiv(K, 1024)):
        inner_k = group * 1024 + lanes * 32
        scale_idx = outs * blocks_per_row + (inner_k >> block_log2)
        scale_q = tl.load(absmax8_ptr + scale_idx,
                          mask=(outs < N) & (inner_k < K), other=0).to(tl.int32)
        scale_code = tl.load(state2code_ptr + scale_q)
        scale_state = tl.load(state2absmax_ptr + (scale_idx >> 8))
        scale_f = tl.fma(scale_code, scale_state, offset)
        scale_h = scale_f.to(tl.float16)

        # BNB loads 16 packed bytes per lane as one uint4. Load the equivalent
        # four uint32 words once here instead of issuing 16 byte loads.
        word_base = outs * (K // 8) + group * 128 + lanes * 4
        word0 = tl.load(packed_ptr + word_base + 0,
                        mask=(outs < N) & (inner_k < K), other=0).to(tl.uint32)
        word1 = tl.load(packed_ptr + word_base + 1,
                        mask=(outs < N) & (inner_k < K), other=0).to(tl.uint32)
        word2 = tl.load(packed_ptr + word_base + 2,
                        mask=(outs < N) & (inner_k < K), other=0).to(tl.uint32)
        word3 = tl.load(packed_ptr + word_base + 3,
                        mask=(outs < N) & (inner_k < K), other=0).to(tl.uint32)

        for pair in tl.static_range(16):
            k0 = inner_k + pair * 2
            if pair < 4:
                packed_word = word0
            elif pair < 8:
                packed_word = word1
            elif pair < 12:
                packed_word = word2
            else:
                packed_word = word3
            byte = ((packed_word >> ((pair % 4) * 8)) & 0xFF).to(tl.int32)
            code0 = (byte >> 4) & 0x0F
            code1 = byte & 0x0F
            if USE_SHFL_LUT:
                bits0 = tl.inline_asm_elementwise(
                    "shfl.sync.idx.b32 $0, $1, $2, 0x1f, -1;",
                    constraints="=r,r,r", args=[lut_bits, code0], dtype=tl.uint32,
                    is_pure=True, pack=1,
                )
                bits1 = tl.inline_asm_elementwise(
                    "shfl.sync.idx.b32 $0, $1, $2, 0x1f, -1;",
                    constraints="=r,r,r", args=[lut_bits, code1], dtype=tl.uint32,
                    is_pure=True, pack=1,
                )
                centroid0 = bits0.to(tl.uint16).to(tl.float16, bitcast=True)
                centroid1 = bits1.to(tl.uint16).to(tl.float16, bitcast=True)
            else:
                centroid0 = tl.load(codebook_ptr + code0).to(tl.float16)
                centroid1 = tl.load(codebook_ptr + code1).to(tl.float16)
            weight0 = (centroid0.to(tl.float32) * scale_h.to(tl.float32)).to(tl.float16).to(tl.float32)
            weight1 = (centroid1.to(tl.float32) * scale_h.to(tl.float32)).to(tl.float16).to(tl.float32)

            x00 = tl.load(x_ptr + k0, mask=k0 < K, other=0).to(tl.float16).to(tl.float32)
            x01 = tl.load(x_ptr + k0 + 1, mask=k0 + 1 < K, other=0).to(tl.float16).to(tl.float32)
            x10 = tl.load(x_ptr + K + k0, mask=k0 < K, other=0).to(tl.float16).to(tl.float32)
            x11 = tl.load(x_ptr + K + k0 + 1, mask=k0 + 1 < K, other=0).to(tl.float16).to(tl.float32)
            x20 = tl.load(x_ptr + 2 * K + k0, mask=k0 < K, other=0).to(tl.float16).to(tl.float32)
            x21 = tl.load(x_ptr + 2 * K + k0 + 1, mask=k0 + 1 < K, other=0).to(tl.float16).to(tl.float32)
            prod00 = (x00 * weight0).to(tl.float16).to(tl.float32)
            prod01 = (x01 * weight1).to(tl.float16).to(tl.float32)
            prod10 = (x10 * weight0).to(tl.float16).to(tl.float32)
            prod11 = (x11 * weight1).to(tl.float16).to(tl.float32)
            prod20 = (x20 * weight0).to(tl.float16).to(tl.float32)
            prod21 = (x21 * weight1).to(tl.float16).to(tl.float32)
            acc0 = acc0 + (prod00 + prod01)
            acc1 = acc1 + (prod10 + prod11)
            acc2 = acc2 + (prod20 + prod21)

    for offset_idx in tl.static_range(5):
        shift = 16 >> offset_idx
        src = tl.broadcast_to((lanes + shift) & 31, (1, OUT_TILE, 32))
        acc0 = tl.where(lanes < 32 - shift, acc0 + tl.gather(acc0, src, axis=2), acc0)
        acc1 = tl.where(lanes < 32 - shift, acc1 + tl.gather(acc1, src, axis=2), acc1)
        acc2 = tl.where(lanes < 32 - shift, acc2 + tl.gather(acc2, src, axis=2), acc2)
    lane_idx0 = tl.full((1, OUT_TILE, 1), 0, tl.int32)
    lane0 = tl.gather(acc0, lane_idx0, axis=2)
    lane1 = tl.gather(acc1, lane_idx0, axis=2)
    lane2 = tl.gather(acc2, lane_idx0, axis=2)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + outs, mask=outs < N, other=0).to(tl.float32)
        lane0 = lane0 + bias
        lane1 = lane1 + bias
        lane2 = lane2 + bias
    tl.store(y_ptr + outs, lane0.to(tl.float16), outs < N)
    tl.store(y_ptr + N + outs, lane1.to(tl.float16), outs < N)
    tl.store(y_ptr + 2 * N + outs, lane2.to(tl.float16), outs < N)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--projection", default="gate_proj")
    ap.add_argument("--shfl-lut", action="store_true", help="experimental physical-lane warp-shuffle LUT lookup")
    ap.add_argument("--output", type=Path, required=True)
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
    layer = model.model.layers[args.layer]
    module = getattr(layer.self_attn if args.projection.endswith("_proj") and args.projection in {"q_proj", "k_proj", "v_proj", "o_proj"} else layer.mlp, args.projection)
    capture: dict[str, torch.Tensor] = {}

    def pre_hook(_module, inputs):
        capture["x"] = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).clone()

    def post_hook(_module, _inputs, output):
        capture["ref"] = output.detach().reshape(-1, output.shape[-1]).clone()

    h1 = module.register_forward_pre_hook(pre_hook)
    h2 = module.register_forward_hook(post_hook)
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
        block_ids = torch.tensor([forced[:3]], device="cuda:0", dtype=torch.long)
        model(block_ids, past_key_values=block_cache.cache, use_cache=True)
    h1.remove(); h2.remove()

    x_raw, ref = capture["x"], capture["ref"]
    x = x_raw.to(module.compute_dtype).contiguous()
    state = module.weight.quant_state
    n, k = int(module.out_features), int(module.in_features)
    if state.quant_type != "nf4" or not state.nested or module.compute_dtype != torch.float16:
        raise RuntimeError("Prototype currently requires nested NF4 with FP16 compute")
    if k % 64 or k % 2 or int(state.blocksize) != 64:
        raise RuntimeError(f"Unsupported K/blocksize: K={k}, blocksize={state.blocksize}")
    packed = module.weight.data.reshape(-1).contiguous().view(torch.uint32)
    absmax8 = state.absmax.contiguous()
    state2absmax = state.state2.absmax.contiguous()
    state2code = state.state2.code.contiguous()
    offset = state.offset.float().contiguous()
    codebook = state.code.contiguous()
    bias = module.bias.to(torch.float16).contiguous() if module.bias is not None else torch.empty((1,), device="cuda:0", dtype=torch.float16)
    configs = [(1, 1), (2, 2), (4, 4), (8, 4), (8, 8), (16, 4), (16, 8), (32, 8), (32, 16), (64, 16), (64, 32)]
    kernel_results = []
    for out_tile, num_warps in configs:
        output = torch.empty((3, n), device="cuda:0", dtype=torch.float16)
        grid = (triton.cdiv(n, out_tile),)

        def launch():
            _fused_nf4_m3_kernel[grid](
                x, packed, absmax8, state2absmax, state2code, offset, codebook, bias, output,
                k, n, int(state.blocksize), module.bias is not None,
                OUT_TILE=out_tile, USE_SHFL_LUT=args.shfl_lut,
                num_warps=num_warps, enable_fp_fusion=False,
            )
            return output.to(x_raw.dtype)

        candidate = launch()
        torch.cuda.synchronize()
        exact = int(torch.count_nonzero(candidate == ref).item())
        max_diff = float((candidate.float() - ref.float()).abs().max().item())
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        triton_ms = triton.testing.do_bench(launch, warmup=10, rep=60)
        kernel_results.append({"output_tile": out_tile, "num_warps": num_warps,
                               "exact_elements": exact, "total_elements": ref.numel(),
                               "bitwise_exact": exact == ref.numel(), "max_abs_diff": max_diff,
                               "triton_operator_ms": triton_ms})

    for _ in range(5):
        module(x_raw)
    torch.cuda.synchronize()
    bnb_ms = triton.testing.do_bench(lambda: module(x_raw), warmup=20, rep=60)
    exact_candidates = [row for row in kernel_results if row["bitwise_exact"]]
    if not exact_candidates:
        raise RuntimeError("No candidate configuration preserved bitwise output identity")
    best = min(exact_candidates, key=lambda row: row["triton_operator_ms"])

    # Interleave repeated native BNB and best exact Triton calls in one process
    # to control clock/thermal drift seen in across-process do_bench samples.
    best_tile, best_warps = best["output_tile"], best["num_warps"]
    best_output = torch.empty((3, n), device="cuda:0", dtype=torch.float16)
    best_grid = (triton.cdiv(n, best_tile),)

    def best_launch():
        _fused_nf4_m3_kernel[best_grid](
            x, packed, absmax8, state2absmax, state2code, offset, codebook, bias, best_output,
            k, n, int(state.blocksize), module.bias is not None,
            OUT_TILE=best_tile, USE_SHFL_LUT=args.shfl_lut,
            num_warps=best_warps, enable_fp_fusion=False,
        )
        return best_output.to(x_raw.dtype)

    for _ in range(20):
        best_launch()
        module(x_raw)
    torch.cuda.synchronize()
    paired = {"triton": [], "bnb": []}
    per_round_ratios = []
    for round_index in range(30):
        order = ("triton", "bnb") if round_index % 2 == 0 else ("bnb", "triton")
        round_values = {}
        for method in order:
            fn = best_launch if method == "triton" else lambda: module(x_raw)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(8):
                fn()
            end.record()
            end.synchronize()
            value = start.elapsed_time(end) / 8.0
            paired[method].append(value)
            round_values[method] = value
        per_round_ratios.append(round_values["bnb"] / round_values["triton"])
    paired_summary = {
        "rounds": 30, "iterations_per_sample": 8,
        "order": "alternating triton/bnb and bnb/triton",
        "median_triton_ms": statistics.median(paired["triton"]),
        "median_bnb_ms": statistics.median(paired["bnb"]),
        "median_paired_speedup": statistics.median(per_round_ratios),
        "geomean_paired_speedup": math.exp(sum(math.log(x) for x in per_round_ratios) / len(per_round_ratios)),
        "paired_speedup_samples": per_round_ratios,
        "triton_samples_ms": paired["triton"], "bnb_samples_ms": paired["bnb"],
    }

    result = {
        "schema": "zassd-bnb-fused-nf4-simt-triton-prototype-v2",
        "classification": "single-operator fused prototype; output identity is tested, timing is diagnostic only",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "layer": args.layer, "projection": args.projection, "forced_tokens": forced[:3],
        "nf4_lookup": "physical-lane warp shuffle (experimental)" if args.shfl_lut else "cached per-nibble codebook loads",
        "activation_shape": list(x.shape), "output_shape": list(ref.shape),
        "model_compute_dtype": str(module.compute_dtype), "activation_dtype_before_cast": str(x_raw.dtype),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"), "triton": importlib.metadata.version("triton"),
        "kernel_candidates": kernel_results, "bnb_operator_ms": bnb_ms,
        "paired_confirmation": paired_summary,
        "best_triton_fused_operator_ms": best["triton_operator_ms"],
        "best_operator_speedup": bnb_ms / best["triton_operator_ms"],
        "best_exact_configuration": {"output_tile": best["output_tile"], "num_warps": best["num_warps"]},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["One Linear4bit module, one prompt, one GPU and one software/model configuration.",
                   "Kernel has not been integrated into model runtime; no KV/state or end-to-end performance result.",
                   "Timing uses a preallocated Triton output and is only an operator diagnostic.",
                   "GSM8K held-out indices 300-349 were not used."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
