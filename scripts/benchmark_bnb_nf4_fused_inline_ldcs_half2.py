#!/usr/bin/env python3
"""Paired timing of exact fused NF4 M=3 using explicit LDG.CS.V4.U32."""

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
):
    outs = tl.program_id(0) * OUT_TILE + tl.arange(0, OUT_TILE)[None, :, None]
    lanes = tl.arange(0, 32)[None, None, :]
    acc0 = tl.zeros((1, OUT_TILE, 32), dtype=tl.float32)
    acc1 = tl.zeros((1, OUT_TILE, 32), dtype=tl.float32)
    acc2 = tl.zeros((1, OUT_TILE, 32), dtype=tl.float32)
    offset = tl.load(offset_ptr).to(tl.float32)
    block_log2: tl.constexpr = 6
    blocks_per_row: tl.constexpr = K // BLOCKSIZE

    for group in range(tl.cdiv(K, 1024)):
        inner_k = group * 1024 + lanes * 32
        scale_idx = outs * blocks_per_row + (inner_k >> block_log2)
        valid = (outs < N) & (inner_k < K)
        scale_q = tl.load(absmax8_ptr + scale_idx, mask=valid, other=0).to(tl.int32)
        scale_code = tl.load(state2code_ptr + scale_q)
        scale_state = tl.load(state2absmax_ptr + (scale_idx >> 8))
        scale_f = tl.fma(scale_code, scale_state, offset)
        scale_h = scale_f.to(tl.float16)

        # Match BNB's 16-byte streaming load at each (output, lane) position.
        word_base = outs * (K // 8) + group * 128 + lanes * 4
        packed0, packed1, packed2, packed3 = tl.inline_asm_elementwise(
            asm="""
            {
                .reg .pred p;
                setp.ne.u32 p, $5, 0;
                mov.b32 $0, 0;
                mov.b32 $1, 0;
                mov.b32 $2, 0;
                mov.b32 $3, 0;
                @p ld.global.cs.v4.u32 {$0, $1, $2, $3}, [$4];
            }
            """,
            constraints="=r,=r,=r,=r,l,r",
            args=[packed_ptr + word_base, valid.to(tl.uint32)],
            dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
            is_pure=True,
            pack=1,
        )

        for pair in tl.static_range(16):
            k0 = inner_k + pair * 2
            if pair < 4:
                packed_word = packed0
            elif pair < 8:
                packed_word = packed1
            elif pair < 12:
                packed_word = packed2
            else:
                packed_word = packed3
            byte = ((packed_word >> ((pair % 4) * 8)) & 0xFF).to(tl.int32)
            code0 = (byte >> 4) & 0x0F
            code1 = byte & 0x0F
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
            packed_w = (weight0.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                        | (weight1.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32) << 16))
            packed_x0 = (x00.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                         | (x01.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32) << 16))
            packed_x1 = (x10.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                         | (x11.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32) << 16))
            packed_x2 = (x20.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32)
                         | (x21.to(tl.float16).to(tl.uint16, bitcast=True).to(tl.uint32) << 16))
            prod_pair0 = tl.inline_asm_elementwise("mul.rn.f16x2 $0, $1, $2;", "=r,r,r", [packed_x0, packed_w], tl.uint32, True, 1)
            prod_pair1 = tl.inline_asm_elementwise("mul.rn.f16x2 $0, $1, $2;", "=r,r,r", [packed_x1, packed_w], tl.uint32, True, 1)
            prod_pair2 = tl.inline_asm_elementwise("mul.rn.f16x2 $0, $1, $2;", "=r,r,r", [packed_x2, packed_w], tl.uint32, True, 1)
            prod00 = prod_pair0.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            prod01 = (prod_pair0 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            prod10 = prod_pair1.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            prod11 = (prod_pair1 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            prod20 = prod_pair2.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
            prod21 = (prod_pair2 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
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
    lane0, lane1, lane2 = tl.gather(acc0, lane_idx0, axis=2), tl.gather(acc1, lane_idx0, axis=2), tl.gather(acc2, lane_idx0, axis=2)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + outs, mask=outs < N, other=0).to(tl.float32)
        lane0, lane1, lane2 = lane0 + bias, lane1 + bias, lane2 + bias
    tl.store(y_ptr + outs, lane0.to(tl.float16), outs < N)
    tl.store(y_ptr + N + outs, lane1.to(tl.float16), outs < N)
    tl.store(y_ptr + 2 * N + outs, lane2.to(tl.float16), outs < N)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--projection", default="gate_proj")
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
    module = getattr(layer.self_attn if args.projection in {"q_proj", "k_proj", "v_proj", "o_proj"} else layer.mlp, args.projection)
    capture: dict[str, torch.Tensor] = {}
    h1 = module.register_forward_pre_hook(lambda _m, inputs: capture.update(x=inputs[0].detach().reshape(-1, inputs[0].shape[-1]).clone()))
    h2 = module.register_forward_hook(lambda _m, _i, output: capture.update(ref=output.detach().reshape(-1, output.shape[-1]).clone()))
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
    h1.remove(); h2.remove()
    x_raw, ref = capture["x"], capture["ref"]
    x = x_raw.to(module.compute_dtype).contiguous()
    state = module.weight.quant_state
    n, k = int(module.out_features), int(module.in_features)
    if state.quant_type != "nf4" or not state.nested or module.compute_dtype != torch.float16 or k % 1024 or int(state.blocksize) != 64:
        raise RuntimeError(f"Unsupported projection contract: K={k}, blocksize={state.blocksize}")
    packed = module.weight.data.reshape(-1).contiguous().view(torch.uint32)
    args_kernel = (packed, state.absmax.contiguous(), state.state2.absmax.contiguous(), state.state2.code.contiguous(), state.offset.float().contiguous(), state.code.contiguous(), module.bias.to(torch.float16).contiguous() if module.bias is not None else torch.empty((1,), device="cuda:0", dtype=torch.float16))
    out_tile, num_warps = 8, 8
    output = torch.empty((3, n), device="cuda:0", dtype=torch.float16)
    grid = (triton.cdiv(n, out_tile),)
    def launch():
        _fused_nf4_m3_kernel[grid](x, *args_kernel, output, k, n, int(state.blocksize), module.bias is not None, OUT_TILE=out_tile, num_warps=num_warps, enable_fp_fusion=False)
        return output.to(x_raw.dtype)
    candidate = launch(); torch.cuda.synchronize()
    exact = int(torch.count_nonzero(candidate == ref).item())
    max_diff = float((candidate.float() - ref.float()).abs().max().item())
    for _ in range(5): launch()
    torch.cuda.synchronize()
    triton_ms = triton.testing.do_bench(launch, warmup=10, rep=60)
    for _ in range(5): module(x_raw)
    torch.cuda.synchronize()
    bnb_ms = triton.testing.do_bench(lambda: module(x_raw), warmup=10, rep=60)
    paired = {"triton": [], "bnb": []}
    for i in range(30):
        order = ("triton", "bnb") if i % 2 == 0 else ("bnb", "triton")
        for method in order:
            fn = launch if method == "triton" else lambda: module(x_raw)
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(8): fn()
            end.record(); end.synchronize()
            paired[method].append(start.elapsed_time(end) / 8)
    ratios = [b / t for t, b in zip(paired["triton"], paired["bnb"])]
    result = {"prompt_id": args.prompt_id, "layer": args.layer, "projection": args.projection, "shape": list(x.shape), "output_shape": list(ref.shape), "exact_elements": exact, "total_elements": ref.numel(), "bitwise_exact": exact == ref.numel(), "max_abs_diff": max_diff, "unpaired_triton_ms": triton_ms, "unpaired_bnb_ms": bnb_ms, "paired_triton_median_ms": statistics.median(paired["triton"]), "paired_bnb_median_ms": statistics.median(paired["bnb"]), "paired_speedup_bnb_over_triton_median": statistics.median(ratios), "paired_ratios": ratios, "paired_rounds": 30, "repeats_per_round": 8, "runtime": {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda, "bnb": importlib.metadata.version("bitsandbytes"), "gpu": torch.cuda.get_device_name(0)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if exact != ref.numel(): raise RuntimeError("inline LDG.CS.V4 candidate is not bitwise exact")


if __name__ == "__main__":
    main()
