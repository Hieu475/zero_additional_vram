#!/usr/bin/env python3
"""Check reconstructed BNB CUDA SIMT FP16 arithmetic against real gate/up calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from bitsandbytes.nn import Linear4bit
from audit_bnb_nf4_gate_reduction_orders import (
    MODEL,
    REVISION,
    _bnb_simt_lane_schedule_kernel,
    original_bnb_forward,
)


def bnb_half2_weight_slice(module, start: int, stop: int) -> torch.Tensor:
    state = module.weight.quant_state
    _, k = (int(v) for v in state.shape)
    rows = torch.arange(start, stop, device="cuda:0", dtype=torch.long)[:, None]
    cols = torch.arange(k, device="cuda:0", dtype=torch.long)[None, :]
    flat = rows * k + cols
    packed = module.weight.data.reshape(-1)
    byte = packed[flat // 2].to(torch.int32)
    nibble = torch.where(flat % 2 == 0, byte >> 4, byte & 0x0F).long()
    centroid = state.code[nibble].to(torch.float16)
    scale_ids = flat // int(state.blocksize)
    scale_code = state.state2.code[state.absmax[scale_ids].long()].float()
    scale_state = state.state2.absmax[scale_ids // int(state.state2.blocksize)].float()
    scale = torch.addcmul(state.offset.float().expand_as(scale_state), scale_state, scale_code)
    return (centroid * scale.to(torch.float16)).to(torch.float16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--layers", default="0,17,35")
    ap.add_argument("--all-linears", action="store_true", help="audit every Linear4bit module instead of gate/up only")
    ap.add_argument("--output-chunk", type=int, default=256)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((r for r in records if str(r["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    layers = [int(v) for v in args.layers.split(",") if v]

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    cache = TargetKVCache(backend="static", max_capacity=int(ids.shape[-1]) + 8)
    captures: dict[str, dict[str, torch.Tensor]] = {}
    handles = []
    selected = {}
    if args.all_linears:
        selected = {name: module for name, module in model.named_modules() if isinstance(module, Linear4bit)}
    else:
        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            for proj_name in ("gate_proj", "up_proj"):
                module = getattr(layer.mlp, proj_name)
                key = f"layers.{layer_idx}.mlp.{proj_name}"
                selected[key] = module
    for key in selected:
        captures[key] = {}
    for key, module in selected.items():

        def pre_hook(_module, inputs, key=key):
            captures[key]["input"] = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).clone()

        def post_hook(_module, _inputs, output, key=key):
            captures[key]["output"] = output.detach().reshape(-1, output.shape[-1]).clone()

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

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
    for handle in handles:
        handle.remove()

    records_out = []
    for key, module in selected.items():
        raw_x = captures[key]["input"]
        x = raw_x
        ref = captures[key]["output"]
        x = x.to(module.compute_dtype).contiguous()
        n = int(module.out_features)
        k = int(module.in_features)
        matched = total = 0
        max_diff = 0.0
        for start in range(0, n, args.output_chunk):
            stop = min(start + args.output_chunk, n)
            weight = bnb_half2_weight_slice(module, start, stop)
            bias = (module.bias[start:stop].to(torch.float16) if module.bias is not None
                    else torch.empty((1,), device="cuda:0", dtype=torch.float16))
            actual = torch.empty((3, stop - start), device="cuda:0", dtype=torch.float16)
            _bnb_simt_lane_schedule_kernel[(3, stop - start)](
                x, weight, bias, actual, 3, stop - start, k, module.bias is not None,
                num_warps=4, enable_fp_fusion=False,
            )
            candidate = actual.to(ref.dtype)
            expected = ref[:, start:stop]
            matched += int(torch.count_nonzero(candidate == expected).item())
            total += expected.numel()
            max_diff = max(max_diff, float((candidate.float() - expected.float()).abs().max().item()))
        m1_rows = []
        with torch.inference_mode(), original_bnb_forward(model):
            for row in range(3):
                m1_rows.append(module(raw_x[row:row + 1]).detach().reshape(1, -1))
        m1 = torch.cat(m1_rows, dim=0)
        m1_exact = int(torch.count_nonzero(m1 == ref).item())
        m1_max_diff = float((m1.float() - ref.float()).abs().max().item())
        records_out.append({"module": key, "activation_shape": list(x.shape), "output_features": n,
                            "exact_elements": matched, "total_elements": total,
                            "max_abs_diff": max_diff, "bitwise_exact": matched == total,
                            "m1_vs_m3_exact_elements": m1_exact, "m1_vs_m3_total_elements": ref.numel(),
                            "m1_vs_m3_max_abs_diff": m1_max_diff, "m1_vs_m3_bitwise_equal": m1_exact == ref.numel()})
        print(json.dumps(records_out[-1]), flush=True)

    payload = {
        "schema": "zassd-bnb-simt-linear4bit-audit-v1",
        "classification": "actual M=3 BNB CUDA SIMT output comparison; finite operator evidence only",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "prompt_tokens": int(ids.shape[-1]), "forced_bnb_tokens": forced[:3], "layers": layers,
        "selection": "all Linear4bit modules" if args.all_linears else "gate_proj and up_proj at selected layers",
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "arithmetic": "FP16 LUT centroid * FP16 nested scale rounded to FP16; FP16 activation * weight rounded to FP16; serial per-lane pair accumulation then SHFL.DOWN 16,8,4,2,1.",
        "modules": records_out,
        "overall": {"exact_elements": sum(r["exact_elements"] for r in records_out),
                    "total_elements": sum(r["total_elements"] for r in records_out),
                    "bitwise_exact": all(r["bitwise_exact"] for r in records_out),
                    "m1_vs_m3_exact_elements": sum(r["m1_vs_m3_exact_elements"] for r in records_out),
                    "m1_vs_m3_total_elements": sum(r["m1_vs_m3_total_elements"] for r in records_out),
                    "m1_vs_m3_bitwise_equal": all(r["m1_vs_m3_bitwise_equal"] for r in records_out)},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limits": ["One prompt; module selection is recorded above.",
                   "M=1 vs M=3 equality is checked at Linear4bit module outputs on these captured activations only; no whole-model token/KV identity or performance claim.",
                   "GSM8K held-out indices 300-349 were not used."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "overall": payload["overall"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
