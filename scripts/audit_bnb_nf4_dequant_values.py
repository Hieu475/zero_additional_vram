#!/usr/bin/env python3
"""Check whether the Triton NF4 code/scale reconstruction matches BNB dequantization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from bitsandbytes.functional import dequantize_4bit

from zassd.backends.base import load_backend


MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-count", type=int, default=65536)
    ap.add_argument("--seed", type=int, default=9217)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model = loaded.model.eval()
    modules = {
        "q_proj": model.model.layers[0].self_attn.q_proj,
        "gate_proj": model.model.layers[0].mlp.gate_proj,
        "up_proj": model.model.layers[0].mlp.up_proj,
        "down_proj": model.model.layers[0].mlp.down_proj,
    }
    generator = torch.Generator(device="cuda:0").manual_seed(args.seed)
    results = {}
    with torch.inference_mode():
        for name, module in modules.items():
            state = module.weight.quant_state
            out_features, in_features = (int(x) for x in state.shape)
            total = out_features * in_features
            count = min(args.sample_count, total)
            indices = torch.randperm(total, generator=generator, device="cuda:0")[:count]
            packed = module.weight.data.reshape(-1)
            byte = packed[indices // 2].to(torch.int32)
            nibble = torch.where(indices % 2 == 0, byte >> 4, byte & 0x0F).long()
            code = state.code[nibble].float()
            quant_block = int(state.blocksize)
            scale_ids = indices // quant_block
            if state.nested:
                quantized_scale = state.absmax[scale_ids].long()
                nested_ids = scale_ids // int(state.state2.blocksize)
                nested_code = state.state2.code[quantized_scale].float()
                nested_scale = state.state2.absmax[nested_ids].float()
                scale = nested_code * nested_scale + state.offset.float()
            else:
                scale = state.absmax[scale_ids].float()
            reconstructed_fp32 = code * scale
            reconstructed_typed = reconstructed_fp32.to(state.dtype)
            full_reference = dequantize_4bit(module.weight.data, quant_state=state).reshape(-1)
            actual = full_reference[indices]
            results[name] = {
                "weight_shape": [out_features, in_features],
                "weight_dtype": str(state.dtype),
                "nested_quantization": bool(state.nested),
                "sample_count": count,
                "reconstruction_fp32_vs_bnb": {
                    "bitwise_equal": torch.equal(reconstructed_fp32, actual.float()),
                    "max_abs_diff": float((reconstructed_fp32 - actual.float()).abs().max().item()),
                    "mean_abs_diff": float((reconstructed_fp32 - actual.float()).abs().mean().item()),
                },
                "reconstruction_cast_to_bnb_dtype_vs_bnb": {
                    "bitwise_equal": torch.equal(reconstructed_typed, actual),
                    "max_abs_diff": float((reconstructed_typed.float() - actual.float()).abs().max().item()),
                    "mean_abs_diff": float((reconstructed_typed.float() - actual.float()).abs().mean().item()),
                },
            }
            print(name, json.dumps(results[name]), flush=True)

    payload = {
        "schema": "zassd-bnb-nf4-dequant-values-audit-v1",
        "classification": "finite sampled dequantization-value audit; no GEMM/reduction or decoder check",
        "model": MODEL,
        "revision": REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": __import__("importlib.metadata", fromlist=["version"]).version("bitsandbytes"),
        "seed": args.seed,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": results,
        "limits": [
            "Samples values from four layer-0 quantized matrices; not an exhaustive full-weight comparison.",
            "Does not test GEMM reduction order, output rounding after accumulation, or full-model state.",
            "Held-out GSM8K indices 300-349 were not used.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "modules": list(results)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
