#!/usr/bin/env python3
"""Launch the contract's exact R=2/N=16/one-warp NF4 kernel for Nsight Compute."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from bitsandbytes.nn import Linear4bit

from zassd.backends.base import load_backend
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.certification.triton_b1_reference_contract import (
    verify_triton_b1_reference_contract,
)


MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
KERNEL_SOURCE = Path(
    "src/zassd/certification/nf4_fixed_rows_group_scale_packed_rows_warp_sweep.py"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--layer", choices=("q_proj", "down_proj"), required=True)
    parser.add_argument("--rows", type=int, choices=(1, 2), default=2)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("data/benchmarks/cnndm_eval.jsonl"))
    parser.add_argument("--prompt-id", default="cnndm_003")
    parser.add_argument("--output-metadata", type=Path, required=True)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.output_metadata.exists():
        raise FileExistsError(args.output_metadata)

    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model = loaded.model.eval()
    verified = verify_triton_b1_reference_contract(
        model, loaded.tokenizer, args.contract, device="cuda:0"
    )
    records = [json.loads(line) for line in args.benchmark.read_text().splitlines()
               if line.strip()]
    row = next((item for item in records if str(item["id"]) == args.prompt_id), None)
    if row is None:
        raise ValueError(f"prompt {args.prompt_id!r} not found in {args.benchmark}")
    prefill_ids = loaded.tokenizer(row["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    # BNB's normal prefill establishes FP16 bias storage for Linear4bit modules
    # before the fixed-row adapter takes over decode calls.
    with torch.inference_mode():
        model(prefill_ids, use_cache=False)
    torch.cuda.synchronize()
    layer = (model.model.layers[0].self_attn.q_proj if args.layer == "q_proj"
             else model.model.layers[0].mlp.down_proj)
    if args.inspect_only:
        print(json.dumps({
            "layer": args.layer,
            "layer_class": f"{type(layer).__module__}.{type(layer).__qualname__}",
            "is_linear4bit": isinstance(layer, Linear4bit),
            "bias_dtype": None if layer.bias is None else str(layer.bias.dtype),
            "bias_device": None if layer.bias is None else str(layer.bias.device),
            "weight_dtype": str(layer.weight.dtype),
            "weight_device": str(layer.weight.device),
            "compute_dtype": str(getattr(layer, "compute_dtype", None)),
            "configured_model_dtype": str(getattr(model.config, "torch_dtype", None)),
            "prefill_prompt_id": args.prompt_id,
            "prefill_prompt_tokens": int(prefill_ids.shape[1]),
            "bias_dtype_after_prefill": None if layer.bias is None else str(layer.bias.dtype),
            "contract_id": verified.contract_id,
        }, indent=2), flush=True)
        return
    torch.manual_seed(9061)
    torch.cuda.manual_seed_all(9061)
    inputs = torch.randn(
        (1, args.rows, layer.in_features), device="cuda:0", dtype=torch.float16
    )
    with torch.inference_mode():
        with fixed_row_schedule_for_model(
            model,
            physical_rows=2,
            attention_strategy="triton_batched",
            quantized_linear_strategy=(
                "triton_nf4_group_scale_packed_rows2_n16_warp1_k256"
            ),
        ):
            for _ in range(3):
                output = layer(inputs)
            torch.cuda.synchronize()
            output_digest = hashlib.sha256(
                output.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            ).hexdigest()
            layer(inputs)
            torch.cuda.synchronize()

    payload = {
        "status": "single exact-schedule launch emitted after three warmups; NCU owns timing",
        "model": MODEL,
        "revision": REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "layer": args.layer,
        "input_shape": list(inputs.shape),
        "input_dtype": str(inputs.dtype),
        "output_shape": list(output.shape),
        "output_sha256": output_digest,
        "bias_dtype_after_hooked_call": None if layer.bias is None else str(layer.bias.dtype),
        "contract_id": verified.contract_id,
        "prefill_prompt_id": args.prompt_id,
        "prefill_prompt_tokens": int(prefill_ids.shape[1]),
        "physical_rows": 2,
        "block_n": 16,
        "block_k": 256,
        "num_warps": 1,
        "fp_fusion": False,
        "kernel_source": str(KERNEL_SOURCE),
        "kernel_source_sha256": hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest(),
        "ncu_capture_instruction": "profile the fourth matching _packed_two_rows_kernel launch after three warmups",
    }
    args.output_metadata.parent.mkdir(parents=True, exist_ok=True)
    args.output_metadata.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
