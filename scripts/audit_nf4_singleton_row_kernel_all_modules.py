"""Finite all-module M=1 equality screen for a one-row NF4 kernel prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear as packed_two_row,
)
from zassd.certification.nf4_fixed_row_singleton_group_scale_packed import (
    nf4_fixed_row_singleton_group_scale_packed_rows_linear as packed_one_row,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmarks/cnndm_eval.jsonl")
    parser.add_argument("--prompt-id", default="cnndm_003")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in records if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt {args.prompt_id!r} not found")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    loaded = load_backend("hf_bnb_nf4", model_name, revision, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    input_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    with torch.inference_mode():
        # Exercise every quantized layer through BNB first so its lazy bias cast is initialized.
        model(input_ids, use_cache=False)
        modules = [
            (name, module)
            for name, module in model.named_modules()
            if module.__class__.__name__ == "Linear4bit"
        ]
        rows = []
        torch.manual_seed(77109)
        torch.cuda.manual_seed_all(77109)
        for name, module in modules:
            x = torch.randn((1, 1, int(module.in_features)), device="cuda:0", dtype=torch.float16)
            old = packed_two_row(
                x, module, physical_rows=2, block_n=16, block_k=256, num_warps=1
            )
            new = packed_one_row(x, module, block_n=16, block_k=256, num_warps=1)
            torch.cuda.synchronize()
            rows.append({
                "module": name,
                "in_features": int(module.in_features),
                "out_features": int(module.out_features),
                "bias_dtype": str(module.bias.dtype) if module.bias is not None else None,
                "bitwise_equal": torch.equal(old, new),
                "max_abs_diff": float((old - new).abs().max()),
                "input_sha256": hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            })
    payload = {
        "status": "finite all-module operator screen; random M=1 inputs, not a formal proof",
        "model": model_name,
        "revision": revision,
        "prompt_id_for_lazy_initialization": args.prompt_id,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": __import__("triton").__version__,
        "strategy": {"block_n": 16, "block_k": 256, "num_warps": 1},
        "compared_modules": len(rows),
        "all_bitwise_equal": all(row["bitwise_equal"] for row in rows),
        "mismatch_count": sum(not row["bitwise_equal"] for row in rows),
        "modules": rows,
        "limitations": [
            "Random per-module activations; does not prove equality on all floating-point inputs.",
            "Does not establish whole-model closed-loop equivalence or BNB equivalence.",
            "Held-out GSM8K indices 300-349 were not used.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "status", "model", "revision", "prompt_id_for_lazy_initialization", "compared_modules",
        "all_bitwise_equal", "mismatch_count", "strategy"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
