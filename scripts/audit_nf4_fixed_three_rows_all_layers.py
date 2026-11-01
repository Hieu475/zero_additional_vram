#!/usr/bin/env python3
"""Finite all-layer shape-invariance audit for the fixed-three-row NF4 prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_warp_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear as pairwise_linear,
)
from zassd.certification.nf4_fixed_three_rows_group_scale_packed import (
    nf4_fixed_three_rows_group_scale_packed_linear as three_rows_linear,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa",
        model_dtype=torch.float16,
    )
    model = loaded.model.eval()
    torch.manual_seed(50813)
    torch.cuda.manual_seed_all(50813)
    rows = []
    failures = []
    with torch.inference_mode():
        for layer_index, layer in enumerate(model.model.layers):
            modules = {
                "q_proj": layer.self_attn.q_proj,
                "k_proj": layer.self_attn.k_proj,
                "v_proj": layer.self_attn.v_proj,
                "o_proj": layer.self_attn.o_proj,
                "gate_proj": layer.mlp.gate_proj,
                "up_proj": layer.mlp.up_proj,
                "down_proj": layer.mlp.down_proj,
            }
            for module_name, module in modules.items():
                x = torch.randn((1, 3, module.in_features), device="cuda:0", dtype=torch.float16)
                candidate_outputs = {}
                shape_checks = {}
                for m in (1, 2, 3):
                    x_m = x[:, :m].contiguous()
                    output = three_rows_linear(x_m, module)
                    candidate_outputs[m] = output.detach()
                    singleton_rows = [
                        three_rows_linear(x[:, row:row + 1].contiguous(), module).detach()
                        for row in range(m)
                    ]
                    shape_checks[str(m)] = [
                        torch.equal(output[:, row:row + 1], singleton_rows[row])
                        for row in range(m)
                    ]
                pairwise = pairwise_linear(
                    x, module, physical_rows=3, block_n=16, block_k=256, num_warps=1
                )
                record = {
                    "layer": layer_index,
                    "module": module_name,
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "m1_m2_m3_rowwise_singleton_equal": shape_checks,
                    "m3_bitwise_equal_to_pairwise_reference": torch.equal(candidate_outputs[3], pairwise),
                    "max_abs_diff_vs_pairwise": float(
                        (candidate_outputs[3].float() - pairwise.float()).abs().max().item()
                    ),
                }
                rows.append(record)
                if not all(all(checks) for checks in shape_checks.values()) or not record["m3_bitwise_equal_to_pairwise_reference"]:
                    failures.append(record)
            print(f"audited layer {layer_index + 1}/{len(model.model.layers)}", flush=True)
    payload = {
        "schema": "zassd-nf4-fixed-three-rows-all-layer-audit-v1",
        "status": "finite random-activation operator audit; not a proof or runtime integration",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": __import__("importlib.metadata", fromlist=["version"]).version("triton"),
        "source_sha256": {
            "kernel": hashlib.sha256(Path("src/zassd/certification/nf4_fixed_three_rows_group_scale_packed.py").read_bytes()).hexdigest(),
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "modules_checked": len(rows),
        "all_shape_and_reference_checks_pass": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "results": rows,
        "limits": [
            "One random M=3 activation tensor per Linear4bit module.",
            "Checks the prototype against its own singleton calls at M=1/2/3 and against the current pairwise kernel at M=3.",
            "Does not cover model-generated activations, attention, KV state, decoder branches, or end-to-end latency.",
            "GSM8K held-out indices 300-349 were not used.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "modules_checked": len(rows),
        "all_shape_and_reference_checks_pass": not failures,
        "failure_count": len(failures),
    }, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
