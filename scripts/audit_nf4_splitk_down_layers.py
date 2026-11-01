"""Check singleton-row invariance of split-K NF4 on every Qwen2 down_proj."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_splitk import (
    nf4_fixed_rows_group_scale_packed_rows_linear,
)


def main() -> None:
    output = Path("experiments/exact_quantized_feasibility/triton_nf4_packed_rows2_splitk_v1/down_layers_shape_audit.json")
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa",
        model_dtype=torch.float16,
    )
    model = loaded.model.eval()
    layers = [layer.mlp.down_proj for layer in model.model.layers]
    torch.manual_seed(2031)
    torch.cuda.manual_seed_all(2031)
    layer_results = []
    with torch.inference_mode():
        for layer_idx, layer in enumerate(layers):
            inputs = torch.randn((1, 2, layer.in_features), device="cuda:0", dtype=torch.float16)
            block = nf4_fixed_rows_group_scale_packed_rows_linear(
                inputs, layer, physical_rows=2, block_n=16, block_k=256,
                num_warps=1, split_parts=8,
            )
            singleton = [
                nf4_fixed_rows_group_scale_packed_rows_linear(
                    inputs[:, row:row + 1].contiguous(), layer,
                    physical_rows=2, block_n=16, block_k=256,
                    num_warps=1, split_parts=8,
                )
                for row in range(2)
            ]
            equal_rows = [
                bool(torch.equal(block[:, row:row + 1], singleton[row]))
                for row in range(2)
            ]
            layer_results.append({
                "layer": layer_idx,
                "in_features": int(layer.in_features),
                "out_features": int(layer.out_features),
                "rowwise_singleton_equal": equal_rows,
                "max_abs_diff": [
                    float((block[:, row:row + 1].float() - singleton[row].float()).abs().max().item())
                    for row in range(2)
                ],
            })
    torch.cuda.synchronize()
    payload = {
        "status": "finite shape-invariance audit only; not a whole-model or BNB-equivalence proof",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": importlib.metadata.version("triton"),
        "schedule": {"block_n": 16, "block_k": 256, "num_warps": 1, "split_parts": 8},
        "layers_checked": len(layer_results),
        "all_rows_equal": all(all(row["rowwise_singleton_equal"]) for row in layer_results),
        "results": layer_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
