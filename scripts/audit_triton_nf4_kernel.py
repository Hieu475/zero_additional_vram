"""Compile and save the actual Triton NF4 linear specialization for a Qwen2 layer."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows import (
    _nf4_fixed_rows_kernel,
    nf4_fixed_rows_linear,
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-rows", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    loaded = load_backend(
        "hf_bnb_nf4", args.model, args.revision, args.device, "sdpa",
        model_dtype=torch.float16,
    )
    model = loaded.model.eval()
    rows = args.physical_rows
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = {
        "q_proj": model.model.layers[0].self_attn.q_proj,
        "gate_proj": model.model.layers[0].mlp.gate_proj,
        "down_proj": model.model.layers[0].mlp.down_proj,
    }
    compiled_before = set()
    specializations = []
    for layer_name, layer in layers.items():
        x = torch.zeros((1, rows, layer.in_features), device=args.device, dtype=torch.float16)
        with torch.inference_mode():
            output = nf4_fixed_rows_linear(x, layer, physical_rows=rows)
        torch.cuda.synchronize(args.device)

        compiled_entries = []
        for device_cache in _nf4_fixed_rows_kernel.device_caches.values():
            cache_entries = device_cache[0] if isinstance(device_cache, tuple) else device_cache.entries
            compiled_entries.extend(cache_entries.items())
        new_entries = [(key, value) for key, value in compiled_entries if key not in compiled_before]
        compiled_before.update(key for key, _ in new_entries)
        if len(new_entries) != 1:
            raise RuntimeError(f"{layer_name}: expected one new specialization, found {len(new_entries)}")
        _, compiled = new_entries[0]
        ptx = compiled.asm["ptx"]
        cubin = compiled.asm["cubin"]
        stem = f"{layer_name}_sm89"
        ptx_path = output_dir / f"{stem}.ptx"
        cubin_path = output_dir / f"{stem}.cubin"
        ptx_path.write_text(ptx)
        cubin_path.write_bytes(cubin)

        sass_path = output_dir / f"{stem}.sass.txt"
        cuobjdump = subprocess.run(
            ["cuobjdump", "--dump-sass", str(cubin_path)],
            text=True, capture_output=True, check=False,
        )
        sass_status = (
            "cuobjdump --dump-sass succeeded"
            if cuobjdump.returncode == 0 and cuobjdump.stdout.strip()
            else f"cuobjdump failed ({cuobjdump.returncode}): {cuobjdump.stderr.strip()}"
        )
        if cuobjdump.returncode == 0 and cuobjdump.stdout.strip():
            sass_path.write_text(cuobjdump.stdout)

        metadata = compiled.metadata
        ptx_lines = [line.strip() for line in ptx.splitlines() if line.strip()]
        instruction_lines = [line for line in ptx_lines if line.endswith(";") and not line.startswith(".")]
        sass_mnemonics = re.findall(r"^\s*/\*[0-9a-f]+\*/\s+([A-Z][A-Z0-9_.]*)", cuobjdump.stdout, re.MULTILINE)
        counts = {
            "FFMA": sass_mnemonics.count("FFMA"),
            "FADD": sass_mnemonics.count("FADD"),
            "FMUL": sass_mnemonics.count("FMUL"),
            "SHFL_family": sum(name.startswith("SHFL") for name in sass_mnemonics),
            "BAR_SYNC_family": sum(name.startswith("BAR.SYNC") for name in sass_mnemonics),
            "HMMA_family": sum(name.startswith("HMMA") for name in sass_mnemonics),
        }
        specializations.append({
            "layer": {
                "q_proj": "model.layers.0.self_attn.q_proj",
                "gate_proj": "model.layers.0.mlp.gate_proj",
                "down_proj": "model.layers.0.mlp.down_proj",
            }[layer_name],
            "input_shape": [1, rows, layer.in_features],
            "output_shape": list(output.shape),
            "input_features": int(layer.in_features),
            "output_features": int(layer.out_features),
            "weight_quant_type": layer.weight.quant_state.quant_type,
            "weight_quant_blocksize": int(layer.weight.quant_state.blocksize),
            "nested_quantization": bool(layer.weight.quant_state.nested),
            "launch_grid": [rows, (layer.out_features + 15) // 16],
            "tile": {"block_n": 16, "block_k": 128, "num_warps": metadata.num_warps},
            "compiler_options": {"enable_fp_fusion": metadata.enable_fp_fusion, "num_stages": metadata.num_stages},
            "registers": getattr(compiled, "n_regs", None),
            "shared_memory_bytes": metadata.shared,
            "ptx_instruction_line_count": len(instruction_lines),
            "sass_instruction_count": len(sass_mnemonics),
            "sass_instruction_counts": counts,
            "ptx_sha256": sha256(ptx.encode()),
            "cubin_sha256": sha256(cubin),
            "sass_status": sass_status,
            "artifacts": {
                "ptx": ptx_path.name,
                "cubin": cubin_path.name,
                "sass": sass_path.name if sass_path.exists() else None,
            },
        })

    summary = {
        "status": "compiled specialization inspection; not a correctness proof",
        "model": args.model,
        "revision": args.revision,
        "backend_metadata": loaded.metadata,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "compute_capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "bitsandbytes": importlib.metadata.version("bitsandbytes"),
        "triton": metadata.triton_version,
        "target": metadata.target.arch,
        "kernel_source_sha256": sha256((Path(__file__).resolve().parents[1] / "src/zassd/certification/nf4_fixed_rows.py").read_bytes()),
        "audit_script_sha256": sha256(Path(__file__).read_bytes()),
        "physical_rows": rows,
        "specializations": specializations,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
