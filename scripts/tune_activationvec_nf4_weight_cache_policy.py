#!/usr/bin/env python3
"""Screen cache-all policy for packed NF4 weight loads, leaving activations unchanged."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import tune_activationvec_nf4_cache_policy as sweep


def build_weight_ca_kernel():
    helper_source = sweep.base._ldcs_v4.src
    kernel_source = sweep.base._fused_nf4_m3_kernel.src
    kernel_source = kernel_source.replace(
        "def _fused_nf4_m3_kernel(", "def _fused_nf4_m3_weight_ca_kernel("
    )
    old = "ld.global.cs.v4.u32 {$0, $1, $2, $3}, [$4];"
    new = "ld.global.ca.v4.u32 {$0, $1, $2, $3}, [$4];"
    if kernel_source.count(old) != 1:
        raise RuntimeError("expected exactly one packed-weight load cache modifier")
    kernel_source = kernel_source.replace(old, new, 1)
    generated_source = (
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\n" + helper_source + "\n\n@triton.jit\n" + kernel_source + "\n"
    )
    generated_path = Path(__file__).with_name("_nf4_activationvec_weight_ca_kernel.py")
    if generated_path.exists() and generated_path.read_text() != generated_source:
        raise RuntimeError(f"generated weight-cache kernel differs: {generated_path}")
    if not generated_path.exists():
        generated_path.write_text(generated_source)
    spec = importlib.util.spec_from_file_location("_nf4_activationvec_weight_ca_kernel", generated_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import generated weight-cache kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fused_nf4_m3_weight_ca_kernel, generated_source


def main():
    sweep.build_ca_kernel = build_weight_ca_kernel
    sweep.main()
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[1:])
    payload = json.loads(args.output.read_text())
    payload["classification"] = (
        "paired operator screen of packed NF4 weight cache policy; exactness gated, no end-to-end claim"
    )
    payload["cache_policy"] = {
        "activation_load_policy": "ld.global.cs.v4.u32 unchanged",
        "packed_weight_candidate": "ld.global.ca.v4.u32",
        "packed_weight_baseline": "ld.global.cs.v4.u32",
        "nested_scale_policy": "unchanged",
    }
    payload["source_hashes"]["weight_cache_driver"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    payload["source_hashes"]["generated_weight_ca_kernel"] = hashlib.sha256(
        Path(__file__).with_name("_nf4_activationvec_weight_ca_kernel.py").read_bytes()
    ).hexdigest()
    payload["source_hashes"].pop("generated_ca_kernel", None)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompt_id": args.prompt_id, "results": payload["results"],
                      "cache_policy": payload["cache_policy"], "output": str(args.output)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
