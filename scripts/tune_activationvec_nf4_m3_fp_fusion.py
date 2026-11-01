#!/usr/bin/env python3
"""Exactness-gated screen of FP32 instruction fusion for the NF4 M=3 kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import tune_activationvec_nf4_m3_launch as sweep


def run_fp_fusion(x_raw, module, state, kernel_args, config):
    out_tile, num_warps = config
    n, k = int(module.out_features), int(module.in_features)
    x = x_raw.to(module.compute_dtype).contiguous()
    output = sweep.torch.empty((3, n), device=x.device, dtype=sweep.torch.float16)
    grid = (sweep.triton.cdiv(n, out_tile),)
    sweep._fused_nf4_m3_kernel[grid](
        x, *kernel_args, output, k, n, int(state.blocksize), module.bias is not None,
        OUT_TILE=out_tile, num_warps=num_warps, enable_fp_fusion=True,
    )
    return output.to(x_raw.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    # Keep the launch geometry fixed; this experiment changes only the compiler
    # FP fusion flag and gates all latency measurements on bitwise BNB equality.
    sweep.CONFIGS = ((8, 8),)
    sweep.run_kernel = run_fp_fusion
    sweep.main()

    payload = json.loads(args.output.read_text())
    payload["classification"] = (
        "single-prompt operator screen; FP32 fusion candidate latency is measured "
        "only after bitwise equality to BNB M=3"
    )
    payload["arithmetic_schedule"] = {
        "baseline": "enable_fp_fusion=False",
        "candidate": "enable_fp_fusion=True",
        "launch_geometry": {"out_tile": 8, "num_warps": 8},
        "acceptance_gate": "candidate must match native BNB M=3 bitwise for each projection",
    }
    payload["source_hashes"]["fp_fusion_driver"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompt_id": payload["prompt_id"],
                      "arithmetic_schedule": payload["arithmetic_schedule"],
                      "results": payload["results"], "output": str(args.output)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
