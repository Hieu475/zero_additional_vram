#!/usr/bin/env python3
"""Exactness-gated screen replacing Triton gather reduction with warp shuffles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2 as kernel_base
import tune_activationvec_nf4_m3_launch as sweep


REDUCTION_OLD = '''for offset_idx in tl.static_range(5):
        shift = 16 >> offset_idx
        src = tl.broadcast_to((lanes + shift) & 31, (1, OUT_TILE, 32))
        acc0 = tl.where(lanes < 32 - shift, acc0 + tl.gather(acc0, src, axis=2), acc0)
        acc1 = tl.where(lanes < 32 - shift, acc1 + tl.gather(acc1, src, axis=2), acc1)
        acc2 = tl.where(lanes < 32 - shift, acc2 + tl.gather(acc2, src, axis=2), acc2)'''

REDUCTION_NEW = '''for offset_idx in tl.static_range(5):
        shift = 16 >> offset_idx
        acc0 = acc0 + _shfl_xor_f32(acc0, shift)
        acc1 = acc1 + _shfl_xor_f32(acc1, shift)
        acc2 = acc2 + _shfl_xor_f32(acc2, shift)'''

SHUFFLE_HELPER = '''@triton.jit
def _shfl_xor_f32(value, LANE_MASK: tl.constexpr):
    mask = tl.full(value.shape, LANE_MASK, tl.int32)
    bits = tl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
        "=r,r,r",
        [value.to(tl.uint32, bitcast=True), mask],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return bits.to(tl.float32, bitcast=True)
'''


def build_kernel():
    source = sweep._fused_nf4_m3_kernel.src
    if source.count(REDUCTION_OLD) != 1:
        raise RuntimeError("expected one staged tl.gather reduction block")
    kernel_source = source.replace(
        "def _fused_nf4_m3_kernel(", "def _fused_nf4_m3_shfl_kernel(", 1
    ).replace(REDUCTION_OLD, REDUCTION_NEW, 1)
    generated = (
        "import triton\nimport triton.language as tl\n\n"
        + "@triton.jit\n" + kernel_base._ldcs_v4.src + "\n\n"
        + SHUFFLE_HELPER + "\n@triton.jit\n" + kernel_source + "\n"
    )
    path = Path(__file__).with_name("_nf4_activationvec_warp_shuffle_kernel.py")
    if path.exists() and path.read_text() != generated:
        raise RuntimeError(f"generated shuffle kernel differs: {path}")
    if not path.exists():
        path.write_text(generated)
    spec = importlib.util.spec_from_file_location("_nf4_activationvec_warp_shuffle_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import generated warp-shuffle kernel")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._fused_nf4_m3_shfl_kernel, generated


SHFL_KERNEL = None


def run_shuffle(x_raw, module, state, kernel_args, config):
    out_tile, num_warps = config
    n, k = int(module.out_features), int(module.in_features)
    x = x_raw.to(module.compute_dtype).contiguous()
    output = sweep.torch.empty((3, n), device=x.device, dtype=sweep.torch.float16)
    grid = (sweep.triton.cdiv(n, out_tile),)
    SHFL_KERNEL[grid](
        x, *kernel_args, output, k, n, int(state.blocksize), module.bias is not None,
        OUT_TILE=out_tile, num_warps=num_warps, enable_fp_fusion=False,
    )
    return output.to(x_raw.dtype)


def main():
    global SHFL_KERNEL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(sys.argv[1:])
    if args.output.exists():
        raise FileExistsError(args.output)
    SHFL_KERNEL, generated = build_kernel()
    sweep.CONFIGS = ((8, 8),)
    sweep.run_kernel = run_shuffle
    sweep.main()
    payload = json.loads(args.output.read_text())
    payload["classification"] = (
        "paired operator screen; staged reduction order retained, warp-shuffle candidate measured only after BNB bitwise gate"
    )
    payload["reduction_schedule"] = {
        "baseline": "five tl.gather stages at xor distances 16,8,4,2,1 with invalid-lane guards",
        "candidate": "five shfl.sync.bfly stages at xor distances 16,8,4,2,1",
        "arithmetic": "FP fusion disabled; additions kept in the same staged order",
        "launch_geometry": {"out_tile": 8, "num_warps": 8},
    }
    payload.setdefault("source_hashes", {})
    payload["source_hashes"]["base_kernel"] = hashlib.sha256(
        Path(kernel_base.__file__).read_bytes()
    ).hexdigest()
    payload["source_hashes"]["warp_shuffle_driver"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    payload["source_hashes"]["generated_warp_shuffle_kernel"] = hashlib.sha256(
        generated.encode()
    ).hexdigest()
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"prompt_id": payload["prompt_id"], "results": payload["results"],
                      "reduction_schedule": payload["reduction_schedule"],
                      "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
