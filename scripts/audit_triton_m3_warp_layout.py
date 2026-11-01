#!/usr/bin/env python3
"""Record how a Triton [1, OUT_TILE, 32] tensor maps to physical warp lanes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _record_layout(out_lane_ptr, out_warp_ptr, out_shuffle_ptr, OUT_TILE: tl.constexpr):
    outs = tl.arange(0, OUT_TILE)[None, :, None]
    lanes = tl.arange(0, 32)[None, None, :]
    linear = tl.broadcast_to(outs * 32 + lanes, (1, OUT_TILE, 32))
    physical_lane, physical_warp = tl.inline_asm_elementwise(
        "mov.u32 $0, %laneid; mov.u32 $1, %warpid;",
        "=r,=r,r",
        [linear.to(tl.int32)],
        dtype=(tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    tl.store(out_lane_ptr + linear, physical_lane)
    tl.store(out_warp_ptr + linear, physical_warp)
    lane_values = lanes.to(tl.float32)
    for stage in tl.static_range(5):
        shift = 16 >> stage
        mask = tl.full(lane_values.shape, shift, tl.int32)
        shuffled = tl.inline_asm_elementwise(
            "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
            "=r,r,r",
            [lane_values.to(tl.uint32, bitcast=True), mask],
            dtype=tl.uint32,
            is_pure=False,
            pack=1,
        ).to(tl.float32, bitcast=True)
        tl.store(out_shuffle_ptr + stage * OUT_TILE * 32 + linear, shuffled)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-tile", type=int, default=8)
    ap.add_argument("--num-warps", type=int, default=8)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.out_tile * 32 != args.num_warps * 32:
        raise ValueError("probe expects one tensor element per thread: OUT_TILE == num_warps")

    lanes = torch.empty((args.out_tile, 32), dtype=torch.int32, device="cuda")
    warps = torch.empty_like(lanes)
    shuffled = torch.empty((5, args.out_tile, 32), dtype=torch.float32, device="cuda")
    _record_layout[(1,)](lanes, warps, shuffled, OUT_TILE=args.out_tile, num_warps=args.num_warps)
    torch.cuda.synchronize()
    lane_rows = lanes.cpu().tolist()
    warp_rows = warps.cpu().tolist()
    shuffle_rows = shuffled.cpu().tolist()
    expected_shuffle = [
        [[float(lane ^ (16 >> stage)) for lane in range(32)] for _ in range(args.out_tile)]
        for stage in range(5)
    ]
    payload = {
        "schema": "zassd-triton-warp-layout-probe-v1",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "shape": [1, args.out_tile, 32],
        "num_warps": args.num_warps,
        "physical_lane_by_logical_output_and_reduction_index": lane_rows,
        "physical_warp_by_logical_output_and_reduction_index": warp_rows,
        "shuffle_result_by_stage_output_and_reduction_index": shuffle_rows,
        "shuffle_xor_expected_by_stage_output_and_reduction_index": expected_shuffle,
        "lane_axis_identity": all(row == list(range(32)) for row in lane_rows),
        "warp_axis_constant_per_output": all(len(set(row)) == 1 for row in warp_rows),
        "ptx_bfly_matches_xor": shuffle_rows == expected_shuffle,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
