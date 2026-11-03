#!/usr/bin/env python3
"""Emulate the per-lane fp32 accumulation orders of BNB kgemm_4bit_inference_naive and the Triton M=3 kernel (development only).

Both kernels use 32 lanes; lane l owns elements [g*1024 + l*32, +32) of each 1024-wide group g; per-lane sums are
combined with a 16/8/4/2/1 shuffle tree. BNB accumulates the 32 fp16 products one at a time; the Triton kernel adds
adjacent pairs first. This script compares both emulations with the real kernels' outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_bnb_nf4_activationvec_block_integration import PROJECTIONS, module_for, triton_m3_overrides


def lane_sums(prod: torch.Tensor, pairwise: bool) -> torch.Tensor:
    n, k = prod.shape
    groups = (k + 1023) // 1024
    padded = torch.zeros(n, groups * 1024, device=prod.device, dtype=torch.float16)
    padded[:, :k] = prod
    p = padded.view(n, groups, 32, 32)
    acc = torch.zeros(n, 32, device=prod.device, dtype=torch.float32)
    for g in range(groups):
        if pairwise:
            for j in range(16):
                acc = acc + (p[:, g, :, 2 * j].float() + p[:, g, :, 2 * j + 1].float())
        else:
            for e in range(32):
                acc = acc + p[:, g, :, e].float()
    return acc


def tree(acc: torch.Tensor) -> torch.Tensor:
    lanes = torch.arange(32, device=acc.device)
    for shift in (16, 8, 4, 2, 1):
        src = (lanes + shift) & 31
        acc = torch.where(lanes < 32 - shift, acc + acc[:, src], acc)
    return acc[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 18, 35])
    ap.add_argument("--draws", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa").model.eval()
    device = "cuda:0"
    gen = torch.Generator(device=device).manual_seed(1)
    rows, agg = [], {"elements": 0, "bnb_vs_seq": 0, "bnb_vs_pair": 0, "triton_vs_seq": 0, "triton_vs_pair": 0,
                     "seq_vs_pair_fp16": 0, "seq_vs_pair_fp32_lane0": 0}
    with torch.inference_mode():
        for li in args.layers:
            for proj in PROJECTIONS:
                module = module_for(model.model.layers[li], proj)
                k, n = int(module.in_features), int(module.out_features)
                weight = module.weight
                st = weight.quant_state
                q = weight.data.view(-1)
                codes = torch.stack([q >> 4, q & 15], dim=-1).view(n, k).long()
                c16 = st.code.to(torch.float16)
                scale_idx = torch.arange(n * (k // 64), device=device)
                scale_f = torch.addcmul(st.offset.float(), st.state2.code[st.absmax.long()],
                                        st.state2.absmax[scale_idx >> 8])
                scale16 = scale_f.to(torch.float16).view(n, k // 64, 1).expand(n, k // 64, 64).reshape(n, k)
                w = (c16[codes].float() * scale16.float()).to(torch.float16)  # (n, k)
                for _ in range(args.draws):
                    x = (torch.randn(1, k, device=device, generator=gen)
                         * torch.exp(torch.randn(1, k, device=device, generator=gen))).to(torch.float16)
                    bnb = module(x.view(1, 1, k)).view(-1)
                    x3 = x.view(1, 1, k).expand(1, 3, k).contiguous()
                    with triton_m3_overrides(model):
                        tri = module(x3)[0, 0]
                    prod = (w * x).to(torch.float16)  # fp16 product, round-to-nearest
                    seq32 = tree(lane_sums(prod, False))
                    pair32 = tree(lane_sums(prod, True))
                    seq, pair = seq32.to(torch.float16), pair32.to(torch.float16)
                    if module.bias is not None:
                        seq = (seq32 + module.bias.float()).to(torch.float16)
                        pair = (pair32 + module.bias.float()).to(torch.float16)
                    agg["elements"] += n
                    agg["bnb_vs_seq"] += int((bnb != seq).sum()); agg["bnb_vs_pair"] += int((bnb != pair).sum())
                    agg["triton_vs_seq"] += int((tri != seq).sum()); agg["triton_vs_pair"] += int((tri != pair).sum())
                    agg["seq_vs_pair_fp16"] += int((seq != pair).sum()); agg["seq_vs_pair_fp32_lane0"] += int((seq32 != pair32).sum())
            print(json.dumps({"layer": li, **agg}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "zassd-nf4-gemv-order-emulation-v1", "layers": args.layers, "draws": args.draws, **agg}, indent=2) + "\n")


if __name__ == "__main__":
    main()
