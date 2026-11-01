"""Launch one Qwen2.5-3B Triton NF4 projection kernel for Nsight profiling."""

import argparse
import torch

from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_rows import nf4_fixed_rows_linear
from zassd.certification.nf4_fixed_rows_group_scale import nf4_fixed_rows_group_scale_linear
from zassd.certification.nf4_fixed_rows_group_scale_packed import (
    nf4_fixed_rows_group_scale_packed_linear,
)
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_pipeline_sweep import (
    nf4_fixed_rows_group_scale_packed_rows_linear,
)
from zassd.certification.nf4_fixed_rows_group_scale_packed_rows_pairwise import (
    nf4_fixed_rows_group_scale_packed_rows_linear as nf4_pairwise_linear,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=("triton_nf4", "group_scale", "group_scale_k256",
                 "group_scale_packed_k256", "packed_rows2_n16_k256",
                 "pairwise_rows2_n16_k256"),
        default="triton_nf4",
    )
    parser.add_argument("--layer", choices=("q_proj", "down_proj"), default="q_proj")
    parser.add_argument("--physical-rows", type=int, default=5)
    parser.add_argument("--rows", type=int, default=None,
                        help="Actual input rows; defaults to physical-rows")
    parser.add_argument("--block-k", type=int, default=256)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    args = parser.parse_args()
    rows = args.physical_rows if args.rows is None else args.rows
    if rows < 1 or rows > args.physical_rows:
        raise ValueError("rows must be within 1..physical-rows")
    loaded = load_backend(
        "hf_bnb_nf4",
        "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "cuda:0",
        "sdpa",
        model_dtype=torch.float16,
    )
    model = loaded.model.eval()
    layer = (model.model.layers[0].self_attn.q_proj if args.layer == "q_proj"
             else model.model.layers[0].mlp.down_proj)
    x = torch.randn((1, rows, layer.in_features), device="cuda:0", dtype=torch.float16)
    with torch.inference_mode():
        if args.strategy == "pairwise_rows2_n16_k256":
            nf4_pairwise_linear(
                x, layer, physical_rows=args.physical_rows, block_n=16,
                block_k=args.block_k, num_warps=args.num_warps,
            )
        elif args.strategy == "group_scale_packed_k256":
            nf4_fixed_rows_group_scale_packed_linear(
                x, layer, physical_rows=args.physical_rows, block_n=8, block_k=256
            )
        elif args.strategy == "packed_rows2_n16_k256":
            nf4_fixed_rows_group_scale_packed_rows_linear(
                x, layer, physical_rows=args.physical_rows, block_n=16,
                block_k=args.block_k, num_warps=args.num_warps,
                num_stages=args.num_stages,
            )
        elif args.strategy.startswith("group_scale"):
            nf4_fixed_rows_group_scale_linear(
                x,
                layer,
                physical_rows=args.physical_rows,
                block_n=8,
                block_k=256 if args.strategy.endswith("k256") else 128,
            )
        else:
            nf4_fixed_rows_linear(x, layer, physical_rows=args.physical_rows)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
