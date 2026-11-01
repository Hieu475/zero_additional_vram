#!/usr/bin/env python3
"""Compare fixed-three-row NF4 singleton rows with the prior one-row kernel."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from zassd.backends.base import load_backend
from zassd.certification.nf4_fixed_row_singleton_group_scale_packed import (
    nf4_fixed_row_singleton_group_scale_packed_rows_linear as singleton_linear,
)
from zassd.certification.nf4_fixed_three_rows_group_scale_packed import (
    nf4_fixed_three_rows_group_scale_packed_linear as three_rows_linear,
)


def main() -> None:
    loaded = load_backend(
        "hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa",
        model_dtype=torch.float16,
    )
    model = loaded.model.eval()
    torch.manual_seed(50813)
    torch.cuda.manual_seed_all(50813)
    results=[]
    with torch.inference_mode():
        for layer_index, layer in enumerate(model.model.layers):
            modules={
                "q_proj":layer.self_attn.q_proj,"k_proj":layer.self_attn.k_proj,
                "v_proj":layer.self_attn.v_proj,"o_proj":layer.self_attn.o_proj,
                "gate_proj":layer.mlp.gate_proj,"up_proj":layer.mlp.up_proj,
                "down_proj":layer.mlp.down_proj,
            }
            for name,module in modules.items():
                x=torch.randn((1,3,module.in_features),device="cuda:0",dtype=torch.float16)
                old=singleton_linear(x[:,:1].contiguous(),module,block_n=16,block_k=256,num_warps=1)
                new_m1=three_rows_linear(x[:,:1].contiguous(),module,block_n=16,block_k=256,num_warps=1)
                new_m3=three_rows_linear(x,module,block_n=16,block_k=256,num_warps=1)[:,:1]
                results.append({
                    "layer":layer_index,"module":name,
                    "old_one_row_equals_three_row_m1":torch.equal(old,new_m1),
                    "old_one_row_equals_three_row_m3_row0":torch.equal(old,new_m3),
                    "max_abs_diff_m1":float((old.float()-new_m1.float()).abs().max().item()),
                    "max_abs_diff_m3_row0":float((old.float()-new_m3.float()).abs().max().item()),
                })
            print(f"audited layer {layer_index+1}/{len(model.model.layers)}",flush=True)
    failures=[r for r in results if not r["old_one_row_equals_three_row_m1"] or not r["old_one_row_equals_three_row_m3_row0"]]
    out=Path("experiments/exact_quantized_feasibility/triton_nf4_singleton_k2_r3_v1/three_row_kernel/operator_cross_kernel_audit.json")
    out.write_text(json.dumps({
        "schema":"zassd-nf4-three-row-vs-singleton-all-layer-audit-v1",
        "status":"finite random-activation operator audit; not a proof",
        "model":"Qwen/Qwen2.5-3B-Instruct","revision":"aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu":torch.cuda.get_device_name(0),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
        "triton":__import__("importlib.metadata",fromlist=["version"]).version("triton"),
        "source_sha256":{
          "singleton_kernel":hashlib.sha256(Path("src/zassd/certification/nf4_fixed_row_singleton_group_scale_packed.py").read_bytes()).hexdigest(),
          "three_row_kernel":hashlib.sha256(Path("src/zassd/certification/nf4_fixed_three_rows_group_scale_packed.py").read_bytes()).hexdigest(),
          "runner":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "modules_checked":len(results),"failures":len(failures),
        "all_singleton_comparisons_bitwise_equal":not failures,
        "limits":["One random activation per projection module.","Does not cover model-generated activations, attention, or whole-sequence state."],
        "results":results,
    },indent=2)+"\n")
    print(json.dumps({"output":str(out),"modules_checked":len(results),"failures":len(failures),"all_equal":not failures},indent=2),flush=True)

if __name__=="__main__":main()
