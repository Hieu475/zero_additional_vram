#!/usr/bin/env python3
"""Compare BNB and Triton SDPA on the actual first forced decode transition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
import audit_triton_b1_closed_loop as runner
from audit_actual_fixed_three_rows_bnb_first_decode import fixed_three_row_linear_schedule


REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    af, bf = a.float(), b.float()
    delta = af - bf
    denom = torch.linalg.vector_norm(af).clamp_min(1e-30)
    return {
        "shape_a": list(a.shape), "shape_b": list(b.shape),
        "dtype_a": str(a.dtype), "dtype_b": str(b.dtype),
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs_diff": float(delta.abs().max().item()),
        "mean_abs_diff": float(delta.abs().mean().item()),
        "rms_diff": float(delta.square().mean().sqrt().item()),
        "relative_l2_diff": float(torch.linalg.vector_norm(delta).div(denom).item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="append", required=True)
    ap.add_argument("--artifact", action="append", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    source_name = "scripts/audit_actual_fixed_three_rows_sdpa_inputs.py"
    if manifest.get("source_sha256", {}).get(source_name) != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("SDPA audit source is not pinned by the experiment manifest")
    prompts = {}
    for filename in args.benchmark:
        prompts.update({str(row["id"]): row["prompt"] for row in
                        (json.loads(line) for line in Path(filename).read_text().splitlines() if line.strip())})
    loaded = load_backend("hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct", REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    old_sdpa = F.scaled_dot_product_attention
    old_tiled = runner.fixed_row_attention_tiled
    details = []
    try:
        for filename in args.artifact:
            artifact = Path(filename)
            if not artifact.is_absolute():
                artifact = repo / artifact
            saved = json.loads(artifact.read_text())
            pid = str(saved["prompt_id"])
            token_id = int(saved["timing"]["paired_rounds"][0]["methods"]["original_bnb_greedy"]["token_ids"][0])
            ids = tokenizer(prompts[pid], return_tensors="pt")["input_ids"].to("cuda:0")
            cap = int(ids.shape[-1]) + 16
            ref_cache = TargetKVCache(backend="static", max_capacity=cap)
            fix_cache = TargetKVCache(backend="static", max_capacity=cap)
            with torch.no_grad():
                model(ids, past_key_values=ref_cache.cache, use_cache=True)
                model(ids, past_key_values=fix_cache.cache, use_cache=True)
            token = torch.tensor([[token_id]], device="cuda:0")
            native_records, tiled_records = [], []

            def native_spy(q, k, v, *pos, **kw):
                out = old_sdpa(q, k, v, *pos, **kw)
                native_records.append({"q": q.detach().clone(), "k": k.detach().clone(),
                                       "v": v.detach().clone(), "out": out.detach().clone(),
                                       "kwargs": {str(key): repr(value) for key, value in kw.items()}})
                return out

            F.scaled_dot_product_attention = native_spy
            try:
                with torch.no_grad():
                    model(token, past_key_values=ref_cache.cache, use_cache=True)
            finally:
                F.scaled_dot_product_attention = old_sdpa

            def tiled_spy(q, k, v, scale, block_k=16, physical_rows=None):
                out = old_tiled(q, k, v, scale, block_k=block_k, physical_rows=physical_rows)
                tiled_records.append({"q": q.detach().clone(), "k": k.detach().clone(),
                                     "v": v.detach().clone(), "out": out.detach().clone(),
                                     "scale": float(scale), "block_k": int(block_k),
                                     "physical_rows": physical_rows})
                return out

            runner.fixed_row_attention_tiled = tiled_spy
            try:
                with (fixed_three_row_linear_schedule(model),
                      runner.fixed_row_sdpa_decode(model, physical_rows=3, block_k=32,
                                                   attention_strategy="triton_batched"),
                      torch.no_grad()):
                    model(token, past_key_values=fix_cache.cache, use_cache=True)
            finally:
                runner.fixed_row_attention_tiled = old_tiled

            if len(native_records) != len(tiled_records):
                raise RuntimeError(f"SDPA call count differs for {pid}: {len(native_records)} vs {len(tiled_records)}")
            per_layer = []
            for layer, (native, tiled) in enumerate(zip(native_records, tiled_records)):
                nk = native["k"].repeat_interleave(native["q"].shape[1] // native["k"].shape[1], dim=1)
                nv = native["v"].repeat_interleave(native["q"].shape[1] // native["v"].shape[1], dim=1)
                qdiff = metrics(native["q"], tiled["q"])
                kdiff = metrics(nk, tiled["k"])
                vdiff = metrics(nv, tiled["v"])
                odiff = metrics(native["out"], tiled["out"])
                # Re-evaluate both implementations on precisely the captured BNB operands.
                native_same = native["out"]
                k_repeated = native["k"].repeat_interleave(
                    native["q"].shape[1] // native["k"].shape[1], dim=1
                )
                v_repeated = native["v"].repeat_interleave(
                    native["q"].shape[1] // native["v"].shape[1], dim=1
                )
                scale = tiled["scale"]
                triton_same = old_tiled(native["q"], k_repeated, v_repeated, scale,
                                        block_k=32, physical_rows=None)
                same_input_diff = metrics(native_same, triton_same)
                per_layer.append({
                    "layer": layer,
                    "native_shapes": {name: list(native[name].shape) for name in ("q", "k", "v", "out")},
                    "triton_shapes": {name: list(tiled[name].shape) for name in ("q", "k", "v", "out")},
                    "q_same_after_capture": qdiff,
                    "k_same_after_gqa_expansion": kdiff,
                    "v_same_after_gqa_expansion": vdiff,
                    "actual_forward_output_diff": odiff,
                    "triton_vs_bnb_native_on_same_bnb_qkv": same_input_diff,
                    "triton_scale": scale,
                    "native_kwargs": native["kwargs"],
                })
            details.append({
                "prompt_id": pid, "forced_token_id": token_id,
                "sdpa_calls": len(per_layer), "layers": per_layer,
                "all_q_equal": all(x["q_same_after_capture"]["bitwise_equal"] for x in per_layer),
                "all_k_equal_after_gqa": all(x["k_same_after_gqa_expansion"]["bitwise_equal"] for x in per_layer),
                "all_v_equal_after_gqa": all(x["v_same_after_gqa_expansion"]["bitwise_equal"] for x in per_layer),
                "first_qkv_input_difference_layer": next((x["layer"] for x in per_layer if not (
                    x["q_same_after_capture"]["bitwise_equal"] and
                    x["k_same_after_gqa_expansion"]["bitwise_equal"] and
                    x["v_same_after_gqa_expansion"]["bitwise_equal"])), None),
                "first_native_vs_triton_output_difference_layer": next((x["layer"] for x in per_layer if not x["actual_forward_output_diff"]["bitwise_equal"]), None),
            })
            print(json.dumps({k: v for k, v in details[-1].items() if k != "layers"}, indent=2), flush=True)
    finally:
        F.scaled_dot_product_attention = old_sdpa
        runner.fixed_row_attention_tiled = old_tiled

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({
        "classification": "same-transition SDPA operand/kernel comparison; finite empirical diagnostic",
        "model": "Qwen/Qwen2.5-3B-Instruct", "revision": REVISION,
        "gpu": torch.cuda.get_device_name(0),
        "native": "PyTorch SDPA BNB singleton GQA call",
        "candidate": "Triton fixed_row_attention_tiled, block_k=32, fixed-three-row schedule",
        "details": details,
        "limitations": ["Two development prompts only; no held-out data used.",
                        "Does not establish cross-GPU or cross-runtime equivalence."],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
