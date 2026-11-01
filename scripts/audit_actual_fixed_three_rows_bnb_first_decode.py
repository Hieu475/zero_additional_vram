#!/usr/bin/env python3
"""Trace module outputs on a forced BNB token under the deployed fixed-three-row schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
from bitsandbytes.nn import Linear4bit

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
import zassd.certification.fixed_row_dispatch as fixed_row_dispatch
from audit_triton_b1_closed_loop import fixed_row_sdpa_decode


STRATEGY = "triton_nf4_fixed_three_rows_group_scale_packed"


@contextmanager
def fixed_three_row_linear_schedule(model):
    original_dispatch = fixed_row_dispatch.fixed_row_schedule_for_model

    @contextmanager
    def dispatch(model, *, physical_rows=3, attention_strategy="rowwise", observer=None,
                 reuse_linear_padding_buffer=False, quantized_linear_strategy="fixed_rows"):
        with original_dispatch(
            model, physical_rows=physical_rows, attention_strategy=attention_strategy,
            observer=observer, reuse_linear_padding_buffer=reuse_linear_padding_buffer,
            quantized_linear_strategy=quantized_linear_strategy,
        ):
            restored = []
            try:
                if quantized_linear_strategy == STRATEGY:
                    for module in model.modules():
                        if not isinstance(module, Linear4bit):
                            continue
                        base = module.forward

                        def one_row_or_base(self, x, _base=base):
                            if x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == 1:
                                from zassd.certification.nf4_fixed_three_rows_group_scale_packed import (
                                    nf4_fixed_three_rows_group_scale_packed_linear,
                                )
                                return nf4_fixed_three_rows_group_scale_packed_linear(
                                    x, self, block_n=16, block_k=256, num_warps=1
                                )
                            return _base(x)

                        restored.append((module, base))
                        module.forward = MethodType(one_row_or_base, module)
                yield model
            finally:
                for module, base in reversed(restored):
                    module.forward = base

    fixed_row_dispatch.fixed_row_schedule_for_model = dispatch
    try:
        yield
    finally:
        fixed_row_dispatch.fixed_row_schedule_for_model = original_dispatch


def tensor_hash(value) -> str:
    h = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        v = value.detach().contiguous()
        h.update(f"{tuple(v.shape)}:{v.dtype}:".encode())
        h.update(v.view(torch.uint8).cpu().numpy().tobytes())
    elif isinstance(value, (tuple, list)):
        for child in value:
            h.update(tensor_hash(child).encode())
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            h.update(str(key).encode())
            h.update(tensor_hash(value[key]).encode())
    else:
        h.update(repr(value).encode())
    return h.hexdigest()


def trace_forward(model, token, cache, module_names):
    rows, handles = [], []
    for module in model.modules():
        name = module_names.get(id(module), "<unnamed>")
        is_leaf = not any(True for _ in module.children())
        def pre(_module, inputs, _name=name, _leaf=is_leaf):
            row = {"name": _name, "input": tensor_hash(inputs), "output": None,
                   "leaf": _leaf}
            if _name == "model.layers.0.self_attn.o_proj" and inputs and isinstance(inputs[0], torch.Tensor):
                row["captured_input"] = inputs[0].detach().clone()
            rows.append(row)
        def post(_module, _inputs, output, _name=name):
            for row in reversed(rows):
                if row["name"] == _name and row["output"] is None:
                    row["output"] = tensor_hash(output)
                    return
        handles.extend([module.register_forward_pre_hook(pre), module.register_forward_hook(post)])
    try:
        with torch.no_grad():
            logits = model(token, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
    finally:
        for handle in handles:
            handle.remove()
    return rows, logits


def compare_modules(reference, candidate):
    right = {row["name"]: row for row in candidate}
    differences = []
    for row in reference:
        if not row.get("leaf"):
            continue
        other = right.get(row["name"])
        if other is None or row["input"] != other["input"]:
            record = {"module": row["name"], "difference": "input",
                      "reference": row["input"],
                      "fixed_three_row": None if other is None else other["input"]}
            left = row.get("captured_input")
            right_input = None if other is None else other.get("captured_input")
            if left is not None and right_input is not None:
                record.update({"tensor_shape": list(left.shape),
                               "tensor_dtype": str(left.dtype),
                               "tensor_max_abs_diff": float((left.float() - right_input.float()).abs().max().item()),
                               "tensor_bitwise_equal": bool(torch.equal(left, right_input))})
            differences.append(record)
        if other is None or row["output"] != other["output"]:
            differences.append({"module": row["name"], "difference": "output",
                                "reference": row["output"],
                                "fixed_three_row": None if other is None else other["output"]})
    return differences


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="append", required=True)
    ap.add_argument("--artifact", action="append", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads(Path(args.manifest).read_text())
    expected_hash = manifest.get("source_sha256", {}).get("scripts/audit_actual_fixed_three_rows_bnb_first_decode.py")
    actual_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if expected_hash != actual_hash:
        raise RuntimeError("operator trace script is not pinned in the manifest")
    prompts = {}
    for filename in args.benchmark:
        prompts.update({str(row["id"]): row["prompt"] for row in
                        (json.loads(line) for line in Path(filename).read_text().splitlines() if line.strip())})
    loaded = load_backend("hf_bnb_nf4", "Qwen/Qwen2.5-3B-Instruct",
                          "aa8e72537993ba99e69dfaafa59ed015b17504d1", "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    names = {id(m): n or "<root>" for n, m in model.named_modules()}
    details = []
    for filename in args.artifact:
        path = Path(filename)
        if not path.is_absolute():
            path = repo / path
        saved = json.loads(path.read_text())
        prompt_id = str(saved["prompt_id"])
        token_id = int(saved["timing"]["paired_rounds"][0]["methods"]["original_bnb_greedy"]["token_ids"][0])
        ids = tokenizer(prompts[prompt_id], return_tensors="pt")["input_ids"].to("cuda:0")
        ref_cache = TargetKVCache(backend="static", max_capacity=int(ids.shape[-1]) + 16)
        fix_cache = TargetKVCache(backend="static", max_capacity=int(ids.shape[-1]) + 16)
        with torch.no_grad():
            ref_prefill = model(ids, past_key_values=ref_cache.cache, use_cache=True).logits[:, -1:, :]
        with (fixed_three_row_linear_schedule(model),
              fixed_row_sdpa_decode(model, physical_rows=3, block_k=32,
                                    attention_strategy="triton_batched"),
              torch.no_grad()):
            fix_prefill = model(ids, past_key_values=fix_cache.cache, use_cache=True).logits[:, -1:, :]
        if not torch.equal(ref_prefill, fix_prefill):
            raise RuntimeError(f"prefill logits differ for {prompt_id}")
        token = torch.tensor([[token_id]], device="cuda:0")
        ref_rows, ref_logits = trace_forward(model, token, ref_cache, names)
        with (fixed_three_row_linear_schedule(model),
              fixed_row_sdpa_decode(model, physical_rows=3, block_k=32,
                                    attention_strategy="triton_batched")):
            fix_rows, fix_logits = trace_forward(model, token, fix_cache, names)
        diffs = compare_modules(ref_rows, fix_rows)
        details.append({
            "prompt_id": prompt_id, "forced_bnb_token_index": 0, "forced_token_id": token_id,
            "prefill_logits_equal": True, "first_module_differences": diffs[:12],
            "total_module_input_output_differences": len(diffs),
            "logits_equal_after_transition": torch.equal(ref_logits, fix_logits),
            "logits_max_abs_diff": float((ref_logits.float() - fix_logits.float()).abs().max().item()),
        })
        print(json.dumps(details[-1], indent=2), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({
        "classification": "single forced decode transition module trace; empirical diagnostic, not proof",
        "model": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0), "quantized_linear_strategy": STRATEGY,
        "attention_strategy": "triton_batched + fixed_row_sdpa_decode(block_k=32)",
        "details": details,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
