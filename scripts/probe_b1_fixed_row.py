#!/usr/bin/env python3
"""Audit fixed-row Qwen2 block logits and KV against its singleton reference."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import torch
from bitsandbytes.nn import Linear4bit
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2RMSNorm

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.qwen2_padded import fixed_row_qwen2_schedule
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from zassd.decoding.speculative import diagnostic_cache_digest
from build_b1_reference_contract import canonical_hash, reference_payload
from run_b1_block_pilot import runtime_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--prompt-id", default="gsm8k_000")
    parser.add_argument("--input", default="data/benchmarks/gsm8k_eval.jsonl")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("config", "float16"), default="config")
    parser.add_argument("--physical-rows", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--attention-strategy", choices=("rowwise", "padded"), default="rowwise")
    parser.add_argument("--quantized-linear-strategy",
                        choices=("fixed_rows", "triton_nf4", "triton_nf4_group_scale_packed_rows2_n16_warp1_k256",
                                 ),
                        default="fixed_rows")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--reference-contract", default=None)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    prompts = {str(row["id"]): row["prompt"] for line in Path(args.input).read_text().splitlines()
               if (row := json.loads(line))}
    prompt = prompts[args.prompt_id]
    model_dtype = torch.float16 if args.model_dtype == "float16" else None
    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, args.device, "sdpa",
                          model_dtype=model_dtype)
    model, tokenizer = loaded.model, loaded.tokenizer
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(args.device)
    if prompt_ids.shape[1] + 2 > args.capacity:
        raise ValueError("capacity must leave room for the two-token strategy warmup")
    if args.quantized_linear_strategy in {
        "triton_nf4", "triton_nf4_group_scale_packed_rows2_n16_warp1_k256",
        }:
        # Match the pilot runner's warmup before hashing loaded model state.
        b1_fixed_row_generate(
            model, tokenizer, prompt_ids, k=max(1, args.physical_rows - 1),
            max_new_tokens=2, device=args.device,
            kv_cache_max_capacity=args.capacity,
            physical_rows=args.physical_rows,
            attention_strategy=args.attention_strategy,
            quantized_linear_strategy=args.quantized_linear_strategy,
        )
        b1_fixed_row_block_generate(
            model, tokenizer, prompt_ids, k=max(1, args.physical_rows - 1),
            max_new_tokens=2, device=args.device,
            kv_cache_max_capacity=args.capacity,
            physical_rows=args.physical_rows,
            attention_strategy=args.attention_strategy,
            quantized_linear_strategy=args.quantized_linear_strategy,
        )
        torch.cuda.synchronize(args.device)
    reference_contract_id = None
    if args.reference_contract:
        def package_version(name):
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                return None
        identity = runtime_identity(
            model, tokenizer, args.device, package_version,
            quantized_linear_strategy=args.quantized_linear_strategy,
        )
        contract_manifest = {
            "model": args.model, "revision": args.revision,
            "backend": loaded.metadata, "runtime_identity": identity,
            "reference_schedule": (
                "qwen2_fixed_row_singleton"
                if args.quantized_linear_strategy == "fixed_rows"
                else f"qwen2_fixed_row_{args.attention_strategy}_"
                     f"{args.quantized_linear_strategy}_singleton"
            ),
            "attention_strategy": args.attention_strategy,
            "physical_rows": args.physical_rows,
            "cache_backend": "StaticPreallocatedKVCache", "capacity": args.capacity,
            "greedy_tie_rule": "torch.argmax_first_index",
            "eos_token_id": tokenizer.eos_token_id,
        }
        contract = json.loads(Path(args.reference_contract).read_text())
        expected = {key: value for key, value in contract.items()
                    if key not in {"reference_contract_id", "provenance"}}
        actual = reference_payload(contract_manifest)
        if actual != expected or canonical_hash(expected) != contract.get("reference_contract_id"):
            mismatches = [key for key in expected if expected[key] != actual.get(key)]
            if canonical_hash(expected) != contract.get("reference_contract_id"):
                mismatches.append("reference_contract_id integrity")
            raise ValueError(
                f"operator probe runtime does not match reference contract: {mismatches}"
            )
        reference_contract_id = contract["reference_contract_id"]
    rows = []
    phase = ["idle"]
    active_row = [0]
    attention_captures = {"sequential": [], "block": []}

    def observe_attention(layer, row, query, key, value, mask, result):
        row_index = active_row[0] if phase[0] == "sequential" else row
        def copy(value):
            return None if value is None else value.detach().cpu().clone()
        attention_captures[phase[0]].append(
            (layer, row_index, copy(query), copy(key), copy(value), copy(mask), copy(result)))

    with fixed_row_qwen2_schedule(model, physical_rows=args.physical_rows,
                                  attention_strategy=args.attention_strategy,
                                  quantized_linear_strategy=args.quantized_linear_strategy,
                                  observer=observe_attention if args.diagnose else None):
        for m in range(1, args.physical_rows + 1):
            sequential = TargetKVCache(backend="static", max_capacity=args.capacity)
            block = TargetKVCache(backend="static", max_capacity=args.capacity)
            captures = {"sequential": {}, "block": {}}
            attention_captures = {"sequential": [], "block": []}
            phase[0] = "sequential"
            hooks = []
            with torch.no_grad():
                first = model(prompt_ids, past_key_values=sequential.cache,
                              use_cache=True).logits[:, -1:, :].argmax(-1)
                model(prompt_ids, past_key_values=block.cache, use_cache=True)
                if args.diagnose and m == 2:
                    for name, module in model.named_modules():
                        if not isinstance(module, (Linear4bit, torch.nn.Linear,
                                                   Qwen2Attention, Qwen2RMSNorm)):
                            continue
                        def capture(_module, inputs, keywords, output, _name=name):
                            first_output = output[0] if isinstance(output, tuple) else output
                            first_input = (inputs[0] if inputs else
                                           keywords.get("hidden_states", keywords.get("input")))
                            if isinstance(first_output, torch.Tensor) and isinstance(first_input, torch.Tensor):
                                captures[phase[0]].setdefault(_name, []).append((
                                    first_input.detach().cpu().clone(),
                                    first_output.detach().cpu().clone()))
                        hooks.append(module.register_forward_hook(capture, with_kwargs=True))
                token = first
                inputs, logits = [], []
                for index in range(m):
                    active_row[0] = index
                    inputs.append(token)
                    out = model(token, past_key_values=sequential.cache, use_cache=True)
                    logits.append(out.logits)
                    token = out.logits[:, -1:, :].argmax(-1)
                phase[0] = "block"
                block_logits = model(torch.cat(inputs, dim=1), past_key_values=block.cache,
                                     use_cache=True).logits
                for hook in hooks:
                    hook.remove()
            sequential_logits = torch.cat(logits, dim=1)
            torch.cuda.synchronize(args.device)
            row_result = {
                "m": m,
                "token_inputs": torch.cat(inputs, dim=1)[0].tolist(),
                "bitwise_logits": bool(torch.equal(sequential_logits, block_logits)),
                "max_logit_difference": float((sequential_logits.float() - block_logits.float()).abs().max()),
                "same_argmax": bool(torch.equal(sequential_logits.argmax(-1), block_logits.argmax(-1))),
                "bitwise_kv": diagnostic_cache_digest(sequential) == diagnostic_cache_digest(block),
                "sequential_cache_length": sequential.get_seq_length(),
                "block_cache_length": block.get_seq_length(),
            }
            if captures["block"]:
                first_differences = []
                for name, block_calls in captures["block"].items():
                    seq_calls = captures["sequential"].get(name, [])
                    if len(block_calls) != 1 or len(seq_calls) != m:
                        continue
                    block_input, block_output = block_calls[0]
                    for index in range(m):
                        seq_input, seq_output = seq_calls[index]
                        if not torch.equal(seq_output[:, 0], block_output[:, index]):
                            first_differences.append({
                                "module": name, "row": index,
                                "same_input": bool(torch.equal(seq_input[:, 0], block_input[:, index])),
                                "max_output_difference": float((seq_output[:, 0].float() -
                                                                block_output[:, index].float()).abs().max()),
                            })
                row_result["first_module_differences"] = first_differences[:12]
            if args.diagnose and m == 2:
                attention_diffs = []
                seq_by_key = {(item[0], item[1]): item[2:] for item in attention_captures["sequential"]}
                for layer, row, *block_tensors in attention_captures["block"]:
                    seq_tensors = seq_by_key[(layer, row)]
                    for name, seq_tensor, block_tensor in zip(
                            ("query", "key", "value", "mask", "output"),
                            seq_tensors, block_tensors):
                        if seq_tensor is None and block_tensor is None:
                            continue
                        same = (seq_tensor is None and block_tensor is None) or (
                            seq_tensor is not None and block_tensor is not None and
                            seq_tensor.shape == block_tensor.shape and
                            torch.equal(seq_tensor, block_tensor))
                        if not same:
                            attention_diffs.append({
                                "layer": layer, "row": row, "tensor": name,
                                "seq_shape": None if seq_tensor is None else list(seq_tensor.shape),
                                "block_shape": None if block_tensor is None else list(block_tensor.shape),
                                "max_difference": None if seq_tensor is None or block_tensor is None or
                                seq_tensor.shape != block_tensor.shape
                                else float((seq_tensor.float() - block_tensor.float()).abs().max()),
                            })
                row_result["attention_tensor_differences"] = attention_diffs[:12]
            rows.append(row_result)
            print(rows[-1], flush=True)
    result = {
        "status": "experimental fixed-row schedule; finite test, not universal proof",
        "model": args.model, "revision": args.revision,
        "backend": loaded.metadata, "gpu": torch.cuda.get_device_name(args.device),
        "prompt_id": args.prompt_id, "prompt_length": int(prompt_ids.shape[1]),
        "reference_contract_id": reference_contract_id,
        "physical_rows": args.physical_rows, "attention_strategy": args.attention_strategy,
        "quantized_linear_strategy": args.quantized_linear_strategy,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
