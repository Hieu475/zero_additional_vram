#!/usr/bin/env python3
"""Find the first module output that differs on the first forced decode step."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as torch_f

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.decoding.speculative import diagnostic_cache_digest


def digest_value(value) -> str:
    h = hashlib.sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().contiguous()
            h.update(f"tensor:{tuple(tensor.shape)}:{tensor.dtype}\n".encode())
            h.update(tensor.view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(item, (tuple, list)):
            h.update(f"seq:{type(item).__name__}:{len(item)}\n".encode())
            for child in item:
                add(child)
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                h.update(f"key:{key}\n".encode())
                add(item[key])
        elif hasattr(item, "to_tuple"):
            add(item.to_tuple())
        elif item is None or isinstance(item, (str, int, float, bool)):
            h.update(f"value:{type(item).__name__}:{item!r}\n".encode())
        else:
            h.update(f"type:{type(item).__module__}.{type(item).__qualname__}\n".encode())

    add(value)
    return h.hexdigest()


def first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (tuple, list)):
        for child in value:
            result = first_tensor(child)
            if result is not None:
                return result
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            result = first_tensor(value[key])
            if result is not None:
                return result
    return None


def collect_step(model, token, cache, module_names):
    events = []
    sdpa_records = []
    handles = []
    order = {"input": 0, "output": 0}
    for module in model.modules():
        name = module_names.get(id(module), "<unnamed>")

        def pre_hook(_module, inputs, _name=name):
            event = {"name": _name, "input_sha256": digest_value(inputs),
                     "input_order": order["input"]}
            if _name == "model.layers.0.self_attn.o_proj":
                event["captured_input"] = first_tensor(inputs)
            events.append(event)
            order["input"] += 1

        def output_hook(_module, _inputs, output, _name=name):
            # Forward hooks execute after the matching pre-hook; preserve order.
            for event in reversed(events):
                if event["name"] == _name and "output_sha256" not in event:
                    event["output_sha256"] = digest_value(output)
                    event["output_order"] = order["output"]
                    if _name == "model.layers.0.self_attn.o_proj":
                        event["captured_output"] = first_tensor(output)
                    order["output"] += 1
                    return

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(output_hook))
    original_sdpa = torch_f.scaled_dot_product_attention

    def traced_sdpa(*args, **kwargs):
        output = original_sdpa(*args, **kwargs)
        query = args[0]
        key = args[1]
        value = args[2]
        mask = kwargs.get("attn_mask", args[3] if len(args) > 3 else None)
        sdpa_records.append({
            "query_shape": list(query.shape),
            "key_shape": list(key.shape),
            "value_shape": list(value.shape),
            "mask_shape": None if mask is None else list(mask.shape),
            "is_causal": kwargs.get("is_causal", args[5] if len(args) > 5 else False),
            "enable_gqa": kwargs.get("enable_gqa", args[7] if len(args) > 7 else False),
            "output": output.detach().clone(),
            "output_sha256": digest_value(output),
        })
        return output

    torch_f.scaled_dot_product_attention = traced_sdpa
    try:
        with torch.no_grad():
            logits = model(token, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]
    finally:
        torch_f.scaled_dot_product_attention = original_sdpa
        for handle in handles:
            handle.remove()
    return events, logits, sdpa_records


def first_difference(reference, candidate, key):
    right = {event["name"]: event for event in candidate}
    order_key = "input_order" if key == "input_sha256" else "output_order"
    for event in sorted(reference, key=lambda item: item.get(order_key, -1)):
        other = right.get(event["name"])
        if other is None or event.get(key) != other.get(key):
            record = {
                "module": event["name"],
                "reference_hash": event.get(key),
                "fixed_row_hash": None if other is None else other.get(key),
            }
            capture_key = "captured_input" if key == "input_sha256" else "captured_output"
            left_tensor = event.get(capture_key)
            right_tensor = None if other is None else other.get(capture_key)
            if left_tensor is not None and right_tensor is not None:
                record.update({
                    "tensor_shape": list(left_tensor.shape),
                    "tensor_dtype": str(left_tensor.dtype),
                    "tensor_bitwise_equal": bool(torch.equal(left_tensor, right_tensor)),
                    "tensor_max_abs_diff": float((left_tensor.float() - right_tensor.float()).abs().max().item()),
                })
            return record
    return None


def model_logits(model, token_ids, cache):
    with torch.no_grad():
        return model(token_ids, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-run", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-rows", type=int, default=3)
    parser.add_argument("--capacity", type=int, default=768)
    parser.add_argument("--attention-strategy",
                        choices=("padded", "rowwise", "bnb_rowwise"), default="padded")
    parser.add_argument("--warmup-tokens", type=int, default=0,
                        help="Forced BNB prefix tokens consumed before the traced decode")
    args = parser.parse_args()

    prompts = {str(row["id"]): row["prompt"] for row in
               (json.loads(line) for line in Path(args.prompt_file).read_text().splitlines()
                if line.strip())}
    source_rows = {}
    for path in sorted(Path(args.saved_run).glob("session*/rows.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                source_rows.setdefault(str(row["id"]), row)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model, loaded.tokenizer
    module_names = {id(module): name or "<root>" for name, module in model.named_modules()}
    details = []
    for prompt_id, source in sorted(source_rows.items()):
        ids = tokenizer(prompts[prompt_id], return_tensors="pt")["input_ids"].to(args.device)
        cache_ref = TargetKVCache(backend="static", max_capacity=args.capacity)
        cache_fixed = TargetKVCache(backend="static", max_capacity=args.capacity)
        with torch.no_grad():
            logits_ref = model(ids, past_key_values=cache_ref.cache, use_cache=True).logits[:, -1:, :]
        with fixed_row_schedule_for_model(
            model, physical_rows=args.physical_rows,
            attention_strategy=args.attention_strategy,
            quantized_linear_strategy="fixed_rows",
        ):
            with torch.no_grad():
                logits_fixed = model(ids, past_key_values=cache_fixed.cache, use_cache=True).logits[:, -1:, :]
        if not torch.equal(logits_ref, logits_fixed):
            raise RuntimeError(f"prefill logits differ for {prompt_id}")
        if diagnostic_cache_digest(cache_ref) != diagnostic_cache_digest(cache_fixed):
            raise RuntimeError(f"prefill KV differs for {prompt_id}")

        bnb_tokens = source["results"]["original_greedy"]["token_ids"]
        if args.warmup_tokens >= len(bnb_tokens):
            raise ValueError(f"warmup prefix is too long for {prompt_id}")
        for index, token_id in enumerate(bnb_tokens[:args.warmup_tokens]):
            warmup = torch.tensor([[int(token_id)]], device=args.device)
            logits_ref = model_logits(model, warmup, cache_ref)
            with fixed_row_schedule_for_model(
                model, physical_rows=args.physical_rows,
                attention_strategy=args.attention_strategy,
                quantized_linear_strategy="fixed_rows",
            ):
                logits_fixed = model_logits(model, warmup, cache_fixed)
            if not torch.equal(logits_ref, logits_fixed):
                raise RuntimeError(
                    f"warmup logits diverged for {prompt_id} after token {index + 1}"
                )
            if diagnostic_cache_digest(cache_ref) != diagnostic_cache_digest(cache_fixed):
                raise RuntimeError(
                    f"warmup KV diverged for {prompt_id} after token {index + 1}"
                )
        traced_token = int(bnb_tokens[args.warmup_tokens])
        token = torch.tensor([[traced_token]], device=args.device)
        ref_events, ref_logits, ref_sdpa = collect_step(model, token, cache_ref, module_names)
        with fixed_row_schedule_for_model(
            model, physical_rows=args.physical_rows,
            attention_strategy=args.attention_strategy,
            quantized_linear_strategy="fixed_rows",
        ):
            fixed_events, fixed_logits, fixed_sdpa = collect_step(model, token, cache_fixed, module_names)
        output_diff = first_difference(ref_events, fixed_events, "output_sha256")
        input_diff = first_difference(ref_events, fixed_events, "input_sha256")
        first_sdpa_diff = None
        for index, (left, right) in enumerate(zip(ref_sdpa, fixed_sdpa)):
            real_rows = left["output"].shape[2]
            fixed_real_rows = right["output"][:, :, :real_rows, :]
            if not torch.equal(left["output"], fixed_real_rows):
                first_sdpa_diff = {
                    "call_index": index,
                    "reference_query_shape": left["query_shape"],
                    "fixed_row_query_shape": right["query_shape"],
                    "reference_key_shape": left["key_shape"],
                    "fixed_row_key_shape": right["key_shape"],
                    "reference_mask_shape": left["mask_shape"],
                    "fixed_row_mask_shape": right["mask_shape"],
                    "reference_is_causal": left["is_causal"],
                    "fixed_row_is_causal": right["is_causal"],
                    "reference_enable_gqa": left["enable_gqa"],
                    "fixed_row_enable_gqa": right["enable_gqa"],
                    "reference_output_shape": list(left["output"].shape),
                    "fixed_row_output_shape": list(right["output"].shape),
                    "compared_real_row_shape": list(fixed_real_rows.shape),
                    "output_bitwise_equal": torch.equal(left["output"], fixed_real_rows),
                    "output_max_abs_diff": float(
                        (left["output"].float() - fixed_real_rows.float()).abs().max().item()
                    ),
                }
                break
        record = {
            "prompt_id": prompt_id,
            "forced_bnb_token_index": args.warmup_tokens,
            "warmup_tokens_exact": args.warmup_tokens,
            "prefill_logits_and_kv_equal": True,
            "first_module_input_difference": input_diff,
            "first_module_output_difference": output_diff,
            "first_sdpa_output_difference": first_sdpa_diff,
            "sdpa_call_count_reference": len(ref_sdpa),
            "sdpa_call_count_fixed_row": len(fixed_sdpa),
            "module_events_reference": len(ref_events),
            "module_events_fixed_row": len(fixed_events),
            "final_logits_equal": torch.equal(ref_logits, fixed_logits),
            "final_logits_max_abs_diff": float(
                (ref_logits.float() - fixed_logits.float()).abs().max().item()
            ),
            "kv_equal_after_first_decode": diagnostic_cache_digest(cache_ref)
            == diagnostic_cache_digest(cache_fixed),
        }
        for event in ref_events + fixed_events:
            event.pop("captured_input", None)
            event.pop("captured_output", None)
        details.append(record)
        print(json.dumps(record, indent=2), flush=True)

    payload = {
        "status": "single-transition module-output trace; finite diagnostic, not proof",
        "model": args.model,
        "revision": args.revision,
        "gpu": torch.cuda.get_device_name(args.device),
        "physical_rows": args.physical_rows,
        "attention_strategy": args.attention_strategy,
        "quantized_linear_strategy": "fixed_rows",
        "saved_run": args.saved_run,
        "details": details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
