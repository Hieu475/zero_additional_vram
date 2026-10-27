#!/usr/bin/env python3
"""Probe Phi-3 fixed-row logits and every committed KV prefix against singleton calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.phi3_padded import fixed_row_phi3_schedule

from run_kv_factorial_intervention import restore_static_cache, snapshot_cache


def tensor_errors(left, right):
    rows = []
    for layer, (a, b) in enumerate(zip(left, right)):
        ak, av, alen = a
        bk, bv, blen = b
        if alen != blen:
            rows.append({"layer": layer, "length_mismatch": [alen, blen],
                         "bitwise_equal": False})
            continue
        if alen == 0:
            rows.append({"layer": layer, "key_max_abs": 0.0,
                         "value_max_abs": 0.0, "bitwise_equal": True})
            continue
        key_delta = (ak.float() - bk.float()).abs()
        value_delta = (av.float() - bv.float()).abs()
        rows.append({
            "layer": layer,
            "key_max_abs": float(key_delta.max()),
            "value_max_abs": float(value_delta.max()),
            "bitwise_equal": torch.equal(ak, bk) and torch.equal(av, bv),
        })
    return rows


def prefix_snapshot(snapshot, prefix_length):
    result = []
    for key, value, length in snapshot:
        if prefix_length > length:
            raise ValueError("requested prefix exceeds cache snapshot length")
        result.append((key[:, :, :prefix_length, :].clone(),
                       value[:, :, :prefix_length, :].clone(), prefix_length))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--revision", default="2fe192450127e6a83f7441aef6e3ca586c338b77")
    parser.add_argument("--benchmark", default="data/benchmarks/gsm8k_ext150.jsonl")
    parser.add_argument("--prompt-id", default="gsm8k_050")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-rows", type=int, default=9)
    parser.add_argument("--attention-strategy", choices=("rowwise", "padded"), default="padded")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this BNB NF4 shape probe")

    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    record = next((row for row in records if row["id"] == args.prompt_id), None)
    if record is None:
        raise ValueError(f"prompt {args.prompt_id!r} not found")
    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(record["prompt"], return_tensors="pt")["input_ids"].to(args.device)
    rows = args.physical_rows
    capacity = int(prompt_ids.shape[1]) + rows + 8

    with fixed_row_phi3_schedule(model, physical_rows=rows,
                                  attention_strategy=args.attention_strategy):
        source_cache = TargetKVCache(backend="static", max_capacity=capacity)
        with torch.no_grad():
            first_logits = model(prompt_ids, past_key_values=source_cache.cache,
                                 use_cache=True).logits[:, -1, :]
        source = snapshot_cache(source_cache.cache)
        candidates = [int(first_logits.argmax(-1).item())]
        candidate_cache = restore_static_cache(source, capacity, args.device)
        with torch.no_grad():
            for _ in range(rows - 1):
                token = torch.tensor([[candidates[-1]]], device=args.device,
                                     dtype=prompt_ids.dtype)
                output = model(token, past_key_values=candidate_cache.cache,
                               use_cache=True)
                candidates.append(int(output.logits[0, -1].argmax().item()))

        measurements = []
        for m in range(1, rows + 1):
            selected = candidates[:m]
            sequential = restore_static_cache(source, capacity, args.device)
            sequential_logits, sequential_prefixes = [], []
            with torch.no_grad():
                for token_id in selected:
                    token = torch.tensor([[token_id]], device=args.device,
                                         dtype=prompt_ids.dtype)
                    out = model(token, past_key_values=sequential.cache, use_cache=True)
                    sequential_logits.append(out.logits[0, -1].float().cpu())
                    sequential_prefixes.append(snapshot_cache(sequential.cache))

            block = restore_static_cache(source, capacity, args.device)
            block_ids = torch.tensor([selected], device=args.device, dtype=prompt_ids.dtype)
            with torch.no_grad():
                block_logits = model(block_ids, past_key_values=block.cache,
                                     use_cache=True).logits[0].float().cpu()
            sequential_logits = torch.stack(sequential_logits)
            block_snapshot = snapshot_cache(block.cache)
            prefix_comparisons = []
            for index, sequential_snapshot in enumerate(sequential_prefixes, start=1):
                expected_length = int(prompt_ids.shape[1]) + index
                block_prefix = prefix_snapshot(block_snapshot, expected_length)
                layer_errors = tensor_errors(sequential_snapshot, block_prefix)
                prefix_comparisons.append({
                    "committed_output_prefix": index,
                    "cache_length": expected_length,
                    "bitwise_equal": all(row.get("bitwise_equal", False) for row in layer_errors),
                    "max_key_abs": max((row.get("key_max_abs", 0.0) for row in layer_errors), default=0.0),
                    "max_value_abs": max((row.get("value_max_abs", 0.0) for row in layer_errors), default=0.0),
                    "first_differing_layer": next((row["layer"] for row in layer_errors
                        if not row.get("bitwise_equal", False)), None),
                })
            logit_difference = (sequential_logits - block_logits).abs()
            measurements.append({
                "M": m,
                "candidate_ids": selected,
                "logits_bitwise_equal": torch.equal(sequential_logits, block_logits),
                "argmax_equal_by_row": (sequential_logits.argmax(-1) == block_logits.argmax(-1)).tolist(),
                "per_row_logit_max_abs": logit_difference.amax(-1).tolist(),
                "kv_bitwise_equal": torch.equal(
                    torch.tensor([x["bitwise_equal"] for x in prefix_comparisons]),
                    torch.ones(m, dtype=torch.bool)),
                "committed_prefixes": prefix_comparisons,
            })
            print(json.dumps({
                "M": m,
                "logits_bitwise_equal": measurements[-1]["logits_bitwise_equal"],
                "all_argmax_equal": all(measurements[-1]["argmax_equal_by_row"]),
                "kv_prefixes_bitwise_equal": measurements[-1]["kv_bitwise_equal"],
                "max_logit_abs": max(measurements[-1]["per_row_logit_max_abs"]),
            }), flush=True)

    report = {
        "status": "experimental fixed-row probe; finite development evidence, not a proof",
        "model": args.model, "revision": args.revision,
        "backend": loaded.metadata, "gpu": torch.cuda.get_device_name(args.device),
        "attention_implementation": model.config._attn_implementation,
        "prompt_id": args.prompt_id, "prompt_length": int(prompt_ids.shape[1]),
        "prompt_token_ids_sha256": hashlib.sha256(json.dumps(prompt_ids[0].tolist()).encode()).hexdigest(),
        "physical_rows": rows, "attention_strategy": args.attention_strategy,
        "candidate_ids": candidates, "measurements": measurements,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output_path), "M_values": len(measurements),
                      "logit_bitwise_passes": sum(x["logits_bitwise_equal"] for x in measurements),
                      "KV_prefix_bitwise_passes": sum(x["kv_bitwise_equal"] for x in measurements)}, indent=2))


if __name__ == "__main__":
    main()
