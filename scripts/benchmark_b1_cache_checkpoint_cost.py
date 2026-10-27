#!/usr/bin/env python3
"""Measure zero-copy fork, logical crop and KV checkpoint/restore costs."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from pathlib import Path
from typing import Callable

import torch

from build_b1_reference_contract import canonical_hash, reference_payload
from run_b1_block_pilot import runtime_identity
from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.b1_exact import b1_fixed_row_generate


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]


def measure(fn: Callable[[], None], repeats: int, device: str, synchronize: bool = True) -> list[float]:
    values = []
    for _ in range(repeats):
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        if synchronize and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        values.append(time.perf_counter() - start)
    return values


def summarize(values: list[float]) -> dict[str, float]:
    return {"median_ms": statistics.median(values) * 1000,
            "p95_ms": percentile(values, 0.95) * 1000,
            "min_ms": min(values) * 1000}


def copy_active(cache: TargetKVCache, length: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    result = []
    for layer in cache.cache.layers:
        result.append((layer.keys[:, :, :length, :].clone(),
                       layer.values[:, :, :length, :].clone()))
    return result


def restore_active(cache: TargetKVCache, snapshot: list[tuple[torch.Tensor, torch.Tensor]], length: int) -> None:
    for layer, (keys, values) in zip(cache.cache.layers, snapshot):
        layer.keys[:, :, :length, :].copy_(keys)
        layer.values[:, :, :length, :].copy_(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--capacity", type=int, default=2048)
    parser.add_argument("--new-tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    row = json.loads(args.prompt_file.read_text().splitlines()[0])
    if not isinstance(row.get("prompt"), str) or not row["prompt"]:
        raise ValueError("prompt file must begin with a nonempty prompt row")
    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, "cuda:0", "sdpa")
    model, tokenizer = loaded.model, loaded.tokenizer

    def package_version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    identity = runtime_identity(model, tokenizer, "cuda:0", package_version)
    contract = json.loads(args.contract.read_text())
    source_manifest = {
        "model": args.model, "revision": args.revision, "backend": loaded.metadata,
        "runtime_identity": identity, "reference_schedule": "qwen2_fixed_row_singleton",
        "attention_strategy": "padded", "physical_rows": 3,
        "cache_backend": "StaticPreallocatedKVCache", "capacity": args.capacity,
        "greedy_tie_rule": "torch.argmax_first_index", "eos_token_id": tokenizer.eos_token_id,
    }
    expected = {key: value for key, value in contract.items()
                if key not in {"reference_contract_id", "provenance"}}
    if reference_payload(source_manifest) != expected or canonical_hash(expected) != contract.get("reference_contract_id"):
        raise ValueError("cache benchmark runtime does not match pinned reference contract")

    input_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].cpu()
    if input_ids.shape[1] + args.new_tokens > args.capacity:
        raise ValueError("prompt plus generation budget exceeds cache capacity")
    holder: dict[str, TargetKVCache] = {}

    def capture(_, cache):
        holder["cache"] = cache

    _, metrics = b1_fixed_row_generate(
        model, tokenizer, input_ids, k=2, max_new_tokens=args.new_tokens,
        device="cuda:0", kv_cache_max_capacity=args.capacity,
        physical_rows=3, attention_strategy="padded", state_observer=capture,
    )
    cache = holder["cache"]
    prompt_len = int(input_ids.shape[1])
    active_len = cache.get_seq_length()
    lengths = sorted({prompt_len, min(prompt_len + 64, active_len), active_len})
    torch.cuda.reset_peak_memory_stats("cuda:0")
    results = []
    for length in lengths:
        layer_bytes = sum(
            2 * layer.keys.element_size() * layer.keys[:, :, :length, :].numel()
            for layer in cache.cache.layers
        )

        def fork():
            branch = cache.fork_ephemeral_draft_kv()
            del branch

        def logical_crop_restore():
            prior = [layer.get_seq_length() for layer in cache.cache.layers]
            cache.crop(max(0, length - 1))
            for layer, seq_len in zip(cache.cache.layers, prior):
                layer.seq_len = seq_len

        fork_values = measure(fork, max(args.repeats, 100), "cuda:0", synchronize=False)
        crop_values = measure(logical_crop_restore, max(args.repeats, 100), "cuda:0", synchronize=False)
        snapshot = copy_active(cache, length)
        torch.cuda.synchronize("cuda:0")
        restore_values = measure(lambda: restore_active(cache, snapshot, length), args.repeats, "cuda:0")
        combined_values = measure(
            lambda: restore_active(cache, copy_active(cache, length), length),
            args.repeats, "cuda:0",
        )
        del snapshot
        results.append({
            "active_length": length,
            "checkpoint_bytes": layer_bytes,
            "zero_copy_fork": summarize(fork_values),
            "logical_crop_and_length_restore": summarize(crop_values),
            "restore_from_existing_checkpoint": summarize(restore_values),
            "checkpoint_plus_restore": summarize(combined_values),
        })

    report = {
        "status": "diagnostic cache primitive timing under pinned B1 runtime; not end-to-end policy cost",
        "reference_contract_id": contract["reference_contract_id"],
        "prompt_id": row.get("id"),
        "prompt_tokens": prompt_len,
        "generated_tokens": len(metrics.generated_token_ids),
        "cache_active_length": active_len,
        "cache_capacity": args.capacity,
        "repeats": args.repeats,
        "results": results,
        "limitations": [
            "Checkpointing clones active KV tensors; this may not match a production checkpoint design.",
            "Zero-copy fork and logical crop timings are host-side metadata operations.",
            "No selector, synchronization policy, allocator pressure or full decode replay is included.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"prompt_tokens": prompt_len, "active_length": active_len,
                      "results": results}, indent=2))


if __name__ == "__main__":
    main()
