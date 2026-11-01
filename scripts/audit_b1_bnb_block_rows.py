#!/usr/bin/env python3
"""Compare BNB greedy singleton rows with experimental BNB-rowwise blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model


def digest_prefix(cache: TargetKVCache, length: int) -> str:
    digest = hashlib.sha256()
    for layer in cache.cache.layers:
        if length > layer.get_seq_length():
            raise ValueError("prefix exceeds active KV length")
        digest.update(length.to_bytes(8, "little"))
        if not getattr(layer, "is_initialized", False):
            continue
        for tensor in (layer.keys, layer.values):
            raw = tensor[:, :, :length, :].detach().contiguous().view(torch.uint8)
            digest.update(raw.cpu().numpy().tobytes())
    return digest.hexdigest()


def model_logits(model, input_ids, cache):
    with torch.no_grad():
        return model(input_ids, past_key_values=cache.cache, use_cache=True).logits


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
    parser.add_argument("--max-rows", type=int, default=3)
    args = parser.parse_args()

    prompt_by_id = {str(row["id"]): row["prompt"] for row in
                    (json.loads(line) for line in Path(args.prompt_file).read_text().splitlines()
                     if line.strip())}
    source_by_id = {}
    for path in sorted(Path(args.saved_run).glob("session*/rows.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                source_by_id.setdefault(str(row["id"]), row)
    if args.max_rows > args.physical_rows:
        raise ValueError("max-rows must not exceed physical-rows")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model, loaded.tokenizer
    details = []
    for prompt_id, source in sorted(source_by_id.items()):
        prompt_ids = tokenizer(prompt_by_id[prompt_id], return_tensors="pt")["input_ids"].to(args.device)
        bnb_tokens = source["results"]["original_greedy"]["token_ids"]
        if len(bnb_tokens) < args.max_rows + 1:
            raise ValueError(f"too few BNB tokens for {prompt_id}")
        prompt_prefill_ref = TargetKVCache(backend="static", max_capacity=args.capacity)
        prompt_prefill_fixed = TargetKVCache(backend="static", max_capacity=args.capacity)
        logits_ref = model_logits(model, prompt_ids, prompt_prefill_ref)[:, -1:, :]
        with fixed_row_schedule_for_model(
            model, physical_rows=args.physical_rows,
            attention_strategy="bnb_rowwise", quantized_linear_strategy="fixed_rows",
        ):
            logits_fixed = model_logits(model, prompt_ids, prompt_prefill_fixed)[:, -1:, :]
        if not torch.equal(logits_ref, logits_fixed):
            raise RuntimeError(f"prefill logits differ for {prompt_id}")
        if digest_prefix(prompt_prefill_ref, int(prompt_ids.shape[1])) != digest_prefix(
            prompt_prefill_fixed, int(prompt_ids.shape[1])
        ):
            raise RuntimeError(f"prefill KV differs for {prompt_id}")

        prompt_records = []
        for rows in range(1, args.max_rows + 1):
            ref_cache = TargetKVCache(backend="static", max_capacity=args.capacity)
            fixed_cache = TargetKVCache(backend="static", max_capacity=args.capacity)
            _ = model_logits(model, prompt_ids, ref_cache)
            _ = model_logits(model, prompt_ids, fixed_cache)
            reference_logits = []
            reference_prefixes = []
            for index, token_id in enumerate(bnb_tokens[:rows]):
                token = torch.tensor([[int(token_id)]], device=args.device)
                out = model_logits(model, token, ref_cache)
                reference_logits.append(out[:, -1:, :].detach())
                reference_prefixes.append(digest_prefix(
                    ref_cache, int(prompt_ids.shape[1]) + index + 1
                ))
            block_tokens = torch.tensor([bnb_tokens[:rows]], device=args.device)
            with fixed_row_schedule_for_model(
                model, physical_rows=args.physical_rows,
                attention_strategy="bnb_rowwise", quantized_linear_strategy="fixed_rows",
            ):
                block_logits = model_logits(model, block_tokens, fixed_cache)
            row_results = []
            for index in range(rows):
                expected = reference_logits[index]
                actual = block_logits[:, index:index + 1, :]
                row_results.append({
                    "row": index,
                    "logits_bitwise_equal": torch.equal(expected, actual),
                    "max_abs_diff": float((expected.float() - actual.float()).abs().max().item()),
                    "reference_argmax": int(expected.argmax(dim=-1).item()),
                    "block_argmax": int(actual.argmax(dim=-1).item()),
                    "kv_prefix_exact": reference_prefixes[index] == digest_prefix(
                        fixed_cache, int(prompt_ids.shape[1]) + index + 1
                    ),
                })
            prompt_records.append({
                "physical_rows": args.physical_rows,
                "real_rows": rows,
                "all_row_logits_exact": all(r["logits_bitwise_equal"] for r in row_results),
                "all_row_argmax_equal": all(r["reference_argmax"] == r["block_argmax"]
                                             for r in row_results),
                "all_kv_prefixes_exact": all(r["kv_prefix_exact"] for r in row_results),
                "row_results": row_results,
            })
        details.append({"prompt_id": prompt_id, "row_sweep": prompt_records})
        print(json.dumps({
            "prompt_id": prompt_id,
            "all_logits_exact": all(r["all_row_logits_exact"] for r in prompt_records),
            "all_kv_prefixes_exact": all(r["all_kv_prefixes_exact"] for r in prompt_records),
        }, indent=2), flush=True)

    payload = {
        "status": "finite BNB-token forced block-vs-singleton audit; not a proof",
        "model": args.model,
        "revision": args.revision,
        "gpu": torch.cuda.get_device_name(args.device),
        "physical_rows": args.physical_rows,
        "attention_strategy": "bnb_rowwise",
        "quantized_linear_strategy": "fixed_rows with per-real-row singleton lm_head",
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
