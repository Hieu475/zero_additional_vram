#!/usr/bin/env python3
"""Replay saved BNB token paths through vanilla and fixed-row singleton schedules.

Development-only numerical diagnostic. The saved BNB token sequence is forced
into both executions so logits and KV can be compared at identical prefixes,
including before greedy argmax first diverges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.decoding.speculative import diagnostic_cache_digest


def forward(model, token_ids: torch.Tensor, cache: TargetKVCache):
    with torch.no_grad():
        output = model(token_ids, past_key_values=cache.cache, use_cache=True)
    return output.logits[:, -1:, :]


def compare_logits(reference: torch.Tensor, candidate: torch.Tensor, forced_id: int) -> dict:
    ref = reference.float().reshape(-1)
    got = candidate.float().reshape(-1)
    top2 = torch.topk(ref, k=2).values
    return {
        "bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_abs_diff": float((ref - got).abs().max().item()),
        "argmax_reference": int(ref.argmax().item()),
        "argmax_fixed_row": int(got.argmax().item()),
        "forced_bnb_token": int(forced_id),
        "bnb_top1_margin": float((top2[0] - top2[1]).item()),
        "forced_token_logit_reference": float(ref[forced_id].item()),
        "forced_token_logit_fixed_row": float(got[forced_id].item()),
    }


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
    parser.add_argument("--max-steps", type=int, default=120,
                        help="Maximum saved BNB tokens replayed per prompt")
    args = parser.parse_args()

    saved_root = Path(args.saved_run)
    prompt_rows = [json.loads(line) for line in Path(args.prompt_file).read_text().splitlines()
                   if line.strip()]
    prompts_by_id = {str(row["id"]): row["prompt"] for row in prompt_rows}
    saved_rows: dict[str, list[dict]] = {}
    for path in sorted(saved_root.glob("session*/rows.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            saved_rows.setdefault(str(row["id"]), []).append(row)
    if not saved_rows:
        raise ValueError(f"no saved session*/rows.jsonl under {saved_root}")
    missing = sorted(set(saved_rows) - set(prompts_by_id))
    if missing:
        raise ValueError(f"prompt file is missing saved IDs: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model, loaded.tokenizer
    details = []
    for prompt_id, source_rows in sorted(saved_rows.items()):
        source_rows.sort(key=lambda r: r["session_index"])
        row = source_rows[0]
        source_token_paths_identical = all(
            r["results"]["original_greedy"]["token_ids"]
            == row["results"]["original_greedy"]["token_ids"]
            for r in source_rows
        )
        if not source_token_paths_identical:
            raise ValueError(f"saved BNB token paths differ across sessions for {prompt_id}")
        forced_tokens = row["results"]["original_greedy"]["token_ids"][:args.max_steps]
        ids = tokenizer(prompts_by_id[prompt_id], return_tensors="pt")["input_ids"].to(args.device)
        capacity = args.capacity
        if int(ids.shape[1]) + len(forced_tokens) > capacity:
            raise ValueError(f"{prompt_id}: prompt and replay exceed cache capacity")
        vanilla_cache = TargetKVCache(backend="static", max_capacity=capacity)
        fixed_cache = TargetKVCache(backend="static", max_capacity=capacity)
        step_records = []
        vanilla_logits = forward(model, ids, vanilla_cache)
        with fixed_row_schedule_for_model(
            model, physical_rows=args.physical_rows,
            attention_strategy=args.attention_strategy,
            quantized_linear_strategy="fixed_rows",
        ):
            fixed_logits = forward(model, ids, fixed_cache)
        step_records.append({
            "output_index": 0,
            "stage": "prompt_prefill",
            **compare_logits(vanilla_logits, fixed_logits, forced_tokens[0]),
            "vanilla_kv_sha256": diagnostic_cache_digest(vanilla_cache),
            "fixed_row_kv_sha256": diagnostic_cache_digest(fixed_cache),
        })
        for index, token_id in enumerate(forced_tokens):
            token = torch.tensor([[int(token_id)]], device=args.device)
            vanilla_logits = forward(model, token, vanilla_cache)
            with fixed_row_schedule_for_model(
                model, physical_rows=args.physical_rows,
                attention_strategy=args.attention_strategy,
                quantized_linear_strategy="fixed_rows",
            ):
                fixed_logits = forward(model, token, fixed_cache)
            step_records.append({
                "output_index": index + 1,
                "stage": "decode_after_forced_token",
                **compare_logits(vanilla_logits, fixed_logits,
                                 forced_tokens[index + 1] if index + 1 < len(forced_tokens)
                                 else int(vanilla_logits.argmax().item())),
                "vanilla_kv_sha256": diagnostic_cache_digest(vanilla_cache),
                "fixed_row_kv_sha256": diagnostic_cache_digest(fixed_cache),
            })
        logit_drift = next((r["output_index"] for r in step_records
                            if not r["bitwise_equal"]), None)
        kv_drift = next((r["output_index"] for r in step_records
                         if r["vanilla_kv_sha256"] != r["fixed_row_kv_sha256"]), None)
        argmax_drift = next((r["output_index"] for r in step_records
                             if r["argmax_reference"] != r["argmax_fixed_row"]), None)
        saved_divergence = next((i for i, (a, b) in enumerate(zip(
            row["results"]["b1_block"]["token_ids"],
            row["results"]["original_greedy"]["token_ids"])) if a != b), None)
        details.append({
            "prompt_id": prompt_id,
            "source_session_indices": [r["session_index"] for r in source_rows],
            "source_token_paths_identical_across_sessions": source_token_paths_identical,
            "saved_bnb_first_token_divergence_from_b1_index": saved_divergence,
            "replayed_forced_bnb_tokens": len(forced_tokens),
            "first_logit_bitwise_drift_output_index": logit_drift,
            "first_argmax_drift_output_index": argmax_drift,
            "first_kv_state_drift_after_consumed_tokens": kv_drift,
            "step_trace": step_records,
        })
        print(json.dumps({k: v for k, v in details[-1].items() if k != "step_trace"}, indent=2),
              flush=True)

    payload = {
        "status": "forced-prefix numerical replay; empirical diagnostic, not a proof",
        "model": args.model,
        "revision": args.revision,
        "gpu": torch.cuda.get_device_name(args.device),
        "physical_rows": args.physical_rows,
        "attention_strategy": args.attention_strategy,
        "quantized_linear_strategy": "fixed_rows",
        "saved_run": str(saved_root),
        "prompt_file": args.prompt_file,
        "details": details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
