#!/usr/bin/env python3
"""Paired A/B timing and state audit for reusable Llama linear pad buffers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from zassd.decoding.speculative import diagnostic_cache_digest
from run_b1_block_pilot import diagnostic_cache_prefix_digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="unsloth/Llama-3.2-3B-Instruct")
    parser.add_argument("--revision", default="006f5dcd1393c3add266de40994ba96225e9689d")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--physical-rows", type=int, default=3)
    parser.add_argument("--attention-strategy", choices=("rowwise", "padded"), default="padded")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--capacity", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--session-index", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output directory {output}")
    if args.physical_rows < args.k + 1:
        raise ValueError("physical_rows must be at least k+1")
    prompts = [json.loads(line) for line in Path(args.prompt_file).read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError("prompt file has no rows")
    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    ids_by_row = []
    for row in prompts:
        ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].cpu()
        if ids.shape[1] + args.max_new_tokens > args.capacity:
            raise ValueError(f"{row['id']} exceeds cache capacity")
        ids_by_row.append(ids)

    # Warm both paths once before measurement.
    first_ids = ids_by_row[0]
    for reuse in (False, True):
        b1_fixed_row_block_generate(
            model, tokenizer, first_ids, k=args.k, max_new_tokens=8,
            device=args.device, kv_cache_max_capacity=args.capacity,
            physical_rows=args.physical_rows, attention_strategy=args.attention_strategy,
            reuse_linear_padding_buffer=reuse,
        )

    output.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for index, (prompt, ids) in enumerate(zip(prompts, ids_by_row)):
        reference_states: dict[int, str] = {}
        _, reference = b1_fixed_row_generate(
            model, tokenizer, ids, k=args.k, max_new_tokens=args.max_new_tokens,
            device=args.device, kv_cache_max_capacity=args.capacity,
            physical_rows=args.physical_rows, attention_strategy=args.attention_strategy,
            state_observer=lambda n, cache: reference_states.__setitem__(
                n, diagnostic_cache_digest(cache)),
        )
        row = {
            "task": prompt.get("task"), "prompt_id": prompt["id"],
            "prompt_token_count": int(ids.shape[1]),
            "prompt_token_ids_sha256": hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
            "reference_token_ids": reference.generated_token_ids,
            "reference_state_prefixes": sorted(reference_states),
            "measurements": [],
        }
        for repeat in range(args.repeats):
            order = [False, True]
            if (repeat + index + args.session_index) % 2:
                order.reverse()
            pair = {}
            for reuse in order:
                name = "reused_buffer" if reuse else "fresh_pad"
                block_states: dict[int, str] = {}

                def observe_prefixes(n, cache):
                    first = 0 if not block_states else max(block_states) + 1
                    for count in range(first, n + 1):
                        block_states[count] = diagnostic_cache_prefix_digest(
                            cache, int(ids.shape[1]) + count)

                _, metrics = b1_fixed_row_block_generate(
                    model, tokenizer, ids, k=args.k,
                    max_new_tokens=args.max_new_tokens, device=args.device,
                    kv_cache_max_capacity=args.capacity, physical_rows=args.physical_rows,
                    attention_strategy=args.attention_strategy,
                    reuse_linear_padding_buffer=reuse, state_observer=observe_prefixes,
                )
                shared = sorted(set(reference_states) & set(block_states))
                pair[name] = {
                    "total_s": metrics.total_time_s,
                    "token_ids": metrics.generated_token_ids,
                    "matches_reference_tokens": metrics.generated_token_ids == reference.generated_token_ids,
                    "state_prefix_count": len(shared),
                    "state_exact_at_all_reference_prefixes":
                        set(reference_states) == set(block_states) and
                        all(reference_states[n] == block_states[n] for n in reference_states),
                    "mismatch_prefixes": [n for n in shared
                        if reference_states[n] != block_states[n]],
                }
            pair["reuse_speedup_vs_fresh_pad"] = (
                pair["fresh_pad"]["total_s"] / pair["reused_buffer"]["total_s"]
            )
            row["measurements"].append(pair)
        rows_out.append(row)
        print(json.dumps({
            "prompt_id": row["prompt_id"],
            "mean_reuse_speedup": sum(m["reuse_speedup_vs_fresh_pad"] for m in row["measurements"]) / args.repeats,
            "all_tokens_match": all(m[v]["matches_reference_tokens"]
                                     for m in row["measurements"] for v in ("fresh_pad", "reused_buffer")),
            "all_state_prefixes_match": all(m[v]["state_exact_at_all_reference_prefixes"]
                                             for m in row["measurements"] for v in ("fresh_pad", "reused_buffer")),
        }), flush=True)

    ratios = [m["reuse_speedup_vs_fresh_pad"] for row in rows_out for m in row["measurements"]]
    summary = {
        "status": "paired experimental A/B; not a speedup claim",
        "model": args.model, "revision": args.revision,
        "backend": loaded.metadata,
        "attention_strategy": args.attention_strategy,
        "k": args.k, "physical_rows": args.physical_rows,
        "max_new_tokens": args.max_new_tokens,
        "prompt_count": len(rows_out), "repeats_per_prompt": args.repeats,
        "paired_measurements": len(ratios),
        "mean_reuse_speedup_vs_fresh_pad": sum(ratios) / len(ratios),
        "all_tokens_match_reference": all(
            m[v]["matches_reference_tokens"] for r in rows_out for m in r["measurements"]
            for v in ("fresh_pad", "reused_buffer")),
        "all_states_match_reference": all(
            m[v]["state_exact_at_all_reference_prefixes"] for r in rows_out for m in r["measurements"]
            for v in ("fresh_pad", "reused_buffer")),
        "scope": "three development prompts, one pinned model/runtime, within-process paired timing",
    }
    (output / "rows.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows_out))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
