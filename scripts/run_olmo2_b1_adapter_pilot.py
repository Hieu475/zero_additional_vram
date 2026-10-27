#!/usr/bin/env python3
"""Run multi-cycle OLMo-2 B1 singleton/block comparison with prefix-KV auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from zassd.decoding.speculative import diagnostic_cache_digest
from run_b1_block_pilot import diagnostic_cache_prefix_digest


def digest_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct")
    parser.add_argument("--revision", default="48d788eca847d4d7548f375ad03d3c9312f6139e")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--physical-rows", type=int, default=5)
    parser.add_argument("--attention-strategy", choices=("rowwise", "padded"), default="padded")
    parser.add_argument("--capacity", type=int, default=1024)
    args = parser.parse_args()
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated output directory: {out}")
    if args.physical_rows < args.k + 1:
        raise ValueError("physical_rows must be at least k+1")
    prompts_path = Path(args.prompt_file)
    prompts = [json.loads(line) for line in prompts_path.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError("validation prompt file is empty")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the OLMo-2 NF4 adapter pilot")

    loaded = load_backend("hf_bnb_nf4", args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "olmo2-fixed-row-adapter-pilot-v1",
        "status": "experimental development validation; not a correctness proof",
        "model": args.model, "revision": args.revision,
        "backend": loaded.metadata, "gpu": torch.cuda.get_device_name(args.device),
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "bitsandbytes": __import__("bitsandbytes").__version__,
        "attention_implementation": model.config._attn_implementation,
        "k": args.k, "physical_rows": args.physical_rows,
        "attention_strategy": args.attention_strategy,
        "max_new_tokens": args.max_new_tokens, "capacity": args.capacity,
        "prompt_file": str(prompts_path), "prompt_file_sha256": digest_file(prompts_path),
        "source_sha256": {
            path: digest_file(path) for path in (
                "src/zassd/certification/olmo2_padded.py",
                "src/zassd/certification/fixed_row_dispatch.py",
                "src/zassd/certification/model_capabilities.py",
                "src/zassd/decoding/b1_exact.py",
            )
        },
    }
    try:
        manifest["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest["git_commit"] = None
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    for prompt_spec in prompts:
        prompt = prompt_spec["prompt"]
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cpu()
        if ids.shape[1] + args.max_new_tokens > args.capacity:
            raise ValueError(f"{prompt_spec['id']} exceeds KV capacity")
        ref_states: dict[int, str] = {}
        _, ref = b1_fixed_row_generate(
            model, tokenizer, ids, k=args.k, max_new_tokens=args.max_new_tokens,
            device=args.device, kv_cache_max_capacity=args.capacity,
            physical_rows=args.physical_rows, attention_strategy=args.attention_strategy,
            state_observer=lambda n, cache: ref_states.__setitem__(
                n, diagnostic_cache_digest(cache)),
        )

        block_states: dict[int, str] = {}

        def observe_block_prefixes(n, cache):
            first = 0 if not block_states else max(block_states) + 1
            for count in range(first, n + 1):
                block_states[count] = diagnostic_cache_prefix_digest(
                    cache, int(ids.shape[1]) + count)

        _, block = b1_fixed_row_block_generate(
            model, tokenizer, ids, k=args.k, max_new_tokens=args.max_new_tokens,
            device=args.device, kv_cache_max_capacity=args.capacity,
            physical_rows=args.physical_rows, attention_strategy=args.attention_strategy,
            trace=True, state_observer=observe_block_prefixes,
        )
        shared = sorted(set(ref_states) & set(block_states))
        row = {
            "task": prompt_spec["task"], "prompt_id": prompt_spec["id"],
            "prompt_token_count": int(ids.shape[1]),
            "prompt_token_ids_sha256": hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
            "reference_token_ids": ref.generated_token_ids,
            "block_token_ids": block.generated_token_ids,
            "block_matches_reference": block.generated_token_ids == ref.generated_token_ids,
            "reference_schedule": ref.reference_schedule,
            "block_schedule": block.reference_schedule,
            "capability_report": block.capability_report,
            "reference_state_prefixes": sorted(ref_states),
            "block_state_prefixes": sorted(block_states),
            "shared_state_prefixes": len(shared),
            "all_shared_states_equal": all(ref_states[n] == block_states[n] for n in shared),
            "state_exact_at_all_reference_prefixes": set(ref_states) == set(block_states)
                and all(ref_states[n] == block_states[n] for n in ref_states),
            "mismatch_prefixes": [n for n in shared if ref_states[n] != block_states[n]],
            "cycle_trace": block.cycle_trace,
            "cycles": block.cycles,
            "draft_tokens": block.draft_tokens,
            "accepted_draft_tokens": block.accepted_draft_tokens,
        }
        rows.append(row)
        with (out / "rows.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(json.dumps({"prompt_id": row["prompt_id"],
                          "block_matches_reference": row["block_matches_reference"],
                          "state_exact_at_all_reference_prefixes": row["state_exact_at_all_reference_prefixes"],
                          "shared_state_prefixes": row["shared_state_prefixes"],
                          "cycles": row["cycles"], "draft_tokens": row["draft_tokens"],
                          "accepted_draft_tokens": row["accepted_draft_tokens"]}), flush=True)

    summary = {
        "status": "complete" if len(rows) == len(prompts) else "partial",
        "prompt_count": len(rows),
        "exact_token_matches": sum(row["block_matches_reference"] for row in rows),
        "all_prefix_state_matches": sum(row["state_exact_at_all_reference_prefixes"] for row in rows),
        "total_shared_prefix_states": sum(row["shared_state_prefixes"] for row in rows),
        "scope": "three development prompts, one model/runtime, finite empirical adapter validation",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
