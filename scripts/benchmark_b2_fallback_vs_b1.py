#!/usr/bin/env python3
"""Paired latency/exactness comparison: original BNB greedy vs B2 activationvec+fallback vs B1 Triton block."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from benchmark_bnb_nf4_activationvec_fallback_closed_loop import generate
from _common import run_vanilla_timed

METHODS = ("original_bnb_greedy", "b2_activationvec_fallback", "b1_triton_block")
B1_KW = dict(physical_rows=3, attention_strategy="rowwise",
             quantized_linear_strategy="triton_nf4_fixed_three_rows_group_scale_packed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-ids", nargs="+", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--timing-rounds", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = {str(r["id"]): r for r in (json.loads(l) for l in Path(args.benchmark).read_text().splitlines() if l.strip())}
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    device = "cuda:0"
    per_prompt = []
    for pid in args.prompt_ids:
        prompt_ids = tokenizer(rows[pid]["prompt"], return_tensors="pt")["input_ids"].to(device)
        capacity = int(prompt_ids.shape[-1]) + args.max_new_tokens + 8
        rounds = []
        b1_singleton_tokens = None
        for r in range(args.timing_rounds):
            order = list(METHODS[r % 3:] + METHODS[:r % 3])
            sample = {"round": r, "order": order, "methods": {}}
            for m in order:
                if m == "original_bnb_greedy":
                    tokens, _, _, ms = run_vanilla_timed(model, tokenizer, prompt_ids, args.max_new_tokens, device)
                    sample["methods"][m] = {"tokens": tokens, "total_s": ms / 1000}
                elif m == "b2_activationvec_fallback":
                    tokens, st = generate(model, tokenizer, prompt_ids, k=2, ngram_size=3,
                                          max_new_tokens=args.max_new_tokens, device=device, capacity=capacity,
                                          fallback_on_empty=True)
                    sample["methods"][m] = {"tokens": tokens, **st}
                else:
                    with torch.inference_mode():
                        _, metrics = b1_fixed_row_block_generate(
                            model, tokenizer, prompt_ids, k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                            device=device, kv_cache_max_capacity=capacity, **B1_KW)
                    sample["methods"][m] = {"tokens": metrics.generated_token_ids, "total_s": metrics.total_time_s,
                                            "cycles": metrics.cycles, "draft_tokens": metrics.draft_tokens,
                                            "accepted_draft_tokens": metrics.accepted_draft_tokens}
            ref = sample["methods"]["original_bnb_greedy"]["tokens"]
            for m in METHODS[1:]:
                sample[f"{m}_matches_bnb"] = sample["methods"][m]["tokens"] == ref
            rounds.append(sample)
        if b1_singleton_tokens is None:
            with torch.inference_mode():
                _, sm = b1_fixed_row_generate(model, tokenizer, prompt_ids, k=2, ngram_size=3,
                                              max_new_tokens=args.max_new_tokens, device=device,
                                              kv_cache_max_capacity=capacity, **B1_KW)
            b1_singleton_tokens = sm.generated_token_ids
        med = {m: statistics.median(s["methods"][m]["total_s"] for s in rounds) for m in METHODS}
        per_prompt.append({
            "prompt_id": pid, "rounds": rounds, "median_s": med,
            "b2_speedup_vs_bnb": med[METHODS[0]] / med[METHODS[1]],
            "b1_speedup_vs_bnb": med[METHODS[0]] / med[METHODS[2]],
            "b2_matches_bnb_all_rounds": all(s["b2_activationvec_fallback_matches_bnb"] for s in rounds),
            "b1_matches_bnb_all_rounds": all(s["b1_triton_block_matches_bnb"] for s in rounds),
            "b1_block_matches_b1_singleton": all(s["methods"]["b1_triton_block"]["tokens"] == b1_singleton_tokens
                                                 for s in rounds)})
        print(json.dumps({k: per_prompt[-1][k] for k in per_prompt[-1] if k not in ("rounds", "median_s")}))

    def gm(key):
        return statistics.geometric_mean(p[key] for p in per_prompt)

    result = {
        "schema": "zassd-b2-fallback-vs-b1-v1",
        "classification": "development-only paired comparison; finite evidence",
        "model": MODEL, "revision": REVISION, "benchmark": args.benchmark,
        "max_new_tokens": args.max_new_tokens, "timing_rounds": args.timing_rounds, "per_prompt": per_prompt,
        "geomean_b2_speedup_vs_bnb": gm("b2_speedup_vs_bnb"), "geomean_b1_speedup_vs_bnb": gm("b1_speedup_vs_bnb"),
        "b2_exact_to_bnb_prompts": sum(p["b2_matches_bnb_all_rounds"] for p in per_prompt),
        "b1_exact_to_bnb_prompts": sum(p["b1_matches_bnb_all_rounds"] for p in per_prompt),
        "b1_block_exact_to_b1_singleton_prompts": sum(p["b1_block_matches_b1_singleton"] for p in per_prompt),
        "prompt_count": len(per_prompt),
        "source": str(Path(__file__).resolve()), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {"torch": torch.__version__, "bitsandbytes": importlib.metadata.version("bitsandbytes"),
                    "gpu": torch.cuda.get_device_name(0)},
        "scope_limits": ["Development prompts only; GSM8K held-out indices 300-349 excluded.",
                         "B1 speedup is latency vs BNB with a different arithmetic reference; not BNB-exact."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in result if k.startswith(("geomean", "b2_exact", "b1_", "prompt_count"))}, indent=2))


if __name__ == "__main__":
    main()
