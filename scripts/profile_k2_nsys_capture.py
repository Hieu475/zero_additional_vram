#!/usr/bin/env python3
"""Capture one warmed exact-B1 K2 decode under Nsight Systems CUDA range."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION
from audit_activationvec_fallback_fraction_ablation_512 import (
    cache_observer,
    fallback_fraction_schedule,
)
from run_strong_baseline_comparison import run_vanilla_timed
from zassd.backends.base import load_backend
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/humaneval_ext114.jsonl"
PROMPTS = ("HumanEval/123", "HumanEval/114")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-id", choices=PROMPTS, required=True)
    ap.add_argument("--max-new-tokens", type=int, choices=(256, 512), required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    output = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    if output.exists():
        raise FileExistsError(output)
    row = next(json.loads(line) for line in BENCHMARK.read_text().splitlines()
               if line.strip() and str(json.loads(line)["id"]) == args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(row["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + args.max_new_tokens + 8
    common = dict(k=2, ngram_size=3, max_new_tokens=args.max_new_tokens,
                  device="cuda:0", kv_cache_max_capacity=capacity, physical_rows=3,
                  attention_strategy="bnb_rowwise", quantized_linear_strategy="fixed_rows")

    ref_states = {"reference": {}}
    with torch.inference_mode():
        _, ref = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **common,
            state_observer=cache_observer(ref_states["reference"], prompt_len))
    candidate_states = {"candidate": {}}
    control = {"no_proposal_cycles": 0, "fallback_forwards": 0,
               "empty_cycle_index": 0, "use_native": False}
    with fallback_fraction_schedule(model, control, 1), torch.inference_mode():
        _, preflight = b1_fixed_row_block_generate(
            model, tokenizer, prompt_ids, **{**common, "trace": True},
            state_observer=cache_observer(candidate_states["candidate"], prompt_len))
    mismatches = [i for i in sorted(set(ref_states["reference"]) & set(candidate_states["candidate"]))
                  if ref_states["reference"][i] != candidate_states["candidate"][i]]
    exact = (preflight.generated_token_ids == ref.generated_token_ids
             and set(ref_states["reference"]) == set(candidate_states["candidate"])
             and not mismatches)
    if not exact:
        raise RuntimeError(f"B1 exactness preflight failed: state mismatches={mismatches[:8]}")

    with torch.inference_mode():
        run_vanilla_timed(model, tokenizer, prompt_ids, 8, "cuda:0")
        warm_control = {"no_proposal_cycles": 0, "fallback_forwards": 0,
                        "empty_cycle_index": 0, "use_native": False}
        with fallback_fraction_schedule(model, warm_control, 1):
            b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids,
                **{**common, "max_new_tokens": 8, "trace": True})
    torch.cuda.synchronize("cuda:0")

    measured_control = {"no_proposal_cycles": 0, "fallback_forwards": 0,
                        "empty_cycle_index": 0, "use_native": False}
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push(f"zassd_k2_r3_fallback_all_{args.max_new_tokens}")
    try:
        with fallback_fraction_schedule(model, measured_control, 1), torch.inference_mode():
            _, metrics = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **{**common, "trace": True})
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize("cuda:0")
        torch.cuda.profiler.stop()

    if metrics.generated_token_ids != ref.generated_token_ids:
        raise RuntimeError("captured candidate tokens differ from B1 singleton")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "zassd-k2-nsys-capture-v1",
        "classification": "Nsight Systems diagnostic capture; profiled latency is not benchmark timing",
        "model": MODEL, "revision": REVISION, "gpu": torch.cuda.get_device_name(0),
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
        "max_new_tokens": args.max_new_tokens,
        "schedule": {"k": 2, "ngram_size": 3, "physical_rows": 3,
                     "attention": "bnb_rowwise", "fallback": "native BNB M=1 on every empty-proposal cycle"},
        "b1_token_and_committed_kv_exact": True,
        "shared_state_prefixes": len(ref_states["reference"]),
        "state_mismatch_positions": mismatches,
        "empty_proposal_cycles": measured_control["no_proposal_cycles"],
        "fallback_forwards": measured_control["fallback_forwards"],
        "metrics": {"output_tokens": len(metrics.generated_token_ids),
                    "total_s_under_capture": metrics.total_time_s,
                    "prefill_s": metrics.prefill_time_s,
                    "draft_s": metrics.draft_time_s,
                    "verify_s": metrics.verify_time_s,
                    "cycles": metrics.cycles, "draft_tokens": metrics.draft_tokens,
                    "accepted_draft_tokens": metrics.accepted_draft_tokens,
                    "target_rows": metrics.target_rows},
        "runner_sha256": sha(Path(__file__)),
        "heldout_used": False,
        "limits": ["Nsight timing is diagnostic and instrumented; do not compare it with unprofiled latency.",
                   "One selected prompt, one model and GPU; no BNB KV-state identity claim.",
                   "GSM8K held-out indices 300-349 excluded."],
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "prompt_id": args.prompt_id,
                      "max_new_tokens": args.max_new_tokens, "b1_exact": exact,
                      "metrics": report["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
