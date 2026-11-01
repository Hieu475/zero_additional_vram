#!/usr/bin/env python3
"""Replay original-BNB token prefixes through BNB and fixed-three-row singleton."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from audit_triton_b1_closed_loop import fixed_row_sdpa_decode
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.decoding.speculative import diagnostic_cache_digest


STRATEGY = "triton_nf4_fixed_three_rows_group_scale_packed"
EOS_ID = 151645


def forward(model, token_ids: torch.Tensor, cache: TargetKVCache) -> torch.Tensor:
    with torch.no_grad():
        output = model(token_ids, past_key_values=cache.cache, use_cache=True)
    return output.logits[:, -1:, :]


def fixed_forward(model, token_ids: torch.Tensor, cache: TargetKVCache) -> torch.Tensor:
    with (
        fixed_row_schedule_for_model(
            model,
            physical_rows=3,
            attention_strategy="triton_batched",
            quantized_linear_strategy=STRATEGY,
        ),
        fixed_row_sdpa_decode(model, physical_rows=3, block_k=32,
                              attention_strategy="triton_batched"),
    ):
        return forward(model, token_ids, cache)


def compare(reference: torch.Tensor, candidate: torch.Tensor, expected_token: int | None) -> dict:
    ref = reference.float().reshape(-1)
    got = candidate.float().reshape(-1)
    ref_top = torch.topk(ref, k=2).values
    got_top = torch.topk(got, k=2).values
    ref_argmax = int(ref.argmax().item())
    got_argmax = int(got.argmax().item())
    forced_ref = float(ref[expected_token].item()) if expected_token is not None else None
    forced_got = float(got[expected_token].item()) if expected_token is not None else None
    return {
        "logits_bitwise_equal": bool(torch.equal(reference, candidate)),
        "max_abs_logit_diff": float((ref - got).abs().max().item()),
        "argmax_bnb": ref_argmax,
        "argmax_fixed_three_row": got_argmax,
        "argmax_equal": ref_argmax == got_argmax,
        "forced_bnb_token": expected_token,
        "bnb_top1_margin": float((ref_top[0] - ref_top[1]).item()),
        "fixed_top1_margin": float((got_top[0] - got_top[1]).item()),
        "forced_token_logit_bnb": forced_ref,
        "forced_token_logit_fixed_three_row": forced_got,
        "forced_token_logit_diff": None if forced_ref is None else forced_got - forced_ref,
    }


def first_index(records: list[dict], key: str) -> int | None:
    return next((int(row["output_index"]) for row in records if not row[key]), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", required=True,
                        help="Prompt JSONL; pass multiple files to combine development cohorts")
    parser.add_argument("--artifact", action="append", required=True,
                        help="Corrected actual_fixed_three_rows_*.json artifact; repeat per prompt")
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    validator = repo / "scripts/validate_nf4_singleton_k2_r3_manifest.py"
    subprocess.run(
        [sys.executable, str(validator), "--manifest", str(args.manifest)],
        cwd=repo,
        check=True,
    )
    manifest = json.loads(args.manifest.read_text())
    manifest_sources = manifest.get("source_sha256", {})
    if "scripts/audit_actual_fixed_three_rows_bnb_replay.py" not in manifest_sources:
        raise ValueError("Replay script must be pinned in the experiment manifest before running")
    prompts = {}
    for benchmark in args.benchmark:
        benchmark_rows = [
            json.loads(line)
            for line in Path(benchmark).read_text().splitlines()
            if line.strip()
        ]
        prompts.update({str(row["id"]): row["prompt"] for row in benchmark_rows})
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    loaded = load_backend(
        "hf_bnb_nf4",
        "Qwen/Qwen2.5-3B-Instruct",
        "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "cuda:0",
        "sdpa",
    )
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    details = []
    for artifact_rel in args.artifact:
        artifact_path = Path(artifact_rel)
        if not artifact_path.is_absolute():
            artifact_path = repo / artifact_path
        saved = json.loads(artifact_path.read_text())
        if saved.get("parameters", {}).get("quantized_linear_strategy") != STRATEGY:
            raise ValueError(f"artifact does not record corrected fixed-three-row strategy: {artifact_path}")
        prompt_id = str(saved["prompt_id"])
        if prompt_id not in prompts:
            raise KeyError(f"prompt {prompt_id!r} missing from benchmarks {args.benchmark}")
        rounds = saved["timing"]["paired_rounds"]
        bnb_tokens = rounds[0]["methods"]["original_bnb_greedy"]["token_ids"]
        if not bnb_tokens:
            raise ValueError(f"empty BNB token path in {artifact_path}")
        # End at the first EOS; forced replay after EOS would not describe greedy decoding.
        forced = [int(x) for x in bnb_tokens[: args.max_steps]]
        if EOS_ID in forced:
            forced = forced[: forced.index(EOS_ID) + 1]

        ids = tokenizer(prompts[prompt_id], return_tensors="pt")["input_ids"].to("cuda:0")
        capacity = int(ids.shape[-1]) + len(forced) + 4
        bnb_cache = TargetKVCache(backend="static", max_capacity=capacity)
        fixed_cache = TargetKVCache(backend="static", max_capacity=capacity)

        bnb_logits = forward(model, ids, bnb_cache)
        fixed_logits = fixed_forward(model, ids, fixed_cache)
        trace = [
            {
                "output_index": 0,
                "stage": "prompt_prefill",
                **compare(bnb_logits, fixed_logits, forced[0]),
                "bnb_kv_sha256": diagnostic_cache_digest(bnb_cache),
                "fixed_three_row_kv_sha256": diagnostic_cache_digest(fixed_cache),
            }
        ]
        for consumed_index, token_id in enumerate(forced):
            token = torch.tensor([[token_id]], device="cuda:0")
            bnb_logits = forward(model, token, bnb_cache)
            fixed_logits = fixed_forward(model, token, fixed_cache)
            output_index = consumed_index + 1
            expected = forced[output_index] if output_index < len(forced) else None
            trace.append(
                {
                    "output_index": output_index,
                    "stage": "decode_after_forced_token",
                    "consumed_bnb_token": token_id,
                    **compare(bnb_logits, fixed_logits, expected),
                    "bnb_kv_sha256": diagnostic_cache_digest(bnb_cache),
                    "fixed_three_row_kv_sha256": diagnostic_cache_digest(fixed_cache),
                }
            )

        for row in trace:
            row["kv_equal"] = row["bnb_kv_sha256"] == row["fixed_three_row_kv_sha256"]
        detail = {
            "artifact": str(artifact_path.relative_to(repo)),
            "prompt_id": prompt_id,
            "session": saved["session"],
            "budget": saved["parameters"]["max_new_tokens"],
            "forced_bnb_tokens": len(forced),
            "first_logit_bitwise_drift_index": first_index(trace, "logits_bitwise_equal"),
            "first_argmax_drift_index": first_index(trace, "argmax_equal"),
            "first_kv_drift_output_index": first_index(trace, "kv_equal"),
            "first_argmax_drift_record": next((r for r in trace if not r["argmax_equal"]), None),
            "first_8_trace_rows": trace[:8],
            "trace": trace,
        }
        details.append(detail)
        print(
            json.dumps(
                {k: v for k, v in detail.items() if k not in {"trace", "first_8_trace_rows", "first_argmax_drift_record"}},
                indent=2,
            ),
            flush=True,
        )

    result = {
        "classification": "forced BNB-prefix logit/KV replay; empirical diagnostic, not proof",
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "gpu": torch.cuda.get_device_name(0),
        "reference": "original BNB NF4 singleton",
        "candidate": STRATEGY,
        "attention_strategy_candidate": "triton_batched + fixed_row_sdpa_decode(block_k=32)",
        "details": details,
        "limitations": [
            "Forced-token replay aligns token inputs, but KV states may diverge before argmax changes.",
            "This compares BNB SDPA with the fixed-three-row B1 Triton attention schedule as deployed.",
            "Finite prompts, one GPU, and one model revision only; held-out GSM8K was not used.",
        ],
    }
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
