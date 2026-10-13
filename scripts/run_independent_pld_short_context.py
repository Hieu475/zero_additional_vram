#!/usr/bin/env python3
"""Paired short-context PLD validation on source documents unused in prior runs."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from prepare_natural_context_inputs import prepare  # noqa: E402
from run_heldout_roofline_validation import (  # noqa: E402
    append_jsonl, gpu_snapshot, measure_vanilla, package_version,
)
from zassd.baselines.prompt_lookup import prompt_lookup_generate  # noqa: E402
from zassd.decoding.speculative import self_speculative_generate  # noqa: E402

PLD_K_VALUES = (2, 4, 6, 8)
OLD_INPUTS = (
    ROOT / "experiments/natural_context_validation/pilot_rtx4050/prepared_inputs.jsonl",
    ROOT / "experiments/natural_context_validation/full_n20_pld_rtx4050/prepared_inputs.jsonl",
    ROOT / "experiments/natural_context_validation/repeat2_n20_pld_rtx4050/prepared_inputs.jsonl",
)


def speculative_core(model, tokenizer, prompt: torch.Tensor, max_new_tokens: int,
                     device: str) -> dict[str, Any]:
    _, metrics = self_speculative_generate(
        model=model, tokenizer=tokenizer, layer_mgr=None, skip_indices=[], prompt=prompt,
        k=0, max_new_tokens=max_new_tokens, temperature=0.0, device=device,
        kv_cache_backend="static", kv_cache_max_capacity=int(prompt.shape[-1]) + max_new_tokens + 16,
    )
    decode_s = max(1e-9, metrics.total_time_s - metrics.prefill_time_s)
    return {
        "output_tokens": metrics.total_tokens, "decode_tokens": max(0, metrics.total_tokens - 1),
        "prefill_ms": metrics.prefill_time_s * 1000, "decode_ms": decode_s * 1000,
        "total_ms": metrics.total_time_s * 1000,
        "decode_tps": max(0, metrics.total_tokens - 1) / decode_s,
        "token_ids": metrics.generated_token_ids,
    }


def measure_pld(model, tokenizer, prompt: torch.Tensor, k: int, max_new_tokens: int,
                device: str, force_miss: bool = False) -> dict[str, Any]:
    _, metrics = prompt_lookup_generate(
        model, tokenizer, prompt, k=k, max_new_tokens=max_new_tokens, temperature=0.0,
        device=device, kv_cache_backend="static", candidate_source="prompt+generated",
        force_miss=force_miss,
        kv_cache_max_capacity=int(prompt.shape[-1]) + max_new_tokens + k + 16,
    )
    decode_s = max(1e-9, metrics.total_time_s - metrics.prefill_time_s)
    cycles = metrics.per_iteration_stats
    return {
        "output_tokens": metrics.total_tokens, "decode_tokens": max(0, metrics.total_tokens - 1),
        "prefill_ms": metrics.prefill_time_s * 1000, "decode_ms": decode_s * 1000,
        "total_ms": metrics.total_time_s * 1000,
        "decode_tps": max(0, metrics.total_tokens - 1) / decode_s,
        "token_ids": metrics.generated_token_ids,
        "verification_cycles": metrics.num_verification_cycles,
        "pld_cycles": metrics.pld_cycles,
        "lookup_hit_rate": metrics.pld_cycles / max(1, metrics.num_verification_cycles),
        "accepted_tokens": sum(int(c.get("accepted", 0)) for c in cycles),
        "draft_tokens": sum(int(c.get("k", 0)) for c in cycles),
        "accepted_tokens_per_cycle": sum(int(c.get("accepted", 0)) for c in cycles)
                                    / max(1, metrics.num_verification_cycles),
        "cycle_stats": cycles,
    }


def bootstrap_mean(values: list[float], seed: int, draws: int = 10000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = [sum(values[rng.randrange(len(values))] for _ in values) / len(values)
             for _ in range(draws)]
    means.sort()
    return [means[int(.025 * draws)], means[int(.975 * draws)]]


def analyze(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    paired: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        paired.setdefault(row["prompt_id"], {})[row["method_key"]] = row
    if any("vanilla" not in pair for pair in paired.values()):
        raise ValueError("Every prompt must have a vanilla measurement")
    results: dict[str, Any] = {}
    keys = sorted({key for pair in paired.values() for key in pair if key != "vanilla"})
    for method in keys:
        dec, end, exact = [], [], []
        for pair in paired.values():
            if method not in pair:
                raise ValueError(f"Missing method {method} in a paired prompt")
            vanilla, action = pair["vanilla"]["measured"], pair[method]["measured"]
            dec.append(action["decode_tps"] / max(vanilla["decode_tps"], 1e-9))
            end.append(vanilla["total_ms"] / max(action["total_ms"], 1e-9))
            exact.append(action["token_ids"] == vanilla["token_ids"])
        results[method] = {
            "n_prompts": len(dec), "mean_decode_speedup": sum(dec) / len(dec),
            "decode_speedup_prompt_bootstrap_95ci": bootstrap_mean(dec, seed),
            "median_decode_speedup": sorted(dec)[len(dec) // 2],
            "mean_end_to_end_speedup": sum(end) / len(end),
            "end_to_end_prompt_bootstrap_95ci": bootstrap_mean(end, seed + 1),
            "sequence_exact_match": sum(exact) / len(exact),
            "exact_match_count": sum(exact),
            "measured_profitable_count": sum(x > 1.0 for x in dec),
            "speedup_p10_p90": [sorted(dec)[int(.1 * (len(dec) - 1))],
                                sorted(dec)[int(.9 * (len(dec) - 1))]],
        }
    return {"n_prompts": len(paired), "methods": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-bucket", type=int, default=60)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--order-seed", type=int, default=20261014)
    parser.add_argument("--prepared-inputs", type=Path, default=None,
                        help="Reuse a previously prepared independent prompt file for a session repeat")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA device is required")
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    inputs_path = out / "prepared_inputs.jsonl"
    if args.prepared_inputs:
        source_inputs = args.prepared_inputs if args.prepared_inputs.is_absolute() else ROOT / args.prepared_inputs
        records = [json.loads(line) for line in source_inputs.read_text().splitlines() if line.strip()]
        inputs_path.write_text(source_inputs.read_text())
        source_manifest = {"reused_prepared_inputs": str(source_inputs),
                           "prepared_sha256": __import__("hashlib").sha256(inputs_path.read_bytes()).hexdigest(),
                           "n_examples": len(records),
                           "bucket_counts": {str(b): sum(r["prompt_tokens"] == b for r in records)
                                             for b in sorted({r["prompt_tokens"] for r in records})}}
    else:
        exclusions = tuple(p for p in OLD_INPUTS if p.exists())
        source_manifest = prepare(inputs_path, args.model, args.revision, args.per_bucket,
                                  (384,), args.seed, exclusions)
        records = [json.loads(line) for line in inputs_path.read_text().splitlines() if line.strip()]
    if len(records) != args.per_bucket and not args.prepared_inputs:
        raise ValueError("Prepared prompt count differs from requested short-context sample size")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.strip()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=quant,
        device_map=args.device, attn_implementation="sdpa").eval()
    config = {
        "execution_commit": commit, "model": args.model, "model_revision": args.revision,
        "quantization": "NF4", "compute_dtype": "float16", "attention": "sdpa",
        "kv_backend": "static", "max_new_tokens": args.max_new_tokens,
        "context_tokens": 384, "sample_seed": args.seed, "method_order_seed": args.order_seed,
        "methods": ["vanilla", "vanilla_speculative_core_K0", "PLD_forced_miss_K4",
                    *[f"PLD_prompt_plus_generated_K{k}" for k in PLD_K_VALUES]],
        "source_manifest": source_manifest,
        "excluded_source_files": [str(p.relative_to(ROOT)) for p in OLD_INPUTS if p.exists()],
        "measurement": "one paired observation per prompt; shuffled method order; separate prefill/decode/e2e",
        "prior_pilot_policy": "All LongBench source IDs from pilot and both previous full sessions excluded",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    env = {"commit_sha": commit, "python": platform.python_version(), "torch": torch.__version__,
           "transformers": package_version("transformers"), "bitsandbytes": package_version("bitsandbytes"),
           "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(args.device),
           "initial_telemetry": gpu_snapshot(args.device)}
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")

    warm = torch.tensor([records[0]["input_ids"]], dtype=torch.long, device=args.device)
    measure_vanilla(model, tokenizer, warm, min(8, args.max_new_tokens), args.device)
    speculative_core(model, tokenizer, warm, min(8, args.max_new_tokens), args.device)
    measure_pld(model, tokenizer, warm, 4, min(8, args.max_new_tokens), args.device, force_miss=True)
    for k in PLD_K_VALUES:
        measure_pld(model, tokenizer, warm, k, min(8, args.max_new_tokens), args.device)

    rng = random.Random(args.order_seed)
    raw_path = out / "raw_measurements.jsonl"
    rows: list[dict[str, Any]] = []
    for record in records:
        prompt = torch.tensor([record["input_ids"]], dtype=torch.long, device=args.device)
        jobs = ["vanilla", "vanilla_speculative_core_K0", "PLD_forced_miss_K4",
                *[f"PLD_prompt_plus_generated_K{k}" for k in PLD_K_VALUES]]
        rng.shuffle(jobs)
        pending = []
        for method in jobs:
            before = gpu_snapshot(args.device)
            if method == "vanilla":
                measured = measure_vanilla(model, tokenizer, prompt, args.max_new_tokens, args.device)
                measured["total_ms"] = measured["prefill_ms"] + measured["decode_ms"]
            elif method == "vanilla_speculative_core_K0":
                measured = speculative_core(model, tokenizer, prompt, args.max_new_tokens, args.device)
            elif method == "PLD_forced_miss_K4":
                measured = measure_pld(model, tokenizer, prompt, 4, args.max_new_tokens,
                                       args.device, force_miss=True)
            else:
                k = int(method.rsplit("K", 1)[1])
                measured = measure_pld(model, tokenizer, prompt, k, args.max_new_tokens,
                                       args.device)
            row = {"prompt_id": record["prompt_id"], "source_task": record["source_task"],
                   "source_id": record["source_id"], "context_bucket": record["context_bucket"],
                   "prompt_tokens": record["prompt_tokens"], "method_key": method,
                   "method": "vanilla" if method == "vanilla" else (
                       "vanilla_core" if method == "vanilla_speculative_core_K0" else "prompt_lookup"),
                   "k": (int(method.rsplit("K", 1)[1]) if "_K" in method else 0),
                   "measured": measured, "method_order_seed": args.order_seed,
                   "telemetry_before": before, "telemetry_after": gpu_snapshot(args.device)}
            pending.append(row)
        for row in pending:
            append_jsonl(raw_path, row)
            rows.append(row)
        print(f"completed {record['prompt_id']} ({record['prompt_tokens']} tokens)", flush=True)
    summary = analyze(rows, args.seed)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"DONE: {out}")


if __name__ == "__main__":
    main()
