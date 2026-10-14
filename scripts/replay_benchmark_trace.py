#!/usr/bin/env python3
"""Replay saved benchmark prompts through production and two tracing modes.

Requires the model dependencies and the configured CUDA GPU. Each mode calls
prompt_lookup_generate independently, which creates a fresh TargetKVCache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments/reproduction_audit/live_replay.json"


def json_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":")).encode()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_cycle_difference(left: list[dict], right: list[dict]) -> dict | None:
    keys = ("prefix_token_ids_hash", "verification_input_token_id", "draft_candidate_ids",
            "accepted_prefix_length", "emitted_token_ids", "selected_next_token_id",
            "kv_length_before", "kv_length_after",
            "remaining_budget_before", "eos_reached", "verification_mode")
    for i in range(max(len(left), len(right))):
        a = left[i] if i < len(left) else None
        b = right[i] if i < len(right) else None
        if a is None or b is None:
            return {"cycle_index": i, "field": "cycle_count", "left": len(left), "right": len(right)}
        for key in keys:
            if key not in a or key not in b or a[key] is None or b[key] is None:
                continue
            if a.get(key) != b.get(key):
                return {"cycle_index": i, "field": key, "left": a.get(key), "right": b.get(key)}
    return None


def normalize_numerical_trace(rows: list[dict], eos_id: int, max_new_tokens: int) -> list[dict]:
    normalized = []
    emitted_so_far = 0
    for cycle in rows:
        mode = cycle["verification_mode"]
        raw_candidates = cycle.get("candidate_tokens", [])
        drafts = raw_candidates[1:] if mode != "vanilla_fallback" else []
        before = cycle.get("cache_before", [])
        after = cycle.get("cache_after", [])
        cache_before = before[0][2] if before else None
        cache_after = after[0][2] if after else None
        emitted = list(cycle.get("emitted_tokens", []))
        prefix = cycle.get("prefix_tokens", [])
        normalized.append({
            "cycle_index": cycle["cycle_index"],
            "prefix_token_ids_hash": json_hash(prefix),
            "verification_input_token_id": raw_candidates[0] if raw_candidates else None,
            "draft_candidate_ids": drafts,
            "accepted_prefix_length": cycle.get("accepted_count", 0),
            "emitted_token_ids": emitted,
            "kv_length_before": cache_before,
            "kv_length_after": cache_after,
            "remaining_budget_before": max_new_tokens - emitted_so_far,
            "eos_reached": eos_id in emitted,
            "selected_next_token_id": cycle.get("selected_next_token_id"),
            "verification_mode": mode,
        })
        emitted_so_far += len(emitted)
    return normalized


def normalize_archived_trace(rows: list[dict], max_new_tokens: int) -> list[dict]:
    normalized = []
    emitted_so_far = 0
    for cycle in rows:
        mode = cycle.get("verification_mode")
        raw_candidates = cycle.get("candidate_tokens", [])
        observed = cycle.get("observed_batch_token_ids", [])
        if mode == "vanilla_fallback":
            selected = observed[0] if observed else None
            drafts = []
        else:
            drafts = raw_candidates[1:]
            row_index = cycle.get("rejected_at")
            if row_index is None:
                row_index = max(0, len(raw_candidates) - 1)
            selected = observed[row_index] if row_index < len(observed) else None
        emitted = list(cycle.get("emitted_tokens", []))
        normalized.append({
            "cycle_index": cycle.get("cycle_index"),
            "verification_input_token_id": raw_candidates[0] if raw_candidates else None,
            "draft_candidate_ids": drafts,
            "accepted_prefix_length": cycle.get("accepted_count", 0),
            "emitted_token_ids": emitted,
            "selected_next_token_id": selected,
            "kv_length_before": cycle.get("cache_length_before"),
            "kv_length_after": cycle.get("cache_length_after"),
            "remaining_budget_before": max_new_tokens - emitted_so_far,
            "eos_reached": None,
            "verification_mode": mode,
        })
        emitted_so_far += len(emitted)
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--double-quant", action=argparse.BooleanOptionalAction, default=True,
                        help="Match the independent benchmark's NF4 double quantization setting")
    parser.add_argument("--historical-trace", type=Path,
                        default=ROOT / "experiments/numerical_causality/traces/multicycle_nf4_sdpa_static.json")
    args = parser.parse_args()
    inputs_path = args.inputs if args.inputs.is_absolute() else ROOT / args.inputs
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this replay")

    rows = read_rows(inputs_path)
    wanted = set(args.ids or [])
    prompts = [row for row in rows if not wanted or row["prompt_id"] in wanted][:args.samples]
    if wanted and {r["prompt_id"] for r in prompts} != wanted:
        missing = sorted(wanted - {r["prompt_id"] for r in prompts})
        raise ValueError(f"Prompt IDs not found in input file: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=args.double_quant)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=quant,
        device_map=args.device, attn_implementation=args.attention).eval()

    # Import after model setup to keep this script runnable from a repository checkout.
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from zassd.baselines.prompt_lookup import prompt_lookup_generate

    historical_path = args.historical_trace if args.historical_trace.is_absolute() else ROOT / args.historical_trace
    historical = json.loads(historical_path.read_text()) if historical_path.is_file() else None
    historical_by_id = {p["prompt_id"]: p for p in historical.get("prompts", [])} if historical else {}
    prompt_results = []
    for row in prompts:
        prompt = torch.tensor([row["input_ids"]], dtype=torch.long, device=args.device)
        capacity = int(prompt.shape[-1]) + args.max_new_tokens + args.k + 16
        runs = {}
        for mode in ("production", "lightweight_trace", "causal_trace"):
            lightweight: list[dict] = []
            numerical: list[dict] = []
            kwargs = {"lightweight_trace": lightweight} if mode == "lightweight_trace" else {}
            if mode == "causal_trace":
                kwargs["numerical_trace"] = numerical
            _, metrics = prompt_lookup_generate(
                model, tokenizer, prompt, k=args.k, ngram_size=3,
                max_new_tokens=args.max_new_tokens, temperature=0.0,
                device=args.device, kv_cache_backend="static",
                candidate_source="prompt+generated", sequential_verify=False,
                kv_cache_max_capacity=capacity, **kwargs)
            trajectory = lightweight if mode == "lightweight_trace" else (
                normalize_numerical_trace(numerical, tokenizer.eos_token_id, args.max_new_tokens)
                if mode == "causal_trace" else None)
            runs[mode] = {
                "generated_token_ids": metrics.generated_token_ids,
                "generated_token_ids_sha256": json_hash(metrics.generated_token_ids),
                "cycle_count": len(metrics.per_iteration_stats),
                "trajectory": trajectory,
            }
        light = runs["lightweight_trace"]["trajectory"]
        causal = runs["causal_trace"]["trajectory"]
        historical_prompt = historical_by_id.get(row["prompt_id"])
        historical_trajectory = normalize_archived_trace(historical_prompt.get("cycles", []), args.max_new_tokens) if historical_prompt else None
        prompt_results.append({
            "prompt_id": row["prompt_id"],
            "prompt_token_ids_sha256": json_hash(row["input_ids"]),
            "prompt_length": len(row["input_ids"]),
            "cache_capacity": capacity,
            "output_equal_production_vs_lightweight": runs["production"]["generated_token_ids"] == runs["lightweight_trace"]["generated_token_ids"],
            "output_equal_lightweight_vs_causal": runs["lightweight_trace"]["generated_token_ids"] == runs["causal_trace"]["generated_token_ids"],
            "first_cycle_difference_lightweight_vs_causal": first_cycle_difference(light, causal),
            "first_cycle_difference_current_vs_archived_trace": first_cycle_difference(light, historical_trajectory) if historical_trajectory is not None else None,
            "archived_trace_trajectory": historical_trajectory,
            "runs": runs,
        })
        print(f"replayed {row['prompt_id']}", flush=True)

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                text=True, capture_output=True, check=True).stdout.strip()
    except Exception:
        commit = None
    result = {
        "schema_version": 1,
        "model": args.model,
        "model_revision": args.revision,
        "quantization": "NF4",
        "bnb_4bit_use_double_quant": args.double_quant,
        "compute_dtype": "float16",
        "attention": args.attention,
        "kv_backend": "static",
        "k": args.k,
        "ngram_size": 3,
        "candidate_source": "prompt+generated",
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "execution_commit": commit,
        "input_path": str(inputs_path.relative_to(ROOT) if inputs_path.is_relative_to(ROOT) else inputs_path),
        "input_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "prompts": prompt_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
