#!/usr/bin/env python3
"""Paired diagnostic pilot for exact quantized speculative decoding.

This is a screening experiment on previously available prompts. The
"free_restore_ceiling" removes observed canonical replay time from a guard
run while keeping its output trajectory fixed. It is a counterfactual cost
ceiling, not an executable exact decoder or a correctness guarantee.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch

try:
    from run_strong_baseline_comparison import run_vanilla_timed
except ModuleNotFoundError:
    from scripts.run_strong_baseline_comparison import run_vanilla_timed
from zassd.backends.base import load_backend
from zassd.baselines.prompt_lookup import prompt_lookup_generate
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter


DEFAULT_INPUTS = {
    "gsm8k": "data/benchmarks/gsm8k_eval.jsonl",
    "humaneval": "data/benchmarks/humaneval_eval.jsonl",
    "cnndm": "data/benchmarks/cnndm_eval.jsonl",
}
METHODS = ("vanilla", "pld_batched", "pld_sequential", "margin_guard")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_prompts(per_task: int) -> list[dict]:
    if per_task < 1:
        raise ValueError("per_task must be positive")
    prompts: list[dict] = []
    for task, path in DEFAULT_INPUTS.items():
        with open(path, encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if len(rows) < per_task:
            raise ValueError(f"{path} contains only {len(rows)} prompts")
        for row in rows[:per_task]:
            prompts.append({"task": task, "id": str(row["id"]),
                            "prompt": row["prompt"]})
    return prompts


def free_restore_ceiling(tokens: int, total_s: float, replay_s: float,
                         vanilla_total_s: float) -> dict:
    """Cost ceiling conditional on the observed guard trajectory.

    Assumes restoration of exactly the same canonical state costs zero and
    causes no change in any subsequent action. It is intentionally optimistic.
    """
    if tokens <= 0 or total_s <= 0 or vanilla_total_s <= 0:
        raise ValueError("nonpositive token count or latency")
    if not 0 <= replay_s < total_s:
        raise ValueError("replay time must be in [0, total time)")
    return {
        "replay_fraction_of_guard_time": replay_s / total_s,
        "free_restore_time_s": total_s - replay_s,
        "free_restore_tps": tokens / (total_s - replay_s),
        "free_restore_speedup_vs_vanilla": vanilla_total_s / (total_s - replay_s),
        "guard_speedup_vs_vanilla": vanilla_total_s / total_s,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    ap.add_argument("--backend", choices=["hf_bnb_nf4", "hqq_gemlite_w4a16"],
                    default="hf_bnb_nf4")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--per-task", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--margin-threshold", type=float, default=1.5)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.k < 1 or args.max_new_tokens < 2:
        raise ValueError("k must be >=1 and max-new-tokens >=2")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.per_task)
    loaded = load_backend(args.backend, args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model, loaded.tokenizer
    manager = LayerManager(ModelAdapter(model))
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    input_files = {task: {"path": path,
                          "sha256": sha256_bytes(Path(path).read_bytes())}
                   for task, path in DEFAULT_INPUTS.items()}
    manifest = {
        "purpose": "diagnostic pilot; prompts are not held out from prior ZASSD work",
        "git_commit": git_commit,
        "model": args.model, "revision": args.revision,
        "backend": args.backend, "backend_metadata": loaded.metadata,
        "reference": "same quantized checkpoint/backend, SDPA, static KV=2048, "
                     "prompt prefill then one-token greedy target forwards",
        "candidate_source": "PLD history, ngram_size=3",
        "methods": METHODS, "method_order": "rotated once by prompt index",
        "per_task": args.per_task, "max_new_tokens": args.max_new_tokens,
        "k": args.k, "margin_threshold": args.margin_threshold,
        "input_files": input_files,
        "python": platform.python_version(),
        "torch": torch.__version__, "transformers": version("transformers"),
        "bitsandbytes": version("bitsandbytes"),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
        "gpu_total_memory_mb": torch.cuda.get_device_properties(args.device).total_memory / 1e6,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # One short warmup for all paths, outside all timed rows.
    warm_ids = tokenizer(prompts[0]["prompt"], return_tensors="pt")["input_ids"]
    run_vanilla_timed(model, tokenizer, warm_ids, 8, args.device)
    prompt_lookup_generate(model, tokenizer, warm_ids, k=args.k, max_new_tokens=8,
                           device=args.device, kv_cache_max_capacity=2048)
    self_speculative_generate(model, tokenizer, manager, [], warm_ids, k=args.k,
                              draft_mode="prompt_lookup", max_new_tokens=8,
                              device=args.device, margin_fallback_threshold=args.margin_threshold)

    rows: list[dict] = []
    for index, item in enumerate(prompts):
        ids = tokenizer(item["prompt"], return_tensors="pt")["input_ids"].cpu()
        if ids.shape[1] + args.max_new_tokens + args.k + 1 > 2048:
            raise ValueError(f"prompt {item['id']} exceeds static KV capacity")
        input_hash = sha256_bytes(json.dumps(ids[0].tolist(), separators=(",", ":")).encode())
        method_order = METHODS[index % len(METHODS):] + METHODS[:index % len(METHODS)]
        method_results: dict[str, dict] = {}
        for method in method_order:
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            if method == "vanilla":
                out_ids, prefill_ms, decode_ms, total_ms = run_vanilla_timed(
                    model, tokenizer, ids, args.max_new_tokens, args.device)
                method_results[method] = {
                    "token_ids": out_ids, "token_count": len(out_ids),
                    "total_s": total_ms / 1000, "prefill_s": prefill_ms / 1000,
                    "decode_s": decode_ms / 1000,
                }
            elif method in {"pld_batched", "pld_sequential"}:
                _, metrics = prompt_lookup_generate(
                    model, tokenizer, ids, k=args.k, max_new_tokens=args.max_new_tokens,
                    device=args.device, kv_cache_max_capacity=2048,
                    sequential_verify=(method == "pld_sequential"),
                )
                method_results[method] = {
                    "token_ids": metrics.generated_token_ids,
                    "token_count": len(metrics.generated_token_ids),
                    "total_s": metrics.total_time_s,
                    "prefill_s": metrics.prefill_time_s,
                    "decode_s": metrics.total_time_s - metrics.prefill_time_s,
                    "draft_s": metrics.draft_time_s,
                    "verify_s": metrics.verify_time_s,
                    "cycles": len(metrics.per_iteration_stats),
                }
            else:
                _, metrics = self_speculative_generate(
                    model, tokenizer, manager, [], ids, k=args.k,
                    draft_mode="prompt_lookup", max_new_tokens=args.max_new_tokens,
                    device=args.device, margin_fallback_threshold=args.margin_threshold,
                    kv_cache_max_capacity=2048,
                )
                method_results[method] = {
                    "token_ids": metrics.generated_token_ids,
                    "token_count": len(metrics.generated_token_ids),
                    "total_s": metrics.total_time_s,
                    "prefill_s": metrics.prefill_time_s,
                    "decode_s": metrics.total_time_s - metrics.prefill_time_s,
                    "draft_s": metrics.draft_time_s,
                    "verify_s": metrics.verify_time_s,
                    "canonical_replay_s": metrics.canonical_replay_time_s,
                    "fallback_cycles": metrics.margin_fallback_cycles,
                    "cycles": len(metrics.per_iteration_stats),
                }
            method_results[method]["peak_allocated_mb"] = (
                torch.cuda.max_memory_allocated(args.device) / 1e6
            )
        ref = method_results["vanilla"]
        for method in METHODS:
            result = method_results[method]
            result["exact_token_match"] = result["token_ids"] == ref["token_ids"]
            result["speedup_vs_vanilla"] = ref["total_s"] / result["total_s"]
            result["task"] = item["task"]
            result["id"] = item["id"]
            result["input_token_ids_sha256"] = input_hash
            result["method"] = method
            if method == "margin_guard":
                result["free_restore_ceiling"] = free_restore_ceiling(
                    result["token_count"], result["total_s"],
                    result["canonical_replay_s"], ref["total_s"])
            rows.append(result)
        with (output / "rows.jsonl").open("a", encoding="utf-8") as stream:
            for row in rows[-len(METHODS):]:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(f"{index + 1}/{len(prompts)} {item['task']} {item['id']}", flush=True)
    summary: dict[str, dict] = {}
    for method in METHODS:
        subset = [row for row in rows if row["method"] == method]
        summary[method] = {
            "n_prompts": len(subset),
            "exact_count": sum(row["exact_token_match"] for row in subset),
            "mean_speedup_vs_vanilla": sum(row["speedup_vs_vanilla"] for row in subset) / len(subset),
        }
    guard_rows = [row for row in rows if row["method"] == "margin_guard"]
    summary["margin_guard"]["mean_replay_fraction"] = sum(
        row["free_restore_ceiling"]["replay_fraction_of_guard_time"] for row in guard_rows
    ) / len(guard_rows)
    summary["margin_guard"]["mean_free_restore_ceiling_speedup"] = sum(
        row["free_restore_ceiling"]["free_restore_speedup_vs_vanilla"] for row in guard_rows
    ) / len(guard_rows)
    summary["interpretation"] = (
        "Free restoration is a cost-only ceiling conditional on the measured guard trajectory; "
        "it does not establish that the corresponding decoder exists or is exact. "
        "This is a small diagnostic cohort, not held-out evidence."
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
