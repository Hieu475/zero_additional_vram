#!/usr/bin/env python3
"""Paired diagnostic run of the B1 singleton schedule against greedy static KV.

Use development prompts only. This runner never reads the frozen held-out set.
The verifier is serial; a speedup is neither expected nor claimed.
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
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.b1_exact import b1_exact_generate, one_token_transition
from zassd.decoding.speculative import diagnostic_cache_digest


INPUTS = {
    "gsm8k": "data/benchmarks/gsm8k_eval.jsonl",
    "humaneval": "data/benchmarks/humaneval_eval.jsonl",
    "cnndm": "data/benchmarks/cnndm_eval.jsonl",
}


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def audit_state(model, tokenizer, ids, max_new_tokens, device, capacity, k):
    """Untimed prefix-by-prefix KV comparison against singleton greedy."""
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.no_grad():
        prefill = model(ids.to(device), past_key_values=cache.cache, use_cache=True).logits
    token = prefill[:, -1:, :].argmax(-1)
    reference_states = [(0, diagnostic_cache_digest(cache))]
    reference_ids = []
    while len(reference_ids) < max_new_tokens:
        value = int(token.item())
        reference_ids.append(value)
        if value == tokenizer.eos_token_id or len(reference_ids) == max_new_tokens:
            break
        token = one_token_transition(model, token, cache)
        reference_states.append((len(reference_ids), diagnostic_cache_digest(cache)))
    b1_states = []
    _, result = b1_exact_generate(
        model, tokenizer, ids, k=k, max_new_tokens=max_new_tokens,
        device=device, kv_cache_max_capacity=capacity,
        state_observer=lambda consumed, state: b1_states.append(
            (consumed, diagnostic_cache_digest(state))),
    )
    first_difference = next((index for index, (a, b) in enumerate(zip(reference_states, b1_states))
                             if a != b), None)
    if first_difference is None and len(reference_states) != len(b1_states):
        first_difference = min(len(reference_states), len(b1_states))
    return {
        "token_exact": reference_ids == result.generated_token_ids,
        "state_exact": first_difference is None,
        "first_state_divergence": first_difference,
        "reference_state_count": len(reference_states),
        "b1_state_count": len(b1_states),
        "reference_state_digests": reference_states,
        "b1_state_digests": b1_states,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--backend", choices=["hf_bnb_nf4", "hqq_gemlite_w4a16"],
                        default="hf_bnb_nf4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-task", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--audit-state", action="store_true")
    args = parser.parse_args()
    capacity = 2048  # run_vanilla_timed uses the same static capacity.
    if args.per_task < 1 or args.max_new_tokens < 1 or args.k < 0:
        raise ValueError("invalid per-task, max-new-tokens or k")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this pilot")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)

    prompts = []
    for task, path in INPUTS.items():
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if len(rows) < args.per_task:
            raise ValueError(f"{path} has fewer than {args.per_task} prompts")
        prompts.extend((task, row["id"], row["prompt"]) for row in rows[:args.per_task])
    loaded = load_backend(args.backend, args.model, args.revision, args.device, "sdpa")
    model, tokenizer = loaded.model, loaded.tokenizer
    manifest = {
        "status": "diagnostic; singleton target schedule, no batched 4-bit kernel",
        "reference": "same model object, static KV, prompt prefill, singleton token forward",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "model": args.model, "revision": args.revision,
        "backend": args.backend, "backend_metadata": loaded.metadata,
        "device": args.device, "gpu": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": _version("transformers"), "bitsandbytes": _version("bitsandbytes"),
        "python": platform.python_version(),
        "k": args.k, "max_new_tokens": args.max_new_tokens,
        "audit_state": args.audit_state,
        "capacity": capacity, "per_task": args.per_task,
        "inputs": {task: {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                   for task, path in INPUTS.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    warmup = tokenizer(prompts[0][2], return_tensors="pt")["input_ids"]
    run_vanilla_timed(model, tokenizer, warmup, 4, args.device)
    b1_exact_generate(model, tokenizer, warmup, k=args.k, max_new_tokens=4,
                      device=args.device, kv_cache_max_capacity=capacity)

    results = []
    for index, (task, prompt_id, prompt) in enumerate(prompts):
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].cpu()
        if ids.shape[1] + args.max_new_tokens > capacity:
            raise ValueError(f"{prompt_id} exceeds static capacity")
        methods = ("greedy", "b1") if index % 2 == 0 else ("b1", "greedy")
        pair = {}
        for method in methods:
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)
            if method == "greedy":
                tokens, prefill_ms, decode_ms, total_ms = run_vanilla_timed(
                    model, tokenizer, ids, args.max_new_tokens, args.device)
                pair[method] = {"token_ids": tokens, "total_s": total_ms / 1000,
                                "prefill_s": prefill_ms / 1000,
                                "decode_s": decode_ms / 1000}
            else:
                _, metrics = b1_exact_generate(
                    model, tokenizer, ids, k=args.k,
                    max_new_tokens=args.max_new_tokens, device=args.device,
                    kv_cache_max_capacity=capacity, trace=True)
                pair[method] = {
                    "token_ids": metrics.generated_token_ids,
                    "total_s": metrics.total_time_s,
                    "prefill_s": metrics.prefill_time_s,
                    "draft_s": metrics.draft_time_s,
                    "verify_s": metrics.verify_time_s,
                    "target_forwards": metrics.target_forwards,
                    "draft_tokens": metrics.draft_tokens,
                    "accepted_draft_tokens": metrics.accepted_draft_tokens,
                    "cycles": metrics.cycles,
                    "cycle_trace": metrics.cycle_trace,
                }
            pair[method]["peak_allocated_mb"] = torch.cuda.max_memory_allocated(args.device) / 1e6
        result = {
            "task": task, "id": prompt_id,
            "input_token_ids_sha256": hashlib.sha256(json.dumps(ids[0].tolist()).encode()).hexdigest(),
            "method_order": methods,
            "exact_token_match": pair["b1"]["token_ids"] == pair["greedy"]["token_ids"],
            "speedup_vs_greedy": pair["greedy"]["total_s"] / pair["b1"]["total_s"],
            "greedy": pair["greedy"], "b1": pair["b1"],
        }
        if args.audit_state:
            result["state_audit"] = audit_state(
                model, tokenizer, ids, args.max_new_tokens, args.device, capacity, args.k)
        results.append(result)
        with (output / "rows.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
        print(f"{index + 1}/{len(prompts)} {task} {prompt_id}: "
              f"exact={result['exact_token_match']}, speedup={result['speedup_vs_greedy']:.3f}",
              flush=True)
    summary = {
        "n": len(results),
        "exact": sum(row["exact_token_match"] for row in results),
        "mean_paired_speedup": sum(row["speedup_vs_greedy"] for row in results) / len(results),
        "scope": "single-session development diagnostic; equality is not a universal kernel proof",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
