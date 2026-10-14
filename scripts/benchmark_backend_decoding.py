#!/usr/bin/env python3
"""Paired per-backend diagnostic benchmark for vanilla, PLD, and CKA90/K2."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zassd.backends import load_backend  # noqa: E402
from zassd.baselines.prompt_lookup import prompt_lookup_generate  # noqa: E402
from zassd.cache.kv_cache import TargetKVCache  # noqa: E402
from zassd.decoding.speculative import self_speculative_generate  # noqa: E402
from zassd.models.layer_manager import LayerManager  # noqa: E402
from zassd.models.model_adapter import ModelAdapter  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
INPUTS = ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl"
SKIP = [4, 5, 6, 7]
METHODS = ("vanilla", "pld_k4", "cka90_k2")


def direct_vanilla(model: Any, prompt: torch.Tensor, eos_id: int | None,
                   budget: int, capacity: int, device: str) -> dict:
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    generated: list[int] = []
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(prompt, past_key_values=cache.cache, use_cache=True)
        torch.cuda.synchronize(device)
        prefill_end = time.perf_counter()
        next_id = int(out.logits[0, -1].argmax().item())
        while len(generated) < budget:
            generated.append(next_id)
            if next_id == eos_id or len(generated) >= budget:
                break
            token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            out = model(token, past_key_values=cache.cache, use_cache=True)
            next_id = int(out.logits[0, -1].argmax().item())
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    return {
        "token_ids": generated, "prefill_time_s": prefill_end - start,
        "total_time_s": end - start,
        "decode_time_s": end - prefill_end,
        "decode_tokens_per_second": max(len(generated) - 1, 0) / max(end - prefill_end, 1e-12),
        "final_kv_length": cache.get_seq_length(),
    }


def invoke(method: str, loaded: Any, manager: LayerManager, ids: list[int],
           budget: int, capacity: int, device: str) -> dict:
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    if method == "vanilla":
        return direct_vanilla(loaded.model, prompt, loaded.tokenizer.eos_token_id,
                              budget, capacity, device)
    if method == "pld_k4":
        _, metrics = prompt_lookup_generate(
            loaded.model, loaded.tokenizer, prompt, k=4, ngram_size=3,
            max_new_tokens=budget, temperature=0.0, device=device,
            kv_cache_backend="static", candidate_source="prompt+generated",
            kv_cache_max_capacity=capacity,
        )
    else:
        _, metrics = self_speculative_generate(
            loaded.model, loaded.tokenizer, manager, SKIP, prompt, k=2,
            max_new_tokens=budget, temperature=0.0, device=device,
            kv_cache_backend="static", kv_cache_max_capacity=capacity,
            draft_mode="layer_skip", config_name="cka_90",
        )
    return {
        "token_ids": list(metrics.generated_token_ids),
        "prefill_time_s": float(metrics.prefill_time_s),
        "total_time_s": float(metrics.total_time_s),
        "decode_time_s": max(float(metrics.total_time_s) - float(metrics.prefill_time_s), 0.0),
        "decode_tokens_per_second": max(len(metrics.generated_token_ids) - 1, 0)
        / max(float(metrics.total_time_s) - float(metrics.prefill_time_s), 1e-12),
        "acceptance_rate": float(metrics.acceptance_rate),
        "verification_cycles": int(metrics.num_verification_cycles),
        "draft_time_s": float(metrics.draft_time_s),
        "verify_time_s": float(metrics.verify_time_s),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", choices=("hf_bnb_nf4", "hqq_gemlite_w4a16"), required=True)
    ap.add_argument("--prompt-id", default="qasper_119_ctx384")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--capacity", type=int, default=436)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--warmup-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    rows = [json.loads(line) for line in INPUTS.read_text().splitlines() if line.strip()]
    row = next(item for item in rows if item["prompt_id"] == args.prompt_id)
    ids = list(row["input_ids"])
    if args.warmup_tokens <= 0:
        raise ValueError("warmup token count must be positive")
    if len(ids) + max(args.max_new_tokens, args.warmup_tokens) + 4 > args.capacity:
        raise ValueError("KV capacity must cover prompt, generation budget, and safety margin")
    loaded = load_backend(args.backend, MODEL, REVISION, args.device, "sdpa")
    if len(ModelAdapter(loaded.model).get_layers()) != 36:
        raise RuntimeError("unexpected Qwen layer count; layer-skip compatibility gate failed")
    adapter = ModelAdapter(loaded.model)
    manager = LayerManager(adapter)
    prompt_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    rng = random.Random(args.seed)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for repetition in range(args.repeats):
        # Warm each path once, outside the measured trial, to trigger lazy kernels/JIT.
        for method in METHODS:
            warm = invoke(method, loaded, manager, ids, args.warmup_tokens,
                          args.capacity, args.device)
            if not warm["token_ids"]:
                raise RuntimeError(f"warmup emitted no tokens: {method}")
        torch.cuda.synchronize(args.device)
        order = list(METHODS)
        rng.shuffle(order)
        measured: dict[str, dict] = {}
        for method in order:
            torch.cuda.reset_peak_memory_stats(args.device)
            torch.cuda.synchronize(args.device)
            result = invoke(method, loaded, manager, ids, args.max_new_tokens,
                            args.capacity, args.device)
            torch.cuda.synchronize(args.device)
            result["peak_allocated_mb"] = torch.cuda.max_memory_allocated(args.device) / 2**20
            result["peak_reserved_mb"] = torch.cuda.max_memory_reserved(args.device) / 2**20
            measured[method] = result
        vanilla_ids = measured["vanilla"]["token_ids"]
        for method in METHODS:
            result = measured[method]
            tokens = result["token_ids"]
            mismatch = next((i for i, (a, b) in enumerate(zip(tokens, vanilla_ids)) if a != b), None)
            if mismatch is None and len(tokens) != len(vanilla_ids):
                mismatch = min(len(tokens), len(vanilla_ids))
            rows_out.append({
                "backend": args.backend, "backend_metadata": loaded.metadata,
                "model": MODEL, "model_revision": REVISION,
                "prompt_id": args.prompt_id, "prompt_token_ids_sha256": prompt_hash,
                "prompt_tokens": len(ids), "max_new_tokens": args.max_new_tokens,
                "kv_capacity": args.capacity, "method": method,
                "repetition": repetition, "balanced_order_seed": args.seed,
                "execution_order": order, "token_ids": tokens,
                "warmup_tokens": args.warmup_tokens,
                "within_backend_exact_vs_vanilla": tokens == vanilla_ids,
                "first_divergence_vs_vanilla": mismatch,
                **{k: v for k, v in result.items() if k != "token_ids"},
            })
    runtime = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device),
        "compute_capability": list(torch.cuda.get_device_capability(args.device)),
        "execution_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                            capture_output=True, text=True, check=True).stdout.strip(),
        "measurement_scope": "single prompt/backend; full-budget warmup excluded; paired methods with randomized order; diagnostic pilot",
        "working_tree_dirty": subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                                             capture_output=True, text=True, check=True).stdout.strip() != "",
    }
    with out.open("a") as handle:
        for record in rows_out:
            record["runtime"] = runtime
            handle.write(json.dumps(record) + "\n")
    print(json.dumps({"backend": args.backend, "prompt_id": args.prompt_id,
                      "repetitions": args.repeats, "records_written": len(rows_out),
                      "output": str(out), "exact": {r["method"]: r["within_backend_exact_vs_vanilla"]
                                                     for r in rows_out[-len(METHODS):]}}, indent=2))


if __name__ == "__main__":
    main()
