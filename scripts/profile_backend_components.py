#!/usr/bin/env python3
"""Capture an untraced BNB/NF4 method with CUDA/NVTX ranges for Nsight."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zassd.baselines.prompt_lookup import prompt_lookup_generate  # noqa: E402
from zassd.cache.kv_cache import TargetKVCache  # noqa: E402
from zassd.decoding.speculative import self_speculative_generate  # noqa: E402
from zassd.models.layer_manager import LayerManager  # noqa: E402
from zassd.models.model_adapter import ModelAdapter  # noqa: E402
from zassd.backends.base import load_backend  # noqa: E402

MODEL = "Qwen/Qwen2.5-3B-Instruct"
REVISION = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
INPUTS = ROOT / "experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl"
DEFAULT_OUT = ROOT / "experiments/backend_validation/kernel_profile"
CKA90_SKIP = [4, 5, 6, 7]


def direct_vanilla(model: Any, prompt_ids: torch.Tensor, eos: int | None,
                   budget: int, capacity: int, device: str) -> dict[str, Any]:
    cache = TargetKVCache(backend="static", max_capacity=capacity)
    out_ids: list[int] = []
    torch.cuda.nvtx.range_push("vanilla_prefill")
    with torch.inference_mode():
        out = model(prompt_ids, past_key_values=cache.cache, use_cache=True)
    torch.cuda.nvtx.range_pop()
    next_id = int(out.logits[0, -1].argmax().item())
    torch.cuda.nvtx.range_push("vanilla_decode")
    while len(out_ids) < budget:
        out_ids.append(next_id)
        if next_id == eos or len(out_ids) >= budget:
            break
        token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        with torch.inference_mode():
            out = model(token, past_key_values=cache.cache, use_cache=True)
        next_id = int(out.logits[0, -1].argmax().item())
    torch.cuda.nvtx.range_pop()
    return {"token_ids": out_ids, "final_kv_length": cache.get_seq_length()}


def run_method(name: str, model: Any, tokenizer: Any, ids: list[int], budget: int,
               capacity: int, device: str) -> dict[str, Any]:
    prompt = torch.tensor([ids], dtype=torch.long, device=device)
    if name == "vanilla":
        return direct_vanilla(model, prompt, tokenizer.eos_token_id, budget, capacity, device)
    if name == "pld_k4":
        torch.cuda.nvtx.range_push("pld_k4_total")
        try:
            _, metrics = prompt_lookup_generate(
                model, tokenizer, prompt, k=4, ngram_size=3,
                max_new_tokens=budget, temperature=0.0, device=device,
                kv_cache_backend="static", candidate_source="prompt+generated",
                kv_cache_max_capacity=capacity)
        finally:
            torch.cuda.nvtx.range_pop()
        return {"token_ids": list(metrics.generated_token_ids), "metrics": asdict(metrics)}
    manager = LayerManager(ModelAdapter(model))
    torch.cuda.nvtx.range_push("cka90_k2_total")
    try:
        _, metrics = self_speculative_generate(
            model, tokenizer, manager, CKA90_SKIP, prompt, k=2,
            max_new_tokens=budget, temperature=0.0, device=device,
            kv_cache_backend="static", kv_cache_max_capacity=capacity,
            draft_mode="layer_skip", config_name="cka_90", profile_nvtx=True)
    finally:
        torch.cuda.nvtx.range_pop()
    return {"token_ids": list(metrics.generated_token_ids), "metrics": asdict(metrics)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", choices=("vanilla", "pld_k4", "cka90_k2"), required=True)
    ap.add_argument("--weight-backend", choices=("hf_bnb_nf4", "hqq_gemlite_w4a16"),
                    default="hf_bnb_nf4")
    ap.add_argument("--prompt-id", default="qasper_119_ctx384")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--warmup-tokens", type=int, default=4)
    ap.add_argument("--capacity", type=int, default=436)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    rows = [json.loads(x) for x in INPUTS.read_text().splitlines() if x.strip()]
    row = next(r for r in rows if r["prompt_id"] == args.prompt_id)
    ids = list(row["input_ids"])
    if len(ids) + args.max_new_tokens + 4 > args.capacity:
        raise ValueError("KV capacity does not cover prompt and output budget")
    loaded = load_backend(args.weight_backend, MODEL, REVISION,
                          device=args.device, attention="sdpa")
    tokenizer, model = loaded.tokenizer, loaded.model
    # Identical method warmup before nsys starts the CUDA capture range.
    warm = run_method(args.method, model, tokenizer, ids, args.warmup_tokens,
                      args.capacity, args.device)
    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push("ncu_measurement")
    started = time.perf_counter()
    measured = run_method(args.method, model, tokenizer, ids, args.max_new_tokens,
                          args.capacity, args.device)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(args.device)
    elapsed = time.perf_counter() - started
    torch.cuda.profiler.stop()
    payload = {
        "method": args.method, "prompt_id": args.prompt_id,
        "prompt_token_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "prompt_tokens": len(ids), "max_new_tokens": args.max_new_tokens,
        "kv_capacity": args.capacity, "weight_backend": args.weight_backend,
        "backend_metadata": loaded.metadata,
        "attention": "SDPA", "model_revision": REVISION,
        "gpu": torch.cuda.get_device_name(args.device), "compute_capability": list(torch.cuda.get_device_capability(args.device)),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "backend_versions": {
            name: importlib.metadata.version(name)
            for name in (("bitsandbytes",) if args.weight_backend == "hf_bnb_nf4" else ("hqq", "gemlite"))
        },
        "execution_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                           check=True, capture_output=True, text=True).stdout.strip(),
        "capture_range": "cudaProfilerApi; warmup excluded; NVTX ncu_measurement range; Nsight-instrumented diagnostics, not production timing",
        "warmup_token_count": len(warm["token_ids"]),
        "measured_token_ids": measured["token_ids"],
        "measured_e2e_seconds_under_profiler": elapsed,
        "tokens_per_second_under_profiler": len(measured["token_ids"]) / max(elapsed, 1e-12),
        "peak_allocated_mb": torch.cuda.max_memory_allocated(args.device) / 2**20,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(args.device) / 2**20,
        "method_metrics": measured.get("metrics"),
    }
    output = args.output_json if args.output_json.is_absolute() else ROOT / args.output_json
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
