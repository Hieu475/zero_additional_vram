#!/usr/bin/env python3
"""Capture a single, labeled decode with Nsight Systems' CUDA profiler range.

Example:
  nsys profile --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi \
    --capture-range-end=stop -o experiments/layer_skip_profile/cka90_k2 \
    python scripts/profile_layer_skip_nsight.py --output-json \
    experiments/layer_skip_profile/cka90_k2.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter


CONFIGS = {
    "cka_60": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 21],
    "cka_67": [3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16],
    "cka_90": [4, 5, 6, 7],
}


def vanilla_decode(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> dict:
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    cache = TargetKVCache(backend="static", max_capacity=2048)
    start = time.perf_counter()
    torch.cuda.nvtx.range_push("vanilla_prefill_and_decode")
    with torch.no_grad():
        out = model(ids, past_key_values=cache.cache, use_cache=True)
    token = out.logits[:, -1:, :].argmax(dim=-1)
    output = [int(token.item())]
    while len(output) < max_new_tokens and output[-1] != tokenizer.eos_token_id:
        with torch.no_grad():
            out = model(token, past_key_values=cache.cache, use_cache=True)
        token = out.logits[:, -1:, :].argmax(dim=-1)
        output.append(int(token.item()))
    torch.cuda.synchronize(device)
    torch.cuda.nvtx.range_pop()
    elapsed = time.perf_counter() - start
    return {"output_tokens": len(output), "decode_ms": elapsed * 1000,
            "decode_tps": max(0, len(output) - 1) / max(elapsed, 1e-9),
            "token_ids": output}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("vanilla", "layer_skip"), default="layer_skip")
    parser.add_argument("--config", choices=tuple(CONFIGS), default="cka_90")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--prompt-id", default="1")
    parser.add_argument("--sync-variant", choices=("reference", "reduced_sync_conservative"),
                        default="reference")
    parser.add_argument("--collect-acceptance-cycles", action="store_true",
                        help="Collect entropy/survival metadata; adds synchronization overhead")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nsight Systems capture")

    records = [json.loads(line) for line in (ROOT / "data/benchmarks/prompts.jsonl").read_text().splitlines() if line.strip()]
    prompt_row = next((r for r in records if str(r["id"]) == args.prompt_id), None)
    if prompt_row is None:
        raise ValueError(f"Prompt id {args.prompt_id} not found")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=quant,
        device_map=args.device, attn_implementation="sdpa").eval()
    manager = LayerManager(ModelAdapter(model))
    prompt = prompt_row["prompt"]

    # Warm the selected execution path before CUDA profiler capture.
    if args.mode == "vanilla":
        vanilla_decode(model, tokenizer, prompt, min(4, args.max_new_tokens), args.device)
    else:
        self_speculative_generate(
            model, tokenizer, manager, CONFIGS[args.config], prompt, k=args.k,
            max_new_tokens=min(8, args.max_new_tokens), temperature=0.0,
            device=args.device, kv_cache_backend="static", config_name=args.config,
            reduced_sync=args.sync_variant == "reduced_sync_conservative")
    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)

    torch.cuda.profiler.start()
    if args.mode == "vanilla":
        measured = vanilla_decode(model, tokenizer, prompt, args.max_new_tokens, args.device)
    else:
        _, metrics = self_speculative_generate(
            model, tokenizer, manager, CONFIGS[args.config], prompt, k=args.k,
            max_new_tokens=args.max_new_tokens, temperature=0.0,
            device=args.device, kv_cache_backend="static", config_name=args.config,
            collect_acceptance_cycles=args.collect_acceptance_cycles, profile_nvtx=True,
            reduced_sync=args.sync_variant == "reduced_sync_conservative")
        measured = asdict(metrics)
    torch.cuda.synchronize(args.device)
    torch.cuda.profiler.stop()

    output = Path(args.output_json)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode, "sync_variant": args.sync_variant,
        "collect_acceptance_cycles": args.collect_acceptance_cycles,
        "config": args.config, "k": args.k,
        "prompt_id": args.prompt_id, "prompt_tokens": len(tokenizer(prompt)["input_ids"]),
        "model": args.model, "model_revision": args.revision,
        "gpu_name": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "commit_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                      capture_output=True, text=True, check=True).stdout.strip(),
        "capture_range": "CUDA profiler API; warmup excluded",
        "metrics": measured,
    }
    output.write_text(json.dumps(payload, indent=2))
    print(output)


if __name__ == "__main__":
    main()
