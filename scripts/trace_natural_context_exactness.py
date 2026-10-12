#!/usr/bin/env python3
"""Trace NF4 layer-skip mismatches on prepared natural long-context inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from zassd.cache.kv_cache import TargetKVCache  # noqa: E402
from zassd.decoding.speculative import diagnostic_cache_digest, self_speculative_generate  # noqa: E402
from zassd.models.layer_manager import LayerManager  # noqa: E402
from zassd.models.model_adapter import ModelAdapter  # noqa: E402


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def vanilla_tokens(model, tokenizer, prompt_ids: torch.Tensor, max_new_tokens: int,
                   cache_capacity: int) -> tuple[list[int], list[dict[str, Any]]]:
    cache = TargetKVCache(backend="static", max_capacity=cache_capacity)
    output: list[int] = []
    states: list[dict[str, Any]] = []
    with torch.no_grad():
        result = model(prompt_ids, past_key_values=cache.cache, use_cache=True)
        for index in range(max_new_tokens):
            logits = result.logits[0, -1, :].float()
            top = torch.topk(logits, k=2)
            token = int(logits.argmax().item())
            output.append(token)
            states.append({
                "position": index, "token_id": token,
                "top1_margin": float((top.values[0] - top.values[1]).item()),
                "cache_length": cache.get_seq_length(),
                "cache_digest": diagnostic_cache_digest(cache),
            })
            if token == tokenizer.eos_token_id or index + 1 >= max_new_tokens:
                break
            result = model(torch.tensor([[token]], dtype=torch.long, device=prompt_ids.device),
                           past_key_values=cache.cache, use_cache=True)
    return output, states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=3072)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for the NF4 trace")
    records = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    records = [row for row in records if row["prompt_tokens"] == args.prompt_tokens]
    if not records:
        raise ValueError(f"No rows with prompt_tokens={args.prompt_tokens} in {args.inputs}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=quant,
        device_map=args.device, attn_implementation="sdpa").eval()
    manager = LayerManager(ModelAdapter(model))
    skip_indices = [4, 5, 6, 7]  # Same CKA90 action as the context pilot.
    rows: list[dict[str, Any]] = []
    for source in records:
        prompt = torch.tensor([source["input_ids"]], dtype=torch.long, device=args.device)
        cap = int(prompt.shape[-1]) + args.max_new_tokens + 8
        reference, reference_states = vanilla_tokens(model, tokenizer, prompt,
                                                     args.max_new_tokens, cap)
        methods: dict[str, Any] = {}
        for name, canonical in (("batched", False), ("sequential", True)):
            _, metrics = self_speculative_generate(
                model, tokenizer, manager, skip_indices, prompt, k=2,
                max_new_tokens=args.max_new_tokens, temperature=0.0,
                device=args.device, kv_cache_backend="static",
                kv_cache_max_capacity=cap, diagnostic_trace=True,
                diagnostic_hash_kv=True, canonical_verify=canonical,
            )
            generated = metrics.generated_token_ids
            mismatch = first_difference(reference, generated)
            cycle = None
            if mismatch is not None:
                cycle = next((item for item in metrics.per_iteration_stats
                              if item.get("output_start", -1) <= mismatch <
                              item.get("output_start", -1) + len(item.get("emitted_tokens", []))), None)
            methods[name] = {
                "generated_token_ids": generated,
                "first_divergence": mismatch,
                "sequence_exact_match": mismatch is None,
                "total_time_s": metrics.total_time_s,
                "prefill_time_s": metrics.prefill_time_s,
                "draft_time_s": metrics.draft_time_s,
                "verify_time_s": metrics.verify_time_s,
                "canonical_replay_time_s": metrics.canonical_replay_time_s,
                "first_divergent_cycle": cycle,
                "cycles": metrics.per_iteration_stats,
            }
        rows.append({
            "prompt_id": source["prompt_id"], "source_task": source["source_task"],
            "source_id": source["source_id"], "prompt_tokens": int(prompt.shape[-1]),
            "vanilla_token_ids": reference, "vanilla_steps": reference_states,
            "vanilla_repeat_token_ids": vanilla_tokens(model, tokenizer, prompt,
                                                       args.max_new_tokens, cap)[0],
            "methods": methods,
        })
        print(f"{source['prompt_id']}: batched={methods['batched']['first_divergence']} "
              f"sequential={methods['sequential']['first_divergence']}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    result = {
        "commit_sha": commit, "model": args.model, "model_revision": args.revision,
        "quantization": "NF4", "compute_dtype": "float16", "attention": "SDPA",
        "device": torch.cuda.get_device_name(args.device),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "method": "CKA90/K2; same static KV backend; batch and sequential target verification",
        "rows": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
