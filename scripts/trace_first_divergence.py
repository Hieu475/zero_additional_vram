#!/usr/bin/env python3
"""Locate the first greedy token divergence and the responsible decode cycle.

This is a diagnostic run, not a throughput benchmark. It records token IDs,
target argmax/margins, and canonical KV lengths without retokenizing output text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.speculative import diagnostic_cache_digest, self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.model_adapter import ModelAdapter
from zassd.profiling.action_cost_model import MeasuredActionCostModel
from zassd.routing.hybrid_router import HybridDraftRouter


def vanilla_trace(model, tokenizer, prompt: str, max_new_tokens: int, device: str,
                  hash_kv: bool = False) -> dict:
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    cache = TargetKVCache(backend="static")
    token_ids: list[int] = []
    steps: list[dict] = []
    with torch.no_grad():
        output = model(prompt_ids, past_key_values=cache.cache, use_cache=True)
        for index in range(max_new_tokens):
            logits = output.logits[0, -1, :].float()
            top = torch.topk(logits, 2)
            # topk has a different tie-breaking order from argmax. The
            # production greedy baseline uses argmax, including exact ties.
            token = int(logits.argmax(dim=-1).item())
            token_ids.append(token)
            steps.append({
                "index": index,
                "token_id": token,
                "margin": float((top.values[0] - top.values[1]).item()),
                "cache_len_when_selected": cache.get_seq_length(),
                "cache_digest_when_selected": diagnostic_cache_digest(cache) if hash_kv else None,
            })
            if token == tokenizer.eos_token_id or index + 1 == max_new_tokens:
                break
            output = model(torch.tensor([[token]], device=device),
                           past_key_values=cache.cache, use_cache=True)
    return {"token_ids": token_ids, "steps": steps, "prompt_len": prompt_ids.shape[1]}


def first_difference(left: list[int], right: list[int]) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--benchmark", default="data/benchmarks/gsm8k_eval.jsonl")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--ids", nargs="*", default=None,
                        help="Trace only these benchmark record IDs")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--method", choices=["layer_skip", "prompt_lookup", "routed"], default="routed")
    parser.add_argument("--hash-kv", action="store_true",
                        help="Hash active KV prefixes on every cycle (slow)")
    parser.add_argument("--canonical-kv-replay", action="store_true",
                        help="Rebuild KV token-by-token after every speculative cycle")
    parser.add_argument("--canonical-verify", action="store_true",
                        help="Verify draft positions sequentially using vanilla target forwards")
    parser.add_argument("--margin-fallback-threshold", type=float, default=None,
                        help="Rebuild canonical KV and verify sequentially when batch top-1 margin is low")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="experiments/first_divergence_trace.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the NF4 divergence trace requires a working GPU driver")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if "bnb-4bit" in args.model.lower() or "4bit" in args.model.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model, device_map=args.device)
    else:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=quant, device_map=args.device)
    model.eval()
    adapter = ModelAdapter(model)
    manager = LayerManager(adapter)
    skip_count = min(6, adapter.num_layers)
    skip_start = (adapter.num_layers - skip_count) // 2
    skip_indices = list(range(skip_start, skip_start + skip_count))
    model_key = "llama32_3b" if "llama" in args.model.lower() else "qwen25_3b"
    records = []
    with open(args.benchmark, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                item = json.loads(line)
                if args.ids is not None and str(item.get("id")) not in args.ids:
                    continue
                records.append(item)
            if len(records) >= args.samples:
                break

    results = []
    for index, item in enumerate(records):
        reference = vanilla_trace(model, tokenizer, item["prompt"],
                                  args.max_new_tokens, args.device, args.hash_kv)
        reference_repeat = vanilla_trace(model, tokenizer, item["prompt"],
                                         args.max_new_tokens, args.device)
        router = (HybridDraftRouter(cost_model=MeasuredActionCostModel.for_model(model_key),
                                    pld_k=4) if args.method == "routed" else None)
        _, metrics = self_speculative_generate(
            model, tokenizer, manager, skip_indices, item["prompt"],
            k=args.k, draft_mode=args.method, router=router,
            max_new_tokens=args.max_new_tokens, device=args.device,
            diagnostic_trace=True,
            diagnostic_hash_kv=args.hash_kv,
            canonical_kv_replay=args.canonical_kv_replay,
            canonical_verify=args.canonical_verify,
            margin_fallback_threshold=args.margin_fallback_threshold,
        )
        output_ids = metrics.generated_token_ids
        position = first_difference(reference["token_ids"], output_ids)
        cycle = None
        if position is not None:
            for candidate in metrics.per_iteration_stats:
                start = candidate.get("output_start", -1)
                if start <= position < start + len(candidate.get("emitted_tokens", [])):
                    cycle = candidate
                    break
            if cycle is None:
                cycle = next((c for c in metrics.per_iteration_stats
                              if c.get("output_start", -1) + len(c.get("emitted_tokens", [])) == position), None)
        cache_differences = []
        if args.hash_kv:
            for candidate in metrics.per_iteration_stats:
                start = candidate.get("output_start", -1)
                if 0 <= start < len(reference["steps"]):
                    same = (candidate.get("cache_digest_before") ==
                            reference["steps"][start]["cache_digest_when_selected"])
                    if not same:
                        cache_differences.append({
                            "output_start": start,
                            "source": candidate.get("router_source", candidate.get("source")),
                            "cache_len_before_verify": candidate.get("cache_len_before_verify"),
                            "reference_cache_len": reference["steps"][start]["cache_len_when_selected"],
                        })
        results.append({
            "id": item.get("id", index),
            "first_divergence": position,
            "vanilla_repeat_first_divergence": first_difference(
                reference["token_ids"], reference_repeat["token_ids"]),
            "vanilla_token_ids": reference["token_ids"],
            "spec_token_ids": output_ids,
            "canonical_replay_time_s": metrics.canonical_replay_time_s,
            "margin_fallback_cycles": metrics.margin_fallback_cycles,
            "total_time_s": metrics.total_time_s,
            "spec_prefill": metrics.diagnostic_prefill,
            "vanilla_step": (reference["steps"][position]
                             if position is not None and position < len(reference["steps"]) else None),
            "spec_cycle": cycle,
            "cycle_trace": metrics.per_iteration_stats,
            "first_cache_difference": cache_differences[0] if cache_differences else None,
            "expected_cache_len_before_cycle": (
                reference["prompt_len"] + cycle["output_start"]
                if cycle is not None else None),
        })
        print(f"{item.get('id', index)}: first divergence = {position}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "model": args.model, "method": args.method, "k": args.k,
        "canonical_kv_replay": args.canonical_kv_replay,
        "canonical_verify": args.canonical_verify,
        "margin_fallback_threshold": args.margin_fallback_threshold,
        "benchmark": args.benchmark, "samples": results,
    }, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
