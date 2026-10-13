#!/usr/bin/env python3
"""Trace PLD KV/logit divergence and run same-prefix canonical-KV replay.

This is a diagnostic experiment. It stores per-cycle cache snapshots on CPU,
so use a small number of prompts and a modest output length.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from zassd.baselines.prompt_lookup import prompt_lookup_generate
from zassd.cache.kv_cache import TargetKVCache


def cache_error(left, right):
    first_layer = None
    max_abs = 0.0
    max_rel = 0.0
    length_mismatches = 0
    for index, (a, b) in enumerate(zip(left, right)):
        ak, av, alen = a
        bk, bv, blen = b
        if alen != blen:
            length_mismatches += 1
            if first_layer is None:
                first_layer = index
            continue
        if alen == 0:
            continue
        layer_abs = 0.0
        layer_rel = 0.0
        for x, y in ((ak, bk), (av, bv)):
            delta = (x.float() - y.float()).abs()
            layer_abs = max(layer_abs, float(delta.max()))
            denom = y.float().abs().clamp_min(1e-12)
            layer_rel = max(layer_rel, float((delta / denom).max()))
        max_abs = max(max_abs, layer_abs)
        max_rel = max(max_rel, layer_rel)
        if first_layer is None and layer_abs != 0.0:
            first_layer = index
    return first_layer, max_abs, max_rel, length_mismatches


def tensor_cache(cache):
    out = []
    for layer in cache.layers:
        length = int(layer.get_seq_length())
        if getattr(layer, "is_initialized", False):
            out.append((layer.keys[:, :, :length, :].detach().cpu().clone(),
                        layer.values[:, :, :length, :].detach().cpu().clone(), length))
        else:
            out.append((torch.empty(0), torch.empty(0), 0))
    return out


def margin(logits):
    top = torch.topk(logits.float().flatten(), 2).values
    return float(top[0] - top[1])


def canonical_replay(model, prefix_ids, prompt_len, backend, capacity):
    """Rebuild vanilla-style cache: one prompt prefill, then one token per call."""
    cache = TargetKVCache(backend=backend, max_capacity=capacity)
    with torch.no_grad():
        out = model(prefix_ids[:, :prompt_len], past_key_values=cache.cache, use_cache=True)
        for index in range(prompt_len, prefix_ids.shape[1]):
            token = prefix_ids[:, index:index + 1]
            out = model(token, past_key_values=cache.cache, use_cache=True)
    return cache, out


def replay_candidate(model, cache, cycle, device):
    """Replay identical candidate IDs using the recorded verification mode."""
    candidate_ids = cycle["candidate_tokens"]
    if cycle["verification_mode"] == "batched":
        candidate = torch.tensor([candidate_ids], device=device, dtype=torch.long)
        return model(candidate, past_key_values=cache.cache, use_cache=True).logits
    logits = []
    actual_k = max(0, len(candidate_ids) - 1)
    with torch.no_grad():
        for index, token_id in enumerate(candidate_ids):
            token = torch.tensor([[token_id]], device=device, dtype=torch.long)
            out = model(token, past_key_values=cache.cache, use_cache=True)
            logits.append(out.logits[:, -1:, :])
            if index < actual_k:
                predicted = int(out.logits[:, -1, :].argmax(dim=-1).item())
                if predicted != candidate_ids[index + 1]:
                    break
    return torch.cat(logits, dim=1)


def vanilla_trace(model, prompt_ids, max_new_tokens, backend, capacity, device, eos_id):
    cache = TargetKVCache(backend=backend, max_capacity=capacity)
    tokens, margins = [], []
    with torch.no_grad():
        out = model(prompt_ids, past_key_values=cache.cache, use_cache=True)
        for index in range(max_new_tokens):
            logits = out.logits[0, -1].float()
            token = int(logits.argmax())
            tokens.append(token)
            margins.append(margin(logits))
            if token == eos_id or index + 1 == max_new_tokens:
                break
            out = model(torch.tensor([[token]], device=device),
                        past_key_values=cache.cache, use_cache=True)
    return tokens, margins


def first_difference(left, right):
    for i, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return i
    return min(len(left), len(right)) if len(left) != len(right) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--precision", choices=["nf4", "fp16", "bf16", "fp32"], default="nf4")
    ap.add_argument("--attention-implementation", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--benchmark", default="experiments/natural_context_validation/independent_pld_short_context/session1_rtx4050/prepared_inputs.jsonl")
    ap.add_argument("--ids", nargs="*", help="Benchmark IDs; omit to use first --samples")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ngram-size", type=int, default=3)
    ap.add_argument("--candidate-source", choices=["history", "prompt", "prompt+generated"],
                    default="prompt+generated")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--backend", choices=["static", "dynamic"], default="static")
    ap.add_argument("--verification-mode", choices=["batched", "sequential"], default="batched")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default="experiments/numerical_causality/trace.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this NF4 trace")

    revision = args.revision
    if revision is None and args.model == "Qwen/Qwen2.5-3B-Instruct":
        revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    load_args = {"device_map": args.device, "attn_implementation": args.attention_implementation}
    if revision is not None:
        load_args["revision"] = revision
    if args.precision == "nf4":
        load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16)
    else:
        load_args["torch_dtype"] = {
            "fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32,
        }[args.precision]
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_args)
    model.eval()
    records = []
    with open(args.benchmark, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            item = json.loads(line)
            record_id = item.get("id", item.get("prompt_id"))
            if args.ids and str(record_id) not in args.ids:
                continue
            records.append(item)
            if len(records) >= args.samples:
                break

    summaries = []
    for prompt_index, item in enumerate(records):
        if "input_ids" in item:
            prompt_ids = torch.tensor([item["input_ids"]], device=args.device, dtype=torch.long)
        else:
            prompt_ids = tokenizer(item["prompt"], return_tensors="pt")["input_ids"].to(args.device)
        prompt_len = int(prompt_ids.shape[1])
        # Match run_independent_pld_short_context.py so static-cache capacity
        # cannot become an unnoticed backend/kernel confound.
        cache_capacity = prompt_len + args.max_new_tokens + args.k + 16
        cycles: list[dict] = []
        _, metrics = prompt_lookup_generate(
            model, tokenizer, prompt_ids, k=args.k, ngram_size=args.ngram_size,
            max_new_tokens=args.max_new_tokens, device=args.device,
            kv_cache_backend=args.backend, candidate_source=args.candidate_source,
            sequential_verify=args.verification_mode == "sequential",
            kv_cache_max_capacity=cache_capacity,
            numerical_trace=cycles,
        )
        vanilla_ids, vanilla_margins = vanilla_trace(
            model, prompt_ids, args.max_new_tokens, args.backend,
            cache_capacity, args.device, tokenizer.eos_token_id)
        token_divergence = first_difference(vanilla_ids, metrics.generated_token_ids)
        trace_rows = []
        first_kv_cycle = first_logit_cycle = None
        for cycle in cycles:
            prefix = cycle["prefix_tokens"]
            prefix_tensor = torch.tensor([prefix], device=args.device, dtype=torch.long)
            ref_cache, _ = canonical_replay(
                model, prefix_tensor, prompt_len, args.backend, cache_capacity)
            with torch.no_grad():
                kv_layer, kv_abs, kv_rel, len_mismatch = cache_error(
                    cycle["cache_before"], tensor_cache(ref_cache.cache))
                kv_diff = kv_layer is not None or len_mismatch > 0
                if kv_diff and first_kv_cycle is None:
                    first_kv_cycle = cycle["cycle_index"]
                replay_logits = replay_candidate(model, ref_cache, cycle, args.device)
                ref_argmax = replay_logits[0].argmax(dim=-1).detach().cpu().tolist()
                ref_logits = replay_logits.detach().float().cpu()
                observed = cycle["verify_logits"]
                logit_delta = (observed[..., :ref_logits.shape[-2], :] - ref_logits).abs()
                logit_abs = float(logit_delta.max()) if logit_delta.numel() else 0.0
                logits_diff = logit_abs != 0.0
                if logits_diff and first_logit_cycle is None:
                    first_logit_cycle = cycle["cycle_index"]
                ref_preds = ref_argmax
                obs_preds = cycle["verify_argmax"]
                flip_at = next((i for i, (a, b) in enumerate(zip(ref_preds, obs_preds)) if a != b), None)
                cycle_start = len(prefix) - prompt_len
                prefix_matches_vanilla = prefix == (prompt_ids[0].tolist() +
                    vanilla_ids[:max(0, cycle_start)])
                expected_next = [vanilla_ids[cycle_start + i + 1]
                                 if cycle_start + i + 1 < len(vanilla_ids) else None
                                 for i in range(len(ref_preds))]
                canonical_matches_vanilla = [
                    expected is not None and pred == expected
                    for pred, expected in zip(ref_preds, expected_next)]
                active_matches_vanilla = [
                    expected is not None and pred == expected
                    for pred, expected in zip(obs_preds, expected_next)]
                trace_rows.append({
                    "cycle_index": cycle["cycle_index"],
                    "verification_mode": cycle["verification_mode"],
                    "prefix_length": len(prefix), "draft_k": len(cycle["candidate_tokens"]) - 1,
                    "accepted_count": cycle["accepted_count"],
                    "candidate_tokens": cycle["candidate_tokens"],
                    "emitted_tokens": cycle["emitted_tokens"],
                    "rejected_at": cycle["rejected_at"],
                    "cache_length_before": cycle["cache_before"][0][2],
                    "cache_length_after": cycle["cache_after"][0][2],
                    "first_kv_different_layer": kv_layer,
                    "kv_max_abs_error": kv_abs, "kv_relative_error": kv_rel,
                    "cache_length_mismatch_count": len_mismatch,
                    "logits_max_abs_error": logit_abs,
                    "vanilla_top1_margin": margin(ref_logits[0, 0]),
                    "argmax_equal": flip_at is None,
                    "kv_substitution_argmax_equal": flip_at is None,
                    "kv_intervention": {
                        "same_token_prefix": prefix_matches_vanilla,
                        "active_kv_argmax": obs_preds,
                        "canonical_kv_argmax": ref_preds,
                        "active_matches_vanilla": active_matches_vanilla,
                        "canonical_matches_vanilla": canonical_matches_vanilla,
                        "argmax_restored": [(not active_ok) and canonical_ok
                                            for active_ok, canonical_ok in zip(
                                                active_matches_vanilla,
                                                canonical_matches_vanilla)],
                    },
                    "first_argmax_difference_in_cycle": flip_at,
                    "prefix_matches_vanilla": prefix_matches_vanilla,
                    "emitted_matches_vanilla": (cycle["emitted_tokens"] == vanilla_ids[
                        len(prefix) - prompt_len:len(prefix) - prompt_len + len(cycle["emitted_tokens"])]),
                    "canonical_replay_token_ids": ref_preds,
                    "observed_batch_token_ids": obs_preds,
                    "vanilla_next_token_ids": expected_next,
                })
        divergence_cycle = None
        divergence_row = None
        if token_divergence is not None:
            for index, source_cycle in enumerate(cycles):
                cycle_start = len(source_cycle["prefix_tokens"]) - prompt_len
                row_index = token_divergence - cycle_start - 1
                if 0 <= row_index < len(source_cycle["candidate_tokens"]):
                    divergence_cycle = source_cycle["cycle_index"]
                    divergence_row = row_index
                    break
        summaries.append({
            "prompt_id": item.get("id", item.get("prompt_id", prompt_index)),
            "prompt_length": prompt_len,
            "token_ids": metrics.generated_token_ids,
            "vanilla_token_ids": vanilla_ids,
            "first_kv_divergence_cycle": first_kv_cycle,
            "first_logit_divergence_cycle": first_logit_cycle,
            "first_token_divergence_index": token_divergence,
            "first_token_divergence_cycle": divergence_cycle,
            "first_token_divergence_logit_row": divergence_row,
            "argmax_agreement_at_first_divergence": (
                trace_rows[divergence_cycle]["argmax_equal"]
                if divergence_cycle is not None and divergence_cycle < len(trace_rows) else None),
            "vanilla_top1_margin_at_divergence": (
                vanilla_margins[token_divergence]
                if token_divergence is not None and token_divergence < len(vanilla_margins) else None),
            "cycles": trace_rows,
        })
        print(f"{item.get('id', item.get('prompt_id', prompt_index))}: kv={first_kv_cycle}, logits={first_logit_cycle}, token={token_divergence}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "model": args.model, "model_revision": revision,
        "precision": args.precision, "backend": args.backend,
        "attention_implementation": args.attention_implementation,
        "verification_mode": args.verification_mode,
        "k": args.k, "benchmark": args.benchmark, "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(args.device),
        }, "prompts": summaries,
    }, indent=2), encoding="utf-8")
    output.parent.joinpath("config.json").write_text(json.dumps({
        "model": args.model, "model_revision": revision,
        "benchmark": args.benchmark, "ids": args.ids,
        "samples": args.samples, "k": args.k, "ngram_size": args.ngram_size,
        "candidate_source": args.candidate_source,
        "max_new_tokens": args.max_new_tokens, "backend": args.backend,
        "precision": args.precision,
        "attention_implementation": args.attention_implementation,
        "verification_mode": args.verification_mode,
        "kv_cache_max_capacity": "prompt_length + max_new_tokens + k + 16",
        "device": args.device,
    }, indent=2), encoding="utf-8")
    output.parent.joinpath("environment.json").write_text(json.dumps({
        "python": platform.python_version(), "torch": torch.__version__,
        "transformers": transformers.__version__, "platform": platform.platform(),
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(args.device),
        "attention_implementation": getattr(model.config, "_attn_implementation", None),
        "cache_backend": args.backend,
    }, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
