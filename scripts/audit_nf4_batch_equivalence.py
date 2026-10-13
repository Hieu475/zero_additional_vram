#!/usr/bin/env python3
"""Compare same-prefix single-token and batched NF4 forward passes.

For each prompt, candidates are taken from the target's own greedy continuation
so this isolates numerical/cache-path differences from draft quality. The
report compares per-position logits and per-layer K/V values for K=1,2,4.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from zassd.cache.kv_cache import TargetKVCache


def forward(model, ids: torch.Tensor, cache: TargetKVCache):
    with torch.no_grad():
        return model(ids, past_key_values=cache.cache, use_cache=True)


def greedy(logits: torch.Tensor) -> int:
    return int(logits[0, -1].float().argmax().item())


def active_kv(cache: TargetKVCache, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer = cache.cache.layers[layer_index]
    return layer.keys, layer.values


def error_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    delta = (a.float() - b.float()).abs()
    denom = b.float().abs().square().mean().sqrt().clamp_min(1e-12)
    return {
        "max_abs": float(delta.max().item()),
        "rms_abs": float(delta.square().mean().sqrt().item()),
        "relative_rms": float((delta.square().mean().sqrt() / denom).item()),
        "bitwise_equal": bool(torch.equal(a, b)),
    }


def audit_prefix(model, prompt_ids: torch.Tensor, k_values: list[int], backend: str) -> dict:
    # Generate a same-model greedy continuation to use as a perfect draft.
    gold_cache = TargetKVCache(backend=backend, max_capacity=max(2048, prompt_ids.shape[1] + max(k_values) + 8))
    gold_out = forward(model, prompt_ids, gold_cache)
    curr = greedy(gold_out.logits[:, -1:, :])
    draft_tokens: list[int] = []
    cursor = curr
    for _ in range(max(k_values)):
        out = forward(model, torch.tensor([[cursor]], device=prompt_ids.device), gold_cache)
        cursor = greedy(out.logits[:, -1:, :])
        draft_tokens.append(cursor)

    reports = {}
    for k in k_values:
        # Independent caches ensure the tested paths start from the same prefix.
        capacity = max(2048, prompt_ids.shape[1] + k + 8)
        batched_cache = TargetKVCache(backend=backend, max_capacity=capacity)
        sequential_cache = TargetKVCache(backend=backend, max_capacity=capacity)
        batch_prefill = forward(model, prompt_ids, batched_cache)
        seq_prefill = forward(model, prompt_ids, sequential_cache)
        prefill_cache_equal = all(
            torch.equal(
                active_kv(batched_cache, i)[j][:, :, :prompt_ids.shape[1], :],
                active_kv(sequential_cache, i)[j][:, :, :prompt_ids.shape[1], :],
            )
            for i in range(len(batched_cache.cache.layers)) for j in (0, 1)
        )
        batch_logits = forward(model, torch.tensor([[curr] + draft_tokens[:k],], device=prompt_ids.device), batched_cache).logits[0]

        seq_logits = []
        input_tokens = [curr] + draft_tokens[:k]
        for token in input_tokens:
            out = forward(model, torch.tensor([[token]], device=prompt_ids.device), sequential_cache)
            seq_logits.append(out.logits[0, -1])
        seq_logits_tensor = torch.stack(seq_logits)

        logit_rows = []
        for position in range(k + 1):
            a = batch_logits[position].float()
            b = seq_logits_tensor[position].float()
            top = torch.topk(b, 2).values
            logit_rows.append({
                "position": position,
                "greedy_batch": int(a.argmax().item()),
                "greedy_sequential": int(b.argmax().item()),
                "reference_margin": float((top[0] - top[1]).item()),
                **error_stats(a, b),
            })

        layer_rows = []
        for layer_index in range(len(batched_cache.cache.layers)):
            bk, bv = active_kv(batched_cache, layer_index)
            sk, sv = active_kv(sequential_cache, layer_index)
            lo, hi = prompt_ids.shape[1], prompt_ids.shape[1] + k + 1
            layer_rows.append({
                "layer": layer_index,
                "key": error_stats(bk[:, :, lo:hi, :], sk[:, :, lo:hi, :]),
                "value": error_stats(bv[:, :, lo:hi, :], sv[:, :, lo:hi, :]),
            })
        first_kv_diff = next((x["layer"] for x in layer_rows
                              if not x["key"]["bitwise_equal"] or not x["value"]["bitwise_equal"]), None)
        reports[str(k)] = {
            "candidate_token_ids": draft_tokens[:k],
            "prefill_logits_bitwise_equal": bool(torch.equal(batch_prefill.logits, seq_prefill.logits)),
            "prefill_cache_bitwise_equal": prefill_cache_equal,
            "logits_by_position": logit_rows,
            "first_kv_differing_layer": first_kv_diff,
            "layer_kv": layer_rows,
        }
    return {"prompt_tokens": int(prompt_ids.shape[1]), "initial_token": curr, "horizons": reports}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--revision", default="aa8e72537993ba99e69dfaafa59ed015b17504d1")
    ap.add_argument("--benchmark", default="data/benchmarks/cnndm_eval.jsonl")
    ap.add_argument("--prepared-inputs", default=None,
                    help="Use exact token-ID prompts from a prepared JSONL artifact")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--max-k", type=int, default=4)
    ap.add_argument("--attention-backend", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--cache-backend", choices=["static", "dynamic"], default="static")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default="experiments/diagnostics/nf4_batch_equivalence.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this NF4 numerical audit")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, quantization_config=quant, device_map=args.device,
        attn_implementation=args.attention_backend)
    model.eval()
    input_path = args.prepared_inputs or args.benchmark
    with open(input_path, encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if args.ids is not None:
        wanted = set(args.ids)
        rows = [r for r in rows if str(r.get("id", r.get("prompt_id"))) in wanted]
    rows = rows[:args.samples]
    results = []
    for row in rows:
        ids = (torch.tensor([row["input_ids"]], dtype=torch.long, device=args.device)
               if "input_ids" in row else
               tokenizer(row["prompt"], return_tensors="pt")["input_ids"].to(args.device))
        result = audit_prefix(model, ids, [k for k in (1, 2, 4) if k <= args.max_k], args.cache_backend)
        result["id"] = row.get("id", row.get("prompt_id", index))
        results.append(result)
        print(f"{row.get('id')}: " + ", ".join(
            f"K={k}, first differing KV layer={v['first_kv_differing_layer']}"
            for k, v in result["horizons"].items()), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "model": args.model, "model_revision": args.revision,
        "attention_backend": args.attention_backend,
        "cache_backend": args.cache_backend, "results": results,
    }, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
