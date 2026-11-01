#!/usr/bin/env python3
"""Closed-loop B1 singleton vs block audit with a fixed Triton SDPA schedule.

This is a development-only experiment. The Triton attention schedule defines
the B1 reference used by both paths; it does not claim equality to native BNB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
from prototype_fixed_row_sdpa_triton import fixed_row_attention_tiled
try:
    from run_strong_baseline_comparison import run_vanilla_timed
except ModuleNotFoundError:
    from scripts.run_strong_baseline_comparison import run_vanilla_timed


@contextmanager
def fixed_row_sdpa_decode(model, physical_rows: int, block_k: int,
                           attention_strategy: str):
    """Use the custom kernel for decode SDPA and preserve native prefill SDPA."""
    original = F.scaled_dot_product_attention
    original_lm_head = model.lm_head.forward

    def dispatch(query, key, value, *args, **kwargs):
        if query.shape[-2] > physical_rows:
            return original(query, key, value, *args, **kwargs)
        attn_mask = kwargs.get("attn_mask", args[0] if args else None)
        dropout = kwargs.get("dropout_p", args[1] if len(args) > 1 else 0.0)
        is_causal = kwargs.get("is_causal", args[3] if len(args) > 3 else False)
        if dropout != 0.0 or is_causal:
            raise RuntimeError("custom B1 SDPA requires zero dropout and explicit causal prefixes")
        if attention_strategy == "bnb_rowwise" and attn_mask is not None:
            raise RuntimeError("bnb_rowwise custom SDPA requires no mask")
        fixed_attention_rows = attention_strategy in {"padded", "triton_padded"}
        if fixed_attention_rows:
            if attn_mask is None:
                raise RuntimeError("fixed-row custom SDPA requires a causal mask")
            q_len, k_len = query.shape[-2], key.shape[-2]
            if (q_len > physical_rows or
                    (attention_strategy == "padded" and q_len != physical_rows)):
                raise RuntimeError("query shape violates fixed-row attention contract")
            prefix = k_len - physical_rows
            expected = torch.full_like(attn_mask, float("-inf"))
            for row in range(q_len):
                expected[..., row, :prefix + row + 1] = 0
            torch._assert_async(
                torch.all(attn_mask == expected),
                "padded SDPA mask violates the contiguous causal-prefix contract",
            )
        elif attention_strategy == "triton_batched" and attn_mask is not None:
            raise RuntimeError("triton_batched SDPA requires the adapter's unpadded prefix call")
        elif attention_strategy not in {"bnb_rowwise", "triton_batched"}:
            raise ValueError(f"unsupported attention strategy: {attention_strategy}")
        if key.shape[1] != query.shape[1]:
            if query.shape[1] % key.shape[1]:
                raise RuntimeError("unsupported GQA head ratio")
            groups = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        scale = kwargs.get("scale")
        if scale is None:
            scale = query.shape[-1] ** -0.5
        return fixed_row_attention_tiled(
            query, key, value, float(scale), block_k=block_k,
            physical_rows=physical_rows if fixed_attention_rows else None,
        )

    def rowwise_lm_head(hidden_states, *args, **kwargs):
        if hidden_states.ndim >= 3 and hidden_states.shape[-2] > 1:
            return torch.cat([
                original_lm_head(hidden_states[..., row:row + 1, :], *args, **kwargs)
                for row in range(hidden_states.shape[-2])
            ], dim=-2)
        return original_lm_head(hidden_states, *args, **kwargs)

    F.scaled_dot_product_attention = dispatch
    if attention_strategy in {"padded", "triton_padded"}:
        model.lm_head.forward = rowwise_lm_head
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original
        model.lm_head.forward = original_lm_head


def _cache_prefixes(cache: TargetKVCache, prompt_len: int, from_count: int,
                    to_count: int) -> list[dict]:
    return [
        {"generated_prefix_tokens": n,
         "sha256": diagnostic_cache_prefix_digest(cache, prompt_len + n)}
        for n in range(from_count, to_count + 1)
    ]


def diagnostic_cache_prefix_digest(cache: TargetKVCache, prefix_length: int) -> str:
    digest = hashlib.sha256()
    for layer in cache.cache.layers:
        active = int(layer.get_seq_length())
        if prefix_length > active:
            raise ValueError(f"requested KV prefix {prefix_length} exceeds active length {active}")
        digest.update(prefix_length.to_bytes(8, "little"))
        if not getattr(layer, "is_initialized", False):
            continue
        for tensor in (layer.keys, layer.values):
            raw = tensor[:, :, :prefix_length, :].detach().contiguous().view(torch.uint8)
            digest.update(raw.cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/cnndm_eval.jsonl")
    parser.add_argument("--prompt-id", default="cnndm_000")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--ngram-size", type=int, default=3)
    parser.add_argument("--physical-rows", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=32)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--timing-rounds", type=int, default=3)
    parser.add_argument("--attention-strategy",
                        choices=("bnb_rowwise", "padded", "triton_padded", "triton_batched"),
                        default="bnb_rowwise")
    parser.add_argument("--quantized-linear-strategy",
                        choices=("fixed_rows", "triton_nf4_group_scale_packed_k256",
                                 "triton_nf4_group_scale_packed_rows2_k256",
                                 "triton_nf4_group_scale_packed_rows2_n16_k256",
                                 "triton_nf4_group_scale_packed_rows2_n16_warp1_k256"),
                        default="fixed_rows")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    if args.physical_rows < args.k + 1:
        raise ValueError("physical_rows must cover pending token plus K draft rows")
    if args.ngram_size < 2:
        raise ValueError("ngram_size must be at least 2")

    benchmark_path = Path(args.benchmark)
    rows = [json.loads(line) for line in benchmark_path.read_text().splitlines() if line.strip()]
    prompt = next((r for r in rows if str(r["id"]) == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt ID {args.prompt_id!r} not found")

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    loaded = load_backend("hf_bnb_nf4", model_name, revision, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    max_capacity = int(prompt_ids.shape[1]) + args.max_new_tokens + 4
    if max_capacity > 2048:
        raise ValueError("prompt plus generation exceeds selected cache capacity")
    reference_states: list[dict] = []
    block_states: list[dict] = []
    reference_lengths: dict[int, dict] = {}
    block_lengths: dict[int, dict] = {}

    def cache_lengths(cache: TargetKVCache) -> dict:
        return {
            "cache_seq_length": int(cache.get_seq_length()),
            "layer_seq_lengths": [int(layer.get_seq_length()) for layer in cache.cache.layers],
        }

    def observe_reference(n: int, cache: TargetKVCache) -> None:
        reference_lengths[n] = cache_lengths(cache)
        reference_states.extend(_cache_prefixes(cache, int(prompt_ids.shape[1]), n, n))

    def observe_block(n: int, cache: TargetKVCache) -> None:
        block_lengths[n] = cache_lengths(cache)
        start = 0 if not block_states else block_states[-1]["generated_prefix_tokens"] + 1
        block_states.extend(_cache_prefixes(cache, int(prompt_ids.shape[1]), start, n))

    common = dict(
        k=args.k, ngram_size=args.ngram_size, max_new_tokens=args.max_new_tokens,
        device="cuda:0", kv_cache_max_capacity=max_capacity,
        physical_rows=args.physical_rows, attention_strategy=args.attention_strategy,
        quantized_linear_strategy=args.quantized_linear_strategy,
    )
    with fixed_row_sdpa_decode(model, args.physical_rows, args.block_k,
                               args.attention_strategy):
        _, reference_metrics = b1_fixed_row_generate(
            model, tokenizer, prompt_ids, **common, state_observer=observe_reference
        )
    with fixed_row_sdpa_decode(model, args.physical_rows, args.block_k,
                               args.attention_strategy):
        _, block_metrics = b1_fixed_row_block_generate(
            model, tokenizer, prompt_ids, **common, state_observer=observe_block, trace=True
        )

    ref_map = {row["generated_prefix_tokens"]: row["sha256"] for row in reference_states}
    block_map = {row["generated_prefix_tokens"]: row["sha256"] for row in block_states}
    shared = sorted(set(ref_map) & set(block_map))
    length_frontiers = sorted(set(reference_lengths) & set(block_lengths))
    lengths_exact = all(
        reference_lengths[n] == block_lengths[n] for n in length_frontiers
    )
    source_files = [
        "scripts/audit_triton_b1_closed_loop.py",
        "scripts/prototype_fixed_row_sdpa_triton.py",
        "src/zassd/decoding/b1_exact.py",
        "src/zassd/certification/qwen2_padded.py",
        "src/zassd/certification/fixed_row_dispatch.py",
    ]
    file_hashes = {
        name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in source_files
    }

    # Warm each execution path, then measure without cache observers or hashes.
    timing_methods = ["original_bnb_greedy", "b1_fixed_row_singleton", "b1_fixed_row_block"]
    for method in timing_methods:
        if method == "original_bnb_greedy":
            run_vanilla_timed(model, tokenizer, prompt_ids, 8, "cuda:0")
        elif method == "b1_fixed_row_singleton":
            with fixed_row_sdpa_decode(model, args.physical_rows, args.block_k,
                                       args.attention_strategy):
                b1_fixed_row_generate(
                    model, tokenizer, prompt_ids, **{**common, "max_new_tokens": 8}
                )
        else:
            with fixed_row_sdpa_decode(model, args.physical_rows, args.block_k,
                                       args.attention_strategy):
                b1_fixed_row_block_generate(
                    model, tokenizer, prompt_ids, **{**common, "max_new_tokens": 8}
                )

    timing_samples = []
    for round_index in range(args.timing_rounds):
        order = timing_methods[round_index % len(timing_methods):] + timing_methods[:round_index % len(timing_methods)]
        if args.session % 2:
            order.reverse()
        round_result = {"round": round_index, "method_order": order, "methods": {}}
        for method in order:
            torch.cuda.synchronize("cuda:0")
            if method == "original_bnb_greedy":
                tokens, prefill_ms, decode_ms, total_ms = run_vanilla_timed(
                    model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0"
                )
                sample = {"token_ids": tokens, "total_s": total_ms / 1000,
                          "prefill_s": prefill_ms / 1000, "decode_s": decode_ms / 1000}
            elif method == "b1_fixed_row_singleton":
                with fixed_row_sdpa_decode(model, args.physical_rows, args.block_k,
                                           args.attention_strategy):
                    _, metrics = b1_fixed_row_generate(model, tokenizer, prompt_ids, **common)
                sample = {"token_ids": metrics.generated_token_ids,
                          "total_s": metrics.total_time_s, "prefill_s": metrics.prefill_time_s}
            else:
                with fixed_row_sdpa_decode(model, args.physical_rows, args.block_k,
                                           args.attention_strategy):
                    _, metrics = b1_fixed_row_block_generate(
                        model, tokenizer, prompt_ids, **common, trace=True
                    )
                sample = {"token_ids": metrics.generated_token_ids,
                          "total_s": metrics.total_time_s, "prefill_s": metrics.prefill_time_s,
                          "verify_s": metrics.verify_time_s,
                          "accepted_draft_tokens": metrics.accepted_draft_tokens,
                          "draft_tokens": metrics.draft_tokens}
            round_result["methods"][method] = sample
        round_result["block_matches_b1"] = (
            round_result["methods"]["b1_fixed_row_block"]["token_ids"] ==
            round_result["methods"]["b1_fixed_row_singleton"]["token_ids"]
        )
        round_result["block_matches_original_bnb"] = (
            round_result["methods"]["b1_fixed_row_block"]["token_ids"] ==
            round_result["methods"]["original_bnb_greedy"]["token_ids"]
        )
        timing_samples.append(round_result)

    b1_ratios = [
        row["methods"]["b1_fixed_row_singleton"]["total_s"] /
        row["methods"]["b1_fixed_row_block"]["total_s"]
        for row in timing_samples
    ]
    bnb_ratios = [
        row["methods"]["original_bnb_greedy"]["total_s"] /
        row["methods"]["b1_fixed_row_block"]["total_s"]
        for row in timing_samples
    ]
    result = {
        "experiment": "triton_fixed_row_sdpa_b1_closed_loop",
        "scope": "development_prompt_only; B1 fixed-row schedule; not BNB equivalence",
        "model": model_name,
        "revision": revision,
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "input_ids_sha256": hashlib.sha256(
            json.dumps(prompt_ids[0].tolist()).encode()
        ).hexdigest(),
        "session": args.session,
        "parameters": {
            "k": args.k, "ngram_size": args.ngram_size,
            "physical_rows": args.physical_rows,
            "block_k": args.block_k, "max_new_tokens": args.max_new_tokens,
            "attention_strategy": args.attention_strategy,
            "quantized_linear_strategy": args.quantized_linear_strategy,
            "timing_rounds": args.timing_rounds,
        },
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "triton": __import__("triton").__version__,
        },
        "source_sha256": file_hashes,
        "reference": {
            "token_ids": reference_metrics.generated_token_ids,
            "instrumented_audit_wall_s": reference_metrics.total_time_s,
            "instrumented_audit_prefill_s": reference_metrics.prefill_time_s,
            "target_forwards": reference_metrics.target_forwards,
            "state_prefixes": len(ref_map),
        },
        "block": {
            "token_ids": block_metrics.generated_token_ids,
            "instrumented_audit_wall_s": block_metrics.total_time_s,
            "instrumented_audit_prefill_s": block_metrics.prefill_time_s,
            "instrumented_audit_verify_s": block_metrics.verify_time_s,
            "cycles": block_metrics.cycle_trace,
            "state_prefixes": len(block_map),
        },
        "token_match": reference_metrics.generated_token_ids == block_metrics.generated_token_ids,
        "shared_state_prefixes": len(shared),
        "all_shared_states_equal": all(ref_map[n] == block_map[n] for n in shared),
        "same_prefix_domain": set(ref_map) == set(block_map),
        "state_mismatches": [n for n in shared if ref_map[n] != block_map[n]],
        "logical_lengths_exact_at_shared_commit_frontiers": lengths_exact,
        "shared_commit_frontiers": len(length_frontiers),
        "logical_length_trace": [
            {"generated_prefix_tokens": n,
             "reference": reference_lengths[n], "block": block_lengths[n],
             "equal": reference_lengths[n] == block_lengths[n]}
            for n in length_frontiers
        ],
        "state_trace": [
            {"generated_prefix_tokens": n, "reference_sha256": ref_map.get(n),
             "block_sha256": block_map.get(n), "equal": ref_map.get(n) == block_map.get(n)}
            for n in sorted(set(ref_map) | set(block_map))
        ],
        "timing": {
            "observer_hashing_in_timed_region": False,
            "note": "Separate uninstrumented pass; audit_wall_s above includes KV hash callbacks and is not benchmark data.",
            "paired_rounds": timing_samples,
            "block_speedup_vs_b1_samples": b1_ratios,
            "block_speedup_vs_original_bnb_samples": bnb_ratios,
            "geometric_mean_speedup_vs_b1": __import__("math").exp(
                sum(__import__("math").log(x) for x in b1_ratios) / len(b1_ratios)
            ),
            "geometric_mean_speedup_vs_original_bnb": __import__("math").exp(
                sum(__import__("math").log(x) for x in bnb_ratios) / len(bnb_ratios)
            ),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(out), "token_match": result["token_match"],
        "shared_state_prefixes": result["shared_state_prefixes"],
        "all_shared_states_equal": result["all_shared_states_equal"],
        "logical_lengths_exact_at_shared_commit_frontiers": lengths_exact,
        "same_prefix_domain": result["same_prefix_domain"],
        "timing_speedup_vs_b1": result["timing"]["geometric_mean_speedup_vs_b1"],
        "timing_speedup_vs_bnb": result["timing"]["geometric_mean_speedup_vs_original_bnb"],
        "reference_s": reference_metrics.total_time_s,
        "block_s": block_metrics.total_time_s,
    }, indent=2))


if __name__ == "__main__":
    main()
