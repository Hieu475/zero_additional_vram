"""Closed-loop B1 audit with a one-row singleton NF4 kernel prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import torch
from bitsandbytes.nn import Linear4bit

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.b1_exact import b1_fixed_row_block_generate, b1_fixed_row_generate
import zassd.certification.fixed_row_dispatch as fixed_row_dispatch
from zassd.certification.triton_b1_reference_contract import (
    verify_triton_b1_reference_contract,
)
from zassd.certification.nf4_fixed_row_singleton_group_scale_packed import (
    nf4_fixed_row_singleton_group_scale_packed_rows_linear,
)
from audit_triton_b1_closed_loop import diagnostic_cache_prefix_digest, fixed_row_sdpa_decode
from run_strong_baseline_comparison import run_vanilla_timed


STRATEGY = "triton_nf4_group_scale_packed_rows2_n16_warp1_k256"


@contextmanager
def singleton_kernel_schedule():
    """Intercept the experimental strategy without editing the pinned adapter."""
    original_dispatch = fixed_row_dispatch.fixed_row_schedule_for_model

    @contextmanager
    def adaptive_dispatch(model, *, physical_rows=2, attention_strategy="rowwise",
                          observer=None, reuse_linear_padding_buffer=False,
                          quantized_linear_strategy="fixed_rows"):
        with original_dispatch(
            model,
            physical_rows=physical_rows,
            attention_strategy=attention_strategy,
            observer=observer,
            reuse_linear_padding_buffer=reuse_linear_padding_buffer,
            quantized_linear_strategy=quantized_linear_strategy,
        ):
            restored = []
            try:
                if quantized_linear_strategy == STRATEGY:
                    if physical_rows != 2:
                        raise ValueError("one-row prototype requires physical_rows=2")
                    for module in model.modules():
                        if not isinstance(module, Linear4bit):
                            continue
                        base_forward = module.forward

                        def one_row_or_base(self, x, _base=base_forward):
                            if x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == 1:
                                return nf4_fixed_row_singleton_group_scale_packed_rows_linear(
                                    x, self, block_n=16, block_k=256, num_warps=1
                                )
                            return _base(x)

                        restored.append((module, base_forward))
                        module.forward = MethodType(one_row_or_base, module)
                yield model
            finally:
                for module, base_forward in reversed(restored):
                    module.forward = base_forward

    fixed_row_dispatch.fixed_row_schedule_for_model = adaptive_dispatch
    try:
        yield
    finally:
        fixed_row_dispatch.fixed_row_schedule_for_model = original_dispatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/benchmarks/cnndm_eval.jsonl")
    parser.add_argument("--prompt-id", default="cnndm_003")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--session", type=int, default=0)
    parser.add_argument("--timing-rounds", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--enable-experimental-singleton", action="store_true", required=True,
                        help="explicitly opt in to the non-contract experimental kernel")
    parser.add_argument("--manifest", type=Path, default=Path(
        "experiments/exact_quantized_feasibility/triton_nf4_singleton_row_kernel_v1/experiment_manifest.json"))
    args = parser.parse_args()
    validator = Path(__file__).resolve().with_name("validate_nf4_singleton_experiment_manifest.py")
    checked = subprocess.run([sys.executable, str(validator), "--manifest", str(args.manifest)],
                             cwd=Path(__file__).resolve().parents[1], check=False)
    if checked.returncode != 0:
        raise SystemExit(checked.returncode)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    rows = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((row for row in rows if str(row["id"]) == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt ID {args.prompt_id!r} absent")
    model_name = "Qwen/Qwen2.5-3B-Instruct"
    revision = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    loaded = load_backend("hf_bnb_nf4", model_name, revision, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    base_contract_path = Path(
        "experiments/exact_quantized_feasibility/b1_triton_batched_k1r2_n16_reference_contract_v5.json"
    )
    base_contract = verify_triton_b1_reference_contract(
        model, tokenizer, base_contract_path, device="cuda:0"
    )
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    max_capacity = int(prompt_ids.shape[1]) + args.max_new_tokens + 4
    reference_states, block_states = {}, {}

    def reference_observer(n, cache):
        reference_states[n] = diagnostic_cache_prefix_digest(
            cache, int(prompt_ids.shape[1]) + n
        )

    def block_observer(n, cache):
        first_missing = max(block_states, default=-1) + 1
        for prefix in range(first_missing, n + 1):
            block_states[prefix] = diagnostic_cache_prefix_digest(
                cache, int(prompt_ids.shape[1]) + prefix
            )

    common = {
        "k": 1,
        "ngram_size": 3,
        "max_new_tokens": args.max_new_tokens,
        "device": "cuda:0",
        "kv_cache_max_capacity": max_capacity,
        "physical_rows": 2,
        "attention_strategy": "triton_batched",
        "quantized_linear_strategy": STRATEGY,
    }
    with singleton_kernel_schedule():
        with fixed_row_sdpa_decode(model, 2, 32, "triton_batched"), torch.inference_mode():
            _, reference_metrics = b1_fixed_row_generate(
                model, tokenizer, prompt_ids, **common,
                state_observer=reference_observer,
            )
        with fixed_row_sdpa_decode(model, 2, 32, "triton_batched"), torch.inference_mode():
            _, block_metrics = b1_fixed_row_block_generate(
                model, tokenizer, prompt_ids, **common,
                state_observer=block_observer,
            )

        methods = ("original_bnb_greedy", "optimized_b1_singleton", "optimized_b1_block")
        for method in methods:
            if method == "original_bnb_greedy":
                run_vanilla_timed(model, tokenizer, prompt_ids, 8, "cuda:0")
            else:
                with fixed_row_sdpa_decode(model, 2, 32, "triton_batched"), torch.inference_mode():
                    decode = b1_fixed_row_generate if method.endswith("singleton") else b1_fixed_row_block_generate
                    decode(model, tokenizer, prompt_ids, **{**common, "max_new_tokens": 8})
        timing = []
        for round_index in range(args.timing_rounds):
            order = list(methods[round_index % len(methods):] + methods[:round_index % len(methods)])
            if args.session % 2:
                order.reverse()
            result = {"round": round_index, "method_order": order, "methods": {}}
            for method in order:
                torch.cuda.synchronize("cuda:0")
                torch.cuda.reset_peak_memory_stats("cuda:0")
                memory_allocated_before = torch.cuda.memory_allocated("cuda:0")
                memory_reserved_before = torch.cuda.memory_reserved("cuda:0")
                if method == "original_bnb_greedy":
                    token_ids, prefill_s, decode_s, total_s = run_vanilla_timed(
                        model, tokenizer, prompt_ids, args.max_new_tokens, "cuda:0"
                    )
                    method_result = {"token_ids": token_ids, "total_s": total_s / 1000,
                                     "prefill_s": prefill_s / 1000, "decode_s": decode_s / 1000}
                else:
                    decode = b1_fixed_row_generate if method.endswith("singleton") else b1_fixed_row_block_generate
                    with fixed_row_sdpa_decode(model, 2, 32, "triton_batched"), torch.inference_mode():
                        _, metrics = decode(model, tokenizer, prompt_ids, **common)
                    method_result = {
                        "token_ids": metrics.generated_token_ids,
                        "total_s": metrics.total_time_s,
                        "prefill_s": metrics.prefill_time_s,
                        "verify_s": getattr(metrics, "verify_time_s", None),
                        "accepted_draft_tokens": getattr(metrics, "accepted_draft_tokens", None),
                        "draft_tokens": getattr(metrics, "draft_tokens", None),
                    }
                torch.cuda.synchronize("cuda:0")
                peak_allocated = torch.cuda.max_memory_allocated("cuda:0")
                peak_reserved = torch.cuda.max_memory_reserved("cuda:0")
                method_result["memory"] = {
                    "baseline_allocated_bytes": memory_allocated_before,
                    "peak_allocated_bytes": peak_allocated,
                    "incremental_peak_allocated_bytes": max(0, peak_allocated - memory_allocated_before),
                    "baseline_reserved_bytes": memory_reserved_before,
                    "peak_reserved_bytes": peak_reserved,
                    "incremental_peak_reserved_bytes": max(0, peak_reserved - memory_reserved_before),
                }
                result["methods"][method] = method_result
            result["block_matches_singleton"] = (
                result["methods"]["optimized_b1_block"]["token_ids"] ==
                result["methods"]["optimized_b1_singleton"]["token_ids"]
            )
            result["block_matches_original_bnb"] = (
                result["methods"]["optimized_b1_block"]["token_ids"] ==
                result["methods"]["original_bnb_greedy"]["token_ids"]
            )
            timing.append(result)

    shared = sorted(set(reference_states) & set(block_states))
    lengths_equal = set(reference_states) == set(block_states)
    b1_ratios = [
        row["methods"]["optimized_b1_singleton"]["total_s"] /
        row["methods"]["optimized_b1_block"]["total_s"] for row in timing
    ]
    bnb_ratios = [
        row["methods"]["original_bnb_greedy"]["total_s"] /
        row["methods"]["optimized_b1_block"]["total_s"] for row in timing
    ]
    payload = {
        "status": "experimental singleton optimization; no contract update or runtime integration",
        "model": model_name,
        "revision": revision,
        "prompt_id": args.prompt_id,
        "prompt_sha256": hashlib.sha256(prompt["prompt"].encode()).hexdigest(),
        "input_prompt_characters": len(prompt["prompt"]),
        "input_prompt_tokens": int(prompt_ids.shape[1]),
        "session": args.session,
        "experiment_manifest": str(args.manifest),
        "manifest_preflight": "passed",
        "base_b1_reference_contract": {
            "contract_id": base_contract.contract_id,
            "model_state_sha256": base_contract.model_state_sha256,
            "verified_source_count": base_contract.verified_source_count,
        },
        "parameters": {"k": 1, "ngram_size": 3, "physical_rows": 2,
                       "max_new_tokens": args.max_new_tokens, "block_k": 32,
                       "attention_strategy": "triton_batched",
                       "quantized_linear_strategy": STRATEGY,
                       "singleton_nf4_kernel": "one-row experimental kernel",
                       "block_nf4_kernel": "existing two-row kernel",
                       "timing_rounds": args.timing_rounds},
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "reference": {"token_ids": reference_metrics.generated_token_ids,
                      "state_prefixes": len(reference_states)},
        "block": {"token_ids": block_metrics.generated_token_ids,
                  "state_prefixes": len(block_states)},
        "token_match": reference_metrics.generated_token_ids == block_metrics.generated_token_ids,
        "shared_state_prefixes": len(shared),
        "all_shared_states_equal": all(reference_states[n] == block_states[n] for n in shared),
        "same_prefix_domain": lengths_equal,
        "state_mismatches": [n for n in shared if reference_states[n] != block_states[n]],
        "memory_telemetry": {
            "scope": "per-method decode call after model load; peak includes persistent model allocations",
            "incremental_peak_allocated_bytes": "peak_allocated_bytes minus allocated baseline immediately before the call",
            "reserved_memory_note": "allocator-reserved values can remain cached and are not equivalent to live tensor bytes",
        },
        "timing": {"paired_rounds": timing,
                   "geometric_mean_speedup_vs_optimized_b1_singleton": math.exp(statistics.mean(map(math.log, b1_ratios))),
                   "geometric_mean_speedup_vs_original_bnb": math.exp(statistics.mean(map(math.log, bnb_ratios))),
                   "speedup_vs_b1_samples": b1_ratios,
                   "speedup_vs_bnb_samples": bnb_ratios},
        "limitations": [
            "Finite development-only closed loop; singleton uses the prototype and block uses the current two-row kernel.",
            "Not yet pinned by a live B1 reference contract; no BNB state-equivalence claim.",
            "Held-out GSM8K indices 300-349 were not used.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "prompt_id": args.prompt_id,
        "session": args.session,
        "max_new_tokens": args.max_new_tokens,
        "token_match": payload["token_match"],
        "shared_state_prefixes": payload["shared_state_prefixes"],
        "all_shared_states_equal": payload["all_shared_states_equal"],
        "same_prefix_domain": payload["same_prefix_domain"],
        "timing_speedup_vs_optimized_b1_singleton": b1_ratios,
        "timing_speedup_vs_original_bnb": bnb_ratios,
        "block_matches_bnb_by_round": [row["block_matches_original_bnb"] for row in timing],
        "output": str(out),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
