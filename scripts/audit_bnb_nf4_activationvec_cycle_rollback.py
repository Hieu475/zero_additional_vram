#!/usr/bin/env python3
"""Audit accepted-prefix and rejected-suffix rollback for one M=3 cycle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
import triton

from zassd.backends.base import load_backend
from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model
from zassd.decoding.speculative import diagnostic_cache_digest
from audit_bnb_nf4_gate_reduction_orders import MODEL, REVISION, original_bnb_forward
from audit_bnb_nf4_activationvec_block_integration import triton_m3_overrides


EOS_ID = 151645


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = [json.loads(line) for line in Path(args.benchmark).read_text().splitlines() if line.strip()]
    prompt = next((item for item in records if str(item["id"]) == args.prompt_id), None)
    if prompt is None:
        raise KeyError(args.prompt_id)
    loaded = load_backend("hf_bnb_nf4", MODEL, REVISION, "cuda:0", "sdpa")
    model, tokenizer = loaded.model.eval(), loaded.tokenizer
    prompt_ids = tokenizer(prompt["prompt"], return_tensors="pt")["input_ids"].to("cuda:0")
    prompt_len = int(prompt_ids.shape[-1])
    capacity = prompt_len + 12

    singleton_cache = TargetKVCache(backend="static", max_capacity=capacity)
    singleton_logits_after_token = []
    singleton_prefix_digests = []
    greedy_tokens = []
    with torch.inference_mode(), original_bnb_forward(model):
        logits = model(prompt_ids, past_key_values=singleton_cache.cache, use_cache=True).logits[:, -1:, :]
        for _ in range(4):
            token = logits.argmax(dim=-1)
            greedy_tokens.append(int(token.item()))
            logits = model(token, past_key_values=singleton_cache.cache, use_cache=True).logits[:, -1:, :]
            singleton_logits_after_token.append(logits.detach().clone())
            singleton_prefix_digests.append(diagnostic_cache_digest(singleton_cache))

    base_cache = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.inference_mode(), original_bnb_forward(model):
        model(prompt_ids, past_key_values=base_cache.cache, use_cache=True)

    wrong_first = EOS_ID if greedy_tokens[1] != EOS_ID else (greedy_tokens[1] + 1) % int(model.config.vocab_size)
    wrong_second = EOS_ID if greedy_tokens[2] != EOS_ID else (greedy_tokens[2] + 1) % int(model.config.vocab_size)
    scenarios = [
        {"name": "accept_both_bonus", "draft_tokens": [greedy_tokens[1], greedy_tokens[2]]},
        {"name": "reject_first_eos_draft", "draft_tokens": [wrong_first, greedy_tokens[2]]},
        {"name": "accept_first_reject_second_eos_draft", "draft_tokens": [greedy_tokens[1], wrong_second]},
    ]
    scenario_rows = []
    for scenario in scenarios:
        branch_cache = TargetKVCache(cache=base_cache.cache.fork_ephemeral_draft_kv(), max_capacity=capacity)
        block_tokens = torch.tensor([[greedy_tokens[0], *scenario["draft_tokens"]],], device="cuda:0", dtype=torch.long)
        with (torch.inference_mode(), original_bnb_forward(model),
              fixed_row_schedule_for_model(model, physical_rows=3, attention_strategy="bnb_rowwise",
                                          quantized_linear_strategy="fixed_rows"),
              triton_m3_overrides(model)):
            block_logits = model(block_tokens, past_key_values=branch_cache.cache, use_cache=True).logits.detach()

        draft_tokens = scenario["draft_tokens"]
        accepted = 0
        for index in range(2):
            target_token = int(block_logits[0, index].argmax().item())
            if draft_tokens[index] != target_token:
                break
            accepted += 1
        keep_rows = 1 + accepted  # pending input plus accepted draft prefix
        emitted = int(block_logits[0, accepted].argmax().item())
        expected_emitted = greedy_tokens[keep_rows]
        valid_logits = block_logits[:, :keep_rows, :]
        reference_logits = torch.cat(singleton_logits_after_token[:keep_rows], dim=1)
        logits_prefix_equal = torch.equal(valid_logits, reference_logits)

        branch_cache.rollback(prompt_len, keep_rows)
        branch_digest = diagnostic_cache_digest(branch_cache)
        expected_digest = singleton_prefix_digests[keep_rows - 1]
        branch_lengths = [int(layer.get_seq_length()) for layer in branch_cache.cache.layers]
        scenario_rows.append({
            "name": scenario["name"], "draft_tokens": draft_tokens,
            "accepted_draft_count": accepted, "kept_input_rows": keep_rows,
            "emitted_token": emitted, "expected_singleton_token": expected_emitted,
            "emitted_token_equal": emitted == expected_emitted,
            "accepted_prefix_logits_bitwise_equal": logits_prefix_equal,
            "accepted_prefix_max_abs_logit_diff": float((valid_logits.float() - reference_logits.float()).abs().max().item()),
            "rolled_back_cache_sha256": branch_digest,
            "singleton_reference_cache_sha256": expected_digest,
            "rolled_back_cache_bitwise_equal": branch_digest == expected_digest,
            "cache_lengths": branch_lengths,
            "expected_cache_length": prompt_len + keep_rows,
            "cache_lengths_equal": all(length == prompt_len + keep_rows for length in branch_lengths),
            "rejected_draft_was_eos": any(draft_tokens[i] == EOS_ID for i in range(accepted, 2)) if accepted < 2 else False,
        })

    result = {
        "schema": "zassd-bnb-fused-nf4-activationvec-cycle-rollback-v1",
        "classification": "development-only one-cycle accepted-prefix and rollback audit; finite evidence",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "prompt_length": prompt_len, "greedy_bnb_tokens": greedy_tokens,
        "physical_rows": 3, "patched_linear_count": len(model.model.layers) * 7,
        "scenario_count": len(scenario_rows), "all_scenarios_pass": all(
            row["emitted_token_equal"] and row["accepted_prefix_logits_bitwise_equal"] and
            row["rolled_back_cache_bitwise_equal"] and row["cache_lengths_equal"] for row in scenario_rows),
        "scenarios": scenario_rows,
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "integration_source": "scripts/audit_bnb_nf4_activationvec_block_integration.py",
        "kernel_source": "scripts/benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2.py",
        "runtime": {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
                    "bitsandbytes": importlib.metadata.version("bitsandbytes"), "gpu": torch.cuda.get_device_name(0)},
        "scope_limits": ["One cycle on two-candidate/three-row schedule; tests accept-both, reject-first, and accept-one/reject-second paths.",
                         "EOS is tested as a rejected draft candidate; accepted target EOS and termination are not covered.",
                         "No multi-cycle pending-token/bonus transition, long closed-loop or end-to-end latency coverage.",
                         "One prompt, one GPU/toolchain; GSM8K held-out indices 300-349 excluded."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_scenarios_pass"]:
        raise RuntimeError("At least one speculative accepted-prefix/rollback case diverged")


if __name__ == "__main__":
    main()
