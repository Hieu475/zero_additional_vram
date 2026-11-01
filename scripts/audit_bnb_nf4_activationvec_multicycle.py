#!/usr/bin/env python3
"""Audit pending-token, bonus-token and rejection rollback over three cycles."""

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
    capacity = prompt_len + 16

    singleton_cache = TargetKVCache(backend="static", max_capacity=capacity)
    greedy_tokens = []
    singleton_logits_after_token = []
    singleton_prefix_digests = []
    with torch.inference_mode(), original_bnb_forward(model):
        logits = model(prompt_ids, past_key_values=singleton_cache.cache, use_cache=True).logits[:, -1:, :]
        for _ in range(7):
            token = logits.argmax(dim=-1)
            greedy_tokens.append(int(token.item()))
            logits = model(token, past_key_values=singleton_cache.cache, use_cache=True).logits[:, -1:, :]
            singleton_logits_after_token.append(logits.detach().clone())
            singleton_prefix_digests.append(diagnostic_cache_digest(singleton_cache))

    canonical = TargetKVCache(backend="static", max_capacity=capacity)
    with torch.inference_mode(), original_bnb_forward(model):
        model(prompt_ids, past_key_values=canonical.cache, use_cache=True)

    alternative_first = EOS_ID if greedy_tokens[1] != EOS_ID else (greedy_tokens[1] + 1) % int(model.config.vocab_size)
    alternative_second = EOS_ID if greedy_tokens[5] != EOS_ID else (greedy_tokens[5] + 1) % int(model.config.vocab_size)
    cycle_plan = [
        {"name": "reject_first_emit_pending", "drafts": [alternative_first, greedy_tokens[2]]},
        {"name": "accept_both_emit_bonus", "drafts": [greedy_tokens[2], greedy_tokens[3]]},
        {"name": "accept_one_reject_second_eos", "drafts": [greedy_tokens[5], alternative_second]},
    ]
    cycles = []
    consumed = 0
    for cycle_index, plan in enumerate(cycle_plan):
        prefix_len = canonical.get_seq_length()
        expected_prefix_len = prompt_len + consumed
        if prefix_len != expected_prefix_len:
            raise RuntimeError(f"cycle {cycle_index}: canonical prefix length {prefix_len}, expected {expected_prefix_len}")
        branch = TargetKVCache(cache=canonical.cache.fork_ephemeral_draft_kv(), max_capacity=capacity)
        verify_ids = torch.tensor([[greedy_tokens[consumed], *plan["drafts"]]], device="cuda:0", dtype=torch.long)
        with (torch.inference_mode(), original_bnb_forward(model),
              fixed_row_schedule_for_model(model, physical_rows=3, attention_strategy="bnb_rowwise",
                                          quantized_linear_strategy="fixed_rows"),
              triton_m3_overrides(model)):
            block_logits = model(verify_ids, past_key_values=branch.cache, use_cache=True).logits.detach()

        accepted = 0
        for candidate_idx in range(2):
            target = int(block_logits[0, candidate_idx].argmax().item())
            if plan["drafts"][candidate_idx] != target:
                break
            accepted += 1
        keep_rows = 1 + accepted
        emitted = int(block_logits[0, accepted].argmax().item())
        expected_emitted = greedy_tokens[consumed + keep_rows]
        reference_logits = singleton_logits_after_token[consumed + keep_rows - 1]
        selected_logits = block_logits[:, accepted:accepted + 1, :]
        logits_equal = torch.equal(selected_logits, reference_logits)

        branch.rollback(prefix_len, keep_rows)
        digest = diagnostic_cache_digest(branch)
        expected_digest = singleton_prefix_digests[consumed + keep_rows - 1]
        lengths = [int(layer.get_seq_length()) for layer in branch.cache.layers]
        cycles.append({
            "cycle": cycle_index,
            "name": plan["name"],
            "pending_input_token": greedy_tokens[consumed],
            "draft_tokens": plan["drafts"],
            "accepted_draft_count": accepted,
            "kept_input_rows": keep_rows,
            "emitted_token": emitted,
            "expected_singleton_token": expected_emitted,
            "emitted_token_equal": emitted == expected_emitted,
            "selected_next_logits_bitwise_equal": logits_equal,
            "selected_next_logits_max_abs_diff": float((selected_logits.float() - reference_logits.float()).abs().max().item()),
            "rollback_kv_sha256": digest,
            "singleton_reference_kv_sha256": expected_digest,
            "rollback_kv_bitwise_equal": digest == expected_digest,
            "cache_length": lengths[0] if lengths else None,
            "expected_cache_length": prompt_len + consumed + keep_rows,
            "all_layer_cache_lengths_equal": all(length == prompt_len + consumed + keep_rows for length in lengths),
            "rejected_eos_draft": accepted < 2 and plan["drafts"][accepted] == EOS_ID,
        })
        canonical = branch
        consumed += keep_rows

    result = {
        "schema": "zassd-bnb-fused-nf4-activationvec-multicycle-v1",
        "classification": "development-only three-cycle greedy identity/state audit; finite evidence",
        "model": MODEL, "revision": REVISION, "prompt_id": args.prompt_id,
        "prompt_length": prompt_len, "greedy_bnb_tokens": greedy_tokens,
        "physical_rows": 3, "patched_linear_count": len(model.model.layers) * 7,
        "cycle_count": len(cycles), "all_cycles_pass": all(
            row["emitted_token_equal"] and row["selected_next_logits_bitwise_equal"] and
            row["rollback_kv_bitwise_equal"] and row["all_layer_cache_lengths_equal"] for row in cycles),
        "final_processed_greedy_tokens": consumed,
        "final_cache_sha256": diagnostic_cache_digest(canonical),
        "singleton_final_reference_sha256": singleton_prefix_digests[consumed - 1],
        "final_state_bitwise_equal": diagnostic_cache_digest(canonical) == singleton_prefix_digests[consumed - 1],
        "cycles": cycles,
        "source": str(Path(__file__).resolve()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "integration_source": "scripts/audit_bnb_nf4_activationvec_block_integration.py",
        "kernel_source": "scripts/benchmark_bnb_nf4_fused_inline_ldcs_activationvec_tailmask_half2.py",
        "runtime": {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda,
                    "bitsandbytes": importlib.metadata.version("bitsandbytes"), "gpu": torch.cuda.get_device_name(0)},
        "scope_limits": ["Three scripted cycles: reject-first, accept-both/bonus, accept-one then reject-second EOS draft.",
                         "One prompt and finite greedy prefix; does not cover accepted target EOS or all draft patterns.",
                         "No end-to-end latency measurement or held-out GSM8K use."]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_cycles_pass"] or not result["final_state_bitwise_equal"]:
        raise RuntimeError("Multi-cycle Triton state diverged from greedy BNB singleton")


if __name__ == "__main__":
    main()
