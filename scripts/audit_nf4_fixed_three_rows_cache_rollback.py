#!/usr/bin/env python3
"""Directly audit prefix preservation and crop rollback for fixed-three-row K2/R3."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import audit_nf4_fixed_three_rows_closed_loop as three_row_runner
from zassd.cache.kv_cache import TargetKVCache


def cache_prefix_sha256(cache, prefix_length: int) -> str:
    digest = hashlib.sha256()
    for layer in cache.layers:
        active = int(layer.get_seq_length())
        if prefix_length > active:
            raise ValueError(f"prefix {prefix_length} exceeds active layer length {active}")
        digest.update(prefix_length.to_bytes(8, "little"))
        if not getattr(layer, "is_initialized", False):
            continue
        for tensor in (layer.keys, layer.values):
            raw = tensor[:, :, :prefix_length, :].detach().contiguous().view(__import__("torch").uint8)
            digest.update(raw.cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    base = three_row_runner.base_runner
    audit = {
        "phase": "idle",
        "prompt_len": None,
        "reference_states": {},
        "verify_context": None,
        "block_verifications": [],
        "rollback_events": [],
        "cycle_trace": [],
        "current_model_call": None,
    }

    original_digest = base.diagnostic_cache_prefix_digest

    def observe_digest(cache, prefix_length):
        value = original_digest(cache, prefix_length)
        generated = int(prefix_length) - int(audit["prompt_len"] or 0)
        if audit["phase"] == "reference" and generated >= 0:
            audit["reference_states"][generated] = value
        return value

    base.diagnostic_cache_prefix_digest = observe_digest
    original_reference = base.b1_fixed_row_generate
    original_block = base.b1_fixed_row_block_generate

    def traced_reference(*args, **kwargs):
        audit["phase"] = "reference"
        try:
            return original_reference(*args, **kwargs)
        finally:
            audit["phase"] = "idle"

    def traced_block(*args, **kwargs):
        audit["phase"] = "block"
        kwargs["trace"] = True
        try:
            result = original_block(*args, **kwargs)
            audit["cycle_trace"] = list(getattr(result[1], "cycle_trace", []))
            return result
        finally:
            audit["phase"] = "idle"

    base.b1_fixed_row_generate = traced_reference
    base.b1_fixed_row_block_generate = traced_block
    original_crop = TargetKVCache.crop

    def audited_crop(self, max_length: int) -> None:
        before_length = self.get_seq_length()
        before_layer_lengths = [int(layer.get_seq_length()) for layer in self.cache.layers]
        original_crop(self, max_length)
        after_length = self.get_seq_length()
        after_layer_lengths = [int(layer.get_seq_length()) for layer in self.cache.layers]
        if audit["phase"] != "block" or after_length >= before_length:
            return
        context = audit.get("verify_context")
        active_digest = cache_prefix_sha256(self.cache, after_length)
        generated_prefix = after_length - int(audit["prompt_len"] or 0)
        expected = audit["reference_states"].get(generated_prefix)
        discarded = None
        if context and context.get("post_seq_len", 0) == before_length:
            committed_after_verify = max(0, after_length - context["pre_seq_len"])
            discarded = max(0, context["written_rows"] - committed_after_verify)
        audit["rollback_events"].append({
            "before_length": before_length,
            "requested_length": int(max_length),
            "after_length": after_length,
            "before_layer_lengths_uniform": len(set(before_layer_lengths)) <= 1,
            "after_layer_lengths_uniform": len(set(after_layer_lengths)) <= 1,
            "after_prefix_matches_singleton_reference": expected is not None and active_digest == expected,
            "generated_prefix_after_crop": generated_prefix,
            "discarded_speculative_rows_from_latest_verify": discarded,
            "latest_verify_rows": context.get("input_rows") if context else None,
            "latest_verify_preserved_existing_prefix": context.get("prefix_unchanged") if context else None,
        })
        audit["verify_context"] = None

    TargetKVCache.crop = audited_crop
    original_load = base.load_backend
    hook_handles = []

    def load_and_hook(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        model = loaded.model

        def before_forward(module, inputs, call_kwargs):
            audit["current_model_call"] = None
            input_ids = inputs[0] if inputs else call_kwargs.get("input_ids")
            cache = call_kwargs.get("past_key_values")
            if input_ids is None or input_ids.ndim != 2:
                return
            if (audit["phase"] == "reference" and cache is not None
                    and int(cache.get_seq_length()) == 0 and input_ids.shape[1] > 1):
                audit["prompt_len"] = int(input_ids.shape[1])
            if audit["phase"] != "block" or cache is None or input_ids.shape[1] <= 1:
                return
            pre_len = int(cache.get_seq_length())
            if input_ids.shape[1] != 3:
                return
            record = {
                "pre_seq_len": pre_len,
                "pre_layer_lengths": [int(layer.get_seq_length()) for layer in cache.layers],
                "pre_prefix_sha256": cache_prefix_sha256(cache, pre_len),
                "input_rows": int(input_ids.shape[1]),
            }
            audit["verify_context"] = record
            audit["current_model_call"] = (cache, record)

        def after_forward(module, inputs, call_kwargs, output):
            current = audit.get("current_model_call")
            if audit["phase"] != "block" or current is None:
                return
            cache, record = current
            post_len = int(cache.get_seq_length())
            record["post_seq_len"] = post_len
            record["post_layer_lengths"] = [int(layer.get_seq_length()) for layer in cache.layers]
            record["written_rows"] = post_len - record["pre_seq_len"]
            record["post_prefix_sha256"] = cache_prefix_sha256(cache, record["pre_seq_len"])
            record["prefix_unchanged"] = record["pre_prefix_sha256"] == record["post_prefix_sha256"]
            audit["block_verifications"].append(record.copy())
            audit["current_model_call"] = None

        hook_handles.append(model.register_forward_pre_hook(before_forward, with_kwargs=True))
        hook_handles.append(model.register_forward_hook(after_forward, with_kwargs=True))
        return loaded

    base.load_backend = load_and_hook
    try:
        three_row_runner.main()
    finally:
        base.load_backend = original_load
        base.diagnostic_cache_prefix_digest = original_digest
        base.b1_fixed_row_generate = original_reference
        base.b1_fixed_row_block_generate = original_block
        TargetKVCache.crop = original_crop
        for handle in hook_handles:
            handle.remove()

    output = Path(sys.argv[sys.argv.index("--output") + 1])
    payload = json.loads(output.read_text())
    prompt_len = int(payload["input_prompt_tokens"])
    rollbacks = audit["rollback_events"]
    payload["cache_transaction_audit"] = {
        "scope": "diagnostic hooks around actual model verification and TargetKVCache.crop; only canonical active prefix is compared after crop",
        "verification_calls_with_three_input_rows": len(audit["block_verifications"]),
        "all_verifications_preserved_preexisting_kv_prefix": all(v["prefix_unchanged"] for v in audit["block_verifications"]),
        "verification_layer_lengths_uniform_before_and_after": all(
            len(set(v["pre_layer_lengths"])) <= 1 and len(set(v["post_layer_lengths"])) <= 1
            for v in audit["block_verifications"]
        ),
        "verification_events": audit["block_verifications"],
        "rollback_events": rollbacks,
        "rollback_event_count": len(rollbacks),
        "all_cropped_active_prefixes_match_singleton_reference": all(
            e["after_prefix_matches_singleton_reference"] for e in rollbacks
        ),
        "rollback_layer_lengths_uniform": all(e["after_layer_lengths_uniform"] for e in rollbacks),
        "cycle_trace": audit["cycle_trace"],
        "singleton_reference_prefixes_captured": len(audit["reference_states"]),
        "prompt_tokens": prompt_len,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "output": str(output),
        "token_match": payload["token_match"],
        "all_shared_states_equal": payload["all_shared_states_equal"],
        "verification_calls_with_three_rows": len(audit["block_verifications"]),
        "prefix_preserved": payload["cache_transaction_audit"]["all_verifications_preserved_preexisting_kv_prefix"],
        "rollback_events": len(rollbacks),
        "all_rollback_prefixes_match_reference": payload["cache_transaction_audit"]["all_cropped_active_prefixes_match_singleton_reference"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
