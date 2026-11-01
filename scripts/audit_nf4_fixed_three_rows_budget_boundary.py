#!/usr/bin/env python3
"""Run budget-boundary audits with enough KV capacity for the shared 8-token warmup."""

from __future__ import annotations

import audit_nf4_fixed_three_rows_eos_trace as trace_runner


def main() -> None:
    base = trace_runner.three_row_runner.base_runner
    original_reference = base.b1_fixed_row_generate
    original_block = base.b1_fixed_row_block_generate

    def with_warmup_capacity(function):
        def wrapped(model, tokenizer, prompt, *args, **kwargs):
            prompt_tokens = int(prompt.shape[-1]) if hasattr(prompt, "shape") else int(
                tokenizer(prompt, return_tensors="pt")["input_ids"].shape[-1]
            )
            needed = prompt_tokens + int(kwargs.get("max_new_tokens", 128)) + 4
            kwargs["kv_cache_max_capacity"] = max(
                int(kwargs.get("kv_cache_max_capacity", 0)), needed
            )
            return function(model, tokenizer, prompt, *args, **kwargs)
        return wrapped

    base.b1_fixed_row_generate = with_warmup_capacity(original_reference)
    base.b1_fixed_row_block_generate = with_warmup_capacity(original_block)
    try:
        trace_runner.main()
    finally:
        base.b1_fixed_row_generate = original_reference
        base.b1_fixed_row_block_generate = original_block


if __name__ == "__main__":
    main()
