#!/usr/bin/env python3
"""Run benchmark_b2_lazy_lm_head.py with the eager BNB timing path given a cache capacity large enough for long prompts.

_common.run_vanilla_timed uses the default 2048-token static cache; this wrapper replaces it in-process with a
byte-for-byte copy of its loop except for the capacity. Usage: run_long_context_lazy_benchmark.py [benchmark args...]
"""

import runpy
import sys
import time

import torch

import _common
from zassd.cache.kv_cache import TargetKVCache


def run_vanilla_timed_long(model, tokenizer, prompt, max_new_tokens, device):
    ids = (prompt if isinstance(prompt, torch.Tensor) else _common.tokenize_prompt(tokenizer, prompt)).to(device)
    cache = TargetKVCache(backend="static", max_capacity=int(ids.shape[-1]) + max_new_tokens + 8)
    torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()
    with torch.no_grad():
        out = model(ids, past_key_values=cache.cache, use_cache=True)
    torch.cuda.synchronize(device)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000
    decode_start = time.perf_counter()
    token = out.logits[:, -1:, :].argmax(dim=-1)
    token_id = int(token.item())
    emitted = [token_id]
    eos = tokenizer.eos_token_id
    while len(emitted) < max_new_tokens and token_id != eos:
        with torch.no_grad():
            out = model(token, past_key_values=cache.cache, use_cache=True)
        token = out.logits[:, -1:, :].argmax(dim=-1)
        token_id = int(token.item())
        emitted.append(token_id)
    torch.cuda.synchronize(device)
    decode_ms = (time.perf_counter() - decode_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000
    return emitted, prefill_ms, decode_ms, total_ms


_common.run_vanilla_timed = run_vanilla_timed_long
sys.argv = ["scripts/benchmark_b2_lazy_lm_head.py", *sys.argv[1:]]
runpy.run_path("scripts/benchmark_b2_lazy_lm_head.py", run_name="__main__")
