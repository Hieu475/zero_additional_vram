from __future__ import annotations

import pytest
import torch

from zassd.baselines import prompt_lookup
from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.ngram import NgramMatch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sequential PLD test requires CUDA")
@pytest.mark.parametrize("reject_position", [None, 0, 1])
def test_sequential_pld_verification_matches_vanilla(session_pipeline, monkeypatch, reject_position):
    model, tokenizer, _, _ = session_pipeline
    device = "cuda:0"
    prompt = "The capital of France is"
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    budget = 16

    cache = TargetKVCache(backend="static", max_capacity=128)
    with torch.no_grad():
        out = model(prompt_ids, past_key_values=cache.cache, use_cache=True)
    next_token = int(out.logits[:, -1, :].argmax(dim=-1).item())
    reference = [next_token]
    while len(reference) < budget and next_token != tokenizer.eos_token_id:
        token = torch.tensor([[next_token]], device=device)
        with torch.no_grad():
            out = model(token, past_key_values=cache.cache, use_cache=True)
        next_token = int(out.logits[:, -1, :].argmax(dim=-1).item())
        reference.append(next_token)

    def exact_reference_candidates(history_tokens, prompt_tokens, generated_tokens, *,
                                   ngram_size, max_candidates, include_generated):
        start = len(generated_tokens) + 1
        candidates = reference[start:start + max_candidates]
        if not candidates:
            return NgramMatch([], "forced_test_miss", 0, 0, 0.0)
        if reject_position is not None and reject_position < len(candidates):
            candidates[reject_position] = (candidates[reject_position] + 1) % len(tokenizer)
        return NgramMatch(candidates, "forced_test_match", ngram_size, 1, 1.0)

    monkeypatch.setattr(prompt_lookup, "find_candidate_match", exact_reference_candidates)
    _, metrics = prompt_lookup.prompt_lookup_generate(
        model, tokenizer, prompt_ids, k=4, max_new_tokens=budget, temperature=0.0,
        device=device, kv_cache_backend="static", candidate_source="prompt+generated",
        sequential_verify=True, kv_cache_max_capacity=128,
    )
    assert metrics.pld_cycles > 0
    assert metrics.generated_token_ids == reference
    assert all(c.get("verification_mode") == "sequential"
               for c in metrics.per_iteration_stats if int(c.get("k", 0)) > 0)
