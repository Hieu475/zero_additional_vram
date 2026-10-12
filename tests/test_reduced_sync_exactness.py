"""Reference versus conservative reduced-sync output/cache invariants."""

from __future__ import annotations

import pytest
import torch

from zassd.decoding.speculative import self_speculative_generate


SKIP_CKA90 = [4, 5, 6, 7]


def _run(model, tokenizer, layer_mgr, *, k: int, reduced_sync: bool,
         max_new_tokens: int = 8) -> dict:
    _, metrics = self_speculative_generate(
        model=model,
        tokenizer=tokenizer,
        layer_mgr=layer_mgr,
        skip_indices=SKIP_CKA90,
        prompt="The quick brown fox jumps over the",
        k=k,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        device="cuda:0",
        kv_cache_backend="static",
        kv_cache_max_capacity=128,
        config_name="cka_90",
        diagnostic_trace=True,
        diagnostic_hash_kv=True,
        reduced_sync=reduced_sync,
    )
    return {
        "tokens": metrics.generated_token_ids,
        "cycles": metrics.per_iteration_stats,
        "mode": metrics.sync_mode,
    }


@pytest.fixture(scope="module")
def pipeline(session_pipeline):
    if not torch.cuda.is_available():
        pytest.skip("reduced-sync integration requires CUDA")
    model, tokenizer, _, layer_mgr = session_pipeline
    return model, tokenizer, layer_mgr


@pytest.mark.parametrize("k", [0, 1, 2, 4])
def test_reference_and_reduced_sync_have_same_tokens_and_kv(pipeline, k):
    model, tokenizer, layer_mgr = pipeline
    reference = _run(model, tokenizer, layer_mgr, k=k, reduced_sync=False)
    reduced = _run(model, tokenizer, layer_mgr, k=k, reduced_sync=True)
    assert reduced["tokens"] == reference["tokens"]
    assert reduced["mode"] == "reduced_sync_conservative"
    assert reference["mode"] == "reference"
    assert [c.get("cache_len_after_commit") for c in reduced["cycles"]] == [
        c.get("cache_len_after_commit") for c in reference["cycles"]
    ]
    assert [c.get("cache_digest_after") for c in reduced["cycles"]] == [
        c.get("cache_digest_after") for c in reference["cycles"]
    ]


@pytest.mark.parametrize("eos_position", [0, 2])
def test_reference_and_reduced_sync_preserve_eos_stopping(pipeline, eos_position):
    model, tokenizer, layer_mgr = pipeline
    original_eos = tokenizer.eos_token_id
    try:
        tokenizer.eos_token_id = original_eos
        baseline = _run(model, tokenizer, layer_mgr, k=0, reduced_sync=False,
                        max_new_tokens=6)["tokens"]
        assert len(baseline) > eos_position
        tokenizer.eos_token_id = baseline[eos_position]
        reference = _run(model, tokenizer, layer_mgr, k=4, reduced_sync=False,
                         max_new_tokens=6)
        reduced = _run(model, tokenizer, layer_mgr, k=4, reduced_sync=True,
                       max_new_tokens=6)
        expected = baseline[:eos_position + 1]
        assert reference["tokens"] == expected
        assert reduced["tokens"] == expected
        assert [c.get("cache_digest_after") for c in reduced["cycles"]] == [
            c.get("cache_digest_after") for c in reference["cycles"]
        ]
    finally:
        tokenizer.eos_token_id = original_eos


@pytest.mark.parametrize("token_budget", [1, 3])
def test_reference_and_reduced_sync_respect_short_token_budget(pipeline, token_budget):
    model, tokenizer, layer_mgr = pipeline
    reference = _run(model, tokenizer, layer_mgr, k=4, reduced_sync=False,
                     max_new_tokens=token_budget)
    reduced = _run(model, tokenizer, layer_mgr, k=4, reduced_sync=True,
                   max_new_tokens=token_budget)
    assert len(reference["tokens"]) <= token_budget
    assert reduced["tokens"] == reference["tokens"]
