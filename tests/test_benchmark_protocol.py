"""Regression checks for the paired benchmark protocol and controls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scripts.run_strong_baseline_comparison import (
    METHODS,
    prompt_level_means,
    run_vanilla_timed,
    summarize,
    tokenize_prompt,
    validate_complete_pairs,
)
from zassd.baselines.prompt_lookup import force_candidate_miss
from zassd.baselines.prompt_lookup import prompt_lookup_generate
from zassd.decoding.ngram import NgramMatch


class TinyTokenizer:
    eos_token_id = 2

    def __call__(self, text, return_tensors=None):
        return {"input_ids": torch.tensor([[1, 3]])}

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(map(str, token_ids))


class TinyModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        token = 7 if self.calls == 0 else 2
        logits = torch.zeros((1, input_ids.shape[1], 10))
        logits[:, -1, token] = 1.0
        self.calls += 1
        return SimpleNamespace(logits=logits)


def _complete_rows() -> list[dict]:
    rows = []
    for sample_id in ("p0", "p1"):
        for repeat in (0, 1):
            for method in METHODS:
                rows.append({
                    "benchmark": "toy", "id": sample_id, "repeat": repeat,
                    "order_seed": 42, "method": method,
                    "input_token_ids": [1, 3], "tps": 10.0,
                    "speedup_vs_vanilla": 1.0 if method == "vanilla" else 0.9,
                    "exact_token_match": True,
                })
    return rows


def test_methods_use_one_shared_input_tokenization():
    tokenizer = TinyTokenizer()
    ids = tokenize_prompt(tokenizer, "same prompt")
    rows = _complete_rows()
    for row in rows:
        row["input_token_ids"] = ids[0].tolist()
    validate_complete_pairs(rows)
    assert ids.tolist() == [[1, 3]]


def test_pair_validation_requires_all_methods_once_and_same_input():
    rows = _complete_rows()
    validate_complete_pairs(rows)
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        validate_complete_pairs(rows[:-1])
    rows[-1]["input_token_ids"] = [99]
    with pytest.raises(ValueError, match="different input token IDs"):
        validate_complete_pairs(rows)


def test_forced_miss_never_returns_draft_candidates():
    match = NgramMatch([8, 9], "prompt", 3, 2, 0.9)
    forced = force_candidate_miss(match)
    assert forced.candidates == []
    assert forced.source == "forced_miss"
    assert forced.confidence == 0.0


def test_pld_forced_miss_accepts_no_draft_tokens(monkeypatch):
    import zassd.baselines.prompt_lookup as prompt_lookup
    import zassd.cache.kv_cache as kv_cache

    class TinyKVCache:
        def __init__(self, backend="static"):
            self.cache = object()

    class SequenceModel:
        def __init__(self):
            self.next_tokens = [4, 5, 2]

        def __call__(self, input_ids, past_key_values=None, use_cache=True):
            token = self.next_tokens.pop(0)
            logits = torch.zeros((1, input_ids.shape[1], 8))
            logits[:, -1, token] = 1.0
            return SimpleNamespace(logits=logits)

    monkeypatch.setattr(kv_cache, "TargetKVCache", TinyKVCache)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_args, **_kwargs: 0)
    _, metrics = prompt_lookup_generate(
        SequenceModel(), TinyTokenizer(), torch.tensor([[1, 3]]),
        k=4, max_new_tokens=8, device="cpu", force_miss=True,
    )
    assert metrics.total_draft_tokens == 0
    assert metrics.total_accepted_tokens == 0
    assert metrics.pld_cycles == 0
    assert metrics.generated_token_ids == [4, 5, 2]


@pytest.mark.parametrize(
    ("max_new_tokens", "expected_ids", "expected_calls"),
    [(1, [7], 1), (5, [7, 2], 2)],
)
def test_vanilla_control_preserves_token_limit_and_eos(
    monkeypatch, max_new_tokens, expected_ids, expected_calls
):
    import zassd.cache.kv_cache as kv_cache

    class TinyKVCache:
        def __init__(self, backend="static"):
            self.cache = object()

    monkeypatch.setattr(kv_cache, "TargetKVCache", TinyKVCache)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    ids, _prefill_ms, _decode_ms, _total_ms = run_vanilla_timed(
        TinyModel(), TinyTokenizer(), torch.tensor([[1, 3]]), max_new_tokens, "cpu"
    )
    assert ids == expected_ids


def test_bootstrap_uses_prompt_as_sampling_unit():
    rows = _complete_rows()
    pld_rows = [row for row in rows if row["method"] == "pld"]
    means = prompt_level_means(pld_rows, "speedup_vs_vanilla")
    assert means == {("toy", "p0"): pytest.approx(0.9),
                     ("toy", "p1"): pytest.approx(0.9)}
    summary = summarize(rows)
    assert summary["pld"]["n"] == 4
    assert summary["pld"]["n_prompts"] == 2
    assert summary["pld"]["bootstrap_unit"].startswith("prompt")


def test_paired_ratios_average_runs_within_prompt():
    from scripts.run_exactness_latency_tradeoff import METHODS as TRADEOFF_METHODS
    from scripts.run_exactness_latency_tradeoff import summarize as summarize_tradeoff

    rows = []
    for sample_id in ("p0", "p1"):
        for repeat in range(2):
            for seed in (42, 43):
                for method in TRADEOFF_METHODS:
                    tps = 10.0 if method == "vanilla" else (8.0 if method == "pld_batched" else 9.0)
                    rows.append({
                        "benchmark": "toy", "id": sample_id, "repeat": repeat,
                        "order_seed": seed, "method": method,
                        "exact_token_match": method != "pld_batched",
                        "end_to_end_tps": tps, "decode_tps": tps - 1,
                    })
    result = summarize_tradeoff(rows)
    assert result["n_mismatch_prompts"] == 2
    assert result["methods"]["pld_batched"]["exact_match_pct"] == 0.0
    assert result["methods"]["pld_batched"]["end_to_end_speedup_mean"] == pytest.approx(0.8)
    assert result["methods"]["pld_sequential"]["exact_match_pct"] == 100.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the benchmark GPU")
def test_vanilla_via_speculative_core_matches_direct_vanilla(session_pipeline):
    from zassd.decoding.speculative import self_speculative_generate

    model, tokenizer, _adapter, manager = session_pipeline
    device = "cuda:0"
    prompt = "Reply with one short greeting."
    direct_ids, *_ = run_vanilla_timed(model, tokenizer, prompt, 12, device)
    _, metrics = self_speculative_generate(
        model, tokenizer, manager, [], prompt,
        k=0, draft_mode="layer_skip", max_new_tokens=12, device=device,
    )
    assert metrics.generated_token_ids == direct_ids
