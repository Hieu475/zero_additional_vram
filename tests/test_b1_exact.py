"""Branch tests for the B1 singleton target schedule."""

from types import SimpleNamespace

import pytest
import torch

from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding import b1_exact


class ToyTokenizer:
    eos_token_id = 5

    def decode(self, ids, skip_special_tokens=True):
        return ",".join(map(str, ids))


class ToyModel:
    """One-token transition: next ID is input ID plus one, modulo six."""

    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, *, past_key_values, use_cache):
        assert use_cache
        self.calls.append((input_ids.shape[1], past_key_values.get_seq_length()))
        ids = input_ids[0].tolist()
        keys = input_ids.float().reshape(1, 1, -1, 1)
        past_key_values.update(keys, keys.clone(), 0)
        logits = torch.full((1, len(ids), 6), -100.0)
        for row, token in enumerate(ids):
            logits[0, row, (token + 1) % 6] = 100.0
        return SimpleNamespace(logits=logits)


def reference(model, prompt, budget, eos):
    cache = TargetKVCache(backend="static", max_capacity=32)
    token = model(prompt, past_key_values=cache.cache, use_cache=True).logits[:, -1:, :].argmax(-1)
    output = []
    while len(output) < budget:
        value = int(token.item())
        output.append(value)
        if value == eos or len(output) == budget:
            break
        token = b1_exact.one_token_transition(model, token, cache)
    return output


@pytest.mark.parametrize("proposals", [
    [], [4, 5], [3, 5], [3, 4, 5], [3, 4, 0],
])
def test_b1_matches_reference_for_accept_reject_and_eos(monkeypatch, proposals):
    monkeypatch.setattr(b1_exact, "find_candidate_tokens",
                        lambda *_args, **_kwargs: proposals)
    prompt = torch.tensor([[0, 1]])
    expected = reference(ToyModel(), prompt, 8, ToyTokenizer.eos_token_id)
    model = ToyModel()
    _, metrics = b1_exact.b1_exact_generate(
        model, ToyTokenizer(), prompt, k=4, max_new_tokens=8,
        device="cpu", kv_cache_max_capacity=32, trace=True,
    )
    assert metrics.generated_token_ids == expected
    assert metrics.target_forwards == max(0, len(expected) - 1)
    assert all(shape == 1 for shape, _ in model.calls[1:])
    for cycle in metrics.cycle_trace:
        expected_len = prompt.shape[1] + cycle["output_start"] + len(cycle["emitted_tokens"])
        if cycle["next_pending"] is None:
            expected_len -= 1  # EOS/budget token is emitted but never consumed.
        assert cycle["cache_len_after"] == expected_len


def test_b1_respects_budget_and_rejects_invalid_inputs(monkeypatch):
    monkeypatch.setattr(b1_exact, "find_candidate_tokens",
                        lambda *_args, **_kwargs: [3, 4, 5])
    prompt = torch.tensor([[0, 1]])
    _, metrics = b1_exact.b1_exact_generate(
        ToyModel(), ToyTokenizer(), prompt, k=4, max_new_tokens=2,
        device="cpu", kv_cache_max_capacity=32,
    )
    assert metrics.generated_token_ids == [2, 3]
    assert metrics.target_forwards == 1
    with pytest.raises(ValueError, match="capacity"):
        b1_exact.b1_exact_generate(ToyModel(), ToyTokenizer(), prompt,
                                    max_new_tokens=31, device="cpu",
                                    kv_cache_max_capacity=32)
    with pytest.raises(ValueError, match="single token row"):
        b1_exact.one_token_transition(ToyModel(), torch.tensor([[1, 2]]),
                                      TargetKVCache(backend="static", max_capacity=32))


def test_b1_observes_every_canonical_state(monkeypatch):
    monkeypatch.setattr(b1_exact, "find_candidate_tokens",
                        lambda *_args, **_kwargs: [3, 4, 5])
    states = []

    def observe(consumed, cache):
        layer = cache.cache.layers[0]
        states.append((consumed, cache.get_seq_length(),
                       layer.keys[:, :, :layer.seq_len, :].clone()))

    _, metrics = b1_exact.b1_exact_generate(
        ToyModel(), ToyTokenizer(), torch.tensor([[0, 1]]),
        k=4, max_new_tokens=8, device="cpu", kv_cache_max_capacity=32,
        state_observer=observe,
    )
    assert metrics.generated_token_ids == [2, 3, 4, 5]
    assert [count for count, _, _ in states] == [0, 1, 2, 3]
    for consumed, length, keys in states:
        assert length == 2 + consumed
        assert keys.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0][:length]
