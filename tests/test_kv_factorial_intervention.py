"""Offline invariants for the KV × verification factorial runner."""

import torch

from scripts.run_kv_factorial_intervention import (
    position_ids_for,
    restore_static_cache,
    selected_next_row,
    snapshot_cache,
    snapshot_fingerprint,
)


def _snapshot():
    return [(torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4),
             torch.ones((1, 1, 3, 4), dtype=torch.float32), 3)]


def test_factorial_conditions_get_independent_cache_clones():
    source = _snapshot()
    source_hash = snapshot_fingerprint(source)
    active_batch = restore_static_cache(source, capacity=16, device="cpu")
    active_sequential = restore_static_cache(source, capacity=16, device="cpu")

    assert active_batch.get_seq_length() == active_sequential.get_seq_length() == 3
    assert active_batch.cache.layers[0].keys.data_ptr() != active_sequential.cache.layers[0].keys.data_ptr()
    active_batch.cache.layers[0].keys[0, 0, 0, 0] = -99
    assert active_sequential.cache.layers[0].keys[0, 0, 0, 0] == source[0][0][0, 0, 0, 0]
    assert snapshot_fingerprint(source) == source_hash


def test_position_ids_preserve_logical_prefix_for_k1_k2_k4():
    for k in (1, 2, 4):
        positions = position_ids_for(prefix_length=384, input_length=k + 1)
        assert positions == list(range(384, 384 + k + 1))


def test_logit_row_and_rejection_index_select_target_prediction():
    assert selected_next_row({"verification_mode": "vanilla_fallback", "candidate_tokens": [10]}) == 0
    assert selected_next_row({"verification_mode": "batched", "candidate_tokens": [10, 11, 12, 13],
                              "rejected_at": 1}) == 1
    assert selected_next_row({"verification_mode": "batched", "candidate_tokens": [10, 11, 12, 13],
                              "rejected_at": None}) == 3
