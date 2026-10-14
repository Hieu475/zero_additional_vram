"""Offline checks for replay trajectory normalization and first-difference logic."""

from scripts.replay_benchmark_trace import first_cycle_difference, normalize_numerical_trace


def test_first_cycle_difference_reports_first_state_field():
    base = [{
        "cycle_index": 0,
        "prefix_token_ids_hash": "same",
        "draft_candidate_ids": [10, 11],
        "accepted_prefix_length": 1,
        "emitted_token_ids": [7, 10],
        "kv_length_before": 20,
        "kv_length_after": 22,
        "remaining_budget_before": 32,
        "eos_reached": False,
    }]
    changed = [dict(base[0], accepted_prefix_length=0)]
    assert first_cycle_difference(base, changed) == {
        "cycle_index": 0,
        "field": "accepted_prefix_length",
        "left": 1,
        "right": 0,
    }


def test_first_cycle_difference_detects_cycle_count_change():
    assert first_cycle_difference([], [{"cycle_index": 0}]) == {
        "cycle_index": 0,
        "field": "cycle_count",
        "left": 0,
        "right": 1,
    }


def test_normalize_numerical_trace_uses_logical_cache_lengths_and_drafts():
    raw = [{
        "cycle_index": 0,
        "verification_mode": "batched",
        "candidate_tokens": [5, 6, 7],
        "accepted_count": 1,
        "emitted_tokens": [5, 6],
        "prefix_tokens": [1, 2, 3],
        "cache_before": [(None, None, 3)],
        "cache_after": [(None, None, 5)],
    }]
    row = normalize_numerical_trace(raw, eos_id=99, max_new_tokens=32)[0]
    assert row["draft_candidate_ids"] == [6, 7]
    assert row["accepted_prefix_length"] == 1
    assert row["kv_length_before"] == 3
    assert row["kv_length_after"] == 5
    assert row["remaining_budget_before"] == 32
    assert row["eos_reached"] is False
