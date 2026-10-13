from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_numerical_causality import classify
from trace_multicycle_numerical_divergence import cache_error


def test_cache_error_is_zero_for_identical_prefixes():
    keys = values = torch.ones((1, 1, 2, 2))
    snapshot = [(keys, values, 2)]
    assert cache_error(snapshot, snapshot) == (None, 0.0, 0.0, 0)


def test_cache_error_reports_layer_and_length_mismatch():
    base = [(torch.zeros((1, 1, 2, 2)), torch.zeros((1, 1, 2, 2)), 2)]
    changed = [(torch.ones((1, 1, 2, 2)), torch.zeros((1, 1, 2, 2)), 2)]
    layer, abs_error, rel_error, length_mismatches = cache_error(base, changed)
    assert (layer, abs_error, length_mismatches) == (0, 1.0, 0)
    assert rel_error == 1.0
    shorter = [(base[0][0], base[0][1], 1)]
    assert cache_error(base, shorter)[3] == 1


def test_canonical_kv_substitution_classifies_restored_argmax():
    prompt = {
        "prompt_id": "cache-case", "first_kv_divergence_cycle": 2,
        "first_logit_divergence_cycle": 2, "first_token_divergence_index": 4,
        "first_token_divergence_cycle": 2,
        "first_token_divergence_logit_row": 0,
        "vanilla_top1_margin_at_divergence": 0.1,
        "vanilla_token_ids": [10, 11, 12, 13, 14],
        "cycles": [{"cycle_index": 2, "verification_mode": "vanilla_fallback",
                    "prefix_matches_vanilla": True, "observed_batch_token_ids": [99],
                    "canonical_replay_token_ids": [14],
                    "first_kv_different_layer": 1, "kv_max_abs_error": 0.2,
                    "logits_max_abs_error": 0.3}],
    }
    assert classify(prompt)["classification"] == "canonical_kv_substitution_restores_vanilla_argmax"


def test_canonical_kv_does_not_overclaim_batch_execution_case():
    prompt = {
        "prompt_id": "batch-case", "first_kv_divergence_cycle": 2,
        "first_logit_divergence_cycle": 2, "first_token_divergence_index": 4,
        "first_token_divergence_cycle": 2,
        "first_token_divergence_logit_row": 0,
        "vanilla_top1_margin_at_divergence": 0.1,
        "vanilla_token_ids": [10, 11, 12, 13, 14],
        "cycles": [{"cycle_index": 2, "verification_mode": "batched",
                    "prefix_matches_vanilla": True, "observed_batch_token_ids": [99],
                    "canonical_replay_token_ids": [98],
                    "first_kv_different_layer": 1, "kv_max_abs_error": 0.2,
                    "logits_max_abs_error": 0.3}],
    }
    assert classify(prompt)["classification"] == "batch_execution_difference_persists_with_canonical_kv"
