"""Test 3 — End-to-End Speculative Decoding Integration Test.

Tests end-to-end generation, verification, and latency decomposition:
T_total = T_draft + T_verify + T_cache + T_controller + T_other
across K in {1, 2, 4} under the 6GB VRAM budget.
"""

from __future__ import annotations

import pytest
import torch

from zassd.decoding.speculative import self_speculative_generate
from zassd.decoding.vanilla import vanilla_generate
from zassd.models.layer_manager import LayerManager
from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter


class TestEndToEndSpeculative:
    """End-to-end pipeline validation."""

    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_speculative_generation_k(self, session_pipeline, k):
        """Test self-speculative generation with different draft lengths K."""
        model, tok, _, layer_mgr = session_pipeline
        skip_indices = [3, 5, 7, 9, 11, 13, 16, 18, 21]
        prompt = "Explain why deep neural networks require regularization during training."

        text, metrics = self_speculative_generate(
            model=model,
            tokenizer=tok,
            layer_mgr=layer_mgr,
            skip_indices=skip_indices,
            prompt=prompt,
            k=k,
            max_new_tokens=32,
            temperature=0.0,
        )

        assert len(text) > 0, "Generated text should not be empty"
        assert metrics.total_tokens > 0, "Generated tokens should be positive"
        assert metrics.tokens_per_second > 0, "Throughput should be positive"
        assert metrics.num_verification_cycles > 0, "Should run at least 1 verification cycle"
        assert metrics.tokens_per_step >= 1.0, "Tokens per step should be >= 1.0"
        assert metrics.peak_vram_mb < 5500.0, f"VRAM exceeded budget: {metrics.peak_vram_mb} MB"

        # Check latency decomposition
        accounted_time = (
            metrics.draft_time_s
            + metrics.verify_time_s
            + metrics.cache_time_s
            + metrics.controller_time_s
            + metrics.other_time_s
        )
        assert accounted_time >= metrics.total_time_s * 0.95, "Latency decomposition does not account for total time"
        assert len(metrics.per_iteration_stats) == metrics.num_verification_cycles
