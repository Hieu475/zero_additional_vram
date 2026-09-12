"""Tests for layer selection strategies."""

from __future__ import annotations

import pytest


class TestRandomSelection:
    """Test random layer selection."""

    def test_basic_selection(self):
        from zassd.layer_selection.random import select_random_layers
        kept, skipped = select_random_layers(
            num_layers=32, skip_ratio=0.25, seed=42
        )
        assert len(kept) + len(skipped) == 32
        assert 0 in kept  # First layer kept
        assert 31 in kept  # Last layer kept

    def test_skip_ratio(self):
        from zassd.layer_selection.random import select_random_layers
        kept, skipped = select_random_layers(
            num_layers=32, skip_ratio=0.5, seed=42
        )
        # Should skip approximately 50% of eligible layers
        assert len(skipped) > 0
        assert len(kept) > 0

    def test_reproducibility(self):
        from zassd.layer_selection.random import select_random_layers
        kept1, skipped1 = select_random_layers(
            num_layers=32, skip_ratio=0.25, seed=42
        )
        kept2, skipped2 = select_random_layers(
            num_layers=32, skip_ratio=0.25, seed=42
        )
        assert kept1 == kept2
        assert skipped1 == skipped2


class TestStaticSelection:
    """Test static layer selection."""

    def test_even_layers(self):
        from zassd.layer_selection.static import select_even_layers
        kept, skipped = select_even_layers(num_layers=8)
        assert kept == [0, 2, 4, 6]
        assert skipped == [1, 3, 5, 7]

    def test_middle_skip(self):
        from zassd.layer_selection.static import select_middle_skip
        kept, skipped = select_middle_skip(
            num_layers=32, skip_start=12, skip_end=24
        )
        assert all(i in skipped for i in range(12, 24))
        assert 0 in kept
        assert 31 in kept


class TestLayerSelector:
    """Test unified selector."""

    def test_random_method(self):
        from zassd.layer_selection.selector import LayerSelector, SelectionMethod
        selector = LayerSelector(
            num_layers=32,
            method=SelectionMethod.RANDOM,
            skip_ratio=0.25,
            seed=42,
        )
        kept, skipped = selector.select()
        assert len(kept) + len(skipped) == 32
        assert selector.skip_ratio > 0
