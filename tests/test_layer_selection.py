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

    def test_cka_method(self):
        import numpy as np
        from zassd.layer_selection.selector import LayerSelector, SelectionMethod

        # Mock CKA matrix with 8 layers
        cka = np.eye(8, dtype=np.float32)
        # Middle layers 3 and 4 are very similar (redundant)
        cka[3, 4] = cka[4, 3] = 0.98
        cka[2, 3] = cka[3, 2] = 0.95

        selector = LayerSelector(
            num_layers=8,
            method=SelectionMethod.CKA,
            cka_matrix=cka,
            num_to_skip=2,
            always_keep=[0, 7],
        )
        kept, skipped = selector.select()
        assert len(kept) + len(skipped) == 8
        assert len(skipped) == 2
        assert 0 in kept
        assert 7 in kept


class TestCKA:
    """Test CKA math and ranking functions."""

    def test_linear_cka_identity(self):
        import torch
        from zassd.layer_selection.cka import linear_cka

        X = torch.randn(100, 64)
        sim = linear_cka(X, X)
        assert abs(sim - 1.0) < 1e-4

    def test_linear_cka_orthogonal(self):
        import torch
        from zassd.layer_selection.cka import linear_cka

        # Two distinct random sets with large N
        torch.manual_seed(42)
        X = torch.randn(1000, 10)
        Y = torch.randn(1000, 10)
        sim = linear_cka(X, Y)
        assert sim < 0.1

    def test_compute_cka_matrix(self):
        import torch
        from zassd.layer_selection.cka import compute_cka_matrix

        activations = {
            0: torch.randn(50, 32),
            1: torch.randn(50, 32),
            2: torch.randn(50, 32),
        }
        matrix = compute_cka_matrix(activations)
        assert matrix.shape == (3, 3)
        assert abs(matrix[0, 0] - 1.0) < 1e-4
        assert abs(matrix[0, 1] - matrix[1, 0]) < 1e-4

    def test_rank_layers(self):
        import numpy as np
        from zassd.layer_selection.cka import rank_layers_by_redundancy, select_layers_cka

        cka = np.eye(6, dtype=np.float32)
        # Layer 2 and 3 are very similar
        cka[1, 2] = cka[2, 1] = 0.99
        cka[2, 3] = cka[3, 2] = 0.99

        ranked = rank_layers_by_redundancy(cka, always_keep=[0, 5])
        # Layer 2 should be highest or near highest redundancy
        assert ranked[0][0] in [2, 3]

        kept, skipped = select_layers_cka(cka, num_to_skip=2, always_keep=[0, 5])
        assert len(kept) == 4
        assert len(skipped) == 2
        assert 0 in kept
        assert 5 in kept


class TestEmpiricalSelection:
    """Test empirical layer selection."""

    def test_precalibrated_qwen(self):
        from zassd.layer_selection.empirical import select_empirical_layers

        kept, skipped = select_empirical_layers(num_layers=36, num_to_skip=6, model_name="Qwen2.5-3B")
        assert len(skipped) == 6
        assert len(kept) == 30
        assert 0 in kept
        assert 35 in kept
        assert skipped == [4, 5, 19, 20, 22, 23]

    def test_selector_empirical_method(self):
        from zassd.layer_selection.selector import LayerSelector, SelectionMethod

        selector = LayerSelector(
            num_layers=36,
            method=SelectionMethod.EMPIRICAL,
            num_to_skip=6,
            model_name="Qwen2.5-3B",
        )
        kept, skipped = selector.select()
        assert len(skipped) == 6
        assert 0 in kept
        assert 35 in kept

    def test_fallback_middle(self):
        from zassd.layer_selection.empirical import select_empirical_layers

        kept, skipped = select_empirical_layers(num_layers=20, num_to_skip=4, model_name="UnknownArchitecture")
        assert len(skipped) == 4
        assert 0 in kept
        assert 19 in kept

