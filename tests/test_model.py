"""Tests for model loading."""

from __future__ import annotations

import pytest


class TestModelLoader:
    """Test model loading utilities."""

    def test_quantization_config_4bit(self):
        from zassd.models.loader import get_quantization_config
        config = get_quantization_config(bits=4)
        assert config.load_in_4bit is True

    def test_quantization_config_8bit(self):
        from zassd.models.loader import get_quantization_config
        config = get_quantization_config(bits=8)
        assert config.load_in_8bit is True

    def test_quantization_config_invalid(self):
        from zassd.models.loader import get_quantization_config
        with pytest.raises(ValueError):
            get_quantization_config(bits=3)


class TestLayerManager:
    """Test logical layer skipping via LayerManager."""

    def test_logical_skipping_preserves_layer_idx(self):
        import torch
        import torch.nn as nn
        from zassd.models.layer_manager import LayerManager

        class MockDecoderLayer(nn.Module):
            def __init__(self, idx: int):
                super().__init__()
                self.layer_idx = idx

            def forward(self, hidden_states, *args, **kwargs):
                return hidden_states * 2

        class MockAdapter:
            def __init__(self, layers):
                self.layers = layers
                self.num_layers = len(layers)

            def get_layers(self):
                return self.layers

        mock_layers = [MockDecoderLayer(i) for i in range(4)]
        adapter = MockAdapter(mock_layers)
        layer_mgr = LayerManager(adapter)

        # Skip layers 1 and 3
        with layer_mgr.skip_layers([1, 3]):
            assert layer_mgr.is_skipped(1)
            assert not layer_mgr.is_skipped(0)
            # Crucial: layer_idx attribute MUST remain identical
            for i, l in enumerate(mock_layers):
                assert l.layer_idx == i

            x = torch.tensor([5.0])
            # Skipped layer returns identity
            assert mock_layers[1](x).item() == 5.0
            # Active layer returns hidden_states * 2
            assert mock_layers[0](x).item() == 10.0

        # After exiting context, original forwards are restored
        assert mock_layers[1](x).item() == 10.0
        assert len(layer_mgr.active_skip_indices) == 0

    def test_logical_skipping_tuple_return(self):
        import torch
        import torch.nn as nn
        from zassd.models.layer_manager import LayerManager

        class MockTupleLayer(nn.Module):
            def __init__(self, idx: int):
                super().__init__()
                self.layer_idx = idx

            def forward(self, hidden_states, *args, **kwargs):
                return (hidden_states + 1, "attn_weights")

        class MockAdapter:
            def __init__(self, layers):
                self.layers = layers
                self.num_layers = len(layers)

            def get_layers(self):
                return self.layers

        mock_layers = [MockTupleLayer(0)]
        adapter = MockAdapter(mock_layers)
        layer_mgr = LayerManager(adapter)

        with layer_mgr.skip_layers([0]):
            x = (torch.tensor([3.0]), "dummy")
            out = mock_layers[0](x)
            assert isinstance(out, tuple)
            assert out[0].item() == 3.0
            assert out[1] is None

