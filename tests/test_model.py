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
