"""Pytest fixtures shared across test modules.

Provides session-scoped model loading to avoid redundant model weight re-loading
and eliminate GPU memory fragmentation during full test suite runs.
"""

from __future__ import annotations

import gc
import pytest
import torch

from zassd.models.loader import load_model, load_tokenizer
from zassd.models.model_adapter import ModelAdapter
from zassd.models.layer_manager import LayerManager


@pytest.fixture(scope="session")
def session_model_and_tok():
    """Load Qwen2.5-3B model once per pytest session."""
    model = load_model("Qwen/Qwen2.5-3B-Instruct", quantize=True, bits=4)
    tok = load_tokenizer("Qwen/Qwen2.5-3B-Instruct")
    yield model, tok
    del model
    del tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def session_pipeline(session_model_and_tok):
    """Provide shared adapter and layer manager wrapping the session model."""
    model, tok = session_model_and_tok
    adapter = ModelAdapter(model)
    layer_mgr = LayerManager(adapter)
    return model, tok, adapter, layer_mgr
