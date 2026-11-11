"""Pure unit checks for the opt-in backend loader boundary."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from zassd.backends import load_backend


def test_unknown_backend_fails_before_model_or_tokenizer_load(monkeypatch):
    from zassd.backends import base

    def fail_if_called(*args, **kwargs):
        raise AssertionError("model/tokenizer loader must not run for an unknown backend")

    monkeypatch.setattr(base.AutoTokenizer, "from_pretrained", fail_if_called)
    with pytest.raises(ValueError, match="unknown backend"):
        load_backend("not_a_backend", "model", "revision")
