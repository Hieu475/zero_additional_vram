"""Tests for cache management."""

from __future__ import annotations

import pytest

from zassd.cache.draft_state import DraftState


class TestDraftState:
    """Test draft state management."""

    def test_initial_state(self):
        state = DraftState()
        assert state.draft_tokens == []
        assert state.draft_time_ms == 0.0

    def test_clear(self):
        state = DraftState()
        state.draft_tokens = [1, 2, 3]
        state.draft_time_ms = 10.0
        state.clear()
        assert state.draft_tokens == []
        assert state.draft_time_ms == 0.0
