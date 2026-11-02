import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
bench = pytest.importorskip("benchmark_bnb_nf4_activationvec_fallback_closed_loop")


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(2, 2)
        self.b = torch.nn.Linear(2, 2)


def test_switch_swaps_only_patched_modules_and_restores(monkeypatch):
    model = Tiny()
    original_a = model.a.forward
    original_b = model.b.forward

    @contextmanager
    def fake_ctx(m):
        saved = m.a.forward
        m.a.forward = MethodType(lambda self, x: x * 0 + 7, m.a)
        try:
            yield
        finally:
            m.a.forward = saved

    monkeypatch.setattr(bench, "block_ctx", fake_ctx)
    x = torch.ones(1, 2)
    with bench.SwitchableSchedule(model) as schedule:
        assert schedule.mode == "patched" and len(schedule._entries) == 1
        assert torch.equal(model.a(x), torch.full((1, 2), 7.0))
        schedule.use("plain")
        assert torch.equal(model.a(x), original_a(x))
        schedule.use("plain")
        schedule.use("patched")
        assert torch.equal(model.a(x), torch.full((1, 2), 7.0))
        assert model.b.forward == original_b
        schedule.use("plain")
    assert torch.equal(model.a(x), original_a(x))
    assert model.a.forward == original_a
