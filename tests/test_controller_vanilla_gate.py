"""Tests for the HW controller VANILLA feasibility gate — CPU-only.

The gate (review item 7): if every speculative action is predicted slower
than plain autoregressive decoding, select_action must return K=0
(a plain vanilla step; speculative.py already executes draft_length=0
as a single-token vanilla fallback) instead of knowingly running at <1.0x.
"""
import sys
from types import SimpleNamespace

sys.path.insert(0, "src")

from zassd.controllers.hardware_controller import (
    HardwareAwareJointController,
    HardwareState,
)

CONFIGS = {"cka_75": list(range(9)), "cka_90": list(range(3))}
HW = HardwareState(vram_used_mb=2000.0, gpu_power_w=50.0,
                   gpu_temperature_c=55.0, power_budget_w=80.0)


def _stub_model(tps: float, baseline: float = 41.5):
    m = SimpleNamespace(
        baseline_tps=baseline,
        total_layers=36,
        kv_cache_mb_per_token=45.0,
    )
    m.resolve_kept_layers = lambda cfg: 27
    m.evaluate_action = lambda **kw: SimpleNamespace(
        expected_speedup=tps / baseline,
        total_cycle_ms=45.0,
        expected_energy_j_tok=1.6,
        expected_tokens_per_step=1.8,
        expected_tps=tps,
    )
    return m


def _ctrl(tps, **kw):
    return HardwareAwareJointController(
        candidate_layer_configs=CONFIGS,
        cost_model=_stub_model(tps),
        hardware_override=HW,
        **kw,
    )


def test_gate_triggers_when_speculation_predicted_to_lose():
    c = _ctrl(36.0)  # below 41.5 vanilla baseline
    a = c.select_action(entropy=1.0)
    assert a.draft_length == 0, f"expected VANILLA skip, got K={a.draft_length}"
    assert a.vanilla_skip is True
    assert a.config_name == "vanilla"
    assert a.skip_indices == []


def test_no_gate_when_speculation_predicted_to_win():
    c = _ctrl(45.0)  # above baseline
    a = c.select_action(entropy=1.0)
    assert a.vanilla_skip is False
    assert a.draft_length in (1, 2, 3, 4)


def test_gate_disabled_returns_speculative_action():
    c = _ctrl(36.0, enable_vanilla_skip=False)
    a = c.select_action(entropy=1.0)
    assert a.vanilla_skip is False
    assert a.draft_length > 0


def test_margin_requires_beating_vanilla_by_bar():
    c = _ctrl(43.0, vanilla_margin=1.1)  # 43 < 41.5*1.1=45.65 -> skip
    assert c.select_action(entropy=1.0).vanilla_skip is True
    c2 = _ctrl(43.0, vanilla_margin=1.0)  # 43 > 41.5 -> speculate
    assert c2.select_action(entropy=1.0).vanilla_skip is False


def test_gate_with_real_cost_model_runs():
    # smoke test: real parametric model, no crash, valid action shape
    from zassd.profiling.action_cost_model import MeasuredActionCostModel
    c = HardwareAwareJointController(
        candidate_layer_configs=CONFIGS,
        cost_model=MeasuredActionCostModel.for_model("qwen25_3b"),
        hardware_override=HW,
    )
    a = c.select_action(entropy=1.0)
    assert a.draft_length in (0, 1, 2, 3, 4)
    assert isinstance(a.vanilla_skip, bool)
