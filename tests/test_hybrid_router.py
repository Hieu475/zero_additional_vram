"""Tests for HybridDraftRouter (Direction 1) — CPU-only, no GPU required."""
import sys
sys.path.insert(0, "src")

from zassd.routing.hybrid_router import HybridDraftRouter


def _hist(seq):
    return list(seq)


def test_pld_hit_routes_to_pld_when_strong():
    r = HybridDraftRouter(cost_model=None)
    # history with repeating pattern: last 3 tokens appeared before with followers
    hist = [10, 20, 30, 40, 10, 20, 30, 50, 10, 20, 30]
    dec = r.select_source(hist, 40, step_k=2, entropy=1.0)
    assert dec.source == "pld", f"expected pld, got {dec.source} ({dec.reason})"
    assert len(dec.candidates) > 0
    assert dec.ngram_match_len == 3


def test_pld_miss_skips_to_vanilla_when_ls_weak():
    # No cost model (default LS 19.7ms/token) -> E[TPS]_LS < vanilla -> skip
    r = HybridDraftRouter(cost_model=None)
    hist = [101, 102, 103, 104, 105, 106, 107]  # no repetition
    dec = r.select_source(hist, 999, step_k=2, entropy=1.0)
    assert dec.source == 'vanilla', f'expected vanilla-skip, got {dec.source}'
    assert dec.ngram_match_len == 0


def test_pld_miss_routes_to_layer_skip_when_ls_strong():
    # Cost model + low entropy -> LS beats vanilla -> use LS
    from zassd.profiling.action_cost_model import MeasuredActionCostModel
    cm = MeasuredActionCostModel.for_model('qwen25_3b')
    r = HybridDraftRouter(cost_model=cm)
    hist = [101, 102, 103, 104, 105, 106, 107]
    dec = r.select_source(hist, 999, step_k=1, entropy=0.2, config_name='cka_75')
    assert dec.source == 'layer_skip', f'expected layer_skip, got {dec.source} ({dec.reason})'


def test_panic_entropy_forces_vanilla():
    r = HybridDraftRouter(cost_model=None)
    hist = [1, 2, 3, 4, 5]
    dec = r.select_source(hist, 999, step_k=2, entropy=5.0)
    assert dec.source == "vanilla"


def test_ema_update_tracks_sources():
    r = HybridDraftRouter(cost_model=None)
    r.update("pld", accepted=2, proposed=2, ngram_match_len=3)
    r.update("layer_skip", accepted=1, proposed=2)
    assert r.stats.pld_cycles == 1
    assert r.stats.layer_skip_cycles == 1
    assert r.stats.pld_strong_hits == 1
    assert r.stats.alpha_pld_ema > r.stats.alpha_ls_ema


def test_router_with_real_cost_model():
    from zassd.profiling.action_cost_model import MeasuredActionCostModel
    cm = MeasuredActionCostModel.for_model("qwen25_3b")
    r = HybridDraftRouter(cost_model=cm)
    hist = [5, 6, 7, 8, 5, 6, 7, 9, 5, 6, 7]
    dec = r.select_source(hist, 8, step_k=2, entropy=0.3, config_name="cka_75")
    assert dec.source in ("pld", "layer_skip", "vanilla")
    assert dec.expected_tps_ls > 0


def _stub_cost_model(verify_ms, draft_ms, ls_alpha, baseline=41.5):
    from types import SimpleNamespace
    m = SimpleNamespace(baseline_tps=baseline)
    m.predict_verify_ms = lambda k: verify_ms(k) if callable(verify_ms) else verify_ms
    m.predict_draft_ms = lambda cfg, k: draft_ms
    m.predict_acceptance_rate = lambda cfg, k, entropy=1.0: ls_alpha
    return m


def test_oracle_differs_from_router_when_pld_below_vanilla():
    # Both drafts predicted far below vanilla: router's PLD-vs-LS branch
    # still returns PLD (it never compares PLD against vanilla), while the
    # bare-argmax oracle correctly returns vanilla. Proves the two decision
    # rules are distinct functions (the 100% GSM8K agreement is empirical,
    # not vacuous).
    cm = _stub_cost_model(verify_ms=lambda k: 25.0 + 475.0 * max(0, k - 1),
                          draft_ms=20.0, ls_alpha=0.8)
    r = HybridDraftRouter(cost_model=cm)
    hist = [10, 20, 30, 40, 10, 20, 30, 50, 10, 20, 30]
    dec = r.select_source(hist, 40, step_k=2, entropy=1.0)
    assert dec.source == "pld", f"router should take PLD branch, got {dec.source}"
    o_src, o_tps = r.oracle_select(hist, 40, step_k=2, entropy=1.0)
    assert o_src == "vanilla", f"oracle should skip, got {o_src}"
    assert o_tps["vanilla"] > o_tps["pld"]


def test_oracle_differs_from_router_on_panic():
    # Panic entropy + no candidates + strong LS: router panics to vanilla,
    # oracle takes layer_skip. Distinct rules again.
    cm = _stub_cost_model(verify_ms=25.0, draft_ms=5.0, ls_alpha=0.95)
    r = HybridDraftRouter(cost_model=cm)
    hist = [101, 102, 103, 104, 105, 106, 107]  # no repetition
    dec = r.select_source(hist, 999, step_k=1, entropy=5.0)
    assert dec.source == "vanilla"
    o_src, _ = r.oracle_select(hist, 999, step_k=1, entropy=5.0)
    assert o_src == "layer_skip", f"oracle should speculate, got {o_src}"
