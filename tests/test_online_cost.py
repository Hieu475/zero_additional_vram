"""Tests for online-calibrated cost/acceptance estimation (CPU-only)."""

from __future__ import annotations

import pytest

from zassd.profiling.online_cost import (
    AcceptanceFeatures,
    CalibratedAcceptancePredictor,
    OnlineLatencyTracker,
    RegretDecomposition,
)


class TestOnlineLatencyTracker:
    def test_no_obs_returns_fallback(self):
        t = OnlineLatencyTracker()
        p = t.predict_draft("cka_75", 1, 19.72)
        assert p.value_ms == 19.72 and p.n_obs == 0

    def test_ema_converges_to_observations(self):
        t = OnlineLatencyTracker(ema_beta=0.5, min_samples=2)
        for _ in range(20):
            t.update_draft("cka_75", 1, 10.0)
        p = t.predict_draft("cka_75", 1, 19.72)
        assert abs(p.value_ms - 10.0) < 1.0
        assert p.n_obs == 20

    def test_blend_moves_from_fallback_to_ema(self):
        t = OnlineLatencyTracker(min_samples=1000)
        t.update_verify(2, 10.0)
        p = t.predict_verify(2, 30.0)
        # With 1 obs vs 1000 min_samples, weight ~0 -> near fallback.
        assert abs(p.value_ms - 30.0) < 1.0

    def test_vanilla_tracking(self):
        t = OnlineLatencyTracker()
        t.update_vanilla(24.0)
        t.update_vanilla(26.0)
        assert t.n_vanilla_obs == 2
        p = t.predict_vanilla(30.0)
        assert 20.0 < p.value_ms < 30.0


class TestCalibratedAcceptancePredictor:
    def test_predictions_bounded(self):
        pr = CalibratedAcceptancePredictor()
        for h in (0.1, 1.0, 4.0):
            for m in (None, 0.0, 2.5):
                p = pr.predict(AcceptanceFeatures(entropy=h, margin=m))
                assert 0.0 < p < 1.0

    def test_prior_directions(self):
        pr = CalibratedAcceptancePredictor()
        base = AcceptanceFeatures(entropy=1.0, margin=1.0, ngram_hit=0.0,
                                 alpha_ema=0.7, k=2)
        low_h = pr.predict(AcceptanceFeatures(entropy=0.2, margin=1.0,
                                              ngram_hit=0.0, alpha_ema=0.7, k=2))
        high_h = pr.predict(AcceptanceFeatures(entropy=3.0, margin=1.0,
                                               ngram_hit=0.0, alpha_ema=0.7, k=2))
        assert low_h > pr.predict(base) > high_h
        hit = pr.predict(AcceptanceFeatures(entropy=1.0, margin=1.0,
                                           ngram_hit=1.0, alpha_ema=0.7, k=2))
        assert hit > pr.predict(base)

    def test_sgd_moves_toward_observations(self):
        pr = CalibratedAcceptancePredictor()
        f = AcceptanceFeatures(entropy=1.0, margin=1.0, ngram_hit=0.0,
                              alpha_ema=0.7, k=1)
        before = pr.predict(f)
        for _ in range(50):
            pr.update(f, accepted=0, proposed=2)
        after = pr.predict(f)
        assert after < before

    def test_ece_in_range_and_computable(self):
        pr = CalibratedAcceptancePredictor()
        assert pr.expected_calibration_error() == 0.0
        f = AcceptanceFeatures(entropy=0.5, margin=2.0, ngram_hit=1.0,
                              alpha_ema=0.8, k=1)
        for _ in range(20):
            pr.update(f, accepted=2, proposed=2)
        ece = pr.expected_calibration_error()
        assert 0.0 <= ece <= 1.0
        assert pr.n_observations == 40


class TestRegretDecomposition:
    def test_identity(self):
        r = RegretDecomposition(router_tps=40.0, oracle_tps=41.0,
                                best_fixed_tps=44.0, n_prompts=10)
        assert r.r_total == pytest.approx(4.0)
        assert r.r_decision == pytest.approx(1.0)
        assert r.r_estimation == pytest.approx(3.0)
        d = r.to_dict()
        assert d["R_total"] == pytest.approx(d["R_decision"] + d["R_estimation"])


class TestRouterOnlineIntegration:
    def test_router_without_online_modules_unchanged(self):
        from zassd.profiling.action_cost_model import MeasuredActionCostModel
        from zassd.routing.hybrid_router import HybridDraftRouter
        r = HybridDraftRouter(cost_model=MeasuredActionCostModel.for_model("qwen25_3b"))
        d = r.select_source([1, 2, 3], 4, step_k=2, entropy=1.0)
        assert d.source in ("pld", "layer_skip", "vanilla")
        r.update("layer_skip", 1, 2)

    def test_router_with_online_modules_runs_and_reports(self):
        from zassd.profiling.action_cost_model import MeasuredActionCostModel
        from zassd.routing.hybrid_router import HybridDraftRouter
        r = HybridDraftRouter(
            cost_model=MeasuredActionCostModel.for_model("qwen25_3b"),
            latency_tracker=OnlineLatencyTracker(),
            alpha_predictor=CalibratedAcceptancePredictor(),
        )
        d = r.select_source([1, 2, 3], 4, step_k=2, entropy=1.0, margin=0.5)
        assert d.source in ("pld", "layer_skip", "vanilla")
        r.update("layer_skip", 1, 2, draft_ms=19.0, verify_ms=26.0,
                 config_name="cka_75", step_k=2)
        r.observe_vanilla(24.0)
        rep = r.calibration_report()
        assert rep["alpha_obs"] == 2
        assert rep["vanilla_obs"] == 1
        assert 0.0 <= rep["alpha_ece"] <= 1.0
