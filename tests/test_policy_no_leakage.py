from __future__ import annotations

from zassd.profiling.acceptance_survival import AcceptanceSurvivalPredictor


def test_acceptance_prediction_does_not_read_current_or_future_outcomes():
    training = [
        {"action": "cka_90", "k": 2, "context_length": 128, "entropy": .4,
         "accepted_survival": [1, 1]},
        {"action": "cka_90", "k": 2, "context_length": 128, "entropy": .6,
         "accepted_survival": [0, 0]},
    ]
    predictor = AcceptanceSurvivalPredictor("combined").fit(training)
    before = {"action": "cka_90", "k": 2, "prediction_horizon": 2,
              "context_length": 256, "entropy": .8,
              "accepted_tokens": 0, "accepted_survival": [0, 0],
              "draft_ms": 80, "verify_ms": 40}
    leaked = {**before, "accepted_tokens": 2, "accepted_survival": [1, 1],
              "draft_ms": 1, "verify_ms": 1}
    assert predictor.predict(before) == predictor.predict(leaked)


def test_online_predictor_is_updated_only_after_observation():
    training = [{"action": "cka_60", "k": 2, "context_length": 300, "entropy": 1.0,
                 "accepted_survival": [1, 0]}]
    predictor = AcceptanceSurvivalPredictor("action").fit(training)
    row = {"action": "cka_60", "k": 2, "prediction_horizon": 2,
           "draft_tokens": 2, "context_length": 300, "entropy": 1.0,
           "accepted_survival": [0, 0]}
    ema = dict(predictor.global_rates)
    first_prediction = predictor.predict_online(row, ema)
    assert first_prediction == [1.0, 0.0]
    predictor.update_online(row, ema)
    second_prediction = predictor.predict_online(row, ema)
    assert second_prediction == [.8, 0.0]
