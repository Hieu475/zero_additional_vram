from __future__ import annotations

from zassd.profiling.acceptance_survival import AcceptanceSurvivalPredictor


TRAIN = [
    {"action": "cka_90", "k": 2, "context_length": 120, "entropy": .5,
     "accepted_survival": [1, 1]},
    {"action": "cka_90", "k": 2, "context_length": 160, "entropy": .7,
     "accepted_survival": [0, 0]},
    {"action": "cka_60", "k": 4, "context_length": 900, "entropy": 2.2,
     "accepted_survival": [1, 0, 0, 0]},
]


def test_all_predictors_return_bounded_monotone_survival():
    query = {"action": "cka_90", "k": 4, "prediction_horizon": 4,
             "context_length": 200, "entropy": 1.1}
    for mode in AcceptanceSurvivalPredictor.MODES:
        values = AcceptanceSurvivalPredictor(mode).fit(TRAIN).predict(query)
        assert len(values) == 4
        assert all(0.0 <= value <= 1.0 for value in values)
        assert all(left >= right for left, right in zip(values, values[1:]))


def test_online_prediction_uses_predecision_horizon_and_not_current_outcome():
    predictor = AcceptanceSurvivalPredictor("action").fit(TRAIN)
    ema = dict(predictor.global_rates)
    row = {"action": "cka_90", "k": 4, "prediction_horizon": 2,
           "draft_tokens": 4, "context_length": 240, "entropy": 1.1,
           "accepted_survival": [0, 0, 0, 0]}
    before = predictor.predict_online(row, ema)
    row_with_different_outcome = {**row, "accepted_survival": [1, 1, 1, 1]}
    assert predictor.predict_online(row_with_different_outcome, ema) == before
    assert len(before) == 2
    predictor.update_online(row, ema)
    after_observation = predictor.predict_online(row, ema)
    assert len(after_observation) == 2
