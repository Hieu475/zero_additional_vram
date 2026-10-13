"""Prompt-split predictors for speculative acceptance survival probabilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def context_bin(length: int) -> str:
    if length <= 512:
        return "le512"
    if length <= 1024:
        return "513_1024"
    if length <= 2048:
        return "1025_2048"
    return "gt2048"


def entropy_bin(value: Any) -> str:
    if value is None:
        return "missing"
    value = float(value)
    if value < 1.0:
        return "lt1"
    if value < 2.0:
        return "1_2"
    if value < 3.0:
        return "2_3"
    return "ge3"


def survival_labels(row: dict[str, Any]) -> list[int]:
    labels = row.get("accepted_survival")
    if labels is not None:
        return [int(bool(x)) for x in labels]
    accepted = int(row.get("accepted_tokens", 0))
    return [int(accepted >= i) for i in range(1, int(row.get("draft_tokens", 0)) + 1)]


class AcceptanceSurvivalPredictor:
    """Smoothed empirical survival predictor with global backoff."""

    MODES = ("global_mean", "global", "action", "context", "entropy", "combined")

    def __init__(self, mode: str = "global", prior_strength: float = 5.0):
        if mode not in self.MODES:
            raise ValueError(f"unknown predictor mode: {mode}")
        self.mode = mode
        self.prior_strength = float(prior_strength)
        self.global_rates: dict[int, float] = {}
        self.global_acceptance: float = 0.0
        self.groups: dict[tuple[Any, ...], dict[int, float]] = {}

    def _key(self, row: dict[str, Any]) -> tuple[Any, ...]:
        action = (str(row.get("action", "unknown")), int(row.get("k", 0)))
        context = context_bin(int(row.get("context_length", 0)))
        entropy = entropy_bin(row.get("entropy"))
        if self.mode == "action":
            return action
        if self.mode == "context":
            return (context,)
        if self.mode == "entropy":
            return (entropy,)
        if self.mode == "combined":
            return action + (context, entropy)
        return ()

    def fit(self, rows: list[dict[str, Any]]) -> "AcceptanceSurvivalPredictor":
        counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        grouped: dict[tuple[Any, ...], dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(lambda: [0, 0]))
        accepted_total = proposed_total = 0
        for row in rows:
            key = self._key(row)
            labels = survival_labels(row)
            accepted_total += sum(labels)
            proposed_total += len(labels)
            for position, accepted in enumerate(labels, start=1):
                counts[position][0] += accepted
                counts[position][1] += 1
                grouped[key][position][0] += accepted
                grouped[key][position][1] += 1
        self.global_rates = {i: yes / total for i, (yes, total) in counts.items() if total}
        self.global_acceptance = accepted_total / proposed_total if proposed_total else 0.0
        self.groups = {}
        for key, positions in grouped.items():
            self.groups[key] = {}
            for i, (yes, total) in positions.items():
                prior = self.global_rates.get(i, 0.0)
                self.groups[key][i] = (yes + self.prior_strength * prior) / (total + self.prior_strength)
        return self

    def predict(self, row: dict[str, Any]) -> list[float]:
        key = self._key(row)
        horizon = int(row.get("prediction_horizon", row.get("draft_tokens", row.get("k", 0))))
        if self.mode == "global_mean":
            return [self.global_acceptance ** i for i in range(1, horizon + 1)]
        survival = [self.groups.get(key, {}).get(i, self.global_rates.get(i, 0.0))
                    for i in range(1, horizon + 1)]
        # Survival probabilities must be non-increasing with draft position.
        for i in range(1, len(survival)):
            survival[i] = min(survival[i], survival[i - 1])
        return survival

    def update_online(self, row: dict[str, Any], ema: dict[int, float], rate: float = 0.2) -> None:
        for position, accepted in enumerate(survival_labels(row), start=1):
            old = ema.get(position, self.global_rates.get(position, float(accepted)))
            ema[position] = (1.0 - rate) * old + rate * accepted

    def predict_online(self, row: dict[str, Any], ema: dict[int, float]) -> list[float]:
        horizon = int(row.get("prediction_horizon", row.get("draft_tokens", row.get("k", 0))))
        survival = [ema.get(i, self.global_rates.get(i, 0.0))
                    for i in range(1, horizon + 1)]
        for i in range(1, len(survival)):
            survival[i] = min(survival[i], survival[i - 1])
        return survival


def expected_emitted(survival: list[float]) -> float:
    """Greedy speculative expectation before EOS/output-budget truncation."""
    return 1.0 + sum(survival)
