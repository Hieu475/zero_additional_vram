"""Online-calibrated cost and acceptance estimation (Phase P0).

Two small, CPU-only, dependency-light components that upgrade the router
from fixed-profile estimates to online-calibrated ones:

1. :class:`OnlineLatencyTracker` -- EMA tracking of the three quantities
   the utility formula actually consumes: vanilla step time, draft time
   per (config, K), and verify time per K. When the GPU heats up, clocks
   shift, or context grows, predictions adapt within a few cycles instead
   of trusting a frozen profile.

2. :class:`CalibratedAcceptancePredictor` -- a tiny logistic model over
   the per-cycle feature vector::

       x_t = [H_t, margin, retained_frac, ngram_hit, alpha_EMA, K]

   with online SGD updates and binned reliability tracking so we can
   report *calibration error* (ECE), not just accuracy. Starts from
   sensible priors (matches the old heuristic direction) and improves
   as cycles accumulate; with no observations it reduces to the prior.

Regret framing used in the paper::

    R_total = R_estimation + R_decision

where R_decision is the gap between the router and the per-cycle oracle
(same estimates, bare argmax) and R_estimation is the gap between the
oracle and the best-fixed hindsight baseline. The existing oracle logs
show R_decision ~= 0, so remaining regret is estimation error -- exactly
what these two classes attack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _logit(p: float) -> float:
    p = min(0.98, max(0.02, p))
    return math.log(p / (1.0 - p))


@dataclass
class LatencyPrediction:
    """One blended latency prediction with provenance for the paper."""

    value_ms: float
    ema_ms: float | None
    fallback_ms: float
    n_obs: int


class OnlineLatencyTracker:
    """EMA tracker for T_vanilla, T_draft(config, K), T_verify(K).

    ``predict_*`` blends the EMA with a profile fallback::

        pred = w * ema + (1 - w) * fallback,  w = n / (n + min_samples)

    so early cycles trust the profile and later cycles trust silicon.
    Pure Python, no GPU, safe to call per cycle (dict lookups only).
    """

    def __init__(self, ema_beta: float = 0.2, min_samples: int = 5) -> None:
        self.ema_beta = ema_beta
        self.min_samples = min_samples
        self._vanilla_ms: float | None = None
        self._vanilla_n = 0
        self._draft: dict[tuple[str, int], tuple[float, int]] = {}
        self._verify: dict[int, tuple[float, int]] = {}

    # -- updates (called once per cycle with measured wall times) ---------
    def update_vanilla(self, ms: float) -> None:
        self._vanilla_ms = ms if self._vanilla_ms is None else (
            (1 - self.ema_beta) * self._vanilla_ms + self.ema_beta * ms)
        self._vanilla_n += 1

    def update_draft(self, config: str, k: int, ms: float) -> None:
        key = (config, k)
        prev, n = self._draft.get(key, (ms, 0))
        ema = ms if n == 0 else (1 - self.ema_beta) * prev + self.ema_beta * ms
        self._draft[key] = (ema, n + 1)

    def update_verify(self, k: int, ms: float) -> None:
        prev, n = self._verify.get(k, (ms, 0))
        ema = ms if n == 0 else (1 - self.ema_beta) * prev + self.ema_beta * ms
        self._verify[k] = (ema, n + 1)

    # -- predictions -------------------------------------------------------
    def _blend(self, ema: float | None, n: int, fallback: float) -> LatencyPrediction:
        if ema is None or n <= 0:
            return LatencyPrediction(fallback, None, fallback, 0)
        w = n / (n + self.min_samples)
        return LatencyPrediction(w * ema + (1 - w) * fallback, ema, fallback, n)

    def predict_vanilla(self, fallback_ms: float) -> LatencyPrediction:
        return self._blend(self._vanilla_ms, self._vanilla_n, fallback_ms)

    def predict_draft(self, config: str, k: int, fallback_ms: float) -> LatencyPrediction:
        ema, n = self._draft.get((config, k), (None, 0))  # type: ignore[misc]
        return self._blend(ema, n, fallback_ms)

    def predict_verify(self, k: int, fallback_ms: float) -> LatencyPrediction:
        ema, n = self._verify.get(k, (None, 0))  # type: ignore[misc]
        return self._blend(ema, n, fallback_ms)

    @property
    def n_vanilla_obs(self) -> int:
        return self._vanilla_n


@dataclass
class AcceptanceFeatures:
    """Per-cycle feature vector for the acceptance predictor."""

    entropy: float = 1.0
    margin: float | None = None  # top1 - top2 logit gap; None = unknown
    retained_frac: float = 0.75  # kept layers / total layers
    ngram_hit: float = 0.0  # 1.0 strong (n>=3), 0.25 weak (n==2), 0.0 miss
    alpha_ema: float = 0.7
    k: int = 1


class CalibratedAcceptancePredictor:
    """Tiny online logistic acceptance model with reliability bins.

    logit(alpha_hat) = b + w_ema*logit(ema) + w_H*(H_ref - H)
                       + w_m*margin + w_hit*hit + w_K*(K-1) + w_r*(retained-0.75)

    Priors reproduce the old heuristic directions (low entropy / large
    margin / strong hit / small K / deep draft -> higher acceptance).
    ``update`` takes one SGD step on log-loss weighted by ``proposed``
    and records the (prediction, outcome) pair in reliability bins so
    :meth:`expected_calibration_error` is always available for the paper.
    """

    N_BINS = 10

    def __init__(
        self,
        b: float = 0.0,
        w_ema: float = 1.0,
        w_h: float = 0.25,
        w_m: float = 0.30,
        w_hit: float = 0.60,
        w_k: float = -0.15,
        w_r: float = 1.25,
        h_ref: float = 1.0,
        lr: float = 0.05,
    ) -> None:
        self.b = b
        self.w_ema = w_ema
        self.w_h = w_h
        self.w_m = w_m
        self.w_hit = w_hit
        self.w_k = w_k
        self.w_r = w_r
        self.h_ref = h_ref
        self.lr = lr
        self._bin_pred: list[float] = [0.0] * self.N_BINS
        self._bin_obs: list[float] = [0.0] * self.N_BINS
        self._bin_n: list[int] = [0] * self.N_BINS
        self.n_updates = 0

    def _linear(self, f: AcceptanceFeatures) -> float:
        z = self.b + self.w_ema * _logit(f.alpha_ema)
        z += self.w_h * (self.h_ref - f.entropy)
        if f.margin is not None:
            z += self.w_m * min(5.0, max(0.0, f.margin))
        z += self.w_hit * f.ngram_hit
        z += self.w_k * (f.k - 1)
        z += self.w_r * (f.retained_frac - 0.75)
        return z

    def predict(self, f: AcceptanceFeatures) -> float:
        return _sigmoid(self._linear(f))

    def update(self, f: AcceptanceFeatures, accepted: int, proposed: int) -> float:
        """One SGD step; returns the pre-update prediction (for logging)."""
        if proposed <= 0:
            return self.predict(f)
        p = self.predict(f)
        y = accepted / proposed
        err = (p - y) * proposed  # gradient scaled by #trials
        lr = self.lr
        # Update along each active feature direction.
        self.b -= lr * err
        self.w_ema -= lr * err * _logit(f.alpha_ema)
        self.w_h -= lr * err * (self.h_ref - f.entropy)
        if f.margin is not None:
            self.w_m -= lr * err * min(5.0, max(0.0, f.margin))
        self.w_hit -= lr * err * f.ngram_hit
        self.w_k -= lr * err * (f.k - 1)
        self.w_r -= lr * err * (f.retained_frac - 0.75)
        # Reliability bin records the *pre-update* prediction vs outcome.
        b = min(self.N_BINS - 1, int(p * self.N_BINS))
        self._bin_pred[b] += p * proposed
        self._bin_obs[b] += accepted
        self._bin_n[b] += proposed
        self.n_updates += 1
        return p

    def expected_calibration_error(self) -> float:
        """Mean |accuracy - confidence| weighted by bin mass in [0, 1]."""
        total = sum(self._bin_n)
        if total <= 0:
            return 0.0
        ece = 0.0
        for i in range(self.N_BINS):
            n = self._bin_n[i]
            if n > 0:
                ece += (n / total) * abs(self._bin_pred[i] / n - self._bin_obs[i] / n)
        return ece

    @property
    def n_observations(self) -> int:
        return sum(self._bin_n)

    def reliability_table(self) -> list[dict]:
        rows = []
        for i in range(self.N_BINS):
            n = self._bin_n[i]
            rows.append({
                "bin": i,
                "n": n,
                "mean_pred": (self._bin_pred[i] / n) if n else 0.0,
                "mean_obs": (self._bin_obs[i] / n) if n else 0.0,
            })
        return rows


@dataclass
class RegretDecomposition:
    """R_total = R_estimation + R_decision for one evaluation slice."""

    router_tps: float
    oracle_tps: float  # same estimates, bare argmax (no heuristics)
    best_fixed_tps: float  # hindsight best single strategy per prompt
    n_prompts: int = 0

    @property
    def r_total(self) -> float:
        return self.best_fixed_tps - self.router_tps

    @property
    def r_decision(self) -> float:
        return self.oracle_tps - self.router_tps

    @property
    def r_estimation(self) -> float:
        return self.r_total - self.r_decision

    def to_dict(self) -> dict:
        return {
            "router_tps": self.router_tps,
            "oracle_tps": self.oracle_tps,
            "best_fixed_tps": self.best_fixed_tps,
            "n_prompts": self.n_prompts,
            "R_total": self.r_total,
            "R_decision": self.r_decision,
            "R_estimation": self.r_estimation,
        }
