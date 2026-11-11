"""Exact host-side arithmetic for a conditional greedy-token certificate.

The certificate theorem is simple: if every reference logit differs from its
fast-path counterpart by at most ``epsilon``, a strict top-two gap greater
than ``2 * epsilon`` fixes the reference argmax. The hard research problem is
to produce a *valid* epsilon from 4-bit kernels and accumulated KV drift.

Fractions avoid accidentally rounding a host-side bound downward. GPU
roundoff bounds must be supplied separately; empirical error maxima are not
acceptable proof inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence


def _nonnegative(value: Fraction | int | float, name: str) -> Fraction:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    result = Fraction(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


@dataclass(frozen=True)
class StateEnvelope:
    """Conditional bound on the canonical KV/state error at one token prefix.

    ``radius`` and ``gain`` are in a caller-specified norm. The caller must
    prove that the chosen norm and transition bound cover *all* state that
    affects future forward passes. A nonempty contract ID prevents accidental
    comparison of evidence from different backends or reference executions.
    """

    radius: Fraction
    reference_contract_id: str
    logical_prefix_length: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius", _nonnegative(self.radius, "radius"))
        if not self.reference_contract_id:
            raise ValueError("reference_contract_id is required")
        if self.logical_prefix_length < 0:
            raise ValueError("logical_prefix_length must be nonnegative")

    def after_committed_token(
        self, transition_gain: Fraction | int | float,
        local_state_error: Fraction | int | float,
    ) -> "StateEnvelope":
        """Apply an externally justified recurrence ``e' <= gain*e + error``."""
        gain = _nonnegative(transition_gain, "transition_gain")
        error = _nonnegative(local_state_error, "local_state_error")
        return StateEnvelope(
            radius=gain * self.radius + error,
            reference_contract_id=self.reference_contract_id,
            logical_prefix_length=self.logical_prefix_length + 1,
        )


@dataclass(frozen=True)
class TokenCertificate:
    reference_contract_id: str
    logical_prefix_length: int
    token_id: int
    fast_margin: Fraction
    logit_error_bound: Fraction


def certify_greedy_token(
    fast_logits: Sequence[float],
    state: StateEnvelope,
    *,
    state_to_logit_gain: Fraction | int | float,
    local_logit_error: Fraction | int | float,
    reference_contract_id: str,
    logical_prefix_length: int,
) -> TokenCertificate | None:
    """Certify a greedy argmax, conditional on valid supplied error bounds.

    Returns ``None`` on close margins, ties, or nonfinite logits. It never
    promotes an empirical margin threshold into a correctness certificate.
    The caller must verify that ``state_to_logit_gain`` and
    ``local_logit_error`` bound the actual quantized execution.
    """
    if reference_contract_id != state.reference_contract_id:
        raise ValueError("reference contract mismatch")
    if logical_prefix_length != state.logical_prefix_length:
        raise ValueError("logical prefix length mismatch")
    gain = _nonnegative(state_to_logit_gain, "state_to_logit_gain")
    local = _nonnegative(local_logit_error, "local_logit_error")
    if len(fast_logits) < 2:
        raise ValueError("at least two logits are required")
    winner = -1
    first = second = -math.inf
    for index, raw in enumerate(fast_logits):
        value = float(raw)
        if not math.isfinite(value):
            return None
        if value > first:
            second = first
            first = value
            winner = index
        elif value > second:
            second = value
    # Convert only the top two values to exact rationals. A tie gets margin 0.
    margin = Fraction.from_float(first) - Fraction.from_float(second)
    error_bound = gain * state.radius + local
    if margin <= 2 * error_bound:
        return None
    return TokenCertificate(reference_contract_id, logical_prefix_length,
                            winner, margin, error_bound)
