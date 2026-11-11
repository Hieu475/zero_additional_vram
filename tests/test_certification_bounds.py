"""Check the conditional certificate and accumulated-state failure mode."""

from fractions import Fraction

import pytest

from zassd.certification import StateEnvelope, certify_greedy_token


def test_certificate_survives_every_reference_logit_inside_bound():
    state = StateEnvelope(Fraction(1, 4), "synthetic-int4", 7)
    certificate = certify_greedy_token(
        [3.0, 1.0, -2.0], state, state_to_logit_gain=1,
        local_logit_error=Fraction(1, 4),
        reference_contract_id="synthetic-int4", logical_prefix_length=7,
    )
    assert certificate is not None and certificate.token_id == 0
    epsilon = certificate.logit_error_bound
    fast = [Fraction(3), Fraction(1), Fraction(-2)]
    # An adversarial reference moves the winner down and competitors up.
    reference = [fast[0] - epsilon, fast[1] + epsilon, fast[2] + epsilon]
    assert reference.index(max(reference)) == certificate.token_id


def test_kv_error_accumulates_even_when_previous_token_was_certified():
    state = StateEnvelope(Fraction(0), "synthetic-int4", 0)
    first = certify_greedy_token(
        [2.0, 0.0], state, state_to_logit_gain=1,
        local_logit_error=Fraction(1, 10),
        reference_contract_id="synthetic-int4", logical_prefix_length=0,
    )
    assert first is not None
    state = state.after_committed_token(transition_gain=2,
                                        local_state_error=Fraction(3, 5))
    second = certify_greedy_token(
        [1.0, 0.0], state, state_to_logit_gain=1,
        local_logit_error=Fraction(1, 10),
        reference_contract_id="synthetic-int4", logical_prefix_length=1,
    )
    assert state.radius == Fraction(3, 5)
    assert second is None  # margin 1 is below 2*(0.6+0.1)


def test_certificate_fails_closed_on_ties_nonfinite_and_wrong_reference():
    state = StateEnvelope(0, "backend-a", 3)
    kwargs = dict(state_to_logit_gain=0, local_logit_error=0,
                  reference_contract_id="backend-a", logical_prefix_length=3)
    assert certify_greedy_token([1.0, 1.0], state, **kwargs) is None
    assert certify_greedy_token([float("nan"), 1.0], state, **kwargs) is None
    with pytest.raises(ValueError, match="contract"):
        certify_greedy_token([2.0, 1.0], state, **{**kwargs, "reference_contract_id": "backend-b"})
    with pytest.raises(ValueError, match="prefix"):
        certify_greedy_token([2.0, 1.0], state, **{**kwargs, "logical_prefix_length": 4})


def test_state_bound_rejects_negative_inputs():
    with pytest.raises(ValueError):
        StateEnvelope(-1, "backend-a", 0)
    with pytest.raises(ValueError):
        StateEnvelope(0, "backend-a", 0).after_committed_token(-1, 0)
