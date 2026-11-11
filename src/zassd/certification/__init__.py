"""Conditional, finite-arithmetic certification primitives.

These helpers only prove token equality when their input error bounds have
already been established for the active model, backend, and KV state. No
quantized backend in ZASSD currently supplies such a bound.
"""

from .greedy import StateEnvelope, TokenCertificate, certify_greedy_token
from .fixed_row_dispatch import fixed_row_schedule_for_model
from .model_capabilities import (
    ARCHITECTURE_OPERATOR_PROFILES,
    B1CapabilityError,
    inspect_architecture_operator_layout,
    inspect_b1_fixed_row_capability,
    require_b1_fixed_row_capability,
)
from .qwen2_padded import fixed_row_qwen2_schedule

__all__ = [
    "ARCHITECTURE_OPERATOR_PROFILES", "B1CapabilityError", "StateEnvelope", "TokenCertificate",
    "certify_greedy_token", "fixed_row_qwen2_schedule", "fixed_row_schedule_for_model",
    "inspect_architecture_operator_layout",
    "inspect_b1_fixed_row_capability", "require_b1_fixed_row_capability",
]
