"""Fail-closed architecture capabilities for research B1 schedules.

The registry describes which architecture has an implemented experimental
fixed-row adapter. It does not certify mathematical or whole-sequence exactness.
Runtime preconditions are checked on the loaded model and the adapter performs
additional cache/input checks while it is active.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any


@dataclass(frozen=True)
class FixedRowAdapterSpec:
    model_type: str
    allowed_model_ids: tuple[str, ...]
    allowed_model_revisions: tuple[str, ...]
    attention_class: str
    allowed_attention_implementations: tuple[str, ...]
    attention_strategies: tuple[str, ...]
    adapter_name: str
    status: str
    evidence: str


@dataclass(frozen=True)
class ArchitectureOperatorProfile:
    model_type: str
    family: str
    causal_lm_class: str
    attention_class: str
    allowed_attention_implementations: tuple[str, ...]
    qkv_projection_modules: tuple[str, ...]
    attention_output_module: str
    mlp_up_projection_modules: tuple[str, ...]
    mlp_down_projection_module: str
    evidence_artifact: str


FIXED_ROW_ADAPTERS: dict[str, FixedRowAdapterSpec] = {
    "qwen2": FixedRowAdapterSpec(
        model_type="qwen2",
        allowed_model_ids=(),
        allowed_model_revisions=(),
        attention_class="Qwen2Attention",
        allowed_attention_implementations=("sdpa",),
        attention_strategies=("rowwise", "padded"),
        adapter_name="qwen2_padded.fixed_row_qwen2_schedule",
        status="experimental_empirically_validated_prototype",
        evidence="Pinned Qwen2 BNB NF4/SDPA probes and B1 development runs; no formal all-input proof.",
    ),
    "llama": FixedRowAdapterSpec(
        model_type="llama",
        allowed_model_ids=("unsloth/Llama-3.2-1B-Instruct", "unsloth/Llama-3.2-3B-Instruct"),
        allowed_model_revisions=(
            "5a8abab4a5d6f164389b1079fb721cfab8d7126c",
            "006f5dcd1393c3add266de40994ba96225e9689d",
        ),
        attention_class="LlamaAttention",
        allowed_attention_implementations=("sdpa",),
        attention_strategies=("rowwise", "padded"),
        adapter_name="llama_padded.fixed_row_llama_schedule",
        status="experimental_empirically_validated_prototype",
        evidence=("Pinned Llama-3.2-1B and 3B NF4/SDPA bitwise M=1..9 probes and 64-token "
                  "B1 pilots on GSM8K, CNN/DailyMail, and HumanEval, for both rowwise "
                  "and padded attention; 192/192 committed prefixes matched per strategy."),
    ),
    "phi3": FixedRowAdapterSpec(
        model_type="phi3",
        allowed_model_ids=("microsoft/Phi-3.5-mini-instruct",),
        allowed_model_revisions=("2fe192450127e6a83f7441aef6e3ca586c338b77",),
        attention_class="Phi3Attention",
        allowed_attention_implementations=("sdpa",),
        attention_strategies=("rowwise", "padded"),
        adapter_name="phi3_padded.fixed_row_phi3_schedule",
        status="experimental_empirically_validated_prototype",
        evidence=("Pinned Phi-3.5-Mini NF4/SDPA bitwise M=1..9 probes and 64-token B1 pilots "
                  "on GSM8K, CNN/DailyMail, and HumanEval, for both rowwise and padded "
                  "attention; 192/192 committed prefixes matched per strategy."),
    ),
    "olmo2": FixedRowAdapterSpec(
        model_type="olmo2",
        allowed_model_ids=("allenai/OLMo-2-0425-1B-Instruct",),
        allowed_model_revisions=("48d788eca847d4d7548f375ad03d3c9312f6139e",),
        attention_class="Olmo2Attention",
        allowed_attention_implementations=("sdpa",),
        attention_strategies=("rowwise", "padded"),
        adapter_name="olmo2_padded.fixed_row_olmo2_schedule",
        status="experimental_empirically_validated_prototype",
        evidence=("Pinned OLMo-2-0425-1B NF4/SDPA bitwise M=1..9 probes and 64-token B1 pilots "
                  "on GSM8K, CNN/DailyMail, and HumanEval, for both rowwise and padded "
                  "attention; 192/192 committed prefixes matched per strategy."),
    ),
}

ARCHITECTURE_OPERATOR_PROFILES: dict[str, ArchitectureOperatorProfile] = {
    "qwen2": ArchitectureOperatorProfile(
        model_type="qwen2", family="Qwen2", causal_lm_class="Qwen2ForCausalLM",
        attention_class="Qwen2Attention", allowed_attention_implementations=("sdpa",),
        qkv_projection_modules=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        attention_output_module="self_attn.o_proj",
        mlp_up_projection_modules=("mlp.gate_proj", "mlp.up_proj"),
        mlp_down_projection_module="mlp.down_proj",
        evidence_artifact="crossmodel_operator_qwen25_3b_cnndm002_m5_v1.json",
    ),
    "llama": ArchitectureOperatorProfile(
        model_type="llama", family="Llama", causal_lm_class="LlamaForCausalLM",
        attention_class="LlamaAttention", allowed_attention_implementations=("sdpa",),
        qkv_projection_modules=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        attention_output_module="self_attn.o_proj",
        mlp_up_projection_modules=("mlp.gate_proj", "mlp.up_proj"),
        mlp_down_projection_module="mlp.down_proj",
        evidence_artifact="crossmodel_operator_llama32_3b_cnndm002_m5_v1.json",
    ),
    "phi3": ArchitectureOperatorProfile(
        model_type="phi3", family="Phi3", causal_lm_class="Phi3ForCausalLM",
        attention_class="Phi3Attention", allowed_attention_implementations=("sdpa",),
        qkv_projection_modules=("self_attn.qkv_proj",),
        attention_output_module="self_attn.o_proj",
        mlp_up_projection_modules=("mlp.gate_up_proj",),
        mlp_down_projection_module="mlp.down_proj",
        evidence_artifact="phi35_cnndm002_operator_shape_k4_v1.json",
    ),
    "olmo2": ArchitectureOperatorProfile(
        model_type="olmo2", family="OLMo2", causal_lm_class="Olmo2ForCausalLM",
        attention_class="Olmo2Attention", allowed_attention_implementations=("sdpa",),
        qkv_projection_modules=("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        attention_output_module="self_attn.o_proj",
        mlp_up_projection_modules=("mlp.gate_proj", "mlp.up_proj"),
        mlp_down_projection_module="mlp.down_proj",
        evidence_artifact="crossmodel_operator_olmo2_1b_cnndm002_m5_v1.json",
    ),
}

SHAPE_SENSITIVITY_EVIDENCE: dict[str, str] = {
    "qwen2": "Cross-model NF4/FP16/SDPA shape sweep; Qwen2 is the only family with a fixed-row prototype.",
    "llama": "Cross-model NF4/FP16/SDPA shape sweep and operator-shape audits; no B1 fixed-row adapter registered.",
    "phi3": "Cross-model NF4/FP16/SDPA shape sweep and fused qkv_proj audit; B1 adapter validated only for pinned Phi-3.5-Mini checkpoint.",
    "olmo2": "Cross-model NF4/FP16/SDPA shape sweep and q_proj audit; fixed-row adapter validated only for the pinned OLMo-2-0425-1B checkpoint.",
}


class B1CapabilityError(RuntimeError):
    """Raised when an experimental B1 schedule is not established for a model."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(
            f"B1 fixed-row capability unavailable: "
            f"{'; '.join(report['rejection_reasons']) or 'unsupported capability profile'}"
        )


def _capability_id(report: dict[str, Any]) -> str:
    stable = {key: value for key, value in report.items() if key != "capability_id"}
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def inspect_architecture_operator_layout(model: Any) -> dict[str, Any]:
    """Inventory the first-layer projection layout against measured family profiles."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    profile = ARCHITECTURE_OPERATOR_PROFILES.get(model_type)
    if profile is None:
        return {
            "recognized": False,
            "model_type": model_type,
            "complete": False,
            "missing_modules": ["no registered architecture operator profile"],
        }
    named_modules = dict(model.named_modules())
    required = [
        *profile.qkv_projection_modules,
        profile.attention_output_module,
        *profile.mlp_up_projection_modules,
        profile.mlp_down_projection_module,
    ]
    prefix = "model.layers.0."
    present = {suffix: prefix + suffix in named_modules for suffix in required}
    attention_modules = [module for module in model.modules()
                         if module.__class__.__name__ == profile.attention_class]
    return {
        "recognized": True,
        "model_type": model_type,
        "family": profile.family,
        "complete": all(present.values()) and bool(attention_modules),
        "first_layer_projection_layout": {
            "qkv": list(profile.qkv_projection_modules),
            "attention_output": profile.attention_output_module,
            "mlp_up": list(profile.mlp_up_projection_modules),
            "mlp_down": profile.mlp_down_projection_module,
        },
        "module_presence": present,
        "attention_class": profile.attention_class,
        "attention_module_count": len(attention_modules),
        "evidence_artifact": profile.evidence_artifact,
        "missing_modules": [suffix for suffix, exists in present.items() if not exists],
    }


def inspect_b1_fixed_row_capability(
    model: Any, *, attention_strategy: str | None = None,
) -> dict[str, Any]:
    """Describe whether this model meets the registered fixed-row prototype gate."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    model_id = getattr(config, "_name_or_path", None)
    model_revision = getattr(config, "_commit_hash", None)
    spec = FIXED_ROW_ADAPTERS.get(model_type)
    architecture_profile = ARCHITECTURE_OPERATOR_PROFILES.get(model_type)
    shape_evidence = SHAPE_SENSITIVITY_EVIDENCE.get(model_type)
    attention_impl = getattr(config, "_attn_implementation", None)
    layout_report = inspect_architecture_operator_layout(model)
    expected_attention_class = (
        architecture_profile.attention_class if architecture_profile else
        spec.attention_class if spec else ""
    )
    attention_modules = [m for m in model.modules()
                         if m.__class__.__name__ == expected_attention_class]
    model_architectures = list(getattr(config, "architectures", None) or [])
    checks = {
        "registered_adapter": spec is not None,
        "checkpoint_id_supported": bool(
            spec is None or not spec.allowed_model_ids or model_id in spec.allowed_model_ids
        ),
        "checkpoint_revision_supported": bool(
            spec is None or not spec.allowed_model_revisions or
            model_revision in spec.allowed_model_revisions
        ),
        "model_class_matches": bool(architecture_profile and any(
            name == architecture_profile.causal_lm_class for name in model_architectures
        ) and type(model).__name__ == architecture_profile.causal_lm_class),
        "registered_operator_layout_matches_model": bool(layout_report.get("complete")),
        "eval_mode": not bool(getattr(model, "training", True)),
        "attention_implementation_supported": bool(
            architecture_profile and
            attention_impl in architecture_profile.allowed_attention_implementations
        ),
        "attention_strategy_supported": bool(
            attention_strategy is None or
            bool(spec and attention_strategy in spec.attention_strategies)
        ),
        "expected_attention_modules_present": bool(attention_modules),
        "no_sliding_window_attention": bool(attention_modules) and all(
            not bool(getattr(module, "sliding_window", None)) for module in attention_modules
        ),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    report: dict[str, Any] = {
        "profile": "b1_fixed_row_experimental",
        "supported": not reasons,
        "mode": "experimental_only",
        "proof_status": "not_formally_proven",
        "model_type": model_type,
        "model_id": model_id,
        "model_revision": model_revision,
        "model_class": type(model).__name__,
        "model_architectures": model_architectures,
        "attention_implementation": attention_impl,
        "attention_strategy": attention_strategy,
        "supported_attention_strategies": list(spec.attention_strategies) if spec else [],
        "attention_module_count": len(attention_modules),
        "operator_layout": layout_report,
        "adapter": spec.adapter_name if spec else None,
        "adapter_status": spec.status if spec else (
            "unregistered_shape_sensitivity_observed" if shape_evidence else "unregistered"
        ),
        "evidence": spec.evidence if spec else "No architecture-specific fixed-row adapter is registered.",
        "shape_sensitivity_observed": shape_evidence is not None,
        "shape_sensitivity_evidence": shape_evidence,
        "checks": checks,
        "rejection_reasons": reasons,
        "runtime_checks_still_required": [
            "batch size one and unpadded prompt",
            "implicit positions and no caller-supplied attention mask",
            "static preallocated KV cache and valid crop/rollback behavior",
            "physical verification rows within the adapter's fixed row count",
            "greedy output and committed KV comparison against the singleton reference",
        ],
    }
    report["capability_id"] = _capability_id(report)
    return report


def require_b1_fixed_row_capability(
    model: Any, *, attention_strategy: str | None = None,
) -> dict[str, Any]:
    """Return the capability report or fail closed before fixed-row execution."""
    report = inspect_b1_fixed_row_capability(model, attention_strategy=attention_strategy)
    if not report["supported"]:
        raise B1CapabilityError(report)
    return report
