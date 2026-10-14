"""Small loading boundary for backend comparisons.

Backends return the ordinary Hugging Face causal-LM object used by the current
ZASSD cache, layer adapter, and decoding implementations. The existing CLI and
reference loading path are untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class LoadedBackend:
    name: str
    model: Any
    tokenizer: Any
    metadata: dict[str, Any]


def load_backend(name: str, model_id: str, revision: str, device: str = "cuda:0",
                 attention: str = "sdpa") -> LoadedBackend:
    """Load a model with a named backend while preserving the HF model API."""
    if name not in {"hf_bnb_nf4", "hqq_gemlite_w4a16"}:
        raise ValueError(f"unknown backend {name!r}; expected hf_bnb_nf4 or hqq_gemlite_w4a16")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if name == "hf_bnb_nf4":
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, quantization_config=quant,
            device_map=device, attn_implementation=attention,
        ).eval()
        return LoadedBackend(name, model, tokenizer, {
            "quantization": "bitsandbytes NF4, double_quant=True, FP16 compute",
            "attention": attention,
        })
    if name == "hqq_gemlite_w4a16":
        try:
            from gemlite.helper import A16W4_HQQ_INT, patch_model
            import hqq
        except ImportError as exc:
            raise RuntimeError(
                "HQQ+GemLite requires optional hqq and gemlite packages; "
                "install compatible versions in the active environment"
            ) from exc
        # Transformers 5.14's HqqConfig route currently raises NotImplementedError
        # in get_quantize_ops(). The low-level supported GemLite patcher keeps
        # Qwen's HF module tree and replaces torch Linear modules in place.
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.float16,
            device_map="cpu", low_cpu_mem_usage=True,
            attn_implementation=attention,
        ).eval()
        model = patch_model(
            model, device=device,
            processor=A16W4_HQQ_INT(device=device, dtype=torch.float16),
            group_size=64,
        )
        return LoadedBackend(name, model, tokenizer, {
            "quantization": "HQQ W4A16, group_size=64, axis=1, GemLite",
            "attention": attention,
            "hqq_version": getattr(hqq, "__version__", "unknown"),
        })
    raise AssertionError("backend validation and dispatch are inconsistent")
