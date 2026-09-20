"""Model loading utilities with quantization support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)


def get_quantization_config(bits: int = 4) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig for quantized loading.

    Args:
        bits: Number of bits for quantization (4 or 8).

    Returns:
        BitsAndBytesConfig instance.
    """
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unsupported quantization bits: {bits}")


def load_model(
    model_name: str,
    revision: str = "main",
    quantize: bool = True,
    bits: int = 4,
    device: str = "cuda:0",
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> PreTrainedModel:
    """Load a causal language model with optional quantization.

    Args:
        model_name: HuggingFace model name or local path.
        revision: Model revision for reproducibility.
        quantize: Whether to apply quantization.
        bits: Quantization bits (4 or 8).
        device: Target device.
        trust_remote_code: Whether to trust remote code.
        **kwargs: Additional arguments for from_pretrained.

    Returns:
        Loaded model.
    """
    load_kwargs: dict[str, Any] = {
        "revision": revision,
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
    }

    if quantize:
        load_kwargs["quantization_config"] = get_quantization_config(bits)
        logger.info(f"Loading {model_name} with {bits}-bit quantization")
    else:
        load_kwargs["torch_dtype"] = torch.float16
        logger.info(f"Loading {model_name} in FP16")

    load_kwargs.update(kwargs)

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    logger.info(
        f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    return model


def load_tokenizer(
    model_name: str,
    padding_side: str = "left",
    trust_remote_code: bool = True,
) -> PreTrainedTokenizer:
    """Load tokenizer for the specified model.

    Args:
        model_name: HuggingFace model name or local path.
        padding_side: Padding side for batch processing.
        trust_remote_code: Whether to trust remote code.

    Returns:
        Loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side=padding_side,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer
