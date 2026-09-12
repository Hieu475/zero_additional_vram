"""Vanilla autoregressive decoding for baseline measurement."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class GenerationMetrics:
    """Metrics collected during generation."""
    total_tokens: int = 0
    total_time_s: float = 0.0
    ttft_s: float = 0.0           # Time to first token
    tpot_ms: float = 0.0          # Time per output token
    tokens_per_second: float = 0.0
    peak_vram_mb: float = 0.0
    per_token_times_ms: list[float] = field(default_factory=list)


def vanilla_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    device: str = "cuda:0",
) -> tuple[str, GenerationMetrics]:
    """Generate text using vanilla autoregressive decoding.

    This is the baseline against which all speculative methods
    are compared.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: Input prompt text.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        device: Target device.

    Returns:
        Tuple of (generated_text, metrics).
    """
    metrics = GenerationMetrics()

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Reset VRAM tracking
    torch.cuda.reset_peak_memory_stats(device)

    # Generate token by token for detailed timing
    generated_ids = inputs["input_ids"].clone()
    past_key_values = None

    start_time = time.perf_counter()

    with torch.no_grad():
        for step in range(max_new_tokens):
            step_start = time.perf_counter()

            if past_key_values is None:
                outputs = model(
                    input_ids=generated_ids,
                    use_cache=True,
                )
            else:
                outputs = model(
                    input_ids=generated_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Greedy or sampling
            if temperature == 0.0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            step_time = (time.perf_counter() - step_start) * 1000
            metrics.per_token_times_ms.append(step_time)

            if step == 0:
                metrics.ttft_s = time.perf_counter() - start_time

            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

    end_time = time.perf_counter()

    # Compute metrics
    new_tokens = generated_ids.shape[1] - input_len
    metrics.total_tokens = new_tokens
    metrics.total_time_s = end_time - start_time
    metrics.tokens_per_second = (
        new_tokens / metrics.total_time_s if metrics.total_time_s > 0 else 0
    )
    if len(metrics.per_token_times_ms) > 1:
        metrics.tpot_ms = sum(metrics.per_token_times_ms[1:]) / len(
            metrics.per_token_times_ms[1:]
        )
    metrics.peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    # Decode output
    output_text = tokenizer.decode(
        generated_ids[0, input_len:], skip_special_tokens=True
    )

    return output_text, metrics
