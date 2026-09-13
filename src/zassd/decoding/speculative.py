"""Self-speculative decoding implementation.

Zero-Additional-VRAM self-speculative decoding using a shared model checkpoint:
1. Draft phase: Generate K candidate tokens using subnetwork (skipped layers via LayerManager)
2. Verify phase: Verify all K tokens in parallel using full model (all layers)
3. Accept / Reject: Update sequence with accepted draft tokens + target correction/bonus token
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from zassd.models.layer_manager import LayerManager
from zassd.decoding.verification import verify_tokens_greedy
from zassd.decoding.rejection_sampling import verify_tokens_rejection_sampling

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeMetrics:
    """Comprehensive metrics for speculative decoding run."""
    total_tokens: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    num_verification_cycles: int = 0
    tokens_per_step: float = 0.0  # E[accepted tokens + 1 per verification cycle]
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0
    draft_time_s: float = 0.0
    verify_time_s: float = 0.0
    peak_vram_mb: float = 0.0
    speedup_vs_vanilla: float = 0.0
    k_value: int = 4
    per_iteration_stats: list[dict] = field(default_factory=list)


def self_speculative_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_mgr: LayerManager,
    skip_indices: list[int],
    prompt: str,
    k: int = 4,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    device: str = "cuda:0",
) -> tuple[str, SpeculativeMetrics]:
    """Generate text using Zero-Additional-VRAM self-speculative decoding.

    Args:
        model: Pre-trained causal language model.
        tokenizer: Tokenizer.
        layer_mgr: LayerManager instance wrapping the model.
        skip_indices: Layers to skip during draft phase (e.g. CKA-selected).
        prompt: Input text.
        k: Number of draft tokens per speculation step.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (0.0 = greedy, exactness guaranteed).
        device: Device to run generation on.

    Returns:
        Tuple of (generated_text, SpeculativeMetrics).
    """
    metrics = SpeculativeMetrics(k_value=k)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    current_ids = inputs["input_ids"].clone()
    prompt_len = current_ids.shape[1]

    torch.cuda.reset_peak_memory_stats(device)
    total_start = time.perf_counter()

    while (current_ids.shape[1] - prompt_len) < max_new_tokens:
        tokens_needed = max_new_tokens - (current_ids.shape[1] - prompt_len)
        step_k = min(k, tokens_needed)
        if step_k <= 0:
            break

        # -------------------------------------------------------------------
        # Phase 1: Draft generation (using skipped layer subnetwork)
        # -------------------------------------------------------------------
        draft_tokens: list[int] = []
        draft_probs_list: list[torch.Tensor] = []
        draft_input = current_ids.clone()
        pkv_draft = None

        t_draft_start = time.perf_counter()
        with torch.no_grad():
            with layer_mgr.skip_layers(skip_indices):
                for d_step in range(step_k):
                    if pkv_draft is None:
                        d_out = model(input_ids=draft_input, use_cache=True)
                    else:
                        d_out = model(
                            input_ids=draft_input[:, -1:],
                            past_key_values=pkv_draft,
                            use_cache=True,
                        )
                    pkv_draft = d_out.past_key_values
                    d_logits = d_out.logits[:, -1, :].float()

                    if temperature == 0.0:
                        d_next = int(d_logits.argmax(dim=-1).item())
                    else:
                        probs = F.softmax(d_logits / temperature, dim=-1)
                        d_next = int(torch.multinomial(probs, num_samples=1).item())
                        draft_probs_list.append(probs[0])

                    draft_tokens.append(d_next)
                    d_next_tensor = torch.tensor([[d_next]], device=device)
                    draft_input = torch.cat([draft_input, d_next_tensor], dim=-1)

                    if d_next == tokenizer.eos_token_id:
                        break

        t_draft_end = time.perf_counter()
        draft_elapsed = t_draft_end - t_draft_start
        metrics.draft_time_s += draft_elapsed
        metrics.total_draft_tokens += len(draft_tokens)

        if not draft_tokens:
            break

        actual_k = len(draft_tokens)

        # -------------------------------------------------------------------
        # Phase 2: Parallel target verification (using full model)
        # -------------------------------------------------------------------
        # Full model evaluates [current_ids, draft_tokens] in a single forward pass
        candidate_ids = torch.cat(
            [current_ids, torch.tensor([draft_tokens], device=device)], dim=-1
        )

        t_verify_start = time.perf_counter()
        with torch.no_grad():
            verify_out = model(input_ids=candidate_ids, use_cache=False)
            # We need logits at positions from current_ids.shape[1] - 1 to candidate_ids.shape[1] - 1
            # Total positions: actual_k + 1
            start_pos = current_ids.shape[1] - 1
            target_logits = verify_out.logits[0, start_pos : start_pos + actual_k + 1, :].float()

        t_verify_end = time.perf_counter()
        verify_elapsed = t_verify_end - t_verify_start
        metrics.verify_time_s += verify_elapsed
        metrics.num_verification_cycles += 1

        # -------------------------------------------------------------------
        # Phase 3: Accept / Reject
        # -------------------------------------------------------------------
        if temperature == 0.0:
            accepted_tokens, next_token, rejected_at = verify_tokens_greedy(
                target_logits=target_logits,
                draft_tokens=draft_tokens,
            )
        else:
            target_probs = F.softmax(target_logits / temperature, dim=-1)
            draft_probs_tensor = torch.stack(draft_probs_list, dim=0)
            accepted_tokens, next_token, rejected_at = verify_tokens_rejection_sampling(
                target_probs=target_probs,
                draft_probs=draft_probs_tensor,
                draft_tokens=draft_tokens,
            )

        num_accepted = len(accepted_tokens)
        metrics.total_accepted_tokens += num_accepted

        # Append accepted tokens + next target token
        emitted_tokens = accepted_tokens + [next_token]
        new_ids = torch.tensor([emitted_tokens], device=device)
        current_ids = torch.cat([current_ids, new_ids], dim=-1)

        iter_stat = {
            "k": actual_k,
            "accepted": num_accepted,
            "rejected_at": rejected_at,
            "draft_time_ms": draft_elapsed * 1000,
            "verify_time_ms": verify_elapsed * 1000,
        }
        metrics.per_iteration_stats.append(iter_stat)

        # Stop if EOS emitted
        if next_token == tokenizer.eos_token_id or (
            tokenizer.eos_token_id in accepted_tokens
        ):
            break

    total_end = time.perf_counter()
    metrics.total_time_s = total_end - total_start

    # Ensure exact token length boundary matching max_new_tokens
    if (current_ids.shape[1] - prompt_len) > max_new_tokens:
        current_ids = current_ids[:, : prompt_len + max_new_tokens]

    new_tokens_count = current_ids.shape[1] - prompt_len
    metrics.total_tokens = new_tokens_count
    metrics.tokens_per_second = (
        new_tokens_count / metrics.total_time_s if metrics.total_time_s > 0 else 0.0
    )
    metrics.acceptance_rate = (
        metrics.total_accepted_tokens / metrics.total_draft_tokens
        if metrics.total_draft_tokens > 0
        else 0.0
    )
    metrics.tokens_per_step = (
        new_tokens_count / metrics.num_verification_cycles
        if metrics.num_verification_cycles > 0
        else 0.0
    )
    metrics.peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    output_text = tokenizer.decode(
        current_ids[0, prompt_len:], skip_special_tokens=True
    )
    return output_text, metrics
