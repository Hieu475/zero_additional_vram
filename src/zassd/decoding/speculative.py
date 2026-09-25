"""Self-speculative decoding implementation with canonical Target KV and Ephemeral Draft KV.

Zero-Additional-VRAM self-speculative decoding using a shared model checkpoint:
1. Prefill phase: Populate canonical Target KV cache with input prompt.
2. Speculative cycle:
   a. Fork ephemeral Draft KV cache from canonical Target KV (zero-copy references).
   b. Draft phase: Generate K candidate tokens with logical layer skipping.
   c. Verify phase: Verify candidate tokens in parallel using full model & canonical Target KV.
   d. Commit / Rollback: Truncate unaccepted candidate tokens, advance bonus/correction token,
      and update canonical Target KV.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from zassd.cache.kv_cache import TargetKVCache
from zassd.controllers.entropy import compute_entropy
from zassd.models.layer_manager import LayerManager

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeMetrics:
    """Comprehensive metrics and cost decomposition for speculative decoding run."""
    total_tokens: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    num_verification_cycles: int = 0
    tokens_per_step: float = 0.0  # E[tokens emitted per cycle]
    total_time_s: float = 0.0
    tokens_per_second: float = 0.0

    # Latency decomposition: T_total = T_prefill + T_draft + T_verify + T_cache + T_controller + T_other
    prefill_time_s: float = 0.0
    draft_time_s: float = 0.0
    verify_time_s: float = 0.0
    cache_time_s: float = 0.0
    controller_time_s: float = 0.0
    other_time_s: float = 0.0

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
    controller: Optional[Any] = None,
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
        k: Default or maximum number of draft tokens per speculation step.
        controller: Optional AdaptiveKController or HardwareAwareJointController.
        max_new_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        device: Device to run generation on.

    Returns:
        Tuple of (generated_text, SpeculativeMetrics).
    """
    metrics = SpeculativeMetrics(k_value=k)

    if isinstance(prompt, torch.Tensor):
        prompt_ids = prompt.to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Canonical Prefill: Populate Target KV Cache
    # -----------------------------------------------------------------------
    t_prefill_start = time.perf_counter()
    target_kv = TargetKVCache()
    with torch.no_grad():
        prefill_out = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)
    target_prefix_logit = prefill_out.logits[0, -1, :]
    torch.cuda.synchronize()
    t_prefill_end = time.perf_counter()
    metrics.prefill_time_s = (t_prefill_end - t_prefill_start)

    generated_token_ids: list[int] = []
    current_prefix_len = prompt_len
    if temperature == 0.0:
        curr_target_tok = int(target_prefix_logit.argmax(dim=-1).item())
    else:
        probs = F.softmax(target_prefix_logit / temperature, dim=-1)
        curr_target_tok = int(torch.multinomial(probs, num_samples=1).item())

    curr_entropy = float(compute_entropy(target_prefix_logit.float()).item())
    last_accepted = 1
    last_proposed = k
    last_draft_ms = 20.0
    last_verify_ms = 25.0

    while len(generated_token_ids) < max_new_tokens:
        # Early exit: if the pending token is EOS, emit it and stop immediately
        # (avoids wasting a draft+verify cycle on hallucinated post-EOS tokens)
        if curr_target_tok == tokenizer.eos_token_id:
            generated_token_ids.append(curr_target_tok)
            break

        rem_tokens = max_new_tokens - len(generated_token_ids)

        if controller is not None:
            t_ctrl_start = time.perf_counter()
            if hasattr(controller, "select_action"):
                action = controller.select_action(
                    entropy=curr_entropy,
                    last_accepted=last_accepted,
                    last_proposed=last_proposed,
                    draft_ms=last_draft_ms,
                    verify_ms=last_verify_ms,
                )
                step_k = action.draft_length
                active_skip = action.skip_indices
            else:
                step_k = controller.update(
                    entropy=curr_entropy,
                    accepted=last_accepted,
                    proposed=last_proposed,
                )
                active_skip = skip_indices
            t_ctrl_end = time.perf_counter()
            metrics.controller_time_s += (t_ctrl_end - t_ctrl_start)
        else:
            step_k = k
            active_skip = skip_indices

        step_k = min(step_k, rem_tokens)
        if step_k <= 0 and rem_tokens <= 0:
            break

        if step_k == 0:
            # Fallback to single-token vanilla step with Target KV
            t_v_start = time.perf_counter()
            curr_tensor = torch.tensor([[curr_target_tok]], device=device)
            with torch.no_grad():
                out = model(curr_tensor, past_key_values=target_kv.cache, use_cache=True)
            torch.cuda.synchronize()
            t_v_end = time.perf_counter()
            verify_elapsed = t_v_end - t_v_start
            metrics.verify_time_s += verify_elapsed
            metrics.num_verification_cycles += 1

            target_logit = out.logits[0, -1, :].float()
            curr_entropy = float(compute_entropy(target_logit).item())
            if temperature == 0.0:
                next_tok = int(target_logit.argmax(dim=-1).item())
            else:
                probs = F.softmax(target_logit / temperature, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())

            generated_token_ids.append(curr_target_tok)
            current_prefix_len += 1
            if curr_target_tok == tokenizer.eos_token_id:
                break
            curr_target_tok = next_tok
            last_accepted = 1
            last_proposed = 0
            last_draft_ms = 0.0
            last_verify_ms = verify_elapsed * 1000
            continue

        # -------------------------------------------------------------------
        # Cache Phase: Fork Ephemeral Draft KV (Zero-copy prefix sharing)
        # -------------------------------------------------------------------
        t_cache_fork_start = time.perf_counter()
        draft_kv = target_kv.fork_ephemeral_draft_kv()
        t_cache_fork_end = time.perf_counter()
        cache_cycle_s = (t_cache_fork_end - t_cache_fork_start)

        # -------------------------------------------------------------------
        # Phase 1: Draft generation (using logical layer skipping)
        # -------------------------------------------------------------------
        draft_tokens: list[int] = []
        draft_probs_list: list[torch.Tensor] = []

        t_draft_start = time.perf_counter()
        with torch.no_grad():
            with layer_mgr.skip_layers(active_skip):
                curr_d = torch.tensor([[curr_target_tok]], device=device)
                for d_step in range(step_k):
                    d_out = model(curr_d, past_key_values=draft_kv, use_cache=True)
                    d_logits = d_out.logits[0, -1, :].float()

                    if temperature == 0.0:
                        d_next = int(d_logits.argmax(dim=-1).item())
                    else:
                        d_probs = F.softmax(d_logits / temperature, dim=-1)
                        d_next = int(torch.multinomial(d_probs, num_samples=1).item())
                        draft_probs_list.append(d_probs)

                    draft_tokens.append(d_next)
                    curr_d = torch.tensor([[d_next]], device=device)

                    if d_next == tokenizer.eos_token_id:
                        break

        torch.cuda.synchronize()
        t_draft_end = time.perf_counter()
        draft_elapsed = t_draft_end - t_draft_start
        metrics.draft_time_s += draft_elapsed
        metrics.total_draft_tokens += len(draft_tokens)

        actual_k = len(draft_tokens)

        # -------------------------------------------------------------------
        # Phase 2: Parallel target verification (using full model)
        # Verify inputs: [curr_target_tok] + draft_tokens (length actual_k + 1)
        # Directly computes all verification logits and the bonus token in 1 pass.
        # -------------------------------------------------------------------
        verify_inputs = [curr_target_tok] + draft_tokens
        cand_tensor = torch.tensor([verify_inputs], device=device)

        t_verify_start = time.perf_counter()
        with torch.no_grad():
            v_out = model(cand_tensor, past_key_values=target_kv.cache, use_cache=True)

        torch.cuda.synchronize()
        t_verify_end = time.perf_counter()
        verify_elapsed = t_verify_end - t_verify_start
        metrics.verify_time_s += verify_elapsed
        metrics.num_verification_cycles += 1

        # -------------------------------------------------------------------
        # Phase 3: Accept / Reject & Target KV Crop / Advance
        # -------------------------------------------------------------------
        t_commit_start = time.perf_counter()
        cycle_emitted = [curr_target_tok]
        rejected_at: Optional[int] = None
        num_accepted_draft = 0

        if temperature == 0.0:
            for i in range(actual_k):
                pred = int(v_out.logits[0, i, :].argmax(dim=-1).item())
                if pred == draft_tokens[i]:
                    cycle_emitted.append(draft_tokens[i])
                    num_accepted_draft += 1
                else:
                    # Mismatch at candidate i
                    target_kv.crop(current_prefix_len + len(cycle_emitted))
                    curr_target_tok = pred
                    rejected_at = i
                    break
        else:
            target_probs_tensor = F.softmax(v_out.logits[0, :actual_k, :] / temperature, dim=-1)
            for i in range(actual_k):
                t_prob = target_probs_tensor[i]
                d_prob = draft_probs_list[i]
                tok_id = draft_tokens[i]

                p_t = t_prob[tok_id].item()
                p_d = d_prob[tok_id].item()

                if torch.rand(1).item() < min(1.0, p_t / max(p_d, 1e-8)):
                    cycle_emitted.append(tok_id)
                    num_accepted_draft += 1
                else:
                    target_kv.crop(current_prefix_len + len(cycle_emitted))
                    residual_probs = torch.clamp(t_prob - d_prob, min=0.0)
                    res_sum = residual_probs.sum()
                    if res_sum > 0:
                        residual_probs = residual_probs / res_sum
                        curr_target_tok = int(torch.multinomial(residual_probs, num_samples=1).item())
                    else:
                        curr_target_tok = int(t_prob.argmax().item())
                    rejected_at = i
                    break

        if rejected_at is None:
            # All actual_k draft tokens accepted!
            # The bonus token is already computed at index actual_k of v_out.logits.
            # Zero additional model forward passes needed!
            if temperature == 0.0:
                bonus_tok = int(v_out.logits[0, actual_k, :].argmax(dim=-1).item())
            else:
                b_probs = F.softmax(v_out.logits[0, actual_k, :] / temperature, dim=-1)
                bonus_tok = int(torch.multinomial(b_probs, num_samples=1).item())
            curr_target_tok = bonus_tok

        torch.cuda.synchronize()
        t_commit_end = time.perf_counter()
        commit_elapsed = t_commit_end - t_commit_start
        cache_cycle_s += commit_elapsed
        metrics.cache_time_s += cache_cycle_s

        if rejected_at is not None:
            curr_entropy = float(compute_entropy(v_out.logits[0, rejected_at, :].float()).item())
        else:
            curr_entropy = float(compute_entropy(v_out.logits[0, actual_k, :].float()).item())

        last_accepted = num_accepted_draft
        last_proposed = actual_k
        last_draft_ms = draft_elapsed * 1000
        last_verify_ms = verify_elapsed * 1000

        metrics.total_accepted_tokens += num_accepted_draft
        generated_token_ids.extend(cycle_emitted)
        current_prefix_len += len(cycle_emitted)

        iter_stat = {
            "k": actual_k,
            "accepted": num_accepted_draft,
            "rejected_at": rejected_at,
            "emitted_count": len(cycle_emitted),
            "draft_time_ms": draft_elapsed * 1000,
            "verify_time_ms": verify_elapsed * 1000,
            "cache_time_ms": cache_cycle_s * 1000,
            "skip_indices": active_skip,
            "entropy": curr_entropy,
        }
        metrics.per_iteration_stats.append(iter_stat)

        if any(t == tokenizer.eos_token_id for t in cycle_emitted):
            break

    # If loop finished and still room for the final pending curr_target_tok
    if len(generated_token_ids) < max_new_tokens and curr_target_tok != tokenizer.eos_token_id:
        if not (generated_token_ids and generated_token_ids[-1] == tokenizer.eos_token_id):
            generated_token_ids.append(curr_target_tok)

    torch.cuda.synchronize()
    total_end = time.perf_counter()
    metrics.total_time_s = total_end - total_start

    # Trim to exact max_new_tokens boundary
    if len(generated_token_ids) > max_new_tokens:
        generated_token_ids = generated_token_ids[:max_new_tokens]

    new_tokens_count = len(generated_token_ids)
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
    metrics.other_time_s = max(
        0.0,
        metrics.total_time_s
        - (metrics.prefill_time_s + metrics.draft_time_s + metrics.verify_time_s + metrics.cache_time_s + metrics.controller_time_s),
    )
    metrics.peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    output_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return output_text, metrics
