"""Prompt Lookup Decoding (PLD) baseline.

Zero-Additional-Weight-VRAM speculative decoding using prompt and history n-gram matching.
Reference: Saxena (2023), "Prompt Lookup Decoding".

Extracts candidate tokens directly from the input prompt and generation history
via exact n-gram matching, then verifies them in parallel with TargetKVCache.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.speculative import SpeculativeMetrics

logger = logging.getLogger(__name__)


from zassd.decoding.ngram import find_candidate_tokens


def prompt_lookup_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str | torch.Tensor,
    k: int = 4,
    ngram_size: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    device: str = "cuda:0",
    kv_cache_backend: str = "static",
) -> tuple[str, SpeculativeMetrics]:
    """Generate text using Prompt Lookup Decoding (PLD).

    Args:
        model: Target language model.
        tokenizer: Tokenizer.
        prompt: Text or tokenized tensor input prompt.
        k: Maximum number of speculative candidate tokens per cycle.
        ngram_size: Context n-gram size used for lookup.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0.0 = greedy).
        device: Execution device.
        kv_cache_backend: Cache backend ('static' or 'dynamic').

    Returns:
        Tuple of (generated text, SpeculativeMetrics).
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

    # 1. Prefill Target KV Cache
    t_prefill_start = time.perf_counter()
    target_kv = TargetKVCache(backend=kv_cache_backend)
    with torch.no_grad():
        prefill_out = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)
    target_prefix_logit = prefill_out.logits[0, -1, :]
    torch.cuda.synchronize()
    t_prefill_end = time.perf_counter()
    metrics.prefill_time_s = (t_prefill_end - t_prefill_start)

    generated_token_ids: list[int] = []
    current_prefix_len = prompt_len
    if temperature == 0.0:
        curr_target_tok_tensor = prefill_out.logits[:, -1:, :].argmax(dim=-1)
        curr_target_tok = int(curr_target_tok_tensor.item())
    else:
        probs = F.softmax(target_prefix_logit / temperature, dim=-1)
        curr_target_tok = int(torch.multinomial(probs, num_samples=1).item())
        curr_target_tok_tensor = torch.tensor([[curr_target_tok]], device=device)

    # Full history tracking for n-gram lookup
    full_history = prompt_ids[0].tolist()

    while len(generated_token_ids) < max_new_tokens:
        if curr_target_tok == tokenizer.eos_token_id:
            generated_token_ids.append(curr_target_tok)
            break

        rem_tokens = max_new_tokens - len(generated_token_ids)
        step_k = min(k, rem_tokens)

        # Lookup candidate tokens from history
        full_tokens = full_history + [curr_target_tok]
        t_draft_start = time.perf_counter()
        candidate_tokens = find_candidate_tokens(full_tokens, ngram_size=ngram_size, max_candidates=step_k)
        t_draft_end = time.perf_counter()
        draft_elapsed = (t_draft_end - t_draft_start)
        metrics.draft_time_s += draft_elapsed

        actual_k = len(candidate_tokens)

        if actual_k == 0:
            # Fallback single-token forward pass
            t_v_start = time.perf_counter()
            with torch.no_grad():
                out = model(curr_target_tok_tensor, past_key_values=target_kv.cache, use_cache=True)
            torch.cuda.synchronize()
            t_v_end = time.perf_counter()
            verify_elapsed = t_v_end - t_v_start
            metrics.verify_time_s += verify_elapsed
            metrics.num_verification_cycles += 1

            if temperature == 0.0:
                next_tok_tensor = out.logits[:, -1:, :].argmax(dim=-1)
                next_tok = int(next_tok_tensor.item())
                curr_target_tok_tensor = next_tok_tensor
            else:
                target_logit = out.logits[0, -1, :].float()
                probs = F.softmax(target_logit / temperature, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())
                curr_target_tok_tensor = torch.tensor([[next_tok]], device=device)

            generated_token_ids.append(curr_target_tok)
            full_history.append(curr_target_tok)
            current_prefix_len += 1
            if curr_target_tok == tokenizer.eos_token_id:
                break
            curr_target_tok = next_tok
            continue

        metrics.total_draft_tokens += actual_k

        # Parallel verification with TargetKVCache
        cand_ids = [curr_target_tok] + candidate_tokens
        cand_tensor = torch.tensor([cand_ids], device=device)

        t_verify_start = time.perf_counter()
        with torch.no_grad():
            v_out = model(cand_tensor, past_key_values=target_kv.cache, use_cache=True)
        torch.cuda.synchronize()
        t_verify_end = time.perf_counter()
        verify_elapsed = t_verify_end - t_verify_start
        metrics.verify_time_s += verify_elapsed
        metrics.num_verification_cycles += 1

        # Phase 3: Accept / Reject & Target KV Crop
        cycle_emitted = [curr_target_tok]
        rejected_at: Optional[int] = None
        num_accepted_draft = 0

        if temperature == 0.0:
            v_preds = v_out.logits[0, :actual_k, :].argmax(dim=-1)
            d_preds = torch.tensor(candidate_tokens, device=device)
            matches = (v_preds == d_preds)

            if bool(matches.all().item()):
                num_accepted_draft = actual_k
                cycle_emitted.extend(candidate_tokens)
                bonus_tok_tensor = v_out.logits[:, actual_k:, :].argmax(dim=-1)
                curr_target_tok_tensor = bonus_tok_tensor
                curr_target_tok = int(bonus_tok_tensor.item())
            else:
                mismatch_idx = int((~matches).nonzero()[0].item())
                num_accepted_draft = mismatch_idx
                if mismatch_idx > 0:
                    cycle_emitted.extend(candidate_tokens[:mismatch_idx])
                target_kv.crop(current_prefix_len + len(cycle_emitted))
                rej_tok_tensor = v_preds[mismatch_idx].view(1, 1)
                curr_target_tok_tensor = rej_tok_tensor
                curr_target_tok = int(rej_tok_tensor.item())
                rejected_at = mismatch_idx
        else:
            target_probs_tensor = F.softmax(v_out.logits[0, :actual_k, :] / temperature, dim=-1)
            for i in range(actual_k):
                t_prob = target_probs_tensor[i]
                tok_id = candidate_tokens[i]
                p_t = t_prob[tok_id].item()

                if torch.rand(1).item() < p_t:
                    cycle_emitted.append(tok_id)
                    num_accepted_draft += 1
                else:
                    target_kv.crop(current_prefix_len + len(cycle_emitted))
                    curr_target_tok = int(t_prob.argmax().item())
                    curr_target_tok_tensor = torch.tensor([[curr_target_tok]], device=device)
                    rejected_at = i
                    break

            if rejected_at is None:
                b_probs = F.softmax(v_out.logits[0, actual_k, :] / temperature, dim=-1)
                bonus_tok = int(torch.multinomial(b_probs, num_samples=1).item())
                curr_target_tok = bonus_tok
                curr_target_tok_tensor = torch.tensor([[curr_target_tok]], device=device)

        metrics.total_accepted_tokens += num_accepted_draft
        generated_token_ids.extend(cycle_emitted)
        full_history.extend(cycle_emitted)
        current_prefix_len += len(cycle_emitted)

        if any(t == tokenizer.eos_token_id for t in cycle_emitted):
            break

    if len(generated_token_ids) < max_new_tokens and curr_target_tok != tokenizer.eos_token_id:
        if not (generated_token_ids and generated_token_ids[-1] == tokenizer.eos_token_id):
            generated_token_ids.append(curr_target_tok)

    torch.cuda.synchronize()
    total_end = time.perf_counter()
    metrics.total_time_s = total_end - total_start

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
    metrics.peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    output_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return output_text, metrics
