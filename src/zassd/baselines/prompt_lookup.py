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


from zassd.decoding.ngram import find_candidate_match, find_candidate_tokens


def force_candidate_miss(match):
    """Keep match metadata while removing all draft candidates for a control."""
    return type(match)([], "forced_miss", match.ngram_size, 0, 0.0)


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
    k_policy: str = "fixed",
    candidate_source: str = "history",
    adaptive_profile: dict | None = None,
    force_miss: bool = False,
    sequential_verify: bool = False,
    kv_cache_max_capacity: int = 2048,
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
        kv_cache_max_capacity: Static-cache capacity for long prompts plus decode.

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
    target_kv = TargetKVCache(backend=kv_cache_backend,
                              max_capacity=kv_cache_max_capacity)
    with torch.no_grad():
        prefill_out = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)
    target_prefix_logit = prefill_out.logits[0, -1, :]
    torch.cuda.synchronize()
    t_prefill_end = time.perf_counter()
    metrics.prefill_time_s = (t_prefill_end - t_prefill_start)

    if k_policy not in {"fixed", "confidence_cost"}:
        raise ValueError("k_policy must be 'fixed' or 'confidence_cost'")
    if sequential_verify and temperature != 0.0:
        raise ValueError("sequential verification currently supports greedy decoding only")
    if candidate_source not in {"history", "prompt", "prompt+generated"}:
        raise ValueError("candidate_source must be 'history', 'prompt', or 'prompt+generated'")
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
    prompt_token_list = full_history.copy()

    measured_verify_ms: dict[int, float] = {}
    measured_vanilla_ms: list[float] = []
    while len(generated_token_ids) < max_new_tokens:
        if curr_target_tok == tokenizer.eos_token_id:
            generated_token_ids.append(curr_target_tok)
            break

        rem_tokens = max_new_tokens - len(generated_token_ids)
        step_k = min(k, max(0, rem_tokens - 1))

        # Lookup candidate tokens from history
        full_tokens = full_history + [curr_target_tok]
        t_draft_start = time.perf_counter()
        generated_list = full_history[prompt_len:]
        match = find_candidate_match(
            full_tokens,
            prompt_token_list if candidate_source != "history" else full_tokens,
            generated_list if candidate_source == "prompt+generated" else [],
            ngram_size=ngram_size, max_candidates=step_k,
            include_generated=(candidate_source == "prompt+generated"),
        )
        if force_miss:
            # Negative control: retain lookup and fallback overhead while
            # suppressing every speculative candidate.
            match = force_candidate_miss(match)
        if candidate_source == "history" and match.candidates:
            match = type(match)(match.candidates, "history", match.ngram_size,
                                match.repetitions, match.confidence)
        candidate_tokens = match.candidates
        lookup_so_far_ms = (time.perf_counter() - t_draft_start) * 1000.0
        if k_policy == "confidence_cost" and candidate_tokens:
            # Select the horizon with the best measured cost estimate. The
            # confidence score is an input feature; calibration is reported
            # separately by the paired development-set sweep.
            available = len(candidate_tokens)
            best_k, best_rate = 0, 0.0
            vanilla_ms = ((adaptive_profile or {}).get("vanilla_token_ms")
                          or (sum(measured_vanilla_ms[-8:]) / len(measured_vanilla_ms[-8:])
                              if measured_vanilla_ms else None))
            for candidate_k in range(1, available + 1):
                verify_ms = (adaptive_profile or {}).get("verify_ms_by_k", {}).get(str(candidate_k))
                verify_ms = verify_ms or measured_verify_ms.get(candidate_k)
                if verify_ms is None and measured_verify_ms:
                    nearest = min(measured_verify_ms, key=lambda x: abs(x - candidate_k))
                    verify_ms = measured_verify_ms[nearest] * candidate_k / nearest
                if verify_ms is None:
                    continue
                profiles = (adaptive_profile or {}).get("acceptance_by_feature_k", {})
                rep_bin = min(match.repetitions, 3)
                feature_key = f"{match.ngram_size}:{rep_bin}:{candidate_k}"
                order_key = f"{match.ngram_size}:*:{candidate_k}"
                global_key = f"*:*:{candidate_k}"
                accepted_est = profiles.get(feature_key, profiles.get(order_key,
                                profiles.get(global_key)))
                if accepted_est is None:
                    alpha = min(0.95, max(0.05, match.confidence))
                    accepted_est = sum(alpha ** i for i in range(1, candidate_k + 1))
                cycle_ms = lookup_so_far_ms + float(verify_ms)
                rate = (1.0 + accepted_est) / cycle_ms if cycle_ms > 0 else 0.0
                if rate > best_rate:
                    best_k, best_rate = candidate_k, rate
            if vanilla_ms and best_rate <= 1.0 / vanilla_ms:
                candidate_tokens = []
            elif best_k:
                candidate_tokens = candidate_tokens[:best_k]
        if tokenizer.eos_token_id in candidate_tokens:
            candidate_tokens = candidate_tokens[:candidate_tokens.index(tokenizer.eos_token_id) + 1]
        t_draft_end = time.perf_counter()
        draft_elapsed = (t_draft_end - t_draft_start)
        metrics.draft_time_s += draft_elapsed

        actual_k = len(candidate_tokens)
        cycle_lookup_ms = draft_elapsed * 1000.0
        cycle_verify_ms = 0.0
        cycle_crop_ms = 0.0
        cycle_sync_ms = 0.0

        if actual_k == 0:
            # Fallback single-token forward pass
            t_v_start = time.perf_counter()
            with torch.no_grad():
                out = model(curr_target_tok_tensor, past_key_values=target_kv.cache, use_cache=True)
            t_sync_start = time.perf_counter()
            torch.cuda.synchronize()
            t_sync_end = time.perf_counter()
            cycle_sync_ms += (t_sync_end - t_sync_start) * 1000.0
            t_v_end = time.perf_counter()
            verify_elapsed = t_v_end - t_v_start
            cycle_verify_ms = verify_elapsed * 1000.0
            metrics.verify_time_s += verify_elapsed
            measured_vanilla_ms.append(cycle_verify_ms)
            metrics.num_verification_cycles += 1
            t_decision = time.perf_counter()
            if temperature == 0.0:
                next_tok_tensor = out.logits[:, -1:, :].argmax(dim=-1)
                next_tok = int(next_tok_tensor.item())
                curr_target_tok_tensor = next_tok_tensor
            else:
                target_logit = out.logits[0, -1, :].float()
                probs = F.softmax(target_logit / temperature, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())
                curr_target_tok_tensor = torch.tensor([[next_tok]], device=device)

            metrics.per_iteration_stats.append({
                "k": 0, "accepted": 0, "rejected_at": None,
                "lookup_ms": cycle_lookup_ms, "verify_ms": cycle_verify_ms,
                "crop_ms": 0.0,
                "decision_ms": (time.perf_counter() - t_decision) * 1000.0,
                "sync_wait_ms": cycle_sync_ms,
                "match_source": match.source, "match_order": match.ngram_size,
                "match_repetitions": match.repetitions,
                "match_confidence": match.confidence,
                "context_len": current_prefix_len,
            })
            generated_token_ids.append(curr_target_tok)
            full_history.append(curr_target_tok)
            current_prefix_len += 1
            if curr_target_tok == tokenizer.eos_token_id:
                break
            curr_target_tok = next_tok
            continue

        metrics.total_draft_tokens += actual_k
        metrics.pld_cycles += 1

        # Parallel verification with TargetKVCache
        cand_ids = [curr_target_tok] + candidate_tokens
        cand_tensor = torch.tensor([cand_ids], device=device)

        t_verify_start = time.perf_counter()
        if sequential_verify:
            # Advance the target cache one token at a time. Each row's argmax
            # is then computed from the exact canonical prefix, and a rejected
            # candidate is never committed to the cache.
            sequential_logits = []
            with torch.no_grad():
                for index, token in enumerate(cand_ids):
                    token_tensor = torch.tensor([[token]], device=device)
                    sequential_out = model(token_tensor, past_key_values=target_kv.cache,
                                           use_cache=True)
                    sequential_logits.append(sequential_out.logits[:, -1:, :])
                    if index < actual_k:
                        observed_next = int(sequential_out.logits[:, -1, :].argmax(dim=-1).item())
                        if observed_next != candidate_tokens[index]:
                            break
            verify_logits = torch.cat(sequential_logits, dim=1)
        else:
            with torch.no_grad():
                v_out = model(cand_tensor, past_key_values=target_kv.cache, use_cache=True)
            verify_logits = v_out.logits
        t_sync_start = time.perf_counter()
        torch.cuda.synchronize()
        t_sync_end = time.perf_counter()
        cycle_sync_ms += (t_sync_end - t_sync_start) * 1000.0
        t_verify_end = time.perf_counter()
        verify_elapsed = t_verify_end - t_verify_start
        metrics.verify_time_s += verify_elapsed
        cycle_verify_ms = verify_elapsed * 1000.0
        measured_verify_ms[actual_k] = cycle_verify_ms
        metrics.num_verification_cycles += 1

        # Phase 3: Accept / Reject & Target KV Crop
        t_decision = time.perf_counter()
        cycle_emitted = [curr_target_tok]
        rejected_at: Optional[int] = None
        num_accepted_draft = 0

        if temperature == 0.0:
            v_preds = verify_logits[0, :actual_k, :].argmax(dim=-1)
            d_preds = torch.tensor(candidate_tokens, device=device)
            # Sequential verification stops at the first rejection, so only
            # the candidate prefix that was actually evaluated has logits.
            matches = (v_preds == d_preds[:v_preds.shape[0]])

            if bool(matches.all().item()):
                num_accepted_draft = actual_k
                cycle_emitted.extend(candidate_tokens)
                bonus_tok_tensor = verify_logits[:, actual_k:, :].argmax(dim=-1)
                curr_target_tok_tensor = bonus_tok_tensor
                curr_target_tok = int(bonus_tok_tensor.item())
            else:
                mismatch_idx = int((~matches).nonzero()[0].item())
                num_accepted_draft = mismatch_idx
                if mismatch_idx > 0:
                    cycle_emitted.extend(candidate_tokens[:mismatch_idx])
                t_crop = time.perf_counter()
                target_kv.crop(current_prefix_len + len(cycle_emitted))
                cycle_crop_ms += (time.perf_counter() - t_crop) * 1000.0
                rej_tok_tensor = v_preds[mismatch_idx].view(1, 1)
                curr_target_tok_tensor = rej_tok_tensor
                curr_target_tok = int(rej_tok_tensor.item())
                rejected_at = mismatch_idx
        else:
            target_probs_tensor = F.softmax(verify_logits[0, :actual_k, :] / temperature, dim=-1)
            for i in range(actual_k):
                t_prob = target_probs_tensor[i]
                tok_id = candidate_tokens[i]
                p_t = t_prob[tok_id].item()

                if torch.rand(1).item() < p_t:
                    cycle_emitted.append(tok_id)
                    num_accepted_draft += 1
                else:
                    t_crop = time.perf_counter()
                    target_kv.crop(current_prefix_len + len(cycle_emitted))
                    cycle_crop_ms += (time.perf_counter() - t_crop) * 1000.0
                    curr_target_tok = int(t_prob.argmax().item())
                    curr_target_tok_tensor = torch.tensor([[curr_target_tok]], device=device)
                    rejected_at = i
                    break

            if rejected_at is None:
                b_probs = F.softmax(verify_logits[0, actual_k, :] / temperature, dim=-1)
                bonus_tok = int(torch.multinomial(b_probs, num_samples=1).item())
                curr_target_tok = bonus_tok
                curr_target_tok_tensor = torch.tensor([[curr_target_tok]], device=device)

        metrics.total_accepted_tokens += num_accepted_draft
        generated_token_ids.extend(cycle_emitted)
        full_history.extend(cycle_emitted)
        current_prefix_len += len(cycle_emitted)
        metrics.per_iteration_stats.append({
            "k": actual_k, "accepted": num_accepted_draft,
            "rejected_at": rejected_at, "lookup_ms": cycle_lookup_ms,
            "verify_ms": cycle_verify_ms, "crop_ms": cycle_crop_ms,
            "decision_ms": (time.perf_counter() - t_decision) * 1000.0,
            "sync_wait_ms": cycle_sync_ms, "match_source": match.source,
            "match_order": match.ngram_size,
            "match_repetitions": match.repetitions,
            "match_confidence": match.confidence,
            "verification_mode": "sequential" if sequential_verify else "batched",
            "context_len": current_prefix_len - len(cycle_emitted),
        })

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

    metrics.generated_token_ids = generated_token_ids

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
