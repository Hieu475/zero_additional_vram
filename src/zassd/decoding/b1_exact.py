"""B1 correctness baseline: a shape-invariant *execution schedule*.

Every target transition consumes exactly one token with the same model object,
cache class, and forward call as the pinned greedy reference. PLD only proposes
token IDs; it never changes target arithmetic. This establishes trajectory
equivalence by induction, conditional on deterministic one-token forwards.

This is deliberately a correctness baseline, not a parallel 4-bit kernel or a
speedup claim. A future batched B1 kernel may replace ``one_token_transition``
only after proving both its logits and relevant KV state equivalent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from zassd.cache.kv_cache import TargetKVCache
from zassd.decoding.ngram import find_candidate_tokens


@dataclass
class B1Metrics:
    generated_token_ids: list[int] = field(default_factory=list)
    prefill_time_s: float = 0.0
    draft_time_s: float = 0.0
    verify_time_s: float = 0.0
    total_time_s: float = 0.0
    target_forwards: int = 0
    target_rows: int = 0
    reference_schedule: str = "singleton"
    physical_rows: int = 1
    draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    cycles: int = 0
    peak_vram_mb: float | None = None
    cycle_trace: list[dict] = field(default_factory=list)
    capability_report: dict | None = None


def one_token_transition(model, token: torch.Tensor, cache: TargetKVCache) -> torch.Tensor:
    """Return the next greedy token while advancing canonical KV by one row."""
    if token.ndim != 2 or token.shape != (1, 1):
        raise ValueError("B1 target transition requires a single token row")
    with torch.no_grad():
        logits = model(token, past_key_values=cache.cache, use_cache=True).logits
    if logits.shape[0] != 1 or logits.shape[1] != 1:
        raise ValueError("target model returned a non-singleton logit row")
    return logits[:, -1:, :].argmax(dim=-1)


def b1_exact_generate(
    model,
    tokenizer,
    prompt: str | torch.Tensor,
    *,
    k: int = 4,
    ngram_size: int = 3,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    kv_cache_max_capacity: int = 2048,
    trace: bool = False,
    state_observer: Callable[[int, TargetKVCache], None] | None = None,
) -> tuple[str, B1Metrics]:
    """Greedy-equivalent PLD under the B1 singleton target schedule.

    The reference is prompt prefill on ``TargetKVCache(static)`` followed by
    ``one_token_transition`` for each emitted nonfinal token. Draft candidates
    are CPU token IDs from the already committed history. The pending token is
    always selected by the reference transition before it can be emitted.
    """
    if k < 0 or ngram_size < 2 or max_new_tokens < 1:
        raise ValueError("require k >= 0, ngram_size >= 2, max_new_tokens >= 1")
    if isinstance(prompt, torch.Tensor):
        prompt_ids = prompt.to(device)
    else:
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
        raise ValueError("B1 requires one nonempty prompt")
    if prompt_ids.shape[1] + max_new_tokens > kv_cache_max_capacity:
        raise ValueError("prompt plus output budget exceeds static KV capacity")

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    metrics = B1Metrics()
    cache = TargetKVCache(backend="static", max_capacity=kv_cache_max_capacity)
    prefill_started = time.perf_counter()
    with torch.no_grad():
        prefill_logits = model(prompt_ids, past_key_values=cache.cache, use_cache=True).logits
    pending_tensor = prefill_logits[:, -1:, :].argmax(dim=-1)
    pending = int(pending_tensor.item())
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    metrics.prefill_time_s = time.perf_counter() - prefill_started
    if state_observer is not None:
        state_observer(0, cache)

    history = prompt_ids[0].tolist()
    emitted: list[int] = []
    eos = tokenizer.eos_token_id
    while len(emitted) < max_new_tokens:
        if pending == eos or len(emitted) + 1 == max_new_tokens:
            emitted.append(pending)
            break

        draft_started = time.perf_counter()
        candidates = find_candidate_tokens(
            history + [pending], ngram_size=ngram_size,
            max_candidates=min(k, max_new_tokens - len(emitted) - 1),
        )
        if eos in candidates:
            candidates = candidates[:candidates.index(eos) + 1]
        metrics.draft_time_s += time.perf_counter() - draft_started
        metrics.draft_tokens += len(candidates)
        metrics.cycles += 1
        cycle_start = len(emitted)
        cache_len_before = cache.get_seq_length()

        # This transition is exactly the next reference call. It consumes the
        # already certified pending token, then chooses the following token.
        verify_started = time.perf_counter()
        next_tensor = one_token_transition(model, pending_tensor, cache)
        metrics.target_forwards += 1
        metrics.target_rows += 1
        emitted.append(pending)
        history.append(pending)
        next_id = int(next_tensor.item())
        if state_observer is not None:
            state_observer(len(emitted), cache)
        accepted = 0
        rejected_at = None

        for position, candidate in enumerate(candidates):
            if next_id != candidate:
                rejected_at = position
                break
            # The candidate equals a token already chosen by the reference.
            emitted.append(candidate)
            history.append(candidate)
            accepted += 1
            metrics.accepted_draft_tokens += 1
            if candidate == eos or len(emitted) == max_new_tokens:
                break
            next_tensor = one_token_transition(model, next_tensor, cache)
            metrics.target_forwards += 1
            metrics.target_rows += 1
            next_id = int(next_tensor.item())
            if state_observer is not None:
                state_observer(len(emitted), cache)

        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        metrics.verify_time_s += time.perf_counter() - verify_started
        if trace:
            metrics.cycle_trace.append({
                "output_start": cycle_start,
                "pending": pending,
                "draft_tokens": candidates,
                "accepted": accepted,
                "rejected_at": rejected_at,
                "emitted_tokens": emitted[cycle_start:],
                "cache_len_before": cache_len_before,
                "cache_len_after": cache.get_seq_length(),
                "next_pending": None if emitted[-1] == eos or len(emitted) == max_new_tokens else next_id,
            })
        if emitted[-1] == eos or len(emitted) == max_new_tokens:
            break
        pending_tensor = next_tensor
        pending = next_id

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        metrics.peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
    metrics.generated_token_ids = emitted
    metrics.total_time_s = time.perf_counter() - started
    return tokenizer.decode(emitted, skip_special_tokens=True), metrics


def b1_fixed_row_block_generate(
    model,
    tokenizer,
    prompt: str | torch.Tensor,
    *,
    k: int = 4,
    ngram_size: int = 3,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    kv_cache_max_capacity: int = 2048,
    physical_rows: int = 9,
    attention_strategy: str = "rowwise",
    reuse_linear_padding_buffer: bool = False,
    quantized_linear_strategy: str = "fixed_rows",
    trace: bool = False,
    state_observer: Callable[[int, TargetKVCache], None] | None = None,
) -> tuple[str, B1Metrics]:
    """Experimental architecture-specific fixed-row block verifier.

    The singleton schedule remains the correctness reference. The temporary
    fixed-row operator patch attempts to make M<=K+1 block outputs and the
    committed KV prefix bitwise equal to that reference. ``attention_strategy``
    is part of that schedule. It is experimental:
    until the operator contract is formally audited, callers must treat this
    as a research prototype and compare its output to
    ``b1_fixed_row_generate`` with the same physical row count.
    """
    from zassd.certification.model_capabilities import require_b1_fixed_row_capability
    from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model

    if k < 1 or ngram_size < 2 or max_new_tokens < 1:
        raise ValueError("require k >= 1, ngram_size >= 2, max_new_tokens >= 1")
    if physical_rows < k + 1:
        raise ValueError("physical_rows must cover pending token plus K draft tokens")
    capability_report = require_b1_fixed_row_capability(
        model, attention_strategy=attention_strategy)
    if isinstance(prompt, torch.Tensor):
        prompt_ids = prompt.to(device)
    else:
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1:
        raise ValueError("B1 requires one nonempty prompt")
    if prompt_ids.shape[1] + max_new_tokens > kv_cache_max_capacity:
        raise ValueError("prompt plus output budget exceeds static KV capacity")

    metrics = B1Metrics()
    metrics.reference_schedule = f"{model.config.model_type}_fixed_row_{attention_strategy}"
    if reuse_linear_padding_buffer:
        metrics.reference_schedule += "_reused_linear_padding"
    if quantized_linear_strategy != "fixed_rows":
        metrics.reference_schedule += f"_quantized_linear_{quantized_linear_strategy}"
    metrics.physical_rows = physical_rows
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    cache = TargetKVCache(backend="static", max_capacity=kv_cache_max_capacity)
    history = prompt_ids[0].tolist()
    emitted: list[int] = []
    eos = tokenizer.eos_token_id

    with fixed_row_schedule_for_model(
        model, physical_rows=physical_rows, attention_strategy=attention_strategy,
        reuse_linear_padding_buffer=reuse_linear_padding_buffer,
        quantized_linear_strategy=quantized_linear_strategy,
    ):
        prefill_started = time.perf_counter()
        with torch.no_grad():
            prefill = model(prompt_ids, past_key_values=cache.cache, use_cache=True)
        pending_tensor = prefill.logits[:, -1:, :].argmax(dim=-1)
        pending = int(pending_tensor.item())
        # ``item`` has already waited for the prefill logits.
        metrics.prefill_time_s = time.perf_counter() - prefill_started
        if state_observer is not None:
            state_observer(0, cache)

        while len(emitted) < max_new_tokens:
            if pending == eos or len(emitted) + 1 == max_new_tokens:
                emitted.append(pending)
                break

            cycle_started = time.perf_counter()
            cycle_start = len(emitted)
            draft_started = time.perf_counter()
            candidates = find_candidate_tokens(
                history + [pending], ngram_size=ngram_size,
                max_candidates=min(k, max_new_tokens - len(emitted) - 1),
            )
            if eos in candidates:
                candidates = candidates[:candidates.index(eos) + 1]
            draft_elapsed = time.perf_counter() - draft_started
            metrics.draft_time_s += draft_elapsed
            metrics.draft_tokens += len(candidates)
            metrics.cycles += 1

            prefix_len = cache.get_seq_length()
            # Each row predicts one candidate. A final non-EOS candidate also
            # gets consumed to produce the bonus token. EOS must never be fed.
            has_eos_candidate = bool(candidates and candidates[-1] == eos)
            verify_inputs = [pending] + (candidates[:-1] if has_eos_candidate else candidates)
            verify_tensor = torch.tensor([verify_inputs], device=device, dtype=prompt_ids.dtype)
            verify_started = time.perf_counter()
            with torch.no_grad():
                logits = model(verify_tensor, past_key_values=cache.cache,
                               use_cache=True).logits[0]
            metrics.target_forwards += 1
            metrics.target_rows += int(verify_tensor.shape[1])
            target_predictions = logits.argmax(dim=-1).tolist()
            # ``tolist`` synchronizes the prediction tensor once for host-side
            # acceptance; an explicit synchronize here would wait a second time.
            verify_elapsed = time.perf_counter() - verify_started
            metrics.verify_time_s += verify_elapsed

            commit_started = time.perf_counter()
            emitted.append(pending)
            history.append(pending)
            rejected_at = None
            accepted = 0
            for position, candidate in enumerate(candidates):
                if target_predictions[position] != candidate:
                    rejected_at = position
                    break
                emitted.append(candidate)
                history.append(candidate)
                accepted += 1
                metrics.accepted_draft_tokens += 1
                if candidate == eos:
                    break

            if rejected_at is not None:
                # Only pending + accepted candidate tokens were consumed by the
                # reference path; discard all later speculative KV writes.
                cache.crop(prefix_len + rejected_at + 1)
                pending = target_predictions[rejected_at]
                pending_tensor = torch.tensor([[pending]], device=device, dtype=prompt_ids.dtype)
            elif has_eos_candidate and accepted == len(candidates):
                # The EOS row was predicted, never consumed.
                # Continue through trace/terminal bookkeeping before exiting.
                pass
            elif accepted == len(candidates):
                # All block inputs were committed; the last row is the bonus.
                pending = target_predictions[-1]
                pending_tensor = torch.tensor([[pending]], device=device, dtype=prompt_ids.dtype)
            else:
                # Output budget ended inside an accepted prefix. The cache may
                # include the final token only when it was actually consumed.
                cache.crop(prefix_len + len(emitted) - cycle_start)
            commit_elapsed = time.perf_counter() - commit_started
            cycle_elapsed = time.perf_counter() - cycle_started

            terminal_cycle = emitted[-1] == eos or len(emitted) == max_new_tokens
            if state_observer is not None and not terminal_cycle:
                state_observer(cache.get_seq_length() - int(prompt_ids.shape[1]), cache)

            if trace:
                metrics.cycle_trace.append({
                    "output_start": cycle_start,
                    "pending": emitted[cycle_start],
                    "draft_tokens": candidates,
                    "accepted": accepted,
                    "rejected_at": rejected_at,
                    "emitted_tokens": emitted[cycle_start:],
                    "draft_s": draft_elapsed,
                    "verify_s": verify_elapsed,
                    "commit_s": commit_elapsed,
                    "cycle_s": cycle_elapsed,
                    "cache_len_after": cache.get_seq_length(),
                    "next_pending": pending if emitted[-1] != eos else None,
                })
            if emitted and emitted[-1] == eos:
                break

        # No pending prediction is needed after EOS/budget completion. Match
        # the canonical greedy cache length by dropping the final emitted token.
        if emitted and (emitted[-1] == eos or len(emitted) == max_new_tokens):
            cache.crop(int(prompt_ids.shape[1]) + len(emitted) - 1)
            if state_observer is not None:
                state_observer(len(emitted) - 1, cache)

    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        metrics.peak_vram_mb = torch.cuda.max_memory_allocated(device) / 1e6
    metrics.generated_token_ids = emitted[:max_new_tokens]
    metrics.total_time_s = time.perf_counter() - started
    metrics.capability_report = capability_report
    return tokenizer.decode(metrics.generated_token_ids, skip_special_tokens=True), metrics


def b1_fixed_row_generate(
    model,
    tokenizer,
    prompt: str | torch.Tensor,
    *,
    k: int = 4,
    ngram_size: int = 3,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    kv_cache_max_capacity: int = 2048,
    physical_rows: int = 9,
    attention_strategy: str = "rowwise",
    reuse_linear_padding_buffer: bool = False,
    quantized_linear_strategy: str = "fixed_rows",
    state_observer: Callable[[int, TargetKVCache], None] | None = None,
) -> tuple[str, B1Metrics]:
    """Greedy B1 reference under the pinned fixed-row operator schedule."""
    from zassd.certification.model_capabilities import require_b1_fixed_row_capability
    from zassd.certification.fixed_row_dispatch import fixed_row_schedule_for_model

    if physical_rows < 2:
        raise ValueError("physical_rows must be >= 2")
    capability_report = require_b1_fixed_row_capability(
        model, attention_strategy=attention_strategy)
    with fixed_row_schedule_for_model(
        model, physical_rows=physical_rows, attention_strategy=attention_strategy,
        reuse_linear_padding_buffer=reuse_linear_padding_buffer,
        quantized_linear_strategy=quantized_linear_strategy,
    ):
        text, metrics = b1_exact_generate(
            model, tokenizer, prompt, k=k, ngram_size=ngram_size,
            max_new_tokens=max_new_tokens, device=device,
            kv_cache_max_capacity=kv_cache_max_capacity,
            state_observer=state_observer,
        )
    metrics.reference_schedule = f"{model.config.model_type}_fixed_row_{attention_strategy}"
    if reuse_linear_padding_buffer:
        metrics.reference_schedule += "_reused_linear_padding"
    if quantized_linear_strategy != "fixed_rows":
        metrics.reference_schedule += f"_quantized_linear_{quantized_linear_strategy}"
    metrics.physical_rows = physical_rows
    metrics.capability_report = capability_report
    return text, metrics
