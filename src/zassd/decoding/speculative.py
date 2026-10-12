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
import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

from zassd.cache.kv_cache import TargetKVCache
from zassd.controllers.entropy import compute_entropy
from zassd.decoding.ngram import find_candidate_tokens
from zassd.models.layer_manager import LayerManager

def _logit_margin(logit_1d: torch.Tensor) -> float | None:
    """Top1-top2 logit gap (confidence signal for the calibrated predictor).

    Returns None on any failure so callers can fall back to prior-only
    estimates. Kept separate from compute_entropy to avoid slowing
    non-routed paths (only called when a router is attached).
    """
    try:
        top2 = torch.topk(logit_1d.float().flatten(), 2).values
        return float((top2[0] - top2[1]).item())
    except Exception:
        return None


def diagnostic_cache_digest(target_kv: TargetKVCache) -> str:
    """Hash only the active KV prefix for a slow, diagnostic-only audit."""
    digest = hashlib.sha256()
    for layer in target_kv.cache.layers:
        length = layer.get_seq_length()
        digest.update(length.to_bytes(8, "little"))
        if not getattr(layer, "is_initialized", False):
            continue
        for tensor in (layer.keys, layer.values):
            raw_bytes = tensor[:, :, :length, :].detach().contiguous().view(torch.uint8)
            digest.update(raw_bytes.cpu().numpy().tobytes())
    return digest.hexdigest()


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

    # Latency decomposition includes diagnostic canonical cache replay when enabled.
    prefill_time_s: float = 0.0
    draft_time_s: float = 0.0
    verify_time_s: float = 0.0
    cache_time_s: float = 0.0
    canonical_replay_time_s: float = 0.0
    margin_fallback_cycles: int = 0
    controller_time_s: float = 0.0
    other_time_s: float = 0.0

    pld_cycles: int = 0
    layer_skip_cycles: int = 0

    peak_vram_mb: float = 0.0
    speedup_vs_vanilla: float = 0.0
    k_value: int = 4
    per_iteration_stats: list[dict] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    diagnostic_prefill: dict = field(default_factory=dict)


def self_speculative_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_mgr: LayerManager,
    skip_indices: list[int],
    prompt: str | torch.Tensor,
    k: int = 4,
    controller: Optional[Any] = None,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    device: str = "cuda:0",
    kv_cache_backend: str = "static",
    kv_cache_max_capacity: int = 2048,
    draft_mode: str = "layer_skip",
    ngram_size: int = 3,
    router: Optional[Any] = None,
    config_name: str = "cka_75",
    diagnostic_trace: bool = False,
    diagnostic_hash_kv: bool = False,
    canonical_kv_replay: bool = False,
    canonical_verify: bool = False,
    margin_fallback_threshold: Optional[float] = None,
    collect_acceptance_cycles: bool = False,
    profile_nvtx: bool = False,
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
        kv_cache_backend: KV cache backend ('static' for pre-allocated or 'dynamic').
        kv_cache_max_capacity: Static cache capacity; must cover prompt plus generated tokens.
        draft_mode: Speculative draft mode ('layer_skip', 'hybrid', 'prompt_lookup', or 'routed').
        router: Optional HybridDraftRouter for draft_mode='routed' (cost-aware PLD vs layer-skip).
        config_name: CKA config name used by router cost estimates.
        ngram_size: Context n-gram length used for prompt lookup.
        canonical_kv_replay: Diagnostic full-prefix KV reconstruction after each speculative cycle.
        canonical_verify: Verify draft positions sequentially; intended for greedy exactness audits.
        margin_fallback_threshold: Rebuild canonical KV and verify sequentially when any batched
            top-1 margin in a cycle is at or below this value (greedy only).
        collect_acceptance_cycles: Include pre-cycle context/confidence and survival labels in
            per_iteration_stats. Intended for offline predictor evaluation.
        profile_nvtx: Add stage labels for Nsight Systems profiling; use only for profiling runs.

    Returns:
        Tuple of (generated_text, SpeculativeMetrics).
    """
    metrics = SpeculativeMetrics(k_value=k)
    if (canonical_verify or margin_fallback_threshold is not None) and temperature != 0.0:
        raise ValueError("canonical verification and margin fallback currently support greedy decoding only")
    if canonical_verify and margin_fallback_threshold is not None:
        raise ValueError("choose canonical_verify or margin_fallback_threshold, not both")
    if margin_fallback_threshold is not None and margin_fallback_threshold < 0:
        raise ValueError("margin_fallback_threshold must be nonnegative")

    if isinstance(prompt, torch.Tensor):
        prompt_ids = prompt.to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]
    full_history: list[int] = prompt_ids[0].tolist()

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    total_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Canonical Prefill: Populate Target KV Cache
    # -----------------------------------------------------------------------
    t_prefill_start = time.perf_counter()
    if profile_nvtx:
        torch.cuda.nvtx.range_push("target_prefill")
    target_kv = TargetKVCache(backend=kv_cache_backend, max_capacity=kv_cache_max_capacity)
    with torch.no_grad():
        prefill_out = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)
    target_prefix_logit = prefill_out.logits[0, -1, :]
    torch.cuda.synchronize()
    if profile_nvtx:
        torch.cuda.nvtx.range_pop()
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

    curr_entropy = float(compute_entropy(target_prefix_logit.float()).item())
    _need_margin = router is not None and draft_mode == "routed"
    curr_margin: float | None = (
        _logit_margin(target_prefix_logit) if (_need_margin or collect_acceptance_cycles) else None
    )
    if diagnostic_trace:
        prefill_top2 = torch.topk(target_prefix_logit.float(), 2)
        metrics.diagnostic_prefill = {
            "token_id": curr_target_tok,
            "margin": float((prefill_top2.values[0] - prefill_top2.values[1]).item()),
            "cache_len": target_kv.get_seq_length(),
        }
        if diagnostic_hash_kv:
            metrics.diagnostic_prefill["cache_digest"] = diagnostic_cache_digest(target_kv)
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
        cycle_entropy = curr_entropy
        cycle_margin = curr_margin if collect_acceptance_cycles else None

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

        # The pending target token already occupies one output position.
        # Do not draft tokens that cannot be emitted within the budget.
        step_k = min(step_k, max(0, rem_tokens - 1))
        if step_k <= 0 and rem_tokens <= 0:
            break

        if step_k == 0:
            # Fallback to single-token vanilla step with Target KV
            trace_prefix_len = target_kv.get_seq_length() if diagnostic_trace else 0
            trace_cache_digest = diagnostic_cache_digest(target_kv) if diagnostic_hash_kv else None
            trace_output_start = len(generated_token_ids)
            t_v_start = time.perf_counter()
            curr_tensor = curr_target_tok_tensor
            if profile_nvtx:
                torch.cuda.nvtx.range_push("vanilla_fallback")
            with torch.no_grad():
                out = model(curr_tensor, past_key_values=target_kv.cache, use_cache=True)
            torch.cuda.synchronize()
            if profile_nvtx:
                torch.cuda.nvtx.range_pop()
            t_v_end = time.perf_counter()
            verify_elapsed = t_v_end - t_v_start
            metrics.verify_time_s += verify_elapsed
            metrics.num_verification_cycles += 1

            target_logit = out.logits[0, -1, :].float()
            curr_entropy = float(compute_entropy(target_logit).item())
            if _need_margin or collect_acceptance_cycles:
                curr_margin = _logit_margin(target_logit)
            if _need_margin and router is not None:
                router.observe_vanilla(verify_elapsed * 1000.0)
            if temperature == 0.0:
                next_tok_tensor = out.logits[:, -1:, :].argmax(dim=-1)
                next_tok = int(next_tok_tensor.item())
                curr_target_tok_tensor = next_tok_tensor
            else:
                probs = F.softmax(target_logit / temperature, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())
                curr_target_tok_tensor = torch.tensor([[next_tok]], device=device)

            generated_token_ids.append(curr_target_tok)
            full_history.append(curr_target_tok)
            current_prefix_len += 1
            vanilla_stat = {
                "k": 0, "accepted": 0, "rejected_at": None,
                "emitted_count": 1, "draft_time_ms": 0.0,
                "verify_time_ms": verify_elapsed * 1000, "cache_time_ms": 0.0,
                "skip_indices": active_skip, "entropy": curr_entropy,
                "cycle_context_length": current_prefix_len - 1,
                "cycle_entropy": cycle_entropy, "cycle_logit_margin": cycle_margin,
                "draft_tokens": [], "accepted_survival": [], "source": "vanilla",
                "router_source": getattr(getattr(router, "_pending_decision", None), "source", "vanilla"),
                "oracle_source": (getattr(router, "_pending_oracle", None) or {}).get("source"),
                "oracle_tps": (getattr(router, "_pending_oracle", None) or {}).get("tps"),
            }
            if diagnostic_trace:
                vanilla_stat.update({
                    "source": "vanilla", "output_start": trace_output_start,
                    "cache_len_before_verify": trace_prefix_len,
                    "cache_len_after_commit": target_kv.get_seq_length(),
                    "emitted_tokens": [curr_target_tok],
                    "verify_argmax": [next_tok],
                    "next_pending_token": next_tok,
                    "cache_digest_before": trace_cache_digest,
                    "cache_digest_after": diagnostic_cache_digest(target_kv) if diagnostic_hash_kv else None,
                })
            metrics.per_iteration_stats.append(vanilla_stat)
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
        # Phase 1: Draft generation (Logical Layer-Skip or Hybrid PLD)
        # Keeps intermediate candidate tensors directly on GPU to avoid CPU-GPU sync.
        # -------------------------------------------------------------------
        draft_tokens: list[int] = []
        draft_token_tensors: list[torch.Tensor] = []
        draft_probs_list: list[torch.Tensor] = []

        if profile_nvtx:
            torch.cuda.nvtx.range_push("draft_forward")
        t_draft_start = time.perf_counter()

        # Routed mode: cost-aware choice between PLD and layer-skip (Huong 1)
        used_pld = False
        routed_force_vanilla = False
        if draft_mode == "routed" and router is not None:
            decision = router.select_source(
                full_history, curr_target_tok, step_k,
                entropy=curr_entropy, config_name=config_name,
                pld_k=min(getattr(router, "pld_k", step_k) or step_k, max(0, rem_tokens - 1)),
                margin=curr_margin,
            )
            if decision.source == "pld" and decision.candidates:
                draft_tokens = list(decision.candidates)
                draft_token_tensors = [torch.tensor([[t]], device=device) for t in draft_tokens]
                used_pld = True
                metrics.pld_cycles += 1
            elif decision.source == "vanilla":
                routed_force_vanilla = True  # skip draft, force vanilla fallback below
                router.update("vanilla", 0, 0)
            # else: fall through to layer-skip draft below
            router._pending_decision = decision
            # Per-cycle oracle log (model-based): same info, bare argmax rule.
            if hasattr(router, "oracle_select"):
                try:
                    _o_src, _o_tps = router.oracle_select(
                        full_history, curr_target_tok, step_k,
                        entropy=curr_entropy, config_name=config_name,
                        pld_k=min(getattr(router, "pld_k", step_k) or step_k, max(0, rem_tokens - 1)),
                        margin=curr_margin,
                    )
                    router._pending_oracle = {"source": _o_src, "tps": _o_tps}
                except Exception:
                    router._pending_oracle = None
        if not used_pld and draft_mode in ("hybrid", "prompt_lookup"):
            cands = find_candidate_tokens(full_history + [curr_target_tok], ngram_size=ngram_size, max_candidates=step_k)
            if len(cands) > 0:
                draft_tokens = cands
                draft_token_tensors = [torch.tensor([[t]], device=device) for t in cands]
                used_pld = True
                metrics.pld_cycles += 1

        # A lookup hit may contain tokens after EOS. Verifying/emitting those
        # tokens changes the stopping semantics relative to vanilla decoding.
        if used_pld and tokenizer.eos_token_id in draft_tokens:
            eos_end = draft_tokens.index(tokenizer.eos_token_id) + 1
            draft_tokens = draft_tokens[:eos_end]
            draft_token_tensors = draft_token_tensors[:eos_end]

        if not used_pld and not routed_force_vanilla and draft_mode in ("layer_skip", "hybrid", "routed"):
            metrics.layer_skip_cycles += 1
            with torch.no_grad():
                with layer_mgr.skip_layers(active_skip):
                    curr_d = curr_target_tok_tensor
                    for d_step in range(step_k):
                        d_out = model(curr_d, past_key_values=draft_kv, use_cache=True)

                        if temperature == 0.0:
                            next_d = d_out.logits[:, -1:, :].argmax(dim=-1)
                            draft_token_tensors.append(next_d)
                            d_next = int(next_d.item())
                            draft_tokens.append(d_next)
                            curr_d = next_d
                        else:
                            d_logits = d_out.logits[0, -1, :].float()
                            d_probs = F.softmax(d_logits / temperature, dim=-1)
                            d_next = int(torch.multinomial(d_probs, num_samples=1).item())
                            draft_probs_list.append(d_probs)
                            draft_tokens.append(d_next)
                            next_d = torch.tensor([[d_next]], device=device)
                            draft_token_tensors.append(next_d)
                            curr_d = next_d

                        if d_next == tokenizer.eos_token_id:
                            break

        torch.cuda.synchronize()
        if profile_nvtx:
            torch.cuda.nvtx.range_pop()
        t_draft_end = time.perf_counter()
        draft_elapsed = t_draft_end - t_draft_start
        metrics.draft_time_s += draft_elapsed
        metrics.total_draft_tokens += len(draft_tokens)

        actual_k = len(draft_tokens)

        if actual_k == 0:
            # Fallback to single-token vanilla step with Target KV
            trace_prefix_len = target_kv.get_seq_length() if diagnostic_trace else 0
            trace_cache_digest = diagnostic_cache_digest(target_kv) if diagnostic_hash_kv else None
            trace_output_start = len(generated_token_ids)
            t_v_start = time.perf_counter()
            if profile_nvtx:
                torch.cuda.nvtx.range_push("vanilla_fallback")
            with torch.no_grad():
                out = model(curr_target_tok_tensor, past_key_values=target_kv.cache, use_cache=True)
            torch.cuda.synchronize()
            if profile_nvtx:
                torch.cuda.nvtx.range_pop()
            t_v_end = time.perf_counter()
            verify_elapsed = t_v_end - t_v_start
            metrics.verify_time_s += verify_elapsed
            metrics.num_verification_cycles += 1

            target_logit = out.logits[0, -1, :].float()
            curr_entropy = float(compute_entropy(target_logit).item())
            if _need_margin or collect_acceptance_cycles:
                curr_margin = _logit_margin(target_logit)
            if temperature == 0.0:
                next_tok_tensor = out.logits[:, -1:, :].argmax(dim=-1)
                next_tok = int(next_tok_tensor.item())
                curr_target_tok_tensor = next_tok_tensor
            else:
                probs = F.softmax(target_logit / temperature, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())
                curr_target_tok_tensor = torch.tensor([[next_tok]], device=device)

            generated_token_ids.append(curr_target_tok)
            full_history.append(curr_target_tok)
            current_prefix_len += 1
            vanilla_stat = {
                "k": 0, "accepted": 0, "rejected_at": None,
                "emitted_count": 1, "draft_time_ms": draft_elapsed * 1000,
                "verify_time_ms": verify_elapsed * 1000, "cache_time_ms": 0.0,
                "skip_indices": active_skip, "entropy": curr_entropy,
                "router_source": getattr(getattr(router, "_pending_decision", None), "source", "vanilla"),
                "oracle_source": (getattr(router, "_pending_oracle", None) or {}).get("source"),
                "oracle_tps": (getattr(router, "_pending_oracle", None) or {}).get("tps"),
            }
            if diagnostic_trace:
                vanilla_stat.update({
                    "source": "vanilla", "output_start": trace_output_start,
                    "cache_len_before_verify": trace_prefix_len,
                    "cache_len_after_commit": target_kv.get_seq_length(),
                    "emitted_tokens": [curr_target_tok],
                    "verify_argmax": [next_tok],
                    "next_pending_token": next_tok,
                    "cache_digest_before": trace_cache_digest,
                    "cache_digest_after": diagnostic_cache_digest(target_kv) if diagnostic_hash_kv else None,
                })
            metrics.per_iteration_stats.append(vanilla_stat)
            if curr_target_tok == tokenizer.eos_token_id:
                break
            curr_target_tok = next_tok
            last_accepted = 1
            last_proposed = 0
            last_draft_ms = 0.0
            last_verify_ms = verify_elapsed * 1000
            if draft_mode == "routed" and router is not None:
                router.observe_vanilla(verify_elapsed * 1000.0)
            continue

        # -------------------------------------------------------------------
        # Phase 2: Parallel target verification (using full model)
        # Verify inputs: [curr_target_tok] + draft_tokens concatenated on GPU.
        # Directly computes all verification logits and the bonus token in 1 pass.
        # -------------------------------------------------------------------
        trace_prefix_len = target_kv.get_seq_length() if diagnostic_trace else 0
        trace_cache_digest = diagnostic_cache_digest(target_kv) if diagnostic_hash_kv else None
        cand_tensor = torch.cat([curr_target_tok_tensor] + draft_token_tensors, dim=-1)

        t_verify_start = time.perf_counter()
        if profile_nvtx:
            torch.cuda.nvtx.range_push("target_verify")
        margin_fallback_triggered = False
        minimum_batch_margin: Optional[float] = None
        cycle_replay_s = 0.0
        with torch.no_grad():
            if canonical_verify:
                # Exact autoregressive reference path: evaluate each candidate
                # position with a single-token query and advance canonical KV.
                # This isolates batch-kernel numerical drift from PLD quality.
                step_logits = []
                for position, input_token in enumerate([curr_target_tok] + draft_tokens):
                    input_tensor = torch.tensor([[input_token]], device=device)
                    step_out = model(input_tensor, past_key_values=target_kv.cache, use_cache=True)
                    position_logits = step_out.logits[0, -1, :]
                    step_logits.append(position_logits)
                    if position < actual_k and int(position_logits.argmax().item()) != draft_tokens[position]:
                        break
                v_logits = torch.stack(step_logits).unsqueeze(0)
            else:
                v_logits = model(cand_tensor, past_key_values=target_kv.cache, use_cache=True).logits
                if margin_fallback_threshold is not None:
                    batch_top2 = torch.topk(v_logits[0].float(), 2, dim=-1).values
                    batch_margins = batch_top2[:, 0] - batch_top2[:, 1]
                    minimum_batch_margin = float(batch_margins.min().item())
                    margin_fallback_triggered = minimum_batch_margin <= margin_fallback_threshold
                if margin_fallback_triggered:
                    # The batched call has mutated the cache. Rebuild the
                    # already-emitted prefix tokenwise before exact verification.
                    replay_start = time.perf_counter()
                    target_kv = TargetKVCache(backend=kv_cache_backend, max_capacity=kv_cache_max_capacity)
                    model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)
                    for replay_token in generated_token_ids:
                        replay_input = torch.tensor([[replay_token]], device=device)
                        model(replay_input, past_key_values=target_kv.cache, use_cache=True)
                    torch.cuda.synchronize()
                    cycle_replay_s = time.perf_counter() - replay_start
                    metrics.canonical_replay_time_s += cycle_replay_s
                    metrics.margin_fallback_cycles += 1
                    step_logits = []
                    for position, input_token in enumerate([curr_target_tok] + draft_tokens):
                        input_tensor = torch.tensor([[input_token]], device=device)
                        step_out = model(input_tensor, past_key_values=target_kv.cache, use_cache=True)
                        position_logits = step_out.logits[0, -1, :]
                        step_logits.append(position_logits)
                        if position < actual_k and int(position_logits.argmax().item()) != draft_tokens[position]:
                            break
                    v_logits = torch.stack(step_logits).unsqueeze(0)

        torch.cuda.synchronize()
        if profile_nvtx:
            torch.cuda.nvtx.range_pop()
        t_verify_end = time.perf_counter()
        verify_elapsed = t_verify_end - t_verify_start - cycle_replay_s
        metrics.verify_time_s += verify_elapsed
        metrics.num_verification_cycles += 1

        # -------------------------------------------------------------------
        # Phase 3: Accept / Reject & Target KV Crop / Advance
        # Vectorized GPU decision for greedy decoding eliminates per-token CPU stalls.
        # -------------------------------------------------------------------
        t_commit_start = time.perf_counter()
        if profile_nvtx:
            torch.cuda.nvtx.range_push("cache_commit_crop")
        cycle_emitted = [curr_target_tok]
        rejected_at: Optional[int] = None
        num_accepted_draft = 0

        if temperature == 0.0:
            v_preds = v_logits[0, :actual_k, :].argmax(dim=-1)
            d_preds = torch.cat(draft_token_tensors, dim=-1).squeeze(0)
            if canonical_verify or margin_fallback_triggered:
                d_preds = d_preds[:v_preds.shape[0]]
            matches = (v_preds == d_preds)

            if bool(matches.all().item()):
                num_accepted_draft = actual_k
                cycle_emitted.extend(draft_tokens)
                bonus_tok_tensor = v_logits[:, actual_k:, :].argmax(dim=-1)
                curr_target_tok_tensor = bonus_tok_tensor
                curr_target_tok = int(bonus_tok_tensor.item())
                rejected_at = None
            else:
                mismatch_idx = int((~matches).nonzero()[0].item())
                num_accepted_draft = mismatch_idx
                if mismatch_idx > 0:
                    cycle_emitted.extend(draft_tokens[:mismatch_idx])
                target_kv.crop(current_prefix_len + len(cycle_emitted))
                rej_tok_tensor = v_preds[mismatch_idx].view(1, 1)
                curr_target_tok_tensor = rej_tok_tensor
                curr_target_tok = int(rej_tok_tensor.item())
                rejected_at = mismatch_idx
        else:
            target_probs_tensor = F.softmax(v_logits[0, :actual_k, :] / temperature, dim=-1)
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
                    curr_target_tok_tensor = torch.tensor([[curr_target_tok]], device=device)
                    rejected_at = i
                    break

            if rejected_at is None:
                b_probs = F.softmax(v_logits[0, actual_k, :] / temperature, dim=-1)
                bonus_tok = int(torch.multinomial(b_probs, num_samples=1).item())
                curr_target_tok = bonus_tok
                curr_target_tok_tensor = torch.tensor([[curr_target_tok]], device=device)

        torch.cuda.synchronize()
        if profile_nvtx:
            torch.cuda.nvtx.range_pop()
        t_commit_end = time.perf_counter()
        commit_elapsed = t_commit_end - t_commit_start
        cache_cycle_s += commit_elapsed
        metrics.cache_time_s += cache_cycle_s

        if rejected_at is not None:
            _rej_logit = v_logits[0, rejected_at, :].float()
            curr_entropy = float(compute_entropy(_rej_logit).item())
            if _need_margin or collect_acceptance_cycles:
                curr_margin = _logit_margin(_rej_logit)
        else:
            _ok_logit = v_logits[0, actual_k, :].float()
            curr_entropy = float(compute_entropy(_ok_logit).item())
            if _need_margin or collect_acceptance_cycles:
                curr_margin = _logit_margin(_ok_logit)

        last_accepted = num_accepted_draft
        last_proposed = actual_k
        last_draft_ms = draft_elapsed * 1000
        last_verify_ms = verify_elapsed * 1000

        if draft_mode == "routed" and router is not None:
            _dec = getattr(router, "_pending_decision", None)
            if _dec is not None and _dec.source in ("pld", "layer_skip"):
                router.update(_dec.source, num_accepted_draft, actual_k,
                              ngram_match_len=_dec.ngram_match_len,
                              draft_ms=draft_elapsed * 1000.0,
                              verify_ms=verify_elapsed * 1000.0,
                              config_name=config_name, step_k=step_k)
        metrics.total_accepted_tokens += num_accepted_draft
        generated_token_ids.extend(cycle_emitted)
        full_history.extend(cycle_emitted)
        current_prefix_len += len(cycle_emitted)

        # Diagnostic mode: rebuild the target cache through the exact same
        # prompt-prefill + one-token decode path used by vanilla generation.
        # Batched verification can produce numerically different NF4 K/V even
        # when its greedy token decisions match; those differences can accrue.
        if canonical_kv_replay:
            replay_start = time.perf_counter()
            target_kv = TargetKVCache(backend=kv_cache_backend, max_capacity=kv_cache_max_capacity)
            with torch.no_grad():
                model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)
                for replay_token in generated_token_ids:
                    replay_input = torch.tensor([[replay_token]], device=device)
                    model(replay_input, past_key_values=target_kv.cache, use_cache=True)
            torch.cuda.synchronize()
            metrics.canonical_replay_time_s += time.perf_counter() - replay_start

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
            "router_source": getattr(getattr(router, "_pending_decision", None), "source", None),
            "oracle_source": (getattr(router, "_pending_oracle", None) or {}).get("source"),
            "oracle_tps": (getattr(router, "_pending_oracle", None) or {}).get("tps"),
            "margin_fallback": margin_fallback_triggered,
            "minimum_batch_margin": minimum_batch_margin,
            "cycle_context_length": current_prefix_len - len(cycle_emitted),
            "cycle_entropy": cycle_entropy,
            "cycle_logit_margin": cycle_margin,
            "source": "pld" if used_pld else ("vanilla" if routed_force_vanilla else "layer_skip"),
        }
        if collect_acceptance_cycles:
            iter_stat["accepted_survival"] = [int(num_accepted_draft >= i) for i in range(1, actual_k + 1)]
        if diagnostic_trace:
            top2 = torch.topk(v_logits[0].float(), 2, dim=-1).values
            iter_stat.update({
                "output_start": len(generated_token_ids) - len(cycle_emitted),
                "cache_len_before_verify": trace_prefix_len,
                "cache_len_after_commit": target_kv.get_seq_length(),
                "pending_token": cycle_emitted[0],
                "draft_tokens": list(draft_tokens),
                "verify_argmax": v_logits[0].argmax(dim=-1).tolist(),
                "verify_margin": (top2[:, 0] - top2[:, 1]).tolist(),
                "emitted_tokens": list(cycle_emitted),
                "next_pending_token": curr_target_tok,
                "cache_digest_before": trace_cache_digest,
                "cache_digest_after": diagnostic_cache_digest(target_kv) if diagnostic_hash_kv else None,
            })
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
    metrics.generated_token_ids = generated_token_ids
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
        - (metrics.prefill_time_s + metrics.draft_time_s + metrics.verify_time_s + metrics.cache_time_s + metrics.controller_time_s + metrics.canonical_replay_time_s),
    )
    metrics.peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    output_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    return output_text, metrics
