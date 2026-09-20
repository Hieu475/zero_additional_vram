"""Test 1 — Cache Equivalence Test.

Verifies that:
1. Path A (full forward) and Path B (prefill + cached decode) produce identical
   or numerically equivalent predictions:
     argmax(logits_A) == argmax(logits_B)
     ||logits_A - logits_B||_inf < epsilon
2. TargetKVCache maintains canonical state and supports exact rollback/cropping.
3. EphemeralDraftKV does not mutate or pollute TargetKVCache.
"""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from zassd.cache.kv_cache import TargetKVCache, EphemeralDraftKV
from zassd.models.loader import load_model, load_tokenizer


class TestCacheEquivalence:
    """Validate cache equivalence and canonical state preservation."""

    def test_full_vs_cached_decode_equivalence(self, session_model_and_tok):
        """Test equivalence between full forward and prefill + cached decode."""
        model, tok = session_model_and_tok
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        prompt = "The fundamental law of mechanics states that force equals mass times"
        inputs = tok(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[1]

        # Path A: Full forward pass without past_key_values
        with torch.no_grad():
            out_a = model(input_ids, use_cache=False)
        logits_a = out_a.logits[0, -1, :].float()
        pred_a = int(logits_a.argmax(dim=-1).item())

        # Path B: Prefill prefix (seq_len - 1) + 1 cached decode step
        target_kv = TargetKVCache()
        with torch.no_grad():
            _ = model(input_ids[:, :-1], past_key_values=target_kv.cache, use_cache=True)
            out_b = model(input_ids[:, -1:], past_key_values=target_kv.cache, use_cache=True)
        logits_b = out_b.logits[0, -1, :].float()
        pred_b = int(logits_b.argmax(dim=-1).item())

        max_abs_diff = float((logits_a - logits_b).abs().max().item())

        # Both paths must predict the exact same next token (" acceleration")
        assert pred_a == pred_b, f"Argmax mismatch: {pred_a} vs {pred_b}"
        # Logit differences in 4-bit quantized models are bounded within quantization grid
        assert max_abs_diff < 1.0, f"Max logit diff too large: {max_abs_diff}"

    def test_target_kv_rollback_integrity(self, session_model_and_tok):
        """Verify that TargetKVCache rolls back cleanly to exact prefix length."""
        model, tok = session_model_and_tok
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        prompt = "In quantum physics, entanglement occurs when pairs of particles"
        inputs = tok(prompt, return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"]
        prompt_len = prompt_ids.shape[1]

        target_kv = TargetKVCache()
        with torch.no_grad():
            _ = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)

        initial_len = target_kv.get_seq_length(0)
        assert initial_len == prompt_len

        # Advance 4 candidate tokens into target_kv
        dummy_candidates = torch.tensor([[100, 200, 300, 400]], device=device)
        with torch.no_grad():
            _ = model(dummy_candidates, past_key_values=target_kv.cache, use_cache=True)

        assert target_kv.get_seq_length(0) == prompt_len + 4

        # Simulate rejection: keep only 1 accepted candidate token
        target_kv.rollback(prefix_len=prompt_len, accepted_count=1)
        assert target_kv.get_seq_length(0) == prompt_len + 1

        # Mathematical Invariant: running next forward pass on rolled-back cache
        # MUST yield predictions identical to full recomputation on (prefix + accepted_candidate)
        eval_tok = torch.tensor([[500]], device=device)
        with torch.no_grad():
            out_cached_after_rollback = model(eval_tok, past_key_values=target_kv.cache, use_cache=True)
            # Ground truth: full recompute over prefix + accepted candidate (dummy_candidates[:, :1]) + eval_tok
            full_seq = torch.cat([prompt_ids, dummy_candidates[:, :1], eval_tok], dim=1)
            out_full_recompute = model(full_seq, use_cache=False)

        logits_cached = out_cached_after_rollback.logits[0, -1, :].float()
        logits_ground_truth = out_full_recompute.logits[0, -1, :].float()

        pred_cached = int(logits_cached.argmax(dim=-1).item())
        pred_ground_truth = int(logits_ground_truth.argmax(dim=-1).item())

        max_diff = float((logits_cached - logits_ground_truth).abs().max().item())
        assert pred_cached == pred_ground_truth, f"Prediction mismatch after rollback: {pred_cached} vs {pred_ground_truth}"
        assert max_diff < 1.0, f"Logit difference after rollback exceeds bound: {max_diff}"

    def test_ephemeral_draft_kv_independence(self, session_model_and_tok):
        """Verify updating ephemeral draft KV does not mutate canonical TargetKVCache."""
        model, tok = session_model_and_tok
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        prompt = "Artificial intelligence is revolutionizing modern computing"
        inputs = tok(prompt, return_tensors="pt").to(device)
        prompt_ids = inputs["input_ids"]
        prompt_len = prompt_ids.shape[1]

        target_kv = TargetKVCache()
        with torch.no_grad():
            _ = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)

        orig_target_len = target_kv.get_seq_length(0)

        # Fork ephemeral draft KV
        draft_kv = EphemeralDraftKV.create_from_target(target_kv)
        assert draft_kv.get_seq_length(0) == orig_target_len

        # Append 3 draft tokens to draft_kv
        curr_tok = torch.tensor([[100]], device=device)
        with torch.no_grad():
            for _ in range(3):
                d_out = model(curr_tok, past_key_values=draft_kv, use_cache=True)
                nxt = int(d_out.logits[0, -1, :].argmax(dim=-1).item())
                curr_tok = torch.tensor([[nxt]], device=device)

        # Draft KV length should have grown
        assert draft_kv.get_seq_length(0) == orig_target_len + 3

        # Canonical target KV MUST remain completely unchanged
        assert target_kv.get_seq_length(0) == orig_target_len
