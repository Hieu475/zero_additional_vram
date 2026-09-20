"""Test B — Target Verification Equivalence Test (Phase 7.2).

Compares position-by-position target probabilities and argmax outputs between:
1. Full recomputation:
     P(d_0 | prefix)
     P(d_1 | prefix, d_0)
     ...
     P(bonus | prefix, d_0, ..., d_{K-1})
2. Target cached verification using TargetKVCache and batched forward pass.

Logs per-position diagnostics:
  {
    "position": pos,
    "target_argmax_full": int,
    "target_argmax_cached": int,
    "max_logit_diff": float,
    "top1_margin": float,
    "match": bool
  }
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from zassd.cache.kv_cache import TargetKVCache
from zassd.models.loader import load_model, load_tokenizer


class TestTargetVerificationEquivalence:
    """Validate equivalence of cached parallel verification against full recomputation."""

    def test_verification_equivalence_across_prompts(self, session_model_and_tok):
        model, tok = session_model_and_tok
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Load fixed test prompts
        prompts_path = Path("data/benchmarks/prompts.jsonl")
        prompts = []
        with open(prompts_path) as f:
            for line in f:
                if line.strip():
                    prompts.append(json.loads(line.strip()))
        eval_prompts = prompts[:5]

        k = 4
        all_diagnostics = []

        for p_idx, p in enumerate(eval_prompts):
            prompt = p["prompt"]
            prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            prompt_len = prompt_ids.shape[1]

            # 1. Populate canonical target KV cache with prompt
            target_kv = TargetKVCache()
            with torch.no_grad():
                prefill_out = model(prompt_ids, past_key_values=target_kv.cache, use_cache=True)
            curr_target_tok = int(prefill_out.logits[0, -1, :].argmax(dim=-1).item())

            # 2. Synthesize draft candidate sequence [d_0, ..., d_{K-1}]
            # We use distinct realistic tokens
            draft_candidates = [curr_target_tok + 1 + i for i in range(k)]

            # 3. Path A: Parallel Cached Verification
            verify_inputs = [curr_target_tok] + draft_candidates
            cand_tensor = torch.tensor([verify_inputs], device=device)
            with torch.no_grad():
                cached_out = model(cand_tensor, past_key_values=target_kv.cache, use_cache=True)
            cached_logits = cached_out.logits[0].float()  # shape: (k + 1, vocab_size)

            # 4. Path B: Full Recomputation (Position by Position, Ground Truth)
            # For position 0: prefix + [curr_target_tok] predicts token after curr_target_tok
            # For position j: prefix + [curr_target_tok, d_0, ..., d_{j-1}]
            full_seq = torch.cat([prompt_ids, cand_tensor], dim=1)
            with torch.no_grad():
                full_out = model(full_seq, use_cache=False)
            # Full logits corresponding to the verification positions
            # The prompt ends at prompt_len - 1.
            # verify_inputs start at prompt_len.
            # Logit at (prompt_len - 1 + pos) predicts token after verify_inputs[pos-1] or verify_inputs[0]
            # Specifically:
            # full_seq[prompt_len] is curr_target_tok.
            # Logit at prompt_len predicts token after curr_target_tok (matches cached_logits[0]).
            full_logits = full_out.logits[0, prompt_len : prompt_len + k + 1].float()

            for pos in range(k + 1):
                c_logit = cached_logits[pos]
                f_logit = full_logits[pos]

                c_argmax = int(c_logit.argmax(dim=-1).item())
                f_argmax = int(f_logit.argmax(dim=-1).item())

                # Compute top-1 vs top-2 margin in full recomputation
                top2_vals, _ = torch.topk(f_logit, 2)
                top1_margin = float((top2_vals[0] - top2_vals[1]).item())
                max_logit_diff = float((c_logit - f_logit).abs().max().item())

                match = (c_argmax == f_argmax)

                diag = {
                    "prompt_id": p.get("id", f"p_{p_idx}"),
                    "position": pos,
                    "target_argmax_full": f_argmax,
                    "target_argmax_cached": c_argmax,
                    "max_logit_diff": round(max_logit_diff, 5),
                    "top1_margin": round(top1_margin, 5),
                    "match": match,
                }
                all_diagnostics.append(diag)

                # Invariant: Logit discrepancy must remain within 4-bit quantization grid bound (< 0.75)
                assert max_logit_diff < 0.75, f"Logit difference too large at pos {pos}: {max_logit_diff}"

                # Invariant: If margin exceeds numerical noise, argmax MUST match 100%
                if top1_margin > max_logit_diff:
                    assert match, (
                        f"Argmax mismatch despite margin ({top1_margin}) > noise ({max_logit_diff}) "
                        f"at prompt {p_idx}, pos {pos}: full={f_argmax} vs cached={c_argmax}"
                    )

        # Save diagnostic analysis
        out_path = Path("results/raw/target_verification_equivalence.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_diagnostics, f, indent=2)

        match_count = sum(1 for d in all_diagnostics if d["match"])
        total_count = len(all_diagnostics)
        assert total_count > 0
        match_rate = match_count / total_count
        assert match_rate >= 0.90, f"Verification match rate {match_rate:.1%} below 90%"
