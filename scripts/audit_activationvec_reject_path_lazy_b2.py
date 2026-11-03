#!/usr/bin/env python3
"""Reject-path B2 audit with the lazy per-row LM head (reuses audit_activationvec_reject_path_b2)."""

import torch

import audit_activationvec_reject_path_b2 as base


def run_block_lazy(model, cache, inputs, device):
    tensor = torch.tensor([inputs], device=device, dtype=torch.long)
    preds = []
    with torch.inference_mode(), base.block_ctx(model):
        hidden = model.model(input_ids=tensor, past_key_values=cache.cache, use_cache=True).last_hidden_state
        for row in range(len(inputs)):
            p = int(model.lm_head(hidden[:, row:row + 1, :]).argmax(dim=-1).item())
            preds.append(p)
            if row < len(inputs) - 1 and p != inputs[row + 1]:
                break
    return preds


base.run_block = run_block_lazy

if __name__ == "__main__":
    base.main()
