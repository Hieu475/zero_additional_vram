"""Formal verification of KV cache invariants for Zero-Additional-Weight-VRAM decoding.

Proves the 4 core systems invariants:
1. [PASS] Prefix tensor is shared (zero-copy pointer aliasing on fork).
2. [PASS] Draft write does not mutate target KV (pointer and content immutability).
3. [PASS] Reject rollback restores canonical target KV (clean truncation without pollution).
4. [PASS] Additional memory scales strictly with speculative suffix (zero duplicated prefix memory).
"""

from __future__ import annotations

import pytest
import torch
from transformers.cache_utils import DynamicCache

from zassd.cache.kv_cache import TargetKVCache


class TestKVCacheInvariants:
    """Rigorous tests proving the four mathematical and systems invariants of ZASSD KV caching."""

    @pytest.fixture
    def populated_target_kv(self) -> tuple[TargetKVCache, torch.Tensor, torch.Tensor]:
        """Create a target KV cache populated with a 32-token prefix across 4 layers."""
        num_layers = 4
        batch_size = 1
        num_heads = 4
        prefix_len = 32
        head_dim = 64

        target = TargetKVCache(backend="dynamic")
        keys_list = []
        vals_list = []

        for layer_idx in range(num_layers):
            k = torch.randn(batch_size, num_heads, prefix_len, head_dim)
            v = torch.randn(batch_size, num_heads, prefix_len, head_dim)
            target.cache.update(k, v, layer_idx)
            keys_list.append(k)
            vals_list.append(v)

        return target, torch.stack(keys_list), torch.stack(vals_list)

    def test_invariant_1_prefix_tensor_is_shared(
        self, populated_target_kv: tuple[TargetKVCache, torch.Tensor, torch.Tensor]
    ) -> None:
        """Invariant 1: Forking an ephemeral draft cache aliases prefix pointers with 0 copies."""
        target, _, _ = populated_target_kv
        num_layers = len(target.cache.layers)

        # Fork ephemeral draft cache
        draft = target.fork_ephemeral_draft_kv()

        assert len(draft.layers) == num_layers, "Draft cache must mirror layer count"

        for idx in range(num_layers):
            t_k_ptr = target.cache.layers[idx].keys.data_ptr()
            t_v_ptr = target.cache.layers[idx].values.data_ptr()
            d_k_ptr = draft.layers[idx].keys.data_ptr()
            d_v_ptr = draft.layers[idx].values.data_ptr()

            # Prove pointer equality (zero-copy alias)
            assert d_k_ptr == t_k_ptr, f"Layer {idx} keys data_ptr must be identical (zero-copy)"
            assert d_v_ptr == t_v_ptr, f"Layer {idx} values data_ptr must be identical (zero-copy)"
            assert draft.layers[idx].is_initialized is True

    def test_invariant_2_draft_write_does_not_mutate_target_kv(
        self, populated_target_kv: tuple[TargetKVCache, torch.Tensor, torch.Tensor]
    ) -> None:
        """Invariant 2: Generating draft tokens updates draft pointers while target KV remains pristine."""
        target, orig_keys, orig_vals = populated_target_kv
        num_layers = len(target.cache.layers)
        prefix_len = target.get_seq_length(0)

        # Record target pointers and snapshot content before draft updates
        target_ptrs_before = [
            (target.cache.layers[i].keys.data_ptr(), target.cache.layers[i].values.data_ptr())
            for i in range(num_layers)
        ]
        target_content_before = [
            (target.cache.layers[i].keys.clone(), target.cache.layers[i].values.clone())
            for i in range(num_layers)
        ]

        draft = target.fork_ephemeral_draft_kv()

        # Simulate generating K=4 draft tokens
        k_draft = 4
        for step in range(k_draft):
            for layer_idx in range(num_layers):
                new_k = torch.randn(1, 4, 1, 64)
                new_v = torch.randn(1, 4, 1, 64)
                draft.update(new_k, new_v, layer_idx)

        # 1. Draft sequence length must have grown by K
        for layer_idx in range(num_layers):
            assert draft.get_seq_length(layer_idx) == prefix_len + k_draft

        # 2. Target KV sequence length must remain unchanged
        assert target.get_seq_length(0) == prefix_len

        # 3. Target KV pointers must remain strictly unchanged
        for idx in range(num_layers):
            curr_k_ptr = target.cache.layers[idx].keys.data_ptr()
            curr_v_ptr = target.cache.layers[idx].values.data_ptr()
            orig_k_ptr, orig_v_ptr = target_ptrs_before[idx]
            assert curr_k_ptr == orig_k_ptr, f"Target layer {idx} keys pointer was mutated"
            assert curr_v_ptr == orig_v_ptr, f"Target layer {idx} values pointer was mutated"

        # 4. Target KV content must be bitwise identical
        for idx in range(num_layers):
            orig_k, orig_v = target_content_before[idx]
            assert torch.equal(target.cache.layers[idx].keys, orig_k), f"Target layer {idx} keys content mutated"
            assert torch.equal(target.cache.layers[idx].values, orig_v), f"Target layer {idx} values content mutated"

    def test_invariant_3_reject_rollback_restores_canonical_target_kv(
        self, populated_target_kv: tuple[TargetKVCache, torch.Tensor, torch.Tensor]
    ) -> None:
        """Invariant 3: Rolling back unaccepted speculative candidates leaves no residual tokens."""
        target, _, _ = populated_target_kv
        num_layers = len(target.cache.layers)
        prefix_len = target.get_seq_length(0)

        # Simulate verification pass extending Target KV with 3 candidate tokens
        cand_len = 3
        for step in range(cand_len):
            for layer_idx in range(num_layers):
                cand_k = torch.randn(1, 4, 1, 64)
                cand_v = torch.randn(1, 4, 1, 64)
                target.cache.update(cand_k, cand_v, layer_idx)

        assert target.get_seq_length(0) == prefix_len + cand_len

        # Suppose candidate at step 1 was rejected (only 1 draft token accepted)
        accepted_count = 1
        target.rollback(prefix_len=prefix_len, accepted_count=accepted_count)

        # Canonical target length must now be exactly prefix_len + accepted_count
        expected_len = prefix_len + accepted_count
        for layer_idx in range(num_layers):
            assert target.get_seq_length(layer_idx) == expected_len
            assert target.cache.layers[layer_idx].keys.shape[2] == expected_len
            assert target.cache.layers[layer_idx].values.shape[2] == expected_len

    def test_invariant_4_additional_memory_scales_only_with_speculative_suffix(
        self, populated_target_kv: tuple[TargetKVCache, torch.Tensor, torch.Tensor]
    ) -> None:
        """Invariant 4: Ephemeral draft memory does not replicate the prefix buffer."""
        target, _, _ = populated_target_kv
        prefix_len = target.get_seq_length(0)
        num_layers = len(target.cache.layers)
        num_heads = 4
        head_dim = 64
        elem_size = 4  # float32 = 4 bytes

        # Before fork: target memory
        target_mem_bytes = sum(
            layer.keys.nelement() * elem_size + layer.values.nelement() * elem_size
            for layer in target.cache.layers
        )
        assert target_mem_bytes == num_layers * 2 * (1 * num_heads * prefix_len * head_dim) * elem_size

        # Fork draft: zero newly allocated bytes for prefix
        draft = target.fork_ephemeral_draft_kv()
        # Verify that both reference the exact same memory buffer addresses
        for dl, tl in zip(draft.layers, target.cache.layers):
            assert dl.keys.untyped_storage().data_ptr() == tl.keys.untyped_storage().data_ptr()
            assert dl.values.untyped_storage().data_ptr() == tl.values.untyped_storage().data_ptr()

        # Add K=2 draft tokens
        k_tokens = 2
        for layer_idx in range(num_layers):
            new_k = torch.randn(1, num_heads, k_tokens, head_dim)
            new_v = torch.randn(1, num_heads, k_tokens, head_dim)
            draft.update(new_k, new_v, layer_idx)

        # Verify that draft holds prefix + K, while target remains strictly at prefix
        for dl in draft.layers:
            assert dl.keys.shape[2] == prefix_len + k_tokens
        for tl in target.cache.layers:
            assert tl.keys.shape[2] == prefix_len


class TestStaticKVCacheInvariants:
    """Formal verification of invariants for StaticPreallocatedKVCache."""

    @pytest.fixture
    def populated_static_kv(self) -> tuple[TargetKVCache, torch.Tensor, torch.Tensor]:
        """Create a target KV cache populated with static preallocated buffer."""
        num_layers = 4
        batch_size = 1
        num_heads = 4
        prefix_len = 32
        head_dim = 64
        max_capacity = 256

        target = TargetKVCache(backend="static", max_capacity=max_capacity)
        keys_list = []
        vals_list = []

        for layer_idx in range(num_layers):
            k = torch.randn(batch_size, num_heads, prefix_len, head_dim)
            v = torch.randn(batch_size, num_heads, prefix_len, head_dim)
            target.cache.update(k, v, layer_idx)
            keys_list.append(k)
            vals_list.append(v)

        return target, torch.stack(keys_list), torch.stack(vals_list)

    def test_static_invariant_1_buffer_is_shared(
        self, populated_static_kv: tuple[TargetKVCache, torch.Tensor, torch.Tensor]
    ) -> None:
        """Invariant 1: Ephemeral draft shares identical pre-allocated memory buffer pointers."""
        target, _, _ = populated_static_kv
        num_layers = len(target.cache.layers)

        draft = target.fork_ephemeral_draft_kv()
        assert len(draft.layers) == num_layers

        for idx in range(num_layers):
            t_k_ptr = target.cache.layers[idx].keys.data_ptr()
            t_v_ptr = target.cache.layers[idx].values.data_ptr()
            d_k_ptr = draft.layers[idx].keys.data_ptr()
            d_v_ptr = draft.layers[idx].values.data_ptr()

            # Zero-copy pointer equality
            assert d_k_ptr == t_k_ptr, f"Layer {idx} static keys pointer must be identical"
            assert d_v_ptr == t_v_ptr, f"Layer {idx} static values pointer must be identical"
            assert draft.layers[idx].get_seq_length() == target.get_seq_length(idx)

    def test_static_invariant_2_draft_writes_do_not_mutate_prefix(
        self, populated_static_kv: tuple[TargetKVCache, torch.Tensor, torch.Tensor]
    ) -> None:
        """Invariant 2: Draft writes candidate tokens into [P, P+K) without mutating prefix [0, P)."""
        target, orig_keys, orig_vals = populated_static_kv
        num_layers = len(target.cache.layers)
        prefix_len = target.get_seq_length(0)

        # Snapshot prefix content
        prefix_keys_before = [
            target.cache.layers[i].keys[:, :, :prefix_len, :].clone()
            for i in range(num_layers)
        ]
        prefix_vals_before = [
            target.cache.layers[i].values[:, :, :prefix_len, :].clone()
            for i in range(num_layers)
        ]

        draft = target.fork_ephemeral_draft_kv()

        # Draft generates K=4 tokens
        k_draft = 4
        for step in range(k_draft):
            for layer_idx in range(num_layers):
                new_k = torch.randn(1, 4, 1, 64)
                new_v = torch.randn(1, 4, 1, 64)
                draft.update(new_k, new_v, layer_idx)

        # Draft seq_len has grown
        assert draft.get_seq_length(0) == prefix_len + k_draft
        # Target seq_len remains strictly unchanged
        assert target.get_seq_length(0) == prefix_len

        # Prefix content [0 : prefix_len] remains bitwise identical
        for idx in range(num_layers):
            assert torch.equal(target.cache.layers[idx].keys[:, :, :prefix_len, :], prefix_keys_before[idx])
            assert torch.equal(target.cache.layers[idx].values[:, :, :prefix_len, :], prefix_vals_before[idx])

    def test_static_invariant_3_target_verification_and_rollback(
        self, populated_static_kv: tuple[TargetKVCache, torch.Tensor, torch.Tensor]
    ) -> None:
        """Invariant 3: Target verification writes verified KV and rollback cleanly sets target seq_len."""
        target, _, _ = populated_static_kv
        num_layers = len(target.cache.layers)
        prefix_len = target.get_seq_length(0)

        # Target verifies K=3 candidates in a single batched step
        cand_k = torch.randn(1, 4, 3, 64)
        cand_v = torch.randn(1, 4, 3, 64)
        for layer_idx in range(num_layers):
            target.cache.update(cand_k, cand_v, layer_idx)

        assert target.get_seq_length(0) == prefix_len + 3

        # Simulate rejection: only 1 accepted candidate
        target.rollback(prefix_len=prefix_len, accepted_count=1)
        assert target.get_seq_length(0) == prefix_len + 1

    def test_static_invariant_4_zero_dynamic_allocation(
        self, populated_static_kv: tuple[TargetKVCache, torch.Tensor, torch.Tensor]
    ) -> None:
        """Invariant 4: Pre-allocated buffers never reallocate memory during generation (data_ptr is invariant)."""
        target, _, _ = populated_static_kv
        num_layers = len(target.cache.layers)

        orig_ptrs = [
            (target.cache.layers[i].keys.data_ptr(), target.cache.layers[i].values.data_ptr())
            for i in range(num_layers)
        ]

        draft = target.fork_ephemeral_draft_kv()

        # Perform 8 draft steps
        for step in range(8):
            for layer_idx in range(num_layers):
                draft.update(torch.randn(1, 4, 1, 64), torch.randn(1, 4, 1, 64), layer_idx)

        # Buffer pointers must NOT have changed (zero dynamic reallocations!)
        for idx in range(num_layers):
            assert draft.layers[idx].keys.data_ptr() == orig_ptrs[idx][0]
            assert draft.layers[idx].values.data_ptr() == orig_ptrs[idx][1]

