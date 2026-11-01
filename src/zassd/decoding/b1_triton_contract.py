"""Contract-bound entry points for the experimental Qwen2 Triton R=2 schedule."""

from __future__ import annotations

from typing import Callable

import torch

from zassd.cache.kv_cache import TargetKVCache
from zassd.certification.triton_b1_reference_contract import (
    VerifiedTritonB1Contract,
    require_verified_triton_b1_contract,
)
from zassd.decoding.b1_exact import (
    B1Metrics,
    b1_fixed_row_block_generate,
    b1_fixed_row_generate,
)


def b1_triton_contract_singleton(
    model,
    tokenizer,
    prompt: str | torch.Tensor,
    *,
    verified_contract: VerifiedTritonB1Contract,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    kv_cache_max_capacity: int = 2048,
    state_observer: Callable[[int, TargetKVCache], None] | None = None,
) -> tuple[str, B1Metrics]:
    """Generate the singleton reference under the exact contract schedule."""
    require_verified_triton_b1_contract(verified_contract, model, tokenizer)
    return b1_fixed_row_generate(
        model, tokenizer, prompt,
        k=1,
        ngram_size=3,
        max_new_tokens=max_new_tokens,
        device=device,
        kv_cache_max_capacity=kv_cache_max_capacity,
        physical_rows=2,
        attention_strategy="triton_batched",
        quantized_linear_strategy=(
            "triton_nf4_group_scale_packed_rows2_n16_warp1_k256"
        ),
        state_observer=state_observer,
    )


def b1_triton_contract_block(
    model,
    tokenizer,
    prompt: str | torch.Tensor,
    *,
    verified_contract: VerifiedTritonB1Contract,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    kv_cache_max_capacity: int = 2048,
    trace: bool = False,
    state_observer: Callable[[int, TargetKVCache], None] | None = None,
) -> tuple[str, B1Metrics]:
    """Verify a block using only the K=1/R=2 schedule in the verified contract."""
    require_verified_triton_b1_contract(verified_contract, model, tokenizer)
    return b1_fixed_row_block_generate(
        model, tokenizer, prompt,
        k=1,
        ngram_size=3,
        max_new_tokens=max_new_tokens,
        device=device,
        kv_cache_max_capacity=kv_cache_max_capacity,
        physical_rows=2,
        attention_strategy="triton_batched",
        quantized_linear_strategy=(
            "triton_nf4_group_scale_packed_rows2_n16_warp1_k256"
        ),
        trace=trace,
        state_observer=state_observer,
    )
