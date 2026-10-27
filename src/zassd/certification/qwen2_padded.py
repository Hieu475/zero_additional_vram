"""Experimental fixed-row Qwen2 verifier operators for B1 research.

All linear calls during decode use the same physical M dimension. Attention
can execute one query row at a time against its causal KV prefix (``rowwise``)
or use a fixed padded Q/K/V shape and explicit causal mask (``padded``). This
is a prototype operator schedule, not a proven production kernel.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)
from zassd.cache.kv_cache import StaticPreallocatedKVCache


@contextmanager
def fixed_row_qwen2_schedule(model, *, physical_rows: int = 9,
                             attention_strategy: str = "rowwise", observer=None):
    """Patch Qwen2 operators for a fixed row count, restoring them on exit.

    Scope: Qwen2 modules, batch=1 unpadded prompt, eval mode, static KV,
    no user attention mask, no sliding window, and no returned attentions.
    Unsupported attention inputs fail closed. Prefill uses the ordinary Qwen2
    attention path; singleton and block decode use the selected fixed-row path.
    """
    if physical_rows < 2 or model.training:
        raise ValueError("fixed-row schedule requires physical_rows >= 2 and eval mode")
    if attention_strategy not in {"rowwise", "padded"}:
        raise ValueError("attention_strategy must be rowwise or padded")
    model_config = getattr(model, "config", None)
    if getattr(model_config, "model_type", None) != "qwen2":
        raise TypeError("fixed-row schedule requires a Qwen2 model configuration")
    if getattr(model_config, "_attn_implementation", None) != "sdpa":
        raise ValueError("fixed-row schedule requires the pinned SDPA model attention implementation")
    attention_modules = [module for module in model.modules()
                         if isinstance(module, Qwen2Attention)]
    if not attention_modules:
        raise TypeError("fixed-row schedule requires at least one Qwen2Attention module")
    if any(module.sliding_window for module in attention_modules):
        raise ValueError("fixed-row schedule does not support sliding-window attention")
    replacements = []

    original_model_forward = model.forward

    def guarded_model_forward(self, *args, _original=original_model_forward, **kwargs):
        forbidden = ("attention_mask", "position_ids", "inputs_embeds", "cache_position",
                     "block_sequence_ids", "position_embeddings")
        if any(kwargs.get(name) is not None for name in forbidden):
            raise ValueError("fixed-row schedule requires implicit positions and no external mask")
        if ((len(args) > 1 and args[1] is not None) or
                (len(args) > 2 and args[2] is not None) or
                (len(args) > 4 and args[4] is not None)):
            raise ValueError("fixed-row schedule requires implicit positions and no external mask")
        return _original(*args, **kwargs)

    model.forward = MethodType(guarded_model_forward, model)
    replacements.append((model, original_model_forward))

    for module in model.modules():
        if isinstance(module, (Linear4bit, torch.nn.Linear)):
            original = module.forward

            def fixed_linear(self, x, _original=original):
                if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] > physical_rows:
                    return _original(x)
                rows = x.shape[1]
                if rows < physical_rows:
                    x = F.pad(x, (0, 0, 0, physical_rows - rows))
                return _original(x.contiguous())[:, :rows, :]

            module.forward = MethodType(fixed_linear, module)
            replacements.append((module, original))
        elif isinstance(module, Qwen2RMSNorm):
            original = module.forward

            def fixed_rmsnorm(self, x, _original=original):
                if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] > physical_rows:
                    return _original(x)
                rows = x.shape[1]
                if rows < physical_rows:
                    x = F.pad(x, (0, 0, 0, physical_rows - rows))
                return _original(x.contiguous())[:, :rows, :]

            module.forward = MethodType(fixed_rmsnorm, module)
            replacements.append((module, original))
        elif isinstance(module, Qwen2Attention):
            original = module.forward

            def row_attention(self, hidden_states, position_embeddings, attention_mask,
                              past_key_values=None, _original=original, **kwargs):
                if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
                    raise ValueError("fixed-row Qwen2 schedule requires batch size one")
                rows = hidden_states.shape[1]
                if kwargs.get("output_attentions") or self.training or self.sliding_window:
                    raise ValueError("unsupported Qwen2 attention mode for fixed-row schedule")
                if past_key_values is not None and not isinstance(
                        past_key_values, StaticPreallocatedKVCache):
                    raise ValueError("fixed-row Qwen2 schedule requires StaticPreallocatedKVCache")
                if past_key_values is None or past_key_values.get_seq_length(self.layer_idx) == 0:
                    return _original(hidden_states, position_embeddings, attention_mask,
                                     past_key_values=past_key_values, **kwargs)
                if rows > physical_rows:
                    raise ValueError("unsupported Qwen2 fixed-row decode shape/options")
                shape = (*hidden_states.shape[:-1], -1, self.head_dim)
                query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
                key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
                value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
                cos, sin = position_embeddings
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
                key, value = past_key_values.update(key, value, self.layer_idx)
                prefix = key.shape[2] - rows
                key = repeat_kv(key, self.num_key_value_groups)
                value = repeat_kv(value, self.num_key_value_groups)
                if attention_strategy == "rowwise":
                    outputs = []
                    for row in range(rows):
                        length = prefix + row + 1
                        # The public B1 generator uses one unpadded prompt.
                        # Truncating K/V makes the causal mask redundant.
                        row_mask = None
                        row_query = query[:, :, row:row + 1, :].contiguous()
                        row_key = key[:, :, :length, :].contiguous()
                        row_value = value[:, :, :length, :].contiguous()
                        row_output = F.scaled_dot_product_attention(
                            row_query, row_key, row_value, attn_mask=row_mask,
                            dropout_p=0.0, scale=self.scaling, is_causal=False,
                        )
                        if observer is not None:
                            observer(self.layer_idx, row, row_query, row_key,
                                     row_value, row_mask, row_output)
                        outputs.append(row_output)
                    attn_output = torch.cat(outputs, dim=2)
                else:
                    query = F.pad(query, (0, 0, 0, physical_rows - rows)).contiguous()
                    key = F.pad(key, (0, 0, 0, physical_rows - rows)).contiguous()
                    value = F.pad(value, (0, 0, 0, physical_rows - rows)).contiguous()
                    mask = torch.full(
                        (1, 1, physical_rows, prefix + physical_rows),
                        float("-inf"), dtype=query.dtype, device=query.device,
                    )
                    for row in range(physical_rows):
                        mask[:, :, row, :prefix + row + 1] = 0
                    attn_output = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=mask,
                        dropout_p=0.0, scale=self.scaling, is_causal=False,
                    )[:, :, :rows, :]
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(*hidden_states.shape[:-1], -1)
                return self.o_proj(attn_output), None

            module.forward = MethodType(row_attention, module)
            replacements.append((module, original))
    try:
        yield model
    finally:
        for module, original in reversed(replacements):
            module.forward = original
