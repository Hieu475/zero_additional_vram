"""Experimental fixed-row Phi-3 operators for B1 research.

This adapter targets Phi3's fused QKV projection and rotary attention. It is
checkpoint/runtime scoped experimental code, not a formal equivalence proof.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import torch.nn.functional as F
from bitsandbytes.nn import Linear4bit
from transformers.models.phi3.modeling_phi3 import (
    Phi3Attention,
    Phi3RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from zassd.cache.kv_cache import StaticPreallocatedKVCache


@contextmanager
def fixed_row_phi3_schedule(model, *, physical_rows: int = 9,
                            attention_strategy: str = "rowwise", observer=None):
    if physical_rows < 2 or model.training:
        raise ValueError("fixed-row schedule requires physical_rows >= 2 and eval mode")
    if attention_strategy not in {"rowwise", "padded"}:
        raise ValueError("attention_strategy must be rowwise or padded")
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "phi3":
        raise TypeError("fixed-row Phi-3 schedule requires model_type='phi3'")
    if type(model).__name__ != "Phi3ForCausalLM":
        raise TypeError("fixed-row Phi-3 schedule requires Phi3ForCausalLM")
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise ValueError("fixed-row Phi-3 schedule requires SDPA")
    attention_modules = [m for m in model.modules() if isinstance(m, Phi3Attention)]
    if not attention_modules:
        raise TypeError("fixed-row Phi-3 schedule requires Phi3Attention modules")
    replacements = []

    original_model_forward = model.forward

    def guarded_model_forward(self, *args, _original=original_model_forward, **kwargs):
        forbidden = ("attention_mask", "position_ids", "inputs_embeds", "cache_position",
                     "block_sequence_ids", "position_embeddings")
        if any(kwargs.get(name) is not None for name in forbidden):
            raise ValueError("fixed-row Phi-3 schedule requires implicit positions and no external mask")
        if ((len(args) > 1 and args[1] is not None) or
                (len(args) > 2 and args[2] is not None) or
                (len(args) > 4 and args[4] is not None)):
            raise ValueError("fixed-row Phi-3 schedule requires implicit positions and no external mask")
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
        elif isinstance(module, Phi3RMSNorm):
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
        elif isinstance(module, Phi3Attention):
            original = module.forward

            def row_attention(self, hidden_states, position_embeddings=None,
                              attention_mask=None, past_key_values=None,
                              _original=original, **kwargs):
                if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
                    raise ValueError("fixed-row Phi-3 schedule requires batch size one")
                rows = hidden_states.shape[1]
                if kwargs.get("output_attentions") or self.training:
                    raise ValueError("unsupported Phi-3 attention mode for fixed-row schedule")
                if past_key_values is not None and not isinstance(
                        past_key_values, StaticPreallocatedKVCache):
                    raise ValueError("fixed-row Phi-3 schedule requires StaticPreallocatedKVCache")
                if past_key_values is None or past_key_values.get_seq_length(self.layer_idx) == 0:
                    return _original(hidden_states, position_embeddings, attention_mask,
                                     past_key_values=past_key_values, **kwargs)
                if rows > physical_rows:
                    raise ValueError("Phi-3 fixed-row decode exceeds configured physical rows")
                if position_embeddings is None:
                    raise ValueError("Phi-3 attention requires model-provided rotary embeddings")

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, self.head_dim)
                qkv = self.qkv_proj(hidden_states)
                query_width = self.config.num_attention_heads * self.head_dim
                kv_width = self.num_key_value_heads * self.head_dim
                query = qkv[..., :query_width].view(hidden_shape).transpose(1, 2)
                key = qkv[..., query_width:query_width + kv_width].view(hidden_shape).transpose(1, 2)
                value = qkv[..., query_width + kv_width:].view(hidden_shape).transpose(1, 2)
                cos, sin = position_embeddings
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
                key, value = past_key_values.update(key, value, self.layer_idx)
                prefix = key.shape[2] - rows
                key = repeat_kv(key, self.num_key_value_groups)
                value = repeat_kv(value, self.num_key_value_groups)
                sliding_window = getattr(self.config, "sliding_window", None)
                if attention_strategy == "rowwise":
                    outputs = []
                    for row in range(rows):
                        length = prefix + row + 1
                        start = max(0, length - sliding_window) if sliding_window else 0
                        row_query = query[:, :, row:row + 1, :].contiguous()
                        row_key = key[:, :, start:length, :].contiguous()
                        row_value = value[:, :, start:length, :].contiguous()
                        row_output = F.scaled_dot_product_attention(
                            row_query, row_key, row_value, attn_mask=None,
                            dropout_p=0.0, scale=self.scaling, is_causal=False,
                        )
                        if observer is not None:
                            observer(self.layer_idx, row, row_query, row_key,
                                     row_value, None, row_output)
                        outputs.append(row_output)
                    attn_output = torch.cat(outputs, dim=2)
                else:
                    query = F.pad(query, (0, 0, 0, physical_rows - rows)).contiguous()
                    key = F.pad(key, (0, 0, 0, physical_rows - rows)).contiguous()
                    value = F.pad(value, (0, 0, 0, physical_rows - rows)).contiguous()
                    mask = torch.full(
                        (1, 1, physical_rows, prefix + physical_rows), float("-inf"),
                        dtype=query.dtype, device=query.device,
                    )
                    for row in range(physical_rows):
                        end = prefix + row + 1
                        start = max(0, end - sliding_window) if sliding_window else 0
                        mask[:, :, row, start:end] = 0
                    attn_output = F.scaled_dot_product_attention(
                        query, key, value, attn_mask=mask, dropout_p=0.0,
                        scale=self.scaling, is_causal=False,
                    )[:, :, :rows, :]
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(*input_shape, -1)
                return self.o_proj(attn_output), None

            module.forward = MethodType(row_attention, module)
            replacements.append((module, original))
    try:
        yield model
    finally:
        for module, original in reversed(replacements):
            module.forward = original
