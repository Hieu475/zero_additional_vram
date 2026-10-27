"""Dispatch experimental fixed-row schedules only to registered architecture adapters."""

from __future__ import annotations

from contextlib import contextmanager

from zassd.certification.model_capabilities import FIXED_ROW_ADAPTERS


@contextmanager
def fixed_row_schedule_for_model(model, *, physical_rows: int = 9,
                                 attention_strategy: str = "rowwise", observer=None,
                                 reuse_linear_padding_buffer: bool = False,
                                 quantized_linear_strategy: str = "fixed_rows"):
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    adapter = FIXED_ROW_ADAPTERS.get(model_type)
    if adapter is None:
        raise TypeError(f"no fixed-row schedule is registered for model_type={model_type!r}")
    if attention_strategy not in adapter.attention_strategies:
        raise ValueError(
            f"attention strategy {attention_strategy!r} is not established for {model_type!r}"
        )
    if reuse_linear_padding_buffer and model_type != "llama":
        raise ValueError("reusable linear padding buffer is implemented only for Llama")
    if quantized_linear_strategy != "fixed_rows" and model_type != "llama":
        raise ValueError("rowwise quantized linears are implemented only for Llama")
    if model_type == "qwen2":
        from zassd.certification.qwen2_padded import fixed_row_qwen2_schedule

        schedule = fixed_row_qwen2_schedule
    elif model_type == "llama":
        from zassd.certification.llama_padded import fixed_row_llama_schedule

        schedule = fixed_row_llama_schedule
    elif model_type == "phi3":
        from zassd.certification.phi3_padded import fixed_row_phi3_schedule

        schedule = fixed_row_phi3_schedule
    elif model_type == "olmo2":
        from zassd.certification.olmo2_padded import fixed_row_olmo2_schedule

        schedule = fixed_row_olmo2_schedule
    else:
        raise TypeError(f"no fixed-row schedule implementation is registered for model_type={model_type!r}")
    with schedule(model, physical_rows=physical_rows,
                  attention_strategy=attention_strategy, observer=observer,
                  **({"reuse_linear_padding_buffer": reuse_linear_padding_buffer}
                     if model_type == "llama" else {}),
                  **({"quantized_linear_strategy": quantized_linear_strategy}
                     if model_type == "llama" else {})):
        yield model
