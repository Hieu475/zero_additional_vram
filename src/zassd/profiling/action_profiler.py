"""Action profiler for measuring real candidate action costs.

Replaces hard-coded heuristic assumptions with empirically measured costs:
  Action: a = (S_i, K_j)
  Measured metrics:
    - draft_ms: mean draft latency per cycle
    - verify_ms: mean target verification latency per cycle
    - cache_ms: mean cache management latency per cycle
    - vram_mb: peak VRAM footprint
    - energy_j_token: estimated energy per token in Joules
    - acceptance_rate: empirical acceptance probability
    - tokens_per_step: effective tokens emitted per verification cycle

Produces results/raw/action_costs.json for consumption by the
HardwareAwareJointController (HECC).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from zassd.decoding.speculative import self_speculative_generate
from zassd.models.layer_manager import LayerManager
from zassd.profiling.gpu import GPUProfiler
from zassd.profiling.memory import get_vram_usage

logger = logging.getLogger(__name__)


@dataclass
class ActionCostRecord:
    """Empirical cost and efficiency profile for a single action (S_i, K_j)."""
    config_name: str
    k: int
    draft_ms: float
    verify_ms: float
    cache_ms: float
    other_ms: float
    total_cycle_ms: float
    tokens_per_step: float
    acceptance_rate: float
    tokens_per_second: float
    vram_mb: float
    energy_j_token: float


class ActionCostDatabase:
    """Database of measured action costs loaded from JSON."""

    def __init__(self, data: Optional[dict[str, dict[str, Any]]] = None) -> None:
        self.data: dict[str, dict[str, Any]] = data if data is not None else {}

    @classmethod
    def from_file(cls, filepath: str | Path) -> ActionCostDatabase:
        """Load database from action_costs.json."""
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"Action cost database not found at {path}. Using empty database.")
            return cls({})
        with open(path) as f:
            raw_data = json.load(f)
        return cls(raw_data)

    def get_action_cost(self, config_name: str, k: int) -> Optional[dict[str, Any]]:
        """Retrieve cost metrics for candidate action (S, K)."""
        k_key = f"K{k}"
        return self.data.get(config_name, {}).get(k_key)

    def get_measured_cycle_ms(self, config_name: str, k: int, default: float = 25.0) -> float:
        """Get measured total cycle latency in ms."""
        cost = self.get_action_cost(config_name, k)
        if cost and "total_cycle_ms" in cost:
            return float(cost["total_cycle_ms"])
        return default

    def get_measured_tokens_per_step(self, config_name: str, k: int, default: float = 1.8) -> float:
        """Get empirical tokens per step."""
        cost = self.get_action_cost(config_name, k)
        if cost and "tokens_per_step" in cost:
            return float(cost["tokens_per_step"])
        return default


class ActionProfiler:
    """Benchmarks and records real hardware costs for action space (S, K)."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layer_mgr: LayerManager,
        candidate_configs: dict[str, list[int]],
        gpu_profiler: Optional[GPUProfiler] = None,
        device: str = "cuda:0",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer_mgr = layer_mgr
        self.candidate_configs = candidate_configs
        self.gpu_profiler = gpu_profiler or GPUProfiler()
        self.device = device

    def profile_action(
        self,
        config_name: str,
        k: int,
        eval_prompts: list[dict[str, Any]],
        max_new_tokens: int = 48,
    ) -> ActionCostRecord:
        """Benchmark a single candidate action (S_i, K_j) over evaluation prompts."""
        skip_indices = self.candidate_configs[config_name]
        logger.info(f"Profiling action ({config_name}, K={k}) across {len(eval_prompts)} prompts...")

        runs = []
        for p in eval_prompts:
            _, s_metrics = self_speculative_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                layer_mgr=self.layer_mgr,
                skip_indices=skip_indices,
                prompt=p["prompt"],
                k=k,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                device=self.device,
            )
            runs.append(s_metrics)

        tot_cycles = max(1, sum(m.num_verification_cycles for m in runs))
        draft_time_sum = sum(m.draft_time_s for m in runs)
        verify_time_sum = sum(m.verify_time_s for m in runs)
        cache_time_sum = sum(m.cache_time_s for m in runs)
        other_time_sum = sum(m.other_time_s for m in runs)
        total_time_sum = sum(m.total_time_s for m in runs)
        total_tok_sum = max(1, sum(m.total_tokens for m in runs))

        # Query GPU power usage if available
        try:
            gpu_power_w = self.gpu_profiler.get_power_usage() if self.gpu_profiler else 60.0
        except Exception:
            gpu_power_w = 60.0
        energy_j_token = (gpu_power_w * total_time_sum) / total_tok_sum

        record = ActionCostRecord(
            config_name=config_name,
            k=k,
            draft_ms=float((draft_time_sum / tot_cycles) * 1000),
            verify_ms=float((verify_time_sum / tot_cycles) * 1000),
            cache_ms=float((cache_time_sum / tot_cycles) * 1000),
            other_ms=float((other_time_sum / tot_cycles) * 1000),
            total_cycle_ms=float((total_time_sum / tot_cycles) * 1000),
            tokens_per_step=float(np.mean([m.tokens_per_step for m in runs])),
            acceptance_rate=float(np.mean([m.acceptance_rate for m in runs])),
            tokens_per_second=float(np.mean([m.tokens_per_second for m in runs])),
            vram_mb=float(np.mean([m.peak_vram_mb for m in runs])),
            energy_j_token=float(energy_j_token),
        )
        return record

    def profile_all(
        self,
        k_values: list[int],
        eval_prompts: list[dict[str, Any]],
        output_file: str | Path = "results/raw/action_costs.json",
    ) -> dict[str, dict[str, Any]]:
        """Profile all candidate pairs (S, K) and save to JSON database."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        database: dict[str, dict[str, Any]] = {}

        for config_name in self.candidate_configs:
            database[config_name] = {}
            for k in k_values:
                rec = self.profile_action(config_name, k, eval_prompts)
                k_key = f"K{k}"
                database[config_name][k_key] = asdict(rec)

        with open(output_path, "w") as f:
            json.dump(database, f, indent=2)

        logger.info(f"Action cost database successfully written to {output_path}")
        return database
