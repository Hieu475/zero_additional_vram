"""Hardware-aware joint speculation controller.

This is the main novel contribution of the research.
The controller jointly optimizes:
- Layer configuration (which layers to skip)
- Draft length K

Subject to hardware constraints:
- VRAM budget
- Latency target
- Energy budget
- GPU utilization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HardwareState:
    """Current hardware state observation."""
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    gpu_utilization: float = 0.0
    gpu_power_w: float = 0.0
    gpu_temperature_c: float = 0.0


@dataclass
class ControllerAction:
    """Controller output action."""
    skip_indices: list[int]     # Layers to skip (S_t)
    draft_length: int           # Number of draft tokens (K_t)


@dataclass
class ControllerObservation:
    """Full observation vector z_t."""
    entropy: float              # H_t
    acceptance_rate: float      # A_{t-1}
    draft_latency_ms: float     # T_draft
    verify_latency_ms: float    # T_verify
    vram_used_mb: float         # VRAM_t
    gpu_power_w: float          # P_t
    gpu_utilization: float      # U_t


# TODO: Implement after experiments 01-05 are complete
# This is the final integration that combines all components:
# - CKA layer selection (from layer_selection/cka.py)
# - Adaptive K (from controllers/adaptive_k.py)
# - Hardware profiling (from profiling/)
# - Utility function U = λ1·Speedup - λ2·Latency - λ3·VRAM - λ4·Energy
