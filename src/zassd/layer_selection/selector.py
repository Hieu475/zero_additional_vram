"""Unified layer selector interface."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SelectionMethod(Enum):
    """Available layer selection methods."""
    RANDOM = "random"
    EVEN = "even"
    ODD = "odd"
    MIDDLE = "middle"
    CKA = "cka"


class LayerSelector:
    """Unified interface for layer selection."""

    def __init__(
        self,
        num_layers: int,
        method: SelectionMethod = SelectionMethod.RANDOM,
        **kwargs: Any,
    ) -> None:
        self.num_layers = num_layers
        self.method = method
        self.kwargs = kwargs
        self._kept: list[int] = []
        self._skipped: list[int] = []

    def select(self) -> tuple[list[int], list[int]]:
        """Perform layer selection.

        Returns:
            Tuple of (kept_indices, skipped_indices).
        """
        if self.method == SelectionMethod.RANDOM:
            from .random import select_random_layers
            self._kept, self._skipped = select_random_layers(
                self.num_layers, **self.kwargs
            )
        elif self.method == SelectionMethod.EVEN:
            from .static import select_even_layers
            self._kept, self._skipped = select_even_layers(self.num_layers)
        elif self.method == SelectionMethod.ODD:
            from .static import select_odd_layers
            self._kept, self._skipped = select_odd_layers(self.num_layers)
        elif self.method == SelectionMethod.MIDDLE:
            from .static import select_middle_skip
            self._kept, self._skipped = select_middle_skip(
                self.num_layers, **self.kwargs
            )
        elif self.method == SelectionMethod.CKA:
            from .cka import select_layers_cka
            cka_matrix = self.kwargs.get("cka_matrix")
            if cka_matrix is None:
                raise ValueError("CKA method requires 'cka_matrix' in kwargs")
            num_to_skip = self.kwargs.get(
                "num_to_skip",
                int(self.num_layers * self.kwargs.get("skip_ratio", 0.5)),
            )
            always_keep = self.kwargs.get("always_keep", [0, -1])
            avoid_consecutive = self.kwargs.get("avoid_consecutive", False)
            self._kept, self._skipped = select_layers_cka(
                cka_matrix=cka_matrix,
                num_to_skip=num_to_skip,
                always_keep=always_keep,
                avoid_consecutive=avoid_consecutive,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self._kept, self._skipped

    @property
    def kept(self) -> list[int]:
        return self._kept

    @property
    def skipped(self) -> list[int]:
        return self._skipped

    @property
    def skip_ratio(self) -> float:
        if not self._skipped:
            return 0.0
        return len(self._skipped) / self.num_layers
