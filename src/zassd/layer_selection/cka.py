"""Centered Kernel Alignment (CKA) for layer similarity analysis.

CKA measures representational similarity between layers.
Highly similar adjacent layers are candidates for skipping.

Reference:
    Kornblith et al., "Similarity of Neural Network Representations Revisited", ICML 2019
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def linear_cka(
    X: torch.Tensor,
    Y: torch.Tensor,
) -> float:
    """Compute linear CKA between two representation matrices.

    Args:
        X: Activations from layer i, shape (n_samples, features_i).
        Y: Activations from layer j, shape (n_samples, features_j).

    Returns:
        CKA similarity score in [0, 1].
    """
    # Center the matrices
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute HSIC
    XtX = X.T @ X
    YtY = Y.T @ Y
    XtY = X.T @ Y

    hsic_xy = torch.sum(XtY ** 2)
    hsic_xx = torch.sum(XtX ** 2)
    hsic_yy = torch.sum(YtY ** 2)

    denominator = torch.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-10:
        return 0.0

    return (hsic_xy / denominator).item()


# TODO: Implement full CKA analysis pipeline
# - collect_activations(model, dataloader) -> dict[int, Tensor]
# - compute_cka_matrix(activations) -> np.ndarray
# - select_layers_by_cka(cka_matrix, threshold) -> list[int]
