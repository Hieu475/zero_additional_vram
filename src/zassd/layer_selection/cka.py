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
    if X.dim() > 2:
        X = X.reshape(-1, X.shape[-1])
    if Y.dim() > 2:
        Y = Y.reshape(-1, Y.shape[-1])

    X = X.float()
    Y = Y.float()

    # Center the matrices
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)

    # Compute HSIC via Gram matrices or feature covariance
    # If n_samples >= features, feature covariance (D x D) is faster
    # HSIC(X, Y) = Tr(X X^T Y Y^T) / (n-1)^2 = ||Y^T X||_F^2 / (n-1)^2
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y

    hsic_xy = torch.sum(XtY ** 2)
    hsic_xx = torch.sum(XtX ** 2)
    hsic_yy = torch.sum(YtY ** 2)

    denominator = torch.sqrt(hsic_xx * hsic_yy)
    if denominator < 1e-10:
        return 0.0

    return float((hsic_xy / denominator).item())


def compute_cka_matrix(
    layer_activations: dict[int, torch.Tensor],
    device: str = "cpu",
) -> np.ndarray:
    """Compute full pairwise CKA matrix across all layers.

    Args:
        layer_activations: dict mapping layer_idx (0..L-1) to activation tensor (N, D).
        device: Device to use for computation ('cpu' or 'cuda').

    Returns:
        Square symmetric matrix of shape (num_layers, num_layers) with values in [0, 1].
    """
    num_layers = len(layer_activations)
    cka_matrix = np.zeros((num_layers, num_layers), dtype=np.float32)

    layer_tensors = {}
    for idx in range(num_layers):
        t = layer_activations[idx]
        if t.dim() > 2:
            t = t.reshape(-1, t.shape[-1])
        layer_tensors[idx] = t.to(device).float()

    for i in range(num_layers):
        cka_matrix[i, i] = 1.0
        for j in range(i + 1, num_layers):
            val = linear_cka(layer_tensors[i], layer_tensors[j])
            cka_matrix[i, j] = val
            cka_matrix[j, i] = val

    return cka_matrix


def compute_adjacent_similarity(cka_matrix: np.ndarray) -> np.ndarray:
    """Extract adjacent layer similarities CKA(i-1, i).

    Args:
        cka_matrix: (L, L) CKA matrix.

    Returns:
        1D array of length L-1 where element i is CKA(i, i+1).
    """
    num_layers = cka_matrix.shape[0]
    adjacent = np.zeros(num_layers - 1, dtype=np.float32)
    for i in range(num_layers - 1):
        adjacent[i] = cka_matrix[i, i + 1]
    return adjacent


def rank_layers_by_redundancy(
    cka_matrix: np.ndarray,
    always_keep: list[int] | None = None,
) -> list[tuple[int, float]]:
    """Rank layers by redundancy based on adjacent CKA similarity.

    A layer i has high redundancy if it is very similar to its adjacent neighbors
    (i-1 and i+1), indicating that the computation in layer i produces very
    little representational change.

    Args:
        cka_matrix: (L, L) pairwise CKA matrix.
        always_keep: Layers protected from skipping (default: [0, num_layers - 1]).

    Returns:
        List of (layer_idx, redundancy_score) sorted from most redundant to least redundant.
    """
    num_layers = cka_matrix.shape[0]
    if always_keep is None:
        always_keep = [0, num_layers - 1]

    # Convert negative indices if any
    protected = set()
    for idx in always_keep:
        if idx < 0:
            protected.add(num_layers + idx)
        else:
            protected.add(idx)

    redundancy_scores = []
    for i in range(num_layers):
        if i in protected:
            continue

        # Score based on similarity with previous and next layer
        prev_sim = cka_matrix[i - 1, i] if i > 0 else 0.0
        next_sim = cka_matrix[i, i + 1] if i < num_layers - 1 else 0.0

        # Also consider direct similarity between skip neighbors CKA(i-1, i+1)
        # If CKA(i-1, i+1) is high, skipping layer i preserves representation well!
        bypass_sim = cka_matrix[i - 1, i + 1] if (i > 0 and i < num_layers - 1) else 0.0

        score = float(0.4 * prev_sim + 0.4 * next_sim + 0.2 * bypass_sim)
        redundancy_scores.append((i, score))

    # Sort descending by redundancy score (highest = most redundant)
    redundancy_scores.sort(key=lambda x: x[1], reverse=True)
    return redundancy_scores


def select_layers_cka(
    cka_matrix: np.ndarray,
    num_to_skip: int,
    always_keep: list[int] | None = None,
    avoid_consecutive: bool = False,
) -> tuple[list[int], list[int]]:
    """Select layers to keep and skip based on CKA redundancy ranking.

    Args:
        cka_matrix: (L, L) CKA similarity matrix.
        num_to_skip: Target number of layers to skip.
        always_keep: Layer indices that must never be skipped.
        avoid_consecutive: If True, tries to avoid skipping adjacent layers.

    Returns:
        Tuple of (kept_layers, skipped_layers).
    """
    num_layers = cka_matrix.shape[0]
    ranked = rank_layers_by_redundancy(cka_matrix, always_keep=always_keep)

    skipped: list[int] = []
    for layer_idx, _ in ranked:
        if len(skipped) >= num_to_skip:
            break

        if avoid_consecutive:
            if (layer_idx - 1 in skipped) or (layer_idx + 1 in skipped):
                continue

        skipped.append(layer_idx)

    # If avoid_consecutive prevented filling the quota, relax it
    if len(skipped) < num_to_skip:
        for layer_idx, _ in ranked:
            if len(skipped) >= num_to_skip:
                break
            if layer_idx not in skipped:
                skipped.append(layer_idx)

    skipped = sorted(skipped)
    kept = sorted(set(range(num_layers)) - set(skipped))
    return kept, skipped
