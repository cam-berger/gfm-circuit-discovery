"""Pure helpers for joint-head and low-rank native-context assays."""

from __future__ import annotations

from itertools import combinations
import hashlib
from typing import Any, Sequence

import numpy as np
import torch


def ordered_nonempty_subsets(heads: Sequence[int]) -> list[tuple[int, ...]]:
    """Return every non-empty subset, ordered first by size then input order."""

    values = tuple(int(head) for head in heads)
    if not values or len(set(values)) != len(values):
        raise ValueError("heads must be a non-empty unique sequence")
    return [
        subset
        for size in range(1, len(values) + 1)
        for subset in combinations(values, size)
    ]


def matched_control_sets(
    candidate_order: Sequence[int],
    control_heads: dict[int, Sequence[int]],
    *,
    size: int,
) -> list[tuple[int, ...]]:
    """Take an outcome-independent size-matched prefix from each control block."""

    candidates = tuple(int(head) for head in candidate_order)
    if not 1 <= size <= len(candidates):
        raise ValueError("control size must match a candidate subset size")
    if set(candidates) != {int(head) for head in control_heads}:
        raise ValueError("candidate/control keys disagree")
    sets = [
        tuple(int(head) for head in control_heads[candidate][:size])
        for candidate in candidates
    ]
    if any(len(heads) != size for heads in sets):
        raise ValueError("a control block is too short")
    if any(len(set(heads)) != size for heads in sets):
        raise ValueError("a matched control set repeats a head")
    return sets


def group_mean_rows(
    matrix: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Average a matrix and scalar outcome within sorted groups."""

    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    labels = np.asarray(groups, dtype=object)
    if x.ndim != 2 or y.ndim != 1 or labels.ndim != 1:
        raise ValueError("matrix must be 2D and values/groups must be 1D")
    if not (x.shape[0] == y.shape[0] == labels.shape[0]):
        raise ValueError("matrix, values, and groups must share rows")
    names = sorted({str(group) for group in labels})
    grouped_x = np.stack(
        [x[labels == name].mean(axis=0) for name in names],
        axis=0,
    )
    grouped_y = np.asarray(
        [y[labels == name].mean() for name in names],
        dtype=np.float64,
    )
    return names, grouped_x, grouped_y


def fit_lowrank_structure(
    matrix: np.ndarray,
    values: np.ndarray,
    groups: np.ndarray,
    *,
    max_rank: int = 64,
) -> dict[str, Any]:
    """Fit an uncentered SVD basis and a task-covariance direction on gene means."""

    names, grouped_x, grouped_y = group_mean_rows(matrix, values, groups)
    available_rank = min(grouped_x.shape)
    if max_rank < 1 or max_rank > available_rank:
        raise ValueError(
            f"max_rank={max_rank} exceeds available rank {available_rank}"
        )
    _, singular_values, vt = np.linalg.svd(grouped_x, full_matrices=False)
    basis = vt[:max_rank]
    direction = grouped_x.T @ grouped_y
    norm = float(np.linalg.norm(direction))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("task-covariance direction has zero norm")
    direction = direction / norm
    if float(grouped_x.mean(axis=0) @ direction) < 0:
        direction = -direction
    energy = np.square(singular_values)
    total_energy = float(energy.sum())
    cumulative = np.cumsum(energy) / total_energy
    return {
        "gene_ids": names,
        "gene_matrix": grouped_x,
        "gene_behavior": grouped_y,
        "basis": basis,
        "singular_values": singular_values,
        "cumulative_energy": cumulative,
        "task_direction": direction,
    }


def deterministic_random_basis(
    width: int,
    rank: int,
    *,
    tag: str,
    draw: int,
) -> np.ndarray:
    """Generate a deterministic Haar-random row-orthonormal basis."""

    if not 1 <= rank <= width:
        raise ValueError("rank must be in [1, width]")
    if draw < 0 or not tag:
        raise ValueError("draw must be non-negative and tag non-empty")
    digest = hashlib.sha256(f"{tag}:{draw}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    generator = np.random.default_rng(seed)
    matrix = generator.standard_normal((width, rank))
    q, _ = np.linalg.qr(matrix, mode="reduced")
    return q.T


def projection_matrix_error(basis: np.ndarray) -> float:
    """Maximum row-orthonormality error for a basis."""

    values = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("basis must be 2D")
    if values.shape[0] == 0:
        return 0.0
    identity = np.eye(values.shape[0], dtype=np.float64)
    return float(np.max(np.abs(values @ values.T - identity)))


def project_rows(matrix: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project row vectors into the span of a row-orthonormal basis."""

    values = np.asarray(matrix, dtype=np.float64)
    vectors = np.asarray(basis, dtype=np.float64)
    if values.ndim != 2 or vectors.ndim != 2:
        raise ValueError("matrix and basis must be 2D")
    if values.shape[1] != vectors.shape[1]:
        raise ValueError("matrix and basis widths disagree")
    return (values @ vectors.T) @ vectors


def replace_target_vectors(
    write: torch.Tensor,
    token_positions: torch.Tensor,
    vectors: torch.Tensor,
) -> torch.Tensor:
    """Return a write tensor with one target vector replaced per example."""

    if write.ndim != 3 or token_positions.ndim != 1 or vectors.ndim != 2:
        raise ValueError("expected write [B,S,D], positions [B], vectors [B,D]")
    if write.shape[0] != token_positions.shape[0] or (
        vectors.shape != (write.shape[0], write.shape[2])
    ):
        raise ValueError("target-vector batch or width mismatch")
    positions = token_positions.to(device=write.device, dtype=torch.long)
    if torch.any(positions < 0) or torch.any(positions >= write.shape[1]):
        raise ValueError("target position outside write tensor")
    output = write.clone()
    rows = torch.arange(write.shape[0], device=write.device)
    output[rows, positions] = vectors.to(
        device=write.device,
        dtype=write.dtype,
    )
    return output


__all__ = [
    "deterministic_random_basis",
    "fit_lowrank_structure",
    "group_mean_rows",
    "matched_control_sets",
    "ordered_nonempty_subsets",
    "project_rows",
    "projection_matrix_error",
    "replace_target_vectors",
]
