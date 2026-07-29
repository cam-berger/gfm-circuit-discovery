"""Exact native-unit helpers for the final SpliceBERT MLP."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np


def composed_classifier_margin(
    residual: np.ndarray,
    write: np.ndarray,
    *,
    layer_norm_weight: np.ndarray,
    layer_norm_bias: np.ndarray,
    layer_norm_eps: float,
    classifier_direction: np.ndarray,
    classifier_bias: float,
) -> np.ndarray:
    """Apply the exact final LayerNorm and linear class-margin readout."""

    residual_values = np.asarray(residual, dtype=np.float64)
    write_values = np.asarray(write, dtype=np.float64)
    if (
        residual_values.ndim < 2
        or write_values.ndim < 2
        or residual_values.shape[-1] != write_values.shape[-1]
    ):
        raise ValueError("residual and write must share a broadcastable width")
    residual_values, write_values = np.broadcast_arrays(
        residual_values, write_values
    )
    width = residual_values.shape[-1]
    weight = np.asarray(layer_norm_weight, dtype=np.float64)
    bias = np.asarray(layer_norm_bias, dtype=np.float64)
    direction = np.asarray(classifier_direction, dtype=np.float64)
    if weight.shape != (width,) or bias.shape != (width,) or direction.shape != (
        width,
    ):
        raise ValueError("LayerNorm/classifier parameters have the wrong width")
    values = residual_values + write_values
    mean = values.mean(axis=-1, keepdims=True)
    variance = np.square(values - mean).mean(axis=-1, keepdims=True)
    hidden = (values - mean) / np.sqrt(variance + float(layer_norm_eps))
    hidden = hidden * weight + bias
    return hidden @ direction + float(classifier_bias)


def unit_group_write_delta(
    activation_delta: np.ndarray,
    output_weight: np.ndarray,
    units: Sequence[int],
) -> np.ndarray:
    """Return the exact MLP-write change contributed by selected native units."""

    delta = np.asarray(activation_delta, dtype=np.float64)
    weight = np.asarray(output_weight, dtype=np.float64)
    selected = np.asarray(tuple(int(unit) for unit in units), dtype=np.int64)
    if delta.ndim != 2 or weight.ndim != 2:
        raise ValueError("activation delta and output weight must be 2D")
    if delta.shape[1] != weight.shape[1]:
        raise ValueError("activation and output-weight unit widths disagree")
    if selected.ndim != 1 or len(np.unique(selected)) != selected.size:
        raise ValueError("units must be a unique 1D sequence")
    if selected.size == 0:
        return np.zeros((delta.shape[0], weight.shape[0]), dtype=np.float64)
    if np.any(selected < 0) or np.any(selected >= delta.shape[1]):
        raise ValueError("unit index outside MLP intermediate width")
    return delta[:, selected] @ weight[:, selected].T


def exact_single_unit_effects(
    activation_delta: np.ndarray,
    output_weight: np.ndarray,
    *,
    clean_residual: np.ndarray,
    clean_write: np.ndarray,
    corrupt_residual: np.ndarray,
    corrupt_write: np.ndarray,
    clean_margin: np.ndarray,
    corrupt_margin: np.ndarray,
    parameters: dict[str, Any],
    chunk_size: int = 128,
) -> dict[str, np.ndarray]:
    """Evaluate every native unit as an exact one-unit interchange intervention."""

    delta = np.asarray(activation_delta, dtype=np.float64)
    weight = np.asarray(output_weight, dtype=np.float64)
    if delta.ndim != 2 or weight.shape[1] != delta.shape[1]:
        raise ValueError("unit activation and output-weight shapes disagree")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    n_examples, n_units = delta.shape
    damage = np.empty((n_examples, n_units), dtype=np.float32)
    recovery = np.empty((n_examples, n_units), dtype=np.float32)
    for start in range(0, n_units, chunk_size):
        stop = min(start + chunk_size, n_units)
        contributions = (
            delta[:, start:stop, None]
            * weight[:, start:stop].T[None, :, :]
        )
        noised = composed_classifier_margin(
            clean_residual[:, None, :],
            clean_write[:, None, :] - contributions,
            **parameters,
        )
        denoised = composed_classifier_margin(
            corrupt_residual[:, None, :],
            corrupt_write[:, None, :] + contributions,
            **parameters,
        )
        damage[:, start:stop] = clean_margin[:, None] - noised
        recovery[:, start:stop] = denoised - corrupt_margin[:, None]
    return {"damage": damage, "recovery": recovery}


def rank_units_bidirectional(
    damage: np.ndarray,
    recovery: np.ndarray,
    groups: np.ndarray,
) -> dict[str, np.ndarray]:
    """Rank all units by their weaker gene-mean causal direction."""

    damage_values = np.asarray(damage, dtype=np.float64)
    recovery_values = np.asarray(recovery, dtype=np.float64)
    labels = np.asarray(groups, dtype=object)
    if damage_values.shape != recovery_values.shape or damage_values.ndim != 2:
        raise ValueError("damage and recovery must be same-shaped 2D arrays")
    if labels.shape != (damage_values.shape[0],):
        raise ValueError("groups must match the example dimension")
    names = sorted({str(value) for value in labels})
    gene_damage = np.stack(
        [damage_values[labels == name].mean(axis=0) for name in names]
    )
    gene_recovery = np.stack(
        [recovery_values[labels == name].mean(axis=0) for name in names]
    )
    mean_damage = gene_damage.mean(axis=0)
    mean_recovery = gene_recovery.mean(axis=0)
    score = np.minimum(mean_damage, mean_recovery)
    unit_ids = np.arange(score.size)
    order = np.lexsort((unit_ids, -score))
    return {
        "order": order,
        "score": score,
        "mean_damage": mean_damage,
        "mean_recovery": mean_recovery,
    }


def output_norm_matched_controls(
    candidates: Sequence[int],
    output_norms: np.ndarray,
    *,
    tag: str,
    draws: int = 3,
) -> list[np.ndarray]:
    """Build deterministic equal-count controls nearest in output-column norm."""

    selected = np.asarray(tuple(int(unit) for unit in candidates), dtype=np.int64)
    norms = np.asarray(output_norms, dtype=np.float64)
    if norms.ndim != 1 or selected.ndim != 1 or selected.size < 1:
        raise ValueError("candidates and output norms must be non-empty 1D arrays")
    if len(np.unique(selected)) != selected.size:
        raise ValueError("candidate units must be unique")
    if selected.size > norms.size // 2:
        raise ValueError("matched controls require at least as many non-candidates")
    if np.any(selected < 0) or np.any(selected >= norms.size):
        raise ValueError("candidate unit outside output norm array")
    if not tag or draws < 1:
        raise ValueError("tag must be non-empty and draws positive")
    available = np.setdiff1d(np.arange(norms.size), selected, assume_unique=True)
    controls = []
    for draw in range(draws):
        digest = hashlib.sha256(f"{tag}:{draw}".encode("utf-8")).digest()
        generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        jitter = generator.uniform(0.0, 1e-12, size=available.size)
        remaining = available.copy()
        remaining_jitter = jitter.copy()
        chosen: list[int] = []
        candidate_order = selected[generator.permutation(selected.size)]
        for candidate in candidate_order:
            distances = np.abs(norms[remaining] - norms[candidate])
            index = int(np.argmin(distances + remaining_jitter))
            chosen.append(int(remaining[index]))
            remaining = np.delete(remaining, index)
            remaining_jitter = np.delete(remaining_jitter, index)
        controls.append(np.asarray(sorted(chosen), dtype=np.int64))
    return controls


__all__ = [
    "composed_classifier_margin",
    "exact_single_unit_effects",
    "output_norm_matched_controls",
    "rank_units_bidirectional",
    "unit_group_write_delta",
]
