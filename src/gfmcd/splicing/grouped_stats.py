"""Gene-grouped uncertainty helpers for paired splicing assays."""

from __future__ import annotations

import numpy as np


def grouped_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    """Gene-grouped percentile interval for the exon-weighted mean."""

    values_array = np.asarray(values, dtype=np.float64)
    groups_array = np.asarray(groups, dtype=object)
    keep = np.isfinite(values_array)
    values_array, groups_array = values_array[keep], groups_array[keep]
    if values_array.size == 0:
        raise ValueError("no finite values")
    unique = np.unique(groups_array)
    by_group = {
        group: values_array[groups_array == group]
        for group in unique
    }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        bootstrap[index] = np.concatenate(
            [by_group[group] for group in sampled]
        ).mean()
    return {
        "estimate": float(values_array.mean()),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "n": int(values_array.size),
        "n_groups": int(len(unique)),
        "n_boot": int(n_boot),
    }


def grouped_signflip_pvalue(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> float:
    """Two-sided gene-level sign flip with the exact add-one correction."""

    values_array = np.asarray(values, dtype=np.float64)
    groups_array = np.asarray(groups, dtype=object)
    keep = np.isfinite(values_array)
    values_array, groups_array = values_array[keep], groups_array[keep]
    if values_array.size == 0:
        raise ValueError("no finite values")
    unique, inverse = np.unique(groups_array, return_inverse=True)
    observed = abs(float(values_array.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(unique))
        permuted = float(np.mean(values_array * signs[inverse]))
        exceed += abs(permuted) >= observed
    return float((exceed + 1) / (n_perm + 1))
