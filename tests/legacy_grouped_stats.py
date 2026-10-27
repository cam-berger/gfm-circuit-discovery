"""Verbatim copy of grouped_stats.py at 081e85e, before it became an adapter over gfmcd.inference.

A same-platform oracle for the adapter parity test: golden floats captured on one CPU do not
survive another, but this code run beside the adapter must agree bit for bit. Do not edit.
"""

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


def ratio_of_means(numerator: np.ndarray, denominator: np.ndarray) -> float:
    """Ratio of exon-weighted means, keeping the denominator's sign.

    The byte-pinned ``circuit.ratio_of_means`` divides by ``max(sum, 1e-9)``, so a negative
    denominator sum returns about +/-1e9 times the numerator instead of the ratio (R55). This
    returns the arithmetic ratio and refuses only a zero denominator.
    """

    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    if numerator_array.shape != denominator_array.shape:
        raise ValueError("numerator and denominator must have matched shapes")
    keep = np.isfinite(numerator_array) & np.isfinite(denominator_array)
    if not keep.any():
        raise ValueError("no finite paired values")
    denominator_mean = float(denominator_array[keep].mean())
    if abs(denominator_mean) <= np.finfo(np.float64).eps:
        raise ValueError("ratio denominator mean is zero")
    return float(numerator_array[keep].mean()) / denominator_mean


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


def grouped_bootstrap_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float | int]:
    """Gene-grouped interval for a ratio of exon-weighted means.

    The numerator and denominator are resampled together by gene. This is a
    ratio of means, never a mean of per-example ratios.
    """

    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    groups_array = np.asarray(groups, dtype=object)
    if (
        numerator_array.shape != denominator_array.shape
        or numerator_array.shape != groups_array.shape
    ):
        raise ValueError("numerator, denominator, and groups must have matched shapes")
    keep = np.isfinite(numerator_array) & np.isfinite(denominator_array)
    numerator_array = numerator_array[keep]
    denominator_array = denominator_array[keep]
    groups_array = groups_array[keep]
    if numerator_array.size == 0:
        raise ValueError("no finite paired values")
    denominator_mean = float(denominator_array.mean())
    if abs(denominator_mean) <= np.finfo(np.float64).eps:
        raise ValueError("ratio denominator mean is zero")
    unique = np.unique(groups_array)
    by_group = {
        group: (
            numerator_array[groups_array == group],
            denominator_array[groups_array == group],
        )
        for group in unique
    }
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        sampled_numerator = np.concatenate(
            [by_group[group][0] for group in sampled]
        )
        sampled_denominator = np.concatenate(
            [by_group[group][1] for group in sampled]
        )
        denominator_value = float(sampled_denominator.mean())
        bootstrap[index] = (
            float(sampled_numerator.mean()) / denominator_value
            if abs(denominator_value) > np.finfo(np.float64).eps
            else np.nan
        )
    finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
    if finite_bootstrap.size < 0.99 * n_boot:
        raise ValueError("too many zero bootstrap denominators")
    return {
        "estimate": float(numerator_array.mean()) / denominator_mean,
        "ci_low": float(np.quantile(finite_bootstrap, 0.025)),
        "ci_high": float(np.quantile(finite_bootstrap, 0.975)),
        "n": int(numerator_array.size),
        "n_groups": int(len(unique)),
        "n_boot": int(n_boot),
    }
