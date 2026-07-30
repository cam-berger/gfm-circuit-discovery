"""Permutation and multiplicity utilities with the repository's add-one convention."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from itertools import product
from statistics import mean


def add_one_permutation_pvalue(
    observed: float,
    null_statistics: Sequence[float],
    *,
    alternative: str = "greater",
) -> float:
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less, or two-sided")
    if not null_statistics:
        raise ValueError("null distribution cannot be empty")
    if not math.isfinite(observed) or any(
        not math.isfinite(x) for x in null_statistics
    ):
        raise ValueError("permutation statistics must be finite")
    if alternative == "greater":
        extreme = sum(value >= observed for value in null_statistics)
    elif alternative == "less":
        extreme = sum(value <= observed for value in null_statistics)
    else:
        extreme = sum(abs(value) >= abs(observed) for value in null_statistics)
    return (extreme + 1) / (len(null_statistics) + 1)


def max_t_adjusted_pvalues(
    observed: Mapping[str, float],
    null_by_permutation: Sequence[Mapping[str, float]],
    *,
    two_sided: bool = True,
) -> dict[str, float]:
    """Single-step max-T correction over the complete scanned family."""

    family = tuple(sorted(observed))
    if not family or not null_by_permutation:
        raise ValueError("observed family and null permutations cannot be empty")
    for permutation in null_by_permutation:
        if set(permutation) != set(family):
            raise ValueError(
                "every permutation must contain the complete scanned family"
            )
    maxima = [
        max(
            abs(permutation[name]) if two_sided else permutation[name]
            for name in family
        )
        for permutation in null_by_permutation
    ]
    return {
        name: add_one_permutation_pvalue(
            abs(value) if two_sided else value,
            maxima,
            alternative="greater",
        )
        for name, value in observed.items()
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    if not pvalues:
        return {}
    if any(not 0 <= value <= 1 for value in pvalues.values()):
        raise ValueError("p-values must fall in [0, 1]")
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, (total - rank) * pvalues[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def paired_cluster_sign_permutation(
    differences: Sequence[float],
    clusters: Sequence[str],
    *,
    permutations: int = 9999,
    seed: int = 20260727,
) -> tuple[float, list[float]]:
    """Cluster-first sign permutation for a paired mean effect."""

    if len(differences) != len(clusters) or not differences:
        raise ValueError("differences and clusters must be non-empty and aligned")
    grouped: dict[str, list[float]] = defaultdict(list)
    for difference, cluster in zip(differences, clusters):
        if not math.isfinite(difference):
            raise ValueError("differences must be finite")
        grouped[str(cluster)].append(float(difference))
    cluster_values = [mean(grouped[key]) for key in sorted(grouped)]
    observed = mean(cluster_values)
    n = len(cluster_values)
    if n <= 16:
        signs = product((-1.0, 1.0), repeat=n)
        null = [
            mean(sign * value for sign, value in zip(draw, cluster_values))
            for draw in signs
        ]
    else:
        if permutations < 1:
            raise ValueError("permutations must be positive")
        rng = random.Random(seed)
        null = [
            mean(
                (1.0 if rng.getrandbits(1) else -1.0) * value
                for value in cluster_values
            )
            for _ in range(permutations)
        ]
    return observed, null


def cluster_bootstrap_interval(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    statistic: Callable[[Sequence[float]], float] = mean,
    draws: int = 5000,
    seed: int = 20260727,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if len(values) != len(clusters) or not values:
        raise ValueError("values and clusters must be non-empty and aligned")
    if draws < 1 or not 0 < alpha < 1:
        raise ValueError("invalid bootstrap parameters")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[str(cluster)].append(float(value))
    keys = sorted(grouped)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(draws):
        sampled = [rng.choice(keys) for _ in keys]
        pool = [value for key in sampled for value in grouped[key]]
        estimates.append(float(statistic(pool)))
    estimates.sort()
    lo_index = max(0, math.floor((alpha / 2) * draws))
    hi_index = min(draws - 1, math.ceil((1 - alpha / 2) * draws) - 1)
    return estimates[lo_index], estimates[hi_index]
