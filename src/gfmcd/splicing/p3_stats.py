"""Gene-first inference helpers for the P3 closure study."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm

from gfmcd.splicing.p3_closure import local_recovery_contrast


@dataclass(frozen=True)
class SimultaneousResult:
    names: tuple[str, ...]
    means: tuple[float, ...]
    standard_errors: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    adjusted_p: tuple[float, ...]
    critical_value: float
    n_genes: int
    n_bootstrap: int


def gene_paired_phase_table(
    rows: Sequence[Mapping[str, object]],
    *,
    metrics: Sequence[str],
    target_context: str = "cds",
    control_context: str = "upstream_intron",
    centers: Sequence[int] = (3, 6, 9),
) -> tuple[dict[str, object], ...]:
    """Average repeated edits inside gene/condition, then apply the exact local statistic."""

    genes = sorted({str(row["gene_id"]) for row in rows})
    output = []
    for gene in genes:
        gene_rows = [row for row in rows if str(row["gene_id"]) == gene]
        result: dict[str, object] = {"gene_id": gene}
        chromosome = {str(row["chrom"]) for row in gene_rows}
        if len(chromosome) != 1:
            raise ValueError(f"gene {gene} spans multiple chromosomes in result rows")
        result["chrom"] = next(iter(chromosome))
        for metric in metrics:
            context_values = {}
            for context in (target_context, control_context):
                amount_values = {}
                amounts = sorted({offset for center in centers for offset in (
                    int(center) - 1, int(center), int(center) + 1
                )})
                for amount in amounts:
                    values = [
                        float(row[metric]) for row in gene_rows
                        if row["context"] == context and int(row["amount"]) == amount
                    ]
                    if not values or not all(math.isfinite(value) for value in values):
                        raise ValueError(
                            f"missing/nonfinite {gene}/{context}/r={amount}/{metric}"
                        )
                    amount_values[amount] = float(np.mean(values))
                context_values[context] = local_recovery_contrast(
                    amount_values, centers=centers
                )
                result[f"s_{metric}_{context}"] = context_values[context]
            result[f"psi_{metric}"] = (
                context_values[target_context] - context_values[control_context]
            )
        output.append(result)
    return tuple(output)


def bootstrap_studentized_max_t(
    matrix: np.ndarray,
    *,
    names: Sequence[str],
    seed: int,
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
) -> SimultaneousResult:
    """Shared-gene studentized max-|t| intervals with add-one adjusted p-values."""

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("matrix columns must match endpoint names")
    if values.shape[0] < 3 or not np.isfinite(values).all():
        raise ValueError("at least three finite gene rows are required")
    if n_bootstrap < 99 or not (0 < alpha < 1):
        raise ValueError("invalid bootstrap count or alpha")
    n = values.shape[0]
    means = values.mean(axis=0)
    standard_errors = values.std(axis=0, ddof=1) / math.sqrt(n)
    if np.any(standard_errors <= 0):
        raise ValueError("every endpoint needs nonzero between-gene variance")
    rng = np.random.default_rng(seed)
    max_t = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = values[rng.integers(0, n, size=n)]
        sample_mean = sample.mean(axis=0)
        sample_se = sample.std(axis=0, ddof=1) / math.sqrt(n)
        if np.any(sample_se <= 0):
            max_t[index] = np.nan
        else:
            max_t[index] = np.max(np.abs((sample_mean - means) / sample_se))
    max_t = max_t[np.isfinite(max_t)]
    if len(max_t) < int(0.95 * n_bootstrap):
        raise RuntimeError("too many degenerate bootstrap draws")
    critical = float(np.quantile(max_t, 1 - alpha, method="higher"))
    lower = means - critical * standard_errors
    upper = means + critical * standard_errors
    observed_t = np.abs(means / standard_errors)
    adjusted_p = np.asarray([
        (1 + np.sum(max_t >= statistic)) / (len(max_t) + 1)
        for statistic in observed_t
    ])
    return SimultaneousResult(
        names=tuple(names),
        means=tuple(float(x) for x in means),
        standard_errors=tuple(float(x) for x in standard_errors),
        lower=tuple(float(x) for x in lower),
        upper=tuple(float(x) for x in upper),
        adjusted_p=tuple(float(x) for x in adjusted_p),
        critical_value=critical,
        n_genes=n,
        n_bootstrap=n_bootstrap,
    )


def mde80(
    values: Sequence[float],
    *,
    critical_value: float,
    power: float = 0.80,
) -> float:
    """Normal-approximation two-sided MDE using the observed gene-level SD."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 3 or not np.isfinite(array).all():
        raise ValueError("MDE requires at least three finite gene values")
    if not (0.5 < power < 1) or critical_value <= 0:
        raise ValueError("invalid power or critical value")
    return float((critical_value + norm.ppf(power)) * array.std(ddof=1) / math.sqrt(len(array)))


def interval_classification(lower: float, upper: float, *, sesoi: float) -> str:
    if lower > sesoi or upper < -sesoi:
        return "material"
    if lower > 0 or upper < 0:
        return "detectable_sub_sesoi_or_materiality_unresolved"
    if lower >= -sesoi and upper <= sesoi:
        return "bounded_null"
    return "inconclusive"


__all__ = [
    "SimultaneousResult",
    "bootstrap_studentized_max_t",
    "gene_paired_phase_table",
    "interval_classification",
    "mde80",
]
