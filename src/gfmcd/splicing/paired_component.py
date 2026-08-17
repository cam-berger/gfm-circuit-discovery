"""Gene-clustered simultaneous inference for paired component families.

The exchangeable unit is a gene, never a row.  Confidence intervals use a nonparametric
gene-cluster bootstrap.  Tests of a zero effect use one Rademacher sign per gene applied to
the *raw cluster sums*.  Family decisions use the maximum absolute statistic over the full
component family.

Production inference requires at least 30 independent genes by default.  Smoke runs and
unit tests may request a smaller threshold explicitly, but inference with fewer than two
clusters is never defined.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution


DEFAULT_MIN_GENES = 30


def _as_family(values: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.ndim == 1:
        out = out[None, :]
    if out.ndim != 2 or out.shape[0] == 0 or out.shape[1] == 0:
        raise ValueError("values must have shape [components, examples]")
    if not np.isfinite(out).all():
        raise ValueError("values contain non-finite entries")
    return out


def _group_index(groups: Sequence[str], n_examples: int) -> tuple[list[str], np.ndarray]:
    if len(groups) != n_examples:
        raise ValueError("groups must have one entry per example")
    rendered = [str(group) for group in groups]
    if any(not group for group in rendered):
        raise ValueError("group labels must be non-empty")
    labels = sorted(set(rendered))
    if not labels:
        raise ValueError("at least one group is required")
    lookup = {label: i for i, label in enumerate(labels)}
    return labels, np.asarray([lookup[group] for group in rendered], dtype=int)


def _validate_min_genes(min_genes: int) -> int:
    value = int(min_genes)
    if value < 2:
        raise ValueError("min_genes must be at least 2")
    return value


def _cluster_summary(
    family: np.ndarray,
    groups: Sequence[str],
    *,
    min_genes: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    required = _validate_min_genes(min_genes)
    labels, index = _group_index(groups, family.shape[1])
    if len(labels) < required:
        raise ValueError(
            f"at least {required} independent gene clusters are required; got {len(labels)}"
        )
    sizes = np.bincount(index, minlength=len(labels)).astype(float)
    sums = np.vstack(
        [np.bincount(index, weights=row, minlength=len(labels)) for row in family]
    )
    return labels, index, sizes, sums


def _seed(namespace: str, seed: int) -> int:
    raw = hashlib.sha256(f"{namespace}|{int(seed)}".encode()).digest()
    return int.from_bytes(raw[:8], "big", signed=False)


def gene_signs(groups: Sequence[str], *, seed: int, namespace: str) -> np.ndarray:
    """Return one deterministic +/-1 sign per row, constant within gene."""
    labels, index = _group_index(groups, len(groups))
    rng = np.random.default_rng(_seed(namespace, seed))
    by_group = rng.choice(np.asarray([-1.0, 1.0]), size=len(labels), replace=True)
    return by_group[index]


def gene_sign_flip(
    values: np.ndarray | Sequence[float],
    groups: Sequence[str],
    *,
    seed: int,
    namespace: str,
) -> np.ndarray:
    """Flip row-level paired effects with one exchangeable sign per gene."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0 or arr.shape[-1] != len(groups):
        raise ValueError("the last values axis must match groups")
    if not np.isfinite(arr).all():
        raise ValueError("values contain non-finite entries")
    return arr * gene_signs(groups, seed=seed, namespace=namespace)


def _cluster_bootstrap_from_summary(
    sizes: np.ndarray,
    sums: np.ndarray,
    *,
    n_draws: int,
    seed: int,
    namespace: str,
) -> np.ndarray:
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    n_genes = len(sizes)
    rng = np.random.default_rng(_seed(namespace, seed))
    counts = rng.multinomial(
        n_genes,
        np.full(n_genes, 1.0 / n_genes),
        size=n_draws,
    ).astype(float)
    denominators = counts @ sizes
    return (counts @ sums.T) / denominators[:, None]


def cluster_bootstrap_estimates(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> np.ndarray:
    """Resample whole genes and return row-weighted family means, ``[draws, components]``."""
    family = _as_family(values)
    _, _, sizes, sums = _cluster_summary(family, groups, min_genes=min_genes)
    return _cluster_bootstrap_from_summary(
        sizes,
        sums,
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
    )


def _sign_flip_from_sums(
    sums: np.ndarray,
    n_examples: int,
    *,
    n_draws: int,
    seed: int,
    namespace: str,
) -> np.ndarray:
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    signs = _rademacher_sign_matrix(
        n_draws=n_draws,
        n_genes=sums.shape[1],
        seed=seed,
        namespace=namespace,
    )
    return (signs @ sums.T) / float(n_examples)


def _rademacher_sign_matrix(
    *,
    n_draws: int,
    n_genes: int,
    seed: int,
    namespace: str,
) -> np.ndarray:
    """Return the exact deterministic whole-gene sign matrix used by a test."""
    if n_draws < 1 or n_genes < 1:
        raise ValueError("n_draws and n_genes must be positive")
    rng = np.random.default_rng(_seed(namespace, seed))
    return rng.choice(np.asarray([-1.0, 1.0]), size=(n_draws, n_genes))


def gene_sign_flip_null_draws(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> np.ndarray:
    """Gene-level sign-flip null means from raw cluster sums.

    No centering or influence approximation is used: each draw applies one sign to every raw
    row in a gene and then recomputes the row-weighted family mean.
    """
    family = _as_family(values)
    _, _, _, sums = _cluster_summary(family, groups, min_genes=min_genes)
    return _sign_flip_from_sums(
        sums,
        family.shape[1],
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
    )


def add_one_max_abs_pvalues(observed: Sequence[float], null_draws: np.ndarray) -> np.ndarray:
    """Exact add-one Monte-Carlo p-values using the full-family max absolute statistic."""
    obs = np.asarray(observed, dtype=float)
    null = np.asarray(null_draws, dtype=float)
    if obs.ndim != 1 or not np.isfinite(obs).all():
        raise ValueError("observed must be a finite one-dimensional sequence")
    if (
        null.ndim != 2
        or null.shape[1] != obs.size
        or null.shape[0] == 0
        or not np.isfinite(null).all()
    ):
        raise ValueError("null_draws must have finite shape [draws, len(observed)]")
    family_max = np.max(np.abs(null), axis=1)
    return np.asarray(
        [
            (1 + np.count_nonzero(family_max >= abs(value))) / (len(family_max) + 1)
            for value in obs
        ]
    )


Alternative = Literal["greater", "less"]
BoundKind = Literal["lower", "upper"]
BOUNDARY_PVALUE_RULE = (
    "snap an oriented statistic within comparison_tolerance of zero to zero; "
    "exceedance iff oriented_null >= oriented_statistic - comparison_tolerance; "
    "p=(1+exceedances)/(B+1)"
)
BOUNDARY_COMPARISON_TOLERANCE_RULE = (
    "float64_eps * max(128,n_examples) * "
    "max(1,max_abs_centered_values,max_abs_oriented_null,abs_oriented_statistic)"
)
INVERTED_BOUNDARY_PVALUE_RULE = (
    "one shared whole-gene Rademacher sign matrix; each draw's crossing is the "
    "row-weighted mean over genes assigned -1; an all-+1 draw is a tie at every "
    "boundary; one frozen endpoint tolerance adjusts every crossing toward the closed "
    "null; greater exceedance iff crossing-tolerance <= boundary and less exceedance "
    "iff crossing+tolerance >= boundary; numeric ties are exceedances; "
    "p=(1+exceedances)/(B+1)"
)
INVERTED_BOUNDARY_TOLERANCE_RULE = (
    "float64_eps * max(128,n_examples) * max(1,max_abs_values,max_abs_crossings)"
)
INVERTED_BOUNDARY_TRANSITION_RULE = (
    "greater/lower reports the first float64 boundary not rejected (its predecessor "
    "rejects); less/upper reports the last float64 boundary not rejected (its successor "
    "rejects); therefore lower > b or upper < b iff the corresponding test at finite "
    "float64 boundary b has p < alpha"
)


def _boundary_vector(
    boundaries: float | Sequence[float],
    n_components: int,
) -> np.ndarray:
    values = np.asarray(boundaries, dtype=float)
    if values.ndim == 0:
        values = np.full(n_components, float(values), dtype=float)
    if values.ndim != 1 or values.size != n_components:
        raise ValueError("boundaries must be scalar or match the component axis")
    if not np.isfinite(values).all():
        raise ValueError("boundaries contain non-finite entries")
    return values


def _alternative_vector(
    alternatives: Alternative | Sequence[Alternative],
    n_components: int,
) -> tuple[Alternative, ...]:
    if isinstance(alternatives, str):
        rendered = (alternatives,) * n_components
    else:
        rendered = tuple(alternatives)
    if len(rendered) != n_components or any(
        alternative not in {"greater", "less"} for alternative in rendered
    ):
        raise ValueError(
            "alternatives must be 'greater'/'less' and match the component axis"
        )
    return rendered  # type: ignore[return-value]


def _alpha_vector(alphas: float | Sequence[float], n_components: int) -> np.ndarray:
    values = np.asarray(alphas, dtype=float)
    if values.ndim == 0:
        values = np.full(n_components, float(values), dtype=float)
    if (
        values.ndim != 1
        or values.size != n_components
        or not np.isfinite(values).all()
        or np.any(values <= 0.0)
        or np.any(values >= 1.0)
    ):
        raise ValueError("alphas must be scalar or match the component axis and lie in (0, 1)")
    return values


def _family_names(names: Sequence[str], n_components: int) -> tuple[str, ...]:
    rendered = tuple(str(name) for name in names)
    if (
        len(rendered) != n_components
        or len(set(rendered)) != len(rendered)
        or any(not name for name in rendered)
    ):
        raise ValueError("names must be non-empty, unique, and match the component axis")
    return rendered


def _array_sha256(values: np.ndarray) -> str:
    """Hash a numeric array in a platform-independent, shape-aware representation."""
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    shape = ",".join(str(dimension) for dimension in canonical.shape)
    digest = hashlib.sha256()
    digest.update(f"float64|{shape}|".encode())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _sign_matrix_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype=np.int8))
    shape = ",".join(str(dimension) for dimension in canonical.shape)
    digest = hashlib.sha256()
    digest.update(f"int8|{shape}|".encode())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _string_sequence_sha256(values: Sequence[str]) -> str:
    """Hash an ordered string sequence without delimiter ambiguity."""
    digest = hashlib.sha256()
    digest.update(f"utf8-sequence|{len(values)}|".encode())
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def boundary_sign_flip_null_draws(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    *,
    boundaries: float | Sequence[float],
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> np.ndarray:
    """Return shared-sign gene randomisation draws about arbitrary null boundaries.

    The returned array has shape ``[draws, components]``.  Every draw uses one sign
    per gene and reuses that same sign across all component rows.  Centering each row
    at its own null boundary before the flip makes this suitable for tests such as
    ``H0: theta <= 0.05``; it is intentionally separate from the historical
    zero-boundary raw-sum helper above.
    """
    family = _as_family(values)
    boundary = _boundary_vector(boundaries, family.shape[0])
    centered = family - boundary[:, None]
    _, _, _, sums = _cluster_summary(centered, groups, min_genes=min_genes)
    return _sign_flip_from_sums(
        sums,
        family.shape[1],
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
    )


@dataclass(frozen=True)
class BoundarySignFlipTest:
    """Auditable one-sided randomisation test against a declared boundary."""

    estimate: float
    boundary: float
    alternative: Alternative
    oriented_statistic: float
    comparison_tolerance: float
    p_value: float
    exceedances: int
    B: int
    seed: int
    namespace: str
    null_draws_sha256: str
    shared_null_draws_sha256: str
    sign_matrix_sha256: str
    ordered_gene_labels_sha256: str
    n_examples: int
    n_genes: int
    randomisation_unit: str = "gene"
    pvalue_rule: str = BOUNDARY_PVALUE_RULE

    @property
    def add_one_floor(self) -> float:
        return 1.0 / (self.B + 1)

    def to_dict(self) -> dict:
        return asdict(self) | {"add_one_floor": self.add_one_floor}


def boundary_sign_flip_tests(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    names: Sequence[str],
    *,
    boundaries: float | Sequence[float],
    alternatives: Alternative | Sequence[Alternative],
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> dict[str, BoundarySignFlipTest]:
    """Run endpoint-wise one-sided boundary tests with shared whole-gene signs.

    Multiplicity is deliberately not hidden inside this function.  It returns each
    endpoint's boundary test so callers can form scientifically declared IUT paths and
    apply :func:`holm_adjust_named` across those named paths.  This avoids allowing a
    high-variance endpoint to mask a low-variance endpoint through a raw-scale family
    maximum while retaining one joint randomisation draw matrix for algebraically
    related rows.
    """
    family = _as_family(values)
    rendered_names = tuple(str(name) for name in names)
    if (
        len(rendered_names) != family.shape[0]
        or len(set(rendered_names)) != len(rendered_names)
        or any(not name for name in rendered_names)
    ):
        raise ValueError("names must be non-empty, unique, and match the component axis")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    boundary = _boundary_vector(boundaries, family.shape[0])
    alternative = _alternative_vector(alternatives, family.shape[0])
    labels, _, _, _ = _cluster_summary(family, groups, min_genes=min_genes)
    centered_null = boundary_sign_flip_null_draws(
        family,
        groups,
        boundaries=boundary,
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=min_genes,
    )
    direction = np.asarray(
        [1.0 if item == "greater" else -1.0 for item in alternative],
        dtype=float,
    )
    estimate = family.mean(axis=1)
    # Form the observed boundary statistic from per-example centered values, matching the
    # null construction.  ``mean(values) - boundary`` can turn an exact closed-null tie
    # into a tiny positive number (for example, 128 copies of 0.05 average to
    # 0.05000000000000001 on common NumPy builds).
    centered_observed = (family - boundary[:, None]).mean(axis=1)
    statistic = direction * centered_observed
    oriented_null = centered_null * direction[None, :]
    shared_digest = _array_sha256(centered_null)
    signs = _rademacher_sign_matrix(
        n_draws=n_draws,
        n_genes=len(labels),
        seed=seed,
        namespace=namespace,
    )
    sign_digest = _sign_matrix_sha256(signs)
    gene_digest = _string_sequence_sha256(labels)

    results: dict[str, BoundarySignFlipTest] = {}
    for index, name in enumerate(rendered_names):
        numeric_scale = max(
            1.0,
            float(np.max(np.abs(family[index] - boundary[index]))),
            float(np.max(np.abs(oriented_null[:, index]))),
            abs(float(statistic[index])),
        )
        comparison_tolerance = float(
            np.finfo(float).eps
            * max(128, family.shape[1])
            * numeric_scale
        )
        oriented_statistic = float(statistic[index])
        if abs(oriented_statistic) <= comparison_tolerance:
            oriented_statistic = 0.0
        exceedances = int(
            np.count_nonzero(
                oriented_null[:, index]
                >= oriented_statistic - comparison_tolerance
            )
        )
        results[name] = BoundarySignFlipTest(
            estimate=float(estimate[index]),
            boundary=float(boundary[index]),
            alternative=alternative[index],
            oriented_statistic=oriented_statistic,
            comparison_tolerance=comparison_tolerance,
            p_value=float((1 + exceedances) / (n_draws + 1)),
            exceedances=exceedances,
            B=int(n_draws),
            seed=int(seed),
            namespace=str(namespace),
            null_draws_sha256=_array_sha256(oriented_null[:, index]),
            shared_null_draws_sha256=shared_digest,
            sign_matrix_sha256=sign_digest,
            ordered_gene_labels_sha256=gene_digest,
            n_examples=family.shape[1],
            n_genes=len(labels),
        )
    return results


def one_sided_boundary_sign_flip(
    values: np.ndarray | Sequence[float],
    groups: Sequence[str],
    *,
    boundary: float,
    alternative: Alternative,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> BoundarySignFlipTest:
    """Convenience wrapper for one arbitrary-boundary one-sided gene test."""
    family = np.asarray(values, dtype=float)
    if family.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return boundary_sign_flip_tests(
        family,
        groups,
        ["endpoint"],
        boundaries=boundary,
        alternatives=alternative,
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=min_genes,
    )["endpoint"]


@dataclass(frozen=True)
class StudentizedBootstrapOneSided:
    """One-sided studentized whole-gene bootstrap test and inverted bound."""

    estimate: float
    standard_error: float
    boundary: float
    alternative: Alternative
    alpha: float
    bound: float
    bound_kind: BoundKind
    critical_value: float
    observed_studentized_statistic: float
    p_value: float
    exceedances: int
    B: int
    seed: int
    namespace: str
    bootstrap_studentized_sha256: str
    shared_resample_indices_sha256: str
    ordered_gene_labels_sha256: str
    estimator: str
    randomisation_unit: str = "gene"
    inference_kind: str = "studentized_gene_bootstrap_asymptotic"
    pvalue_rule: str = (
        "one-sided add-one tail probability of centered studentized whole-gene "
        "bootstrap draws; confidence bound uses the conservative upper empirical "
        "quantile of the oriented studentized draws"
    )
    degenerate_empirical_point_mass: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RobustGeneBootstrapTest:
    """Conjunction of raw-mean and trimmed-mean boundary evidence."""

    raw: StudentizedBootstrapOneSided
    trimmed: StudentizedBootstrapOneSided
    trim_fraction: float
    joint_p_value: float
    joint_boundary_pass: bool
    median: float
    quantiles: tuple[float, float, float, float, float]
    boundary_exceedance_prevalence: float
    top_one_percent_deletion_estimate: float
    n_genes: int
    inference_kind: str = "studentized_gene_bootstrap_asymptotic"

    def to_dict(self) -> dict:
        return asdict(self)


def _gene_mean_matrix(
    family: np.ndarray,
    groups: Sequence[str],
    *,
    min_genes: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    labels, index, sizes, sums = _cluster_summary(
        family, groups, min_genes=min_genes
    )
    return tuple(labels), sums / sizes[None, :]


def _trimmed_location_and_se(
    samples: np.ndarray,
    *,
    trim_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row-wise trimmed means and winsorized asymptotic standard errors."""

    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    n = values.shape[1]
    k = int(np.floor(float(trim_fraction) * n))
    if k < 1 or 2 * k >= n - 1:
        raise ValueError(
            "trim_fraction must trim at least one observation while retaining two"
        )
    ordered = np.sort(values, axis=1)
    retained = ordered[:, k : n - k]
    estimate = retained.mean(axis=1)
    winsorized = ordered.copy()
    winsorized[:, :k] = ordered[:, [k]]
    winsorized[:, n - k :] = ordered[:, [n - k - 1]]
    standard_error = winsorized.std(axis=1, ddof=1) / (
        (1.0 - 2.0 * float(trim_fraction)) * np.sqrt(float(n))
    )
    return estimate, standard_error


def _raw_location_and_se(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    return values.mean(axis=1), values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])


def studentized_gene_bootstrap_tests(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    names: Sequence[str],
    *,
    boundaries: float | Sequence[float],
    alternatives: Alternative | Sequence[Alternative],
    alpha: float,
    trim_fraction: float,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> dict[str, RobustGeneBootstrapTest]:
    """Require raw and trimmed gene effects to clear the same one-sided boundary.

    Genes, rather than rows, are the estimand and resampling unit.  The function uses one
    shared deterministic gene-resample matrix across every endpoint and both estimators.
    It is an asymptotic studentized bootstrap procedure, not an exact randomisation test.
    """

    family = _as_family(values)
    endpoint_names = _family_names(names, family.shape[0])
    boundary = _boundary_vector(boundaries, family.shape[0])
    alternative = _alternative_vector(alternatives, family.shape[0])
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not np.isfinite(trim_fraction) or not 0.0 < float(trim_fraction) < 0.5:
        raise ValueError("trim_fraction must lie in (0, 0.5)")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    labels, gene_values = _gene_mean_matrix(
        family, groups, min_genes=min_genes
    )
    n_genes = gene_values.shape[1]
    # Validate the requested trim at the observed sample size before allocating draws.
    _trimmed_location_and_se(gene_values[:1], trim_fraction=trim_fraction)
    rng = np.random.default_rng(_seed(namespace, seed))
    resample_indices = rng.integers(
        0,
        n_genes,
        size=(int(n_draws), n_genes),
        dtype=np.int64,
    )
    resample_digest = hashlib.sha256(
        np.ascontiguousarray(resample_indices, dtype="<i8").tobytes(order="C")
    ).hexdigest()
    gene_digest = _string_sequence_sha256(labels)
    results: dict[str, RobustGeneBootstrapTest] = {}

    for endpoint_index, name in enumerate(endpoint_names):
        endpoint = gene_values[endpoint_index]
        raw_estimate, raw_se = _raw_location_and_se(endpoint)
        trimmed_estimate, trimmed_se = _trimmed_location_and_se(
            endpoint, trim_fraction=trim_fraction
        )
        raw_est = float(raw_estimate[0])
        raw_standard_error = float(raw_se[0])
        trim_est = float(trimmed_estimate[0])
        trim_standard_error = float(trimmed_se[0])
        direction = 1.0 if alternative[endpoint_index] == "greater" else -1.0
        if np.all(endpoint == endpoint[0]):
            # An exact empirical point mass has no studentization scale.  Do not
            # crash (or add an arbitrary epsilon): its empirical bootstrap law is
            # also a point mass, so the conservative one-sided bound is the point
            # estimate itself and a strict boundary on the alternative side gets
            # the add-one floor.  The record flags this branch explicitly.
            estimate = float(endpoint[0])
            oriented_distance = direction * (
                estimate - float(boundary[endpoint_index])
            )
            strict_pass = oriented_distance > 0.0
            at_boundary = oriented_distance == 0.0
            exceedances = 0 if strict_pass else int(n_draws)
            observed = (
                0.0
                if at_boundary
                else (np.finfo(float).max if strict_pass else -np.finfo(float).max)
            )

            def point_mass(estimator: str) -> StudentizedBootstrapOneSided:
                return StudentizedBootstrapOneSided(
                    estimate=estimate,
                    standard_error=0.0,
                    boundary=float(boundary[endpoint_index]),
                    alternative=alternative[endpoint_index],
                    alpha=float(alpha),
                    bound=estimate,
                    bound_kind=("lower" if direction > 0 else "upper"),
                    critical_value=0.0,
                    observed_studentized_statistic=float(observed),
                    p_value=float((1 + exceedances) / (n_draws + 1)),
                    exceedances=exceedances,
                    B=int(n_draws),
                    seed=int(seed),
                    namespace=str(namespace),
                    bootstrap_studentized_sha256=_array_sha256(
                        np.zeros(int(n_draws), dtype=float)
                    ),
                    shared_resample_indices_sha256=resample_digest,
                    ordered_gene_labels_sha256=gene_digest,
                    estimator=estimator,
                    pvalue_rule=(
                        "exact empirical point-mass branch; conservative bound equals "
                        "the common gene value; strict alternative receives add-one floor"
                    ),
                    degenerate_empirical_point_mass=True,
                )

            raw = point_mass("unweighted_gene_mean")
            trimmed = point_mass(
                f"{100 * float(trim_fraction):g}%_trimmed_unweighted_gene_mean"
            )
            prevalence = (
                np.mean(endpoint > raw.boundary)
                if direction > 0
                else np.mean(endpoint < raw.boundary)
            )
            results[name] = RobustGeneBootstrapTest(
                raw=raw,
                trimmed=trimmed,
                trim_fraction=float(trim_fraction),
                joint_p_value=float(max(raw.p_value, trimmed.p_value)),
                joint_boundary_pass=bool(
                    strict_pass and raw.p_value < float(alpha)
                ),
                median=estimate,
                quantiles=(estimate, estimate, estimate, estimate, estimate),
                boundary_exceedance_prevalence=float(prevalence),
                top_one_percent_deletion_estimate=estimate,
                n_genes=n_genes,
            )
            continue
        if (
            not np.isfinite(raw_standard_error)
            or not np.isfinite(trim_standard_error)
            or raw_standard_error <= 0.0
            or trim_standard_error <= 0.0
        ):
            raise ValueError("studentized bootstrap requires positive finite standard errors")

        sampled = endpoint[resample_indices]
        raw_boot_estimate, raw_boot_se = _raw_location_and_se(sampled)
        trim_boot_estimate, trim_boot_se = _trimmed_location_and_se(
            sampled, trim_fraction=trim_fraction
        )
        if (
            np.any(~np.isfinite(raw_boot_se))
            or np.any(~np.isfinite(trim_boot_se))
            or np.any(raw_boot_se <= 0.0)
            or np.any(trim_boot_se <= 0.0)
        ):
            raise ValueError(
                "a studentized bootstrap resample has zero or non-finite standard error"
            )
        def one(
            estimator: str,
            estimate: float,
            standard_error: float,
            boot_estimate: np.ndarray,
            boot_se: np.ndarray,
        ) -> StudentizedBootstrapOneSided:
            oriented_bootstrap = direction * (boot_estimate - estimate) / boot_se
            observed = direction * (
                estimate - float(boundary[endpoint_index])
            ) / standard_error
            exceedances = int(np.count_nonzero(oriented_bootstrap >= observed))
            critical = float(
                np.quantile(
                    oriented_bootstrap,
                    1.0 - float(alpha),
                    method="higher",
                )
            )
            bound = float(estimate - direction * critical * standard_error)
            return StudentizedBootstrapOneSided(
                estimate=estimate,
                standard_error=standard_error,
                boundary=float(boundary[endpoint_index]),
                alternative=alternative[endpoint_index],
                alpha=float(alpha),
                bound=bound,
                bound_kind=("lower" if direction > 0 else "upper"),
                critical_value=critical,
                observed_studentized_statistic=float(observed),
                p_value=float((1 + exceedances) / (n_draws + 1)),
                exceedances=exceedances,
                B=int(n_draws),
                seed=int(seed),
                namespace=str(namespace),
                bootstrap_studentized_sha256=_array_sha256(oriented_bootstrap),
                shared_resample_indices_sha256=resample_digest,
                ordered_gene_labels_sha256=gene_digest,
                estimator=estimator,
            )

        raw = one(
            "unweighted_gene_mean",
            raw_est,
            raw_standard_error,
            raw_boot_estimate,
            raw_boot_se,
        )
        trimmed = one(
            f"{100 * float(trim_fraction):g}%_trimmed_unweighted_gene_mean",
            trim_est,
            trim_standard_error,
            trim_boot_estimate,
            trim_boot_se,
        )
        if direction > 0:
            boundary_pass = (
                raw.bound > raw.boundary and trimmed.bound > trimmed.boundary
            )
            ordered_tail = np.sort(endpoint)[: max(1, n_genes - int(np.ceil(0.01 * n_genes)))]
            prevalence = np.mean(endpoint > raw.boundary)
        else:
            boundary_pass = (
                raw.bound < raw.boundary and trimmed.bound < trimmed.boundary
            )
            ordered_tail = np.sort(endpoint)[int(np.ceil(0.01 * n_genes)) :]
            prevalence = np.mean(endpoint < raw.boundary)
        joint_p = max(raw.p_value, trimmed.p_value)
        results[name] = RobustGeneBootstrapTest(
            raw=raw,
            trimmed=trimmed,
            trim_fraction=float(trim_fraction),
            joint_p_value=float(joint_p),
            joint_boundary_pass=bool(boundary_pass and joint_p < float(alpha)),
            median=float(np.median(endpoint)),
            quantiles=tuple(
                float(item)
                for item in np.quantile(endpoint, [0.01, 0.10, 0.50, 0.90, 0.99])
            ),
            boundary_exceedance_prevalence=float(prevalence),
            top_one_percent_deletion_estimate=float(np.mean(ordered_tail)),
            n_genes=n_genes,
        )
    return results


@dataclass(frozen=True)
class RobustGeneBootstrapTOST:
    """Raw+trimmed two-one-sided equivalence test for one endpoint."""

    estimate: float
    sesoi: float
    lower_edge: RobustGeneBootstrapTest
    upper_edge: RobustGeneBootstrapTest
    p_value: float
    equivalence_pass: bool
    inference_kind: str = "studentized_gene_bootstrap_asymptotic"

    def to_dict(self) -> dict:
        return asdict(self)


def studentized_gene_bootstrap_tost(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    names: Sequence[str],
    *,
    sesoi: float,
    alpha: float,
    trim_fraction: float,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> dict[str, RobustGeneBootstrapTOST]:
    """Test strict equivalence for raw and trimmed gene estimands simultaneously."""

    family = _as_family(values)
    endpoint_names = _family_names(names, family.shape[0])
    if not np.isfinite(sesoi) or float(sesoi) <= 0.0:
        raise ValueError("sesoi must be positive and finite")
    expanded_values = np.repeat(family, 2, axis=0)
    expanded_names = tuple(
        edge
        for name in endpoint_names
        for edge in (f"{name}|lower", f"{name}|upper")
    )
    expanded = studentized_gene_bootstrap_tests(
        expanded_values,
        groups,
        expanded_names,
        boundaries=tuple(
            boundary
            for _name in endpoint_names
            for boundary in (-float(sesoi), float(sesoi))
        ),
        alternatives=tuple(
            alternative
            for _name in endpoint_names
            for alternative in ("greater", "less")
        ),
        alpha=alpha,
        trim_fraction=trim_fraction,
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=min_genes,
    )
    results: dict[str, RobustGeneBootstrapTOST] = {}
    for index, name in enumerate(endpoint_names):
        lower = expanded[f"{name}|lower"]
        upper = expanded[f"{name}|upper"]
        p_value = max(lower.joint_p_value, upper.joint_p_value)
        if lower.raw.estimate != upper.raw.estimate:
            raise RuntimeError("TOST edges disagree on the tested gene-mean estimand")
        results[name] = RobustGeneBootstrapTOST(
            estimate=float(lower.raw.estimate),
            sesoi=float(sesoi),
            lower_edge=lower,
            upper_edge=upper,
            p_value=float(p_value),
            equivalence_pass=bool(
                lower.joint_boundary_pass
                and upper.joint_boundary_pass
                and p_value < float(alpha)
            ),
        )
    return results


@dataclass(frozen=True)
class CrossingBoundarySignFlipTest:
    """One-sided boundary test under the exactly invertible crossing convention.

    For one Rademacher draw, moving the arbitrary null boundary changes whether the
    draw exceeds the observed centered mean at exactly one crossing value: the
    row-weighted mean among genes assigned ``-1``.  Representing the test by those
    crossings gives a monotone finite-Monte-Carlo test whose numeric ties can be made
    explicitly conservative and whose confidence bound can be inverted exactly.
    """

    estimate: float
    boundary: float
    alternative: Alternative
    comparison_tolerance: float
    p_value: float
    exceedances: int
    B: int
    seed: int
    namespace: str
    crossing_draws_sha256: str
    adjusted_crossing_draws_sha256: str
    shared_crossing_draws_sha256: str
    exceedance_mask_sha256: str
    shared_negative_example_counts_sha256: str
    always_tie_mask_sha256: str
    sign_matrix_sha256: str
    ordered_gene_labels_sha256: str
    ordered_endpoint_names_sha256: str
    n_examples: int
    n_genes: int
    always_tie_draws: int
    randomisation_unit: str = "gene"
    pvalue_rule: str = INVERTED_BOUNDARY_PVALUE_RULE
    comparison_tolerance_rule: str = INVERTED_BOUNDARY_TOLERANCE_RULE

    @property
    def add_one_floor(self) -> float:
        return 1.0 / (self.B + 1)

    def to_dict(self) -> dict:
        return asdict(self) | {"add_one_floor": self.add_one_floor}


@dataclass(frozen=True)
class BoundarySignFlipInversion:
    """Auditable finite-Monte-Carlo inversion of a one-sided boundary test.

    ``greater`` produces a lower confidence bound and ``less`` produces an upper
    confidence bound.  A finite bound is the *non-rejecting* float64 transition.  This
    convention is load-bearing: strict interval classification is exactly dual to the
    strict test rule ``p < alpha``, including when a reference boundary equals the
    reported bound.

    ``bound`` and the three transition tests are ``None`` only when structural all-+1
    sign draws make rejection impossible at the requested Monte-Carlo resolution.  In
    that case the one-sided bound is vacuous and :meth:`rejects` always returns false.
    """

    estimate: float
    alpha: float
    alternative: Alternative
    bound_kind: BoundKind
    bound: float | None
    comparison_tolerance: float
    transition_status: str
    max_rejecting_exceedances: int
    transition_threshold_index: int | None
    transition_threshold_multiplicity: int
    transition_test: CrossingBoundarySignFlipTest | None
    rejecting_adjacent_test: CrossingBoundarySignFlipTest | None
    nonrejecting_adjacent_test: CrossingBoundarySignFlipTest | None
    B: int
    seed: int
    namespace: str
    sign_matrix_sha256: str
    ordered_gene_labels_sha256: str
    ordered_endpoint_names_sha256: str
    n_examples: int
    n_genes: int
    always_tie_draws: int
    transition_rule: str = INVERTED_BOUNDARY_TRANSITION_RULE

    @property
    def finite_transition(self) -> bool:
        return self.bound is not None

    def rejects(self, boundary: float) -> bool:
        """Return the decision dual to comparing this bound strictly with ``boundary``."""
        candidate = float(boundary)
        if not np.isfinite(candidate):
            raise ValueError("boundary must be finite")
        if self.bound is None:
            return False
        if self.alternative == "greater":
            return bool(self.bound > candidate)
        return bool(self.bound < candidate)

    def to_dict(self) -> dict:
        return asdict(self) | {"finite_transition": self.finite_transition}


@dataclass(frozen=True)
class _CrossingFamily:
    family: np.ndarray
    labels: tuple[str, ...]
    endpoint_names: tuple[str, ...]
    signs: np.ndarray
    crossings: np.ndarray
    comparison_tolerances: np.ndarray
    negative_example_counts: np.ndarray
    always_tie: np.ndarray
    sign_matrix_sha256: str
    ordered_gene_labels_sha256: str
    ordered_endpoint_names_sha256: str
    crossing_draws_sha256: str
    negative_example_counts_sha256: str
    always_tie_mask_sha256: str


def _crossing_family(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    names: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int,
) -> _CrossingFamily:
    """Build the shared-sign crossing representation used by tests and inversions."""
    family = _as_family(values)
    endpoint_names = _family_names(names, family.shape[0])
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    labels, index, sizes, _ = _cluster_summary(family, groups, min_genes=min_genes)
    # Anchor before summing so an exact closed-null constant endpoint remains an exact
    # crossing tie (rather than becoming 0.05000000000000001 after a long raw sum).
    anchors = family[:, 0]
    centered = family - anchors[:, None]
    centered_sums = np.vstack(
        [
            np.bincount(index, weights=row, minlength=len(labels))
            for row in centered
        ]
    )
    signs = _rademacher_sign_matrix(
        n_draws=n_draws,
        n_genes=len(labels),
        seed=seed,
        namespace=namespace,
    )
    negative = signs < 0.0
    # Each row is a draw; a boolean @ float product sums whole-gene row counts and
    # whole-gene endpoint sums without materialising a second float64 sign matrix.
    negative_example_counts = negative @ sizes
    negative_centered_sums = negative @ centered_sums.T
    always_tie = ~negative.any(axis=1)
    crossings = np.zeros_like(negative_centered_sums, dtype=float)
    np.divide(
        negative_centered_sums,
        negative_example_counts[:, None],
        out=crossings,
        where=~always_tie[:, None],
    )
    crossings += anchors[None, :]
    # Structural all-+1 draws do not have a crossing; their numeric placeholder is
    # reset to zero and disambiguated by ``always_tie_mask_sha256``.
    crossings[always_tie, :] = 0.0
    if not np.isfinite(crossings).all():
        raise ValueError("crossing calculation produced non-finite values")
    ordinary_crossings = crossings[~always_tie]
    if ordinary_crossings.shape[0] == 0:
        crossing_scale = np.zeros(family.shape[0], dtype=float)
    else:
        crossing_scale = np.max(np.abs(ordinary_crossings), axis=0)
    comparison_tolerances = (
        np.finfo(float).eps
        * max(128, family.shape[1])
        * np.maximum.reduce(
            (
                np.ones(family.shape[0], dtype=float),
                np.max(np.abs(family), axis=1),
                crossing_scale,
            )
        )
    )
    if (
        not np.isfinite(comparison_tolerances).all()
        or np.any(comparison_tolerances <= 0.0)
    ):
        raise ValueError("crossing comparison tolerance is not finite and positive")

    label_tuple = tuple(labels)
    return _CrossingFamily(
        family=family,
        labels=label_tuple,
        endpoint_names=endpoint_names,
        signs=signs,
        crossings=crossings,
        comparison_tolerances=comparison_tolerances,
        negative_example_counts=negative_example_counts,
        always_tie=always_tie,
        sign_matrix_sha256=_sign_matrix_sha256(signs),
        ordered_gene_labels_sha256=_string_sequence_sha256(label_tuple),
        ordered_endpoint_names_sha256=_string_sequence_sha256(endpoint_names),
        crossing_draws_sha256=_array_sha256(crossings),
        negative_example_counts_sha256=_array_sha256(negative_example_counts),
        always_tie_mask_sha256=_sign_matrix_sha256(always_tie),
    )


def _crossing_test(
    state: _CrossingFamily,
    component_index: int,
    *,
    boundary: float,
    alternative: Alternative,
    seed: int,
    namespace: str,
) -> CrossingBoundarySignFlipTest:
    candidate = float(boundary)
    if not np.isfinite(candidate):
        raise ValueError("boundaries contain non-finite entries")
    crossing = state.crossings[:, component_index]
    comparison_tolerance = float(state.comparison_tolerances[component_index])
    exceedance = state.always_tie.copy()
    ordinary = ~state.always_tie
    if alternative == "greater":
        adjusted_crossing = crossing - comparison_tolerance
        exceedance[ordinary] = adjusted_crossing[ordinary] <= candidate
    else:
        adjusted_crossing = crossing + comparison_tolerance
        exceedance[ordinary] = adjusted_crossing[ordinary] >= candidate
    exceedances = int(np.count_nonzero(exceedance))
    B = state.signs.shape[0]
    return CrossingBoundarySignFlipTest(
        estimate=float(state.family[component_index].mean()),
        boundary=candidate,
        alternative=alternative,
        comparison_tolerance=comparison_tolerance,
        p_value=float((1 + exceedances) / (B + 1)),
        exceedances=exceedances,
        B=B,
        seed=int(seed),
        namespace=str(namespace),
        crossing_draws_sha256=_array_sha256(crossing),
        adjusted_crossing_draws_sha256=_array_sha256(adjusted_crossing),
        shared_crossing_draws_sha256=state.crossing_draws_sha256,
        exceedance_mask_sha256=_sign_matrix_sha256(exceedance),
        shared_negative_example_counts_sha256=(
            state.negative_example_counts_sha256
        ),
        always_tie_mask_sha256=state.always_tie_mask_sha256,
        sign_matrix_sha256=state.sign_matrix_sha256,
        ordered_gene_labels_sha256=state.ordered_gene_labels_sha256,
        ordered_endpoint_names_sha256=state.ordered_endpoint_names_sha256,
        n_examples=state.family.shape[1],
        n_genes=len(state.labels),
        always_tie_draws=int(np.count_nonzero(state.always_tie)),
    )


def crossing_boundary_sign_flip_tests(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    names: Sequence[str],
    *,
    boundaries: float | Sequence[float],
    alternatives: Alternative | Sequence[Alternative],
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> dict[str, CrossingBoundarySignFlipTest]:
    """Test arbitrary boundaries using the convention inverted by R3d-style bounds.

    This is intentionally additive to :func:`boundary_sign_flip_tests`: existing frozen
    studies retain their tolerance-based replay convention, while new studies can use a
    crossing representation whose p-values and confidence bounds are exactly dual.
    """
    state = _crossing_family(
        values,
        groups,
        names,
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=min_genes,
    )
    boundary = _boundary_vector(boundaries, state.family.shape[0])
    alternative = _alternative_vector(alternatives, state.family.shape[0])
    return {
        name: _crossing_test(
            state,
            index,
            boundary=float(boundary[index]),
            alternative=alternative[index],
            seed=seed,
            namespace=namespace,
        )
        for index, name in enumerate(state.endpoint_names)
    }


def _max_exceedances_for_strict_alpha(alpha: float, B: int) -> int:
    """Largest integer ``e`` satisfying ``(1 + e) / (B + 1) < alpha``."""
    candidate = min(B, max(-1, int(np.ceil(alpha * (B + 1))) - 2))
    while candidate >= 0 and (1 + candidate) / (B + 1) >= alpha:
        candidate -= 1
    while candidate < B and (2 + candidate) / (B + 1) < alpha:
        candidate += 1
    return candidate


def invert_boundary_sign_flip_tests(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    names: Sequence[str],
    *,
    alternatives: Alternative | Sequence[Alternative],
    alphas: float | Sequence[float],
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> dict[str, BoundarySignFlipInversion]:
    """Invert shared-sign one-sided tests into exact float64 confidence bounds.

    For ``greater`` the returned lower bound is the first float64 boundary whose test
    does not reject.  For ``less`` the upper bound is the last non-rejecting boundary.
    Thus a strict comparison against any finite float64 reference boundary is exactly
    equivalent to testing that boundary with :func:`crossing_boundary_sign_flip_tests`.
    The transition itself and one representable step on each side are evaluated and
    hashed in every result.
    """
    state = _crossing_family(
        values,
        groups,
        names,
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=min_genes,
    )
    alternative = _alternative_vector(alternatives, state.family.shape[0])
    alpha = _alpha_vector(alphas, state.family.shape[0])
    B = state.signs.shape[0]
    always_tie_draws = int(np.count_nonzero(state.always_tie))
    results: dict[str, BoundarySignFlipInversion] = {}

    for index, name in enumerate(state.endpoint_names):
        endpoint_alpha = float(alpha[index])
        endpoint_alternative = alternative[index]
        comparison_tolerance = float(state.comparison_tolerances[index])
        bound_kind: BoundKind = (
            "lower" if endpoint_alternative == "greater" else "upper"
        )
        max_exceedances = _max_exceedances_for_strict_alpha(endpoint_alpha, B)
        available = max_exceedances - always_tie_draws
        ordinary_crossings = state.crossings[~state.always_tie, index]
        if endpoint_alternative == "greater":
            ordinary_thresholds = ordinary_crossings - comparison_tolerance
        else:
            ordinary_thresholds = ordinary_crossings + comparison_tolerance
        ordered = np.sort(ordinary_thresholds)

        if available < 0:
            results[name] = BoundarySignFlipInversion(
                estimate=float(state.family[index].mean()),
                alpha=endpoint_alpha,
                alternative=endpoint_alternative,
                bound_kind=bound_kind,
                bound=None,
                comparison_tolerance=comparison_tolerance,
                transition_status="VACUOUS_NO_REJECTING_BOUNDARY",
                max_rejecting_exceedances=max_exceedances,
                transition_threshold_index=None,
                transition_threshold_multiplicity=0,
                transition_test=None,
                rejecting_adjacent_test=None,
                nonrejecting_adjacent_test=None,
                B=B,
                seed=int(seed),
                namespace=str(namespace),
                sign_matrix_sha256=state.sign_matrix_sha256,
                ordered_gene_labels_sha256=state.ordered_gene_labels_sha256,
                ordered_endpoint_names_sha256=(
                    state.ordered_endpoint_names_sha256
                ),
                n_examples=state.family.shape[1],
                n_genes=len(state.labels),
                always_tie_draws=always_tie_draws,
            )
            continue

        if endpoint_alternative == "greater":
            transition_index = available
            bound = float(ordered[transition_index])
            rejecting_boundary = float(np.nextafter(bound, -np.inf))
            nonrejecting_boundary = float(np.nextafter(bound, np.inf))
        else:
            transition_index = len(ordered) - available - 1
            bound = float(ordered[transition_index])
            rejecting_boundary = float(np.nextafter(bound, np.inf))
            nonrejecting_boundary = float(np.nextafter(bound, -np.inf))

        if not all(
            np.isfinite(item)
            for item in (bound, rejecting_boundary, nonrejecting_boundary)
        ):
            raise ValueError("a finite adjacent-float transition could not be represented")
        transition_test = _crossing_test(
            state,
            index,
            boundary=bound,
            alternative=endpoint_alternative,
            seed=seed,
            namespace=namespace,
        )
        rejecting_test = _crossing_test(
            state,
            index,
            boundary=rejecting_boundary,
            alternative=endpoint_alternative,
            seed=seed,
            namespace=namespace,
        )
        nonrejecting_test = _crossing_test(
            state,
            index,
            boundary=nonrejecting_boundary,
            alternative=endpoint_alternative,
            seed=seed,
            namespace=namespace,
        )
        if not (
            transition_test.p_value >= endpoint_alpha
            and rejecting_test.p_value < endpoint_alpha
            and nonrejecting_test.p_value >= endpoint_alpha
        ):
            raise RuntimeError("boundary inversion failed to bracket its float64 transition")

        results[name] = BoundarySignFlipInversion(
            estimate=float(state.family[index].mean()),
            alpha=endpoint_alpha,
            alternative=endpoint_alternative,
            bound_kind=bound_kind,
            bound=bound,
            comparison_tolerance=comparison_tolerance,
            transition_status="FINITE_FLOAT64_TRANSITION",
            max_rejecting_exceedances=max_exceedances,
            transition_threshold_index=int(transition_index),
            transition_threshold_multiplicity=int(
                np.count_nonzero(ordinary_thresholds == bound)
            ),
            transition_test=transition_test,
            rejecting_adjacent_test=rejecting_test,
            nonrejecting_adjacent_test=nonrejecting_test,
            B=B,
            seed=int(seed),
            namespace=str(namespace),
            sign_matrix_sha256=state.sign_matrix_sha256,
            ordered_gene_labels_sha256=state.ordered_gene_labels_sha256,
            ordered_endpoint_names_sha256=state.ordered_endpoint_names_sha256,
            n_examples=state.family.shape[1],
            n_genes=len(state.labels),
            always_tie_draws=always_tie_draws,
        )
    return results


def invert_one_sided_boundary_sign_flip(
    values: np.ndarray | Sequence[float],
    groups: Sequence[str],
    *,
    alternative: Alternative,
    alpha: float,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> BoundarySignFlipInversion:
    """Convenience wrapper for one exactly inverted one-sided gene test."""
    endpoint = np.asarray(values, dtype=float)
    if endpoint.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return invert_boundary_sign_flip_tests(
        endpoint,
        groups,
        ["endpoint"],
        alternatives=alternative,
        alphas=alpha,
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=min_genes,
    )["endpoint"]


def intersection_union_pvalue(pvalues: Sequence[float]) -> float:
    """Return the IUT p-value: the maximum constituent one-sided p-value."""
    values = np.asarray(pvalues, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values > 1)
    ):
        raise ValueError("pvalues must be a non-empty finite sequence in [0, 1]")
    return float(np.max(values))


@dataclass(frozen=True)
class TOSTEquivalenceTest:
    """Two one-sided tests for equivalence inside ``(-sesoi, +sesoi)``."""

    estimate: float
    sesoi: float
    lower_edge: BoundarySignFlipTest
    upper_edge: BoundarySignFlipTest
    p_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def tost_equivalence_test(
    values: np.ndarray | Sequence[float],
    groups: Sequence[str],
    *,
    sesoi: float,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int = DEFAULT_MIN_GENES,
) -> TOSTEquivalenceTest:
    """Test ``-sesoi < theta < sesoi`` using a shared-sign TOST/IUT."""
    endpoint = np.asarray(values, dtype=float)
    if endpoint.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if not np.isfinite(sesoi) or sesoi <= 0:
        raise ValueError("sesoi must be positive and finite")
    edges = boundary_sign_flip_tests(
        np.vstack((endpoint, endpoint)),
        groups,
        ["lower_edge", "upper_edge"],
        boundaries=[-float(sesoi), float(sesoi)],
        alternatives=["greater", "less"],
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=min_genes,
    )
    lower = edges["lower_edge"]
    upper = edges["upper_edge"]
    return TOSTEquivalenceTest(
        estimate=float(endpoint.mean()),
        sesoi=float(sesoi),
        lower_edge=lower,
        upper_edge=upper,
        p_value=intersection_union_pvalue([lower.p_value, upper.p_value]),
    )


def strict_interval_decision(lo: float, hi: float, sesoi: float) -> str:
    """Classify an interval using strict SESOI boundaries.

    An interval that merely touches ``-sesoi`` or ``+sesoi`` is unresolved.  The
    historical :func:`simultaneous_family_decision` continues to use its original,
    inclusive classifier.
    """
    if not all(np.isfinite(value) for value in (lo, hi, sesoi)):
        raise ValueError("lo, hi, and sesoi must be finite")
    if lo > hi:
        raise ValueError("lo must not exceed hi")
    if sesoi <= 0:
        raise ValueError("sesoi must be positive")
    if lo > sesoi:
        return "MATERIAL_POSITIVE"
    if hi < -sesoi:
        return "MATERIAL_NEGATIVE"
    if lo > -sesoi and hi < sesoi:
        return "EQUIVALENT_WITHIN_SESOI"
    return "UNRESOLVED"


def holm_adjust_named(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjust a complete, explicitly named family of path p-values."""
    if not pvalues:
        raise ValueError("pvalues must contain at least one named path")
    rendered = {str(name): float(value) for name, value in pvalues.items()}
    if (
        len(rendered) != len(pvalues)
        or any(not name for name in rendered)
        or any(not np.isfinite(value) or value < 0 or value > 1 for value in rendered.values())
    ):
        raise ValueError("named pvalues must have unique non-empty names and lie in [0, 1]")

    ordered = sorted(rendered, key=lambda name: (rendered[name], name))
    adjusted: dict[str, float] = {}
    running_max = 0.0
    family_size = len(ordered)
    for rank, name in enumerate(ordered):
        candidate = min(1.0, (family_size - rank) * rendered[name])
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return {name: float(adjusted[name]) for name in rendered}


@dataclass(frozen=True)
class FamilyDecision:
    estimate: float
    simultaneous_ci: tuple[float, float]
    max_abs_p: float
    classification: str

    def to_dict(self) -> dict:
        return asdict(self)


def _classify_interval(lo: float, hi: float, sesoi: float) -> str:
    if lo >= sesoi:
        return "MATERIAL_POSITIVE"
    if hi <= -sesoi:
        return "MATERIAL_NEGATIVE"
    if lo >= -sesoi and hi <= sesoi:
        return "EQUIVALENT_WITHIN_SESOI"
    return "UNRESOLVED"


def simultaneous_family_decision(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    names: Sequence[str],
    *,
    sesoi: float,
    n_draws: int = 4000,
    alpha: float = 0.05,
    seed: int = 0,
    namespace: str = "paired-component",
    min_genes: int = DEFAULT_MIN_GENES,
    studentized: bool = False,
) -> tuple[dict[str, FamilyDecision], dict]:
    """Classify a component family with one gene-bootstrap simultaneous envelope."""
    family = _as_family(values)
    if len(names) != family.shape[0] or len(set(names)) != len(names):
        raise ValueError("names must be unique and match the component axis")
    if not 0 < alpha < 1 or sesoi <= 0:
        raise ValueError("alpha must be in (0,1) and sesoi must be positive")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    labels, _, sizes, sums = _cluster_summary(family, groups, min_genes=min_genes)
    estimate = family.mean(axis=1)
    bootstrap = _cluster_bootstrap_from_summary(
        sizes,
        sums,
        n_draws=n_draws,
        seed=seed,
        namespace=f"{namespace}:cluster-bootstrap",
    )
    bootstrap_deviation = bootstrap - estimate[None, :]
    if studentized:
        if n_draws < 2:
            raise ValueError("studentized intervals require at least two bootstrap draws")
        scales = np.std(bootstrap, axis=0, ddof=1)
        positive_scale = scales > np.finfo(float).eps
        standardized = np.zeros_like(bootstrap_deviation)
        standardized[:, positive_scale] = (
            bootstrap_deviation[:, positive_scale] / scales[positive_scale]
        )
        family_max = np.max(np.abs(standardized), axis=1)
        critical_value = float(np.quantile(family_max, 1 - alpha, method="higher"))
        half_widths = critical_value * scales
        interval_method = "gene-cluster bootstrap studentized max-deviation interval"
    else:
        family_max = np.max(np.abs(bootstrap_deviation), axis=1)
        critical_value = float(np.quantile(family_max, 1 - alpha, method="higher"))
        half_widths = np.full(family.shape[0], critical_value, dtype=float)
        scales = None
        interval_method = (
            "gene-cluster bootstrap simultaneous max-absolute-deviation interval"
        )
    null_draws = _sign_flip_from_sums(
        sums,
        family.shape[1],
        n_draws=n_draws,
        seed=seed,
        namespace=f"{namespace}:gene-sign-flip",
    )
    pvalues = add_one_max_abs_pvalues(estimate, null_draws)
    decisions: dict[str, FamilyDecision] = {}
    for name, value, pvalue, half_width in zip(
        names, estimate, pvalues, half_widths
    ):
        lo, hi = float(value - half_width), float(value + half_width)
        decisions[name] = FamilyDecision(
            estimate=float(value),
            simultaneous_ci=(lo, hi),
            max_abs_p=float(pvalue),
            classification=_classify_interval(lo, hi, sesoi),
        )
    calibration = {
        "method": interval_method,
        "randomisation_unit": "gene",
        "n_components": family.shape[0],
        "n_examples": family.shape[1],
        "n_genes": len(labels),
        "minimum_genes_required": int(min_genes),
        "n_bootstrap_draws": n_draws,
        "n_sign_flip_draws": n_draws,
        "alpha": alpha,
        "sesoi": sesoi,
        "studentized": bool(studentized),
        "simultaneous_critical_value": critical_value,
        "simultaneous_half_width": float(np.max(half_widths)),
        "simultaneous_half_widths": {
            name: float(width) for name, width in zip(names, half_widths)
        },
        "bootstrap_scales": (
            None
            if scales is None
            else {name: float(scale) for name, scale in zip(names, scales)}
        ),
        "resolution_ok": bool(np.max(half_widths) <= sesoi),
        "pvalue": (
            "gene-level raw-cluster-sum sign flip; exact add-one Monte-Carlo "
            "full-family max-absolute-statistic"
        ),
    }
    return decisions, calibration


def one_sided_binomial_bounds(
    successes: int,
    trials: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Exact Clopper-Pearson one-sided lower and upper bounds."""
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials] and trials must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    lower = 0.0 if successes == 0 else float(
        beta_distribution.ppf(alpha, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta_distribution.ppf(1 - alpha, successes + 1, trials - successes)
    )
    return lower, upper


def decision_path_preflight(
    values: np.ndarray | Sequence[Sequence[float]],
    groups: Sequence[str],
    *,
    sesoi: float,
    n_calibration: int = 4000,
    n_trials: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
    namespace: str = "paired-preflight",
    min_genes: int = DEFAULT_MIN_GENES,
    plant_multiplier: float = 3.0,
    power_floor: float = 0.90,
    plant_vector: Sequence[float] | None = None,
    planted_requirements: Mapping[int, str] | None = None,
    null_declaration_paths: Sequence[Mapping[int, str]] | None = None,
    studentized: bool = False,
) -> dict:
    """Calibrate the exact production classifier on row-level nulls and plants.

    Every trial first applies one sign per gene to the raw row-level family.  The null family
    and a copy with a constant row-level plant are then passed through
    :func:`simultaneous_family_decision` with disjoint deterministic namespaces.  By default
    the first component receives ``plant_multiplier * sesoi`` and must classify materially
    positive.  A caller with a conjunctive decision may instead supply a full
    ``plant_vector`` and the required classifier label for each decision-bearing component.
    FPR and power gates use one-sided exact binomial bounds, not raw Monte-Carlo proportions.
    """
    family = _as_family(values)
    if n_calibration < 1 or n_trials < 1:
        raise ValueError("n_calibration and n_trials must be positive")
    if plant_multiplier <= 0 or not 0 < power_floor < 1:
        raise ValueError("plant_multiplier must be positive and power_floor must be in (0,1)")
    labels, _, _, _ = _cluster_summary(family, groups, min_genes=min_genes)
    names = [f"component_{index}" for index in range(family.shape[0])]
    _, reference_calibration = simultaneous_family_decision(
        family,
        groups,
        names,
        sesoi=sesoi,
        n_draws=n_calibration,
        alpha=alpha,
        seed=seed,
        namespace=f"{namespace}:reference",
        min_genes=min_genes,
        studentized=studentized,
    )

    material = {"MATERIAL_POSITIVE", "MATERIAL_NEGATIVE"}
    null_positives = 0
    planted_positives = 0
    if plant_vector is None:
        planted_offsets = np.zeros(family.shape[0], dtype=float)
        planted_offsets[0] = float(plant_multiplier * sesoi)
    else:
        planted_offsets = np.asarray(plant_vector, dtype=float)
        if (
            planted_offsets.ndim != 1
            or planted_offsets.size != family.shape[0]
            or not np.isfinite(planted_offsets).all()
        ):
            raise ValueError("plant_vector must be finite and match the component family")
    requirements = dict(planted_requirements or {0: "MATERIAL_POSITIVE"})
    allowed_classes = {
        "MATERIAL_POSITIVE",
        "MATERIAL_NEGATIVE",
        "EQUIVALENT_WITHIN_SESOI",
        "UNRESOLVED",
    }
    if (
        not requirements
        or any(index < 0 or index >= family.shape[0] for index in requirements)
        or any(label not in allowed_classes for label in requirements.values())
    ):
        raise ValueError("planted_requirements must name valid component indices and labels")
    declaration_paths = [dict(path) for path in (null_declaration_paths or ())]
    if declaration_paths and any(
        not path
        or any(index < 0 or index >= family.shape[0] for index in path)
        or any(label not in allowed_classes for label in path.values())
        for path in declaration_paths
    ):
        raise ValueError(
            "null_declaration_paths must contain non-empty valid classifier conjunctions"
        )
    for trial in range(n_trials):
        null_values = gene_sign_flip(
            family,
            groups,
            seed=seed,
            namespace=f"{namespace}:row-null:{trial}",
        )
        null_decisions, _ = simultaneous_family_decision(
            null_values,
            groups,
            names,
            sesoi=sesoi,
            n_draws=n_calibration,
            alpha=alpha,
            seed=seed,
            namespace=f"{namespace}:classifier:null:{trial}",
            min_genes=min_genes,
            studentized=studentized,
        )
        if declaration_paths:
            null_positives += int(any(
                all(
                    null_decisions[names[index]].classification == label
                    for index, label in path.items()
                )
                for path in declaration_paths
            ))
        else:
            null_positives += int(
                any(decision.classification in material for decision in null_decisions.values())
            )

        planted_values = null_values.copy()
        planted_values += planted_offsets[:, None]
        planted_decisions, _ = simultaneous_family_decision(
            planted_values,
            groups,
            names,
            sesoi=sesoi,
            n_draws=n_calibration,
            alpha=alpha,
            seed=seed,
            namespace=f"{namespace}:classifier:plant:{trial}",
            min_genes=min_genes,
            studentized=studentized,
        )
        planted_positives += int(all(
            planted_decisions[names[index]].classification == label
            for index, label in requirements.items()
        ))

    fpr = null_positives / n_trials
    power = planted_positives / n_trials
    _, fpr_upper = one_sided_binomial_bounds(null_positives, n_trials, alpha=alpha)
    power_lower, _ = one_sided_binomial_bounds(planted_positives, n_trials, alpha=alpha)
    return {
        "randomisation_unit": "gene",
        "classifier": "simultaneous_family_decision on every row-level null and plant trial",
        "namespace": namespace,
        "alpha": float(alpha),
        "sesoi": float(sesoi),
        "n_components": family.shape[0],
        "n_examples": family.shape[1],
        "n_genes": len(labels),
        "minimum_genes_required": int(min_genes),
        "n_calibration": n_calibration,
        "n_trials": n_trials,
        "production_classifier_calls": 1 + 2 * n_trials,
        "simultaneous_half_width": reference_calibration["simultaneous_half_width"],
        "simultaneous_half_widths": reference_calibration[
            "simultaneous_half_widths"
        ],
        "studentized": bool(studentized),
        "resolution_ok": reference_calibration["resolution_ok"],
        "familywise_false_positives": null_positives,
        "familywise_fpr": fpr,
        "familywise_fpr_upper": fpr_upper,
        "familywise_fpr_ok": fpr_upper <= alpha,
        "planted_effect": (
            float(planted_offsets[0])
            if np.count_nonzero(planted_offsets) == 1 and planted_offsets[0] != 0
            else None
        ),
        "plant_vector": planted_offsets.tolist(),
        "planted_requirements": {
            names[index]: label for index, label in sorted(requirements.items())
        },
        "null_declaration_policy": (
            "any material endpoint"
            if not declaration_paths
            else "any predeclared classifier conjunction"
        ),
        "null_declaration_paths": [
            {names[index]: label for index, label in sorted(path.items())}
            for path in declaration_paths
        ],
        "planted_positive_trials": planted_positives,
        "planted_power": power,
        "planted_power_lower": power_lower,
        "planted_power_ok": power_lower >= power_floor,
        "power_floor": power_floor,
        "binomial_bound": f"one-sided {100 * (1 - alpha):g}% Clopper-Pearson",
    }


__all__ = [
    "Alternative",
    "BoundKind",
    "BoundarySignFlipInversion",
    "BoundarySignFlipTest",
    "CrossingBoundarySignFlipTest",
    "DEFAULT_MIN_GENES",
    "FamilyDecision",
    "INVERTED_BOUNDARY_PVALUE_RULE",
    "INVERTED_BOUNDARY_TOLERANCE_RULE",
    "INVERTED_BOUNDARY_TRANSITION_RULE",
    "TOSTEquivalenceTest",
    "RobustGeneBootstrapTest",
    "RobustGeneBootstrapTOST",
    "StudentizedBootstrapOneSided",
    "add_one_max_abs_pvalues",
    "boundary_sign_flip_null_draws",
    "boundary_sign_flip_tests",
    "cluster_bootstrap_estimates",
    "crossing_boundary_sign_flip_tests",
    "decision_path_preflight",
    "gene_sign_flip",
    "gene_sign_flip_null_draws",
    "gene_signs",
    "holm_adjust_named",
    "invert_boundary_sign_flip_tests",
    "invert_one_sided_boundary_sign_flip",
    "intersection_union_pvalue",
    "one_sided_boundary_sign_flip",
    "one_sided_binomial_bounds",
    "simultaneous_family_decision",
    "strict_interval_decision",
    "studentized_gene_bootstrap_tests",
    "studentized_gene_bootstrap_tost",
    "tost_equivalence_test",
]
