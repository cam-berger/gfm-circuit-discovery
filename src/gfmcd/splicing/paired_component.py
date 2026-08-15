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
BOUNDARY_PVALUE_RULE = (
    "snap an oriented statistic within comparison_tolerance of zero to zero; "
    "exceedance iff oriented_null >= oriented_statistic - comparison_tolerance; "
    "p=(1+exceedances)/(B+1)"
)
BOUNDARY_COMPARISON_TOLERANCE_RULE = (
    "float64_eps * max(128,n_examples) * "
    "max(1,max_abs_centered_values,max_abs_oriented_null,abs_oriented_statistic)"
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
    "BoundarySignFlipTest",
    "DEFAULT_MIN_GENES",
    "FamilyDecision",
    "TOSTEquivalenceTest",
    "add_one_max_abs_pvalues",
    "boundary_sign_flip_null_draws",
    "boundary_sign_flip_tests",
    "cluster_bootstrap_estimates",
    "decision_path_preflight",
    "gene_sign_flip",
    "gene_sign_flip_null_draws",
    "gene_signs",
    "holm_adjust_named",
    "intersection_union_pvalue",
    "one_sided_boundary_sign_flip",
    "one_sided_binomial_bounds",
    "simultaneous_family_decision",
    "strict_interval_decision",
    "tost_equivalence_test",
]
