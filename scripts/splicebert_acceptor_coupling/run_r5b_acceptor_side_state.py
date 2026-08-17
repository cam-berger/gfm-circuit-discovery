#!/usr/bin/env python
"""R5b: cross-fitted acceptor-side residual state for the masked-acceptor G margin.

The only behavioral endpoint in this runner is the hard G margin at the masked acceptor.
It fits two fold-indexed uncentered-SVD projector families from paired-donor residual
differences, evaluates necessity and sufficiency through the checkpoint's complete MLM
head, and compares every selected rank with pseudo-donor, adjacent-basis, and 32 nested
rank/energy-matched Haar controls.

The protocol is deliberately DRAFT.  ``--stage target`` is rejected before any path is
opened while any literal seal pin remains ``None``.  ``--stage audit`` is model- and
data-free and exists so the exact algebra, registries, random pools, and selector can be
reviewed before a freeze.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import math
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.acceptor_coupling_mechanism import (  # noqa: E402
    SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE,
    crossfit_project,
    crossfit_svd,
    deterministic_nested_random_bases,
    energy_match_rows,
    fixed_calibration_overlap_null,
    hard_g_margin_from_base_logits,
    linear_mlm_base_logits,
    mean_squared_principal_cosine,
    ordered_overlap_null_draw_identities,
    orthogonal_projector,
    projector_diagnostics,
    select_crossfit_outcome_rank,
    validate_svd_numerical_rank,
)
from gfmcd.splicing.acceptor_mechanism_cohort import (  # noqa: E402
    R5_EXPECTED_CENSUS_IDENTITY,
    R5_EXPECTED_SOURCE_IDENTITY,
    R5_ROWS_PER_BUCKET,
    R5_VALIDATION_CHROMOSOMES,
    R5CohortRow,
    build_r5_reservations,
    validate_expected_r5_census_identity,
    validate_expected_r5_source_identity,
    validate_r5_reservation_audit,
)
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    load_adapter,
    load_examples_unscored,
    matched_decoy_donor,
    pick_device,
)
from gfmcd.splicing.coupling import CouplingExample  # noqa: E402
from gfmcd.splicing.paired_component import (  # noqa: E402
    BOUNDARY_PVALUE_RULE,
    boundary_sign_flip_tests,
    gene_sign_flip,
    intersection_union_pvalue,
    one_sided_binomial_bounds,
    simultaneous_family_decision,
)
from gfmcd.splicing.r5_stack_binding import (  # noqa: E402
    R3C_ENTRY_SCHEMA,
    R3C_INSTRUMENT_REVISION,
    R5StackBindingError,
    build_r5_stack_binding,
    r5_stack_fingerprint,
    validate_r3c_target_health_contract,
    validate_r3c_target_manifest_contract,
    validate_serialized_r5_stack,
)


RUNNER = Path(__file__).resolve()
PROTOCOL = (
    REPO
    / "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5B_ACCEPTOR_SIDE_STATE.md"
)
MECHANISM_CORE = REPO / "src/gfmcd/splicing/acceptor_coupling_mechanism.py"
STAT_CORE = REPO / "src/gfmcd/splicing/paired_component.py"
COHORT_CORE = REPO / "src/gfmcd/splicing/acceptor_mechanism_cohort.py"
# Every value stays a literal None until one prospective joint R5a--R5c freeze.  The
# normalized runner fingerprint replaces only these assignment values.  A development or
# audit result must never be copied into one of these constants without review.
FROZEN_PROTOCOL_SHA256: str | None = None
FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = None
FROZEN_COHORT_INPUT_SHA256: str | None = None
FROZEN_COHORT_AUDIT_SHA256: str | None = None
FROZEN_R3C_ARTIFACT_SHA256: str | None = None
FROZEN_R5A_ARTIFACT_SHA256: str | None = None
FROZEN_PREFLIGHT_ARTIFACT_SHA256: str | None = None

RANK_GRID = (1, 2, 4, 8, 16, 32)
N_CAUSAL_CONTROLS = 32
N_OVERLAP_CALIBRATION = 999
N_OVERLAP_TEST = 999
N_REFIT_BOOTSTRAP = 4000
N_INFERENCE_DRAWS = 4000
N_FOLDS = 2
ROWS_PER_FOLD = 64
TARGET_N = R5_ROWS_PER_BUCKET
PRODUCTION_BATCH = 16
SEED = 0
SESOI = 0.05
OVERLAP_SESOI = 0.20
STAGE_ALPHA = 0.05
OVERLAP_ALPHA = 0.025
RANK_ALPHA = 0.025 / len(RANK_GRID)
RECOVERY_FRACTION = 0.90
DIAGNOSTIC_ENERGY_TARGET = 0.90
MIN_GENES = 30
PROJECTOR_ATOL = 1e-8
MLM_ATOL = 1e-5
CALIBRATION_EPSILON = 0.01
CALIBRATION_POWER_FLOOR = 0.90
CALIBRATION_PLANT_SUCCESS_RATE = 0.95

CROSSFIT_SALT = "r5b-u-a-crossfit-v1"
CAUSAL_HAAR_NAMESPACE = "r5b-causal-nested-haar-v1"
OVERLAP_CALIBRATION_NAMESPACE = "r5b-overlap-haar-calibration-v1"
OVERLAP_TEST_NAMESPACE = "r5b-overlap-haar-test-v1"
REFIT_NAMESPACE = "r5b-overlap-whole-gene-refit-v1"

# This identity was obtained outcome-blind by the shared v2 cohort builder.  Importing it
# prevents R5b from maintaining a second, silently diverging copy of cohort counts/hashes.
EXPECTED_COHORT_CENSUS = R5_EXPECTED_CENSUS_IDENTITY


class TargetSealError(RuntimeError):
    """Raised before target paths or model outcomes can be inspected."""


class InstrumentError(RuntimeError):
    """Raised when a plumbing, invariant, or provenance requirement fails."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: Any, *, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.ndim}|".encode("ascii"))
    digest.update(",".join(str(int(size)) for size in array.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def ordered_text_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256(b"gfmcd.r5b.ordered-text.v1\0")
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _finite_matrix(name: str, value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a non-empty finite matrix")
    return matrix


def _finite_vector(name: str, value: Any, *, n: int | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    if n is not None and vector.size != n:
        raise ValueError(f"{name} must contain exactly {n} rows")
    return vector


def crossfit_gene_order(genes: Sequence[str]) -> tuple[str, ...]:
    rendered = tuple(str(gene) for gene in genes)
    if len(rendered) != TARGET_N or len(set(rendered)) != TARGET_N:
        raise ValueError("R5b requires exactly 128 unique one-row-per-gene records")
    if any(not gene for gene in rendered):
        raise ValueError("gene identifiers must be non-empty")
    return tuple(
        sorted(
            rendered,
            key=lambda gene: (
                hashlib.sha256(f"{CROSSFIT_SALT}|{gene}".encode()).hexdigest(),
                gene,
            ),
        )
    )


def exact_two_fold_ids(genes: Sequence[str]) -> np.ndarray:
    """Assign alternating hash-ordered genes to two exact 64-gene folds."""
    order = crossfit_gene_order(genes)
    fold_by_gene = {gene: index % N_FOLDS for index, gene in enumerate(order)}
    folds = np.asarray([fold_by_gene[str(gene)] for gene in genes], dtype=np.int64)
    counts = np.bincount(folds, minlength=N_FOLDS)
    if tuple(counts.tolist()) != (ROWS_PER_FOLD, ROWS_PER_FOLD):
        raise RuntimeError("R5b cross-fit assignment must contain two exact 64-gene folds")
    return folds


TARGET_PRIMITIVE_NAMES = (
    "target|S|partner",
    "target|N|partner",
    "target|S|pseudo",
    "target|N|pseudo",
)
ADJACENT_PRIMITIVE_NAMES = (
    "adjacent_basis|S|partner",
    "adjacent_basis|N|partner",
)


def causal_primitive_names() -> tuple[str, ...]:
    names = [*TARGET_PRIMITIVE_NAMES, *ADJACENT_PRIMITIVE_NAMES]
    for draw in range(N_CAUSAL_CONTROLS):
        names.extend(
            (
                f"random|draw={draw:02d}|S|partner",
                f"random|draw={draw:02d}|N|partner",
            )
        )
    if len(names) != 70 or len(set(names)) != 70:
        raise RuntimeError("R5b causal primitive registry drifted")
    return tuple(names)


def causal_family_names() -> tuple[str, ...]:
    names = [
        "target|S|partner",
        "target|N|partner",
        "target|S|pseudo",
        "target|N|pseudo",
        "target|S|specific",
        "target|N|specific",
        "adjacent_basis|S|partner",
        "adjacent_basis|N|partner",
    ]
    for draw in range(N_CAUSAL_CONTROLS):
        names.extend(
            (
                f"random|draw={draw:02d}|S|partner",
                f"random|draw={draw:02d}|N|partner",
            )
        )
    for draw in range(N_CAUSAL_CONTROLS):
        names.extend(
            (
                f"target_minus_random|draw={draw:02d}|S|partner",
                f"target_minus_random|draw={draw:02d}|N|partner",
            )
        )
    if len(names) != 136 or len(set(names)) != 136:
        raise RuntimeError("R5b causal family must contain exactly 136 unique rows")
    return tuple(names)


def derive_causal_family(
    primitive: Any,
    *,
    primitive_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Regenerate specificity and all target-minus-random rows from raw margin cells."""
    values = _finite_matrix("primitive", primitive)
    expected = causal_primitive_names()
    supplied = expected if primitive_names is None else tuple(map(str, primitive_names))
    if supplied != expected or values.shape[0] != len(expected):
        raise ValueError("primitive rows must use the exact frozen 70-row registry")
    by_name = {name: values[index] for index, name in enumerate(expected)}
    target_s = by_name["target|S|partner"]
    target_n = by_name["target|N|partner"]
    rows: list[np.ndarray] = [
        target_s,
        target_n,
        by_name["target|S|pseudo"],
        by_name["target|N|pseudo"],
        target_s - by_name["target|S|pseudo"],
        target_n - by_name["target|N|pseudo"],
        by_name["adjacent_basis|S|partner"],
        by_name["adjacent_basis|N|partner"],
    ]
    for draw in range(N_CAUSAL_CONTROLS):
        rows.extend(
            (
                by_name[f"random|draw={draw:02d}|S|partner"],
                by_name[f"random|draw={draw:02d}|N|partner"],
            )
        )
    for draw in range(N_CAUSAL_CONTROLS):
        rows.extend(
            (
                target_s - by_name[f"random|draw={draw:02d}|S|partner"],
                target_n - by_name[f"random|draw={draw:02d}|N|partner"],
            )
        )
    family = np.vstack(rows)
    names = causal_family_names()
    if family.shape[0] != len(names):
        raise RuntimeError("R5b causal derived-family registry drifted")
    return family, names


def causal_requirement_registry() -> dict[str, str]:
    requirements = {
        "target|S|partner": "MATERIAL_POSITIVE",
        "target|N|partner": "MATERIAL_POSITIVE",
        "target|S|pseudo": "EQUIVALENT_WITHIN_SESOI",
        "target|N|pseudo": "EQUIVALENT_WITHIN_SESOI",
        "target|S|specific": "MATERIAL_POSITIVE",
        "target|N|specific": "MATERIAL_POSITIVE",
        "adjacent_basis|S|partner": "EQUIVALENT_WITHIN_SESOI",
        "adjacent_basis|N|partner": "EQUIVALENT_WITHIN_SESOI",
    }
    for draw in range(N_CAUSAL_CONTROLS):
        requirements[f"target_minus_random|draw={draw:02d}|S|partner"] = (
            "POSITIVE_ABOVE_ZERO"
        )
        requirements[f"target_minus_random|draw={draw:02d}|N|partner"] = (
            "POSITIVE_ABOVE_ZERO"
        )
    if set(requirements) - set(causal_family_names()):
        raise RuntimeError("R5b requirement registry references an unknown family row")
    return requirements


def causal_boundary_edge_registry() -> tuple[dict[str, Any], ...]:
    edges: list[dict[str, Any]] = []
    for row, requirement in causal_requirement_registry().items():
        if requirement == "MATERIAL_POSITIVE":
            edges.append(
                {
                    "edge": f"{row}|positive_edge",
                    "row": row,
                    "boundary": SESOI,
                    "alternative": "greater",
                    "requirement": requirement,
                }
            )
        elif requirement == "EQUIVALENT_WITHIN_SESOI":
            edges.extend(
                (
                    {
                        "edge": f"{row}|lower_edge",
                        "row": row,
                        "boundary": -SESOI,
                        "alternative": "greater",
                        "requirement": requirement,
                    },
                    {
                        "edge": f"{row}|upper_edge",
                        "row": row,
                        "boundary": SESOI,
                        "alternative": "less",
                        "requirement": requirement,
                    },
                )
            )
        elif requirement == "POSITIVE_ABOVE_ZERO":
            edges.append(
                {
                    "edge": f"{row}|positive_edge",
                    "row": row,
                    "boundary": 0.0,
                    "alternative": "greater",
                    "requirement": requirement,
                }
            )
        else:  # pragma: no cover - guarded by the literal registry above
            raise RuntimeError(f"unknown R5b requirement {requirement}")
    if len(edges) != 76 or len({edge["edge"] for edge in edges}) != 76:
        raise RuntimeError("R5b path must contain exactly 76 decision edges")
    return tuple(edges)


def _strict_ci_pass(interval: Sequence[float], requirement: str) -> bool:
    if len(interval) != 2:
        return False
    lo, hi = map(float, interval)
    if not np.isfinite([lo, hi]).all() or lo > hi:
        return False
    if requirement == "MATERIAL_POSITIVE":
        return lo > SESOI
    if requirement == "EQUIVALENT_WITHIN_SESOI":
        return lo > -SESOI and hi < SESOI
    if requirement == "POSITIVE_ABOVE_ZERO":
        return lo > 0.0
    raise ValueError(f"unknown requirement: {requirement}")


def infer_causal_rank_family(
    primitive: Any,
    genes: Sequence[str],
    *,
    rank: int,
    n_draws: int = N_INFERENCE_DRAWS,
    seed: int = SEED,
    namespace: str = "r5b-causal-family",
    min_genes: int = MIN_GENES,
) -> dict[str, Any]:
    """Run one complete rank-specific 136-row family and its 76-edge IUT path."""
    if rank not in RANK_GRID:
        raise ValueError("rank must belong to the frozen R5b rank grid")
    family, names = derive_causal_family(primitive)
    if family.shape[1] != len(genes):
        raise ValueError("family columns and genes disagree")
    intervals, interval_calibration = simultaneous_family_decision(
        family,
        genes,
        names,
        sesoi=SESOI,
        n_draws=n_draws,
        alpha=RANK_ALPHA,
        seed=seed,
        namespace=f"{namespace}:rank={rank}:simultaneous-envelope",
        min_genes=min_genes,
        studentized=True,
    )
    index = {name: position for position, name in enumerate(names)}
    registry = causal_boundary_edge_registry()
    edge_values = np.vstack([family[index[edge["row"]]] for edge in registry])
    tests = boundary_sign_flip_tests(
        edge_values,
        genes,
        [edge["edge"] for edge in registry],
        boundaries=[float(edge["boundary"]) for edge in registry],
        alternatives=[str(edge["alternative"]) for edge in registry],
        n_draws=n_draws,
        seed=seed,
        namespace=f"{namespace}:rank={rank}:shared-boundary-edges",
        min_genes=min_genes,
    )
    path_p = intersection_union_pvalue(
        [tests[edge["edge"]].p_value for edge in registry]
    )
    requirements = causal_requirement_registry()
    row_records: dict[str, Any] = {}
    for row, requirement in requirements.items():
        decision = intervals[row]
        row_edges = [edge for edge in registry if edge["row"] == row]
        row_p = intersection_union_pvalue(
            [tests[edge["edge"]].p_value for edge in row_edges]
        )
        row_records[row] = {
            "estimate": float(decision.estimate),
            "simultaneous_ci": list(map(float, decision.simultaneous_ci)),
            "requirement": requirement,
            "boundary_iut_p_value": row_p,
            "boundary_pass": bool(row_p <= RANK_ALPHA),
            "strict_ci_pass": _strict_ci_pass(decision.simultaneous_ci, requirement),
        }
    all_ci_pass = all(record["strict_ci_pass"] for record in row_records.values())
    gate = bool(path_p <= RANK_ALPHA and all_ci_pass)
    sign_digests = {test.sign_matrix_sha256 for test in tests.values()}
    gene_digests = {test.ordered_gene_labels_sha256 for test in tests.values()}
    if len(sign_digests) != 1 or len(gene_digests) != 1:
        raise RuntimeError("R5b decision edges did not share one gene sign matrix")
    return {
        "schema": "gfmcd.r5b.causal_rank_family/1",
        "rank": int(rank),
        "family_order": list(names),
        "family_order_sha256": canonical_sha256(list(names)),
        "family_size": len(names),
        "primitive_order": list(causal_primitive_names()),
        "primitive_order_sha256": canonical_sha256(list(causal_primitive_names())),
        "primitive_sha256": array_sha256(primitive),
        "rank_alpha": RANK_ALPHA,
        "stagewise_allocation": "0.025 / 6 ranks",
        "boundary_edge_registry": list(registry),
        "boundary_edge_registry_sha256": canonical_sha256(list(registry)),
        "boundary_randomisation": {
            "B": int(n_draws),
            "seed": int(seed),
            "namespace": f"{namespace}:rank={rank}:shared-boundary-edges",
            "all_edges_share_one_gene_sign_matrix": True,
            "sign_matrix_sha256": next(iter(sign_digests)),
            "ordered_gene_labels_sha256": next(iter(gene_digests)),
            "pvalue_rule": BOUNDARY_PVALUE_RULE,
        },
        "edge_tests": {name: test.to_dict() for name, test in tests.items()},
        "interval_calibration": interval_calibration,
        "required_rows": row_records,
        "path_iut_p_value": float(path_p),
        "path_p_pass": bool(path_p <= RANK_ALPHA),
        "all_strict_ci_requirements_pass": bool(all_ci_pass),
        "random_control_gate": gate,
    }


@dataclass(frozen=True)
class RefitOverlapIntervals:
    observed_by_rank: dict[int, float]
    lower_by_rank: dict[int, float]
    simultaneous_deviation_critical: float
    bootstrap_sha256: str
    draw_ids_sha256: str
    n_draws: int
    seed: int
    namespace: str

    def record(self) -> dict[str, Any]:
        return {
            "observed_by_rank": {str(k): v for k, v in self.observed_by_rank.items()},
            "simultaneous_lower_by_rank": {
                str(k): v for k, v in self.lower_by_rank.items()
            },
            "simultaneous_deviation_critical": self.simultaneous_deviation_critical,
            "bootstrap_sha256": self.bootstrap_sha256,
            "ordered_draw_ids_sha256": self.draw_ids_sha256,
            "n_draws": self.n_draws,
            "seed": self.seed,
            "namespace": self.namespace,
            "method": (
                "whole-gene two-half uncentered-SVD refit; 95% simultaneous lower "
                "basic interval from max(observed - refit) across complete rank grid"
            ),
        }


def _svd_basis(rows: np.ndarray, maximum_rank: int) -> np.ndarray:
    """Exact uncentered right-SVD basis, optimized through the smaller Gram matrix."""
    matrix = _finite_matrix("rows", rows)
    if maximum_rank < 1 or maximum_rank > min(matrix.shape):
        raise ValueError("maximum_rank exceeds the available uncentered SVD rank")
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    singular = np.sqrt(eigenvalues[:maximum_rank])
    # Squaring the condition number in the Gram matrix makes ratios near sqrt(machine
    # epsilon) ambiguous.  Only in that narrow zone, fall back to a direct SVD so the
    # frozen 1e-10 numerical-rank rule is actually enforceable without paying for a full
    # SVD on every well-conditioned one of the 8,000 bootstrap refits.
    ambiguity_floor = 10.0 * np.sqrt(np.finfo(np.float64).eps)
    direct_basis: np.ndarray | None = None
    if singular[0] <= 0 or singular[-1] / singular[0] <= ambiguity_floor:
        _, direct_singular, direct_vh = np.linalg.svd(matrix, full_matrices=False)
        singular = direct_singular[:maximum_rank]
        direct_basis = direct_vh[:maximum_rank].copy()
    try:
        validate_svd_numerical_rank(singular, required_rank=maximum_rank)
    except ValueError as exc:
        raise InstrumentError(
            "uncentered SVD refit is numerically rank deficient under the frozen "
            f"relative tolerance {SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE:g}"
        ) from exc
    basis = (
        direct_basis
        if direct_basis is not None
        else (matrix.T @ eigenvectors[:, :maximum_rank] / singular[None, :]).T
    )
    # Do not QR the complete basis: that preserves only the maximum-rank span and can rotate
    # the ordered prefix subspaces, invalidating the rank scan.  The Gram eigenvectors already
    # give ordered right singular vectors; row normalization removes only scalar round-off.
    norms = np.linalg.norm(basis, axis=1)
    basis = basis / norms[:, None]
    if not np.allclose(
        basis @ basis.T,
        np.eye(maximum_rank),
        rtol=0.0,
        atol=PROJECTOR_ATOL,
    ):
        raise InstrumentError("uncentered Gram-SVD basis lost row orthonormality")
    return basis


def _fold_matrices(delta_h: Any, folds: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    delta = _finite_matrix("delta_h", delta_h)
    fold_array = np.asarray(folds)
    if fold_array.shape != (delta.shape[0],) or set(fold_array.tolist()) != {0, 1}:
        raise ValueError("fold IDs must align and contain exactly folds 0 and 1")
    left, right = delta[fold_array == 0], delta[fold_array == 1]
    if left.shape[0] != ROWS_PER_FOLD or right.shape[0] != ROWS_PER_FOLD:
        raise ValueError("R5b overlap refit requires two exact 64-gene halves")
    return left, right


def observed_crossfold_overlap(
    fitted: Any,
    *,
    rank_grid: Sequence[int] = RANK_GRID,
) -> dict[int, float]:
    if len(fitted.folds) != 2:
        raise ValueError("R5b requires exactly two fitted cross-fit folds")
    by_fold = {fold.evaluation_fold: fold for fold in fitted.folds}
    if set(by_fold) != {"0", "1"}:
        raise ValueError("R5b fitted folds must be literal 0 and 1")
    left = by_fold["0"].svd.basis
    right = by_fold["1"].svd.basis
    return {
        int(rank): mean_squared_principal_cosine(
            left[: int(rank)], right[: int(rank)]
        )
        for rank in rank_grid
    }


def refit_overlap_intervals(
    delta_h: Any,
    folds: Sequence[int],
    observed_by_rank: Mapping[int, float],
    *,
    rank_grid: Sequence[int] = RANK_GRID,
    n_draws: int = N_REFIT_BOOTSTRAP,
    seed: int = SEED,
    namespace: str = REFIT_NAMESPACE,
) -> RefitOverlapIntervals:
    """Refit both uncentered SVDs on independently resampled whole genes."""
    ranks = tuple(map(int, rank_grid))
    if ranks != tuple(sorted(set(ranks))) or ranks[-1] > 32:
        raise ValueError("rank_grid must be an increasing prefix grid through at most 32")
    if set(observed_by_rank) != set(ranks) or n_draws < 2:
        raise ValueError("observed overlap/rank registry or refit draw count differs")
    left, right = _fold_matrices(delta_h, folds)
    digest = hashlib.sha256(f"{namespace}|seed={int(seed)}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    bootstrap = np.empty((n_draws, len(ranks)), dtype=np.float64)
    for draw in range(n_draws):
        left_indices = rng.integers(0, len(left), size=len(left))
        right_indices = rng.integers(0, len(right), size=len(right))
        left_basis = _svd_basis(left[left_indices], ranks[-1])
        right_basis = _svd_basis(right[right_indices], ranks[-1])
        for index, rank in enumerate(ranks):
            bootstrap[draw, index] = mean_squared_principal_cosine(
                left_basis[:rank], right_basis[:rank]
            )
    observed = np.asarray([float(observed_by_rank[rank]) for rank in ranks])
    max_deviation = np.max(observed[None, :] - bootstrap, axis=1)
    critical = float(np.quantile(max_deviation, 0.95, method="higher"))
    lower = observed - critical
    return RefitOverlapIntervals(
        observed_by_rank={rank: float(value) for rank, value in zip(ranks, observed)},
        lower_by_rank={rank: float(value) for rank, value in zip(ranks, lower)},
        simultaneous_deviation_critical=critical,
        bootstrap_sha256=array_sha256(bootstrap),
        draw_ids_sha256=canonical_sha256(list(range(n_draws))),
        n_draws=int(n_draws),
        seed=int(seed),
        namespace=str(namespace),
    )


def _haar_overlap_pool(
    width: int,
    *,
    rank_grid: Sequence[int],
    n_draws: int,
    seed: int,
    namespace: str,
) -> dict[int, np.ndarray]:
    ranks = tuple(map(int, rank_grid))
    pool = {rank: np.empty(n_draws, dtype=np.float64) for rank in ranks}
    for draw in range(n_draws):
        left = deterministic_nested_random_bases(
            width,
            ranks,
            seed=seed,
            namespace=f"{namespace}:left",
            draw=draw,
        )
        right = deterministic_nested_random_bases(
            width,
            ranks,
            seed=seed,
            namespace=f"{namespace}:right",
            draw=draw,
        )
        for rank in ranks:
            pool[rank][draw] = mean_squared_principal_cosine(
                left[rank], right[rank]
            )
    return pool


def overlap_family_inference(
    observed_by_rank: Mapping[int, float],
    refit: RefitOverlapIntervals,
    *,
    width: int,
    rank_grid: Sequence[int] = RANK_GRID,
    n_calibration: int = N_OVERLAP_CALIBRATION,
    n_test: int = N_OVERLAP_TEST,
    seed: int = SEED,
    calibration_pool: Mapping[int, np.ndarray] | None = None,
    test_pool: Mapping[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run the disjoint fixed-calibration nested-Haar maximum overlap family."""
    ranks = tuple(map(int, rank_grid))
    if set(observed_by_rank) != set(ranks) or set(refit.lower_by_rank) != set(ranks):
        raise ValueError("overlap maps must contain the complete rank grid")
    if (calibration_pool is None) != (test_pool is None):
        raise ValueError("overlap calibration/test pools must be supplied together")
    calibration = (
        _haar_overlap_pool(
            width,
            rank_grid=ranks,
            n_draws=n_calibration,
            seed=seed,
            namespace=OVERLAP_CALIBRATION_NAMESPACE,
        )
        if calibration_pool is None
        else {rank: np.asarray(calibration_pool[rank]) for rank in ranks}
    )
    test = (
        _haar_overlap_pool(
            width,
            rank_grid=ranks,
            n_draws=n_test,
            seed=seed,
            namespace=OVERLAP_TEST_NAMESPACE,
        )
        if test_pool is None
        else {rank: np.asarray(test_pool[rank]) for rank in ranks}
    )
    if any(
        np.asarray(calibration[rank]).shape != (n_calibration,)
        or np.asarray(test[rank]).shape != (n_test,)
        for rank in ranks
    ):
        raise ValueError("supplied overlap pools differ from their declared draw counts")
    calibration_draw_identities = ordered_overlap_null_draw_identities(
        namespace=OVERLAP_CALIBRATION_NAMESPACE,
        seed=seed,
        n_draws=n_calibration,
    )
    test_draw_identities = ordered_overlap_null_draw_identities(
        namespace=OVERLAP_TEST_NAMESPACE,
        seed=seed,
        n_draws=n_test,
    )
    fixed = fixed_calibration_overlap_null(
        observed_by_rank,
        calibration,
        test,
        rank_grid=ranks,
        calibration_draw_identities=calibration_draw_identities,
        test_draw_identities=test_draw_identities,
    )
    observed_z = dict(fixed.observed_standardized_by_rank)
    critical = float(
        np.quantile(fixed.test_max_standardized, 1.0 - OVERLAP_ALPHA, method="higher")
    )
    by_rank: dict[str, Any] = {}
    gates: dict[int, bool] = {}
    for rank in ranks:
        z = float(observed_z[rank])
        exceedances = int(np.count_nonzero(fixed.test_max_standardized >= z))
        p_value = float((1 + exceedances) / (n_test + 1))
        lower = float(refit.lower_by_rank[rank])
        gate = bool(z > critical and p_value <= OVERLAP_ALPHA and lower > OVERLAP_SESOI)
        gates[rank] = gate
        by_rank[str(rank)] = {
            "observed_overlap": float(observed_by_rank[rank]),
            "fixed_calibration_z": z,
            "max_family_critical_value": critical,
            "max_family_exceedances": exceedances,
            "max_family_add_one_p_value": p_value,
            "simultaneous_refit_lower": lower,
            "absolute_overlap_sesoi": OVERLAP_SESOI,
            "chance_overlap_pass": bool(z > critical and p_value <= OVERLAP_ALPHA),
            "absolute_overlap_pass": bool(lower > OVERLAP_SESOI),
            "random_overlap_gate": gate,
        }
    return {
        "schema": "gfmcd.r5b.overlap_family/1",
        "rank_grid": list(ranks),
        "alpha": OVERLAP_ALPHA,
        "absolute_overlap_sesoi": OVERLAP_SESOI,
        "calibration_namespace": OVERLAP_CALIBRATION_NAMESPACE,
        "test_namespace": OVERLAP_TEST_NAMESPACE,
        "ordered_calibration_draw_ids_sha256": canonical_sha256(
            list(range(n_calibration))
        ),
        "ordered_test_draw_ids_sha256": canonical_sha256(list(range(n_test))),
        "fixed_calibration": fixed.record(),
        "refit": refit.record(),
        "by_rank": by_rank,
        "random_overlap_gate_by_rank": {
            str(rank): gate for rank, gate in gates.items()
        },
        "calibration_and_test_pools_disjoint": bool(
            fixed.calibration_pool_sha256 != fixed.test_pool_sha256
        ),
    }


@dataclass(frozen=True)
class ResidualCapture:
    """Transient final-residual cells for one acceptor-only R5 reservation."""

    genes: tuple[str, ...]
    strands: tuple[str, ...]
    example_ids: tuple[str, ...]
    h_clean: np.ndarray
    h_partner_corrupt: np.ndarray
    h_pseudo_corrupt: np.ndarray
    h_adjacent_clean_mean: np.ndarray
    h_adjacent_partner_corrupt_mean: np.ndarray
    m_clean: np.ndarray
    m_partner_corrupt: np.ndarray
    m_pseudo_corrupt: np.ndarray
    capability: dict[str, Any]

    def validate(self) -> None:
        if (
            len(self.genes) != TARGET_N
            or len(set(self.genes)) != TARGET_N
            or len(self.strands) != TARGET_N
            or set(self.strands) != {"+", "-"}
            or len(self.example_ids) != TARGET_N
            or len(set(self.example_ids)) != TARGET_N
        ):
            raise InstrumentError(
                "R5b capture must contain 128 unique genes/examples and both strands"
            )
        matrices = {
            "h_clean": self.h_clean,
            "h_partner_corrupt": self.h_partner_corrupt,
            "h_pseudo_corrupt": self.h_pseudo_corrupt,
            "h_adjacent_clean_mean": self.h_adjacent_clean_mean,
            "h_adjacent_partner_corrupt_mean": self.h_adjacent_partner_corrupt_mean,
        }
        normalized = {name: _finite_matrix(name, value) for name, value in matrices.items()}
        shapes = {value.shape for value in normalized.values()}
        if len(shapes) != 1 or next(iter(shapes))[0] != TARGET_N:
            raise InstrumentError("all R5b residual captures must share [128, hidden] shape")
        for name, value in (
            ("m_clean", self.m_clean),
            ("m_partner_corrupt", self.m_partner_corrupt),
            ("m_pseudo_corrupt", self.m_pseudo_corrupt),
        ):
            _finite_vector(name, value, n=TARGET_N)
        if self.capability.get("all_controls_pass") is not True:
            raise InstrumentError("full-MLM/dense-residual capability did not pass")

    def digest_record(self) -> dict[str, Any]:
        return {
            "ordered_example_ids_sha256": ordered_text_sha256(self.example_ids),
            "ordered_gene_ids_sha256": ordered_text_sha256(self.genes),
            "h_clean_sha256": array_sha256(self.h_clean),
            "h_partner_corrupt_sha256": array_sha256(self.h_partner_corrupt),
            "h_pseudo_corrupt_sha256": array_sha256(self.h_pseudo_corrupt),
            "h_adjacent_clean_mean_sha256": array_sha256(
                self.h_adjacent_clean_mean
            ),
            "h_adjacent_partner_corrupt_mean_sha256": array_sha256(
                self.h_adjacent_partner_corrupt_mean
            ),
            "m_clean_sha256": array_sha256(self.m_clean),
            "m_partner_corrupt_sha256": array_sha256(self.m_partner_corrupt),
            "m_pseudo_corrupt_sha256": array_sha256(self.m_pseudo_corrupt),
            "activation_arrays_serialized": False,
        }


def _torch_g_margin(logits: torch.Tensor, base_token_ids: Mapping[str, int]) -> torch.Tensor:
    if set(base_token_ids) != {"A", "C", "G", "T"}:
        raise ValueError("base_token_ids must contain exactly A/C/G/T")
    if logits.ndim != 2:
        raise ValueError("logits must have shape [rows, vocabulary]")
    g = logits[:, int(base_token_ids["G"])]
    alternatives = logits[
        :, [int(base_token_ids[base]) for base in ("A", "C", "T")]
    ]
    return g - alternatives.max(dim=-1).values


class FullMLMScorer:
    """Score final residuals through the checkpoint's complete nonlinear MLM head."""

    def __init__(
        self,
        cls: torch.nn.Module,
        base_token_ids: Mapping[str, int],
        *,
        device: str | torch.device,
        batch_size: int,
    ) -> None:
        self.cls = cls
        self.base_token_ids = {base: int(value) for base, value in base_token_ids.items()}
        if set(self.base_token_ids) != {"A", "C", "G", "T"}:
            raise ValueError("base token registry differs")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        try:
            self.dtype = next(cls.parameters()).dtype
        except StopIteration:
            self.dtype = torch.float32

    def logits(self, residuals: Any) -> np.ndarray:
        matrix = _finite_matrix("residuals", residuals)
        rows: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(matrix), self.batch_size):
                tensor = torch.from_numpy(
                    np.ascontiguousarray(matrix[start : start + self.batch_size], dtype=np.float32)
                ).to(self.device, self.dtype)
                value = self.cls(tensor)
                if not isinstance(value, torch.Tensor) or value.ndim != 2:
                    raise InstrumentError("checkpoint cls(final_residual) did not return [B,V]")
                rows.append(value.float().cpu().numpy())
        return np.concatenate(rows, axis=0)

    def margins(self, residuals: Any) -> np.ndarray:
        logits = self.logits(residuals)
        selected = logits[
            :,
            [self.base_token_ids[base] for base in ("A", "C", "G", "T")],
        ]
        return np.asarray(hard_g_margin_from_base_logits(selected), dtype=np.float64)


def full_mlm_capability(
    cls: torch.nn.Module,
    *,
    h_clean: Any,
    h_partner_corrupt: Any,
    native_clean_logits: Any,
    native_partner_logits: Any,
    dense_clean_patch_logits: Any,
    dense_self_patch_logits: Any,
    base_token_ids: Mapping[str, int],
    device: str | torch.device,
    batch_size: int,
    atol: float = MLM_ATOL,
) -> dict[str, Any]:
    """Check direct full-MLM replay, decoder-after-transform, and dense hook replay."""
    clean = _finite_matrix("h_clean", h_clean)
    corrupt = _finite_matrix("h_partner_corrupt", h_partner_corrupt)
    clean_logits = _finite_matrix("native_clean_logits", native_clean_logits)
    corrupt_logits = _finite_matrix("native_partner_logits", native_partner_logits)
    dense_clean = _finite_matrix("dense_clean_patch_logits", dense_clean_patch_logits)
    dense_self = _finite_matrix("dense_self_patch_logits", dense_self_patch_logits)
    if clean.shape != corrupt.shape or not (
        clean_logits.shape
        == corrupt_logits.shape
        == dense_clean.shape
        == dense_self.shape
    ):
        raise ValueError("full-MLM capability cells are not aligned")
    if clean.shape[0] != clean_logits.shape[0] or atol <= 0:
        raise ValueError("residual/logit rows disagree or tolerance is invalid")

    scorer = FullMLMScorer(
        cls,
        base_token_ids,
        device=device,
        batch_size=batch_size,
    )
    replay_clean = scorer.logits(clean)
    replay_corrupt = scorer.logits(corrupt)
    direct_clean_error = float(np.max(np.abs(replay_clean - clean_logits)))
    direct_corrupt_error = float(np.max(np.abs(replay_corrupt - corrupt_logits)))
    dense_clean_error = float(np.max(np.abs(dense_clean - clean_logits)))
    dense_self_error = float(np.max(np.abs(dense_self - corrupt_logits)))

    predictions = getattr(cls, "predictions", None)
    transform = getattr(predictions, "transform", None)
    decoder = getattr(predictions, "decoder", None)
    if transform is None or decoder is None or not hasattr(decoder, "weight"):
        raise InstrumentError("checkpoint MLM head lacks predictions.transform/decoder")
    decoder_bias = getattr(decoder, "bias", None)
    if decoder_bias is None:
        decoder_bias = getattr(predictions, "bias", None)
    if decoder_bias is None:
        raise InstrumentError("checkpoint MLM decoder bias is unavailable")
    transformed: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(clean), batch_size):
            tensor = torch.from_numpy(
                np.ascontiguousarray(clean[start : start + batch_size], dtype=np.float32)
            ).to(scorer.device, scorer.dtype)
            transformed.append(transform(tensor).float().cpu().numpy())
    linear_input = np.concatenate(transformed, axis=0)
    selected_linear = linear_mlm_base_logits(
        linear_input,
        decoder.weight.detach().float().cpu().numpy(),
        decoder_bias.detach().float().cpu().numpy(),
        base_token_ids,
    )
    selected_indices = [int(base_token_ids[base]) for base in ("A", "C", "G", "T")]
    selected_native = clean_logits[:, selected_indices]
    transform_logit_error = float(np.max(np.abs(selected_linear - selected_native)))
    transform_margin_error = float(
        np.max(
            np.abs(
                np.asarray(hard_g_margin_from_base_logits(selected_linear))
                - np.asarray(hard_g_margin_from_base_logits(selected_native))
            )
        )
    )
    errors = {
        "direct_clean_full_logits_max_abs": direct_clean_error,
        "direct_corrupt_full_logits_max_abs": direct_corrupt_error,
        "decoder_after_transform_base_logits_max_abs": transform_logit_error,
        "decoder_after_transform_margin_max_abs": transform_margin_error,
        "dense_clean_residual_patch_full_logits_max_abs": dense_clean_error,
        "dense_corrupt_self_patch_full_logits_max_abs": dense_self_error,
    }
    return {
        "schema": "gfmcd.r5b.full_mlm_dense_residual_capability/1",
        "single_masked_acceptor_G_endpoint": True,
        "masked_donor_endpoint_constructed": False,
        "full_mlm_head_used": True,
        "final_residual_to_margin_called_linear": False,
        "decoder_linearity_scope": "only predictions.transform(h) -> decoder logits",
        "same_row_dense_patch": True,
        "candidate_projector_used": False,
        "atol": float(atol),
        "errors": errors,
        "all_controls_pass": bool(all(error < atol for error in errors.values())),
    }


def _score_projected_component(
    scorer: FullMLMScorer,
    *,
    clean_residual: np.ndarray,
    corrupt_residual: np.ndarray,
    clean_margin: np.ndarray,
    corrupt_margin: np.ndarray,
    projected_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sufficiency = scorer.margins(corrupt_residual + projected_delta) - corrupt_margin
    necessity = clean_margin - scorer.margins(clean_residual - projected_delta)
    return sufficiency, necessity


def _random_crossfit_projection(
    delta_h: np.ndarray,
    folds: np.ndarray,
    *,
    rank: int,
    draw: int,
    seed: int,
    binding: str,
) -> tuple[np.ndarray, dict[str, str]]:
    output = np.empty_like(delta_h)
    digests: dict[str, str] = {}
    covered = np.zeros(len(delta_h), dtype=np.int64)
    for evaluation_fold in (0, 1):
        namespace = (
            f"{CAUSAL_HAAR_NAMESPACE}|binding={binding}|"
            f"evaluation_fold={evaluation_fold}|fit_from_fold={1-evaluation_fold}"
        )
        bases = deterministic_nested_random_bases(
            delta_h.shape[1],
            RANK_GRID,
            seed=seed,
            namespace=namespace,
            draw=draw,
        )
        indices = np.flatnonzero(folds == evaluation_fold)
        projector = orthogonal_projector(bases[rank])
        output[indices] = delta_h[indices] @ projector
        covered[indices] += 1
        digests[str(evaluation_fold)] = array_sha256(bases[RANK_GRID[-1]])
    if not np.all(covered == 1):
        raise InstrumentError("random cross-fit control did not cover every row exactly once")
    return output, digests


def intervention_primitives_by_rank(
    capture: ResidualCapture,
    scorer: FullMLMScorer,
    fitted: Any,
    adjacent_fitted: Any,
    folds: np.ndarray,
    *,
    seed: int,
    binding: str,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Evaluate target, pseudo, adjacent-basis, and nested Haar margin primitives."""
    delta_partner = capture.h_clean - capture.h_partner_corrupt
    delta_pseudo = capture.h_clean - capture.h_pseudo_corrupt
    outputs: dict[int, np.ndarray] = {}
    random_basis_digests: dict[str, Any] = {}
    for rank in RANK_GRID:
        target_partner = crossfit_project(delta_partner, fitted, rank=rank)
        target_pseudo = crossfit_project(delta_pseudo, fitted, rank=rank)
        adjacent_raw = crossfit_project(delta_partner, adjacent_fitted, rank=rank)
        adjacent = energy_match_rows(target_partner, adjacent_raw)
        target_s, target_n = _score_projected_component(
            scorer,
            clean_residual=capture.h_clean,
            corrupt_residual=capture.h_partner_corrupt,
            clean_margin=capture.m_clean,
            corrupt_margin=capture.m_partner_corrupt,
            projected_delta=target_partner,
        )
        pseudo_s, pseudo_n = _score_projected_component(
            scorer,
            clean_residual=capture.h_clean,
            corrupt_residual=capture.h_pseudo_corrupt,
            clean_margin=capture.m_clean,
            corrupt_margin=capture.m_pseudo_corrupt,
            projected_delta=target_pseudo,
        )
        adjacent_s, adjacent_n = _score_projected_component(
            scorer,
            clean_residual=capture.h_clean,
            corrupt_residual=capture.h_partner_corrupt,
            clean_margin=capture.m_clean,
            corrupt_margin=capture.m_partner_corrupt,
            projected_delta=adjacent,
        )
        primitive_rows = [target_s, target_n, pseudo_s, pseudo_n, adjacent_s, adjacent_n]
        rank_digests: dict[str, Any] = {}
        for draw in range(N_CAUSAL_CONTROLS):
            random_raw, basis_digests = _random_crossfit_projection(
                delta_partner,
                folds,
                rank=rank,
                draw=draw,
                seed=seed,
                binding=binding,
            )
            random_vector = energy_match_rows(target_partner, random_raw)
            random_s, random_n = _score_projected_component(
                scorer,
                clean_residual=capture.h_clean,
                corrupt_residual=capture.h_partner_corrupt,
                clean_margin=capture.m_clean,
                corrupt_margin=capture.m_partner_corrupt,
                projected_delta=random_vector,
            )
            primitive_rows.extend((random_s, random_n))
            rank_digests[f"draw={draw:02d}"] = {
                "maximum_rank_basis_sha256_by_evaluation_fold": basis_digests,
                "rank": rank,
                "per_row_energy_match_max_abs": float(
                    np.max(
                        np.abs(
                            np.linalg.norm(random_vector, axis=1)
                            - np.linalg.norm(target_partner, axis=1)
                        )
                    )
                ),
            }
        primitive = np.vstack(primitive_rows)
        if primitive.shape != (len(causal_primitive_names()), TARGET_N):
            raise InstrumentError("R5b intervention primitive shape drifted")
        outputs[rank] = primitive
        random_basis_digests[str(rank)] = rank_digests
    return outputs, {
        "namespace": CAUSAL_HAAR_NAMESPACE,
        "binding": binding,
        "draws_per_rank": N_CAUSAL_CONTROLS,
        "nested_prefix_grid": list(RANK_GRID),
        "no_redraw": True,
        "no_zero_fill": True,
        "basis_records_by_rank": random_basis_digests,
    }


def gene_sufficient_statistics(
    values: Any,
    names: Sequence[str],
    genes: Sequence[str],
) -> dict[str, Any]:
    matrix = _finite_matrix("values", values)
    rendered_names = tuple(map(str, names))
    rendered_genes = tuple(map(str, genes))
    if matrix.shape != (len(rendered_names), len(rendered_genes)):
        raise ValueError("sufficient-statistic dimensions disagree")
    labels = sorted(set(rendered_genes))
    records = []
    gene_array = np.asarray(rendered_genes, dtype=object)
    for gene in labels:
        selected = matrix[:, gene_array == gene]
        records.append(
            {
                "gene_id": gene,
                "n": int(selected.shape[1]),
                "sums": [float(value) for value in selected.sum(axis=1)],
            }
        )
    record = {
        "schema": "gfmcd.r5b.gene_sufficient_statistics/1",
        "row_names": list(rendered_names),
        "row_names_sha256": canonical_sha256(list(rendered_names)),
        "n_examples": matrix.shape[1],
        "n_genes": len(labels),
        "genes": records,
    }
    return record | {"record_sha256": canonical_sha256(record)}


def reconstruct_gene_sufficient_statistics(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, list[str], list[str]]:
    if record.get("schema") != "gfmcd.r5b.gene_sufficient_statistics/1":
        raise ValueError("R5b sufficient-statistic schema differs")
    payload = dict(record)
    stored_digest = payload.pop("record_sha256", None)
    if stored_digest != canonical_sha256(payload):
        raise ValueError("R5b sufficient-statistic record digest differs")
    names = list(map(str, payload.get("row_names", [])))
    if payload.get("row_names_sha256") != canonical_sha256(names):
        raise ValueError("R5b sufficient-statistic row registry digest differs")
    rows: list[np.ndarray] = []
    groups: list[str] = []
    genes = payload.get("genes")
    if not isinstance(genes, list) or len(genes) != payload.get("n_genes"):
        raise ValueError("R5b sufficient-statistic gene registry differs")
    for item in genes:
        gene = str(item.get("gene_id", ""))
        count = item.get("n")
        sums = np.asarray(item.get("sums"), dtype=np.float64)
        if (
            not gene
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            or sums.shape != (len(names),)
            or not np.isfinite(sums).all()
        ):
            raise ValueError("R5b sufficient-statistic gene row is malformed")
        # Exact for R5b because its cohort contract is one row per gene.  Retaining the
        # general n field makes any accidental multi-row gene explicit and fail-closed.
        if count != 1:
            raise ValueError("R5b sufficient statistics require exactly one row per gene")
        rows.append(sums)
        groups.append(gene)
    matrix = np.stack(rows, axis=1)
    if matrix.shape[1] != payload.get("n_examples"):
        raise ValueError("R5b sufficient-statistic example count differs")
    return matrix, names, groups


def analyze_residual_capture(
    capture: ResidualCapture,
    scorer: FullMLMScorer,
    *,
    n_inference_draws: int = N_INFERENCE_DRAWS,
    n_refit_draws: int = N_REFIT_BOOTSTRAP,
    n_overlap_calibration: int = N_OVERLAP_CALIBRATION,
    n_overlap_test: int = N_OVERLAP_TEST,
    seed: int = SEED,
    binding: str,
) -> dict[str, Any]:
    """Run the complete cross-fitted R5b analysis from transient residual captures."""
    capture.validate()
    folds = exact_two_fold_ids(capture.genes)
    for fold in (0, 1):
        if set(np.asarray(capture.strands)[folds == fold].tolist()) != {"+", "-"}:
            raise InstrumentError("each exact 64-gene cross-fit fold must contain both strands")
    delta_partner = capture.h_clean - capture.h_partner_corrupt
    delta_adjacent = (
        capture.h_adjacent_clean_mean - capture.h_adjacent_partner_corrupt_mean
    )
    fitted = crossfit_svd(
        delta_partner,
        capture.genes,
        folds,
        candidate_ranks=RANK_GRID,
        diagnostic_energy_target=DIAGNOSTIC_ENERGY_TARGET,
        group_average=True,
    )
    adjacent_fitted = crossfit_svd(
        delta_adjacent,
        capture.genes,
        folds,
        candidate_ranks=RANK_GRID,
        diagnostic_energy_target=DIAGNOSTIC_ENERGY_TARGET,
        group_average=True,
    )
    for label, fitted_family in (("target", fitted), ("adjacent", adjacent_fitted)):
        if len(fitted_family.folds) != 2:
            raise InstrumentError(f"{label} cross-fit did not produce exactly two folds")
        for fold in fitted_family.folds:
            if len(fold.derivation_groups) != ROWS_PER_FOLD:
                raise InstrumentError(f"{label} SVD did not fit 64 opposite-fold gene means")
            if set(fold.derivation_groups) & set(fold.evaluation_groups):
                raise InstrumentError(f"{label} SVD leaked an evaluation gene")
            for rank in RANK_GRID:
                diagnostics = projector_diagnostics(
                    orthogonal_projector(fold.svd.basis[:rank])
                )
                if (
                    diagnostics["rank"] != rank
                    or diagnostics["symmetry_max_abs"] >= PROJECTOR_ATOL
                    or diagnostics["idempotence_max_abs"] >= PROJECTOR_ATOL
                    or diagnostics["trace_rank_error"] >= PROJECTOR_ATOL
                ):
                    raise InstrumentError(f"{label} projector invariant failed")

    observed_overlap = observed_crossfold_overlap(fitted)
    refit = refit_overlap_intervals(
        delta_partner,
        folds,
        observed_overlap,
        n_draws=n_refit_draws,
        seed=seed,
    )
    overlap = overlap_family_inference(
        observed_overlap,
        refit,
        width=delta_partner.shape[1],
        n_calibration=n_overlap_calibration,
        n_test=n_overlap_test,
        seed=seed,
    )
    primitives, random_controls = intervention_primitives_by_rank(
        capture,
        scorer,
        fitted,
        adjacent_fitted,
        folds,
        seed=seed,
        binding=binding,
    )
    causal: dict[str, Any] = {}
    sufficient_stats: dict[str, Any] = {}
    for rank in RANK_GRID:
        inference = infer_causal_rank_family(
            primitives[rank],
            capture.genes,
            rank=rank,
            n_draws=n_inference_draws,
            seed=seed,
            namespace="r5b:observed-causal-family",
        )
        causal[str(rank)] = inference
        sufficient_stats[str(rank)] = gene_sufficient_statistics(
            primitives[rank], causal_primitive_names(), capture.genes
        )
    overlap_gates = {
        rank: bool(overlap["random_overlap_gate_by_rank"][str(rank)])
        for rank in RANK_GRID
    }
    causal_gates = {
        rank: bool(causal[str(rank)]["random_control_gate"]) for rank in RANK_GRID
    }
    sufficiency = {
        rank: float(np.mean(primitives[rank][0])) for rank in RANK_GRID
    }
    selection = select_crossfit_outcome_rank(
        sufficiency,
        overlap_gates,
        causal_gates,
        rank_grid=RANK_GRID,
        recovery_fraction=RECOVERY_FRACTION,
    )
    selected_rank = selection.selected_rank
    if selected_rank is None:
        verdict = "NO_U_A_RANK"
    elif causal[str(selected_rank)]["random_control_gate"] and overlap_gates[selected_rank]:
        verdict = "ACCEPTOR_SIDE_STATE_SUPPORTED"
    else:  # selection makes this branch unreachable; retain a fail-closed memorial.
        raise InstrumentError("selected rank does not pass both frozen rank gates")
    return {
        "schema": "gfmcd.r5b.acceptor_side_state_analysis/1",
        "endpoint": "single masked acceptor G hard margin",
        "masked_donor_endpoint_constructed": False,
        "n_rows": TARGET_N,
        "n_genes": len(set(capture.genes)),
        "fold_assignment": {
            "policy": (
                "genes ordered by sha256(r5b-u-a-crossfit-v1|gene_id), then alternating 0/1"
            ),
            "fold_ids": [int(value) for value in folds],
            "ordered_fold_ids_sha256": array_sha256(folds, dtype="<i8"),
            "counts": {str(fold): int(np.count_nonzero(folds == fold)) for fold in (0, 1)},
            "both_strands_by_fold": {
                str(fold): sorted(set(np.asarray(capture.strands)[folds == fold].tolist()))
                for fold in (0, 1)
            },
        },
        "capture_digests": capture.digest_record(),
        "full_mlm_capability": capture.capability,
        "target_crossfit": fitted.record(),
        "adjacent_crossfit": adjacent_fitted.record(),
        "random_controls": random_controls,
        "overlap_family": overlap,
        "causal_families_by_rank": causal,
        "primitive_gene_sufficient_statistics_by_rank": sufficient_stats,
        "outcome_rank_selection": selection.record(),
        "verdict": verdict,
        "claim_boundary": (
            "fold-indexed final-acceptor-residual projectors carry a material tested portion; "
            "not completeness, semantics, L3H1 write, downstream delivery, or biology"
        ),
    }


def _selected_token_logits(
    output: Any,
    positions: Sequence[int],
    token_offset: int,
) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise InstrumentError("masked-LM forward did not return [B,T,V] logits")
    rows = torch.arange(len(positions), device=logits.device)
    tokens = torch.as_tensor(
        [int(position) + int(token_offset) for position in positions],
        device=logits.device,
    )
    return logits[rows, tokens].float()


def _selected_final_residuals(
    output: Any,
    positions: Sequence[int],
    token_offset: int,
) -> torch.Tensor:
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states or not isinstance(hidden_states[-1], torch.Tensor):
        raise InstrumentError("masked-LM forward lacks final transformer residuals")
    final = hidden_states[-1]
    rows = torch.arange(len(positions), device=final.device)
    tokens = torch.as_tensor(
        [int(position) + int(token_offset) for position in positions],
        device=final.device,
    )
    return final[rows, tokens]


def _adjacent_mean_residuals(
    output: Any,
    windows: Sequence[Sequence[int]],
    token_offset: int,
) -> torch.Tensor:
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states or not isinstance(hidden_states[-1], torch.Tensor):
        raise InstrumentError("masked-LM forward lacks final transformer residuals")
    final = hidden_states[-1]
    if len(windows) != final.shape[0] or any(len(window) != 4 for window in windows):
        raise ValueError("adjacent windows must contain four positions per row")
    return torch.stack(
        [
            final[row, [int(position) + int(token_offset) for position in window]].mean(
                dim=0
            )
            for row, window in enumerate(windows)
        ],
        dim=0,
    )


def _forward_with_dense_cls_patch(
    model: torch.nn.Module,
    cls: torch.nn.Module,
    encoded: Mapping[str, torch.Tensor],
    positions: Sequence[int],
    token_offset: int,
    replacement: torch.Tensor,
) -> torch.Tensor:
    """Patch whole same-row masked-token residuals at the full MLM-head boundary."""
    called = 0

    def patch(_module: torch.nn.Module, inputs: tuple[Any, ...]):
        nonlocal called
        if len(inputs) != 1 or not isinstance(inputs[0], torch.Tensor):
            raise InstrumentError("unexpected checkpoint cls pre-hook signature")
        hidden = inputs[0]
        if hidden.ndim != 3 or hidden.shape[0] != len(positions):
            raise InstrumentError("dense residual hook received an unaligned hidden tensor")
        updated = hidden.clone()
        rows = torch.arange(len(positions), device=hidden.device)
        tokens = torch.as_tensor(
            [int(position) + int(token_offset) for position in positions],
            device=hidden.device,
        )
        updated[rows, tokens] = replacement.to(hidden.device, hidden.dtype)
        called += 1
        return (updated,)

    handle = cls.register_forward_pre_hook(patch)
    try:
        with torch.no_grad():
            output = model(**encoded)
    finally:
        handle.remove()
    if called != 1:
        raise InstrumentError("dense residual hook did not execute exactly once")
    return _selected_token_logits(output, positions, token_offset)


def capture_model_residuals(
    adapter: Any,
    task: CouplingTask,
    rows: Sequence[R5CohortRow],
    *,
    batch_size: int,
) -> ResidualCapture:
    """Capture the three acceptor-only sequence states and independent dense-hook controls."""
    if len(rows) != TARGET_N:
        raise ValueError("R5b model capture requires the exact 128-row reservation")
    cls = adapter.model.cls
    token_offset = int(adapter.prefix_tokens)
    fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "h_clean",
            "h_partner",
            "h_pseudo",
            "h_adjacent_clean",
            "h_adjacent_partner",
            "m_clean",
            "m_partner",
            "m_pseudo",
        )
    }
    capability_errors: dict[str, list[float]] = {
        name: []
        for name in (
            "direct_clean_full_logits_max_abs",
            "direct_corrupt_full_logits_max_abs",
            "decoder_after_transform_base_logits_max_abs",
            "decoder_after_transform_margin_max_abs",
            "dense_clean_residual_patch_full_logits_max_abs",
            "dense_corrupt_self_patch_full_logits_max_abs",
        )
    }
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        positions = [row.acceptor_g for row in chunk]
        windows = [row.adjacent_window for row in chunk]
        clean_encoded = task.encode([row.clean_sequence for row in chunk], positions)
        partner_encoded = task.encode(
            [row.paired_corrupt_sequence for row in chunk], positions
        )
        pseudo_encoded = task.encode(
            [row.pseudo_corrupt_sequence for row in chunk], positions
        )
        with torch.no_grad():
            clean_output = adapter.model(**clean_encoded, output_hidden_states=True)
            partner_output = adapter.model(**partner_encoded, output_hidden_states=True)
            pseudo_output = adapter.model(**pseudo_encoded, output_hidden_states=True)
        h_clean = _selected_final_residuals(clean_output, positions, token_offset)
        h_partner = _selected_final_residuals(partner_output, positions, token_offset)
        h_pseudo = _selected_final_residuals(pseudo_output, positions, token_offset)
        clean_logits = _selected_token_logits(clean_output, positions, token_offset)
        partner_logits = _selected_token_logits(partner_output, positions, token_offset)
        pseudo_logits = _selected_token_logits(pseudo_output, positions, token_offset)
        dense_clean = _forward_with_dense_cls_patch(
            adapter.model,
            cls,
            partner_encoded,
            positions,
            token_offset,
            h_clean,
        )
        dense_self = _forward_with_dense_cls_patch(
            adapter.model,
            cls,
            partner_encoded,
            positions,
            token_offset,
            h_partner,
        )
        capability = full_mlm_capability(
            cls,
            h_clean=h_clean.detach().float().cpu().numpy(),
            h_partner_corrupt=h_partner.detach().float().cpu().numpy(),
            native_clean_logits=clean_logits.detach().cpu().numpy(),
            native_partner_logits=partner_logits.detach().cpu().numpy(),
            dense_clean_patch_logits=dense_clean.detach().cpu().numpy(),
            dense_self_patch_logits=dense_self.detach().cpu().numpy(),
            base_token_ids=task.bid,
            device=adapter.device,
            batch_size=batch_size,
        )
        for name, value in capability["errors"].items():
            capability_errors[name].append(float(value))
        fields["h_clean"].append(h_clean.detach().float().cpu().numpy())
        fields["h_partner"].append(h_partner.detach().float().cpu().numpy())
        fields["h_pseudo"].append(h_pseudo.detach().float().cpu().numpy())
        fields["h_adjacent_clean"].append(
            _adjacent_mean_residuals(clean_output, windows, token_offset)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        fields["h_adjacent_partner"].append(
            _adjacent_mean_residuals(partner_output, windows, token_offset)
            .detach()
            .float()
            .cpu()
            .numpy()
        )
        fields["m_clean"].append(
            _torch_g_margin(clean_logits, task.bid).detach().cpu().numpy()
        )
        fields["m_partner"].append(
            _torch_g_margin(partner_logits, task.bid).detach().cpu().numpy()
        )
        fields["m_pseudo"].append(
            _torch_g_margin(pseudo_logits, task.bid).detach().cpu().numpy()
        )
    errors = {name: max(values) for name, values in capability_errors.items()}
    capability_record = {
        "schema": "gfmcd.r5b.full_mlm_dense_residual_capability/1",
        "single_masked_acceptor_G_endpoint": True,
        "masked_donor_endpoint_constructed": False,
        "full_mlm_head_used": True,
        "final_residual_to_margin_called_linear": False,
        "decoder_linearity_scope": "only predictions.transform(h) -> decoder logits",
        "same_row_dense_patch": True,
        "candidate_projector_used": False,
        "atol": MLM_ATOL,
        "errors": errors,
        "all_controls_pass": bool(all(error < MLM_ATOL for error in errors.values())),
    }
    return ResidualCapture(
        genes=tuple(str(row.example.gene_id) for row in rows),
        strands=tuple(str(row.example.strand) for row in rows),
        example_ids=tuple(str(row.example.example_id) for row in rows),
        h_clean=np.concatenate(fields["h_clean"], axis=0),
        h_partner_corrupt=np.concatenate(fields["h_partner"], axis=0),
        h_pseudo_corrupt=np.concatenate(fields["h_pseudo"], axis=0),
        h_adjacent_clean_mean=np.concatenate(fields["h_adjacent_clean"], axis=0),
        h_adjacent_partner_corrupt_mean=np.concatenate(
            fields["h_adjacent_partner"], axis=0
        ),
        m_clean=np.concatenate(fields["m_clean"], axis=0),
        m_partner_corrupt=np.concatenate(fields["m_partner"], axis=0),
        m_pseudo_corrupt=np.concatenate(fields["m_pseudo"], axis=0),
        capability=capability_record,
    )


def load_exclusion_registry(path: str | Path) -> tuple[list[str], list[str], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text())
    if record.get("schema") != "gfmcd.r5.exclusion_registry/1":
        raise ValueError("R5 exclusion registry schema differs")
    historical = list(map(str, record.get("historical_r3_gene_ids", [])))
    r3c = list(map(str, record.get("r3c_gene_ids", [])))
    by_bucket = record.get("r3c_gene_ids_by_bucket")
    if (
        historical != sorted(set(historical))
        or r3c != sorted(set(r3c))
        or any(not gene for gene in (*historical, *r3c))
        or not isinstance(by_bucket, Mapping)
        or set(by_bucket) != {"0", "1"}
    ):
        raise ValueError("R5 exclusion gene lists must be sorted, unique, and non-empty")
    normalized_buckets = {
        bucket: list(map(str, by_bucket[bucket])) for bucket in ("0", "1")
    }
    if any(
        genes != sorted(set(genes)) for genes in normalized_buckets.values()
    ) or sorted(set().union(*map(set, normalized_buckets.values()))) != r3c:
        raise ValueError("R5 exclusion bucket registries do not reproduce the R3c union")
    payload = {key: value for key, value in record.items() if key != "record_sha256"}
    if record.get("record_sha256") != canonical_sha256(payload):
        raise ValueError("R5 exclusion registry self-digest differs")
    if (
        record.get("derivation")
        != "authenticated R3c target cohort audit; no free gene-list input"
        or record.get("source_identity") != R5_EXPECTED_SOURCE_IDENTITY
        or record.get("expected_census_identity_sha256")
        != canonical_sha256(R5_EXPECTED_CENSUS_IDENTITY)
    ):
        raise ValueError("R5 exclusion registry derivation/source contract differs")
    for label in ("r3c_artifact", "r3c_manifest"):
        identity = record.get(label, {})
        if (
            not isinstance(identity, Mapping)
            or re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256"))) is None
            or not isinstance(identity.get("size_bytes"), int)
            or identity.get("size_bytes") <= 0
        ):
            raise ValueError(f"R5 exclusion registry {label} identity differs")
    return historical, r3c, {
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
        "record_sha256": record["record_sha256"],
        "r3c_artifact": dict(record["r3c_artifact"]),
        "r3c_manifest": dict(record["r3c_manifest"]),
        "derived_from_authenticated_r3c": True,
    }


def load_shared_r5_reservation(
    *,
    gtf: str,
    fasta: str,
    exclusion_registry: str,
    bucket: int,
) -> tuple[tuple[R5CohortRow, ...], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rebuild the exact shared outcome-blind cohort before loading the model."""
    if bucket not in (0, 1):
        raise ValueError("R5 reservation bucket must be 0 or 1")
    source_identity = validate_expected_r5_source_identity(gtf, fasta)
    historical, r3c, registry_identity = load_exclusion_registry(exclusion_registry)
    examples = load_examples_unscored(
        gtf,
        fasta,
        "val",
        None,
        tag="r5-shared-outcome-blind-census-v2",
        allowed_chromosomes=R5_VALIDATION_CHROMOSOMES,
    )
    pseudo = {
        str(example.example_id): matched_decoy_donor(example) for example in examples
    }
    selected, audit = build_r5_reservations(
        examples,
        pseudo,
        historical_r3_gene_ids=historical,
        r3c_gene_ids=r3c,
        rows_per_bucket=TARGET_N,
        require_capacity=True,
    )
    validate_r5_reservation_audit(audit, rows_per_bucket=TARGET_N)
    census_identity = validate_expected_r5_census_identity(audit)
    return selected[bucket], audit, source_identity, registry_identity


def _easy_causal_primitive_offset() -> np.ndarray:
    lookup = {name: index for index, name in enumerate(causal_primitive_names())}
    offset = np.zeros(len(lookup), dtype=np.float64)
    offset[lookup["target|S|partner"]] = 0.40
    offset[lookup["target|N|partner"]] = 0.40
    # Pseudo and adjacent primitives stay at zero.  Every random primitive stays at zero,
    # so every target-minus-random superiority contrast is +0.40.
    return offset


def causal_boundary_case_specs() -> tuple[dict[str, Any], ...]:
    """Construct all 76 algebraically coherent single-edge boundary faces."""
    primitive_names = causal_primitive_names()
    lookup = {name: index for index, name in enumerate(primitive_names)}
    family_names = causal_family_names()
    family_lookup = {name: index for index, name in enumerate(family_names)}
    requirements = causal_requirement_registry()
    cases: list[dict[str, Any]] = []
    for edge in causal_boundary_edge_registry():
        offset = _easy_causal_primitive_offset()
        row = str(edge["row"])
        edge_name = str(edge["edge"])
        boundary = float(edge["boundary"])
        if row.startswith("target|S|") or row.startswith("target|N|"):
            _, kind, endpoint = row.split("|")
            partner_name = f"target|{kind}|partner"
            pseudo_name = f"target|{kind}|pseudo"
            if endpoint == "partner":
                offset[lookup[partner_name]] = SESOI
                offset[lookup[pseudo_name]] = -SESOI / 2
            elif endpoint == "pseudo":
                offset[lookup[partner_name]] = 0.40
                offset[lookup[pseudo_name]] = boundary
            elif endpoint == "specific":
                offset[lookup[partner_name]] = 1.5 * SESOI
                offset[lookup[pseudo_name]] = SESOI / 2
            else:  # pragma: no cover
                raise RuntimeError(f"unknown target endpoint {endpoint}")
        elif row.startswith("adjacent_basis|"):
            offset[lookup[row]] = boundary
        elif row.startswith("target_minus_random|"):
            _, draw_field, kind, _ = row.split("|")
            target = offset[lookup[f"target|{kind}|partner"]]
            offset[lookup[f"random|{draw_field}|{kind}|partner"]] = target - boundary
        else:  # pragma: no cover
            raise RuntimeError(f"unknown R5b boundary row {row}")
        family, _ = derive_causal_family(offset[:, None])
        observed = family[:, 0]
        if not np.isclose(
            observed[family_lookup[row]], boundary, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"R5b boundary generator failed to place {edge_name}")
        for other_row, requirement in requirements.items():
            value = float(observed[family_lookup[other_row]])
            if other_row == row:
                continue
            if requirement == "MATERIAL_POSITIVE":
                valid = value > SESOI
            elif requirement == "EQUIVALENT_WITHIN_SESOI":
                valid = -SESOI < value < SESOI
            else:
                valid = value > 0.0
            if not valid:
                raise RuntimeError(
                    f"R5b boundary {edge_name} does not keep {other_row} on an easy alternative"
                )
        cases.append(
            {
                "case": edge_name,
                "rank_invariant_edge": dict(edge),
                "primitive_offset": offset.tolist(),
                "primitive_offset_sha256": array_sha256(offset),
            }
        )
    if len(cases) != 76:
        raise RuntimeError("R5b causal calibration requires exactly 76 boundary cases")
    return tuple(cases)


def prospective_calibration_trial_count(
    *,
    alpha_family: float = RANK_ALPHA,
    epsilon: float = CALIBRATION_EPSILON,
    power_floor: float = CALIBRATION_POWER_FLOOR,
    plant_success_rate: float = CALIBRATION_PLANT_SUCCESS_RATE,
    minimum_trials: int | None = None,
) -> dict[str, Any]:
    """Compute the smallest prospective engineering-audit trial count."""
    if not (0 < alpha_family < 1 and 0 < epsilon < 1 and 0 < power_floor < 1):
        raise ValueError("calibration probabilities must lie strictly inside (0,1)")
    if not power_floor < plant_success_rate <= 1:
        raise ValueError("plant_success_rate must exceed the power floor")
    lower_limit = (
        len(RANK_GRID) * len(causal_boundary_case_specs())
        if minimum_trials is None
        else int(minimum_trials)
    )
    if lower_limit < 1:
        raise ValueError("minimum_trials must be positive")
    for trials in range(lower_limit, 200_001):
        false_positives = int(math.floor(trials * alpha_family))
        plant_successes = int(math.floor(trials * plant_success_rate))
        _, false_upper = one_sided_binomial_bounds(false_positives, trials)
        plant_lower, _ = one_sided_binomial_bounds(plant_successes, trials)
        if false_upper <= alpha_family + epsilon and plant_lower >= power_floor:
            return {
                "N_cal": trials,
                "alpha_family": alpha_family,
                "epsilon_cal": epsilon,
                "false_positive_reference_count": false_positives,
                "false_positive_one_sided_95_upper": false_upper,
                "false_positive_engineering_ceiling": alpha_family + epsilon,
                "plant_reference_successes": plant_successes,
                "plant_one_sided_95_lower": plant_lower,
                "plant_power_floor": power_floor,
                "minimum_case_coverage_trials": lower_limit,
                "formula": (
                    "smallest N >= complete rank x edge registry such that CP95-upper("
                    "floor(N*alpha),N)<=alpha+epsilon and CP95-lower("
                    "floor(0.95*N),N)>=0.90"
                ),
                "scientific_fwer_claimed_from_calibration": False,
            }
    raise RuntimeError("prospective R5b calibration trial count is infeasible")


def run_causal_classifier_calibration(
    primitives_by_rank: Mapping[int, np.ndarray],
    genes: Sequence[str],
    *,
    n_draws: int = N_INFERENCE_DRAWS,
    n_trials: int | None = None,
    seed: int = SEED,
    min_genes: int = MIN_GENES,
) -> dict[str, Any]:
    """Exercise the production classifier on every rank/edge boundary and strong plants."""
    if set(primitives_by_rank) != set(RANK_GRID):
        raise ValueError("calibration requires the complete six-rank primitive family")
    plan = prospective_calibration_trial_count()
    trials = int(plan["N_cal"] if n_trials is None else n_trials)
    case_specs = causal_boundary_case_specs()
    complete_cases = [(rank, case) for rank in RANK_GRID for case in case_specs]
    if trials < len(complete_cases):
        raise ValueError("calibration must execute every literal rank/edge case at least once")
    centered = {
        rank: np.asarray(primitives_by_rank[rank], dtype=np.float64)
        - np.asarray(primitives_by_rank[rank], dtype=np.float64).mean(axis=1, keepdims=True)
        for rank in RANK_GRID
    }
    false_pattern: list[int] = []
    plant_pattern: list[int] = []
    null_audit_hashes: list[str] = []
    plant_audit_hashes: list[str] = []
    case_counts: dict[str, dict[str, int]] = {}
    easy = _easy_causal_primitive_offset()
    for trial in range(trials):
        rank, case = complete_cases[trial % len(complete_cases)]
        noise = gene_sign_flip(
            centered[rank],
            genes,
            seed=seed + trial,
            namespace=f"r5b:calibration:outer:rank={rank}:trial={trial}",
        )
        boundary_primitive = noise + np.asarray(case["primitive_offset"])[:, None]
        boundary_result = infer_causal_rank_family(
            boundary_primitive,
            genes,
            rank=rank,
            n_draws=n_draws,
            seed=seed,
            namespace=f"r5b:calibration:null:rank={rank}:trial={trial}",
            min_genes=min_genes,
        )
        planted_result = infer_causal_rank_family(
            noise + easy[:, None],
            genes,
            rank=rank,
            n_draws=n_draws,
            seed=seed,
            namespace=f"r5b:calibration:plant:rank={rank}:trial={trial}",
            min_genes=min_genes,
        )
        false_positive = int(boundary_result["random_control_gate"])
        plant_success = int(planted_result["random_control_gate"])
        false_pattern.append(false_positive)
        plant_pattern.append(plant_success)
        compact_null = {
            "path_iut_p_value": boundary_result["path_iut_p_value"],
            "path_p_pass": boundary_result["path_p_pass"],
            "all_strict_ci_requirements_pass": boundary_result[
                "all_strict_ci_requirements_pass"
            ],
            "random_control_gate": boundary_result["random_control_gate"],
            "primitive_sha256": boundary_result["primitive_sha256"],
        }
        compact_plant = {
            "path_iut_p_value": planted_result["path_iut_p_value"],
            "path_p_pass": planted_result["path_p_pass"],
            "all_strict_ci_requirements_pass": planted_result[
                "all_strict_ci_requirements_pass"
            ],
            "random_control_gate": planted_result["random_control_gate"],
            "primitive_sha256": planted_result["primitive_sha256"],
        }
        null_audit_hashes.append(canonical_sha256(compact_null))
        plant_audit_hashes.append(canonical_sha256(compact_plant))
        stratum = f"rank={rank}|{case['case']}"
        counter = case_counts.setdefault(stratum, {"trials": 0, "false_positives": 0})
        counter["trials"] += 1
        counter["false_positives"] += false_positive
    false_count = int(sum(false_pattern))
    plant_count = int(sum(plant_pattern))
    _, false_upper = one_sided_binomial_bounds(false_count, trials)
    plant_lower, _ = one_sided_binomial_bounds(plant_count, trials)
    return {
        "schema": "gfmcd.r5b.causal_classifier_calibration/1",
        "plan": plan,
        "executed_trials": trials,
        "n_inference_draws": n_draws,
        "rank_edge_case_count": len(complete_cases),
        "rank_edge_case_registry_sha256": canonical_sha256(
            [
                {"rank": rank, "case": case["case"], "offset": case["primitive_offset_sha256"]}
                for rank, case in complete_cases
            ]
        ),
        "every_literal_case_executed": bool(
            set(case_counts)
            == {f"rank={rank}|{case['case']}" for rank, case in complete_cases}
        ),
        "false_positive_trials": false_count,
        "false_positive_pattern_sha256": canonical_sha256(false_pattern),
        "false_positive_one_sided_95_upper": false_upper,
        "false_positive_engineering_ceiling": RANK_ALPHA + CALIBRATION_EPSILON,
        "false_positive_engineering_gate": bool(
            false_upper <= RANK_ALPHA + CALIBRATION_EPSILON
        ),
        "plant_successes": plant_count,
        "plant_pattern_sha256": canonical_sha256(plant_pattern),
        "plant_one_sided_95_lower": plant_lower,
        "plant_power_gate": bool(plant_lower >= CALIBRATION_POWER_FLOOR),
        "per_literal_case_counts": case_counts,
        "null_classifier_audit_digest_sha256": canonical_sha256(null_audit_hashes),
        "plant_classifier_audit_digest_sha256": canonical_sha256(plant_audit_hashes),
        "all_pass": bool(
            false_upper <= RANK_ALPHA + CALIBRATION_EPSILON
            and plant_lower >= CALIBRATION_POWER_FLOOR
            and set(case_counts)
            == {f"rank={rank}|{case['case']}" for rank, case in complete_cases}
        ),
        "interpretation": (
            "engineering audit for gross implementation inflation only; formal nominal "
            "control comes from boundary/IUT inference and the six-rank allocation"
        ),
    }


def selector_registry_audit() -> dict[str, Any]:
    """Exercise every selection branch without model outcomes."""
    all_true = {rank: True for rank in RANK_GRID}
    unique_results: dict[str, Any] = {}
    for rank in RANK_GRID:
        curve = {candidate: 0.50 for candidate in RANK_GRID}
        curve[rank] = 1.0
        for earlier in RANK_GRID:
            if earlier < rank:
                curve[earlier] = 0.89
        overlap = dict(all_true)
        causal = dict(all_true)
        for earlier in RANK_GRID:
            if earlier < rank:
                overlap[earlier] = False
        selected = select_crossfit_outcome_rank(
            curve, overlap, causal, rank_grid=RANK_GRID
        )
        unique_results[str(rank)] = selected.record()
        if selected.selected_rank != rank:
            raise RuntimeError("selector registry failed a unique-rank case")
    at = {rank: 1.0 for rank in RANK_GRID}
    at[1] = RECOVERY_FRACTION
    below = dict(at)
    below[1] = np.nextafter(RECOVERY_FRACTION, -np.inf)
    above = dict(at)
    above[1] = np.nextafter(RECOVERY_FRACTION, np.inf)
    boundary = {
        "at": select_crossfit_outcome_rank(at, all_true, all_true).record(),
        "below": select_crossfit_outcome_rank(below, all_true, all_true).record(),
        "above": select_crossfit_outcome_rank(above, all_true, all_true).record(),
        "non_positive": select_crossfit_outcome_rank(
            {rank: 0.0 for rank in RANK_GRID}, all_true, all_true
        ).record(),
    }
    non_boolean_rejected = False
    incomplete_rejected = False
    try:
        select_crossfit_outcome_rank(
            at,
            {rank: (1 if rank == 1 else True) for rank in RANK_GRID},
            all_true,
        )
    except TypeError:
        non_boolean_rejected = True
    try:
        select_crossfit_outcome_rank(
            {rank: at[rank] for rank in RANK_GRID[:-1]}, all_true, all_true
        )
    except ValueError:
        incomplete_rejected = True
    checks = {
        "each_rank_unique_smallest_eligible": all(
            record["selected_rank"] == int(rank)
            for rank, record in unique_results.items()
        ),
        "touching_ninety_percent_is_eligible": boundary["at"]["selected_rank"] == 1,
        "immediately_below_is_not_eligible": boundary["below"]["selected_rank"] != 1,
        "immediately_above_is_eligible": boundary["above"]["selected_rank"] == 1,
        "non_positive_maximum_returns_no_rank": boundary["non_positive"]["selected_rank"]
        is None,
        "non_boolean_gate_rejected": non_boolean_rejected,
        "incomplete_map_rejected": incomplete_rejected,
    }
    return {
        "unique_rank_cases": unique_results,
        "threshold_cases": boundary,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def overlap_geometry_audit(width: int = 96) -> dict[str, Any]:
    """Model-free exact fixtures for zero, boundary, and strong prefix overlap."""
    if width < 64:
        raise ValueError("overlap geometry audit requires width at least 64")
    left = np.eye(width, dtype=np.float64)[:32]
    orthogonal = np.eye(width, dtype=np.float64)[32:64]
    boundary = math.sqrt(OVERLAP_SESOI) * left + math.sqrt(
        1.0 - OVERLAP_SESOI
    ) * orthogonal
    strong = math.sqrt(0.80) * left + math.sqrt(0.20) * orthogonal
    records: dict[str, Any] = {}
    for rank in RANK_GRID:
        records[str(rank)] = {
            "zero": mean_squared_principal_cosine(left[:rank], orthogonal[:rank]),
            "absolute_boundary": mean_squared_principal_cosine(
                left[:rank], boundary[:rank]
            ),
            "strong": mean_squared_principal_cosine(left[:rank], strong[:rank]),
        }
    checks = {
        "zero_exact": all(np.isclose(row["zero"], 0.0) for row in records.values()),
        "boundary_exact": all(
            np.isclose(row["absolute_boundary"], OVERLAP_SESOI)
            for row in records.values()
        ),
        "touching_boundary_not_sufficient": all(
            not (row["absolute_boundary"] > OVERLAP_SESOI)
            for row in records.values()
        ),
        "strong_above_boundary": all(
            row["strong"] > OVERLAP_SESOI for row in records.values()
        ),
    }
    return {"by_rank": records, "checks": checks, "all_pass": all(checks.values())}


def _overlap_raw_matrix_fixture(
    *,
    width: int,
    squared_overlap: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two 64-gene raw matrices with a fixed population prefix overlap."""
    if width < 64 or not 0.0 <= squared_overlap <= 1.0:
        raise ValueError("overlap fixture requires width>=64 and overlap in [0,1]")
    left_basis = np.eye(width, dtype=np.float64)[:32]
    orthogonal = np.eye(width, dtype=np.float64)[32:64]
    right_basis = math.sqrt(squared_overlap) * left_basis + math.sqrt(
        1.0 - squared_overlap
    ) * orthogonal
    generator = np.random.default_rng(seed)
    q_left, _ = np.linalg.qr(generator.normal(size=(64, 32)), mode="reduced")
    q_right, _ = np.linalg.qr(generator.normal(size=(64, 32)), mode="reduced")
    # Well-separated singular values make every prefix identifiable under whole-gene
    # resampling.  Without gaps, a population prefix has no stable identity and is not a
    # legitimate strong-plant fixture for a prefix-overlap instrument.
    singular = np.geomspace(1_000.0, 1.0, 32, dtype=np.float64)
    left = (q_left * singular[None, :]) @ left_basis
    right = (q_right * singular[None, :]) @ right_basis
    return left, right


def run_overlap_refit_calibration(
    *,
    width: int,
    n_refit_draws: int = N_REFIT_BOOTSTRAP,
    n_calibration: int = N_OVERLAP_CALIBRATION,
    n_test: int = N_OVERLAP_TEST,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run raw-matrix no-share, exact-boundary, and strong overlap fixtures end to end."""
    geometries = {
        "no_shared_subspace": 0.0,
        "absolute_overlap_boundary": OVERLAP_SESOI,
        "strong_shared_subspace": 0.80,
    }
    genes = [f"overlap-audit-gene-{index:03d}" for index in range(TARGET_N)]
    folds = np.asarray([0] * ROWS_PER_FOLD + [1] * ROWS_PER_FOLD, dtype=np.int64)
    calibration_pool = _haar_overlap_pool(
        width,
        rank_grid=RANK_GRID,
        n_draws=n_calibration,
        seed=seed,
        namespace=OVERLAP_CALIBRATION_NAMESPACE,
    )
    test_pool = _haar_overlap_pool(
        width,
        rank_grid=RANK_GRID,
        n_draws=n_test,
        seed=seed,
        namespace=OVERLAP_TEST_NAMESPACE,
    )
    results: dict[str, Any] = {}
    for geometry_index, (name, population_overlap) in enumerate(geometries.items()):
        left, right = _overlap_raw_matrix_fixture(
            width=width,
            squared_overlap=population_overlap,
            seed=seed + 1000 * geometry_index,
        )
        delta = np.vstack((left, right))
        fitted = crossfit_svd(
            delta,
            genes,
            folds,
            candidate_ranks=RANK_GRID,
            diagnostic_energy_target=DIAGNOSTIC_ENERGY_TARGET,
            group_average=True,
        )
        observed = observed_crossfold_overlap(fitted)
        refit = refit_overlap_intervals(
            delta,
            folds,
            observed,
            n_draws=n_refit_draws,
            seed=seed,
            namespace=f"{REFIT_NAMESPACE}:calibration:{name}",
        )
        inference = overlap_family_inference(
            observed,
            refit,
            width=width,
            n_calibration=n_calibration,
            n_test=n_test,
            seed=seed,
            calibration_pool=calibration_pool,
            test_pool=test_pool,
        )
        gates = {
            rank: bool(inference["random_overlap_gate_by_rank"][str(rank)])
            for rank in RANK_GRID
        }
        results[name] = {
            "population_squared_overlap": population_overlap,
            "raw_left_sha256": array_sha256(left),
            "raw_right_sha256": array_sha256(right),
            "raw_combined_sha256": array_sha256(delta),
            "crossfit": fitted.record(),
            "inference": inference,
            "gate_by_rank": {str(rank): value for rank, value in gates.items()},
        }
    zero_pass = not any(
        results["no_shared_subspace"]["gate_by_rank"].values()
    )
    boundary_point_exact = all(
        np.isclose(
            results["absolute_overlap_boundary"]["inference"]["by_rank"][str(rank)][
                "observed_overlap"
            ],
            OVERLAP_SESOI,
            rtol=0.0,
            atol=1e-10,
        )
        for rank in RANK_GRID
    )
    boundary_pass = boundary_point_exact and not any(
        results["absolute_overlap_boundary"]["gate_by_rank"].values()
    )
    strong_pass = all(
        results["strong_shared_subspace"]["gate_by_rank"].values()
    )
    checks = {
        "no_shared_subspace_rejected": zero_pass,
        "absolute_boundary_point_exact": boundary_point_exact,
        "touching_absolute_boundary_not_accepted": boundary_pass,
        "strong_shared_subspace_detected_all_ranks": strong_pass,
        "all_raw_refits_completed": all(
            record["inference"]["refit"]["n_draws"] == n_refit_draws
            for record in results.values()
        ),
        "all_disjoint_haar_pools": all(
            record["inference"]["calibration_and_test_pools_disjoint"]
            for record in results.values()
        ),
    }
    return {
        "schema": "gfmcd.r5b.overlap_refit_calibration/1",
        "width": width,
        "n_refit_draws": n_refit_draws,
        "n_calibration_draws": n_calibration,
        "n_test_draws": n_test,
        "raw_matrix_generator": (
            "dense 64x32 orthonormal gene-score matrices times fixed left/rotated "
            "rank-32 bases; no direct edit of overlap, SVD, CI, or gate"
        ),
        "geometries": results,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def run_model_free_audit() -> dict[str, Any]:
    """Audit registries and exact algebra without reading a model, GTF, FASTA, or bucket."""
    genes = [f"audit-gene-{index:03d}" for index in range(TARGET_N)]
    folds = exact_two_fold_ids(genes)
    rng = np.random.default_rng(20260815)
    delta = rng.normal(size=(TARGET_N, 40))
    fitted = crossfit_svd(
        delta,
        genes,
        folds,
        candidate_ranks=RANK_GRID,
        diagnostic_energy_target=DIAGNOSTIC_ENERGY_TARGET,
        group_average=True,
    )
    projection_checks: list[dict[str, Any]] = []
    for fold in fitted.folds:
        for rank in RANK_GRID:
            diagnostics = projector_diagnostics(
                orthogonal_projector(fold.svd.basis[:rank])
            )
            projection_checks.append(
                {
                    "evaluation_fold": fold.evaluation_fold,
                    "rank": rank,
                    **diagnostics,
                }
            )
    nested = deterministic_nested_random_bases(
        40,
        RANK_GRID,
        seed=SEED,
        namespace="r5b:model-free-audit:nested-haar",
        draw=0,
    )
    nested_exact = all(
        np.array_equal(nested[rank], nested[32][:rank]) for rank in RANK_GRID
    )
    primitive = np.zeros((len(causal_primitive_names()), TARGET_N), dtype=np.float64)
    family, names = derive_causal_family(primitive)
    selector = selector_registry_audit()
    geometry = overlap_geometry_audit()
    plan = prospective_calibration_trial_count()
    checks = {
        "no_model_or_data_access": True,
        "exact_128_unique_genes": len(set(genes)) == TARGET_N,
        "two_exact_64_gene_folds": tuple(np.bincount(folds).tolist()) == (64, 64),
        "opposite_fold_gene_mean_svd": all(
            fold.svd.n_fit_rows == 64
            and fold.derivation_matrix_policy == "gene_mean_rows"
            and not (set(fold.derivation_groups) & set(fold.evaluation_groups))
            for fold in fitted.folds
        ),
        "all_projectors_exact": all(
            item["rank"] == round(item["rank"])
            and item["symmetry_max_abs"] < PROJECTOR_ATOL
            and item["idempotence_max_abs"] < PROJECTOR_ATOL
            and item["trace_rank_error"] < PROJECTOR_ATOL
            for item in projection_checks
        ),
        "nested_haar_prefixes_exact": nested_exact,
        "exact_70_primitive_registry": primitive.shape[0] == 70,
        "exact_136_family_registry": family.shape[0] == 136 and len(names) == 136,
        "exact_76_edge_registry": len(causal_boundary_edge_registry()) == 76,
        "all_76_boundary_offsets_coherent": len(causal_boundary_case_specs()) == 76,
        "selector_registry_pass": selector["all_pass"],
        "overlap_geometry_registry_pass": geometry["all_pass"],
        "prospective_calibration_feasible": plan["N_cal"] >= 456,
        "no_masked_donor_endpoint_constructed": True,
    }
    return {
        "schema": "gfmcd.r5b.model_free_audit/1",
        "status": "DEVELOPMENT_MODEL_FREE_NO_SCIENTIFIC_RESULT",
        "ranks": list(RANK_GRID),
        "cohort_contract": EXPECTED_COHORT_CENSUS,
        "source_contract": R5_EXPECTED_SOURCE_IDENTITY,
        "fold_ids_sha256": array_sha256(folds, dtype="<i8"),
        "crossfit": fitted.record(),
        "projector_checks": projection_checks,
        "nested_haar_max_basis_sha256": array_sha256(nested[32]),
        "causal_registry": {
            "primitive_order": list(causal_primitive_names()),
            "family_order": list(causal_family_names()),
            "edge_registry": list(causal_boundary_edge_registry()),
            "boundary_case_binding_sha256": canonical_sha256(
                [
                    {
                        "case": case["case"],
                        "offset": case["primitive_offset_sha256"],
                    }
                    for case in causal_boundary_case_specs()
                ]
            ),
        },
        "selector_audit": selector,
        "overlap_geometry_audit": geometry,
        "prospective_calibration_plan": plan,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def infer_entry_behavior(
    capture: ResidualCapture,
    *,
    n_draws: int,
    seed: int,
) -> dict[str, Any]:
    partner = capture.m_clean - capture.m_partner_corrupt
    pseudo = capture.m_clean - capture.m_pseudo_corrupt
    family = np.vstack((partner, pseudo, partner - pseudo))
    names = ["partner", "pseudo", "specific"]
    intervals, interval_calibration = simultaneous_family_decision(
        family,
        capture.genes,
        names,
        sesoi=SESOI,
        n_draws=n_draws,
        alpha=STAGE_ALPHA,
        seed=seed,
        namespace="r5b:entry-behavior:simultaneous-envelope",
        min_genes=MIN_GENES,
        studentized=True,
    )
    registry = (
        ("partner|positive_edge", 0, SESOI, "greater"),
        ("pseudo|lower_edge", 1, -SESOI, "greater"),
        ("pseudo|upper_edge", 1, SESOI, "less"),
        ("specific|positive_edge", 2, SESOI, "greater"),
    )
    tests = boundary_sign_flip_tests(
        np.vstack([family[index] for _, index, _, _ in registry]),
        capture.genes,
        [name for name, _, _, _ in registry],
        boundaries=[boundary for _, _, boundary, _ in registry],
        alternatives=[alternative for _, _, _, alternative in registry],
        n_draws=n_draws,
        seed=seed,
        namespace="r5b:entry-behavior:shared-boundary-edges",
        min_genes=MIN_GENES,
    )
    path_p = intersection_union_pvalue([test.p_value for test in tests.values()])
    ci_pass = {
        "partner": _strict_ci_pass(
            intervals["partner"].simultaneous_ci, "MATERIAL_POSITIVE"
        ),
        "pseudo": _strict_ci_pass(
            intervals["pseudo"].simultaneous_ci, "EQUIVALENT_WITHIN_SESOI"
        ),
        "specific": _strict_ci_pass(
            intervals["specific"].simultaneous_ci, "MATERIAL_POSITIVE"
        ),
    }
    supported = bool(path_p <= STAGE_ALPHA and all(ci_pass.values()))
    if supported:
        classification = "ENTRY_BEHAVIOR_REPLICATED"
    elif (
        intervals["pseudo"].simultaneous_ci[0] > SESOI
        or intervals["pseudo"].simultaneous_ci[1] < -SESOI
    ):
        classification = "ENTRY_BEHAVIOR_NONSPECIFIC"
    elif intervals["partner"].simultaneous_ci[1] < SESOI:
        classification = "ENTRY_BEHAVIOR_NOT_REPLICATED"
    else:
        classification = "ENTRY_BEHAVIOR_SCIENTIFICALLY_UNRESOLVED"
    return {
        "schema": "gfmcd.r5b.entry_behavior/1",
        "rows": {
            name: {
                "estimate": intervals[name].estimate,
                "simultaneous_ci": list(intervals[name].simultaneous_ci),
                "strict_ci_pass": ci_pass[name],
            }
            for name in names
        },
        "boundary_tests": {name: test.to_dict() for name, test in tests.items()},
        "path_iut_p_value": path_p,
        "alpha": STAGE_ALPHA,
        "interval_calibration": interval_calibration,
        "supported": supported,
        "classification": classification,
        "gene_sufficient_statistics": gene_sufficient_statistics(
            family, names, capture.genes
        ),
    }


def normalized_runner_record(path: Path = RUNNER) -> dict[str, Any]:
    text = path.read_text()
    normalized = re.sub(
        r"^(FROZEN_(?:PROTOCOL|INSTRUMENT_FINGERPRINT|COHORT_INPUT|COHORT_AUDIT|"
        r"R3C_ARTIFACT|R5A_ARTIFACT|PREFLIGHT_ARTIFACT)_SHA256):[^=]*=.*$",
        r"\1 = <PIN>",
        text,
        flags=re.MULTILINE,
    )
    return {
        "path": str(path),
        "normalization": "replace only literal FROZEN_*_SHA256 assignment values",
        "sha256": hashlib.sha256(normalized.encode()).hexdigest(),
    }


def exact_hash_bundle() -> dict[str, Any]:
    sources = (PROTOCOL, MECHANISM_CORE, STAT_CORE, COHORT_CORE)
    records = {
        str(path.relative_to(REPO)): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sources
    }
    normalized_runner = normalized_runner_record()
    records[str(RUNNER.relative_to(REPO))] = {
        "normalization": normalized_runner["normalization"],
        "normalized_runner_sha256": normalized_runner["sha256"],
    }
    return records


def instrument_binding_payload() -> dict[str, Any]:
    stack = build_r5_stack_binding(repo=REPO)
    return {
        "schema": "gfmcd.r5b.instrument_binding/1",
        "r5_stack_binding": stack,
        "r5_stack_fingerprint_sha256": r5_stack_fingerprint(stack),
        "endpoint": "single masked acceptor G hard margin",
        "masked_donor_endpoint_constructed": False,
        "cohort_source_identity": R5_EXPECTED_SOURCE_IDENTITY,
        "cohort_census_identity": EXPECTED_COHORT_CENSUS,
        "stage_sizes": {"preflight": TARGET_N, "target": TARGET_N},
        "crossfit": {
            "salt": CROSSFIT_SALT,
            "folds": 2,
            "rows_per_fold": ROWS_PER_FOLD,
            "assignment": "hash-order then alternating",
            "fit": "uncentered SVD of opposite-fold gene means only",
        },
        "rank_grid": list(RANK_GRID),
        "outcome_rank_recovery_fraction": RECOVERY_FRACTION,
        "causal_family": {
            "primitive_order": list(causal_primitive_names()),
            "family_order": list(causal_family_names()),
            "requirements": causal_requirement_registry(),
            "edge_registry": list(causal_boundary_edge_registry()),
            "rows_per_rank": 136,
            "alpha_per_rank": RANK_ALPHA,
            "inference_draws": N_INFERENCE_DRAWS,
            "random_controls": N_CAUSAL_CONTROLS,
            "haar_namespace": CAUSAL_HAAR_NAMESPACE,
        },
        "overlap_family": {
            "absolute_sesoi": OVERLAP_SESOI,
            "alpha": OVERLAP_ALPHA,
            "refit_draws": N_REFIT_BOOTSTRAP,
            "calibration_draws": N_OVERLAP_CALIBRATION,
            "test_draws": N_OVERLAP_TEST,
            "calibration_namespace": OVERLAP_CALIBRATION_NAMESPACE,
            "test_namespace": OVERLAP_TEST_NAMESPACE,
        },
        "full_mlm": {
            "atol": MLM_ATOL,
            "direct_full_logits": True,
            "decoder_after_transform": True,
            "dense_same_row_clean_and_self_patch": True,
            "complete_cls_for_every_intervention": True,
        },
        "prospective_calibration_plan": prospective_calibration_trial_count(),
        "exact_source_hashes": exact_hash_bundle(),
    }


def instrument_fingerprint(payload: Mapping[str, Any] | None = None) -> str:
    return canonical_sha256(
        instrument_binding_payload() if payload is None else dict(payload)
    )


def _all_literal_pins(*, target: bool) -> dict[str, str | None]:
    pins = {
        "protocol": FROZEN_PROTOCOL_SHA256,
        "instrument": FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
        "cohort_input": FROZEN_COHORT_INPUT_SHA256,
        "cohort_audit": FROZEN_COHORT_AUDIT_SHA256,
        "r3c_artifact": FROZEN_R3C_ARTIFACT_SHA256,
        "r5a_artifact": FROZEN_R5A_ARTIFACT_SHA256,
    }
    if target:
        pins["preflight_artifact"] = FROZEN_PREFLIGHT_ARTIFACT_SHA256
    return pins


def authorize_model_stage(*, target: bool) -> None:
    """Check literal constants only; safe to call before touching any user path."""
    pins = _all_literal_pins(target=target)
    missing = [name for name, value in pins.items() if value is None]
    malformed = [
        name
        for name, value in pins.items()
        if value is not None and re.fullmatch(r"[0-9a-f]{64}", value) is None
    ]
    if missing or malformed:
        detail = []
        if missing:
            detail.append(f"missing literal pins: {', '.join(missing)}")
        if malformed:
            detail.append(f"malformed literal pins: {', '.join(malformed)}")
        raise TargetSealError(
            "R5b is DRAFT; model execution is hard-blocked before path access ("
            + "; ".join(detail)
            + ")"
        )


def verify_frozen_instrument() -> tuple[dict[str, Any], str]:
    protocol_bytes = PROTOCOL.read_bytes()
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    if protocol_hash != FROZEN_PROTOCOL_SHA256:
        raise TargetSealError("R5b protocol bytes differ from the literal frozen pin")
    first_lines = PROTOCOL.read_text().splitlines()[:8]
    if not any(line.startswith("> **Status: FROZEN") for line in first_lines):
        raise TargetSealError("R5b protocol does not declare FROZEN status")
    payload = instrument_binding_payload()
    fingerprint = instrument_fingerprint(payload)
    if fingerprint != FROZEN_INSTRUMENT_FINGERPRINT_SHA256:
        raise TargetSealError("R5b instrument fingerprint differs from the literal pin")
    return payload, fingerprint


def require_clean_git(repo: Path = REPO) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise TargetSealError("R5b model execution requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise TargetSealError("R5b model execution requires a full git commit identity")
    return {"commit": commit, "dirty": False}


def verify_entry_artifacts(r3c_path: str | Path, r5a_path: str | Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for label, raw_path, expected_hash in (
        ("r3c", r3c_path, FROZEN_R3C_ARTIFACT_SHA256),
        ("r5a", r5a_path, FROZEN_R5A_ARTIFACT_SHA256),
    ):
        path = Path(raw_path).expanduser().resolve()
        manifest_path = path.with_suffix(".manifest.json")
        if (
            not path.is_file()
            or not manifest_path.is_file()
            or sha256_file(path) != expected_hash
        ):
            raise TargetSealError(f"R5b {label} entry artifact differs from its literal pin")
        manifest = json.loads(manifest_path.read_text())
        matching_outputs = [
            output
            for output in manifest.get("outputs", [])
            if isinstance(output, Mapping)
            and Path(str(output.get("path", ""))).expanduser().resolve() == path
            and output.get("sha256") == expected_hash
            and output.get("size_bytes") == path.stat().st_size
        ]
        if len(matching_outputs) != 1 or manifest.get("git_dirty") is not False:
            raise TargetSealError(
                f"R5b {label} entry artifact is not bound by a clean sibling manifest"
            )
        records[label] = json.loads(path.read_text())
        records[f"{label}_manifest_record"] = manifest
        records[f"{label}_identity"] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": expected_hash,
            "manifest_path": str(manifest_path),
            "manifest_size_bytes": manifest_path.stat().st_size,
            "manifest_sha256": sha256_file(manifest_path),
        }
    r3c = records["r3c"]
    r5a = records["r5a"]
    r5a_contract = r5a.get("r5_entry_contract", {})
    current_stack = build_r5_stack_binding(repo=REPO)
    if (
        r3c.get("schema") != R3C_ENTRY_SCHEMA
        or r3c.get("instrument_revision") != R3C_INSTRUMENT_REVISION
        or r3c.get("stage") != "target"
        or r3c.get("status") != "CONFIRMATORY"
        or r3c.get("tier") != "HELD_OUT_TARGET"
        or r3c.get("split") != "val"
        or r3c.get("gene_bucket") != 1
        or r3c.get("verdict")
        != "HELD_OUT_ACCEPTOR_HEAD_CONFIRMATION_COMPLETE"
        or r3c.get("all_gates_pass") is not True
        or r3c.get("sealed_split_opened") is not False
        or r3c.get("protocol_status")
        != "FROZEN_PROTOCOL_AND_INSTRUMENT_VERIFIED"
        or r3c.get("primary_estimand")
        != "raw masked-acceptor-G logit-margin component drop"
    ):
        raise TargetSealError("R5b requires the frozen supported R3c L3H1 entry path")
    try:
        validate_r3c_target_health_contract(r3c)
        validate_r3c_target_manifest_contract(
            r3c, records["r3c_manifest_record"]
        )
    except R5StackBindingError as exc:
        raise TargetSealError(
            "R5b requires the corrected R3c target-health/manifest contract"
        ) from exc
    r3c_gates = r3c.get("preflight_gates", r3c.get("gates", {}))
    if (
        not isinstance(r3c_gates, Mapping)
        or not r3c_gates
        or any(value is not True for value in r3c_gates.values())
    ):
        raise TargetSealError("R5b R3c entry contains a failed instrument gate")
    r3c_construction = r3c.get("construction", {})
    r3c_policy = r3c_construction.get("policy", {})
    if (
        r3c_construction.get("schema")
        != "gfmcd.r3c.acceptor_only_construction/1"
        or r3c_construction.get("all_single_acceptor_G") is not True
        or r3c_policy.get("readout")
        != "mask exactly one G at next_acceptor_pos - 1"
        or r3c_policy.get("paired_operator") != "GT -> AC at donor_pos"
        or r3c_policy.get("donor_readout_constructed") is not False
        or r3c.get("instrument_binding_payload", {}).get("readout")
        != "next_acceptor_pos - 1 only"
    ):
        raise TargetSealError("R5b R3c entry is not acceptor-only/no-donor")
    r3c_sources = r3c.get("hashes", {}).get("sources", {})
    if any(
        r3c_sources.get(key, {}).get("sha256")
        != R5_EXPECTED_SOURCE_IDENTITY[key]["sha256"]
        or r3c_sources.get(key, {}).get("size_bytes")
        != R5_EXPECTED_SOURCE_IDENTITY[key]["size_bytes"]
        for key in ("gtf", "fasta")
    ):
        raise TargetSealError("R5b R3c source identity differs from the shared source")
    r3c_cohort = r3c.get("cohort_audit", {})
    historical_genes = r3c_cohort.get("historical_reconstruction", {}).get(
        "excluded_gene_ids"
    )
    fixed = r3c_cohort.get("current_census", {}).get(
        "fixed_selection_gene_ids_by_bucket"
    )
    if (
        not isinstance(historical_genes, list)
        or historical_genes != sorted(set(map(str, historical_genes)))
        or len(historical_genes)
        != R5_EXPECTED_CENSUS_IDENTITY["historical_r3_gene_count"]
        or canonical_sha256(historical_genes)
        != R5_EXPECTED_CENSUS_IDENTITY["historical_r3_gene_ids_sha256"]
        or not isinstance(fixed, Mapping)
        or set(fixed) != {"0", "1"}
    ):
        raise TargetSealError("R5b R3c cohort exclusion identity differs")
    for bucket in ("0", "1"):
        bucket_genes = list(map(str, fixed[bucket].get("gene_ids", [])))
        if (
            bucket_genes != sorted(set(bucket_genes))
            or fixed[bucket].get("gene_ids_sha256")
            != canonical_sha256(bucket_genes)
        ):
            raise TargetSealError("R5b R3c fixed bucket gene identity differs")
    r3c_genes = sorted(
        set(map(str, fixed["0"].get("gene_ids", [])))
        | set(map(str, fixed["1"].get("gene_ids", [])))
    )
    if (
        len(r3c_genes) != R5_EXPECTED_CENSUS_IDENTITY["r3c_gene_count"]
        or canonical_sha256(r3c_genes)
        != R5_EXPECTED_CENSUS_IDENTITY["r3c_gene_ids_sha256"]
    ):
        raise TargetSealError("R5b R3c fixed cohort gene identity differs")
    if (
        r5a.get("schema") != "gfmcd.r5a.acceptor_address_binding/1"
        or r5a.get("stage") != "target"
        or r5a.get("all_instrument_gates_pass") is not True
        or r5a_contract.get("schema") != "gfmcd.r5.entry_contract/1"
        or r5a_contract.get("instrument_gates_pass") is not True
        or r5a_contract.get("claim_eligible") is not True
        or r5a_contract.get("verdict")
        != "L3H1_DONOR_LOCAL_READER_SUPPORTED"
        or r5a_contract.get("acceptor_G_only") is not True
        or r5a_contract.get("masked_donor_endpoint_constructed") is not False
    ):
        raise TargetSealError("R5b requires the frozen R5a reader-supported entry result")
    r5a_manifest = records["r5a_manifest_record"]
    r5a_manifest_config = r5a_manifest.get("config")
    r5a_git = r5a.get("git")
    if (
        r5a_manifest.get("experiment_id")
        != "splicebert_acceptor_coupling:r5a_acceptor_address_binding"
        or r5a_manifest.get("status") != r5a.get("status")
        or not isinstance(r5a_manifest_config, Mapping)
        or not isinstance(r5a_git, Mapping)
        or r5a_manifest_config.get("stage") != "target"
        or r5a_manifest_config.get("bucket") != 1
        or r5a_manifest_config.get("instrument_fingerprint_sha256")
        != r5a.get("instrument_fingerprint_sha256")
        or r5a_manifest_config.get("r5_stack_fingerprint_sha256")
        != r5a.get("r5_stack_fingerprint_sha256")
        or r5a_manifest_config.get("checkpoint_aggregate_sha256")
        != r5a.get("checkpoint", {}).get("aggregate_sha256")
        or r5a_manifest_config.get("construction_sha256")
        != r5a.get("construction_sha256")
        or r5a_manifest.get("git_commit") != r5a_git.get("commit")
        or r5a_git.get("dirty") is not False
    ):
        raise TargetSealError("R5b R5a sibling manifest context differs")
    try:
        validate_serialized_r5_stack(r5a, current=current_stack)
    except R5StackBindingError as exc:
        raise TargetSealError("R5b R5a joint R5 stack identity differs") from exc
    stack_fingerprint = r5_stack_fingerprint(current_stack)
    if (
        r5a_contract.get("r5_stack_fingerprint_sha256") != stack_fingerprint
        or r5a.get("instrument_binding", {}).get("r5_stack_binding")
        != current_stack
        or r5a.get("instrument_binding", {}).get(
            "r5_stack_fingerprint_sha256"
        )
        != stack_fingerprint
    ):
        raise TargetSealError("R5b R5a handoff is not bound to the joint R5 stack")
    r3c_checkpoint = r3c.get("hashes", {}).get("checkpoint_aggregate_sha256")
    if (
        re.fullmatch(r"[0-9a-f]{64}", str(r3c_checkpoint)) is None
        or r5a_contract.get("checkpoint_sha256") != r3c_checkpoint
    ):
        raise TargetSealError("R5b R3c/R5a entry checkpoint identities differ")
    try:
        r5a_census = validate_expected_r5_census_identity(
            r5a.get("cohort_audit", {})
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TargetSealError("R5b R5a cohort census cannot be validated") from exc
    if (
        r5a.get("checkpoint", {}).get("aggregate_sha256") != r3c_checkpoint
        or r5a.get("source_identity") != R5_EXPECTED_SOURCE_IDENTITY
        or r5a.get("runtime_identity") != current_stack["runtime_identity"]
        or r5a_census != EXPECTED_COHORT_CENSUS
        or r5a.get("bucket") != 1
    ):
        raise TargetSealError("R5b R5a checkpoint/source/cohort/runtime context differs")
    linked_r3c = r5a.get("r3c_entry", {})
    if (
        linked_r3c.get("sha256") != records["r3c_identity"]["sha256"]
        or linked_r3c.get("size_bytes")
        != records["r3c_identity"]["size_bytes"]
        or linked_r3c.get("manifest", {}).get("sha256")
        != records["r3c_identity"]["manifest_sha256"]
        or linked_r3c.get("manifest", {}).get("size_bytes")
        != records["r3c_identity"]["manifest_size_bytes"]
        or r5a_contract.get("upstream_r3c_artifact_sha256")
        != records["r3c_identity"]["sha256"]
        or r5a_contract.get("upstream_r3c_manifest_sha256")
        != records["r3c_identity"]["manifest_sha256"]
    ):
        raise TargetSealError("R5b R5a does not cross-link the exact R3c package")

    registry_package = r5a.get("r5_exclusion_registry", {})
    registry_identity = registry_package.get("identity", {})
    registry_record = registry_package.get("record", {})
    registry_path = Path(str(registry_identity.get("path", ""))).expanduser().resolve()
    registry_outputs = [
        output
        for output in r5a_manifest.get("outputs", [])
        if isinstance(output, Mapping)
        and Path(str(output.get("path", ""))).expanduser().resolve() == registry_path
        and output.get("sha256") == registry_identity.get("sha256")
        and output.get("size_bytes") == registry_identity.get("size_bytes")
    ]
    if (
        registry_package.get("derived_from_authenticated_r3c") is not True
        or not registry_path.is_file()
        or sha256_file(registry_path) != registry_identity.get("sha256")
        or registry_path.stat().st_size != registry_identity.get("size_bytes")
        or len(registry_outputs) != 1
        or registry_record.get("record_sha256")
        != registry_identity.get("record_sha256")
        or registry_record.get("r3c_artifact", {}).get("sha256")
        != records["r3c_identity"]["sha256"]
        or registry_record.get("r3c_manifest", {}).get("sha256")
        != records["r3c_identity"]["manifest_sha256"]
        or r5a_contract.get("exclusion_registry_sha256")
        != registry_identity.get("sha256")
        or r5a_contract.get("exclusion_registry_record_sha256")
        != registry_identity.get("record_sha256")
    ):
        raise TargetSealError("R5b R5a exclusion registry package is not authenticated")
    _, _, loaded_registry_identity = load_exclusion_registry(registry_path)
    if (
        loaded_registry_identity["record_sha256"]
        != registry_identity.get("record_sha256")
        or loaded_registry_identity["r3c_artifact"]["sha256"]
        != records["r3c_identity"]["sha256"]
    ):
        raise TargetSealError("R5b exclusion registry does not replay from R3c")
    return {
        "r3c": records["r3c_identity"],
        "r5a": records["r5a_identity"],
        "checkpoint_sha256": r3c_checkpoint,
        "r5a_cohort_sha256": r5a_contract["cohort_sha256"],
        "r5a_construction_sha256": r5a_contract["construction_sha256"],
        "r5_stack_binding": current_stack,
        "r5_stack_fingerprint_sha256": stack_fingerprint,
        "exclusion_registry": loaded_registry_identity,
        "entry_chain_pass": True,
    }


def _causal_replay_comparable(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"primitive_sha256"}
    }


def validate_analysis_replay(record: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all observed causal classifiers and the selector from sufficient stats."""
    if record.get("schema") != "gfmcd.r5b.acceptor_side_state_analysis/1":
        raise ValueError("R5b analysis schema differs")
    causal = record.get("causal_families_by_rank")
    stats = record.get("primitive_gene_sufficient_statistics_by_rank")
    overlap = record.get("overlap_family", {})
    if not isinstance(causal, Mapping) or not isinstance(stats, Mapping):
        raise ValueError("R5b analysis lacks causal families/sufficient statistics")
    if set(causal) != {str(rank) for rank in RANK_GRID} or set(stats) != set(causal):
        raise ValueError("R5b analysis lacks the complete rank grid")
    sufficiency: dict[int, float] = {}
    causal_gates: dict[int, bool] = {}
    overlap_gates: dict[int, bool] = {}
    replay_hashes: dict[str, str] = {}
    for rank in RANK_GRID:
        primitive, names, genes = reconstruct_gene_sufficient_statistics(stats[str(rank)])
        if tuple(names) != causal_primitive_names():
            raise ValueError("R5b sufficient-statistic primitive order differs")
        stored = causal[str(rank)]
        replay = infer_causal_rank_family(
            primitive,
            genes,
            rank=rank,
            n_draws=int(stored["boundary_randomisation"]["B"]),
            seed=int(stored["boundary_randomisation"]["seed"]),
            namespace="r5b:observed-causal-family",
        )
        if canonical_sha256(_causal_replay_comparable(replay)) != canonical_sha256(
            _causal_replay_comparable(stored)
        ):
            raise ValueError(f"R5b rank {rank} inference does not replay from sufficient stats")
        replay_hashes[str(rank)] = canonical_sha256(replay)
        sufficiency[rank] = float(np.mean(primitive[0]))
        causal_gates[rank] = bool(replay["random_control_gate"])
        overlap_gates[rank] = bool(
            overlap["random_overlap_gate_by_rank"][str(rank)]
        )
    selected = select_crossfit_outcome_rank(
        sufficiency,
        overlap_gates,
        causal_gates,
        rank_grid=RANK_GRID,
        recovery_fraction=RECOVERY_FRACTION,
    )
    if selected.record() != record.get("outcome_rank_selection"):
        raise ValueError("R5b outcome rank selection does not replay")
    fixed = overlap.get("fixed_calibration", {})
    if (
        overlap.get("calibration_and_test_pools_disjoint") is not True
        or fixed.get("calibration_pool_sha256") == fixed.get("test_pool_sha256")
        or fixed.get("n_calibration_draws") != N_OVERLAP_CALIBRATION
        or fixed.get("n_test_draws") != N_OVERLAP_TEST
    ):
        raise ValueError("R5b overlap pool identity/size contract differs")
    return {
        "all_observed_causal_families_replayed": True,
        "selector_replayed": True,
        "replay_sha256_by_rank": replay_hashes,
    }


def verify_preflight_artifact(
    path: str | Path,
    *,
    fingerprint: str,
    binding_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = Path(path).expanduser().resolve()
    manifest_path = artifact.with_suffix(".manifest.json")
    if not artifact.is_file() or not manifest_path.is_file():
        raise TargetSealError("R5b target requires an immutable preflight artifact and manifest")
    artifact_hash = sha256_file(artifact)
    if artifact_hash != FROZEN_PREFLIGHT_ARTIFACT_SHA256:
        raise TargetSealError("R5b preflight artifact differs from its literal reviewed pin")
    manifest = json.loads(manifest_path.read_text())
    matching = [
        output
        for output in manifest.get("outputs", [])
        if Path(output.get("path", "")).resolve() == artifact
    ]
    if (
        len(matching) != 1
        or matching[0].get("sha256") != artifact_hash
        or matching[0].get("size_bytes") != artifact.stat().st_size
        or manifest.get("git_dirty") is not False
    ):
        raise TargetSealError("R5b preflight artifact does not match its sibling manifest")
    record = json.loads(artifact.read_text())
    if (
        record.get("schema") != "gfmcd.r5b.acceptor_side_state/1"
        or record.get("stage") != "preflight"
        or record.get("target_bindable_preflight") is not True
        or record.get("all_instrument_gates_pass") is not True
        or record.get("calibration", {}).get("all_pass") is not True
        or record.get("model_free_audit", {}).get("all_pass") is not True
        or record.get("instrument_fingerprint_sha256") != fingerprint
        or record.get("instrument_binding_payload") != binding_payload
    ):
        raise TargetSealError("R5b preflight did not pass its complete frozen instrument path")
    try:
        validate_serialized_r5_stack(
            record, current=binding_payload["r5_stack_binding"]
        )
    except (KeyError, R5StackBindingError) as exc:
        raise TargetSealError("R5b preflight joint R5 stack differs") from exc
    sources = record.get("r5c_runtime_max_source_by_bucket")
    maps = record.get("r5c_runtime_crossfit_map_by_bucket")
    if (
        not isinstance(sources, Mapping)
        or set(sources) != {"0"}
        or not isinstance(maps, Mapping)
        or set(maps) != {"0"}
    ):
        raise TargetSealError("R5b preflight lacks its exact bucket-0 runtime source")
    source = sources["0"]
    crossfit_map = maps["0"]
    bundle_path = Path(str(source.get("runtime_bundle_path", ""))).expanduser().resolve()
    bundle_outputs = [
        output
        for output in manifest.get("outputs", [])
        if Path(str(output.get("path", ""))).expanduser().resolve() == bundle_path
    ]
    if (
        len(bundle_outputs) != 1
        or bundle_outputs[0].get("sha256") != source.get("runtime_bundle_sha256")
        or bundle_outputs[0].get("size_bytes")
        != source.get("runtime_bundle_size_bytes")
    ):
        raise TargetSealError("R5b preflight runtime source is not bound by its manifest")
    arrays = load_and_validate_runtime_max_source(source)
    try:
        validate_runtime_max_source_binding(
            source,
            arrays,
            crossfit_map,
            bucket=0,
            cohort_audit=record["cohort_audit"],
            checkpoint_sha256=record["checkpoint_identity"]["bundle_sha256"],
            r5_stack_fingerprint_sha256=binding_payload[
                "r5_stack_fingerprint_sha256"
            ],
        )
    except (InstrumentError, KeyError, TypeError, ValueError) as exc:
        raise TargetSealError(
            "R5b preflight runtime source is not bound to bucket 0"
        ) from exc
    analysis = record["analysis"]
    if (
        source.get("ordered_row_ids_sha256")
        != analysis.get("capture_digests", {}).get("ordered_example_ids_sha256")
        or list(map(int, crossfit_map.get("row_fold_ids", [])))
        != list(map(int, analysis.get("fold_assignment", {}).get("fold_ids", [])))
    ):
        raise TargetSealError("R5b preflight runtime source differs from its analysis rows")
    replay = validate_analysis_replay(record["analysis"])
    return record, {
        "artifact": {
            "path": str(artifact),
            "size_bytes": artifact.stat().st_size,
            "sha256": artifact_hash,
        },
        "manifest": {
            "path": str(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
        "replay": replay,
        "runtime_max_source": {
            "bucket": 0,
            "path": str(bundle_path),
            "size_bytes": bundle_path.stat().st_size,
            "sha256": sha256_file(bundle_path),
            "source_payload_sha256": source["source_payload_sha256"],
            "verified": True,
        },
    }


def validate_preflight_context(
    record: Mapping[str, Any],
    *,
    binding_payload: Mapping[str, Any],
    fingerprint: str,
    entry_chain: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    cohort_audit: Mapping[str, Any],
    registry_identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject a bucket-0 preflight from any different method or execution context."""

    expected_bucket0 = cohort_audit["selected_by_bucket"]["0"]
    def artifact_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.get(key)
            for key in (
                "size_bytes",
                "sha256",
                "manifest_size_bytes",
                "manifest_sha256",
            )
        }

    def registry_semantics(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.get(key)
            for key in (
                "size_bytes",
                "sha256",
                "record_sha256",
                "r3c_artifact",
                "r3c_manifest",
                "derived_from_authenticated_r3c",
            )
        }

    expected_entry = {
        "r3c": artifact_semantics(entry_chain["r3c"]),
        "r5a": artifact_semantics(entry_chain["r5a"]),
        "checkpoint_sha256": entry_chain["checkpoint_sha256"],
        "r5a_cohort_sha256": entry_chain["r5a_cohort_sha256"],
        "r5a_construction_sha256": entry_chain["r5a_construction_sha256"],
        "r5_stack_binding": entry_chain["r5_stack_binding"],
        "r5_stack_fingerprint_sha256": entry_chain[
            "r5_stack_fingerprint_sha256"
        ],
        "exclusion_registry": registry_semantics(
            entry_chain["exclusion_registry"]
        ),
        "entry_chain_pass": entry_chain["entry_chain_pass"],
    }
    stored_entry_raw = record.get("entry_chain", {})
    stored_entry = (
        {
            "r3c": artifact_semantics(stored_entry_raw.get("r3c", {})),
            "r5a": artifact_semantics(stored_entry_raw.get("r5a", {})),
            "checkpoint_sha256": stored_entry_raw.get("checkpoint_sha256"),
            "r5a_cohort_sha256": stored_entry_raw.get("r5a_cohort_sha256"),
            "r5a_construction_sha256": stored_entry_raw.get(
                "r5a_construction_sha256"
            ),
            "r5_stack_binding": stored_entry_raw.get("r5_stack_binding"),
            "r5_stack_fingerprint_sha256": stored_entry_raw.get(
                "r5_stack_fingerprint_sha256"
            ),
            "exclusion_registry": registry_semantics(
                stored_entry_raw.get("exclusion_registry", {})
            ),
            "entry_chain_pass": stored_entry_raw.get("entry_chain_pass"),
        }
        if isinstance(stored_entry_raw, Mapping)
        else {}
    )
    stored_checkpoint = record.get("checkpoint_identity", {})
    checkpoint_matches = (
        stored_checkpoint.get("bundle_sha256") == checkpoint.get("bundle_sha256")
        and stored_checkpoint.get("files") == checkpoint.get("files")
        and stored_checkpoint.get("aggregate_schema")
        == checkpoint.get("aggregate_schema")
    )
    checks = {
        "bucket0": record.get("bucket") == 0,
        "method_fingerprint": record.get("instrument_fingerprint_sha256")
        == fingerprint,
        "method_binding": record.get("instrument_binding_payload")
        == binding_payload,
        "joint_stack": record.get("r5_stack_binding")
        == binding_payload["r5_stack_binding"]
        and record.get("r5_stack_fingerprint_sha256")
        == binding_payload["r5_stack_fingerprint_sha256"],
        "entry_chain": stored_entry == expected_entry,
        "source": record.get("source_identity") == source_identity,
        "cohort_census": record.get("cohort_audit") == cohort_audit,
        "cohort_registry": registry_semantics(
            record.get("cohort_registry_identity", {})
        )
        == registry_semantics(registry_identity),
        "bucket0_reservation": record.get("r5_entry_contract", {}).get(
            "cohort_sha256"
        )
        == expected_bucket0["ordered_rows_sha256"]
        and record.get("r5_entry_contract", {}).get("construction_sha256")
        == expected_bucket0["construction_sha256"],
        "checkpoint": checkpoint_matches,
        "runtime": record.get("runtime_identity") == runtime,
        "acceptor_only": record.get("endpoint")
        == "single masked acceptor G hard margin"
        and record.get("masked_donor_endpoint_constructed") is False,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise TargetSealError(f"R5b target/preflight context differs: {failed}")
    return {
        "schema": "gfmcd.r5b.preflight_context/1",
        "all_match": True,
        "checks": checks,
    }


def resolved_model_root(model: str) -> Path:
    path = Path(model).expanduser()
    return path.resolve() if path.is_absolute() else (REPO / path).resolve()


def checkpoint_identity(model: str) -> dict[str, Any]:
    root = resolved_model_root(model)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint directory is missing: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"checkpoint directory contains no files: {root}")
    records = [
        {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "root": str(root),
        "files": records,
        "aggregate_schema": (
            "ordered relative_path/size_bytes/sha256 list (R3c-compatible)"
        ),
        "bundle_sha256": canonical_sha256(records),
    }


def _runtime_max_rank_arrays(
    capture: ResidualCapture,
    *,
    binding: str,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    folds = exact_two_fold_ids(capture.genes)
    fitted = crossfit_svd(
        capture.h_clean - capture.h_partner_corrupt,
        capture.genes,
        folds,
        candidate_ranks=RANK_GRID,
        diagnostic_energy_target=DIAGNOSTIC_ENERGY_TARGET,
        group_average=True,
    )
    by_fold = {int(fold.evaluation_fold): fold for fold in fitted.folds}
    if set(by_fold) != {0, 1}:
        raise InstrumentError("R5b runtime bundle lacks the exact two cross-fit bases")
    maximum_bases = np.stack(
        [by_fold[fold].svd.basis[: RANK_GRID[-1]] for fold in (0, 1)], axis=0
    )
    random_bases = np.empty(
        (
            N_CAUSAL_CONTROLS,
            N_FOLDS,
            RANK_GRID[-1],
            maximum_bases.shape[-1],
        ),
        dtype=np.float64,
    )
    for draw in range(N_CAUSAL_CONTROLS):
        for evaluation_fold in (0, 1):
            namespace = (
                f"{CAUSAL_HAAR_NAMESPACE}|binding={binding}|"
                f"evaluation_fold={evaluation_fold}|fit_from_fold={1-evaluation_fold}"
            )
            random_bases[draw, evaluation_fold] = deterministic_nested_random_bases(
                maximum_bases.shape[-1],
                RANK_GRID,
                seed=seed,
                namespace=namespace,
                draw=draw,
            )[RANK_GRID[-1]]
    row_ids = np.asarray(capture.example_ids, dtype=f"<U{max(map(len, capture.example_ids))}")
    arrays = {
        "max_rank_bases_by_fold": maximum_bases,
        "max_rank_random_bases_by_draw_fold": random_bases,
        "row_fold_ids": folds.astype(np.int64),
        "ordered_row_ids": row_ids,
    }
    map_record = {
        "ordered_row_ids": list(capture.example_ids),
        "ordered_gene_ids": list(capture.genes),
        "row_fold_ids": [int(value) for value in folds],
        "basis_sha256_by_evaluation_fold": {
            str(fold): array_sha256(maximum_bases[fold]) for fold in (0, 1)
        },
        "projector_sha256_by_evaluation_fold_and_rank": {
            str(fold): {
                str(rank): array_sha256(
                    orthogonal_projector(maximum_bases[fold, :rank])
                )
                for rank in RANK_GRID
            }
            for fold in (0, 1)
        },
    }
    return arrays, {
        "crossfit_map": map_record,
        "crossfit_map_sha256": canonical_sha256(map_record),
        "ordered_row_ids_sha256": ordered_text_sha256(capture.example_ids),
        "ordered_gene_ids_sha256": ordered_text_sha256(capture.genes),
    }


def _runtime_array_sha256(name: str, value: np.ndarray) -> str:
    if name == "ordered_row_ids":
        return ordered_text_sha256([str(item) for item in value.tolist()])
    if np.issubdtype(value.dtype, np.integer):
        return array_sha256(value, dtype="<i8")
    return array_sha256(value)


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _validate_runtime_max_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected = {
        "max_rank_bases_by_fold",
        "max_rank_random_bases_by_draw_fold",
        "row_fold_ids",
        "ordered_row_ids",
    }
    if set(arrays) != expected:
        raise InstrumentError("R5b runtime max-rank bundle array registry differs")
    bases = np.asarray(arrays["max_rank_bases_by_fold"], dtype=np.float64)
    random = np.asarray(
        arrays["max_rank_random_bases_by_draw_fold"], dtype=np.float64
    )
    folds = np.asarray(arrays["row_fold_ids"])
    row_ids = np.asarray(arrays["ordered_row_ids"])
    if (
        bases.ndim != 3
        or bases.shape[:2] != (2, 32)
        or random.shape != (32, 2, 32, bases.shape[-1])
        or folds.shape != (128,)
        or tuple(np.bincount(folds.astype(int), minlength=2).tolist()) != (64, 64)
        or row_ids.shape != (128,)
        or len(set(map(str, row_ids.tolist()))) != 128
    ):
        raise InstrumentError("R5b runtime max-rank bundle shapes/identities differ")
    for basis in bases:
        orthogonal_projector(basis)
    for basis in random.reshape(-1, 32, bases.shape[-1]):
        orthogonal_projector(basis)


def _selected_runtime_arrays(
    maximum_arrays: Mapping[str, np.ndarray], selected_rank: int
) -> dict[str, np.ndarray]:
    _validate_runtime_max_arrays(maximum_arrays)
    if selected_rank not in RANK_GRID:
        raise ValueError("R5c runtime bundle requires a selected frozen R5b rank")
    return {
        "selected_rank": np.asarray(selected_rank, dtype=np.int64),
        "selected_bases_by_fold": np.asarray(
            maximum_arrays["max_rank_bases_by_fold"][:, :selected_rank, :],
            dtype=np.float64,
        ),
        "random_bases_by_draw_fold": np.asarray(
            maximum_arrays["max_rank_random_bases_by_draw_fold"][:, :, :selected_rank, :],
            dtype=np.float64,
        ),
        "row_fold_ids": np.asarray(maximum_arrays["row_fold_ids"], dtype=np.int64),
        "ordered_row_ids": np.asarray(maximum_arrays["ordered_row_ids"]),
    }


def runtime_bundle_contract(
    arrays: Mapping[str, np.ndarray],
    bundle_bytes: bytes,
    *,
    bundle_path: Path,
    bucket: int,
    cohort_audit: Mapping[str, Any],
    checkpoint_sha256: str,
    selected_rank: int,
    crossfit_map_sha256: str,
    source_payload_sha256: str,
    source_artifact_sha256: str | None,
    max_rank_source: Mapping[str, Any],
    r5_stack_fingerprint_sha256: str,
) -> dict[str, Any]:
    execution_arrays = (
        "selected_bases_by_fold",
        "random_bases_by_draw_fold",
        "row_fold_ids",
        "ordered_row_ids",
    )
    if set(arrays) != {*execution_arrays, "selected_rank"}:
        raise InstrumentError("R5c selected runtime bundle array registry differs")
    embedded_rank = np.asarray(arrays["selected_rank"])
    selected = np.asarray(arrays["selected_bases_by_fold"], dtype=np.float64)
    random = np.asarray(arrays["random_bases_by_draw_fold"], dtype=np.float64)
    folds = np.asarray(arrays["row_fold_ids"])
    row_ids = np.asarray(arrays["ordered_row_ids"])
    if (
        selected.shape[0] != 2
        or selected.shape[1] != selected_rank
        or embedded_rank.shape != ()
        or int(embedded_rank) != selected_rank
        or random.shape[:3] != (32, 2, selected_rank)
        or random.shape[-1] != selected.shape[-1]
        or folds.shape != (128,)
        or tuple(np.bincount(folds.astype(int), minlength=2).tolist()) != (64, 64)
        or row_ids.shape != (128,)
        or len(set(map(str, row_ids.tolist()))) != 128
    ):
        raise InstrumentError("R5c selected runtime bundle shapes differ")
    for basis in selected:
        orthogonal_projector(basis)
    for basis in random.reshape(-1, selected_rank, selected.shape[-1]):
        orthogonal_projector(basis)
    cohort = cohort_audit["selected_by_bucket"][str(bucket)]
    return {
        "schema": "gfmcd.r5b.r5c_runtime_contract/1",
        "bucket": bucket,
        "cohort_sha256": cohort["ordered_rows_sha256"],
        "construction_sha256": cohort["construction_sha256"],
        "gene_ids_sha256": cohort["gene_ids_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "r5_stack_fingerprint_sha256": r5_stack_fingerprint_sha256,
        "selected_rank": selected_rank,
        "crossfit_map_sha256": crossfit_map_sha256,
        "runtime_bundle_path": str(bundle_path),
        "runtime_bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "runtime_bundle_size_bytes": len(bundle_bytes),
        "runtime_array_sha256": {
            name: _runtime_array_sha256(name, np.asarray(arrays[name]))
            for name in execution_arrays
        },
        "runtime_array_shapes": {
            name: list(np.asarray(arrays[name]).shape) for name in execution_arrays
        },
        "ordered_row_ids_sha256": ordered_text_sha256(map(str, row_ids.tolist())),
        "source_payload_sha256": source_payload_sha256,
        "source_artifact_sha256": source_artifact_sha256,
        "max_rank_source_sha256": canonical_sha256(dict(max_rank_source)),
        "max_rank_runtime_bundle_sha256": max_rank_source[
            "runtime_bundle_sha256"
        ],
        "full_bases_or_activations_committed_to_git": False,
        "runtime_bundle_is_transient_uncommitted": True,
    }


def runtime_max_source_record(
    arrays: Mapping[str, np.ndarray],
    bundle_bytes: bytes,
    *,
    bundle_path: Path,
    bucket: int,
    cohort_audit: Mapping[str, Any],
    checkpoint_sha256: str,
    map_record: Mapping[str, Any],
    r5_stack_fingerprint_sha256: str,
) -> dict[str, Any]:
    _validate_runtime_max_arrays(arrays)
    cohort = cohort_audit["selected_by_bucket"][str(bucket)]
    record = {
        "schema": "gfmcd.r5b.r5c_runtime_max_source/1",
        "bucket": bucket,
        "cohort_sha256": cohort["ordered_rows_sha256"],
        "construction_sha256": cohort["construction_sha256"],
        "gene_ids_sha256": cohort["gene_ids_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "r5_stack_fingerprint_sha256": r5_stack_fingerprint_sha256,
        "stored_rank": 32,
        "crossfit_map_sha256": map_record["crossfit_map_sha256"],
        "ordered_row_ids_sha256": map_record["ordered_row_ids_sha256"],
        "ordered_gene_ids_sha256": map_record["ordered_gene_ids_sha256"],
        "runtime_bundle_path": str(bundle_path),
        "runtime_bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "runtime_bundle_size_bytes": len(bundle_bytes),
        "runtime_array_sha256": {
            name: _runtime_array_sha256(name, np.asarray(value))
            for name, value in arrays.items()
        },
        "runtime_array_shapes": {
            name: list(np.asarray(value).shape) for name, value in arrays.items()
        },
        "runtime_bundle_is_transient_uncommitted": True,
    }
    record["source_payload_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in record.items()
            if key not in {"runtime_bundle_path", "source_payload_sha256"}
        }
    )
    return record


def load_and_validate_runtime_max_source(
    source: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if source.get("schema") != "gfmcd.r5b.r5c_runtime_max_source/1":
        raise TargetSealError("R5b preflight runtime max-source schema differs")
    path = Path(str(source.get("runtime_bundle_path", ""))).expanduser().resolve()
    if (
        not path.is_file()
        or path.stat().st_size != source.get("runtime_bundle_size_bytes")
        or sha256_file(path) != source.get("runtime_bundle_sha256")
    ):
        raise TargetSealError("R5b preflight runtime max-source bytes differ")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    _validate_runtime_max_arrays(arrays)
    observed_hashes = {
        name: _runtime_array_sha256(name, value) for name, value in arrays.items()
    }
    if observed_hashes != source.get("runtime_array_sha256"):
        raise TargetSealError("R5b preflight runtime max-source array hashes differ")
    expected_payload_sha = canonical_sha256(
        {
            key: value
            for key, value in source.items()
            if key not in {"runtime_bundle_path", "source_payload_sha256"}
        }
    )
    if source.get("source_payload_sha256") != expected_payload_sha:
        raise TargetSealError("R5b preflight runtime max-source payload digest differs")
    return arrays


def validate_runtime_max_source_binding(
    source: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    map_record: Mapping[str, Any],
    *,
    bucket: int,
    cohort_audit: Mapping[str, Any],
    checkpoint_sha256: str,
    r5_stack_fingerprint_sha256: str | None = None,
) -> None:
    """Bind a max-rank sibling bundle to one exact reservation and fold map."""
    _validate_runtime_max_arrays(arrays)
    genes = [str(value) for value in map_record["ordered_gene_ids"]]
    expected_folds = exact_two_fold_ids(genes)
    cohort = cohort_audit["selected_by_bucket"][str(bucket)]
    expected = {
        "bucket": bucket,
        "cohort_sha256": cohort["ordered_rows_sha256"],
        "construction_sha256": cohort["construction_sha256"],
        "gene_ids_sha256": cohort["gene_ids_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "stored_rank": RANK_GRID[-1],
        "crossfit_map_sha256": canonical_sha256(dict(map_record)),
        "ordered_row_ids_sha256": ordered_text_sha256(
            [str(value) for value in arrays["ordered_row_ids"].tolist()]
        ),
        "ordered_gene_ids_sha256": ordered_text_sha256(
            [str(value) for value in map_record["ordered_gene_ids"]]
        ),
    }
    if r5_stack_fingerprint_sha256 is not None:
        expected["r5_stack_fingerprint_sha256"] = r5_stack_fingerprint_sha256
    if any(source.get(key) != value for key, value in expected.items()):
        raise InstrumentError("R5b runtime max-source reservation binding differs")
    if (
        list(map(str, arrays["ordered_row_ids"].tolist()))
        != list(map(str, map_record["ordered_row_ids"]))
        or [int(value) for value in arrays["row_fold_ids"]]
        != [int(value) for value in map_record["row_fold_ids"]]
        or not np.array_equal(
            np.asarray(arrays["row_fold_ids"], dtype=np.int64), expected_folds
        )
    ):
        raise InstrumentError("R5b runtime bundle rows/folds differ from its cross-fit map")
    bases = np.asarray(arrays["max_rank_bases_by_fold"], dtype=np.float64)
    observed_basis_hashes = {
        str(fold): array_sha256(bases[fold]) for fold in (0, 1)
    }
    if observed_basis_hashes != map_record.get(
        "basis_sha256_by_evaluation_fold"
    ):
        raise InstrumentError("R5b runtime max bases differ from its cross-fit map")
    observed_projectors = {
        str(fold): {
            str(rank): array_sha256(orthogonal_projector(bases[fold, :rank]))
            for rank in RANK_GRID
        }
        for fold in (0, 1)
    }
    if observed_projectors != map_record.get(
        "projector_sha256_by_evaluation_fold_and_rank"
    ):
        raise InstrumentError("R5b runtime projectors differ from its cross-fit map")


def runtime_identity(device: str | None = None) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ("numpy", "scipy", "torch", "transformers", "tokenizers"):
        try:
            packages[package] = package_version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": device,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def snapshot_input_paths(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        snapshot[str(path)] = {
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        }
    return snapshot


def verify_input_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    for raw, expected in snapshot.items():
        path = Path(raw)
        observed = {
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256_file(path),
        }
        if observed != dict(expected):
            raise InstrumentError(f"R5b input changed during execution: {path}")


def atomic_write_new(path: str | Path, data: bytes) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=f".{secrets.token_hex(8)}.tmp",
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite immutable artifact {destination}")
        os.link(temporary, destination)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return destination


def _write_result_and_manifest(
    result: Mapping[str, Any],
    output: Path,
    *,
    stage: str,
    input_paths: Sequence[str | Path],
    git_identity: Mapping[str, Any] | None,
    extra_outputs: Mapping[Path, bytes] | None = None,
    input_snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[Path, Path]:
    manifest_path = output.with_suffix(".manifest.json")
    extras = {
        Path(path).expanduser().resolve(): bytes(payload)
        for path, payload in (extra_outputs or {}).items()
    }
    destinations = {output.resolve(), manifest_path.resolve(), *extras}
    if len(destinations) != 2 + len(extras):
        raise ValueError("R5b result, manifest, and runtime bundle paths must be distinct")
    if any(path.exists() for path in destinations):
        raise FileExistsError("R5b refuses to overwrite an artifact or sibling manifest")
    result_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    result_binding = result.get("instrument_binding_payload", {})
    stack_fingerprint = result.get(
        "r5_stack_fingerprint_sha256",
        result_binding.get("r5_stack_fingerprint_sha256"),
    )
    manifest = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r5b_acceptor_side_state",
        sys.argv,
        repo=REPO,
        status="CONFIRMATORY" if stage == "target" else "DEVELOPMENT",
        config={
            "stage": stage,
            "n": TARGET_N if stage != "audit" else 0,
            "rank_grid": list(RANK_GRID),
            "inference_draws": N_INFERENCE_DRAWS,
            "refit_draws": N_REFIT_BOOTSTRAP,
            "overlap_calibration_draws": N_OVERLAP_CALIBRATION,
            "overlap_test_draws": N_OVERLAP_TEST,
            "causal_controls": N_CAUSAL_CONTROLS,
            "seed": SEED,
            "instrument_fingerprint_sha256": result.get(
                "instrument_fingerprint_sha256"
            ),
            "r5_stack_fingerprint_sha256": stack_fingerprint,
        },
        seeds={"production": SEED},
        notes=[
            "Every scientific outcome is the single masked-acceptor G margin.",
            "No masked-donor endpoint is constructed.",
            "Full activation captures are not serialized; compact statistics and "
            "digests are recorded.",
            "Selected downstream basis bundles are transient, uncommitted, and "
            "individually manifest-bound.",
        ],
    )
    manifest.inputs = []
    for raw_path in input_paths:
        path = Path(raw_path).expanduser().resolve()
        if input_snapshot is None:
            identity = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        else:
            if str(path) not in input_snapshot:
                raise InstrumentError(f"R5b manifest input lacks its final snapshot: {path}")
            identity = input_snapshot[str(path)]
        manifest.inputs.append(
            {
                "path": str(path),
                "size_bytes": int(identity["size_bytes"]),
                "sha256": str(identity["sha256"]),
            }
        )
    manifest.outputs = [
        {
            "path": str(output),
            "size_bytes": len(result_bytes),
            "sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
    ] + [
        {
            "path": str(path),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload in sorted(extras.items(), key=lambda item: str(item[0]))
    ]
    if git_identity is not None:
        manifest.git_commit = str(git_identity["commit"])
        manifest.git_dirty = False
    manifest_bytes = (json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    atomic_write_new(output, result_bytes)
    try:
        for path, payload in sorted(extras.items(), key=lambda item: str(item[0])):
            atomic_write_new(path, payload)
        atomic_write_new(manifest_path, manifest_bytes)
    except Exception:
        # The result remains immutable and recoverable; never remove it silently.  Surface
        # the incomplete pair so a human can repair provenance explicitly.
        raise
    return output, manifest_path


def _validate_bound_cohort(
    audit: Mapping[str, Any], registry_identity: Mapping[str, Any]
) -> None:
    if registry_identity.get("sha256") != FROZEN_COHORT_INPUT_SHA256:
        raise TargetSealError("R5b exclusion registry differs from the literal cohort-input pin")
    if canonical_sha256(dict(audit)) != FROZEN_COHORT_AUDIT_SHA256:
        raise TargetSealError("R5b reconstructed cohort audit differs from its literal pin")


def _final_scientific_verdict(
    analysis: Mapping[str, Any], entry_behavior: Mapping[str, Any]
) -> str:
    if entry_behavior.get("supported") is not True:
        return str(entry_behavior.get("classification", "ENTRY_BEHAVIOR_UNRESOLVED"))
    return str(analysis["verdict"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("audit", "preflight", "target"), default="audit"
    )
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--exclusion-registry", default=None)
    parser.add_argument("--r3c-artifact", default=None)
    parser.add_argument("--r5a-artifact", default=None)
    parser.add_argument("--preflight-artifact", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=PRODUCTION_BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-inference-draws", type=int, default=None)
    parser.add_argument("--n-refit-draws", type=int, default=None)
    parser.add_argument("--n-overlap-calibration", type=int, default=None)
    parser.add_argument("--n-overlap-test", type=int, default=None)
    parser.add_argument("--n-calibration-trials", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # This branch is deliberately first.  Target/preflight cannot stat, hash, or parse any
    # supplied artifact/model/data path while the protocol is unsealed.
    if args.stage in {"preflight", "target"}:
        try:
            authorize_model_stage(target=args.stage == "target")
        except TargetSealError as exc:
            raise SystemExit(str(exc)) from exc

    if args.stage == "audit":
        audit = run_model_free_audit()
        audit_binding = instrument_binding_payload()
        result = {
            "schema": "gfmcd.r5b.acceptor_side_state/1",
            "stage": "audit",
            "status": "DEVELOPMENT",
            "protocol_status": "DRAFT_NOT_FROZEN",
            "target_bindable_preflight": False,
            "masked_donor_endpoint_constructed": False,
            "model_free_audit": audit,
            "instrument_binding_payload": audit_binding,
            "instrument_fingerprint_sha256": instrument_fingerprint(
                audit_binding
            ),
            "r5_stack_binding": audit_binding["r5_stack_binding"],
            "r5_stack_fingerprint_sha256": audit_binding[
                "r5_stack_fingerprint_sha256"
            ],
            "verdict": "MODEL_FREE_AUDIT_PASS" if audit["all_pass"] else "INSTRUMENT_UNRESOLVED",
        }
        if args.out is None:
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        output = Path(args.out).expanduser().resolve()
        _write_result_and_manifest(
            result,
            output,
            stage="audit",
            input_paths=(RUNNER, PROTOCOL, MECHANISM_CORE, STAT_CORE, COHORT_CORE),
            git_identity=None,
        )
        print(f"R5b model-free audit saved to {output}")
        return

    production_defaults = {
        "batch": (args.batch, PRODUCTION_BATCH),
        "seed": (args.seed, SEED),
        "n_inference_draws": (args.n_inference_draws, N_INFERENCE_DRAWS),
        "n_refit_draws": (args.n_refit_draws, N_REFIT_BOOTSTRAP),
        "n_overlap_calibration": (
            args.n_overlap_calibration,
            N_OVERLAP_CALIBRATION,
        ),
        "n_overlap_test": (args.n_overlap_test, N_OVERLAP_TEST),
        "n_calibration_trials": (
            args.n_calibration_trials,
            prospective_calibration_trial_count()["N_cal"],
        ),
    }
    changed = [
        name
        for name, (value, expected) in production_defaults.items()
        if value not in (None, expected)
    ]
    if changed:
        raise SystemExit(f"R5b model stages forbid production overrides: {', '.join(changed)}")
    if args.batch < 1:
        raise SystemExit("R5b --batch must be positive")
    required_paths = {
        "r3c_artifact": args.r3c_artifact,
        "r5a_artifact": args.r5a_artifact,
        "out": args.out,
    }
    if args.stage == "target":
        required_paths["preflight_artifact"] = args.preflight_artifact
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        raise SystemExit(f"R5b {args.stage} requires: {', '.join(missing)}")
    output = Path(args.out).expanduser().resolve()
    manifest_path = output.with_suffix(".manifest.json")
    runtime_max_path = output.with_suffix(".runtime-max.npz")
    r5c_bundle_paths = {
        bucket: output.with_suffix(f".r5c-bucket{bucket}.npz") for bucket in (0, 1)
    }
    possible_outputs = {output, manifest_path, runtime_max_path}
    if args.stage == "target":
        possible_outputs.update(r5c_bundle_paths.values())
    if any(path.exists() for path in possible_outputs):
        raise SystemExit("R5b requires fresh result, manifest, and runtime-bundle paths")

    try:
        binding_payload, fingerprint = verify_frozen_instrument()
        git_identity = require_clean_git()
        entry_chain = verify_entry_artifacts(args.r3c_artifact, args.r5a_artifact)
        resolved_exclusion_registry = entry_chain["exclusion_registry"]["path"]
        if (
            args.exclusion_registry is not None
            and Path(args.exclusion_registry).expanduser().resolve()
            != Path(resolved_exclusion_registry).expanduser().resolve()
        ):
            raise TargetSealError(
                "R5b forbids a free exclusion registry; use the authenticated R5a output"
            )
        if args.stage == "target":
            preflight_record, preflight_package = verify_preflight_artifact(
                args.preflight_artifact,
                fingerprint=fingerprint,
                binding_payload=binding_payload,
            )
        else:
            preflight_record = None
            preflight_package = None
    except (
        InstrumentError,
        TargetSealError,
        ValueError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        raise SystemExit(f"R5b {args.stage} authorization failed: {exc}") from exc

    input_paths: list[str | Path] = [
        RUNNER,
        PROTOCOL,
        MECHANISM_CORE,
        STAT_CORE,
        COHORT_CORE,
        args.gtf,
        args.fasta,
        resolved_exclusion_registry,
        args.r3c_artifact,
        Path(args.r3c_artifact).with_suffix(".manifest.json"),
        args.r5a_artifact,
        Path(args.r5a_artifact).with_suffix(".manifest.json"),
    ]
    stack_binding = binding_payload["r5_stack_binding"]
    input_paths.extend(
        REPO / relative
        for relative in (
            *stack_binding["local_python_closure"]["files"],
            *stack_binding["protocols"],
        )
    )
    if args.preflight_artifact:
        input_paths.extend(
            (
                args.preflight_artifact,
                Path(args.preflight_artifact).with_suffix(".manifest.json"),
            )
        )
    if preflight_record is not None:
        input_paths.append(
            preflight_record["r5c_runtime_max_source_by_bucket"]["0"][
                "runtime_bundle_path"
            ]
        )
    input_paths = list(
        dict.fromkeys(
            str(Path(path).expanduser().resolve()) for path in input_paths
        )
    )
    snapshot = snapshot_input_paths(input_paths)
    rows, cohort_audit, source_identity, registry_identity = load_shared_r5_reservation(
        gtf=args.gtf,
        fasta=args.fasta,
        exclusion_registry=resolved_exclusion_registry,
        bucket=0 if args.stage == "preflight" else 1,
    )
    _validate_bound_cohort(cohort_audit, registry_identity)
    mechanism_target_identity = cohort_audit["selected_by_bucket"]["1"]
    if (
        entry_chain["r5a_cohort_sha256"]
        != mechanism_target_identity["ordered_rows_sha256"]
        or entry_chain["r5a_construction_sha256"]
        != mechanism_target_identity["construction_sha256"]
    ):
        raise SystemExit("R5b R5a entry is not bound to the shared bucket-1 reservation")
    checkpoint_before = checkpoint_identity(args.model)
    if checkpoint_before["bundle_sha256"] != entry_chain["checkpoint_sha256"]:
        raise SystemExit("R5b supplied checkpoint differs from the R3c/R5a entry chain")
    device = pick_device(args.device)
    current_runtime_identity = runtime_identity(device)
    preflight_context = None
    if preflight_record is not None:
        preflight_context = validate_preflight_context(
            preflight_record,
            binding_payload=binding_payload,
            fingerprint=fingerprint,
            entry_chain=entry_chain,
            source_identity=source_identity,
            cohort_audit=cohort_audit,
            registry_identity=registry_identity,
            checkpoint=checkpoint_before,
            runtime=current_runtime_identity,
        )
        assert preflight_package is not None
        preflight_package["context"] = preflight_context
    adapter = load_adapter(args.model, device, eager=False)
    task = CouplingTask(adapter, batch=args.batch)
    capture = capture_model_residuals(adapter, task, rows, batch_size=args.batch)
    scorer = FullMLMScorer(
        adapter.model.cls,
        task.bid,
        device=adapter.device,
        batch_size=args.batch,
    )
    analysis = analyze_residual_capture(
        capture,
        scorer,
        n_inference_draws=N_INFERENCE_DRAWS,
        n_refit_draws=N_REFIT_BOOTSTRAP,
        n_overlap_calibration=N_OVERLAP_CALIBRATION,
        n_overlap_test=N_OVERLAP_TEST,
        seed=SEED,
        binding=fingerprint,
    )
    entry_behavior = infer_entry_behavior(
        capture, n_draws=N_INFERENCE_DRAWS, seed=SEED
    )
    model_free_audit = run_model_free_audit()
    if args.stage == "preflight":
        primitives = {
            rank: reconstruct_gene_sufficient_statistics(
                analysis["primitive_gene_sufficient_statistics_by_rank"][str(rank)]
            )[0]
            for rank in RANK_GRID
        }
        causal_calibration = run_causal_classifier_calibration(
            primitives,
            sorted(capture.genes),
            n_draws=N_INFERENCE_DRAWS,
            n_trials=args.n_calibration_trials,
            seed=SEED,
        )
        overlap_calibration = run_overlap_refit_calibration(
            width=capture.h_clean.shape[1],
            n_refit_draws=N_REFIT_BOOTSTRAP,
            n_calibration=N_OVERLAP_CALIBRATION,
            n_test=N_OVERLAP_TEST,
            seed=SEED,
        )
        selector_calibration = selector_registry_audit()
        calibration = {
            "schema": "gfmcd.r5b.complete_preflight_calibration/1",
            "causal_boundary_registry": causal_calibration,
            "overlap_refit_registry": overlap_calibration,
            "selector_registry": selector_calibration,
            "all_pass": bool(
                causal_calibration["all_pass"]
                and overlap_calibration["all_pass"]
                and selector_calibration["all_pass"]
            ),
        }
        calibration_origin = "RUN_ON_CURRENT_BUCKET0"
    else:
        assert preflight_record is not None
        calibration = preflight_record["calibration"]
        calibration_origin = "INHERITED_FROM_LITERAL_BUCKET0_PREFLIGHT"

    checkpoint_after = checkpoint_identity(args.model)
    if checkpoint_before != checkpoint_after:
        raise SystemExit("R5b checkpoint changed during execution")
    bucket = 0 if args.stage == "preflight" else 1
    runtime_max_arrays, runtime_map_package = _runtime_max_rank_arrays(
        capture,
        binding=fingerprint,
        seed=SEED,
    )
    runtime_max_bytes = _npz_bytes(runtime_max_arrays)
    runtime_max_source = runtime_max_source_record(
        runtime_max_arrays,
        runtime_max_bytes,
        bundle_path=runtime_max_path,
        bucket=bucket,
        cohort_audit=cohort_audit,
        checkpoint_sha256=checkpoint_before["bundle_sha256"],
        map_record=runtime_map_package,
        r5_stack_fingerprint_sha256=binding_payload[
            "r5_stack_fingerprint_sha256"
        ],
    )
    validate_runtime_max_source_binding(
        runtime_max_source,
        runtime_max_arrays,
        runtime_map_package["crossfit_map"],
        bucket=bucket,
        cohort_audit=cohort_audit,
        checkpoint_sha256=checkpoint_before["bundle_sha256"],
        r5_stack_fingerprint_sha256=binding_payload[
            "r5_stack_fingerprint_sha256"
        ],
    )
    if (
        runtime_max_source["ordered_row_ids_sha256"]
        != analysis["capture_digests"]["ordered_example_ids_sha256"]
        or runtime_map_package["crossfit_map"]["row_fold_ids"]
        != analysis["fold_assignment"]["fold_ids"]
    ):
        raise SystemExit("R5b runtime source differs from the analyzed rows/folds")
    instrument_gates = {
        "joint_r3c_r5a_entry_chain": bool(entry_chain["entry_chain_pass"]),
        "joint_r5_stack_bound": entry_chain["r5_stack_binding"]
        == binding_payload["r5_stack_binding"]
        and entry_chain["r5_stack_fingerprint_sha256"]
        == binding_payload["r5_stack_fingerprint_sha256"],
        "authenticated_r3c_derived_exclusion_registry": registry_identity.get(
            "derived_from_authenticated_r3c"
        )
        is True,
        "exact_shared_cohort_source": source_identity == R5_EXPECTED_SOURCE_IDENTITY,
        "exact_shared_cohort_census": validate_expected_r5_census_identity(cohort_audit)
        == EXPECTED_COHORT_CENSUS,
        "literal_cohort_input_pin": registry_identity["sha256"]
        == FROZEN_COHORT_INPUT_SHA256,
        "literal_cohort_audit_pin": canonical_sha256(cohort_audit)
        == FROZEN_COHORT_AUDIT_SHA256,
        "exact_128_one_row_per_gene": len(rows) == TARGET_N
        and len({row.example.gene_id for row in rows}) == TARGET_N,
        "both_strands": {row.example.strand for row in rows} == {"+", "-"},
        "no_chr8_chr9": cohort_audit["sealed_chr8_chr9_opened"] is False,
        "single_acceptor_G_only": analysis["masked_donor_endpoint_constructed"] is False,
        "full_mlm_dense_residual_capability": capture.capability["all_controls_pass"],
        "model_free_registry_audit": model_free_audit["all_pass"],
        "causal_classifier_calibration": calibration["all_pass"],
        "checkpoint_immutable_during_run": checkpoint_before == checkpoint_after,
        "input_snapshot_immutable_during_run": True,
        "runtime_max_source_bound_to_reservation": True,
        "target_matches_bucket0_preflight_context": bool(
            args.stage == "preflight"
            or (preflight_context is not None and preflight_context["all_match"])
        ),
    }
    all_instrument_gates_pass = all(instrument_gates.values())
    target_bindable = bool(args.stage == "preflight" and all_instrument_gates_pass)
    scientific_verdict = _final_scientific_verdict(analysis, entry_behavior)
    verdict = (
        scientific_verdict
        if all_instrument_gates_pass
        else "INSTRUMENT_UNRESOLVED"
    )
    runtime_max_sources_by_bucket = {str(bucket): runtime_max_source}
    runtime_crossfit_maps_by_bucket = {
        str(bucket): runtime_map_package["crossfit_map"]
    }
    if args.stage == "target":
        assert preflight_record is not None
        runtime_max_sources_by_bucket["0"] = preflight_record[
            "r5c_runtime_max_source_by_bucket"
        ]["0"]
        runtime_crossfit_maps_by_bucket["0"] = preflight_record[
            "r5c_runtime_crossfit_map_by_bucket"
        ]["0"]
    runtime_contracts_by_bucket: dict[str, Any] = {}
    extra_outputs: dict[Path, bytes] = {runtime_max_path: runtime_max_bytes}
    selected_rank_raw = analysis["outcome_rank_selection"]["selected_rank"]
    selected_rank = None if selected_rank_raw is None else int(selected_rank_raw)
    if args.stage == "target" and verdict == "ACCEPTOR_SIDE_STATE_SUPPORTED":
        if selected_rank not in RANK_GRID:
            raise SystemExit("R5b supported target lacks a frozen selected rank")
        assert preflight_record is not None
        preflight_source = preflight_record["r5c_runtime_max_source_by_bucket"]["0"]
        preflight_map = preflight_record["r5c_runtime_crossfit_map_by_bucket"]["0"]
        preflight_max_arrays = load_and_validate_runtime_max_source(preflight_source)
        validate_runtime_max_source_binding(
            preflight_source,
            preflight_max_arrays,
            preflight_map,
            bucket=0,
            cohort_audit=cohort_audit,
            checkpoint_sha256=checkpoint_before["bundle_sha256"],
            r5_stack_fingerprint_sha256=binding_payload[
                "r5_stack_fingerprint_sha256"
            ],
        )
        selected_sources = {
            0: (preflight_source, preflight_map, preflight_max_arrays),
            1: (
                runtime_max_source,
                runtime_map_package["crossfit_map"],
                runtime_max_arrays,
            ),
        }
        for contract_bucket, (source, crossfit_map, maximum_arrays) in (
            selected_sources.items()
        ):
            selected_arrays = _selected_runtime_arrays(maximum_arrays, selected_rank)
            selected_bytes = _npz_bytes(selected_arrays)
            bundle_path = r5c_bundle_paths[contract_bucket]
            runtime_contracts_by_bucket[str(contract_bucket)] = runtime_bundle_contract(
                selected_arrays,
                selected_bytes,
                bundle_path=bundle_path,
                bucket=contract_bucket,
                cohort_audit=cohort_audit,
                checkpoint_sha256=checkpoint_before["bundle_sha256"],
                selected_rank=selected_rank,
                crossfit_map_sha256=canonical_sha256(dict(crossfit_map)),
                source_payload_sha256=source["source_payload_sha256"],
                source_artifact_sha256=(
                    FROZEN_PREFLIGHT_ARTIFACT_SHA256
                    if contract_bucket == 0
                    else None
                ),
                max_rank_source=source,
                r5_stack_fingerprint_sha256=binding_payload[
                    "r5_stack_fingerprint_sha256"
                ],
            )
            extra_outputs[bundle_path] = selected_bytes
        if set(runtime_contracts_by_bucket) != {"0", "1"}:
            raise SystemExit("R5b supported target lacks both R5c runtime contracts")
    current_cohort = cohort_audit["selected_by_bucket"][str(bucket)]
    r5_entry_contract = {
        "schema": "gfmcd.r5.entry_contract/1",
        "instrument_gates_pass": all_instrument_gates_pass,
        "claim_eligible": bool(
            args.stage == "target"
            and verdict == "ACCEPTOR_SIDE_STATE_SUPPORTED"
            and set(runtime_contracts_by_bucket) == {"0", "1"}
        ),
        "verdict": verdict,
        "acceptor_G_only": True,
        "masked_donor_endpoint_constructed": False,
        "checkpoint_sha256": checkpoint_before["bundle_sha256"],
        "cohort_sha256": current_cohort["ordered_rows_sha256"],
        "construction_sha256": current_cohort["construction_sha256"],
        "selected_rank": selected_rank,
        "crossfit_map_sha256": runtime_max_source["crossfit_map_sha256"],
        "r5_stack_fingerprint_sha256": entry_chain[
            "r5_stack_fingerprint_sha256"
        ],
        "upstream_r3c_artifact_sha256": entry_chain["r3c"]["sha256"],
        "upstream_r3c_manifest_sha256": entry_chain["r3c"][
            "manifest_sha256"
        ],
        "upstream_r5a_artifact_sha256": entry_chain["r5a"]["sha256"],
        "upstream_r5a_manifest_sha256": entry_chain["r5a"][
            "manifest_sha256"
        ],
        "exclusion_registry_sha256": entry_chain["exclusion_registry"][
            "sha256"
        ],
        "exclusion_registry_record_sha256": entry_chain[
            "exclusion_registry"
        ]["record_sha256"],
    }
    result = {
        "schema": "gfmcd.r5b.acceptor_side_state/1",
        "stage": args.stage,
        "status": "CONFIRMATORY" if args.stage == "target" else "DEVELOPMENT",
        "protocol_status": "FROZEN_PROTOCOL_AND_INSTRUMENT_VERIFIED",
        "target_bindable_preflight": target_bindable,
        "endpoint": "single masked acceptor G hard margin",
        "masked_donor_endpoint_constructed": False,
        "bucket": bucket,
        "source_identity": source_identity,
        "cohort_audit": cohort_audit,
        "cohort_audit_sha256": canonical_sha256(cohort_audit),
        "cohort_registry_identity": registry_identity,
        "entry_chain": entry_chain,
        "entry_behavior": entry_behavior,
        "analysis": analysis,
        "model_free_audit": model_free_audit,
        "calibration": calibration,
        "calibration_origin": calibration_origin,
        "instrument_gates": instrument_gates,
        "all_instrument_gates_pass": all_instrument_gates_pass,
        "verdict": verdict,
        "scientific_verdict_before_instrument_gate": scientific_verdict,
        "instrument_binding_payload": binding_payload,
        "instrument_fingerprint_sha256": fingerprint,
        "r5_stack_binding": binding_payload["r5_stack_binding"],
        "r5_stack_fingerprint_sha256": binding_payload[
            "r5_stack_fingerprint_sha256"
        ],
        "checkpoint_identity": checkpoint_before,
        "runtime_identity": current_runtime_identity,
        "input_snapshot": snapshot,
        "input_snapshot_sha256": canonical_sha256(snapshot),
        "git_authorization_identity": git_identity,
        "preflight_package": preflight_package,
        "preflight_context": preflight_context,
        "r5c_runtime_max_source_by_bucket": runtime_max_sources_by_bucket,
        "r5c_runtime_crossfit_map_by_bucket": runtime_crossfit_maps_by_bucket,
        "r5c_runtime_contract_by_bucket": runtime_contracts_by_bucket,
        "r5_entry_contract": r5_entry_contract,
        "licenses": (
            "no scientific claim"
            if args.stage == "preflight"
            else (
                "a fold-indexed acceptor-side residual mediator for a material tested portion"
                if verdict == "ACCEPTOR_SIDE_STATE_SUPPORTED"
                else "the exact resolved negative or unresolved outcome named by verdict"
            )
        ),
        "does_not_license": [
            "the complete paired-donor effect",
            "L3H1 writes the state",
            "a downstream delivery route",
            "example-specific content",
            "state semantics",
            "a biological mechanism",
        ],
    }
    if checkpoint_identity(args.model) != checkpoint_before:
        raise SystemExit("R5b checkpoint changed before artifact serialization")
    binding_after = instrument_binding_payload()
    if (
        binding_after != binding_payload
        or instrument_fingerprint(binding_after) != fingerprint
    ):
        raise SystemExit("R5b joint/local instrument sources changed during execution")
    verify_input_snapshot(snapshot)
    _write_result_and_manifest(
        result,
        output,
        stage=args.stage,
        input_paths=input_paths,
        git_identity=git_identity,
        extra_outputs=extra_outputs,
        input_snapshot=snapshot,
    )
    print(json.dumps({"verdict": verdict, "out": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
