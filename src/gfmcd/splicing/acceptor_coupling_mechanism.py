"""Model-free primitives for the masked-acceptor mechanism programme.

The functions in this module define coordinates and estimands; they do not load a model,
choose candidates, perform inference, or assign mechanistic meaning.  In particular:

* ``A @ V`` reconstruction is an architectural identity, not evidence for binding;
* the MLM decoder is linear in the *output of the MLM transform*, not generally in the
  final transformer residual; and
* necessity and sufficiency are returned as separate row-level quantities.

The corresponding staged designs live in the R5a--R5c draft preregistrations under
``studies/splicebert_acceptor_coupling/preregistrations``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from numbers import Integral
from typing import Any, Mapping, Sequence

import numpy as np


DONOR_WINDOW_OFFSETS = (-1, 0, 1, 2)
AV_CELL_ORDER = ("baseline", "address_only", "value_only", "joint")
BASE_ORDER = ("A", "C", "G", "T")

# A requested SVD direction is considered present only when its singular value is
# strictly larger than this scale-relative threshold.  This is deliberately a fixed
# instrument constant rather than NumPy's shape-dependent default so the same fitted
# matrix cannot change validity when it is routed through a differently shaped helper.
SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE = 1e-10
SVD_NUMERICAL_RANK_RULE = (
    "count singular values sigma_i with sigma_i > sigma_1 * 1e-10; "
    "require numerical rank >= maximum requested candidate rank"
)


def _is_integral_not_bool(value: Any) -> bool:
    """Return whether ``value`` is an integer scalar but not a Boolean scalar."""

    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def _array_sha256(value: Any) -> str:
    """Hash an array in a platform-stable float64 representation.

    The digest binds shape and ordered values without placing the fitted array itself in a
    compact result record.  It is an identity/provenance field, not evidence that a fitted
    activation object was committed.
    """

    array = np.ascontiguousarray(_finite_array("digest_array", value), dtype="<f8")
    digest = hashlib.sha256()
    digest.update(b"gfmcd.float64-array.v1\0")
    digest.update(str(array.ndim).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(int(size)) for size in array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _ordered_text_sha256(values: Sequence[str]) -> str:
    """Hash ordered text values with length prefixes (so boundaries are unambiguous)."""

    digest = hashlib.sha256()
    digest.update(b"gfmcd.ordered-text.v1\0")
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _finite_array(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _same_shape_arrays(**values: Any) -> dict[str, np.ndarray]:
    arrays = {name: _finite_array(name, value) for name, value in values.items()}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        rendered = ", ".join(f"{name}={array.shape}" for name, array in arrays.items())
        raise ValueError(f"all cells must have the same shape; got {rendered}")
    return arrays


def _result(array: np.ndarray) -> float | np.ndarray:
    return float(array) if array.ndim == 0 else array


def donor_window_positions(
    donor_position: int,
    *,
    sequence_length: int | None = None,
) -> tuple[int, int, int, int]:
    """Return the fixed donor-local support ``d + [-1, 0, 1, 2]``.

    Positions are zero-based sequence coordinates.  If ``sequence_length`` is supplied,
    every coordinate must be inside ``[0, sequence_length)``.  Tokenizer offsets are
    deliberately excluded from this scalar helper; use :func:`donor_window_indices` when
    constructing batched token indices.
    """

    if not _is_integral_not_bool(donor_position):
        raise TypeError("donor_position must be an integer sequence coordinate")
    donor = int(donor_position)
    positions = tuple(donor + offset for offset in DONOR_WINDOW_OFFSETS)
    if sequence_length is not None:
        if (
            not _is_integral_not_bool(sequence_length)
            or int(sequence_length) < 1
        ):
            raise ValueError("sequence_length must be a positive integer")
        length = int(sequence_length)
        if positions[0] < 0 or positions[-1] >= length:
            raise ValueError(
                f"donor window {positions} falls outside sequence length {length}"
            )
    return positions  # type: ignore[return-value]


def donor_window_indices(
    donor_positions: Sequence[int] | np.ndarray,
    *,
    sequence_length: int,
    token_offset: int = 0,
) -> np.ndarray:
    """Return batched token indices for ``d + [-1, 0, 1, 2]`` as ``[B, 4]``.

    Bounds are checked in sequence coordinates before the fixed tokenizer offset is added.
    """

    donors = np.asarray(donor_positions)
    if donors.ndim != 1 or donors.size == 0:
        raise ValueError("donor_positions must be a non-empty one-dimensional sequence")
    if not np.issubdtype(donors.dtype, np.integer):
        raise TypeError("donor_positions must contain integers")
    if (
        not _is_integral_not_bool(sequence_length)
        or int(sequence_length) < 1
    ):
        raise ValueError("sequence_length must be a positive integer")
    if not _is_integral_not_bool(token_offset):
        raise TypeError("token_offset must be an integer")
    sequence_indices = donors.astype(np.int64, copy=False)[:, None] + np.asarray(
        DONOR_WINDOW_OFFSETS, dtype=np.int64
    )[None, :]
    if np.any(sequence_indices < 0) or np.any(sequence_indices >= int(sequence_length)):
        raise ValueError("at least one donor window falls outside the sequence")
    return sequence_indices + int(token_offset)


def reconstruct_attention_value(attention: Any, values: Any) -> np.ndarray:
    """Reconstruct ``sum_j A[..., j] V[..., j, :]`` without model dependencies."""

    address = _finite_array("attention", attention)
    payload = _finite_array("values", values)
    if address.ndim < 1 or payload.ndim != address.ndim + 1:
        raise ValueError("attention must be [..., source] and values [..., source, head_dim]")
    if payload.shape[:-1] != address.shape or payload.shape[-1] < 1:
        raise ValueError("attention and values must share every axis through source")
    return np.sum(address[..., None] * payload, axis=-2)


def av_four_cell_reconstruction(
    address_plus: Any,
    address_minus: Any,
    value_plus: Any,
    value_minus: Any,
) -> dict[str, np.ndarray]:
    """Return the ordered ``A x V`` reconstruction cells.

    ``plus`` and ``minus`` name the source states chosen by the caller.  The four cells are
    ``baseline=A- V-``, ``address_only=A+ V-``, ``value_only=A- V+``, and
    ``joint=A+ V+``.  This helper performs no causal interpretation of the cells.
    """

    a_plus = _finite_array("address_plus", address_plus)
    a_minus = _finite_array("address_minus", address_minus)
    v_plus = _finite_array("value_plus", value_plus)
    v_minus = _finite_array("value_minus", value_minus)
    if a_plus.shape != a_minus.shape or v_plus.shape != v_minus.shape:
        raise ValueError("plus/minus address and value states must be shape matched")
    return {
        "baseline": reconstruct_attention_value(a_minus, v_minus),
        "address_only": reconstruct_attention_value(a_plus, v_minus),
        "value_only": reconstruct_attention_value(a_minus, v_plus),
        "joint": reconstruct_attention_value(a_plus, v_plus),
    }


def av_factorial_contrasts(cells: Mapping[str, Any]) -> dict[str, float | np.ndarray]:
    """Compute all non-redundant contrasts of the four ``A x V`` outcome cells."""

    if set(cells) != set(AV_CELL_ORDER):
        raise ValueError(f"cells must have exactly the keys {AV_CELL_ORDER}")
    arrays = _same_shape_arrays(**{name: cells[name] for name in AV_CELL_ORDER})
    baseline = arrays["baseline"]
    address = arrays["address_only"]
    value = arrays["value_only"]
    joint = arrays["joint"]
    contrasts = {
        "joint_lift": joint - baseline,
        "address_effect_at_value_minus": address - baseline,
        "value_effect_at_address_minus": value - baseline,
        "value_effect_at_address_plus": joint - address,
        "address_effect_at_value_plus": joint - value,
        "interaction_AxV": joint - address - value + baseline,
    }
    return {name: _result(array) for name, array in contrasts.items()}


def partner_pseudo_specific(
    native_partner: Any,
    intervened_partner: Any,
    native_pseudo: Any,
    intervened_pseudo: Any,
) -> dict[str, float | np.ndarray]:
    """Return partner drop, pseudo drop, and their specificity contrast.

    The sign convention is always native minus intervened.  Inputs are raw per-row outcome
    cells, not ratios or already-normalized fractions.
    """

    cells = _same_shape_arrays(
        native_partner=native_partner,
        intervened_partner=intervened_partner,
        native_pseudo=native_pseudo,
        intervened_pseudo=intervened_pseudo,
    )
    partner = cells["native_partner"] - cells["intervened_partner"]
    pseudo = cells["native_pseudo"] - cells["intervened_pseudo"]
    return {
        "partner": _result(partner),
        "pseudo": _result(pseudo),
        "specific": _result(partner - pseudo),
    }


def orthogonal_projector(basis: Any, *, atol: float = 1e-8) -> np.ndarray:
    """Return ``P = U.T @ U`` for a row-orthonormal basis ``U[k, width]``."""

    vectors = _finite_array("basis", basis)
    if vectors.ndim != 2 or not 1 <= vectors.shape[0] <= vectors.shape[1]:
        raise ValueError("basis must have shape [rank, width] with 1 <= rank <= width")
    if atol <= 0:
        raise ValueError("atol must be positive")
    gram = vectors @ vectors.T
    if not np.allclose(gram, np.eye(vectors.shape[0]), rtol=0.0, atol=atol):
        raise ValueError("basis rows are not orthonormal at the declared tolerance")
    return vectors.T @ vectors


def projector_diagnostics(projector: Any) -> dict[str, float]:
    """Return symmetry, idempotence, and integer-rank errors for an alleged projector."""

    matrix = _finite_array("projector", projector)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("projector must be square")
    rank = int(np.rint(np.trace(matrix)))
    return {
        "symmetry_max_abs": float(np.max(np.abs(matrix - matrix.T))),
        "idempotence_max_abs": float(np.max(np.abs(matrix @ matrix - matrix))),
        "trace_rank_error": float(abs(np.trace(matrix) - rank)),
        "rank": rank,
    }


def project_rows(rows: Any, projector: Any) -> np.ndarray:
    """Project row vectors on their final axis with a square projector."""

    values = _finite_array("rows", rows)
    matrix = _finite_array("projector", projector)
    if values.ndim < 1 or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("rows must be [..., width] and projector must be square")
    if values.shape[-1] != matrix.shape[0]:
        raise ValueError("row width and projector width disagree")
    return values @ matrix


def replace_projected_component(target: Any, source: Any, projector: Any) -> np.ndarray:
    """Replace ``target``'s projected component with ``source``'s component."""

    arrays = _same_shape_arrays(target=target, source=source)
    delta = arrays["source"] - arrays["target"]
    return arrays["target"] + project_rows(delta, projector)


def validate_svd_numerical_rank(
    singular_values: Any,
    *,
    required_rank: int,
) -> dict[str, Any]:
    """Enforce and record the frozen scale-relative SVD rank rule.

    This public validator is also the required route for any optimized or refit SVD path
    that does not call :func:`fit_svd_grid` directly.
    """

    values = _finite_array("singular_values", singular_values)
    if values.ndim != 1 or values.size < 1 or np.any(values < 0.0):
        raise ValueError("singular_values must be a non-empty non-negative vector")
    if np.any(np.diff(values) > 0.0):
        raise ValueError("singular_values must be ordered non-increasingly")
    if (
        not _is_integral_not_bool(required_rank)
        or int(required_rank) < 1
        or int(required_rank) > values.size
    ):
        raise ValueError("required_rank must be an available positive integer rank")
    required = int(required_rank)
    leading = float(values[0])
    if leading == 0.0:
        raise ValueError("cannot validate numerical rank for an exact-zero matrix")
    ratios = values / leading
    numerical_rank = int(
        np.count_nonzero(ratios > SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE)
    )
    threshold = leading * SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE
    required_singular = float(values[required - 1])
    required_ratio = float(ratios[required - 1])
    diagnostics = {
        "required_rank": required,
        "numerical_rank": numerical_rank,
        "numerical_rank_relative_tolerance": (
            SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE
        ),
        "numerical_rank_threshold": threshold,
        "minimum_required_singular_value": required_singular,
        "minimum_required_singular_value_ratio": required_ratio,
        "singular_values_sha256": _array_sha256(values),
        "numerical_rank_rule": SVD_NUMERICAL_RANK_RULE,
    }
    diagnostics["numerical_rank_diagnostics_sha256"] = _array_sha256(
        np.asarray(
            [
                SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE,
                threshold,
                required_singular,
                required_ratio,
                float(numerical_rank),
                float(required),
            ],
            dtype=np.float64,
        )
    )
    if numerical_rank < required:
        raise ValueError(
            "SVD matrix is numerically rank deficient under the frozen "
            f"scale-relative rule: numerical_rank={numerical_rank}, "
            f"required_rank={required}, relative_tolerance="
            f"{SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE:g}, "
            f"minimum_required_singular_value_ratio={required_ratio:.17g}"
        )
    return diagnostics


@dataclass(frozen=True)
class SVDGridFit:
    """An uncentered SVD basis plus a *diagnostic* energy summary.

    ``diagnostic_energy_rank`` is the smallest rank reaching
    ``diagnostic_energy_target``. It is useful for descriptive instrument diagnostics only;
    R5b uses :func:`select_crossfit_outcome_rank` on held-out causal sufficiency.
    ``basis`` contains every row through the largest candidate rank so every candidate can
    be evaluated without refitting.
    """

    basis: np.ndarray
    singular_values: np.ndarray
    candidate_ranks: tuple[int, ...]
    cumulative_energy_by_rank: tuple[tuple[int, float], ...]
    diagnostic_energy_rank: int
    diagnostic_energy_target: float
    diagnostic_energy_target_reached: bool
    n_fit_rows: int
    width: int
    numerical_rank: int
    numerical_rank_relative_tolerance: float
    numerical_rank_threshold: float
    minimum_required_singular_value: float
    minimum_required_singular_value_ratio: float

    def record(self) -> dict[str, Any]:
        """Return JSON-ready selection metadata without serializing the fitted basis."""

        projector_digests = {
            str(rank): _array_sha256(orthogonal_projector(self.basis[:rank]))
            for rank in self.candidate_ranks
        }
        return {
            "candidate_ranks": list(self.candidate_ranks),
            "cumulative_energy_by_rank": {
                str(rank): energy for rank, energy in self.cumulative_energy_by_rank
            },
            "diagnostic_energy_rank": self.diagnostic_energy_rank,
            "diagnostic_energy_target": self.diagnostic_energy_target,
            "diagnostic_energy_target_reached": self.diagnostic_energy_target_reached,
            "n_fit_rows": self.n_fit_rows,
            "width": self.width,
            "maximum_requested_rank": self.candidate_ranks[-1],
            "numerical_rank": self.numerical_rank,
            "numerical_rank_relative_tolerance": (
                self.numerical_rank_relative_tolerance
            ),
            "numerical_rank_threshold": self.numerical_rank_threshold,
            "minimum_required_singular_value": (
                self.minimum_required_singular_value
            ),
            "minimum_required_singular_value_ratio": (
                self.minimum_required_singular_value_ratio
            ),
            "numerical_rank_rule": SVD_NUMERICAL_RANK_RULE,
            "basis_sha256": _array_sha256(self.basis),
            "singular_values_sha256": _array_sha256(self.singular_values),
            "numerical_rank_diagnostics_sha256": _array_sha256(
                np.asarray(
                    [
                        self.numerical_rank_relative_tolerance,
                        self.numerical_rank_threshold,
                        self.minimum_required_singular_value,
                        self.minimum_required_singular_value_ratio,
                        float(self.numerical_rank),
                        float(self.candidate_ranks[-1]),
                    ],
                    dtype=np.float64,
                )
            ),
            "projector_sha256_by_rank": projector_digests,
            "fitted_arrays_serialized": False,
            "diagnostic_energy_summary_rule": (
                "NOT_THE_R5B_SCIENTIFIC_SELECTOR: "
                "smallest_candidate_rank_reaching_uncentered_svd_energy_target; "
                "otherwise_largest_candidate_rank_with_failure_flag"
            ),
        }


def fit_svd_grid(
    rows: Any,
    *,
    candidate_ranks: Sequence[int],
    diagnostic_energy_target: float,
) -> SVDGridFit:
    """Fit an uncentered SVD and summarize its derivation-set energy curve.

    This generic energy diagnostic must not choose the scientific ``U_A`` rank.  Use
    :func:`select_crossfit_outcome_rank` after held-out evaluation for that decision.
    """

    values = _finite_array("rows", rows)
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("rows must be a non-empty two-dimensional matrix")
    raw_ranks = tuple(candidate_ranks)
    if (
        not raw_ranks
        or any(
            not _is_integral_not_bool(rank)
            or int(rank) < 1
            for rank in raw_ranks
        )
    ):
        raise ValueError("candidate_ranks must be a strictly increasing sequence of positive integers")
    ranks = tuple(int(rank) for rank in raw_ranks)
    if tuple(sorted(set(ranks))) != ranks:
        raise ValueError("candidate_ranks must be a strictly increasing sequence of positive integers")
    available = min(values.shape)
    if ranks[-1] > available:
        raise ValueError(
            f"largest candidate rank {ranks[-1]} exceeds available rank {available}"
        )
    if not 0 < float(diagnostic_energy_target) <= 1:
        raise ValueError("diagnostic_energy_target must lie in (0, 1]")
    _, singular_values, vh = np.linalg.svd(values, full_matrices=False)
    if not np.isfinite(singular_values).all() or not np.isfinite(vh).all():
        raise ValueError("SVD produced non-finite fitted values")
    numerical_rank_record = validate_svd_numerical_rank(
        singular_values,
        required_rank=ranks[-1],
    )
    leading = float(singular_values[0])
    # Scale before squaring so rank/energy validity is invariant to multiplying the
    # complete fit matrix by a tiny or huge finite constant.
    squared = np.square(singular_values / leading)
    total = float(squared.sum())
    cumulative = np.cumsum(squared) / total
    curve = tuple((rank, float(cumulative[rank - 1])) for rank in ranks)
    diagnostic_rank = next(
        (rank for rank, energy in curve if energy >= float(diagnostic_energy_target)),
        ranks[-1],
    )
    reached = any(energy >= float(diagnostic_energy_target) for _, energy in curve)
    return SVDGridFit(
        basis=vh[:ranks[-1]].copy(),
        singular_values=singular_values.copy(),
        candidate_ranks=ranks,
        cumulative_energy_by_rank=curve,
        diagnostic_energy_rank=diagnostic_rank,
        diagnostic_energy_target=float(diagnostic_energy_target),
        diagnostic_energy_target_reached=bool(reached),
        n_fit_rows=values.shape[0],
        width=values.shape[1],
        numerical_rank=int(numerical_rank_record["numerical_rank"]),
        numerical_rank_relative_tolerance=(
            SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE
        ),
        numerical_rank_threshold=float(
            numerical_rank_record["numerical_rank_threshold"]
        ),
        minimum_required_singular_value=float(
            numerical_rank_record["minimum_required_singular_value"]
        ),
        minimum_required_singular_value_ratio=float(
            numerical_rank_record["minimum_required_singular_value_ratio"]
        ),
    )


def deterministic_group_folds(
    groups: Sequence[str],
    *,
    n_folds: int = 2,
    namespace: str = "acceptor-u-a-v1",
) -> np.ndarray:
    """Assign every row from a gene to one stable SHA-256 fold."""

    if not _is_integral_not_bool(n_folds):
        raise TypeError("n_folds must be an integer")
    if int(n_folds) < 2 or not isinstance(namespace, str) or not namespace:
        raise ValueError("n_folds must be at least two and namespace must be non-empty")
    rendered = [str(group) for group in groups]
    if not rendered or any(not group for group in rendered):
        raise ValueError("groups must contain non-empty identifiers")
    by_group = {
        group: int.from_bytes(
            hashlib.sha256(f"{namespace}|{group}".encode("utf-8")).digest()[:8],
            "big",
        )
        % int(n_folds)
        for group in sorted(set(rendered))
    }
    return np.asarray([by_group[group] for group in rendered], dtype=np.int64)


@dataclass(frozen=True)
class CrossFitFold:
    evaluation_fold: str
    derivation_indices: np.ndarray
    evaluation_indices: np.ndarray
    derivation_groups: tuple[str, ...]
    evaluation_groups: tuple[str, ...]
    svd: SVDGridFit
    derivation_matrix_policy: str

    def record(self) -> dict[str, Any]:
        return {
            "evaluation_fold": self.evaluation_fold,
            "n_derivation_rows": int(self.derivation_indices.size),
            "n_evaluation_rows": int(self.evaluation_indices.size),
            "n_derivation_groups": len(self.derivation_groups),
            "n_evaluation_groups": len(self.evaluation_groups),
            "derivation_groups": list(self.derivation_groups),
            "evaluation_groups": list(self.evaluation_groups),
            "derivation_matrix_policy": self.derivation_matrix_policy,
            "svd": self.svd.record(),
        }


@dataclass(frozen=True)
class CrossFitSVD:
    folds: tuple[CrossFitFold, ...]
    row_fold_ids: tuple[str, ...]
    group_average: bool

    def record(self) -> dict[str, Any]:
        return {
            "n_folds": len(self.folds),
            "group_average": self.group_average,
            "row_fold_ids": list(self.row_fold_ids),
            "ordered_row_fold_ids_sha256": _ordered_text_sha256(self.row_fold_ids),
            "folds": [fold.record() for fold in self.folds],
        }


def _group_mean_matrix(rows: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    rendered = np.asarray([str(group) for group in groups], dtype=object)
    labels = sorted(set(rendered.tolist()))
    return np.stack([rows[rendered == label].mean(axis=0) for label in labels], axis=0)


def crossfit_svd(
    rows: Any,
    groups: Sequence[str],
    fold_ids: Sequence[str | int],
    *,
    candidate_ranks: Sequence[int],
    diagnostic_energy_target: float,
    group_average: bool = True,
) -> CrossFitSVD:
    """Fit one SVD basis per held-out fold with strict gene disjointness.

    When ``group_average`` is true (the default), derivation genes contribute one mean row
    each to rank selection and basis fitting.  Evaluation remains at the original row level.
    """

    matrix = _finite_array("rows", rows)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("rows must have shape [examples, width] with at least two examples")
    rendered_groups = [str(group) for group in groups]
    rendered_folds = [str(fold) for fold in fold_ids]
    if len(rendered_groups) != matrix.shape[0] or len(rendered_folds) != matrix.shape[0]:
        raise ValueError("groups and fold_ids must have one entry per row")
    if any(not group for group in rendered_groups) or any(not fold for fold in rendered_folds):
        raise ValueError("group and fold identifiers must be non-empty")
    group_to_folds: dict[str, set[str]] = {}
    for group, fold in zip(rendered_groups, rendered_folds):
        group_to_folds.setdefault(group, set()).add(fold)
    split_groups = [group for group, assigned in group_to_folds.items() if len(assigned) != 1]
    if split_groups:
        raise ValueError(f"groups may not cross folds; first split group is {sorted(split_groups)[0]}")
    unique_folds = sorted(set(rendered_folds))
    if len(unique_folds) < 2:
        raise ValueError("cross-fitting requires at least two non-empty folds")

    group_array = np.asarray(rendered_groups, dtype=object)
    fold_array = np.asarray(rendered_folds, dtype=object)
    fitted: list[CrossFitFold] = []
    for held_out in unique_folds:
        evaluation_indices = np.flatnonzero(fold_array == held_out)
        derivation_indices = np.flatnonzero(fold_array != held_out)
        if not len(evaluation_indices) or not len(derivation_indices):
            raise ValueError("every cross-fit fold needs derivation and evaluation rows")
        derivation_groups = tuple(sorted(set(group_array[derivation_indices].tolist())))
        evaluation_groups = tuple(sorted(set(group_array[evaluation_indices].tolist())))
        if set(derivation_groups) & set(evaluation_groups):
            raise ValueError("derivation and evaluation groups are not disjoint")
        derivation_rows = matrix[derivation_indices]
        if group_average:
            derivation_rows = _group_mean_matrix(
                derivation_rows,
                group_array[derivation_indices].tolist(),
            )
        svd = fit_svd_grid(
            derivation_rows,
            candidate_ranks=candidate_ranks,
            diagnostic_energy_target=diagnostic_energy_target,
        )
        fitted.append(
            CrossFitFold(
                evaluation_fold=held_out,
                derivation_indices=derivation_indices,
                evaluation_indices=evaluation_indices,
                derivation_groups=derivation_groups,
                evaluation_groups=evaluation_groups,
                svd=svd,
                derivation_matrix_policy=("gene_mean_rows" if group_average else "raw_rows"),
            )
        )
    return CrossFitSVD(
        folds=tuple(fitted),
        row_fold_ids=tuple(rendered_folds),
        group_average=bool(group_average),
    )


def crossfit_project(rows: Any, fitted: CrossFitSVD, *, rank: int) -> np.ndarray:
    """Project each row at an explicit candidate rank using its held-out-fold basis."""

    matrix = _finite_array("rows", rows)
    if matrix.ndim != 2 or matrix.shape[0] != len(fitted.row_fold_ids):
        raise ValueError("rows must match the fitted cross-fit row count")
    output = np.empty_like(matrix)
    if not _is_integral_not_bool(rank) or int(rank) < 1:
        raise ValueError("rank must be a positive integer")
    requested_rank = int(rank)
    covered = np.zeros(matrix.shape[0], dtype=np.int64)
    for fold in fitted.folds:
        if matrix.shape[1] != fold.svd.width:
            raise ValueError("row width does not match a fitted basis")
        if requested_rank not in fold.svd.candidate_ranks:
            raise ValueError("rank must belong to every fitted candidate-rank grid")
        if (
            fold.evaluation_indices.ndim != 1
            or np.any(fold.evaluation_indices < 0)
            or np.any(fold.evaluation_indices >= matrix.shape[0])
            or len(np.unique(fold.evaluation_indices)) != len(fold.evaluation_indices)
        ):
            raise ValueError("each fold must contain unique in-range evaluation indices")
        projector = orthogonal_projector(fold.svd.basis[:requested_rank])
        output[fold.evaluation_indices] = project_rows(
            matrix[fold.evaluation_indices], projector
        )
        np.add.at(covered, fold.evaluation_indices, 1)
    if not np.all(covered == 1):
        raise ValueError("cross-fit bookkeeping does not cover every row exactly once")
    return output


@dataclass(frozen=True)
class CrossFitOutcomeRankSelection:
    """Frozen R5b rank decision from held-out causal outcomes and matched controls."""

    rank_grid: tuple[int, ...]
    heldout_sufficiency_by_rank: tuple[tuple[int, float], ...]
    random_overlap_gate_by_rank: tuple[tuple[int, bool], ...]
    random_control_gate_by_rank: tuple[tuple[int, bool], ...]
    maximum_heldout_sufficiency: float
    recovery_fraction: float
    recovery_threshold: float
    selected_rank: int | None
    status: str

    def record(self) -> dict[str, Any]:
        return {
            "rank_grid": list(self.rank_grid),
            "heldout_sufficiency_by_rank": {
                str(rank): value for rank, value in self.heldout_sufficiency_by_rank
            },
            "random_overlap_gate_by_rank": {
                str(rank): value for rank, value in self.random_overlap_gate_by_rank
            },
            "random_control_gate_by_rank": {
                str(rank): value for rank, value in self.random_control_gate_by_rank
            },
            "maximum_heldout_sufficiency": self.maximum_heldout_sufficiency,
            "recovery_fraction": self.recovery_fraction,
            "recovery_threshold": self.recovery_threshold,
            "selected_rank": self.selected_rank,
            "status": self.status,
            "selection_rule": (
                "smallest rank reaching recovery_fraction times the maximum held-out "
                "U_A sufficiency and passing both preregistered rank-matched random "
                "overlap and causal-control gates; otherwise no U_A rank"
            ),
        }


def select_crossfit_outcome_rank(
    heldout_sufficiency_by_rank: Mapping[int, float],
    random_overlap_gate_by_rank: Mapping[int, bool],
    random_control_gate_by_rank: Mapping[int, bool],
    *,
    rank_grid: Sequence[int] = (1, 2, 4, 8, 16, 32),
    recovery_fraction: float = 0.90,
) -> CrossFitOutcomeRankSelection:
    """Apply the frozen R5b held-out outcome rank rule.

    This function does not compute the sufficiency or control gates.  It verifies that the
    complete prespecified rank family was supplied and performs only the fixed selection:
    choose the smallest rank whose held-out sufficiency is at least ``recovery_fraction`` of
    the maximum across the grid and which passes both rank-matched control gates.  If no rank
    qualifies (including a non-positive maximum), ``selected_rank`` is ``None`` and no
    ``U_A`` claim may be made.
    """

    raw_ranks = tuple(rank_grid)
    if (
        not raw_ranks
        or any(
            not _is_integral_not_bool(rank)
            or int(rank) < 1
            for rank in raw_ranks
        )
    ):
        raise ValueError("rank_grid must be a strictly increasing positive sequence")
    ranks = tuple(int(rank) for rank in raw_ranks)
    if tuple(sorted(set(ranks))) != ranks:
        raise ValueError("rank_grid must be a strictly increasing positive sequence")
    if not 0 < float(recovery_fraction) <= 1:
        raise ValueError("recovery_fraction must lie in (0, 1]")
    expected = set(ranks)
    supplied = (
        set(heldout_sufficiency_by_rank),
        set(random_overlap_gate_by_rank),
        set(random_control_gate_by_rank),
    )
    if any(keys != expected for keys in supplied):
        raise ValueError("every rank map must contain exactly the complete rank_grid")
    for name, mapping in (
        ("heldout_sufficiency_by_rank", heldout_sufficiency_by_rank),
        ("random_overlap_gate_by_rank", random_overlap_gate_by_rank),
        ("random_control_gate_by_rank", random_control_gate_by_rank),
    ):
        if any(not _is_integral_not_bool(key) for key in mapping):
            raise TypeError(f"{name} keys must be integer ranks, not booleans")
    for name, gate in (
        ("random_overlap_gate_by_rank", random_overlap_gate_by_rank),
        ("random_control_gate_by_rank", random_control_gate_by_rank),
    ):
        if any(type(gate[rank]) is not bool for rank in ranks):
            raise TypeError(f"{name} values must be literal booleans")
    sufficiency = tuple(
        (rank, float(heldout_sufficiency_by_rank[rank])) for rank in ranks
    )
    if not np.isfinite([value for _, value in sufficiency]).all():
        raise ValueError("heldout sufficiency contains non-finite values")
    overlap = tuple((rank, bool(random_overlap_gate_by_rank[rank])) for rank in ranks)
    controls = tuple((rank, bool(random_control_gate_by_rank[rank])) for rank in ranks)
    maximum = max(value for _, value in sufficiency)
    threshold = float(recovery_fraction) * maximum
    overlap_map = dict(overlap)
    control_map = dict(controls)
    eligible = [
        rank
        for rank, value in sufficiency
        if maximum > 0
        and value >= threshold
        and overlap_map[rank]
        and control_map[rank]
    ]
    selected = eligible[0] if eligible else None
    return CrossFitOutcomeRankSelection(
        rank_grid=ranks,
        heldout_sufficiency_by_rank=sufficiency,
        random_overlap_gate_by_rank=overlap,
        random_control_gate_by_rank=controls,
        maximum_heldout_sufficiency=float(maximum),
        recovery_fraction=float(recovery_fraction),
        recovery_threshold=float(threshold),
        selected_rank=selected,
        status=("RANK_SELECTED" if selected is not None else "NO_U_A_RANK"),
    )


def deterministic_random_basis(
    width: int,
    rank: int,
    *,
    seed: int,
    namespace: str,
    draw: int,
) -> np.ndarray:
    """Return a stable Haar-random row-orthonormal basis."""

    if not _is_integral_not_bool(width) or not _is_integral_not_bool(rank):
        raise TypeError("width and rank must be integers")
    if not _is_integral_not_bool(seed) or not _is_integral_not_bool(draw):
        raise TypeError("seed and draw must be integers")
    if not 1 <= int(rank) <= int(width):
        raise ValueError("rank must lie in [1, width]")
    if not isinstance(namespace, str) or not namespace or int(draw) < 0:
        raise ValueError("namespace must be non-empty and draw must be non-negative")
    digest = hashlib.sha256(
        f"{namespace}|seed={int(seed)}|draw={int(draw)}".encode("utf-8")
    ).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    q, r = np.linalg.qr(generator.standard_normal((int(width), int(rank))), mode="reduced")
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    q = q * signs[None, :]
    return q.T


def deterministic_nested_random_bases(
    width: int,
    ranks: Sequence[int],
    *,
    seed: int,
    namespace: str,
    draw: int,
) -> dict[int, np.ndarray]:
    """Return deterministic Haar prefixes from one maximal random basis.

    A single ``width x max(ranks)`` Gaussian matrix is factorized.  Rank ``k`` receives
    the first ``k`` signed-Q columns, so every smaller-rank null is an exact prefix of every
    larger-rank null for the same draw.  This is the required control for a scanned rank
    family; calling :func:`deterministic_random_basis` independently at each rank does not
    create the same nested null object.
    """

    if not _is_integral_not_bool(width):
        raise TypeError("width must be an integer")
    if not _is_integral_not_bool(seed) or not _is_integral_not_bool(draw):
        raise TypeError("seed and draw must be integers")
    raw_ranks = tuple(ranks)
    if (
        not raw_ranks
        or any(not _is_integral_not_bool(rank) or int(rank) < 1 for rank in raw_ranks)
    ):
        raise ValueError("ranks must be a strictly increasing sequence of positive integers")
    normalized = tuple(int(rank) for rank in raw_ranks)
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError("ranks must be a strictly increasing sequence of positive integers")
    if int(width) < 1 or normalized[-1] > int(width):
        raise ValueError("every rank must lie in [1, width]")
    if not isinstance(namespace, str) or not namespace or int(draw) < 0:
        raise ValueError("namespace must be non-empty and draw must be non-negative")
    digest = hashlib.sha256(
        (
            f"{namespace}|nested-haar|max_rank={normalized[-1]}|"
            f"seed={int(seed)}|draw={int(draw)}"
        ).encode("utf-8")
    ).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    q, r = np.linalg.qr(
        generator.standard_normal((int(width), normalized[-1])),
        mode="reduced",
    )
    signs = np.where(np.diag(r) < 0, -1.0, 1.0)
    q = q * signs[None, :]
    return {rank: q[:, :rank].T.copy() for rank in normalized}


def mean_squared_principal_cosine(left_basis: Any, right_basis: Any) -> float:
    """Measure equal-rank subspace overlap as mean squared principal cosine.

    Both inputs must be row-orthonormal ``[rank, width]`` bases of the same shape.  The
    result lies in ``[0, 1]``.  This helper supplies an absolute overlap estimand; sampling
    uncertainty and the preregistered SESOI remain responsibilities of the study runner.
    """

    arrays = _same_shape_arrays(left_basis=left_basis, right_basis=right_basis)
    left = arrays["left_basis"]
    right = arrays["right_basis"]
    if left.ndim != 2 or not 1 <= left.shape[0] <= left.shape[1]:
        raise ValueError("bases must share shape [rank, width] with 1 <= rank <= width")
    # Validate row orthonormality with the same strict default as projector construction.
    orthogonal_projector(left)
    orthogonal_projector(right)
    singular_values = np.linalg.svd(left @ right.T, compute_uv=False)
    overlap = float(np.mean(np.square(np.clip(singular_values, 0.0, 1.0))))
    return float(np.clip(overlap, 0.0, 1.0))


@dataclass(frozen=True, order=True)
class OverlapNullDrawIdentity:
    """Semantic identity of one ordered overlap-null draw.

    Array hashes alone cannot detect a calibration draw reused under a permutation or as
    part of a larger test pool.  The namespace, seed, and draw index are therefore part of
    the statistical object and are checked as a true set before nuisance fitting.
    """

    namespace: str
    seed: int
    draw: int

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace:
            raise ValueError("overlap-null draw namespace must be non-empty")
        if not _is_integral_not_bool(self.seed):
            raise TypeError("overlap-null draw seed must be an integer")
        if not _is_integral_not_bool(self.draw):
            raise TypeError("overlap-null draw index must be an integer")
        if int(self.draw) < 0:
            raise ValueError("overlap-null draw index must be non-negative")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "draw", int(self.draw))

    def canonical_text(self) -> str:
        """Return an unambiguous canonical identity encoding."""

        namespace = self.namespace.encode("utf-8")
        return (
            f"namespace_utf8_length={len(namespace)}|{self.namespace}|"
            f"seed={int(self.seed)}|draw={int(self.draw)}"
        )

    def record(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "seed": int(self.seed),
            "draw": int(self.draw),
        }


def ordered_overlap_null_draw_identities(
    *,
    namespace: str,
    seed: int,
    n_draws: int,
    first_draw: int = 0,
) -> tuple[OverlapNullDrawIdentity, ...]:
    """Construct a contiguous ordered overlap-null draw identity sequence."""

    if not _is_integral_not_bool(n_draws) or int(n_draws) < 2:
        raise ValueError("n_draws must be an integer of at least two")
    if not _is_integral_not_bool(first_draw) or int(first_draw) < 0:
        raise ValueError("first_draw must be a non-negative integer")
    return tuple(
        OverlapNullDrawIdentity(
            namespace=namespace,
            seed=seed,
            draw=int(first_draw) + offset,
        )
        for offset in range(int(n_draws))
    )


@dataclass(frozen=True)
class FixedCalibrationOverlapNull:
    """Rank-wise overlap standardization fit on a calibration-only null pool."""

    rank_grid: tuple[int, ...]
    calibration_mean_by_rank: tuple[tuple[int, float], ...]
    calibration_sd_by_rank: tuple[tuple[int, float], ...]
    observed_standardized_by_rank: tuple[tuple[int, float], ...]
    test_max_standardized: np.ndarray
    n_calibration_draws: int
    n_test_draws: int
    calibration_pool_sha256: str
    test_pool_sha256: str
    calibration_draw_identities: tuple[OverlapNullDrawIdentity, ...]
    test_draw_identities: tuple[OverlapNullDrawIdentity, ...]

    def record(self) -> dict[str, Any]:
        """Return compact JSON-ready parameters and pool identities, not null arrays."""

        calibration_text = tuple(
            identity.canonical_text() for identity in self.calibration_draw_identities
        )
        test_text = tuple(
            identity.canonical_text() for identity in self.test_draw_identities
        )
        return {
            "rank_grid": list(self.rank_grid),
            "calibration_mean_by_rank": {
                str(rank): value for rank, value in self.calibration_mean_by_rank
            },
            "calibration_sd_by_rank": {
                str(rank): value for rank, value in self.calibration_sd_by_rank
            },
            "observed_standardized_by_rank": {
                str(rank): value for rank, value in self.observed_standardized_by_rank
            },
            "n_calibration_draws": self.n_calibration_draws,
            "n_test_draws": self.n_test_draws,
            "calibration_pool_sha256": self.calibration_pool_sha256,
            "test_pool_sha256": self.test_pool_sha256,
            "ordered_calibration_draw_identities_sha256": _ordered_text_sha256(
                calibration_text
            ),
            "ordered_test_draw_identities_sha256": _ordered_text_sha256(test_text),
            "calibration_draw_identity_set_sha256": _ordered_text_sha256(
                tuple(sorted(calibration_text))
            ),
            "test_draw_identity_set_sha256": _ordered_text_sha256(
                tuple(sorted(test_text))
            ),
            "calibration_first_draw_identity": (
                self.calibration_draw_identities[0].record()
            ),
            "calibration_last_draw_identity": (
                self.calibration_draw_identities[-1].record()
            ),
            "test_first_draw_identity": self.test_draw_identities[0].record(),
            "test_last_draw_identity": self.test_draw_identities[-1].record(),
            "draw_identity_schema": "namespace_utf8, signed_integer_seed, nonnegative_draw_v1",
            "calibration_unique_draw_identities": len(
                set(self.calibration_draw_identities)
            ),
            "test_unique_draw_identities": len(set(self.test_draw_identities)),
            "semantic_draw_identity_intersection_count": len(
                set(self.calibration_draw_identities)
                & set(self.test_draw_identities)
            ),
            "draw_identity_sets_disjoint": not bool(
                set(self.calibration_draw_identities)
                & set(self.test_draw_identities)
            ),
            "test_max_standardized_sha256": _array_sha256(
                self.test_max_standardized
            ),
            "null_arrays_serialized": False,
            "standardization_rule": (
                "rank_mean_and_sample_sd_fit_on_calibration_pool_only; observed_and_"
                "independent_test_pool_standardized_with_frozen_calibration_parameters"
            ),
        }


def fixed_calibration_overlap_null(
    observed_by_rank: Mapping[int, float],
    calibration_overlap_by_rank: Mapping[int, Any],
    test_overlap_by_rank: Mapping[int, Any],
    *,
    rank_grid: Sequence[int],
    calibration_draw_identities: Sequence[OverlapNullDrawIdentity],
    test_draw_identities: Sequence[OverlapNullDrawIdentity],
) -> FixedCalibrationOverlapNull:
    """Standardize observed/test overlaps using a disjoint calibration null pool.

    The caller generates calibration and test overlaps from independently seeded namespaces
    and supplies the ordered semantic identity of every draw.  This function rejects duplicate
    identities within a pool and any identity intersection across pools, including permuted or
    partial reuse that unequal array hashes cannot detect.  It fits rank-wise nuisance parameters
    only on the calibration pool, freezes them, and applies them to both the observed statistic and
    every test-null draw.  The returned test maxima can therefore enter an add-one maximum-family
    p-value without estimating nuisance parameters from those same test draws.
    """

    raw_ranks = tuple(rank_grid)
    if (
        not raw_ranks
        or any(not _is_integral_not_bool(rank) or int(rank) < 1 for rank in raw_ranks)
    ):
        raise ValueError("rank_grid must be a strictly increasing positive sequence")
    ranks = tuple(int(rank) for rank in raw_ranks)
    if tuple(sorted(set(ranks))) != ranks:
        raise ValueError("rank_grid must be a strictly increasing positive sequence")
    expected = set(ranks)
    for name, mapping in (
        ("observed_by_rank", observed_by_rank),
        ("calibration_overlap_by_rank", calibration_overlap_by_rank),
        ("test_overlap_by_rank", test_overlap_by_rank),
    ):
        if set(mapping) != expected:
            raise ValueError(f"{name} must contain exactly the complete rank_grid")
        if any(not _is_integral_not_bool(key) for key in mapping):
            raise TypeError(f"{name} keys must be integer ranks, not booleans")

    observed = np.asarray(
        [float(observed_by_rank[rank]) for rank in ranks], dtype=np.float64
    )
    if not np.isfinite(observed).all():
        raise ValueError("observed overlaps contain non-finite values")

    def matrix_from(name: str, mapping: Mapping[int, Any]) -> np.ndarray:
        columns: list[np.ndarray] = []
        draw_count: int | None = None
        for rank in ranks:
            column = _finite_array(f"{name}[{rank}]", mapping[rank])
            if column.ndim != 1 or column.size < 2:
                raise ValueError(f"{name} values must be one-dimensional with at least two draws")
            if draw_count is None:
                draw_count = int(column.size)
            elif column.size != draw_count:
                raise ValueError(f"{name} ranks must have equal draw counts")
            columns.append(column)
        return np.stack(columns, axis=1)

    calibration = matrix_from("calibration_overlap_by_rank", calibration_overlap_by_rank)
    test = matrix_from("test_overlap_by_rank", test_overlap_by_rank)

    def validated_identities(
        name: str,
        supplied: Sequence[OverlapNullDrawIdentity],
        expected_count: int,
    ) -> tuple[OverlapNullDrawIdentity, ...]:
        identities = tuple(supplied)
        if len(identities) != expected_count:
            raise ValueError(
                f"{name} must have one ordered identity per overlap draw"
            )
        if any(not isinstance(identity, OverlapNullDrawIdentity) for identity in identities):
            raise TypeError(
                f"{name} entries must be OverlapNullDrawIdentity records"
            )
        if len(set(identities)) != len(identities):
            raise ValueError(f"{name} contains duplicate semantic draw identities")
        return identities

    calibration_identities = validated_identities(
        "calibration_draw_identities",
        calibration_draw_identities,
        int(calibration.shape[0]),
    )
    test_identities = validated_identities(
        "test_draw_identities",
        test_draw_identities,
        int(test.shape[0]),
    )
    reused_identities = set(calibration_identities) & set(test_identities)
    if reused_identities:
        first_reused = min(reused_identities).canonical_text()
        raise ValueError(
            "calibration and test overlap draw identities must be disjoint; "
            f"first reused identity is {first_reused}"
        )
    calibration_digest = _array_sha256(calibration)
    test_digest = _array_sha256(test)
    if calibration_digest == test_digest:
        raise ValueError("calibration and test overlap pools must be disjoint")
    means = calibration.mean(axis=0)
    sample_sd = calibration.std(axis=0, ddof=1)
    if np.any(sample_sd <= np.finfo(np.float64).eps):
        raise ValueError("calibration overlap pool has a zero-variance rank")
    observed_z = (observed - means) / sample_sd
    test_z = (test - means[None, :]) / sample_sd[None, :]
    return FixedCalibrationOverlapNull(
        rank_grid=ranks,
        calibration_mean_by_rank=tuple(
            (rank, float(value)) for rank, value in zip(ranks, means)
        ),
        calibration_sd_by_rank=tuple(
            (rank, float(value)) for rank, value in zip(ranks, sample_sd)
        ),
        observed_standardized_by_rank=tuple(
            (rank, float(value)) for rank, value in zip(ranks, observed_z)
        ),
        test_max_standardized=np.max(test_z, axis=1),
        n_calibration_draws=int(calibration.shape[0]),
        n_test_draws=int(test.shape[0]),
        calibration_pool_sha256=calibration_digest,
        test_pool_sha256=test_digest,
        calibration_draw_identities=calibration_identities,
        test_draw_identities=test_identities,
    )


def _stable_row_l2_norm(rows: np.ndarray) -> np.ndarray:
    """Compute final-axis L2 norms without treating tiny finite rows as zero."""

    scale = np.max(np.abs(rows), axis=-1)
    scaled = np.divide(
        rows,
        scale[..., None],
        out=np.zeros_like(rows),
        where=scale[..., None] != 0.0,
    )
    norm = scale * np.sqrt(np.sum(np.square(scaled), axis=-1))
    if not np.isfinite(norm).all():
        raise ValueError("row L2 energy is not representable as finite float64")
    return norm


def energy_match_rows(reference: Any, candidate: Any, *, atol: float = 1e-12) -> np.ndarray:
    """Match each candidate direction to the corresponding reference L2 energy.

    Zero is classified exactly from the row entries.  In particular, a norm below ``atol``
    is still nonzero and is never silently zero-filled.  The legacy ``atol`` keyword remains
    accepted for caller compatibility but does not alter this exact-zero decision.
    """

    arrays = _same_shape_arrays(reference=reference, candidate=candidate)
    if arrays["reference"].ndim < 1 or not np.isfinite(float(atol)) or atol <= 0:
        raise ValueError("reference/candidate must have a vector axis and atol must be positive")
    reference_zero = np.all(arrays["reference"] == 0.0, axis=-1)
    candidate_zero = np.all(arrays["candidate"] == 0.0, axis=-1)
    impossible = ~reference_zero & candidate_zero
    if np.any(impossible):
        raise ValueError(
            "cannot energy-match an exact-zero candidate (zero candidate row) "
            "to a nonzero reference"
        )

    reference_norm = _stable_row_l2_norm(arrays["reference"])
    candidate_scale = np.max(np.abs(arrays["candidate"]), axis=-1)
    scaled_candidate = np.divide(
        arrays["candidate"],
        candidate_scale[..., None],
        out=np.zeros_like(arrays["candidate"]),
        where=candidate_scale[..., None] != 0.0,
    )
    scaled_candidate_norm = np.sqrt(
        np.sum(np.square(scaled_candidate), axis=-1)
    )
    direction = np.divide(
        scaled_candidate,
        scaled_candidate_norm[..., None],
        out=np.zeros_like(scaled_candidate),
        where=scaled_candidate_norm[..., None] != 0.0,
    )
    matched = direction * reference_norm[..., None]
    matched = np.where(reference_zero[..., None], 0.0, matched)
    if not np.isfinite(matched).all():
        raise ValueError("energy-matched row is not representable as finite float64")
    return matched


def random_rank_energy_control(
    rows: Any,
    reference_projection: Any,
    *,
    rank: int,
    seed: int,
    namespace: str,
    draw: int,
) -> dict[str, np.ndarray]:
    """Build a rank-matched random projection and row-energy-match it to a reference."""

    arrays = _same_shape_arrays(rows=rows, reference_projection=reference_projection)
    if arrays["rows"].ndim != 2:
        raise ValueError("rows and reference_projection must be two-dimensional")
    basis = deterministic_random_basis(
        arrays["rows"].shape[1],
        rank,
        seed=seed,
        namespace=namespace,
        draw=draw,
    )
    projector = orthogonal_projector(basis)
    raw = project_rows(arrays["rows"], projector)
    matched = energy_match_rows(arrays["reference_projection"], raw)
    return {
        "basis": basis,
        "projector": projector,
        "raw_projection": raw,
        "energy_matched_projection": matched,
    }


def select_linear_mlm_margin_inputs(
    decoder_weight: Any,
    decoder_bias: Any,
    base_token_ids: Mapping[str, int],
) -> dict[str, Any]:
    """Select exact A/C/G/T rows from the MLM decoder's linear layer.

    The caller must supply the output of the checkpoint's MLM transform as the linear input.
    Calling a raw transformer residual a linear input is incorrect for BERT-style MLM heads.
    """

    weight = _finite_array("decoder_weight", decoder_weight)
    bias = _finite_array("decoder_bias", decoder_bias)
    if weight.ndim != 2 or bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
        raise ValueError("decoder_weight must be [vocab, width] and decoder_bias [vocab]")
    if set(base_token_ids) != set(BASE_ORDER):
        raise ValueError(f"base_token_ids must have exactly the keys {BASE_ORDER}")
    raw_ids = tuple(base_token_ids[base] for base in BASE_ORDER)
    if any(not _is_integral_not_bool(token_id) for token_id in raw_ids):
        raise TypeError("base token IDs must be integer decoder rows, not booleans")
    ids = tuple(int(token_id) for token_id in raw_ids)
    if len(set(ids)) != len(ids) or any(index < 0 or index >= weight.shape[0] for index in ids):
        raise ValueError("base token IDs must be unique valid decoder rows")
    index = np.asarray(ids, dtype=np.int64)
    return {
        "base_order": BASE_ORDER,
        "base_token_ids": ids,
        "weight": weight[index].copy(),
        "bias": bias[index].copy(),
    }


def linear_mlm_base_logits(
    linear_input: Any,
    decoder_weight: Any,
    decoder_bias: Any,
    base_token_ids: Mapping[str, int],
) -> np.ndarray:
    """Compute exact A/C/G/T logits from the MLM transform output."""

    hidden = _finite_array("linear_input", linear_input)
    selected = select_linear_mlm_margin_inputs(
        decoder_weight,
        decoder_bias,
        base_token_ids,
    )
    if hidden.ndim < 1 or hidden.shape[-1] != selected["weight"].shape[1]:
        raise ValueError("linear_input width does not match the decoder")
    return hidden @ selected["weight"].T + selected["bias"]


def hard_g_margin_from_base_logits(base_logits: Any) -> np.ndarray | float:
    """Return ``logit(G) - max(logit(A), logit(C), logit(T))`` from A/C/G/T logits."""

    logits = _finite_array("base_logits", base_logits)
    if logits.ndim < 1 or logits.shape[-1] != len(BASE_ORDER):
        raise ValueError("base_logits must end in the ordered A/C/G/T axis")
    g_index = BASE_ORDER.index("G")
    alternatives = [BASE_ORDER.index(base) for base in ("A", "C", "T")]
    margin = logits[..., g_index] - np.max(logits[..., alternatives], axis=-1)
    return _result(margin)


def linear_mlm_g_margin(
    linear_input: Any,
    decoder_weight: Any,
    decoder_bias: Any,
    base_token_ids: Mapping[str, int],
) -> np.ndarray | float:
    """Compute the hard masked-G margin through the selected linear decoder rows."""

    return hard_g_margin_from_base_logits(
        linear_mlm_base_logits(
            linear_input,
            decoder_weight,
            decoder_bias,
            base_token_ids,
        )
    )


def l3_to_u_mediation_estimands(
    *,
    m_minus: Any,
    m_l3_plus: Any,
    m_minus_with_u_from_l3_plus: Any,
    m_l3_plus_with_u_from_minus: Any,
    leg_id: str,
) -> dict[str, Any]:
    """Return the frozen L3H1 -> ``U_A`` sufficiency/necessity estimands.

    ``m_minus`` is the native paired-donor-corrupt margin. ``m_l3_plus`` is that same run
    after restoring clean L3H1.  The other cells interchange only the ``U_A`` component of
    the final masked-acceptor residual:

    ``S_U = m(P_U[h^{L3+}] into h-) - m-``
    ``N_U = m(h^{L3+}) - m(replace its U_A component with h-)``.
    """

    if not isinstance(leg_id, str) or not leg_id:
        raise ValueError("leg_id must be a non-empty string")
    cells = _same_shape_arrays(
        m_minus=m_minus,
        m_l3_plus=m_l3_plus,
        m_minus_with_u_from_l3_plus=m_minus_with_u_from_l3_plus,
        m_l3_plus_with_u_from_minus=m_l3_plus_with_u_from_minus,
    )
    estimands = {
        "direct_l3_rescue": cells["m_l3_plus"] - cells["m_minus"],
        "S_U": cells["m_minus_with_u_from_l3_plus"] - cells["m_minus"],
        "N_U": cells["m_l3_plus"] - cells["m_l3_plus_with_u_from_minus"],
    }
    return {
        "leg_id": leg_id,
        "cells": {name: _result(value) for name, value in cells.items()},
        "estimands": {name: _result(value) for name, value in estimands.items()},
    }


def _require_exact_record_keys(
    name: str,
    value: Any,
    expected: Sequence[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    expected_set = set(expected)
    if set(value) != expected_set:
        raise ValueError(
            f"{name} must have exactly the keys {tuple(expected)}"
        )
    return value


def _require_exact_replayed_numeric(
    name: str,
    stored: Any,
    replayed: Any,
) -> None:
    stored_array = _finite_array(f"stored {name}", stored)
    replayed_array = _finite_array(f"replayed {name}", replayed)
    if (
        stored_array.shape != replayed_array.shape
        or _array_sha256(stored_array) != _array_sha256(replayed_array)
    ):
        raise ValueError(
            f"stored {name} disagrees with exact primitive-cell replay"
        )


def replay_l3_to_u_mediation_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonically replay an L3 -> ``U_A`` record from primitive cells.

    The supplied record must have the exact constructor schema.  Stored estimands are
    treated only as redundancy checks: all three are recomputed from the four primitive
    margin cells and every stored value must match the float64 replay exactly.
    """

    top = _require_exact_record_keys(
        "L3-to-U record",
        record,
        ("leg_id", "cells", "estimands"),
    )
    cells = _require_exact_record_keys(
        "L3-to-U cells",
        top["cells"],
        (
            "m_minus",
            "m_l3_plus",
            "m_minus_with_u_from_l3_plus",
            "m_l3_plus_with_u_from_minus",
        ),
    )
    stored_estimands = _require_exact_record_keys(
        "L3-to-U estimands",
        top["estimands"],
        ("direct_l3_rescue", "S_U", "N_U"),
    )
    replayed = l3_to_u_mediation_estimands(
        m_minus=cells["m_minus"],
        m_l3_plus=cells["m_l3_plus"],
        m_minus_with_u_from_l3_plus=cells["m_minus_with_u_from_l3_plus"],
        m_l3_plus_with_u_from_minus=cells["m_l3_plus_with_u_from_minus"],
        leg_id=top["leg_id"],
    )
    for name, value in replayed["estimands"].items():
        _require_exact_replayed_numeric(
            f"L3-to-U estimand {name}",
            stored_estimands[name],
            value,
        )
    return replayed


def delivery_sandwich_estimands(
    *,
    m_minus: Any,
    m_c_plus: Any,
    m_c_plus_with_d_minus: Any,
    m_minus_with_d_from_c_plus: Any,
    leg_id: str,
    component_id: str,
) -> dict[str, Any]:
    """Return the conditional downstream-delivery necessity/sufficiency sandwich.

    ``c+`` is the paired-donor-corrupt run with clean L3H1 restored.  ``d-`` is downstream
    component ``d`` captured under the native corrupt run, and ``d^{c+}`` is captured under
    ``c+``.  The row-level estimands are

    ``N_d = m(c+) - m(c+ with d <- d-)`` and
    ``S_d = m(m- run with d <- d^{c+}) - m-``.

    They remain separate; this function does not turn their minimum, maximum, or average into
    a composite delivery score.
    """

    if not isinstance(leg_id, str) or not leg_id:
        raise ValueError("leg_id must be a non-empty string")
    if not isinstance(component_id, str) or not component_id:
        raise ValueError("component_id must be a non-empty string")
    cells = _same_shape_arrays(
        m_minus=m_minus,
        m_c_plus=m_c_plus,
        m_c_plus_with_d_minus=m_c_plus_with_d_minus,
        m_minus_with_d_from_c_plus=m_minus_with_d_from_c_plus,
    )
    estimands = {
        "direct_l3_rescue": cells["m_c_plus"] - cells["m_minus"],
        "N_d": cells["m_c_plus"] - cells["m_c_plus_with_d_minus"],
        "S_d": cells["m_minus_with_d_from_c_plus"] - cells["m_minus"],
    }
    return {
        "leg_id": leg_id,
        "component_id": component_id,
        "cells": {name: _result(value) for name, value in cells.items()},
        "estimands": {name: _result(value) for name, value in estimands.items()},
    }


def replay_delivery_sandwich_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonically replay a delivery sandwich from primitive cells.

    All derived rescue, necessity, and sufficiency values are recomputed.  The stored
    estimand block is accepted only when every value exactly agrees with that replay.
    """

    top = _require_exact_record_keys(
        "delivery sandwich record",
        record,
        ("leg_id", "component_id", "cells", "estimands"),
    )
    cells = _require_exact_record_keys(
        "delivery sandwich cells",
        top["cells"],
        (
            "m_minus",
            "m_c_plus",
            "m_c_plus_with_d_minus",
            "m_minus_with_d_from_c_plus",
        ),
    )
    stored_estimands = _require_exact_record_keys(
        "delivery sandwich estimands",
        top["estimands"],
        ("direct_l3_rescue", "N_d", "S_d"),
    )
    replayed = delivery_sandwich_estimands(
        m_minus=cells["m_minus"],
        m_c_plus=cells["m_c_plus"],
        m_c_plus_with_d_minus=cells["m_c_plus_with_d_minus"],
        m_minus_with_d_from_c_plus=cells["m_minus_with_d_from_c_plus"],
        leg_id=top["leg_id"],
        component_id=top["component_id"],
    )
    for name, value in replayed["estimands"].items():
        _require_exact_replayed_numeric(
            f"delivery sandwich estimand {name}",
            stored_estimands[name],
            value,
        )
    return replayed


def delivery_candidate_route_gaps(
    candidate: Mapping[str, Any],
    donor_route: Mapping[str, Any],
) -> dict[str, float | np.ndarray]:
    """Compare a candidate and donor-route comparator on identical sandwich legs."""

    replayed_candidate = replay_delivery_sandwich_record(candidate)
    replayed_donor_route = replay_delivery_sandwich_record(donor_route)
    if (
        replayed_candidate["leg_id"] != replayed_donor_route["leg_id"]
        or not replayed_candidate["leg_id"]
    ):
        raise ValueError("candidate and donor-route sandwiches must share a non-empty leg_id")
    for name in ("m_minus", "m_c_plus"):
        left = _finite_array(
            f"candidate.{name}", replayed_candidate["cells"][name]
        )
        right = _finite_array(
            f"donor_route.{name}", replayed_donor_route["cells"][name]
        )
        if left.shape != right.shape or _array_sha256(left) != _array_sha256(right):
            raise ValueError(f"candidate and donor-route sandwiches do not share the {name} leg")
    c_est = replayed_candidate["estimands"]
    d_est = replayed_donor_route["estimands"]
    arrays = _same_shape_arrays(
        candidate_N=c_est.get("N_d"),
        donor_route_N=d_est.get("N_d"),
        candidate_S=c_est.get("S_d"),
        donor_route_S=d_est.get("S_d"),
    )
    return {
        "N_candidate_minus_donor_route": _result(
            arrays["candidate_N"] - arrays["donor_route_N"]
        ),
        "S_candidate_minus_donor_route": _result(
            arrays["candidate_S"] - arrays["donor_route_S"]
        ),
    }


__all__ = [
    "AV_CELL_ORDER",
    "BASE_ORDER",
    "DONOR_WINDOW_OFFSETS",
    "SVD_NUMERICAL_RANK_RELATIVE_TOLERANCE",
    "SVD_NUMERICAL_RANK_RULE",
    "CrossFitFold",
    "CrossFitOutcomeRankSelection",
    "CrossFitSVD",
    "FixedCalibrationOverlapNull",
    "OverlapNullDrawIdentity",
    "SVDGridFit",
    "av_factorial_contrasts",
    "av_four_cell_reconstruction",
    "crossfit_project",
    "crossfit_svd",
    "delivery_candidate_route_gaps",
    "delivery_sandwich_estimands",
    "deterministic_group_folds",
    "deterministic_nested_random_bases",
    "deterministic_random_basis",
    "donor_window_indices",
    "donor_window_positions",
    "energy_match_rows",
    "fit_svd_grid",
    "fixed_calibration_overlap_null",
    "hard_g_margin_from_base_logits",
    "l3_to_u_mediation_estimands",
    "linear_mlm_base_logits",
    "linear_mlm_g_margin",
    "mean_squared_principal_cosine",
    "orthogonal_projector",
    "ordered_overlap_null_draw_identities",
    "partner_pseudo_specific",
    "project_rows",
    "projector_diagnostics",
    "random_rank_energy_control",
    "reconstruct_attention_value",
    "replay_delivery_sandwich_record",
    "replay_l3_to_u_mediation_record",
    "replace_projected_component",
    "select_crossfit_outcome_rank",
    "select_linear_mlm_margin_inputs",
    "validate_svd_numerical_rank",
]
