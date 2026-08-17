#!/usr/bin/env python
"""R5a: acceptor-only paired-donor reader, address, and value experiment.

The only behavioral readout constructed by this program is the native masked-acceptor
``G`` margin.  QK logits and attention probabilities are internal mechanism readouts;
they are never substituted for the behavioral endpoint.

The checked-in protocol is deliberately DRAFT.  ``--mode model-free-audit`` is the only
mode available while any literal freeze/census pin is ``None``.  In particular, a target
request is rejected before resolving the model, GTF, FASTA, or preflight paths.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.circuits.hf_bert import (  # noqa: E402
    _merge_context,
    edit_heads,
    patch_heads,
)
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.acceptor_coupling_mechanism import (  # noqa: E402
    AV_CELL_ORDER,
    av_factorial_contrasts,
    av_four_cell_reconstruction,
    donor_window_positions,
)
from gfmcd.splicing.acceptor_mechanism_cohort import (  # noqa: E402
    R5_ADJACENT_SALT,
    R5_DONOR_PSEUDO_CALIPER,
    R5_EXPECTED_CENSUS_IDENTITY,
    R5_EXPECTED_SOURCE_IDENTITY,
    R5_ROWS_PER_BUCKET,
    R5_VALIDATION_CHROMOSOMES,
    build_r5_reservations,
    r5_census_identity,
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
from gfmcd.splicing.paired_component import (  # noqa: E402
    boundary_sign_flip_tests,
    holm_adjust_named,
    intersection_union_pvalue,
    one_sided_binomial_bounds,
    simultaneous_family_decision,
    strict_interval_decision,
)
from gfmcd.splicing.provenance import (  # noqa: E402
    capture_embed_components,
    clamp_embed_component,
)
from gfmcd.splicing.r5_stack_binding import (  # noqa: E402
    R3C_ENTRY_SCHEMA,
    R3C_INSTRUMENT_REVISION,
    R5StackBindingError,
    build_r5_stack_binding,
    r5_stack_fingerprint,
    stable_runtime_identity,
    validate_r3c_target_health_contract,
    validate_r3c_target_manifest_contract,
    validate_serialized_r5_stack,
)


RUNNER = Path(__file__).resolve()
PROTOCOL = (
    REPO
    / "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5A_ACCEPTOR_ADDRESS_BINDING.md"
)
R5B_PROTOCOL = (
    REPO
    / "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5B_ACCEPTOR_SIDE_STATE.md"
)
R5C_PROTOCOL = (
    REPO
    / "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5C_ACCEPTOR_WRITE_DELIVERY.md"
)
R3C_RUNNER = (
    REPO
    / "scripts/splicebert_acceptor_coupling/"
    "run_r3c_acceptor_head_confirmation.py"
)
MECHANISM_CORE = REPO / "src/gfmcd/splicing/acceptor_coupling_mechanism.py"
COHORT_CORE = REPO / "src/gfmcd/splicing/acceptor_mechanism_cohort.py"
STAT_CORE = REPO / "src/gfmcd/splicing/paired_component.py"
HOOK_CORE = REPO / "src/gfmcd/circuits/hf_bert.py"
EMBED_CORE = REPO / "src/gfmcd/splicing/provenance.py"
ASSAY_CORE = REPO / "src/gfmcd/splicing/reverse_coupling.py"
CIRCUIT_CORE = REPO / "src/gfmcd/splicing/circuit.py"

# Literal nonbinding values.  A reviewed freeze replaces every value together; model
# scoring cannot be requested while even one remains absent.  The model-free census is
# intentionally the only route that can populate the future cohort pins.
FROZEN_PROTOCOL_SHA256: str | None = None
FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = None
FROZEN_PREFLIGHT_ARTIFACT_SHA256: str | None = None
FROZEN_R3C_ARTIFACT_SHA256: str | None = None
FROZEN_CHECKPOINT_AGGREGATE_SHA256: str | None = None
FROZEN_COMPLETE_ELIGIBLE_GENES: int | None = None
FROZEN_BUCKET_ELIGIBLE_GENES: tuple[int, int] | None = None
FROZEN_SELECTED_COHORT_SHA256: tuple[str, str] | None = None
FROZEN_CONSTRUCTION_SHA256: tuple[str, str] | None = None
FROZEN_EXCLUDED_GENE_REGISTRY_SHA256: str | None = None

PRODUCTION_N = 128
PRODUCTION_BATCH = 8
PRODUCTION_SEED = 0
PRODUCTION_RANDOMISATION = 4000
PRODUCTION_CALIBRATION_TRIALS = 465
MIN_GENES = 128
SESOI = 0.05
STAGE_ALPHA = 0.05
FAMILY_ALPHA = STAGE_ALPHA / 4.0
DECOY_TOL = R5_DONOR_PSEUDO_CALIPER

READER = (3, 1)
ADDRESS_BUILDER = (2, 15)
NEGATIVE_HEAD = (0, 2)
SHIFT_HEADS = ("L3H1", "L0H2")
SHIFT_COMPARISONS = ("paired_adjacent", "pseudo_adjacent")
SHIFT_MEASURES = ("qk_preference", "attention_preference", "acceptor_margin")
SHIFT_CELLS = ("full", "word_shift", "position_shift", "joint_shift")
SHIFT_QUANTITIES = (
    "position_effect_at_native_content",
    "position_effect_at_shifted_content",
    "content_effect_at_native_position",
    "content_effect_at_shifted_position",
    "interaction_position_x_content",
)

QK_ARMS = (
    ("L3H1", "paired"),
    ("L3H1", "pseudo"),
    ("L3H1", "adjacent"),
    ("L0H2", "paired"),
)
WINDOWS = ("paired", "pseudo", "adjacent")
ENDPOINTS = ("partner", "pseudo", "specific")
AV_QUANTITIES = (
    "baseline",
    "address_only",
    "value_only",
    "joint",
    "joint_lift",
    "address_effect_at_value_minus",
    "value_effect_at_address_minus",
    "value_effect_at_address_plus",
    "address_effect_at_value_plus",
    "interaction_AxV",
)
BUILDERS = ("L2H15", "L0H2")

FAMILY_ORDER = ("position_shift", "qk_surgery", "attention_value", "l2_role")
PATH_ORDER = (
    "P_POSITION_SENSITIVE_L3_READER",
    "P_POSITION_DOMINANT_L3_READER",
    "P_QK_L3_READER",
    "P_AV_DONOR_WINDOW_VALUE",
    "P_L2H15_ADDRESS",
    "P_L2H15_MIXED_AV",
)

HOOK_ATOL = 1e-5
MARGIN_ATOL = 1e-4
CALIBRATION_EPSILON = 0.01
CALIBRATION_MAX_FALSE_POSITIVES = 5
CALIBRATION_MIN_POWER_SUCCESSES = 441
CALIBRATION_PRIMITIVE_NOISE_SD = 0.025
HEALTH_MAX_EXAMPLES = 16
HEALTH_PROBES_PER_EXAMPLE = 4
HEALTH_FLOOR = 0.50


class SealError(RuntimeError):
    """Raised before any protected path is resolved or opened."""


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(f"float64|{','.join(map(str, array.shape))}|".encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def head_tuple(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"L(\d+)H(\d+)", name)
    if match is None:
        raise ValueError(f"invalid head name: {name!r}")
    return int(match.group(1)), int(match.group(2))


def position_row_name(head: str, comparison: str, measure: str, quantity: str) -> str:
    return (
        f"POSITION|head={head}|comparison={comparison}|measure={measure}|"
        f"quantity={quantity}"
    )


def qk_row_name(head: str, window: str, endpoint: str) -> str:
    return f"QK|head={head}|window={window}|endpoint={endpoint}"


def av_row_name(window: str, quantity: str, endpoint: str) -> str:
    return f"AV|window={window}|quantity={quantity}|endpoint={endpoint}"


def l2_row_name(builder: str, quantity: str, endpoint: str) -> str:
    return f"L2|builder={builder}|quantity={quantity}|endpoint={endpoint}"


def family_row_names() -> dict[str, list[str]]:
    return {
        "position_shift": [
            position_row_name(head, comparison, measure, quantity)
            for head in SHIFT_HEADS
            for comparison in SHIFT_COMPARISONS
            for measure in SHIFT_MEASURES
            for quantity in SHIFT_QUANTITIES
        ],
        "qk_surgery": [
            qk_row_name(head, window, endpoint)
            for head, window in QK_ARMS
            for endpoint in ENDPOINTS
        ],
        "attention_value": [
            av_row_name(window, quantity, endpoint)
            for window in WINDOWS
            for quantity in AV_QUANTITIES
            for endpoint in ENDPOINTS
        ],
        "l2_role": [
            l2_row_name(builder, quantity, endpoint)
            for builder in BUILDERS
            for quantity in AV_QUANTITIES
            for endpoint in ENDPOINTS
        ],
    }


def _requirement(
    output: dict[str, str], row: str, kind: str
) -> None:
    if kind not in {"MATERIAL_POSITIVE", "EQUIVALENT_WITHIN_SESOI"}:
        raise ValueError(f"invalid requirement: {kind}")
    previous = output.setdefault(row, kind)
    if previous != kind:
        raise RuntimeError(f"conflicting requirements for {row}")


def named_path_specs() -> dict[str, dict[str, Any]]:
    position: dict[str, str] = {}
    for measure in SHIFT_MEASURES:
        for quantity in SHIFT_QUANTITIES[:2]:
            _requirement(
                position,
                position_row_name("L3H1", "paired_adjacent", measure, quantity),
                "MATERIAL_POSITIVE",
            )
    for head, comparison in (
        ("L3H1", "pseudo_adjacent"),
        ("L0H2", "paired_adjacent"),
        ("L0H2", "pseudo_adjacent"),
    ):
        for measure in SHIFT_MEASURES:
            for quantity in SHIFT_QUANTITIES[:2]:
                _requirement(
                    position,
                    position_row_name(head, comparison, measure, quantity),
                    "EQUIVALENT_WITHIN_SESOI",
                )

    # Position sensitivity does not imply content irrelevance.  The stronger nested path
    # adds the two content simple effects at the native and shifted position assignments.
    # The interaction remains a corrected/reportable row but is not a logical prerequisite
    # for either reading because the embedding LayerNorm makes additivity non-structural.
    position_dominant = dict(position)
    for measure in SHIFT_MEASURES:
        for quantity in SHIFT_QUANTITIES[2:4]:
            _requirement(
                position_dominant,
                position_row_name("L3H1", "paired_adjacent", measure, quantity),
                "EQUIVALENT_WITHIN_SESOI",
            )

    qk: dict[str, str] = {}
    for endpoint in ("partner", "specific"):
        _requirement(qk, qk_row_name("L3H1", "paired", endpoint), "MATERIAL_POSITIVE")
    _requirement(qk, qk_row_name("L3H1", "paired", "pseudo"), "EQUIVALENT_WITHIN_SESOI")
    for head, window in QK_ARMS[1:]:
        for endpoint in ENDPOINTS:
            _requirement(qk, qk_row_name(head, window, endpoint), "EQUIVALENT_WITHIN_SESOI")

    av: dict[str, str] = {}
    for quantity in ("joint_lift", "value_effect_at_address_plus"):
        for endpoint in ("partner", "specific"):
            _requirement(av, av_row_name("paired", quantity, endpoint), "MATERIAL_POSITIVE")
        _requirement(av, av_row_name("paired", quantity, "pseudo"), "EQUIVALENT_WITHIN_SESOI")
        for window in ("pseudo", "adjacent"):
            for endpoint in ENDPOINTS:
                _requirement(av, av_row_name(window, quantity, endpoint), "EQUIVALENT_WITHIN_SESOI")

    l2_address: dict[str, str] = {}
    for quantity in ("address_effect_at_value_plus", "joint_lift"):
        for endpoint in ("partner", "specific"):
            _requirement(
                l2_address,
                l2_row_name("L2H15", quantity, endpoint),
                "MATERIAL_POSITIVE",
            )
        _requirement(
            l2_address,
            l2_row_name("L2H15", quantity, "pseudo"),
            "EQUIVALENT_WITHIN_SESOI",
        )
    for endpoint in ENDPOINTS:
        _requirement(
            l2_address,
            l2_row_name("L2H15", "value_effect_at_address_plus", endpoint),
            "EQUIVALENT_WITHIN_SESOI",
        )
    for quantity in AV_QUANTITIES:
        for endpoint in ENDPOINTS:
            _requirement(
                l2_address,
                l2_row_name("L0H2", quantity, endpoint),
                "EQUIVALENT_WITHIN_SESOI",
            )

    l2_mixed: dict[str, str] = {}
    for quantity in (
        "address_effect_at_value_plus",
        "value_effect_at_address_plus",
        "joint_lift",
    ):
        for endpoint in ("partner", "specific"):
            _requirement(
                l2_mixed,
                l2_row_name("L2H15", quantity, endpoint),
                "MATERIAL_POSITIVE",
            )
        _requirement(
            l2_mixed,
            l2_row_name("L2H15", quantity, "pseudo"),
            "EQUIVALENT_WITHIN_SESOI",
        )
    for quantity in AV_QUANTITIES:
        for endpoint in ENDPOINTS:
            _requirement(
                l2_mixed,
                l2_row_name("L0H2", quantity, endpoint),
                "EQUIVALENT_WITHIN_SESOI",
            )

    output = {
        "P_POSITION_SENSITIVE_L3_READER": {
            "family": "position_shift",
            "requirements": position,
        },
        "P_POSITION_DOMINANT_L3_READER": {
            "family": "position_shift",
            "requirements": position_dominant,
        },
        "P_QK_L3_READER": {"family": "qk_surgery", "requirements": qk},
        "P_AV_DONOR_WINDOW_VALUE": {
            "family": "attention_value",
            "requirements": av,
        },
        "P_L2H15_ADDRESS": {"family": "l2_role", "requirements": l2_address},
        "P_L2H15_MIXED_AV": {"family": "l2_role", "requirements": l2_mixed},
    }
    if tuple(output) != PATH_ORDER:
        raise RuntimeError("R5a path order drifted")
    rows = family_row_names()
    for path, spec in output.items():
        if not set(spec["requirements"]).issubset(rows[spec["family"]]):
            raise RuntimeError(f"{path} contains a row outside its frozen family")
    return output


def shift_factorial_contrasts(cells: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if tuple(cells) != SHIFT_CELLS:
        raise ValueError(f"shift cells must have exact order {SHIFT_CELLS}")
    arrays = {name: np.asarray(cells[name], dtype=float) for name in SHIFT_CELLS}
    if len({value.shape for value in arrays.values()}) != 1:
        raise ValueError("shift cells must have identical shapes")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("shift cells contain non-finite values")
    full = arrays["full"]
    word = arrays["word_shift"]
    position = arrays["position_shift"]
    joint = arrays["joint_shift"]
    return {
        "position_effect_at_native_content": full - position,
        "position_effect_at_shifted_content": word - joint,
        "content_effect_at_native_position": full - word,
        "content_effect_at_shifted_position": position - joint,
        "interaction_position_x_content": full - word - position + joint,
    }


def endpoint_vectors(partner: Any, pseudo: Any) -> dict[str, np.ndarray]:
    partner_array = np.asarray(partner, dtype=float)
    pseudo_array = np.asarray(pseudo, dtype=float)
    if partner_array.shape != pseudo_array.shape or partner_array.ndim != 1:
        raise ValueError("partner and pseudo must be shape-matched vectors")
    if not np.isfinite(partner_array).all() or not np.isfinite(pseudo_array).all():
        raise ValueError("endpoint vectors contain non-finite values")
    return {
        "partner": partner_array,
        "pseudo": pseudo_array,
        "specific": partner_array - pseudo_array,
    }


def factorial_quantity_vectors(cells: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if tuple(cells) != AV_CELL_ORDER:
        raise ValueError(f"A x V cells must have exact order {AV_CELL_ORDER}")
    arrays = {name: np.asarray(cells[name], dtype=float) for name in AV_CELL_ORDER}
    if len({array.shape for array in arrays.values()}) != 1:
        raise ValueError("A x V cells must be shape matched")
    contrasts = av_factorial_contrasts(arrays)
    return arrays | {name: np.asarray(value, dtype=float) for name, value in contrasts.items()}


def build_family_matrices(measurements: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Derive all four immutable families from raw intervention cells.

    ``specific`` and every factorial contrast are rebuilt here.  Callers cannot supply or
    overwrite a derived row, which makes stored primitive cells sufficient for exact replay.
    """

    row_names = family_row_names()
    output: dict[str, np.ndarray] = {}

    position_by_name: dict[str, np.ndarray] = {}
    for head in SHIFT_HEADS:
        for comparison in SHIFT_COMPARISONS:
            for measure in SHIFT_MEASURES:
                cells = measurements["position_shift"][head][comparison][measure]
                for quantity, values in shift_factorial_contrasts(cells).items():
                    position_by_name[
                        position_row_name(head, comparison, measure, quantity)
                    ] = values
    output["position_shift"] = np.vstack(
        [position_by_name[name] for name in row_names["position_shift"]]
    )

    qk_by_name: dict[str, np.ndarray] = {}
    for head, window in QK_ARMS:
        endpoints = endpoint_vectors(
            measurements["qk_surgery"][head][window]["partner"],
            measurements["qk_surgery"][head][window]["pseudo"],
        )
        for endpoint, values in endpoints.items():
            qk_by_name[qk_row_name(head, window, endpoint)] = values
    output["qk_surgery"] = np.vstack(
        [qk_by_name[name] for name in row_names["qk_surgery"]]
    )

    av_by_name: dict[str, np.ndarray] = {}
    for window in WINDOWS:
        by_leg = {
            leg: factorial_quantity_vectors(
                measurements["attention_value"][window][leg]
            )
            for leg in ("partner", "pseudo")
        }
        for quantity in AV_QUANTITIES:
            endpoints = endpoint_vectors(
                by_leg["partner"][quantity], by_leg["pseudo"][quantity]
            )
            for endpoint, values in endpoints.items():
                av_by_name[av_row_name(window, quantity, endpoint)] = values
    output["attention_value"] = np.vstack(
        [av_by_name[name] for name in row_names["attention_value"]]
    )

    l2_by_name: dict[str, np.ndarray] = {}
    for builder in BUILDERS:
        by_leg = {
            leg: factorial_quantity_vectors(measurements["l2_role"][builder][leg])
            for leg in ("partner", "pseudo")
        }
        for quantity in AV_QUANTITIES:
            endpoints = endpoint_vectors(
                by_leg["partner"][quantity], by_leg["pseudo"][quantity]
            )
            for endpoint, values in endpoints.items():
                l2_by_name[l2_row_name(builder, quantity, endpoint)] = values
    output["l2_role"] = np.vstack(
        [l2_by_name[name] for name in row_names["l2_role"]]
    )

    sizes = {"position_shift": 60, "qk_surgery": 12, "attention_value": 90, "l2_role": 60}
    for family, expected in sizes.items():
        if output[family].shape[0] != expected:
            raise RuntimeError(f"{family} family row count drifted")
        if not np.isfinite(output[family]).all():
            raise RuntimeError(f"{family} family contains non-finite values")
    return output


def strict_family_inference(
    values: np.ndarray,
    groups: Sequence[str],
    names: Sequence[str],
    *,
    family: str,
    n_draws: int = PRODUCTION_RANDOMISATION,
    seed: int = PRODUCTION_SEED,
    namespace: str = "r5a",
    min_genes: int = MIN_GENES,
) -> dict[str, Any]:
    """Run boundary/TOST -> IUT -> Holm plus strict full-family CI agreement."""

    if family not in FAMILY_ORDER:
        raise ValueError(f"unknown R5a family: {family}")
    matrix = np.asarray(values, dtype=float)
    expected_names = family_row_names()[family]
    if list(names) != expected_names or matrix.shape != (len(names), len(groups)):
        raise ValueError("strict inference requires the exact ordered R5a family")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")

    decisions, interval_audit = simultaneous_family_decision(
        matrix,
        groups,
        names,
        sesoi=SESOI,
        n_draws=n_draws,
        alpha=FAMILY_ALPHA,
        seed=seed,
        namespace=f"{namespace}:{family}:simultaneous-family",
        min_genes=min_genes,
    )
    family_paths = {
        name: spec
        for name, spec in named_path_specs().items()
        if spec["family"] == family
    }
    edge_keys: list[tuple[str, str, str]] = []
    edge_values: list[np.ndarray] = []
    boundaries: list[float] = []
    alternatives: list[str] = []
    index = {name: position for position, name in enumerate(names)}
    seen: set[tuple[str, str]] = set()
    for spec in family_paths.values():
        for row, requirement in spec["requirements"].items():
            key = (row, requirement)
            if key in seen:
                continue
            seen.add(key)
            if requirement == "MATERIAL_POSITIVE":
                edge_keys.append((row, requirement, "boundary"))
                edge_values.append(matrix[index[row]])
                boundaries.append(SESOI)
                alternatives.append("greater")
            else:
                for edge, boundary, alternative in (
                    ("lower", -SESOI, "greater"),
                    ("upper", SESOI, "less"),
                ):
                    edge_keys.append((row, requirement, edge))
                    edge_values.append(matrix[index[row]])
                    boundaries.append(boundary)
                    alternatives.append(alternative)
    edge_names = [f"{row}|requirement={kind}|edge={edge}" for row, kind, edge in edge_keys]
    tests = boundary_sign_flip_tests(
        np.vstack(edge_values),
        groups,
        edge_names,
        boundaries=boundaries,
        alternatives=alternatives,  # type: ignore[arg-type]
        n_draws=n_draws,
        seed=seed,
        namespace=f"{namespace}:{family}:shared-boundary-edges",
        min_genes=min_genes,
    )
    by_requirement: dict[tuple[str, str], list[Any]] = {}
    for key, name in zip(edge_keys, edge_names):
        by_requirement.setdefault(key[:2], []).append(tests[name])

    raw_path_p: dict[str, float] = {}
    for path, spec in family_paths.items():
        raw_path_p[path] = intersection_union_pvalue(
            [
                max(test.p_value for test in by_requirement[(row, requirement)])
                for row, requirement in spec["requirements"].items()
            ]
        )
    adjusted = holm_adjust_named(raw_path_p)
    paths: dict[str, Any] = {}
    for path, spec in family_paths.items():
        requirement_diagnostics = {
            row: {
                "required": requirement,
                "strict_simultaneous_ci_classification": strict_interval_decision(
                    *decisions[row].simultaneous_ci,
                    SESOI,
                ),
            }
            for row, requirement in spec["requirements"].items()
        }
        ci_pass = all(
            item["strict_simultaneous_ci_classification"] == item["required"]
            for item in requirement_diagnostics.values()
        )
        p_pass = adjusted[path] < FAMILY_ALPHA
        active_equivalence_rows = [
            row
            for row, item in requirement_diagnostics.items()
            if item["required"] == "EQUIVALENT_WITHIN_SESOI"
            and item["strict_simultaneous_ci_classification"]
            in {"MATERIAL_POSITIVE", "MATERIAL_NEGATIVE"}
        ]
        unresolved_rows = [
            row
            for row, item in requirement_diagnostics.items()
            if item["strict_simultaneous_ci_classification"] == "UNRESOLVED"
        ]
        inactive_material_rows = [
            row
            for row, item in requirement_diagnostics.items()
            if item["required"] == "MATERIAL_POSITIVE"
            and item["strict_simultaneous_ci_classification"]
            in {"EQUIVALENT_WITHIN_SESOI", "MATERIAL_NEGATIVE"}
        ]
        if p_pass and ci_pass:
            failure_mode = None
        elif active_equivalence_rows:
            failure_mode = (
                "CONTENT_ALSO_MATERIAL"
                if path == "P_POSITION_DOMINANT_L3_READER"
                else "NONSELECTIVE_OR_CONTROL_ACTIVE"
            )
        elif unresolved_rows:
            failure_mode = "SCIENTIFICALLY_UNRESOLVED"
        elif inactive_material_rows:
            failure_mode = "REQUIRED_TARGET_NOT_MATERIAL"
        else:
            failure_mode = "SCIENTIFICALLY_UNRESOLVED"
        paths[path] = {
            "family": family,
            "requirements": dict(spec["requirements"]),
            "iut_p_value": raw_path_p[path],
            "holm_adjusted_p_value": adjusted[path],
            "holm_family_size": len(family_paths),
            "family_alpha": FAMILY_ALPHA,
            "p_gate_pass": p_pass,
            "strict_simultaneous_ci_gate_pass": ci_pass,
            "requirement_diagnostics": requirement_diagnostics,
            "active_equivalence_rows": active_equivalence_rows,
            "unresolved_rows": unresolved_rows,
            "inactive_material_rows": inactive_material_rows,
            "failure_mode": failure_mode,
            "classification": "SUPPORTED" if p_pass and ci_pass else "NOT_SUPPORTED",
        }

    sign_digests = {test.sign_matrix_sha256 for test in tests.values()}
    gene_digests = {test.ordered_gene_labels_sha256 for test in tests.values()}
    if len(sign_digests) != 1 or len(gene_digests) != 1:
        raise RuntimeError("R5a decision edges did not share one gene sign matrix")
    return {
        "family": family,
        "row_names": list(names),
        "row_order_sha256": canonical_sha256(list(names)),
        "estimates": {
            name: float(decisions[name].estimate) for name in names
        },
        "simultaneous_decisions": {
            name: decisions[name].to_dict() for name in names
        },
        "interval_calibration": interval_audit,
        "boundary_tests": {
            name: test.to_dict() for name, test in tests.items()
        },
        "boundary_randomisation_audit": {
            "all_decision_edges_share_one_sign_matrix": True,
            "B": n_draws,
            "seed": seed,
            "namespace": f"{namespace}:{family}:shared-boundary-edges",
            "sign_matrix_sha256": next(iter(sign_digests)),
            "ordered_gene_labels_sha256": next(iter(gene_digests)),
            "edge_count": len(edge_names),
            "edge_order_sha256": canonical_sha256(edge_names),
        },
        "paths": paths,
    }


def infer_all_families(
    matrices: Mapping[str, np.ndarray],
    groups: Sequence[str],
    *,
    n_draws: int = PRODUCTION_RANDOMISATION,
    seed: int = PRODUCTION_SEED,
    namespace: str = "r5a",
    min_genes: int = MIN_GENES,
) -> dict[str, Any]:
    names = family_row_names()
    if tuple(matrices) != FAMILY_ORDER:
        raise ValueError(f"families must have exact order {FAMILY_ORDER}")
    result = {
        family: strict_family_inference(
            matrices[family],
            groups,
            names[family],
            family=family,
            n_draws=n_draws,
            seed=seed,
            namespace=namespace,
            min_genes=min_genes,
        )
        for family in FAMILY_ORDER
    }
    paths = {
        path: result[spec["family"]]["paths"][path]
        for path, spec in named_path_specs().items()
    }
    position_supported = (
        paths["P_POSITION_SENSITIVE_L3_READER"]["classification"] == "SUPPORTED"
    )
    position_dominant = (
        paths["P_POSITION_DOMINANT_L3_READER"]["classification"] == "SUPPORTED"
    )
    qk_supported = paths["P_QK_L3_READER"]["classification"] == "SUPPORTED"
    av_supported = (
        paths["P_AV_DONOR_WINDOW_VALUE"]["classification"] == "SUPPORTED"
    )
    address_supported = paths["P_L2H15_ADDRESS"]["classification"] == "SUPPORTED"
    mixed_supported = paths["P_L2H15_MIXED_AV"]["classification"] == "SUPPORTED"
    reader_paths = (
        paths["P_POSITION_SENSITIVE_L3_READER"],
        paths["P_QK_L3_READER"],
        paths["P_AV_DONOR_WINDOW_VALUE"],
    )
    if position_supported and qk_supported and av_supported:
        reader_decision = "L3H1_DONOR_LOCAL_READER_SUPPORTED"
    elif any(
        path["failure_mode"] == "NONSELECTIVE_OR_CONTROL_ACTIVE"
        for path in reader_paths
    ):
        reader_decision = "NONSELECTIVE_OR_CONTROL_ACTIVE"
    elif any(
        path["failure_mode"] == "SCIENTIFICALLY_UNRESOLVED"
        for path in reader_paths
    ):
        reader_decision = "SCIENTIFICALLY_UNRESOLVED"
    else:
        reader_decision = "L3H1_DONOR_LOCAL_READER_NOT_SUPPORTED"
    if address_supported:
        l2_decision = "L2H15_ADDRESS_REUSE_SUPPORTED"
    elif mixed_supported:
        l2_decision = "L2H15_MIXED_AV"
    elif any(
        paths[path]["failure_mode"] == "NONSELECTIVE_OR_CONTROL_ACTIVE"
        for path in ("P_L2H15_ADDRESS", "P_L2H15_MIXED_AV")
    ):
        l2_decision = "NONSELECTIVE_OR_CONTROL_ACTIVE"
    elif any(
        paths[path]["failure_mode"] == "SCIENTIFICALLY_UNRESOLVED"
        for path in ("P_L2H15_ADDRESS", "P_L2H15_MIXED_AV")
    ):
        l2_decision = "SCIENTIFICALLY_UNRESOLVED"
    else:
        l2_decision = "L2H15_ROLE_NOT_SUPPORTED"
    return {
        "families": result,
        "paths": paths,
        "decisions": {
            "position_sensitive_reader": (
                "POSITION_SENSITIVE_READER_SUPPORTED"
                if position_supported
                else "POSITION_SENSITIVE_READER_NOT_SUPPORTED"
            ),
            "position_dominance": (
                "POSITION_DOMINANT_READER_SUPPORTED"
                if position_dominant
                else "POSITION_DOMINANCE_NOT_SUPPORTED"
            ),
            "reader": (
                reader_decision
            ),
            "l2_role": l2_decision,
            "opens_r5b": bool(position_supported and qk_supported and av_supported),
        },
    }


def _synthetic_measurements(
    n: int,
    *,
    position: tuple[float, float] = (0.0, 0.0),
    content: tuple[float, float] = (0.0, 0.0),
    position_reference: tuple[float, float] = (0.0, 0.0),
    position_negative: tuple[float, float] = (0.0, 0.0),
    qk_partner: float = 0.0,
    qk_pseudo: float = 0.0,
    qk_control: float = 0.0,
    av_partner: float = 0.0,
    av_pseudo: float = 0.0,
    av_control: float = 0.0,
    l2_mode: str = "zero",
    l2_partner: float = 0.0,
    l2_pseudo: float = 0.0,
    l2_value: float = 0.0,
    l2_control: float = 0.0,
) -> dict[str, Any]:
    """Build coherent primitive cells for classifier plants/boundary diagnostics."""

    if n < 2 or l2_mode not in {"zero", "address", "mixed"}:
        raise ValueError("invalid synthetic R5a configuration")
    zero = lambda value=0.0: np.full(n, float(value), dtype=float)
    shift = {
        head: {
            comparison: {
                measure: {cell: zero() for cell in SHIFT_CELLS}
                for measure in SHIFT_MEASURES
            }
            for comparison in SHIFT_COMPARISONS
        }
        for head in SHIFT_HEADS
    }
    def set_shift(
        head: str,
        comparison: str,
        position_effects: tuple[float, float],
        content_effects: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        p_native, p_shifted = map(float, position_effects)
        c_native, c_shifted = map(float, content_effects)
        # joint=0, then solve the four simple-effect equations.  Feasibility requires
        # p_native + c_shifted == c_native + p_shifted.
        full = p_native + c_shifted
        word = p_shifted
        pos = c_shifted
        if not np.isclose(full - word, c_native, atol=1e-12):
            raise ValueError("requested position/content effects violate 2x2 algebra")
        for measure in SHIFT_MEASURES:
            shift[head][comparison][measure] = {
                "full": zero(full),
                "word_shift": zero(word),
                "position_shift": zero(pos),
                "joint_shift": zero(),
            }

    set_shift("L3H1", "paired_adjacent", position, content)
    set_shift("L3H1", "pseudo_adjacent", position_reference)
    for comparison in SHIFT_COMPARISONS:
        set_shift("L0H2", comparison, position_negative)

    qk = {
        head: {
            window: {"partner": zero(), "pseudo": zero()}
            for candidate, window in QK_ARMS
            if candidate == head
        }
        for head, _window in QK_ARMS
    }
    qk["L3H1"]["paired"] = {
        "partner": zero(qk_partner),
        "pseudo": zero(qk_pseudo),
    }
    qk["L3H1"]["pseudo"]["partner"] = zero(qk_control)

    def cells(address: float = 0.0, value: float = 0.0, joint: float = 0.0):
        return {
            "baseline": zero(),
            "address_only": zero(address),
            "value_only": zero(value),
            "joint": zero(joint),
        }

    av = {
        window: {"partner": cells(), "pseudo": cells()} for window in WINDOWS
    }
    av["paired"]["partner"] = cells(value=av_partner, joint=av_partner)
    av["paired"]["pseudo"] = cells(value=av_pseudo, joint=av_pseudo)
    av["pseudo"]["partner"] = cells(value=av_control, joint=av_control)

    l2 = {
        builder: {"partner": cells(), "pseudo": cells()} for builder in BUILDERS
    }
    if l2_mode == "address":
        l2["L2H15"]["partner"] = cells(
            address=l2_partner, joint=l2_partner + l2_value
        )
        l2["L2H15"]["pseudo"] = cells(address=l2_pseudo, joint=l2_pseudo)
    elif l2_mode == "mixed":
        l2["L2H15"]["partner"] = cells(
            address=l2_partner,
            value=l2_partner,
            joint=2.0 * l2_partner,
        )
        l2["L2H15"]["pseudo"] = cells(
            address=l2_pseudo,
            value=l2_pseudo,
            joint=2.0 * l2_pseudo,
        )
    l2["L0H2"]["partner"] = cells(
        address=l2_control, joint=l2_control
    )
    return {
        "position_shift": shift,
        "qk_surgery": qk,
        "attention_value": av,
        "l2_role": l2,
    }


def _add_centered_primitive_noise(
    value: Any,
    *,
    rng: np.random.Generator,
    scale: float,
    shared_gene_signs: np.ndarray | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            key: _add_centered_primitive_noise(
                item,
                rng=rng,
                scale=scale,
                shared_gene_signs=shared_gene_signs,
            )
            for key, item in value.items()
        }
    array = np.asarray(value, dtype=float)
    noise = rng.normal(0.0, scale, size=array.shape)
    noise -= noise.mean()
    if shared_gene_signs is not None:
        signs = np.asarray(shared_gene_signs, dtype=float)
        if signs.shape != array.shape:
            raise ValueError("shared calibration signs must match every primitive row")
        noise *= signs
    return array + noise


def classifier_calibration(
    *,
    n_genes: int = PRODUCTION_N,
    n_draws: int = PRODUCTION_RANDOMISATION,
    n_trials: int = PRODUCTION_CALIBRATION_TRIALS,
    seed: int = PRODUCTION_SEED,
) -> dict[str, Any]:
    """Exercise the exact production classifier on coherent boundary faces and plants.

    The faces are implementation diagnostics, not a claim that a pointwise synthetic DGP
    maximizes rejection over a nuisance continuum.  Nominal control comes from the closed
    boundary tests, IUT paths, Holm correction, and four-family alpha allocation.
    """

    expected_upper = one_sided_binomial_bounds(
        CALIBRATION_MAX_FALSE_POSITIVES,
        PRODUCTION_CALIBRATION_TRIALS,
        alpha=0.05,
    )[1]
    expected_lower = one_sided_binomial_bounds(
        CALIBRATION_MIN_POWER_SUCCESSES,
        PRODUCTION_CALIBRATION_TRIALS,
        alpha=0.05,
    )[0]
    if not (
        expected_upper <= FAMILY_ALPHA + CALIBRATION_EPSILON
        and expected_lower >= 0.90
    ):
        raise RuntimeError("R5a frozen calibration trial/count formula drifted")
    if n_genes < MIN_GENES or n_trials < 1:
        raise ValueError("production calibration requires 128 genes and positive trials")
    groups = [f"calibration-gene-{index:03d}" for index in range(n_genes)]
    surfaces = {
        "position_material_boundary": {
            "family": "position_shift",
            "assigned_paths": [
                "P_POSITION_SENSITIVE_L3_READER",
                "P_POSITION_DOMINANT_L3_READER",
            ],
            "kwargs": {"position": (SESOI, SESOI)},
        },
        "position_content_lower_boundary": {
            "family": "position_shift",
            "assigned_paths": ["P_POSITION_DOMINANT_L3_READER"],
            "kwargs": {
                "position": (0.40, 0.40),
                "content": (-SESOI, -SESOI),
            },
        },
        "position_content_upper_boundary": {
            "family": "position_shift",
            "assigned_paths": ["P_POSITION_DOMINANT_L3_READER"],
            "kwargs": {
                "position": (0.40, 0.40),
                "content": (SESOI, SESOI),
            },
        },
        "position_reference_lower_boundary": {
            "family": "position_shift",
            "assigned_paths": [
                "P_POSITION_SENSITIVE_L3_READER",
                "P_POSITION_DOMINANT_L3_READER",
            ],
            "kwargs": {
                "position": (0.40, 0.40),
                "position_reference": (-SESOI, -SESOI),
            },
        },
        "position_reference_upper_boundary": {
            "family": "position_shift",
            "assigned_paths": [
                "P_POSITION_SENSITIVE_L3_READER",
                "P_POSITION_DOMINANT_L3_READER",
            ],
            "kwargs": {
                "position": (0.40, 0.40),
                "position_reference": (SESOI, SESOI),
            },
        },
        "position_negative_lower_boundary": {
            "family": "position_shift",
            "assigned_paths": [
                "P_POSITION_SENSITIVE_L3_READER",
                "P_POSITION_DOMINANT_L3_READER",
            ],
            "kwargs": {
                "position": (0.40, 0.40),
                "position_negative": (-SESOI, -SESOI),
            },
        },
        "position_negative_upper_boundary": {
            "family": "position_shift",
            "assigned_paths": [
                "P_POSITION_SENSITIVE_L3_READER",
                "P_POSITION_DOMINANT_L3_READER",
            ],
            "kwargs": {
                "position": (0.40, 0.40),
                "position_negative": (SESOI, SESOI),
            },
        },
        "qk_material_boundary": {
            "family": "qk_surgery",
            "assigned_paths": ["P_QK_L3_READER"],
            "kwargs": {"qk_partner": SESOI},
        },
        "qk_pseudo_lower_boundary": {
            "family": "qk_surgery",
            "assigned_paths": ["P_QK_L3_READER"],
            "kwargs": {"qk_partner": 0.40, "qk_pseudo": -SESOI},
        },
        "qk_pseudo_upper_boundary": {
            "family": "qk_surgery",
            "assigned_paths": ["P_QK_L3_READER"],
            "kwargs": {"qk_partner": 0.40, "qk_pseudo": SESOI},
        },
        "qk_control_lower_boundary": {
            "family": "qk_surgery",
            "assigned_paths": ["P_QK_L3_READER"],
            "kwargs": {"qk_partner": 0.40, "qk_control": -SESOI},
        },
        "qk_control_upper_boundary": {
            "family": "qk_surgery",
            "assigned_paths": ["P_QK_L3_READER"],
            "kwargs": {"qk_partner": 0.40, "qk_control": SESOI},
        },
        "av_material_boundary": {
            "family": "attention_value",
            "assigned_paths": ["P_AV_DONOR_WINDOW_VALUE"],
            "kwargs": {"av_partner": SESOI},
        },
        "av_pseudo_lower_boundary": {
            "family": "attention_value",
            "assigned_paths": ["P_AV_DONOR_WINDOW_VALUE"],
            "kwargs": {"av_partner": 0.40, "av_pseudo": -SESOI},
        },
        "av_pseudo_upper_boundary": {
            "family": "attention_value",
            "assigned_paths": ["P_AV_DONOR_WINDOW_VALUE"],
            "kwargs": {"av_partner": 0.40, "av_pseudo": SESOI},
        },
        "av_control_lower_boundary": {
            "family": "attention_value",
            "assigned_paths": ["P_AV_DONOR_WINDOW_VALUE"],
            "kwargs": {"av_partner": 0.40, "av_control": -SESOI},
        },
        "av_control_upper_boundary": {
            "family": "attention_value",
            "assigned_paths": ["P_AV_DONOR_WINDOW_VALUE"],
            "kwargs": {"av_partner": 0.40, "av_control": SESOI},
        },
        "l2_address_material_boundary": {
            "family": "l2_role",
            "assigned_paths": ["P_L2H15_ADDRESS"],
            "kwargs": {"l2_mode": "address", "l2_partner": SESOI},
        },
        "l2_address_value_lower_boundary": {
            "family": "l2_role",
            "assigned_paths": ["P_L2H15_ADDRESS"],
            "kwargs": {
                "l2_mode": "address",
                "l2_partner": 0.40,
                "l2_value": -SESOI,
            },
        },
        "l2_address_value_upper_boundary": {
            "family": "l2_role",
            "assigned_paths": ["P_L2H15_ADDRESS"],
            "kwargs": {
                "l2_mode": "address",
                "l2_partner": 0.40,
                "l2_value": SESOI,
            },
        },
        "l2_address_pseudo_lower_boundary": {
            "family": "l2_role",
            "assigned_paths": ["P_L2H15_ADDRESS"],
            "kwargs": {
                "l2_mode": "address",
                "l2_partner": 0.40,
                "l2_pseudo": -SESOI,
            },
        },
        "l2_address_pseudo_upper_boundary": {
            "family": "l2_role",
            "assigned_paths": ["P_L2H15_ADDRESS"],
            "kwargs": {
                "l2_mode": "address",
                "l2_partner": 0.40,
                "l2_pseudo": SESOI,
            },
        },
        "l2_negative_lower_boundary": {
            "family": "l2_role",
            "assigned_paths": ["P_L2H15_ADDRESS", "P_L2H15_MIXED_AV"],
            "kwargs": {
                "l2_mode": "address",
                "l2_partner": 0.40,
                "l2_control": -SESOI,
            },
        },
        "l2_negative_upper_boundary": {
            "family": "l2_role",
            "assigned_paths": ["P_L2H15_ADDRESS", "P_L2H15_MIXED_AV"],
            "kwargs": {
                "l2_mode": "address",
                "l2_partner": 0.40,
                "l2_control": SESOI,
            },
        },
        "l2_mixed_material_boundary": {
            "family": "l2_role",
            "assigned_paths": ["P_L2H15_MIXED_AV"],
            "kwargs": {"l2_mode": "mixed", "l2_partner": SESOI},
        },
        "l2_mixed_pseudo_lower_boundary": {
            "family": "l2_role",
            "assigned_paths": ["P_L2H15_MIXED_AV"],
            "kwargs": {
                "l2_mode": "mixed",
                "l2_partner": 0.40,
                "l2_pseudo": -SESOI,
            },
        },
        "l2_mixed_pseudo_upper_boundary": {
            "family": "l2_role",
            "assigned_paths": ["P_L2H15_MIXED_AV"],
            "kwargs": {
                "l2_mode": "mixed",
                "l2_partner": 0.40,
                "l2_pseudo": SESOI,
            },
        },
    }
    plants = {
        "P_POSITION_SENSITIVE_L3_READER": {
            "family": "position_shift",
            "kwargs": {"position": (0.40, 0.40), "content": (0.20, 0.20)},
        },
        "P_POSITION_DOMINANT_L3_READER": {
            "family": "position_shift",
            "kwargs": {"position": (0.40, 0.40)},
        },
        "P_QK_L3_READER": {
            "family": "qk_surgery",
            "kwargs": {"qk_partner": 0.40},
        },
        "P_AV_DONOR_WINDOW_VALUE": {
            "family": "attention_value",
            "kwargs": {"av_partner": 0.40},
        },
        "P_L2H15_ADDRESS": {
            "family": "l2_role",
            "kwargs": {"l2_mode": "address", "l2_partner": 0.40},
        },
        "P_L2H15_MIXED_AV": {
            "family": "l2_role",
            "kwargs": {"l2_mode": "mixed", "l2_partner": 0.40},
        },
    }
    surface_counts = {name: 0 for name in surfaces}
    surface_patterns = {name: [] for name in surfaces}
    plant_counts = {name: 0 for name in plants}
    plant_patterns = {name: [] for name in plants}
    for trial in range(n_trials):
        for surface_index, (name, spec) in enumerate(surfaces.items()):
            rng = np.random.default_rng(
                int.from_bytes(
                    hashlib.sha256(
                        f"r5a-calibration|surface|{surface_index}|{trial}|{seed}".encode()
                    ).digest()[:8],
                    "big",
                )
            )
            outer_signs = rng.choice(
                np.asarray([-1.0, 1.0]), size=n_genes, replace=True
            )
            primitive = _add_centered_primitive_noise(
                _synthetic_measurements(n_genes, **spec["kwargs"]),
                rng=rng,
                scale=CALIBRATION_PRIMITIVE_NOISE_SD,
                shared_gene_signs=outer_signs,
            )
            family = build_family_matrices(primitive)[spec["family"]]
            result = strict_family_inference(
                family,
                groups,
                family_row_names()[spec["family"]],
                family=spec["family"],
                n_draws=n_draws,
                seed=seed,
                namespace=f"r5a-calibration:surface:{name}:trial:{trial}",
                min_genes=MIN_GENES,
            )
            rejected = any(
                result["paths"][path]["classification"] == "SUPPORTED"
                for path in spec["assigned_paths"]
            )
            surface_counts[name] += int(rejected)
            surface_patterns[name].append(int(rejected))
        for plant_index, (path, spec) in enumerate(plants.items()):
            rng = np.random.default_rng(
                int.from_bytes(
                    hashlib.sha256(
                        f"r5a-calibration|plant|{plant_index}|{trial}|{seed}".encode()
                    ).digest()[:8],
                    "big",
                )
            )
            outer_signs = rng.choice(
                np.asarray([-1.0, 1.0]), size=n_genes, replace=True
            )
            primitive = _add_centered_primitive_noise(
                _synthetic_measurements(n_genes, **spec["kwargs"]),
                rng=rng,
                scale=CALIBRATION_PRIMITIVE_NOISE_SD,
                shared_gene_signs=outer_signs,
            )
            family = build_family_matrices(primitive)[spec["family"]]
            result = strict_family_inference(
                family,
                groups,
                family_row_names()[spec["family"]],
                family=spec["family"],
                n_draws=n_draws,
                seed=seed,
                namespace=f"r5a-calibration:plant:{path}:trial:{trial}",
                min_genes=MIN_GENES,
            )
            supported = result["paths"][path]["classification"] == "SUPPORTED"
            plant_counts[path] += int(supported)
            plant_patterns[path].append(int(supported))
    surface_records = {}
    for name, count in surface_counts.items():
        upper = one_sided_binomial_bounds(count, n_trials, alpha=0.05)[1]
        surface_records[name] = {
            "family": surfaces[name]["family"],
            "assigned_paths": surfaces[name]["assigned_paths"],
            "false_positive_trials": count,
            "trials": n_trials,
            "one_sided_95pct_clopper_pearson_upper": upper,
            "engineering_limit": FAMILY_ALPHA + CALIBRATION_EPSILON,
            "pass": count <= CALIBRATION_MAX_FALSE_POSITIVES
            and upper <= FAMILY_ALPHA + CALIBRATION_EPSILON,
            "pattern_sha256": canonical_sha256(surface_patterns[name]),
        }
    plant_records = {}
    for path, count in plant_counts.items():
        lower = one_sided_binomial_bounds(count, n_trials, alpha=0.05)[0]
        plant_records[path] = {
            "family": plants[path]["family"],
            "successes": count,
            "trials": n_trials,
            "one_sided_95pct_clopper_pearson_lower": lower,
            "power_floor": 0.90,
            "pass": count >= CALIBRATION_MIN_POWER_SUCCESSES and lower >= 0.90,
            "pattern_sha256": canonical_sha256(plant_patterns[path]),
        }
    return {
        "schema": "gfmcd.r5a.classifier_calibration/1",
        "production_classifier_used": True,
        "primitive_cells_rebuilt_before_every_classifier": True,
        "outer_null_uses_one_shared_sign_per_gene_across_all_primitive_rows": True,
        "derived_rows_directly_shifted": False,
        "n_genes": n_genes,
        "n_draws": n_draws,
        "n_trials": n_trials,
        "family_alpha": FAMILY_ALPHA,
        "epsilon_cal": CALIBRATION_EPSILON,
        "primitive_noise_sd": CALIBRATION_PRIMITIVE_NOISE_SD,
        "trial_count_rule": (
            "smallest N with CP-upper(floor(N*alpha_family),N)<=alpha_family+0.01 "
            "and CP-lower(floor(.95*N),N)>=.90"
        ),
        "boundary_faces_are_implementation_diagnostics_not_global_lfc_proof": True,
        "formal_nominal_control": (
            "closed boundary tests + TOST/IUT + within-family Holm + four-family alpha allocation"
        ),
        "boundary_signature_registry": {
            name: {
                "family": spec["family"],
                "assigned_paths": spec["assigned_paths"],
                "primitive_kwargs": spec["kwargs"],
            }
            for name, spec in surfaces.items()
        },
        "boundary_signature_registry_sha256": canonical_sha256(
            {
                name: {
                    "family": spec["family"],
                    "assigned_paths": spec["assigned_paths"],
                    "primitive_kwargs": spec["kwargs"],
                }
                for name, spec in surfaces.items()
            }
        ),
        "boundary_signature_count": len(surfaces),
        "plant_registry": {
            path: {"family": spec["family"], "primitive_kwargs": spec["kwargs"]}
            for path, spec in plants.items()
        },
        "plant_registry_sha256": canonical_sha256(
            {
                path: {
                    "family": spec["family"],
                    "primitive_kwargs": spec["kwargs"],
                }
                for path, spec in plants.items()
            }
        ),
        "surfaces": surface_records,
        "plants": plant_records,
        "all_pass": all(item["pass"] for item in surface_records.values())
        and all(item["pass"] for item in plant_records.values()),
    }


def swap_component_tensor(
    captured: torch.Tensor,
    left_positions: torch.Tensor,
    right_positions: torch.Tensor,
) -> torch.Tensor:
    """Return a component tensor with two aligned four-token windows exchanged.

    This is used for either word-embedding or position-embedding outputs.  The input is
    never edited in place; swapping twice is an exact identity at tensor precision.
    """

    if captured.ndim != 3:
        raise ValueError("captured embedding component must have shape [B or 1,S,H]")
    left = torch.as_tensor(left_positions, device=captured.device, dtype=torch.long)
    right = torch.as_tensor(right_positions, device=captured.device, dtype=torch.long)
    if left.ndim != 2 or left.shape != right.shape or left.shape[1] != 4:
        raise ValueError("window positions must have shape [B,4]")
    batch = left.shape[0]
    if captured.shape[0] not in {1, batch}:
        raise ValueError("captured component batch is incompatible with positions")
    if torch.any(left < 0) or torch.any(right < 0):
        raise ValueError("window positions must be non-negative")
    if torch.any(left >= captured.shape[1]) or torch.any(right >= captured.shape[1]):
        raise ValueError("window positions exceed the component sequence length")
    for row in range(batch):
        if set(left[row].tolist()) & set(right[row].tolist()):
            raise ValueError("shift windows must be disjoint")
    source = captured.expand(batch, -1, -1).clone()
    rows = torch.arange(batch, device=source.device)[:, None].expand_as(left)
    left_values = source[rows, left].clone()
    right_values = source[rows, right].clone()
    source[rows, left] = right_values
    source[rows, right] = left_values
    return source


def qk_attention_row(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    head: int,
    query_positions: torch.Tensor,
    attention_mask: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact pre-softmax and post-softmax row for one BERT head."""

    if query.shape != key.shape or query.ndim != 3:
        raise ValueError("query/key must have matching [B,S,H] shapes")
    if query.shape[-1] % num_heads:
        raise ValueError("hidden width is not divisible by num_heads")
    if not 0 <= int(head) < int(num_heads):
        raise ValueError("head is out of range")
    positions = torch.as_tensor(query_positions, device=query.device, dtype=torch.long)
    if positions.shape != (query.shape[0],):
        raise ValueError("query_positions must have shape [B]")
    mask = torch.as_tensor(attention_mask, device=query.device)
    if mask.shape != query.shape[:2]:
        raise ValueError("attention_mask must have shape [B,S]")
    width = query.shape[-1] // num_heads
    lo, hi = int(head) * width, (int(head) + 1) * width
    rows = torch.arange(query.shape[0], device=query.device)
    q = query[rows, positions, lo:hi]
    k = key[:, :, lo:hi]
    logits = torch.einsum("bd,bsd->bs", q, k) / float(width) ** 0.5
    logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)
    return logits, torch.softmax(logits, dim=-1)


def replace_selected_logits(
    clean_logits: torch.Tensor,
    source_logits: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Replace exactly four finite row logits and return a fresh tensor."""

    if clean_logits.shape != source_logits.shape or clean_logits.ndim != 2:
        raise ValueError("clean/source logits must be matching [B,S] tensors")
    selected = torch.as_tensor(positions, device=clean_logits.device, dtype=torch.long)
    if selected.shape != (clean_logits.shape[0], 4):
        raise ValueError("positions must have shape [B,4]")
    if torch.any(selected < 0) or torch.any(selected >= clean_logits.shape[1]):
        raise ValueError("selected logit position is out of range")
    if any(len(set(row.tolist())) != 4 for row in selected):
        raise ValueError("each QK surgery row must select four distinct logits")
    output = clean_logits.clone()
    rows = torch.arange(output.shape[0], device=output.device)[:, None].expand_as(selected)
    output[rows, selected] = source_logits[rows, selected]
    return output


def reconstruct_head_context(attention: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    if attention.ndim != 2 or values.ndim != 3 or values.shape[:2] != attention.shape:
        raise ValueError("attention [B,S] and values [B,S,Dh] must align")
    return torch.einsum("bs,bsd->bd", attention, values)


def replace_value_window(
    target: torch.Tensor, source: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    if target.shape != source.shape or target.ndim != 3:
        raise ValueError("target/source values must be matching [B,S,Dh] tensors")
    selected = torch.as_tensor(positions, device=target.device, dtype=torch.long)
    if selected.shape != (target.shape[0], 4):
        raise ValueError("positions must have shape [B,4]")
    output = target.clone()
    rows = torch.arange(output.shape[0], device=output.device)[:, None].expand_as(selected)
    output[rows, selected] = source[rows, selected]
    return output


def _capture_qkv_and_heads(
    adapter, encoded: Mapping[str, torch.Tensor], *, layer: int = READER[0]
) -> dict[str, Any]:
    projections: dict[str, torch.Tensor] = {}
    handles = []
    if not 0 <= int(layer) < adapter.num_layers:
        raise ValueError("capture layer is out of range")
    module = adapter.attention_self_modules[int(layer)]
    for name in ("query", "key", "value"):
        handles.append(
            getattr(module, name).register_forward_hook(
                lambda _module, _inputs, output, key=name: projections.__setitem__(
                    key, output.detach().clone()
                )
            )
        )
    head_cache: dict[int, torch.Tensor] = {}
    dense_cache: dict[str, torch.Tensor] = {}

    def capture_context(_module, _inputs, output):
        context = output if torch.is_tensor(output) else output[0]
        head_cache[int(layer)] = (
            context.view(
                context.shape[0], context.shape[1], adapter.num_heads, -1
            )
            .permute(0, 2, 1, 3)
            .detach()
            .clone()
        )

    handles.append(module.register_forward_hook(capture_context))
    dense_module = adapter.encoder_layers[int(layer)].attention.output.dense

    def capture_dense(_module, inputs, output):
        if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(output):
            raise RuntimeError("attention output projection did not expose tensors")
        dense_cache["input"] = inputs[0].detach().clone()
        dense_cache["output"] = output.detach().clone()

    handles.append(dense_module.register_forward_hook(capture_dense))
    try:
        result = adapter.model(**encoded, output_attentions=True)
    finally:
        for handle in handles:
            handle.remove()
    if (
        set(projections) != {"query", "key", "value"}
        or int(layer) not in head_cache
        or set(dense_cache) != {"input", "output"}
    ):
        raise RuntimeError(f"failed to capture complete R5a layer-{layer} state")
    return {
        "query": projections["query"],
        "key": projections["key"],
        "value": projections["value"],
        "heads": head_cache[int(layer)],
        "attentions": result.attentions[int(layer)].detach().clone(),
        "layer": int(layer),
        "output_projection_input": dense_cache["input"],
        "output_projection_output": dense_cache["output"],
        "logits": result.logits.detach().float(),
    }


def full_mlm_replay_capability(
    adapter,
    task: CouplingTask,
    encoded: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
) -> dict[str, Any]:
    """Verify that the complete checkpoint ``model.cls`` reproduces logits and margins."""

    cls_module = getattr(adapter.model, "cls", None)
    if cls_module is None:
        raise RuntimeError("R5a requires the checkpoint's complete model.cls module")
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs):
        if not inputs or not torch.is_tensor(inputs[0]):
            raise RuntimeError("model.cls input is not a tensor")
        captured["hidden"] = inputs[0].detach().clone()

    handle = cls_module.register_forward_pre_hook(hook)
    try:
        native = adapter.model(**encoded).logits.detach().float()
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("failed to capture model.cls input")
    with torch.no_grad():
        replayed = cls_module(captured["hidden"]).detach().float()
    logits_error = float((native - replayed).abs().max())
    native_margin = task.margin(native, readout_positions)
    replay_margin = task.margin(replayed, readout_positions)
    margin_error = float((native_margin - replay_margin).abs().max())
    return {
        "checked": True,
        "uses_complete_model_cls": True,
        "donor_readout_constructed": False,
        "full_logits_max_abs_error": logits_error,
        "acceptor_G_margin_max_abs_error": margin_error,
        "atol": HOOK_ATOL,
        "ok": logits_error < HOOK_ATOL and margin_error < HOOK_ATOL,
    }


def attention_projection_capability(adapter, state: Mapping[str, Any]) -> dict[str, Any]:
    """Verify that split head contexts merge into the checkpoint output projection."""

    layer = int(state["layer"])
    merged = _merge_context(state["heads"])
    dense = adapter.encoder_layers[layer].attention.output.dense
    with torch.no_grad():
        projected = dense(merged)
    input_error = float(
        (merged.float() - state["output_projection_input"].float()).abs().max()
    )
    output_error = float(
        (projected.float() - state["output_projection_output"].float()).abs().max()
    )
    return {
        "layer": layer,
        "merged_context_sha256": _array_sha256(merged.detach().cpu().numpy()),
        "projected_context_sha256": _array_sha256(projected.detach().cpu().numpy()),
        "merge_to_dense_input_max_abs_error": input_error,
        "dense_replay_max_abs_error": output_error,
        "atol": HOOK_ATOL,
        "finite": bool(torch.isfinite(projected).all()),
        "uses_checkpoint_attention_output_dense": True,
        "ok": bool(
            torch.isfinite(projected).all()
            and input_error < HOOK_ATOL
            and output_error < HOOK_ATOL
        ),
    }


def full_head_noop_capability(
    adapter,
    task: CouplingTask,
    encoded: Mapping[str, torch.Tensor],
    state: Mapping[str, Any],
    readout_positions: Sequence[int],
) -> dict[str, Any]:
    """Replay a complete native L3H1 head through the actual hook path."""

    layer, head = READER
    with patch_heads(adapter, {layer: state["heads"]}, [(layer, head)]):
        replayed = adapter.model(**encoded).logits.detach().float()
    native = state["logits"].detach().float()
    logits_error = float((native - replayed).abs().max())
    margin_error = float(
        (
            task.margin(native, readout_positions)
            - task.margin(replayed, readout_positions)
        )
        .abs()
        .max()
    )
    return {
        "checked": True,
        "head": "L3H1",
        "whole_head_all_tokens_replaced_with_same_run_capture": True,
        "full_logits_max_abs_error": logits_error,
        "acceptor_G_margin_max_abs_error": margin_error,
        "atol": HOOK_ATOL,
        "ok": logits_error < HOOK_ATOL and margin_error < HOOK_ATOL,
    }


def health_probe_positions(
    sequence: str,
    *,
    example_id: str,
    readout_position: int,
    protected_positions: Sequence[int],
    n_probe: int = HEALTH_PROBES_PER_EXAMPLE,
    seed: int = PRODUCTION_SEED,
) -> list[int]:
    """Choose deterministic nucleotide probes away from every R5a site.

    Probe choice is annotation/sequence-only and is fixed before any model output is
    examined.  A seven-token guard surrounds the acceptor query; every paired, pseudo,
    and adjacent protected token receives a four-token guard on both sides.
    """

    if n_probe < 1:
        raise ValueError("R5a health requires at least one probe")
    excluded = set(range(int(readout_position) - 3, int(readout_position) + 4))
    for position in protected_positions:
        excluded.update(range(int(position) - 4, int(position) + 5))
    candidates = [
        position
        for position in range(10, len(sequence) - 10)
        if position not in excluded and sequence[position] in "ACGT"
    ]
    if len(candidates) < n_probe:
        raise RuntimeError(f"{example_id}: insufficient non-site R5a health probes")
    rng_seed = int.from_bytes(
        hashlib.sha256(
            f"r5a-health|{example_id}|{seed}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(rng_seed)
    return sorted(
        int(position) for position in rng.choice(candidates, n_probe, replace=False)
    )


def _top1_nucleotide_token_ids(
    logits: torch.Tensor, base_ids: Mapping[str, int]
) -> torch.Tensor:
    nucleotide_ids = torch.tensor(
        [int(base_ids[base]) for base in "ACGT"], device=logits.device
    )
    nucleotide_logits = torch.stack(
        [logits[..., int(base_ids[base])] for base in "ACGT"], dim=-1
    )
    return nucleotide_ids[nucleotide_logits.argmax(dim=-1)]


def bounded_intervention_health(
    adapter,
    task: CouplingTask,
    rows: Sequence[Any],
    *,
    seed: int = PRODUCTION_SEED,
) -> dict[str, Any]:
    """Bound nucleotide MLM health for every intervention geometry.

    Reader surgeries all replace a single acceptor-query head context, so the shared
    local-query stress policy uses a zero replacement for L3H1 and L0H2.  Builder arms
    are checked under their exact all-token within-example mean intervention, both alone
    and combined with the L3H1 local-query stress.  These are plumbing/health controls;
    candidate R5a outcomes never enter their definition.
    """

    selected = list(rows[:HEALTH_MAX_EXAMPLES])
    if len(selected) != HEALTH_MAX_EXAMPLES:
        raise RuntimeError("R5a bounded health requires 16 selected examples")
    policies: tuple[tuple[str, tuple[int, int] | None, tuple[int, int] | None], ...] = (
        ("native", None, None),
        ("local_query_zero_L3H1", None, READER),
        ("local_query_zero_L0H2", None, NEGATIVE_HEAD),
        ("builder_mean_L2H15", ADDRESS_BUILDER, None),
        ("builder_mean_L2H15_plus_local_query_zero_L3H1", ADDRESS_BUILDER, READER),
        ("builder_mean_L0H2", NEGATIVE_HEAD, None),
        ("builder_mean_L0H2_plus_local_query_zero_L3H1", NEGATIVE_HEAD, READER),
    )
    hits = {name: 0 for name, _builder, _query in policies}
    totals = {name: 0 for name, _builder, _query in policies}
    probe_registry: list[dict[str, Any]] = []
    for start in range(0, len(selected), task.batch):
        batch_rows = selected[start : start + task.batch]
        readout = [int(row.acceptor_g) for row in batch_rows]
        encoded = task.encode([row.clean_sequence for row in batch_rows], readout)
        probes: list[list[int]] = []
        for local_index, row in enumerate(batch_rows):
            protected = (
                *donor_window_positions(int(row.example.donor_pos)),
                *donor_window_positions(int(row.pseudo_donor)),
                *[int(value) for value in row.adjacent_window],
            )
            positions = health_probe_positions(
                row.clean_sequence,
                example_id=str(row.example.example_id),
                readout_position=int(row.acceptor_g),
                protected_positions=protected,
                seed=seed,
            )
            probes.append(positions)
            probe_registry.append(
                {
                    "example_id": str(row.example.example_id),
                    "positions": positions,
                }
            )
            token_positions = [position + task.off for position in positions]
            encoded["input_ids"][local_index, token_positions] = task.mask_id

        query_tokens = torch.tensor(
            [position + task.off for position in readout],
            device=adapter.device,
            dtype=torch.long,
        )
        query_mask = torch.zeros_like(encoded["attention_mask"], dtype=torch.bool)
        query_mask[torch.arange(len(batch_rows), device=adapter.device), query_tokens] = True
        zero_sources: dict[tuple[int, int], dict[int, torch.Tensor]] = {}
        for query_head in (READER, NEGATIVE_HEAD):
            layer, _head = query_head
            state = _capture_qkv_and_heads(adapter, encoded, layer=layer)
            zero_sources[query_head] = {layer: torch.zeros_like(state["heads"])}

        for name, builder, query_head in policies:
            with ExitStack() as stack:
                if builder is not None:
                    stack.enter_context(
                        edit_heads(adapter, [builder], mode="within_example_mean")
                    )
                if query_head is not None:
                    layer, _head = query_head
                    stack.enter_context(
                        patch_heads(
                            adapter,
                            zero_sources[query_head],
                            [query_head],
                            token_mask=query_mask,
                        )
                    )
                logits = adapter.model(**encoded).logits.float()
            for local_index, (row, positions) in enumerate(zip(batch_rows, probes)):
                token_positions = [position + task.off for position in positions]
                predicted = _top1_nucleotide_token_ids(
                    logits[local_index, token_positions], task.bid
                )
                target = torch.tensor(
                    [task.bid[row.clean_sequence[position]] for position in positions],
                    device=predicted.device,
                    dtype=predicted.dtype,
                )
                hits[name] += int((predicted == target).sum().item())
                totals[name] += len(positions)

    records = {
        name: {
            "hits": hits[name],
            "total": totals[name],
            "accuracy": hits[name] / totals[name] if totals[name] else 0.0,
            "floor": HEALTH_FLOOR,
            "ok": bool(totals[name] and hits[name] / totals[name] >= HEALTH_FLOOR),
        }
        for name, _builder, _query in policies
    }
    return {
        "schema": "gfmcd.r5a.bounded_intervention_health/1",
        "candidate_outcomes_used": False,
        "maximum_examples": HEALTH_MAX_EXAMPLES,
        "probes_per_example": HEALTH_PROBES_PER_EXAMPLE,
        "probe_registry": probe_registry,
        "probe_registry_sha256": canonical_sha256(probe_registry),
        "policies": records,
        "all_pass": all(record["ok"] for record in records.values()),
    }


def _inject_query_context_margin(
    adapter,
    task: CouplingTask,
    encoded: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
    query_tokens: torch.Tensor,
    base_heads: torch.Tensor,
    *,
    layer: int,
    head: int,
    context: torch.Tensor,
    outer_context: Callable[[], AbstractContextManager] | None = None,
) -> np.ndarray:
    source = base_heads.clone()
    rows = torch.arange(source.shape[0], device=source.device)
    source[rows, head, query_tokens] = context.to(source.dtype)
    token_mask = torch.zeros(
        source.shape[0], source.shape[2], dtype=torch.bool, device=source.device
    )
    token_mask[rows, query_tokens] = True
    with ExitStack() as stack:
        if outer_context is not None:
            stack.enter_context(outer_context())
        stack.enter_context(
            patch_heads(adapter, {layer: source}, [(layer, head)], token_mask=token_mask)
        )
        logits = adapter.model(**encoded).logits.float()
    return task.margin(logits, readout_positions).detach().cpu().numpy()


def _head_values(state: Mapping[str, Any], head: int, num_heads: int) -> torch.Tensor:
    value = state["value"]
    width = value.shape[-1] // num_heads
    return value[:, :, head * width : (head + 1) * width]


def _head_rows(
    state: Mapping[str, Any],
    encoded: Mapping[str, torch.Tensor],
    query_tokens: torch.Tensor,
    *,
    head: int,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, attention = qk_attention_row(
        state["query"],
        state["key"],
        head=head,
        query_positions=query_tokens,
        attention_mask=encoded["attention_mask"],
        num_heads=num_heads,
    )
    if not torch.isfinite(logits).all() or not torch.isfinite(attention).all():
        raise RuntimeError("R5a QK/attention row contains a non-finite value")
    row_sum_error = float((attention.sum(dim=-1) - 1.0).abs().max())
    if row_sum_error >= HOOK_ATOL or not bool((attention.sum(dim=-1) > 0).all()):
        raise RuntimeError("R5a attention row is zero or does not sum to one")
    return logits, attention


def _window_tensor(
    positions: Sequence[int], *, token_offset: int, device: torch.device
) -> torch.Tensor:
    return torch.tensor(
        [
            [value + token_offset for value in donor_window_positions(position)]
            for position in positions
        ],
        dtype=torch.long,
        device=device,
    )


def _preference(row: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> np.ndarray:
    indices = torch.arange(row.shape[0], device=row.device)[:, None]
    return (
        row[indices, left].mean(dim=1) - row[indices, right].mean(dim=1)
    ).detach().cpu().numpy()


def _swap_context(
    adapter,
    components: Mapping[str, torch.Tensor],
    left: torch.Tensor,
    right: torch.Tensor,
    cell: str,
) -> AbstractContextManager:
    if cell not in SHIFT_CELLS:
        raise ValueError(f"unknown shift cell: {cell}")
    stack = ExitStack()
    tracked = torch.cat((left, right), dim=1)
    if cell in {"word_shift", "joint_shift"}:
        word = swap_component_tensor(components["word"], left, right)
        stack.enter_context(clamp_embed_component(adapter, "word", word, tracked))
    if cell in {"position_shift", "joint_shift"}:
        position = swap_component_tensor(components["position"], left, right)
        stack.enter_context(
            clamp_embed_component(adapter, "position", position, tracked)
        )
    return stack


def run_position_shift_batch(
    adapter,
    task: CouplingTask,
    encoded: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
    paired_positions: Sequence[int],
    pseudo_positions: Sequence[int],
    adjacent_positions: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute the exact 2x2 component shift and fixed-V behavior bridge."""

    device = adapter.device
    query_tokens = torch.tensor(
        [position + task.off for position in readout_positions],
        dtype=torch.long,
        device=device,
    )
    windows = {
        "paired": _window_tensor(paired_positions, token_offset=task.off, device=device),
        "pseudo": _window_tensor(pseudo_positions, token_offset=task.off, device=device),
        "adjacent": _window_tensor(adjacent_positions, token_offset=task.off, device=device),
    }
    components = capture_embed_components(
        adapter, lambda: adapter.model(**encoded)
    )
    layers = tuple(sorted({head_tuple(name)[0] for name in SHIFT_HEADS}))
    clean_states = {
        layer: _capture_qkv_and_heads(adapter, encoded, layer=layer)
        for layer in layers
    }
    clean_margin = task.margin(
        clean_states[READER[0]]["logits"], readout_positions
    ).cpu().numpy()
    output: dict[str, Any] = {head: {} for head in SHIFT_HEADS}
    invariants: dict[str, Any] = {
        "query_token_never_clamped": True,
        "input_ids_sha256": _array_sha256(encoded["input_ids"].cpu().numpy()),
        "fixed_V_sha256_by_head": {},
        "full_cell_margin_identity_max_abs": 0.0,
        "attention_reconstruction_max_abs": 0.0,
        "component_swap_involution_max_abs": 0.0,
        "component_swap_assignment_max_abs": 0.0,
        "component_swap_outside_max_abs": 0.0,
        "bridge_reuses_one_clean_V_tensor_for_all_cells": True,
        "bridge_fixed_V_sha256_by_head_and_cell": {},
    }
    for comparison, left_name in (
        ("paired_adjacent", "paired"),
        ("pseudo_adjacent", "pseudo"),
    ):
        left, right = windows[left_name], windows["adjacent"]
        if any(
            int(query_tokens[row]) in set(torch.cat((left[row], right[row])).tolist())
            for row in range(len(query_tokens))
        ):
            raise RuntimeError("position/content shift would edit the acceptor query")
        component_rows = torch.arange(left.shape[0], device=device)[:, None]
        for captured in components.values():
            expanded = captured.expand(left.shape[0], -1, -1)
            shifted = swap_component_tensor(captured, left, right)
            restored = swap_component_tensor(shifted, left, right)
            invariants["component_swap_involution_max_abs"] = max(
                invariants["component_swap_involution_max_abs"],
                float((restored.float() - expanded.float()).abs().max()),
            )
            assignment_error = max(
                float(
                    (
                        shifted[component_rows, left].float()
                        - expanded[component_rows, right].float()
                    )
                    .abs()
                    .max()
                ),
                float(
                    (
                        shifted[component_rows, right].float()
                        - expanded[component_rows, left].float()
                    )
                    .abs()
                    .max()
                ),
            )
            invariants["component_swap_assignment_max_abs"] = max(
                invariants["component_swap_assignment_max_abs"], assignment_error
            )
            selected_mask = torch.zeros(
                expanded.shape[:2], dtype=torch.bool, device=expanded.device
            )
            selected_mask[component_rows, left] = True
            selected_mask[component_rows, right] = True
            outside_error = float(
                (shifted[~selected_mask].float() - expanded[~selected_mask].float())
                .abs()
                .max()
            )
            invariants["component_swap_outside_max_abs"] = max(
                invariants["component_swap_outside_max_abs"], outside_error
            )
        states: dict[str, dict[int, Any]] = {}
        for cell in SHIFT_CELLS:
            with _swap_context(adapter, components, left, right, cell):
                states[cell] = {
                    layer: _capture_qkv_and_heads(adapter, encoded, layer=layer)
                    for layer in layers
                }
        for head_name in SHIFT_HEADS:
            layer, head = head_tuple(head_name)
            fixed_values = _head_values(clean_states[layer], head, adapter.num_heads)
            invariants["fixed_V_sha256_by_head"][head_name] = _array_sha256(
                fixed_values.cpu().numpy()
            )
            invariants["bridge_fixed_V_sha256_by_head_and_cell"][head_name] = {
                cell: invariants["fixed_V_sha256_by_head"][head_name]
                for cell in SHIFT_CELLS
            }
            cell_values = {
                measure: {} for measure in SHIFT_MEASURES
            }
            for cell in SHIFT_CELLS:
                qk, attention = _head_rows(
                    states[cell][layer],
                    encoded,
                    query_tokens,
                    head=head,
                    num_heads=adapter.num_heads,
                )
                native_attention = states[cell][layer]["attentions"][:, head]
                rows = torch.arange(attention.shape[0], device=attention.device)
                native_row = native_attention[rows, query_tokens]
                error = float((native_row.float() - attention.float()).abs().max())
                invariants["attention_reconstruction_max_abs"] = max(
                    invariants["attention_reconstruction_max_abs"], error
                )
                context = reconstruct_head_context(attention, fixed_values)
                margin = _inject_query_context_margin(
                    adapter,
                    task,
                    encoded,
                    readout_positions,
                    query_tokens,
                    clean_states[layer]["heads"],
                    layer=layer,
                    head=head,
                    context=context,
                )
                if cell == "full":
                    invariants["full_cell_margin_identity_max_abs"] = max(
                        invariants["full_cell_margin_identity_max_abs"],
                        float(np.max(np.abs(margin - clean_margin))),
                    )
                cell_values["qk_preference"][cell] = _preference(qk, left, right)
                cell_values["attention_preference"][cell] = _preference(
                    attention, left, right
                )
                cell_values["acceptor_margin"][cell] = margin
            output[head_name][comparison] = cell_values
    invariants["ok"] = bool(
        invariants["attention_reconstruction_max_abs"] < HOOK_ATOL
        and invariants["full_cell_margin_identity_max_abs"] < MARGIN_ATOL
        and invariants["component_swap_involution_max_abs"] == 0.0
        and invariants["component_swap_assignment_max_abs"] == 0.0
        and invariants["component_swap_outside_max_abs"] == 0.0
    )
    return output, invariants


def run_qk_batch(
    adapter,
    task: CouplingTask,
    encoded_clean: Mapping[str, torch.Tensor],
    encoded_partner: Mapping[str, torch.Tensor],
    encoded_pseudo: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
    paired_positions: Sequence[int],
    pseudo_positions: Sequence[int],
    adjacent_positions: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_tokens = torch.tensor(
        [position + task.off for position in readout_positions],
        dtype=torch.long,
        device=adapter.device,
    )
    windows = {
        "paired": _window_tensor(paired_positions, token_offset=task.off, device=adapter.device),
        "pseudo": _window_tensor(pseudo_positions, token_offset=task.off, device=adapter.device),
        "adjacent": _window_tensor(adjacent_positions, token_offset=task.off, device=adapter.device),
    }
    layers = tuple(sorted({head_tuple(head)[0] for head, _ in QK_ARMS}))
    encodings = {
        "clean": encoded_clean,
        "partner": encoded_partner,
        "pseudo": encoded_pseudo,
    }
    states = {
        name: {
            layer: _capture_qkv_and_heads(adapter, encoded, layer=layer)
            for layer in layers
        }
        for name, encoded in encodings.items()
    }
    clean_margin = task.margin(
        states["clean"][READER[0]]["logits"], readout_positions
    ).cpu().numpy()
    output: dict[str, Any] = {head: {window: {} for h, window in QK_ARMS if h == head} for head, _ in QK_ARMS}
    maximum_unselected_error = 0.0
    maximum_selected_assignment_error = 0.0
    maximum_value_error = 0.0
    fixed_value_sha256_by_arm: dict[str, str] = {}
    for head_name, window_name in QK_ARMS:
        layer, head = head_tuple(head_name)
        clean_logits, _ = _head_rows(
            states["clean"][layer], encoded_clean, query_tokens, head=head, num_heads=adapter.num_heads
        )
        clean_values = _head_values(states["clean"][layer], head, adapter.num_heads)
        fixed_value_sha256_by_arm[f"{head_name}:{window_name}"] = _array_sha256(
            clean_values.cpu().numpy()
        )
        for leg in ("partner", "pseudo"):
            source_logits, _ = _head_rows(
                states[leg][layer],
                encoded_partner if leg == "partner" else encoded_pseudo,
                query_tokens,
                head=head,
                num_heads=adapter.num_heads,
            )
            edited = replace_selected_logits(clean_logits, source_logits, windows[window_name])
            rows = torch.arange(edited.shape[0], device=edited.device)[:, None]
            selected_mask = torch.zeros_like(edited, dtype=torch.bool)
            selected_mask[rows, windows[window_name]] = True
            maximum_selected_assignment_error = max(
                maximum_selected_assignment_error,
                float(
                    (
                        edited[rows, windows[window_name]]
                        - source_logits[rows, windows[window_name]]
                    )
                    .abs()
                    .max()
                ),
            )
            maximum_unselected_error = max(
                maximum_unselected_error,
                float((edited[~selected_mask] - clean_logits[~selected_mask]).abs().max()),
            )
            maximum_value_error = max(
                maximum_value_error,
                float((clean_values - _head_values(states["clean"][layer], head, adapter.num_heads)).abs().max()),
            )
            attention = torch.softmax(edited, dim=-1)
            if (
                not torch.isfinite(attention).all()
                or float((attention.sum(dim=-1) - 1.0).abs().max()) >= HOOK_ATOL
            ):
                raise RuntimeError("R5a edited QK attention row is invalid")
            context = reconstruct_head_context(attention, clean_values)
            margin = _inject_query_context_margin(
                adapter,
                task,
                encoded_clean,
                readout_positions,
                query_tokens,
                states["clean"][layer]["heads"],
                layer=layer,
                head=head,
                context=context,
            )
            output[head_name][window_name][leg] = clean_margin - margin
    return output, {
        "exactly_four_logits_replaced": True,
        "selected_logit_assignment_max_abs_error": maximum_selected_assignment_error,
        "unselected_logit_max_abs_error": maximum_unselected_error,
        "fixed_value_max_abs_error": maximum_value_error,
        "fixed_clean_value_sha256_by_arm": fixed_value_sha256_by_arm,
        "same_clean_value_tensor_used_for_every_leg": True,
        "ok": maximum_selected_assignment_error == 0.0
        and maximum_unselected_error == 0.0
        and maximum_value_error == 0.0,
    }


def run_av_batch(
    adapter,
    task: CouplingTask,
    encoded_clean: Mapping[str, torch.Tensor],
    encoded_partner: Mapping[str, torch.Tensor],
    encoded_pseudo: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
    paired_positions: Sequence[int],
    pseudo_positions: Sequence[int],
    adjacent_positions: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_tokens = torch.tensor(
        [position + task.off for position in readout_positions],
        dtype=torch.long,
        device=adapter.device,
    )
    windows = {
        "paired": _window_tensor(paired_positions, token_offset=task.off, device=adapter.device),
        "pseudo": _window_tensor(pseudo_positions, token_offset=task.off, device=adapter.device),
        "adjacent": _window_tensor(adjacent_positions, token_offset=task.off, device=adapter.device),
    }
    states = {
        "clean": _capture_qkv_and_heads(adapter, encoded_clean),
        "partner": _capture_qkv_and_heads(adapter, encoded_partner),
        "pseudo": _capture_qkv_and_heads(adapter, encoded_pseudo),
    }
    clean_a = _head_rows(
        states["clean"], encoded_clean, query_tokens, head=READER[1], num_heads=adapter.num_heads
    )[1]
    clean_v = _head_values(states["clean"], READER[1], adapter.num_heads)
    output: dict[str, Any] = {window: {} for window in WINDOWS}
    baseline_error = 0.0
    reconstruction_error = 0.0
    value_window_assignment_error = 0.0
    value_outside_window_error = 0.0
    restore_lift: dict[str, np.ndarray] = {}
    rows = torch.arange(query_tokens.shape[0], device=adapter.device)
    native_context = states["clean"]["heads"][rows, READER[1], query_tokens]
    reconstruction_error = max(
        reconstruction_error,
        float((reconstruct_head_context(clean_a, clean_v) - native_context).abs().max()),
    )
    for leg, encoded in (("partner", encoded_partner), ("pseudo", encoded_pseudo)):
        state = states[leg]
        minus_a = _head_rows(
            state, encoded, query_tokens, head=READER[1], num_heads=adapter.num_heads
        )[1]
        minus_v = _head_values(state, READER[1], adapter.num_heads)
        minus_native_context = state["heads"][rows, READER[1], query_tokens]
        reconstruction_error = max(
            reconstruction_error,
            float(
                (
                    reconstruct_head_context(minus_a, minus_v)
                    - minus_native_context
                )
                .abs()
                .max()
            ),
        )
        native_margin = task.margin(state["logits"], readout_positions).cpu().numpy()
        restored_margin = _inject_query_context_margin(
            adapter,
            task,
            encoded,
            readout_positions,
            query_tokens,
            state["heads"],
            layer=READER[0],
            head=READER[1],
            context=native_context,
        )
        restore_lift[leg] = restored_margin - native_margin
        for window in WINDOWS:
            mixed_v = replace_value_window(minus_v, clean_v, windows[window])
            value_rows = torch.arange(mixed_v.shape[0], device=mixed_v.device)[:, None]
            value_window_assignment_error = max(
                value_window_assignment_error,
                float(
                    (
                        mixed_v[value_rows, windows[window]]
                        - clean_v[value_rows, windows[window]]
                    )
                    .abs()
                    .max()
                ),
            )
            selected_mask = torch.zeros(
                mixed_v.shape[:2], dtype=torch.bool, device=mixed_v.device
            )
            selected_mask[value_rows, windows[window]] = True
            value_outside_window_error = max(
                value_outside_window_error,
                float((mixed_v[~selected_mask] - minus_v[~selected_mask]).abs().max()),
            )
            cells = {
                "baseline": reconstruct_head_context(minus_a, minus_v),
                "address_only": reconstruct_head_context(clean_a, minus_v),
                "value_only": reconstruct_head_context(minus_a, mixed_v),
                "joint": reconstruct_head_context(clean_a, mixed_v),
            }
            output[window][leg] = {}
            for name in AV_CELL_ORDER:
                margin = _inject_query_context_margin(
                    adapter,
                    task,
                    encoded,
                    readout_positions,
                    query_tokens,
                    state["heads"],
                    layer=READER[0],
                    head=READER[1],
                    context=cells[name],
                )
                output[window][leg][name] = margin - native_margin
                if name == "baseline":
                    baseline_error = max(
                        baseline_error, float(np.max(np.abs(margin - native_margin)))
                    )
    return output, {
        "a_at_v_reconstruction_max_abs_error": reconstruction_error,
        "baseline_inject_max_abs_error": baseline_error,
        "value_window_assignment_max_abs_error": value_window_assignment_error,
        "value_outside_window_max_abs_error": value_outside_window_error,
        "address_source_scope": "complete L3H1 acceptor-query attention row",
        "value_source_scope": "selected four-token window; recipient values elsewhere",
        "clean_L3H1_same_example_restore_lift": restore_lift,
        "clean_L3H1_restore_is_real_query_head_path": True,
        "ok": reconstruction_error < HOOK_ATOL
        and baseline_error < MARGIN_ATOL
        and value_window_assignment_error == 0.0
        and value_outside_window_error == 0.0,
    }


def run_l2_batch(
    adapter,
    task: CouplingTask,
    encoded_clean: Mapping[str, torch.Tensor],
    encoded_partner: Mapping[str, torch.Tensor],
    encoded_pseudo: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_tokens = torch.tensor(
        [position + task.off for position in readout_positions],
        dtype=torch.long,
        device=adapter.device,
    )
    output: dict[str, Any] = {builder: {} for builder in BUILDERS}
    baseline_error = 0.0
    for builder_name in BUILDERS:
        builder = head_tuple(builder_name)
        intact = {
            "clean": _capture_qkv_and_heads(adapter, encoded_clean),
            "partner": _capture_qkv_and_heads(adapter, encoded_partner),
            "pseudo": _capture_qkv_and_heads(adapter, encoded_pseudo),
        }
        with edit_heads(adapter, [builder], mode="within_example_mean"):
            ablated = {
                "clean": _capture_qkv_and_heads(adapter, encoded_clean),
                "partner": _capture_qkv_and_heads(adapter, encoded_partner),
                "pseudo": _capture_qkv_and_heads(adapter, encoded_pseudo),
            }
        for leg, encoded in (("partner", encoded_partner), ("pseudo", encoded_pseudo)):
            intact_coupling = (
                task.margin(intact["clean"]["logits"], readout_positions)
                - task.margin(intact[leg]["logits"], readout_positions)
            ).cpu().numpy()
            ablated_coupling = (
                task.margin(ablated["clean"]["logits"], readout_positions)
                - task.margin(ablated[leg]["logits"], readout_positions)
            ).cpu().numpy()
            cell_couplings: dict[str, np.ndarray] = {}
            for state_name, state_encoded in (("clean", encoded_clean), (leg, encoded)):
                intact_a = _head_rows(
                    intact[state_name],
                    state_encoded,
                    query_tokens,
                    head=READER[1],
                    num_heads=adapter.num_heads,
                )[1]
                ablated_a = _head_rows(
                    ablated[state_name],
                    state_encoded,
                    query_tokens,
                    head=READER[1],
                    num_heads=adapter.num_heads,
                )[1]
                intact_v = _head_values(intact[state_name], READER[1], adapter.num_heads)
                ablated_v = _head_values(ablated[state_name], READER[1], adapter.num_heads)
                contexts = av_four_cell_reconstruction(
                    intact_a.cpu().numpy(),
                    ablated_a.cpu().numpy(),
                    intact_v.cpu().numpy(),
                    ablated_v.cpu().numpy(),
                )
                margins: dict[str, np.ndarray] = {}
                for cell in AV_CELL_ORDER:
                    margins[cell] = _inject_query_context_margin(
                        adapter,
                        task,
                        state_encoded,
                        readout_positions,
                        query_tokens,
                        ablated[state_name]["heads"],
                        layer=READER[0],
                        head=READER[1],
                        context=torch.as_tensor(
                            contexts[cell], device=adapter.device, dtype=ablated_v.dtype
                        ),
                        outer_context=lambda builder=builder: edit_heads(
                            adapter, [builder], mode="within_example_mean"
                        ),
                    )
                if state_name == "clean":
                    clean_margins = margins
                else:
                    corrupt_margins = margins
            for cell in AV_CELL_ORDER:
                cell_couplings[cell] = clean_margins[cell] - corrupt_margins[cell]
            output[builder_name][leg] = {
                cell: cell_couplings[cell] - ablated_coupling for cell in AV_CELL_ORDER
            }
            baseline_error = max(
                baseline_error,
                float(np.max(np.abs(output[builder_name][leg]["baseline"]))),
            )
            # This scalar is recorded only as path-continuity context, never inserted into
            # the factorial family or used as a substitute for a frozen cell.
            output[builder_name][leg]["native_builder_contribution"] = (
                intact_coupling - ablated_coupling
            )
    return output, {
        "baseline_rescue_max_abs_error": baseline_error,
        "ok": baseline_error < MARGIN_ATOL,
    }


def _normalized_runner_sha256(path: Path = RUNNER) -> str:
    text = path.read_text()
    names = (
        "FROZEN_PROTOCOL_SHA256",
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256",
        "FROZEN_PREFLIGHT_ARTIFACT_SHA256",
        "FROZEN_R3C_ARTIFACT_SHA256",
        "FROZEN_CHECKPOINT_AGGREGATE_SHA256",
        "FROZEN_COMPLETE_ELIGIBLE_GENES",
        "FROZEN_BUCKET_ELIGIBLE_GENES",
        "FROZEN_SELECTED_COHORT_SHA256",
        "FROZEN_CONSTRUCTION_SHA256",
        "FROZEN_EXCLUDED_GENE_REGISTRY_SHA256",
    )
    for name in names:
        text, count = re.subn(
            rf"^{name}:?[^=]*= .*?$",
            f"{name}=<NORMALIZED_PIN>",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(f"could not normalize runner pin {name}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def instrument_binding_payload() -> dict[str, Any]:
    sources = (
        PROTOCOL,
        R5B_PROTOCOL,
        R5C_PROTOCOL,
        R3C_RUNNER,
        MECHANISM_CORE,
        COHORT_CORE,
        STAT_CORE,
        HOOK_CORE,
        EMBED_CORE,
        ASSAY_CORE,
        CIRCUIT_CORE,
    )
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"R5a binding source missing: {missing[0]}")
    stack = build_r5_stack_binding(repo=REPO)
    return {
        "schema": "gfmcd.r5a.instrument_binding/1",
        "r5_stack_binding": stack,
        "r5_stack_fingerprint_sha256": r5_stack_fingerprint(stack),
        "normalized_runner_sha256": _normalized_runner_sha256(),
        "source_sha256": {str(path.relative_to(REPO)): sha256_file(path) for path in sources},
        "constants": {
            "n": PRODUCTION_N,
            "batch": PRODUCTION_BATCH,
            "seed": PRODUCTION_SEED,
            "randomisation": PRODUCTION_RANDOMISATION,
            "calibration_trials": PRODUCTION_CALIBRATION_TRIALS,
            "calibration_epsilon": CALIBRATION_EPSILON,
            "calibration_primitive_noise_sd": CALIBRATION_PRIMITIVE_NOISE_SD,
            "health_maximum_examples": HEALTH_MAX_EXAMPLES,
            "health_probes_per_example": HEALTH_PROBES_PER_EXAMPLE,
            "health_floor": HEALTH_FLOOR,
            "sesoi": SESOI,
            "stage_alpha": STAGE_ALPHA,
            "family_alpha": FAMILY_ALPHA,
            "pseudo_caliper_nt": DECOY_TOL,
            "cohort_schema": "gfmcd.r5.acceptor_mechanism_cohort/2",
            "adjacent_window_policy": {
                "salt": R5_ADJACENT_SALT,
                "shells": 8,
                "tokens_per_window": 4,
                "maximum_absolute_offset_from_donor": 34,
                "selection": "nearest admissible shell; hash-fixed side order within shell",
            },
            "reader": list(READER),
            "address_builder": list(ADDRESS_BUILDER),
            "negative_head": list(NEGATIVE_HEAD),
            "family_order": list(FAMILY_ORDER),
            "family_rows_sha256": {
                name: canonical_sha256(rows) for name, rows in family_row_names().items()
            },
            "path_registry_sha256": canonical_sha256(named_path_specs()),
            "behavioral_endpoint": "masked_acceptor_G_margin_only",
            "donor_readout_constructed": False,
        },
    }


def _all_freeze_pins_present(*, include_preflight: bool = True) -> bool:
    hash_pins = (
        FROZEN_PROTOCOL_SHA256,
        FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
        FROZEN_R3C_ARTIFACT_SHA256,
        FROZEN_CHECKPOINT_AGGREGATE_SHA256,
        FROZEN_EXCLUDED_GENE_REGISTRY_SHA256,
    ) + ((FROZEN_PREFLIGHT_ARTIFACT_SHA256,) if include_preflight else ())
    cohort_hashes = (FROZEN_SELECTED_COHORT_SHA256, FROZEN_CONSTRUCTION_SHA256)
    return bool(
        all(_is_sha256(value) for value in hash_pins)
        and isinstance(FROZEN_COMPLETE_ELIGIBLE_GENES, int)
        and FROZEN_COMPLETE_ELIGIBLE_GENES >= 2 * PRODUCTION_N
        and isinstance(FROZEN_BUCKET_ELIGIBLE_GENES, tuple)
        and len(FROZEN_BUCKET_ELIGIBLE_GENES) == 2
        and all(isinstance(value, int) and value >= PRODUCTION_N for value in FROZEN_BUCKET_ELIGIBLE_GENES)
        and all(
            isinstance(pair, tuple)
            and len(pair) == 2
            and all(_is_sha256(value) for value in pair)
            for pair in cohort_hashes
        )
    )


def authorize_model_scoring(stage: str) -> dict[str, Any]:
    """Fail before resolving any model/data path unless the entire R5a seal is literal."""

    if stage not in {"preflight", "target"}:
        raise SealError("model scoring requires preflight or target stage")
    if not _all_freeze_pins_present(include_preflight=stage == "target"):
        raise SealError(
            "R5a remains DRAFT: model scoring requires all literal method, census, "
            "construction, instrument, and preflight pins"
        )
    protocol_text = PROTOCOL.read_text()
    if "Status: FROZEN" not in protocol_text:
        raise SealError("R5a protocol does not declare FROZEN status")
    protocol_hash = sha256_file(PROTOCOL)
    if protocol_hash != FROZEN_PROTOCOL_SHA256:
        raise SealError("R5a protocol bytes differ from the frozen pin")
    binding = instrument_binding_payload()
    fingerprint = canonical_sha256(binding)
    if fingerprint != FROZEN_INSTRUMENT_FINGERPRINT_SHA256:
        raise SealError("R5a instrument fingerprint differs from the frozen pin")
    if stage == "target" and not _is_sha256(FROZEN_PREFLIGHT_ARTIFACT_SHA256):
        raise SealError("R5a target requires a literal reviewed preflight artifact pin")
    return {
        "binding": binding,
        "fingerprint": fingerprint,
        "r5_stack_binding": binding["r5_stack_binding"],
        "r5_stack_fingerprint_sha256": binding[
            "r5_stack_fingerprint_sha256"
        ],
    }


def _load_r3c_runner():
    path = R3C_RUNNER
    if not path.is_file():
        raise RuntimeError("R5a requires the reviewed R3c cohort runner")
    spec = importlib.util.spec_from_file_location("gfmcd_r3c_cohort_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the R3c cohort runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_model_free_reservations(
    gtf: str, fasta: str
) -> tuple[dict[int, tuple[Any, ...]], dict[str, Any], dict[str, Any]]:
    """Reconstruct historical/R3c exclusions and the exact shared R5 reservations."""

    source_identity = validate_expected_r5_source_identity(gtf, fasta)
    r3c = _load_r3c_runner()
    common = {"gtf": gtf, "fasta": fasta, "n": r3c.PREFLIGHT_N}
    _, _, _, bucket0_audit = r3c.load_cohort(
        SimpleNamespace(stage="preflight", **common)
    )
    _, _, _, bucket1_audit = r3c.load_cohort(
        SimpleNamespace(stage="target", **common)
    )
    historical = bucket0_audit["historical_reconstruction"]["excluded_gene_ids"]
    r3c_genes = sorted(
        set(bucket0_audit["selection"]["gene_ids"])
        | set(bucket1_audit["selection"]["gene_ids"])
    )
    examples = load_examples_unscored(
        gtf,
        fasta,
        "val",
        None,
        tag="r5-acceptor-mechanism-v1",
        allowed_chromosomes=R5_VALIDATION_CHROMOSOMES,
    )
    pseudo_donors = {
        str(example.example_id): matched_decoy_donor(example) for example in examples
    }
    selected, audit = build_r5_reservations(
        examples,
        pseudo_donors,
        historical_r3_gene_ids=historical,
        r3c_gene_ids=r3c_genes,
        rows_per_bucket=R5_ROWS_PER_BUCKET,
        require_capacity=True,
    )
    validate_r5_reservation_audit(audit, rows_per_bucket=R5_ROWS_PER_BUCKET)
    validate_expected_r5_census_identity(audit)
    return selected, audit, source_identity


def model_free_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Run the shared annotation/sequence-only census without loading a model."""

    _selected, audit, source_identity = build_model_free_reservations(
        args.gtf, args.fasta
    )
    identity = r5_census_identity(audit)
    binding = instrument_binding_payload()
    return {
        "schema": "gfmcd.r5a.model_free_audit/1",
        "model_loaded": False,
        "model_outputs_read": False,
        "allowed_chromosomes": list(R5_VALIDATION_CHROMOSOMES),
        "sealed_chr8_chr9_opened": False,
        "pins_are_nonbinding_none": not _all_freeze_pins_present(),
        "instrument_binding": binding,
        "instrument_fingerprint_sha256": canonical_sha256(binding),
        "r5_stack_binding": binding["r5_stack_binding"],
        "r5_stack_fingerprint_sha256": binding[
            "r5_stack_fingerprint_sha256"
        ],
        "cohort_audit": audit,
        "expected_census_identity": R5_EXPECTED_CENSUS_IDENTITY,
        "observed_census_identity": identity,
        "expected_source_identity": R5_EXPECTED_SOURCE_IDENTITY,
        "observed_source_identity": source_identity,
        "future_pin_payload": {
            "complete_eligible_genes": identity["eligible_genes"],
            "bucket_eligible_genes": [
                identity["available_genes_by_bucket"]["0"],
                identity["available_genes_by_bucket"]["1"],
            ],
            "selected_cohort_sha256": [
                identity["selected_by_bucket"][str(bucket)]["ordered_rows_sha256"]
                for bucket in (0, 1)
            ],
            "construction_sha256": [
                identity["selected_by_bucket"][str(bucket)]["construction_sha256"]
                for bucket in (0, 1)
            ],
            "excluded_gene_registry_sha256": identity[
                "combined_excluded_gene_ids_sha256"
            ],
        },
    }


def validate_frozen_cohort_pins(audit: Mapping[str, Any]) -> None:
    identity = validate_expected_r5_census_identity(audit)
    observed_counts = (
        identity["available_genes_by_bucket"]["0"],
        identity["available_genes_by_bucket"]["1"],
    )
    observed_rows = tuple(
        identity["selected_by_bucket"][str(bucket)]["ordered_rows_sha256"]
        for bucket in (0, 1)
    )
    observed_constructions = tuple(
        identity["selected_by_bucket"][str(bucket)]["construction_sha256"]
        for bucket in (0, 1)
    )
    if identity["eligible_genes"] != FROZEN_COMPLETE_ELIGIBLE_GENES:
        raise RuntimeError("R5a complete eligible-gene pin differs")
    if observed_counts != FROZEN_BUCKET_ELIGIBLE_GENES:
        raise RuntimeError("R5a bucket-capacity pins differ")
    if observed_rows != FROZEN_SELECTED_COHORT_SHA256:
        raise RuntimeError("R5a selected-cohort pins differ")
    if observed_constructions != FROZEN_CONSTRUCTION_SHA256:
        raise RuntimeError("R5a construction pins differ")
    if (
        identity["combined_excluded_gene_ids_sha256"]
        != FROZEN_EXCLUDED_GENE_REGISTRY_SHA256
    ):
        raise RuntimeError("R5a excluded-gene registry pin differs")


def entry_behavior_inference(
    clean_margin: Sequence[float],
    partner_margin: Sequence[float],
    pseudo_margin: Sequence[float],
    groups: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    namespace: str,
) -> dict[str, Any]:
    clean = np.asarray(clean_margin, dtype=float)
    partner = clean - np.asarray(partner_margin, dtype=float)
    pseudo = clean - np.asarray(pseudo_margin, dtype=float)
    family = np.vstack((partner, pseudo, partner - pseudo))
    names = ["native_partner", "native_pseudo", "native_specific"]
    decisions, interval = simultaneous_family_decision(
        family,
        groups,
        names,
        sesoi=SESOI,
        n_draws=n_draws,
        alpha=STAGE_ALPHA,
        seed=seed,
        namespace=f"{namespace}:simultaneous",
        min_genes=MIN_GENES,
    )
    edges = boundary_sign_flip_tests(
        np.vstack((partner, pseudo, pseudo, partner - pseudo)),
        groups,
        ["partner", "pseudo_lower", "pseudo_upper", "specific"],
        boundaries=[SESOI, -SESOI, SESOI, SESOI],
        alternatives=["greater", "greater", "less", "greater"],
        n_draws=n_draws,
        seed=seed,
        namespace=f"{namespace}:shared-boundary-edges",
        min_genes=MIN_GENES,
    )
    p_value = intersection_union_pvalue(
        [edge.p_value for edge in edges.values()]
    )
    ci_pass = (
        strict_interval_decision(*decisions["native_partner"].simultaneous_ci, SESOI)
        == "MATERIAL_POSITIVE"
        and strict_interval_decision(*decisions["native_pseudo"].simultaneous_ci, SESOI)
        == "EQUIVALENT_WITHIN_SESOI"
        and strict_interval_decision(*decisions["native_specific"].simultaneous_ci, SESOI)
        == "MATERIAL_POSITIVE"
    )
    return {
        "requirements": {
            "native_partner": "MATERIAL_POSITIVE",
            "native_pseudo": "EQUIVALENT_WITHIN_SESOI",
            "native_specific": "MATERIAL_POSITIVE",
        },
        "iut_p_value": p_value,
        "alpha": STAGE_ALPHA,
        "p_gate_pass": p_value < STAGE_ALPHA,
        "strict_simultaneous_ci_gate_pass": ci_pass,
        "classification": (
            "ENTRY_BEHAVIOR_CONFIRMED"
            if p_value < STAGE_ALPHA and ci_pass
            else "ENTRY_BEHAVIOR_NOT_CONFIRMED"
        ),
        "decisions": {name: decision.to_dict() for name, decision in decisions.items()},
        "interval_calibration": interval,
        "boundary_tests": {name: edge.to_dict() for name, edge in edges.items()},
    }


def _accumulate_tree(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping):
            child = target.setdefault(key, {})
            if not isinstance(child, dict):
                raise RuntimeError("R5a measurement tree type drifted")
            _accumulate_tree(child, value)
        else:
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                array = array[None]
            if array.ndim != 1 or not np.isfinite(array).all():
                raise ValueError("R5a primitive measurement must be a finite vector")
            target.setdefault(key, []).extend(array.tolist())


def _finalize_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finalize_tree(item) for key, item in value.items()}
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise RuntimeError("R5a accumulated primitive is not one-dimensional")
    return array


def _max_invariant(records: Sequence[Mapping[str, Any]], key: str) -> float:
    return max(float(record[key]) for record in records)


def _git_identity_clean() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    if status.stdout.strip():
        raise RuntimeError("R5a model scoring requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if not _is_sha256(commit) and re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise RuntimeError("R5a git commit identity is malformed")
    return {"commit": commit, "branch": branch, "dirty": False}


def _clean_manifest_identity(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise RuntimeError(f"R5a {label} lacks its sibling manifest")
    manifest = json.loads(manifest_path.read_text())
    matches = [
        item
        for item in manifest.get("outputs", [])
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).expanduser().resolve() == path
        and item.get("sha256") == expected_sha256
        and item.get("size_bytes") == path.stat().st_size
    ]
    if len(matches) != 1 or manifest.get("git_dirty") is not False:
        raise RuntimeError(f"R5a {label} is not bound by a clean sibling manifest")
    return {
        "path": str(manifest_path),
        "size_bytes": manifest_path.stat().st_size,
        "sha256": sha256_file(manifest_path),
    }


def derive_r5_exclusion_registry(
    r3c_record: Mapping[str, Any],
    *,
    r3c_artifact_identity: Mapping[str, Any],
    r3c_manifest_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive the only admissible R5 exclusion registry from authenticated R3c."""

    cohort = r3c_record.get("cohort_audit", {})
    historical = list(
        map(
            str,
            cohort.get("historical_reconstruction", {}).get(
                "excluded_gene_ids", []
            ),
        )
    )
    fixed = (
        cohort.get("current_census", {})
        .get("fixed_selection_gene_ids_by_bucket", {})
    )
    if not isinstance(fixed, Mapping) or set(fixed) != {"0", "1"}:
        raise RuntimeError("R5a R3c entry lacks both fixed exclusion buckets")
    per_bucket: dict[str, list[str]] = {}
    for bucket in ("0", "1"):
        genes = list(map(str, fixed[bucket].get("gene_ids", [])))
        if (
            genes != sorted(set(genes))
            or fixed[bucket].get("gene_ids_sha256") != canonical_sha256(genes)
        ):
            raise RuntimeError(f"R5a R3c bucket-{bucket} gene registry differs")
        per_bucket[bucket] = genes
    r3c_genes = sorted(set(per_bucket["0"]) | set(per_bucket["1"]))
    if historical != sorted(set(historical)):
        raise RuntimeError("R5a historical R3 gene registry is not sorted/unique")
    if (
        len(historical) != R5_EXPECTED_CENSUS_IDENTITY["historical_r3_gene_count"]
        or canonical_sha256(historical)
        != R5_EXPECTED_CENSUS_IDENTITY["historical_r3_gene_ids_sha256"]
        or len(r3c_genes) != R5_EXPECTED_CENSUS_IDENTITY["r3c_gene_count"]
        or canonical_sha256(r3c_genes)
        != R5_EXPECTED_CENSUS_IDENTITY["r3c_gene_ids_sha256"]
    ):
        raise RuntimeError("R5a derived exclusion registry differs from the frozen census")
    payload = {
        "schema": "gfmcd.r5.exclusion_registry/1",
        "derivation": "authenticated R3c target cohort audit; no free gene-list input",
        "r3c_artifact": dict(r3c_artifact_identity),
        "r3c_manifest": dict(r3c_manifest_identity),
        "source_identity": R5_EXPECTED_SOURCE_IDENTITY,
        "historical_r3_gene_ids": historical,
        "r3c_gene_ids_by_bucket": per_bucket,
        "r3c_gene_ids": r3c_genes,
        "expected_census_identity_sha256": canonical_sha256(
            R5_EXPECTED_CENSUS_IDENTITY
        ),
    }
    return payload | {"record_sha256": canonical_sha256(payload)}


def _verify_r3c_entry(path_value: str | None) -> dict[str, Any]:
    if path_value is None:
        raise RuntimeError("R5a requires --r3c-artifact")
    path = Path(path_value).resolve()
    if not path.is_file() or sha256_file(path) != FROZEN_R3C_ARTIFACT_SHA256:
        raise RuntimeError("R5a R3c entry artifact differs from its literal pin")
    artifact_sha256 = sha256_file(path)
    manifest_identity = _clean_manifest_identity(
        path, artifact_sha256, label="R3c entry artifact"
    )
    record = json.loads(path.read_text())
    if (
        record.get("schema") != R3C_ENTRY_SCHEMA
        or record.get("instrument_revision") != R3C_INSTRUMENT_REVISION
        or record.get("stage") != "target"
        or record.get("status") != "CONFIRMATORY"
        or record.get("tier") != "HELD_OUT_TARGET"
        or record.get("split") != "val"
        or record.get("gene_bucket") != 1
        or record.get("verdict")
        != "HELD_OUT_ACCEPTOR_HEAD_CONFIRMATION_COMPLETE"
        or record.get("all_gates_pass") is not True
        or record.get("sealed_split_opened") is not False
        or record.get("protocol_status")
        != "FROZEN_PROTOCOL_AND_INSTRUMENT_VERIFIED"
        or record.get("primary_estimand")
        != "raw masked-acceptor-G logit-margin component drop"
    ):
        raise RuntimeError("R5a requires the sealed confirmatory acceptor-only R3c artifact")
    construction = record.get("construction", {})
    policy = construction.get("policy", {})
    if (
        construction.get("schema") != "gfmcd.r3c.acceptor_only_construction/1"
        or construction.get("all_single_acceptor_G") is not True
        or policy.get("readout") != "mask exactly one G at next_acceptor_pos - 1"
        or policy.get("paired_operator") != "GT -> AC at donor_pos"
        or policy.get("donor_readout_constructed") is not False
        or record.get("instrument_binding_payload", {}).get("readout")
        != "next_acceptor_pos - 1 only"
    ):
        raise RuntimeError("R5a requires R3c's exact acceptor-only/no-donor contract")
    try:
        validate_r3c_target_manifest_contract(
            record,
            json.loads(Path(manifest_identity["path"]).read_text()),
        )
    except (OSError, ValueError, R5StackBindingError) as exc:
        raise RuntimeError("R5a R3c sibling manifest context differs") from exc
    paths = record.get("paths", {})
    l3 = paths.get("L3H1_reused_component", {}) if isinstance(paths, Mapping) else {}
    try:
        validate_r3c_target_health_contract(record)
    except R5StackBindingError as exc:
        raise RuntimeError(
            "R5a requires monotone R3c target health and all-path calibration"
        ) from exc
    gates = record.get("preflight_gates", record.get("gates", {}))
    if (
        not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
    ):
        raise RuntimeError("R5a R3c entry artifact contains a failed preflight gate")
    hashes = record.get("hashes", {})
    checkpoint_sha256 = hashes.get("checkpoint_aggregate_sha256")
    sources = hashes.get("sources", {})
    source_sha256 = {key: sources.get(key, {}).get("sha256") for key in ("gtf", "fasta")}
    if not _is_sha256(checkpoint_sha256) or not all(
        _is_sha256(value) for value in source_sha256.values()
    ):
        raise RuntimeError("R5a R3c entry artifact lacks exact checkpoint/source hashes")
    for key in ("gtf", "fasta"):
        observed = sources.get(key, {})
        expected = R5_EXPECTED_SOURCE_IDENTITY[key]
        if (
            observed.get("sha256") != expected["sha256"]
            or observed.get("size_bytes") != expected["size_bytes"]
        ):
            raise RuntimeError("R5a R3c entry uses the wrong source bytes")
    artifact_identity = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": artifact_sha256,
    }
    exclusion_registry = derive_r5_exclusion_registry(
        record,
        r3c_artifact_identity=artifact_identity,
        r3c_manifest_identity=manifest_identity,
    )
    return {
        **artifact_identity,
        "manifest": manifest_identity,
        "L3H1_reused_component": l3,
        "checkpoint_sha256": checkpoint_sha256,
        "source_sha256": source_sha256,
        "cohort_identity": {
            "historical_r3_gene_ids_sha256": R5_EXPECTED_CENSUS_IDENTITY[
                "historical_r3_gene_ids_sha256"
            ],
            "r3c_gene_ids_sha256": R5_EXPECTED_CENSUS_IDENTITY[
                "r3c_gene_ids_sha256"
            ],
        },
        "exclusion_registry_record": exclusion_registry,
    }


def _verify_bound_preflight(
    path_value: str | None,
    fingerprint: str,
    *,
    stack_binding: Mapping[str, Any],
    stack_fingerprint: str,
) -> dict[str, Any]:
    if path_value is None:
        raise RuntimeError("R5a target requires --preflight-artifact")
    path = Path(path_value).resolve()
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError("R5a target requires artifact and sibling manifest")
    if sha256_file(path) != FROZEN_PREFLIGHT_ARTIFACT_SHA256:
        raise RuntimeError("R5a preflight artifact bytes differ from the literal pin")
    manifest = json.loads(manifest_path.read_text())
    expected_outputs = [
        item
        for item in manifest.get("outputs", [])
        if isinstance(item, Mapping)
        and Path(str(item.get("path", ""))).expanduser().resolve() == path
        and item.get("sha256") == sha256_file(path)
        and item.get("size_bytes") == path.stat().st_size
    ]
    if len(expected_outputs) != 1 or manifest.get("git_dirty") is not False:
        raise RuntimeError("R5a preflight sibling manifest does not bind artifact bytes")
    record = json.loads(path.read_text())
    if (
        record.get("schema") != "gfmcd.r5a.acceptor_address_binding/1"
        or record.get("stage") != "preflight"
        or record.get("status") != "PREFLIGHT_PASS"
        or record.get("instrument_fingerprint_sha256") != fingerprint
        or record.get("all_instrument_gates_pass") is not True
    ):
        raise RuntimeError("R5a preflight semantic binding differs")
    try:
        validate_serialized_r5_stack(record, current=stack_binding)
    except R5StackBindingError as exc:
        raise RuntimeError("R5a preflight joint R5 stack differs") from exc
    if record.get("r5_stack_fingerprint_sha256") != stack_fingerprint:
        raise RuntimeError("R5a preflight joint R5 stack fingerprint differs")
    if (
        record.get("entry_behavior", {}).get("classification")
        != "ENTRY_BEHAVIOR_CONFIRMED"
        or record.get("clean_L3H1_path_continuity", {}).get("classification")
        != "L3_PATH_CONFIRMED"
    ):
        raise RuntimeError("R5a preflight entry/path-continuity gate did not pass")
    raw = record.get("raw_family_values")
    groups = record.get("gene_ids")
    if not isinstance(raw, Mapping) or not isinstance(groups, list):
        raise RuntimeError("R5a preflight lacks replayable family rows")
    matrices = {
        family: np.asarray(raw[family], dtype=float) for family in FAMILY_ORDER
    }
    sufficient = record.get("gene_sufficient_statistics", {})
    expected_matrix_hashes = sufficient.get("family_matrix_sha256", {})
    if any(
        _array_sha256(matrices[family]) != expected_matrix_hashes.get(family)
        for family in FAMILY_ORDER
    ):
        raise RuntimeError("R5a preflight gene sufficient-statistic hashes differ")
    replay = infer_all_families(
        matrices,
        groups,
        n_draws=PRODUCTION_RANDOMISATION,
        seed=PRODUCTION_SEED,
        namespace="r5a:preflight:observed",
        min_genes=MIN_GENES,
    )
    if canonical_sha256(replay) != record.get("inference_sha256"):
        raise RuntimeError("R5a preflight inference does not replay from stored rows")
    calibration = record.get("classifier_calibration")
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("all_pass") is not True
        or calibration.get("n_genes") != PRODUCTION_N
        or calibration.get("n_draws") != PRODUCTION_RANDOMISATION
        or calibration.get("n_trials") != PRODUCTION_CALIBRATION_TRIALS
        or calibration.get("family_alpha") != FAMILY_ALPHA
        or calibration.get("epsilon_cal") != CALIBRATION_EPSILON
        or calibration.get("primitive_noise_sd")
        != CALIBRATION_PRIMITIVE_NOISE_SD
        or calibration.get("boundary_signature_count") != 27
        or set(calibration.get("plants", {})) != set(PATH_ORDER)
        or canonical_sha256(calibration.get("boundary_signature_registry", {}))
        != calibration.get("boundary_signature_registry_sha256")
        or canonical_sha256(calibration.get("plant_registry", {}))
        != calibration.get("plant_registry_sha256")
    ):
        raise RuntimeError("R5a preflight classifier calibration is not passing")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "record": record,
    }


def validate_bound_preflight_context(
    preflight: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    r3c_entry: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    cohort_audit: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the inherited bucket-0 preflight to match the current target context."""

    record = preflight.get("record")
    if not isinstance(record, Mapping):
        raise RuntimeError("R5a preflight record is missing")
    expected_bucket0 = cohort_audit["selected_by_bucket"]["0"]
    r3c_fields = (
        "sha256",
        "size_bytes",
        "checkpoint_sha256",
        "source_sha256",
        "cohort_identity",
    )
    r3c_matches = all(
        record.get("r3c_entry", {}).get(field) == r3c_entry.get(field)
        for field in r3c_fields
    ) and all(
        record.get("r3c_entry", {}).get("manifest", {}).get(field)
        == r3c_entry.get("manifest", {}).get(field)
        for field in ("sha256", "size_bytes")
    )
    stored_checkpoint = record.get("checkpoint", {})
    checkpoint_matches = (
        stored_checkpoint.get("aggregate_sha256")
        == checkpoint.get("aggregate_sha256")
        and stored_checkpoint.get("ordered_bundle")
        == checkpoint.get("ordered_bundle")
    )
    checks = {
        "bucket0_preflight": record.get("bucket") == 0,
        "instrument_fingerprint": record.get("instrument_fingerprint_sha256")
        == authorization["fingerprint"],
        "instrument_binding": record.get("instrument_binding")
        == authorization["binding"],
        "joint_stack": record.get("r5_stack_binding")
        == authorization["r5_stack_binding"]
        and record.get("r5_stack_fingerprint_sha256")
        == authorization["r5_stack_fingerprint_sha256"],
        "r3c_artifact_and_manifest": r3c_matches,
        "source_identity": record.get("source_identity") == source_identity,
        "checkpoint_identity": checkpoint_matches,
        "cohort_census": record.get("cohort_audit") == cohort_audit,
        "bucket0_cohort": record.get("construction_sha256")
        == expected_bucket0["construction_sha256"]
        and record.get("r5_entry_contract", {}).get("cohort_sha256")
        == expected_bucket0["ordered_rows_sha256"],
        "runtime_identity": record.get("runtime_identity") == runtime,
        "acceptor_only": record.get("behavioral_endpoint")
        == "masked_acceptor_G_margin_only"
        and record.get("donor_readout_constructed") is False,
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"R5a target/preflight context differs: {failed}")
    return {
        "schema": "gfmcd.r5a.preflight_context/1",
        "all_match": True,
        "checks": checks,
    }


def _checkpoint_identity(model_path: str) -> dict[str, Any]:
    root = Path(model_path).expanduser()
    if not root.is_absolute():
        root = REPO / root
    if not root.is_dir():
        raise RuntimeError("R5a checkpoint root is not a directory")
    root = root.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("R5a checkpoint bundle contains no regular files")
    records = {
        str(path.resolve().relative_to(root)): {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    }
    ordered_bundle = [
        {
            "relative_path": relative_path,
            "size_bytes": identity["size_bytes"],
            "sha256": identity["sha256"],
        }
        for relative_path, identity in records.items()
    ]
    return {
        "root": str(root),
        "files": records,
        "ordered_bundle": ordered_bundle,
        "aggregate_schema": "ordered relative_path/size_bytes/sha256 list (R3c-compatible)",
        "aggregate_sha256": canonical_sha256(ordered_bundle),
    }


def verify_input_snapshots_after_run(
    args: argparse.Namespace,
    *,
    authorization: Mapping[str, Any],
    git_before: Mapping[str, Any],
    source_before: Mapping[str, Any],
    checkpoint_before: Mapping[str, Any],
    r3c_before: Mapping[str, Any],
    preflight_before: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Rehash every long-lived input before permitting an immutable artifact."""

    source_after = validate_expected_r5_source_identity(args.gtf, args.fasta)
    checkpoint_after = _checkpoint_identity(args.model)
    if canonical_sha256(source_after) != canonical_sha256(source_before):
        raise RuntimeError("R5a GTF/FASTA input changed during the run")
    if canonical_sha256(checkpoint_after) != canonical_sha256(checkpoint_before):
        raise RuntimeError("R5a checkpoint input changed during the run")
    if sha256_file(Path(str(r3c_before["path"]))) != r3c_before["sha256"]:
        raise RuntimeError("R5a R3c entry artifact changed during the run")
    r3c_manifest = r3c_before.get("manifest", {})
    if (
        not isinstance(r3c_manifest, Mapping)
        or sha256_file(Path(str(r3c_manifest.get("path", ""))))
        != r3c_manifest.get("sha256")
    ):
        raise RuntimeError("R5a R3c entry manifest changed during the run")
    if preflight_before is not None:
        if sha256_file(Path(str(preflight_before["path"]))) != preflight_before["sha256"]:
            raise RuntimeError("R5a bound preflight artifact changed during the run")
        if (
            sha256_file(Path(str(preflight_before["manifest_path"])))
            != preflight_before["manifest_sha256"]
        ):
            raise RuntimeError("R5a bound preflight manifest changed during the run")
    binding_after = instrument_binding_payload()
    if canonical_sha256(binding_after) != authorization["fingerprint"]:
        raise RuntimeError("R5a instrument sources changed during the run")
    git_after = _git_identity_clean()
    if dict(git_after) != dict(git_before):
        raise RuntimeError("R5a code revision changed during the run")
    return {
        "checked_after_model_and_inference": True,
        "source_identity_unchanged": True,
        "checkpoint_identity_unchanged": True,
        "r3c_entry_unchanged": True,
        "bound_preflight_unchanged": preflight_before is not None,
        "instrument_binding_unchanged": True,
        "git_identity_unchanged": True,
    }


def run_model_stage(
    args: argparse.Namespace,
    *,
    authorization: Mapping[str, Any],
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    git_identity = _git_identity_clean()
    r3c_entry = _verify_r3c_entry(args.r3c_artifact)
    bound_preflight = (
        _verify_bound_preflight(
            args.preflight_artifact,
            authorization["fingerprint"],
            stack_binding=authorization["r5_stack_binding"],
            stack_fingerprint=authorization["r5_stack_fingerprint_sha256"],
        )
        if args.stage == "target"
        else None
    )
    reservations, cohort_audit, source_identity = build_model_free_reservations(
        args.gtf, args.fasta
    )
    if r3c_entry["source_sha256"] != {
        key: source_identity[key]["sha256"] for key in ("gtf", "fasta")
    }:
        raise RuntimeError("R5a source bytes differ from the sealed R3c entry source")
    validate_frozen_cohort_pins(cohort_audit)
    bucket = 0 if args.stage == "preflight" else 1
    rows = list(reservations[bucket])
    if len(rows) != PRODUCTION_N or len({row.example.gene_id for row in rows}) != MIN_GENES:
        raise RuntimeError("R5a selected reservation is not exactly 128 one-row-per-gene rows")

    checkpoint = _checkpoint_identity(args.model)
    if checkpoint["aggregate_sha256"] != FROZEN_CHECKPOINT_AGGREGATE_SHA256:
        raise RuntimeError("R5a checkpoint bundle differs from its literal pin")
    if checkpoint["aggregate_sha256"] != r3c_entry["checkpoint_sha256"]:
        raise RuntimeError("R5a checkpoint differs from the sealed R3c entry checkpoint")
    runtime = stable_runtime_identity()
    preflight_context = (
        validate_bound_preflight_context(
            bound_preflight,
            authorization=authorization,
            r3c_entry=r3c_entry,
            source_identity=source_identity,
            checkpoint=checkpoint,
            cohort_audit=cohort_audit,
            runtime=runtime,
        )
        if bound_preflight is not None
        else None
    )
    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=True)
    adapter.model.eval()
    task = CouplingTask(adapter, batch=args.batch)
    primitive_accumulator: dict[str, Any] = {
        family: {} for family in FAMILY_ORDER
    }
    native = {"clean": [], "partner": [], "pseudo": []}
    l3_restore_lift = {"partner": [], "pseudo": []}
    position_invariants: list[dict[str, Any]] = []
    qk_invariants: list[dict[str, Any]] = []
    av_invariants: list[dict[str, Any]] = []
    l2_invariants: list[dict[str, Any]] = []
    full_mlm: dict[str, Any] | None = None
    output_projection: dict[str, Any] | None = None
    full_head_noop: dict[str, Any] | None = None

    with torch.no_grad():
        for start in range(0, len(rows), args.batch):
            batch_rows = rows[start : start + args.batch]
            readout = [row.acceptor_g for row in batch_rows]
            paired = [int(row.example.donor_pos) for row in batch_rows]
            pseudo = [int(row.pseudo_donor) for row in batch_rows]
            adjacent_anchor = [int(row.adjacent_window[1]) for row in batch_rows]
            clean = task.encode([row.clean_sequence for row in batch_rows], readout)
            partner = task.encode(
                [row.paired_corrupt_sequence for row in batch_rows], readout
            )
            pseudo_encoded = task.encode(
                [row.pseudo_corrupt_sequence for row in batch_rows], readout
            )
            clean_logits = adapter.model(**clean).logits.float()
            partner_logits = adapter.model(**partner).logits.float()
            pseudo_logits = adapter.model(**pseudo_encoded).logits.float()
            native["clean"].extend(task.margin(clean_logits, readout).cpu().tolist())
            native["partner"].extend(task.margin(partner_logits, readout).cpu().tolist())
            native["pseudo"].extend(task.margin(pseudo_logits, readout).cpu().tolist())
            if full_mlm is None:
                full_mlm = full_mlm_replay_capability(adapter, task, clean, readout)
                state = _capture_qkv_and_heads(adapter, clean, layer=READER[0])
                output_projection = attention_projection_capability(adapter, state)
                full_head_noop = full_head_noop_capability(
                    adapter, task, clean, state, readout
                )

            position_values, position_check = run_position_shift_batch(
                adapter,
                task,
                clean,
                readout,
                paired,
                pseudo,
                adjacent_anchor,
            )
            qk_values, qk_check = run_qk_batch(
                adapter,
                task,
                clean,
                partner,
                pseudo_encoded,
                readout,
                paired,
                pseudo,
                adjacent_anchor,
            )
            av_values, av_check = run_av_batch(
                adapter,
                task,
                clean,
                partner,
                pseudo_encoded,
                readout,
                paired,
                pseudo,
                adjacent_anchor,
            )
            l2_values, l2_check = run_l2_batch(
                adapter,
                task,
                clean,
                partner,
                pseudo_encoded,
                readout,
            )
            _accumulate_tree(
                primitive_accumulator["position_shift"], position_values
            )
            _accumulate_tree(primitive_accumulator["qk_surgery"], qk_values)
            _accumulate_tree(
                primitive_accumulator["attention_value"], av_values
            )
            _accumulate_tree(primitive_accumulator["l2_role"], l2_values)
            position_invariants.append(position_check)
            qk_invariants.append(qk_check)
            av_invariants.append(av_check)
            for leg in ("partner", "pseudo"):
                l3_restore_lift[leg].extend(
                    np.asarray(
                        av_check["clean_L3H1_same_example_restore_lift"][leg],
                        dtype=float,
                    ).tolist()
                )
            l2_invariants.append(l2_check)
            print(f"R5a {args.stage}: {min(start + args.batch, len(rows))}/{len(rows)}", flush=True)

        health = bounded_intervention_health(
            adapter,
            task,
            rows,
            seed=PRODUCTION_SEED,
        )

    measurements = _finalize_tree(primitive_accumulator)
    matrices = build_family_matrices(measurements)
    genes = [str(row.example.gene_id) for row in rows]
    inference = infer_all_families(
        matrices,
        genes,
        n_draws=PRODUCTION_RANDOMISATION,
        seed=PRODUCTION_SEED,
        namespace=f"r5a:{args.stage}:observed",
        min_genes=MIN_GENES,
    )
    entry = entry_behavior_inference(
        native["clean"],
        native["partner"],
        native["pseudo"],
        genes,
        n_draws=PRODUCTION_RANDOMISATION,
        seed=PRODUCTION_SEED,
        namespace=f"r5a:{args.stage}:entry",
    )
    l3_restore_path = entry_behavior_inference(
        np.zeros(len(genes), dtype=float),
        -np.asarray(l3_restore_lift["partner"], dtype=float),
        -np.asarray(l3_restore_lift["pseudo"], dtype=float),
        genes,
        n_draws=PRODUCTION_RANDOMISATION,
        seed=PRODUCTION_SEED,
        namespace=f"r5a:{args.stage}:clean-L3H1-restore",
    )
    l3_restore_path["classification"] = (
        "L3_PATH_CONFIRMED"
        if l3_restore_path["classification"] == "ENTRY_BEHAVIOR_CONFIRMED"
        else "L3_PATH_NOT_CONFIRMED"
    )
    l3_restore_path["intervention"] = (
        "same-example clean L3H1 acceptor-query context into matched corrupt recipient"
    )
    if args.stage == "preflight":
        calibration = classifier_calibration(
            n_genes=PRODUCTION_N,
            n_draws=PRODUCTION_RANDOMISATION,
            n_trials=PRODUCTION_CALIBRATION_TRIALS,
            seed=PRODUCTION_SEED,
        )
        calibration_origin = "RUN_ON_BUCKET0_WITH_SYNTHETIC_PRIMITIVES"
    else:
        assert bound_preflight is not None
        calibration = bound_preflight["record"]["classifier_calibration"]
        calibration_origin = "INHERITED_FROM_LITERAL_BUCKET0_PREFLIGHT"

    input_snapshot = verify_input_snapshots_after_run(
        args,
        authorization=authorization,
        git_before=git_identity,
        source_before=source_identity,
        checkpoint_before=checkpoint,
        r3c_before=r3c_entry,
        preflight_before=bound_preflight,
    )

    capability = {
        "full_mlm_replay": full_mlm,
        "attention_output_projection": output_projection,
        "full_head_noop_replay": full_head_noop,
        "position_shift": {
            "all_batches_ok": all(item["ok"] for item in position_invariants),
            "attention_reconstruction_max_abs": _max_invariant(
                position_invariants, "attention_reconstruction_max_abs"
            ),
            "full_cell_margin_identity_max_abs": _max_invariant(
                position_invariants, "full_cell_margin_identity_max_abs"
            ),
            "component_swap_involution_max_abs": _max_invariant(
                position_invariants, "component_swap_involution_max_abs"
            ),
            "component_swap_assignment_max_abs": _max_invariant(
                position_invariants, "component_swap_assignment_max_abs"
            ),
            "component_swap_outside_max_abs": _max_invariant(
                position_invariants, "component_swap_outside_max_abs"
            ),
            "bridge_reuses_one_clean_V_tensor_for_all_cells": all(
                item["bridge_reuses_one_clean_V_tensor_for_all_cells"]
                for item in position_invariants
            ),
            "input_id_digest_by_batch": [
                item["input_ids_sha256"] for item in position_invariants
            ],
            "fixed_V_digest_by_batch": [
                item["fixed_V_sha256_by_head"] for item in position_invariants
            ],
            "fixed_V_digest_by_head_cell_and_batch": [
                item["bridge_fixed_V_sha256_by_head_and_cell"]
                for item in position_invariants
            ],
        },
        "qk_surgery": {
            "all_batches_ok": all(item["ok"] for item in qk_invariants),
            "selected_logit_assignment_max_abs_error": _max_invariant(
                qk_invariants, "selected_logit_assignment_max_abs_error"
            ),
            "unselected_logit_max_abs_error": _max_invariant(
                qk_invariants, "unselected_logit_max_abs_error"
            ),
            "fixed_value_max_abs_error": _max_invariant(
                qk_invariants, "fixed_value_max_abs_error"
            ),
            "fixed_clean_value_sha256_by_arm_and_batch": [
                item["fixed_clean_value_sha256_by_arm"] for item in qk_invariants
            ],
            "same_clean_value_tensor_used_for_every_leg": all(
                item["same_clean_value_tensor_used_for_every_leg"]
                for item in qk_invariants
            ),
        },
        "attention_value": {
            "all_batches_ok": all(item["ok"] for item in av_invariants),
            "a_at_v_reconstruction_max_abs_error": _max_invariant(
                av_invariants, "a_at_v_reconstruction_max_abs_error"
            ),
            "baseline_inject_max_abs_error": _max_invariant(
                av_invariants, "baseline_inject_max_abs_error"
            ),
            "value_window_assignment_max_abs_error": _max_invariant(
                av_invariants, "value_window_assignment_max_abs_error"
            ),
            "value_outside_window_max_abs_error": _max_invariant(
                av_invariants, "value_outside_window_max_abs_error"
            ),
            "address_source_scope": (
                "complete L3H1 acceptor-query attention row"
            ),
            "value_source_scope": (
                "selected four-token window; recipient values elsewhere"
            ),
            "clean_L3H1_restore_is_real_query_head_path": all(
                item["clean_L3H1_restore_is_real_query_head_path"]
                for item in av_invariants
            ),
        },
        "l2_role": {
            "all_batches_ok": all(item["ok"] for item in l2_invariants),
            "baseline_rescue_max_abs_error": _max_invariant(
                l2_invariants, "baseline_rescue_max_abs_error"
            ),
        },
        "bounded_intervention_health": health,
        "candidate_outcomes_used_to_define_capability": False,
        "behavioral_endpoint": "masked_acceptor_G_margin_only",
        "donor_readout_constructed": False,
    }
    capability["all_pass"] = bool(
        full_mlm and full_mlm["ok"]
        and output_projection and output_projection["ok"]
        and full_head_noop and full_head_noop["ok"]
        and capability["position_shift"]["all_batches_ok"]
        and capability["qk_surgery"]["all_batches_ok"]
        and capability["attention_value"]["all_batches_ok"]
        and capability["l2_role"]["all_batches_ok"]
        and health["all_pass"]
    )
    instrument_gates = {
        "exact_source_and_census_identity": True,
        "exact_128_one_row_per_gene_both_strands": bool(
            len(rows) == 128
            and len(set(genes)) == 128
            and {row.example.strand for row in rows} == {"+", "-"}
        ),
        "full_hook_and_decoder_capability": bool(capability["all_pass"]),
        "classifier_calibration": bool(calibration["all_pass"]),
        "bounded_intervention_health": bool(health["all_pass"]),
        "inputs_rehashed_after_run": bool(
            input_snapshot["checked_after_model_and_inference"]
        ),
        "r3c_entry_artifact_bound": True,
        "immutable_method_and_cohort_pins": True,
        "chr8_chr9_closed": True,
    }
    all_instrument = all(instrument_gates.values())
    if not all_instrument:
        status = "INSTRUMENT_UNRESOLVED"
    elif entry["classification"] != "ENTRY_BEHAVIOR_CONFIRMED":
        status = "ENTRY_BEHAVIOR_NOT_REPLICATED"
    elif l3_restore_path["classification"] != "L3_PATH_CONFIRMED":
        status = "L3_PATH_NOT_REPLICATED"
    elif args.stage == "preflight":
        status = "PREFLIGHT_PASS"
    else:
        status = inference["decisions"]["reader"]
    selected_identity = cohort_audit["selected_by_bucket"][str(bucket)]
    entry_contract = {
        "schema": "gfmcd.r5.entry_contract/1",
        "instrument_gates_pass": bool(all_instrument),
        "claim_eligible": bool(
            args.stage == "target"
            and status == "L3H1_DONOR_LOCAL_READER_SUPPORTED"
        ),
        "verdict": status,
        "acceptor_G_only": True,
        "masked_donor_endpoint_constructed": False,
        "checkpoint_sha256": checkpoint["aggregate_sha256"],
        "cohort_sha256": selected_identity["ordered_rows_sha256"],
        "construction_sha256": selected_identity["construction_sha256"],
        "r5_stack_fingerprint_sha256": authorization[
            "r5_stack_fingerprint_sha256"
        ],
        "upstream_r3c_artifact_sha256": r3c_entry["sha256"],
        "upstream_r3c_manifest_sha256": r3c_entry["manifest"]["sha256"],
    }
    exclusion_registry_path = output.with_suffix(".exclusion-registry.json")
    exclusion_registry_record = r3c_entry.pop("exclusion_registry_record")
    exclusion_registry_bytes = (
        json.dumps(exclusion_registry_record, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    exclusion_registry_identity = {
        "path": str(exclusion_registry_path),
        "size_bytes": len(exclusion_registry_bytes),
        "sha256": hashlib.sha256(exclusion_registry_bytes).hexdigest(),
        "record_sha256": exclusion_registry_record["record_sha256"],
    }
    entry_contract["exclusion_registry_sha256"] = exclusion_registry_identity[
        "sha256"
    ]
    entry_contract["exclusion_registry_record_sha256"] = (
        exclusion_registry_identity["record_sha256"]
    )
    record = {
        "schema": "gfmcd.r5a.acceptor_address_binding/1",
        "experiment": "R5a_acceptor_address_binding",
        "stage": args.stage,
        "status": status,
        "r5_entry_contract": entry_contract,
        "behavioral_endpoint": "masked_acceptor_G_margin_only",
        "donor_readout_constructed": False,
        "instrument_fingerprint_sha256": authorization["fingerprint"],
        "instrument_binding": authorization["binding"],
        "r5_stack_binding": authorization["r5_stack_binding"],
        "r5_stack_fingerprint_sha256": authorization[
            "r5_stack_fingerprint_sha256"
        ],
        "git": git_identity,
        "runtime_identity": runtime,
        "checkpoint": checkpoint,
        "source_identity": source_identity,
        "input_snapshot_after_run": input_snapshot,
        "cohort_audit": cohort_audit,
        "bucket": bucket,
        "gene_ids": genes,
        "gene_ids_sha256": canonical_sha256(genes),
        "row_identity": [row.identity_record() for row in rows],
        "construction_sha256": cohort_audit["selected_by_bucket"][str(bucket)][
            "construction_sha256"
        ],
        "r3c_entry": r3c_entry,
        "r5_exclusion_registry": {
            "identity": exclusion_registry_identity,
            "record": exclusion_registry_record,
            "derived_from_authenticated_r3c": True,
        },
        "bound_preflight": (
            None
            if bound_preflight is None
            else {
                key: value for key, value in bound_preflight.items() if key != "record"
            }
        ),
        "preflight_context": preflight_context,
        "native_margin_cells": native,
        "entry_behavior": entry,
        "clean_L3H1_path_continuity": l3_restore_path,
        "capability": capability,
        "classifier_calibration": calibration,
        "classifier_calibration_origin": calibration_origin,
        "raw_primitive_measurements": _json_ready(measurements),
        "raw_family_values": {
            family: matrices[family].tolist() for family in FAMILY_ORDER
        },
        "gene_sufficient_statistics": {
            "schema": "gfmcd.r5a.gene_sufficient_statistics/1",
            "aggregation": (
                "one selected row per gene; raw_family_values are ordered [family-row,gene]"
            ),
            "ordered_gene_ids": genes,
            "one_row_per_gene": len(set(genes)) == len(genes),
            "family_matrix_sha256": {
                family: _array_sha256(matrices[family]) for family in FAMILY_ORDER
            },
            "family_shapes": {
                family: list(matrices[family].shape) for family in FAMILY_ORDER
            },
        },
        "family_row_names": family_row_names(),
        "inference": inference,
        "inference_sha256": canonical_sha256(inference),
        "instrument_gates": instrument_gates,
        "all_instrument_gates_pass": all_instrument,
        "sealed_chr8_chr9_opened": False,
    }
    return record, git_identity


def atomic_write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("model-free-audit", "model"), default="model-free-audit"
    )
    parser.add_argument("--stage", choices=("preflight", "target"), default="preflight")
    parser.add_argument("--n", type=int, default=PRODUCTION_N)
    parser.add_argument("--batch", type=int, default=PRODUCTION_BATCH)
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument("--draws", type=int, default=PRODUCTION_RANDOMISATION)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--r3c-artifact")
    parser.add_argument("--preflight-artifact")
    parser.add_argument("--out")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.n != PRODUCTION_N or args.seed != PRODUCTION_SEED or args.draws != PRODUCTION_RANDOMISATION:
        raise SystemExit("R5a production n/seed/randomisation parameters are fixed")
    if args.batch != PRODUCTION_BATCH:
        raise SystemExit(f"R5a production batch is fixed at {PRODUCTION_BATCH}")
    authorization = None
    # Model authorization is deliberately first: no user-supplied model, data, entry,
    # preflight, or output path is resolved, opened, or stat'ed before this check.
    if args.mode == "model":
        try:
            authorization = authorize_model_scoring(args.stage)
        except SealError as exc:
            label = "target" if args.stage == "target" else "model"
            raise SystemExit(f"R5a {label} authorization failed: {exc}") from exc

    output = Path(args.out).resolve() if args.out else None
    manifest_path = output.with_suffix(".manifest.json") if output is not None else None
    exclusion_registry_path = (
        output.with_suffix(".exclusion-registry.json")
        if output is not None and args.mode == "model"
        else None
    )
    if output is not None and (
        output.exists()
        or (manifest_path is not None and manifest_path.exists())
        or (
            exclusion_registry_path is not None
            and exclusion_registry_path.exists()
        )
    ):
        raise SystemExit("R5a outputs are immutable; choose a fresh path")

    if args.mode == "model-free-audit":
        record = model_free_audit(args)
        payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if output is None:
            print(payload.decode("utf-8"), end="")
        else:
            assert manifest_path is not None
            manifest = ExperimentManifest.create(
                "splicebert_acceptor_coupling:r5a_model_free_audit",
                list(sys.argv if argv is None else [str(RUNNER), *argv]),
                repo=REPO,
                status="MODEL_FREE_AUDIT",
                config={
                    "rows_per_bucket": PRODUCTION_N,
                    "cohort_schema": "gfmcd.r5.acceptor_mechanism_cohort/2",
                    "expected_census_sha256": canonical_sha256(
                        R5_EXPECTED_CENSUS_IDENTITY
                    ),
                    "model_loaded": False,
                    "sealed_chr8_chr9_opened": False,
                },
                seeds={"cohort": PRODUCTION_SEED},
                notes=[
                    "Annotation/sequence-only R5 census; no model outputs or chr8/chr9 accessed.",
                    "Pins remain nonbinding None until the immutable audit is reviewed.",
                ],
            )
            manifest.inputs = [
                {
                    "path": str(Path(args.gtf).resolve()),
                    **record["observed_source_identity"]["gtf"],
                },
                {
                    "path": str(Path(args.fasta).resolve()),
                    **record["observed_source_identity"]["fasta"],
                },
            ]
            manifest.outputs = [
                {
                    "path": str(output),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ]
            manifest_bytes = (
                json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            atomic_write_new(output, payload)
            atomic_write_new(manifest_path, manifest_bytes)
            print(f"R5a model-free audit written to {output}")
        return

    if output is None or manifest_path is None:
        raise SystemExit("R5a model mode requires a fresh --out path")
    assert authorization is not None
    record, git_identity = run_model_stage(
        args, authorization=authorization, output=output
    )
    result_bytes = (
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r5a_acceptor_address_binding",
        list(sys.argv if argv is None else [str(RUNNER), *argv]),
        repo=REPO,
        status=record["status"],
        config={
            "stage": args.stage,
            "n": PRODUCTION_N,
            "batch": args.batch,
            "seed": PRODUCTION_SEED,
            "n_randomisation": PRODUCTION_RANDOMISATION,
            "n_calibration_trials": PRODUCTION_CALIBRATION_TRIALS,
            "bucket": record["bucket"],
            "instrument_fingerprint_sha256": authorization["fingerprint"],
            "r5_stack_fingerprint_sha256": authorization[
                "r5_stack_fingerprint_sha256"
            ],
            "checkpoint_aggregate_sha256": record["checkpoint"][
                "aggregate_sha256"
            ],
            "construction_sha256": record["construction_sha256"],
            "family_row_order_sha256": {
                family: canonical_sha256(rows)
                for family, rows in family_row_names().items()
            },
            "path_registry_sha256": canonical_sha256(named_path_specs()),
        },
        seeds={"randomisation": PRODUCTION_SEED, "calibration": PRODUCTION_SEED},
        notes=[
            "Every behavioral outcome is the single masked-acceptor-G margin; no donor readout is constructed.",
            "R5a POSITION uses paired-to-adjacent word/position component swaps and a fixed-V margin bridge.",
            "QK uses exact four-logit surgery; A/V uses complete address rows and four-token value windows.",
            "Validation chromosomes chr11/chr16/chr19 only; chr8/chr9 remains closed.",
            "The R3c-derived exclusion registry is a separate authenticated output of this manifest.",
        ],
    )
    manifest.git_commit = git_identity["commit"]
    manifest.git_dirty = False
    manifest.inputs = [
        {
            "path": str(Path(args.gtf).resolve()),
            **record["source_identity"]["gtf"],
        },
        {
            "path": str(Path(args.fasta).resolve()),
            **record["source_identity"]["fasta"],
        },
        {
            "path": record["r3c_entry"]["path"],
            "sha256": record["r3c_entry"]["sha256"],
            "size_bytes": Path(record["r3c_entry"]["path"]).stat().st_size,
        },
        dict(record["r3c_entry"]["manifest"]),
    ]
    for filename, identity in record["checkpoint"]["files"].items():
        manifest.inputs.append(
            {
                "path": str(Path(record["checkpoint"]["root"]) / filename),
                **identity,
            }
        )
    if record["bound_preflight"] is not None:
        path = Path(record["bound_preflight"]["path"])
        manifest.inputs.append(
            {
                "path": str(path),
                "sha256": record["bound_preflight"]["sha256"],
                "size_bytes": path.stat().st_size,
            }
        )
        preflight_manifest = Path(record["bound_preflight"]["manifest_path"])
        manifest.inputs.append(
            {
                "path": str(preflight_manifest),
                "sha256": record["bound_preflight"]["manifest_sha256"],
                "size_bytes": preflight_manifest.stat().st_size,
            }
        )
    stack_binding = authorization.get("r5_stack_binding")
    closure = (
        stack_binding.get("local_python_closure", {}).get("files")
        if isinstance(stack_binding, Mapping)
        else None
    )
    protocols = (
        stack_binding.get("protocols")
        if isinstance(stack_binding, Mapping)
        else None
    )
    if not isinstance(closure, Mapping) or not isinstance(protocols, Mapping):
        raise RuntimeError("R5a joint stack lacks its exact source input registry")
    existing_inputs = {
        Path(str(item["path"])).expanduser().resolve()
        for item in manifest.inputs
        if isinstance(item, Mapping) and "path" in item
    }
    for relative in (*closure, *protocols):
        source_path = (REPO / str(relative)).resolve()
        try:
            source_path.relative_to(REPO.resolve())
        except ValueError as exc:
            raise RuntimeError("R5a joint stack source escapes the repository") from exc
        if not source_path.is_file():
            raise RuntimeError(f"R5a joint stack source is missing: {relative}")
        if source_path not in existing_inputs:
            manifest.inputs.append(
                {
                    "path": str(source_path),
                    "size_bytes": source_path.stat().st_size,
                    "sha256": sha256_file(source_path),
                }
            )
            existing_inputs.add(source_path)
    registry_identity = record["r5_exclusion_registry"]["identity"]
    registry_record = record["r5_exclusion_registry"]["record"]
    registry_bytes = (
        json.dumps(registry_record, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if (
        hashlib.sha256(registry_bytes).hexdigest() != registry_identity["sha256"]
        or len(registry_bytes) != registry_identity["size_bytes"]
    ):
        raise RuntimeError("R5a exclusion-registry serialization drifted")
    manifest.outputs = [
        {
            "path": str(output),
            "size_bytes": len(result_bytes),
            "sha256": hashlib.sha256(result_bytes).hexdigest(),
        },
        dict(registry_identity),
    ]
    manifest_bytes = (
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    atomic_write_new(output, result_bytes)
    atomic_write_new(Path(registry_identity["path"]), registry_bytes)
    atomic_write_new(manifest_path, manifest_bytes)
    print(f"R5a status={record['status']} -> {output}")


if __name__ == "__main__":
    main()
