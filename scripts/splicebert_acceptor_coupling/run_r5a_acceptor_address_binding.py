#!/usr/bin/env python
"""R5a: acceptor-only paired-donor reader, address, and value experiment.

The only behavioral readout constructed by this program is the native masked-acceptor
``G`` margin.  QK logits and attention probabilities are internal mechanism readouts;
they are never substituted for the behavioral endpoint.

The checked-in protocol is frozen for preflight only. ``--mode model-free-audit`` binds the
prospective instrument without opening genome or model data.  A target request is
rejected before resolving the model, GTF, FASTA, output, or stage-preflight paths unless
the exact R1c handoffs and all R5a freeze pins are present.
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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.circuits.hf_bert import (  # noqa: E402
    _merge_context,
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
    FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256 as SHARED_R1C_PREFLIGHT_ARTIFACT_SHA256,
    FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256 as SHARED_R1C_PREFLIGHT_MANIFEST_SHA256,
    FROZEN_R1C_TARGET_ARTIFACT_SHA256 as SHARED_R1C_TARGET_ARTIFACT_SHA256,
    FROZEN_R1C_TARGET_ATTEMPT_SHA256 as SHARED_R1C_TARGET_ATTEMPT_SHA256,
    FROZEN_R1C_TARGET_COMPLETION_SHA256 as SHARED_R1C_TARGET_COMPLETION_SHA256,
    FROZEN_R1C_TARGET_MANIFEST_SHA256 as SHARED_R1C_TARGET_MANIFEST_SHA256,
    R5CapstoneCohortError,
    R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
    R5_DONOR_PSEUDO_CALIPER,
    R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
    R5_VALIDATION_CHROMOSOMES,
    load_r1c_capstone_cohort_handoff,
    load_r1c_preflight_cohort_handoff,
    load_r1c_verified_target_common_legs,
    materialize_r5_capstone_rows,
    materialize_r5_preflight_rows,
    validate_r5_capstone_cohort_handoff,
    validate_r5_preflight_cohort_handoff,
    validate_expected_r5_source_identity,
)
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    load_adapter,
    load_examples_unscored,
    pick_device,
)
from gfmcd.splicing.paired_component import (  # noqa: E402
    boundary_sign_flip_tests,
    intersection_union_pvalue,
    simultaneous_family_decision,
    strict_interval_decision,
)
from gfmcd.splicing.provenance import (  # noqa: E402
    capture_embed_components,
    clamp_embed_component,
)
from gfmcd.splicing.r5_stack_binding import (  # noqa: E402
    R5StackBindingError,
    SHARED_R1C_PACKAGE_PIN_NAMES,
    build_r5_stack_binding,
    normalized_literal_pin_source_sha256,
    r5_stack_fingerprint,
    stable_runtime_identity,
    validate_r5_runtime_policy,
    validate_serialized_r5_stack,
)
from gfmcd.splicing.r5_target_ledger import (  # noqa: E402
    R5TargetSpentError,
    build_r5_biological_target_identity,
    complete_r5_target_attempt,
    reserve_r5_target_attempt,
)
from gfmcd.splicing.r5_upstream_provenance import (  # noqa: E402
    R3C_PREFLIGHT_ARTIFACT_SHA256,
    R3C_PREFLIGHT_MANIFEST_SHA256,
    R3D_ENTRY_SCHEMA,
    R3D_REGISTRY_ARTIFACT_SHA256,
    R3D_REGISTRY_MANIFEST_SHA256,
    R3D_TARGET_ARTIFACT_SHA256,
    R3D_TARGET_ATTEMPT_SHA256,
    R3D_TARGET_COMPLETION_SHA256,
    R3D_TARGET_MANIFEST_SHA256,
    R5_ENTRY_SCHEMA,
    load_r3c_exclusion_package,
    load_r3d_entry_package,
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
MECHANISM_CORE = REPO / "src/gfmcd/splicing/acceptor_coupling_mechanism.py"
COHORT_CORE = REPO / "src/gfmcd/splicing/acceptor_mechanism_cohort.py"
STAT_CORE = REPO / "src/gfmcd/splicing/paired_component.py"
HOOK_CORE = REPO / "src/gfmcd/circuits/hf_bert.py"
EMBED_CORE = REPO / "src/gfmcd/splicing/provenance.py"
ASSAY_CORE = REPO / "src/gfmcd/splicing/reverse_coupling.py"
CIRCUIT_CORE = REPO / "src/gfmcd/splicing/circuit.py"
R1C_RUNNER = (
    REPO
    / "scripts/splicebert_acceptor_coupling/run_r1c_sealed_reverse_behavior.py"
)

# Reviewed method, checkpoint, and historical-R3 pins are binding at source freeze.
# R1c package pins are owned by the shared cohort contract and remain unset until their
# producing runs; R5a's own output-derived preflight pins likewise remain unset.
FROZEN_PROTOCOL_SHA256: str | None = "2981e40a65872914b5551d7b463d85b5203ef7630d66a1e9a0b24ae82c348ec6"
FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = "59077228c551f38cda5e136a94e5522f79c4c3984947c156eeae78a0ab61604b"
FROZEN_PREFLIGHT_ARTIFACT_SHA256: str | None = None
FROZEN_PREFLIGHT_MANIFEST_SHA256: str | None = None
FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256 = SHARED_R1C_PREFLIGHT_ARTIFACT_SHA256
FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256 = SHARED_R1C_PREFLIGHT_MANIFEST_SHA256
FROZEN_R1C_TARGET_ARTIFACT_SHA256 = SHARED_R1C_TARGET_ARTIFACT_SHA256
FROZEN_R1C_TARGET_MANIFEST_SHA256 = SHARED_R1C_TARGET_MANIFEST_SHA256
FROZEN_R1C_TARGET_ATTEMPT_SHA256 = SHARED_R1C_TARGET_ATTEMPT_SHA256
FROZEN_R1C_TARGET_COMPLETION_SHA256 = SHARED_R1C_TARGET_COMPLETION_SHA256
FROZEN_R3D_ARTIFACT_SHA256: str = R3D_TARGET_ARTIFACT_SHA256
FROZEN_R3D_MANIFEST_SHA256: str = R3D_TARGET_MANIFEST_SHA256
FROZEN_R3D_REGISTRY_ARTIFACT_SHA256: str = R3D_REGISTRY_ARTIFACT_SHA256
FROZEN_R3D_REGISTRY_MANIFEST_SHA256: str = R3D_REGISTRY_MANIFEST_SHA256
FROZEN_R3D_ATTEMPT_SHA256: str = R3D_TARGET_ATTEMPT_SHA256
FROZEN_R3D_COMPLETION_SHA256: str = R3D_TARGET_COMPLETION_SHA256
FROZEN_R3C_ARTIFACT_SHA256: str = R3C_PREFLIGHT_ARTIFACT_SHA256
FROZEN_R3C_MANIFEST_SHA256: str = R3C_PREFLIGHT_MANIFEST_SHA256
FROZEN_CHECKPOINT_AGGREGATE_SHA256: str | None = "fb2a04e0c9b5247b44f03cd23581632d5924b86fcafaffa4c6f2b8d01ae1da8d"

EXACT_ROWS = 400
PRODUCTION_N = EXACT_ROWS
PRODUCTION_BATCH = 8
PRODUCTION_SEED = 0
PRODUCTION_RANDOMISATION = 10000
PRODUCTION_CALIBRATION_TRIALS = 465
MIN_GENES = 30
MARGIN_SESOI = 0.05
STAGE_ALPHA = 0.05
DECOY_TOL = R5_DONOR_PSEUDO_CALIPER

READER = (3, 1)
SECONDARY_READER = (2, 15)
NEGATIVE_HEAD = (0, 2)
SHIFT_HEADS = ("L3H1", "L0H2")
SHIFT_COMPARISONS = ("paired_pseudo",)
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
    ("L0H2", "paired"),
)
L2_QK_ARMS = (
    ("L2H15", "paired"),
    ("L2H15", "pseudo"),
    ("L0H2", "paired"),
)
WINDOWS = ("paired", "pseudo")
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
FAMILY_ORDER = (
    "position_margin",
    "qk_surgery",
    "attention_value",
    "l2h15_qk_secondary",
)
CORE_PATH_ORDER = (
    "G1_POSITION_CAUSAL_MARGIN",
    "G2_EXACT_QK_CAUSAL_MARGIN",
    "G3_FIXED_V_AXV_CAUSAL_MARGIN",
)
SECONDARY_PATH_ORDER = ("S1_L2H15_EXACT_QK_SECONDARY",)
ALL_PATH_ORDER = CORE_PATH_ORDER + SECONDARY_PATH_ORDER
# Compatibility name for the headline fixed sequence.  Secondary failure never
# invalidates the L3H1 core decision.
PATH_ORDER = CORE_PATH_ORDER

HOOK_ATOL = 1e-5
MARGIN_ATOL = 1e-4
SDPA_EAGER_COMMON_LEG_ATOL = 1e-4
CALIBRATION_EPSILON = 0.01
CALIBRATION_MAX_FALSE_POSITIVES = 5
CALIBRATION_MIN_POWER_SUCCESSES = 441
CALIBRATION_PRIMITIVE_NOISE_SD = 0.025
HEALTH_MAX_EXAMPLES = 16
HEALTH_PROBES_PER_EXAMPLE = 4
HEALTH_NONINFERIORITY_MARGIN = 0.15


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


def position_margin_contrast_name(control: str, quantity: str) -> str:
    return (
        "POSITION_MARGIN_CONTRAST|target=L3H1:paired_pseudo|"
        f"control={control}|measure=acceptor_margin|quantity={quantity}"
    )


def qk_margin_contrast_name(
    control: str, endpoint: str, *, target: str = "L3H1"
) -> str:
    return (
        f"QK_MARGIN_CONTRAST|target={target}:paired|"
        f"control={control}|endpoint={endpoint}"
    )


def av_margin_contrast_name(control: str, quantity: str, endpoint: str) -> str:
    return (
        "AV_MARGIN_CONTRAST|target=paired|"
        f"control={control}|quantity={quantity}|endpoint={endpoint}"
    )


def family_row_names() -> dict[str, list[str]]:
    return {
        "position_margin": [
            position_row_name(head, comparison, "acceptor_margin", quantity)
            for head in SHIFT_HEADS
            for comparison in SHIFT_COMPARISONS
            for quantity in SHIFT_QUANTITIES
        ]
        + [
            position_margin_contrast_name(control, quantity)
            for control in ("L0H2:paired_pseudo",)
            for quantity in SHIFT_QUANTITIES[:2]
        ],
        "qk_surgery": [
            qk_row_name(head, window, endpoint)
            for head, window in QK_ARMS
            for endpoint in ENDPOINTS
        ]
        + [
            qk_margin_contrast_name(control, endpoint)
            for control in ("L3H1:pseudo", "L0H2:paired")
            for endpoint in ENDPOINTS
        ],
        "attention_value": [
            av_row_name(window, quantity, endpoint)
            for window in WINDOWS
            for quantity in AV_QUANTITIES
            for endpoint in ENDPOINTS
        ]
        + [
            av_margin_contrast_name(control, quantity, endpoint)
            for control in ("pseudo",)
            for quantity in (
                "joint_lift",
                "address_effect_at_value_plus",
                "value_effect_at_address_plus",
            )
            for endpoint in ENDPOINTS
        ],
        "l2h15_qk_secondary": [
            qk_row_name(head, window, endpoint)
            for head, window in L2_QK_ARMS
            for endpoint in ENDPOINTS
        ]
        + [
            qk_margin_contrast_name(control, endpoint, target="L2H15")
            for control in ("L2H15:pseudo", "L0H2:paired")
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
    for quantity in SHIFT_QUANTITIES[:2]:
        _requirement(
            position,
            position_row_name(
                "L3H1", "paired_pseudo", "acceptor_margin", quantity
            ),
            "MATERIAL_POSITIVE",
        )
    for head, comparison in (("L0H2", "paired_pseudo"),):
        for quantity in SHIFT_QUANTITIES[:2]:
            _requirement(
                position,
                position_row_name(
                    head, comparison, "acceptor_margin", quantity
                ),
                "EQUIVALENT_WITHIN_SESOI",
            )
    for control in ("L0H2:paired_pseudo",):
        for quantity in SHIFT_QUANTITIES[:2]:
            _requirement(
                position,
                position_margin_contrast_name(control, quantity),
                "MATERIAL_POSITIVE",
            )

    qk: dict[str, str] = {}
    for endpoint in ("partner", "specific"):
        _requirement(qk, qk_row_name("L3H1", "paired", endpoint), "MATERIAL_POSITIVE")
    _requirement(qk, qk_row_name("L3H1", "paired", "pseudo"), "EQUIVALENT_WITHIN_SESOI")
    for head, window in QK_ARMS[1:]:
        for endpoint in ENDPOINTS:
            _requirement(qk, qk_row_name(head, window, endpoint), "EQUIVALENT_WITHIN_SESOI")
    for control in ("L3H1:pseudo", "L0H2:paired"):
        for endpoint in ("partner", "specific"):
            _requirement(
                qk,
                qk_margin_contrast_name(control, endpoint),
                "MATERIAL_POSITIVE",
            )

    av: dict[str, str] = {}
    for quantity in (
        "joint_lift",
        "address_effect_at_value_plus",
        "value_effect_at_address_plus",
    ):
        for endpoint in ("partner", "specific"):
            _requirement(av, av_row_name("paired", quantity, endpoint), "MATERIAL_POSITIVE")
        _requirement(av, av_row_name("paired", quantity, "pseudo"), "EQUIVALENT_WITHIN_SESOI")
        for window in ("pseudo",):
            for endpoint in ENDPOINTS:
                _requirement(av, av_row_name(window, quantity, endpoint), "EQUIVALENT_WITHIN_SESOI")
        for control in ("pseudo",):
            for endpoint in ("partner", "specific"):
                _requirement(
                    av,
                    av_margin_contrast_name(control, quantity, endpoint),
                    "MATERIAL_POSITIVE",
                )

    l2: dict[str, str] = {}
    for endpoint in ("partner", "specific"):
        _requirement(
            l2,
            qk_row_name("L2H15", "paired", endpoint),
            "MATERIAL_POSITIVE",
        )
    _requirement(
        l2,
        qk_row_name("L2H15", "paired", "pseudo"),
        "EQUIVALENT_WITHIN_SESOI",
    )
    for head, window in L2_QK_ARMS[1:]:
        for endpoint in ENDPOINTS:
            _requirement(
                l2,
                qk_row_name(head, window, endpoint),
                "EQUIVALENT_WITHIN_SESOI",
            )
    for control in ("L2H15:pseudo", "L0H2:paired"):
        for endpoint in ("partner", "specific"):
            _requirement(
                l2,
                qk_margin_contrast_name(control, endpoint, target="L2H15"),
                "MATERIAL_POSITIVE",
            )

    output = {
        "G1_POSITION_CAUSAL_MARGIN": {
            "family": "position_margin",
            "requirements": position,
        },
        "G2_EXACT_QK_CAUSAL_MARGIN": {
            "family": "qk_surgery",
            "requirements": qk,
        },
        "G3_FIXED_V_AXV_CAUSAL_MARGIN": {
            "family": "attention_value",
            "requirements": av,
        },
        "S1_L2H15_EXACT_QK_SECONDARY": {
            "family": "l2h15_qk_secondary",
            "requirements": l2,
            "role": "secondary_upstream_reader_role",
            "gates_l3_headline": False,
            "historical_r3c_health_used": False,
        },
    }
    if tuple(output) != ALL_PATH_ORDER:
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


def build_family_matrices(
    measurements: Mapping[str, Any], *, families: Sequence[str] | None = None
) -> dict[str, np.ndarray]:
    """Derive all four immutable families from raw intervention cells.

    ``specific`` and every factorial contrast are rebuilt here.  Callers cannot supply or
    overwrite a derived row, which makes stored primitive cells sufficient for exact replay.
    """

    selected = FAMILY_ORDER if families is None else tuple(families)
    if not selected or any(family not in FAMILY_ORDER for family in selected):
        raise ValueError("families must be a nonempty ordered subset of FAMILY_ORDER")
    if tuple(family for family in FAMILY_ORDER if family in selected) != selected:
        raise ValueError("families must preserve the fixed R5a order")
    row_names = family_row_names()
    output: dict[str, np.ndarray] = {}

    if "position_margin" in selected:
        position_by_name: dict[str, np.ndarray] = {}
        for head in SHIFT_HEADS:
            for comparison in SHIFT_COMPARISONS:
                cells = measurements["position_shift"][head][comparison][
                    "acceptor_margin"
                ]
                for quantity, values in shift_factorial_contrasts(cells).items():
                    position_by_name[
                        position_row_name(
                            head, comparison, "acceptor_margin", quantity
                        )
                    ] = values
        for quantity in SHIFT_QUANTITIES[:2]:
            target = position_by_name[
                position_row_name(
                    "L3H1", "paired_pseudo", "acceptor_margin", quantity
                )
            ]
            position_by_name[
                position_margin_contrast_name("L0H2:paired_pseudo", quantity)
            ] = target - position_by_name[
                position_row_name(
                    "L0H2", "paired_pseudo", "acceptor_margin", quantity
                )
            ]
        output["position_margin"] = np.vstack(
            [position_by_name[name] for name in row_names["position_margin"]]
        )

    if "qk_surgery" in selected:
        qk_by_name: dict[str, np.ndarray] = {}
        for head, window in QK_ARMS:
            endpoints = endpoint_vectors(
                measurements["qk_surgery"][head][window]["partner"],
                measurements["qk_surgery"][head][window]["pseudo"],
            )
            for endpoint, values in endpoints.items():
                qk_by_name[qk_row_name(head, window, endpoint)] = values
        for control in ("L3H1:pseudo", "L0H2:paired"):
            head, window = control.split(":")
            for endpoint in ENDPOINTS:
                qk_by_name[qk_margin_contrast_name(control, endpoint)] = (
                    qk_by_name[qk_row_name("L3H1", "paired", endpoint)]
                    - qk_by_name[qk_row_name(head, window, endpoint)]
                )
        output["qk_surgery"] = np.vstack(
            [qk_by_name[name] for name in row_names["qk_surgery"]]
        )

    if "attention_value" in selected:
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
        for control in ("pseudo",):
            for quantity in (
                "joint_lift",
                "address_effect_at_value_plus",
                "value_effect_at_address_plus",
            ):
                for endpoint in ENDPOINTS:
                    av_by_name[
                        av_margin_contrast_name(control, quantity, endpoint)
                    ] = (
                        av_by_name[av_row_name("paired", quantity, endpoint)]
                        - av_by_name[av_row_name(control, quantity, endpoint)]
                    )
        output["attention_value"] = np.vstack(
            [av_by_name[name] for name in row_names["attention_value"]]
        )

    if "l2h15_qk_secondary" in selected:
        l2_by_name: dict[str, np.ndarray] = {}
        for head, window in L2_QK_ARMS:
            endpoints = endpoint_vectors(
                measurements["qk_surgery"][head][window]["partner"],
                measurements["qk_surgery"][head][window]["pseudo"],
            )
            for endpoint, values in endpoints.items():
                l2_by_name[qk_row_name(head, window, endpoint)] = values
        for control in ("L2H15:pseudo", "L0H2:paired"):
            head, window = control.split(":")
            for endpoint in ENDPOINTS:
                l2_by_name[
                    qk_margin_contrast_name(control, endpoint, target="L2H15")
                ] = (
                    l2_by_name[qk_row_name("L2H15", "paired", endpoint)]
                    - l2_by_name[qk_row_name(head, window, endpoint)]
                )
        output["l2h15_qk_secondary"] = np.vstack(
            [l2_by_name[name] for name in row_names["l2h15_qk_secondary"]]
        )

    sizes = {
        "position_margin": 12,
        "qk_surgery": 15,
        "attention_value": 69,
        "l2h15_qk_secondary": 15,
    }
    for family, expected in sizes.items():
        if family not in selected:
            continue
        if output[family].shape[0] != expected:
            raise RuntimeError(f"{family} family row count drifted")
        if not np.isfinite(output[family]).all():
            raise RuntimeError(f"{family} family contains non-finite values")
    return output


def position_bridge_diagnostics(measurements: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize QK-logit and attention-probability bridges without a margin SESOI.

    These are internal readouts in incompatible units.  They are signed candidate bridges,
    never scientific gates, equivalence tests, or substitutes for the causal margin rows.
    """

    units = {
        "qk_preference": "pre_softmax_logit_difference",
        "attention_preference": "post_softmax_probability_difference",
    }
    rows: dict[str, Any] = {}
    for head in SHIFT_HEADS:
        for comparison in SHIFT_COMPARISONS:
            for measure, unit in units.items():
                contrasts = shift_factorial_contrasts(
                    measurements["position_shift"][head][comparison][measure]
                )
                for quantity, vector in contrasts.items():
                    values = np.asarray(vector, dtype=float)
                    name = position_row_name(head, comparison, measure, quantity)
                    rows[name] = {
                        "unit": unit,
                        "role": "SIGNED_DIAGNOSTIC_ONLY",
                        "mean": float(np.mean(values)),
                        "trimmed_mean_10pct": float(_trimmed_mean(values, 0.10)),
                        "positive_fraction": float(np.mean(values > 0.0)),
                        "zero_null_or_margin_sesoi_applied": False,
                    }
    return {
        "schema": "gfmcd.r5a.position_bridge_diagnostics/2",
        "decision_bearing": False,
        "query_policy": "same numerical clean query vector in every 2x2 cell",
        "shifted_native_query_included_in_decisions": False,
        "scale_specific_thresholds": None,
        "reason": (
            "QK logits and attention probabilities have different units and no "
            "model-free SESOI; only causal masked-acceptor margin interventions decide."
        ),
        "rows": rows,
    }


def _trimmed_mean(values: Any, proportion: float = 0.10) -> np.ndarray | float:
    array = np.asarray(values, dtype=float)
    n = array.shape[-1]
    cut = int(np.floor(proportion * n))
    if n - 2 * cut < 2:
        raise ValueError("too few observations for the prespecified trimmed mean")
    result = np.sort(array, axis=-1)[..., cut : n - cut].mean(axis=-1)
    return float(result) if result.ndim == 0 else result


def _location_and_se(values: Any, estimator: str) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    n = array.shape[-1]
    if n < 3 or not np.isfinite(array).all():
        raise ValueError("studentized inference needs at least three finite genes")
    if estimator == "mean":
        location = array.mean(axis=-1)
        se = array.std(axis=-1, ddof=1) / np.sqrt(n)
    elif estimator == "trimmed_mean_10pct":
        proportion = 0.10
        cut = int(np.floor(proportion * n))
        ordered = np.sort(array, axis=-1)
        location = ordered[..., cut : n - cut].mean(axis=-1)
        winsor = ordered.copy()
        winsor[..., :cut] = ordered[..., cut : cut + 1]
        winsor[..., n - cut :] = ordered[..., n - cut - 1 : n - cut]
        se = winsor.std(axis=-1, ddof=1) / ((1.0 - 2.0 * proportion) * np.sqrt(n))
    else:
        raise ValueError(f"unknown estimator: {estimator}")
    return np.asarray(location, dtype=float), np.asarray(se, dtype=float)


def _gene_aggregate_matrix(
    matrix: np.ndarray, groups: Sequence[str], *, min_genes: int
) -> tuple[np.ndarray, list[str]]:
    labels = sorted(set(map(str, groups)))
    if len(labels) < min_genes:
        raise ValueError(f"need at least {min_genes} independent genes")
    group_array = np.asarray(list(map(str, groups)), dtype=object)
    aggregated = np.stack(
        [matrix[:, group_array == label].mean(axis=1) for label in labels], axis=1
    )
    return aggregated, labels


def dual_estimator_studentized_bootstrap_inversion(
    values: np.ndarray,
    groups: Sequence[str],
    names: Sequence[str],
    requirements: Mapping[str, str],
    *,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int,
    alpha: float = STAGE_ALPHA,
    sesoi: float = MARGIN_SESOI,
) -> dict[str, Any]:
    """Joint bootstrap-t inversion for the mean and 10%-trimmed mean.

    The add-one values quantify finite-draw Monte Carlo error for a nonparametric
    cluster bootstrap.  They are deliberately labelled approximate, never exact.
    """

    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(names), len(groups)) or not np.isfinite(matrix).all():
        raise ValueError("bootstrap inversion requires a finite [row, observation] matrix")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    index = {name: i for i, name in enumerate(names)}
    if not requirements or not set(requirements).issubset(index):
        raise ValueError("requirements must be a nonempty subset of the row registry")
    required_names = list(requirements)
    gene_matrix, gene_labels = _gene_aggregate_matrix(
        matrix[[index[name] for name in required_names]], groups, min_genes=min_genes
    )
    rng_seed = int.from_bytes(
        hashlib.sha256(f"{namespace}|{seed}".encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(rng_seed)
    draw_indices = rng.integers(0, gene_matrix.shape[1], size=(n_draws, gene_matrix.shape[1]))
    estimators = ("mean", "trimmed_mean_10pct")
    observed = np.zeros((2, len(required_names)), dtype=float)
    observed_se = np.zeros_like(observed)
    t_star = np.zeros((2, len(required_names), n_draws), dtype=float)
    for row_index, vector in enumerate(gene_matrix):
        sampled = vector[draw_indices]
        for estimator_index, estimator in enumerate(estimators):
            theta, se = _location_and_se(vector, estimator)
            boot_theta, boot_se = _location_and_se(sampled, estimator)
            observed[estimator_index, row_index] = float(theta)
            observed_se[estimator_index, row_index] = float(se)
            numerator = boot_theta - float(theta)
            t_star[estimator_index, row_index] = np.divide(
                numerator,
                boot_se,
                out=np.where(numerator >= 0.0, np.inf, -np.inf),
                where=boot_se > 0.0,
            )
            t_star[estimator_index, row_index][
                (boot_se == 0.0) & (numerator == 0.0)
            ] = 0.0
    max_abs = np.max(np.abs(t_star), axis=(0, 1))
    critical = float(np.quantile(max_abs, 1.0 - alpha, method="higher"))

    edge_specs: list[tuple[int, int, str, float, str, str]] = []
    for row_index, row in enumerate(required_names):
        requirement = requirements[row]
        for estimator_index, estimator in enumerate(estimators):
            if requirement == "MATERIAL_POSITIVE":
                edge_specs.append((estimator_index, row_index, row, sesoi, "greater", "material"))
            elif requirement == "EQUIVALENT_WITHIN_SESOI":
                edge_specs.extend(
                    (
                        (estimator_index, row_index, row, -sesoi, "greater", "lower"),
                        (estimator_index, row_index, row, sesoi, "less", "upper"),
                    )
                )
            else:
                raise ValueError(f"invalid requirement: {requirement}")
    signed_null = np.stack(
        [
            t_star[e, r] if alternative == "greater" else -t_star[e, r]
            for e, r, _row, _boundary, alternative, _edge in edge_specs
        ]
    )
    max_signed_null = np.max(signed_null, axis=0)
    edge_records: dict[str, Any] = {}
    p_by_row_estimator: dict[tuple[str, str], list[float]] = {}
    for estimator_index, row_index, row, boundary, alternative, edge in edge_specs:
        se = observed_se[estimator_index, row_index]
        delta = observed[estimator_index, row_index] - boundary
        t_observed = delta / se if se > 0.0 else (np.inf if delta > 0 else -np.inf)
        signed_observed = t_observed if alternative == "greater" else -t_observed
        p_value = float((1 + np.count_nonzero(max_signed_null >= signed_observed)) / (n_draws + 1))
        estimator = estimators[estimator_index]
        edge_name = f"{row}|estimator={estimator}|edge={edge}"
        edge_records[edge_name] = {
            "boundary": boundary,
            "alternative": alternative,
            "studentized_statistic": float(signed_observed),
            "maxT_add_one_monte_carlo_p_value": p_value,
        }
        p_by_row_estimator.setdefault((row, estimator), []).append(p_value)

    requirement_records: dict[str, Any] = {}
    all_ci_pass = True
    all_p_pass = True
    for row_index, row in enumerate(required_names):
        per_estimator: dict[str, Any] = {}
        for estimator_index, estimator in enumerate(estimators):
            estimate = observed[estimator_index, row_index]
            se = observed_se[estimator_index, row_index]
            interval = (float(estimate - critical * se), float(estimate + critical * se))
            classification = strict_interval_decision(*interval, sesoi)
            required = requirements[row]
            p_value = max(p_by_row_estimator[(row, estimator)])
            passed = classification == required and p_value < alpha
            per_estimator[estimator] = {
                "estimate": float(estimate),
                "standard_error": float(se),
                "simultaneous_bootstrap_t_ci": list(interval),
                "classification": classification,
                "max_edge_p_value": p_value,
                "passed": bool(passed),
            }
            all_ci_pass &= classification == required
            all_p_pass &= p_value < alpha
        requirement_records[row] = {
            "required": requirements[row],
            "estimators": per_estimator,
            "both_estimators_pass": all(item["passed"] for item in per_estimator.values()),
        }
    path_p = max(record["maxT_add_one_monte_carlo_p_value"] for record in edge_records.values())
    return {
        "schema": "gfmcd.r5a.dual_estimator_bootstrap_t/1",
        "estimators": list(estimators),
        "trim_fraction": 0.10,
        "resampling_unit": "independent_gene",
        "n_genes": len(gene_labels),
        "ordered_gene_labels_sha256": canonical_sha256(gene_labels),
        "B": n_draws,
        "seed": seed,
        "namespace": namespace,
        "alpha": alpha,
        "sesoi_logit_units": sesoi,
        "simultaneous_max_abs_t_critical": critical,
        "finite_draw_language": (
            "nonparametric cluster bootstrap-t with add-one Monte Carlo p-values; "
            "bootstrap inference is not exact and its finite-draw p-values are not exact"
        ),
        "shared_resample_index_sha256": _array_sha256(draw_indices),
        "requirements": requirement_records,
        "edge_tests": edge_records,
        "path_iut_p_value": path_p,
        "p_gate_pass": bool(all_p_pass),
        "simultaneous_ci_gate_pass": bool(all_ci_pass),
        "classification": "SUPPORTED" if all_p_pass and all_ci_pass else "NOT_SUPPORTED",
    }


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
    """Evaluate one ordered scientific gate at the full stagewise alpha."""

    if family not in FAMILY_ORDER:
        raise ValueError(f"unknown R5a family: {family}")
    matrix = np.asarray(values, dtype=float)
    expected_names = family_row_names()[family]
    if list(names) != expected_names or matrix.shape != (len(names), len(groups)):
        raise ValueError("strict inference requires the exact ordered R5a family")
    matching = [(path, spec) for path, spec in named_path_specs().items() if spec["family"] == family]
    if len(matching) != 1:
        raise RuntimeError("each v2 family must implement exactly one fixed-sequence gate")
    path, spec = matching[0]
    robust = dual_estimator_studentized_bootstrap_inversion(
        matrix,
        groups,
        names,
        spec["requirements"],
        n_draws=n_draws,
        seed=seed,
        namespace=f"{namespace}:{path}",
        min_genes=min_genes,
    )
    active_controls: list[str] = []
    unresolved: list[str] = []
    inactive_targets: list[str] = []
    for row, record in robust["requirements"].items():
        classifications = {
            item["classification"] for item in record["estimators"].values()
        }
        if record["required"] == "EQUIVALENT_WITHIN_SESOI" and classifications & {
            "MATERIAL_POSITIVE", "MATERIAL_NEGATIVE"
        }:
            active_controls.append(row)
        if "UNRESOLVED" in classifications:
            unresolved.append(row)
        if record["required"] == "MATERIAL_POSITIVE" and classifications & {
            "EQUIVALENT_WITHIN_SESOI", "MATERIAL_NEGATIVE"
        }:
            inactive_targets.append(row)
    if robust["classification"] == "SUPPORTED":
        failure_mode = None
    elif active_controls:
        failure_mode = "NONSELECTIVE_OR_CONTROL_ACTIVE"
    elif unresolved:
        failure_mode = "SCIENTIFICALLY_UNRESOLVED"
    elif inactive_targets:
        failure_mode = "REQUIRED_TARGET_NOT_MATERIAL"
    else:
        failure_mode = "SCIENTIFICALLY_UNRESOLVED"
    path_record = {
        "family": family,
        "requirements": dict(spec["requirements"]),
        "fixed_sequence_alpha": STAGE_ALPHA,
        "alpha_fragmented_across_families": False,
        "robust_inference": robust,
        "iut_p_value": robust["path_iut_p_value"],
        "p_gate_pass": robust["p_gate_pass"],
        "strict_simultaneous_ci_gate_pass": robust["simultaneous_ci_gate_pass"],
        "active_equivalence_rows": active_controls,
        "unresolved_rows": unresolved,
        "inactive_material_rows": inactive_targets,
        "failure_mode": failure_mode,
        "classification": robust["classification"],
        "gates_l3_headline": bool(spec.get("gates_l3_headline", True)),
        "historical_r3c_health_used": bool(
            spec.get("historical_r3c_health_used", False)
        ),
    }
    return {
        "family": family,
        "row_names": list(names),
        "row_order_sha256": canonical_sha256(list(names)),
        "units": "masked_acceptor_G_margin_logit_units",
        "descriptive_estimates": {
            name: {
                "mean": float(np.mean(matrix[index])),
                "trimmed_mean_10pct": float(_trimmed_mean(matrix[index], 0.10)),
            }
            for index, name in enumerate(names)
        },
        "paths": {path: path_record},
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
    result: dict[str, Any] = {}
    paths: dict[str, Any] = {}
    stopped_after: str | None = None
    for gate_index, path in enumerate(CORE_PATH_ORDER, start=1):
        family = named_path_specs()[path]["family"]
        if stopped_after is not None:
            record = {
                "family": family,
                "status": "NOT_TESTED_DUE_TO_PRIOR_SCIENTIFIC_STOP",
                "prior_failed_gate": stopped_after,
            }
            result[family] = record
            paths[path] = record
            continue
        record = strict_family_inference(
            matrices[family],
            groups,
            names[family],
            family=family,
            n_draws=n_draws,
            seed=seed,
            namespace=namespace,
            min_genes=min_genes,
        )
        result[family] = record
        path_record = record["paths"][path]
        path_record["fixed_sequence_gate_index"] = gate_index
        paths[path] = path_record
        if path_record["classification"] != "SUPPORTED":
            stopped_after = path
    core_supported = stopped_after is None
    tested = [
        path
        for path in CORE_PATH_ORDER
        if paths[path].get("classification") is not None
    ]
    if core_supported:
        reader_decision = "L3H1_DONOR_LOCAL_READER_SUPPORTED"
    else:
        failure = paths[stopped_after]["failure_mode"]
        reader_decision = (
            failure
            if failure in {"NONSELECTIVE_OR_CONTROL_ACTIVE", "SCIENTIFICALLY_UNRESOLVED"}
            else "L3H1_DONOR_LOCAL_READER_NOT_SUPPORTED"
        )
    secondary_path = SECONDARY_PATH_ORDER[0]
    secondary_family = named_path_specs()[secondary_path]["family"]
    if core_supported:
        secondary_record = strict_family_inference(
            matrices[secondary_family],
            groups,
            names[secondary_family],
            family=secondary_family,
            n_draws=n_draws,
            seed=seed,
            namespace=namespace,
            min_genes=min_genes,
        )
        result[secondary_family] = secondary_record
        secondary = secondary_record["paths"][secondary_path]
        secondary["fixed_sequence_gate_index"] = len(CORE_PATH_ORDER) + 1
        paths[secondary_path] = secondary
        if secondary["classification"] == "SUPPORTED":
            l2_decision = "L2H15_UPSTREAM_READER_ROLE_SUPPORTED"
        elif secondary["failure_mode"] == "SCIENTIFICALLY_UNRESOLVED":
            l2_decision = "L2H15_ROLE_UNRESOLVED"
        else:
            l2_decision = "L2H15_ROLE_NOT_SUPPORTED"
        secondary_status = "EXECUTED_AND_CLASSIFIED"
    else:
        secondary = {
            "family": secondary_family,
            "status": "NOT_TESTED_DUE_TO_PRIOR_CORE_SCIENTIFIC_STOP",
            "prior_failed_gate": stopped_after,
            "gates_l3_headline": False,
            "historical_r3c_health_used": False,
            "model_intervention_executed": False,
        }
        result[secondary_family] = secondary
        paths[secondary_path] = secondary
        l2_decision = "L2H15_NOT_TESTED_DUE_TO_PRIOR_CORE_STOP"
        secondary_status = secondary["status"]
    return {
        "families": result,
        "paths": paths,
        "fixed_sequence": {
            "order": list(PATH_ORDER),
            "alpha_each_reached_gate": STAGE_ALPHA,
            "alpha_fragmented": False,
            "tested_gates": tested,
            "stopped_after": stopped_after,
            "downstream_untested_after_stop": True,
        },
        "secondary_sequence": {
            "order": list(SECONDARY_PATH_ORDER),
            "runs_only_after_supported_l3_core": True,
            "gates_l3_headline": False,
            "status": secondary_status,
            "historical_r3c_health_used": False,
        },
        "decisions": {
            "reader": reader_decision,
            "opens_r5b": bool(core_supported),
            "l2h15_role": l2_decision,
        },
    }


def fixed_sequence_stop_records(
    classified_paths: Mapping[str, Any],
    execution_log: Sequence[Mapping[str, Any]],
    *,
    instrument_stop: str | None,
    scientific_stop: str | None,
) -> dict[str, dict[str, Any]]:
    """Describe unclassified gates without erasing whether surgery already ran."""

    if instrument_stop is not None and scientific_stop is not None:
        raise ValueError("an R5a run cannot have both an instrument and scientific stop")
    executed = {
        str(item["path"]): bool(item.get("model_intervention_executed"))
        for item in execution_log
    }
    if not set(executed).issubset(PATH_ORDER):
        raise ValueError("execution log contains an unknown R5a gate")
    missing = [path for path in PATH_ORDER if path not in classified_paths]
    if missing and instrument_stop is None and scientific_stop is None:
        raise ValueError("missing R5a gates require an explicit stop")
    records: dict[str, dict[str, Any]] = {}
    stop_reason = instrument_stop or scientific_stop
    for gate_index, path in enumerate(PATH_ORDER, start=1):
        if path in classified_paths:
            continue
        family = named_path_specs()[path]["family"]
        reached_instrument_failure = bool(
            instrument_stop is not None and executed.get(path, False)
        )
        if reached_instrument_failure:
            status = "INSTRUMENT_UNRESOLVED_AT_REACHED_GATE"
        elif instrument_stop is not None:
            status = "NOT_TESTED_DUE_TO_PRIOR_INSTRUMENT_FAILURE"
        else:
            status = "NOT_TESTED_DUE_TO_PRIOR_SCIENTIFIC_STOP"
        records[path] = {
            "family": family,
            "status": status,
            "prior_stop": stop_reason,
            "model_intervention_executed": reached_instrument_failure,
            "fixed_sequence_gate_index": gate_index,
        }
    return records


def _synthetic_measurements(
    n: int,
    *,
    position: tuple[float, float] = (0.0, 0.0),
    content: tuple[float, float] = (0.0, 0.0),
    position_negative: tuple[float, float] = (0.0, 0.0),
    qk_partner: float = 0.0,
    qk_pseudo: float = 0.0,
    qk_control: float = 0.0,
    l2_qk_partner: float = 0.0,
    l2_qk_pseudo: float = 0.0,
    av_partner: float = 0.0,
    av_pseudo: float = 0.0,
    av_control: float = 0.0,
) -> dict[str, Any]:
    """Build coherent primitive cells for classifier plants/boundary diagnostics."""

    if n < 2:
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

    set_shift("L3H1", "paired_pseudo", position, content)
    for comparison in SHIFT_COMPARISONS:
        set_shift("L0H2", comparison, position_negative)

    all_qk_arms = tuple(dict.fromkeys((*QK_ARMS, *L2_QK_ARMS)))
    qk = {
        head: {
            window: {"partner": zero(), "pseudo": zero()}
            for candidate, window in all_qk_arms
            if candidate == head
        }
        for head, _window in all_qk_arms
    }
    qk["L3H1"]["paired"] = {
        "partner": zero(qk_partner),
        "pseudo": zero(qk_pseudo),
    }
    qk["L3H1"]["pseudo"]["partner"] = zero(qk_control)
    qk["L2H15"]["paired"] = {
        "partner": zero(l2_qk_partner),
        "pseudo": zero(l2_qk_pseudo),
    }
    qk["L2H15"]["pseudo"]["partner"] = zero(qk_control)

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
    av["paired"]["partner"] = cells(
        address=av_partner, value=av_partner, joint=2.0 * av_partner
    )
    av["paired"]["pseudo"] = cells(
        address=av_pseudo, value=av_pseudo, joint=2.0 * av_pseudo
    )
    av["pseudo"]["partner"] = cells(
        address=av_control, value=av_control, joint=2.0 * av_control
    )
    return {
        "position_shift": shift,
        "qk_surgery": qk,
        "attention_value": av,
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
    """Replay strong coherent plants and exact-boundary non-rejection diagnostics.

    This is an implementation calibration, not an empirical type-I-error proof.  The
    production decision is the dual-estimator bootstrap-t inversion itself.
    """

    if n_genes < MIN_GENES or n_draws < 1 or n_trials < 1:
        raise ValueError("R5a calibration requires its complete gene count and positive settings")
    groups = [f"calibration-gene-{index:03d}" for index in range(n_genes)]
    registry = {
        "G1_POSITION_CAUSAL_MARGIN": {
            "family": "position_margin",
            "plant": {"position": (0.40, 0.40)},
            "boundary": {"position": (MARGIN_SESOI, MARGIN_SESOI)},
        },
        "G2_EXACT_QK_CAUSAL_MARGIN": {
            "family": "qk_surgery",
            "plant": {"qk_partner": 0.40},
            "boundary": {"qk_partner": MARGIN_SESOI},
        },
        "G3_FIXED_V_AXV_CAUSAL_MARGIN": {
            "family": "attention_value",
            "plant": {"av_partner": 0.40},
            "boundary": {"av_partner": MARGIN_SESOI},
        },
    }
    records: dict[str, Any] = {}
    for gate_index, (path, spec) in enumerate(registry.items()):
        gate_records: dict[str, Any] = {}
        for label in ("plant", "boundary"):
            rng = np.random.default_rng(
                int.from_bytes(
                    hashlib.sha256(
                        f"r5a-v2-calibration|{path}|{label}|{seed}".encode("utf-8")
                    ).digest()[:8],
                    "big",
                )
            )
            primitive = _add_centered_primitive_noise(
                _synthetic_measurements(n_genes, **spec[label]),
                rng=rng,
                scale=CALIBRATION_PRIMITIVE_NOISE_SD,
            )
            matrix = build_family_matrices(primitive)[spec["family"]]
            result = strict_family_inference(
                matrix,
                groups,
                family_row_names()[spec["family"]],
                family=spec["family"],
                n_draws=n_draws,
                seed=seed,
                namespace=f"r5a-v2-calibration:{path}:{label}",
                min_genes=MIN_GENES,
            )
            classification = result["paths"][path]["classification"]
            gate_records[label] = {
                "classification": classification,
                "pass": classification == ("SUPPORTED" if label == "plant" else "NOT_SUPPORTED"),
            }
        records[path] = {
            "fixed_sequence_gate_index": gate_index + 1,
            "family": spec["family"],
            **gate_records,
            "pass": gate_records["plant"]["pass"] and gate_records["boundary"]["pass"],
        }
    return {
        "schema": "gfmcd.r5a.classifier_calibration/2",
        "production_classifier_used": True,
        "calibration_kind": "strong-plant_and_boundary_nonrejection_implementation_check",
        "not_a_global_type_i_error_proof": True,
        "dual_estimators": ["mean", "trimmed_mean_10pct"],
        "bootstrap_kind": "studentized nonparametric gene bootstrap; not exact",
        "n_genes": n_genes,
        "n_draws": n_draws,
        "n_trials_argument_retained_for_stack_compatibility": n_trials,
        "fixed_sequence_alpha": STAGE_ALPHA,
        "alpha_fragmented": False,
        "registry": registry,
        "registry_sha256": canonical_sha256(registry),
        "gates": records,
        "all_pass": all(record["pass"] for record in records.values()),
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
    examined.  A seven-token guard surrounds the acceptor query; every paired and
    pseudo protected token receives a four-token guard on both sides.
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


def paired_noop_noninferiority(
    noop_correct: Sequence[float],
    intervention_correct: Sequence[float],
    groups: Sequence[str],
    *,
    margin: float = HEALTH_NONINFERIORITY_MARGIN,
    n_draws: int = PRODUCTION_RANDOMISATION,
    seed: int = PRODUCTION_SEED,
    namespace: str = "r5a-health",
) -> dict[str, Any]:
    """One-sided paired gene-bootstrap noninferiority against the exact hook no-op."""

    noop = np.asarray(noop_correct, dtype=float)
    intervention = np.asarray(intervention_correct, dtype=float)
    labels = np.asarray(list(map(str, groups)), dtype=object)
    if (
        noop.shape != intervention.shape
        or noop.ndim != 1
        or labels.shape != noop.shape
        or not np.isfinite(noop).all()
        or not np.isfinite(intervention).all()
    ):
        raise ValueError("paired health vectors must be finite and shape matched")
    gene_ids = sorted(set(labels.tolist()))
    delta = np.asarray(
        [
            np.mean(intervention[labels == gene] - noop[labels == gene])
            for gene in gene_ids
        ],
        dtype=float,
    )
    rng_seed = int.from_bytes(
        hashlib.sha256(f"{namespace}|{seed}".encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(delta), size=(n_draws, len(delta)))
    observed, observed_se = _location_and_se(delta, "mean")
    boot, boot_se = _location_and_se(delta[indices], "mean")
    numerator = boot - float(observed)
    t_star = np.divide(
        numerator,
        boot_se,
        out=np.where(numerator >= 0.0, np.inf, -np.inf),
        where=boot_se > 0.0,
    )
    t_star[(boot_se == 0.0) & (numerator == 0.0)] = 0.0
    critical = float(np.quantile(t_star, 1.0 - STAGE_ALPHA, method="higher"))
    lower = float(observed - critical * observed_se)
    return {
        "noop_accuracy": float(np.mean(noop)),
        "intervention_accuracy": float(np.mean(intervention)),
        "paired_gene_mean_difference": float(observed),
        "one_sided_95pct_studentized_bootstrap_lower": lower,
        "noninferiority_margin": float(margin),
        "criterion": "lower_bound > -noninferiority_margin",
        "pass": bool(lower > -margin),
        "B": n_draws,
        "resampling_unit": "gene",
        "bootstrap_is_exact": False,
    }


def health_policy_registry() -> dict[str, Any]:
    """Freeze core and secondary health gates without importing historical R3c health."""

    return {
        "core": {
            "local_query_zero_L3H1": READER,
            "local_query_zero_L0H2": NEGATIVE_HEAD,
        },
        "secondary": {"local_query_zero_L2H15": SECONDARY_READER},
        "historical_r3c_health_used": False,
        "secondary_failure_gates_l3_headline": False,
    }


def bounded_intervention_health(
    adapter,
    task: CouplingTask,
    rows: Sequence[Any],
    *,
    seed: int = PRODUCTION_SEED,
) -> dict[str, Any]:
    """Compare each query-head stress to its paired exact-hook no-op."""

    selected = list(rows[:HEALTH_MAX_EXAMPLES])
    if len(selected) != HEALTH_MAX_EXAMPLES:
        raise RuntimeError("R5a paired health requires 16 selected examples")
    policy_registry = health_policy_registry()
    policies = tuple(
        (name, head)
        for family in ("core", "secondary")
        for name, head in policy_registry[family].items()
    )
    correct = {
        name: {"noop": [], "intervention": [], "groups": []}
        for name, _head in policies
    }
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
                {"example_id": str(row.example.example_id), "positions": positions}
            )
            encoded["input_ids"][
                local_index, [position + task.off for position in positions]
            ] = task.mask_id

        query_tokens = torch.tensor(
            [position + task.off for position in readout],
            device=adapter.device,
            dtype=torch.long,
        )
        query_mask = torch.zeros_like(encoded["attention_mask"], dtype=torch.bool)
        query_mask[torch.arange(len(batch_rows), device=adapter.device), query_tokens] = True
        for name, query_head in policies:
            layer, _head = query_head
            state = _capture_qkv_and_heads(adapter, encoded, layer=layer)
            sources = {
                "noop": {layer: state["heads"]},
                "intervention": {layer: torch.zeros_like(state["heads"])},
            }
            logits_by_arm = {}
            for arm, source in sources.items():
                with patch_heads(
                    adapter, source, [query_head], token_mask=query_mask
                ):
                    logits_by_arm[arm] = adapter.model(**encoded).logits.float()
            for local_index, (row, positions) in enumerate(zip(batch_rows, probes)):
                token_positions = [position + task.off for position in positions]
                target = torch.tensor(
                    [task.bid[row.clean_sequence[position]] for position in positions],
                    device=adapter.device,
                )
                for arm, logits in logits_by_arm.items():
                    predicted = _top1_nucleotide_token_ids(
                        logits[local_index, token_positions], task.bid
                    )
                    correct[name][arm].extend(
                        (predicted == target).float().cpu().tolist()
                    )
                correct[name]["groups"].extend(
                    [str(row.example.gene_id)] * len(positions)
                )

    records = {
        name: paired_noop_noninferiority(
            values["noop"],
            values["intervention"],
            values["groups"],
            margin=HEALTH_NONINFERIORITY_MARGIN,
            n_draws=PRODUCTION_RANDOMISATION,
            seed=seed,
            namespace=f"r5a-health:{name}",
        )
        for name, values in correct.items()
    }
    core_names = tuple(policy_registry["core"])
    secondary_names = tuple(policy_registry["secondary"])
    core_all_pass = all(records[name]["pass"] for name in core_names)
    secondary_all_pass = all(records[name]["pass"] for name in secondary_names)
    return {
        "schema": "gfmcd.r5a.paired_noop_health/2",
        "candidate_outcomes_used": False,
        "comparison": "intervention minus exact same-hook no-op on paired probes",
        "absolute_accuracy_floor_used": False,
        "maximum_examples": HEALTH_MAX_EXAMPLES,
        "probes_per_example": HEALTH_PROBES_PER_EXAMPLE,
        "probe_registry": probe_registry,
        "probe_registry_sha256": canonical_sha256(probe_registry),
        "policy_registry": policy_registry,
        "policies": records,
        "core_all_pass": core_all_pass,
        "secondary_all_pass": secondary_all_pass,
        "all_pass": core_all_pass and secondary_all_pass,
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
    query_state: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a query-head row, optionally fixing Q to another captured state."""

    query_source = state if query_state is None else query_state
    logits, attention = qk_attention_row(
        query_source["query"],
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


def _head_query_at_tokens(
    state: Mapping[str, Any],
    query_tokens: torch.Tensor,
    *,
    head: int,
    num_heads: int,
) -> torch.Tensor:
    query = state["query"]
    if query.ndim != 3 or query.shape[-1] % num_heads:
        raise RuntimeError("R5a captured query tensor shape differs")
    width = query.shape[-1] // num_heads
    rows = torch.arange(query.shape[0], device=query.device)
    return query[
        rows,
        query_tokens,
        head * width : (head + 1) * width,
    ]


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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exchange paired/pseudo embeddings with clean numerical Q and V fixed."""

    device = adapter.device
    query_tokens = torch.tensor(
        [position + task.off for position in readout_positions],
        dtype=torch.long,
        device=device,
    )
    windows = {
        "paired": _window_tensor(paired_positions, token_offset=task.off, device=device),
        "pseudo": _window_tensor(pseudo_positions, token_offset=task.off, device=device),
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
        "decision_query_source": "numerically identical clean query row",
        "shifted_native_query_role": "DESCRIPTIVE_ONLY",
        "shifted_native_query_decision_bearing": False,
        "clean_recipient_for_every_cell": True,
        "input_ids_sha256": _array_sha256(encoded["input_ids"].cpu().numpy()),
        "fixed_V_sha256_by_head": {},
        "fixed_query_sha256_by_head": {},
        "fixed_query_sha256_by_head_and_cell": {},
        "fixed_query_assignment_max_abs_error": 0.0,
        "shifted_native_query_change_max_abs_by_head_and_cell": {},
        "full_cell_margin_identity_max_abs": 0.0,
        "full_cell_attention_reconstruction_max_abs": 0.0,
        "component_swap_involution_max_abs": 0.0,
        "component_swap_assignment_max_abs": 0.0,
        "component_swap_outside_max_abs": 0.0,
        "bridge_reuses_one_clean_V_tensor_for_all_cells": True,
        "bridge_fixed_V_sha256_by_head_and_cell": {},
    }
    for comparison, left_name, right_name in (
        ("paired_pseudo", "paired", "pseudo"),
    ):
        left, right = windows[left_name], windows[right_name]
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
            fixed_query = _head_query_at_tokens(
                clean_states[layer],
                query_tokens,
                head=head,
                num_heads=adapter.num_heads,
            )
            fixed_query_sha256 = _array_sha256(
                fixed_query.detach().cpu().numpy()
            )
            invariants["fixed_query_sha256_by_head"][head_name] = (
                fixed_query_sha256
            )
            invariants["fixed_query_sha256_by_head_and_cell"][head_name] = {}
            invariants[
                "shifted_native_query_change_max_abs_by_head_and_cell"
            ][head_name] = {}
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
                cell_fixed_query = _head_query_at_tokens(
                    clean_states[layer],
                    query_tokens,
                    head=head,
                    num_heads=adapter.num_heads,
                )
                cell_fixed_query_sha256 = _array_sha256(
                    cell_fixed_query.detach().cpu().numpy()
                )
                invariants["fixed_query_sha256_by_head_and_cell"][head_name][
                    cell
                ] = cell_fixed_query_sha256
                fixed_query_error = float(
                    (cell_fixed_query.float() - fixed_query.float()).abs().max()
                )
                invariants["fixed_query_assignment_max_abs_error"] = max(
                    invariants["fixed_query_assignment_max_abs_error"],
                    fixed_query_error,
                )
                if cell_fixed_query_sha256 != fixed_query_sha256:
                    raise RuntimeError("R5a fixed query digest drifted across G1 cells")
                shifted_native_query = _head_query_at_tokens(
                    states[cell][layer],
                    query_tokens,
                    head=head,
                    num_heads=adapter.num_heads,
                )
                invariants[
                    "shifted_native_query_change_max_abs_by_head_and_cell"
                ][head_name][cell] = float(
                    (shifted_native_query.float() - fixed_query.float()).abs().max()
                )
                qk, attention = _head_rows(
                    states[cell][layer],
                    encoded,
                    query_tokens,
                    head=head,
                    num_heads=adapter.num_heads,
                    query_state=clean_states[layer],
                )
                if cell == "full":
                    native_attention = states[cell][layer]["attentions"][:, head]
                    rows = torch.arange(
                        attention.shape[0], device=attention.device
                    )
                    native_row = native_attention[rows, query_tokens]
                    error = float(
                        (native_row.float() - attention.float()).abs().max()
                    )
                    invariants[
                        "full_cell_attention_reconstruction_max_abs"
                    ] = max(
                        invariants[
                            "full_cell_attention_reconstruction_max_abs"
                        ],
                        error,
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
        invariants["full_cell_attention_reconstruction_max_abs"] < HOOK_ATOL
        and invariants["full_cell_margin_identity_max_abs"] < MARGIN_ATOL
        and invariants["fixed_query_assignment_max_abs_error"] == 0.0
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
    *,
    arms: Sequence[tuple[str, str]] = QK_ARMS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_tokens = torch.tensor(
        [position + task.off for position in readout_positions],
        dtype=torch.long,
        device=adapter.device,
    )
    windows = {
        "paired": _window_tensor(paired_positions, token_offset=task.off, device=adapter.device),
        "pseudo": _window_tensor(pseudo_positions, token_offset=task.off, device=adapter.device),
    }
    selected_arms = tuple((str(head), str(window)) for head, window in arms)
    if not selected_arms or any(window not in windows for _head, window in selected_arms):
        raise ValueError("QK arms must be a nonempty paired/pseudo registry")
    layers = tuple(sorted({head_tuple(head)[0] for head, _ in selected_arms}))
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
    # Every capture carries the same complete-model logits.  Index through an actually
    # requested layer so the separately registered L2H15/L0H2 family does not depend on
    # L3H1 being present in its arm registry.
    clean_margin = task.margin(
        states["clean"][layers[0]]["logits"], readout_positions
    ).cpu().numpy()
    output: dict[str, Any] = {
        head: {window: {} for candidate, window in selected_arms if candidate == head}
        for head, _ in selected_arms
    }
    maximum_unselected_error = 0.0
    maximum_selected_assignment_error = 0.0
    maximum_value_error = 0.0
    fixed_value_sha256_by_arm: dict[str, str] = {}
    for head_name, window_name in selected_arms:
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_tokens = torch.tensor(
        [position + task.off for position in readout_positions],
        dtype=torch.long,
        device=adapter.device,
    )
    windows = {
        "paired": _window_tensor(paired_positions, token_offset=task.off, device=adapter.device),
        "pseudo": _window_tensor(pseudo_positions, token_offset=task.off, device=adapter.device),
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


def _normalized_runner_sha256(path: Path = RUNNER) -> str:
    text = path.read_text()
    names = (
        "FROZEN_PROTOCOL_SHA256",
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256",
        "FROZEN_PREFLIGHT_ARTIFACT_SHA256",
        "FROZEN_PREFLIGHT_MANIFEST_SHA256",
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256",
        "FROZEN_R1C_TARGET_ARTIFACT_SHA256",
        "FROZEN_R1C_TARGET_MANIFEST_SHA256",
        "FROZEN_R1C_TARGET_ATTEMPT_SHA256",
        "FROZEN_R1C_TARGET_COMPLETION_SHA256",
        "FROZEN_R3D_ARTIFACT_SHA256",
        "FROZEN_R3D_MANIFEST_SHA256",
        "FROZEN_R3D_REGISTRY_ARTIFACT_SHA256",
        "FROZEN_R3D_REGISTRY_MANIFEST_SHA256",
        "FROZEN_R3D_ATTEMPT_SHA256",
        "FROZEN_R3D_COMPLETION_SHA256",
        "FROZEN_R3C_ARTIFACT_SHA256",
        "FROZEN_R3C_MANIFEST_SHA256",
        "FROZEN_CHECKPOINT_AGGREGATE_SHA256",
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
        "schema": "gfmcd.r5a.instrument_binding/2",
        "r5_stack_binding": stack,
        "r5_stack_fingerprint_sha256": r5_stack_fingerprint(stack),
        "normalized_runner_sha256": _normalized_runner_sha256(),
        "source_sha256": {
            str(path.relative_to(REPO)): (
                normalized_literal_pin_source_sha256(
                    path,
                    permitted_pin_names=SHARED_R1C_PACKAGE_PIN_NAMES,
                )
                if path == COHORT_CORE
                else sha256_file(path)
            )
            for path in sources
        },
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
            "health_noninferiority_margin": HEALTH_NONINFERIORITY_MARGIN,
            "r1c_sdpa_to_r5a_eager_common_leg_atol": SDPA_EAGER_COMMON_LEG_ATOL,
            "margin_sesoi": MARGIN_SESOI,
            "stage_alpha": STAGE_ALPHA,
            "fixed_sequence_alpha_fragmented": False,
            "pseudo_caliper_nt": DECOY_TOL,
            "cohort_handoffs": {
                "preflight": R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
                "target": R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
                "exact_ordered_rows": EXACT_ROWS,
                "adjacent_diagnostic_required": False,
            },
            "reader": list(READER),
            "secondary_reader": list(SECONDARY_READER),
            "negative_head": list(NEGATIVE_HEAD),
            "family_order": list(FAMILY_ORDER),
            "family_rows_sha256": {
                name: canonical_sha256(rows) for name, rows in family_row_names().items()
            },
            "path_registry_sha256": canonical_sha256(named_path_specs()),
            "health_policy_registry": health_policy_registry(),
            "behavioral_endpoint": "masked_acceptor_G_margin_only",
            "donor_readout_constructed": False,
        },
    }


def _all_freeze_pins_present(*, include_preflight: bool = True) -> bool:
    hash_pins: tuple[Any, ...] = (
        FROZEN_PROTOCOL_SHA256,
        FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
        FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256,
        FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256,
        FROZEN_R3D_ARTIFACT_SHA256,
        FROZEN_R3D_MANIFEST_SHA256,
        FROZEN_R3D_REGISTRY_ARTIFACT_SHA256,
        FROZEN_R3D_REGISTRY_MANIFEST_SHA256,
        FROZEN_R3D_ATTEMPT_SHA256,
        FROZEN_R3D_COMPLETION_SHA256,
        FROZEN_R3C_ARTIFACT_SHA256,
        FROZEN_R3C_MANIFEST_SHA256,
        FROZEN_CHECKPOINT_AGGREGATE_SHA256,
    )
    if include_preflight:
        hash_pins += (
            FROZEN_R1C_TARGET_ARTIFACT_SHA256,
            FROZEN_R1C_TARGET_MANIFEST_SHA256,
            FROZEN_R1C_TARGET_ATTEMPT_SHA256,
            FROZEN_R1C_TARGET_COMPLETION_SHA256,
            FROZEN_PREFLIGHT_ARTIFACT_SHA256,
            FROZEN_PREFLIGHT_MANIFEST_SHA256,
        )
    return bool(
        all(_is_sha256(value) for value in hash_pins)
        and FROZEN_R3D_ARTIFACT_SHA256 == R3D_TARGET_ARTIFACT_SHA256
        and FROZEN_R3D_MANIFEST_SHA256 == R3D_TARGET_MANIFEST_SHA256
        and FROZEN_R3D_REGISTRY_ARTIFACT_SHA256 == R3D_REGISTRY_ARTIFACT_SHA256
        and FROZEN_R3D_REGISTRY_MANIFEST_SHA256 == R3D_REGISTRY_MANIFEST_SHA256
        and FROZEN_R3D_ATTEMPT_SHA256 == R3D_TARGET_ATTEMPT_SHA256
        and FROZEN_R3D_COMPLETION_SHA256 == R3D_TARGET_COMPLETION_SHA256
        and FROZEN_R3C_ARTIFACT_SHA256 == R3C_PREFLIGHT_ARTIFACT_SHA256
        and FROZEN_R3C_MANIFEST_SHA256 == R3C_PREFLIGHT_MANIFEST_SHA256
    )


def authorize_model_scoring(stage: str) -> dict[str, Any]:
    """Fail before resolving any model/data path unless the entire R5a seal is literal."""

    if stage not in {"preflight", "target"}:
        raise SealError("model scoring requires preflight or target stage")
    if not _all_freeze_pins_present(include_preflight=stage == "target"):
        raise SealError(
            "R5a scoring is hard closed: model scoring requires all literal method, R1c "
            "handoff, checkpoint, instrument, and stage-preflight pins"
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


def load_authenticated_r1c_preflight_handoff(
    artifact_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Authenticate R1c's exact nonsealed validation rows for method preflight only."""

    try:
        handoff = load_r1c_preflight_cohort_handoff(
            artifact_path, manifest_path
        )
        verified = validate_r5_preflight_cohort_handoff(handoff)
    except (R5CapstoneCohortError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"R1c validation-preflight handoff authentication failed: {exc}"
        ) from exc
    if (
        handoff.get("schema") != R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA
        or handoff.get("row_count") != EXACT_ROWS
        or verified.get("row_count") != EXACT_ROWS
        or verified.get("verified") is not True
        or handoff.get("scientific_entry_authorized") is not False
        or handoff.get("sealed_split_opened") is not False
    ):
        raise RuntimeError("R1c validation-preflight handoff disposition differs")
    package_inputs = [
        _pinned_input_identity(
            artifact_path,
            FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256,
            label="R1c validation-preflight artifact",
        ),
        _pinned_input_identity(
            manifest_path,
            FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256,
            label="R1c validation-preflight manifest",
        ),
    ]
    return {
        "handoff": handoff,
        "verified": verified,
        "row_count": EXACT_ROWS,
        "exact_order_preserved": True,
        "scientific_entry_authorized": False,
        "sealed_split_opened": False,
        "package_inputs": package_inputs,
    }


def build_r5a_target_identity(
    capstone_handoff: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Bind R5a spend to R1c's exact sealed 400-row biological target."""

    if capstone_handoff.get("row_count") != EXACT_ROWS:
        raise RuntimeError("R5a target identity requires the exact 400-row handoff")
    return build_r5_biological_target_identity(
        stage="R5a",
        checkpoint_sha256=checkpoint_sha256,
        target_gene_ids_sha256=str(
            capstone_handoff.get("sorted_unique_gene_ids_sha256")
        ),
        ordered_rows_sha256=str(
            capstone_handoff.get("ordered_row_ids_sha256")
        ),
        construction_sha256=str(
            capstone_handoff.get("capstone_construction_sha256")
        ),
        endpoint_policy="masked_acceptor_G_margin_only",
    )


R1C_CONFIRMED_BEHAVIOR_VERDICTS = frozenset(
    {
        "CONFIRMED_PAIRED_DONOR_ACCEPTOR_BEHAVIOR",
        "CONFIRMED_DOWNSTREAM_ACCEPTOR_PREFERENCE",
    }
)
R1C_VALID_SCIENTIFIC_NULL_VERDICTS = frozenset(
    {
        "PAIRED_DONOR_PATH_UNRESOLVED_ROBUSTNESS",
        "PAIRED_DONOR_PATH_UNRESOLVED",
        "NO_CONFIRMED_PAIRED_DONOR_PATH",
    }
)


def _pinned_input_identity(
    path_value: str | Path | None,
    expected_sha256: str | None,
    *,
    label: str,
) -> dict[str, Any]:
    if path_value is None or not _is_sha256(expected_sha256):
        raise RuntimeError(f"R5a requires a literal-pinned {label}")
    path = Path(path_value).expanduser().resolve(strict=True)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(f"R5a {label} differs from its literal pin")
    return {
        "path": str(path),
        "logical_name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": observed,
    }


def classify_r1c_target_behavior_gate(
    attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify only the public R1c successor attestation, never an invented schema."""

    if attestation.get("schema") != "gfmcd.r1c.verified_target_handoff/1":
        raise RuntimeError("R1c successor target attestation schema differs")
    verdict = attestation.get("verdict")
    if verdict in R1C_CONFIRMED_BEHAVIOR_VERDICTS:
        if attestation.get("all_gates_pass") is not True:
            raise RuntimeError("R1c confirmed verdict lacks passing frozen gates")
        return {
            "verdict": verdict,
            "scientific_gate_pass": True,
            "failure_is_scientific_not_instrumental": False,
            "instrument_failure": False,
        }
    if verdict in R1C_VALID_SCIENTIFIC_NULL_VERDICTS:
        return {
            "verdict": verdict,
            "scientific_gate_pass": False,
            "failure_is_scientific_not_instrumental": True,
            "instrument_failure": False,
        }
    if verdict == "INCONCLUSIVE":
        raise RuntimeError("R1c target instrument was inconclusive")
    raise RuntimeError("R1c successor target verdict is not prespecified")


def _verify_r1c_successor_target_package(
    artifact_path: str | Path,
    manifest_path: str | Path,
    attempt_path: str | Path,
    completion_path: str | Path,
    *,
    preflight_artifact_path: str | Path,
) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("_gfmcd_r1c_successor", R1C_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("R1c successor verifier could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verifier = getattr(module, "verify_successor_target_package", None)
    if not callable(verifier):
        raise RuntimeError("R1c successor target verifier is unavailable")
    result = verifier(
        artifact_path,
        manifest_path,
        attempt_path,
        completion_path,
        preflight_artifact_path=preflight_artifact_path,
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("R1c successor target verifier returned a malformed result")
    return dict(result)


def load_authenticated_r1c_target_handoff(
    artifact_path: str | Path,
    manifest_path: str | Path,
    attempt_path: str | Path,
    completion_path: str | Path,
    preflight_artifact_path: str | Path,
    preflight_manifest_path: str | Path,
) -> dict[str, Any]:
    """Authenticate the exact R1c six-file target package and scientific gate."""

    specs = (
        (artifact_path, FROZEN_R1C_TARGET_ARTIFACT_SHA256, "R1c target artifact"),
        (manifest_path, FROZEN_R1C_TARGET_MANIFEST_SHA256, "R1c target manifest"),
        (attempt_path, FROZEN_R1C_TARGET_ATTEMPT_SHA256, "R1c target attempt"),
        (completion_path, FROZEN_R1C_TARGET_COMPLETION_SHA256, "R1c target completion"),
        (
            preflight_artifact_path,
            FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256,
            "R1c predecessor preflight artifact",
        ),
        (
            preflight_manifest_path,
            FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256,
            "R1c predecessor preflight manifest",
        ),
    )
    package_inputs = [
        _pinned_input_identity(path, pin, label=label)
        for path, pin, label in specs
    ]
    attestation = _verify_r1c_successor_target_package(
        artifact_path,
        manifest_path,
        attempt_path,
        completion_path,
        preflight_artifact_path=preflight_artifact_path,
    )
    behavior_gate = classify_r1c_target_behavior_gate(attestation)
    if not behavior_gate["scientific_gate_pass"]:
        return {
            "handoff": None,
            "verified": None,
            "attestation": attestation,
            "behavior_gate": behavior_gate,
            "package_inputs": package_inputs,
        }
    try:
        handoff = load_r1c_capstone_cohort_handoff(
            artifact_path,
            manifest_path,
            attempt_path,
            completion_path,
            preflight_artifact_path,
            preflight_manifest_path,
        )
        verified = validate_r5_capstone_cohort_handoff(handoff)
    except (R5CapstoneCohortError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(f"R1c target handoff authentication failed: {exc}") from exc
    if (
        handoff.get("schema") != R5_CAPSTONE_COHORT_HANDOFF_SCHEMA
        or handoff.get("row_count") != EXACT_ROWS
        or verified.get("row_count") != EXACT_ROWS
        or verified.get("verified") is not True
    ):
        raise RuntimeError("R1c target handoff disposition differs")
    return {
        "handoff": handoff,
        "verified": verified,
        "attestation": attestation,
        "behavior_gate": behavior_gate,
        "package_inputs": package_inputs,
    }


def load_exposure_packages(args: argparse.Namespace) -> dict[str, Any]:
    """Authenticate R3 provenance without deriving an R5 cohort from exclusions."""

    r3d = load_r3d_entry_package(
        args.r3d_artifact,
        args.r3d_manifest,
        registry_path=args.r3d_registry,
        registry_manifest_path=args.r3d_registry_manifest,
        attempt_path=args.r3d_attempt,
        completion_path=args.r3d_completion,
    )
    if r3d.get("attempt") is None or r3d.get("completion") is None:
        raise RuntimeError(
            "R5a requires the exact R3d target-attempt and completion receipts"
        )
    r3c = load_r3c_exclusion_package(args.r3c_artifact, args.r3c_manifest)
    chain = {
        "r3d_scientific_entry": r3d,
        "r3c_exclusion_provenance": r3c,
    }
    chain["exposure_chain_sha256"] = canonical_sha256(
        {
            "r3d_package_contract_sha256": r3d["package_contract_sha256"],
            "r3c_package_contract_sha256": r3c["package_contract_sha256"],
        }
    )
    return chain


def load_upstream_chain(
    exposure_chain: Mapping[str, Any], r1c_package: Mapping[str, Any]
) -> dict[str, Any]:
    """Combine already-authenticated R3 provenance with the later R1c package."""

    chain = dict(exposure_chain)
    chain["r1c_package"] = dict(r1c_package)
    chain["chain_sha256"] = canonical_sha256(
        {
            "exposure_chain_sha256": chain["exposure_chain_sha256"],
            "r1c_package_sha256": canonical_sha256(
                r1c_package.get("package_inputs", [])
            ),
            "r1c_handoff_record_sha256": (
                r1c_package.get("handoff", {}) or {}
            ).get("record_sha256"),
        }
    )
    return chain


def materialize_authenticated_cohort(
    gtf: str,
    fasta: str,
    *,
    stage: str,
    handoff: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]]:
    """Materialize all 400 authenticated R1c rows in byte-frozen order."""

    if stage not in {"preflight", "target"}:
        raise ValueError("R5a cohort stage must be preflight or target")
    source_identity = validate_expected_r5_source_identity(gtf, fasta)
    split = "val" if stage == "preflight" else "test"
    allowed = R5_VALIDATION_CHROMOSOMES if stage == "preflight" else ("chr8", "chr9")
    tag = (
        "r1c-preflight-r1c-single-g-primary-family-v2"
        if stage == "preflight"
        else "r1c-target-r1c-single-g-primary-family-v2"
    )
    examples = load_examples_unscored(
        gtf,
        fasta,
        split,
        None,
        tag=tag,
        allowed_chromosomes=allowed,
    )
    materializer = (
        materialize_r5_preflight_rows
        if stage == "preflight"
        else materialize_r5_capstone_rows
    )
    rows, audit = materializer(examples, handoff)
    ordered_ids = [str(row.example.example_id) for row in rows]
    expected_ids = [str(row["example_id"]) for row in handoff["rows"]]
    if (
        len(rows) != EXACT_ROWS
        or ordered_ids != expected_ids
        or audit.get("handoff_order_preserved") is not True
        or audit.get("rows_filtered_or_reordered") is not False
        or audit.get("adjacent_window_required") is not False
    ):
        raise RuntimeError("R5a cohort did not preserve the exact 400-row R1c handoff")
    return tuple(rows), dict(audit), source_identity

def model_free_audit(args: argparse.Namespace) -> dict[str, Any]:
    """Publish the prospective binding state without opening genome or model data."""

    binding = instrument_binding_payload()
    return {
        "schema": "gfmcd.r5a.prospective_instrument_audit/2",
        "model_loaded": False,
        "model_outputs_read": False,
        "genome_opened": False,
        "sealed_chr8_chr9_opened": False,
        "preflight_pins_ready": _all_freeze_pins_present(include_preflight=False),
        "target_pins_ready": _all_freeze_pins_present(include_preflight=True),
        "preflight_handoff": {
            "schema": R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
            "row_count": EXACT_ROWS,
            "scientific_entry_authorized": False,
        },
        "target_handoff": {
            "schema": R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
            "row_count": EXACT_ROWS,
            "requires_prior_r5a_target_receipt_before_genome_or_model": True,
        },
        "instrument_binding": binding,
        "instrument_fingerprint_sha256": canonical_sha256(binding),
        "r5_stack_binding": binding["r5_stack_binding"],
        "r5_stack_fingerprint_sha256": binding[
            "r5_stack_fingerprint_sha256"
        ],
    }


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
        sesoi=MARGIN_SESOI,
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
        boundaries=[MARGIN_SESOI, -MARGIN_SESOI, MARGIN_SESOI, MARGIN_SESOI],
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
        strict_interval_decision(*decisions["native_partner"].simultaneous_ci, MARGIN_SESOI)
        == "MATERIAL_POSITIVE"
        and strict_interval_decision(*decisions["native_pseudo"].simultaneous_ci, MARGIN_SESOI)
        == "EQUIVALENT_WITHIN_SESOI"
        and strict_interval_decision(*decisions["native_specific"].simultaneous_ci, MARGIN_SESOI)
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


def _verify_bound_preflight(
    path_value: str | None,
    fingerprint: str,
    *,
    stack_binding: Mapping[str, Any],
    stack_fingerprint: str,
) -> dict[str, Any]:
    """Authenticate an instrument-only R5a preflight; no observed science may gate."""

    if path_value is None:
        raise RuntimeError("R5a target requires --preflight-artifact")
    path = Path(path_value).resolve()
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise RuntimeError("R5a target requires artifact and sibling manifest")
    if sha256_file(path) != FROZEN_PREFLIGHT_ARTIFACT_SHA256:
        raise RuntimeError("R5a preflight artifact bytes differ from the literal pin")
    if sha256_file(manifest_path) != FROZEN_PREFLIGHT_MANIFEST_SHA256:
        raise RuntimeError("R5a preflight manifest bytes differ from the literal pin")
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
        record.get("schema") != "gfmcd.r5a.acceptor_address_binding/2"
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
    contract = record.get("r5_entry_contract")
    cohort = record.get("cohort_handoff")
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema") != R5_ENTRY_SCHEMA
        or contract.get("claim_eligible") is not False
        or contract.get("verdict") != "PREFLIGHT_PASS"
        or record.get("scientific_entry_authorized") is not False
        or record.get("r5a_opened_chr8_chr9") is not False
        or record.get("observed_scientific_inference") is not None
        or not isinstance(cohort, Mapping)
        or cohort.get("schema") != R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA
        or cohort.get("row_count") != EXACT_ROWS
        or cohort.get("scientific_entry_authorized") is not False
        or cohort.get("sealed_split_opened") is not False
        or cohort.get("handoff_order_preserved") is not True
    ):
        raise RuntimeError("R5a preflight nonentry handoff contract differs")
    common_replay = record.get("same_row_r1c_common_leg_replay")
    av_capability = record.get("capability", {}).get("families", {}).get(
        "attention_value"
    )
    if (
        not isinstance(common_replay, Mapping)
        or common_replay.get("source_execution") != "R1c eager=False (SDPA)"
        or common_replay.get("replay_execution") != "R5a eager=True (hook-capable)"
        or common_replay.get("atol") != SDPA_EAGER_COMMON_LEG_ATOL
        or common_replay.get("pass") is not True
        or common_replay.get("mismatch_disposition") != "INSTRUMENT_UNRESOLVED"
        or not isinstance(av_capability, Mapping)
        or av_capability.get("all_batches_ok") is not True
        or float(av_capability.get("a_at_v_reconstruction_max_abs_error", np.inf))
        >= HOOK_ATOL
    ):
        raise RuntimeError("R5a preflight SDPA/eager or A@V reconstruction gate differs")
    calibration = record.get("classifier_calibration")
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("all_pass") is not True
        or calibration.get("n_genes") != PRODUCTION_N
        or calibration.get("n_draws") != PRODUCTION_RANDOMISATION
        or calibration.get("n_trials_argument_retained_for_stack_compatibility")
        != PRODUCTION_CALIBRATION_TRIALS
        or calibration.get("fixed_sequence_alpha") != STAGE_ALPHA
        or calibration.get("alpha_fragmented") is not False
        or set(calibration.get("gates", {})) != set(CORE_PATH_ORDER)
        or canonical_sha256(calibration.get("registry", {}))
        != calibration.get("registry_sha256")
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
    upstream_chain: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    capstone_handoff: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind target execution to the exact earlier nonentry instrument preflight."""

    record = preflight.get("record")
    if not isinstance(record, Mapping):
        raise RuntimeError("R5a preflight record is missing")
    stored_checkpoint = record.get("checkpoint", {})
    checkpoint_matches = (
        stored_checkpoint.get("aggregate_sha256")
        == checkpoint.get("aggregate_sha256")
        and stored_checkpoint.get("ordered_bundle")
        == checkpoint.get("ordered_bundle")
    )
    checks = {
        "validation_preflight_nonentry": record.get("stage") == "preflight"
        and record.get("scientific_entry_authorized") is False
        and record.get("r5a_opened_chr8_chr9") is False,
        "instrument_fingerprint": record.get("instrument_fingerprint_sha256")
        == authorization["fingerprint"],
        "instrument_binding": record.get("instrument_binding")
        == authorization["binding"],
        "joint_stack": record.get("r5_stack_binding")
        == authorization["r5_stack_binding"]
        and record.get("r5_stack_fingerprint_sha256")
        == authorization["r5_stack_fingerprint_sha256"],
        "upstream_exposure_provenance": record.get("upstream_chain", {}).get(
            "exposure_chain_sha256"
        )
        == upstream_chain.get("exposure_chain_sha256"),
        "r3d_path_a_scientific_entry": upstream_chain.get(
            "r3d_scientific_entry", {}
        ).get("entry", {}).get("classification")
        == "SUPPORTED"
        and upstream_chain.get("r3d_scientific_entry", {}).get("entry", {}).get(
            "path_b_used_to_gate_entry"
        )
        is False,
        "r1c_target_behavior_gate": upstream_chain.get("r1c_package", {})
        .get("behavior_gate", {})
        .get("scientific_gate_pass")
        is True,
        "source_identity": record.get("source_identity") == source_identity,
        "checkpoint_identity": checkpoint_matches,
        "r1c_predecessor_package": record.get("cohort_handoff", {}).get(
            "source_artifact_sha256"
        )
        == capstone_handoff.get("source", {}).get("preflight_artifact", {}).get(
            "sha256"
        )
        and record.get("cohort_handoff", {}).get("source_manifest_sha256")
        == capstone_handoff.get("source", {}).get("preflight_manifest", {}).get(
            "sha256"
        ),
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
    upstream_before: Mapping[str, Any],
    preflight_before: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Rehash every long-lived input before permitting an immutable artifact."""

    source_after = validate_expected_r5_source_identity(args.gtf, args.fasta)
    checkpoint_after = _checkpoint_identity(args.model)
    if canonical_sha256(source_after) != canonical_sha256(source_before):
        raise RuntimeError("R5a GTF/FASTA input changed during the run")
    if canonical_sha256(checkpoint_after) != canonical_sha256(checkpoint_before):
        raise RuntimeError("R5a checkpoint input changed during the run")
    exposure_after = load_exposure_packages(args)
    if exposure_after.get("exposure_chain_sha256") != upstream_before.get(
        "exposure_chain_sha256"
    ):
        raise RuntimeError("R5a authenticated R3 provenance changed during the run")
    package_inputs = upstream_before.get("r1c_package", {}).get(
        "package_inputs", []
    )
    if not isinstance(package_inputs, list) or any(
        not isinstance(identity, Mapping)
        or sha256_file(Path(str(identity.get("path")))) != identity.get("sha256")
        for identity in package_inputs
    ):
        raise RuntimeError("R5a authenticated R1c package changed during the run")
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
        "upstream_package_chain_unchanged": True,
        "bound_preflight_unchanged": preflight_before is not None,
        "instrument_binding_unchanged": True,
        "git_identity_unchanged": True,
    }


def _compact_upstream_chain(chain: Mapping[str, Any]) -> dict[str, Any]:
    r3d = chain["r3d_scientific_entry"]
    r3c = chain["r3c_exclusion_provenance"]
    r1c = chain["r1c_package"]
    handoff = r1c.get("handoff") or {}
    attestation = r1c.get("attestation")
    return {
        "chain_sha256": chain["chain_sha256"],
        "exposure_chain_sha256": chain["exposure_chain_sha256"],
        "r3d_scientific_entry": {
            key: r3d[key]
            for key in (
                "artifact",
                "manifest",
                "generating_commit",
                "publication_commit",
                "entry",
                "checkpoint_and_data",
                "target_gene_count",
                "target_gene_ids_sha256",
                "registry",
                "attempt",
                "completion",
                "package_contract_sha256",
            )
        },
        "r1c_package": {
            "behavior_gate": r1c.get("behavior_gate"),
            "package_inputs": r1c.get("package_inputs"),
            "attestation_sha256": (
                attestation.get("attestation_sha256")
                if isinstance(attestation, Mapping)
                else None
            ),
            "handoff_schema": handoff.get("schema"),
            "handoff_record_sha256": handoff.get("record_sha256"),
            "row_count": handoff.get("row_count"),
        },
        "r3c_exclusion_provenance": {
            key: r3c[key]
            for key in (
                "artifact",
                "manifest",
                "generating_commit",
                "role",
                "scientific_entry_authorized",
                "historical_r3_gene_count",
                "historical_r3_gene_ids_sha256",
                "r3c_gene_count",
                "r3c_gene_ids_sha256",
                "package_contract_sha256",
            )
        },
        "scientific_roles": {
            "r3d_path_a": "prior component premise only",
            "r3d_path_b": "candidate generation only; never gates R5a",
            "r1c": (
                "nonentry validation handoff" if r1c.get("behavior_gate") is None
                else "sealed target masked-acceptor behavior gate"
            ),
            "r3c": "exclusion provenance only; never scientific entry",
        },
    }


def _summarize_family_capability(
    family: str, invariants: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not invariants:
        return {"status": "NOT_TESTED", "all_batches_ok": False}
    if family == "position_margin":
        return {
            "all_batches_ok": all(item["ok"] for item in invariants),
            "full_cell_attention_reconstruction_max_abs": _max_invariant(
                invariants, "full_cell_attention_reconstruction_max_abs"
            ),
            "full_cell_margin_identity_max_abs": _max_invariant(
                invariants, "full_cell_margin_identity_max_abs"
            ),
            "component_swap_involution_max_abs": _max_invariant(
                invariants, "component_swap_involution_max_abs"
            ),
            "component_swap_assignment_max_abs": _max_invariant(
                invariants, "component_swap_assignment_max_abs"
            ),
            "component_swap_outside_max_abs": _max_invariant(
                invariants, "component_swap_outside_max_abs"
            ),
            "fixed_query_assignment_max_abs_error": _max_invariant(
                invariants, "fixed_query_assignment_max_abs_error"
            ),
            "fixed_query_sha256_by_head_and_batch": [
                item["fixed_query_sha256_by_head"] for item in invariants
            ],
            "fixed_query_sha256_by_head_and_cell_and_batch": [
                item["fixed_query_sha256_by_head_and_cell"]
                for item in invariants
            ],
            "shifted_native_query_change_max_abs_by_head_and_cell_and_batch": [
                item[
                    "shifted_native_query_change_max_abs_by_head_and_cell"
                ]
                for item in invariants
            ],
            "decision_query_source": "numerically identical clean query row",
            "shifted_native_query_decision_bearing": False,
            "clean_recipient_for_every_cell": all(
                item["clean_recipient_for_every_cell"] for item in invariants
            ),
            "bridge_reuses_one_clean_V_tensor_for_all_cells": all(
                item["bridge_reuses_one_clean_V_tensor_for_all_cells"]
                for item in invariants
            ),
        }
    if family in {"qk_surgery", "l2h15_qk_secondary"}:
        return {
            "all_batches_ok": all(item["ok"] for item in invariants),
            "exactly_four_logits_replaced": all(
                item["exactly_four_logits_replaced"] for item in invariants
            ),
            "selected_logit_assignment_max_abs_error": _max_invariant(
                invariants, "selected_logit_assignment_max_abs_error"
            ),
            "unselected_logit_max_abs_error": _max_invariant(
                invariants, "unselected_logit_max_abs_error"
            ),
            "fixed_value_max_abs_error": _max_invariant(
                invariants, "fixed_value_max_abs_error"
            ),
            "same_clean_value_tensor_used_for_every_leg": all(
                item["same_clean_value_tensor_used_for_every_leg"]
                for item in invariants
            ),
        }
    if family == "attention_value":
        return {
            "all_batches_ok": all(item["ok"] for item in invariants),
            "a_at_v_reconstruction_max_abs_error": _max_invariant(
                invariants, "a_at_v_reconstruction_max_abs_error"
            ),
            "baseline_inject_max_abs_error": _max_invariant(
                invariants, "baseline_inject_max_abs_error"
            ),
            "value_window_assignment_max_abs_error": _max_invariant(
                invariants, "value_window_assignment_max_abs_error"
            ),
            "value_outside_window_max_abs_error": _max_invariant(
                invariants, "value_outside_window_max_abs_error"
            ),
            "address_source_scope": "complete L3H1 acceptor-query attention row",
            "value_source_scope": "selected four-token window; recipient values elsewhere",
        }
    raise ValueError(f"unknown capability family: {family}")


def _execute_family_measurements(
    adapter,
    task: CouplingTask,
    rows: Sequence[Any],
    *,
    family: str,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[float]]]:
    """Execute one query-fixed paired/pseudo family and replay its hook invariants."""

    accumulator: dict[str, Any] = {}
    invariants: list[dict[str, Any]] = []
    restore_lift: dict[str, list[float]] = {"partner": [], "pseudo": []}
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            readout = [int(row.acceptor_g) for row in batch_rows]
            paired = [int(row.example.donor_pos) for row in batch_rows]
            pseudo = [int(row.pseudo_donor) for row in batch_rows]
            clean = task.encode([row.clean_sequence for row in batch_rows], readout)
            partner = task.encode(
                [row.paired_corrupt_sequence for row in batch_rows], readout
            )
            pseudo_encoded = task.encode(
                [row.pseudo_corrupt_sequence for row in batch_rows], readout
            )
            if family == "position_margin":
                values, check = run_position_shift_batch(
                    adapter, task, clean, readout, paired, pseudo
                )
                key = "position_shift"
            elif family in {"qk_surgery", "l2h15_qk_secondary"}:
                values, check = run_qk_batch(
                    adapter,
                    task,
                    clean,
                    partner,
                    pseudo_encoded,
                    readout,
                    paired,
                    pseudo,
                    arms=(QK_ARMS if family == "qk_surgery" else L2_QK_ARMS),
                )
                key = "qk_surgery"
            elif family == "attention_value":
                values, check = run_av_batch(
                    adapter,
                    task,
                    clean,
                    partner,
                    pseudo_encoded,
                    readout,
                    paired,
                    pseudo,
                )
                key = "attention_value"
                for leg in ("partner", "pseudo"):
                    restore_lift[leg].extend(
                        np.asarray(
                            check["clean_L3H1_same_example_restore_lift"][leg],
                            dtype=float,
                        ).tolist()
                    )
            else:
                raise RuntimeError(f"unknown reached R5a family: {family}")
            _accumulate_tree(accumulator, values)
            invariants.append(check)
    measurements = _finalize_tree(accumulator)
    wrapper = {key: measurements}
    matrix = build_family_matrices(wrapper, families=(family,))[family]
    capability = _summarize_family_capability(family, invariants)
    return {key: measurements}, capability, restore_lift


def same_row_common_leg_replay(
    rows: Sequence[Any],
    native: Mapping[str, Sequence[float]],
    common_leg_source: Mapping[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Bound R1c SDPA common legs to R5a eager execution on the exact same rows."""

    if stage not in {"preflight", "target"}:
        raise ValueError("R5a common-leg replay stage differs")
    common = common_leg_source.get(
        "common_leg_rows" if stage == "preflight" else "rows"
    )
    if not isinstance(common, list) or len(common) != EXACT_ROWS:
        raise RuntimeError("R1c preflight common-leg handoff is incomplete")
    row_ids = [str(row.example.example_id) for row in rows]
    common_ids = [str(row.get("example_id")) for row in common]
    if row_ids != common_ids:
        raise RuntimeError("R5a preflight common-leg order differs from R1c")
    mapping = {
        "clean": "P_clean",
        "partner": "P_under_D",
        "pseudo": "P_under_N",
    }
    errors = {
        local: float(
            np.max(
                np.abs(
                    np.asarray(native[local], dtype=float)
                    - np.asarray(
                        [row["margin_levels"][source] for row in common],
                        dtype=float,
                    )
                )
            )
        )
        for local, source in mapping.items()
    }
    return {
        "role": "SDPA_TO_EAGER_COMMON_LEG_INSTRUMENT_REPLAY",
        "stage": stage,
        "source_execution": "R1c eager=False (SDPA)",
        "replay_execution": "R5a eager=True (hook-capable)",
        "scientific_entry_authorized": stage == "target",
        "row_order_exact": True,
        "max_abs_error_by_common_leg": errors,
        "maximum_max_abs_error": max(errors.values()),
        "atol": SDPA_EAGER_COMMON_LEG_ATOL,
        "criterion": "maximum_max_abs_error <= atol",
        "pass": max(errors.values()) <= SDPA_EAGER_COMMON_LEG_ATOL,
        "mismatch_disposition": "INSTRUMENT_UNRESOLVED",
        "mismatch_is_scientific_null": False,
    }


def same_row_preflight_margin_replay(
    rows: Sequence[Any],
    native: Mapping[str, Sequence[float]],
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility wrapper for the validation-preflight common-leg replay."""

    return same_row_common_leg_replay(
        rows, native, handoff, stage="preflight"
    )


def run_model_stage(
    args: argparse.Namespace,
    *,
    authorization: Mapping[str, Any],
    rows: Sequence[Any],
    cohort_materialization: Mapping[str, Any],
    cohort_handoff: Mapping[str, Any],
    upstream_chain: Mapping[str, Any],
    bound_preflight: Mapping[str, Any] | None,
    target_receipt: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run reached R5a gates in order; later interventions are never scored after a stop."""

    git_identity = _git_identity_clean()
    if upstream_chain["r3d_scientific_entry"]["entry"]["classification"] != "SUPPORTED":
        raise RuntimeError("R5a requires authenticated R3d Path A support")
    if upstream_chain["r3d_scientific_entry"]["entry"]["path_b_used_to_gate_entry"] is not False:
        raise RuntimeError("R5a cannot use R3d Path B as scientific entry")
    r1c_gate = upstream_chain.get("r1c_package", {}).get("behavior_gate")
    if args.stage == "target" and (
        not isinstance(r1c_gate, Mapping)
        or r1c_gate.get("scientific_gate_pass") is not True
    ):
        raise RuntimeError("R1C_ACCEPTOR_BEHAVIOR_NOT_CONFIRMED: scientific stop before R5a")
    if args.stage == "preflight" and r1c_gate is not None:
        raise RuntimeError("R5a preflight cannot consume the sealed R1c target verdict")
    if args.stage == "target" and target_receipt is None:
        raise RuntimeError("R5a target model access requires a prior biological-spend receipt")
    if args.stage == "preflight" and target_receipt is not None:
        raise RuntimeError("R5a preflight must not carry a target receipt")

    rows = list(rows)
    source_identity = validate_expected_r5_source_identity(args.gtf, args.fasta)
    if (
        len(rows) != EXACT_ROWS
        or cohort_handoff.get("row_count") != EXACT_ROWS
        or cohort_materialization.get("handoff_order_preserved") is not True
        or cohort_materialization.get("rows_filtered_or_reordered") is not False
        or len({str(row.example.gene_id) for row in rows}) < MIN_GENES
    ):
        raise RuntimeError("R5a requires the exact ordered 400-row R1c handoff")

    runtime_policy = validate_r5_runtime_policy(device=args.device)
    checkpoint = _checkpoint_identity(args.model)
    if checkpoint["aggregate_sha256"] != FROZEN_CHECKPOINT_AGGREGATE_SHA256:
        raise RuntimeError("R5a checkpoint bundle differs from its literal pin")
    upstream_checkpoint = upstream_chain["r3d_scientific_entry"][
        "checkpoint_and_data"
    ]["checkpoint_aggregate_sha256"]
    if checkpoint["aggregate_sha256"] != upstream_checkpoint:
        raise RuntimeError("R5a checkpoint differs from authenticated R3d")
    runtime = stable_runtime_identity()
    preflight_context = (
        validate_bound_preflight_context(
            bound_preflight,
            authorization=authorization,
            upstream_chain=upstream_chain,
            source_identity=source_identity,
            checkpoint=checkpoint,
            capstone_handoff=cohort_handoff,
            runtime=runtime,
        )
        if bound_preflight is not None
        else None
    )

    device = pick_device(args.device)
    adapter = load_adapter(args.model, device, eager=True)
    adapter.model.eval()
    task = CouplingTask(adapter, batch=args.batch)
    genes = [str(row.example.gene_id) for row in rows]
    native = {"clean": [], "partner": [], "pseudo": []}
    full_mlm: dict[str, Any] | None = None
    output_projection: dict[str, Any] | None = None
    full_head_noop: dict[str, Any] | None = None

    with torch.no_grad():
        for start in range(0, len(rows), args.batch):
            batch_rows = rows[start : start + args.batch]
            readout = [row.acceptor_g for row in batch_rows]
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

        health = bounded_intervention_health(
            adapter, task, rows, seed=PRODUCTION_SEED
        )

    if args.stage == "preflight":
        calibration = classifier_calibration(
            n_genes=PRODUCTION_N,
            n_draws=PRODUCTION_RANDOMISATION,
            n_trials=PRODUCTION_CALIBRATION_TRIALS,
            seed=PRODUCTION_SEED,
        )
        calibration_origin = "RUN_ON_R1C_VALIDATION_HANDOFF_WITH_SYNTHETIC_PRIMITIVES"
    else:
        assert bound_preflight is not None
        calibration = bound_preflight["record"]["classifier_calibration"]
        calibration_origin = "INHERITED_FROM_LITERAL_R1C_HANDOFF_PREFLIGHT"

    common_leg_source = (
        cohort_handoff
        if args.stage == "preflight"
        else upstream_chain.get("r1c_package", {}).get("common_legs")
    )
    if not isinstance(common_leg_source, Mapping):
        raise RuntimeError("R5a stage lacks the authorized R1c common-leg replay")
    common_leg_replay = same_row_common_leg_replay(
        rows,
        native,
        common_leg_source,
        stage=args.stage,
    )
    base_capability_pass = bool(
        full_mlm
        and full_mlm["ok"]
        and output_projection
        and output_projection["ok"]
        and full_head_noop
        and full_head_noop["ok"]
        and health["core_all_pass"]
        and calibration["all_pass"]
        and common_leg_replay["pass"] is True
    )

    primitive_measurements: dict[str, Any] = {}
    matrices: dict[str, np.ndarray] = {}
    family_capabilities: dict[str, Any] = {}
    family_results: dict[str, Any] = {}
    paths: dict[str, Any] = {}
    execution_log: list[dict[str, Any]] = []
    scientific_stop: str | None = None
    instrument_stop: str | None = None
    position_diagnostics: dict[str, Any] | None = None
    l3_restore_lift = {"partner": [], "pseudo": []}

    if not base_capability_pass:
        instrument_stop = "BASE_CORE_INSTRUMENT_CAPABILITY"
    else:
        for gate_index, path in enumerate(CORE_PATH_ORDER, start=1):
            family = named_path_specs()[path]["family"]
            wrapper, capability, family_restore = _execute_family_measurements(
                adapter,
                task,
                rows,
                family=family,
                batch_size=args.batch,
            )
            primitive_measurements.update(wrapper)
            matrix = build_family_matrices(wrapper, families=(family,))[family]
            matrices[family] = matrix
            family_capabilities[family] = capability
            for leg in l3_restore_lift:
                l3_restore_lift[leg].extend(family_restore[leg])
            execution_log.append(
                {
                    "gate_index": gate_index,
                    "path": path,
                    "family": family,
                    "model_intervention_executed": True,
                    "capability_pass": capability["all_batches_ok"],
                }
            )
            if not capability["all_batches_ok"]:
                instrument_stop = family
                break
            if args.stage == "preflight":
                path_record = {
                    "path": path,
                    "family": family,
                    "status": "PREFLIGHT_CAPABILITY_PASS",
                    "classification": "NOT_SCIENTIFICALLY_CLASSIFIED",
                    "model_intervention_executed": True,
                    "scientific_entry_authorized": False,
                    "fixed_sequence_gate_index": gate_index,
                }
                paths[path] = path_record
                family_results[family] = path_record
                if family == "position_margin":
                    position_diagnostics = position_bridge_diagnostics(wrapper)
                continue
            result = strict_family_inference(
                matrix,
                genes,
                family_row_names()[family],
                family=family,
                n_draws=PRODUCTION_RANDOMISATION,
                seed=PRODUCTION_SEED,
                namespace=f"r5a:{args.stage}:observed",
                min_genes=MIN_GENES,
            )
            family_results[family] = result
            path_record = result["paths"][path]
            path_record["fixed_sequence_gate_index"] = gate_index
            paths[path] = path_record
            if family == "position_margin":
                position_diagnostics = position_bridge_diagnostics(wrapper)
            if path_record["classification"] != "SUPPORTED":
                scientific_stop = path
                break

    secondary_path = SECONDARY_PATH_ORDER[0]
    secondary_family = named_path_specs()[secondary_path]["family"]
    inherited_secondary_ready = (
        True
        if args.stage == "preflight"
        else bool(
            bound_preflight
            and bound_preflight["record"]
            .get("secondary_prospective_eligibility", {})
            .get("eligible_for_target")
            is True
        )
    )
    secondary_ready = bool(health["secondary_all_pass"] and inherited_secondary_ready)
    if instrument_stop is not None or scientific_stop is not None:
        secondary_record = {
            "path": secondary_path,
            "family": secondary_family,
            "status": "NOT_TESTED_DUE_TO_CORE_STOP",
            "classification": "NOT_TESTED",
            "gates_l3_headline": False,
            "historical_r3c_health_used": False,
        }
    elif not secondary_ready:
        secondary_record = {
            "path": secondary_path,
            "family": secondary_family,
            "status": "NOT_TESTED_SECONDARY_HEALTH_OR_PREFLIGHT",
            "classification": "INSTRUMENT_UNRESOLVED",
            "gates_l3_headline": False,
            "historical_r3c_health_used": False,
        }
    else:
        wrapper, secondary_capability, _restore = _execute_family_measurements(
            adapter,
            task,
            rows,
            family=secondary_family,
            batch_size=args.batch,
        )
        primitive_measurements["l2_qk_surgery"] = wrapper["qk_surgery"]
        secondary_matrix = build_family_matrices(
            wrapper, families=(secondary_family,)
        )[secondary_family]
        matrices[secondary_family] = secondary_matrix
        family_capabilities[secondary_family] = secondary_capability
        if not secondary_capability["all_batches_ok"]:
            secondary_record = {
                "path": secondary_path,
                "family": secondary_family,
                "status": "SECONDARY_INSTRUMENT_UNRESOLVED",
                "classification": "INSTRUMENT_UNRESOLVED",
                "gates_l3_headline": False,
                "historical_r3c_health_used": False,
            }
        elif args.stage == "preflight":
            secondary_record = {
                "path": secondary_path,
                "family": secondary_family,
                "status": "PREFLIGHT_CAPABILITY_PASS",
                "classification": "NOT_SCIENTIFICALLY_CLASSIFIED",
                "model_intervention_executed": True,
                "gates_l3_headline": False,
                "historical_r3c_health_used": False,
            }
        else:
            secondary_result = strict_family_inference(
                secondary_matrix,
                genes,
                family_row_names()[secondary_family],
                family=secondary_family,
                n_draws=PRODUCTION_RANDOMISATION,
                seed=PRODUCTION_SEED,
                namespace="r5a:target:observed-secondary",
                min_genes=MIN_GENES,
            )
            family_results[secondary_family] = secondary_result
            secondary_record = secondary_result["paths"][secondary_path]
            secondary_record["gates_l3_headline"] = False
            secondary_record["historical_r3c_health_used"] = False
    paths[secondary_path] = secondary_record
    family_results.setdefault(secondary_family, secondary_record)
    family_capabilities.setdefault(
        secondary_family,
        {
            "status": secondary_record["status"],
            "all_batches_ok": None,
        },
    )

    placeholders = fixed_sequence_stop_records(
        paths,
        execution_log,
        instrument_stop=instrument_stop,
        scientific_stop=scientific_stop,
    )
    for path, placeholder in placeholders.items():
        family = placeholder["family"]
        paths[path] = placeholder
        family_results.setdefault(family, placeholder)
        family_capabilities.setdefault(
            family, {"status": placeholder["status"], "all_batches_ok": None}
        )

    if instrument_stop is not None:
        status = "INSTRUMENT_UNRESOLVED"
    elif scientific_stop is not None:
        status = f"SCIENTIFIC_STOP_AFTER_{scientific_stop}"
    elif args.stage == "preflight":
        status = "PREFLIGHT_PASS"
    else:
        status = "L3H1_DONOR_LOCAL_READER_SUPPORTED"

    inference = (
        None
        if args.stage == "preflight"
        else {
            "families": family_results,
            "paths": paths,
            "fixed_sequence": {
                "order": list(CORE_PATH_ORDER),
                "alpha_each_reached_gate": STAGE_ALPHA,
                "alpha_fragmented": False,
                "scientific_stop": scientific_stop,
                "instrument_stop": instrument_stop,
                "downstream_model_interventions_not_executed_after_stop": True,
                "execution_log": execution_log,
            },
            "secondary_sequence": {
                "order": list(SECONDARY_PATH_ORDER),
                "gates_l3_headline": False,
                "historical_r3c_health_used": False,
                "record": secondary_record,
            },
            "decisions": {
                "reader": (
                    "L3H1_DONOR_LOCAL_READER_SUPPORTED"
                    if status == "L3H1_DONOR_LOCAL_READER_SUPPORTED"
                    else status
                ),
                "opens_r5b": status == "L3H1_DONOR_LOCAL_READER_SUPPORTED",
                "l2h15_role": (
                    "L2H15_UPSTREAM_READER_ROLE_SUPPORTED"
                    if secondary_record.get("classification") == "SUPPORTED"
                    else (
                        "L2H15_ROLE_UNRESOLVED"
                        if secondary_record.get("classification")
                        in {"UNRESOLVED", "INSTRUMENT_UNRESOLVED"}
                        else "L2H15_ROLE_NOT_SUPPORTED"
                    )
                ),
            },
        }
    )

    input_snapshot = verify_input_snapshots_after_run(
        args,
        authorization=authorization,
        git_before=git_identity,
        source_before=source_identity,
        checkpoint_before=checkpoint,
        upstream_before=upstream_chain,
        preflight_before=bound_preflight,
    )
    capability = {
        "full_mlm_replay": full_mlm,
        "attention_output_projection": output_projection,
        "full_head_noop_replay": full_head_noop,
        "paired_noop_intervention_health": health,
        "families": family_capabilities,
        "candidate_outcomes_used_to_define_capability": False,
        "behavioral_endpoint": "masked_acceptor_G_margin_only",
        "donor_readout_constructed": False,
        "all_reached_capabilities_pass": instrument_stop is None,
    }
    av_capability = family_capabilities.get("attention_value", {})
    a_at_v_reconstruction = {
        "role": "EXACT_QUERY_CONTEXT_CAPABILITY",
        "reached_in_current_stage": isinstance(
            av_capability.get("a_at_v_reconstruction_max_abs_error"), (int, float)
        ),
        "max_abs_error": av_capability.get(
            "a_at_v_reconstruction_max_abs_error"
        ),
        "atol": HOOK_ATOL,
        "pass": (
            av_capability.get("all_batches_ok") is True
            if "all_batches_ok" in av_capability
            else None
        ),
        "frozen_preflight_pass": (
            None
            if bound_preflight is None
            else bound_preflight["record"]
            .get("a_at_v_reconstruction", {})
            .get("pass")
        ),
    }
    verified_handoff = upstream_chain["r1c_package"]["verified"]
    construction_field = (
        "preflight_construction_sha256"
        if args.stage == "preflight"
        else "capstone_construction_sha256"
    )
    construction_sha256 = str(verified_handoff[construction_field])
    instrument_gates = {
        "exact_source_identity": True,
        "exact_400_row_handoff_order": bool(
            len(rows) == EXACT_ROWS
            and cohort_materialization.get("handoff_order_preserved") is True
            and cohort_materialization.get("rows_filtered_or_reordered") is False
        ),
        "runtime_policy_mps_without_fallback": runtime_policy["verified"],
        "base_hook_and_decoder_capability": bool(base_capability_pass),
        "classifier_calibration": bool(calibration["all_pass"]),
        "core_paired_noop_intervention_health": bool(health["core_all_pass"]),
        "same_row_validation_replay_or_target_behavior_gate": bool(
            common_leg_replay["pass"]
            and (
                args.stage == "preflight"
                or r1c_gate and r1c_gate.get("scientific_gate_pass") is True
            )
        ),
        "reached_family_capabilities": instrument_stop is None,
        "inputs_rehashed_after_run": bool(
            input_snapshot["checked_after_model_and_inference"]
        ),
        "r3d_path_a_and_stage_appropriate_r1c_handoff_bound": True,
        "r3c_exclusion_provenance_only": upstream_chain[
            "r3c_exclusion_provenance"
        ]["scientific_entry_authorized"]
        is False,
        "immutable_method_and_handoff_pins": True,
        "stage_data_boundary_respected": (
            args.stage == "target"
            or all(str(row.example.chrom).replace("chr", "") in {"11", "16", "19"} for row in rows)
        ),
    }
    all_instrument = all(instrument_gates.values())
    if all_instrument != (instrument_stop is None):
        raise RuntimeError("R5a instrument disposition is internally inconsistent")

    claim_eligible = bool(
        args.stage == "target" and status == "L3H1_DONOR_LOCAL_READER_SUPPORTED"
    )
    entry_contract = {
        "schema": R5_ENTRY_SCHEMA,
        "instrument_gates_pass": bool(all_instrument),
        "claim_eligible": claim_eligible,
        "verdict": status,
        "scientific_entry_order": (
            [
                "authenticated_R3d_Path_A_prior",
                "authenticated_R1c_validation_preflight_nonentry",
                "R5a_instrument_preflight",
            ]
            if args.stage == "preflight"
            else [
                "authenticated_R3d_Path_A_prior",
                "sealed_R1c_chr8_chr9_acceptor_behavior",
                *CORE_PATH_ORDER,
            ]
        ),
        "r3d_path_b_used_to_gate": False,
        "r3c_used_as_scientific_entry": False,
        "acceptor_G_only": True,
        "masked_donor_endpoint_constructed": False,
        "checkpoint_sha256": checkpoint["aggregate_sha256"],
        "target_gene_ids_sha256": verified_handoff[
            "sorted_unique_gene_ids_sha256"
        ],
        "cohort_sha256": verified_handoff["cohort_sha256"],
        "ordered_row_ids_sha256": verified_handoff["ordered_row_ids_sha256"],
        "ordered_gene_ids_sha256": verified_handoff["ordered_gene_ids_sha256"],
        "construction_sha256": construction_sha256,
        "r5_stack_fingerprint_sha256": authorization[
            "r5_stack_fingerprint_sha256"
        ],
        "upstream_package_chain_sha256": upstream_chain["chain_sha256"],
    }
    if args.stage == "target":
        entry_contract.update(
            {
                "capstone_cohort_schema": cohort_handoff["schema"],
                "capstone_row_count": cohort_handoff["row_count"],
                "capstone_cohort_sha256": cohort_handoff["cohort_sha256"],
                "capstone_ordered_row_ids_sha256": cohort_handoff[
                    "ordered_row_ids_sha256"
                ],
                "capstone_ordered_gene_ids_sha256": cohort_handoff[
                    "ordered_gene_ids_sha256"
                ],
                "capstone_sorted_unique_gene_ids_sha256": cohort_handoff[
                    "sorted_unique_gene_ids_sha256"
                ],
                "capstone_construction_sha256": cohort_handoff[
                    "capstone_construction_sha256"
                ],
            }
        )
    source = cohort_handoff.get("source", {})
    cohort_binding = {
        "schema": cohort_handoff["schema"],
        "row_count": cohort_handoff["row_count"],
        "cohort_sha256": cohort_handoff["cohort_sha256"],
        "ordered_row_ids_sha256": cohort_handoff["ordered_row_ids_sha256"],
        "ordered_gene_ids_sha256": cohort_handoff["ordered_gene_ids_sha256"],
        "sorted_unique_gene_ids_sha256": cohort_handoff[
            "sorted_unique_gene_ids_sha256"
        ],
        "construction_sha256": construction_sha256,
        "handoff_order_preserved": cohort_materialization[
            "handoff_order_preserved"
        ],
        "rows_filtered_or_reordered": cohort_materialization[
            "rows_filtered_or_reordered"
        ],
        "scientific_entry_authorized": args.stage == "target",
        "sealed_split_opened": args.stage == "target",
        "source_package_sha256": cohort_handoff["source_package_sha256"],
        "source_artifact_sha256": source.get("artifact", {}).get("sha256"),
        "source_manifest_sha256": source.get("manifest", {}).get("sha256"),
    }
    record = {
        "schema": "gfmcd.r5a.acceptor_address_binding/2",
        "experiment": "R5a_acceptor_address_binding",
        "stage": args.stage,
        "status": status,
        "r5_entry_contract": entry_contract,
        "behavioral_endpoint": "masked_acceptor_G_margin_only",
        "donor_readout_constructed": False,
        "masked_donor_endpoint_constructed": False,
        "scientific_entry_authorized": claim_eligible,
        "instrument_fingerprint_sha256": authorization["fingerprint"],
        "instrument_binding": authorization["binding"],
        "r5_stack_binding": authorization["r5_stack_binding"],
        "r5_stack_fingerprint_sha256": authorization[
            "r5_stack_fingerprint_sha256"
        ],
        "git": git_identity,
        "runtime_identity": runtime,
        "runtime_policy": runtime_policy,
        "checkpoint": checkpoint,
        "source_identity": source_identity,
        "input_snapshot_after_run": input_snapshot,
        "cohort_handoff": cohort_binding,
        "cohort_materialization": cohort_materialization,
        "gene_ids": genes,
        "gene_ids_sha256": canonical_sha256(genes),
        "row_identity": [row.identity_record() for row in rows],
        "construction_sha256": construction_sha256,
        "upstream_chain": _compact_upstream_chain(upstream_chain),
        "target_attempt_receipt": target_receipt,
        "bound_preflight": (
            None
            if bound_preflight is None
            else {
                key: value
                for key, value in bound_preflight.items()
                if key != "record"
            }
        ),
        "preflight_context": preflight_context,
        "native_margin_cells": native,
        "same_row_preflight_margin_replay": (
            common_leg_replay if args.stage == "preflight" else None
        ),
        "same_row_r1c_common_leg_replay": common_leg_replay,
        "r1c_target_behavior_gate": r1c_gate,
        "capability": capability,
        "a_at_v_reconstruction": a_at_v_reconstruction,
        "preflight_capability_sequence": (
            {path: paths[path] for path in ALL_PATH_ORDER}
            if args.stage == "preflight"
            else None
        ),
        "secondary_prospective_eligibility": {
            "head": "L2H15",
            "eligible_for_target": bool(
                args.stage == "preflight"
                and health["secondary_all_pass"]
                and secondary_record.get("status") == "PREFLIGHT_CAPABILITY_PASS"
            ),
            "own_health_gate_pass": bool(health["secondary_all_pass"]),
            "own_exact_qk_capability": secondary_record.get("status"),
            "gates_l3_headline": False,
            "historical_r3c_health_used": False,
        },
        "classifier_calibration": calibration,
        "classifier_calibration_origin": calibration_origin,
        "raw_primitive_measurements": _json_ready(primitive_measurements),
        "position_internal_bridge_diagnostics": position_diagnostics,
        "raw_family_values": {
            family: matrix.tolist() for family, matrix in matrices.items()
        },
        "gene_sufficient_statistics": {
            "schema": "gfmcd.r5a.gene_sufficient_statistics/2",
            "aggregation": "row-level storage; inference aggregates repeated rows by gene",
            "ordered_gene_ids": genes,
            "repeated_genes_allowed": True,
            "family_matrix_sha256": {
                family: _array_sha256(matrix)
                for family, matrix in matrices.items()
            },
            "family_shapes": {
                family: list(matrix.shape)
                for family, matrix in matrices.items()
            },
        },
        "family_row_names": family_row_names(),
        "family_execution_status": {
            family: family_results[family].get(
                "status", "EXECUTED_AND_CLASSIFIED"
            )
            for family in FAMILY_ORDER
        },
        "observed_scientific_inference": inference,
        "inference": inference,
        "inference_sha256": None if inference is None else canonical_sha256(inference),
        "instrument_gates": instrument_gates,
        "all_instrument_gates_pass": all_instrument,
        "r5a_opened_chr8_chr9": args.stage == "target",
        "prior_sealed_r1c_chr8_chr9_result_consumed": args.stage == "target",
    }
    if l3_restore_lift["partner"]:
        record["same_example_l3_restore_diagnostic"] = {
            leg: values for leg, values in l3_restore_lift.items()
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
    parser.add_argument("--device", default="mps")
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--r3d-artifact")
    parser.add_argument("--r3d-manifest")
    parser.add_argument("--r3d-registry")
    parser.add_argument("--r3d-registry-manifest")
    parser.add_argument("--r3d-attempt")
    parser.add_argument("--r3d-completion")
    parser.add_argument("--r3c-artifact")
    parser.add_argument("--r3c-manifest")
    parser.add_argument("--r1c-preflight-artifact")
    parser.add_argument("--r1c-preflight-manifest")
    parser.add_argument("--r1c-target-artifact")
    parser.add_argument("--r1c-target-manifest")
    parser.add_argument("--r1c-target-attempt")
    parser.add_argument("--r1c-target-completion")
    parser.add_argument("--preflight-artifact")
    parser.add_argument("--out")
    return parser.parse_args(argv)


def _output_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None, Path | None]:
    output = Path(args.out).expanduser().resolve() if args.out else None
    manifest = output.with_suffix(".manifest.json") if output is not None else None
    registry = None
    if output is not None and any(
        path is not None and path.exists() for path in (output, manifest, registry)
    ):
        raise SystemExit("R5a outputs are immutable; choose a fresh path")
    return output, manifest, registry


def _upstream_manifest_inputs(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    upstream = record["upstream_chain"]
    r3d = upstream["r3d_scientific_entry"]
    r1c = upstream["r1c_package"]
    r3c = upstream["r3c_exclusion_provenance"]
    values = [
        r3d["artifact"],
        r3d["manifest"],
        r3d["registry"]["artifact"],
        r3d["registry"]["manifest"],
        r3d["attempt"],
        r3d["completion"],
        r3c["artifact"],
        r3c["manifest"],
        *r1c["package_inputs"],
    ]
    if any(not isinstance(value, Mapping) for value in values):
        raise RuntimeError("R5a upstream manifest input identity is incomplete")
    return [dict(value) for value in values]


def _model_free_exposure_manifest_inputs(
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    exposed = record["authenticated_exposure_chain"]
    keys = (
        "r3d_artifact",
        "r3d_manifest",
        "r3d_registry_artifact",
        "r3d_registry_manifest",
        "r3d_attempt",
        "r3d_completion",
        "r3c_artifact",
        "r3c_manifest",
    )
    values = [exposed.get(key) for key in keys]
    if any(not isinstance(value, Mapping) for value in values):
        raise RuntimeError("R5a model-free exposure manifest registry is incomplete")
    return [dict(value) for value in values]


def _target_receipt_manifest_input(record: Mapping[str, Any]) -> dict[str, Any] | None:
    receipt = record.get("target_attempt_receipt")
    if receipt is None:
        return None
    if not isinstance(receipt, Mapping) or not isinstance(
        receipt.get("receipt_path"), str
    ):
        raise RuntimeError("R5a target attempt receipt identity is incomplete")
    path = Path(receipt["receipt_path"]).expanduser().resolve(strict=True)
    persisted = json.loads(path.read_text())
    if persisted != {key: value for key, value in receipt.items() if key != "receipt_path"}:
        raise RuntimeError("R5a target attempt receipt bytes drifted before publication")
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _append_joint_stack_inputs(
    manifest: ExperimentManifest, authorization: Mapping[str, Any]
) -> None:
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
    existing = {
        Path(str(item["path"])).expanduser().resolve()
        for item in manifest.inputs
        if isinstance(item, Mapping) and "path" in item
    }
    for relative in (*closure, *protocols):
        source = (REPO / str(relative)).resolve()
        try:
            source.relative_to(REPO.resolve())
        except ValueError as exc:
            raise RuntimeError("R5a joint stack source escapes the repository") from exc
        if not source.is_file():
            raise RuntimeError(f"R5a joint stack source is missing: {relative}")
        if source not in existing:
            manifest.inputs.append(
                {
                    "path": str(source),
                    "size_bytes": source.stat().st_size,
                    "sha256": sha256_file(source),
                }
            )
            existing.add(source)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if (
        args.n != PRODUCTION_N
        or args.seed != PRODUCTION_SEED
        or args.draws != PRODUCTION_RANDOMISATION
    ):
        raise SystemExit("R5a production n/seed/bootstrap parameters are fixed")
    if args.batch != PRODUCTION_BATCH:
        raise SystemExit(f"R5a production batch is fixed at {PRODUCTION_BATCH}")

    authorization: Mapping[str, Any] | None = None
    upstream_chain: Mapping[str, Any] | None = None
    bound_preflight: Mapping[str, Any] | None = None
    target_receipt: Mapping[str, Any] | None = None

    # Fail-closed seal validation is first.  In target mode, exact upstream packages and
    # the bound preflight are then authenticated before the irreversible receipt, which
    # itself precedes output, cohort, checkpoint, or model inspection.
    if args.mode == "model":
        if args.out is None:
            raise SystemExit("R5a model mode requires --out")
        try:
            authorization = authorize_model_scoring(args.stage)
        except SealError as exc:
            label = "target" if args.stage == "target" else "model"
            raise SystemExit(f"R5a {label} authorization failed: {exc}") from exc
        # Scientific/prior order is executable order: authenticate exact R3d Path A
        # and R3c nonentry provenance before touching either R1c handoff.
        exposure_chain = load_exposure_packages(args)
        if args.stage == "preflight":
            r1c_package = load_authenticated_r1c_preflight_handoff(
                args.r1c_preflight_artifact,
                args.r1c_preflight_manifest,
            )
            r1c_package["behavior_gate"] = None
            r1c_package["attestation"] = None
        else:
            r1c_package = load_authenticated_r1c_target_handoff(
                args.r1c_target_artifact,
                args.r1c_target_manifest,
                args.r1c_target_attempt,
                args.r1c_target_completion,
                args.r1c_preflight_artifact,
                args.r1c_preflight_manifest,
            )
            if not r1c_package["behavior_gate"]["scientific_gate_pass"]:
                raise SystemExit(
                    "R5a scientific stop: authenticated R1c target behavior was not "
                    f"confirmed ({r1c_package['behavior_gate']['verdict']}); no R5a "
                    "target receipt or reader intervention was opened"
                )
        upstream_chain = load_upstream_chain(exposure_chain, r1c_package)
        if args.stage == "target":
            bound_preflight = _verify_bound_preflight(
                args.preflight_artifact,
                authorization["fingerprint"],
                stack_binding=authorization["r5_stack_binding"],
                stack_fingerprint=authorization["r5_stack_fingerprint_sha256"],
            )
            assert FROZEN_CHECKPOINT_AGGREGATE_SHA256 is not None
            assert r1c_package["handoff"] is not None
            identity = build_r5a_target_identity(
                r1c_package["handoff"],
                checkpoint_sha256=FROZEN_CHECKPOINT_AGGREGATE_SHA256,
            )
            try:
                target_receipt = reserve_r5_target_attempt(
                    repo=REPO,
                    identity=identity,
                    attempt_context={
                        "r5_stack_fingerprint_sha256": authorization[
                            "r5_stack_fingerprint_sha256"
                        ],
                        "stage_preflight_artifact_sha256": bound_preflight["sha256"],
                        "stage_preflight_manifest_sha256": bound_preflight[
                            "manifest_sha256"
                        ],
                        "upstream_package_chain_sha256": upstream_chain[
                            "chain_sha256"
                        ],
                    },
                )
            except R5TargetSpentError as exc:
                raise SystemExit(f"R5a target is already spent: {exc}") from exc
            # Outcome-bearing common legs are authorized only after this downstream
            # stage's irreversible receipt.  They memorialize R1c SDPA execution for
            # the exact same-row eager replay before R5a reader interpretation.
            upstream_chain["r1c_package"]["common_legs"] = (
                load_r1c_verified_target_common_legs(
                    args.r1c_target_artifact,
                    args.r1c_target_manifest,
                    args.r1c_target_attempt,
                    args.r1c_target_completion,
                    args.r1c_preflight_artifact,
                    args.r1c_preflight_manifest,
                )
            )

    output, manifest_path, _registry_path = _output_paths(args)

    if args.mode == "model-free-audit":
        record = model_free_audit(args)
        payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8")
        if output is None:
            print(payload.decode("utf-8"), end="")
            return
        assert manifest_path is not None
        manifest = ExperimentManifest.create(
            "splicebert_acceptor_coupling:r5a_model_free_audit",
            list(sys.argv if argv is None else [str(RUNNER), *argv]),
            repo=REPO,
            status="PROSPECTIVE_INSTRUMENT_AUDIT",
            config={
                "exact_handoff_rows": EXACT_ROWS,
                "preflight_cohort_schema": R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
                "target_cohort_schema": R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
                "model_loaded": False,
                "genome_opened": False,
                "r5a_opened_chr8_chr9": False,
            },
            seeds={"instrument": PRODUCTION_SEED},
            notes=[
                "Prospective method binding only; no genome, model, or chr8/chr9 accessed.",
                "R1c supplies exact ordered 400-row preflight and target handoffs.",
            ],
        )
        manifest.inputs = []
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

    assert output is not None and manifest_path is not None
    assert authorization is not None and upstream_chain is not None
    handoff = upstream_chain["r1c_package"]["handoff"]
    assert isinstance(handoff, Mapping)
    rows, cohort_materialization, _source_identity = materialize_authenticated_cohort(
        args.gtf,
        args.fasta,
        stage=args.stage,
        handoff=handoff,
    )
    record, git_identity = run_model_stage(
        args,
        authorization=authorization,
        rows=rows,
        cohort_materialization=cohort_materialization,
        cohort_handoff=handoff,
        upstream_chain=upstream_chain,
        bound_preflight=bound_preflight,
        target_receipt=target_receipt,
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
            "n_bootstrap": PRODUCTION_RANDOMISATION,
            "bootstrap_estimators": ["mean", "trimmed_mean_10pct"],
            "bootstrap_exact": False,
            "exact_handoff_rows": EXACT_ROWS,
            "cohort_handoff_schema": record["cohort_handoff"]["schema"],
            "instrument_fingerprint_sha256": authorization["fingerprint"],
            "r5_stack_fingerprint_sha256": authorization[
                "r5_stack_fingerprint_sha256"
            ],
            "checkpoint_aggregate_sha256": record["checkpoint"][
                "aggregate_sha256"
            ],
            "construction_sha256": record["construction_sha256"],
            "fixed_sequence_alpha": STAGE_ALPHA,
            "alpha_fragmented": False,
            "family_row_order_sha256": {
                family: canonical_sha256(rows)
                for family, rows in family_row_names().items()
            },
            "path_registry_sha256": canonical_sha256(named_path_specs()),
        },
        seeds={"bootstrap": PRODUCTION_SEED, "calibration": PRODUCTION_SEED},
        notes=[
            "Every decision-bearing endpoint is a masked-acceptor-G margin intervention.",
            "QK logits and attention probabilities are signed diagnostics in separate units.",
            "R5a uses a fixed sequence and never executes a downstream intervention after a failed gate.",
            "R3d Path A is prior component evidence; Path B does not gate. R3c is provenance only.",
            "Preflight uses the exact R1c validation handoff without scientific entry; target uses the exact sealed R1c 400-row handoff.",
            "L2H15 is a separate secondary reader-role family and never gates the L3H1 headline.",
            "No masked-donor endpoint is constructed.",
        ],
    )
    manifest.git_commit = git_identity["commit"]
    manifest.git_dirty = False
    source_paths = {
        "gtf": Path(args.gtf).expanduser().resolve(),
        "fasta": Path(args.fasta).expanduser().resolve(),
        "fasta_fai": Path(f"{Path(args.fasta).expanduser()}.fai").resolve(),
    }
    manifest.inputs = [
        {"path": str(source_paths[key]), **identity}
        for key, identity in record["source_identity"].items()
    ]
    manifest.inputs.extend(_upstream_manifest_inputs(record))
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
    receipt_input = _target_receipt_manifest_input(record)
    if receipt_input is not None:
        manifest.inputs.append(receipt_input)
    _append_joint_stack_inputs(manifest, authorization)

    result_identity = {
        "size_bytes": len(result_bytes),
        "sha256": hashlib.sha256(result_bytes).hexdigest(),
    }
    manifest.outputs = [{"path": str(output), **result_identity}]
    manifest_bytes = (
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    manifest_identity = {
        "size_bytes": len(manifest_bytes),
        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    atomic_write_new(output, result_bytes)
    atomic_write_new(manifest_path, manifest_bytes)

    completion = None
    if args.stage == "target":
        assert target_receipt is not None
        completion = complete_r5_target_attempt(
            receipt=target_receipt,
            result_identity=result_identity,
            manifest_identity=manifest_identity,
            official_disposition=record["status"],
        )
    suffix = (
        ""
        if completion is None
        else f" completion={completion['completion_path']}"
    )
    print(f"R5a status={record['status']} -> {output}{suffix}")


if __name__ == "__main__":
    main()
