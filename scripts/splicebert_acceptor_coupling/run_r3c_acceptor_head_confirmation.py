#!/usr/bin/env python
"""R3c v2: acceptor-only confirmation of fixed retrieval and delivery head candidates.

This instrument never constructs a donor readout.  Every model endpoint masks the G in
the downstream intron-mate acceptor (``next_acceptor_pos - 1``), then compares the paired
upstream donor ``GT -> AC`` edit with the same edit at a distance-calipered non-partner
donor motif.  Component interventions are whole-head within-example token-mean ablation
and same-example corrupt-to-clean activation noising.

The v2 successor makes target health eligibility monotone: a target path must have passed
both the reviewed bucket-0 health gate and its own current-bucket health gate.  A target
invocation exits before inspecting user data paths while any literal seal pin below is absent.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import AbstractContextManager
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.circuits.hf_bert import (  # noqa: E402
    _merge_context,
    _split_context,
    capture_head_outputs,
    edit_heads,
    patch_heads,
)
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA,
    DEFAULT_GTF,
    DEFAULT_MODEL,
    CouplingTask,
    load_adapter,
    load_examples_unscored,
    matched_decoy_acceptor,
    matched_decoy_donor,
    noop,
    pick_device,
    substitute,
)
from gfmcd.splicing.coupling import (  # noqa: E402
    CouplingConfig,
    MOTIF_CLEAN_DONOR_ALT,
    planted_donor_motifs,
)
from gfmcd.splicing.paired_component import (  # noqa: E402
    boundary_sign_flip_tests,
    gene_sign_flip,
    holm_adjust_named,
    intersection_union_pvalue,
    one_sided_binomial_bounds,
    simultaneous_family_decision,
    strict_interval_decision,
)
from gfmcd.splicing.reverse_coupling import (  # noqa: E402
    acceptor_readout,
    donor_alt_admissible,
)


RUNNER = Path(__file__).resolve()
PROTOCOL = (
    REPO
    / "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R3C_ACCEPTOR_HEAD_CONFIRMATION.md"
)
STAT_CORE = REPO / "src/gfmcd/splicing/paired_component.py"
HOOK_CORE = REPO / "src/gfmcd/circuits/hf_bert.py"
ASSAY_CORE = REPO / "src/gfmcd/splicing/reverse_coupling.py"
CIRCUIT_CORE = REPO / "src/gfmcd/splicing/circuit.py"
OTHER_BOUND_SOURCES = (
    REPO / "src/gfmcd/__init__.py",
    REPO / "src/gfmcd/adapters/__init__.py",
    REPO / "src/gfmcd/adapters/base.py",
    REPO / "src/gfmcd/adapters/splicebert.py",
    REPO / "src/gfmcd/circuits/__init__.py",
    REPO / "src/gfmcd/circuits/guards.py",
    REPO / "src/gfmcd/splicing/__init__.py",
    REPO / "src/gfmcd/splicing/candidates.py",
    REPO / "src/gfmcd/splicing/coupling.py",
    REPO / "src/gfmcd/splicing/dataset.py",
    REPO / "src/gfmcd/splicing/gtf.py",
    REPO / "src/gfmcd/splicing/io.py",
    REPO / "src/gfmcd/splicing/schema.py",
    REPO / "src/gfmcd/provenance.py",
    REPO / "pyproject.toml",
    REPO / "uv.lock",
)
TARGET_ATTEMPT_LEDGER = (
    REPO
    / "results/splicebert_acceptor_coupling/target_attempts/"
    "r3c_acceptor_head_confirmation"
)

RESULT_SCHEMA = "gfmcd.r3c.acceptor_head_confirmation/2"
INSTRUMENT_REVISION = "r3c-v2-monotone-health-authorization"

# These values remain literal ``None`` until a reviewed freeze.  Do not replace them from
# a development run.  The normalized runner hash excludes only their assignment values.
FROZEN_PROTOCOL_SHA256: str | None = "4968e6a04175818a8585cd835026503ef63cc277996dde8dc32c6da55ed8a234"
FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = "8050ec7bce63e541004d39157b1f828a257004783cab036a6b5cbe248a0740d4"
FROZEN_PREFLIGHT_ARTIFACT_SHA256: str | None = None

PREFLIGHT_N = 500
TARGET_N = 500
PRODUCTION_BATCH = 32
PRODUCTION_SEED = 0
PRODUCTION_RANDOMISATION = 4000
PRODUCTION_TRIALS = 200
PREFLIGHT_POWER_FLOOR = 0.90
MAX_SURFACE_FALSE_POSITIVES = 4
MIN_PATH_PLANT_SUCCESSES = 188
SESOI = 0.05
FAMILY_ALPHA = 0.05
MIN_GENES = 30
PLANT_DELTA = 0.40
DECOY_TOL = 40

HISTORICAL_TAG = "r1b-d0"
HISTORICAL_SCREEN_N = 1000
HISTORICAL_EXPECTED_ROWS = 974
HISTORICAL_EXPECTED_GENES = 645
HISTORICAL_EXPECTED_EXAMPLE_ID_SHA256 = (
    "161dff038d747dbaa7b91a96d24b34d45f55de5482b5140410641354e46d2a8e"
)
CURRENT_TAG = "r3c-acceptor-head-confirmation-v1"
GENE_BUCKET_SALT = "r3b-gene"
TARGET_HOLDOUT_ID = "r3c-acceptor-head-confirmation-val-bucket1-first500"
HISTORICAL_V1_HEALTH_SUPPRESSED_PATHS = ("L2H15_mean_component",)
VALIDATION_CHROMOSOMES = tuple(CouplingConfig().val_chromosomes)
HISTORICAL_EXCLUSION_RULE = (
    "exclude every gene represented by a row among the historical r1b-d0 first 1000 "
    "validation rows for which both legacy matched decoys exist"
)

# Independent, model-free full-GTF census identity.  ``load_cohort`` recomputes every value
# and aborts on drift; these are not capacity assumptions used to truncate the census.
CURRENT_EXPECTED_CENSUS_IDENTITY = {
    "screened_rows": 6064,
    "historical_excluded_gene_sha256": (
        "0964c06136fd9a8fa9251ad13a2ca4d45e3f0c8cea45323dbdc1fe2092b8fda0"
    ),
    "eligible_rows_by_bucket": {"0": 1010, "1": 1058},
    "eligible_genes_by_bucket": {"0": 417, "1": 443},
    "fixed_selection_by_bucket": {
        "0": {
            "rows": 500,
            "genes": 284,
            "ordered_rows_sha256": (
                "c4b7d17a14dc73b1326a2135cf3a8c49c46d5d78fa894f4879b3c574d4df07e2"
            ),
            "gene_ids_sha256": (
                "8745477f1fe458173774a1a905d569574ed917169621cfbfacbd7b8a65746d6d"
            ),
        },
        "1": {
            "rows": 500,
            "genes": 304,
            "ordered_rows_sha256": (
                "02d4c776d911f261bdd513e3c993a18cab8373e9d242b22ccbdf2fd90f448f8f"
            ),
            "gene_ids_sha256": (
                "484b2c4a11f47c608ea51bcaf2a4234d3197a56f6a10ffb38be9706e5fbd2c5e"
            ),
        },
    },
}

HEALTH_FLOOR = 0.50
HEALTH_MAX_EXAMPLES = 16
HEALTH_PROBES_PER_EXAMPLE = 4
HOOK_SELF_PATCH_ATOL = 1e-5
HOOK_BATCH_INVARIANCE_ATOL = 1e-5
HOOK_RESTORATION_PARTNER_FLOOR = 0.20
HOOK_BATCH_INVARIANCE_EXAMPLES = 16
EXPECTED_PREFLIGHT_MODEL_FORWARD_CALLS = 1110
PREFLIGHT_COST_BENCHMARK = {
    "status": "NON_GATE_LINEAR_EXTRAPOLATION_BEFORE_BUCKET0",
    "date": "2026-08-15",
    "synthetic_shape": {"primitive_rows": 40, "examples": 500, "genes": 284},
    "benchmark_draws": 400,
    "benchmark_trials": 2,
    "benchmark_classifier_calls": 58,
    "three_run_median_wall_seconds": 0.40049979113973677,
    "production_linear_calibration_seconds": 400.4997911397368,
    "production_linear_calibration_minutes": 6.674996518995613,
    "serialized_bytes_trials_1": 213173,
    "serialized_bytes_trials_2": 357697,
    "production_linear_calibration_bytes": 28973449,
    "expected_model_forward_calls_before_calibration": EXPECTED_PREFLIGHT_MODEL_FORWARD_CALLS,
    "scope": "calibration only for timing/bytes; model execution time excluded",
}

MEAN_MODE = "within_example_token_mean_ablation"
NOISING_MODE = "activation_noising"
MODES = (MEAN_MODE, NOISING_MODE)
PRIMITIVE_ENDPOINTS = ("partner", "pseudo")
ENDPOINTS = ("partner", "pseudo", "specific")

# Frozen order.  No MLP appears in this experiment.
HEAD_COMPONENTS = (
    ("head", 3, 1),
    ("head", 2, 15),
    ("head", 1, 12),
    ("head", 4, 11),
    ("head", 4, 9),
    ("head", 4, 7),
    ("head", 4, 0),
    ("head", 4, 2),
    ("head", 5, 11),
    ("head", 0, 2),
)
COMPONENTS = HEAD_COMPONENTS
DELIVERY_CANDIDATES = ("L4H11", "L4H9")
DONOR_ROUTE_COMPARATORS = ("L4H0", "L4H2", "L5H11")
DELIVERY_GAP_ENDPOINTS = ("partner", "specific")

PATH_NAMES = (
    "L3H1_reused_component",
    "L2H15_mean_component",
    "L1H12_mean_component",
    "L4H11_noising_component",
    "L4H9_noising_component",
    "L4H7_noising_component",
    "distinct_acceptor_route_candidate",
)

PREFLIGHT_GATE_NAMES = (
    "minimum_30_independent_genes",
    "exact_full_gtf_census_identity",
    "native_paired_donor_positive_control",
    "matched_pseudo_donor_negative_control",
    "single_acceptor_G_readout_and_GT_to_AC_construction",
    "zero_planted_donor_motifs",
    "bounded_identical_policy_health_executed",
    "exact_62_row_family",
    "analytic_path_fwer_and_boundary_face_diagnostics",
    "all_health_eligible_path_plants_powered",
    "bound_bucket0_live_hook_capability",
)


class TargetSealError(RuntimeError):
    """Raised before target data paths or model outcomes can be opened."""


def component_name(component: tuple[str, int, int | None]) -> str:
    kind, layer, head = component
    if kind != "head" or head is None:
        raise ValueError("R3c accepts attention heads only")
    return f"L{layer}H{head}"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    shape = ",".join(str(value) for value in array.shape)
    digest = hashlib.sha256()
    digest.update(f"float64|{shape}|".encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def primitive_row_name(mode: str, head: str, endpoint: str) -> str:
    return f"{mode}|{head}|{endpoint}"


def family_row_name(mode: str, head: str, endpoint: str) -> str:
    return f"{mode}|{head}|{endpoint}"


def delivery_gap_row_name(endpoint: str) -> str:
    return f"{NOISING_MODE}|delivery_gap|{endpoint}"


def primitive_row_names(
    components: Sequence[tuple[str, int, int | None]] = COMPONENTS,
) -> list[str]:
    return [
        primitive_row_name(mode, component_name(component), endpoint)
        for mode in MODES
        for component in components
        for endpoint in PRIMITIVE_ENDPOINTS
    ]


def family_row_names(
    components: Sequence[tuple[str, int, int | None]] = COMPONENTS,
) -> list[str]:
    rows = [
        family_row_name(mode, component_name(component), endpoint)
        for mode in MODES
        for component in components
        for endpoint in ENDPOINTS
    ]
    rows.extend(delivery_gap_row_name(endpoint) for endpoint in DELIVERY_GAP_ENDPOINTS)
    return rows


def _component_requirements(mode: str, head: str) -> dict[str, str]:
    return {
        family_row_name(mode, head, "partner"): "MATERIAL_POSITIVE",
        family_row_name(mode, head, "pseudo"): "EQUIVALENT_WITHIN_SESOI",
        family_row_name(mode, head, "specific"): "MATERIAL_POSITIVE",
    }


def named_path_specs() -> dict[str, dict[str, Any]]:
    """Return the exact seven path-level claims in their frozen Holm order."""
    l3h1 = {
        **_component_requirements(MEAN_MODE, "L3H1"),
        **_component_requirements(NOISING_MODE, "L3H1"),
    }
    h11 = _component_requirements(NOISING_MODE, "L4H11")
    h9 = _component_requirements(NOISING_MODE, "L4H9")
    distinct = {
        **h11,
        **h9,
        delivery_gap_row_name("partner"): "MATERIAL_POSITIVE",
        delivery_gap_row_name("specific"): "MATERIAL_POSITIVE",
    }
    for comparator in DONOR_ROUTE_COMPARATORS:
        for endpoint in ENDPOINTS:
            distinct[family_row_name(NOISING_MODE, comparator, endpoint)] = (
                "EQUIVALENT_WITHIN_SESOI"
            )
    specs = {
        "L3H1_reused_component": {
            "requirements": l3h1,
            "tier": "PRIMARY_REUSE_COMPONENT_CONFIRMATION",
        },
        "L2H15_mean_component": {
            "requirements": _component_requirements(MEAN_MODE, "L2H15"),
            "tier": "PRIMARY_RETRIEVAL_CANDIDATE_CONFIRMATION",
        },
        "L1H12_mean_component": {
            "requirements": _component_requirements(MEAN_MODE, "L1H12"),
            "tier": "SECONDARY_FIXED_COMPONENT_CONFIRMATION",
        },
        "L4H11_noising_component": {
            "requirements": h11,
            "tier": "PRIMARY_ACCEPTOR_ROUTE_CANDIDATE_CONFIRMATION",
        },
        "L4H9_noising_component": {
            "requirements": h9,
            "tier": "PRIMARY_ACCEPTOR_ROUTE_CANDIDATE_CONFIRMATION",
        },
        "L4H7_noising_component": {
            "requirements": _component_requirements(NOISING_MODE, "L4H7"),
            "tier": "SECONDARY_FIXED_COMPONENT_CONFIRMATION",
        },
        "distinct_acceptor_route_candidate": {
            "requirements": distinct,
            "tier": "PRIMARY_DISTINCT_ROUTE_PATTERN_CONFIRMATION",
        },
    }
    if tuple(specs) != PATH_NAMES:
        raise RuntimeError("R3c named-path order drifted")
    return specs


def build_primitive_matrix(
    measurements: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    components: Sequence[tuple[str, int, int | None]] = COMPONENTS,
) -> tuple[np.ndarray, list[str]]:
    values: list[np.ndarray] = []
    names = primitive_row_names(components)
    for mode in MODES:
        for component in components:
            head = component_name(component)
            for endpoint in PRIMITIVE_ENDPOINTS:
                values.append(np.asarray(measurements[mode][head][endpoint], dtype=float))
    if not values:
        raise ValueError("primitive family cannot be empty")
    family = np.vstack(values)
    if family.shape[0] != len(names) or not np.isfinite(family).all():
        raise ValueError("primitive values do not match the frozen finite row family")
    return family, names


def endpoint_vectors(partner: np.ndarray, pseudo: np.ndarray) -> dict[str, np.ndarray]:
    """Return the two primitives and their sole derived specificity contrast."""
    partner_values = np.asarray(partner, dtype=float)
    pseudo_values = np.asarray(pseudo, dtype=float)
    if (
        partner_values.ndim != 1
        or pseudo_values.shape != partner_values.shape
        or not np.isfinite(partner_values).all()
        or not np.isfinite(pseudo_values).all()
    ):
        raise ValueError("partner and pseudo must be aligned finite vectors")
    return {
        "partner": partner_values,
        "pseudo": pseudo_values,
        "specific": partner_values - pseudo_values,
    }


def mean_component_endpoint_drops(
    native_partner: np.ndarray,
    native_pseudo: np.ndarray,
    intervened_partner: np.ndarray,
    intervened_pseudo: np.ndarray,
) -> dict[str, np.ndarray]:
    """Mean-ablation effect: native coupling minus intervened coupling."""
    return endpoint_vectors(
        np.asarray(native_partner, dtype=float)
        - np.asarray(intervened_partner, dtype=float),
        np.asarray(native_pseudo, dtype=float)
        - np.asarray(intervened_pseudo, dtype=float),
    )


def noising_component_endpoint_drops(
    native_clean_margin: np.ndarray,
    partner_patched_clean_margin: np.ndarray,
    pseudo_patched_clean_margin: np.ndarray,
) -> dict[str, np.ndarray]:
    """Noising effect: native clean margin minus aligned patched-clean margin."""
    clean = np.asarray(native_clean_margin, dtype=float)
    return endpoint_vectors(
        clean - np.asarray(partner_patched_clean_margin, dtype=float),
        clean - np.asarray(pseudo_patched_clean_margin, dtype=float),
    )


def derive_family_from_primitives(
    primitive: np.ndarray,
    *,
    components: Sequence[tuple[str, int, int | None]] = COMPONENTS,
) -> tuple[np.ndarray, list[str]]:
    """Derive all specifics and the two noising delivery gaps after any outer edit."""
    values = np.asarray(primitive, dtype=float)
    primitive_names = primitive_row_names(components)
    if (
        values.ndim != 2
        or values.shape[0] != len(primitive_names)
        or values.shape[1] == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("primitive must have frozen [40, examples] finite shape")
    by_name = {name: values[index] for index, name in enumerate(primitive_names)}
    derived: list[np.ndarray] = []
    names = family_row_names(components)
    for mode in MODES:
        for component in components:
            head = component_name(component)
            partner = by_name[primitive_row_name(mode, head, "partner")]
            pseudo = by_name[primitive_row_name(mode, head, "pseudo")]
            derived.extend((partner, pseudo, partner - pseudo))
    for endpoint in DELIVERY_GAP_ENDPOINTS:
        candidates = np.mean(
            [
                derived_endpoint_from_primitives(by_name, NOISING_MODE, head, endpoint)
                for head in DELIVERY_CANDIDATES
            ],
            axis=0,
        )
        comparators = np.mean(
            [
                derived_endpoint_from_primitives(by_name, NOISING_MODE, head, endpoint)
                for head in DONOR_ROUTE_COMPARATORS
            ],
            axis=0,
        )
        derived.append(candidates - comparators)
    family = np.vstack(derived)
    if family.shape[0] != 62 or len(names) != 62:
        raise RuntimeError("R3c family must contain exactly 60 head rows plus two gaps")
    return family, names


def derived_endpoint_from_primitives(
    by_name: Mapping[str, np.ndarray], mode: str, head: str, endpoint: str
) -> np.ndarray:
    partner = np.asarray(by_name[primitive_row_name(mode, head, "partner")])
    pseudo = np.asarray(by_name[primitive_row_name(mode, head, "pseudo")])
    if endpoint == "partner":
        return partner
    if endpoint == "pseudo":
        return pseudo
    if endpoint == "specific":
        return partner - pseudo
    raise ValueError(f"unknown endpoint: {endpoint}")


def center_primitive_rows(primitive: np.ndarray) -> np.ndarray:
    """Center primitives before a shared whole-gene outer sign flip."""
    values = np.asarray(primitive, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(primitive_row_names()):
        raise ValueError("primitive rows must use the frozen 40-row order")
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("primitive rows must contain finite example values")
    return values - values.mean(axis=1, keepdims=True)


def coherent_path_offset(delta: float) -> np.ndarray:
    """Build one primitive-space plant that exercises all seven paths coherently.

    A positive component receives ``partner=delta, pseudo=0``.  Specificity is therefore
    derived as ``delta``; it is never independently planted.  H11/H9 receive the same
    noising plant while the three donor-route comparators remain zero, so both delivery-gap
    rows also derive to exactly ``delta``.
    """
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be positive and finite")
    names = primitive_row_names()
    offset = np.zeros(len(names), dtype=float)
    positive_component_modes = (
        (MEAN_MODE, "L3H1"),
        (NOISING_MODE, "L3H1"),
        (MEAN_MODE, "L2H15"),
        (MEAN_MODE, "L1H12"),
        (NOISING_MODE, "L4H11"),
        (NOISING_MODE, "L4H9"),
        (NOISING_MODE, "L4H7"),
    )
    lookup = {name: index for index, name in enumerate(names)}
    for mode, head in positive_component_modes:
        offset[lookup[primitive_row_name(mode, head, "partner")]] = float(delta)
    family, family_names = derive_family_from_primitives(offset[:, None])
    observed = {name: float(family[index, 0]) for index, name in enumerate(family_names)}
    expected_value = {
        "MATERIAL_POSITIVE": float(delta),
        "EQUIVALENT_WITHIN_SESOI": 0.0,
    }
    for path in named_path_specs().values():
        for row, classification in path["requirements"].items():
            if not np.isclose(
                observed[row], expected_value[classification], rtol=0.0, atol=1e-12
            ):
                raise RuntimeError(f"incoherent R3c plant at {row}")
    return offset


def _empty_primitive_offset() -> tuple[np.ndarray, dict[str, int]]:
    names = primitive_row_names()
    return np.zeros(len(names), dtype=float), {
        name: index for index, name in enumerate(names)
    }


def _set_component_primitives(
    offset: np.ndarray,
    lookup: Mapping[str, int],
    mode: str,
    head: str,
    *,
    partner: float,
    pseudo: float,
) -> None:
    offset[lookup[primitive_row_name(mode, head, "partner")]] = float(partner)
    offset[lookup[primitive_row_name(mode, head, "pseudo")]] = float(pseudo)


def _easy_path_offset(path_names: Sequence[str]) -> tuple[np.ndarray, dict[str, int]]:
    """Place every listed path strictly inside its alternative using coherent primitives."""
    offset, lookup = _empty_primitive_offset()
    specs = named_path_specs()
    for path_name in path_names:
        for row, expected in specs[path_name]["requirements"].items():
            if expected != "MATERIAL_POSITIVE":
                continue
            mode, head, endpoint = row.split("|")
            if head == "delivery_gap":
                continue
            if endpoint not in {"partner", "specific"}:
                raise RuntimeError(f"positive R3c path row is not primitive-derivable: {row}")
            offset[lookup[primitive_row_name(mode, head, "partner")]] = PLANT_DELTA
    return offset, lookup


def _component_face(
    offset: np.ndarray,
    lookup: Mapping[str, int],
    mode: str,
    head: str,
    face: str,
) -> dict[str, str]:
    """Put exactly one component-path constituent on one coherent null face."""
    if face == "partner_boundary":
        _set_component_primitives(
            offset,
            lookup,
            mode,
            head,
            partner=SESOI,
            pseudo=-SESOI / 2,
        )
        return {
            family_row_name(mode, head, "partner"): "positive_boundary",
        }
    if face == "specific_boundary":
        _set_component_primitives(
            offset,
            lookup,
            mode,
            head,
            partner=1.5 * SESOI,
            pseudo=SESOI / 2,
        )
        return {
            family_row_name(mode, head, "specific"): "positive_boundary",
        }
    if face == "pseudo_lower_edge":
        _set_component_primitives(
            offset,
            lookup,
            mode,
            head,
            partner=PLANT_DELTA,
            pseudo=-SESOI,
        )
        return {
            family_row_name(mode, head, "pseudo"): "equivalence_lower_edge",
        }
    if face == "pseudo_upper_edge":
        _set_component_primitives(
            offset,
            lookup,
            mode,
            head,
            partner=PLANT_DELTA,
            pseudo=SESOI,
        )
        return {
            family_row_name(mode, head, "pseudo"): "equivalence_upper_edge",
        }
    raise ValueError(f"unknown component boundary face: {face}")


def _validate_surface(
    *,
    name: str,
    offset: np.ndarray,
    assigned_paths: Sequence[str],
    targeted_boundaries: Mapping[str, str],
) -> None:
    family, names = derive_family_from_primitives(offset[:, None])
    observed = {row: float(family[index, 0]) for index, row in enumerate(names)}
    requirements = named_path_specs()
    if not targeted_boundaries:
        raise RuntimeError(f"surface {name} has no null boundary")
    for path_name in assigned_paths:
        path_targets = set(targeted_boundaries) & set(
            requirements[path_name]["requirements"]
        )
        if len(path_targets) != 1:
            raise RuntimeError(
                f"surface {name} must put exactly one constituent/edge of assigned path "
                f"{path_name} on its null; observed {sorted(path_targets)}"
            )
        for row, expected in requirements[path_name]["requirements"].items():
            value = observed[row]
            target = targeted_boundaries.get(row)
            if target == "positive_boundary":
                valid = np.isclose(value, SESOI, rtol=0.0, atol=1e-12)
            elif target == "equivalence_lower_edge":
                valid = np.isclose(value, -SESOI, rtol=0.0, atol=1e-12)
            elif target == "equivalence_upper_edge":
                valid = np.isclose(value, SESOI, rtol=0.0, atol=1e-12)
            elif expected == "MATERIAL_POSITIVE":
                valid = value > SESOI
            else:
                valid = -SESOI < value < SESOI
            if not valid:
                raise RuntimeError(
                    f"surface {name} does not keep {path_name}:{row} on its declared "
                    f"boundary/interior (value={value}, target={target}, expected={expected})"
                )


def boundary_face_surface_specs() -> dict[str, dict[str, Any]]:
    """Return every coherent constituent-boundary surface in frozen execution order.

    The suite cycles each algebraically distinct null face while all other constituents of
    the assigned path are strictly inside their alternatives.  Homologous single-component
    paths may share a surface; a multi-component conjunction never has two independent
    component units put on their boundary merely for speed.
    """
    surfaces: dict[str, dict[str, Any]] = {}
    component_faces = (
        "partner_boundary",
        "specific_boundary",
        "pseudo_lower_edge",
        "pseudo_upper_edge",
    )

    def add(
        name: str,
        offset: np.ndarray,
        assigned_paths: Sequence[str],
        targeted: Mapping[str, str],
        rationale: str,
    ) -> None:
        _validate_surface(
            name=name,
            offset=offset,
            assigned_paths=assigned_paths,
            targeted_boundaries=targeted,
        )
        surfaces[name] = {
            "assigned_paths": list(assigned_paths),
            "targeted_boundaries": dict(targeted),
            "offset": offset.tolist(),
            "offset_sha256": _array_sha256(offset),
            "rationale": rationale,
        }

    # The distinct route has 28 algebraically distinct constituent faces: four component
    # faces for each of H11/H9, two gap faces, and six TOST edges for each of three
    # comparator heads.  Those 28 faces establish a lower bound on the suite size because
    # no surface may put two constituents of the same assigned path on its null.
    route_surfaces: list[dict[str, Any]] = []
    for head, individual_path in (
        ("L4H11", "L4H11_noising_component"),
        ("L4H9", "L4H9_noising_component"),
    ):
        for face in component_faces:
            offset, lookup = _easy_path_offset(
                (individual_path, "distinct_acceptor_route_candidate")
            )
            targeted = _component_face(offset, lookup, NOISING_MODE, head, face)
            route_surfaces.append(
                {
                    "name": f"{head}_route_{face}",
                    "offset": offset,
                    "lookup": lookup,
                    "assigned_paths": [
                        individual_path,
                        "distinct_acceptor_route_candidate",
                    ],
                    "targeted": targeted,
                    "rationale": (
                        "one isolated head face belongs to both its component claim and "
                        "the distinct-route path"
                    ),
                }
            )

    gap_faces = {
        "partner": (0.075, -0.025),
        "specific": (0.10, 0.025),
    }
    for endpoint, (candidate_partner, candidate_pseudo) in gap_faces.items():
        offset, lookup = _easy_path_offset(("distinct_acceptor_route_candidate",))
        for head in DELIVERY_CANDIDATES:
            _set_component_primitives(
                offset,
                lookup,
                NOISING_MODE,
                head,
                partner=candidate_partner,
                pseudo=candidate_pseudo,
            )
        for head in DONOR_ROUTE_COMPARATORS:
            _set_component_primitives(
                offset, lookup, NOISING_MODE, head, partner=0.025, pseudo=0.0
            )
        route_surfaces.append(
            {
                "name": f"distinct_route_gap_{endpoint}_boundary",
                "offset": offset,
                "lookup": lookup,
                "assigned_paths": ["distinct_acceptor_route_candidate"],
                "targeted": {
                    delivery_gap_row_name(endpoint): "positive_boundary"
                },
                "rationale": (
                    f"isolated {endpoint} delivery-gap face; the other gap and every "
                    "head endpoint remain strict alternatives"
                ),
            }
        )

    # Each comparator endpoint has two independently isolatable TOST edges.
    comparator_faces = {
        "partner_lower": (-SESOI, -SESOI / 2, "partner", "equivalence_lower_edge"),
        "partner_upper": (SESOI, SESOI / 2, "partner", "equivalence_upper_edge"),
        "pseudo_lower": (-SESOI / 2, -SESOI, "pseudo", "equivalence_lower_edge"),
        "pseudo_upper": (SESOI / 2, SESOI, "pseudo", "equivalence_upper_edge"),
        "specific_lower": (-SESOI / 2, SESOI / 2, "specific", "equivalence_lower_edge"),
        "specific_upper": (SESOI / 2, -SESOI / 2, "specific", "equivalence_upper_edge"),
    }
    for head in DONOR_ROUTE_COMPARATORS:
        for face, (partner, pseudo, endpoint, edge) in comparator_faces.items():
            offset, lookup = _easy_path_offset(("distinct_acceptor_route_candidate",))
            _set_component_primitives(
                offset,
                lookup,
                NOISING_MODE,
                head,
                partner=partner,
                pseudo=pseudo,
            )
            targeted = {
                family_row_name(NOISING_MODE, head, endpoint): edge
            }
            route_surfaces.append(
                {
                    "name": f"distinct_route_{head}_{face}",
                    "offset": offset,
                    "lookup": lookup,
                    "assigned_paths": ["distinct_acceptor_route_candidate"],
                    "targeted": targeted,
                    "rationale": (
                        "one isolated comparator TOST edge; its other two endpoints are "
                        "strictly inside equivalence"
                    ),
                }
            )

    if len(route_surfaces) != 28:
        raise RuntimeError("R3c requires exactly 28 algebraically isolated route surfaces")

    # Pack the twelve path-disjoint non-route faces into the first twelve route surfaces.
    # Packing cannot add a second null constituent to any assigned path and therefore reduces
    # production classifier calls without appealing to distributional symmetry.
    packed: list[tuple[list[tuple[str, str, str]], str]] = []
    for face in component_faces:
        packed.append(
            ([("L3H1_reused_component", MEAN_MODE, "L3H1")], face)
        )
    for face in component_faces:
        packed.append(
            ([("L3H1_reused_component", NOISING_MODE, "L3H1")], face)
        )
    shared = [
        ("L2H15_mean_component", MEAN_MODE, "L2H15"),
        ("L1H12_mean_component", MEAN_MODE, "L1H12"),
        ("L4H7_noising_component", NOISING_MODE, "L4H7"),
    ]
    for face in component_faces:
        packed.append((shared, face))

    for surface, (assignments, face) in zip(route_surfaces, packed):
        for path_name, _, _ in assignments:
            easy, _ = _easy_path_offset((path_name,))
            nonzero = easy != 0.0
            surface["offset"][nonzero] = easy[nonzero]
        for path_name, mode, head in assignments:
            if path_name not in surface["assigned_paths"]:
                surface["assigned_paths"].append(path_name)
            surface["targeted"].update(
                _component_face(
                    surface["offset"], surface["lookup"], mode, head, face
                )
            )
        surface["rationale"] += (
            "; packed with path-disjoint component faces so every assigned path still "
            "has exactly one null constituent/edge"
        )

    for surface in route_surfaces:
        add(
            surface["name"],
            surface["offset"],
            surface["assigned_paths"],
            surface["targeted"],
            surface["rationale"],
        )
    return surfaces


def boundary_face_suite_binding() -> dict[str, Any]:
    specs = boundary_face_surface_specs()
    return {
        "surface_order": list(specs),
        "surface_count": len(specs),
        "surfaces": {
            name: {
                key: value
                for key, value in spec.items()
                if key != "offset"
            }
            for name, spec in specs.items()
        },
        "suite_sha256": canonical_sha256(specs),
    }


def path_health_participants() -> dict[str, list[tuple[str, str]]]:
    """Return the ordered component/mode health dependencies of every named path."""
    participants: dict[str, list[tuple[str, str]]] = {}
    for path_name, spec in named_path_specs().items():
        seen: list[tuple[str, str]] = []
        for row in spec["requirements"]:
            mode, head, _ = row.split("|")
            if head == "delivery_gap":
                for participant in (*DELIVERY_CANDIDATES, *DONOR_ROUTE_COMPARATORS):
                    key = (NOISING_MODE, participant)
                    if key not in seen:
                        seen.append(key)
            else:
                key = (mode, head)
                if key not in seen:
                    seen.append(key)
        participants[path_name] = seen
    return participants


def path_health_eligibility(
    measurements: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, dict[str, Any]]:
    """Apply live health locally to only the component/mode rows used by each path."""
    participants = path_health_participants()
    return {
        path_name: {
            "ok": all(
                bool(measurements[mode][head]["health"]["ok"])
                for mode, head in entries
            ),
            "participants": [f"{mode}|{head}" for mode, head in entries],
            "failed_participants": [
                f"{mode}|{head}"
                for mode, head in entries
                if not bool(measurements[mode][head]["health"]["ok"])
            ],
        }
        for path_name, entries in participants.items()
    }


def calibration_path_health_eligibility() -> dict[str, dict[str, Any]]:
    """Exercise every frozen path in synthetic calibration, independent of live health.

    Live health can only suppress a scientific declaration.  It must not make the
    classifier's false-positive or planted-power calibration easier by silently removing
    a path from the engineering preflight.
    """
    return {
        path_name: {
            "ok": True,
            "participants": [f"{mode}|{head}" for mode, head in entries],
            "failed_participants": [],
        }
        for path_name, entries in path_health_participants().items()
    }


def monotone_target_path_health(
    current_health: Mapping[str, Mapping[str, Any]],
    preflight_paths: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Authorize target paths only when both preflight and target live health pass.

    A target may lose an otherwise authorized path when its own health fails, but it may
    never gain a path whose bucket-0 health failed.  This prevents target outcomes from
    reviving a path that lacked a live preflight authorization.
    """
    if tuple(current_health) != PATH_NAMES or tuple(preflight_paths) != PATH_NAMES:
        raise ValueError("target health requires the exact seven-path registry")
    effective: dict[str, dict[str, Any]] = {}
    preflight_authorized: list[str] = []
    current_healthy: list[str] = []
    effective_authorized: list[str] = []
    for path_name in PATH_NAMES:
        current = current_health[path_name]
        preflight = preflight_paths[path_name].get("health")
        if not isinstance(preflight, Mapping):
            raise ValueError(f"preflight health is missing for {path_name}")
        current_participants = current.get("participants")
        preflight_participants = preflight.get("participants")
        if (
            not isinstance(current.get("ok"), bool)
            or not isinstance(preflight.get("ok"), bool)
            or current_participants != preflight_participants
        ):
            raise ValueError(f"preflight/current health registry differs for {path_name}")
        preflight_ok = bool(preflight["ok"])
        current_ok = bool(current["ok"])
        if preflight_ok:
            preflight_authorized.append(path_name)
        if current_ok:
            current_healthy.append(path_name)
        failed = list(
            dict.fromkeys(
                [
                    *list(preflight.get("failed_participants", [])),
                    *list(current.get("failed_participants", [])),
                ]
            )
        )
        effective[path_name] = {
            "ok": preflight_ok and current_ok,
            "participants": list(current_participants),
            "failed_participants": failed,
            "preflight_health_ok": preflight_ok,
            "current_health_ok": current_ok,
            "authorization_rule": "preflight_health_ok AND current_health_ok",
        }
        if effective[path_name]["ok"]:
            effective_authorized.append(path_name)
    authorization = {
        "schema": "gfmcd.r3c.target_path_health_authorization/1",
        "rule": "effective target health = reviewed preflight health AND current health",
        "can_only_shrink_from_preflight": True,
        "preflight_authorized_paths": preflight_authorized,
        "preflight_suppressed_paths": [
            name for name in PATH_NAMES if name not in preflight_authorized
        ],
        "current_healthy_paths": current_healthy,
        "effective_authorized_paths": effective_authorized,
        "effective_suppressed_paths": [
            name for name in PATH_NAMES if name not in effective_authorized
        ],
    }
    return effective, authorization


def reviewed_preflight_path_health(
    current_health: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Carry the reviewed v1 health suppression into the prospective v2 ceiling.

    V2 was created after observing the v1 bucket-0 L2H15 health failure.  Repeating the
    same deterministic health sample may diagnose reproducibility, but it cannot turn that
    observed failure into a fresh authorization.
    """
    if tuple(current_health) != PATH_NAMES:
        raise ValueError("preflight health requires the exact seven-path registry")
    effective: dict[str, dict[str, Any]] = {}
    for path_name in PATH_NAMES:
        current = current_health[path_name]
        current_ok = current.get("ok")
        if not isinstance(current_ok, bool):
            raise ValueError(f"current preflight health is malformed for {path_name}")
        historical_ok = path_name not in HISTORICAL_V1_HEALTH_SUPPRESSED_PATHS
        participants = list(current.get("participants", []))
        failed = list(current.get("failed_participants", []))
        if not historical_ok:
            failed = list(dict.fromkeys([*participants, *failed]))
        effective[path_name] = {
            "ok": historical_ok and current_ok,
            "participants": participants,
            "failed_participants": failed,
            "reviewed_v1_health_ceiling_ok": historical_ok,
            "current_v2_preflight_health_ok": current_ok,
            "authorization_rule": "reviewed v1 health ceiling AND current v2 preflight health",
        }
    return effective


def preflight_path_health_authorization(
    observed_health: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Record the reviewed bucket-0 live-health ceiling for a future target."""
    if tuple(observed_health) != PATH_NAMES:
        raise ValueError("preflight health requires the exact seven-path registry")
    authorized = [name for name in PATH_NAMES if observed_health[name].get("ok") is True]
    return {
        "schema": "gfmcd.r3c.preflight_path_health_authorization/1",
        "rule": (
            "reviewed v1 health ceiling AND current v2 bucket-0 health freezes the "
            "maximum target-authorized path set"
        ),
        "can_only_shrink_at_target": True,
        "historical_v1_suppressed_paths": list(
            HISTORICAL_V1_HEALTH_SUPPRESSED_PATHS
        ),
        "preflight_authorized_paths": authorized,
        "preflight_suppressed_paths": [name for name in PATH_NAMES if name not in authorized],
    }


def _required_row_test_kinds() -> dict[str, str]:
    kinds: dict[str, str] = {}
    for spec in named_path_specs().values():
        for row, expected in spec["requirements"].items():
            kind = (
                "positive_boundary"
                if expected == "MATERIAL_POSITIVE"
                else "tost_equivalence_iut"
            )
            prior = kinds.setdefault(row, kind)
            if prior != kind:
                raise RuntimeError(f"R3c row {row} has conflicting test directions")
    return kinds


def strict_path_family_inference(
    family: np.ndarray,
    groups: Sequence[str],
    names: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int,
    path_health: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the 62-row envelope and the exact seven-path boundary/IUT/Holm family."""
    values = np.asarray(family, dtype=float)
    ordered_names = list(names)
    if (
        values.ndim != 2
        or values.shape[0] != 62
        or ordered_names != family_row_names()
        or values.shape[1] != len(groups)
        or not np.isfinite(values).all()
    ):
        raise ValueError("strict inference requires the exact finite 62-row R3c family")
    if n_draws < 2:
        raise ValueError("R3c inference requires at least two draws")

    interval_decisions, interval_calibration = simultaneous_family_decision(
        values,
        groups,
        ordered_names,
        sesoi=SESOI,
        n_draws=n_draws,
        alpha=FAMILY_ALPHA,
        seed=seed,
        namespace=f"{namespace}:simultaneous-estimation-envelope",
        min_genes=min_genes,
        studentized=True,
    )
    interval_calibration = dict(interval_calibration)
    interval_calibration["method"] = (
        "gene-cluster bootstrap endpoint-scale-normalized max-deviation interval"
    )
    interval_calibration["nested_bootstrap_t"] = False
    interval_calibration["inclusive_historical_classification_used"] = False
    interval_calibration["strict_classification_rule"] = (
        "lo > +SESOI; hi < -SESOI; or -SESOI < lo <= hi < +SESOI"
    )

    index = {name: position for position, name in enumerate(ordered_names)}
    kinds = _required_row_test_kinds()
    positive_names = [name for name in ordered_names if kinds.get(name) == "positive_boundary"]
    equivalent_names = [name for name in ordered_names if kinds.get(name) == "tost_equivalence_iut"]

    # Every decision-bearing edge, including both TOST edges, shares one exact
    # whole-gene Rademacher matrix.  Do not split these into helper calls with different
    # namespaces: the vector sign-flip contract is stronger than Holm's dependence needs.
    edge_names: list[str] = []
    edge_values: list[np.ndarray] = []
    edge_boundaries: list[float] = []
    edge_alternatives: list[str] = []
    for name in positive_names:
        edge_names.append(f"{name}|positive_edge")
        edge_values.append(values[index[name]])
        edge_boundaries.append(SESOI)
        edge_alternatives.append("greater")
    for name in equivalent_names:
        row = values[index[name]]
        edge_names.extend((f"{name}|lower_edge", f"{name}|upper_edge"))
        edge_values.extend((row, row))
        edge_boundaries.extend((-SESOI, SESOI))
        edge_alternatives.extend(("greater", "less"))
    edge_tests = boundary_sign_flip_tests(
        np.vstack(edge_values),
        groups,
        edge_names,
        boundaries=edge_boundaries,
        alternatives=edge_alternatives,
        n_draws=n_draws,
        seed=seed,
        namespace=f"{namespace}:shared-vector-boundary-edges",
        min_genes=min_genes,
    )

    sign_digests = {test.sign_matrix_sha256 for test in edge_tests.values()}
    gene_digests = {
        test.ordered_gene_labels_sha256 for test in edge_tests.values()
    }
    namespaces = {test.namespace for test in edge_tests.values()}
    if len(sign_digests) != 1 or len(gene_digests) != 1 or len(namespaces) != 1:
        raise RuntimeError(
            "R3c decision-bearing edges did not reuse one whole-gene sign matrix"
        )
    required_boundary_metadata = {
        "sign_matrix_sha256",
        "ordered_gene_labels_sha256",
        "shared_null_draws_sha256",
        "null_draws_sha256",
        "namespace",
        "B",
    }
    if any(
        not required_boundary_metadata.issubset(test.to_dict())
        for test in edge_tests.values()
    ):
        raise RuntimeError("shared boundary-test provenance metadata is incomplete")

    row_tests: dict[str, dict[str, Any]] = {}
    for name in positive_names:
        test = edge_tests[f"{name}|positive_edge"]
        row_tests[name] = {
            "test": "H0: mean <= +SESOI; H1: mean > +SESOI",
            "expected_classification": "MATERIAL_POSITIVE",
            "p_value": test.p_value,
            "boundary_test": test.to_dict(),
        }
    for name in equivalent_names:
        lower = edge_tests[f"{name}|lower_edge"]
        upper = edge_tests[f"{name}|upper_edge"]
        p_value = intersection_union_pvalue([lower.p_value, upper.p_value])
        row_tests[name] = {
            "test": "TOST/IUT for -SESOI < mean < +SESOI",
            "expected_classification": "EQUIVALENT_WITHIN_SESOI",
            "p_value": p_value,
            "lower_edge": lower.to_dict(),
            "upper_edge": upper.to_dict(),
            "iut_rule": "max(lower_edge_p, upper_edge_p)",
        }

    interval_rows: dict[str, dict[str, Any]] = {}
    for name in ordered_names:
        historical = interval_decisions[name]
        lo, hi = historical.simultaneous_ci
        interval_rows[name] = {
            "estimate": historical.estimate,
            "simultaneous_ci": [lo, hi],
            "strict_classification": strict_interval_decision(lo, hi, SESOI),
            "historical_inclusive_classification": historical.classification,
            "historical_zero_boundary_max_abs_p": historical.max_abs_p,
            "decision_authority": (
                "strict CI concordance only; boundary path p-values are the test authority"
            ),
        }

    specs = named_path_specs()
    raw_path_p = {
        path_name: intersection_union_pvalue(
            [row_tests[row]["p_value"] for row in spec["requirements"]]
        )
        for path_name, spec in specs.items()
    }
    adjusted_path_p = holm_adjust_named(raw_path_p)
    health = path_health or {
        path_name: {"ok": True, "participants": [], "failed_participants": []}
        for path_name in PATH_NAMES
    }
    if tuple(health) != PATH_NAMES:
        raise ValueError("path health must cover the exact seven paths in frozen order")

    path_outputs: dict[str, dict[str, Any]] = {}
    for path_name, spec in specs.items():
        requirements = spec["requirements"]
        ci_matches = {
            row: interval_rows[row]["strict_classification"] == expected
            for row, expected in requirements.items()
        }
        p_pass = bool(adjusted_path_p[path_name] < FAMILY_ALPHA)
        ci_pass = all(ci_matches.values())
        health_ok = bool(health[path_name]["ok"])
        if not health_ok:
            classification = "HEALTH_FAILED"
        elif p_pass and ci_pass:
            classification = "SUPPORTED"
        elif p_pass != ci_pass:
            classification = "INCONCLUSIVE_METHOD_DISAGREEMENT"
        else:
            classification = "NOT_SUPPORTED_AT_FROZEN_THRESHOLDS"
        path_outputs[path_name] = {
            "tier": spec["tier"],
            "requirements": dict(requirements),
            "iut_p_value": raw_path_p[path_name],
            "holm_adjusted_p": adjusted_path_p[path_name],
            "holm_family_size": len(PATH_NAMES),
            "strict_p_threshold": f"adjusted_p < {FAMILY_ALPHA}",
            "p_pass": p_pass,
            "strict_simultaneous_ci_matches": ci_matches,
            "strict_simultaneous_ci_pass": ci_pass,
            "health": dict(health[path_name]),
            "classification": classification,
        }
    return {
        "row_names": ordered_names,
        "row_order_sha256": canonical_sha256(ordered_names),
        "intervals": interval_rows,
        "interval_calibration": interval_calibration,
        "row_boundary_tests": row_tests,
        "boundary_randomisation_audit": {
            "schema": "gfmcd.r3c.shared_boundary_randomisation/1",
            "all_decision_edges_share_one_sign_matrix": True,
            "edge_count": len(edge_tests),
            "B": n_draws,
            "namespace": next(iter(namespaces)),
            "sign_matrix_sha256": next(iter(sign_digests)),
            "ordered_gene_labels_sha256": next(iter(gene_digests)),
            "edge_order_sha256": canonical_sha256(edge_names),
        },
        "untested_rows": [name for name in ordered_names if name not in row_tests],
        "path_family": {
            "method": "path IUT p-values followed by Holm across exactly seven named claims",
            "alpha": FAMILY_ALPHA,
            "strict_threshold": "adjusted_p < alpha",
            "path_names": list(PATH_NAMES),
            "raw_iut_p_values": raw_path_p,
            "holm_adjusted_p_values": adjusted_path_p,
        },
        "paths": path_outputs,
    }


def compact_classifier_audit(result: Mapping[str, Any]) -> dict[str, Any]:
    """Retain enough of every calibration classifier call for semantic replay checks."""
    return {
        "boundary_randomisation_audit": dict(
            result["boundary_randomisation_audit"]
        ),
        "interval_calibration": {
            key: result["interval_calibration"][key]
            for key in (
                "n_components",
                "n_examples",
                "n_genes",
                "minimum_genes_required",
                "n_bootstrap_draws",
                "n_sign_flip_draws",
                "alpha",
                "sesoi",
                "studentized",
            )
        },
        "path_family": dict(result["path_family"]),
        "paths": {
            name: {
                key: result["paths"][name][key]
                for key in (
                    "iut_p_value",
                    "holm_adjusted_p",
                    "holm_family_size",
                    "p_pass",
                    "strict_simultaneous_ci_pass",
                    "classification",
                )
            }
            | {"health_ok": bool(result["paths"][name]["health"]["ok"])}
            for name in PATH_NAMES
        },
    }


def r3c_path_preflight(
    primitive: np.ndarray,
    groups: Sequence[str],
    *,
    path_health: Mapping[str, Mapping[str, Any]],
    n_calibration: int,
    n_trials: int,
    seed: int,
    namespace: str,
    min_genes: int,
) -> dict[str, Any]:
    """Calibrate the exact seven-path classifier on centered primitive residuals."""
    values = np.asarray(primitive, dtype=float)
    if values.shape[0] != len(primitive_row_names()) or values.shape[1] != len(groups):
        raise ValueError("preflight primitive matrix and groups disagree")
    if n_calibration < 2 or n_trials < 1:
        raise ValueError("preflight draw counts are invalid")
    residuals = center_primitive_rows(values)
    surface_specs = boundary_face_surface_specs()
    plant = coherent_path_offset(PLANT_DELTA)
    planted_successes = {name: 0 for name in PATH_NAMES}
    case_audit: dict[str, dict[str, Any]] = {
        case: {
            "assigned_paths": list(spec["assigned_paths"]),
            "targeted_boundaries": dict(spec["targeted_boundaries"]),
            "offset_sha256": spec["offset_sha256"],
            "rationale": spec["rationale"],
            "outer_namespace_template": (
                f"{namespace}:boundary-face:{case}:outer:<trial>"
            ),
            "classifier_namespace_template": (
                f"{namespace}:boundary-face:{case}:classifier:<trial>"
            ),
            "false_positive_trials": 0,
            "per_path_false_positive_trials": {
                name: 0 for name in spec["assigned_paths"]
            },
            "outer_null_sha256_by_trial": [],
            "classifier_audit_sha256_by_trial": [],
            "classifier_records_by_trial": [],
            "false_positive_pattern": [],
        }
        for case, spec in surface_specs.items()
    }
    plant_outer_hashes: list[str] = []
    plant_classifier_hashes: list[str] = []
    plant_classifier_records: list[dict[str, Any]] = []
    for trial in range(n_trials):
        for case, spec in surface_specs.items():
            case_paths = spec["assigned_paths"]
            null = gene_sign_flip(
                residuals,
                groups,
                seed=seed,
                namespace=f"{namespace}:boundary-face:{case}:outer:{trial}",
            )
            case_audit[case]["outer_null_sha256_by_trial"].append(
                _array_sha256(null)
            )
            boundary_face_family, row_names = derive_family_from_primitives(
                null + np.asarray(spec["offset"], dtype=float)[:, None]
            )
            boundary_face_result = strict_path_family_inference(
                boundary_face_family,
                groups,
                row_names,
                n_draws=n_calibration,
                seed=seed,
                namespace=(
                    f"{namespace}:boundary-face:{case}:classifier:{trial}"
                ),
                min_genes=min_genes,
                path_health=path_health,
            )
            compact_audit = compact_classifier_audit(boundary_face_result)
            case_audit[case]["classifier_records_by_trial"].append(compact_audit)
            case_audit[case]["classifier_audit_sha256_by_trial"].append(
                canonical_sha256(compact_audit)
            )
            supported = {
                name: (
                    boundary_face_result["paths"][name]["classification"]
                    == "SUPPORTED"
                )
                for name in case_paths
            }
            case_positive = any(supported.values())
            case_audit[case]["false_positive_trials"] += int(case_positive)
            case_audit[case]["false_positive_pattern"].append(int(case_positive))
            for name, value in supported.items():
                case_audit[case]["per_path_false_positive_trials"][name] += int(
                    value
                )

        plant_null = gene_sign_flip(
            residuals,
            groups,
            seed=seed,
            namespace=f"{namespace}:coherent-plant:outer:{trial}",
        )
        plant_outer_hashes.append(_array_sha256(plant_null))
        planted_family, row_names = derive_family_from_primitives(
            plant_null + plant[:, None]
        )
        planted_result = strict_path_family_inference(
            planted_family,
            groups,
            row_names,
            n_draws=n_calibration,
            seed=seed,
            namespace=f"{namespace}:coherent-plant:classifier:{trial}",
            min_genes=min_genes,
            path_health=path_health,
        )
        plant_compact_audit = compact_classifier_audit(planted_result)
        plant_classifier_records.append(plant_compact_audit)
        plant_classifier_hashes.append(
            canonical_sha256(plant_compact_audit)
        )
        for name in PATH_NAMES:
            planted_successes[name] += int(
                planted_result["paths"][name]["classification"] == "SUPPORTED"
            )

    planted_power: dict[str, dict[str, Any]] = {}
    for name in PATH_NAMES:
        count = planted_successes[name]
        lower, _ = one_sided_binomial_bounds(count, n_trials, alpha=FAMILY_ALPHA)
        health_ok = bool(path_health[name]["ok"])
        planted_power[name] = {
            "successes": count,
            "trials": n_trials,
            "power": count / n_trials,
            "one_sided_clopper_pearson_lower": lower,
            "health_eligible": health_ok,
            "power_ok_if_health_eligible": bool(
                (not health_ok) or lower >= PREFLIGHT_POWER_FLOOR
            ),
        }
    eligible = [name for name in PATH_NAMES if path_health[name]["ok"]]
    pattern_digests: dict[str, str] = {}
    for case in surface_specs:
        record = case_audit[case]
        record["false_positive_rate"] = record["false_positive_trials"] / n_trials
        _, upper = one_sided_binomial_bounds(
            record["false_positive_trials"], n_trials, alpha=FAMILY_ALPHA
        )
        record["one_sided_clopper_pearson_upper"] = upper
        record["fpr_ok"] = bool(upper <= FAMILY_ALPHA)
        record["outer_null_digest_sha256"] = canonical_sha256(
            record.pop("outer_null_sha256_by_trial")
        )
        record["classifier_audit_digest_sha256"] = canonical_sha256(
            record.pop("classifier_audit_sha256_by_trial")
        )
        record["classifier_records_sha256"] = canonical_sha256(
            record["classifier_records_by_trial"]
        )
        pattern_digest = canonical_sha256(
            record.pop("false_positive_pattern")
        )
        record["false_positive_pattern_sha256"] = pattern_digest
        pattern_digests[case] = pattern_digest
    worst_surface = max(
        surface_specs,
        key=lambda case: (
            case_audit[case]["one_sided_clopper_pearson_upper"],
            case_audit[case]["false_positive_trials"],
            -list(surface_specs).index(case),
        ),
    )
    worst_count = int(case_audit[worst_surface]["false_positive_trials"])
    worst_upper = float(
        case_audit[worst_surface]["one_sided_clopper_pearson_upper"]
    )
    return {
        "randomisation_unit": "gene",
        "outer_trial_unit": "endpoint-centered primitive row residual",
        "outer_trial_centering": (
            "subtract each of 40 primitive partner/pseudo row means before sign flipping"
        ),
        "outer_trial_algebra": (
            "one sign per gene shared across all 40 primitives; derive 20 specifics and "
            "two delivery gaps only after the outer flip and coherent offset"
        ),
        "translation_invariant": True,
        "classifier": (
            "62-row endpoint-scaled simultaneous envelope plus strict boundary/TOST/IUT "
            "tests and exact seven-path Holm family"
        ),
        "namespace": namespace,
        "alpha": FAMILY_ALPHA,
        "sesoi": SESOI,
        "n_primitive_rows": len(primitive_row_names()),
        "n_family_rows": len(family_row_names()),
        "n_examples": values.shape[1],
        "n_genes": len(set(map(str, groups))),
        "minimum_genes_required": min_genes,
        "n_calibration": n_calibration,
        "n_trials": n_trials,
        "production_classifier_calls": (len(surface_specs) + 1)
        * n_trials,
        "boundary_face_cases": case_audit,
        "boundary_face_case_order": list(surface_specs),
        "boundary_face_suite_binding": boundary_face_suite_binding(),
        "boundary_face_semantics": (
            "each independently randomized surface puts exactly one constituent/TOST "
            "edge of every assigned path on its null boundary, holds every other "
            "conjunct strictly inside its alternative, derives specifics/gaps from "
            "primitives, and invokes the complete seven-path classifier"
        ),
        "analytic_fwer_argument": (
            "under any true path null at least one positive-boundary or TOST-union-null "
            "constituent is valid and super-uniform; the path IUT maximum is therefore "
            "valid, Holm strongly controls the seven-path FWER under arbitrary dependence, "
            "and strict-CI concordance plus health gates can only remove declarations"
        ),
        "boundary_face_role": (
            "prespecified end-to-end implementation diagnostics, not an assertion that "
            "one nuisance midpoint globally maximizes declaration probability"
        ),
        "surface_fpr_gate": (
            "every surface independently requires the one-sided Clopper-Pearson upper "
            "bound for any assigned null path declared SUPPORTED to be <= alpha"
        ),
        "worst_surface": worst_surface,
        "worst_surface_false_positives": worst_count,
        "surface_pattern_manifest_sha256": canonical_sha256(
            pattern_digests
        ),
        "worst_surface_fpr": worst_count / n_trials,
        "worst_surface_fpr_upper": worst_upper,
        "all_surface_fpr_diagnostics_ok": bool(
            all(record["fpr_ok"] for record in case_audit.values())
        ),
        "plant_delta": PLANT_DELTA,
        "coherent_primitive_plant": plant.tolist(),
        "coherent_primitive_plant_sha256": _array_sha256(plant),
        "plant_semantics": "partner-only primitive plants; pseudo=0; specifics/gaps derived",
        "plant_outer_namespace_template": (
            f"{namespace}:coherent-plant:outer:<trial>"
        ),
        "plant_classifier_namespace_template": (
            f"{namespace}:coherent-plant:classifier:<trial>"
        ),
        "plant_outer_null_digest_sha256": canonical_sha256(plant_outer_hashes),
        "plant_classifier_audit_digest_sha256": canonical_sha256(
            plant_classifier_hashes
        ),
        "plant_classifier_records_by_trial": plant_classifier_records,
        "plant_classifier_records_sha256": canonical_sha256(
            plant_classifier_records
        ),
        "planted_power_by_path": planted_power,
        "health_eligible_paths": eligible,
        "all_health_eligible_path_plants_powered": bool(
            eligible
            and all(
                planted_power[name]["power_ok_if_health_eligible"] for name in eligible
            )
        ),
        "power_floor": PREFLIGHT_POWER_FLOOR,
        "maximum_false_positives_per_surface": MAX_SURFACE_FALSE_POSITIVES,
        "minimum_plant_successes_per_health_eligible_path": MIN_PATH_PLANT_SUCCESSES,
        "preflight_cost_benchmark": PREFLIGHT_COST_BENCHMARK,
        "binomial_bound": "one-sided 95% Clopper-Pearson",
    }


def gene_bucket(gene_id: str) -> int:
    digest = hashlib.sha256(f"{GENE_BUCKET_SALT}|{gene_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2


def historical_r3_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(row["example_id"]) for row in rows]
    genes = {str(row["gene_id"]) for row in rows}
    return {
        "rows": len(rows),
        "genes": len(genes),
        "example_id_sha256": hashlib.sha256("|".join(ids).encode()).hexdigest(),
    }


def validate_historical_r3_identity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = historical_r3_identity(rows)
    expected = {
        "rows": HISTORICAL_EXPECTED_ROWS,
        "genes": HISTORICAL_EXPECTED_GENES,
        "example_id_sha256": HISTORICAL_EXPECTED_EXAMPLE_ID_SHA256,
    }
    if observed != expected:
        raise RuntimeError(
            "historical R3 reconstruction differs from immutable identity: "
            f"observed={observed}, expected={expected}"
        )
    return observed


def current_census_identity(audit: Mapping[str, Any]) -> dict[str, Any]:
    current = audit["current_census"]
    historical = audit["historical_reconstruction"]
    return {
        "screened_rows": current["screened_rows"],
        "historical_excluded_gene_sha256": historical[
            "ordered_excluded_genes_sha256"
        ],
        "eligible_rows_by_bucket": current["eligible_rows_by_bucket"],
        "eligible_genes_by_bucket": current["eligible_genes_by_bucket"],
        "fixed_selection_by_bucket": current["fixed_selection_by_bucket"],
    }


def validate_current_census_identity(audit: Mapping[str, Any]) -> dict[str, Any]:
    observed = current_census_identity(audit)
    if observed != CURRENT_EXPECTED_CENSUS_IDENTITY:
        raise RuntimeError(
            "R3c full-GTF current census identity drifted: "
            f"observed={observed}, expected={CURRENT_EXPECTED_CENSUS_IDENTITY}"
        )
    return observed


def _current_row_record(example, decoy_donor: int) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "gene_id": example.gene_id,
        "decoy_donor": int(decoy_donor),
    }


def load_cohort(args):
    """Construct the outcome-blind acceptor-only cohort before loading the model."""
    bucket = 1 if args.stage == "target" else 0
    historical_screen = load_examples_unscored(
        args.gtf,
        args.fasta,
        "val",
        HISTORICAL_SCREEN_N,
        tag=HISTORICAL_TAG,
        allowed_chromosomes=VALIDATION_CHROMOSOMES,
    )
    historical_rows: list[dict[str, Any]] = []
    historical_exclusions: Counter[str] = Counter()
    for example in historical_screen:
        decoy_donor = matched_decoy_donor(example)
        decoy_acceptor = matched_decoy_acceptor(example, DECOY_TOL)
        if decoy_donor is None:
            historical_exclusions["missing_legacy_decoy_donor"] += 1
            continue
        if decoy_acceptor is None:
            historical_exclusions["missing_legacy_decoy_acceptor"] += 1
            continue
        historical_rows.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "decoy_donor": int(decoy_donor),
                "decoy_acceptor": int(decoy_acceptor),
            }
        )
    historical_identity = validate_historical_r3_identity(historical_rows)
    historical_genes = sorted({row["gene_id"] for row in historical_rows})
    historical_gene_set = set(historical_genes)

    census = load_examples_unscored(
        args.gtf,
        args.fasta,
        "val",
        None,
        tag=CURRENT_TAG,
        allowed_chromosomes=VALIDATION_CHROMOSOMES,
    )
    exclusions: Counter[str] = Counter()
    eligible: dict[int, list[tuple[Any, int]]] = {0: [], 1: []}
    for example in census:
        if example.gene_id in historical_gene_set:
            exclusions["historical_R3_gene"] += 1
            continue
        decoy_donor = matched_decoy_donor(example)
        if decoy_donor is None:
            exclusions["missing_matched_decoy_donor"] += 1
            continue
        decoy_donor = int(decoy_donor)
        if decoy_donor == int(example.donor_pos):
            exclusions["pseudo_donor_equals_paired_donor"] += 1
            continue
        if abs(decoy_donor - int(example.donor_pos)) > DECOY_TOL:
            exclusions["donor_pseudo_motif_outside_distance_caliper"] += 1
            continue
        if not donor_alt_admissible(example, MOTIF_CLEAN_DONOR_ALT):
            exclusions["paired_donor_edit_inadmissible"] += 1
            continue
        if planted_donor_motifs(
            example.sequence, decoy_donor, MOTIF_CLEAN_DONOR_ALT
        ):
            exclusions["pseudo_donor_edit_plants_splice_motif"] += 1
            continue
        eligible[gene_bucket(example.gene_id)].append((example, decoy_donor))

    fixed_selection: dict[str, dict[str, Any]] = {}
    fixed_selection_genes: dict[str, dict[str, Any]] = {}
    for value in (0, 1):
        rows = eligible[value][: PREFLIGHT_N if value == 0 else TARGET_N]
        genes = sorted({str(row[0].gene_id) for row in rows})
        fixed_selection[str(value)] = {
            "rows": len(rows),
            "genes": len(genes),
            "ordered_rows_sha256": canonical_sha256(
                [_current_row_record(example, donor) for example, donor in rows]
            ),
            "gene_ids_sha256": canonical_sha256(genes),
        }
        fixed_selection_genes[str(value)] = {
            "gene_ids": genes,
            "gene_ids_sha256": canonical_sha256(genes),
        }
    available = eligible[bucket]
    if len(available) < args.n:
        raise RuntimeError(
            f"only {len(available)} acceptor-only rows in bucket {bucket}; asked for {args.n}"
        )
    selected_rows = available[: args.n]
    selected = [row[0] for row in selected_rows]
    n_genes = len({example.gene_id for example in selected})
    if n_genes < MIN_GENES:
        raise RuntimeError(
            f"R3c requires at least {MIN_GENES} independent genes; selected {n_genes}"
        )
    audit = {
        "schema": "gfmcd.r3c.acceptor_only_cohort_audit/1",
        "selection_reads_model_outputs": False,
        "preconstruction_allowed_chromosomes": list(VALIDATION_CHROMOSOMES),
        "sealed_chr8_chr9_opened": False,
        "historical_reconstruction": {
            "tag": HISTORICAL_TAG,
            "screened_rows": len(historical_screen),
            "both_legacy_decoys_complete_rows": len(historical_rows),
            "genes": len(historical_genes),
            "immutable_identity": historical_identity,
            "immutable_identity_matches": True,
            "exclusions": dict(sorted(historical_exclusions.items())),
            "ordered_rows_sha256": canonical_sha256(historical_rows),
            "ordered_excluded_genes_sha256": canonical_sha256(historical_genes),
            "excluded_gene_ids": historical_genes,
            "rule": HISTORICAL_EXCLUSION_RULE,
        },
        "current_census": {
            "tag": CURRENT_TAG,
            "screened_rows": len(census),
            "eligibility_requires_matched_acceptor": False,
            "eligibility_requires_only_donor_pseudo": True,
            "exclusions": dict(sorted(exclusions.items())),
            "eligible_rows_by_bucket": {
                str(value): len(eligible[value]) for value in (0, 1)
            },
            "eligible_genes_by_bucket": {
                str(value): len({row[0].gene_id for row in eligible[value]})
                for value in (0, 1)
            },
            "fixed_selection_by_bucket": fixed_selection,
            "fixed_selection_gene_ids_by_bucket": fixed_selection_genes,
        },
        "selection": {
            "bucket": bucket,
            "rows": len(selected_rows),
            "genes": n_genes,
            "ordered_rows_sha256": canonical_sha256(
                [_current_row_record(example, donor) for example, donor in selected_rows]
            ),
            "gene_ids": sorted({str(example.gene_id) for example in selected}),
            "gene_ids_sha256": canonical_sha256(
                sorted({str(example.gene_id) for example in selected})
            ),
            "donor_distance_mismatch": {
                "caliper": DECOY_TOL,
                "median": float(
                    np.median(
                        [
                            abs(int(example.donor_pos) - donor)
                            for example, donor in selected_rows
                        ]
                    )
                ),
                "p95": float(
                    np.quantile(
                        [
                            abs(int(example.donor_pos) - donor)
                            for example, donor in selected_rows
                        ],
                        0.95,
                    )
                ),
                "maximum": max(
                    abs(int(example.donor_pos) - donor)
                    for example, donor in selected_rows
                ),
            },
            "annotated_pseudo_donor_count": sum(
                donor in {int(position) for position in example.real_don}
                for example, donor in selected_rows
            ),
        },
    }
    validate_current_census_identity(audit)
    return selected, [row[1] for row in selected_rows], bucket, audit


def mean_intervention(
    adapter, component: tuple[str, int, int | None]
) -> AbstractContextManager:
    kind, layer, head = component
    if kind != "head" or head is None:
        raise ValueError("R3c can intervene on attention heads only")
    return edit_heads(adapter, [(layer, int(head))], mode="within_example_mean")


def _namespace_seed(namespace: str, seed: int) -> int:
    digest = hashlib.sha256(f"{namespace}|{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def health_probe_positions(
    sequence: str,
    *,
    readout_position: int,
    corruption_positions: Sequence[int],
    n_probe: int,
    seed: int,
    namespace: str,
) -> list[int]:
    excluded = set(range(readout_position - 3, readout_position + 4))
    for position in corruption_positions:
        excluded.update(range(position - 3, position + 5))
    candidates = [
        position
        for position in range(10, len(sequence) - 10)
        if position not in excluded and sequence[position] in "ACGT"
    ]
    if not candidates:
        return []
    rng = np.random.default_rng(_namespace_seed(namespace, seed))
    count = min(n_probe, len(candidates))
    return sorted(int(value) for value in rng.choice(candidates, count, replace=False))


def top1_nucleotide_token_ids(
    logits: torch.Tensor, base_ids: Mapping[str, int]
) -> torch.Tensor:
    ids = torch.tensor([base_ids[base] for base in "ACGT"], device=logits.device)
    scores = torch.stack([logits[..., base_ids[base]] for base in "ACGT"], dim=-1)
    return ids[scores.argmax(-1)]


def _health_record(hits: Mapping[str, int], totals: Mapping[str, int]) -> dict[str, Any]:
    policies = {
        policy: {
            "hits": int(hits[policy]),
            "total": int(totals[policy]),
            "accuracy": hits[policy] / totals[policy] if totals[policy] else 0.0,
            "ok": bool(
                totals[policy]
                and hits[policy] / totals[policy] >= HEALTH_FLOOR
            ),
        }
        for policy in hits
    }
    return {
        "hits": sum(hits.values()),
        "total": sum(totals.values()),
        "accuracy": min(record["accuracy"] for record in policies.values()),
        "floor": HEALTH_FLOOR,
        "source_policies": policies,
        "ok": all(record["ok"] for record in policies.values()),
    }


class AcceptorArm:
    """The sole model endpoint: masked acceptor G under real/pseudo donor edits."""

    def __init__(self, task, examples, decoy_donors):
        self.task = task
        self.examples = list(examples)
        self.mask = [int(example.next_acceptor_pos) - 1 for example in examples]
        if self.mask != [acceptor_readout(example) for example in examples]:
            raise RuntimeError("acceptor readout helper disagrees with next_acceptor_pos - 1")
        self.partner_positions = [int(example.donor_pos) for example in examples]
        self.pseudo_positions = [int(value) for value in decoy_donors]
        self.clean = [example.sequence for example in examples]
        if any(sequence[position] != "G" for sequence, position in zip(self.clean, self.mask)):
            raise RuntimeError("R3c readout is not the acceptor G")
        if any(
            sequence[position : position + 2] != "GT"
            for sequence, position in zip(self.clean, self.partner_positions)
        ):
            raise RuntimeError("R3c paired corruption target is not GT")
        if any(
            sequence[position : position + 2] != "GT"
            for sequence, position in zip(self.clean, self.pseudo_positions)
        ):
            raise RuntimeError("R3c pseudo corruption target is not GT")
        self.partner_corrupt = [
            substitute(sequence, position, MOTIF_CLEAN_DONOR_ALT)
            for sequence, position in zip(self.clean, self.partner_positions)
        ]
        self.pseudo_corrupt = [
            substitute(sequence, position, MOTIF_CLEAN_DONOR_ALT)
            for sequence, position in zip(self.clean, self.pseudo_positions)
        ]

    def margins(
        self,
        sequences: Sequence[str],
        ctx: Callable[[], AbstractContextManager] = noop,
    ) -> np.ndarray:
        output: list[float] = []
        with torch.no_grad():
            for start in range(0, len(self.mask), self.task.batch):
                sl = slice(start, start + self.task.batch)
                encoded = self.task.encode(list(sequences[sl]), self.mask[sl])
                with ctx():
                    logits = self.task.ad.model(**encoded).logits.float()
                output.extend(self.task.margin(logits, self.mask[sl]).tolist())
        return np.asarray(output, dtype=float)

    def native(self) -> dict[str, np.ndarray]:
        clean = self.margins(self.clean)
        partner = self.margins(self.partner_corrupt)
        pseudo = self.margins(self.pseudo_corrupt)
        return {
            "clean_margin": clean,
            "partner_corrupt_margin": partner,
            "pseudo_corrupt_margin": pseudo,
            "partner": clean - partner,
            "pseudo": clean - pseudo,
        }

    def couplings(
        self, ctx: Callable[[], AbstractContextManager]
    ) -> tuple[np.ndarray, np.ndarray]:
        clean = self.margins(self.clean, ctx)
        return (
            clean - self.margins(self.partner_corrupt, ctx),
            clean - self.margins(self.pseudo_corrupt, ctx),
        )

    def activation_noising_drops(
        self,
        components: Sequence[tuple[str, int, int | None]],
        native_clean_margin: np.ndarray,
    ) -> dict[str, dict[str, np.ndarray]]:
        output = {
            component_name(component): {"partner": [], "pseudo": []}
            for component in components
        }
        with torch.no_grad():
            for start in range(0, len(self.mask), self.task.batch):
                stop = min(start + self.task.batch, len(self.mask))
                sl = slice(start, stop)
                clean_encoded = self.task.encode(self.clean[sl], self.mask[sl])
                sources = {
                    "partner": self.task.encode(
                        self.partner_corrupt[sl], self.mask[sl]
                    ),
                    "pseudo": self.task.encode(self.pseudo_corrupt[sl], self.mask[sl]),
                }
                caches = {
                    endpoint: capture_head_outputs(
                        self.task.ad,
                        lambda encoded=encoded: self.task.ad.model(**encoded),
                    )
                    for endpoint, encoded in sources.items()
                }
                native = np.asarray(native_clean_margin[sl], dtype=float)
                for component in components:
                    name = component_name(component)
                    _, layer, head = component
                    for endpoint, cache in caches.items():
                        with patch_heads(
                            self.task.ad, cache, [(int(layer), int(head))]
                        ):
                            logits = self.task.ad.model(**clean_encoded).logits.float()
                        patched = self.task.margin(logits, self.mask[sl]).cpu().numpy()
                        output[name][endpoint].extend((native - patched).tolist())
        return {
            name: {
                endpoint: np.asarray(values, dtype=float)
                for endpoint, values in endpoints.items()
            }
            for name, endpoints in output.items()
        }

    def bounded_health(
        self,
        components: Sequence[tuple[str, int, int | None]],
        *,
        mode: str,
        seed: int,
    ) -> dict[str, dict[str, Any]]:
        if mode not in MODES:
            raise ValueError(f"unknown intervention mode: {mode}")
        policies = ("mean",) if mode == MEAN_MODE else ("partner", "pseudo")
        hits = {
            component_name(component): {policy: 0 for policy in policies}
            for component in components
        }
        totals = {
            component_name(component): {policy: 0 for policy in policies}
            for component in components
        }
        limit = min(HEALTH_MAX_EXAMPLES, len(self.examples))
        with torch.no_grad():
            for start in range(0, limit, self.task.batch):
                stop = min(start + self.task.batch, limit)
                indices = list(range(start, stop))
                masks = [self.mask[index] for index in indices]
                clean_encoded = self.task.encode(
                    [self.clean[index] for index in indices], masks
                )
                source_encoded = {
                    "partner": self.task.encode(
                        [self.partner_corrupt[index] for index in indices], masks
                    ),
                    "pseudo": self.task.encode(
                        [self.pseudo_corrupt[index] for index in indices], masks
                    ),
                }
                probes: list[list[int]] = []
                for local_index, index in enumerate(indices):
                    example = self.examples[index]
                    positions = health_probe_positions(
                        self.clean[index],
                        readout_position=self.mask[index],
                        corruption_positions=(
                            self.partner_positions[index],
                            self.pseudo_positions[index],
                        ),
                        n_probe=HEALTH_PROBES_PER_EXAMPLE,
                        seed=seed,
                        namespace=f"r3c-health:{example.example_id}",
                    )
                    probes.append(positions)
                    for position in positions:
                        token_position = position + self.task.off
                        clean_encoded["input_ids"][local_index, token_position] = (
                            self.task.mask_id
                        )
                        for encoded in source_encoded.values():
                            encoded["input_ids"][local_index, token_position] = (
                                self.task.mask_id
                            )
                caches = {}
                if mode == NOISING_MODE:
                    caches = {
                        policy: capture_head_outputs(
                            self.task.ad,
                            lambda encoded=source_encoded[policy]: self.task.ad.model(
                                **encoded
                            ),
                        )
                        for policy in policies
                    }
                for component in components:
                    name = component_name(component)
                    _, layer, head = component
                    for policy in policies:
                        ctx = (
                            mean_intervention(self.task.ad, component)
                            if mode == MEAN_MODE
                            else patch_heads(
                                self.task.ad,
                                caches[policy],
                                [(int(layer), int(head))],
                            )
                        )
                        with ctx:
                            logits = self.task.ad.model(**clean_encoded).logits.float()
                        for local_index, (index, positions) in enumerate(
                            zip(indices, probes)
                        ):
                            if not positions:
                                continue
                            token_positions = [
                                position + self.task.off for position in positions
                            ]
                            predicted = top1_nucleotide_token_ids(
                                logits[local_index, token_positions], self.task.bid
                            )
                            target = torch.tensor(
                                [
                                    self.task.bid[self.clean[index][position]]
                                    for position in positions
                                ],
                                device=predicted.device,
                            )
                            hits[name][policy] += int((predicted == target).sum().item())
                            totals[name][policy] += len(positions)
        return {
            name: _health_record(hits[name], totals[name]) for name in hits
        }


def _tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def _encoded_same_example_alignment(
    clean: Mapping[str, torch.Tensor],
    corrupt: Mapping[str, torch.Tensor],
    positions: Sequence[int],
    *,
    prefix_tokens: int,
) -> bool:
    if clean["input_ids"].shape != corrupt["input_ids"].shape:
        return False
    if not torch.equal(clean["attention_mask"], corrupt["attention_mask"]):
        return False
    differing = clean["input_ids"] != corrupt["input_ids"]
    for row, position in enumerate(positions):
        observed = set(torch.nonzero(differing[row], as_tuple=False).flatten().tolist())
        expected = {int(position) + prefix_tokens, int(position) + prefix_tokens + 1}
        if observed != expected:
            return False
    return True


def _measure_hook_controls(
    adapter,
    *,
    sequences: Sequence[str],
    masks: Sequence[int],
    partner_corrupt: Sequence[str],
    pseudo_corrupt: Sequence[str],
    partner_positions: Sequence[int],
    pseudo_positions: Sequence[int],
    example_ids: Sequence[str],
    batch: int,
    check_full_mlm: bool,
) -> dict[str, Any]:
    """Exercise clean-source patches without referring to any frozen candidate head."""
    task = CouplingTask(adapter, batch=batch)
    all_heads = [
        (layer, head)
        for layer in range(adapter.num_layers)
        for head in range(adapter.num_heads)
    ]
    if adapter.num_layers <= 3 or adapter.num_heads != 16:
        raise RuntimeError("R3c live hook capability requires the 6-layer/16-head family")
    all_l3_heads = [(3, head) for head in range(adapter.num_heads)]
    arrays: dict[str, list[float]] = {
        "clean_margin": [],
        "partner_corrupt_margin": [],
        "pseudo_corrupt_margin": [],
        "self_patched_clean_margin": [],
        "partner_restored_margin": [],
        "pseudo_restored_margin": [],
    }
    split_merge_exact = True
    source_alignment_exact = True
    self_patch_max_abs_logit_error = 0.0
    mlm_replay_checked = False
    mlm_full_logits_max_abs_error = 0.0
    mlm_margin_max_abs_error = 0.0
    mlm_examples = 0

    with torch.no_grad():
        for start in range(0, len(sequences), batch):
            stop = min(start + batch, len(sequences))
            sl = slice(start, stop)
            local_masks = list(masks[sl])
            clean_encoded = task.encode(list(sequences[sl]), local_masks)
            partner_encoded = task.encode(list(partner_corrupt[sl]), local_masks)
            pseudo_encoded = task.encode(list(pseudo_corrupt[sl]), local_masks)
            source_alignment_exact = bool(
                source_alignment_exact
                and _encoded_same_example_alignment(
                    clean_encoded,
                    partner_encoded,
                    partner_positions[sl],
                    prefix_tokens=task.off,
                )
                and _encoded_same_example_alignment(
                    clean_encoded,
                    pseudo_encoded,
                    pseudo_positions[sl],
                    prefix_tokens=task.off,
                )
            )

            captured_hidden: dict[str, torch.Tensor] = {}
            cls_module = getattr(adapter.model, "cls", None)
            cls_handle = None
            if check_full_mlm and not mlm_replay_checked:
                if cls_module is None or not hasattr(cls_module, "register_forward_pre_hook"):
                    raise RuntimeError("R3c requires the checkpoint's full model.cls head")

                def capture_cls_input(_module, inputs):
                    if not inputs or not torch.is_tensor(inputs[0]):
                        raise TypeError("model.cls did not receive final hidden states")
                    captured_hidden["value"] = inputs[0].detach().clone()

                cls_handle = cls_module.register_forward_pre_hook(capture_cls_input)

            holder: dict[str, Any] = {}

            def run_clean():
                holder["output"] = adapter.model(**clean_encoded)
                return holder["output"]

            try:
                clean_cache = capture_head_outputs(adapter, run_clean)
            finally:
                if cls_handle is not None:
                    cls_handle.remove()
            clean_logits = holder["output"].logits.float()
            clean_margin = task.margin(clean_logits, local_masks)

            split_merge_exact = bool(
                split_merge_exact
                and all(
                    torch.equal(
                        heads,
                        _split_context(_merge_context(heads), adapter.num_heads),
                    )
                    for heads in clean_cache.values()
                )
            )

            with patch_heads(adapter, clean_cache, all_heads):
                self_logits = adapter.model(**clean_encoded).logits.float()
            with patch_heads(adapter, clean_cache, all_l3_heads):
                partner_restored_logits = adapter.model(**partner_encoded).logits.float()
            with patch_heads(adapter, clean_cache, all_l3_heads):
                pseudo_restored_logits = adapter.model(**pseudo_encoded).logits.float()
            partner_logits = adapter.model(**partner_encoded).logits.float()
            pseudo_logits = adapter.model(**pseudo_encoded).logits.float()

            self_patch_max_abs_logit_error = max(
                self_patch_max_abs_logit_error,
                _tensor_max_abs(clean_logits, self_logits),
            )
            batch_arrays = {
                "clean_margin": clean_margin,
                "partner_corrupt_margin": task.margin(partner_logits, local_masks),
                "pseudo_corrupt_margin": task.margin(pseudo_logits, local_masks),
                "self_patched_clean_margin": task.margin(self_logits, local_masks),
                "partner_restored_margin": task.margin(
                    partner_restored_logits, local_masks
                ),
                "pseudo_restored_margin": task.margin(
                    pseudo_restored_logits, local_masks
                ),
            }
            for name, tensor in batch_arrays.items():
                arrays[name].extend(tensor.detach().cpu().tolist())

            if check_full_mlm and not mlm_replay_checked:
                if "value" not in captured_hidden:
                    raise RuntimeError("R3c failed to capture final model.cls input")
                replayed = cls_module(captured_hidden["value"])
                replayed_logits = (
                    replayed.logits
                    if hasattr(replayed, "logits")
                    else replayed[0]
                    if isinstance(replayed, (tuple, list))
                    else replayed
                )
                if not torch.is_tensor(replayed_logits):
                    raise TypeError("full model.cls replay did not return logits")
                replayed_logits = replayed_logits.float()
                mlm_full_logits_max_abs_error = _tensor_max_abs(
                    clean_logits, replayed_logits
                )
                mlm_margin_max_abs_error = _tensor_max_abs(
                    clean_margin, task.margin(replayed_logits, local_masks)
                )
                mlm_examples = stop - start
                mlm_replay_checked = True

    rendered = {
        name: np.asarray(values, dtype=float) for name, values in arrays.items()
    }
    if any(values.shape != (len(sequences),) for values in rendered.values()):
        raise RuntimeError("R3c hook capability produced incomplete aligned vectors")
    return {
        "arrays": rendered,
        "split_merge_exact": split_merge_exact,
        "source_alignment_exact": source_alignment_exact,
        "example_order_sha256": canonical_sha256(list(map(str, example_ids))),
        "self_patch_max_abs_logit_error": self_patch_max_abs_logit_error,
        "full_mlm_replay": {
            "checked": mlm_replay_checked,
            "examples": mlm_examples,
            "full_logits_max_abs_error": mlm_full_logits_max_abs_error,
            "acceptor_G_margin_max_abs_error": mlm_margin_max_abs_error,
        },
    }


def hook_capability_inference(
    partner_recovery: np.ndarray,
    pseudo_recovery: np.ndarray,
    groups: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int,
) -> dict[str, Any]:
    """Test the all-L3 clean-to-corrupt live path independently of candidate heads."""
    values = np.vstack(
        (
            np.asarray(partner_recovery, dtype=float),
            np.asarray(pseudo_recovery, dtype=float),
        )
    )
    names = ["all_L3_partner_recovery", "all_L3_pseudo_recovery"]
    intervals, calibration = simultaneous_family_decision(
        values,
        groups,
        names,
        sesoi=SESOI,
        n_draws=n_draws,
        alpha=FAMILY_ALPHA,
        seed=seed,
        namespace=f"{namespace}:simultaneous-estimation-envelope",
        min_genes=min_genes,
        studentized=True,
    )
    edge_names = [
        "all_L3_partner_recovery|positive_edge",
        "all_L3_pseudo_recovery|lower_edge",
        "all_L3_pseudo_recovery|upper_edge",
    ]
    # Writing all three edges in one vector call keeps the shared-sign contract visible and
    # prevents helper-level namespace drift between the two TOST edges.
    tests = boundary_sign_flip_tests(
        np.vstack((values[0], values[1], values[1])),
        groups,
        edge_names,
        boundaries=(HOOK_RESTORATION_PARTNER_FLOOR, -SESOI, SESOI),
        alternatives=("greater", "greater", "less"),
        n_draws=n_draws,
        seed=seed,
        namespace=f"{namespace}:shared-vector-boundary-edges",
        min_genes=min_genes,
    )
    sign_digests = {test.sign_matrix_sha256 for test in tests.values()}
    gene_digests = {test.ordered_gene_labels_sha256 for test in tests.values()}
    if len(sign_digests) != 1 or len(gene_digests) != 1:
        raise RuntimeError("R3c hook capability edges did not share one gene sign matrix")
    pseudo_p = intersection_union_pvalue(
        [tests[edge_names[1]].p_value, tests[edge_names[2]].p_value]
    )
    path_p = intersection_union_pvalue(
        [tests[edge_names[0]].p_value, pseudo_p]
    )
    partner_lo, partner_hi = intervals[names[0]].simultaneous_ci
    pseudo_lo, pseudo_hi = intervals[names[1]].simultaneous_ci
    ci_pass = bool(
        partner_lo > HOOK_RESTORATION_PARTNER_FLOOR
        and -SESOI < pseudo_lo <= pseudo_hi < SESOI
    )
    p_pass = bool(path_p < FAMILY_ALPHA)
    if p_pass and ci_pass:
        classification = "LIVE_HOOK_CAPABILITY_CONFIRMED"
    elif p_pass != ci_pass:
        classification = "INCONCLUSIVE_METHOD_DISAGREEMENT"
    else:
        classification = "LIVE_HOOK_CAPABILITY_NOT_CONFIRMED"
    return {
        "partner_floor": HOOK_RESTORATION_PARTNER_FLOOR,
        "pseudo_equivalence_sesoi": SESOI,
        "estimates": {
            name: float(values[index].mean()) for index, name in enumerate(names)
        },
        "simultaneous_cis": {
            name: list(intervals[name].simultaneous_ci) for name in names
        },
        "interval_calibration": dict(calibration),
        "boundary_tests": {name: tests[name].to_dict() for name in edge_names},
        "boundary_randomisation_audit": {
            "all_decision_edges_share_one_sign_matrix": True,
            "B": n_draws,
            "namespace": tests[edge_names[0]].namespace,
            "sign_matrix_sha256": next(iter(sign_digests)),
            "ordered_gene_labels_sha256": next(iter(gene_digests)),
            "edge_order_sha256": canonical_sha256(edge_names),
        },
        "partner_p_value": tests[edge_names[0]].p_value,
        "pseudo_tost_iut_p_value": pseudo_p,
        "capability_iut_p_value": path_p,
        "strict_p_threshold": f"p < {FAMILY_ALPHA}",
        "p_pass": p_pass,
        "strict_simultaneous_ci_pass": ci_pass,
        "classification": classification,
    }


def run_hook_capability(
    adapter,
    arm: AcceptorArm,
    groups: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    min_genes: int,
) -> dict[str, Any]:
    """Run the independent bucket-0 hook battery that a target later inherits."""
    example_ids = [str(example.example_id) for example in arm.examples]
    full = _measure_hook_controls(
        adapter,
        sequences=arm.clean,
        masks=arm.mask,
        partner_corrupt=arm.partner_corrupt,
        pseudo_corrupt=arm.pseudo_corrupt,
        partner_positions=arm.partner_positions,
        pseudo_positions=arm.pseudo_positions,
        example_ids=example_ids,
        batch=arm.task.batch,
        check_full_mlm=True,
    )
    arrays = full["arrays"]
    partner_recovery = (
        arrays["partner_restored_margin"] - arrays["partner_corrupt_margin"]
    )
    pseudo_recovery = (
        arrays["pseudo_restored_margin"] - arrays["pseudo_corrupt_margin"]
    )
    restoration_statistics = gene_sufficient_statistics(
        np.vstack((partner_recovery, pseudo_recovery)),
        groups,
        ["all_L3_partner_recovery", "all_L3_pseudo_recovery"],
    )
    restoration_values, restoration_groups = _reconstruct_cluster_array(
        restoration_statistics
    )
    inference = hook_capability_inference(
        restoration_values[0],
        restoration_values[1],
        restoration_groups,
        n_draws=n_draws,
        seed=seed,
        namespace="r3c:bucket0-independent-live-hook-capability",
        min_genes=min_genes,
    )

    limit = min(HOOK_BATCH_INVARIANCE_EXAMPLES, len(arm.clean))
    subset = {
        "sequences": arm.clean[:limit],
        "masks": arm.mask[:limit],
        "partner_corrupt": arm.partner_corrupt[:limit],
        "pseudo_corrupt": arm.pseudo_corrupt[:limit],
        "partner_positions": arm.partner_positions[:limit],
        "pseudo_positions": arm.pseudo_positions[:limit],
        "example_ids": example_ids[:limit],
    }
    batch_one = _measure_hook_controls(
        adapter, **subset, batch=1, check_full_mlm=False
    )
    batch_many = _measure_hook_controls(
        adapter, **subset, batch=max(1, min(PRODUCTION_BATCH, limit)), check_full_mlm=False
    )
    batch_differences = {
        name: float(
            np.max(np.abs(batch_one["arrays"][name] - batch_many["arrays"][name]))
        )
        for name in batch_one["arrays"]
    }
    batch_max = max(batch_differences.values(), default=0.0)
    self_margin_error = float(
        np.max(
            np.abs(
                arrays["clean_margin"] - arrays["self_patched_clean_margin"]
            )
        )
    )
    controls = {
        "clean_to_clean_whole_head_self_patch": bool(
            self_margin_error <= HOOK_SELF_PATCH_ATOL
            and full["self_patch_max_abs_logit_error"] <= HOOK_SELF_PATCH_ATOL
        ),
        "post_attention_head_split_merge_identity": bool(full["split_merge_exact"]),
        "same_example_source_target_alignment": bool(
            full["source_alignment_exact"]
            and batch_one["source_alignment_exact"]
            and batch_many["source_alignment_exact"]
            and batch_one["example_order_sha256"]
            == batch_many["example_order_sha256"]
        ),
        "batch_size_invariance": bool(batch_max <= HOOK_BATCH_INVARIANCE_ATOL),
        "final_hidden_through_full_model_cls_reproduces_logits_and_margin": bool(
            full["full_mlm_replay"]["checked"]
            and full["full_mlm_replay"]["full_logits_max_abs_error"]
            <= HOOK_SELF_PATCH_ATOL
            and full["full_mlm_replay"]["acceptor_G_margin_max_abs_error"]
            <= HOOK_SELF_PATCH_ATOL
        ),
        "all_L3_clean_to_corrupt_live_restoration": (
            inference["classification"] == "LIVE_HOOK_CAPABILITY_CONFIRMED"
        ),
    }
    return {
        "schema": "gfmcd.r3c.independent_hook_capability/1",
        "candidate_outcomes_used": False,
        "source_direction": "clean -> same-example paired/pseudo donor corrupt target",
        "patched_heads": "all 16 heads in layer 3",
        "self_patch_scope": "all heads in all layers, clean -> identical clean",
        "n_examples": len(arm.clean),
        "n_genes": len(set(map(str, groups))),
        "example_order_sha256": full["example_order_sha256"],
        "same_example_alignment": {
            "exact": full["source_alignment_exact"],
            "rule": "same row/order; identical attention mask; only the declared two donor tokens differ",
        },
        "split_merge": {
            "exact": full["split_merge_exact"],
            "scope": "captured post-attention per-head contexts at every layer",
        },
        "self_patch": {
            "margin_max_abs_error": self_margin_error,
            "full_logits_max_abs_error": full["self_patch_max_abs_logit_error"],
            "atol": HOOK_SELF_PATCH_ATOL,
        },
        "batch_invariance": {
            "n_examples": limit,
            "batch_sizes": [1, max(1, min(PRODUCTION_BATCH, limit))],
            "max_abs_differences": batch_differences,
            "maximum": batch_max,
            "atol": HOOK_BATCH_INVARIANCE_ATOL,
        },
        "full_mlm_replay": full["full_mlm_replay"] | {"atol": HOOK_SELF_PATCH_ATOL},
        "all_L3_restoration": inference
        | {
            "gene_sufficient_statistics": restoration_statistics
        },
        "a_at_v_scope": (
            "post-attention head context split/merge identity only; explicit A@V value-product "
            "reconstruction is deferred to R5a"
        ),
        "controls": controls,
        "all_controls_pass": bool(all(controls.values())),
    }


def construction_record(examples, arm: AcceptorArm) -> dict[str, Any]:
    rows = []
    for index, example in enumerate(examples):
        rows.append(
            {
                "example_id": example.example_id,
                "gene_id": example.gene_id,
                "readout": {
                    "role": "downstream_intron_mate_acceptor_G",
                    "position": arm.mask[index],
                    "expected_position": int(example.next_acceptor_pos) - 1,
                    "base": example.sequence[arm.mask[index]],
                },
                "paired_donor": {
                    "position": arm.partner_positions[index],
                    "clean_motif": example.sequence[
                        arm.partner_positions[index] : arm.partner_positions[index] + 2
                    ],
                    "alt": MOTIF_CLEAN_DONOR_ALT,
                    "planted_motifs": planted_donor_motifs(
                        example.sequence,
                        arm.partner_positions[index],
                        MOTIF_CLEAN_DONOR_ALT,
                    ),
                    "corrupt_sequence_sha256": hashlib.sha256(
                        arm.partner_corrupt[index].encode()
                    ).hexdigest(),
                },
                "pseudo_donor": {
                    "position": arm.pseudo_positions[index],
                    "clean_motif": example.sequence[
                        arm.pseudo_positions[index] : arm.pseudo_positions[index] + 2
                    ],
                    "alt": MOTIF_CLEAN_DONOR_ALT,
                    "planted_motifs": planted_donor_motifs(
                        example.sequence,
                        arm.pseudo_positions[index],
                        MOTIF_CLEAN_DONOR_ALT,
                    ),
                    "corrupt_sequence_sha256": hashlib.sha256(
                        arm.pseudo_corrupt[index].encode()
                    ).hexdigest(),
                },
            }
        )
    policy = {
        "readout": "mask exactly one G at next_acceptor_pos - 1",
        "paired_operator": "GT -> AC at donor_pos",
        "pseudo_operator": "GT -> AC at matched_decoy_donor",
        "pseudo_distance_caliper": DECOY_TOL,
        "donor_readout_constructed": False,
    }
    return {
        "schema": "gfmcd.r3c.acceptor_only_construction/1",
        "policy": policy,
        "policy_sha256": canonical_sha256(policy),
        "rows": len(rows),
        "ordered_rows_sha256": canonical_sha256(rows),
        "row_sha256": [
            {"example_id": row["example_id"], "sha256": canonical_sha256(row)}
            for row in rows
        ],
        "all_single_acceptor_G": all(
            row["readout"]["base"] == "G"
            and row["readout"]["position"] == row["readout"]["expected_position"]
            for row in rows
        ),
        "all_donor_targets_GT": all(
            row["paired_donor"]["clean_motif"] == "GT"
            and row["pseudo_donor"]["clean_motif"] == "GT"
            for row in rows
        ),
        "all_motif_clean": all(
            not row["paired_donor"]["planted_motifs"]
            and not row["pseudo_donor"]["planted_motifs"]
            for row in rows
        ),
    }


def gene_sufficient_statistics(
    values: np.ndarray, groups: Sequence[str], names: Sequence[str]
) -> dict[str, Any]:
    family = np.asarray(values, dtype=float)
    if family.ndim != 2 or family.shape != (len(names), len(groups)):
        raise ValueError("family, row names, and gene labels disagree")
    rendered = [str(group) for group in groups]
    labels = sorted(set(rendered))
    lookup = {label: index for index, label in enumerate(labels)}
    index = np.asarray([lookup[label] for label in rendered], dtype=int)
    counts = np.bincount(index, minlength=len(labels))
    sums = np.vstack(
        [np.bincount(index, weights=row, minlength=len(labels)) for row in family]
    )
    records = []
    for gene_index, gene_id in enumerate(labels):
        records.append(
            {
                "gene_id": gene_id,
                "n_examples": int(counts[gene_index]),
                "raw_component_sums": [
                    float(value) for value in sums[:, gene_index]
                ],
            }
        )
    return {
        "schema": "gene-cluster-raw-sums-v1",
        "row_names": list(names),
        "genes": records,
    }


def snapshot_input_paths(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"bound input is missing or not a file: {path}")
        record = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        prior = snapshot.setdefault(str(path), record)
        if prior != record:
            raise RuntimeError(f"bound input changed during snapshot: {path}")
    return dict(sorted(snapshot.items()))


def snapshot_record(
    snapshot: Mapping[str, Mapping[str, Any]], path: str | Path
) -> dict[str, Any]:
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in snapshot:
        raise RuntimeError(f"path is absent from input snapshot: {resolved}")
    return dict(snapshot[resolved])


def verify_input_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    for raw, expected in snapshot.items():
        path = Path(raw)
        if (
            not path.is_file()
            or path.stat().st_size != expected["size_bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise RuntimeError(f"bound input changed during execution: {path}")


def atomic_write_new(path: str | Path, data: bytes) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact {destination}")
    temporary = destination.parent / (
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, destination, follow_symlinks=False)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
    return destination


def checkpoint_files(model: str) -> list[Path]:
    root = Path(model).expanduser()
    if not root.is_absolute():
        root = REPO / root
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(path for path in root.rglob("*") if path.is_file())
    else:
        raise RuntimeError(f"checkpoint path is missing: {root}")
    if not files:
        raise RuntimeError("checkpoint bundle contains no regular files")
    return files


def resolved_model_root(model: str) -> Path:
    root = Path(model).expanduser()
    if not root.is_absolute():
        root = REPO / root
    return root.resolve()


def checkpoint_bundle_from_snapshot(
    model: str,
    model_files: Sequence[Path],
    snapshot: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    root = resolved_model_root(model)
    bundle_root = root if root.is_dir() else root.parent
    records = [
        {
            "path": str(path.resolve()),
            **snapshot_record(snapshot, path),
            "relative_path": str(path.resolve().relative_to(bundle_root.resolve())),
        }
        for path in model_files
    ]
    digest = canonical_sha256(
        [
            {
                "relative_path": record["relative_path"],
                "size_bytes": record["size_bytes"],
                "sha256": record["sha256"],
            }
            for record in records
        ]
    )
    return records, digest


def stage_checkpoint_bundle(
    model: str,
    model_files: Sequence[Path],
    snapshot: Mapping[str, Mapping[str, Any]],
) -> tuple[tempfile.TemporaryDirectory, Path, str]:
    source_root = resolved_model_root(model)
    staging = tempfile.TemporaryDirectory(prefix="gfmcd-r3c-checkpoint-")
    destination = Path(staging.name) / (
        "model" if source_root.is_dir() else source_root.name
    )
    if source_root.is_dir():
        destination.mkdir(parents=True)
        for source in model_files:
            target = destination / source.resolve().relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    else:
        shutil.copyfile(source_root, destination)
    _, expected = checkpoint_bundle_from_snapshot(model, model_files, snapshot)
    staged_files = checkpoint_files(str(destination))
    staged_root = destination if destination.is_dir() else destination.parent
    staged = [
        {
            "relative_path": str(path.resolve().relative_to(staged_root.resolve())),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in staged_files
    ]
    if canonical_sha256(staged) != expected:
        staging.cleanup()
        raise RuntimeError("staged checkpoint differs from bound source bytes")
    return staging, destination, expected


def runtime_identity(device: str | None = None) -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "scipy", "torch", "transformers", "tokenizers", "pyfaidx"):
        try:
            packages[name] = package_version(name)
        except PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "torch_runtime": {
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn_runtime": torch.backends.cudnn.version(),
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "device": {"selected": device},
    }


def normalized_runner_record(
    path: str | Path = RUNNER,
    *,
    snapshot: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    data = resolved.read_bytes()
    if snapshot is not None:
        expected = snapshot_record(snapshot, resolved)
        if len(data) != expected["size_bytes"] or hashlib.sha256(data).hexdigest() != expected[
            "sha256"
        ]:
            raise RuntimeError("runner bytes changed after snapshot")
    text = data.decode()
    normalized = re.sub(
        r"^(FROZEN_(?:PROTOCOL|INSTRUMENT_FINGERPRINT|PREFLIGHT_ARTIFACT)_SHA256):[^=]*=.*$",
        r'\1: str | None = "<SEAL_LITERAL>"',
        text,
        flags=re.MULTILINE,
    ).encode()
    return {
        "path": str(resolved),
        "normalization": "replace only FROZEN_*_SHA256 assignment values",
        "normalized_size_bytes": len(normalized),
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
    }


def exact_hash_bundle(
    model: str,
    gtf: str,
    fasta: str,
    *,
    snapshot: Mapping[str, Mapping[str, Any]],
    model_files: Sequence[Path],
) -> dict[str, Any]:
    checkpoint_records, checkpoint_digest = checkpoint_bundle_from_snapshot(
        model, model_files, snapshot
    )
    sources = {
        "gtf": {"path": str(Path(gtf).resolve()), **snapshot_record(snapshot, gtf)},
        "fasta": {
            "path": str(Path(fasta).resolve()),
            **snapshot_record(snapshot, fasta),
        },
    }
    bound = {
        str(path.relative_to(REPO)): {
            "path": str(path.resolve()),
            **snapshot_record(snapshot, path),
        }
        for path in OTHER_BOUND_SOURCES
    }
    runner = normalized_runner_record(snapshot=snapshot)
    return {
        "checkpoint_files": checkpoint_records,
        "checkpoint_aggregate_sha256": checkpoint_digest,
        "sources": sources,
        "source_aggregate_sha256": canonical_sha256(
            {key: value["sha256"] for key, value in sources.items()}
        ),
        "bound_sources": bound,
        "bound_sources_aggregate_sha256": canonical_sha256(
            {key: value["sha256"] for key, value in bound.items()}
        ),
        "protocol_sha256": snapshot_record(snapshot, PROTOCOL)["sha256"],
        "runner": runner,
        "runner_sha256": runner["normalized_sha256"],
        "stat_core_sha256": snapshot_record(snapshot, STAT_CORE)["sha256"],
        "hook_core_sha256": snapshot_record(snapshot, HOOK_CORE)["sha256"],
        "assay_core_sha256": snapshot_record(snapshot, ASSAY_CORE)["sha256"],
        "circuit_core_sha256": snapshot_record(snapshot, CIRCUIT_CORE)["sha256"],
    }


def instrument_binding_payload(
    hashes: Mapping[str, Any],
    *,
    batch: int,
    device: str,
    seed: int,
    n_randomisation: int,
    n_trials: int,
) -> dict[str, Any]:
    return {
        "schema": "r3c-acceptor-head-confirmation-instrument-v2",
        "result_schema": RESULT_SCHEMA,
        "instrument_revision": INSTRUMENT_REVISION,
        "hashes": {
            key: hashes[key]
            for key in (
                "checkpoint_aggregate_sha256",
                "source_aggregate_sha256",
                "bound_sources_aggregate_sha256",
                "protocol_sha256",
                "runner_sha256",
                "stat_core_sha256",
                "hook_core_sha256",
                "assay_core_sha256",
                "circuit_core_sha256",
            )
        },
        "component_order": [component_name(component) for component in COMPONENTS],
        "component_kinds": [component[0] for component in COMPONENTS],
        "modes": list(MODES),
        "primitive_endpoints": list(PRIMITIVE_ENDPOINTS),
        "derived_endpoints": list(ENDPOINTS),
        "primitive_row_names": primitive_row_names(),
        "family_row_names": family_row_names(),
        "path_specs": named_path_specs(),
        "path_names": list(PATH_NAMES),
        "primary_units": "masked-acceptor-G logit-margin component drop",
        "readout": "next_acceptor_pos - 1 only",
        "operator": "paired and pseudo donor GT -> AC",
        "sesoi": SESOI,
        "family_alpha": FAMILY_ALPHA,
        "strict_path_p_threshold": "Holm-adjusted p < alpha",
        "interval_rule": "strict_interval_decision on 62-row simultaneous envelope",
        "interval_scaling": "fixed endpoint bootstrap SD max-deviation",
        "interval_historical_classifier_used": False,
        "minimum_genes": MIN_GENES,
        "production_stage_sizes": {"preflight": PREFLIGHT_N, "target": TARGET_N},
        "batch": batch,
        "device": device,
        "seed": seed,
        "n_randomisation": n_randomisation,
        "n_trials": n_trials,
        "plant_delta": PLANT_DELTA,
        "coherent_primitive_plant": coherent_path_offset(PLANT_DELTA).tolist(),
        "boundary_face_suite": boundary_face_suite_binding(),
        "outer_calibration": (
            "center primitive rows, whole-gene shared sign flip, derive family afterward"
        ),
        "power_floor": PREFLIGHT_POWER_FLOOR,
        "gene_partition": "sha256('r3b-gene|' + gene_id) mod 2",
        "current_tag": CURRENT_TAG,
        "validation_chromosomes": list(VALIDATION_CHROMOSOMES),
        "historical_identity": {
            "rows": HISTORICAL_EXPECTED_ROWS,
            "genes": HISTORICAL_EXPECTED_GENES,
            "example_id_sha256": HISTORICAL_EXPECTED_EXAMPLE_ID_SHA256,
        },
        "current_census_identity": CURRENT_EXPECTED_CENSUS_IDENTITY,
        "current_eligibility_requires_only_donor_pseudo": True,
        "donor_pseudo_caliper": DECOY_TOL,
        "health_policy": {
            "maximum_examples": HEALTH_MAX_EXAMPLES,
            "probes_per_example": HEALTH_PROBES_PER_EXAMPLE,
            "floor": HEALTH_FLOOR,
            "top1_vocabulary": ["A", "C", "G", "T"],
            "mean_source": "clean",
            "noising_sources": ["paired_donor_corrupt", "pseudo_donor_corrupt"],
            "suppression": "affected component/mode and dependent paths only",
            "synthetic_calibration": "all seven paths health-enabled",
            "target_authorization": (
                "reviewed preflight health AND current target health; target can only shrink"
            ),
            "reviewed_v1_suppressed_paths": list(
                HISTORICAL_V1_HEALTH_SUPPRESSED_PATHS
            ),
        },
        "independent_hook_capability_policy": {
            "candidate_outcomes_used": False,
            "clean_to_clean_self_patch_atol": HOOK_SELF_PATCH_ATOL,
            "batch_invariance_atol": HOOK_BATCH_INVARIANCE_ATOL,
            "batch_invariance_examples": HOOK_BATCH_INVARIANCE_EXAMPLES,
            "all_L3_partner_recovery_floor": HOOK_RESTORATION_PARTNER_FLOOR,
            "all_L3_pseudo_equivalence_sesoi": SESOI,
            "source_direction": "clean -> same-example paired/pseudo donor corrupt target",
            "full_mlm_replay": "final hidden through full model.cls",
            "a_at_v_scope": "split/merge identity now; explicit value-product reconstruction deferred to R5a",
            "target_policy": "inherit bound bucket-0 capability; never gate on candidate path outcomes",
        },
        "runtime": runtime_identity(device),
    }


def authorize_target(
    protocol_sha256: str,
    *,
    protocol_text: str,
    frozen_protocol_sha256: str | None = FROZEN_PROTOCOL_SHA256,
    frozen_preflight_artifact_sha256: str | None = FROZEN_PREFLIGHT_ARTIFACT_SHA256,
) -> None:
    if frozen_protocol_sha256 is None:
        raise TargetSealError(
            "R3c is DRAFT and target execution is hard-blocked; review and freeze the "
            "protocol and instrument before opening bucket 1."
        )
    status = [line.strip() for line in protocol_text.splitlines()[:12]]
    if not any(line.startswith("> **Status: FROZEN") for line in status):
        raise TargetSealError("R3c protocol does not declare FROZEN status")
    if protocol_sha256 != frozen_protocol_sha256:
        raise TargetSealError("R3c protocol bytes differ from the frozen pin")
    if frozen_preflight_artifact_sha256 is None:
        raise TargetSealError(
            "R3c target requires a literal reviewed preflight-artifact SHA-256 pin"
        )
    if re.fullmatch(r"[0-9a-f]{64}", frozen_preflight_artifact_sha256) is None:
        raise TargetSealError("R3c preflight-artifact SHA-256 pin is malformed")


def verify_frozen_instrument(
    fingerprint: str,
    *,
    frozen_fingerprint: str | None = FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
) -> None:
    if frozen_fingerprint is None:
        raise TargetSealError("R3c target requires a literal instrument fingerprint pin")
    if fingerprint != frozen_fingerprint:
        raise TargetSealError("R3c instrument fingerprint differs from the frozen pin")


def authorize_bindable_preflight(
    protocol_sha256: str,
    fingerprint: str,
    *,
    protocol_text: str,
    frozen_protocol_sha256: str | None = FROZEN_PROTOCOL_SHA256,
    frozen_fingerprint: str | None = FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
) -> bool:
    if frozen_protocol_sha256 is None and frozen_fingerprint is None:
        return False
    if frozen_protocol_sha256 is None or frozen_fingerprint is None:
        raise TargetSealError("R3c bindable preflight requires both literal pins")
    authorize_target(
        protocol_sha256,
        protocol_text=protocol_text,
        frozen_protocol_sha256=frozen_protocol_sha256,
        # A bindable bucket-0 run is the artifact that will subsequently receive this
        # third literal pin; it cannot require its own not-yet-existing byte hash.
        frozen_preflight_artifact_sha256="0" * 64,
    )
    verify_frozen_instrument(fingerprint, frozen_fingerprint=frozen_fingerprint)
    return True


def require_clean_git(repo: Path = REPO) -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if status.returncode or status.stdout.strip():
        raise RuntimeError("R3c target authorization requires a clean worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if commit.returncode or re.fullmatch(r"[0-9a-f]{40}", commit.stdout.strip()) is None:
        raise RuntimeError("R3c target authorization requires a full git identity")
    return {"commit": commit.stdout.strip(), "dirty": False}


def _require_preflight(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"R3c preflight semantic validation failed: {message}")


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _ordered_gene_labels_sha256(labels: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(f"utf8-sequence|{len(labels)}|".encode())
    for label in labels:
        encoded = str(label).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _expected_sign_matrix_sha256(
    *, namespace: str, seed: int, B: int, n_genes: int
) -> str:
    raw_seed = hashlib.sha256(f"{namespace}|{int(seed)}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(raw_seed[:8], "big", signed=False))
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(B, n_genes))
    canonical = np.ascontiguousarray(signs, dtype=np.int8)
    shape = ",".join(str(dimension) for dimension in canonical.shape)
    digest = hashlib.sha256()
    digest.update(f"int8|{shape}|".encode())
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _decision_edge_order() -> list[str]:
    kinds = _required_row_test_kinds()
    positives = [name for name in family_row_names() if kinds.get(name) == "positive_boundary"]
    equivalents = [
        name for name in family_row_names() if kinds.get(name) == "tost_equivalence_iut"
    ]
    return [f"{name}|positive_edge" for name in positives] + [
        edge
        for name in equivalents
        for edge in (f"{name}|lower_edge", f"{name}|upper_edge")
    ]


def _validate_shared_boundary_audit(
    audit: Any,
    *,
    namespace: str,
    B: int,
    ordered_gene_digest: str,
    n_genes: int,
    replay_sign_digest: bool = True,
) -> None:
    _require_preflight(isinstance(audit, dict), "shared boundary audit is missing")
    _require_preflight(
        audit.get("all_decision_edges_share_one_sign_matrix") is True,
        "decision-bearing edges did not attest one shared sign matrix",
    )
    _require_preflight(audit.get("B") == B, "shared boundary draw count differs")
    _require_preflight(
        audit.get("namespace") == namespace,
        "shared boundary namespace is non-canonical",
    )
    _require_preflight(
        audit.get("edge_count") == len(_decision_edge_order()),
        "shared boundary edge count differs",
    )
    _require_preflight(
        audit.get("edge_order_sha256") == canonical_sha256(_decision_edge_order()),
        "shared boundary edge order differs",
    )
    _require_preflight(
        _is_sha256(audit.get("sign_matrix_sha256")),
        "shared sign-matrix digest is missing",
    )
    if replay_sign_digest:
        _require_preflight(
            audit.get("sign_matrix_sha256")
            == _expected_sign_matrix_sha256(
                namespace=namespace,
                seed=PRODUCTION_SEED,
                B=B,
                n_genes=n_genes,
            ),
            "shared sign-matrix digest does not replay from seed/namespace/gene order",
        )
    _require_preflight(
        audit.get("ordered_gene_labels_sha256") == ordered_gene_digest,
        "shared boundary ordered-gene digest differs",
    )


def _validate_gene_statistics(
    stats: Any,
    *,
    names: Sequence[str],
    gene_ids: Sequence[str],
    n_examples: int,
) -> list[float]:
    _require_preflight(isinstance(stats, dict), "gene sufficient statistics are missing")
    _require_preflight(
        stats.get("schema") == "gene-cluster-raw-sums-v1"
        and stats.get("row_names") == list(names),
        "gene sufficient-statistic schema or row order differs",
    )
    genes = stats.get("genes")
    _require_preflight(isinstance(genes, list), "gene sufficient-statistic rows are missing")
    _require_preflight(
        [entry.get("gene_id") for entry in genes] == list(gene_ids),
        "gene sufficient-statistic identities differ from the cohort",
    )
    totals = np.zeros(len(names), dtype=float)
    total_examples = 0
    for entry in genes:
        count = entry.get("n_examples")
        sums = entry.get("raw_component_sums")
        _require_preflight(
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            and isinstance(sums, list)
            and len(sums) == len(names)
            and all(_is_finite_number(value) for value in sums),
            "gene sufficient-statistic row is malformed",
        )
        total_examples += count
        totals += np.asarray(sums, dtype=float)
    _require_preflight(total_examples == n_examples, "gene row counts do not sum to n")
    return (totals / n_examples).tolist()


def _reconstruct_cluster_array(stats: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Reconstruct an inference-equivalent row array from exact gene counts and sums."""
    genes = stats["genes"]
    n_rows = len(stats["row_names"])
    columns: list[np.ndarray] = []
    groups: list[str] = []
    for entry in genes:
        count = int(entry["n_examples"])
        sums = np.asarray(entry["raw_component_sums"], dtype=float)
        # All R3c inference depends only on gene size and raw cluster sums.  Placing the
        # exact sum in one member and zeros in the remaining members preserves both without
        # introducing mean-times-count rounding.
        block = np.zeros((n_rows, count), dtype=float)
        block[:, 0] = sums
        columns.append(block)
        groups.extend([str(entry["gene_id"])] * count)
    return np.concatenate(columns, axis=1), groups


def _validate_observed_inference_replay(
    *,
    stats: Mapping[str, Any],
    stored_inference: Any,
    stored_paths: Any,
    stored_path_family: Any,
    path_health: Mapping[str, Mapping[str, Any]],
    n_draws: int,
    seed: int,
    namespace: str,
    min_genes: int,
) -> dict[str, Any]:
    values, groups = _reconstruct_cluster_array(stats)
    replay = strict_path_family_inference(
        values,
        groups,
        family_row_names(),
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=min_genes,
        path_health=path_health,
    )
    embedded = {
        key: value
        for key, value in replay.items()
        if key not in {"paths", "row_names", "row_order_sha256"}
    }
    _require_preflight(
        stored_inference == embedded
        and stored_paths == replay["paths"]
        and stored_path_family == replay["path_family"],
        "observed inference does not replay from exact gene sufficient statistics",
    )
    return replay


def _validate_compact_classifier_audit(
    audit: Any,
    *,
    classifier_namespace: str,
    n_examples: int,
    n_genes: int,
    n_draws: int,
    ordered_gene_digest: str,
    expected_health: Mapping[str, bool],
    replay_sign_digest: bool = False,
) -> dict[str, bool]:
    _require_preflight(isinstance(audit, dict), "calibration classifier audit is missing")
    _validate_shared_boundary_audit(
        audit.get("boundary_randomisation_audit"),
        namespace=f"{classifier_namespace}:shared-vector-boundary-edges",
        B=n_draws,
        ordered_gene_digest=ordered_gene_digest,
        n_genes=n_genes,
        replay_sign_digest=replay_sign_digest,
    )
    interval = audit.get("interval_calibration")
    expected_interval = {
        "n_components": 62,
        "n_examples": n_examples,
        "n_genes": n_genes,
        "minimum_genes_required": MIN_GENES,
        "n_bootstrap_draws": n_draws,
        "n_sign_flip_draws": n_draws,
        "alpha": FAMILY_ALPHA,
        "sesoi": SESOI,
        "studentized": True,
    }
    _require_preflight(interval == expected_interval, "calibration interval contract differs")
    family = audit.get("path_family")
    _require_preflight(
        isinstance(family, dict)
        and family.get("path_names") == list(PATH_NAMES)
        and family.get("alpha") == FAMILY_ALPHA,
        "calibration Holm family differs",
    )
    raw = family.get("raw_iut_p_values")
    adjusted = family.get("holm_adjusted_p_values")
    _require_preflight(
        isinstance(raw, dict)
        and tuple(raw) == PATH_NAMES
        and all(_is_finite_number(value) and 0 <= value <= 1 for value in raw.values()),
        "calibration path IUT p-values are malformed",
    )
    expected_adjusted = holm_adjust_named(raw)
    _require_preflight(
        isinstance(adjusted, dict)
        and tuple(adjusted) == PATH_NAMES
        and all(
            np.isclose(adjusted[name], expected_adjusted[name], rtol=0.0, atol=1e-15)
            for name in PATH_NAMES
        ),
        "calibration Holm values are inconsistent",
    )
    paths = audit.get("paths")
    _require_preflight(
        isinstance(paths, dict) and tuple(paths) == PATH_NAMES,
        "calibration classifier lacks the seven path records",
    )
    supported: dict[str, bool] = {}
    for name in PATH_NAMES:
        path = paths[name]
        _require_preflight(isinstance(path, dict), f"calibration {name} record is malformed")
        p_pass = bool(adjusted[name] < FAMILY_ALPHA)
        ci_pass = path.get("strict_simultaneous_ci_pass")
        health_ok = expected_health[name]
        expected_class = (
            "HEALTH_FAILED"
            if not health_ok
            else "SUPPORTED"
            if p_pass and ci_pass is True
            else "INCONCLUSIVE_METHOD_DISAGREEMENT"
            if p_pass != (ci_pass is True)
            else "NOT_SUPPORTED_AT_FROZEN_THRESHOLDS"
        )
        _require_preflight(
            _is_finite_number(path.get("iut_p_value"))
            and np.isclose(path["iut_p_value"], raw[name], rtol=0.0, atol=1e-15)
            and _is_finite_number(path.get("holm_adjusted_p"))
            and np.isclose(
                path["holm_adjusted_p"], adjusted[name], rtol=0.0, atol=1e-15
            )
            and path.get("holm_family_size") == 7
            and path.get("p_pass") is p_pass
            and isinstance(ci_pass, bool)
            and path.get("health_ok") is health_ok
            and path.get("classification") == expected_class,
            f"calibration {name} decision is internally inconsistent",
        )
        supported[name] = expected_class == "SUPPORTED"
    return supported


def _validate_all_path_calibration_health(
    calibration: Mapping[str, Any], expected_health: Mapping[str, bool]
) -> None:
    """Reject any production calibration that suppressed a frozen path by live health."""
    planted = calibration.get("planted_power_by_path")
    _require_preflight(
        tuple(expected_health) == PATH_NAMES
        and all(expected_health[name] is True for name in PATH_NAMES)
        and calibration.get("health_eligible_paths") == list(PATH_NAMES)
        and isinstance(planted, Mapping)
        and tuple(planted) == PATH_NAMES
        and all(planted[name].get("health_eligible") is True for name in PATH_NAMES),
        "synthetic calibration must keep all seven paths health-enabled",
    )


def _validate_production_path_preflight(
    calibration: Any,
    *,
    n_examples: int,
    n_genes: int,
    ordered_gene_digest: str,
    expected_health: Mapping[str, bool],
) -> None:
    _require_preflight(isinstance(calibration, dict), "production calibration is missing")
    _validate_all_path_calibration_health(calibration, expected_health)
    suite = boundary_face_surface_specs()
    _require_preflight(
        calibration.get("n_calibration") == PRODUCTION_RANDOMISATION
        and calibration.get("n_trials") == PRODUCTION_TRIALS
        and calibration.get("n_primitive_rows") == 40
        and calibration.get("n_family_rows") == 62
        and calibration.get("n_examples") == n_examples
        and calibration.get("n_genes") == n_genes
        and calibration.get("production_classifier_calls")
        == (len(suite) + 1) * PRODUCTION_TRIALS,
        "production calibration dimensions or call count differ",
    )
    _require_preflight(
        calibration.get("boundary_face_case_order") == list(suite)
        and calibration.get("boundary_face_suite_binding")
        == boundary_face_suite_binding(),
        "boundary-face suite order or binding differs",
    )
    _require_preflight(
        calibration.get("analytic_fwer_argument")
        == (
            "under any true path null at least one positive-boundary or TOST-union-null "
            "constituent is valid and super-uniform; the path IUT maximum is therefore "
            "valid, Holm strongly controls the seven-path FWER under arbitrary dependence, "
            "and strict-CI concordance plus health gates can only remove declarations"
        )
        and calibration.get("boundary_face_role")
        == (
            "prespecified end-to-end implementation diagnostics, not an assertion that "
            "one nuisance midpoint globally maximizes declaration probability"
        ),
        "analytic FWER argument or boundary-face role differs",
    )
    cases = calibration.get("boundary_face_cases")
    _require_preflight(
        isinstance(cases, dict) and set(cases) == set(suite),
        "boundary-face case family differs",
    )
    seen_namespaces: set[str] = set()
    pattern_digests: dict[str, str] = {}
    for case_name in suite:
        spec = suite[case_name]
        case = cases[case_name]
        _require_preflight(
            isinstance(case, dict)
            and case.get("assigned_paths") == spec["assigned_paths"]
            and case.get("targeted_boundaries") == spec["targeted_boundaries"]
            and case.get("offset_sha256") == spec["offset_sha256"]
            and case.get("rationale") == spec["rationale"],
            f"boundary-face metadata differs for {case_name}",
        )
        _require_preflight(
            case.get("outer_namespace_template")
            == f"r3c:production-path-preflight:boundary-face:{case_name}:outer:<trial>"
            and case.get("classifier_namespace_template")
            == f"r3c:production-path-preflight:boundary-face:{case_name}:classifier:<trial>",
            f"boundary-face namespaces differ for {case_name}",
        )
        records = case.get("classifier_records_by_trial")
        _require_preflight(
            isinstance(records, list) and len(records) == PRODUCTION_TRIALS,
            f"boundary-face classifier records are incomplete for {case_name}",
        )
        _require_preflight(
            case.get("classifier_records_sha256") == canonical_sha256(records)
            and case.get("classifier_audit_digest_sha256")
            == canonical_sha256([canonical_sha256(item) for item in records]),
            f"boundary-face classifier digests differ for {case_name}",
        )
        path_counts = {name: 0 for name in spec["assigned_paths"]}
        pattern: list[int] = []
        for trial, classifier in enumerate(records):
            classifier_namespace = (
                f"r3c:production-path-preflight:boundary-face:{case_name}:"
                f"classifier:{trial}"
            )
            _require_preflight(
                classifier_namespace not in seen_namespaces,
                "calibration classifier namespaces are not unique",
            )
            seen_namespaces.add(classifier_namespace)
            supported = _validate_compact_classifier_audit(
                classifier,
                classifier_namespace=classifier_namespace,
                n_examples=n_examples,
                n_genes=n_genes,
                n_draws=PRODUCTION_RANDOMISATION,
                ordered_gene_digest=ordered_gene_digest,
                expected_health=expected_health,
                replay_sign_digest=trial == 0,
            )
            for name in path_counts:
                path_counts[name] += int(supported[name])
            pattern.append(int(any(supported[name] for name in path_counts)))
        count = sum(pattern)
        _, upper = one_sided_binomial_bounds(
            count, PRODUCTION_TRIALS, alpha=FAMILY_ALPHA
        )
        _require_preflight(
            case.get("per_path_false_positive_trials") == path_counts
            and case.get("false_positive_trials") == count
            and np.isclose(
                case.get("false_positive_rate", float("nan")),
                count / PRODUCTION_TRIALS,
                rtol=0.0,
                atol=1e-15,
            )
            and np.isclose(
                case.get("one_sided_clopper_pearson_upper", float("nan")),
                upper,
                rtol=0.0,
                atol=1e-15,
            )
            and case.get("fpr_ok") is (upper <= FAMILY_ALPHA)
            and count <= MAX_SURFACE_FALSE_POSITIVES
            and _is_sha256(case.get("outer_null_digest_sha256")),
            f"boundary-face FPR accounting differs for {case_name}",
        )
        pattern_digest = canonical_sha256(pattern)
        _require_preflight(
            case.get("false_positive_pattern_sha256") == pattern_digest,
            f"boundary-face outcome digest differs for {case_name}",
        )
        pattern_digests[case_name] = pattern_digest

    plant_records = calibration.get("plant_classifier_records_by_trial")
    _require_preflight(
        isinstance(plant_records, list) and len(plant_records) == PRODUCTION_TRIALS,
        "coherent plant classifier records are incomplete",
    )
    _require_preflight(
        calibration.get("plant_classifier_records_sha256")
        == canonical_sha256(plant_records)
        and calibration.get("plant_classifier_audit_digest_sha256")
        == canonical_sha256([canonical_sha256(item) for item in plant_records]),
        "coherent plant classifier digests differ",
    )
    successes = {name: 0 for name in PATH_NAMES}
    for trial, classifier in enumerate(plant_records):
        classifier_namespace = (
            f"r3c:production-path-preflight:coherent-plant:classifier:{trial}"
        )
        _require_preflight(
            classifier_namespace not in seen_namespaces,
            "plant classifier namespace collides with a boundary-face namespace",
        )
        seen_namespaces.add(classifier_namespace)
        supported = _validate_compact_classifier_audit(
            classifier,
            classifier_namespace=classifier_namespace,
            n_examples=n_examples,
            n_genes=n_genes,
            n_draws=PRODUCTION_RANDOMISATION,
            ordered_gene_digest=ordered_gene_digest,
            expected_health=expected_health,
            replay_sign_digest=trial == 0,
        )
        for name in PATH_NAMES:
            successes[name] += int(supported[name])
    planted = calibration.get("planted_power_by_path")
    _require_preflight(
        isinstance(planted, dict) and tuple(planted) == PATH_NAMES,
        "coherent plant path family differs",
    )
    for name in PATH_NAMES:
        count = successes[name]
        lower, _ = one_sided_binomial_bounds(
            count, PRODUCTION_TRIALS, alpha=FAMILY_ALPHA
        )
        health_ok = expected_health[name]
        expected = {
            "successes": count,
            "trials": PRODUCTION_TRIALS,
            "power": count / PRODUCTION_TRIALS,
            "one_sided_clopper_pearson_lower": lower,
            "health_eligible": health_ok,
            "power_ok_if_health_eligible": bool(
                (not health_ok) or lower >= PREFLIGHT_POWER_FLOOR
            ),
        }
        _require_preflight(
            planted[name] == expected
            and ((not health_ok) or count >= MIN_PATH_PLANT_SUCCESSES),
            f"coherent plant power accounting differs for {name}",
        )

    worst_surface = max(
        suite,
        key=lambda name: (
            cases[name]["one_sided_clopper_pearson_upper"],
            cases[name]["false_positive_trials"],
            -list(suite).index(name),
        ),
    )
    worst_count = cases[worst_surface]["false_positive_trials"]
    worst_upper = cases[worst_surface]["one_sided_clopper_pearson_upper"]
    eligible = [name for name in PATH_NAMES if expected_health[name]]
    _require_preflight(
        calibration.get("worst_surface") == worst_surface
        and calibration.get("worst_surface_false_positives") == worst_count
        and np.isclose(
            calibration.get("worst_surface_fpr", float("nan")),
            worst_count / PRODUCTION_TRIALS,
            rtol=0.0,
            atol=1e-15,
        )
        and np.isclose(
            calibration.get("worst_surface_fpr_upper", float("nan")),
            worst_upper,
            rtol=0.0,
            atol=1e-15,
        )
        and calibration.get("all_surface_fpr_diagnostics_ok") is True
        and calibration.get("surface_pattern_manifest_sha256")
        == canonical_sha256(pattern_digests)
        and calibration.get("health_eligible_paths") == eligible
        and calibration.get("all_health_eligible_path_plants_powered") is True,
        "worst-surface FPR or aggregate plant gates differ",
    )
    plant = coherent_path_offset(PLANT_DELTA)
    _require_preflight(
        calibration.get("plant_delta") == PLANT_DELTA
        and calibration.get("coherent_primitive_plant") == plant.tolist()
        and calibration.get("coherent_primitive_plant_sha256") == _array_sha256(plant)
        and _is_sha256(calibration.get("plant_outer_null_digest_sha256")),
        "coherent primitive plant differs",
    )


def validate_preflight_binding(
    record: Mapping[str, Any], fingerprint: str, binding_payload: Mapping[str, Any]
) -> None:
    expected = {
        "schema": RESULT_SCHEMA,
        "experiment": "R3c_acceptor_head_confirmation",
        "instrument_revision": INSTRUMENT_REVISION,
        "protocol_status": "FROZEN_PROTOCOL_AND_INSTRUMENT_VERIFIED",
        "stage": "preflight",
        "status": "DEVELOPMENT",
        "tier": "DEVELOPMENT",
        "split": "val",
        "gene_bucket": 0,
        "n": PREFLIGHT_N,
        "target_bindable_preflight": True,
        "verdict": "TARGET_BINDABLE_PREFLIGHT_GO",
        "all_gates_pass": True,
        "sealed_split_opened": False,
    }
    for key, value in expected.items():
        _require_preflight(record.get(key) == value, f"top-level {key} differs")
    _require_preflight(
        record.get("instrument_fingerprint_sha256") == fingerprint
        and canonical_sha256(binding_payload) == fingerprint
        and record.get("instrument_binding_payload") == binding_payload,
        "instrument fingerprint or binding payload differs",
    )
    artifact_hashes = record.get("hashes")
    _require_preflight(isinstance(artifact_hashes, dict), "bound hash bundle is missing")
    _require_preflight(
        all(
            artifact_hashes.get(key) == value
            for key, value in binding_payload.get("hashes", {}).items()
        ),
        "artifact hashes disagree with the binding payload",
    )
    _require_preflight(
        record.get("input_snapshot_sha256")
        == canonical_sha256(record.get("input_snapshot")),
        "input snapshot digest differs",
    )

    cohort = record.get("cohort_audit")
    _require_preflight(
        isinstance(cohort, dict)
        and cohort.get("schema") == "gfmcd.r3c.acceptor_only_cohort_audit/1"
        and cohort.get("selection_reads_model_outputs") is False
        and cohort.get("preconstruction_allowed_chromosomes")
        == list(VALIDATION_CHROMOSOMES)
        and cohort.get("sealed_chr8_chr9_opened") is False,
        "cohort construction policy differs",
    )
    _require_preflight(
        current_census_identity(cohort) == CURRENT_EXPECTED_CENSUS_IDENTITY,
        "full-GTF census identity differs",
    )
    historical = cohort.get("historical_reconstruction", {})
    historical_genes = historical.get("excluded_gene_ids")
    _require_preflight(
        historical.get("tag") == HISTORICAL_TAG
        and historical.get("both_legacy_decoys_complete_rows") == HISTORICAL_EXPECTED_ROWS
        and historical.get("genes") == HISTORICAL_EXPECTED_GENES
        and historical.get("immutable_identity")
        == {
            "rows": HISTORICAL_EXPECTED_ROWS,
            "genes": HISTORICAL_EXPECTED_GENES,
            "example_id_sha256": HISTORICAL_EXPECTED_EXAMPLE_ID_SHA256,
        }
        and isinstance(historical_genes, list)
        and historical_genes == sorted(set(historical_genes))
        and historical.get("ordered_excluded_genes_sha256")
        == canonical_sha256(historical_genes),
        "historical R3 exclusion memorial differs",
    )
    current = cohort.get("current_census", {})
    _require_preflight(
        current.get("tag") == CURRENT_TAG
        and current.get("eligibility_requires_matched_acceptor") is False
        and current.get("eligibility_requires_only_donor_pseudo") is True,
        "current acceptor-only eligibility differs",
    )
    fixed_genes = current.get("fixed_selection_gene_ids_by_bucket")
    _require_preflight(
        isinstance(fixed_genes, dict) and set(fixed_genes) == {"0", "1"},
        "fixed cohort gene lists are missing",
    )
    for bucket in ("0", "1"):
        gene_ids = fixed_genes[bucket].get("gene_ids")
        _require_preflight(
            isinstance(gene_ids, list)
            and gene_ids == sorted(set(gene_ids))
            and len(gene_ids)
            == CURRENT_EXPECTED_CENSUS_IDENTITY["fixed_selection_by_bucket"][bucket][
                "genes"
            ]
            and fixed_genes[bucket].get("gene_ids_sha256")
            == canonical_sha256(gene_ids)
            and all(gene_bucket(gene_id) == int(bucket) for gene_id in gene_ids),
            f"fixed bucket-{bucket} gene list differs",
        )
    selection = cohort.get("selection", {})
    gene_ids = selection.get("gene_ids")
    _require_preflight(
        selection.get("bucket") == 0
        and selection.get("rows") == PREFLIGHT_N
        and selection.get("genes") == len(fixed_genes["0"]["gene_ids"])
        and selection.get("ordered_rows_sha256")
        == CURRENT_EXPECTED_CENSUS_IDENTITY["fixed_selection_by_bucket"]["0"][
            "ordered_rows_sha256"
        ]
        and gene_ids == fixed_genes["0"]["gene_ids"]
        and selection.get("gene_ids_sha256") == canonical_sha256(gene_ids),
        "realized bucket-0 cohort differs",
    )
    n_genes = len(gene_ids)
    _require_preflight(record.get("n_genes") == n_genes, "top-level gene count differs")
    ordered_gene_digest = _ordered_gene_labels_sha256(gene_ids)

    construction = record.get("construction")
    expected_policy = {
        "readout": "mask exactly one G at next_acceptor_pos - 1",
        "paired_operator": "GT -> AC at donor_pos",
        "pseudo_operator": "GT -> AC at matched_decoy_donor",
        "pseudo_distance_caliper": DECOY_TOL,
        "donor_readout_constructed": False,
    }
    _require_preflight(
        isinstance(construction, dict)
        and construction.get("schema") == "gfmcd.r3c.acceptor_only_construction/1"
        and construction.get("policy") == expected_policy
        and construction.get("policy_sha256") == canonical_sha256(expected_policy)
        and construction.get("rows") == PREFLIGHT_N
        and construction.get("all_single_acceptor_G") is True
        and construction.get("all_donor_targets_GT") is True
        and construction.get("all_motif_clean") is True
        and _is_sha256(construction.get("ordered_rows_sha256"))
        and isinstance(construction.get("row_sha256"), list)
        and len(construction["row_sha256"]) == PREFLIGHT_N
        and all(_is_sha256(item.get("sha256")) for item in construction["row_sha256"]),
        "acceptor-only construction or no-donor-readout attestation differs",
    )

    expected_components = [component_name(component) for component in COMPONENTS]
    _require_preflight(
        record.get("component_names") == expected_components
        and record.get("component_family_sha256") == canonical_sha256(expected_components)
        and record.get("modes") == list(MODES)
        and record.get("primary_estimand")
        == "raw masked-acceptor-G logit-margin component drop",
        "component or estimand registry differs",
    )
    components = record.get("components")
    _require_preflight(
        isinstance(components, dict) and set(components) == set(expected_components),
        "component records differ from the fixed head registry",
    )
    health_measurements: dict[str, dict[str, dict[str, Any]]] = {
        mode: {} for mode in MODES
    }
    health_executed = True
    for component in COMPONENTS:
        name = component_name(component)
        item = components[name]
        _require_preflight(
            (item.get("kind"), item.get("layer"), item.get("head")) == component,
            f"component metadata differs for {name}",
        )
        modes = item.get("modes")
        _require_preflight(
            isinstance(modes, dict) and set(modes) == set(MODES),
            f"component modes differ for {name}",
        )
        for mode in MODES:
            health = modes[mode].get("health")
            endpoints = modes[mode].get("endpoints")
            _require_preflight(
                isinstance(health, dict)
                and isinstance(health.get("ok"), bool)
                and isinstance(health.get("total"), int)
                and health.get("total") > 0
                and isinstance(endpoints, dict)
                and tuple(endpoints) == ENDPOINTS,
                f"component health/endpoints differ for {mode}|{name}",
            )
            health_executed = health_executed and health["total"] > 0
            health_measurements[mode][name] = {"health": health}
    expected_path_health = reviewed_preflight_path_health(
        path_health_eligibility(health_measurements)
    )
    expected_health_ok = {
        name: bool(expected_path_health[name]["ok"]) for name in PATH_NAMES
    }
    _require_preflight(
        record.get("path_health_authorization")
        == preflight_path_health_authorization(expected_path_health),
        "bucket-0 path-health authorization differs",
    )

    family = record.get("family")
    _require_preflight(
        isinstance(family, dict)
        and family.get("primitive_row_names") == primitive_row_names()
        and family.get("primitive_row_order_sha256")
        == canonical_sha256(primitive_row_names())
        and family.get("row_names") == family_row_names()
        and family.get("row_order_sha256") == canonical_sha256(family_row_names())
        and family.get("n_primitive_rows") == 40
        and family.get("n_family_rows") == 62,
        "40/62 row family differs",
    )
    family_stats = family.get("gene_sufficient_statistics")
    _validate_gene_statistics(
        family_stats,
        names=family_row_names(),
        gene_ids=gene_ids,
        n_examples=PREFLIGHT_N,
    )
    inference = family.get("inference")
    _require_preflight(isinstance(inference, dict), "observed family inference is missing")
    _validate_observed_inference_replay(
        stats=family_stats,
        stored_inference=inference,
        stored_paths=record.get("paths"),
        stored_path_family=record.get("path_family"),
        path_health=expected_path_health,
        n_draws=PRODUCTION_RANDOMISATION,
        seed=PRODUCTION_SEED,
        namespace="r3c:observed-seven-path-family",
        min_genes=MIN_GENES,
    )
    _validate_shared_boundary_audit(
        inference.get("boundary_randomisation_audit"),
        namespace="r3c:observed-seven-path-family:shared-vector-boundary-edges",
        B=PRODUCTION_RANDOMISATION,
        ordered_gene_digest=ordered_gene_digest,
        n_genes=n_genes,
    )
    _require_preflight(
        isinstance(inference.get("intervals"), dict)
        and tuple(inference["intervals"]) == tuple(family_row_names())
        and isinstance(inference.get("row_boundary_tests"), dict)
        and set(inference["row_boundary_tests"]) == set(_required_row_test_kinds()),
        "observed interval or boundary-test family differs",
    )
    for row, item in inference["intervals"].items():
        lo, hi = item.get("simultaneous_ci", (None, None))
        _require_preflight(
            _is_finite_number(item.get("estimate"))
            and _is_finite_number(lo)
            and _is_finite_number(hi)
            and lo <= hi
            and item.get("strict_classification")
            == strict_interval_decision(lo, hi, SESOI),
            f"strict interval decision differs for {row}",
        )

    path_family = record.get("path_family")
    paths = record.get("paths")
    _require_preflight(
        isinstance(path_family, dict)
        and path_family.get("path_names") == list(PATH_NAMES)
        and isinstance(paths, dict)
        and tuple(paths) == PATH_NAMES,
        "observed seven-path Holm family differs",
    )
    raw = path_family.get("raw_iut_p_values")
    adjusted = path_family.get("holm_adjusted_p_values")
    _require_preflight(
        isinstance(raw, dict)
        and tuple(raw) == PATH_NAMES
        and isinstance(adjusted, dict)
        and tuple(adjusted) == PATH_NAMES,
        "observed path p-value registries differ",
    )
    expected_adjusted = holm_adjust_named(raw)
    for name in PATH_NAMES:
        path = paths[name]
        p_pass = bool(expected_adjusted[name] < FAMILY_ALPHA)
        ci_matches = {
            row: inference["intervals"][row]["strict_classification"] == expected
            for row, expected in named_path_specs()[name]["requirements"].items()
        }
        ci_pass = all(ci_matches.values())
        health_ok = expected_health_ok[name]
        expected_class = (
            "HEALTH_FAILED"
            if not health_ok
            else "SUPPORTED"
            if p_pass and ci_pass
            else "INCONCLUSIVE_METHOD_DISAGREEMENT"
            if p_pass != ci_pass
            else "NOT_SUPPORTED_AT_FROZEN_THRESHOLDS"
        )
        _require_preflight(
            np.isclose(adjusted[name], expected_adjusted[name], rtol=0.0, atol=1e-15)
            and path.get("requirements") == named_path_specs()[name]["requirements"]
            and path.get("holm_family_size") == 7
            and path.get("p_pass") is p_pass
            and path.get("strict_simultaneous_ci_matches") == ci_matches
            and path.get("strict_simultaneous_ci_pass") is ci_pass
            and path.get("health") == expected_path_health[name]
            and path.get("classification") == expected_class,
            f"observed scientific decision is inconsistent for {name}",
        )

    calibration = family.get("production_path_preflight")
    _require_preflight(
        family.get("production_path_preflight_origin") == "RUN_ON_CURRENT_BUCKET",
        "preflight calibration origin differs",
    )
    _validate_production_path_preflight(
        calibration,
        n_examples=PREFLIGHT_N,
        n_genes=n_genes,
        ordered_gene_digest=ordered_gene_digest,
        expected_health={name: True for name in PATH_NAMES},
    )

    capability = record.get("hook_capability")
    _require_preflight(
        record.get("hook_capability_origin") == "RUN_ON_CURRENT_BUCKET"
        and isinstance(capability, dict)
        and capability.get("schema") == "gfmcd.r3c.independent_hook_capability/1"
        and capability.get("candidate_outcomes_used") is False
        and capability.get("n_examples") == PREFLIGHT_N
        and capability.get("n_genes") == n_genes
        and capability.get("example_order_sha256")
        == canonical_sha256([item["example_id"] for item in construction["row_sha256"]])
        and capability.get("all_controls_pass") is True,
        "independent live-hook capability record differs",
    )
    controls = capability.get("controls")
    expected_control_names = {
        "clean_to_clean_whole_head_self_patch",
        "post_attention_head_split_merge_identity",
        "same_example_source_target_alignment",
        "batch_size_invariance",
        "final_hidden_through_full_model_cls_reproduces_logits_and_margin",
        "all_L3_clean_to_corrupt_live_restoration",
    }
    _require_preflight(
        isinstance(controls, dict)
        and set(controls) == expected_control_names
        and all(value is True for value in controls.values()),
        "independent live-hook controls did not all pass",
    )
    _require_preflight(
        capability.get("split_merge", {}).get("exact") is True
        and capability.get("same_example_alignment", {}).get("exact") is True
        and capability.get("self_patch", {}).get("atol") == HOOK_SELF_PATCH_ATOL
        and capability.get("self_patch", {}).get("margin_max_abs_error")
        <= HOOK_SELF_PATCH_ATOL
        and capability.get("self_patch", {}).get("full_logits_max_abs_error")
        <= HOOK_SELF_PATCH_ATOL
        and capability.get("batch_invariance", {}).get("atol")
        == HOOK_BATCH_INVARIANCE_ATOL
        and capability.get("batch_invariance", {}).get("maximum")
        <= HOOK_BATCH_INVARIANCE_ATOL
        and capability.get("full_mlm_replay", {}).get("checked") is True
        and capability.get("full_mlm_replay", {}).get("full_logits_max_abs_error")
        <= HOOK_SELF_PATCH_ATOL
        and capability.get("full_mlm_replay", {}).get(
            "acceptor_G_margin_max_abs_error"
        )
        <= HOOK_SELF_PATCH_ATOL,
        "hook identity, batching, or full-MLM reproduction metrics differ",
    )
    restoration = capability.get("all_L3_restoration")
    _require_preflight(
        isinstance(restoration, dict)
        and restoration.get("classification") == "LIVE_HOOK_CAPABILITY_CONFIRMED"
        and restoration.get("p_pass") is True
        and restoration.get("strict_simultaneous_ci_pass") is True
        and restoration.get("partner_floor") == HOOK_RESTORATION_PARTNER_FLOOR
        and restoration.get("pseudo_equivalence_sesoi") == SESOI,
        "all-L3 live restoration capability differs",
    )
    restoration_stats = restoration.get("gene_sufficient_statistics")
    capability_means = _validate_gene_statistics(
        restoration_stats,
        names=["all_L3_partner_recovery", "all_L3_pseudo_recovery"],
        gene_ids=gene_ids,
        n_examples=PREFLIGHT_N,
    )
    _require_preflight(
        all(
            np.isclose(
                restoration["estimates"][name], capability_means[index], rtol=0.0, atol=1e-12
            )
            for index, name in enumerate(
                ("all_L3_partner_recovery", "all_L3_pseudo_recovery")
            )
        ),
        "all-L3 restoration estimates disagree with gene sums",
    )
    replay_capability_values, replay_capability_groups = _reconstruct_cluster_array(
        restoration_stats
    )
    replay_restoration = hook_capability_inference(
        replay_capability_values[0],
        replay_capability_values[1],
        replay_capability_groups,
        n_draws=PRODUCTION_RANDOMISATION,
        seed=PRODUCTION_SEED,
        namespace="r3c:bucket0-independent-live-hook-capability",
        min_genes=MIN_GENES,
    )
    _require_preflight(
        all(restoration.get(key) == value for key, value in replay_restoration.items()),
        "all-L3 capability inference does not replay from gene sufficient statistics",
    )
    cap_audit = restoration.get("boundary_randomisation_audit")
    _require_preflight(
        isinstance(cap_audit, dict)
        and cap_audit.get("all_decision_edges_share_one_sign_matrix") is True
        and cap_audit.get("B") == PRODUCTION_RANDOMISATION
        and cap_audit.get("namespace")
        == "r3c:bucket0-independent-live-hook-capability:shared-vector-boundary-edges"
        and _is_sha256(cap_audit.get("sign_matrix_sha256"))
        and cap_audit.get("ordered_gene_labels_sha256") == ordered_gene_digest,
        "all-L3 restoration boundary namespace/digests differ",
    )
    _require_preflight(
        cap_audit.get("sign_matrix_sha256")
        == _expected_sign_matrix_sha256(
            namespace=cap_audit["namespace"],
            seed=PRODUCTION_SEED,
            B=PRODUCTION_RANDOMISATION,
            n_genes=n_genes,
        ),
        "all-L3 restoration sign-matrix digest does not replay",
    )

    baselines = record.get("baselines")
    baseline_means = _validate_gene_statistics(
        baselines.get("gene_sufficient_statistics") if isinstance(baselines, dict) else None,
        names=["baseline_partner", "baseline_pseudo"],
        gene_ids=gene_ids,
        n_examples=PREFLIGHT_N,
    )
    _require_preflight(
        np.isclose(baselines.get("partner"), baseline_means[0], rtol=0.0, atol=1e-12)
        and np.isclose(baselines.get("pseudo"), baseline_means[1], rtol=0.0, atol=1e-12),
        "baseline means disagree with gene sums",
    )
    expected_gates = {
        "minimum_30_independent_genes": n_genes >= MIN_GENES,
        "exact_full_gtf_census_identity": True,
        "native_paired_donor_positive_control": baselines["partner"] > 0.20,
        "matched_pseudo_donor_negative_control": abs(baselines["pseudo"]) < SESOI,
        "single_acceptor_G_readout_and_GT_to_AC_construction": True,
        "zero_planted_donor_motifs": True,
        "bounded_identical_policy_health_executed": health_executed,
        "exact_62_row_family": True,
        "analytic_path_fwer_and_boundary_face_diagnostics": calibration[
            "all_surface_fpr_diagnostics_ok"
        ],
        "all_health_eligible_path_plants_powered": calibration[
            "all_health_eligible_path_plants_powered"
        ],
        "bound_bucket0_live_hook_capability": capability["all_controls_pass"],
    }
    _require_preflight(
        tuple(expected_gates) == PREFLIGHT_GATE_NAMES
        and record.get("gates") == expected_gates
        and all(expected_gates.values()),
        "ordered gates are not semantic consequences of the stored evidence",
    )


def verify_preflight_manifest(
    path: str | Path,
    *,
    frozen_artifact_sha256: str | None = FROZEN_PREFLIGHT_ARTIFACT_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = Path(path).expanduser().resolve()
    manifest_path = artifact.with_suffix(".manifest.json")
    if not artifact.is_file() or not manifest_path.is_file():
        raise ValueError("target requires immutable preflight artifact and sibling manifest")
    record = json.loads(artifact.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("experiment_id") != (
        "splicebert_acceptor_coupling:r3c_acceptor_head_confirmation"
    ):
        raise ValueError("preflight manifest belongs to another experiment")
    matches = []
    for output in manifest.get("outputs", []):
        output_path = Path(str(output.get("path", ""))).expanduser()
        if not output_path.is_absolute():
            output_path = REPO / output_path
        if output_path.resolve() == artifact:
            matches.append(output)
    artifact_sha256 = sha256_file(artifact)
    if len(matches) != 1 or matches[0].get("sha256") != artifact_sha256:
        raise ValueError("preflight bytes do not match sibling manifest")
    if frozen_artifact_sha256 is None:
        raise ValueError("target requires a literal reviewed preflight-artifact SHA-256 pin")
    if artifact_sha256 != frozen_artifact_sha256:
        raise ValueError("preflight artifact differs from the literal reviewed SHA-256 pin")
    return record, {
        "artifact": {
            "path": str(artifact),
            "size_bytes": artifact.stat().st_size,
            "sha256": artifact_sha256,
        },
        "manifest": {
            "path": str(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        },
    }


def _component_output(
    measurements: Mapping[str, Mapping[str, Mapping[str, Any]]],
    inference: Mapping[str, Any],
) -> dict[str, Any]:
    memberships: dict[tuple[str, str], list[str]] = {}
    for path_name, spec in named_path_specs().items():
        for row in spec["requirements"]:
            mode, head, _ = row.split("|")
            if head != "delivery_gap":
                memberships.setdefault((mode, head), []).append(path_name)
    output = {}
    for component in COMPONENTS:
        head = component_name(component)
        output[head] = {
            "kind": "head",
            "layer": component[1],
            "head": component[2],
            "modes": {},
        }
        for mode in MODES:
            endpoints = {}
            for endpoint in ENDPOINTS:
                row = family_row_name(mode, head, endpoint)
                endpoints[endpoint] = {
                    "raw_mean": float(
                        np.mean(measurements[mode][head][endpoint])
                    ),
                    "simultaneous_estimate": inference["intervals"][row],
                    "boundary_test": inference["row_boundary_tests"].get(row),
                }
            output[head]["modes"][mode] = {
                "health": measurements[mode][head]["health"],
                "endpoints": endpoints,
                "frozen_path_memberships": memberships.get((mode, head), []),
            }
    return output


def build_preflight_gates(
    *,
    n_genes: int,
    cohort_audit: Mapping[str, Any],
    native_partner_mean: float,
    native_pseudo_mean: float,
    construction: Mapping[str, Any],
    health_executed: bool,
    exact_family: bool,
    calibration: Mapping[str, Any],
    hook_capability: Mapping[str, Any],
) -> dict[str, bool]:
    """Build instrument gates without accepting any candidate/path outcome as input."""
    gates = {
        "minimum_30_independent_genes": n_genes >= MIN_GENES,
        "exact_full_gtf_census_identity": (
            current_census_identity(cohort_audit) == CURRENT_EXPECTED_CENSUS_IDENTITY
        ),
        "native_paired_donor_positive_control": native_partner_mean > 0.20,
        "matched_pseudo_donor_negative_control": abs(native_pseudo_mean) < SESOI,
        "single_acceptor_G_readout_and_GT_to_AC_construction": bool(
            construction["all_single_acceptor_G"]
            and construction["all_donor_targets_GT"]
            and construction["policy"]["donor_readout_constructed"] is False
        ),
        "zero_planted_donor_motifs": bool(construction["all_motif_clean"]),
        "bounded_identical_policy_health_executed": bool(health_executed),
        "exact_62_row_family": bool(exact_family),
        "analytic_path_fwer_and_boundary_face_diagnostics": bool(
            calibration["all_surface_fpr_diagnostics_ok"]
        ),
        "all_health_eligible_path_plants_powered": bool(
            calibration["all_health_eligible_path_plants_powered"]
        ),
        "bound_bucket0_live_hook_capability": bool(
            hook_capability["all_controls_pass"]
        ),
    }
    if tuple(gates) != PREFLIGHT_GATE_NAMES:
        raise RuntimeError("R3c gate order drifted")
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "preflight", "target"), default="smoke")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument("--n-randomisation", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--preflight-artifact", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # Authorization-only protocol read.  This happens before any user-supplied path, output,
    # cohort, checkpoint, or model outcome can be inspected.
    if args.stage == "target":
        try:
            authorize_target(
                sha256_file(PROTOCOL), protocol_text=PROTOCOL.read_text()
            )
        except TargetSealError as exc:
            raise SystemExit(str(exc)) from exc

    args.n = args.n if args.n is not None else (
        30 if args.stage == "smoke" else PREFLIGHT_N if args.stage == "preflight" else TARGET_N
    )
    args.batch = args.batch if args.batch is not None else (
        8 if args.stage == "smoke" else PRODUCTION_BATCH
    )
    args.n_randomisation = args.n_randomisation if args.n_randomisation is not None else (
        199 if args.stage == "smoke" else PRODUCTION_RANDOMISATION
    )
    args.n_trials = args.n_trials if args.n_trials is not None else (
        5 if args.stage == "smoke" else PRODUCTION_TRIALS
    )
    if args.n < MIN_GENES or min(args.batch, args.n_randomisation, args.n_trials) < 1:
        raise SystemExit("R3c requires n>=30 and positive execution counts")
    if args.stage != "smoke":
        fixed = {
            "n": (args.n, PREFLIGHT_N if args.stage == "preflight" else TARGET_N),
            "batch": (args.batch, PRODUCTION_BATCH),
            "seed": (args.seed, PRODUCTION_SEED),
            "n-randomisation": (args.n_randomisation, PRODUCTION_RANDOMISATION),
            "n-trials": (args.n_trials, PRODUCTION_TRIALS),
        }
        drift = [
            f"--{key}={actual} (fixed {expected})"
            for key, (actual, expected) in fixed.items()
            if actual != expected
        ]
        if drift:
            raise SystemExit("R3c production parameters are fixed: " + "; ".join(drift))
        if args.out is None:
            raise SystemExit("R3c preflight/target requires a fresh --out path")
    output = Path(args.out).resolve() if args.out else None
    manifest_path = output.with_suffix(".manifest.json") if output else None
    if output is not None and (output.exists() or manifest_path.exists()):
        raise SystemExit("R3c outputs are immutable; choose a fresh versioned path")

    model_files = checkpoint_files(args.model)
    bound_paths = [
        *model_files,
        Path(args.gtf),
        Path(args.fasta),
        PROTOCOL,
        RUNNER,
        STAT_CORE,
        HOOK_CORE,
        ASSAY_CORE,
        CIRCUIT_CORE,
        *OTHER_BOUND_SOURCES,
    ]
    if args.preflight_artifact:
        preflight_path = Path(args.preflight_artifact).expanduser().resolve()
        bound_paths.extend((preflight_path, preflight_path.with_suffix(".manifest.json")))
    try:
        snapshot = snapshot_input_paths(bound_paths)
        verify_input_snapshot(snapshot)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    device = pick_device(args.device)
    hashes = exact_hash_bundle(
        args.model,
        args.gtf,
        args.fasta,
        snapshot=snapshot,
        model_files=model_files,
    )
    binding_payload = instrument_binding_payload(
        hashes,
        batch=args.batch,
        device=device,
        seed=args.seed,
        n_randomisation=args.n_randomisation,
        n_trials=args.n_trials,
    )
    fingerprint = canonical_sha256(binding_payload)
    protocol_text = PROTOCOL.read_text()
    try:
        target_bindable_preflight = (
            authorize_bindable_preflight(
                hashes["protocol_sha256"],
                fingerprint,
                protocol_text=protocol_text,
            )
            if args.stage == "preflight"
            else args.stage == "target"
        )
    except TargetSealError as exc:
        raise SystemExit(f"R3c preflight seal validation failed: {exc}") from exc

    preflight_record = None
    preflight_package = None
    git_identity = None
    if args.stage == "target":
        try:
            verify_frozen_instrument(fingerprint)
            if not args.preflight_artifact:
                raise ValueError("R3c target requires --preflight-artifact")
            preflight_record, preflight_package = verify_preflight_manifest(
                args.preflight_artifact
            )
            validate_preflight_binding(preflight_record, fingerprint, binding_payload)
            git_identity = require_clean_git()
        except (TargetSealError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"R3c target binding failed: {exc}") from exc
    elif target_bindable_preflight:
        try:
            git_identity = require_clean_git()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    staging = None
    model_path: str | Path = resolved_model_root(args.model)
    staged_checkpoint_sha = None
    if args.stage != "smoke":
        try:
            staging, model_path, staged_checkpoint_sha = stage_checkpoint_bundle(
                args.model, model_files, snapshot
            )
            verify_input_snapshot(snapshot)
        except RuntimeError as exc:
            if staging is not None:
                staging.cleanup()
            raise SystemExit(str(exc)) from exc

    examples, decoy_donors, bucket, cohort_audit = load_cohort(args)
    groups = [example.gene_id for example in examples]
    adapter = load_adapter(model_path, device, eager=False)
    task = CouplingTask(adapter, batch=args.batch)
    arm = AcceptorArm(task, examples, decoy_donors)
    construction = construction_record(examples, arm)
    native = arm.native()
    print(
        f"stage={args.stage} bucket={bucket} n={len(examples)} "
        f"genes={len(set(groups))} device={device} "
        f"partner={native['partner'].mean():+.4f} pseudo={native['pseudo'].mean():+.4f}",
        flush=True,
    )

    if args.stage == "target":
        assert preflight_record is not None
        hook_capability = preflight_record["hook_capability"]
        hook_capability_origin = "INHERITED_FROM_BOUND_BUCKET_0_PREFLIGHT"
    else:
        print("  capability: independent clean-source hook battery", flush=True)
        hook_capability = run_hook_capability(
            adapter,
            arm,
            groups,
            n_draws=args.n_randomisation,
            seed=args.seed,
            min_genes=MIN_GENES,
        )
        hook_capability_origin = "RUN_ON_CURRENT_BUCKET"

    measurements: dict[str, dict[str, dict[str, Any]]] = {
        mode: {} for mode in MODES
    }
    for index, component in enumerate(COMPONENTS):
        name = component_name(component)
        ctx = lambda component=component: mean_intervention(adapter, component)
        partner_intervened, pseudo_intervened = arm.couplings(ctx)
        measurements[MEAN_MODE][name] = mean_component_endpoint_drops(
            native["partner"],
            native["pseudo"],
            partner_intervened,
            pseudo_intervened,
        )
        print(f"  mean {index + 1:02d}/{len(COMPONENTS)} {name}", flush=True)

    print("  noising: capture same-example paired/pseudo donor sources", flush=True)
    noised = arm.activation_noising_drops(COMPONENTS, native["clean_margin"])
    for component in COMPONENTS:
        name = component_name(component)
        measurements[NOISING_MODE][name] = endpoint_vectors(
            noised[name]["partner"], noised[name]["pseudo"]
        )
    for mode in MODES:
        print(f"  health {mode}", flush=True)
        health = arm.bounded_health(COMPONENTS, mode=mode, seed=args.seed)
        for component in COMPONENTS:
            measurements[mode][component_name(component)]["health"] = health[
                component_name(component)
            ]

    primitive, primitive_names = build_primitive_matrix(measurements)
    family, row_names = derive_family_from_primitives(primitive)
    observed_path_health = path_health_eligibility(measurements)
    if args.stage == "target":
        assert preflight_record is not None
        path_health, path_health_authorization = monotone_target_path_health(
            observed_path_health, preflight_record["paths"]
        )
    else:
        path_health = reviewed_preflight_path_health(observed_path_health)
        path_health_authorization = preflight_path_health_authorization(
            path_health
        )
    family_gene_statistics = gene_sufficient_statistics(family, groups, row_names)
    inference_family, inference_groups = _reconstruct_cluster_array(
        family_gene_statistics
    )
    inference = strict_path_family_inference(
        inference_family,
        inference_groups,
        row_names,
        n_draws=args.n_randomisation,
        seed=args.seed,
        namespace="r3c:observed-seven-path-family",
        min_genes=MIN_GENES,
        path_health=path_health,
    )

    if args.stage == "target":
        assert preflight_record is not None
        calibration = preflight_record["family"]["production_path_preflight"]
        calibration_origin = "INHERITED_FROM_BOUND_BUCKET_0_PREFLIGHT"
    else:
        calibration = r3c_path_preflight(
            primitive,
            groups,
            path_health=calibration_path_health_eligibility(),
            n_calibration=args.n_randomisation,
            n_trials=args.n_trials,
            seed=args.seed,
            namespace="r3c:production-path-preflight",
            min_genes=MIN_GENES,
        )
        calibration_origin = "RUN_ON_CURRENT_BUCKET"

    components_output = _component_output(measurements, inference)
    paths_output = inference["paths"]
    health_executed = all(
        measurements[mode][component_name(component)]["health"]["total"] > 0
        for mode in MODES
        for component in COMPONENTS
    )
    exact_family = (
        primitive.shape[0] == 40
        and family.shape[0] == 62
        and primitive_names == primitive_row_names()
        and row_names == family_row_names()
    )
    gates = build_preflight_gates(
        n_genes=len(set(groups)),
        cohort_audit=cohort_audit,
        native_partner_mean=float(native["partner"].mean()),
        native_pseudo_mean=float(native["pseudo"].mean()),
        construction=construction,
        health_executed=health_executed,
        exact_family=exact_family,
        calibration=calibration,
        hook_capability=hook_capability,
    )
    all_gates_pass = all(gates.values())
    if not all_gates_pass:
        verdict = "INCONCLUSIVE"
    elif args.stage == "smoke":
        verdict = "SMOKE_MECHANICS_PASS"
    elif args.stage == "preflight":
        verdict = (
            "TARGET_BINDABLE_PREFLIGHT_GO"
            if target_bindable_preflight
            else "DRAFT_PREFLIGHT_PASS_NOT_TARGET_BINDABLE"
        )
    else:
        verdict = "HELD_OUT_ACCEPTOR_HEAD_CONFIRMATION_COMPLETE"

    result = {
        "schema": RESULT_SCHEMA,
        "experiment": "R3c_acceptor_head_confirmation",
        "instrument_revision": INSTRUMENT_REVISION,
        "protocol_status": (
            "FROZEN_PROTOCOL_AND_INSTRUMENT_VERIFIED"
            if target_bindable_preflight
            else "DRAFT_NOT_FROZEN"
        ),
        "target_bindable_preflight": bool(target_bindable_preflight),
        "stage": args.stage,
        "status": "CONFIRMATORY" if args.stage == "target" else "DEVELOPMENT",
        "tier": "HELD_OUT_TARGET" if args.stage == "target" else "DEVELOPMENT",
        "split": "val",
        "gene_bucket": bucket,
        "sealed_split_opened": False,
        "n": len(examples),
        "n_genes": len(set(groups)),
        "cohort_audit": cohort_audit,
        "construction": construction,
        "hook_capability": hook_capability,
        "hook_capability_origin": hook_capability_origin,
        "component_names": [component_name(component) for component in COMPONENTS],
        "component_family_sha256": canonical_sha256(
            [component_name(component) for component in COMPONENTS]
        ),
        "primary_estimand": "raw masked-acceptor-G logit-margin component drop",
        "modes": list(MODES),
        "baselines": {
            "partner": float(native["partner"].mean()),
            "pseudo": float(native["pseudo"].mean()),
            "clean_margin": float(native["clean_margin"].mean()),
            "gene_sufficient_statistics": gene_sufficient_statistics(
                np.vstack((native["partner"], native["pseudo"])),
                groups,
                ["baseline_partner", "baseline_pseudo"],
            ),
        },
        "family": {
            "family_definition": (
                "10 heads x 2 modes x partner/pseudo/specific plus two noising "
                "acceptor-route delivery-gap rows"
            ),
            "primitive_row_names": primitive_names,
            "primitive_row_order_sha256": canonical_sha256(primitive_names),
            "row_names": row_names,
            "row_order_sha256": canonical_sha256(row_names),
            "n_primitive_rows": primitive.shape[0],
            "n_family_rows": family.shape[0],
            "inference": {
                key: value
                for key, value in inference.items()
                if key not in {"paths", "row_names", "row_order_sha256"}
            },
            "production_path_preflight": calibration,
            "production_path_preflight_origin": calibration_origin,
            "gene_sufficient_statistics": family_gene_statistics,
        },
        "components": components_output,
        "delivery_gap": {
            endpoint: inference["intervals"][delivery_gap_row_name(endpoint)]
            | {
                "boundary_test": inference["row_boundary_tests"].get(
                    delivery_gap_row_name(endpoint)
                )
            }
            for endpoint in DELIVERY_GAP_ENDPOINTS
        },
        "paths": paths_output,
        "path_health_authorization": path_health_authorization,
        "path_family": inference["path_family"],
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "verdict": verdict,
        "hashes": hashes,
        "instrument_binding_payload": binding_payload,
        "instrument_fingerprint_sha256": fingerprint,
        "input_snapshot": snapshot,
        "input_snapshot_sha256": canonical_sha256(snapshot),
        "git_authorization_identity": git_identity,
        "preflight_package": preflight_package,
        "checkpoint_execution": {
            "private_staged_copy_loaded": args.stage != "smoke",
            "staged_checkpoint_sha256": staged_checkpoint_sha,
        },
        "licenses": (
            "development instrument behavior only"
            if args.stage != "target"
            else "the supported fixed component/path patterns under the two declared interventions"
        ),
        "does_not_license": [
            "attention as mechanism",
            "address or payload identity",
            "acceptor-side write",
            "path mediation",
            "sufficiency",
            "faithfulness",
            "a complete circuit",
            "biological mechanism",
        ],
        "next_if_supported": (
            "freeze separate QK/address, donor-value-to-acceptor write, acceptor-side "
            "subspace, and downstream path-mediation assays"
        ),
    }

    if args.stage == "smoke":
        print(
            json.dumps(
                {
                    "verdict": verdict,
                    "gates": gates,
                    "path_classifications": {
                        name: paths_output[name]["classification"] for name in PATH_NAMES
                    },
                    "fingerprint": fingerprint,
                },
                indent=2,
            )
        )
        if staging is not None:
            staging.cleanup()
        return

    assert output is not None and manifest_path is not None
    manifest = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r3c_acceptor_head_confirmation",
        sys.argv,
        repo=REPO,
        status=result["status"],
        config={
            "stage": args.stage,
            "n": args.n,
            "batch": args.batch,
            "seed": args.seed,
            "n_randomisation": args.n_randomisation,
            "n_trials": args.n_trials,
            "gene_bucket": bucket,
            "component_family_sha256": result["component_family_sha256"],
            "family_row_order_sha256": canonical_sha256(row_names),
            "path_family_sha256": canonical_sha256(named_path_specs()),
            "instrument_fingerprint_sha256": fingerprint,
            "construction_sha256": construction["ordered_rows_sha256"],
            "input_snapshot_sha256": result["input_snapshot_sha256"],
        },
        seeds={"randomisation": args.seed, "health_probe_selection": args.seed},
        notes=[
            "Masked acceptor G only; no donor readout is constructed.",
            "Both paired and matched pseudo donor edits are GT->AC.",
            "All historical R3 genes are excluded before current eligibility.",
            "The current census requires only a donor pseudo, not an acceptor pseudo.",
            "Path decisions use strict boundary tests, TOST/IUT, and Holm over seven claims.",
            "The 62-row simultaneous envelope uses strict CI classification as a required concordance check.",
            "Outer nulls center primitive rows before shared whole-gene sign flips.",
            "Validation chromosomes chr11/16/19 only; sealed chr8/9 remains closed.",
        ],
    )
    if git_identity is not None:
        manifest.git_commit = git_identity["commit"]
        manifest.git_dirty = False
    manifest.inputs = [
        {"path": path, **record} for path, record in sorted(snapshot.items())
    ]
    result_bytes = (json.dumps(result, indent=2) + "\n").encode()
    manifest.outputs = [
        {
            "path": str(output),
            "size_bytes": len(result_bytes),
            "sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
    ]
    manifest_bytes = (
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    try:
        verify_input_snapshot(snapshot)
        atomic_write_new(output, result_bytes)
        atomic_write_new(manifest_path, manifest_bytes)
    finally:
        if staging is not None:
            staging.cleanup()
    print(f"verdict={verdict} -> {output}")


if __name__ == "__main__":
    main()
