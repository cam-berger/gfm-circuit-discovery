#!/usr/bin/env python
"""R3d: prospective endpoint-specific train verification of two R3c paths.

R3d keeps the frozen R3c v2.1 masked-acceptor-G measurement but replaces its
62-row simultaneous interval with confidence bounds obtained by inverting the
same exact whole-gene boundary tests used for each decision.  The two paths are
gate-kept in a fixed sequence: L3H1 reuse first, then the original 17-row route
conjunction only if L3H1 is supported.

This is deliberately *not* a sealed-split or independent confirmation.  Its
target uses canonical train chromosomes, excludes the immutable two-class registry of
Claude inverse-acceptor and 400-shard clean-alt head-screen direct exposures, and takes
one outcome-blind hash-ranked row per gene.  HCA18/broad Atlas, donor-circuit, and earlier
R3 train selection exposures are outside that registry, so the target is registry-disjoint
rather than candidate-selection-disjoint or globally gene-fresh.
No masked-donor endpoint is constructed anywhere in this file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).resolve()
R3C_RUNNER = RUNNER.with_name("run_r3c_acceptor_head_confirmation.py")
PROTOCOL = (
    REPO
    / "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R3D_ACCEPTOR_SPECIFICITY_ADJUDICATION.md"
)
STAT_CORE = REPO / "src/gfmcd/splicing/paired_component.py"
TARGET_ATTEMPT_LEDGER_RELATIVE_TO_GIT_COMMON_DIR = Path(
    "gfmcd_target_attempts/r3d"
)
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(R3C_RUNNER.parent))

import run_r3c_acceptor_head_confirmation as r3c  # noqa: E402
from gfmcd.provenance import ExperimentManifest, sha256_file  # noqa: E402
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
from gfmcd.splicing.coupling import (  # noqa: E402
    CouplingConfig,
    MOTIF_CLEAN_DONOR_ALT,
    planted_donor_motifs,
)
from gfmcd.splicing.paired_component import (  # noqa: E402
    crossing_boundary_sign_flip_tests,
    gene_sign_flip,
    invert_boundary_sign_flip_tests,
    one_sided_binomial_bounds,
)
from gfmcd.splicing.reverse_coupling import donor_alt_admissible  # noqa: E402


BOUND_CODE_SOURCES = tuple(
    dict.fromkeys(
        (
            R3C_RUNNER,
            STAT_CORE,
            r3c.HOOK_CORE,
            r3c.ASSAY_CORE,
            r3c.CIRCUIT_CORE,
            *r3c.OTHER_BOUND_SOURCES,
        )
    )
)


RESULT_SCHEMA = "gfmcd.r3d.acceptor_specificity_adjudication/1"
REGISTRY_SCHEMA = "gfmcd.r3d.exposed_gene_registry/1"
REGISTRY_BUILDER_RELATIVE = (
    "scripts/splicebert_acceptor_coupling/build_r3d_train_exposure_registry.py"
)
REGISTRY_SCOPE = (
    "claude_inverse_acceptor_and_clean_alt_head_screen_direct_exposures_v1"
)
INSTRUMENT_REVISION = (
    "r3d-v1-train-endpoint-test-inversion-fixed-sequence-smoke-mechanics-split"
)

# The protocol/input/instrument pins authorize one bindable production preflight.
# Its result-package pins remain None until that preflight is independently audited.
# The normalized runner fingerprint ignores only these assignments, never executable
# code.
FROZEN_PROTOCOL_SHA256: str | None = "e319694f3f0b53695e787a67b5bcb6f1d016ac5e148e36e3050e090284f66592"
FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = "0f35295643238e9360d22bfe550e981e7788fa0975a3a9fdfe96c5f5f94fa5de"
FROZEN_PREFLIGHT_ARTIFACT_SHA256: str | None = None
FROZEN_PREFLIGHT_MANIFEST_SHA256: str | None = None
FROZEN_EXPOSED_GENE_REGISTRY_SHA256: str | None = "17ad316d5f9c478e37b91f34d87bc1d0c6a714f411947419e9dbcced2443598c"
FROZEN_EXPOSED_GENE_REGISTRY_MANIFEST_SHA256: str | None = "745d7a31837c3575edcd9816464f82fc7f23b440b00b06fde4f6185a5f50afde"
FROZEN_R3C_V2_1_RUNNER_SHA256: str | None = "25db5d95e7c5a99697f69d2153ec0d770571e3063106aa13c7f00f74e5065e31"
FROZEN_R3C_V2_1_PREFLIGHT_SHA256: str | None = "c0c9306fe894ce73bfd981a14c85e2aec6660da33f01d9716cee4327b3ff2d3e"

PREFLIGHT_N_GENES = 256
TARGET_MIN_GENES = 2500
SMOKE_N_GENES = 32
PRODUCTION_BATCH = 32
PRODUCTION_SEED = 0
PRODUCTION_RANDOMISATION = 4000
PRODUCTION_TRIALS = 200
PRODUCTION_PLANT_TRIALS = 500
SMOKE_RANDOMISATION = 199
SMOKE_TRIALS = 2
MIN_GENES = 30
SESOI = 0.05
PATH_ALPHA = 0.05
PLANT_DELTA = 0.40
MAX_FACE_FALSE_POSITIVES = 20
MIN_PLANT_SUCCESSES = 462
FACE_FAIL_PROBABILITY_AT_POINT05 = 0.001159908250747922
FACE_SUITE_UNION_BOUND_AT_POINT05 = 0.041756697026925194
FACE_SUITE_PASS_LOWER_AT_POINT05 = 0.9582433029730748
FACE_CP_UPPER_AT_MAX_COUNT = 0.14198925159094505
PLANT_CP_LOWER_AT_THRESHOLD = 0.9015717845992148
PLANT_PASS_PROBABILITY_AT_POINT95 = 0.9953942562690719
DECOY_TOL = r3c.DECOY_TOL

MEAN_MODE = r3c.MEAN_MODE
NOISING_MODE = r3c.NOISING_MODE
PRIMITIVE_ENDPOINTS = ("partner", "pseudo")
ENDPOINTS = ("partner", "pseudo", "specific")

L3H1 = ("head", 3, 1)
ROUTE_COMPONENTS = (
    ("head", 4, 11),
    ("head", 4, 9),
    ("head", 4, 0),
    ("head", 4, 2),
    ("head", 5, 11),
)
NOISING_COMPONENTS = (L3H1,) + ROUTE_COMPONENTS
COMPONENTS = NOISING_COMPONENTS
DELIVERY_CANDIDATES = ("L4H11", "L4H9")
DONOR_ROUTE_COMPARATORS = ("L4H0", "L4H2", "L5H11")
DELIVERY_GAP_ENDPOINTS = ("partner", "specific")

PATH_A = "L3H1_reused_component"
PATH_B = "distinct_acceptor_route_candidate"
PATH_NAMES = (PATH_A, PATH_B)
PREFLIGHT_GATE_NAMES = (
    "minimum_independent_genes",
    "explicit_preconstruction_train_chromosome_restriction",
    "one_hash_ranked_row_per_gene",
    "frozen_registry_train_capacity_replayed",
    "target_excludes_registry_or_preflight_uses_registry",
    "native_paired_donor_positive_control",
    "matched_pseudo_donor_negative_control",
    "single_acceptor_G_and_GT_to_AC_only",
    "zero_planted_donor_motifs",
    "exact_14_primitive_23_row_family",
    "three_way_monotone_health_registry_valid",
    "preflight_path_a_health_authorizes_target_entry",
    "r3c_v2_1_preflight_health_bound",
    "bounded_identical_policy_health_executed",
    "all_36_boundary_faces_at_most_20_of_200",
    "coherent_plant_at_least_462_of_500",
    "test_inversion_is_decision_instrument",
)

CANONICAL_CHROMOSOMES = tuple(
    [f"chr{index}" for index in range(1, 23)] + ["chrX", "chrY"]
)
VALIDATION_CHROMOSOMES = tuple(CouplingConfig().val_chromosomes)
TEST_CHROMOSOMES = tuple(CouplingConfig().test_chromosomes)
FORBIDDEN_CHROMOSOMES = frozenset(
    (*VALIDATION_CHROMOSOMES, *TEST_CHROMOSOMES)
)
TRAIN_CHROMOSOMES = tuple(
    chromosome
    for chromosome in CANONICAL_CHROMOSOMES
    if chromosome not in FORBIDDEN_CHROMOSOMES
)
if set(TRAIN_CHROMOSOMES) & FORBIDDEN_CHROMOSOMES:  # pragma: no cover
    raise RuntimeError("R3d train chromosome restriction includes a forbidden chromosome")

CURRENT_TAG = "r3d-train-component-adjudication-v1"
ROW_RANK_SALT = "r3d-train-one-row-per-gene-v1"
GENE_RANK_SALT = "r3d-train-gene-order-v1"


class TargetSealError(RuntimeError):
    """Raised before any caller-controlled path is inspected."""


def canonical_sha256(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(rendered).hexdigest()


def face_max_count_pass(false_positives: int, trials: int) -> bool:
    if not isinstance(false_positives, int) or isinstance(false_positives, bool):
        raise ValueError("face false-positive count must be an integer")
    if false_positives < 0 or false_positives > trials or trials < 1:
        raise ValueError("face false-positive count/trials are invalid")
    threshold = MAX_FACE_FALSE_POSITIVES if trials == PRODUCTION_TRIALS else trials
    return false_positives <= threshold


def plant_count_pass(successes: int, trials: int) -> bool:
    if not isinstance(successes, int) or isinstance(successes, bool):
        raise ValueError("plant success count must be an integer")
    if successes < 0 or successes > trials or trials < 1:
        raise ValueError("plant success count/trials are invalid")
    threshold = MIN_PLANT_SUCCESSES if trials == PRODUCTION_PLANT_TRIALS else trials
    return successes >= threshold


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    shape = ",".join(str(value) for value in array.shape)
    digest = hashlib.sha256()
    digest.update(f"float64|{shape}|".encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def component_name(component: tuple[str, int, int | None]) -> str:
    return r3c.component_name(component)


def primitive_row_name(mode: str, head: str, endpoint: str) -> str:
    return f"{mode}|{head}|{endpoint}"


def family_row_name(mode: str, head: str, endpoint: str) -> str:
    return f"{mode}|{head}|{endpoint}"


def delivery_gap_row_name(endpoint: str) -> str:
    return f"{NOISING_MODE}|delivery_gap|{endpoint}"


def primitive_row_names() -> list[str]:
    rows = [
        primitive_row_name(MEAN_MODE, "L3H1", endpoint)
        for endpoint in PRIMITIVE_ENDPOINTS
    ]
    rows.extend(
        primitive_row_name(NOISING_MODE, component_name(component), endpoint)
        for component in NOISING_COMPONENTS
        for endpoint in PRIMITIVE_ENDPOINTS
    )
    if len(rows) != 14 or len(set(rows)) != 14:
        raise RuntimeError("R3d requires exactly 14 unique primitive rows")
    return rows


def family_row_names() -> list[str]:
    rows = [
        family_row_name(MEAN_MODE, "L3H1", endpoint) for endpoint in ENDPOINTS
    ]
    rows.extend(
        family_row_name(NOISING_MODE, component_name(component), endpoint)
        for component in NOISING_COMPONENTS
        for endpoint in ENDPOINTS
    )
    rows.extend(delivery_gap_row_name(endpoint) for endpoint in DELIVERY_GAP_ENDPOINTS)
    if len(rows) != 23 or len(set(rows)) != 23:
        raise RuntimeError("R3d requires exactly 23 unique derived rows")
    return rows


def _component_requirements(mode: str, head: str) -> dict[str, str]:
    return {
        family_row_name(mode, head, "partner"): "MATERIAL_POSITIVE",
        family_row_name(mode, head, "pseudo"): "EQUIVALENT_WITHIN_SESOI",
        family_row_name(mode, head, "specific"): "MATERIAL_POSITIVE",
    }


def named_path_specs() -> dict[str, dict[str, Any]]:
    path_a = {
        **_component_requirements(MEAN_MODE, "L3H1"),
        **_component_requirements(NOISING_MODE, "L3H1"),
    }
    path_b = {
        **_component_requirements(NOISING_MODE, "L4H11"),
        **_component_requirements(NOISING_MODE, "L4H9"),
        delivery_gap_row_name("partner"): "MATERIAL_POSITIVE",
        delivery_gap_row_name("specific"): "MATERIAL_POSITIVE",
    }
    for head in DONOR_ROUTE_COMPARATORS:
        for endpoint in ENDPOINTS:
            path_b[family_row_name(NOISING_MODE, head, endpoint)] = (
                "EQUIVALENT_WITHIN_SESOI"
            )
    specs = {
        PATH_A: {
            "requirements": path_a,
            "alpha": PATH_ALPHA,
            "gate": None,
            "tier": "PROSPECTIVE_TRAIN_ENDPOINT_REUSE_VERIFICATION",
        },
        PATH_B: {
            "requirements": path_b,
            "alpha": PATH_ALPHA,
            "gate": PATH_A,
            "tier": "PROSPECTIVE_TRAIN_ENDPOINT_ROUTE_VERIFICATION",
        },
    }
    if len(path_a) != 6 or len(path_b) != 17:
        raise RuntimeError("R3d path sizes must remain 6 and 17")
    return specs


def endpoint_vectors(partner: np.ndarray, pseudo: np.ndarray) -> dict[str, np.ndarray]:
    return r3c.endpoint_vectors(partner, pseudo)


def build_primitive_matrix(
    measurements: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> tuple[np.ndarray, list[str]]:
    values = [
        np.asarray(measurements[MEAN_MODE]["L3H1"][endpoint], dtype=float)
        for endpoint in PRIMITIVE_ENDPOINTS
    ]
    values.extend(
        np.asarray(
            measurements[NOISING_MODE][component_name(component)][endpoint],
            dtype=float,
        )
        for component in NOISING_COMPONENTS
        for endpoint in PRIMITIVE_ENDPOINTS
    )
    matrix = np.vstack(values)
    names = primitive_row_names()
    if matrix.shape[0] != 14 or matrix.shape[1] == 0 or not np.isfinite(matrix).all():
        raise ValueError("R3d primitive matrix must have finite [14, examples] shape")
    return matrix, names


def _derived_endpoint(
    primitives: Mapping[str, np.ndarray], mode: str, head: str, endpoint: str
) -> np.ndarray:
    partner = np.asarray(primitives[primitive_row_name(mode, head, "partner")])
    pseudo = np.asarray(primitives[primitive_row_name(mode, head, "pseudo")])
    if endpoint == "partner":
        return partner
    if endpoint == "pseudo":
        return pseudo
    if endpoint == "specific":
        return partner - pseudo
    raise ValueError(f"unknown endpoint: {endpoint}")


def derive_family_from_primitives(
    primitive: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(primitive, dtype=float)
    primitive_names = primitive_row_names()
    if (
        values.ndim != 2
        or values.shape[0] != len(primitive_names)
        or values.shape[1] == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("primitive must use the finite frozen 14-row order")
    by_name = {name: values[index] for index, name in enumerate(primitive_names)}
    derived = [
        _derived_endpoint(by_name, MEAN_MODE, "L3H1", endpoint)
        for endpoint in ENDPOINTS
    ]
    derived.extend(
        _derived_endpoint(by_name, NOISING_MODE, component_name(component), endpoint)
        for component in NOISING_COMPONENTS
        for endpoint in ENDPOINTS
    )
    for endpoint in DELIVERY_GAP_ENDPOINTS:
        candidates = np.mean(
            [
                _derived_endpoint(by_name, NOISING_MODE, head, endpoint)
                for head in DELIVERY_CANDIDATES
            ],
            axis=0,
        )
        comparators = np.mean(
            [
                _derived_endpoint(by_name, NOISING_MODE, head, endpoint)
                for head in DONOR_ROUTE_COMPARATORS
            ],
            axis=0,
        )
        derived.append(candidates - comparators)
    family = np.vstack(derived)
    names = family_row_names()
    if family.shape[0] != 23:
        raise RuntimeError("R3d family algebra drifted from 23 rows")
    return family, names


def center_primitive_rows(primitive: np.ndarray) -> np.ndarray:
    values = np.asarray(primitive, dtype=float)
    if values.ndim != 2 or values.shape[0] != 14 or not np.isfinite(values).all():
        raise ValueError("R3d centering requires a finite 14-row primitive matrix")
    return values - values.mean(axis=1, keepdims=True)


def _empty_offset() -> tuple[np.ndarray, dict[str, int]]:
    names = primitive_row_names()
    return np.zeros(len(names), dtype=float), {name: i for i, name in enumerate(names)}


def _set_component(
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


def coherent_path_offset(delta: float = PLANT_DELTA) -> np.ndarray:
    if not np.isfinite(delta) or delta <= SESOI:
        raise ValueError("coherent R3d plant must be finite and exceed the SESOI")
    offset, lookup = _empty_offset()
    for mode, head in (
        (MEAN_MODE, "L3H1"),
        (NOISING_MODE, "L3H1"),
        (NOISING_MODE, "L4H11"),
        (NOISING_MODE, "L4H9"),
    ):
        _set_component(offset, lookup, mode, head, partner=delta, pseudo=0.0)
    family, names = derive_family_from_primitives(offset[:, None])
    observed = {name: float(family[index, 0]) for index, name in enumerate(names)}
    for spec in named_path_specs().values():
        for row, required in spec["requirements"].items():
            expected = delta if required == "MATERIAL_POSITIVE" else 0.0
            if not np.isclose(observed[row], expected, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"incoherent R3d plant at {row}")
    return offset


def _easy_path_offset(path: str) -> tuple[np.ndarray, dict[str, int]]:
    offset, lookup = _empty_offset()
    for row, requirement in named_path_specs()[path]["requirements"].items():
        if requirement != "MATERIAL_POSITIVE":
            continue
        mode, head, endpoint = row.split("|")
        if head == "delivery_gap":
            continue
        if endpoint not in {"partner", "specific"}:
            raise RuntimeError(f"positive row is not primitive-derivable: {row}")
        offset[lookup[primitive_row_name(mode, head, "partner")]] = PLANT_DELTA
    return offset, lookup


def _component_face(
    offset: np.ndarray,
    lookup: Mapping[str, int],
    mode: str,
    head: str,
    face: str,
) -> tuple[str, str]:
    if face == "partner_boundary":
        _set_component(
            offset, lookup, mode, head, partner=SESOI, pseudo=-SESOI / 2
        )
        return family_row_name(mode, head, "partner"), "positive_boundary"
    if face == "specific_boundary":
        _set_component(
            offset, lookup, mode, head, partner=1.5 * SESOI, pseudo=SESOI / 2
        )
        return family_row_name(mode, head, "specific"), "positive_boundary"
    if face == "pseudo_lower_edge":
        _set_component(
            offset, lookup, mode, head, partner=PLANT_DELTA, pseudo=-SESOI
        )
        return family_row_name(mode, head, "pseudo"), "equivalence_lower_edge"
    if face == "pseudo_upper_edge":
        _set_component(
            offset, lookup, mode, head, partner=PLANT_DELTA, pseudo=SESOI
        )
        return family_row_name(mode, head, "pseudo"), "equivalence_upper_edge"
    raise ValueError(f"unknown component face: {face}")


def _validate_surface(name: str, spec: Mapping[str, Any]) -> None:
    family, names = derive_family_from_primitives(
        np.asarray(spec["offset"], dtype=float)[:, None]
    )
    observed = {row: float(family[index, 0]) for index, row in enumerate(names)}
    path = str(spec["path"])
    target_row = str(spec["target_row"])
    edge = str(spec["edge"])
    requirements = named_path_specs()[path]["requirements"]
    if target_row not in requirements:
        raise RuntimeError(f"surface {name} targets a row outside {path}")
    for row, required in requirements.items():
        value = observed[row]
        if row == target_row and edge == "positive_boundary":
            ok = np.isclose(value, SESOI, rtol=0.0, atol=1e-12)
        elif row == target_row and edge == "equivalence_lower_edge":
            ok = np.isclose(value, -SESOI, rtol=0.0, atol=1e-12)
        elif row == target_row and edge == "equivalence_upper_edge":
            ok = np.isclose(value, SESOI, rtol=0.0, atol=1e-12)
        elif required == "MATERIAL_POSITIVE":
            ok = value > SESOI
        else:
            ok = -SESOI < value < SESOI
        if not ok:
            raise RuntimeError(
                f"surface {name} violates {path}:{row} "
                f"(value={value}, target={target_row}, edge={edge})"
            )


def boundary_face_surface_specs() -> dict[str, dict[str, Any]]:
    """Return 8 L3H1 plus 28 route faces, never packed across gate levels."""
    surfaces: dict[str, dict[str, Any]] = {}

    def add(
        name: str,
        *,
        path: str,
        offset: np.ndarray,
        target_row: str,
        edge: str,
    ) -> None:
        record = {
            "path": path,
            "target_row": target_row,
            "edge": edge,
            "offset": offset.tolist(),
            "offset_sha256": _array_sha256(offset),
        }
        _validate_surface(name, record)
        surfaces[name] = record

    component_faces = (
        "partner_boundary",
        "specific_boundary",
        "pseudo_lower_edge",
        "pseudo_upper_edge",
    )
    for mode in (MEAN_MODE, NOISING_MODE):
        for face in component_faces:
            offset, lookup = _easy_path_offset(PATH_A)
            row, edge = _component_face(offset, lookup, mode, "L3H1", face)
            add(
                f"path_a_{mode}_{face}",
                path=PATH_A,
                offset=offset,
                target_row=row,
                edge=edge,
            )

    for head in DELIVERY_CANDIDATES:
        for face in component_faces:
            offset, lookup = _easy_path_offset(PATH_B)
            row, edge = _component_face(offset, lookup, NOISING_MODE, head, face)
            add(
                f"path_b_{head}_{face}",
                path=PATH_B,
                offset=offset,
                target_row=row,
                edge=edge,
            )

    gap_faces = {
        "partner": (0.075, -0.025),
        "specific": (0.10, 0.025),
    }
    for endpoint, (candidate_partner, candidate_pseudo) in gap_faces.items():
        offset, lookup = _easy_path_offset(PATH_B)
        for head in DELIVERY_CANDIDATES:
            _set_component(
                offset,
                lookup,
                NOISING_MODE,
                head,
                partner=candidate_partner,
                pseudo=candidate_pseudo,
            )
        for head in DONOR_ROUTE_COMPARATORS:
            _set_component(
                offset, lookup, NOISING_MODE, head, partner=0.025, pseudo=0.0
            )
        add(
            f"path_b_delivery_gap_{endpoint}_boundary",
            path=PATH_B,
            offset=offset,
            target_row=delivery_gap_row_name(endpoint),
            edge="positive_boundary",
        )

    comparator_faces = {
        "partner_lower": (-SESOI, -SESOI / 2, "partner", "equivalence_lower_edge"),
        "partner_upper": (SESOI, SESOI / 2, "partner", "equivalence_upper_edge"),
        "pseudo_lower": (-SESOI / 2, -SESOI, "pseudo", "equivalence_lower_edge"),
        "pseudo_upper": (SESOI / 2, SESOI, "pseudo", "equivalence_upper_edge"),
        "specific_lower": (
            -SESOI / 2,
            SESOI / 2,
            "specific",
            "equivalence_lower_edge",
        ),
        "specific_upper": (
            SESOI / 2,
            -SESOI / 2,
            "specific",
            "equivalence_upper_edge",
        ),
    }
    for head in DONOR_ROUTE_COMPARATORS:
        for face, (partner, pseudo, endpoint, edge) in comparator_faces.items():
            offset, lookup = _easy_path_offset(PATH_B)
            _set_component(
                offset,
                lookup,
                NOISING_MODE,
                head,
                partner=partner,
                pseudo=pseudo,
            )
            add(
                f"path_b_{head}_{face}",
                path=PATH_B,
                offset=offset,
                target_row=family_row_name(NOISING_MODE, head, endpoint),
                edge=edge,
            )
    if len(surfaces) != 36:
        raise RuntimeError("R3d preflight requires exactly 36 boundary faces")
    if sum(spec["path"] == PATH_A for spec in surfaces.values()) != 8:
        raise RuntimeError("R3d boundary suite must contain 8 path-A faces")
    if sum(spec["path"] == PATH_B for spec in surfaces.values()) != 28:
        raise RuntimeError("R3d boundary suite must contain 28 path-B faces")
    return surfaces


def boundary_face_suite_binding() -> dict[str, Any]:
    surfaces = boundary_face_surface_specs()
    compact = {
        name: {key: value for key, value in spec.items() if key != "offset"}
        for name, spec in surfaces.items()
    }
    return {
        "surface_order": list(surfaces),
        "surface_count": len(surfaces),
        "path_a_faces": 8,
        "path_b_faces": 28,
        "surfaces": compact,
        "suite_sha256": canonical_sha256(surfaces),
    }


def decision_edges(path: str) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for row, requirement in named_path_specs()[path]["requirements"].items():
        if requirement == "MATERIAL_POSITIVE":
            edges.append(
                {
                    "name": f"{row}|positive_edge",
                    "row": row,
                    "boundary": SESOI,
                    "alternative": "greater",
                    "bound": "lower",
                }
            )
        elif requirement == "EQUIVALENT_WITHIN_SESOI":
            edges.extend(
                (
                    {
                        "name": f"{row}|lower_edge",
                        "row": row,
                        "boundary": -SESOI,
                        "alternative": "greater",
                        "bound": "lower",
                    },
                    {
                        "name": f"{row}|upper_edge",
                        "row": row,
                        "boundary": SESOI,
                        "alternative": "less",
                        "bound": "upper",
                    },
                )
            )
        else:  # pragma: no cover - guarded by the frozen registry
            raise RuntimeError(f"unknown path requirement: {requirement}")
    expected = 8 if path == PATH_A else 28
    if len(edges) != expected:
        raise RuntimeError(f"{path} must have exactly {expected} decision edges")
    return edges


def _dataclass_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"cannot serialize inference record {type(value)!r}")


def _inversion_bound(record: Any, side: str) -> float:
    """Read the deliberately small public contract of the paired-component helper."""
    payload = _dataclass_dict(record)
    candidates = (
        ("lower_bound", "bound", "transition_boundary")
        if side == "lower"
        else ("upper_bound", "bound", "transition_boundary")
    )
    for key in candidates:
        value = payload.get(key)
        if value is not None and np.isfinite(float(value)):
            return float(value)
    raise RuntimeError(f"inversion record omits its finite {side} bound")


def _strict_classification(lo: float, hi: float) -> str:
    if lo > SESOI:
        return "MATERIAL_POSITIVE"
    if hi < -SESOI:
        return "MATERIAL_NEGATIVE"
    if lo > -SESOI and hi < SESOI:
        return "EQUIVALENT_WITHIN_SESOI"
    return "UNRESOLVED"


def infer_one_path(
    family: np.ndarray,
    groups: Sequence[str],
    row_names: Sequence[str],
    path: str,
    *,
    n_draws: int,
    seed: int,
    namespace: str,
    health: Mapping[str, Any],
) -> dict[str, Any]:
    """Test and invert one path; p-value and bound use the same strict edge test."""
    values = np.asarray(family, dtype=float)
    if values.shape != (23, len(groups)) or list(row_names) != family_row_names():
        raise ValueError("R3d inference requires the exact aligned 23-row family")
    by_name = {name: values[index] for index, name in enumerate(row_names)}
    edges = decision_edges(path)
    edge_values = np.vstack([by_name[edge["row"]] for edge in edges])
    edge_names = [edge["name"] for edge in edges]
    alternatives = [edge["alternative"] for edge in edges]
    boundaries = [edge["boundary"] for edge in edges]
    tests = crossing_boundary_sign_flip_tests(
        edge_values,
        groups,
        edge_names,
        boundaries=boundaries,
        alternatives=alternatives,
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=MIN_GENES,
    )
    inversions = invert_boundary_sign_flip_tests(
        edge_values,
        groups,
        edge_names,
        alternatives=alternatives,
        alphas=[PATH_ALPHA] * len(edges),
        n_draws=n_draws,
        seed=seed,
        namespace=namespace,
        min_genes=MIN_GENES,
    )
    requirements = named_path_specs()[path]["requirements"]
    rows: dict[str, Any] = {}
    all_edges_reject = True
    coherence = True
    for row, requirement in requirements.items():
        row_edges = [edge for edge in edges if edge["row"] == row]
        lower = float("-inf")
        upper = float("inf")
        edge_output: dict[str, Any] = {}
        for edge in row_edges:
            test = tests[edge["name"]]
            inversion = inversions[edge["name"]]
            if (
                inversion.sign_matrix_sha256 != test.sign_matrix_sha256
                or inversion.ordered_gene_labels_sha256
                != test.ordered_gene_labels_sha256
            ):
                raise RuntimeError(
                    "reference test and inversion did not share exact signs/gene order"
                )
            rejected = bool(test.p_value < PATH_ALPHA)
            if rejected != bool(inversion.rejects(edge["boundary"])):
                raise RuntimeError(
                    "reference-boundary p-value and inverted bound are not exact duals"
                )
            all_edges_reject &= rejected
            bound = _inversion_bound(inversion, edge["bound"])
            if edge["bound"] == "lower":
                lower = bound
            else:
                upper = bound
            edge_output[edge["name"]] = {
                "reference_test": _dataclass_dict(test),
                "inversion": _dataclass_dict(inversion),
                "strict_reject": rejected,
            }
        classification = _strict_classification(lower, upper)
        row_pass = classification == requirement
        edge_pass = all(
            edge_output[edge["name"]]["strict_reject"] for edge in row_edges
        )
        row_edge_p_values = {
            edge["name"]: float(tests[edge["name"]].p_value)
            for edge in row_edges
        }
        coherence &= row_pass == edge_pass
        rows[row] = {
            "estimate": float(np.mean(by_name[row])),
            "test_inverted_interval": [
                lower if np.isfinite(lower) else None,
                upper if np.isfinite(upper) else None,
            ],
            "required": requirement,
            "classification": classification,
            "strict_requirement_pass": row_pass,
            "strict_test_interval_coherence": row_pass == edge_pass,
            "edge_p_values": row_edge_p_values,
            "intersection_union_p_value": max(row_edge_p_values.values()),
            "edges": edge_output,
        }
    supported = bool(
        health.get("ok") is True
        and all_edges_reject
        and coherence
        and all(row["strict_requirement_pass"] for row in rows.values())
    )
    sign_digests = {
        test.sign_matrix_sha256 for test in tests.values()
    }
    gene_digests = {
        test.ordered_gene_labels_sha256 for test in tests.values()
    }
    edge_p_values = {
        name: float(test.p_value) for name, test in tests.items()
    }
    return {
        "path": path,
        "alpha": PATH_ALPHA,
        "strict_threshold": "p < alpha",
        "requirements": requirements,
        "health": dict(health),
        "rows": rows,
        "all_edges_reject": bool(all_edges_reject),
        "edge_p_values": edge_p_values,
        "intersection_union_p_value": max(edge_p_values.values()),
        "test_interval_coherence": bool(coherence),
        "classification": (
            "SUPPORTED"
            if supported
            else "HEALTH_FAILED"
            if health.get("ok") is not True
            else "NOT_SUPPORTED"
        ),
        "randomisation_audit": {
            "unit": "whole gene",
            "B": n_draws,
            "seed": seed,
            "namespace": namespace,
            "edge_order": edge_names,
            "edge_order_sha256": canonical_sha256(edge_names),
            "one_shared_sign_matrix": len(sign_digests) == 1,
            "sign_matrix_sha256": next(iter(sign_digests)) if len(sign_digests) == 1 else None,
            "one_ordered_gene_digest": len(gene_digests) == 1,
            "ordered_gene_labels_sha256": (
                next(iter(gene_digests)) if len(gene_digests) == 1 else None
            ),
            "add_one_floor": 1 / (n_draws + 1),
        },
    }


def fixed_sequence_inference(
    family: np.ndarray,
    groups: Sequence[str],
    row_names: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    namespace: str,
    path_health: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    path_a = infer_one_path(
        family,
        groups,
        row_names,
        PATH_A,
        n_draws=n_draws,
        seed=seed,
        namespace=f"{namespace}:path-a",
        health=path_health[PATH_A],
    )
    if path_a["classification"] == "SUPPORTED":
        path_b = infer_one_path(
            family,
            groups,
            row_names,
            PATH_B,
            n_draws=n_draws,
            seed=seed,
            namespace=f"{namespace}:path-b-after-path-a",
            health=path_health[PATH_B],
        )
    else:
        path_b = {
            "path": PATH_B,
            "alpha": PATH_ALPHA,
            "gate": PATH_A,
            "classification": "NOT_TESTED_GATE_CLOSED",
            "requirements": named_path_specs()[PATH_B]["requirements"],
            "health": dict(path_health[PATH_B]),
            "rows": {},
            "reason": "fixed-sequence gate requires path A SUPPORTED",
        }
    return {
        "schema": "gfmcd.r3d.fixed_sequence_inference/1",
        "studywise_alpha": PATH_ALPHA,
        "gatekeeping": (
            "Path A tested at alpha=.05; Path B tested at alpha=.05 only after Path A "
            "is supported. This fixed sequence strongly controls studywise FWER."
        ),
        "path_order": list(PATH_NAMES),
        "paths": {PATH_A: path_a, PATH_B: path_b},
    }


def _path_health_participants() -> dict[str, list[tuple[str, str]]]:
    return {
        PATH_A: [(MEAN_MODE, "L3H1"), (NOISING_MODE, "L3H1")],
        PATH_B: [
            (NOISING_MODE, "L4H11"),
            (NOISING_MODE, "L4H9"),
            (NOISING_MODE, "L4H0"),
            (NOISING_MODE, "L4H2"),
            (NOISING_MODE, "L5H11"),
        ],
    }


def observed_path_health(
    measurements: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path, participants in _path_health_participants().items():
        failed = [
            f"{mode}|{head}"
            for mode, head in participants
            if measurements[mode][head]["health"].get("ok") is not True
        ]
        output[path] = {
            "ok": not failed,
            "participants": [f"{mode}|{head}" for mode, head in participants],
            "failed_participants": failed,
        }
    return output


def r3c_preflight_health(record: Mapping[str, Any]) -> dict[str, bool]:
    if (
        record.get("schema") != r3c.RESULT_SCHEMA
        or record.get("stage") != "preflight"
        or record.get("verdict") != "TARGET_BINDABLE_PREFLIGHT_GO"
        or record.get("all_gates_pass") is not True
        or record.get("target_bindable_preflight") is not True
        or record.get("construction", {}).get("all_single_acceptor_G") is not True
        or record.get("construction", {}).get("policy", {}).get(
            "donor_readout_constructed"
        )
        is not False
        or record.get("hook_capability", {}).get("all_controls_pass") is not True
    ):
        raise ValueError("R3d requires the clean target-bindable R3c v2.1 preflight")
    mapping = {PATH_A: PATH_A, PATH_B: PATH_B}
    health: dict[str, bool] = {}
    for path, source in mapping.items():
        path_record = record.get("paths", {}).get(source, {})
        health[path] = bool(path_record.get("health", {}).get("ok") is True)
    if health[PATH_A] is not True:
        raise ValueError("R3c v2.1 Path-A health makes every R3d path unreachable")
    return health


def three_way_monotone_health(
    *,
    r3c_health: Mapping[str, bool],
    current_health: Mapping[str, Mapping[str, Any]],
    r3d_preflight_paths: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for path in PATH_NAMES:
        prior_r3d = (
            True
            if r3d_preflight_paths is None
            else r3d_preflight_paths.get(path, {}).get("health", {}).get("ok") is True
        )
        current = current_health[path].get("ok") is True
        ok = bool(r3c_health.get(path) is True and prior_r3d and current)
        output[path] = {
            "ok": ok,
            "r3c_v2_1_preflight_health_ok": r3c_health.get(path) is True,
            "r3d_preflight_health_ok": prior_r3d,
            "current_bucket_health_ok": current,
            "can_only_shrink": True,
            "participants": list(current_health[path]["participants"]),
            "failed_participants": list(current_health[path]["failed_participants"]),
            "authorization_rule": (
                "R3c v2.1 preflight AND R3d preflight AND current target health; "
                "no later stage may revive a suppressed path"
            ),
        }
    return output


def _hash_rank(salt: str, *values: Any) -> str:
    return hashlib.sha256(
        "|".join([salt, *(str(value) for value in values)]).encode()
    ).hexdigest()


def read_exposed_gene_registry(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    record = json.loads(source.read_text())
    genes = record.get("gene_ids")
    if (
        record.get("schema") != REGISTRY_SCHEMA
        or record.get("scope_id") != REGISTRY_SCOPE
        or not isinstance(record.get("scope"), Mapping)
        or record.get("scope", {}).get("global_freshness_claimed") is not False
        or not isinstance(genes, list)
        or not genes
        or any(not isinstance(gene, str) or not gene for gene in genes)
        or genes != sorted(set(genes))
    ):
        raise ValueError(
            "exposed-gene registry must use the frozen direct-exposure scope and sorted unique gene_ids"
        )
    if record.get("gene_ids_sha256") != canonical_sha256(genes):
        raise ValueError("exposed-gene registry gene_ids digest does not replay")
    return record


def _eligible_train_rows(args: SimpleNamespace) -> tuple[list[tuple[Any, int]], Counter]:
    """Apply the chromosome restriction during GTF load, before any FASTA fetch."""
    census = load_examples_unscored(
        args.gtf,
        args.fasta,
        "train",
        None,
        tag=CURRENT_TAG,
        allowed_chromosomes=TRAIN_CHROMOSOMES,
    )
    exclusions: Counter[str] = Counter()
    eligible: list[tuple[Any, int]] = []
    for example in census:
        if str(example.chrom) not in set(TRAIN_CHROMOSOMES):
            raise RuntimeError("preconstruction train restriction was not respected")
        decoy = matched_decoy_donor(example)
        if decoy is None:
            exclusions["missing_matched_decoy_donor"] += 1
            continue
        decoy = int(decoy)
        if decoy == int(example.donor_pos):
            exclusions["pseudo_donor_equals_paired_donor"] += 1
            continue
        if abs(decoy - int(example.donor_pos)) > DECOY_TOL:
            exclusions["donor_pseudo_motif_outside_distance_caliper"] += 1
            continue
        if not donor_alt_admissible(example, MOTIF_CLEAN_DONOR_ALT):
            exclusions["paired_donor_edit_inadmissible"] += 1
            continue
        if planted_donor_motifs(example.sequence, decoy, MOTIF_CLEAN_DONOR_ALT):
            exclusions["pseudo_donor_edit_plants_splice_motif"] += 1
            continue
        eligible.append((example, decoy))
    return eligible, exclusions


def one_hash_ranked_row_per_gene(
    rows: Sequence[tuple[Any, int]],
) -> list[tuple[Any, int]]:
    selected: dict[str, tuple[Any, int]] = {}
    ranks: dict[str, str] = {}
    for example, decoy in rows:
        gene = str(example.gene_id)
        rank = _hash_rank(
            ROW_RANK_SALT, gene, example.example_id, int(decoy)
        )
        if gene not in ranks or rank < ranks[gene]:
            selected[gene] = (example, int(decoy))
            ranks[gene] = rank
    return sorted(
        selected.values(),
        key=lambda row: (
            _hash_rank(GENE_RANK_SALT, str(row[0].gene_id)),
            str(row[0].gene_id),
        ),
    )


def _row_record(example: Any, decoy: int) -> dict[str, Any]:
    return {
        "example_id": str(example.example_id),
        "gene_id": str(example.gene_id),
        "chrom": str(example.chrom),
        "decoy_donor": int(decoy),
    }


def train_capacity_record(
    eligible_rows: Sequence[tuple[Any, int]],
    registry_genes: Sequence[str],
    exclusions: Mapping[str, int],
) -> dict[str, Any]:
    """Replay the model-free registry builder's exact train-capacity contract."""
    registry = set(map(str, registry_genes))
    raw_records = sorted(
        (_row_record(example, decoy) for example, decoy in eligible_rows),
        key=lambda row: (row["example_id"], row["decoy_donor"]),
    )
    ranked = one_hash_ranked_row_per_gene(eligible_rows)
    exposed_ranked = [row for row in ranked if str(row[0].gene_id) in registry]
    disjoint_ranked = [row for row in ranked if str(row[0].gene_id) not in registry]
    exposed_rows = [
        row for row in eligible_rows if str(row[0].gene_id) in registry
    ]
    disjoint_rows = [
        row for row in eligible_rows if str(row[0].gene_id) not in registry
    ]

    def ranked_hash(rows: Sequence[tuple[Any, int]]) -> str:
        return canonical_sha256(
            [_row_record(example, decoy) for example, decoy in rows]
        )

    return {
        "preconstruction_allowed_chromosomes": list(TRAIN_CHROMOSOMES),
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMOSOMES),
        "forbidden_chromosomes_materialized": False,
        "eligible_rows": len(eligible_rows),
        "eligible_row_records_sha256": canonical_sha256(raw_records),
        "eligible_genes_one_row_each": len(ranked),
        "eligible_gene_ids_sha256": canonical_sha256(
            sorted({str(row[0].gene_id) for row in eligible_rows})
        ),
        "exclusions": dict(
            sorted((str(key), int(value)) for key, value in exclusions.items())
        ),
        "exposed_development": {
            "rows_available": len(exposed_rows),
            "genes_one_row_each": len(exposed_ranked),
            "gene_ids_sha256": canonical_sha256(
                sorted({str(row[0].gene_id) for row in exposed_ranked})
            ),
            "ordered_one_row_per_gene_sha256": ranked_hash(exposed_ranked),
        },
        # Preserve the registry-builder field name for exact artifact replay.  R3d's
        # result language calls this registry-disjoint, never globally fresh.
        "fresh_target": {
            "rows_available": len(disjoint_rows),
            "genes_one_row_each": len(disjoint_ranked),
            "gene_ids_sha256": canonical_sha256(
                sorted({str(row[0].gene_id) for row in disjoint_ranked})
            ),
            "ordered_one_row_per_gene_sha256": ranked_hash(disjoint_ranked),
        },
        "one_row_per_gene_policy": {
            "row_rank_salt": ROW_RANK_SALT,
            "gene_rank_salt": GENE_RANK_SALT,
        },
    }


def load_cohort(args: SimpleNamespace, registry: Mapping[str, Any]):
    eligible_rows, exclusions = _eligible_train_rows(args)
    replayed_capacity = train_capacity_record(
        eligible_rows, registry["gene_ids"], exclusions
    )
    if registry.get("train_capacity") != replayed_capacity:
        raise RuntimeError(
            "R3d train census/ranking does not replay the frozen registry capacity"
        )
    ranked = one_hash_ranked_row_per_gene(eligible_rows)
    exposed = set(map(str, registry["gene_ids"]))
    exposed_rows = [row for row in ranked if str(row[0].gene_id) in exposed]
    registry_disjoint_rows = [
        row for row in ranked if str(row[0].gene_id) not in exposed
    ]
    if args.stage == "target":
        if len(registry_disjoint_rows) < TARGET_MIN_GENES:
            raise RuntimeError(
                f"R3d target requires >={TARGET_MIN_GENES} registry-disjoint genes; "
                f"found {len(registry_disjoint_rows)}"
            )
        selected_rows = registry_disjoint_rows
        selection = "every eligible gene absent from the frozen direct-exposure registry"
    else:
        required = PREFLIGHT_N_GENES if args.stage == "preflight" else args.n_genes
        if len(exposed_rows) < required:
            raise RuntimeError(
                f"R3d {args.stage} requires {required} exposed eligible genes; "
                f"found {len(exposed_rows)}"
            )
        selected_rows = exposed_rows[:required]
        selection = f"first {required} registry genes in frozen hash rank"
    selected_genes = [str(row[0].gene_id) for row in selected_rows]
    if len(selected_genes) != len(set(selected_genes)):
        raise RuntimeError("R3d must select exactly one row per independent gene")
    if args.stage == "target" and set(selected_genes) & exposed:
        raise RuntimeError("R3d target intersects the exposed-gene registry")
    records = [_row_record(*row) for row in selected_rows]
    audit = {
        "schema": "gfmcd.r3d.train_cohort_audit/1",
        "selection_reads_model_outputs": False,
        "split": "train",
        "preconstruction_allowed_chromosomes": list(TRAIN_CHROMOSOMES),
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMOSOMES),
        "provenance_hash_reads_full_fasta_bytes": True,
        "forbidden_chromosomes_materialized_as_examples": False,
        "forbidden_chromosomes_model_scored": False,
        "sealed_or_independent_claim": False,
        "screened_eligible_rows": len(eligible_rows),
        "eligible_genes_one_row_each": len(ranked),
        "eligible_exposed_genes": len(exposed_rows),
        "eligible_registry_disjoint_genes": len(registry_disjoint_rows),
        "exclusions": dict(sorted(exclusions.items())),
        "selection_rule": selection,
        "row_rank_salt": ROW_RANK_SALT,
        "gene_rank_salt": GENE_RANK_SALT,
        "selected_rows": len(selected_rows),
        "selected_genes": len(selected_genes),
        "selected_row_records": records,
        "selected_gene_ids": selected_genes,
        "one_row_per_gene": True,
        "ordered_rows_sha256": canonical_sha256(records),
        "ordered_gene_ids_sha256": canonical_sha256(selected_genes),
        "registry_gene_intersection_count": len(set(selected_genes) & exposed),
        "target_uses_every_eligible_registry_disjoint_gene": args.stage == "target",
        "target_minimum_genes": TARGET_MIN_GENES,
        "frozen_registry_capacity_replayed": True,
        "frozen_registry_capacity_sha256": canonical_sha256(replayed_capacity),
    }
    return (
        [row[0] for row in selected_rows],
        [int(row[1]) for row in selected_rows],
        audit,
    )


def authorize_target(
    protocol_sha256: str,
    *,
    protocol_text: str,
    frozen_protocol_sha256: str | None = FROZEN_PROTOCOL_SHA256,
    frozen_fingerprint_sha256: str | None = FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
    frozen_preflight_sha256: str | None = FROZEN_PREFLIGHT_ARTIFACT_SHA256,
    frozen_preflight_manifest_sha256: str | None = FROZEN_PREFLIGHT_MANIFEST_SHA256,
    frozen_registry_sha256: str | None = FROZEN_EXPOSED_GENE_REGISTRY_SHA256,
    frozen_registry_manifest_sha256: str | None = (
        FROZEN_EXPOSED_GENE_REGISTRY_MANIFEST_SHA256
    ),
    frozen_r3c_runner_sha256: str | None = FROZEN_R3C_V2_1_RUNNER_SHA256,
    frozen_r3c_preflight_sha256: str | None = FROZEN_R3C_V2_1_PREFLIGHT_SHA256,
) -> None:
    pins = {
        "protocol": frozen_protocol_sha256,
        "instrument fingerprint": frozen_fingerprint_sha256,
        "R3d preflight artifact": frozen_preflight_sha256,
        "R3d preflight manifest": frozen_preflight_manifest_sha256,
        "exposed-gene registry": frozen_registry_sha256,
        "exposed-gene registry manifest": frozen_registry_manifest_sha256,
        "R3c v2.1 runner": frozen_r3c_runner_sha256,
        "R3c v2.1 preflight": frozen_r3c_preflight_sha256,
    }
    missing = [name for name, value in pins.items() if value is None]
    if missing:
        raise TargetSealError(
            "R3d is DRAFT and target execution is hard-blocked; missing literal pins: "
            + ", ".join(missing)
        )
    malformed = [
        name
        for name, value in pins.items()
        if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
    ]
    if malformed:
        raise TargetSealError("malformed R3d literal pins: " + ", ".join(malformed))
    if not any(
        line.strip().startswith("> **Status: FROZEN")
        for line in protocol_text.splitlines()[:12]
    ):
        raise TargetSealError("R3d protocol does not declare FROZEN status")
    if protocol_sha256 != frozen_protocol_sha256:
        raise TargetSealError("R3d protocol bytes differ from the frozen pin")


def authorize_bindable_preflight(
    *,
    hashes: Mapping[str, Any],
    fingerprint: str,
    protocol_text: str,
    frozen_protocol_sha256: str | None = FROZEN_PROTOCOL_SHA256,
    frozen_fingerprint_sha256: str | None = FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
    frozen_registry_sha256: str | None = FROZEN_EXPOSED_GENE_REGISTRY_SHA256,
    frozen_registry_manifest_sha256: str | None = (
        FROZEN_EXPOSED_GENE_REGISTRY_MANIFEST_SHA256
    ),
    frozen_r3c_runner_sha256: str | None = FROZEN_R3C_V2_1_RUNNER_SHA256,
    frozen_r3c_preflight_sha256: str | None = FROZEN_R3C_V2_1_PREFLIGHT_SHA256,
) -> bool:
    """Return False for an all-draft instrument; reject every partial or drifted freeze."""
    pins = {
        "protocol": frozen_protocol_sha256,
        "instrument fingerprint": frozen_fingerprint_sha256,
        "exposed-gene registry": frozen_registry_sha256,
        "exposed-gene registry manifest": frozen_registry_manifest_sha256,
        "R3c v2.1 runner": frozen_r3c_runner_sha256,
        "R3c v2.1 preflight": frozen_r3c_preflight_sha256,
    }
    if all(value is None for value in pins.values()):
        return False
    missing = [name for name, value in pins.items() if value is None]
    if missing:
        raise TargetSealError(
            "R3d bindable preflight has a partial freeze; missing: "
            + ", ".join(missing)
        )
    malformed = [
        name
        for name, value in pins.items()
        if re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
    ]
    if malformed:
        raise TargetSealError(
            "R3d bindable preflight has malformed pins: " + ", ".join(malformed)
        )
    if not any(
        line.strip().startswith("> **Status: FROZEN")
        for line in protocol_text.splitlines()[:12]
    ):
        raise TargetSealError("R3d bindable preflight protocol is not FROZEN")
    expected = {
        "protocol_sha256": frozen_protocol_sha256,
        "exposed_gene_registry_sha256": frozen_registry_sha256,
        "exposed_gene_registry_manifest_sha256": frozen_registry_manifest_sha256,
        "r3c_v2_1_runner_sha256": frozen_r3c_runner_sha256,
        "r3c_v2_1_preflight_sha256": frozen_r3c_preflight_sha256,
    }
    drift = [
        key for key, value in expected.items() if hashes.get(key) != value
    ]
    if drift:
        raise TargetSealError(
            "R3d bindable preflight input hashes differ from literal pins: "
            + ", ".join(drift)
        )
    if fingerprint != frozen_fingerprint_sha256:
        raise TargetSealError(
            "R3d bindable preflight instrument fingerprint differs from its literal pin"
        )
    return True


def normalized_runner_sha256(path: str | Path = RUNNER) -> str:
    text = Path(path).read_text()
    normalized = re.sub(
        r"^(FROZEN_[A-Z0-9_]+_SHA256):[^=]*=.*$",
        r'\1: str | None = "<SEAL_LITERAL>"',
        text,
        flags=re.MULTILINE,
    ).encode()
    return hashlib.sha256(normalized).hexdigest()


def instrument_binding_payload(
    *,
    hashes: Mapping[str, Any],
    batch: int,
    device: str,
    seed: int,
    n_randomisation: int,
    n_trials: int,
) -> dict[str, Any]:
    return {
        "schema": "gfmcd.r3d.instrument_binding/1",
        "result_schema": RESULT_SCHEMA,
        "instrument_revision": INSTRUMENT_REVISION,
        "hashes": dict(hashes),
        "component_order": [component_name(component) for component in COMPONENTS],
        "primitive_row_names": primitive_row_names(),
        "family_row_names": family_row_names(),
        "path_specs": named_path_specs(),
        "fixed_sequence": list(PATH_NAMES),
        "path_alpha": PATH_ALPHA,
        "sesoi": SESOI,
        "decision_rule": "strict p < alpha; confidence bound inverts that same test",
        "boundary_face_suite": boundary_face_suite_binding(),
        "calibration": {
            "boundary_face_trials": PRODUCTION_TRIALS,
            "maximum_false_positive_trials_per_face": MAX_FACE_FALSE_POSITIVES,
            "cp_upper_is_descriptive_only": True,
            "plant_trials": PRODUCTION_PLANT_TRIALS,
            "minimum_plant_successes": MIN_PLANT_SUCCESSES,
            "operating_characteristics": {
                "face_suite_pass_probability_lower_bound_at_point05": (
                    FACE_SUITE_PASS_LOWER_AT_POINT05
                ),
                "plant_pass_probability_at_true_joint_point95": (
                    PLANT_PASS_PROBABILITY_AT_POINT95
                ),
            },
        },
        "cohort": {
            "split": "train",
            "preconstruction_allowed_chromosomes": list(TRAIN_CHROMOSOMES),
            "forbidden_chromosomes": sorted(FORBIDDEN_CHROMOSOMES),
            "one_hash_ranked_row_per_gene": True,
            "preflight_genes": PREFLIGHT_N_GENES,
            "target": "all eligible genes absent from the direct-exposure registry",
            "target_minimum_genes": TARGET_MIN_GENES,
            "registry_schema": REGISTRY_SCHEMA,
            "registry_scope_id": REGISTRY_SCOPE,
        },
        "measurement": {
            "readout": "mask exactly the G at next_acceptor_pos - 1",
            "operator": "paired and pseudo donor GT -> AC",
            "donor_readout_constructed": False,
            "r3c_runner_bound": True,
        },
        "health": (
            "R3c v2.1 preflight AND R3d preflight AND target; monotone suppression"
        ),
        "production": {
            "batch": batch,
            "device": device,
            "seed": seed,
            "n_randomisation": n_randomisation,
            "n_trials": n_trials,
        },
        "runtime": r3c.runtime_identity(device),
    }


def _snapshot(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    return r3c.snapshot_input_paths(paths)


def fasta_index_path(fasta: str | Path) -> Path:
    """Return the exact sidecar path PyfaidxGenome consumes for a plain FASTA."""
    source = Path(fasta).expanduser().resolve()
    return Path(f"{source}.fai")


def _verify_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    r3c.verify_input_snapshot(snapshot)


def _atomic_write(path: str | Path, data: bytes) -> Path:
    return r3c.atomic_write_new(path, data)


def target_attempt_ledger(repo: str | Path = REPO) -> Path:
    """Keep spend receipts in the shared git dir, outside every clean worktree."""
    root = Path(repo).expanduser().resolve()
    completed = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    raw = completed.stdout.strip()
    if completed.returncode or not raw:
        raise RuntimeError("R3d target attempt ledger requires a git common directory")
    common = Path(raw).expanduser()
    if not common.is_absolute():
        common = (root / common).resolve()
    else:
        common = common.resolve()
    return common / TARGET_ATTEMPT_LEDGER_RELATIVE_TO_GIT_COMMON_DIR


def target_identity_from_registry(
    registry: Mapping[str, Any], hashes: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze the biological target independently of code revision/output choice."""
    capacity = registry.get("train_capacity")
    target = capacity.get("fresh_target") if isinstance(capacity, Mapping) else None
    if (
        not isinstance(target, Mapping)
        or not isinstance(target.get("genes_one_row_each"), int)
        or target["genes_one_row_each"] < TARGET_MIN_GENES
        or not _is_sha256(target.get("gene_ids_sha256"))
        or not _is_sha256(target.get("ordered_one_row_per_gene_sha256"))
    ):
        raise ValueError("registry omits a valid >=2500-gene target identity")
    source_keys = (
        "gtf_sha256",
        "fasta_sha256",
        "fasta_index_sha256",
        "checkpoint_aggregate_sha256",
    )
    source_identity = {key: hashes.get(key) for key in source_keys}
    if any(not _is_sha256(value) for value in source_identity.values()):
        raise ValueError("target identity source hashes are incomplete")
    return {
        "schema": "gfmcd.r3d.biological_target_identity/1",
        "experiment": "R3d_acceptor_specificity_adjudication",
        "target_version": "r3d-v1",
        "checkpoint_and_data": source_identity,
        "split": "train",
        "masked_endpoint": "acceptor_G_at_next_acceptor_pos_minus_1",
        "paired_and_pseudo_operator": "GT_to_AC",
        "registry_scope_id": registry.get("scope_id"),
        "registry_gene_ids_sha256": registry.get("gene_ids_sha256"),
        "preconstruction_allowed_chromosomes": list(TRAIN_CHROMOSOMES),
        "forbidden_chromosomes": sorted(FORBIDDEN_CHROMOSOMES),
        "one_row_per_gene_policy": {
            "row_rank_salt": ROW_RANK_SALT,
            "gene_rank_salt": GENE_RANK_SALT,
        },
        "target": {
            "selection": "every eligible gene absent from the two-class direct-exposure registry",
            "genes_one_row_each": target["genes_one_row_each"],
            "gene_ids_sha256": target["gene_ids_sha256"],
            "ordered_one_row_per_gene_sha256": target[
                "ordered_one_row_per_gene_sha256"
            ],
            "target_minimum_genes": TARGET_MIN_GENES,
        },
    }


def canonical_target_attempt_path(
    target_identity: Mapping[str, Any], *, repo: str | Path = REPO
) -> Path:
    if target_identity.get("schema") != "gfmcd.r3d.biological_target_identity/1":
        raise ValueError("R3d target attempt key requires the canonical target identity")
    return target_attempt_ledger(repo) / f"{canonical_sha256(target_identity)}.json"


def verify_loaded_target_identity(
    cohort_audit: Mapping[str, Any], target_identity: Mapping[str, Any]
) -> None:
    target = target_identity["target"]
    selected_ids = cohort_audit.get("selected_gene_ids")
    if (
        not isinstance(selected_ids, list)
        or cohort_audit.get("selected_genes") != target["genes_one_row_each"]
        or cohort_audit.get("selected_rows") != target["genes_one_row_each"]
        or canonical_sha256(sorted(selected_ids)) != target["gene_ids_sha256"]
        or cohort_audit.get("ordered_rows_sha256")
        != target["ordered_one_row_per_gene_sha256"]
        or cohort_audit.get("registry_gene_intersection_count") != 0
        or cohort_audit.get("target_uses_every_eligible_registry_disjoint_gene")
        is not True
    ):
        raise RuntimeError("loaded target cohort differs from the spent target identity")


def write_target_attempt(
    path: str | Path,
    *,
    target_identity: Mapping[str, Any],
    invocation: Sequence[str],
    authorized_git_identity: Mapping[str, Any],
    instrument_fingerprint: str,
    pins: Mapping[str, Any],
    input_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Irreversibly reserve the target before cohort construction/model scoring."""
    attempt_path = Path(path).expanduser().resolve()
    record = {
        "schema": "gfmcd.r3d.target_attempt/1",
        "status": "TARGET_SPENT_BEFORE_COHORT_OR_MODEL_OUTCOME",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_materialized_before_record": False,
        "model_outcome_opened_before_record": False,
        "target_identity": dict(target_identity),
        "target_identity_sha256": canonical_sha256(target_identity),
        "invocation": list(invocation),
        "authorized_git_identity": dict(authorized_git_identity),
        "instrument_fingerprint_sha256": instrument_fingerprint,
        "literal_pins": dict(pins),
        "pre_attempt_input_snapshot": dict(input_snapshot),
        "pre_attempt_input_snapshot_sha256": canonical_sha256(input_snapshot),
    }
    record["attempt_id"] = canonical_sha256(record)
    _atomic_write(
        attempt_path,
        (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    return {
        "path": str(attempt_path),
        "size_bytes": attempt_path.stat().st_size,
        "sha256": sha256_file(attempt_path),
        "attempt_id": record["attempt_id"],
        "status": record["status"],
        "target_identity_sha256": record["target_identity_sha256"],
        "completion_path": str(attempt_path.with_suffix(".completion.json")),
        "receipt": record,
    }


def write_target_attempt_completion(
    target_attempt: Mapping[str, Any],
    *,
    status: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    """Close a spent attempt create-only; the start receipt remains authoritative."""
    if status not in {"TARGET_PUBLICATION_COMPLETE", "TARGET_EXECUTION_FAILED"}:
        raise ValueError("unknown target-attempt completion status")
    path = Path(str(target_attempt["completion_path"])).expanduser().resolve()
    record = {
        "schema": "gfmcd.r3d.target_attempt_completion/1",
        "status": status,
        "closed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt_id": target_attempt["attempt_id"],
        "attempt_sha256": target_attempt["sha256"],
        "details": dict(details),
    }
    _atomic_write(
        path,
        (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "status": status,
    }


def install_target_attempt_failure_hook(target_attempt: Mapping[str, Any]) -> None:
    """Best-effort error closure; the immutable start receipt handles hard crashes."""
    prior = sys.excepthook

    def close_and_report(exc_type: type[BaseException], exc: BaseException, traceback: Any) -> None:
        completion = Path(str(target_attempt["completion_path"]))
        if not completion.exists():
            try:
                write_target_attempt_completion(
                    target_attempt,
                    status="TARGET_EXECUTION_FAILED",
                    details={
                        "exception_type": exc_type.__name__,
                        "message": str(exc),
                    },
                )
            except Exception:
                pass
        prior(exc_type, exc, traceback)

    sys.excepthook = close_and_report


def _load_json_bound(path: str | Path, expected_sha256: str | None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    digest = sha256_file(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"bound JSON input differs from its literal pin: {source}")
    return json.loads(source.read_text())


def _load_artifact_package(
    path: str | Path,
    expected_sha256: str | None,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load an immutable result and require its sibling manifest to bind its bytes."""
    source = Path(path).expanduser().resolve()
    manifest_path = source.with_suffix(".manifest.json")
    record = _load_json_bound(source, expected_sha256)
    manifest = json.loads(manifest_path.read_text())
    manifest_digest = sha256_file(manifest_path)
    if (
        expected_manifest_sha256 is not None
        and manifest_digest != expected_manifest_sha256
    ):
        raise ValueError("artifact sibling manifest differs from its literal pin")
    digest = sha256_file(source)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("artifact sibling manifest outputs must be a list")

    def resolves_to_source(value: Any) -> bool:
        candidate = Path(str(value)).expanduser()
        if candidate.is_absolute():
            if candidate.resolve() == source:
                return True
            candidate_parts = candidate.parts
        else:
            resolved = (REPO / candidate).resolve()
            if resolved == source:
                return True
            candidate_parts = candidate.parts
        try:
            logical_source = source.relative_to(REPO)
        except ValueError:
            return False
        if "results" not in candidate_parts:
            return False
        logical_candidate = Path(*candidate_parts[candidate_parts.index("results") :])
        return logical_candidate == logical_source

    matches = [
        output
        for output in outputs
        if isinstance(output, Mapping) and resolves_to_source(output.get("path", ""))
    ]
    if (
        len(matches) != 1
        or matches[0].get("sha256") != digest
        or matches[0].get("size_bytes") != source.stat().st_size
    ):
        raise ValueError("artifact bytes do not match exactly one sibling-manifest output")
    if (
        not isinstance(manifest.get("command"), list)
        or not manifest["command"]
        or manifest.get("status") != record.get("status")
        or (
            "command" in record
            and manifest.get("command") != record.get("command")
        )
    ):
        raise ValueError("artifact sibling manifest command/status semantics differ")
    if (
        record.get("target_bindable_preflight") is True
        or record.get("stage") == "target"
    ):
        identity = record.get("git_authorization_identity")
        snapshot = record.get("input_snapshot")
        manifest_inputs = manifest.get("inputs")
        rendered_inputs = {
            str(Path(str(item.get("path", ""))).expanduser().resolve()): {
                "size_bytes": item.get("size_bytes"),
                "sha256": item.get("sha256"),
            }
            for item in manifest_inputs
            if isinstance(item, Mapping)
        } if isinstance(manifest_inputs, list) else None
        if (
            not isinstance(identity, Mapping)
            or manifest.get("git_dirty") is not False
            or manifest.get("git_commit") != identity.get("commit")
            or not isinstance(snapshot, Mapping)
            or rendered_inputs != dict(snapshot)
            or manifest.get("config", {}).get("input_snapshot_sha256")
            != canonical_sha256(snapshot)
            or manifest.get("config", {}).get("instrument_fingerprint_sha256")
            != record.get("instrument_fingerprint_sha256")
            or manifest.get("config", {}).get("stage") != record.get("stage")
        ):
            raise ValueError(
                "target-bindable artifact manifest does not replay clean git/input/instrument semantics"
            )
    return record, {
        "artifact": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": digest,
        },
        "manifest": {
            "path": str(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": manifest_digest,
        },
    }


def git_blob_sha256(
    revision: str, path: str, *, repo: str | Path = REPO
) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None or not path:
        raise ValueError("committed blob identity is malformed")
    completed = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=Path(repo).expanduser().resolve(),
        capture_output=True,
        check=False,
    )
    if completed.returncode:
        raise ValueError("committed builder blob is unavailable")
    return hashlib.sha256(completed.stdout).hexdigest()


def _load_registry_package(
    path: str | Path,
    expected_sha256: str | None,
    expected_manifest_sha256: str | None,
    *,
    require_clean_generator: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the registry's dedicated model-free manifest contract."""
    source = Path(path).expanduser().resolve()
    manifest_path = source.with_suffix(".manifest.json")
    record = _load_json_bound(source, expected_sha256)
    manifest = json.loads(manifest_path.read_text())
    manifest_digest = sha256_file(manifest_path)
    if (
        expected_manifest_sha256 is not None
        and manifest_digest != expected_manifest_sha256
    ):
        raise ValueError("registry sibling manifest differs from its literal pin")
    outputs = manifest.get("outputs")
    matches = []
    if isinstance(outputs, list):
        for output in outputs:
            if not isinstance(output, Mapping):
                continue
            raw = Path(str(output.get("path", ""))).expanduser()
            resolved = raw.resolve() if raw.is_absolute() else (REPO / raw).resolve()
            if resolved == source:
                matches.append(output)
    code = manifest.get("code")
    parameters = manifest.get("parameters")
    inputs = manifest.get("inputs")
    generator = record.get("generator")
    input_bindings = record.get("input_bindings")
    def binding_pairs(value: Any) -> set[tuple[Any, Any]]:
        pairs: set[tuple[Any, Any]] = set()
        if isinstance(value, Mapping):
            if "size_bytes" in value and "sha256" in value:
                pairs.add((value.get("size_bytes"), value.get("sha256")))
            for child in value.values():
                pairs.update(binding_pairs(child))
        elif isinstance(value, list):
            for child in value:
                pairs.update(binding_pairs(child))
        return pairs

    input_pairs = binding_pairs(inputs)
    bound_pairs = binding_pairs(
        {
            "input_bindings": input_bindings,
            "registered_source_bindings": record.get("sources"),
        }
    )
    head_source = record.get("sources", {}).get(
        "B_head_screen_clean_alt_candidate_selection", {}
    )
    claude_source = record.get("sources", {}).get(
        "A_claude_inverse_acceptor", {}
    )
    endpoint_strata = (
        head_source.get("endpoint_strata", {})
        if isinstance(head_source, Mapping)
        else {}
    )
    head_manifest = inputs.get("head_screen_shards", {}) if isinstance(inputs, Mapping) else {}
    shard_files = head_manifest.get("files") if isinstance(head_manifest, Mapping) else None
    bound_inputs = input_bindings if isinstance(input_bindings, Mapping) else {}
    exact_input_mappings = (
        isinstance(inputs, Mapping)
        and isinstance(claude_source, Mapping)
        and isinstance(head_source, Mapping)
        and inputs.get("annotation") == bound_inputs.get("annotation")
        and inputs.get("genome") == bound_inputs.get("genome")
        and inputs.get("genome_index") == bound_inputs.get("genome_index")
        and inputs.get("claude_artifacts_and_manifests")
        == claude_source.get("artifact_bindings")
        and inputs.get("claude_reconstruction_git_blobs")
        == claude_source.get("reconstruction_git_blobs")
        and inputs.get("claude_result_manifest_semantic_bindings")
        == claude_source.get("result_manifest_semantic_bindings")
        and inputs.get("claude_transcript")
        == claude_source.get("transcript_binding")
        and inputs.get("head_screen_comparison_artifact")
        == head_source.get("comparison_artifact")
        and inputs.get("head_screen_comparison_manifest")
        == head_source.get("comparison_manifest")
    )
    exact_shard_inventory = (
        isinstance(shard_files, list)
        and len(shard_files) == 400
        and all(
            isinstance(item, Mapping)
            and set(item) == {"name", "sha256", "size_bytes"}
            and isinstance(item.get("name"), str)
            and item["name"]
            and _is_sha256(item.get("sha256"))
            and isinstance(item.get("size_bytes"), int)
            and item["size_bytes"] > 0
            for item in shard_files
        )
        and [item["name"] for item in shard_files]
        == sorted({item["name"] for item in shard_files})
        and head_manifest.get("files_count") == 400
        and head_manifest.get("inventory_sha256") == canonical_sha256(shard_files)
        and head_manifest.get("inventory_sha256")
        == head_source.get("shard_inventory_sha256")
    )
    committed_builder_ok = (
        isinstance(code, Mapping) and code.get("git_dirty") is not False
    )
    if isinstance(code, Mapping) and code.get("git_dirty") is False:
        try:
            committed_builder_ok = (
                git_blob_sha256(
                    str(code.get("git_revision")), str(code.get("builder"))
                )
                == code.get("builder_sha256")
            )
        except ValueError:
            committed_builder_ok = False
    if (
        record.get("schema") != REGISTRY_SCHEMA
        or record.get("scope_id") != REGISTRY_SCOPE
        or record.get("gene_ids_sha256")
        != canonical_sha256(record.get("gene_ids", []))
        or manifest.get("schema")
        != "gfmcd.r3d.exposed_gene_registry_manifest/1"
        or manifest.get("experiment") != "r3d_train_exposure_registry_v1"
        or manifest.get("status") != record.get("status")
        or not isinstance(manifest.get("command"), list)
        or not manifest["command"]
        or not isinstance(code, Mapping)
        or not isinstance(generator, Mapping)
        or code.get("builder") != REGISTRY_BUILDER_RELATIVE
        or generator.get("script") != REGISTRY_BUILDER_RELATIVE
        or code.get("builder") != generator.get("script")
        or code.get("builder_sha256") != generator.get("script_sha256")
        or re.fullmatch(r"[0-9a-f]{40}", str(code.get("git_revision"))) is None
        or (require_clean_generator and code.get("git_dirty") is not False)
        or committed_builder_ok is not True
        or not isinstance(parameters, Mapping)
        or parameters.get("scope_id") != REGISTRY_SCOPE
        or parameters.get("source_b_shards") != 400
        or parameters.get("source_b_candidate_selection_uses_both_endpoint_strata")
        is not True
        or not isinstance(head_source, Mapping)
        or head_source.get("shards") != 400
        or head_source.get("rows") != 400
        or head_source.get("exact_masked_acceptor_endpoint_is_stratum_only") is not True
        or endpoint_strata.get("masked_acceptor", {}).get("rows") != 222
        or endpoint_strata.get("masked_donor", {}).get("rows") != 178
        or exact_input_mappings is not True
        or exact_shard_inventory is not True
        or generator.get("model_loaded_or_run") is not False
        or len(matches) != 1
        or matches[0].get("sha256") != sha256_file(source)
        or matches[0].get("size_bytes") != source.stat().st_size
        or not bound_pairs
        or not bound_pairs.issubset(input_pairs)
    ):
        raise ValueError("registry sibling manifest semantic binding does not replay")
    return record, {
        "artifact": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        },
        "manifest": {
            "path": str(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "sha256": manifest_digest,
        },
    }


def _require_r3d_preflight(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"R3d preflight semantic validation failed: {message}")


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(float(value))
    )


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_gene_statistics(
    stats: Any,
    *,
    names: Sequence[str],
    n_examples: int,
    expected_gene_ids: Sequence[str] | None = None,
    one_example_per_gene: bool = False,
) -> tuple[np.ndarray, list[str]]:
    _require_r3d_preflight(isinstance(stats, Mapping), "gene sufficient statistics are missing")
    _require_r3d_preflight(
        stats.get("schema") == "gene-cluster-raw-sums-v1"
        and stats.get("row_names") == list(names),
        "gene sufficient-statistic schema or row order differs",
    )
    genes = stats.get("genes")
    _require_r3d_preflight(isinstance(genes, list) and genes, "gene records are missing")
    gene_ids = [entry.get("gene_id") for entry in genes if isinstance(entry, Mapping)]
    _require_r3d_preflight(
        len(gene_ids) == len(genes)
        and all(isinstance(gene, str) and gene for gene in gene_ids)
        and gene_ids == sorted(set(gene_ids)),
        "gene records are not sorted unique identities",
    )
    if expected_gene_ids is not None:
        _require_r3d_preflight(
            gene_ids == sorted(map(str, expected_gene_ids)),
            "gene statistics disagree with the selected cohort",
        )
    total = 0
    for entry in genes:
        count = entry.get("n_examples")
        sums = entry.get("raw_component_sums")
        _require_r3d_preflight(
            isinstance(count, int)
            and not isinstance(count, bool)
            and count > 0
            and (not one_example_per_gene or count == 1)
            and isinstance(sums, list)
            and len(sums) == len(names)
            and all(_is_finite_number(value) for value in sums),
            "gene sufficient-statistic row is malformed",
        )
        total += count
    _require_r3d_preflight(total == n_examples, "gene counts do not sum to n")
    values, groups = r3c._reconstruct_cluster_array(stats)
    _require_r3d_preflight(
        values.shape == (len(names), n_examples) and len(groups) == n_examples,
        "gene sufficient-statistic reconstruction has the wrong dimensions",
    )
    return values, groups


def _reorder_one_row_per_gene_statistics(
    values: np.ndarray,
    groups: Sequence[str],
    ordered_gene_ids: Sequence[str],
    *,
    context: str,
) -> tuple[np.ndarray, list[str]]:
    """Restore the outcome-blind cohort order after sorted compact storage."""
    ordered = list(map(str, ordered_gene_ids))
    group_list = list(map(str, groups))
    _require_r3d_preflight(
        len(group_list) == len(set(group_list))
        and len(ordered) == len(set(ordered))
        and set(group_list) == set(ordered),
        f"{context} gene identities cannot be restored to cohort order",
    )
    column_by_gene = {gene: index for index, gene in enumerate(group_list)}
    indices = [column_by_gene[gene] for gene in ordered]
    return np.asarray(values)[:, indices], ordered


def _validate_r3d_calibration(
    calibration: Any,
    *,
    primitive_values: np.ndarray,
    primitive_groups: Sequence[str],
) -> None:
    _require_r3d_preflight(isinstance(calibration, Mapping), "production calibration is missing")
    namespace = "r3d:production-boundary-face-preflight"
    suite = boundary_face_surface_specs()
    _require_r3d_preflight(
        calibration.get("schema") == "gfmcd.r3d.boundary_face_calibration/1"
        and calibration.get("randomisation_unit") == "whole gene"
        and calibration.get("translation_invariant_centering") is True
        and calibration.get("n_primitive_rows") == 14
        and calibration.get("n_family_rows") == 23
        and calibration.get("n_boundary_faces") == 36
        and calibration.get("trials_per_surface") == PRODUCTION_TRIALS
        and calibration.get("plant_trials") == PRODUCTION_PLANT_TRIALS
        and calibration.get("n_calibration_draws") == PRODUCTION_RANDOMISATION
        and calibration.get("seed") == PRODUCTION_SEED
        and calibration.get("namespace") == namespace
        and calibration.get("boundary_face_suite") == boundary_face_suite_binding()
        and calibration.get("engineering_not_inferential") is True,
        "production calibration dimensions, namespace, or suite binding differ",
    )
    faces = calibration.get("boundary_faces")
    _require_r3d_preflight(
        isinstance(faces, Mapping) and list(faces) == list(suite),
        "boundary-face record order or membership differs",
    )
    all_engineering_ok = True
    for face_name, spec in suite.items():
        face = faces[face_name]
        _require_r3d_preflight(
            isinstance(face, Mapping)
            and face.get("path") == spec["path"]
            and face.get("target_row") == spec["target_row"]
            and face.get("edge") == spec["edge"]
            and face.get("trials") == PRODUCTION_TRIALS
            and face.get("outer_namespace_template")
            == f"{namespace}:face:{face_name}:outer:<trial>"
            and face.get("classifier_namespace_template")
            == f"{namespace}:face:{face_name}:classifier:<trial>",
            f"boundary-face metadata differs for {face_name}",
        )
        pattern = face.get("support_pattern")
        _require_r3d_preflight(
            isinstance(pattern, list)
            and len(pattern) == PRODUCTION_TRIALS
            and all(value in (0, 1) and not isinstance(value, bool) for value in pattern)
            and _is_sha256(face.get("outer_null_digest_sha256"))
            and _is_sha256(face.get("classifier_audit_digest_sha256")),
            f"boundary-face compact trial summaries are incomplete for {face_name}",
        )
        count = sum(pattern)
        _, upper = one_sided_binomial_bounds(
            count, PRODUCTION_TRIALS, alpha=PATH_ALPHA
        )
        _require_r3d_preflight(
            face.get("support_pattern") == pattern
            and face.get("pattern_sha256") == canonical_sha256(pattern)
            and face.get("false_positive_trials") == count
            and np.isclose(
                face.get("false_positive_rate", float("nan")),
                count / PRODUCTION_TRIALS,
                rtol=0.0,
                atol=1e-15,
            )
            and np.isclose(
                face.get("one_sided_95pct_clopper_pearson_upper", float("nan")),
                upper,
                rtol=0.0,
                atol=1e-15,
            )
            and face.get("one_sided_cp_upper_is_descriptive_only") is True
            and face.get("maximum_false_positive_trials")
            == MAX_FACE_FALSE_POSITIVES
            and face.get("engineering_fpr_ok")
            is face_max_count_pass(count, PRODUCTION_TRIALS),
            f"boundary-face decisions, digests, or CP gate differ for {face_name}",
        )
        all_engineering_ok &= face_max_count_pass(count, PRODUCTION_TRIALS)

    plant_patterns = calibration.get("plant_support_patterns")
    _require_r3d_preflight(
        isinstance(plant_patterns, Mapping)
        and tuple(plant_patterns) == PATH_NAMES
        and all(
            isinstance(plant_patterns[path], list)
            and len(plant_patterns[path]) == PRODUCTION_PLANT_TRIALS
            and all(
                value in (0, 1) and not isinstance(value, bool)
                for value in plant_patterns[path]
            )
            for path in PATH_NAMES
        )
        and calibration.get("plant_outer_namespace_template")
        == f"{namespace}:plant:outer:<trial>"
        and calibration.get("plant_classifier_namespace_template")
        == f"{namespace}:plant:classifier:<trial>"
        and _is_sha256(calibration.get("plant_classifier_audit_digest_sha256"))
        and _is_sha256(calibration.get("plant_outer_null_digest_sha256")),
        "coherent-plant compact patterns or namespaces are incomplete",
    )
    successes = {path: sum(plant_patterns[path]) for path in PATH_NAMES}
    plants = calibration.get("plant_successes")
    _require_r3d_preflight(
        isinstance(plants, Mapping) and tuple(plants) == PATH_NAMES,
        "coherent-plant success family differs",
    )
    for path, count in successes.items():
        expected = {
            "successes": count,
            "trials": PRODUCTION_PLANT_TRIALS,
            "power": count / PRODUCTION_PLANT_TRIALS,
            "minimum_successes": MIN_PLANT_SUCCESSES,
            "powered": plant_count_pass(count, PRODUCTION_PLANT_TRIALS),
        }
        _require_r3d_preflight(
            plants[path] == expected,
            f"coherent-plant arithmetic differs for {path}",
        )
    plant = coherent_path_offset()
    all_plants = all(
        plant_count_pass(count, PRODUCTION_PLANT_TRIALS)
        for count in successes.values()
    )
    _require_r3d_preflight(
        calibration.get("all_boundary_face_engineering_fpr_ok") is all_engineering_ok
        and calibration.get("face_familywise_max_count_operating_characteristics")
        == {
            "point_null_per_face_probability": PATH_ALPHA,
            "maximum_false_positive_trials": MAX_FACE_FALSE_POSITIVES,
            "single_face_fail_probability": FACE_FAIL_PROBABILITY_AT_POINT05,
            "union_bound_any_of_36_faces_fail": FACE_SUITE_UNION_BOUND_AT_POINT05,
            "suite_pass_probability_lower_bound": FACE_SUITE_PASS_LOWER_AT_POINT05,
            "cp_upper_at_maximum_count_descriptive_only": FACE_CP_UPPER_AT_MAX_COUNT,
            "does_not_assume_cross_face_independence": True,
        }
        and calibration.get("plant_delta") == PLANT_DELTA
        and calibration.get("plant_operating_characteristics")
        == {
            "trials": PRODUCTION_PLANT_TRIALS,
            "minimum_successes": MIN_PLANT_SUCCESSES,
            "one_sided_95pct_cp_lower_at_threshold": PLANT_CP_LOWER_AT_THRESHOLD,
            "pass_probability_at_true_joint_success_point95": (
                PLANT_PASS_PROBABILITY_AT_POINT95
            ),
            "path_b_successes_must_not_exceed_path_a_successes": True,
        }
        and successes[PATH_B] <= successes[PATH_A]
        and calibration.get("coherent_primitive_plant_sha256") == _array_sha256(plant)
        and calibration.get("plant_support_pattern_sha256")
        == {
            path: canonical_sha256(plant_patterns[path]) for path in PATH_NAMES
        }
        and calibration.get("all_plants_at_least_462_of_500") is all_plants
        and calibration.get("all_plants_meet_stage_threshold") is all_plants,
        "aggregate calibration decisions or digests do not replay",
    )
    replay = calibration_preflight(
        primitive_values,
        primitive_groups,
        n_draws=PRODUCTION_RANDOMISATION,
        n_trials=PRODUCTION_TRIALS,
        n_plant_trials=PRODUCTION_PLANT_TRIALS,
        seed=PRODUCTION_SEED,
        namespace=namespace,
    )
    _require_r3d_preflight(
        calibration == replay,
        "production calibration does not replay through the exact classifier from primitive statistics",
    )


def validate_r3d_preflight(
    record: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    registry_package: Mapping[str, Any],
    fingerprint: str,
    binding_payload: Mapping[str, Any],
    r3c_health_ok: Mapping[str, bool],
    r3c_preflight_package: Mapping[str, Any],
) -> None:
    """Independently replay every decision-bearing compact preflight field."""
    expected_top = {
        "schema": RESULT_SCHEMA,
        "experiment": "R3d_acceptor_specificity_adjudication",
        "instrument_revision": INSTRUMENT_REVISION,
        "stage": "preflight",
        "status": "DEVELOPMENT",
        "tier": "PROSPECTIVE_ENDPOINT_SPECIFIC_TRAIN_VERIFICATION",
        "not_sealed_or_independent_confirmation": True,
        "target_bindable_preflight": True,
        "split": "train",
        "n": PREFLIGHT_N_GENES,
        "n_genes": PREFLIGHT_N_GENES,
        "all_gates_pass": True,
        "official_scientific_disposition_gates_pass": True,
        "components_are_diagnostic_when_global_gate_fails": False,
        "verdict": "TARGET_BINDABLE_PREFLIGHT_GO",
        "licenses": "no positive scientific disposition; development mechanics only",
    }
    for key, expected in expected_top.items():
        _require_r3d_preflight(record.get(key) == expected, f"top-level {key} differs")
    command = record.get("command")
    _require_r3d_preflight(
        isinstance(command, list)
        and command
        and "--stage" in command
        and command.index("--stage") + 1 < len(command)
        and command[command.index("--stage") + 1] == "preflight",
        "preflight exact argv is missing or has the wrong stage",
    )
    _require_r3d_preflight(
        record.get("instrument_fingerprint_sha256") == fingerprint
        and canonical_sha256(binding_payload) == fingerprint
        and record.get("instrument_binding_payload") == binding_payload,
        "instrument fingerprint or binding payload differs",
    )
    hashes = record.get("hashes")
    _require_r3d_preflight(
        isinstance(hashes, Mapping)
        and all(
            hashes.get(key) == value
            for key, value in binding_payload.get("hashes", {}).items()
        ),
        "preflight hash bundle disagrees with the current binding",
    )
    _require_r3d_preflight(
        record.get("input_snapshot_sha256")
        == canonical_sha256(record.get("input_snapshot")),
        "input snapshot digest differs",
    )
    snapshot = record.get("input_snapshot")
    recorded_registry_package = record.get("registry", {}).get("package")
    recorded_r3c_package = record.get("r3c_v2_1_preflight_package")
    _require_r3d_preflight(
        isinstance(snapshot, Mapping)
        and _packages_semantically_equal(
            recorded_registry_package, registry_package
        )
        and _packages_semantically_equal(
            recorded_r3c_package, r3c_preflight_package
        )
        and _snapshot_contains_package(snapshot, recorded_registry_package)
        and _snapshot_contains_package(snapshot, recorded_r3c_package)
        and hashes.get("r3c_v2_1_preflight_sha256")
        == r3c_preflight_package.get("artifact", {}).get("sha256")
        and hashes.get("r3c_v2_1_preflight_manifest_sha256")
        == r3c_preflight_package.get("manifest", {}).get("sha256"),
        "preflight snapshot does not authenticate its predecessor packages",
    )
    git_identity = record.get("git_authorization_identity")
    _require_r3d_preflight(
        isinstance(git_identity, Mapping)
        and re.fullmatch(r"[0-9a-f]{40}", str(git_identity.get("commit"))) is not None
        and git_identity.get("dirty") is False,
        "bindable preflight lacks its clean git authorization identity",
    )

    registry_summary = record.get("registry")
    expected_registry_summary = {
        "schema": registry["schema"],
        "scope_id": registry["scope_id"],
        "scope": registry["scope"],
        "global_gene_freshness_claimed": False,
        "gene_count": len(registry["gene_ids"]),
        "gene_ids_sha256": registry["gene_ids_sha256"],
        "file_sha256": registry_package["artifact"]["sha256"],
    }
    _require_r3d_preflight(
        isinstance(registry_summary, Mapping)
        and set(registry_summary) == {*expected_registry_summary, "package"}
        and all(
            registry_summary.get(key) == value
            for key, value in expected_registry_summary.items()
        )
        and _packages_semantically_equal(
            registry_summary.get("package"), registry_package
        ),
        "registry summary/package differs from the current immutable registry",
    )
    cohort = record.get("cohort_audit")
    selected_ids = cohort.get("selected_gene_ids") if isinstance(cohort, Mapping) else None
    selected_rows = cohort.get("selected_row_records") if isinstance(cohort, Mapping) else None
    capacity = registry.get("train_capacity")
    _require_r3d_preflight(
        isinstance(cohort, Mapping)
        and cohort.get("schema") == "gfmcd.r3d.train_cohort_audit/1"
        and cohort.get("selection_reads_model_outputs") is False
        and cohort.get("split") == "train"
        and cohort.get("preconstruction_allowed_chromosomes") == list(TRAIN_CHROMOSOMES)
        and cohort.get("forbidden_chromosomes") == sorted(FORBIDDEN_CHROMOSOMES)
        and cohort.get("provenance_hash_reads_full_fasta_bytes") is True
        and cohort.get("forbidden_chromosomes_materialized_as_examples") is False
        and cohort.get("forbidden_chromosomes_model_scored") is False
        and cohort.get("sealed_or_independent_claim") is False
        and cohort.get("selected_rows") == PREFLIGHT_N_GENES
        and cohort.get("selected_genes") == PREFLIGHT_N_GENES
        and cohort.get("one_row_per_gene") is True
        and cohort.get("registry_gene_intersection_count") == PREFLIGHT_N_GENES
        and cohort.get("target_uses_every_eligible_registry_disjoint_gene") is False
        and cohort.get("target_minimum_genes") == TARGET_MIN_GENES
        and cohort.get("row_rank_salt") == ROW_RANK_SALT
        and cohort.get("gene_rank_salt") == GENE_RANK_SALT
        and cohort.get("frozen_registry_capacity_replayed") is True
        and isinstance(capacity, Mapping)
        and cohort.get("frozen_registry_capacity_sha256") == canonical_sha256(capacity)
        and cohort.get("eligible_genes_one_row_each") == capacity.get("eligible_genes_one_row_each")
        and cohort.get("eligible_exposed_genes") == capacity.get("exposed_development", {}).get("genes_one_row_each")
        and cohort.get("eligible_registry_disjoint_genes") == capacity.get("fresh_target", {}).get("genes_one_row_each")
        and isinstance(selected_ids, list)
        and len(selected_ids) == PREFLIGHT_N_GENES
        and len(set(selected_ids)) == PREFLIGHT_N_GENES
        and set(selected_ids).issubset(set(registry["gene_ids"]))
        and selected_ids
        == sorted(
            selected_ids,
            key=lambda gene: (_hash_rank(GENE_RANK_SALT, gene), gene),
        )
        and isinstance(selected_rows, list)
        and len(selected_rows) == PREFLIGHT_N_GENES
        and [row.get("gene_id") for row in selected_rows] == selected_ids
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("example_id"), str)
            and isinstance(row.get("decoy_donor"), int)
            and row.get("chrom") in TRAIN_CHROMOSOMES
            for row in selected_rows
        )
        and cohort.get("ordered_rows_sha256") == canonical_sha256(selected_rows)
        and cohort.get("ordered_gene_ids_sha256") == canonical_sha256(selected_ids),
        "preflight cohort/registry allocation does not replay",
    )

    family_record = record.get("family")
    _require_r3d_preflight(
        isinstance(family_record, Mapping)
        and family_record.get("definition")
        == "14 primitive partner/pseudo rows derive exactly 23 decision rows"
        and family_record.get("primitive_row_names") == primitive_row_names()
        and family_record.get("primitive_row_order_sha256")
        == canonical_sha256(primitive_row_names())
        and family_record.get("row_names") == family_row_names()
        and family_record.get("row_order_sha256") == canonical_sha256(family_row_names())
        and family_record.get("n_primitive_rows") == 14
        and family_record.get("n_family_rows") == 23
        and family_record.get("production_preflight_origin") == "RUN_ON_CURRENT_BUCKET",
        "14-to-23 family contract differs",
    )
    primitive_values, primitive_groups = _validate_gene_statistics(
        family_record.get("primitive_gene_sufficient_statistics"),
        names=primitive_row_names(),
        n_examples=PREFLIGHT_N_GENES,
        expected_gene_ids=selected_ids,
        one_example_per_gene=True,
    )
    family_values, family_groups = _validate_gene_statistics(
        family_record.get("gene_sufficient_statistics"),
        names=family_row_names(),
        n_examples=PREFLIGHT_N_GENES,
        expected_gene_ids=selected_ids,
        one_example_per_gene=True,
    )
    primitive_values, primitive_groups = _reorder_one_row_per_gene_statistics(
        primitive_values,
        primitive_groups,
        selected_ids,
        context="primitive preflight statistics",
    )
    family_values, family_groups = _reorder_one_row_per_gene_statistics(
        family_values,
        family_groups,
        selected_ids,
        context="derived preflight statistics",
    )
    derived, derived_names = derive_family_from_primitives(primitive_values)
    _require_r3d_preflight(
        primitive_groups == family_groups
        and derived_names == family_row_names()
        and np.array_equal(derived, family_values),
        "23-row family does not derive exactly from the 14 primitive rows",
    )

    health = record.get("three_way_monotone_health")
    components = record.get("components")
    _require_r3d_preflight(
        isinstance(health, Mapping)
        and tuple(health) == PATH_NAMES
        and isinstance(components, Mapping),
        "three-way health registry is missing",
    )
    for path, participants in _path_health_participants().items():
        item = health[path]
        participant_names = [f"{mode}|{head}" for mode, head in participants]
        current_failures: list[str] = []
        for mode, head in participants:
            component_health = (
                components.get(head, {}).get("modes", {}).get(mode, {}).get("health", {})
            )
            _require_r3d_preflight(
                isinstance(component_health, Mapping)
                and component_health.get("total", 0) > 0
                and isinstance(component_health.get("ok"), bool),
                f"bounded health was not executed for {mode}|{head}",
            )
            if component_health.get("ok") is not True:
                current_failures.append(f"{mode}|{head}")
        expected_ok = bool(
            r3c_health_ok.get(path) is True and not current_failures
        )
        _require_r3d_preflight(
            isinstance(item, Mapping)
            and item.get("r3c_v2_1_preflight_health_ok") is (r3c_health_ok.get(path) is True)
            and item.get("r3d_preflight_health_ok") is True
            and item.get("current_bucket_health_ok") is (not current_failures)
            and item.get("ok") is expected_ok
            and item.get("can_only_shrink") is True
            and item.get("participants") == participant_names
            and item.get("failed_participants") == current_failures,
            f"three-way monotone health does not replay for {path}",
        )
    _require_r3d_preflight(
        health[PATH_A].get("ok") is True,
        "Path-A preflight health cannot authorize target entry",
    )

    replay = fixed_sequence_inference(
        family_values,
        family_groups,
        family_row_names(),
        n_draws=PRODUCTION_RANDOMISATION,
        seed=PRODUCTION_SEED,
        namespace="r3d:observed-fixed-sequence",
        path_health=health,
    )
    _require_r3d_preflight(
        family_record.get("fixed_sequence_inference") == replay
        and record.get("diagnostic_paths") == replay["paths"]
        and record.get("paths")
        == official_paths_after_global_gates(replay["paths"], record.get("gates", {})),
        "observed fixed-sequence inference does not replay from gene sufficient statistics",
    )
    _require_r3d_preflight(
        record.get("path_family")
        == {
            "path_order": list(PATH_NAMES),
            "studywise_alpha": PATH_ALPHA,
            "strong_fwer_control": "fixed-sequence gatekeeping",
            "path_b_tested_only_after_path_a_supported": (
                replay["paths"][PATH_B]["classification"]
                != "NOT_TESTED_GATE_CLOSED"
            ),
        },
        "preflight fixed-sequence path-family declaration differs",
    )
    replay_rows: dict[str, Any] = {}
    for path_record in replay["paths"].values():
        replay_rows.update(path_record.get("rows", {}))
    expected_component_modes = {
        "L3H1": [MEAN_MODE, NOISING_MODE],
        **{
            component_name(component): [NOISING_MODE]
            for component in ROUTE_COMPONENTS
        },
    }
    _require_r3d_preflight(
        list(components) == list(expected_component_modes),
        "publication-facing component registry/order differs",
    )
    family_lookup = {
        name: family_values[index] for index, name in enumerate(family_row_names())
    }
    for head, expected_modes in expected_component_modes.items():
        component = components[head]
        _require_r3d_preflight(
            component.get("kind") == "head"
            and list(component.get("modes", {})) == expected_modes,
            f"publication-facing modes differ for {head}",
        )
        for mode in expected_modes:
            mode_record = component["modes"][mode]
            endpoints = mode_record.get("endpoints")
            _require_r3d_preflight(
                isinstance(endpoints, Mapping) and tuple(endpoints) == ENDPOINTS,
                f"publication-facing endpoints differ for {mode}|{head}",
            )
            for endpoint in ENDPOINTS:
                row_name = family_row_name(mode, head, endpoint)
                endpoint_record = endpoints[endpoint]
                _require_r3d_preflight(
                    np.isclose(
                        endpoint_record.get("raw_mean", float("nan")),
                        float(np.mean(family_lookup[row_name])),
                        rtol=0.0,
                        atol=1e-15,
                    )
                    and endpoint_record.get("decision") == replay_rows.get(row_name),
                    f"publication-facing summary differs for {row_name}",
                )
    _validate_r3d_calibration(
        family_record.get("production_preflight"),
        primitive_values=primitive_values,
        primitive_groups=primitive_groups,
    )

    baselines = record.get("baselines")
    baseline_values, baseline_groups = _validate_gene_statistics(
        baselines.get("gene_sufficient_statistics") if isinstance(baselines, Mapping) else None,
        names=(
            "baseline_partner",
            "baseline_pseudo",
            "baseline_clean_margin",
        ),
        n_examples=PREFLIGHT_N_GENES,
        expected_gene_ids=selected_ids,
        one_example_per_gene=True,
    )
    baseline_values, baseline_groups = _reorder_one_row_per_gene_statistics(
        baseline_values,
        baseline_groups,
        selected_ids,
        context="baseline preflight statistics",
    )
    partner_mean = float(np.mean(baseline_values[0]))
    pseudo_mean = float(np.mean(baseline_values[1]))
    clean_margin_mean = float(np.mean(baseline_values[2]))
    _require_r3d_preflight(
        np.isclose(baselines.get("partner", float("nan")), partner_mean, rtol=0.0, atol=1e-15)
        and np.isclose(baselines.get("pseudo", float("nan")), pseudo_mean, rtol=0.0, atol=1e-15)
        and np.isclose(
            baselines.get("clean_margin", float("nan")),
            clean_margin_mean,
            rtol=0.0,
            atol=1e-15,
        )
        and baseline_groups == family_groups,
        "baseline means do not replay from gene sufficient statistics",
    )
    construction = record.get("construction")
    policy = construction.get("policy", {}) if isinstance(construction, Mapping) else {}
    row_hashes = construction.get("row_sha256") if isinstance(construction, Mapping) else None
    _require_r3d_preflight(
        isinstance(construction, Mapping)
        and construction.get("schema") == "gfmcd.r3c.acceptor_only_construction/1"
        and construction.get("rows") == PREFLIGHT_N_GENES
        and policy
        == {
            "readout": "mask exactly one G at next_acceptor_pos - 1",
            "paired_operator": "GT -> AC at donor_pos",
            "pseudo_operator": "GT -> AC at matched_decoy_donor",
            "pseudo_distance_caliper": DECOY_TOL,
            "donor_readout_constructed": False,
        }
        and construction.get("policy_sha256") == canonical_sha256(policy)
        and construction.get("all_single_acceptor_G") is True
        and construction.get("all_donor_targets_GT") is True
        and construction.get("all_motif_clean") is True
        and _is_sha256(construction.get("ordered_rows_sha256"))
        and isinstance(row_hashes, list)
        and len(row_hashes) == PREFLIGHT_N_GENES
        and [item.get("example_id") for item in row_hashes]
        == [item["example_id"] for item in selected_rows]
        and all(_is_sha256(item.get("sha256")) for item in row_hashes),
        "acceptor-only construction contract differs",
    )

    measurements: dict[str, dict[str, Any]] = {MEAN_MODE: {}, NOISING_MODE: {}}
    for mode, head in {
        participant
        for participants in _path_health_participants().values()
        for participant in participants
    }:
        measurements[mode][head] = {
            "health": components[head]["modes"][mode]["health"]
        }
    expected_gates = build_gates(
        n_genes=PREFLIGHT_N_GENES,
        stage="preflight",
        native={
            "partner": np.full(PREFLIGHT_N_GENES, partner_mean),
            "pseudo": np.full(PREFLIGHT_N_GENES, pseudo_mean),
        },
        construction=construction,
        cohort_audit=cohort,
        measurements=measurements,
        calibration=family_record["production_preflight"],
        health=health,
        r3c_health_ok=r3c_health_ok,
    )
    _require_r3d_preflight(
        tuple(expected_gates) == PREFLIGHT_GATE_NAMES
        and record.get("gates") == expected_gates
        and all(expected_gates.values()),
        "preflight gates do not replay from stored sufficient records",
    )


def _require_r3d_target(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"R3d target semantic validation failed: {message}")


def _snapshot_contains_package(
    snapshot: Mapping[str, Any], package: Mapping[str, Any]
) -> bool:
    for role in ("artifact", "manifest"):
        item = package.get(role)
        if not isinstance(item, Mapping):
            return False
        try:
            path = str(Path(str(item.get("path"))).expanduser().resolve())
        except (TypeError, ValueError):
            return False
        if snapshot.get(path) != {
            "size_bytes": item.get("size_bytes"),
            "sha256": item.get("sha256"),
        }:
            return False
    return True


def _package_semantic_identity(package: Any) -> dict[str, Any] | None:
    """Bind immutable package bytes while permitting a worktree relocation."""
    if not isinstance(package, Mapping) or set(package) != {"artifact", "manifest"}:
        return None
    identity: dict[str, Any] = {}
    for role in ("artifact", "manifest"):
        item = package.get(role)
        if not isinstance(item, Mapping):
            return None
        raw_path = Path(str(item.get("path", ""))).expanduser()
        parts = raw_path.parts
        logical_path = (
            str(Path(*parts[parts.index("results") :]))
            if "results" in parts
            else raw_path.name
        )
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            not logical_path
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            or not _is_sha256(digest)
        ):
            return None
        identity[role] = {
            "logical_path": logical_path,
            "size_bytes": size,
            "sha256": digest,
        }
    return identity


def _packages_semantically_equal(left: Any, right: Any) -> bool:
    identity = _package_semantic_identity(left)
    return identity is not None and identity == _package_semantic_identity(right)


def validate_r3d_target(
    record: Mapping[str, Any],
    *,
    registry: Mapping[str, Any],
    registry_package: Mapping[str, Any],
    fingerprint: str,
    binding_payload: Mapping[str, Any],
    r3c_health_ok: Mapping[str, bool],
    r3c_preflight_package: Mapping[str, Any],
    r3d_preflight: Mapping[str, Any],
    r3d_preflight_package: Mapping[str, Any],
) -> None:
    """Model-free replay of every publication-facing target disposition."""
    validate_r3d_preflight(
        r3d_preflight,
        registry=registry,
        registry_package=registry_package,
        fingerprint=fingerprint,
        binding_payload=binding_payload,
        r3c_health_ok=r3c_health_ok,
        r3c_preflight_package=r3c_preflight_package,
    )
    n_genes = record.get("n_genes")
    expected_top = {
        "schema": RESULT_SCHEMA,
        "experiment": "R3d_acceptor_specificity_adjudication",
        "instrument_revision": INSTRUMENT_REVISION,
        "stage": "target",
        "status": "PROSPECTIVE_VERIFICATION",
        "tier": "PROSPECTIVE_ENDPOINT_SPECIFIC_TRAIN_VERIFICATION",
        "not_sealed_or_independent_confirmation": True,
        "target_bindable_preflight": True,
        "split": "train",
    }
    for key, expected in expected_top.items():
        _require_r3d_target(record.get(key) == expected, f"top-level {key} differs")
    _require_r3d_target(
        isinstance(n_genes, int)
        and n_genes >= TARGET_MIN_GENES
        and record.get("n") == n_genes
        and record.get("primary_estimand")
        == "raw masked-acceptor-G logit-margin component drop"
        and record.get("component_names")
        == [component_name(component) for component in COMPONENTS],
        "target does not contain its full >=2500-gene cohort",
    )
    command = record.get("command")
    _require_r3d_target(
        isinstance(command, list)
        and "--stage" in command
        and command.index("--stage") + 1 < len(command)
        and command[command.index("--stage") + 1] == "target",
        "target exact argv is missing or has the wrong stage",
    )
    hashes = record.get("hashes")
    snapshot = record.get("input_snapshot")
    identity = record.get("git_authorization_identity")
    recorded_registry_package = record.get("registry", {}).get("package")
    recorded_r3c_package = record.get("r3c_v2_1_preflight_package")
    recorded_r3d_package = record.get("r3d_preflight_package")
    _require_r3d_target(
        record.get("instrument_fingerprint_sha256") == fingerprint
        and canonical_sha256(binding_payload) == fingerprint
        and record.get("instrument_binding_payload") == binding_payload
        and isinstance(hashes, Mapping)
        and all(
            hashes.get(key) == value
            for key, value in binding_payload.get("hashes", {}).items()
        )
        and isinstance(snapshot, Mapping)
        and record.get("input_snapshot_sha256") == canonical_sha256(snapshot)
        and isinstance(identity, Mapping)
        and identity.get("dirty") is False
        and re.fullmatch(r"[0-9a-f]{40}", str(identity.get("commit"))) is not None,
        "target binding, package, snapshot, or clean-git identity differs",
    )
    _require_r3d_target(
        _packages_semantically_equal(recorded_registry_package, registry_package)
        and _packages_semantically_equal(recorded_r3c_package, r3c_preflight_package)
        and _packages_semantically_equal(recorded_r3d_package, r3d_preflight_package)
        and _snapshot_contains_package(snapshot, recorded_registry_package)
        and _snapshot_contains_package(snapshot, recorded_r3c_package)
        and _snapshot_contains_package(snapshot, recorded_r3d_package)
        and hashes.get("exposed_gene_registry_sha256")
        == registry_package.get("artifact", {}).get("sha256")
        and hashes.get("exposed_gene_registry_manifest_sha256")
        == registry_package.get("manifest", {}).get("sha256")
        and hashes.get("r3c_v2_1_preflight_sha256")
        == r3c_preflight_package.get("artifact", {}).get("sha256")
        and hashes.get("r3c_v2_1_preflight_manifest_sha256")
        == r3c_preflight_package.get("manifest", {}).get("sha256"),
        "target snapshot does not authenticate every predecessor package",
    )

    expected_registry = {
        "schema": registry["schema"],
        "scope_id": registry["scope_id"],
        "scope": registry["scope"],
        "global_gene_freshness_claimed": False,
        "gene_count": len(registry["gene_ids"]),
        "gene_ids_sha256": registry["gene_ids_sha256"],
        "file_sha256": registry_package["artifact"]["sha256"],
    }
    registry_summary = record.get("registry")
    _require_r3d_target(
        isinstance(registry_summary, Mapping)
        and set(registry_summary) == {*expected_registry, "package"}
        and all(
            registry_summary.get(key) == value
            for key, value in expected_registry.items()
        )
        and _packages_semantically_equal(
            registry_summary.get("package"), registry_package
        ),
        "target registry summary/package differs",
    )
    cohort = record.get("cohort_audit")
    selected_ids = cohort.get("selected_gene_ids") if isinstance(cohort, Mapping) else None
    selected_rows = cohort.get("selected_row_records") if isinstance(cohort, Mapping) else None
    capacity = registry.get("train_capacity", {})
    target_capacity = capacity.get("fresh_target", {})
    _require_r3d_target(
        isinstance(cohort, Mapping)
        and cohort.get("schema") == "gfmcd.r3d.train_cohort_audit/1"
        and cohort.get("selection_reads_model_outputs") is False
        and cohort.get("split") == "train"
        and cohort.get("preconstruction_allowed_chromosomes") == list(TRAIN_CHROMOSOMES)
        and cohort.get("forbidden_chromosomes") == sorted(FORBIDDEN_CHROMOSOMES)
        and cohort.get("provenance_hash_reads_full_fasta_bytes") is True
        and cohort.get("forbidden_chromosomes_materialized_as_examples") is False
        and cohort.get("forbidden_chromosomes_model_scored") is False
        and cohort.get("sealed_or_independent_claim") is False
        and cohort.get("selected_rows") == n_genes
        and cohort.get("selected_genes") == n_genes
        and cohort.get("one_row_per_gene") is True
        and cohort.get("registry_gene_intersection_count") == 0
        and cohort.get("target_uses_every_eligible_registry_disjoint_gene") is True
        and cohort.get("target_minimum_genes") == TARGET_MIN_GENES
        and cohort.get("row_rank_salt") == ROW_RANK_SALT
        and cohort.get("gene_rank_salt") == GENE_RANK_SALT
        and cohort.get("frozen_registry_capacity_replayed") is True
        and cohort.get("frozen_registry_capacity_sha256") == canonical_sha256(capacity)
        and cohort.get("eligible_genes_one_row_each") == capacity.get("eligible_genes_one_row_each")
        and cohort.get("eligible_exposed_genes")
        == capacity.get("exposed_development", {}).get("genes_one_row_each")
        and cohort.get("eligible_registry_disjoint_genes") == n_genes
        and n_genes == target_capacity.get("genes_one_row_each")
        and isinstance(selected_ids, list)
        and len(selected_ids) == n_genes
        and len(set(selected_ids)) == n_genes
        and set(selected_ids).isdisjoint(set(registry["gene_ids"]))
        and selected_ids
        == sorted(
            selected_ids,
            key=lambda gene: (_hash_rank(GENE_RANK_SALT, gene), gene),
        )
        and isinstance(selected_rows, list)
        and len(selected_rows) == n_genes
        and [row.get("gene_id") for row in selected_rows] == selected_ids
        and all(
            isinstance(row, Mapping)
            and isinstance(row.get("example_id"), str)
            and isinstance(row.get("decoy_donor"), int)
            and row.get("chrom") in TRAIN_CHROMOSOMES
            for row in selected_rows
        )
        and cohort.get("ordered_rows_sha256") == canonical_sha256(selected_rows)
        and cohort.get("ordered_gene_ids_sha256") == canonical_sha256(selected_ids)
        and canonical_sha256(sorted(selected_ids)) == target_capacity.get("gene_ids_sha256")
        and cohort.get("ordered_rows_sha256")
        == target_capacity.get("ordered_one_row_per_gene_sha256"),
        "target cohort is not the complete registry-disjoint frozen cohort",
    )

    target_identity = record.get("target_identity")
    _require_r3d_target(
        target_identity == target_identity_from_registry(registry, hashes),
        "biological target identity does not replay",
    )
    try:
        verify_loaded_target_identity(cohort, target_identity)
    except RuntimeError as exc:
        raise ValueError(f"R3d target semantic validation failed: {exc}") from exc
    attempt = record.get("target_attempt")
    receipt = attempt.get("receipt") if isinstance(attempt, Mapping) else None
    expected_attempt_pins = {
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "instrument_fingerprint_sha256": FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
        "preflight_artifact_sha256": FROZEN_PREFLIGHT_ARTIFACT_SHA256,
        "preflight_manifest_sha256": FROZEN_PREFLIGHT_MANIFEST_SHA256,
        "registry_artifact_sha256": FROZEN_EXPOSED_GENE_REGISTRY_SHA256,
        "registry_manifest_sha256": FROZEN_EXPOSED_GENE_REGISTRY_MANIFEST_SHA256,
        "r3c_runner_sha256": FROZEN_R3C_V2_1_RUNNER_SHA256,
        "r3c_preflight_sha256": FROZEN_R3C_V2_1_PREFLIGHT_SHA256,
    }
    _require_r3d_target(
        isinstance(attempt, Mapping)
        and isinstance(receipt, Mapping)
        and receipt.get("schema") == "gfmcd.r3d.target_attempt/1"
        and receipt.get("status") == "TARGET_SPENT_BEFORE_COHORT_OR_MODEL_OUTCOME"
        and receipt.get("cohort_materialized_before_record") is False
        and receipt.get("model_outcome_opened_before_record") is False
        and receipt.get("target_identity") == target_identity
        and receipt.get("target_identity_sha256") == canonical_sha256(target_identity)
        and receipt.get("invocation") == command
        and receipt.get("authorized_git_identity") == identity
        and receipt.get("instrument_fingerprint_sha256") == fingerprint
        and receipt.get("literal_pins") == expected_attempt_pins
        and receipt.get("pre_attempt_input_snapshot_sha256")
        == canonical_sha256(receipt.get("pre_attempt_input_snapshot"))
        and attempt.get("attempt_id") == receipt.get("attempt_id")
        and attempt.get("target_identity_sha256") == canonical_sha256(target_identity),
        "target-attempt receipt semantics differ",
    )
    receipt_without_id = dict(receipt)
    receipt_without_id.pop("attempt_id", None)
    receipt_bytes = (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    attempt_path = str(Path(str(attempt.get("path"))).expanduser().resolve())
    expected_attempt_snapshot = {
        "size_bytes": len(receipt_bytes),
        "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }
    _require_r3d_target(
        receipt.get("attempt_id") == canonical_sha256(receipt_without_id)
        and attempt.get("size_bytes") == len(receipt_bytes)
        and attempt.get("sha256") == expected_attempt_snapshot["sha256"]
        and snapshot.get(attempt_path) == expected_attempt_snapshot
        and dict(receipt["pre_attempt_input_snapshot"])
        == {key: value for key, value in snapshot.items() if key != attempt_path}
        and attempt.get("completion_path")
        == str(Path(attempt_path).with_suffix(".completion.json")),
        "target-attempt bytes or pre-outcome snapshot do not replay",
    )

    family_record = record.get("family")
    _require_r3d_target(
        isinstance(family_record, Mapping)
        and family_record.get("primitive_row_names") == primitive_row_names()
        and family_record.get("primitive_row_order_sha256")
        == canonical_sha256(primitive_row_names())
        and family_record.get("row_names") == family_row_names()
        and family_record.get("row_order_sha256") == canonical_sha256(family_row_names())
        and family_record.get("n_primitive_rows") == 14
        and family_record.get("n_family_rows") == 23
        and family_record.get("production_preflight_origin")
        == "INHERITED_FROM_BOUND_R3D_PREFLIGHT"
        and family_record.get("production_preflight")
        == r3d_preflight.get("family", {}).get("production_preflight"),
        "target family or inherited preflight calibration differs",
    )
    primitive_values, primitive_groups = _validate_gene_statistics(
        family_record.get("primitive_gene_sufficient_statistics"),
        names=primitive_row_names(),
        n_examples=n_genes,
        expected_gene_ids=selected_ids,
        one_example_per_gene=True,
    )
    family_values, family_groups = _validate_gene_statistics(
        family_record.get("gene_sufficient_statistics"),
        names=family_row_names(),
        n_examples=n_genes,
        expected_gene_ids=selected_ids,
        one_example_per_gene=True,
    )
    primitive_values, primitive_groups = _reorder_one_row_per_gene_statistics(
        primitive_values,
        primitive_groups,
        selected_ids,
        context="primitive target statistics",
    )
    family_values, family_groups = _reorder_one_row_per_gene_statistics(
        family_values,
        family_groups,
        selected_ids,
        context="derived target statistics",
    )
    derived, names = derive_family_from_primitives(primitive_values)
    _require_r3d_target(
        primitive_groups == family_groups
        and names == family_row_names()
        and np.array_equal(derived, family_values),
        "target 23-row family does not derive from its 14 primitive rows",
    )

    components = record.get("components")
    health = record.get("three_way_monotone_health")
    _require_r3d_target(
        isinstance(components, Mapping)
        and isinstance(health, Mapping)
        and tuple(health) == PATH_NAMES,
        "target components or three-way health are missing",
    )
    for path, participants in _path_health_participants().items():
        participant_names = [f"{mode}|{head}" for mode, head in participants]
        current_failures = []
        for mode, head in participants:
            component_health = components[head]["modes"][mode]["health"]
            _require_r3d_target(
                isinstance(component_health.get("total"), int)
                and not isinstance(component_health.get("total"), bool)
                and component_health.get("total") >= 0
                and isinstance(component_health.get("ok"), bool),
                f"target health record is malformed for {mode}|{head}",
            )
            if component_health.get("ok") is not True:
                current_failures.append(f"{mode}|{head}")
        prior_ok = (
            r3d_preflight.get("paths", {}).get(path, {}).get("health", {}).get("ok")
            is True
        )
        expected_ok = bool(
            r3c_health_ok.get(path) is True and prior_ok and not current_failures
        )
        item = health[path]
        _require_r3d_target(
            item.get("r3c_v2_1_preflight_health_ok") is (r3c_health_ok.get(path) is True)
            and item.get("r3d_preflight_health_ok") is prior_ok
            and item.get("current_bucket_health_ok") is (not current_failures)
            and item.get("ok") is expected_ok
            and item.get("can_only_shrink") is True
            and item.get("participants") == participant_names
            and item.get("failed_participants") == current_failures,
            f"target three-way health does not replay for {path}",
        )
    replay = fixed_sequence_inference(
        family_values,
        family_groups,
        family_row_names(),
        n_draws=PRODUCTION_RANDOMISATION,
        seed=PRODUCTION_SEED,
        namespace="r3d:observed-fixed-sequence",
        path_health=health,
    )
    _require_r3d_target(
        family_record.get("fixed_sequence_inference") == replay
        and record.get("diagnostic_paths") == replay["paths"],
        "target fixed-sequence inference does not replay",
    )
    expected_path_family = {
        "path_order": list(PATH_NAMES),
        "studywise_alpha": PATH_ALPHA,
        "strong_fwer_control": "fixed-sequence gatekeeping",
        "path_b_tested_only_after_path_a_supported": (
            replay["paths"][PATH_B]["classification"]
            != "NOT_TESTED_GATE_CLOSED"
        ),
    }
    _require_r3d_target(
        record.get("path_family") == expected_path_family,
        "target fixed-sequence path-family declaration differs",
    )

    replay_rows: dict[str, Any] = {}
    for path_record in replay["paths"].values():
        replay_rows.update(path_record.get("rows", {}))
    expected_component_modes = {
        "L3H1": [MEAN_MODE, NOISING_MODE],
        **{
            component_name(component): [NOISING_MODE]
            for component in ROUTE_COMPONENTS
        },
    }
    family_lookup = {
        name: family_values[index] for index, name in enumerate(family_row_names())
    }
    _require_r3d_target(
        list(components) == list(expected_component_modes),
        "target component order differs",
    )
    for head, modes in expected_component_modes.items():
        _require_r3d_target(
            components[head].get("kind") == "head"
            and list(components[head].get("modes", {})) == modes,
            f"target component modes differ for {head}",
        )
        for mode in modes:
            endpoints = components[head]["modes"][mode].get("endpoints", {})
            _require_r3d_target(
                tuple(endpoints) == ENDPOINTS,
                f"target endpoints differ for {mode}|{head}",
            )
            for endpoint in ENDPOINTS:
                row = family_row_name(mode, head, endpoint)
                summary = endpoints[endpoint]
                _require_r3d_target(
                    np.isclose(
                        summary.get("raw_mean", float("nan")),
                        float(np.mean(family_lookup[row])),
                        rtol=0.0,
                        atol=1e-15,
                    )
                    and summary.get("decision") == replay_rows.get(row),
                    f"target component summary differs for {row}",
                )

    baselines = record.get("baselines")
    baseline_values, baseline_groups = _validate_gene_statistics(
        baselines.get("gene_sufficient_statistics") if isinstance(baselines, Mapping) else None,
        names=(
            "baseline_partner",
            "baseline_pseudo",
            "baseline_clean_margin",
        ),
        n_examples=n_genes,
        expected_gene_ids=selected_ids,
        one_example_per_gene=True,
    )
    baseline_values, baseline_groups = _reorder_one_row_per_gene_statistics(
        baseline_values,
        baseline_groups,
        selected_ids,
        context="baseline target statistics",
    )
    means = [float(np.mean(row)) for row in baseline_values]
    _require_r3d_target(
        baseline_groups == family_groups
        and np.isclose(baselines.get("partner", float("nan")), means[0], rtol=0.0, atol=1e-15)
        and np.isclose(baselines.get("pseudo", float("nan")), means[1], rtol=0.0, atol=1e-15)
        and np.isclose(baselines.get("clean_margin", float("nan")), means[2], rtol=0.0, atol=1e-15),
        "target baseline means do not replay",
    )
    construction = record.get("construction")
    policy = construction.get("policy", {}) if isinstance(construction, Mapping) else {}
    row_hashes = construction.get("row_sha256") if isinstance(construction, Mapping) else None
    expected_policy_without_readout_flag = {
        "readout": "mask exactly one G at next_acceptor_pos - 1",
        "paired_operator": "GT -> AC at donor_pos",
        "pseudo_operator": "GT -> AC at matched_decoy_donor",
        "pseudo_distance_caliper": DECOY_TOL,
    }
    _require_r3d_target(
        isinstance(construction, Mapping)
        and construction.get("schema") == "gfmcd.r3c.acceptor_only_construction/1"
        and construction.get("rows") == n_genes
        and isinstance(policy, Mapping)
        and set(policy)
        == {*expected_policy_without_readout_flag, "donor_readout_constructed"}
        and all(
            policy.get(key) == value
            for key, value in expected_policy_without_readout_flag.items()
        )
        and isinstance(policy.get("donor_readout_constructed"), bool)
        and construction.get("policy_sha256") == canonical_sha256(policy)
        and isinstance(construction.get("all_single_acceptor_G"), bool)
        and isinstance(construction.get("all_donor_targets_GT"), bool)
        and isinstance(construction.get("all_motif_clean"), bool)
        and _is_sha256(construction.get("ordered_rows_sha256"))
        and isinstance(row_hashes, list)
        and len(row_hashes) == n_genes
        and [item.get("example_id") for item in row_hashes]
        == [item["example_id"] for item in selected_rows]
        and all(_is_sha256(item.get("sha256")) for item in row_hashes),
        "target acceptor-only construction differs",
    )

    measurements: dict[str, dict[str, Any]] = {MEAN_MODE: {}, NOISING_MODE: {}}
    for mode, head in {
        participant
        for participants in _path_health_participants().values()
        for participant in participants
    }:
        measurements[mode][head] = {
            "health": components[head]["modes"][mode]["health"]
        }
    expected_gates = build_gates(
        n_genes=n_genes,
        stage="target",
        native={
            "partner": np.full(n_genes, means[0]),
            "pseudo": np.full(n_genes, means[1]),
        },
        construction=construction,
        cohort_audit=cohort,
        measurements=measurements,
        calibration=family_record["production_preflight"],
        health=health,
        r3c_health_ok=r3c_health_ok,
    )
    all_gates = all(expected_gates.values())
    official = official_paths_after_global_gates(replay["paths"], expected_gates)
    expected_verdict = (
        "PROSPECTIVE_TRAIN_ENDPOINT_VERIFICATION_COMPLETE"
        if all_gates
        else "INCONCLUSIVE"
    )
    _require_r3d_target(
        tuple(expected_gates) == PREFLIGHT_GATE_NAMES
        and record.get("gates") == expected_gates
        and record.get("all_gates_pass") is all_gates
        and record.get("official_scientific_disposition_gates_pass") is all_gates
        and record.get("components_are_diagnostic_when_global_gate_fails")
        is (not all_gates)
        and record.get("paths") == official
        and record.get("verdict") == expected_verdict
        and record.get("licenses")
        == scientific_license(
            stage="target",
            all_gates_pass=all_gates,
            official_paths=official,
        ),
        "target gates, official paths, verdict, or license do not replay",
    )


def _health_measurements(arm: r3c.AcceptorArm, seed: int) -> dict[str, Any]:
    return {
        MEAN_MODE: arm.bounded_health((L3H1,), mode=MEAN_MODE, seed=seed),
        NOISING_MODE: arm.bounded_health(
            NOISING_COMPONENTS, mode=NOISING_MODE, seed=seed
        ),
    }


def measure_family(adapter: Any, arm: r3c.AcceptorArm, seed: int):
    native = arm.native()
    measurements: dict[str, dict[str, dict[str, Any]]] = {
        MEAN_MODE: {},
        NOISING_MODE: {},
    }
    partner_intervened, pseudo_intervened = arm.couplings(
        lambda: r3c.mean_intervention(adapter, L3H1)
    )
    measurements[MEAN_MODE]["L3H1"] = r3c.mean_component_endpoint_drops(
        native["partner"],
        native["pseudo"],
        partner_intervened,
        pseudo_intervened,
    )
    noised = arm.activation_noising_drops(
        NOISING_COMPONENTS, native["clean_margin"]
    )
    for component in NOISING_COMPONENTS:
        head = component_name(component)
        measurements[NOISING_MODE][head] = endpoint_vectors(
            noised[head]["partner"], noised[head]["pseudo"]
        )
    health = _health_measurements(arm, seed)
    measurements[MEAN_MODE]["L3H1"]["health"] = health[MEAN_MODE]["L3H1"]
    for component in NOISING_COMPONENTS:
        head = component_name(component)
        measurements[NOISING_MODE][head]["health"] = health[NOISING_MODE][head]
    primitive, primitive_names = build_primitive_matrix(measurements)
    family, row_names = derive_family_from_primitives(primitive)
    return native, measurements, primitive, primitive_names, family, row_names


def baseline_sufficient_statistics(
    native: Mapping[str, np.ndarray], groups: Sequence[str]
) -> dict[str, Any]:
    return r3c.gene_sufficient_statistics(
        np.vstack(
            (
                native["partner"],
                native["pseudo"],
                native["clean_margin"],
            )
        ),
        groups,
        [
            "baseline_partner",
            "baseline_pseudo",
            "baseline_clean_margin",
        ],
    )


def _compact_path(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": result.get("path"),
        "classification": result["classification"],
        "health_ok": result.get("health", {}).get("ok"),
        "all_edges_reject": result.get("all_edges_reject"),
        "edge_p_values": result.get("edge_p_values"),
        "test_interval_coherence": result.get("test_interval_coherence"),
        "intersection_union_p_value": result.get("intersection_union_p_value"),
        "randomisation_audit": result.get("randomisation_audit"),
    }


def calibration_preflight(
    primitive: np.ndarray,
    groups: Sequence[str],
    *,
    n_draws: int,
    n_trials: int,
    n_plant_trials: int | None = None,
    seed: int,
    namespace: str,
) -> dict[str, Any]:
    """Exercise all 36 null faces and a coherent plant through production inference."""
    plant_trials = n_trials if n_plant_trials is None else n_plant_trials
    if n_trials < 1 or plant_trials < 1 or n_draws < 1:
        raise ValueError("calibration draw counts must be positive")
    residuals = center_primitive_rows(primitive)
    health = {
        path: {"ok": True, "source": "synthetic calibration"} for path in PATH_NAMES
    }
    surfaces = boundary_face_surface_specs()
    face_records: dict[str, Any] = {}
    for face_name, spec in surfaces.items():
        positives = 0
        patterns: list[int] = []
        audits: list[str] = []
        outer_null_hashes: list[str] = []
        for trial in range(n_trials):
            null = gene_sign_flip(
                residuals,
                groups,
                seed=seed,
                namespace=f"{namespace}:face:{face_name}:outer:{trial}",
            )
            family, names = derive_family_from_primitives(
                null + np.asarray(spec["offset"], dtype=float)[:, None]
            )
            outer_null_hashes.append(_array_sha256(null))
            path = str(spec["path"])
            result = infer_one_path(
                family,
                groups,
                names,
                path,
                n_draws=n_draws,
                seed=seed,
                namespace=f"{namespace}:face:{face_name}:classifier:{trial}",
                health=health[path],
            )
            declared = result["classification"] == "SUPPORTED"
            positives += int(declared)
            patterns.append(int(declared))
            compact = _compact_path(result)
            audits.append(canonical_sha256(compact))
        _, upper = one_sided_binomial_bounds(positives, n_trials, alpha=0.05)
        count_gate = MAX_FACE_FALSE_POSITIVES if n_trials == PRODUCTION_TRIALS else n_trials
        face_records[face_name] = {
            "path": spec["path"],
            "target_row": spec["target_row"],
            "edge": spec["edge"],
            "trials": n_trials,
            "false_positive_trials": positives,
            "false_positive_rate": positives / n_trials,
            "one_sided_95pct_clopper_pearson_upper": upper,
            "one_sided_cp_upper_is_descriptive_only": True,
            "maximum_false_positive_trials": count_gate,
            "engineering_fpr_ok": face_max_count_pass(positives, n_trials),
            "outer_namespace_template": (
                f"{namespace}:face:{face_name}:outer:<trial>"
            ),
            "classifier_namespace_template": (
                f"{namespace}:face:{face_name}:classifier:<trial>"
            ),
            "support_pattern": patterns,
            "pattern_sha256": canonical_sha256(patterns),
            "outer_null_digest_sha256": canonical_sha256(outer_null_hashes),
            "classifier_audit_digest_sha256": canonical_sha256(audits),
        }

    plant = coherent_path_offset()
    successes = {path: 0 for path in PATH_NAMES}
    plant_patterns = {path: [] for path in PATH_NAMES}
    plant_audits: list[str] = []
    plant_outer_null_hashes: list[str] = []
    for trial in range(plant_trials):
        null = gene_sign_flip(
            residuals,
            groups,
            seed=seed,
            namespace=f"{namespace}:plant:outer:{trial}",
        )
        family, names = derive_family_from_primitives(null + plant[:, None])
        plant_outer_null_hashes.append(_array_sha256(null))
        result = fixed_sequence_inference(
            family,
            groups,
            names,
            n_draws=n_draws,
            seed=seed,
            namespace=f"{namespace}:plant:classifier:{trial}",
            path_health=health,
        )
        for path in PATH_NAMES:
            supported = int(
                result["paths"][path]["classification"] == "SUPPORTED"
            )
            successes[path] += supported
            plant_patterns[path].append(supported)
        compact = {
            path: _compact_path(result["paths"][path]) for path in PATH_NAMES
        }
        plant_audits.append(canonical_sha256(compact))
    plants = {
        path: {
            "successes": count,
            "trials": plant_trials,
            "power": count / plant_trials,
            "minimum_successes": (
                MIN_PLANT_SUCCESSES
                if plant_trials == PRODUCTION_PLANT_TRIALS
                else plant_trials
            ),
            "powered": plant_count_pass(count, plant_trials),
        }
        for path, count in successes.items()
    }
    if successes[PATH_B] > successes[PATH_A]:
        raise RuntimeError("fixed-sequence plant successes violate B <= A")
    return {
        "schema": "gfmcd.r3d.boundary_face_calibration/1",
        "randomisation_unit": "whole gene",
        "translation_invariant_centering": True,
        "n_primitive_rows": 14,
        "n_family_rows": 23,
        "n_boundary_faces": 36,
        "trials_per_surface": n_trials,
        "plant_trials": plant_trials,
        "n_calibration_draws": n_draws,
        "seed": seed,
        "namespace": namespace,
        "boundary_face_suite": boundary_face_suite_binding(),
        "boundary_faces": face_records,
        "all_boundary_face_engineering_fpr_ok": all(
            record["engineering_fpr_ok"] for record in face_records.values()
        ),
        "face_familywise_max_count_operating_characteristics": {
            "point_null_per_face_probability": PATH_ALPHA,
            "maximum_false_positive_trials": MAX_FACE_FALSE_POSITIVES,
            "single_face_fail_probability": FACE_FAIL_PROBABILITY_AT_POINT05,
            "union_bound_any_of_36_faces_fail": FACE_SUITE_UNION_BOUND_AT_POINT05,
            "suite_pass_probability_lower_bound": FACE_SUITE_PASS_LOWER_AT_POINT05,
            "cp_upper_at_maximum_count_descriptive_only": FACE_CP_UPPER_AT_MAX_COUNT,
            "does_not_assume_cross_face_independence": True,
        },
        "plant_delta": PLANT_DELTA,
        "coherent_primitive_plant_sha256": _array_sha256(plant),
        "plant_successes": plants,
        "plant_operating_characteristics": {
            "trials": PRODUCTION_PLANT_TRIALS,
            "minimum_successes": MIN_PLANT_SUCCESSES,
            "one_sided_95pct_cp_lower_at_threshold": PLANT_CP_LOWER_AT_THRESHOLD,
            "pass_probability_at_true_joint_success_point95": (
                PLANT_PASS_PROBABILITY_AT_POINT95
            ),
            "path_b_successes_must_not_exceed_path_a_successes": True,
        },
        "plant_outer_namespace_template": f"{namespace}:plant:outer:<trial>",
        "plant_classifier_namespace_template": (
            f"{namespace}:plant:classifier:<trial>"
        ),
        "plant_support_patterns": plant_patterns,
        "plant_support_pattern_sha256": {
            path: canonical_sha256(pattern) for path, pattern in plant_patterns.items()
        },
        "plant_classifier_audit_digest_sha256": canonical_sha256(plant_audits),
        "plant_outer_null_digest_sha256": canonical_sha256(
            plant_outer_null_hashes
        ),
        "all_plants_at_least_462_of_500": bool(
            plant_trials == PRODUCTION_PLANT_TRIALS
            and all(
                plant_count_pass(record["successes"], plant_trials)
                for record in plants.values()
            )
        ),
        "all_plants_meet_stage_threshold": all(
            record["powered"] for record in plants.values()
        ),
        "engineering_not_inferential": True,
    }


def calibration_for_stage(
    primitive: np.ndarray,
    groups: Sequence[str],
    *,
    stage: str,
    n_draws: int,
    n_trials: int,
    seed: int,
    namespace: str,
) -> dict[str, Any]:
    """Separate smoke mechanics from observed-residual sensitivity."""
    if stage == "preflight":
        observed = calibration_preflight(
            primitive,
            groups,
            n_draws=n_draws,
            n_trials=n_trials,
            n_plant_trials=PRODUCTION_PLANT_TRIALS,
            seed=seed,
            namespace=namespace,
        )
        return {
            "gating_basis": "OBSERVED_CENTERED_RESIDUALS_PRODUCTION",
            "gating_calibration": observed,
            "observed_centered_residual_non_gating_diagnostic": None,
        }
    if stage != "smoke":
        raise ValueError("stage calibration supports smoke or preflight only")
    observed = calibration_preflight(
        primitive,
        groups,
        n_draws=n_draws,
        n_trials=n_trials,
        n_plant_trials=n_trials,
        seed=seed,
        namespace=namespace,
    )
    mechanics = calibration_preflight(
        np.zeros_like(np.asarray(primitive, dtype=float)),
        groups,
        n_draws=n_draws,
        n_trials=n_trials,
        n_plant_trials=n_trials,
        seed=seed,
        namespace=f"{namespace}:smoke-zero-residual-mechanics",
    )
    return {
        "gating_basis": "ZERO_RESIDUAL_SMOKE_MECHANICS",
        "gating_calibration": mechanics,
        "observed_centered_residual_non_gating_diagnostic": observed,
    }


def smoke_calibration_report(staged: Mapping[str, Any]) -> dict[str, Any]:
    """Return the explicit smoke-only distinction between mechanics and power."""
    if staged.get("gating_basis") != "ZERO_RESIDUAL_SMOKE_MECHANICS":
        raise ValueError("smoke calibration report requires zero-residual mechanics")
    mechanics = staged.get("gating_calibration")
    observed = staged.get("observed_centered_residual_non_gating_diagnostic")
    if not isinstance(mechanics, Mapping) or not isinstance(observed, Mapping):
        raise ValueError("smoke calibration report is incomplete")

    def compact(record: Mapping[str, Any], *, gating: bool) -> dict[str, Any]:
        return {
            "gating": gating,
            "can_authorize_target": False,
            "n_boundary_faces": record.get("n_boundary_faces"),
            "trials_per_surface": record.get("trials_per_surface"),
            "plant_trials": record.get("plant_trials"),
            "plant_successes": record.get("plant_successes"),
            "plant_support_patterns": record.get("plant_support_patterns"),
            "all_boundary_face_engineering_fpr_ok": record.get(
                "all_boundary_face_engineering_fpr_ok"
            ),
            "all_plants_meet_stage_threshold": record.get(
                "all_plants_meet_stage_threshold"
            ),
        }

    return {
        "gating_basis": staged["gating_basis"],
        "zero_residual_mechanics": compact(mechanics, gating=True),
        "observed_centered_residual_non_gating_diagnostic": compact(
            observed, gating=False
        ),
    }


def build_gates(
    *,
    n_genes: int,
    stage: str,
    native: Mapping[str, np.ndarray],
    construction: Mapping[str, Any],
    cohort_audit: Mapping[str, Any],
    measurements: Mapping[str, Mapping[str, Mapping[str, Any]]],
    calibration: Mapping[str, Any],
    health: Mapping[str, Mapping[str, Any]],
    r3c_health_ok: Mapping[str, bool],
) -> dict[str, bool]:
    health_executed = all(
        measurements[mode][head]["health"].get("total", 0) > 0
        for mode, head in {
            participant
            for values in _path_health_participants().values()
            for participant in values
        }
    )
    target_size_ok = (
        n_genes >= TARGET_MIN_GENES if stage == "target" else n_genes >= MIN_GENES
    )
    gates = {
        "minimum_independent_genes": target_size_ok,
        "explicit_preconstruction_train_chromosome_restriction": (
            cohort_audit.get("preconstruction_allowed_chromosomes")
            == list(TRAIN_CHROMOSOMES)
            and cohort_audit.get("provenance_hash_reads_full_fasta_bytes") is True
            and cohort_audit.get("forbidden_chromosomes_materialized_as_examples")
            is False
            and cohort_audit.get("forbidden_chromosomes_model_scored") is False
        ),
        "one_hash_ranked_row_per_gene": (
            cohort_audit.get("one_row_per_gene") is True and n_genes == cohort_audit.get("selected_rows")
        ),
        "frozen_registry_train_capacity_replayed": (
            cohort_audit.get("frozen_registry_capacity_replayed") is True
        ),
        "target_excludes_registry_or_preflight_uses_registry": (
            cohort_audit.get("registry_gene_intersection_count") == (0 if stage == "target" else n_genes)
        ),
        "native_paired_donor_positive_control": float(np.mean(native["partner"])) > 0.20,
        "matched_pseudo_donor_negative_control": abs(float(np.mean(native["pseudo"]))) < SESOI,
        "single_acceptor_G_and_GT_to_AC_only": bool(
            construction.get("all_single_acceptor_G")
            and construction.get("all_donor_targets_GT")
            and construction.get("policy", {}).get("donor_readout_constructed") is False
        ),
        "zero_planted_donor_motifs": construction.get("all_motif_clean") is True,
        "exact_14_primitive_23_row_family": True,
        "three_way_monotone_health_registry_valid": all(
            item.get("can_only_shrink") is True
            and isinstance(item.get("ok"), bool)
            and isinstance(item.get("participants"), list)
            and isinstance(item.get("failed_participants"), list)
            for item in health.values()
        ),
        "preflight_path_a_health_authorizes_target_entry": (
            health[PATH_A].get("ok") is True
            if stage == "preflight"
            else True
        ),
        "r3c_v2_1_preflight_health_bound": (
            r3c_health_ok.get(PATH_A) is True
            and all(isinstance(r3c_health_ok.get(path), bool) for path in PATH_NAMES)
        ),
        "bounded_identical_policy_health_executed": health_executed,
        "all_36_boundary_faces_at_most_20_of_200": (
            calibration.get("n_boundary_faces") == 36
            if stage == "smoke"
            else calibration.get("all_boundary_face_engineering_fpr_ok") is True
        ),
        "coherent_plant_at_least_462_of_500": (
            calibration.get("all_plants_meet_stage_threshold") is True
            if stage == "smoke"
            else calibration.get("all_plants_at_least_462_of_500") is True
        ),
        "test_inversion_is_decision_instrument": True,
    }
    return gates


def _component_output(
    measurements: Mapping[str, Mapping[str, Mapping[str, Any]]],
    inference: Mapping[str, Any],
) -> dict[str, Any]:
    row_lookup: dict[str, Any] = {}
    for path in inference["paths"].values():
        row_lookup.update(path.get("rows", {}))
    output: dict[str, Any] = {}
    for mode, head in [(MEAN_MODE, "L3H1")] + [
        (NOISING_MODE, component_name(component)) for component in NOISING_COMPONENTS
    ]:
        component = output.setdefault(head, {"kind": "head", "modes": {}})
        component["modes"][mode] = {
            "health": measurements[mode][head]["health"],
            "endpoints": {
                endpoint: {
                    "raw_mean": float(np.mean(measurements[mode][head][endpoint])),
                    "decision": row_lookup.get(family_row_name(mode, head, endpoint)),
                }
                for endpoint in ENDPOINTS
            },
        }
    return output


def official_paths_after_global_gates(
    diagnostic_paths: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, bool],
) -> dict[str, Any]:
    official = deepcopy(dict(diagnostic_paths))
    failures = [name for name, passed in gates.items() if not passed]
    if failures:
        for path in PATH_NAMES:
            official[path]["diagnostic_classification"] = official[path].get(
                "classification"
            )
            official[path]["classification"] = "INCONCLUSIVE_GLOBAL_GATE_FAILED"
            official[path]["official_scientific_disposition"] = False
            official[path]["global_gate_failures"] = failures
    else:
        for path in PATH_NAMES:
            official[path]["official_scientific_disposition"] = True
    return official


def scientific_license(
    *, stage: str, all_gates_pass: bool, official_paths: Mapping[str, Mapping[str, Any]]
) -> str:
    if stage != "target" or not all_gates_pass:
        return "no positive scientific disposition; development mechanics only"
    if official_paths[PATH_A].get("classification") != "SUPPORTED":
        return "no positive scientific disposition"
    if official_paths[PATH_B].get("classification") == "SUPPORTED":
        return (
            "registry-disjoint masked-acceptor endpoint support for L3H1 reuse and "
            "the original distinct-route component pattern"
        )
    return "registry-disjoint masked-acceptor endpoint support for L3H1 reuse only"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "preflight", "target"), default="smoke")
    parser.add_argument("--n-genes", type=int, default=None)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument("--n-randomisation", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--exposed-gene-registry", default=None)
    parser.add_argument("--r3c-preflight-artifact", default=None)
    parser.add_argument("--preflight-artifact", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # This authorization block is intentionally before every caller-controlled path,
    # output reservation, checkpoint discovery, cohort load, or artifact read.
    if args.stage == "target":
        try:
            authorize_target(
                sha256_file(PROTOCOL), protocol_text=PROTOCOL.read_text()
            )
        except TargetSealError as exc:
            raise SystemExit(str(exc)) from exc

    args.n_genes = args.n_genes if args.n_genes is not None else (
        SMOKE_N_GENES if args.stage == "smoke" else PREFLIGHT_N_GENES if args.stage == "preflight" else None
    )
    args.batch = args.batch if args.batch is not None else (
        8 if args.stage == "smoke" else PRODUCTION_BATCH
    )
    args.n_randomisation = args.n_randomisation if args.n_randomisation is not None else (
        SMOKE_RANDOMISATION if args.stage == "smoke" else PRODUCTION_RANDOMISATION
    )
    args.n_trials = args.n_trials if args.n_trials is not None else (
        SMOKE_TRIALS if args.stage == "smoke" else PRODUCTION_TRIALS
    )
    if args.stage == "target" and args.n_genes is not None:
        raise SystemExit(
            "R3d target forbids --n-genes and uses every eligible registry-disjoint gene"
        )
    if args.stage == "preflight" and args.n_genes != PREFLIGHT_N_GENES:
        raise SystemExit(f"R3d preflight is fixed at {PREFLIGHT_N_GENES} genes")
    if args.stage != "target" and (args.n_genes is None or args.n_genes < MIN_GENES):
        raise SystemExit(f"R3d requires at least {MIN_GENES} independent genes")
    if min(args.batch, args.n_randomisation, args.n_trials) < 1:
        raise SystemExit("R3d execution counts must be positive")
    if args.stage != "smoke":
        fixed = {
            "batch": (args.batch, PRODUCTION_BATCH),
            "seed": (args.seed, PRODUCTION_SEED),
            "n-randomisation": (args.n_randomisation, PRODUCTION_RANDOMISATION),
            "n-trials": (args.n_trials, PRODUCTION_TRIALS),
        }
        drift = [
            f"--{name}={actual} (fixed {expected})"
            for name, (actual, expected) in fixed.items()
            if actual != expected
        ]
        if drift:
            raise SystemExit("R3d production parameters are fixed: " + "; ".join(drift))
    if not args.exposed_gene_registry or not args.r3c_preflight_artifact:
        raise SystemExit(
            "R3d requires --exposed-gene-registry and --r3c-preflight-artifact"
        )
    if args.stage == "target" and not args.preflight_artifact:
        raise SystemExit("R3d target requires --preflight-artifact")
    if args.stage != "smoke" and args.out is None:
        raise SystemExit("R3d preflight/target requires a fresh --out path")

    output = Path(args.out).expanduser().resolve() if args.out else None
    manifest_path = output.with_suffix(".manifest.json") if output else None
    if output and (output.exists() or manifest_path.exists()):
        raise SystemExit("R3d outputs are immutable; choose a fresh versioned path")

    registry_path = Path(args.exposed_gene_registry).expanduser().resolve()
    r3c_preflight_path = Path(args.r3c_preflight_artifact).expanduser().resolve()
    fasta_index = fasta_index_path(args.fasta)
    model_files = r3c.checkpoint_files(args.model)
    bound_paths: list[Path] = [
        *model_files,
        Path(args.gtf),
        Path(args.fasta),
        fasta_index,
        registry_path,
        registry_path.with_suffix(".manifest.json"),
        r3c_preflight_path,
        r3c_preflight_path.with_suffix(".manifest.json"),
        R3C_RUNNER,
        RUNNER,
        PROTOCOL,
        STAT_CORE,
        *BOUND_CODE_SOURCES,
    ]
    if args.preflight_artifact:
        preflight_path = Path(args.preflight_artifact).expanduser().resolve()
        bound_paths.extend((preflight_path, preflight_path.with_suffix(".manifest.json")))
    snapshot = _snapshot(bound_paths)
    registry_artifact, registry_package = _load_registry_package(
        registry_path,
        FROZEN_EXPOSED_GENE_REGISTRY_SHA256,
        FROZEN_EXPOSED_GENE_REGISTRY_MANIFEST_SHA256,
        require_clean_generator=(
            args.stage == "target"
            or FROZEN_EXPOSED_GENE_REGISTRY_SHA256 is not None
            or FROZEN_EXPOSED_GENE_REGISTRY_MANIFEST_SHA256 is not None
        ),
    )
    registry = read_exposed_gene_registry(registry_path)
    if registry_artifact != registry:
        raise SystemExit("exposed-gene registry changed while its package was loaded")
    r3c_preflight, r3c_preflight_package = _load_artifact_package(
        r3c_preflight_path, FROZEN_R3C_V2_1_PREFLIGHT_SHA256
    )
    try:
        r3c.validate_preflight_binding(
            r3c_preflight,
            r3c_preflight["instrument_fingerprint_sha256"],
            r3c_preflight["instrument_binding_payload"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"R3c v2.1 predecessor semantic replay failed: {exc}") from exc
    r3c_health = r3c_preflight_health(r3c_preflight)
    if (
        FROZEN_EXPOSED_GENE_REGISTRY_SHA256 is not None
        and sha256_file(registry_path) != FROZEN_EXPOSED_GENE_REGISTRY_SHA256
    ):
        raise SystemExit("exposed-gene registry differs from its literal pin")
    if (
        FROZEN_R3C_V2_1_RUNNER_SHA256 is not None
        and sha256_file(R3C_RUNNER) != FROZEN_R3C_V2_1_RUNNER_SHA256
    ):
        raise SystemExit("R3c v2.1 runner differs from its literal R3d pin")

    r3d_preflight = None
    r3d_preflight_package = None
    if args.stage == "target":
        r3d_preflight, r3d_preflight_package = _load_artifact_package(
            args.preflight_artifact,
            FROZEN_PREFLIGHT_ARTIFACT_SHA256,
            FROZEN_PREFLIGHT_MANIFEST_SHA256,
        )

    # Compute and verify the complete source/instrument binding before cohort sequence
    # construction, adapter loading, or any model outcome.  A wrong frozen fingerprint
    # therefore cannot spend even one target row.
    device = pick_device(args.device)
    hashes = {
        "protocol_sha256": sha256_file(PROTOCOL),
        "runner_normalized_sha256": normalized_runner_sha256(),
        "r3c_v2_1_runner_sha256": sha256_file(R3C_RUNNER),
        "r3c_v2_1_preflight_sha256": sha256_file(r3c_preflight_path),
        "r3c_v2_1_preflight_manifest_sha256": sha256_file(
            r3c_preflight_path.with_suffix(".manifest.json")
        ),
        "exposed_gene_registry_sha256": sha256_file(registry_path),
        "exposed_gene_registry_manifest_sha256": sha256_file(
            registry_path.with_suffix(".manifest.json")
        ),
        "stat_core_sha256": sha256_file(STAT_CORE),
        "bound_code_sources": {
            str(path.resolve().relative_to(REPO)): snapshot[str(path.resolve())]
            for path in BOUND_CODE_SOURCES
        },
        "gtf_sha256": sha256_file(args.gtf),
        "fasta_sha256": sha256_file(args.fasta),
        "fasta_index_sha256": sha256_file(fasta_index),
        "checkpoint_aggregate_sha256": r3c.checkpoint_bundle_from_snapshot(
            args.model, model_files, snapshot
        )[1],
    }
    binding = instrument_binding_payload(
        hashes=hashes,
        batch=args.batch,
        device=device,
        seed=args.seed,
        n_randomisation=args.n_randomisation,
        n_trials=args.n_trials,
    )
    fingerprint = canonical_sha256(binding)
    try:
        target_bindable_preflight = bool(
            args.stage == "preflight"
            and authorize_bindable_preflight(
                hashes=hashes,
                fingerprint=fingerprint,
                protocol_text=PROTOCOL.read_text(),
            )
        )
    except TargetSealError as exc:
        raise SystemExit(f"R3d bindable preflight authorization failed: {exc}") from exc
    if args.stage == "target":
        if fingerprint != FROZEN_INSTRUMENT_FINGERPRINT_SHA256:
            raise SystemExit("R3d instrument fingerprint differs from the frozen pin")
        if (
            r3d_preflight.get("instrument_fingerprint_sha256") != fingerprint
            or r3d_preflight.get("instrument_binding_payload") != binding
        ):
            raise SystemExit(
                "R3d target binding differs from the exact bound R3d preflight"
            )
        try:
            validate_r3d_preflight(
                r3d_preflight,
                registry=registry,
                registry_package=registry_package,
                fingerprint=fingerprint,
                binding_payload=binding,
                r3c_health_ok=r3c_health,
                r3c_preflight_package=r3c_preflight_package,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    git_authorization_identity = None
    if target_bindable_preflight or args.stage == "target":
        try:
            git_authorization_identity = r3c.require_clean_git()
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    target_attempt = None
    target_identity = None
    if args.stage == "target":
        assert git_authorization_identity is not None
        target_identity = target_identity_from_registry(registry, hashes)
        attempt_path = canonical_target_attempt_path(target_identity)
        try:
            target_attempt = write_target_attempt(
                attempt_path,
                target_identity=target_identity,
                invocation=sys.argv,
                authorized_git_identity=git_authorization_identity,
                instrument_fingerprint=fingerprint,
                pins={
                    "protocol_sha256": FROZEN_PROTOCOL_SHA256,
                    "instrument_fingerprint_sha256": (
                        FROZEN_INSTRUMENT_FINGERPRINT_SHA256
                    ),
                    "preflight_artifact_sha256": FROZEN_PREFLIGHT_ARTIFACT_SHA256,
                    "preflight_manifest_sha256": FROZEN_PREFLIGHT_MANIFEST_SHA256,
                    "registry_artifact_sha256": (
                        FROZEN_EXPOSED_GENE_REGISTRY_SHA256
                    ),
                    "registry_manifest_sha256": (
                        FROZEN_EXPOSED_GENE_REGISTRY_MANIFEST_SHA256
                    ),
                    "r3c_runner_sha256": FROZEN_R3C_V2_1_RUNNER_SHA256,
                    "r3c_preflight_sha256": FROZEN_R3C_V2_1_PREFLIGHT_SHA256,
                },
                input_snapshot=snapshot,
            )
        except FileExistsError as exc:
            raise SystemExit(
                "R3d biological target has already been spent; immutable attempt receipt exists"
            ) from exc
        snapshot = dict(
            sorted(
                {
                    **snapshot,
                    **_snapshot([target_attempt["path"]]),
                }.items()
            )
        )
        install_target_attempt_failure_hook(target_attempt)

    examples, decoy_donors, cohort_audit = load_cohort(args, registry)
    if target_identity is not None:
        verify_loaded_target_identity(cohort_audit, target_identity)
    groups = [str(example.gene_id) for example in examples]
    if len(groups) != len(set(groups)):
        raise RuntimeError("R3d independent-unit contract requires one row per gene")
    adapter = load_adapter(args.model, device, eager=False)
    task = CouplingTask(adapter, batch=args.batch)
    arm = r3c.AcceptorArm(task, examples, decoy_donors)
    construction = r3c.construction_record(examples, arm)
    (
        native,
        measurements,
        primitive,
        primitive_names,
        family,
        row_names,
    ) = measure_family(adapter, arm, args.seed)
    current_health = observed_path_health(measurements)
    health = three_way_monotone_health(
        r3c_health=r3c_health,
        current_health=current_health,
        r3d_preflight_paths=(
            None if r3d_preflight is None else r3d_preflight.get("paths", {})
        ),
    )
    inference = fixed_sequence_inference(
        family,
        groups,
        row_names,
        n_draws=args.n_randomisation,
        seed=args.seed,
        namespace="r3d:observed-fixed-sequence",
        path_health=health,
    )
    staged_calibration: dict[str, Any] | None = None
    if args.stage == "target":
        calibration = r3d_preflight["family"]["production_preflight"]
        calibration_origin = "INHERITED_FROM_BOUND_R3D_PREFLIGHT"
    else:
        staged_calibration = calibration_for_stage(
            primitive,
            groups,
            stage=args.stage,
            n_draws=args.n_randomisation,
            n_trials=args.n_trials,
            seed=args.seed,
            namespace="r3d:production-boundary-face-preflight",
        )
        calibration = staged_calibration["gating_calibration"]
        calibration_origin = (
            "RUN_ON_ZERO_RESIDUAL_SMOKE_MECHANICS"
            if args.stage == "smoke"
            else "RUN_ON_CURRENT_BUCKET"
        )

    gates = build_gates(
        n_genes=len(groups),
        stage=args.stage,
        native=native,
        construction=construction,
        cohort_audit=cohort_audit,
        measurements=measurements,
        calibration=calibration,
        health=health,
        r3c_health_ok=r3c_health,
    )
    all_gates_pass = all(gates.values())
    if args.stage == "preflight":
        target_bindable_preflight = bool(
            target_bindable_preflight and all_gates_pass
        )
    elif args.stage == "target":
        target_bindable_preflight = True
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
        verdict = "PROSPECTIVE_TRAIN_ENDPOINT_VERIFICATION_COMPLETE"

    diagnostic_paths = inference["paths"]
    official_paths = official_paths_after_global_gates(diagnostic_paths, gates)
    family_stats = r3c.gene_sufficient_statistics(family, groups, row_names)
    result = {
        "schema": RESULT_SCHEMA,
        "experiment": "R3d_acceptor_specificity_adjudication",
        "instrument_revision": INSTRUMENT_REVISION,
        "stage": args.stage,
        "status": "DEVELOPMENT" if args.stage != "target" else "PROSPECTIVE_VERIFICATION",
        "tier": "PROSPECTIVE_ENDPOINT_SPECIFIC_TRAIN_VERIFICATION",
        "not_sealed_or_independent_confirmation": True,
        "command": list(sys.argv),
        "target_bindable_preflight": target_bindable_preflight,
        "split": "train",
        "n": len(examples),
        "n_genes": len(groups),
        "cohort_audit": cohort_audit,
        "registry": {
            "schema": registry["schema"],
            "scope_id": registry["scope_id"],
            "scope": registry["scope"],
            "global_gene_freshness_claimed": registry["scope"][
                "global_freshness_claimed"
            ],
            "gene_count": len(registry["gene_ids"]),
            "gene_ids_sha256": registry["gene_ids_sha256"],
            "file_sha256": hashes["exposed_gene_registry_sha256"],
            "package": registry_package,
        },
        "r3c_v2_1_preflight_package": r3c_preflight_package,
        "r3d_preflight_package": r3d_preflight_package,
        "construction": construction,
        "primary_estimand": "raw masked-acceptor-G logit-margin component drop",
        "component_names": [component_name(component) for component in COMPONENTS],
        "baselines": {
            "partner": float(np.mean(native["partner"])),
            "pseudo": float(np.mean(native["pseudo"])),
            "clean_margin": float(np.mean(native["clean_margin"])),
            "gene_sufficient_statistics": baseline_sufficient_statistics(
                native, groups
            ),
        },
        "family": {
            "definition": "14 primitive partner/pseudo rows derive exactly 23 decision rows",
            "primitive_row_names": primitive_names,
            "primitive_row_order_sha256": canonical_sha256(primitive_names),
            "primitive_gene_sufficient_statistics": (
                r3c.gene_sufficient_statistics(primitive, groups, primitive_names)
            ),
            "row_names": row_names,
            "row_order_sha256": canonical_sha256(row_names),
            "n_primitive_rows": primitive.shape[0],
            "n_family_rows": family.shape[0],
            "gene_sufficient_statistics": family_stats,
            "fixed_sequence_inference": inference,
            "production_preflight": calibration,
            "production_preflight_origin": calibration_origin,
        },
        "components": _component_output(measurements, inference),
        "components_are_diagnostic_when_global_gate_fails": not all_gates_pass,
        "diagnostic_paths": diagnostic_paths,
        "paths": official_paths,
        "path_family": {
            "path_order": list(PATH_NAMES),
            "studywise_alpha": PATH_ALPHA,
            "strong_fwer_control": "fixed-sequence gatekeeping",
            "path_b_tested_only_after_path_a_supported": (
                diagnostic_paths[PATH_B]["classification"]
                != "NOT_TESTED_GATE_CLOSED"
            ),
        },
        "three_way_monotone_health": health,
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "official_scientific_disposition_gates_pass": all_gates_pass,
        "verdict": verdict,
        "hashes": hashes,
        "instrument_binding_payload": binding,
        "instrument_fingerprint_sha256": fingerprint,
        "git_authorization_identity": git_authorization_identity,
        "target_identity": target_identity,
        "target_attempt": target_attempt,
        "input_snapshot": snapshot,
        "input_snapshot_sha256": canonical_sha256(snapshot),
        "licenses": scientific_license(
            stage=args.stage,
            all_gates_pass=all_gates_pass,
            official_paths=official_paths,
        ),
        "does_not_license": [
            "sealed-split or independent confirmation",
            "attention as mechanism",
            "address or payload identity",
            "acceptor-side write",
            "delivery or mediation",
            "sufficiency or faithfulness",
            "biological mechanism",
        ],
    }
    if args.stage == "smoke":
        assert staged_calibration is not None
        result["family"]["smoke_calibration"] = smoke_calibration_report(
            staged_calibration
        )
    if args.stage == "preflight" and target_bindable_preflight:
        try:
            validate_r3d_preflight(
                result,
                registry=registry,
                registry_package=registry_package,
                fingerprint=fingerprint,
                binding_payload=binding,
                r3c_health_ok=r3c_health,
                r3c_preflight_package=r3c_preflight_package,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    if args.stage == "target":
        assert r3d_preflight is not None
        assert r3d_preflight_package is not None
        try:
            validate_r3d_target(
                result,
                registry=registry,
                registry_package=registry_package,
                fingerprint=fingerprint,
                binding_payload=binding,
                r3c_health_ok=r3c_health,
                r3c_preflight_package=r3c_preflight_package,
                r3d_preflight=r3d_preflight,
                r3d_preflight_package=r3d_preflight_package,
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    if args.stage == "smoke":
        print(
            json.dumps(
                {
                    "verdict": verdict,
                    "gates": gates,
                    "paths": {
                        path: inference["paths"][path]["classification"]
                        for path in PATH_NAMES
                    },
                    "smoke_calibration": result["family"]["smoke_calibration"],
                    "fingerprint": fingerprint,
                },
                indent=2,
            )
        )
        return

    assert output is not None and manifest_path is not None
    result_bytes = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode()
    manifest = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r3d_acceptor_specificity_adjudication",
        sys.argv,
        repo=REPO,
        status=result["status"],
        config={
            "stage": args.stage,
            "n_genes": len(groups),
            "batch": args.batch,
            "seed": args.seed,
            "n_randomisation": args.n_randomisation,
            "n_trials": args.n_trials,
            "n_plant_trials": calibration.get("plant_trials"),
            "instrument_fingerprint_sha256": fingerprint,
            "family_row_order_sha256": canonical_sha256(row_names),
            "cohort_rows_sha256": cohort_audit["ordered_rows_sha256"],
            "input_snapshot_sha256": canonical_sha256(snapshot),
            "target_attempt_id": (
                None if target_attempt is None else target_attempt["attempt_id"]
            ),
        },
        seeds={"randomisation": args.seed, "health_probe_selection": args.seed},
        notes=[
            "Masked acceptor G only; no masked-donor endpoint is constructed.",
            "Canonical train chromosomes are restricted before example construction and FASTA-window fetch.",
            "One outcome-blind hash-ranked row is selected per gene.",
            "Target uses every eligible gene absent from the immutable two-class direct-exposure registry.",
            "HCA18/broad Atlas, donor-circuit, and earlier R3 train candidate-generation exposures are outside the blacklist.",
            "This is registry-disjoint, not candidate-selection-disjoint or globally gene-fresh.",
            "Path B is tested only after Path A is supported.",
            "Intervals invert the exact decision tests; R3d is not a sealed confirmation.",
        ],
    )
    if git_authorization_identity is not None:
        manifest.git_commit = git_authorization_identity["commit"]
        manifest.git_dirty = False
    manifest.inputs = [
        {"path": path, **record} for path, record in sorted(snapshot.items())
    ]
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
    _verify_snapshot(snapshot)
    if git_authorization_identity is not None:
        final_git_identity = r3c.require_clean_git()
        if final_git_identity != git_authorization_identity:
            raise RuntimeError("R3d git identity changed during the authorized run")
    _atomic_write(output, result_bytes)
    _atomic_write(manifest_path, manifest_bytes)
    if target_attempt is not None:
        write_target_attempt_completion(
            target_attempt,
            status="TARGET_PUBLICATION_COMPLETE",
            details={
                "verdict": verdict,
                "official_path_classifications": {
                    path: official_paths[path]["classification"] for path in PATH_NAMES
                },
                "result": {
                    "path": str(output),
                    "size_bytes": len(result_bytes),
                    "sha256": hashlib.sha256(result_bytes).hexdigest(),
                },
                "manifest": {
                    "path": str(manifest_path),
                    "size_bytes": len(manifest_bytes),
                    "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                },
            },
        )
    print(f"verdict={verdict} -> {output}")


if __name__ == "__main__":
    main()
