#!/usr/bin/env python
"""R5c: conditional L3H1 write and masked-acceptor delivery instrument.

The only behavioral readout in this module is the hard masked-acceptor-G margin.
The scientific execution order is R3c -> R5a -> R5b -> R5c.  At draft status
all literal seal pins are ``None``; consequently model preflight and target both
stop before opening an upstream artifact, checkpoint, annotation, FASTA, cohort,
runtime bundle, preflight, or output.

``--stage audit`` is intentionally model-free.  It binds and checks the component,
ensemble, row, path, boundary, common-leg, and calibration registries.  Once the
literal seals are filled, the same CLI authenticates the upstream chain and the
per-bucket R5b runtime, reconstructs the shared reservation, privately stages the
checkpoint, runs the exact query-local ``c+`` rescue and N/S sandwich, replays the
preflight for target authorization, and writes an immutable result plus manifest.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
from numbers import Integral
from pathlib import Path
import platform
import re
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.circuits.hf_bert import capture_head_outputs, patch_heads  # noqa: E402
from gfmcd.splicing.acceptor_coupling_mechanism import (  # noqa: E402
    delivery_candidate_route_gaps,
    delivery_sandwich_estimands,
    energy_match_rows,
    l3_to_u_mediation_estimands,
    replay_delivery_sandwich_record,
    replay_l3_to_u_mediation_record,
)
from gfmcd.splicing.acceptor_mechanism_cohort import (  # noqa: E402
    R5_EXPECTED_CENSUS_IDENTITY,
    R5_EXPECTED_SOURCE_IDENTITY,
    R5_ROWS_PER_BUCKET,
    R5_VALIDATION_CHROMOSOMES,
    build_r5_reservations,
    validate_expected_r5_census_identity,
    validate_expected_r5_source_identity,
    validate_r5_reservation_audit,
)
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
from gfmcd.splicing.paired_component import (  # noqa: E402
    boundary_sign_flip_tests,
    holm_adjust_named,
    intersection_union_pvalue,
    one_sided_binomial_bounds,
    simultaneous_family_decision,
    strict_interval_decision,
)
from gfmcd.splicing.r5_stack_binding import (  # noqa: E402
    R3C_INSTRUMENT_REVISION,
    R5StackBindingError,
    STACK_BINDING_FIELD,
    STACK_FINGERPRINT_FIELD,
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
    "DRAFT_R5C_ACCEPTOR_WRITE_DELIVERY.md"
)
STAT_CORE = REPO / "src/gfmcd/splicing/paired_component.py"
MECHANISM_CORE = REPO / "src/gfmcd/splicing/acceptor_coupling_mechanism.py"
COHORT_CORE = REPO / "src/gfmcd/splicing/acceptor_mechanism_cohort.py"
HOOK_CORE = REPO / "src/gfmcd/circuits/hf_bert.py"
CIRCUIT_CORE = REPO / "src/gfmcd/splicing/circuit.py"
COUPLING_CORE = REPO / "src/gfmcd/splicing/coupling.py"
REVERSE_COUPLING_CORE = REPO / "src/gfmcd/splicing/reverse_coupling.py"
ADAPTER_CORE = REPO / "src/gfmcd/adapters/splicebert.py"
PROVENANCE_CORE = REPO / "src/gfmcd/provenance.py"

# Literal non-wildcard seals.  Target authorization checks all of these before it
# even resolves a user-supplied path.  A development artifact may never fill them.
FROZEN_PROTOCOL_SHA256: str | None = None
FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = None
FROZEN_PREFLIGHT_ARTIFACT_SHA256: str | None = None
FROZEN_R3C_ENTRY_ARTIFACT_SHA256: str | None = None
FROZEN_R5A_ENTRY_ARTIFACT_SHA256: str | None = None
FROZEN_R5B_ENTRY_ARTIFACT_SHA256: str | None = None

SCHEMA = "gfmcd.r5c.acceptor_write_delivery/1"
AUDIT_SCHEMA = "gfmcd.r5c.model_free_registry_audit/1"
SESOI = 0.05
STAGE_ALPHA = 0.05
FAMILY_ALPHA = STAGE_ALPHA / 5
PRODUCTION_RANDOMISATION = 4000
PRODUCTION_SEED = 0
PRODUCTION_BATCH = 8
MIN_GENES = R5_ROWS_PER_BUCKET
CALIBRATION_EPSILON = 0.01
CALIBRATION_POWER_FLOOR = 0.90
CALIBRATION_PLANT_SUCCESS_RATE = 0.95

L3_READER = (3, 1)
DIRECT_NEGATIVE = (0, 2)
ACCEPTOR_CANDIDATES = ((4, 11), (4, 9), (4, 7))
DONOR_ROUTE_COMPARATORS = ((4, 0), (4, 2), (5, 11))
PRIMARY_ACCEPTOR_PAIR = ((4, 11), (4, 9))
SECONDARY_ACCEPTOR_TRIPLE = ACCEPTOR_CANDIDATES
DONOR_ROUTE_PAIRS = (
    ((4, 0), (4, 2)),
    ((4, 0), (5, 11)),
    ((4, 2), (5, 11)),
)
DONOR_ROUTE_TRIPLE = DONOR_ROUTE_COMPARATORS
REGISTERED_DOWNSTREAM_HEADS = frozenset(
    ACCEPTOR_CANDIDATES + DONOR_ROUTE_COMPARATORS
)

ENDPOINTS = ("partner", "pseudo", "specific")
SANDWICH_ESTIMANDS = ("N", "S")
SANDWICH_ROWS = (
    "N_partner",
    "S_partner",
    "N_pseudo",
    "S_pseudo",
    "N_specific",
    "S_specific",
)
FAMILY_NAMES = ("C1", "C2", "C3", "C4", "C5")
EXPECTED_FAMILY_SIZES = {"C1": 6, "C2": 134, "C3": 54, "C4": 158, "C5": 142}
R5B_RUNTIME_CONTRACT_SCHEMA = "gfmcd.r5b.r5c_runtime_contract/1"
R5_ENTRY_CONTRACT_SCHEMA = "gfmcd.r5.entry_contract/1"


class TargetSealError(RuntimeError):
    """Raised before any target-supplied path can be inspected."""


class InstrumentError(RuntimeError):
    """A failed hook, identity, health, provenance, or calibration invariant."""


class ScientificStop(RuntimeError):
    """A healthy but negative prerequisite outcome that validly closes R5c."""


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if not np.isfinite(array).all():
        raise ValueError("cannot hash a non-finite numeric array")
    shape = ",".join(map(str, array.shape))
    digest = hashlib.sha256()
    digest.update(f"float64|{shape}|".encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _r5b_numeric_array_sha256(value: Any, *, dtype: str) -> str:
    """Reproduce the public R5b runtime-contract numeric digest."""

    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if not np.isfinite(array).all():
        raise InstrumentError("R5b runtime array contains non-finite values")
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.ndim}|".encode("ascii"))
    digest.update(",".join(str(int(size)) for size in array.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _r5b_ordered_text_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256(b"gfmcd.r5b.ordered-text.v1\0")
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def r5b_runtime_array_sha256(name: str, value: Any) -> str:
    array = np.asarray(value)
    if name == "ordered_row_ids":
        return _r5b_ordered_text_sha256([str(item) for item in array.tolist()])
    if np.issubdtype(array.dtype, np.integer):
        return _r5b_numeric_array_sha256(array, dtype="<i8")
    return _r5b_numeric_array_sha256(array, dtype="<f8")


def head_name(head: tuple[int, int]) -> str:
    layer, index = head
    if not _is_integer(layer) or not _is_integer(index):
        raise TypeError("head coordinates must be integers, not booleans")
    if int(layer) < 0 or int(index) < 0:
        raise ValueError("head coordinates must be non-negative")
    return f"L{int(layer)}H{int(index)}"


def ensemble_name(heads: Sequence[tuple[int, int]]) -> str:
    normalized = tuple(heads)
    if len(normalized) not in (1, 2, 3) or len(set(normalized)) != len(normalized):
        raise ValueError("an ensemble must contain one, two, or three unique heads")
    return "+".join(head_name(head) for head in normalized)


def _hashed_ensemble_order(namespace: str, heads: Sequence[tuple[int, int]]) -> tuple[str, str]:
    identity = ensemble_name(heads)
    return hashlib.sha256(f"{namespace}|{identity}".encode()).hexdigest(), identity


def random_ensemble_registry() -> dict[str, tuple[tuple[tuple[int, int], ...], ...]]:
    """Return the frozen deterministic matched-layer random pairs and triples."""

    l4 = tuple(
        (4, head) for head in range(16) if (4, head) not in REGISTERED_DOWNSTREAM_HEADS
    )
    l5 = tuple(
        (5, head) for head in range(16) if (5, head) not in REGISTERED_DOWNSTREAM_HEADS
    )

    def choose(
        namespace: str,
        candidates: Iterable[tuple[tuple[int, int], ...]],
        count: int,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        unique = tuple(sorted(set(candidates), key=lambda item: _hashed_ensemble_order(namespace, item)))
        if len(unique) < count:
            raise RuntimeError(f"insufficient eligible ensembles for {namespace}")
        return unique[:count]

    import itertools

    pair_l4_l4 = choose("r5c-random-pair-l4-l4-v1", itertools.combinations(l4, 2), 8)
    pair_l4_l5 = choose(
        "r5c-random-pair-l4-l5-v1",
        ((left, right) for left in l4 for right in l5),
        8,
    )
    triple_l4_l4_l4 = choose(
        "r5c-random-triple-l4-l4-l4-v1", itertools.combinations(l4, 3), 8
    )
    triple_l4_l4_l5 = choose(
        "r5c-random-triple-l4-l4-l5-v1",
        ((a, b, c) for a, b in itertools.combinations(l4, 2) for c in l5),
        8,
    )
    return {
        "pairs_l4_l4": pair_l4_l4,
        "pairs_l4_l5": pair_l4_l5,
        "triples_l4_l4_l4": triple_l4_l4_l4,
        "triples_l4_l4_l5": triple_l4_l4_l5,
    }


def random_pairs() -> tuple[tuple[tuple[int, int], ...], ...]:
    registry = random_ensemble_registry()
    return registry["pairs_l4_l4"] + registry["pairs_l4_l5"]


def random_triples() -> tuple[tuple[tuple[int, int], ...], ...]:
    registry = random_ensemble_registry()
    return registry["triples_l4_l4_l4"] + registry["triples_l4_l4_l5"]


def complete_downstream_registry() -> tuple[tuple[tuple[int, int], ...], ...]:
    """All singleton, pair, and triple arms in their fixed family order."""

    return (
        tuple((head,) for head in ACCEPTOR_CANDIDATES + DONOR_ROUTE_COMPARATORS)
        + (PRIMARY_ACCEPTOR_PAIR,)
        + DONOR_ROUTE_PAIRS
        + random_pairs()
        + (SECONDARY_ACCEPTOR_TRIPLE, DONOR_ROUTE_TRIPLE)
        + random_triples()
    )


def _direct_row(component: tuple[int, int], endpoint: str) -> str:
    return f"C1|component={head_name(component)}|estimand=R_L3|endpoint={endpoint}"


def _u_row(estimand: str, endpoint: str) -> str:
    return f"C2|component=U_A|estimand={estimand}|endpoint={endpoint}"


def _u_random_row(draw: int, estimand: str) -> str:
    return f"C2|component=random_projector_{draw:02d}|estimand={estimand}|endpoint=partner"


def _u_gap_row(draw: int, estimand: str) -> str:
    return f"C2|gap=U_A-random_projector_{draw:02d}|estimand={estimand}|endpoint=partner"


def _sandwich_row(family: str, component: str, estimand: str, endpoint: str) -> str:
    return f"{family}|component={component}|estimand={estimand}|endpoint={endpoint}"


def _gap_row(family: str, target: str, control: str, estimand: str) -> str:
    # All delivery gaps use the partner-source sandwich.  This preserves the exact
    # helper-level common-leg identity; pseudo and specificity remain separately tested.
    return f"{family}|gap={target}-{control}|estimand={estimand}|endpoint=partner"


def family_row_names() -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    rows["C1"] = tuple(
        _direct_row(component, endpoint)
        for component in (L3_READER, DIRECT_NEGATIVE)
        for endpoint in ENDPOINTS
    )
    c2 = [
        _u_row(estimand, endpoint)
        for estimand in ("S_U", "N_U")
        for endpoint in ENDPOINTS
    ]
    c2.extend(
        _u_random_row(draw, estimand)
        for draw in range(32)
        for estimand in ("S_U", "N_U")
    )
    c2.extend(
        _u_gap_row(draw, estimand)
        for draw in range(32)
        for estimand in ("S_U", "N_U")
    )
    rows["C2"] = tuple(c2)

    c3 = [
        _sandwich_row("C3", head_name(component), estimand, endpoint)
        for component in ACCEPTOR_CANDIDATES + DONOR_ROUTE_COMPARATORS
        for endpoint in ENDPOINTS
        for estimand in SANDWICH_ESTIMANDS
    ]
    c3.extend(
        _gap_row("C3", head_name(candidate), head_name(route), estimand)
        for candidate in ACCEPTOR_CANDIDATES
        for route in DONOR_ROUTE_COMPARATORS
        for estimand in SANDWICH_ESTIMANDS
    )
    rows["C3"] = tuple(c3)

    pair_components = (PRIMARY_ACCEPTOR_PAIR,) + DONOR_ROUTE_PAIRS + random_pairs()
    c4 = [
        _sandwich_row("C4", ensemble_name(component), estimand, endpoint)
        for component in pair_components
        for endpoint in ENDPOINTS
        for estimand in SANDWICH_ESTIMANDS
    ]
    pair_target = ensemble_name(PRIMARY_ACCEPTOR_PAIR)
    c4.extend(
        _gap_row("C4", pair_target, ensemble_name(control), estimand)
        for control in DONOR_ROUTE_PAIRS + random_pairs()
        for estimand in SANDWICH_ESTIMANDS
    )
    rows["C4"] = tuple(c4)

    triple_components = (
        SECONDARY_ACCEPTOR_TRIPLE,
        DONOR_ROUTE_TRIPLE,
    ) + random_triples()
    c5 = [
        _sandwich_row("C5", ensemble_name(component), estimand, endpoint)
        for component in triple_components
        for endpoint in ENDPOINTS
        for estimand in SANDWICH_ESTIMANDS
    ]
    triple_target = ensemble_name(SECONDARY_ACCEPTOR_TRIPLE)
    c5.extend(
        _gap_row("C5", triple_target, ensemble_name(control), estimand)
        for control in (DONOR_ROUTE_TRIPLE,) + random_triples()
        for estimand in SANDWICH_ESTIMANDS
    )
    rows["C5"] = tuple(c5)
    return rows


@dataclass(frozen=True)
class Requirement:
    row: str
    decision: str

    @property
    def boundary(self) -> float:
        if self.decision == "MATERIAL_POSITIVE":
            return SESOI
        if self.decision == "SUPERIOR_TO_CONTROL":
            return 0.0
        raise ValueError("equivalence has two boundaries")


def _target_sandwich_requirements(family: str, component: str) -> list[Requirement]:
    output: list[Requirement] = []
    for endpoint in ENDPOINTS:
        expected = "EQUIVALENT_WITHIN_SESOI" if endpoint == "pseudo" else "MATERIAL_POSITIVE"
        for estimand in SANDWICH_ESTIMANDS:
            output.append(
                Requirement(
                    _sandwich_row(family, component, estimand, endpoint), expected
                )
            )
    return output


def _equivalent_sandwich_requirements(
    family: str, component: str
) -> list[Requirement]:
    return [
        Requirement(
            _sandwich_row(family, component, estimand, endpoint),
            "EQUIVALENT_WITHIN_SESOI",
        )
        for endpoint in ENDPOINTS
        for estimand in SANDWICH_ESTIMANDS
    ]


def path_registry() -> dict[str, dict[str, tuple[Requirement, ...]]]:
    """Return the exact path IUT/Holm registry, grouped by intervention size."""

    c1 = [
        Requirement(
            _direct_row(L3_READER, endpoint),
            "EQUIVALENT_WITHIN_SESOI" if endpoint == "pseudo" else "MATERIAL_POSITIVE",
        )
        for endpoint in ENDPOINTS
    ]
    c1.extend(
        Requirement(_direct_row(DIRECT_NEGATIVE, endpoint), "EQUIVALENT_WITHIN_SESOI")
        for endpoint in ENDPOINTS
    )

    c2 = [
        Requirement(
            _u_row(estimand, endpoint),
            "EQUIVALENT_WITHIN_SESOI" if endpoint == "pseudo" else "MATERIAL_POSITIVE",
        )
        for estimand in ("S_U", "N_U")
        for endpoint in ENDPOINTS
    ]
    c2.extend(
        Requirement(_u_gap_row(draw, estimand), "SUPERIOR_TO_CONTROL")
        for draw in range(32)
        for estimand in ("S_U", "N_U")
    )

    c3_paths: dict[str, tuple[Requirement, ...]] = {}
    for candidate in ACCEPTOR_CANDIDATES:
        name = head_name(candidate)
        requirements = _target_sandwich_requirements("C3", name)
        for route in DONOR_ROUTE_COMPARATORS:
            requirements.extend(_equivalent_sandwich_requirements("C3", head_name(route)))
            requirements.extend(
                Requirement(
                    _gap_row("C3", name, head_name(route), estimand),
                    "SUPERIOR_TO_CONTROL",
                )
                for estimand in SANDWICH_ESTIMANDS
            )
        c3_paths[f"P_C3_{name}"] = tuple(requirements)

    pair_target = ensemble_name(PRIMARY_ACCEPTOR_PAIR)
    c4 = _target_sandwich_requirements("C4", pair_target)
    for route in DONOR_ROUTE_PAIRS:
        route_name = ensemble_name(route)
        c4.extend(_equivalent_sandwich_requirements("C4", route_name))
    for control in DONOR_ROUTE_PAIRS + random_pairs():
        c4.extend(
            Requirement(
                _gap_row("C4", pair_target, ensemble_name(control), estimand),
                "SUPERIOR_TO_CONTROL",
            )
            for estimand in SANDWICH_ESTIMANDS
        )

    triple_target = ensemble_name(SECONDARY_ACCEPTOR_TRIPLE)
    triple_route = ensemble_name(DONOR_ROUTE_TRIPLE)
    c5 = _target_sandwich_requirements("C5", triple_target)
    c5.extend(_equivalent_sandwich_requirements("C5", triple_route))
    for control in (DONOR_ROUTE_TRIPLE,) + random_triples():
        c5.extend(
            Requirement(
                _gap_row("C5", triple_target, ensemble_name(control), estimand),
                "SUPERIOR_TO_CONTROL",
            )
            for estimand in SANDWICH_ESTIMANDS
        )

    return {
        "C1": {"P_C1_L3_DIRECT": tuple(c1)},
        "C2": {"P_C2_L3_WRITES_U_A": tuple(c2)},
        "C3": c3_paths,
        "C4": {"P_C4_PRIMARY_PAIR": tuple(c4)},
        "C5": {"P_C5_SECONDARY_TRIPLE": tuple(c5)},
    }


def _requirement_edges(requirement: Requirement) -> tuple[tuple[str, float, str], ...]:
    if requirement.decision == "MATERIAL_POSITIVE":
        return (("positive_edge", SESOI, "greater"),)
    if requirement.decision == "SUPERIOR_TO_CONTROL":
        return (("superiority_edge", 0.0, "greater"),)
    if requirement.decision == "EQUIVALENT_WITHIN_SESOI":
        return (
            ("lower_edge", -SESOI, "greater"),
            ("upper_edge", SESOI, "less"),
        )
    raise ValueError(f"unknown decision {requirement.decision}")


def _ci_matches(requirement: Requirement, lo: float, hi: float) -> bool:
    if requirement.decision == "MATERIAL_POSITIVE":
        return lo > SESOI
    if requirement.decision == "SUPERIOR_TO_CONTROL":
        return lo > 0.0
    return strict_interval_decision(lo, hi, SESOI) == "EQUIVALENT_WITHIN_SESOI"


def strict_family_inference(
    family_name: str,
    values: Any,
    groups: Sequence[str],
    *,
    n_draws: int = PRODUCTION_RANDOMISATION,
    seed: int = PRODUCTION_SEED,
    namespace: str = "r5c-observed",
    min_genes: int = MIN_GENES,
    health_ok: bool = True,
) -> dict[str, Any]:
    """Run full-family intervals, arbitrary-boundary IUT paths, and Holm."""

    rows_by_family = family_row_names()
    if family_name not in FAMILY_NAMES:
        raise ValueError("unknown R5c family")
    names = rows_by_family[family_name]
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(names), len(groups)) or not np.isfinite(matrix).all():
        raise ValueError("family values must be finite [exact_rows, examples]")
    if n_draws < 1 or not _is_integer(n_draws):
        raise ValueError("n_draws must be a positive integer")

    intervals, interval_audit = simultaneous_family_decision(
        matrix,
        groups,
        names,
        sesoi=SESOI,
        n_draws=int(n_draws),
        alpha=FAMILY_ALPHA,
        seed=int(seed),
        namespace=f"{namespace}:{family_name}:simultaneous-intervals",
        min_genes=min_genes,
        studentized=True,
    )
    paths = path_registry()[family_name]
    by_row: dict[str, Requirement] = {}
    for requirements in paths.values():
        for requirement in requirements:
            prior = by_row.setdefault(requirement.row, requirement)
            if prior.decision != requirement.decision:
                raise RuntimeError("one family row has contradictory path requirements")

    edge_names: list[str] = []
    edge_rows: list[np.ndarray] = []
    boundaries: list[float] = []
    alternatives: list[str] = []
    row_index = {name: index for index, name in enumerate(names)}
    for row in names:
        requirement = by_row.get(row)
        if requirement is None:
            continue
        for edge, boundary, alternative in _requirement_edges(requirement):
            edge_names.append(f"{row}|{edge}")
            edge_rows.append(matrix[row_index[row]])
            boundaries.append(boundary)
            alternatives.append(alternative)
    tests = boundary_sign_flip_tests(
        np.vstack(edge_rows),
        groups,
        edge_names,
        boundaries=boundaries,
        alternatives=alternatives,
        n_draws=int(n_draws),
        seed=int(seed),
        namespace=f"{namespace}:{family_name}:shared-boundary-edges",
        min_genes=min_genes,
    )
    sign_hashes = {test.sign_matrix_sha256 for test in tests.values()}
    gene_hashes = {test.ordered_gene_labels_sha256 for test in tests.values()}
    if len(sign_hashes) != 1 or len(gene_hashes) != 1:
        raise InstrumentError("decision edges did not share one gene sign matrix")

    row_p: dict[str, float] = {}
    row_tests: dict[str, Any] = {}
    for row, requirement in by_row.items():
        selected = [tests[f"{row}|{edge}"] for edge, _, _ in _requirement_edges(requirement)]
        row_p[row] = intersection_union_pvalue([test.p_value for test in selected])
        row_tests[row] = {
            "expected": requirement.decision,
            "iut_p_value": row_p[row],
            "edges": {test_name.split("|")[-1]: test.to_dict() for test_name, test in zip(
                (f"{row}|{edge}" for edge, _, _ in _requirement_edges(requirement)), selected
            )},
        }
    raw_path_p = {
        path: intersection_union_pvalue([row_p[item.row] for item in requirements])
        for path, requirements in paths.items()
    }
    adjusted = holm_adjust_named(raw_path_p)
    path_results: dict[str, Any] = {}
    for path, requirements in paths.items():
        ci_matches = {}
        for requirement in requirements:
            lo, hi = intervals[requirement.row].simultaneous_ci
            ci_matches[requirement.row] = _ci_matches(requirement, lo, hi)
        p_pass = adjusted[path] < FAMILY_ALPHA
        ci_pass = all(ci_matches.values())
        if not health_ok:
            classification = "INSTRUMENT_UNRESOLVED"
        elif p_pass and ci_pass:
            classification = "SUPPORTED"
        elif p_pass != ci_pass:
            classification = "SCIENTIFICALLY_UNRESOLVED_METHOD_DISAGREEMENT"
        else:
            classification = "NOT_SUPPORTED_AT_FROZEN_THRESHOLDS"
        path_results[path] = {
            "iut_p_value": raw_path_p[path],
            "holm_adjusted_p": adjusted[path],
            "p_pass": p_pass,
            "strict_simultaneous_ci_pass": ci_pass,
            "strict_simultaneous_ci_matches": ci_matches,
            "classification": classification,
        }
    return {
        "family": family_name,
        "alpha": FAMILY_ALPHA,
        "row_names": list(names),
        "row_order_sha256": canonical_sha256(names),
        "n_rows": len(names),
        "intervals": {
            name: {
                "estimate": intervals[name].estimate,
                "simultaneous_ci": list(intervals[name].simultaneous_ci),
                "strict_classification": strict_interval_decision(
                    *intervals[name].simultaneous_ci, SESOI
                ),
            }
            for name in names
        },
        "interval_audit": interval_audit,
        "row_boundary_tests": row_tests,
        "boundary_randomisation_audit": {
            "all_decision_edges_share_one_sign_matrix": True,
            "B": int(n_draws),
            "namespace": next(iter(tests.values())).namespace,
            "sign_matrix_sha256": next(iter(sign_hashes)),
            "ordered_gene_labels_sha256": next(iter(gene_hashes)),
            "edge_order_sha256": canonical_sha256(edge_names),
        },
        "paths": path_results,
    }


def _torch_array_sha256(value: torch.Tensor) -> str:
    if not torch.is_tensor(value):
        raise TypeError("activation digest requires a tensor")
    return array_sha256(value.detach().float().cpu().numpy())


def _boolean_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(f"uint8|{array.ndim}|".encode("ascii"))
    digest.update(",".join(str(int(size)) for size in array.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _condition_capture_identity(capture: "ConditionCapture") -> dict[str, Any]:
    head_hashes = {
        str(layer): _torch_array_sha256(capture.heads[layer])
        for layer in sorted(capture.heads)
    }
    record = {
        "schema": "gfmcd.r5c.condition_capture_identity/1",
        "margin_sha256": array_sha256(capture.margin),
        "query_residual_sha256": array_sha256(capture.query_residual),
        "query_logits_sha256": capture.query_logits_sha256,
        "head_outputs_sha256_by_layer": head_hashes,
        "full_mlm_margin_replay_max_abs_error": float(
            capture.full_mlm_margin_replay_max_abs_error
        ),
        "full_mlm_logits_replay_exact": bool(capture.full_mlm_logits_replay_exact),
    }
    record["capture_sha256"] = canonical_sha256(record)
    return record


def _attention_heads_from_output(output: Any, *, num_heads: int) -> torch.Tensor:
    context = output if torch.is_tensor(output) else output[0]
    if not torch.is_tensor(context) or context.ndim != 3:
        raise InstrumentError("attention audit expected a [batch,sequence,width] tensor")
    batch, sequence, width = context.shape
    if width % int(num_heads):
        raise InstrumentError("attention audit width is not divisible by the head count")
    return (
        context.view(batch, sequence, int(num_heads), width // int(num_heads))
        .permute(0, 2, 1, 3)
        .detach()
        .clone()
    )


def _tensor_part_registry_sha256(parts: Sequence[tuple[str, torch.Tensor]]) -> str:
    return canonical_sha256([[name, _torch_array_sha256(value)] for name, value in parts])


def _head_mutation_audit_record(
    *,
    intervention_id: str,
    role: str,
    component_id: str,
    heads: Sequence[tuple[int, int]],
    token_mask: torch.Tensor,
    source_cache: Mapping[int, torch.Tensor],
    before_by_layer: Mapping[int, torch.Tensor],
    after_by_layer: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    """Summarize the tensors actually seen immediately before and after a patch hook."""

    selected = tuple(sorted(set(heads)))
    if not intervention_id or not role or not component_id or not selected:
        raise InstrumentError("head-mutation audit identity is incomplete")
    if len(selected) != len(tuple(heads)):
        raise InstrumentError("head-mutation audit received duplicate requested heads")
    mask = token_mask.detach().to(dtype=torch.bool, device="cpu")
    if mask.ndim != 2 or not torch.all(mask.sum(dim=1) == 1):
        raise InstrumentError("R5c head patches must request exactly one token per row")
    token_indices = torch.argmax(mask.to(dtype=torch.int64), dim=1).tolist()
    selected_by_layer: dict[int, set[int]] = {}
    for layer, head in selected:
        selected_by_layer.setdefault(int(layer), set()).add(int(head))
    if set(before_by_layer) != set(selected_by_layer) or set(after_by_layer) != set(
        selected_by_layer
    ):
        raise InstrumentError("head-mutation audit did not observe every requested layer")

    selected_query_source: list[tuple[str, torch.Tensor]] = []
    selected_query_after: list[tuple[str, torch.Tensor]] = []
    selected_offmask_before: list[tuple[str, torch.Tensor]] = []
    selected_offmask_after: list[tuple[str, torch.Tensor]] = []
    unselected_before: list[tuple[str, torch.Tensor]] = []
    unselected_after: list[tuple[str, torch.Tensor]] = []
    changed_mask_hashes: dict[str, str] = {}
    changed_counts: dict[str, int] = {}
    changed_indices: dict[str, list[list[int]]] = {}
    off_mask_change_count = 0
    unexpected_head_token_change_count = 0
    replacement_mismatch_count = 0
    mask_device_cache: dict[torch.device, torch.Tensor] = {}

    for layer in sorted(selected_by_layer):
        before = before_by_layer[layer]
        after = after_by_layer[layer]
        source = source_cache[layer].to(device=after.device, dtype=after.dtype)
        if before.shape != after.shape or source.shape != after.shape or before.ndim != 4:
            raise InstrumentError("head-mutation audit tensors are not aligned [B,H,S,D]")
        local_mask = mask_device_cache.setdefault(after.device, mask.to(after.device))
        if tuple(local_mask.shape) != (after.shape[0], after.shape[2]):
            raise InstrumentError("head-mutation audit mask shape differs from activations")
        changed = torch.any(before != after, dim=-1)
        for head in range(after.shape[1]):
            identity = head_name((layer, head))
            head_changed = changed[:, head]
            if head in selected_by_layer[layer]:
                selected_query_source.append((identity, source[:, head][local_mask]))
                selected_query_after.append((identity, after[:, head][local_mask]))
                selected_offmask_before.append((identity, before[:, head][~local_mask]))
                selected_offmask_after.append((identity, after[:, head][~local_mask]))
                off_mask_change_count += int(torch.count_nonzero(head_changed & ~local_mask))
                replacement_mismatch_count += int(
                    torch.count_nonzero(
                        torch.any(after[:, head][local_mask] != source[:, head][local_mask], dim=-1)
                    )
                )
                changed_mask_hashes[identity] = _boolean_array_sha256(
                    head_changed.detach().cpu().numpy()
                )
                changed_counts[identity] = int(torch.count_nonzero(head_changed))
                changed_indices[identity] = [
                    [int(row), int(token)]
                    for row, token in torch.nonzero(
                        head_changed, as_tuple=False
                    ).detach().cpu().tolist()
                ]
            else:
                unselected_before.append((identity, before[:, head]))
                unselected_after.append((identity, after[:, head]))
                unexpected_head_token_change_count += int(torch.count_nonzero(head_changed))

    record: dict[str, Any] = {
        "schema": "gfmcd.r5c.head_mutation_audit/1",
        "intervention_id": intervention_id,
        "role": role,
        "component_id": component_id,
        "requested_heads": [head_name(head) for head in selected],
        "requested_head_count": len(selected),
        "n_rows": int(mask.shape[0]),
        "sequence_tokens": int(mask.shape[1]),
        "requested_token_indices": [int(value) for value in token_indices],
        "requested_token_mask_sha256": _boolean_array_sha256(mask.numpy()),
        "selected_query_source_sha256": _tensor_part_registry_sha256(
            selected_query_source
        ),
        "selected_query_after_sha256": _tensor_part_registry_sha256(
            selected_query_after
        ),
        "selected_offmask_before_sha256": _tensor_part_registry_sha256(
            selected_offmask_before
        ),
        "selected_offmask_after_sha256": _tensor_part_registry_sha256(
            selected_offmask_after
        ),
        "unselected_heads_before_sha256": _tensor_part_registry_sha256(
            unselected_before
        ),
        "unselected_heads_after_sha256": _tensor_part_registry_sha256(
            unselected_after
        ),
        "observed_changed_token_mask_sha256_by_head": changed_mask_hashes,
        "observed_changed_token_count_by_head": changed_counts,
        "observed_changed_token_indices_by_head": changed_indices,
        "off_mask_change_count": off_mask_change_count,
        "unexpected_head_token_change_count": unexpected_head_token_change_count,
        "replacement_mismatch_count": replacement_mismatch_count,
        "before_layer_sha256": {
            str(layer): _torch_array_sha256(before_by_layer[layer])
            for layer in sorted(before_by_layer)
        },
        "after_layer_sha256": {
            str(layer): _torch_array_sha256(after_by_layer[layer])
            for layer in sorted(after_by_layer)
        },
    }
    record["audit_sha256"] = canonical_sha256(record)
    return record


@contextmanager
def audited_patch_heads(
    adapter: Any,
    source_cache: Mapping[int, torch.Tensor],
    heads: Sequence[tuple[int, int]],
    *,
    token_mask: torch.Tensor,
    intervention_id: str,
    role: str,
    component_id: str,
    audit_sink: dict[str, Any],
):
    """Patch heads while independently observing the hook's pre/post tensors."""

    selected = tuple(heads)
    layers = sorted({int(layer) for layer, _ in selected})
    before: dict[int, torch.Tensor] = {}
    after: dict[int, torch.Tensor] = {}
    before_handles = []
    after_handles = []

    def observer(store: dict[int, torch.Tensor], layer: int):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if layer in store:
                raise InstrumentError("a patched attention module executed more than once")
            store[layer] = _attention_heads_from_output(
                output, num_heads=int(adapter.num_heads)
            )

        return hook

    completed = False
    try:
        for layer in layers:
            before_handles.append(
                adapter.attention_self_modules[layer].register_forward_hook(
                    observer(before, layer)
                )
            )
        with patch_heads(adapter, source_cache, selected, token_mask=token_mask):
            for layer in layers:
                after_handles.append(
                    adapter.attention_self_modules[layer].register_forward_hook(
                        observer(after, layer)
                    )
                )
            yield
            completed = True
    finally:
        for handle in after_handles:
            handle.remove()
        for handle in before_handles:
            handle.remove()
    if completed:
        audit_sink.clear()
        audit_sink.update(
            _head_mutation_audit_record(
                intervention_id=intervention_id,
                role=role,
                component_id=component_id,
                heads=selected,
                token_mask=token_mask,
                source_cache=source_cache,
                before_by_layer=before,
                after_by_layer=after,
            )
        )


def common_leg_record(
    leg_id: str,
    m_minus: Any,
    m_c_plus: Any,
    *,
    m_minus_capture_sha256: str | None = None,
    m_c_plus_capture_sha256: str | None = None,
) -> dict[str, str]:
    if not isinstance(leg_id, str) or not leg_id:
        raise ValueError("leg_id must be non-empty")
    left = np.asarray(m_minus, dtype=float)
    right = np.asarray(m_c_plus, dtype=float)
    if left.ndim != 1 or right.shape != left.shape or not np.isfinite(np.vstack((left, right))).all():
        raise ValueError("common legs must be finite aligned one-dimensional arrays")
    left_hash = array_sha256(left)
    right_hash = array_sha256(right)
    minus_capture = m_minus_capture_sha256 or canonical_sha256(
        ["margin-only-capture", leg_id, left_hash]
    )
    plus_capture = m_c_plus_capture_sha256 or canonical_sha256(
        ["margin-only-capture", leg_id, right_hash]
    )
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in (minus_capture, plus_capture)
    ):
        raise ValueError("common-leg capture identities must be SHA-256 digests")
    record = {
        "leg_id": leg_id,
        "m_minus_sha256": left_hash,
        "m_c_plus_sha256": right_hash,
        "m_minus_capture_sha256": minus_capture,
        "m_c_plus_capture_sha256": plus_capture,
    }
    record["common_leg_pair_sha256"] = canonical_sha256(record)
    return record


def validate_common_legs(
    records: Sequence[Mapping[str, Any]],
    *,
    m_minus: Any | None = None,
    m_c_plus: Any | None = None,
    capture_identity: Mapping[str, str] | None = None,
) -> dict[str, str]:
    if not records:
        raise InstrumentError("at least one common-leg record is required")
    required = {
        "leg_id",
        "m_minus_sha256",
        "m_c_plus_sha256",
        "m_minus_capture_sha256",
        "m_c_plus_capture_sha256",
        "common_leg_pair_sha256",
    }
    if any(set(record) != required for record in records):
        raise InstrumentError("common-leg record schema differs")
    first = dict(records[0])
    if any(dict(record) != first for record in records[1:]):
        raise InstrumentError("candidate/comparator sandwich common legs differ")
    if any(not re.fullmatch(r"[0-9a-f]{64}", first[key]) for key in required - {"leg_id"}):
        raise InstrumentError("common-leg hashes are malformed")
    expected_pair = canonical_sha256(
        {key: first[key] for key in required if key != "common_leg_pair_sha256"}
    )
    if first["common_leg_pair_sha256"] != expected_pair:
        raise InstrumentError("common-leg pair digest does not replay")
    if m_minus is not None and first["m_minus_sha256"] != array_sha256(m_minus):
        raise InstrumentError("common-leg m_minus capture differs from primitive cells")
    if m_c_plus is not None and first["m_c_plus_sha256"] != array_sha256(m_c_plus):
        raise InstrumentError("common-leg m_c_plus capture differs from primitive cells")
    if capture_identity is not None:
        expected_capture_keys = {"m_minus_capture_sha256", "m_c_plus_capture_sha256"}
        if set(capture_identity) != expected_capture_keys or any(
            first[key] != capture_identity[key] for key in expected_capture_keys
        ):
            raise InstrumentError("common-leg tensor capture identities differ")
    return first


def endpoint_vectors(partner: Any, pseudo: Any) -> dict[str, np.ndarray]:
    left = np.asarray(partner, dtype=float)
    right = np.asarray(pseudo, dtype=float)
    if left.ndim != 1 or right.shape != left.shape or not np.isfinite(np.vstack((left, right))).all():
        raise ValueError("partner/pseudo vectors must be finite and aligned")
    return {"partner": left, "pseudo": right, "specific": left - right}


def row_project(delta: Any, projectors: Any) -> np.ndarray:
    """Apply one cross-fitted projector or row-orthonormal basis per row."""

    values = np.asarray(delta, dtype=float)
    matrices = np.asarray(projectors, dtype=float)
    if values.ndim != 2 or matrices.ndim != 3 or matrices.shape[0] != values.shape[0]:
        raise ValueError("operators must have shape [rows,width,width] or [rows,rank,width]")
    if not np.isfinite(values).all() or not np.isfinite(matrices).all():
        raise ValueError("delta/projectors must be finite")
    if matrices.shape[1:] == (values.shape[1], values.shape[1]):
        return np.einsum("ni,nij->nj", values, matrices)
    if matrices.shape[2] != values.shape[1] or not 1 <= matrices.shape[1] <= values.shape[1]:
        raise ValueError("basis operators must have shape [rows,rank,width]")
    gram = np.einsum("nkd,nld->nkl", matrices, matrices)
    if not np.allclose(
        gram,
        np.broadcast_to(np.eye(matrices.shape[1]), gram.shape),
        rtol=0.0,
        atol=1e-7,
    ):
        raise ValueError("row bases are not orthonormal")
    coefficients = np.einsum("nd,nkd->nk", values, matrices)
    return np.einsum("nk,nkd->nd", coefficients, matrices)


def validate_crossfit_projectors(
    projectors: Any,
    *,
    selected_rank: int,
    expected_rows: int = R5_ROWS_PER_BUCKET,
    atol: float = 1e-7,
) -> dict[str, Any]:
    matrices = np.asarray(projectors, dtype=float)
    if not _is_integer(selected_rank) or int(selected_rank) < 1:
        raise InstrumentError("selected R5b rank must be a positive integer")
    if (
        not _is_integer(expected_rows)
        or int(expected_rows) < 1
        or matrices.ndim != 3
        or matrices.shape[0] != int(expected_rows)
    ):
        raise InstrumentError("R5c needs one cross-fitted operator per reserved row")
    if matrices.shape[1] == matrices.shape[2]:
        representation = "square_projector"
        symmetry = float(np.max(np.abs(matrices - matrices.swapaxes(1, 2))))
        idempotence = float(np.max(np.abs(matrices @ matrices - matrices)))
        ranks = [int(np.linalg.matrix_rank(matrix, tol=atol)) for matrix in matrices]
    else:
        representation = "row_orthonormal_basis"
        if matrices.shape[1] != int(selected_rank) or matrices.shape[2] < matrices.shape[1]:
            raise InstrumentError("cross-fitted basis shape differs from selected rank")
        gram = np.einsum("nkd,nld->nkl", matrices, matrices)
        idempotence = float(
            np.max(
                np.abs(
                    gram
                    - np.broadcast_to(np.eye(int(selected_rank)), gram.shape)
                )
            )
        )
        symmetry = float(np.max(np.abs(gram - gram.swapaxes(1, 2))))
        ranks = [int(np.linalg.matrix_rank(local_gram, tol=atol)) for local_gram in gram]
    if symmetry > atol or idempotence > atol or set(ranks) != {int(selected_rank)}:
        raise InstrumentError("cross-fitted projectors fail symmetry/idempotence/rank")
    return {
        "n_projectors": matrices.shape[0],
        "selected_rank": int(selected_rank),
        "representation": representation,
        "symmetry_max_abs": symmetry,
        "idempotence_max_abs": idempotence,
        "operator_array_sha256": array_sha256(matrices),
    }


def runtime_projector_capability_audit(runtime: "R5bRuntimeBundle") -> dict[str, Any]:
    """Validate every authenticated selected/random row projector used by R5c."""

    indices = np.arange(R5_ROWS_PER_BUCKET, dtype=np.int64)
    selected_bases = runtime.row_bases(indices)
    selected_audit = validate_crossfit_projectors(
        selected_bases, selected_rank=runtime.selected_rank
    )
    random_bases = runtime.random_row_bases(indices)
    random_audits = []
    for draw, bases in enumerate(random_bases):
        audit = validate_crossfit_projectors(
            bases, selected_rank=runtime.selected_rank
        )
        random_audits.append({"draw": draw, **audit})
    authenticated = dict(runtime.authenticated_runtime_array_sha256)
    required_hashes = {
        "selected_bases_by_fold",
        "random_bases_by_draw_fold",
        "row_fold_ids",
        "ordered_row_ids",
    }
    if set(authenticated) != required_hashes:
        raise InstrumentError("authenticated R5b runtime hash registry differs")
    observed_fold_hashes = {
        "selected_bases_by_fold": r5b_runtime_array_sha256(
            "selected_bases_by_fold", runtime.selected_bases_by_fold
        ),
        "random_bases_by_draw_fold": r5b_runtime_array_sha256(
            "random_bases_by_draw_fold", runtime.random_bases_by_draw_fold
        ),
        "row_fold_ids": r5b_runtime_array_sha256(
            "row_fold_ids", runtime.row_fold_ids
        ),
        "ordered_row_ids": r5b_runtime_array_sha256(
            "ordered_row_ids", np.asarray(runtime.ordered_row_ids)
        ),
    }
    if observed_fold_hashes != authenticated:
        raise InstrumentError("projector audit inputs differ from authenticated R5b arrays")
    return {
        "schema": "gfmcd.r5c.crossfit_projector_capability/1",
        "selected": selected_audit,
        "random": random_audits,
        "authenticated_runtime_array_sha256": authenticated,
        "observed_runtime_array_sha256": observed_fold_hashes,
        "all_projectors_validated": 1 + len(random_audits),
        "audit_sha256": canonical_sha256(
            {"selected": selected_audit, "random": random_audits}
        ),
    }


@dataclass(frozen=True)
class R5bRuntimeBundle:
    selected_rank: int
    selected_bases_by_fold: np.ndarray
    random_bases_by_draw_fold: np.ndarray
    row_fold_ids: np.ndarray
    ordered_row_ids: tuple[str, ...]
    bundle_sha256: str
    authenticated_runtime_array_sha256: Mapping[str, str]

    @property
    def width(self) -> int:
        return int(self.selected_bases_by_fold.shape[-1])

    def row_bases(self, indices: Sequence[int]) -> np.ndarray:
        idx = np.asarray(indices, dtype=int)
        return self.selected_bases_by_fold[self.row_fold_ids[idx]]

    def random_row_bases(self, indices: Sequence[int]) -> tuple[np.ndarray, ...]:
        idx = np.asarray(indices, dtype=int)
        return tuple(
            self.random_bases_by_draw_fold[draw, self.row_fold_ids[idx]]
            for draw in range(32)
        )


def _validate_orthonormal_basis_array(name: str, bases: np.ndarray, rank: int) -> None:
    if bases.shape[-2] != rank:
        raise InstrumentError(f"{name} rank differs from selected R5b rank")
    gram = np.einsum("...kd,...ld->...kl", bases, bases)
    expected = np.broadcast_to(np.eye(rank), gram.shape)
    if not np.allclose(gram, expected, rtol=0.0, atol=1e-7):
        raise InstrumentError(f"{name} rows are not orthonormal")


def load_r5b_runtime_bundle(
    path: str | Path,
    *,
    r5b_record: Mapping[str, Any],
    ordered_row_ids: Sequence[str],
) -> R5bRuntimeBundle:
    """Load the transient fold bases needed to execute a bound R5b state.

    The NPZ is runtime state, not evidence.  Its exact byte hash, array hashes, rank,
    folds, and row order must already be authenticated by the frozen compact R5b result.
    """

    runtime_path = Path(path).expanduser().resolve()
    expected_hash = r5b_record.get("runtime_bundle_sha256")
    try:
        observed_hash = sha256_file(runtime_path)
    except OSError as exc:
        raise InstrumentError(f"R5b transient runtime bundle cannot be read: {exc}") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)) or observed_hash != expected_hash:
        raise InstrumentError("R5b transient runtime bundle hash differs")
    if (
        "runtime_bundle_size_bytes" in r5b_record
        and runtime_path.stat().st_size != r5b_record["runtime_bundle_size_bytes"]
    ):
        raise InstrumentError("R5b transient runtime bundle size differs")
    with np.load(runtime_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "selected_rank",
            "selected_bases_by_fold",
            "random_bases_by_draw_fold",
            "row_fold_ids",
            "ordered_row_ids",
        }:
            raise InstrumentError("R5b runtime NPZ schema differs")
        selected_rank_raw = np.asarray(archive["selected_rank"])
        if selected_rank_raw.shape != () or not _is_integer(selected_rank_raw.item()):
            raise InstrumentError("R5b runtime selected rank is malformed")
        rank = int(selected_rank_raw.item())
        selected = np.asarray(archive["selected_bases_by_fold"], dtype=float)
        random = np.asarray(archive["random_bases_by_draw_fold"], dtype=float)
        fold_ids = np.asarray(archive["row_fold_ids"])
        row_ids = tuple(str(item) for item in np.asarray(archive["ordered_row_ids"]).tolist())
    if rank != r5b_record.get("selected_rank") or rank < 1:
        raise InstrumentError("R5b runtime selected rank differs from compact result")
    if selected.ndim != 3 or selected.shape[0] != 2 or selected.shape[1] != rank:
        raise InstrumentError("selected bases must have shape [2,rank,width]")
    if random.shape != (32, 2, rank, selected.shape[-1]):
        raise InstrumentError("random bases must have shape [32,2,rank,width]")
    if fold_ids.shape != (R5_ROWS_PER_BUCKET,) or not np.issubdtype(fold_ids.dtype, np.integer):
        raise InstrumentError("R5b runtime row folds are malformed")
    fold_ids = fold_ids.astype(np.int64, copy=False)
    if set(fold_ids.tolist()) != {0, 1} or any(np.count_nonzero(fold_ids == fold) != 64 for fold in (0, 1)):
        raise InstrumentError("R5b runtime must have two exact 64-row folds")
    if row_ids != tuple(map(str, ordered_row_ids)):
        raise InstrumentError("R5b runtime row order differs from the R5 reservation")
    _validate_orthonormal_basis_array("selected bases", selected, rank)
    _validate_orthonormal_basis_array("random bases", random, rank)
    expected_arrays = r5b_record.get("runtime_array_sha256", {})
    observed_arrays = {
        "selected_bases_by_fold": r5b_runtime_array_sha256(
            "selected_bases_by_fold", selected
        ),
        "random_bases_by_draw_fold": r5b_runtime_array_sha256(
            "random_bases_by_draw_fold", random
        ),
        "row_fold_ids": r5b_runtime_array_sha256("row_fold_ids", fold_ids),
        "ordered_row_ids": r5b_runtime_array_sha256(
            "ordered_row_ids", np.asarray(row_ids)
        ),
    }
    if expected_arrays != observed_arrays:
        raise InstrumentError("R5b runtime array hashes differ from compact result")
    if (
        "ordered_row_ids_sha256" in r5b_record
        and r5b_record["ordered_row_ids_sha256"]
        != observed_arrays["ordered_row_ids"]
    ):
        raise InstrumentError("R5b ordered-row digest differs from its runtime arrays")
    expected_shapes = r5b_record.get("runtime_array_shapes")
    observed_shapes = {
        "selected_bases_by_fold": list(selected.shape),
        "random_bases_by_draw_fold": list(random.shape),
        "row_fold_ids": list(fold_ids.shape),
        "ordered_row_ids": [len(row_ids)],
    }
    if expected_shapes is not None and expected_shapes != observed_shapes:
        raise InstrumentError("R5b runtime array shapes differ from compact result")
    return R5bRuntimeBundle(
        selected_rank=rank,
        selected_bases_by_fold=selected,
        random_bases_by_draw_fold=random,
        row_fold_ids=fold_ids,
        ordered_row_ids=row_ids,
        bundle_sha256=observed_hash,
        authenticated_runtime_array_sha256=dict(observed_arrays),
    )


def validate_runtime_fold_assignment(
    rows: Sequence[Any], runtime: R5bRuntimeBundle
) -> dict[str, Any]:
    """Recompute the fixed two-fold assignment from the shared cohort genes."""

    if len(rows) != R5_ROWS_PER_BUCKET:
        raise InstrumentError("R5b fold validation requires the exact R5 row count")
    genes = [str(row.example.gene_id) for row in rows]
    if len(set(genes)) != R5_ROWS_PER_BUCKET:
        raise InstrumentError("R5b fold validation requires one row per gene")
    ordered = sorted(
        genes,
        key=lambda gene: (
            hashlib.sha256(f"r5b-u-a-crossfit-v1|{gene}".encode()).hexdigest(),
            gene,
        ),
    )
    fold_by_gene = {gene: index % 2 for index, gene in enumerate(ordered)}
    expected = np.asarray([fold_by_gene[gene] for gene in genes], dtype=np.int64)
    if not np.array_equal(expected, runtime.row_fold_ids):
        raise InstrumentError("R5b runtime fold IDs differ from the frozen gene assignment")
    strands_by_fold = {
        str(fold): sorted(
            {str(row.example.strand) for row, value in zip(rows, expected) if value == fold}
        )
        for fold in (0, 1)
    }
    if any(value != ["+", "-"] for value in strands_by_fold.values()):
        raise InstrumentError("an R5b cross-fit fold does not contain both strands")
    return {
        "algorithm": "alternating sha256(r5b-u-a-crossfit-v1|gene_id) order",
        "fold_counts": {
            str(fold): int(np.count_nonzero(expected == fold)) for fold in (0, 1)
        },
        "both_strands_by_fold": {key: True for key in strands_by_fold},
        "row_fold_ids_sha256": r5b_runtime_array_sha256("row_fold_ids", expected),
        "ordered_gene_assignment_sha256": canonical_sha256(
            [[gene, fold_by_gene[gene]] for gene in ordered]
        ),
        "status": "PASS",
    }


def r5b_runtime_contract_for_bucket(
    record: Mapping[str, Any],
    *,
    bucket: int,
    cohort_binding: Mapping[str, Any],
    entry_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Select and validate the compact R5b -> R5c contract for one bucket.

    The target R5b artifact must carry both contracts: bucket 0 authenticated from
    its literal preflight and bucket 1 from its confirmatory result.  Requiring both
    prevents a target-only set of bases from being silently reused for instrument
    preflight rows with different identities.
    """

    if not _is_integer(bucket) or int(bucket) not in (0, 1):
        raise InstrumentError("R5b runtime contract bucket must be literal 0 or 1")
    contracts = record.get("r5c_runtime_contract_by_bucket")
    if not isinstance(contracts, Mapping) or set(contracts) != {"0", "1"}:
        raise InstrumentError(
            "R5b compact artifact must authenticate distinct bucket-0 and bucket-1 "
            "R5c runtime contracts"
        )
    contract = contracts.get(str(int(bucket)))
    if not isinstance(contract, Mapping):  # pragma: no cover - guarded by exact keys
        raise InstrumentError("R5b runtime contract is missing")
    contract = dict(contract)
    required = {
        "schema",
        "bucket",
        "cohort_sha256",
        "construction_sha256",
        "gene_ids_sha256",
        "checkpoint_sha256",
        "selected_rank",
        "crossfit_map_sha256",
        "runtime_bundle_sha256",
        "runtime_array_sha256",
        STACK_FINGERPRINT_FIELD,
    }
    if not required <= set(contract):
        raise InstrumentError(
            "R5b runtime contract lacks: "
            + ", ".join(sorted(required - set(contract)))
        )
    if contract["schema"] != R5B_RUNTIME_CONTRACT_SCHEMA:
        raise InstrumentError("R5b runtime contract schema differs")
    if contract["bucket"] != int(bucket):
        raise InstrumentError("R5b runtime contract bucket differs")
    expected_binding = {
        "cohort_sha256": cohort_binding.get("ordered_rows_sha256"),
        "construction_sha256": cohort_binding.get("construction_sha256"),
        "gene_ids_sha256": cohort_binding.get("gene_ids_sha256"),
    }
    if any(contract[key] != value for key, value in expected_binding.items()):
        raise InstrumentError("R5b runtime contract differs from the shared reservation")
    if contract["checkpoint_sha256"] != entry_authorization.get("checkpoint_sha256"):
        raise InstrumentError("R5b runtime contract checkpoint differs from the entry chain")
    if contract["selected_rank"] != entry_authorization.get("selected_rank"):
        raise InstrumentError("R5b runtime contract rank differs from the entry chain")
    if contract[STACK_FINGERPRINT_FIELD] != entry_authorization.get(
        STACK_FINGERPRINT_FIELD
    ):
        raise InstrumentError("R5b runtime contract differs from the joint R5 stack")
    if not re.fullmatch(r"[0-9a-f]{64}", str(contract["crossfit_map_sha256"])):
        raise InstrumentError("R5b runtime contract cross-fit map is malformed")
    if int(bucket) == 1 and contract["crossfit_map_sha256"] != entry_authorization.get(
        "crossfit_map_sha256"
    ):
        raise InstrumentError("R5b target runtime cross-fit map differs from its entry contract")
    if not re.fullmatch(r"[0-9a-f]{64}", str(contract["runtime_bundle_sha256"])):
        raise InstrumentError("R5b runtime bundle digest is malformed")
    array_hashes = contract["runtime_array_sha256"]
    expected_arrays = {
        "selected_bases_by_fold",
        "random_bases_by_draw_fold",
        "row_fold_ids",
        "ordered_row_ids",
    }
    if not isinstance(array_hashes, Mapping) or set(array_hashes) != expected_arrays:
        raise InstrumentError("R5b runtime array-hash registry differs")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value))
        for value in array_hashes.values()
    ):
        raise InstrumentError("R5b runtime array digest is malformed")
    return contract


def _masked_query_token_mask(
    readout_positions: Sequence[int], *, sequence_tokens: int, token_offset: int
) -> torch.Tensor:
    if not _is_integer(sequence_tokens) or int(sequence_tokens) < 1:
        raise ValueError("sequence_tokens must be a positive integer")
    if not _is_integer(token_offset):
        raise TypeError("token_offset must be an integer")
    rendered = tuple(readout_positions)
    if not rendered or any(not _is_integer(position) for position in rendered):
        raise TypeError("readout positions must be non-Boolean integers")
    indices = [int(position) + int(token_offset) for position in rendered]
    if any(index < 0 or index >= int(sequence_tokens) for index in indices):
        raise ValueError("a masked-acceptor query lies outside the token sequence")
    mask = torch.zeros((len(indices), int(sequence_tokens)), dtype=torch.bool)
    mask[torch.arange(len(indices)), torch.tensor(indices)] = True
    if not torch.all(mask.sum(dim=1) == 1):  # pragma: no cover - construction identity
        raise RuntimeError("query patch mask does not select exactly one token per row")
    return mask


def _hard_acceptor_g_margin(logits: torch.Tensor, base_ids: Mapping[str, int]) -> torch.Tensor:
    if set(base_ids) != {"A", "C", "G", "T"}:
        raise InstrumentError("base token registry must be exactly A/C/G/T")
    if logits.ndim != 2:
        raise InstrumentError("R5c residual scorer expects [batch,vocabulary] logits")
    g = logits[:, int(base_ids["G"])]
    others = torch.stack(
        [logits[:, int(base_ids[base])] for base in ("A", "C", "T")], dim=-1
    )
    return g - others.max(dim=-1).values


def _score_query_residuals(adapter: Any, residuals: np.ndarray, base_ids: Mapping[str, int]) -> tuple[np.ndarray, str]:
    cls = getattr(adapter.model, "cls", None)
    if cls is None:
        raise InstrumentError("checkpoint lacks the complete model.cls MLM head")
    parameter = next(cls.parameters())
    tensor = torch.from_numpy(np.ascontiguousarray(residuals, dtype=np.float32)).to(
        adapter.device, dtype=parameter.dtype
    )
    with torch.no_grad():
        raw = cls(tensor)
    logits = raw.logits if hasattr(raw, "logits") else raw[0] if isinstance(raw, (tuple, list)) else raw
    if not torch.is_tensor(logits):
        raise InstrumentError("model.cls did not return logits")
    logits = logits.float()
    margin = _hard_acceptor_g_margin(logits, base_ids).detach().cpu().numpy()
    return margin, array_sha256(logits.detach().cpu().numpy())


@dataclass(frozen=True)
class ConditionCapture:
    margin: np.ndarray
    query_residual: np.ndarray
    query_logits_sha256: str
    heads: Mapping[int, torch.Tensor]
    full_mlm_margin_replay_max_abs_error: float
    full_mlm_logits_replay_exact: bool


def _capture_condition(
    adapter: Any,
    task: Any,
    encoded: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
    *,
    contexts: Sequence[Any] = (),
) -> ConditionCapture:
    """Capture final query residual and every head under an exact context stack."""

    holder: dict[str, Any] = {}
    final_hidden: dict[str, torch.Tensor] = {}
    cls = getattr(adapter.model, "cls", None)
    if cls is None:
        raise InstrumentError("checkpoint lacks model.cls")

    def prehook(_module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].ndim != 3:
            raise InstrumentError("model.cls did not receive [batch,sequence,width] hidden states")
        final_hidden["value"] = inputs[0].detach().clone()

    handle = cls.register_forward_pre_hook(prehook)
    try:
        with ExitStack() as stack:
            for context in contexts:
                stack.enter_context(context)

            def run() -> Any:
                holder["output"] = adapter.model(**encoded)
                return holder["output"]

            heads = capture_head_outputs(adapter, run)
    finally:
        handle.remove()
    if "output" not in holder or "value" not in final_hidden:
        raise InstrumentError("condition capture is incomplete")
    output = holder["output"]
    logits = output.logits.float()
    native_margin = task.margin(logits, readout_positions).detach().cpu().numpy()
    token_indices = torch.tensor(
        [int(position) + int(task.off) for position in readout_positions],
        device=final_hidden["value"].device,
    )
    batch_indices = torch.arange(len(readout_positions), device=token_indices.device)
    query_hidden = final_hidden["value"][batch_indices, token_indices]
    query_logits = logits[batch_indices, token_indices].detach().cpu().numpy()
    replay_margin, replay_hash = _score_query_residuals(
        adapter, query_hidden.float().cpu().numpy(), task.bid
    )
    replay_error = float(np.max(np.abs(replay_margin - native_margin)))
    logits_exact = replay_hash == array_sha256(query_logits)
    if replay_error > 1e-5:
        raise InstrumentError("full model.cls query replay does not reproduce acceptor-G margin")
    if not logits_exact:
        raise InstrumentError("full model.cls query replay does not reproduce full logits")
    return ConditionCapture(
        margin=np.asarray(native_margin, dtype=float),
        query_residual=query_hidden.float().cpu().numpy(),
        query_logits_sha256=array_sha256(query_logits),
        heads=heads,
        full_mlm_margin_replay_max_abs_error=replay_error,
        full_mlm_logits_replay_exact=logits_exact,
    )


def execute_conditional_leg(
    adapter: Any,
    task: Any,
    *,
    clean_encoded: Mapping[str, torch.Tensor],
    corrupt_encoded: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
    projectors: Any,
    random_projectors: Sequence[Any],
    downstream_registry: Sequence[Sequence[tuple[int, int]]],
    leg_id: str,
) -> dict[str, Any]:
    """Execute one same-example source leg of the exact R5c interventions.

    This function never constructs or scores a donor readout.  ``c+`` is the corrupt
    recipient with only clean L3H1 at the masked-acceptor query.  Necessity composes
    that patch with ``d <- d-``; sufficiency patches ``d^{c+}`` into the native
    corrupt recipient.  All ensembles therefore share the exact native-corrupt and
    L3-rescued common legs.
    """

    sequence_tokens = int(clean_encoded["input_ids"].shape[1])
    if corrupt_encoded["input_ids"].shape != clean_encoded["input_ids"].shape:
        raise InstrumentError("clean/corrupt token shapes differ")
    query_mask = _masked_query_token_mask(
        readout_positions, sequence_tokens=sequence_tokens, token_offset=int(task.off)
    ).to(adapter.device)
    clean = _capture_condition(adapter, task, clean_encoded, readout_positions)
    minus = _capture_condition(adapter, task, corrupt_encoded, readout_positions)
    captures: list[ConditionCapture] = [clean, minus]
    mutation_audits: list[dict[str, Any]] = []
    chunk_identity = canonical_sha256(
        {
            "leg_id": leg_id,
            "readout_positions": [int(value) for value in readout_positions],
            "corrupt_input_ids_sha256": array_sha256(
                corrupt_encoded["input_ids"].detach().cpu().numpy()
            ),
        }
    )
    expected_downstream = complete_downstream_registry()
    supplied_downstream = tuple(tuple(heads) for heads in downstream_registry)
    if supplied_downstream != expected_downstream:
        raise InstrumentError("downstream execution registry is incomplete or reordered")
    if len(random_projectors) != 32:
        raise InstrumentError("R5c requires exactly 32 selected-rank random projectors")
    registered_heads = tuple(
        sorted(
            {L3_READER, DIRECT_NEGATIVE}
            | {head for ensemble in expected_downstream for head in ensemble}
        )
    )
    clean_self_audit: dict[str, Any] = {}
    clean_self = _capture_condition(
        adapter,
        task,
        clean_encoded,
        readout_positions,
        contexts=(
            audited_patch_heads(
                adapter,
                clean.heads,
                registered_heads,
                token_mask=query_mask,
                intervention_id=f"{leg_id}:{chunk_identity}:clean_self_patch",
                role="registered_clean_self_patch",
                component_id="ALL_REGISTERED_HEADS",
                audit_sink=clean_self_audit,
            ),
        ),
    )
    minus_self_audit: dict[str, Any] = {}
    minus_self = _capture_condition(
        adapter,
        task,
        corrupt_encoded,
        readout_positions,
        contexts=(
            audited_patch_heads(
                adapter,
                minus.heads,
                registered_heads,
                token_mask=query_mask,
                intervention_id=f"{leg_id}:{chunk_identity}:minus_self_patch",
                role="registered_minus_self_patch",
                component_id="ALL_REGISTERED_HEADS",
                audit_sink=minus_self_audit,
            ),
        ),
    )
    captures.extend((clean_self, minus_self))
    mutation_audits.extend((clean_self_audit, minus_self_audit))
    self_patch_margin_error = max(
        float(np.max(np.abs(clean_self.margin - clean.margin))),
        float(np.max(np.abs(minus_self.margin - minus.margin))),
    )
    self_patch_logits_exact = bool(
        clean_self.query_logits_sha256 == clean.query_logits_sha256
        and minus_self.query_logits_sha256 == minus.query_logits_sha256
    )
    l3_audit: dict[str, Any] = {}
    l3_context = audited_patch_heads(
        adapter,
        clean.heads,
        [L3_READER],
        token_mask=query_mask,
        intervention_id=f"{leg_id}:{chunk_identity}:c_plus",
        role="l3_c_plus",
        component_id=head_name(L3_READER),
        audit_sink=l3_audit,
    )
    c_plus = _capture_condition(
        adapter,
        task,
        corrupt_encoded,
        readout_positions,
        contexts=(l3_context,),
    )
    mutation_audits.append(l3_audit)
    captures.append(c_plus)
    direct_negative_audit: dict[str, Any] = {}
    direct_negative = _capture_condition(
        adapter,
        task,
        corrupt_encoded,
        readout_positions,
        contexts=(
            audited_patch_heads(
                adapter,
                clean.heads,
                [DIRECT_NEGATIVE],
                token_mask=query_mask,
                intervention_id=f"{leg_id}:{chunk_identity}:direct_negative",
                role="direct_negative",
                component_id=head_name(DIRECT_NEGATIVE),
                audit_sink=direct_negative_audit,
            ),
        ),
    )
    mutation_audits.append(direct_negative_audit)
    captures.append(direct_negative)

    delta = c_plus.query_residual - minus.query_residual
    projected = row_project(delta, projectors)
    m_sufficient, suff_logits = _score_query_residuals(
        adapter, minus.query_residual + projected, task.bid
    )
    m_without, without_logits = _score_query_residuals(
        adapter, c_plus.query_residual - projected, task.bid
    )
    u_estimands = l3_to_u_mediation_estimands(
        m_minus=minus.margin,
        m_l3_plus=c_plus.margin,
        m_minus_with_u_from_l3_plus=m_sufficient,
        m_l3_plus_with_u_from_minus=m_without,
        leg_id=leg_id,
    )
    random_u: list[dict[str, Any]] = []
    for draw, random_set in enumerate(random_projectors):
        raw_random_projected = row_project(delta, random_set)
        random_projected = energy_match_rows(projected, raw_random_projected)
        energy_error = float(
            np.max(
                np.abs(
                    np.linalg.norm(random_projected, axis=1)
                    - np.linalg.norm(projected, axis=1)
                )
            )
        )
        random_sufficient, random_suff_logits = _score_query_residuals(
            adapter, minus.query_residual + random_projected, task.bid
        )
        random_without, random_without_logits = _score_query_residuals(
            adapter, c_plus.query_residual - random_projected, task.bid
        )
        random_record = l3_to_u_mediation_estimands(
            m_minus=minus.margin,
            m_l3_plus=c_plus.margin,
            m_minus_with_u_from_l3_plus=random_sufficient,
            m_l3_plus_with_u_from_minus=random_without,
            leg_id=leg_id,
        )
        random_record.update(
            {
                "draw": draw,
                "projector_sha256": array_sha256(random_set),
                "per_row_energy_match_max_abs_error": energy_error,
                "redrawn": False,
                "sufficiency_logits_sha256": random_suff_logits,
                "necessity_logits_sha256": random_without_logits,
            }
        )
        random_u.append(random_record)
    minus_capture_identity = _condition_capture_identity(minus)
    c_plus_capture_identity = _condition_capture_identity(c_plus)
    common_capture_identity = {
        "m_minus_capture_sha256": minus_capture_identity["capture_sha256"],
        "m_c_plus_capture_sha256": c_plus_capture_identity["capture_sha256"],
    }
    common = common_leg_record(
        leg_id,
        minus.margin,
        c_plus.margin,
        **common_capture_identity,
    )

    sandwich: dict[str, Any] = {}
    common_records: list[Mapping[str, Any]] = []
    for heads in downstream_registry:
        component_id = ensemble_name(heads)
        necessity_l3_audit: dict[str, Any] = {}
        necessity_delivery_audit: dict[str, Any] = {}
        necessity_contexts = (
            audited_patch_heads(
                adapter,
                clean.heads,
                [L3_READER],
                token_mask=query_mask,
                intervention_id=(
                    f"{leg_id}:{chunk_identity}:{component_id}:necessity_l3"
                ),
                role="sandwich_necessity_l3",
                component_id=component_id,
                audit_sink=necessity_l3_audit,
            ),
            audited_patch_heads(
                adapter,
                minus.heads,
                tuple(heads),
                token_mask=query_mask,
                intervention_id=(
                    f"{leg_id}:{chunk_identity}:{component_id}:necessity_delivery"
                ),
                role="sandwich_necessity_delivery",
                component_id=component_id,
                audit_sink=necessity_delivery_audit,
            ),
        )
        necessity = _capture_condition(
            adapter,
            task,
            corrupt_encoded,
            readout_positions,
            contexts=necessity_contexts,
        )
        sufficiency_delivery_audit: dict[str, Any] = {}
        sufficiency = _capture_condition(
            adapter,
            task,
            corrupt_encoded,
            readout_positions,
            contexts=(
                audited_patch_heads(
                    adapter,
                    c_plus.heads,
                    tuple(heads),
                    token_mask=query_mask,
                    intervention_id=(
                        f"{leg_id}:{chunk_identity}:{component_id}:sufficiency_delivery"
                    ),
                    role="sandwich_sufficiency_delivery",
                    component_id=component_id,
                    audit_sink=sufficiency_delivery_audit,
                ),
            ),
        )
        captures.extend((necessity, sufficiency))
        mutation_audits.extend(
            (
                necessity_l3_audit,
                necessity_delivery_audit,
                sufficiency_delivery_audit,
            )
        )
        record = delivery_sandwich_estimands(
            m_minus=minus.margin,
            m_c_plus=c_plus.margin,
            m_c_plus_with_d_minus=necessity.margin,
            m_minus_with_d_from_c_plus=sufficiency.margin,
            leg_id=leg_id,
            component_id=component_id,
        )
        record["common_leg"] = dict(common)
        record["mutation_audit_sha256"] = {
            "necessity_l3": necessity_l3_audit["audit_sha256"],
            "necessity_delivery": necessity_delivery_audit["audit_sha256"],
            "sufficiency_delivery": sufficiency_delivery_audit["audit_sha256"],
        }
        record["condition_capture_sha256"] = {
            "necessity": _condition_capture_identity(necessity)["capture_sha256"],
            "sufficiency": _condition_capture_identity(sufficiency)["capture_sha256"],
        }
        sandwich[component_id] = record
        common_records.append(record["common_leg"])
    validate_common_legs(
        common_records,
        m_minus=minus.margin,
        m_c_plus=c_plus.margin,
        capture_identity=common_capture_identity,
    )
    replay_records = [_condition_capture_identity(capture) for capture in captures]
    replay_error = max(
        float(record["full_mlm_margin_replay_max_abs_error"])
        for record in replay_records
    )
    return {
        "leg_id": leg_id,
        "readout": "single masked acceptor G hard margin",
        "masked_donor_endpoint_constructed": False,
        "query_patch_tokens_per_row": 1,
        "hook_capability": {
            "registered_heads": [head_name(head) for head in registered_heads],
            "registered_query_self_patch_margin_max_abs_error": self_patch_margin_error,
            "registered_query_self_patch_full_logits_exact": self_patch_logits_exact,
            "l3_query_self_patch_included": head_name(L3_READER)
            in {head_name(head) for head in registered_heads},
            "candidate_outcomes_used": False,
            "dense_full_residual_replay": {
                "condition_count": len(replay_records),
                "margin_max_abs_error": replay_error,
                "full_logits_exact": all(
                    record["full_mlm_logits_replay_exact"]
                    for record in replay_records
                ),
                "condition_capture_registry_sha256": canonical_sha256(
                    [record["capture_sha256"] for record in replay_records]
                ),
            },
            "condition_capture_records": replay_records,
            "mutation_audits": mutation_audits,
            "expected_query_token_indices": [
                int(position) + int(task.off) for position in readout_positions
            ],
        },
        "common_leg": common,
        "conditions": {
            "clean_margin": clean.margin,
            "m_minus_margin": minus.margin,
            "m_c_plus_margin": c_plus.margin,
            "clean_margin_sha256": array_sha256(clean.margin),
            "m_minus_margin_sha256": array_sha256(minus.margin),
            "m_c_plus_margin_sha256": array_sha256(c_plus.margin),
            "m_minus_query_residual_sha256": array_sha256(minus.query_residual),
            "m_c_plus_query_residual_sha256": array_sha256(c_plus.query_residual),
            "u_sufficiency_logits_sha256": suff_logits,
            "u_necessity_logits_sha256": without_logits,
            "clean_capture_identity": _condition_capture_identity(clean),
            "m_minus_capture_identity": minus_capture_identity,
            "m_c_plus_capture_identity": c_plus_capture_identity,
        },
        "u_a": u_estimands,
        "direct_negative": {
            "component_id": head_name(DIRECT_NEGATIVE),
            "cells": {
                "m_minus": minus.margin,
                "m_direct_negative": direct_negative.margin,
            },
            "estimands": {
                "direct_negative_rescue": direct_negative.margin - minus.margin,
            },
            "margin_sha256": array_sha256(direct_negative.margin),
        },
        "random_u_a": random_u,
        "sandwich": sandwich,
    }


def _replay_l3_envelope(record: Mapping[str, Any]) -> dict[str, Any]:
    core_keys = {"leg_id", "cells", "estimands"}
    allowed_metadata = {
        "draw",
        "projector_sha256",
        "per_row_energy_match_max_abs_error",
        "redrawn",
        "sufficiency_logits_sha256",
        "necessity_logits_sha256",
    }
    if not isinstance(record, Mapping) or not core_keys.issubset(record) or not set(
        record
    ).issubset(core_keys | allowed_metadata):
        raise InstrumentError("L3-to-U primitive envelope schema differs")
    try:
        return replay_l3_to_u_mediation_record(
            {key: record[key] for key in ("leg_id", "cells", "estimands")}
        )
    except (TypeError, ValueError) as exc:
        raise InstrumentError(f"L3-to-U primitive replay failed: {exc}") from exc


def _replay_delivery_envelope(record: Mapping[str, Any]) -> dict[str, Any]:
    core_keys = {"leg_id", "component_id", "cells", "estimands"}
    allowed_metadata = {
        "common_leg",
        "mutation_audit_sha256",
        "mutation_audit_sha256_by_chunk",
        "condition_capture_sha256",
    }
    if not isinstance(record, Mapping) or not core_keys.issubset(record) or not set(
        record
    ).issubset(core_keys | allowed_metadata):
        raise InstrumentError("delivery primitive envelope schema differs")
    try:
        return replay_delivery_sandwich_record(
            {
                key: record[key]
                for key in ("leg_id", "component_id", "cells", "estimands")
            }
        )
    except (TypeError, ValueError) as exc:
        raise InstrumentError(f"delivery primitive replay failed: {exc}") from exc


def _replay_direct_negative(record: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {"component_id", "cells", "estimands", "margin_sha256"}
    if not isinstance(record, Mapping) or set(record) != allowed:
        raise InstrumentError("direct-negative primitive schema differs")
    if record.get("component_id") != head_name(DIRECT_NEGATIVE):
        raise InstrumentError("direct-negative component identity differs")
    cells = record.get("cells")
    estimands = record.get("estimands")
    if not isinstance(cells, Mapping) or set(cells) != {
        "m_minus",
        "m_direct_negative",
    }:
        raise InstrumentError("direct-negative primitive cells differ")
    if not isinstance(estimands, Mapping) or set(estimands) != {
        "direct_negative_rescue"
    }:
        raise InstrumentError("direct-negative estimand schema differs")
    m_minus = np.asarray(cells["m_minus"], dtype=float)
    m_direct = np.asarray(cells["m_direct_negative"], dtype=float)
    stored = np.asarray(estimands["direct_negative_rescue"], dtype=float)
    if (
        m_minus.ndim != 1
        or m_direct.shape != m_minus.shape
        or stored.shape != m_minus.shape
        or not np.isfinite(np.vstack((m_minus, m_direct, stored))).all()
        or not np.array_equal(stored, m_direct - m_minus)
    ):
        raise InstrumentError("direct-negative estimand does not replay from primitive cells")
    if record.get("margin_sha256") != array_sha256(m_direct):
        raise InstrumentError("direct-negative margin digest differs from primitive cells")
    return {
        "component_id": head_name(DIRECT_NEGATIVE),
        "cells": {"m_minus": m_minus, "m_direct_negative": m_direct},
        "estimands": {"direct_negative_rescue": m_direct - m_minus},
        "margin_sha256": array_sha256(m_direct),
    }


def _leg_estimand(leg: Mapping[str, Any], component: str, estimand: str) -> np.ndarray:
    replayed = _replay_delivery_envelope(leg["sandwich"][component])
    return np.asarray(replayed["estimands"][f"{estimand}_d"], dtype=float)


def derive_family_matrices(
    partner_leg: Mapping[str, Any], pseudo_leg: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """Derive all five families from the two raw source legs.

    Specificity and candidate-control gaps are always regenerated here.  Neither a
    caller-provided specific row nor a caller-provided gap can enter inference.
    """

    if partner_leg.get("leg_id") != "partner" or pseudo_leg.get("leg_id") != "pseudo":
        raise InstrumentError("R5c needs literal partner and pseudo source legs")
    if any(
        leg.get("masked_donor_endpoint_constructed") is not False
        or leg.get("readout") != "single masked acceptor G hard margin"
        for leg in (partner_leg, pseudo_leg)
    ):
        raise InstrumentError("R5c source legs are not acceptor-G-only")
    expected_components = {ensemble_name(item) for item in complete_downstream_registry()}
    replayed_u: dict[str, dict[str, Any]] = {}
    replayed_random: dict[str, list[dict[str, Any]]] = {}
    replayed_delivery: dict[str, dict[str, dict[str, Any]]] = {}
    replayed_direct_negative: dict[str, dict[str, Any]] = {}
    for leg in (partner_leg, pseudo_leg):
        if set(leg.get("sandwich", {})) != expected_components:
            raise InstrumentError("source leg lacks the complete downstream registry")
        source_id = str(leg["leg_id"])
        canonical_delivery = {
            component: _replay_delivery_envelope(record)
            for component, record in leg["sandwich"].items()
        }
        common = validate_common_legs(
            [record["common_leg"] for record in leg["sandwich"].values()],
            m_minus=next(iter(canonical_delivery.values()))["cells"]["m_minus"],
            m_c_plus=next(iter(canonical_delivery.values()))["cells"]["m_c_plus"],
        )
        if "common_leg" in leg and dict(leg["common_leg"]) != common:
            raise InstrumentError("source-leg common capture differs from its sandwiches")
        if len(leg.get("random_u_a", ())) != 32:
            raise InstrumentError("source leg lacks the exact random-projector registry")
        replayed_u[source_id] = _replay_l3_envelope(leg["u_a"])
        replayed_random[source_id] = [
            _replay_l3_envelope(record) for record in leg["random_u_a"]
        ]
        replayed_delivery[source_id] = canonical_delivery
        replayed_direct_negative[source_id] = _replay_direct_negative(
            leg["direct_negative"]
        )

    output: dict[str, np.ndarray] = {}

    def add(rows: dict[str, np.ndarray], name: str, value: Any) -> None:
        vector = np.asarray(value, dtype=float)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise InstrumentError(f"derived row {name} is not a finite vector")
        if rows and vector.shape != next(iter(rows.values())).shape:
            raise InstrumentError("derived family rows are not aligned")
        if name in rows:
            raise InstrumentError(f"duplicate derived row {name}")
        rows[name] = vector

    # C1: direct query-local L3H1 rescue and the one-head L0H2 control.
    c1: dict[str, np.ndarray] = {}
    direct_source = {
        head_name(L3_READER): (
            np.asarray(replayed_u["partner"]["estimands"]["direct_l3_rescue"], dtype=float),
            np.asarray(replayed_u["pseudo"]["estimands"]["direct_l3_rescue"], dtype=float),
        ),
        head_name(DIRECT_NEGATIVE): (
            np.asarray(
                replayed_direct_negative["partner"]["estimands"][
                    "direct_negative_rescue"
                ],
                dtype=float,
            ),
            np.asarray(
                replayed_direct_negative["pseudo"]["estimands"][
                    "direct_negative_rescue"
                ],
                dtype=float,
            ),
        ),
    }
    for component in (L3_READER, DIRECT_NEGATIVE):
        vectors = endpoint_vectors(*direct_source[head_name(component)])
        for endpoint in ENDPOINTS:
            add(c1, _direct_row(component, endpoint), vectors[endpoint])

    # C2: selected U_A, 32 random selected-rank/energy controls, and target gaps.
    c2: dict[str, np.ndarray] = {}
    for estimand in ("S_U", "N_U"):
        target = endpoint_vectors(
            replayed_u["partner"]["estimands"][estimand],
            replayed_u["pseudo"]["estimands"][estimand],
        )
        for endpoint in ENDPOINTS:
            add(c2, _u_row(estimand, endpoint), target[endpoint])
    random_partner_by_draw: dict[tuple[int, str], np.ndarray] = {}
    for draw in range(32):
        p_record = partner_leg["random_u_a"][draw]
        q_record = pseudo_leg["random_u_a"][draw]
        if p_record.get("draw") != draw or q_record.get("draw") != draw:
            raise InstrumentError("random-projector order differs across source legs")
        if p_record.get("projector_sha256") != q_record.get("projector_sha256"):
            raise InstrumentError("random-projector identity differs across source legs")
        for estimand in ("S_U", "N_U"):
            random_partner = np.asarray(
                replayed_random["partner"][draw]["estimands"][estimand], dtype=float
            )
            add(c2, _u_random_row(draw, estimand), random_partner)
            random_partner_by_draw[(draw, estimand)] = random_partner
    for draw in range(32):
        for estimand in ("S_U", "N_U"):
            target_partner = np.asarray(
                replayed_u["partner"]["estimands"][estimand], dtype=float
            )
            add(
                c2,
                _u_gap_row(draw, estimand),
                target_partner - random_partner_by_draw[(draw, estimand)],
            )

    def build_delivery_family(
        family: str,
        components: Sequence[Sequence[tuple[int, int]]],
        target: Sequence[tuple[int, int]],
        controls: Sequence[Sequence[tuple[int, int]]],
    ) -> dict[str, np.ndarray]:
        rows: dict[str, np.ndarray] = {}
        for component in components:
            identity = ensemble_name(component)
            vectors_by_estimand = {
                estimand: endpoint_vectors(
                    _leg_estimand(partner_leg, identity, estimand),
                    _leg_estimand(pseudo_leg, identity, estimand),
                )
                for estimand in SANDWICH_ESTIMANDS
            }
            for endpoint in ENDPOINTS:
                for estimand in SANDWICH_ESTIMANDS:
                    vectors = vectors_by_estimand[estimand]
                    add(
                        rows,
                        _sandwich_row(family, identity, estimand, endpoint),
                        vectors[endpoint],
                    )
        target_name = ensemble_name(target)
        for control in controls:
            control_name = ensemble_name(control)
            # This helper independently asserts exact raw m-/m(c+) common legs.
            gaps = delivery_candidate_route_gaps(
                replayed_delivery["partner"][target_name],
                replayed_delivery["partner"][control_name],
            )
            for estimand in SANDWICH_ESTIMANDS:
                add(
                    rows,
                    _gap_row(family, target_name, control_name, estimand),
                    gaps[f"{estimand}_candidate_minus_donor_route"],
                )
        return rows

    c3_components = tuple((head,) for head in ACCEPTOR_CANDIDATES + DONOR_ROUTE_COMPARATORS)
    # C3 contains all 3x3 candidate/comparator gaps, so construct it explicitly.
    c3: dict[str, np.ndarray] = {}
    for component in c3_components:
        identity = ensemble_name(component)
        vectors_by_estimand = {
            estimand: endpoint_vectors(
                _leg_estimand(partner_leg, identity, estimand),
                _leg_estimand(pseudo_leg, identity, estimand),
            )
            for estimand in SANDWICH_ESTIMANDS
        }
        for endpoint in ENDPOINTS:
            for estimand in SANDWICH_ESTIMANDS:
                vectors = vectors_by_estimand[estimand]
                add(c3, _sandwich_row("C3", identity, estimand, endpoint), vectors[endpoint])
    for candidate in ACCEPTOR_CANDIDATES:
        for route in DONOR_ROUTE_COMPARATORS:
            candidate_name = head_name(candidate)
            route_name = head_name(route)
            gaps = delivery_candidate_route_gaps(
                replayed_delivery["partner"][candidate_name],
                replayed_delivery["partner"][route_name],
            )
            for estimand in SANDWICH_ESTIMANDS:
                add(
                    c3,
                    _gap_row("C3", candidate_name, route_name, estimand),
                    gaps[f"{estimand}_candidate_minus_donor_route"],
                )

    c4_components = (PRIMARY_ACCEPTOR_PAIR,) + DONOR_ROUTE_PAIRS + random_pairs()
    c4 = build_delivery_family(
        "C4",
        c4_components,
        PRIMARY_ACCEPTOR_PAIR,
        DONOR_ROUTE_PAIRS + random_pairs(),
    )
    c5_components = (SECONDARY_ACCEPTOR_TRIPLE, DONOR_ROUTE_TRIPLE) + random_triples()
    c5 = build_delivery_family(
        "C5",
        c5_components,
        SECONDARY_ACCEPTOR_TRIPLE,
        (DONOR_ROUTE_TRIPLE,) + random_triples(),
    )

    derived = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    expected = family_row_names()
    for family in FAMILY_NAMES:
        if tuple(derived[family]) != expected[family]:
            raise InstrumentError(f"{family} derived row order differs from its registry")
        output[family] = np.vstack([derived[family][name] for name in expected[family]])
    return output


def _concatenate_vectors(values: Sequence[Any], *, label: str) -> np.ndarray:
    arrays = [np.asarray(value, dtype=float) for value in values]
    if not arrays or any(array.ndim != 1 or not np.isfinite(array).all() for array in arrays):
        raise InstrumentError(f"{label} contains an incomplete vector chunk")
    return np.concatenate(arrays)


def concatenate_leg_chunks(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine batch-local captures while rebuilding full common-leg hashes."""

    if not chunks:
        raise InstrumentError("cannot concatenate an empty R5c source leg")
    leg_id = chunks[0].get("leg_id")
    if leg_id not in {"partner", "pseudo"} or any(
        chunk.get("leg_id") != leg_id for chunk in chunks
    ):
        raise InstrumentError("R5c leg chunks are mixed or unlabeled")
    component_ids = tuple(chunks[0].get("sandwich", {}))
    if any(tuple(chunk.get("sandwich", {})) != component_ids for chunk in chunks):
        raise InstrumentError("downstream component order differs across chunks")

    def merge_mediation(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        canonical_records = [_replay_l3_envelope(record) for record in records]
        merged = {
            "leg_id": leg_id,
            "cells": {
                name: _concatenate_vectors(
                    [record["cells"][name] for record in canonical_records],
                    label=f"{leg_id}:{name}",
                )
                for name in canonical_records[0]["cells"]
            },
            "estimands": {
                name: _concatenate_vectors(
                    [record["estimands"][name] for record in canonical_records],
                    label=f"{leg_id}:{name}",
                )
                for name in canonical_records[0]["estimands"]
            },
        }
        return replay_l3_to_u_mediation_record(merged)

    u_a = merge_mediation([chunk["u_a"] for chunk in chunks])
    random_u: list[dict[str, Any]] = []
    for draw in range(32):
        records = [chunk["random_u_a"][draw] for chunk in chunks]
        if any(record.get("draw") != draw for record in records):
            raise InstrumentError("random-projector draw order differs across chunks")
        merged = merge_mediation(records)
        merged.update(
            {
                "draw": draw,
                "projector_sha256": canonical_sha256(
                    [record["projector_sha256"] for record in records]
                ),
                "per_row_energy_match_max_abs_error": max(
                    float(record["per_row_energy_match_max_abs_error"])
                    for record in records
                ),
                "redrawn": any(record.get("redrawn") is not False for record in records),
            }
        )
        random_u.append(merged)

    sandwich: dict[str, Any] = {}
    chunk_common_records = [chunk["common_leg"] for chunk in chunks]
    aggregate_capture_identity = {
        "m_minus_capture_sha256": canonical_sha256(
            [record["m_minus_capture_sha256"] for record in chunk_common_records]
        ),
        "m_c_plus_capture_sha256": canonical_sha256(
            [record["m_c_plus_capture_sha256"] for record in chunk_common_records]
        ),
    }
    for component in component_ids:
        records = [chunk["sandwich"][component] for chunk in chunks]
        canonical_records = [_replay_delivery_envelope(record) for record in records]
        cells = {
            name: _concatenate_vectors(
                [record["cells"][name] for record in canonical_records],
                label=f"{leg_id}:{component}:{name}",
            )
            for name in canonical_records[0]["cells"]
        }
        estimands = {
            name: _concatenate_vectors(
                [record["estimands"][name] for record in canonical_records],
                label=f"{leg_id}:{component}:{name}",
            )
            for name in canonical_records[0]["estimands"]
        }
        primitive = replay_delivery_sandwich_record(
            {
                "leg_id": leg_id,
                "component_id": component,
                "cells": cells,
                "estimands": estimands,
            }
        )
        sandwich[component] = primitive | {
            "leg_id": leg_id,
            "component_id": component,
            "common_leg": common_leg_record(
                leg_id,
                cells["m_minus"],
                cells["m_c_plus"],
                **aggregate_capture_identity,
            ),
            "mutation_audit_sha256_by_chunk": {
                direction: [
                    record["mutation_audit_sha256"][direction]
                    for record in records
                ]
                for direction in (
                    "necessity_l3",
                    "necessity_delivery",
                    "sufficiency_delivery",
                )
            },
            "condition_capture_sha256": {
                direction: canonical_sha256(
                    [record["condition_capture_sha256"][direction] for record in records]
                )
                for direction in ("necessity", "sufficiency")
            },
        }
    common = validate_common_legs(
        [record["common_leg"] for record in sandwich.values()],
        m_minus=next(iter(sandwich.values()))["cells"]["m_minus"],
        m_c_plus=next(iter(sandwich.values()))["cells"]["m_c_plus"],
        capture_identity=aggregate_capture_identity,
    )
    capability_chunks = [chunk["hook_capability"] for chunk in chunks]
    direct_records = [
        _replay_direct_negative(chunk["direct_negative"]) for chunk in chunks
    ]
    direct_cells = {
        name: _concatenate_vectors(
            [record["cells"][name] for record in direct_records],
            label=f"{leg_id}:direct-negative:{name}",
        )
        for name in ("m_minus", "m_direct_negative")
    }
    direct_estimand = (
        direct_cells["m_direct_negative"] - direct_cells["m_minus"]
    )
    replay_chunks = [record["dense_full_residual_replay"] for record in capability_chunks]
    return {
        "leg_id": leg_id,
        "readout": "single masked acceptor G hard margin",
        "masked_donor_endpoint_constructed": False,
        "query_patch_tokens_per_row": 1,
        "common_leg": common,
        "u_a": u_a,
        "direct_negative": {
            "component_id": head_name(DIRECT_NEGATIVE),
            "cells": direct_cells,
            "estimands": {"direct_negative_rescue": direct_estimand},
            "margin_sha256": array_sha256(direct_cells["m_direct_negative"]),
        },
        "random_u_a": random_u,
        "sandwich": sandwich,
        "hook_capability": {
            "registered_heads": (
                list(capability_chunks[0]["registered_heads"])
                if all(
                    record["registered_heads"]
                    == capability_chunks[0]["registered_heads"]
                    for record in capability_chunks
                )
                else []
            ),
            "registered_query_self_patch_margin_max_abs_error": max(
                float(record["registered_query_self_patch_margin_max_abs_error"])
                for record in capability_chunks
            ),
            "registered_query_self_patch_full_logits_exact": all(
                record["registered_query_self_patch_full_logits_exact"]
                for record in capability_chunks
            ),
            "l3_query_self_patch_included": all(
                record["l3_query_self_patch_included"] for record in capability_chunks
            ),
            "candidate_outcomes_used": False,
            "dense_full_residual_replay": {
                "condition_count": sum(
                    int(record["condition_count"]) for record in replay_chunks
                ),
                "margin_max_abs_error": max(
                    float(record["margin_max_abs_error"]) for record in replay_chunks
                ),
                "full_logits_exact": all(
                    record["full_logits_exact"] is True for record in replay_chunks
                ),
                "condition_capture_registry_sha256": canonical_sha256(
                    [
                        record["condition_capture_registry_sha256"]
                        for record in replay_chunks
                    ]
                ),
            },
            "mutation_audit_chunks": [
                record["mutation_audits"] for record in capability_chunks
            ],
            "expected_query_token_indices_by_chunk": [
                list(record["expected_query_token_indices"])
                for record in capability_chunks
            ],
            "condition_capture_records": [
                capture
                for record in capability_chunks
                for capture in record["condition_capture_records"]
            ],
        },
        "chunk_condition_hashes": [chunk["conditions"] for chunk in chunks],
        "native_conditions": {
            name: _concatenate_vectors(
                [chunk["conditions"][name] for chunk in chunks],
                label=f"{leg_id}:{name}",
            )
            for name in ("clean_margin", "m_minus_margin", "m_c_plus_margin")
        },
    }


def _health_probe_positions(row: Any, *, n_probe: int = 2) -> tuple[int, ...]:
    sequence = row.clean_sequence
    protected = (
        set(range(int(row.example.donor_pos) - 4, int(row.example.donor_pos) + 6))
        | set(range(int(row.pseudo_donor) - 4, int(row.pseudo_donor) + 6))
        | set(range(int(row.acceptor_g) - 4, int(row.acceptor_g) + 5))
    )
    eligible = [
        position
        for position, base in enumerate(sequence)
        if base in "ACGT" and 2 <= position < len(sequence) - 2 and position not in protected
    ]
    ordered = sorted(
        eligible,
        key=lambda position: (
            hashlib.sha256(
                f"r5c-health-v1|{row.example.example_id}|{position}".encode()
            ).hexdigest(),
            position,
        ),
    )
    if len(ordered) < n_probe:
        raise InstrumentError("an R5c health row lacks enough non-site nucleotide probes")
    return tuple(ordered[:n_probe])


def run_bounded_hook_health(
    adapter: Any,
    task: Any,
    rows: Sequence[Any],
    *,
    max_examples: int = 8,
    floor: float = 0.50,
) -> dict[str, Any]:
    """Exercise exact head-patch directions on masked non-site nucleotides."""

    selected = tuple(rows[: min(max_examples, len(rows))])
    if not selected:
        raise InstrumentError("R5c health requires at least one row")
    readouts = [int(row.acceptor_g) for row in selected]
    probes = [_health_probe_positions(row) for row in selected]
    clean_sequences = [row.clean_sequence for row in selected]
    sources = {
        "partner": [row.paired_corrupt_sequence for row in selected],
        "pseudo": [row.pseudo_corrupt_sequence for row in selected],
    }
    registry = complete_downstream_registry()
    arm_names = (
        "L3H1_direct",
        "L0H2_direct",
    ) + tuple(
        f"{ensemble_name(ensemble)}|{direction}"
        for ensemble in registry
        for direction in ("necessity", "sufficiency")
    )
    hits = {
        source: {arm: 0 for arm in arm_names} for source in ("partner", "pseudo")
    }
    totals = {
        source: {arm: 0 for arm in arm_names} for source in ("partner", "pseudo")
    }

    def encoded_with_probes(sequences: Sequence[str]) -> Mapping[str, torch.Tensor]:
        encoded = task.encode(sequences, readouts)
        for row_index, positions in enumerate(probes):
            for position in positions:
                encoded["input_ids"][row_index, position + task.off] = task.mask_id
        return encoded

    def score(logits: torch.Tensor, source: str, arm: str) -> None:
        for row_index, (row, positions) in enumerate(zip(selected, probes)):
            for position in positions:
                token_logits = logits[row_index, position + task.off]
                predicted = max(
                    "ACGT", key=lambda base: float(token_logits[task.bid[base]].item())
                )
                hits[source][arm] += int(predicted == row.clean_sequence[position])
                totals[source][arm] += 1

    with torch.no_grad():
        clean_encoded = encoded_with_probes(clean_sequences)
        clean_holder: dict[str, Any] = {}
        clean_cache = capture_head_outputs(
            adapter,
            lambda: clean_holder.setdefault("output", adapter.model(**clean_encoded)),
        )
        query_mask = _masked_query_token_mask(
            readouts,
            sequence_tokens=int(clean_encoded["input_ids"].shape[1]),
            token_offset=int(task.off),
        ).to(adapter.device)
        for source, sequences in sources.items():
            corrupt_encoded = encoded_with_probes(sequences)
            minus_holder: dict[str, Any] = {}
            minus_cache = capture_head_outputs(
                adapter,
                lambda: minus_holder.setdefault(
                    "output", adapter.model(**corrupt_encoded)
                ),
            )
            c_holder: dict[str, Any] = {}
            with patch_heads(adapter, clean_cache, [L3_READER], token_mask=query_mask):
                c_cache = capture_head_outputs(
                    adapter,
                    lambda: c_holder.setdefault(
                        "output", adapter.model(**corrupt_encoded)
                    ),
                )
            score(c_holder["output"].logits.float(), source, "L3H1_direct")
            with patch_heads(
                adapter, clean_cache, [DIRECT_NEGATIVE], token_mask=query_mask
            ):
                negative = adapter.model(**corrupt_encoded).logits.float()
            score(negative, source, "L0H2_direct")
            for ensemble in registry:
                identity = ensemble_name(ensemble)
                with ExitStack() as stack:
                    stack.enter_context(
                        patch_heads(
                            adapter, clean_cache, [L3_READER], token_mask=query_mask
                        )
                    )
                    stack.enter_context(
                        patch_heads(adapter, minus_cache, ensemble, token_mask=query_mask)
                    )
                    necessity = adapter.model(**corrupt_encoded).logits.float()
                score(necessity, source, f"{identity}|necessity")
                with patch_heads(adapter, c_cache, ensemble, token_mask=query_mask):
                    sufficiency = adapter.model(**corrupt_encoded).logits.float()
                score(sufficiency, source, f"{identity}|sufficiency")
    records = {
        source: {
            arm: {
                "hits": hits[source][arm],
                "total": totals[source][arm],
                "accuracy": hits[source][arm] / totals[source][arm],
                "floor": floor,
                "ok": bool(
                    totals[source][arm]
                    and hits[source][arm] / totals[source][arm] >= floor
                ),
            }
            for arm in arm_names
        }
        for source in ("partner", "pseudo")
    }
    return {
        "schema": "gfmcd.r5c.bounded_hook_health/1",
        "examples": len(selected),
        "probes_per_example": 2,
        "candidate_outcomes_used": False,
        "records": records,
        "all_arms_ok": all(
            record["ok"]
            for source_records in records.values()
            for record in source_records.values()
        ),
    }


def validate_entry_chain(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the scientific R3c -> R5a -> R5b entry chain fail closed."""

    expected = {
        "r3c": "HELD_OUT_ACCEPTOR_HEAD_CONFIRMATION_COMPLETE",
        "r5a": "L3H1_DONOR_LOCAL_READER_SUPPORTED",
        "r5b": "ACCEPTOR_SIDE_STATE_SUPPORTED",
    }
    stages: dict[str, Mapping[str, Any]] = {}
    for stage, verdict in expected.items():
        candidate = record.get(stage)
        if not isinstance(candidate, Mapping):
            raise InstrumentError(f"entry chain lacks {stage}")
        stages[stage] = candidate
        if stage in {"r5a", "r5b"} and candidate.get("schema") != R5_ENTRY_CONTRACT_SCHEMA:
            raise InstrumentError(f"{stage} entry-contract schema differs")
        if candidate.get("instrument_gates_pass") is not True:
            raise InstrumentError(f"{stage} instrument gates did not pass")
        if candidate.get("claim_eligible") is not True:
            raise ScientificStop(f"{stage} is not claim eligible")
        if candidate.get("verdict") != verdict:
            raise ScientificStop(f"{stage} scientific prerequisite did not pass")
        if candidate.get("acceptor_G_only") is not True:
            raise InstrumentError(f"{stage} is not bound to the acceptor-G endpoint")
        if candidate.get("masked_donor_endpoint_constructed") is not False:
            raise InstrumentError(f"{stage} constructed a forbidden masked-donor endpoint")
    checkpoint_hashes = {stages[stage].get("checkpoint_sha256") for stage in stages}
    if len(checkpoint_hashes) != 1 or not re.fullmatch(
        r"[0-9a-f]{64}", str(next(iter(checkpoint_hashes)))
    ):
        raise InstrumentError("entry stages do not share one bound checkpoint")
    for key in ("cohort_sha256", "construction_sha256"):
        values = {stages[stage].get(key) for stage in ("r5a", "r5b")}
        if len(values) != 1 or not re.fullmatch(
            r"[0-9a-f]{64}", str(next(iter(values)))
        ):
            raise InstrumentError(f"R5a/R5b {key} differs or is missing")
    stack_fingerprints = {
        stages[stage].get(STACK_FINGERPRINT_FIELD) for stage in ("r5a", "r5b")
    }
    if len(stack_fingerprints) != 1 or not re.fullmatch(
        r"[0-9a-f]{64}", str(next(iter(stack_fingerprints)))
    ):
        raise InstrumentError("R5a/R5b entry contracts do not share one joint stack")
    rank = stages["r5b"].get("selected_rank")
    if not _is_integer(rank) or int(rank) < 1:
        raise ScientificStop("R5b selected no non-null U_A rank")
    crossfit = stages["r5b"].get("crossfit_map_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", str(crossfit)):
        raise InstrumentError("R5b cross-fit map is not literally bound")
    return {
        "status": "R5C_ENTRY_AUTHORIZED",
        "checkpoint_sha256": next(iter(checkpoint_hashes)),
        "cohort_sha256": stages["r5b"]["cohort_sha256"],
        "construction_sha256": stages["r5b"]["construction_sha256"],
        "selected_rank": int(rank),
        "crossfit_map_sha256": crossfit,
        STACK_FINGERPRINT_FIELD: next(iter(stack_fingerprints)),
    }


def _load_exact_json_artifact(
    path: str | Path, *, expected_sha256: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = Path(path).expanduser().resolve()
    manifest_path = artifact.with_suffix(".manifest.json")
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)):
        raise InstrumentError(f"{label} literal artifact pin is malformed")
    try:
        observed_sha256 = sha256_file(artifact)
    except OSError as exc:
        raise InstrumentError(f"{label} artifact cannot be read: {exc}") from exc
    if observed_sha256 != expected_sha256:
        raise InstrumentError(f"{label} artifact bytes differ from the literal pin")
    try:
        record = json.loads(artifact.read_text())
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise InstrumentError(f"{label} artifact/manifest cannot be read: {exc}") from exc
    if not isinstance(record, dict) or not isinstance(manifest, dict):
        raise InstrumentError(f"{label} artifact/manifest schema is not an object")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not any(
        Path(str(item.get("path", ""))).expanduser().resolve() == artifact
        and item.get("sha256") == expected_sha256
        and item.get("size_bytes") == artifact.stat().st_size
        for item in outputs
        if isinstance(item, Mapping)
    ):
        raise InstrumentError(f"{label} manifest does not independently bind its result")
    if manifest.get("git_dirty") is not False:
        raise InstrumentError(f"{label} was not produced from a clean worktree")
    return record, manifest


def _r3c_entry_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    construction = record.get("construction", {})
    policy = construction.get("policy", {}) if isinstance(construction, Mapping) else {}
    l3_path = record.get("paths", {}).get("L3H1_reused_component", {})
    checkpoint = record.get("hashes", {}).get("checkpoint_aggregate_sha256")
    target = (
        record.get("stage") == "target"
        and record.get("status") == "CONFIRMATORY"
        and record.get("verdict") == "HELD_OUT_ACCEPTOR_HEAD_CONFIRMATION_COMPLETE"
        and record.get("instrument_revision") == R3C_INSTRUMENT_REVISION
    )
    return {
        "instrument_gates_pass": record.get("all_gates_pass") is True,
        "claim_eligible": bool(
            target and l3_path.get("classification") == "SUPPORTED"
        ),
        "verdict": record.get("verdict"),
        "acceptor_G_only": construction.get("all_single_acceptor_G") is True,
        "masked_donor_endpoint_constructed": policy.get("donor_readout_constructed")
        is not False,
        "checkpoint_sha256": checkpoint,
    }


def load_and_validate_entry_artifacts(
    *,
    r3c_path: str | Path,
    r5a_path: str | Path,
    r5b_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate all upstream artifacts and apply the conditional entry rule."""

    r3c, r3c_manifest = _load_exact_json_artifact(
        r3c_path,
        expected_sha256=str(FROZEN_R3C_ENTRY_ARTIFACT_SHA256),
        label="R3c entry",
    )
    r5a, _r5a_manifest = _load_exact_json_artifact(
        r5a_path,
        expected_sha256=str(FROZEN_R5A_ENTRY_ARTIFACT_SHA256),
        label="R5a entry",
    )
    r5b, _r5b_manifest = _load_exact_json_artifact(
        r5b_path,
        expected_sha256=str(FROZEN_R5B_ENTRY_ARTIFACT_SHA256),
        label="R5b entry",
    )
    if (
        r3c.get("schema") != "gfmcd.r3c.acceptor_head_confirmation/2"
        or r3c.get("stage") != "target"
        or r3c.get("status") != "CONFIRMATORY"
        or r3c.get("tier") != "HELD_OUT_TARGET"
        or r3c.get("split") != "val"
        or r3c.get("gene_bucket") != 1
        or r3c.get("verdict")
        != "HELD_OUT_ACCEPTOR_HEAD_CONFIRMATION_COMPLETE"
        or r3c.get("all_gates_pass") is not True
        or r3c.get("sealed_split_opened") is not False
        or r3c.get("instrument_revision") != R3C_INSTRUMENT_REVISION
        or r3c.get("primary_estimand")
        != "raw masked-acceptor-G logit-margin component drop"
    ):
        raise InstrumentError("R3c entry artifact is not the corrected frozen target schema")
    try:
        validate_r3c_target_health_contract(r3c)
        validate_r3c_target_manifest_contract(r3c, r3c_manifest)
    except R5StackBindingError as exc:
        raise InstrumentError(f"R3c target/manifest contract differs: {exc}") from exc
    if (
        r5a.get("schema") != "gfmcd.r5a.acceptor_address_binding/1"
        or r5a.get("stage") != "target"
    ):
        raise InstrumentError("R5a entry artifact is not its frozen target schema")
    if (
        r5b.get("schema") != "gfmcd.r5b.acceptor_side_state/1"
        or r5b.get("stage") != "target"
    ):
        raise InstrumentError("R5b entry artifact is not its frozen target schema")
    r5a_contract = r5a.get("r5_entry_contract")
    r5b_contract = r5b.get("r5_entry_contract")
    if not isinstance(r5a_contract, Mapping) or not isinstance(r5b_contract, Mapping):
        raise InstrumentError("R5a/R5b compact artifacts lack the standard entry contract")
    if any(
        contract.get("schema") != R5_ENTRY_CONTRACT_SCHEMA
        for contract in (r5a_contract, r5b_contract)
    ):
        raise InstrumentError("R5a/R5b entry-contract schema differs")
    manifest_hashes = {
        "r3c": sha256_file(Path(r3c_path).expanduser().resolve().with_suffix(".manifest.json")),
        "r5a": sha256_file(Path(r5a_path).expanduser().resolve().with_suffix(".manifest.json")),
        "r5b": sha256_file(Path(r5b_path).expanduser().resolve().with_suffix(".manifest.json")),
    }
    if (
        r5a_contract.get("upstream_r3c_artifact_sha256")
        != str(FROZEN_R3C_ENTRY_ARTIFACT_SHA256)
        or r5a_contract.get("upstream_r3c_manifest_sha256")
        != manifest_hashes["r3c"]
        or r5b_contract.get("upstream_r3c_artifact_sha256")
        != str(FROZEN_R3C_ENTRY_ARTIFACT_SHA256)
        or r5b_contract.get("upstream_r3c_manifest_sha256")
        != manifest_hashes["r3c"]
        or r5b_contract.get("upstream_r5a_artifact_sha256")
        != str(FROZEN_R5A_ENTRY_ARTIFACT_SHA256)
        or r5b_contract.get("upstream_r5a_manifest_sha256")
        != manifest_hashes["r5a"]
    ):
        raise InstrumentError("R3c/R5a/R5b artifact-manifest chain is mixed")
    current_stack = build_r5_stack_binding(repo=REPO)
    current_stack_fingerprint = r5_stack_fingerprint(current_stack)
    try:
        for upstream in (r5a, r5b):
            validate_serialized_r5_stack(upstream, current=current_stack)
    except R5StackBindingError as exc:
        raise InstrumentError(f"upstream joint R5 stack differs: {exc}") from exc
    if any(
        contract.get(STACK_FINGERPRINT_FIELD) != current_stack_fingerprint
        for contract in (r5a_contract, r5b_contract)
    ):
        raise InstrumentError("R5a/R5b entry contracts differ from the joint R5 stack")
    validate_upstream_instrument_stack_fingerprints(
        r5a,
        r5b,
        current_stack_fingerprint=current_stack_fingerprint,
    )
    entry = {
        "r3c": _r3c_entry_contract(r3c),
        "r5a": dict(r5a_contract),
        "r5b": dict(r5b_contract),
    }
    authorized = validate_entry_chain(entry)
    authorized[STACK_FINGERPRINT_FIELD] = current_stack_fingerprint
    authorized["upstream_packages"] = {
        "r3c": {
            "artifact_sha256": str(FROZEN_R3C_ENTRY_ARTIFACT_SHA256),
            "artifact_size_bytes": Path(r3c_path).expanduser().resolve().stat().st_size,
            "manifest_sha256": manifest_hashes["r3c"],
            "manifest_size_bytes": Path(r3c_path)
            .expanduser()
            .resolve()
            .with_suffix(".manifest.json")
            .stat()
            .st_size,
        },
        "r5a": {
            "artifact_sha256": str(FROZEN_R5A_ENTRY_ARTIFACT_SHA256),
            "artifact_size_bytes": Path(r5a_path).expanduser().resolve().stat().st_size,
            "manifest_sha256": manifest_hashes["r5a"],
            "manifest_size_bytes": Path(r5a_path)
            .expanduser()
            .resolve()
            .with_suffix(".manifest.json")
            .stat()
            .st_size,
        },
        "r5b": {
            "artifact_sha256": str(FROZEN_R5B_ENTRY_ARTIFACT_SHA256),
            "artifact_size_bytes": Path(r5b_path).expanduser().resolve().stat().st_size,
            "manifest_sha256": manifest_hashes["r5b"],
            "manifest_size_bytes": Path(r5b_path)
            .expanduser()
            .resolve()
            .with_suffix(".manifest.json")
            .stat()
            .st_size,
        },
    }
    return r3c, r5a, r5b, authorized


def validate_upstream_instrument_stack_fingerprints(
    r5a: Mapping[str, Any],
    r5b: Mapping[str, Any],
    *,
    current_stack_fingerprint: str,
) -> None:
    """Validate the shared stack fingerprint in each stage's serialized payload."""

    bindings = (
        r5a.get("instrument_binding"),
        r5b.get("instrument_binding_payload"),
    )
    if any(
        not isinstance(binding, Mapping)
        or binding.get(STACK_FINGERPRINT_FIELD) != current_stack_fingerprint
        for binding in bindings
    ):
        raise InstrumentError("R5a/R5b instrument bindings differ from the joint R5 stack")


def validate_r5_cohort_binding(
    audit: Mapping[str, Any], *, bucket: int
) -> dict[str, Any]:
    """Bind R5c to the shared outcome-blind census and exactly one reservation."""

    if not _is_integer(bucket) or int(bucket) not in (0, 1):
        raise InstrumentError("R5 cohort bucket must be literal 0 or 1")
    try:
        validate_r5_reservation_audit(audit)
        identity = validate_expected_r5_census_identity(audit)
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise InstrumentError(f"R5 cohort binding failed: {exc}") from exc
    selected = identity["selected_by_bucket"][str(int(bucket))]
    return {
        "schema": identity["schema"],
        "bucket": int(bucket),
        "rows": selected["rows"],
        "genes": selected["genes"],
        "gene_ids_sha256": selected["gene_ids_sha256"],
        "ordered_rows_sha256": selected["ordered_rows_sha256"],
        "construction_sha256": selected["construction_sha256"],
        "full_census_identity_sha256": canonical_sha256(identity),
    }


def validate_r5_source_binding(gtf: str | Path, fasta: str | Path) -> dict[str, Any]:
    """Require the comprehensive-GTF and hg38 source bytes used for the census."""

    try:
        return validate_expected_r5_source_identity(gtf, fasta)
    except (OSError, RuntimeError, ValueError) as exc:
        raise InstrumentError(f"R5 source binding failed: {exc}") from exc


def _r3c_excluded_gene_sets(record: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    audit = record.get("cohort_audit")
    if not isinstance(audit, Mapping):
        raise InstrumentError("R3c artifact lacks its cohort audit")
    historical = audit.get("historical_reconstruction", {}).get("excluded_gene_ids")
    by_bucket = audit.get("current_census", {}).get(
        "fixed_selection_gene_ids_by_bucket"
    )
    if not isinstance(historical, list) or not isinstance(by_bucket, Mapping):
        raise InstrumentError("R3c artifact lacks complete R3/R3c gene exclusions")
    if set(by_bucket) != {"0", "1"}:
        raise InstrumentError("R3c artifact lacks both fixed gene buckets")
    r3c_genes: list[str] = []
    for bucket in ("0", "1"):
        values = by_bucket[bucket].get("gene_ids")
        if not isinstance(values, list):
            raise InstrumentError("R3c bucket gene IDs are missing")
        if canonical_sha256(sorted(map(str, values))) != by_bucket[bucket].get(
            "gene_ids_sha256"
        ):
            raise InstrumentError("R3c bucket gene-ID digest differs")
        r3c_genes.extend(map(str, values))
    historical = list(map(str, historical))
    if len(set(historical)) != 645 or len(set(r3c_genes)) != 588:
        raise InstrumentError("R3/R3c exclusion cardinality differs from the R5 census")
    return historical, sorted(set(r3c_genes))


def load_shared_r5_reservation(
    *,
    gtf: str | Path,
    fasta: str | Path,
    r3c_record: Mapping[str, Any],
    bucket: int,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Reconstruct the exact shared reservation before loading the adapter."""

    source_identity = validate_r5_source_binding(gtf, fasta)
    historical, r3c_genes = _r3c_excluded_gene_sets(r3c_record)
    examples = load_examples_unscored(
        str(gtf),
        str(fasta),
        "val",
        None,
        tag="r5-acceptor-mechanism-v1",
        allowed_chromosomes=R5_VALIDATION_CHROMOSOMES,
    )
    pseudo = {
        str(example.example_id): matched_decoy_donor(example) for example in examples
    }
    reservations, audit = build_r5_reservations(
        examples,
        pseudo,
        historical_r3_gene_ids=historical,
        r3c_gene_ids=r3c_genes,
    )
    binding = validate_r5_cohort_binding(audit, bucket=bucket)
    rows = reservations[int(bucket)]
    if len(rows) != R5_ROWS_PER_BUCKET:
        raise InstrumentError("shared R5 reservation is incomplete")
    return rows, {
        "source_identity": source_identity,
        "audit": audit,
        "binding": binding,
    }


CAPABILITY_GATES = (
    "single_masked_acceptor_G_readout",
    "no_masked_donor_endpoint",
    "full_mlm_logits_and_margin_replay",
    "dense_full_residual_replay",
    "l3_query_self_patch_noop",
    "all_registered_head_query_self_patch_noop",
    "l3_c_plus_query_only_isolation",
    "common_legs_bit_identical",
    "crossfit_projectors_valid",
    "random_projectors_rank_energy_matched_without_redraw",
    "ensemble_membership_and_patch_size_exact",
    "necessity_and_sufficiency_health_complete",
    "boundary_binding_suite_pass",
    "engineering_calibration_pass",
    "candidate_outcomes_not_used_for_capability",
)


def validate_head_mutation_audit(
    record: Mapping[str, Any],
    *,
    expected_heads: Sequence[tuple[int, int]] | None = None,
    expected_role: str | None = None,
    expected_component_id: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema",
        "intervention_id",
        "role",
        "component_id",
        "requested_heads",
        "requested_head_count",
        "n_rows",
        "sequence_tokens",
        "requested_token_indices",
        "requested_token_mask_sha256",
        "selected_query_source_sha256",
        "selected_query_after_sha256",
        "selected_offmask_before_sha256",
        "selected_offmask_after_sha256",
        "unselected_heads_before_sha256",
        "unselected_heads_after_sha256",
        "observed_changed_token_mask_sha256_by_head",
        "observed_changed_token_count_by_head",
        "observed_changed_token_indices_by_head",
        "off_mask_change_count",
        "unexpected_head_token_change_count",
        "replacement_mismatch_count",
        "before_layer_sha256",
        "after_layer_sha256",
        "audit_sha256",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != required
        or record.get("schema") != "gfmcd.r5c.head_mutation_audit/1"
    ):
        raise InstrumentError("head-mutation audit schema differs")
    unsigned = {key: record[key] for key in required if key != "audit_sha256"}
    if record["audit_sha256"] != canonical_sha256(unsigned):
        raise InstrumentError("head-mutation audit digest does not replay")
    requested = record["requested_heads"]
    if (
        not isinstance(requested, list)
        or len(requested) != record["requested_head_count"]
        or len(requested) != len(set(requested))
        or not requested
    ):
        raise InstrumentError("head-mutation requested-head registry differs")
    if expected_heads is not None and requested != [
        head_name(head) for head in sorted(expected_heads)
    ]:
        raise InstrumentError("head-mutation audit contains a wrong or extra head")
    if not all(
        isinstance(record.get(field), str) and record.get(field)
        for field in ("intervention_id", "role", "component_id")
    ):
        raise InstrumentError("head-mutation audit identity is incomplete")
    if expected_role is not None and record["role"] != expected_role:
        raise InstrumentError("head-mutation audit role differs")
    if expected_component_id is not None and record["component_id"] != expected_component_id:
        raise InstrumentError("head-mutation audit component differs")
    n_rows = record["n_rows"]
    sequence_tokens = record["sequence_tokens"]
    indices = record["requested_token_indices"]
    if (
        not _is_integer(n_rows)
        or not _is_integer(sequence_tokens)
        or int(n_rows) < 1
        or int(sequence_tokens) < 1
        or not isinstance(indices, list)
        or len(indices) != int(n_rows)
        or any(
            not _is_integer(value) or not 0 <= int(value) < int(sequence_tokens)
            for value in indices
        )
    ):
        raise InstrumentError("head-mutation audit token coordinates differ")
    reconstructed_mask = np.zeros((int(n_rows), int(sequence_tokens)), dtype=np.uint8)
    reconstructed_mask[np.arange(int(n_rows)), np.asarray(indices, dtype=int)] = 1
    if record["requested_token_mask_sha256"] != _boolean_array_sha256(
        reconstructed_mask
    ):
        raise InstrumentError("head-mutation requested token mask does not replay")
    digest_fields = (
        "selected_query_source_sha256",
        "selected_query_after_sha256",
        "selected_offmask_before_sha256",
        "selected_offmask_after_sha256",
        "unselected_heads_before_sha256",
        "unselected_heads_after_sha256",
    )
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(record[key])) for key in digest_fields):
        raise InstrumentError("head-mutation tensor digest is malformed")
    if record["selected_query_source_sha256"] != record["selected_query_after_sha256"]:
        raise InstrumentError("head patch did not write the requested source tensor")
    if record["selected_offmask_before_sha256"] != record["selected_offmask_after_sha256"]:
        raise InstrumentError("head patch changed a wrong token")
    if record["unselected_heads_before_sha256"] != record["unselected_heads_after_sha256"]:
        raise InstrumentError("head patch changed an extra head")
    if any(
        record[key] != 0
        for key in (
            "off_mask_change_count",
            "unexpected_head_token_change_count",
            "replacement_mismatch_count",
        )
    ):
        raise InstrumentError("head patch mutation counts show a wrong head/token change")
    changed_hashes = record["observed_changed_token_mask_sha256_by_head"]
    changed_counts = record["observed_changed_token_count_by_head"]
    changed_indices = record["observed_changed_token_indices_by_head"]
    if (
        not isinstance(changed_hashes, Mapping)
        or not isinstance(changed_counts, Mapping)
        or not isinstance(changed_indices, Mapping)
        or set(changed_hashes) != set(requested)
        or set(changed_counts) != set(requested)
        or set(changed_indices) != set(requested)
        or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(changed_hashes[name]))
            or not _is_integer(changed_counts[name])
            or not 0 <= int(changed_counts[name]) <= int(n_rows)
            for name in requested
        )
    ):
        raise InstrumentError("observed head/token mutation registry differs")
    for name in requested:
        coordinates = changed_indices[name]
        if not isinstance(coordinates, list) or any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not _is_integer(value) for value in pair)
            for pair in coordinates
        ):
            raise InstrumentError("observed head/token mutation coordinates differ")
        observed_mask = np.zeros((int(n_rows), int(sequence_tokens)), dtype=np.uint8)
        for row, token in coordinates:
            if (
                not 0 <= int(row) < int(n_rows)
                or int(token) != int(indices[int(row)])
                or observed_mask[int(row), int(token)]
            ):
                raise InstrumentError("observed mutation contains a wrong token")
            observed_mask[int(row), int(token)] = 1
        if (
            int(np.count_nonzero(observed_mask)) != int(changed_counts[name])
            or _boolean_array_sha256(observed_mask) != changed_hashes[name]
        ):
            raise InstrumentError("observed changed-token mask does not replay")
    for field in ("before_layer_sha256", "after_layer_sha256"):
        values = record[field]
        if not isinstance(values, Mapping) or any(
            not re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in values.values()
        ):
            raise InstrumentError("head-mutation layer digest registry differs")
    if expected_heads is not None:
        expected_layers = {str(layer) for layer, _ in expected_heads}
        if any(
            set(record[field]) != expected_layers
            for field in ("before_layer_sha256", "after_layer_sha256")
        ):
            raise InstrumentError("head-mutation observed-layer registry differs")
    return {
        "audit_sha256": record["audit_sha256"],
        "requested_token_indices": [int(value) for value in indices],
        "requested_heads": list(requested),
    }


def validate_mutation_audit_registry(leg: Mapping[str, Any]) -> dict[str, Any]:
    hook = leg.get("hook_capability")
    chunks = hook.get("mutation_audit_chunks") if isinstance(hook, Mapping) else None
    if not isinstance(chunks, list) or not chunks:
        raise InstrumentError("source leg lacks measured mutation-audit chunks")
    expected_query_indices_by_chunk = hook.get("expected_query_token_indices_by_chunk")
    if (
        not isinstance(expected_query_indices_by_chunk, list)
        or len(expected_query_indices_by_chunk) != len(chunks)
    ):
        raise InstrumentError("source leg lacks expected masked-acceptor query coordinates")
    registered = tuple(
        sorted(
            {L3_READER, DIRECT_NEGATIVE}
            | {head for ensemble in complete_downstream_registry() for head in ensemble}
        )
    )
    components = {
        ensemble_name(ensemble): tuple(ensemble)
        for ensemble in complete_downstream_registry()
    }
    all_digests: list[str] = []
    intervention_ids: list[str] = []
    total_rows = 0
    for chunk_index, raw_records in enumerate(chunks):
        if not isinstance(raw_records, list):
            raise InstrumentError("mutation-audit chunk is not an ordered list")
        records = [dict(record) for record in raw_records]
        intervention_ids.extend(str(record.get("intervention_id")) for record in records)
        by_role: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            by_role.setdefault(str(record.get("role")), []).append(record)
        singleton_roles = {
            "registered_clean_self_patch": (registered, "ALL_REGISTERED_HEADS"),
            "registered_minus_self_patch": (registered, "ALL_REGISTERED_HEADS"),
            "l3_c_plus": ((L3_READER,), head_name(L3_READER)),
            "direct_negative": ((DIRECT_NEGATIVE,), head_name(DIRECT_NEGATIVE)),
        }
        query_indices: list[int] | None = None
        for role, (heads, component_id) in singleton_roles.items():
            if len(by_role.get(role, ())) != 1:
                raise InstrumentError(f"mutation-audit chunk lacks exactly one {role}")
            validated = validate_head_mutation_audit(
                by_role[role][0],
                expected_heads=heads,
                expected_role=role,
                expected_component_id=component_id,
            )
            if query_indices is None:
                query_indices = validated["requested_token_indices"]
                total_rows += len(query_indices)
            elif validated["requested_token_indices"] != query_indices:
                raise InstrumentError("mutation audit used a wrong token within a chunk")
            all_digests.append(validated["audit_sha256"])
        if query_indices != expected_query_indices_by_chunk[chunk_index]:
            raise InstrumentError("mutation audit used a wrong masked-acceptor token")
        for role in (
            "sandwich_necessity_l3",
            "sandwich_necessity_delivery",
            "sandwich_sufficiency_delivery",
        ):
            records_for_role = by_role.get(role, ())
            if len(records_for_role) != len(components):
                raise InstrumentError(f"mutation-audit {role} registry is incomplete")
            by_component = {str(record.get("component_id")): record for record in records_for_role}
            if set(by_component) != set(components):
                raise InstrumentError(f"mutation-audit {role} component registry differs")
            for component_id, ensemble in components.items():
                expected_heads = (L3_READER,) if role == "sandwich_necessity_l3" else ensemble
                validated = validate_head_mutation_audit(
                    by_component[component_id],
                    expected_heads=expected_heads,
                    expected_role=role,
                    expected_component_id=component_id,
                )
                if validated["requested_token_indices"] != query_indices:
                    raise InstrumentError("mutation audit used a wrong token within a chunk")
                all_digests.append(validated["audit_sha256"])
        expected_roles = set(singleton_roles) | {
            "sandwich_necessity_l3",
            "sandwich_necessity_delivery",
            "sandwich_sufficiency_delivery",
        }
        if set(by_role) != expected_roles:
            raise InstrumentError("mutation-audit chunk contains an undeclared role")

        for component_id in components:
            envelope = leg["sandwich"][component_id]
            audit_lists = envelope.get("mutation_audit_sha256_by_chunk", {})
            if not isinstance(audit_lists, Mapping):
                raise InstrumentError("sandwich lacks independent mutation records")
            direction_map = {
                "necessity_l3": "sandwich_necessity_l3",
                "necessity_delivery": "sandwich_necessity_delivery",
                "sufficiency_delivery": "sandwich_sufficiency_delivery",
            }
            for direction, role in direction_map.items():
                values = audit_lists.get(direction)
                if (
                    not isinstance(values, list)
                    or len(values) != len(chunks)
                    or values[chunk_index]
                    != next(
                        record["audit_sha256"]
                        for record in by_role[role]
                        if record["component_id"] == component_id
                    )
                ):
                    raise InstrumentError("sandwich/mutation audit cross-link differs")
    if total_rows != R5_ROWS_PER_BUCKET:
        raise InstrumentError("mutation audits do not cover the exact R5 reservation")
    if len(intervention_ids) != len(set(intervention_ids)):
        raise InstrumentError("mutation-audit intervention identities are duplicated")
    return {
        "schema": "gfmcd.r5c.mutation_registry_validation/1",
        "leg_id": leg.get("leg_id"),
        "chunks": len(chunks),
        "rows": total_rows,
        "audit_count": len(all_digests),
        "ordered_audit_sha256": canonical_sha256(all_digests),
        "query_only": True,
        "ensemble_membership_and_patch_size_exact": True,
    }


def validate_residual_replay_registry(leg: Mapping[str, Any]) -> dict[str, Any]:
    hook = leg.get("hook_capability")
    records = hook.get("condition_capture_records") if isinstance(hook, Mapping) else None
    claimed = hook.get("dense_full_residual_replay") if isinstance(hook, Mapping) else None
    if not isinstance(records, list) or not records or not isinstance(claimed, Mapping):
        raise InstrumentError("source leg lacks dense residual replay records")
    identities: list[str] = []
    errors: list[float] = []
    exact: list[bool] = []
    required = {
        "schema",
        "margin_sha256",
        "query_residual_sha256",
        "query_logits_sha256",
        "head_outputs_sha256_by_layer",
        "full_mlm_margin_replay_max_abs_error",
        "full_mlm_logits_replay_exact",
        "capture_sha256",
    }
    for record in records:
        if (
            not isinstance(record, Mapping)
            or set(record) != required
            or record.get("schema") != "gfmcd.r5c.condition_capture_identity/1"
        ):
            raise InstrumentError("condition-capture replay schema differs")
        unsigned = {key: record[key] for key in required if key != "capture_sha256"}
        if record["capture_sha256"] != canonical_sha256(unsigned):
            raise InstrumentError("condition-capture digest does not replay")
        head_hashes = record["head_outputs_sha256_by_layer"]
        if not isinstance(head_hashes, Mapping) or set(head_hashes) != {
            str(layer) for layer in range(6)
        }:
            raise InstrumentError("condition-capture head-layer registry differs")
        digest_values = [
            record["margin_sha256"],
            record["query_residual_sha256"],
            record["query_logits_sha256"],
            *head_hashes.values(),
        ]
        if any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in digest_values):
            raise InstrumentError("condition-capture tensor digest is malformed")
        error = float(record["full_mlm_margin_replay_max_abs_error"])
        if not np.isfinite(error) or error < 0 or type(record["full_mlm_logits_replay_exact"]) is not bool:
            raise InstrumentError("condition-capture replay diagnostic is malformed")
        identities.append(str(record["capture_sha256"]))
        errors.append(error)
        exact.append(bool(record["full_mlm_logits_replay_exact"]))
    derived = {
        "condition_count": len(records),
        "margin_max_abs_error": max(errors),
        "full_logits_exact": all(exact),
        "condition_capture_registry_sha256": canonical_sha256(identities),
    }
    if dict(claimed) != derived:
        raise InstrumentError("dense residual replay summary does not rederive")
    return derived


def validate_common_leg_capture_registry(leg: Mapping[str, Any]) -> dict[str, Any]:
    sandwich = leg.get("sandwich")
    chunks = leg.get("chunk_condition_hashes")
    hook_capability = leg.get("hook_capability")
    if (
        not isinstance(sandwich, Mapping)
        or not isinstance(chunks, list)
        or not chunks
        or not isinstance(hook_capability, Mapping)
    ):
        raise InstrumentError("source leg lacks captured common-leg identities")
    canonical = {
        component: _replay_delivery_envelope(record)
        for component, record in sandwich.items()
    }
    first = next(iter(canonical.values()))
    captured_records = [
        chunk[field]
        for chunk in chunks
        for field in ("m_minus_capture_identity", "m_c_plus_capture_identity")
    ]
    for identity in captured_records:
        if (
            not isinstance(identity, Mapping)
            or identity.get("schema") != "gfmcd.r5c.condition_capture_identity/1"
            or identity.get("capture_sha256")
            != canonical_sha256(
                {
                    key: value
                    for key, value in identity.items()
                    if key != "capture_sha256"
                }
            )
        ):
            raise InstrumentError("common-leg condition-capture identity does not replay")
    replay_capture_hashes = {
        str(record.get("capture_sha256"))
        for record in hook_capability.get("condition_capture_records", ())
        if isinstance(record, Mapping)
    }
    if any(
        identity["capture_sha256"] not in replay_capture_hashes
        for identity in captured_records
    ):
        raise InstrumentError("common-leg captures are absent from residual replay registry")
    capture_identity = {
        "m_minus_capture_sha256": canonical_sha256(
            [
                chunk["m_minus_capture_identity"]["capture_sha256"]
                for chunk in chunks
            ]
        ),
        "m_c_plus_capture_sha256": canonical_sha256(
            [
                chunk["m_c_plus_capture_identity"]["capture_sha256"]
                for chunk in chunks
            ]
        ),
    }
    common = validate_common_legs(
        [record["common_leg"] for record in sandwich.values()],
        m_minus=first["cells"]["m_minus"],
        m_c_plus=first["cells"]["m_c_plus"],
        capture_identity=capture_identity,
    )
    if dict(leg.get("common_leg", {})) != common:
        raise InstrumentError("source common-leg identity differs from captured tensors")
    return {
        "leg_id": leg.get("leg_id"),
        "common_leg_pair_sha256": common["common_leg_pair_sha256"],
        "captured_tensor_identity_sha256": canonical_sha256(capture_identity),
        "components": len(sandwich),
    }


def validate_projector_capability_audit(
    record: Mapping[str, Any],
    *,
    expected_runtime_array_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    required = {
        "schema",
        "selected",
        "random",
        "authenticated_runtime_array_sha256",
        "observed_runtime_array_sha256",
        "all_projectors_validated",
        "audit_sha256",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != required
        or record.get("schema") != "gfmcd.r5c.crossfit_projector_capability/1"
    ):
        raise InstrumentError("cross-fit projector capability schema differs")
    selected = record["selected"]
    random_records = record["random"]
    if not isinstance(selected, Mapping) or not isinstance(random_records, list) or len(random_records) != 32:
        raise InstrumentError("cross-fit projector audit registry differs")
    if not _is_integer(selected.get("selected_rank")) or int(
        selected["selected_rank"]
    ) < 1:
        raise InstrumentError("cross-fit projector selected rank is malformed")
    all_records = [selected, *random_records]
    for index, audit in enumerate(all_records):
        try:
            symmetry = float(audit.get("symmetry_max_abs"))
            idempotence = float(audit.get("idempotence_max_abs"))
        except (TypeError, ValueError):
            raise InstrumentError("cross-fit projector diagnostic is malformed") from None
        if (
            audit.get("n_projectors") != R5_ROWS_PER_BUCKET
            or audit.get("selected_rank") != selected.get("selected_rank")
            or not np.isfinite([symmetry, idempotence]).all()
            or symmetry > 1e-7
            or idempotence > 1e-7
            or audit.get("representation")
            not in {"square_projector", "row_orthonormal_basis"}
            or not re.fullmatch(r"[0-9a-f]{64}", str(audit.get("operator_array_sha256")))
            or (index and audit.get("draw") != index - 1)
        ):
            raise InstrumentError("cross-fit projector diagnostic differs")
    authenticated = record["authenticated_runtime_array_sha256"]
    observed = record["observed_runtime_array_sha256"]
    if (
        not isinstance(authenticated, Mapping)
        or dict(authenticated) != dict(observed)
        or (
            expected_runtime_array_sha256 is not None
            and dict(authenticated) != dict(expected_runtime_array_sha256)
        )
    ):
        raise InstrumentError("cross-fit projectors differ from authenticated runtime arrays")
    if (
        record["all_projectors_validated"] != 33
        or record["audit_sha256"]
        != canonical_sha256({"selected": selected, "random": random_records})
    ):
        raise InstrumentError("cross-fit projector audit digest differs")
    return {"selected_rank": selected["selected_rank"], "validated": 33}


def validate_capability(
    record: Mapping[str, Any],
    *,
    partner_leg: Mapping[str, Any] | None = None,
    pseudo_leg: Mapping[str, Any] | None = None,
    expected_runtime_array_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    gates = record.get("gates")
    if not isinstance(gates, Mapping) or set(gates) != set(CAPABILITY_GATES):
        raise InstrumentError("capability record does not contain the exact gate registry")
    evidence = record.get("evidence")
    if any(type(gates[name]) is not bool for name in CAPABILITY_GATES):
        raise InstrumentError("capability gates must be literal booleans")
    if not isinstance(evidence, Mapping):
        raise InstrumentError("capability lacks measured evidence")
    if partner_leg is None or pseudo_leg is None:
        raise InstrumentError("capability replay requires both primitive source legs")
    replay = {
        "partner": validate_residual_replay_registry(partner_leg),
        "pseudo": validate_residual_replay_registry(pseudo_leg),
    }
    mutations = {
        "partner": validate_mutation_audit_registry(partner_leg),
        "pseudo": validate_mutation_audit_registry(pseudo_leg),
    }
    common = {
        "partner": validate_common_leg_capture_registry(partner_leg),
        "pseudo": validate_common_leg_capture_registry(pseudo_leg),
    }
    if evidence.get("residual_replay") != replay:
        raise InstrumentError("capability residual replay evidence differs")
    if evidence.get("mutation_audit") != mutations:
        raise InstrumentError("capability mutation evidence differs")
    if evidence.get("common_leg_capture") != common:
        raise InstrumentError("capability common-leg evidence differs")
    projector_validation = validate_projector_capability_audit(
        evidence.get("crossfit_projectors", {}),
        expected_runtime_array_sha256=expected_runtime_array_sha256,
    )
    readout = evidence.get("readout_validation")
    self_patch = evidence.get("self_patch")
    random_projectors = evidence.get("random_projectors")
    diagnostics = evidence.get("diagnostics")
    health = evidence.get("health")
    if not all(
        isinstance(value, Mapping)
        for value in (readout, self_patch, random_projectors, diagnostics, health)
    ):
        raise InstrumentError("capability measured-evidence schema differs")
    derived_gates = {
        "single_masked_acceptor_G_readout": readout.get("acceptor_g_mismatch_count") == 0,
        "no_masked_donor_endpoint": all(
            leg.get("masked_donor_endpoint_constructed") is False
            for leg in (partner_leg, pseudo_leg)
        ),
        "full_mlm_logits_and_margin_replay": all(
            item["full_logits_exact"] is True and item["margin_max_abs_error"] <= 1e-5
            for item in replay.values()
        ),
        "dense_full_residual_replay": all(
            item["condition_count"] > 0 and item["margin_max_abs_error"] <= 1e-5
            for item in replay.values()
        ),
        "l3_query_self_patch_noop": self_patch.get("l3_included") is True
        and self_patch.get("margin_max_abs_error", np.inf) <= 1e-5
        and self_patch.get("full_logits_exact") is True,
        "all_registered_head_query_self_patch_noop": self_patch.get(
            "registered_heads_exact"
        )
        is True
        and self_patch.get("margin_max_abs_error", np.inf) <= 1e-5
        and self_patch.get("full_logits_exact") is True,
        "l3_c_plus_query_only_isolation": all(
            item["query_only"] is True for item in mutations.values()
        ),
        "common_legs_bit_identical": all(
            item["components"] == len(complete_downstream_registry())
            for item in common.values()
        ),
        "crossfit_projectors_valid": projector_validation.get("validated") == 33,
        "random_projectors_rank_energy_matched_without_redraw": random_projectors.get(
            "redrawn_count"
        )
        == 0
        and random_projectors.get("energy_match_max_abs_error", np.inf) <= 1e-7,
        "ensemble_membership_and_patch_size_exact": all(
            item["ensemble_membership_and_patch_size_exact"] is True
            for item in mutations.values()
        ),
        "necessity_and_sufficiency_health_complete": health.get("all_arms_ok") is True,
        "boundary_binding_suite_pass": diagnostics.get("binding_suite_status") == "PASS",
        "engineering_calibration_pass": diagnostics.get("engineering_calibration_status")
        == "PASS"
        and diagnostics.get("production_trial_count") is True,
        "candidate_outcomes_not_used_for_capability": record.get(
            "candidate_outcomes_used"
        )
        is False
        and health.get("candidate_outcomes_used") is False
        and all(
            leg.get("hook_capability", {}).get("candidate_outcomes_used") is False
            for leg in (partner_leg, pseudo_leg)
        ),
    }
    if dict(gates) != derived_gates:
        raise InstrumentError("capability gates do not rederive from measured evidence")
    if not all(derived_gates.values()):
        failed = [name for name in CAPABILITY_GATES if not derived_gates[name]]
        raise InstrumentError("R5c capability failed: " + ", ".join(failed))
    if record.get("candidate_outcomes_used") is not False:
        raise InstrumentError("candidate outcomes may not establish instrument capability")
    return {"status": "R5C_CAPABILITY_CONFIRMED", "gates": derived_gates}


def computed_calibration_trials(
    *,
    alpha_family: float = FAMILY_ALPHA,
    epsilon: float = CALIBRATION_EPSILON,
    power_floor: float = CALIBRATION_POWER_FLOOR,
    plant_success_rate: float = CALIBRATION_PLANT_SUCCESS_RATE,
    confidence_alpha: float = 0.05,
    maximum: int = 1_000_000,
) -> int:
    """Prospectively derive the exact engineering-calibration trial count."""

    if not 0 < alpha_family < 1 or not 0 < epsilon < 1 or not 0 < power_floor < plant_success_rate < 1:
        raise ValueError("invalid calibration design constants")
    for trials in range(1, maximum + 1):
        false_positives = int(np.floor(trials * alpha_family))
        plant_successes = int(np.floor(trials * plant_success_rate))
        _, upper = one_sided_binomial_bounds(
            false_positives, trials, alpha=confidence_alpha
        )
        lower, _ = one_sided_binomial_bounds(
            plant_successes, trials, alpha=confidence_alpha
        )
        if upper <= alpha_family + epsilon and lower >= power_floor:
            return trials
    raise RuntimeError("the frozen calibration precision rule is infeasible")


def calibration_registry() -> dict[str, Any]:
    """Bind coherent primitive faces and formal validity separately.

    Partner, pseudo, specificity, and comparator gaps are algebraically linked.
    Therefore the diagnostics cover every literal decision edge with coherent faces,
    but do not falsely assert that every edge can be the sole null edge.  Nominal
    control comes from super-uniform boundary constituents, IUT=max, Holm, and the
    fixed five-family allocation—not from this engineering simulation.
    """

    paths = path_registry()
    edges: list[dict[str, str]] = []
    for family in FAMILY_NAMES:
        seen: set[tuple[str, str]] = set()
        for path, requirements in paths[family].items():
            for requirement in requirements:
                for edge, _, _ in _requirement_edges(requirement):
                    key = (requirement.row, edge)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append(
                        {
                            "family": family,
                            "path": path,
                            "row": requirement.row,
                            "edge": edge,
                            "primitive_face_policy": (
                                "coherent_partner_pseudo_cells; linked null edges permitted"
                            ),
                        }
                    )
    trials = computed_calibration_trials()
    return {
        "schema": "gfmcd.r5c.calibration_registry/1",
        "engineering_not_scientific_fwer_evidence": True,
        "formal_nominal_control": (
            "valid super-uniform arbitrary-boundary constituents; path IUT=max; "
            "Holm within family; alpha=.05/5 fixed family allocation"
        ),
        "primitive_generator": (
            "centered partner/pseudo raw margin cells; specificity and all gaps rederived"
        ),
        "isolated_edge_maximum_claimed": False,
        "literal_edge_count": len(edges),
        "literal_edges_sha256": canonical_sha256(edges),
        "coherent_face_registry": edges,
        "alpha_family": FAMILY_ALPHA,
        "epsilon_cal": CALIBRATION_EPSILON,
        "power_floor": CALIBRATION_POWER_FLOOR,
        "plant_success_rate": CALIBRATION_PLANT_SUCCESS_RATE,
        "N_cal": trials,
    }


def _row_fields(row: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in row.split("|")[1:]:
        if "=" not in part:
            raise InstrumentError(f"malformed R5c row ID: {row}")
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def coherent_synthetic_family(
    family: str,
    *,
    boundary_row: str | None = None,
    boundary_edge: str | None = None,
    n_examples: int = MIN_GENES,
    residual_scale: float = 0.0,
    outer_trial: int = 0,
) -> np.ndarray:
    """Generate a family from raw partner/pseudo primitive cells.

    This is the binding-suite generator.  It never shifts a caller-visible derived
    specificity or gap row: a requested face changes an underlying partner/pseudo
    component cell and regenerates the complete family.
    """

    names = family_row_names()[family]
    paths = path_registry()[family]
    requirements = {
        requirement.row: requirement
        for path_requirements in paths.values()
        for requirement in path_requirements
    }
    if not _is_integer(n_examples) or int(n_examples) < MIN_GENES:
        raise ValueError("synthetic family requires at least the frozen gene count")

    raw: dict[tuple[str, str], list[np.ndarray]] = {}
    raw_base: dict[tuple[str, str], list[float]] = {}
    sign_seed = int.from_bytes(
        hashlib.sha256(
            f"r5c-calibration-outer|{family}|{outer_trial}".encode()
        ).digest()[:8],
        "big",
    )
    outer_signs = np.random.default_rng(sign_seed).choice(
        (-1.0, 1.0), size=int(n_examples)
    )

    def ensure(
        component: str, estimand: str, *, target: bool = False
    ) -> list[np.ndarray]:
        key = (component, estimand)
        if key not in raw:
            initial = [0.20 if target else 0.0, 0.0]
            raw_base[key] = initial.copy()
            arrays: list[np.ndarray] = []
            for endpoint_index, location in enumerate(("partner", "pseudo")):
                residual_seed = int.from_bytes(
                    hashlib.sha256(
                        f"r5c-calibration-residual|{family}|{component}|{estimand}|{location}".encode()
                    ).digest()[:8],
                    "big",
                )
                residual = np.random.default_rng(residual_seed).normal(
                    0.0, float(residual_scale), size=int(n_examples)
                )
                residual -= residual.mean()
                residual *= outer_signs
                arrays.append(initial[endpoint_index] + residual)
            raw[key] = arrays
        return raw[key]

    def set_base(component: str, estimand: str, endpoint_index: int, value: float) -> None:
        primitive = ensure(component, estimand)
        key = (component, estimand)
        old = raw_base[key][endpoint_index]
        primitive[endpoint_index] = primitive[endpoint_index] + (float(value) - old)
        raw_base[key][endpoint_index] = float(value)

    # Material target primitives; all controls begin at equivalence.
    if family == "C1":
        ensure(head_name(L3_READER), "R_L3", target=True)
        ensure(head_name(DIRECT_NEGATIVE), "R_L3")
    elif family == "C2":
        for estimand in ("S_U", "N_U"):
            ensure("U_A", estimand, target=True)
            for draw in range(32):
                ensure(f"random_projector_{draw:02d}", estimand)
    elif family == "C3":
        for component in map(head_name, ACCEPTOR_CANDIDATES):
            for estimand in SANDWICH_ESTIMANDS:
                ensure(component, estimand, target=True)
        for component in map(head_name, DONOR_ROUTE_COMPARATORS):
            for estimand in SANDWICH_ESTIMANDS:
                ensure(component, estimand)
    elif family == "C4":
        target = ensemble_name(PRIMARY_ACCEPTOR_PAIR)
        for estimand in SANDWICH_ESTIMANDS:
            ensure(target, estimand, target=True)
        for component in DONOR_ROUTE_PAIRS + random_pairs():
            for estimand in SANDWICH_ESTIMANDS:
                ensure(ensemble_name(component), estimand)
    elif family == "C5":
        target = ensemble_name(SECONDARY_ACCEPTOR_TRIPLE)
        for estimand in SANDWICH_ESTIMANDS:
            ensure(target, estimand, target=True)
        for component in (DONOR_ROUTE_TRIPLE,) + random_triples():
            for estimand in SANDWICH_ESTIMANDS:
                ensure(ensemble_name(component), estimand)
    else:  # pragma: no cover - caller checked by registry lookup
        raise ValueError("unknown family")

    if boundary_row is not None:
        if boundary_row not in requirements or boundary_edge is None:
            raise InstrumentError("boundary face does not name a path constituent")
        edge_specs = {
            edge: boundary
            for edge, boundary, _ in _requirement_edges(requirements[boundary_row])
        }
        if boundary_edge not in edge_specs:
            raise InstrumentError("boundary edge is not a constituent of the named row")
        boundary = edge_specs[boundary_edge]
        fields = _row_fields(boundary_row)
        estimand = fields["estimand"]
        if "component" in fields:
            component = fields["component"]
            ensure(component, estimand)
            endpoint = fields["endpoint"]
            if endpoint == "partner":
                set_base(component, estimand, 0, boundary)
            elif endpoint == "pseudo":
                set_base(component, estimand, 1, boundary)
            elif endpoint == "specific":
                set_base(
                    component,
                    estimand,
                    0,
                    raw_base[(component, estimand)][1] + boundary,
                )
            else:  # pragma: no cover
                raise InstrumentError("unknown endpoint in row ID")
        else:
            target, control = fields["gap"].split("-", 1)
            ensure(target, estimand)
            ensure(control, estimand)
            set_base(
                control,
                estimand,
                0,
                raw_base[(target, estimand)][0] - boundary,
            )

    def endpoint(primitive: Sequence[np.ndarray], name: str) -> np.ndarray:
        if name == "partner":
            return primitive[0]
        if name == "pseudo":
            return primitive[1]
        return primitive[0] - primitive[1]

    values: list[np.ndarray] = []
    for name in names:
        fields = _row_fields(name)
        estimand = fields["estimand"]
        if "component" in fields:
            value = endpoint(ensure(fields["component"], estimand), fields["endpoint"])
        else:
            target, control = fields["gap"].split("-", 1)
            value = ensure(target, estimand)[0] - ensure(control, estimand)[0]
        values.append(np.asarray(value, dtype=float))
    return np.vstack(values)


def run_constituent_binding_suite(
    *, n_draws: int = PRODUCTION_RANDOMISATION, seed: int = PRODUCTION_SEED
) -> dict[str, Any]:
    """Exercise all coherent faces and production inference on each signature."""

    groups = [f"synthetic-gene-{index:03d}" for index in range(MIN_GENES)]
    registry = calibration_registry()
    cases = registry["coherent_face_registry"]
    face_hashes: list[str] = []
    signatures: dict[tuple[str, str, str], dict[str, str]] = {}
    for case in cases:
        matrix = coherent_synthetic_family(
            case["family"],
            boundary_row=case["row"],
            boundary_edge=case["edge"],
        )
        names = family_row_names()[case["family"]
        ]
        row_index = names.index(case["row"])
        expected_boundary = {
            edge: boundary
            for edge, boundary, _ in _requirement_edges(
                next(
                    requirement
                    for requirements in path_registry()[case["family"]].values()
                    for requirement in requirements
                    if requirement.row == case["row"]
                )
            )
        }[case["edge"]]
        if not np.allclose(
            matrix[row_index], expected_boundary, rtol=0.0, atol=1e-12
        ):
            raise InstrumentError("coherent primitive face did not regenerate its edge")
        face_hashes.append(array_sha256(matrix))
        fields = _row_fields(case["row"])
        signature = (
            case["family"],
            case["edge"],
            "gap" if "gap" in fields else fields.get("endpoint", "unknown"),
        )
        signatures.setdefault(signature, case)

    # One production-classifier call per algebra/boundary signature, rather than
    # pretending algebraically linked literal rows are independent DGPs.
    diagnostic_results: dict[str, Any] = {}
    for signature, case in signatures.items():
        matrix = coherent_synthetic_family(
            case["family"],
            boundary_row=case["row"],
            boundary_edge=case["edge"],
        )
        result = strict_family_inference(
            case["family"],
            matrix,
            groups,
            n_draws=n_draws,
            seed=seed,
            namespace="r5c-binding:" + ":".join(signature),
        )
        requiring_paths = [
            path
            for path, requirements in path_registry()[case["family"]].items()
            if any(requirement.row == case["row"] for requirement in requirements)
        ]
        if any(result["paths"][path]["classification"] == "SUPPORTED" for path in requiring_paths):
            raise InstrumentError("a coherent boundary face was falsely supported")
        diagnostic_results["|".join(signature)] = {
            "representative_row": case["row"],
            "representative_edge": case["edge"],
            "requiring_paths": requiring_paths,
            "inference_sha256": canonical_sha256(result),
        }

    plant_results: dict[str, Any] = {}
    for family in FAMILY_NAMES:
        result = strict_family_inference(
            family,
            coherent_synthetic_family(family),
            groups,
            n_draws=n_draws,
            seed=seed,
            namespace=f"r5c-binding:{family}:all-alternative",
        )
        if any(path["classification"] != "SUPPORTED" for path in result["paths"].values()):
            raise InstrumentError(f"{family} coherent all-alternative plant was not supported")
        plant_results[family] = canonical_sha256(result)
    return {
        "schema": "gfmcd.r5c.constituent_binding_suite/1",
        "production_classifier_reused": True,
        "n_draws": n_draws,
        "literal_faces_checked": len(cases),
        "literal_face_matrices_sha256": canonical_sha256(face_hashes),
        "signature_diagnostics": diagnostic_results,
        "signature_count": len(signatures),
        "all_alternative_plant_inference_sha256": plant_results,
        "all_boundary_signatures_rejected": True,
        "all_path_plants_supported": True,
        "engineering_not_scientific_fwer_evidence": True,
        "status": "PASS",
    }


def run_engineering_calibration(
    *,
    n_draws: int = PRODUCTION_RANDOMISATION,
    seed: int = PRODUCTION_SEED,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Run the prospective coherent-face Type-I/power implementation audit."""

    trials = computed_calibration_trials() if n_trials is None else int(n_trials)
    if trials < 1:
        raise ValueError("calibration trials must be positive")
    production = trials == computed_calibration_trials()
    groups = [f"synthetic-gene-{index:03d}" for index in range(MIN_GENES)]
    cases_by_family = {
        family: [
            case
            for case in calibration_registry()["coherent_face_registry"]
            if case["family"] == family
        ]
        for family in FAMILY_NAMES
    }
    records: dict[str, Any] = {}
    for family in FAMILY_NAMES:
        false_positives = 0
        plant_successes = 0
        false_pattern: list[int] = []
        plant_pattern: list[int] = []
        cases = cases_by_family[family]
        for trial in range(trials):
            case = cases[trial % len(cases)]
            boundary_result = strict_family_inference(
                family,
                coherent_synthetic_family(
                    family,
                    boundary_row=case["row"],
                    boundary_edge=case["edge"],
                    residual_scale=0.08,
                    outer_trial=trial,
                ),
                groups,
                n_draws=n_draws,
                seed=seed,
                namespace=f"r5c-calibration:{family}:boundary:{trial}",
            )
            requiring = [
                path
                for path, requirements in path_registry()[family].items()
                if any(requirement.row == case["row"] for requirement in requirements)
            ]
            false_positive = any(
                boundary_result["paths"][path]["classification"] == "SUPPORTED"
                for path in requiring
            )
            false_positives += int(false_positive)
            false_pattern.append(int(false_positive))

            plant_result = strict_family_inference(
                family,
                coherent_synthetic_family(
                    family, residual_scale=0.08, outer_trial=trial
                ),
                groups,
                n_draws=n_draws,
                seed=seed,
                namespace=f"r5c-calibration:{family}:plant:{trial}",
            )
            plant_success = all(
                path["classification"] == "SUPPORTED"
                for path in plant_result["paths"].values()
            )
            plant_successes += int(plant_success)
            plant_pattern.append(int(plant_success))
        _, false_upper = one_sided_binomial_bounds(
            false_positives, trials, alpha=0.05
        )
        plant_lower, _ = one_sided_binomial_bounds(
            plant_successes, trials, alpha=0.05
        )
        records[family] = {
            "trials": trials,
            "false_positive_trials": false_positives,
            "false_positive_rate": false_positives / trials,
            "one_sided_clopper_pearson_upper": false_upper,
            "boundary_gate_ok": false_upper <= FAMILY_ALPHA + CALIBRATION_EPSILON,
            "plant_success_trials": plant_successes,
            "plant_success_rate": plant_successes / trials,
            "one_sided_clopper_pearson_lower": plant_lower,
            "plant_gate_ok": plant_lower >= CALIBRATION_POWER_FLOOR,
            "false_positive_pattern_sha256": canonical_sha256(false_pattern),
            "plant_success_pattern_sha256": canonical_sha256(plant_pattern),
            "balanced_case_cycle_sha256": canonical_sha256(
                [
                    [cases[trial % len(cases)]["row"], cases[trial % len(cases)]["edge"]]
                    for trial in range(trials)
                ]
            ),
        }
    all_pass = all(
        record["boundary_gate_ok"] and record["plant_gate_ok"]
        for record in records.values()
    )
    return {
        "schema": "gfmcd.r5c.engineering_calibration/1",
        "production_trial_count": production,
        "N_cal": trials,
        "n_draws_per_classifier": n_draws,
        "family_records": records,
        "all_family_gates_pass": all_pass,
        "engineering_not_scientific_fwer_evidence": True,
        "status": "PASS" if all_pass else "FAIL",
    }


def validate_binding_and_calibration_records(
    binding: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate every cheap, replayable invariant of the preflight diagnostics."""

    registry = calibration_registry()
    plants = binding.get("all_alternative_plant_inference_sha256")
    diagnostics = binding.get("signature_diagnostics")
    if (
        not isinstance(plants, Mapping)
        or not isinstance(diagnostics, Mapping)
        or binding.get("schema") != "gfmcd.r5c.constituent_binding_suite/1"
        or binding.get("status") != "PASS"
        or binding.get("production_classifier_reused") is not True
        or binding.get("n_draws") != PRODUCTION_RANDOMISATION
        or binding.get("literal_faces_checked") != registry["literal_edge_count"]
        or binding.get("all_boundary_signatures_rejected") is not True
        or binding.get("all_path_plants_supported") is not True
        or binding.get("engineering_not_scientific_fwer_evidence") is not True
        or set(plants) != set(FAMILY_NAMES)
    ):
        raise InstrumentError("R5c constituent binding-suite record differs")
    hash_values = [
        binding.get("literal_face_matrices_sha256"),
        *plants.values(),
        *[
            item.get("inference_sha256")
            for item in diagnostics.values()
            if isinstance(item, Mapping)
        ],
    ]
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in hash_values):
        raise InstrumentError("R5c constituent binding-suite hashes are malformed")
    expected_diagnostics: dict[str, dict[str, Any]] = {}
    for case in registry["coherent_face_registry"]:
        fields = _row_fields(case["row"])
        signature = (
            case["family"],
            case["edge"],
            "gap" if "gap" in fields else fields.get("endpoint", "unknown"),
        )
        key = "|".join(signature)
        if key not in expected_diagnostics:
            expected_diagnostics[key] = {
                "representative_row": case["row"],
                "representative_edge": case["edge"],
                "requiring_paths": [
                    path
                    for path, requirements in path_registry()[case["family"]].items()
                    if any(
                        requirement.row == case["row"]
                        for requirement in requirements
                    )
                ],
            }
    if (
        binding.get("signature_count") != len(expected_diagnostics)
        or set(diagnostics) != set(expected_diagnostics)
        or any(
            not isinstance(item, Mapping) for item in diagnostics.values()
        )
        or any(
            any(diagnostics[key].get(field) != value for field, value in expected.items())
            for key, expected in expected_diagnostics.items()
        )
    ):
        raise InstrumentError("R5c constituent binding signature count differs")

    trials = computed_calibration_trials()
    family_records = calibration.get("family_records")
    if (
        calibration.get("schema") != "gfmcd.r5c.engineering_calibration/1"
        or calibration.get("status") != "PASS"
        or calibration.get("production_trial_count") is not True
        or calibration.get("N_cal") != trials
        or calibration.get("n_draws_per_classifier") != PRODUCTION_RANDOMISATION
        or calibration.get("all_family_gates_pass") is not True
        or calibration.get("engineering_not_scientific_fwer_evidence") is not True
        or not isinstance(family_records, Mapping)
        or set(family_records) != set(FAMILY_NAMES)
    ):
        raise InstrumentError("R5c engineering-calibration record differs")
    cases = registry["coherent_face_registry"]
    for family in FAMILY_NAMES:
        record = family_records[family]
        if not isinstance(record, Mapping):
            raise InstrumentError(f"R5c {family} calibration record is malformed")
        false = record.get("false_positive_trials")
        plant = record.get("plant_success_trials")
        if (
            not _is_integer(false)
            or not _is_integer(plant)
            or not 0 <= int(false) <= trials
            or not 0 <= int(plant) <= trials
            or record.get("trials") != trials
        ):
            raise InstrumentError(f"R5c {family} calibration counts differ")
        _, false_upper = one_sided_binomial_bounds(int(false), trials, alpha=0.05)
        plant_lower, _ = one_sided_binomial_bounds(int(plant), trials, alpha=0.05)
        expected_cases = [case for case in cases if case["family"] == family]
        expected_cycle_hash = canonical_sha256(
            [
                [
                    expected_cases[trial % len(expected_cases)]["row"],
                    expected_cases[trial % len(expected_cases)]["edge"],
                ]
                for trial in range(trials)
            ]
        )
        if (
            record.get("false_positive_rate") != int(false) / trials
            or record.get("plant_success_rate") != int(plant) / trials
            or record.get("one_sided_clopper_pearson_upper") != false_upper
            or record.get("one_sided_clopper_pearson_lower") != plant_lower
            or record.get("boundary_gate_ok")
            is not bool(false_upper <= FAMILY_ALPHA + CALIBRATION_EPSILON)
            or record.get("plant_gate_ok")
            is not bool(plant_lower >= CALIBRATION_POWER_FLOOR)
            or record.get("boundary_gate_ok") is not True
            or record.get("plant_gate_ok") is not True
            or record.get("balanced_case_cycle_sha256") != expected_cycle_hash
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("false_positive_pattern_sha256"))
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("plant_success_pattern_sha256"))
            )
        ):
            raise InstrumentError(f"R5c {family} calibration replay differs")
    return {
        "status": "R5C_BINDING_AND_CALIBRATION_RECORDS_VALID",
        "binding_sha256": canonical_sha256(binding),
        "calibration_sha256": canonical_sha256(calibration),
    }


def native_behavior_inference(
    partner: Any,
    pseudo: Any,
    groups: Sequence[str],
    *,
    n_draws: int,
    seed: int,
    namespace: str,
) -> dict[str, Any]:
    values = np.vstack((np.asarray(partner, dtype=float), np.asarray(pseudo, dtype=float)))
    names = ("native_partner", "native_pseudo")
    intervals, interval_audit = simultaneous_family_decision(
        values,
        groups,
        names,
        sesoi=SESOI,
        n_draws=n_draws,
        alpha=FAMILY_ALPHA,
        seed=seed,
        namespace=f"{namespace}:intervals",
        min_genes=MIN_GENES,
        studentized=True,
    )
    tests = boundary_sign_flip_tests(
        np.vstack((values[0], values[1], values[1])),
        groups,
        ("partner_positive", "pseudo_lower", "pseudo_upper"),
        boundaries=(SESOI, -SESOI, SESOI),
        alternatives=("greater", "greater", "less"),
        n_draws=n_draws,
        seed=seed,
        namespace=f"{namespace}:shared-boundary-edges",
        min_genes=MIN_GENES,
    )
    pseudo_p = intersection_union_pvalue(
        [tests["pseudo_lower"].p_value, tests["pseudo_upper"].p_value]
    )
    path_p = intersection_union_pvalue(
        [tests["partner_positive"].p_value, pseudo_p]
    )
    partner_ci = intervals["native_partner"].simultaneous_ci
    pseudo_ci = intervals["native_pseudo"].simultaneous_ci
    ci_pass = bool(
        partner_ci[0] > SESOI
        and pseudo_ci[0] > -SESOI
        and pseudo_ci[1] < SESOI
    )
    p_pass = path_p < FAMILY_ALPHA
    if p_pass and ci_pass:
        classification = "PASS"
    elif p_pass != ci_pass:
        classification = "SCIENTIFICALLY_UNRESOLVED"
    elif strict_interval_decision(*pseudo_ci, SESOI) in {
        "MATERIAL_POSITIVE",
        "MATERIAL_NEGATIVE",
    }:
        classification = "NONSPECIFIC"
    else:
        classification = "NOT_REPLICATED"
    return {
        "partner": values[0].tolist(),
        "pseudo": values[1].tolist(),
        "specific": (values[0] - values[1]).tolist(),
        "estimates": {name: float(values[index].mean()) for index, name in enumerate(names)},
        "simultaneous_cis": {
            name: list(intervals[name].simultaneous_ci) for name in names
        },
        "interval_audit": interval_audit,
        "boundary_tests": {name: test.to_dict() for name, test in tests.items()},
        "iut_p_value": path_p,
        "strict_p_threshold": f"p < {FAMILY_ALPHA}",
        "p_pass": p_pass,
        "strict_ci_pass": ci_pass,
        "classification": classification,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def run_model_measurements(
    *,
    adapter: Any,
    rows: Sequence[Any],
    runtime: R5bRuntimeBundle,
    batch: int,
    n_draws: int,
    seed: int,
    binding_suite: Mapping[str, Any],
    engineering_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute both source legs, exact families, health, and observed inference."""

    if len(rows) != R5_ROWS_PER_BUCKET or batch < 1:
        raise InstrumentError("R5c model execution needs 128 rows and positive batch size")
    row_ids = tuple(str(row.example.example_id) for row in rows)
    if row_ids != runtime.ordered_row_ids:
        raise InstrumentError("R5b runtime and shared reservation row order differ")
    task = CouplingTask(adapter, batch=batch)
    registry = complete_downstream_registry()
    projector_capability = runtime_projector_capability_audit(runtime)
    chunks: dict[str, list[Mapping[str, Any]]] = {"partner": [], "pseudo": []}
    for start in range(0, len(rows), batch):
        stop = min(start + batch, len(rows))
        local = rows[start:stop]
        readouts = [int(row.acceptor_g) for row in local]
        clean = task.encode([row.clean_sequence for row in local], readouts)
        sources = {
            "partner": task.encode(
                [row.paired_corrupt_sequence for row in local], readouts
            ),
            "pseudo": task.encode(
                [row.pseudo_corrupt_sequence for row in local], readouts
            ),
        }
        indices = list(range(start, stop))
        selected_bases = runtime.row_bases(indices)
        random_bases = runtime.random_row_bases(indices)
        for leg_id in ("partner", "pseudo"):
            chunks[leg_id].append(
                execute_conditional_leg(
                    adapter,
                    task,
                    clean_encoded=clean,
                    corrupt_encoded=sources[leg_id],
                    readout_positions=readouts,
                    projectors=selected_bases,
                    random_projectors=random_bases,
                    downstream_registry=registry,
                    leg_id=leg_id,
                )
            )
    partner_leg = concatenate_leg_chunks(chunks["partner"])
    pseudo_leg = concatenate_leg_chunks(chunks["pseudo"])
    families = derive_family_matrices(partner_leg, pseudo_leg)
    groups = [str(row.example.gene_id) for row in rows]
    if len(set(groups)) != R5_ROWS_PER_BUCKET:
        raise InstrumentError("R5c inference requires exactly one row per gene")
    inference = {
        family: strict_family_inference(
            family,
            families[family],
            groups,
            n_draws=n_draws,
            seed=seed,
            namespace=f"r5c-observed:{family}",
        )
        for family in FAMILY_NAMES
    }
    clean_partner = partner_leg["native_conditions"]["clean_margin"]
    clean_pseudo = pseudo_leg["native_conditions"]["clean_margin"]
    if not np.array_equal(clean_partner, clean_pseudo):
        raise InstrumentError("partner and pseudo legs do not share the native clean margin")
    native = native_behavior_inference(
        clean_partner - partner_leg["native_conditions"]["m_minus_margin"],
        clean_partner - pseudo_leg["native_conditions"]["m_minus_margin"],
        groups,
        n_draws=n_draws,
        seed=seed,
        namespace="r5c-native-entry-behavior",
    )
    health = run_bounded_hook_health(adapter, task, rows)
    self_patch_error = max(
        partner_leg["hook_capability"][
            "registered_query_self_patch_margin_max_abs_error"
        ],
        pseudo_leg["hook_capability"][
            "registered_query_self_patch_margin_max_abs_error"
        ],
    )
    self_patch_exact = all(
        leg["hook_capability"]["registered_query_self_patch_full_logits_exact"]
        for leg in (partner_leg, pseudo_leg)
    )
    energy_ok = all(
        item["per_row_energy_match_max_abs_error"] <= 1e-7
        and item["redrawn"] is False
        for leg in (partner_leg, pseudo_leg)
        for item in leg["random_u_a"]
    )
    residual_replay = {
        "partner": validate_residual_replay_registry(partner_leg),
        "pseudo": validate_residual_replay_registry(pseudo_leg),
    }
    mutation_audit = {
        "partner": validate_mutation_audit_registry(partner_leg),
        "pseudo": validate_mutation_audit_registry(pseudo_leg),
    }
    common_leg_capture = {
        "partner": validate_common_leg_capture_registry(partner_leg),
        "pseudo": validate_common_leg_capture_registry(pseudo_leg),
    }
    registered_heads = [
        head_name(head)
        for head in sorted(
            {L3_READER, DIRECT_NEGATIVE}
            | {head for ensemble in registry for head in ensemble}
        )
    ]
    self_patch_evidence = {
        "margin_max_abs_error": self_patch_error,
        "full_logits_exact": self_patch_exact,
        "l3_included": all(
            leg["hook_capability"]["l3_query_self_patch_included"]
            for leg in (partner_leg, pseudo_leg)
        ),
        "registered_heads_exact": all(
            leg["hook_capability"]["registered_heads"] == registered_heads
            for leg in (partner_leg, pseudo_leg)
        ),
    }
    random_projector_evidence = {
        "energy_match_max_abs_error": max(
            float(item["per_row_energy_match_max_abs_error"])
            for leg in (partner_leg, pseudo_leg)
            for item in leg["random_u_a"]
        ),
        "redrawn_count": sum(
            item.get("redrawn") is not False
            for leg in (partner_leg, pseudo_leg)
            for item in leg["random_u_a"]
        ),
        "draws_per_leg": 32,
    }
    capability_evidence = {
        "readout_validation": {
            "acceptor_g_mismatch_count": sum(
                row.clean_sequence[row.acceptor_g] != "G" for row in rows
            ),
            "rows_checked": len(rows),
        },
        "residual_replay": residual_replay,
        "self_patch": self_patch_evidence,
        "mutation_audit": mutation_audit,
        "common_leg_capture": common_leg_capture,
        "crossfit_projectors": projector_capability,
        "random_projectors": random_projector_evidence,
        "health": health,
        "diagnostics": {
            "binding_suite_status": binding_suite.get("status"),
            "engineering_calibration_status": engineering_calibration.get("status"),
            "production_trial_count": engineering_calibration.get(
                "production_trial_count"
            ),
        },
    }
    projector_validation = validate_projector_capability_audit(
        projector_capability,
        expected_runtime_array_sha256=runtime.authenticated_runtime_array_sha256,
    )
    capability_gates = {
        "single_masked_acceptor_G_readout": capability_evidence[
            "readout_validation"
        ]["acceptor_g_mismatch_count"]
        == 0,
        "no_masked_donor_endpoint": all(
            leg["masked_donor_endpoint_constructed"] is False
            for leg in (partner_leg, pseudo_leg)
        ),
        "full_mlm_logits_and_margin_replay": all(
            item["full_logits_exact"] is True and item["margin_max_abs_error"] <= 1e-5
            for item in residual_replay.values()
        ),
        "dense_full_residual_replay": all(
            item["condition_count"] > 0 and item["margin_max_abs_error"] <= 1e-5
            for item in residual_replay.values()
        ),
        "l3_query_self_patch_noop": self_patch_evidence["l3_included"]
        and self_patch_exact
        and self_patch_error <= 1e-5,
        "all_registered_head_query_self_patch_noop": self_patch_evidence[
            "registered_heads_exact"
        ]
        and self_patch_exact
        and self_patch_error <= 1e-5,
        "l3_c_plus_query_only_isolation": all(
            item["query_only"] for item in mutation_audit.values()
        ),
        "common_legs_bit_identical": all(
            item["components"] == len(registry)
            for item in common_leg_capture.values()
        ),
        "crossfit_projectors_valid": projector_validation.get("validated") == 33,
        "random_projectors_rank_energy_matched_without_redraw": energy_ok,
        "ensemble_membership_and_patch_size_exact": all(
            item["ensemble_membership_and_patch_size_exact"]
            for item in mutation_audit.values()
        ),
        "necessity_and_sufficiency_health_complete": health["all_arms_ok"],
        "boundary_binding_suite_pass": binding_suite.get("status") == "PASS",
        "engineering_calibration_pass": engineering_calibration.get("status") == "PASS"
        and engineering_calibration.get("production_trial_count") is True,
        "candidate_outcomes_not_used_for_capability": health.get(
            "candidate_outcomes_used"
        )
        is False
        and all(
            leg["hook_capability"]["candidate_outcomes_used"] is False
            for leg in (partner_leg, pseudo_leg)
        ),
    }
    capability_record = {
        "gates": capability_gates,
        "evidence": capability_evidence,
        "candidate_outcomes_used": False,
    }
    capability_status = None
    try:
        capability_status = validate_capability(
            capability_record,
            partner_leg=partner_leg,
            pseudo_leg=pseudo_leg,
            expected_runtime_array_sha256=runtime.authenticated_runtime_array_sha256,
        )
        instrument_ok = True
    except InstrumentError as exc:
        capability_status = {"status": "INSTRUMENT_UNRESOLVED", "reason": str(exc)}
        instrument_ok = False
    path_classes = {
        family: {
            path: result["classification"]
            for path, result in inference[family]["paths"].items()
        }
        for family in FAMILY_NAMES
    }
    controls_active = any(
        inference[family]["intervals"][row]["strict_classification"]
        in {"MATERIAL_POSITIVE", "MATERIAL_NEGATIVE"}
        for family in FAMILY_NAMES
        for row, requirement in {
            requirement.row: requirement
            for requirements in path_registry()[family].values()
            for requirement in requirements
        }.items()
        if requirement.decision == "EQUIVALENT_WITHIN_SESOI"
    )
    random_gap_rows = {
        "C2": [
            _u_gap_row(draw, estimand)
            for draw in range(32)
            for estimand in ("S_U", "N_U")
        ],
        "C4": [
            _gap_row(
                "C4",
                ensemble_name(PRIMARY_ACCEPTOR_PAIR),
                ensemble_name(control),
                estimand,
            )
            for control in random_pairs()
            for estimand in SANDWICH_ESTIMANDS
        ],
        "C5": [
            _gap_row(
                "C5",
                ensemble_name(SECONDARY_ACCEPTOR_TRIPLE),
                ensemble_name(control),
                estimand,
            )
            for control in random_triples()
            for estimand in SANDWICH_ESTIMANDS
        ],
    }
    random_gap_diagnostics = {
        family: {
            row: bool(
                inference[family]["row_boundary_tests"][row]["iut_p_value"]
                < FAMILY_ALPHA
                and any(
                    path["strict_simultaneous_ci_matches"].get(row) is True
                    for path in inference[family]["paths"].values()
                )
            )
            for row in names
        }
        for family, names in random_gap_rows.items()
    }
    randoms_beaten = all(
        passed
        for family in random_gap_diagnostics.values()
        for passed in family.values()
    )
    verdict = scientific_verdict(
        instrument_ok=instrument_ok,
        entry_behavior=native["classification"],
        c1=path_classes["C1"]["P_C1_L3_DIRECT"],
        c2=path_classes["C2"]["P_C2_L3_WRITES_U_A"],
        c3_paths=path_classes["C3"],
        c4=path_classes["C4"]["P_C4_PRIMARY_PAIR"],
        c5=path_classes["C5"]["P_C5_SECONDARY_TRIPLE"],
        controls_active=controls_active,
        randoms_beaten=randoms_beaten,
    )
    return {
        "groups": groups,
        "ordered_gene_ids_sha256": canonical_sha256(groups),
        "native_behavior": native,
        "partner_leg": _json_ready(partner_leg),
        "pseudo_leg": _json_ready(pseudo_leg),
        "families": {
            family: {
                "row_names": list(family_row_names()[family]),
                "raw_values": families[family].tolist(),
                "raw_values_sha256": array_sha256(families[family]),
                "inference": inference[family],
                "inference_sha256": canonical_sha256(inference[family]),
            }
            for family in FAMILY_NAMES
        },
        "gene_sufficient_statistics": {
            "one_row_per_gene": True,
            "n_genes": len(groups),
            "raw_gene_values_stored_in_families": True,
            "family": {
                family: {
                    "row_sum": np.sum(families[family], axis=1).tolist(),
                    "row_sum_squares": np.sum(
                        np.square(families[family]), axis=1
                    ).tolist(),
                    "raw_values_sha256": array_sha256(families[family]),
                }
                for family in FAMILY_NAMES
            },
        },
        "capability": capability_record,
        "capability_status": capability_status,
        "instrument_ok": instrument_ok,
        "path_classifications": path_classes,
        "controls_active": controls_active,
        "random_gap_diagnostics": random_gap_diagnostics,
        "randoms_beaten": randoms_beaten,
        "verdict": verdict,
    }


def model_free_registry_audit() -> dict[str, Any]:
    ensembles = random_ensemble_registry()
    flat_random = [item for values in ensembles.values() for item in values]
    if len(flat_random) != 32 or len({ensemble_name(item) for item in flat_random}) != 32:
        raise InstrumentError("random ensemble identities are incomplete or duplicated")
    if any(set(item) & REGISTERED_DOWNSTREAM_HEADS for item in flat_random):
        raise InstrumentError("a random ensemble contains a registered downstream head")
    rows = family_row_names()
    if {name: len(rows[name]) for name in rows} != EXPECTED_FAMILY_SIZES:
        raise InstrumentError("R5c family cardinality drift")
    if any(len(set(rows[name])) != len(rows[name]) for name in FAMILY_NAMES):
        raise InstrumentError("an R5c family has duplicate row IDs")
    paths = path_registry()
    for family in FAMILY_NAMES:
        allowed = set(rows[family])
        if not paths[family]:
            raise InstrumentError(f"{family} has no declared paths")
        for path, requirements in paths[family].items():
            if not requirements or any(requirement.row not in allowed for requirement in requirements):
                raise InstrumentError(f"{path} refers outside its exact family")
    calibration = calibration_registry()
    return {
        "schema": AUDIT_SCHEMA,
        "status": "DRAFT_MODEL_FREE_REGISTRY_AUDIT_PASS",
        "model_loaded": False,
        "annotation_read": False,
        "fasta_read": False,
        "cohort_read": False,
        "chr8_chr9_opened": False,
        "readout": "single masked acceptor G hard margin",
        "masked_donor_endpoint_constructed": False,
        "entry_order": ["R3c", "R5a", "R5b", "R5c"],
        "shared_cohort": {
            "schema": R5_EXPECTED_CENSUS_IDENTITY["schema"],
            "expected_census_identity_sha256": canonical_sha256(
                R5_EXPECTED_CENSUS_IDENTITY
            ),
            "expected_census_identity": R5_EXPECTED_CENSUS_IDENTITY,
            "validated_by": "validate_expected_r5_census_identity",
        },
        "shared_sources": {
            "expected_source_identity": R5_EXPECTED_SOURCE_IDENTITY,
            "expected_source_identity_sha256": canonical_sha256(
                R5_EXPECTED_SOURCE_IDENTITY
            ),
            "validated_by": "validate_expected_r5_source_identity",
            "comprehensive_gtf_required": True,
        },
        "fixed_heads": {
            "reader": head_name(L3_READER),
            "direct_negative": head_name(DIRECT_NEGATIVE),
            "acceptor_candidates": [head_name(item) for item in ACCEPTOR_CANDIDATES],
            "donor_route_comparators": [head_name(item) for item in DONOR_ROUTE_COMPARATORS],
        },
        "random_ensembles": {
            key: [ensemble_name(item) for item in values] for key, values in ensembles.items()
        },
        "random_ensemble_registry_sha256": canonical_sha256(
            {key: [ensemble_name(item) for item in values] for key, values in ensembles.items()}
        ),
        "families": {
            name: {
                "n_rows": len(rows[name]),
                "row_order_sha256": canonical_sha256(rows[name]),
                "path_names": list(paths[name]),
                "path_registry_sha256": canonical_sha256(
                    {
                        path: [requirement.__dict__ for requirement in requirements]
                        for path, requirements in paths[name].items()
                    }
                ),
            }
            for name in FAMILY_NAMES
        },
        "calibration_registry": calibration,
        "capability_gate_names": list(CAPABILITY_GATES),
        "capability_gate_registry_sha256": canonical_sha256(CAPABILITY_GATES),
    }


def normalized_runner_sha256(text: str | None = None) -> str:
    source = RUNNER.read_text() if text is None else str(text)
    pin_names = (
        "FROZEN_PROTOCOL_SHA256",
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256",
        "FROZEN_PREFLIGHT_ARTIFACT_SHA256",
        "FROZEN_R3C_ENTRY_ARTIFACT_SHA256",
        "FROZEN_R5A_ENTRY_ARTIFACT_SHA256",
        "FROZEN_R5B_ENTRY_ARTIFACT_SHA256",
    )
    for name in pin_names:
        source, count = re.subn(
            rf"^{name}: str \| None = .*$",
            f"{name}: str | None = <FROZEN_PIN>",
            source,
            flags=re.MULTILINE,
        )
        if count != 1:
            raise RuntimeError(f"normalized runner could not locate exactly one {name}")
    return hashlib.sha256(source.encode()).hexdigest()


def instrument_binding_payload() -> dict[str, Any]:
    audit = model_free_registry_audit()
    stack = build_r5_stack_binding(repo=REPO)
    return {
        "schema": "gfmcd.r5c.instrument_binding/1",
        STACK_BINDING_FIELD: stack,
        STACK_FINGERPRINT_FIELD: r5_stack_fingerprint(stack),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "normalized_runner_sha256": normalized_runner_sha256(),
        "stat_core_sha256": hashlib.sha256(STAT_CORE.read_bytes()).hexdigest(),
        "mechanism_core_sha256": hashlib.sha256(MECHANISM_CORE.read_bytes()).hexdigest(),
        "cohort_core_sha256": hashlib.sha256(COHORT_CORE.read_bytes()).hexdigest(),
        "hook_core_sha256": hashlib.sha256(HOOK_CORE.read_bytes()).hexdigest(),
        "circuit_core_sha256": hashlib.sha256(CIRCUIT_CORE.read_bytes()).hexdigest(),
        "coupling_core_sha256": hashlib.sha256(COUPLING_CORE.read_bytes()).hexdigest(),
        "reverse_coupling_core_sha256": hashlib.sha256(
            REVERSE_COUPLING_CORE.read_bytes()
        ).hexdigest(),
        "adapter_core_sha256": hashlib.sha256(ADAPTER_CORE.read_bytes()).hexdigest(),
        "provenance_core_sha256": hashlib.sha256(
            PROVENANCE_CORE.read_bytes()
        ).hexdigest(),
        "registry_audit_sha256": canonical_sha256(audit),
        "runtime_identity": stable_runtime_identity(),
        "production_randomisation": PRODUCTION_RANDOMISATION,
        "seed": PRODUCTION_SEED,
        "sesoi": SESOI,
        "family_alpha": FAMILY_ALPHA,
    }


def stable_runtime_identity() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in (
        "numpy",
        "scipy",
        "torch",
        "transformers",
        "tokenizers",
        "pyfaidx",
    ):
        try:
            packages[package] = package_version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def authorize_target() -> dict[str, str]:
    pins = {
        "FROZEN_PROTOCOL_SHA256": FROZEN_PROTOCOL_SHA256,
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256": FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
        "FROZEN_PREFLIGHT_ARTIFACT_SHA256": FROZEN_PREFLIGHT_ARTIFACT_SHA256,
        "FROZEN_R3C_ENTRY_ARTIFACT_SHA256": FROZEN_R3C_ENTRY_ARTIFACT_SHA256,
        "FROZEN_R5A_ENTRY_ARTIFACT_SHA256": FROZEN_R5A_ENTRY_ARTIFACT_SHA256,
        "FROZEN_R5B_ENTRY_ARTIFACT_SHA256": FROZEN_R5B_ENTRY_ARTIFACT_SHA256,
    }
    missing = [name for name, value in pins.items() if value is None]
    if missing:
        raise TargetSealError(
            "R5c target is hard closed: literal seals remain None ("
            + ", ".join(missing)
            + ")"
        )
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in pins.values()):
        raise TargetSealError("R5c target seal has a malformed SHA-256")
    payload = instrument_binding_payload()
    observed_protocol = payload["protocol_sha256"]
    observed_fingerprint = canonical_sha256(payload)
    if observed_protocol != FROZEN_PROTOCOL_SHA256:
        raise TargetSealError("R5c protocol hash differs from the literal freeze")
    if observed_fingerprint != FROZEN_INSTRUMENT_FINGERPRINT_SHA256:
        raise TargetSealError("R5c instrument fingerprint differs from the literal freeze")
    return {name: str(value) for name, value in pins.items()}


def authorize_preflight() -> dict[str, str]:
    """Authorize a bindable bucket-0 run without requiring its future output hash."""

    pins = {
        "FROZEN_PROTOCOL_SHA256": FROZEN_PROTOCOL_SHA256,
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256": FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
        "FROZEN_R3C_ENTRY_ARTIFACT_SHA256": FROZEN_R3C_ENTRY_ARTIFACT_SHA256,
        "FROZEN_R5A_ENTRY_ARTIFACT_SHA256": FROZEN_R5A_ENTRY_ARTIFACT_SHA256,
        "FROZEN_R5B_ENTRY_ARTIFACT_SHA256": FROZEN_R5B_ENTRY_ARTIFACT_SHA256,
    }
    missing = [name for name, value in pins.items() if value is None]
    if missing:
        raise TargetSealError(
            "R5c preflight is hard closed: literal seals remain None ("
            + ", ".join(missing)
            + ")"
        )
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in pins.values()):
        raise TargetSealError("R5c preflight seal has a malformed SHA-256")
    payload = instrument_binding_payload()
    if payload["protocol_sha256"] != FROZEN_PROTOCOL_SHA256:
        raise TargetSealError("R5c protocol hash differs from the literal freeze")
    if canonical_sha256(payload) != FROZEN_INSTRUMENT_FINGERPRINT_SHA256:
        raise TargetSealError("R5c instrument fingerprint differs from the literal freeze")
    return {name: str(value) for name, value in pins.items()}


def require_clean_git() -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InstrumentError(f"cannot establish git identity: {exc}") from exc
    if status.strip():
        raise InstrumentError("R5c bindable execution requires a clean worktree")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise InstrumentError("git commit identity is malformed")
    return {"commit": commit, "worktree": "clean"}


def checkpoint_identity(model: str | Path) -> dict[str, Any]:
    root = Path(model).expanduser().resolve()
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(path for path in root.rglob("*") if path.is_file())
    else:
        raise InstrumentError("model checkpoint path does not exist")
    if not files:
        raise InstrumentError("model checkpoint contains no files")
    records = [
        {
            "relative_path": str(path.relative_to(root)) if root.is_dir() else path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    return {
        "root": str(root),
        "files": records,
        "checkpoint_aggregate_sha256": canonical_sha256(records),
    }


def stage_checkpoint_bundle(
    model: str | Path, identity: Mapping[str, Any]
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    source = Path(model).expanduser().resolve()
    staging = tempfile.TemporaryDirectory(prefix="gfmcd-r5c-checkpoint-")
    root = Path(staging.name) / ("model" if source.is_dir() else source.name)
    try:
        if source.is_dir():
            root.mkdir(parents=True)
            for record in identity["files"]:
                relative = Path(record["relative_path"])
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / relative, target)
        else:
            shutil.copyfile(source, root)
        staged = checkpoint_identity(root)
        if staged["checkpoint_aggregate_sha256"] != identity[
            "checkpoint_aggregate_sha256"
        ]:
            raise InstrumentError("private staged checkpoint differs from bound source bytes")
        return staging, root
    except Exception:
        staging.cleanup()
        raise


def validate_preflight_replay(
    record: Mapping[str, Any], *, fingerprint: str
) -> dict[str, Any]:
    current_stack = build_r5_stack_binding(repo=REPO)
    try:
        validate_serialized_r5_stack(record, current=current_stack)
    except R5StackBindingError as exc:
        raise InstrumentError(f"R5c preflight joint stack differs: {exc}") from exc
    if (
        record.get("schema") != SCHEMA
        or record.get("stage") != "preflight"
        or record.get("bucket") != 0
        or record.get("instrument_fingerprint_sha256") != fingerprint
        or record.get("sealed_chr8_chr9_opened") is not False
    ):
        raise InstrumentError("R5c preflight identity differs")
    binding = record.get("instrument_binding")
    registry_audit = record.get("registry_audit")
    if (
        not isinstance(binding, Mapping)
        or dict(binding) != instrument_binding_payload()
        or canonical_sha256(binding) != fingerprint
        or not isinstance(registry_audit, Mapping)
        or canonical_sha256(registry_audit)
        != binding.get("registry_audit_sha256")
    ):
        raise InstrumentError("R5c preflight instrument/registry binding differs")
    measurement = record.get("measurement")
    if not isinstance(measurement, Mapping):
        raise InstrumentError("R5c preflight lacks its measurement record")
    capability_status = measurement.get("capability_status")
    if (
        measurement.get("instrument_ok") is not True
        or not isinstance(capability_status, Mapping)
        or capability_status.get("status") != "R5C_CAPABILITY_CONFIRMED"
    ):
        raise InstrumentError("R5c preflight capability status is inconsistent")
    binding = record.get("binding_suite")
    calibration = record.get("engineering_calibration")
    if not isinstance(binding, Mapping) or not isinstance(calibration, Mapping):
        raise InstrumentError("R5c preflight lacks diagnostic records")
    diagnostic_validation = validate_binding_and_calibration_records(
        binding,
        calibration,
    )
    groups = measurement.get("groups")
    if (
        not isinstance(groups, list)
        or len(groups) != R5_ROWS_PER_BUCKET
        or len(set(map(str, groups))) != R5_ROWS_PER_BUCKET
        or canonical_sha256(groups) != measurement.get("ordered_gene_ids_sha256")
    ):
        raise InstrumentError("R5c preflight lacks exact inference groups")
    partner_leg = measurement.get("partner_leg")
    pseudo_leg = measurement.get("pseudo_leg")
    if not isinstance(partner_leg, Mapping) or not isinstance(pseudo_leg, Mapping):
        raise InstrumentError("R5c preflight lacks primitive partner/pseudo legs")
    runtime_contract = record.get("r5b_runtime_contract")
    expected_runtime_hashes = (
        runtime_contract.get("runtime_array_sha256")
        if isinstance(runtime_contract, Mapping)
        else None
    )
    if not isinstance(expected_runtime_hashes, Mapping):
        raise InstrumentError("R5c preflight lacks authenticated runtime array hashes")
    if runtime_contract.get(STACK_FINGERPRINT_FIELD) != record.get(
        STACK_FINGERPRINT_FIELD
    ):
        raise InstrumentError("R5c preflight runtime differs from the joint R5 stack")
    validate_capability(
        measurement.get("capability", {}),
        partner_leg=partner_leg,
        pseudo_leg=pseudo_leg,
        expected_runtime_array_sha256=expected_runtime_hashes,
    )
    try:
        rederived_families = derive_family_matrices(partner_leg, pseudo_leg)
    except (KeyError, TypeError, ValueError, InstrumentError) as exc:
        raise InstrumentError(f"R5c preflight primitive replay failed: {exc}") from exc
    family_records = measurement.get("families")
    if not isinstance(family_records, Mapping):
        raise InstrumentError("R5c preflight lacks its family records")
    sufficient = measurement.get("gene_sufficient_statistics")
    if (
        not isinstance(sufficient, Mapping)
        or sufficient.get("one_row_per_gene") is not True
        or sufficient.get("n_genes") != R5_ROWS_PER_BUCKET
        or sufficient.get("raw_gene_values_stored_in_families") is not True
        or not isinstance(sufficient.get("family"), Mapping)
    ):
        raise InstrumentError("R5c preflight lacks exact gene sufficient statistics")
    replay_hashes: dict[str, str] = {}
    for family in FAMILY_NAMES:
        family_record = family_records.get(family)
        if not isinstance(family_record, Mapping):
            raise InstrumentError(f"R5c preflight lacks {family}")
        values = np.asarray(family_record.get("raw_values"), dtype=float)
        if values.shape != (
            EXPECTED_FAMILY_SIZES[family],
            R5_ROWS_PER_BUCKET,
        ) or array_sha256(values) != family_record.get("raw_values_sha256"):
            raise InstrumentError(f"R5c preflight {family} raw values differ")
        if (
            family_record.get("row_names") != list(family_row_names()[family])
            or not np.array_equal(values, rederived_families[family])
        ):
            raise InstrumentError(
                f"R5c preflight {family} values do not rederive from primitive legs"
            )
        expected_sufficient = {
            "row_sum": np.sum(rederived_families[family], axis=1).tolist(),
            "row_sum_squares": np.sum(
                np.square(rederived_families[family]), axis=1
            ).tolist(),
            "raw_values_sha256": array_sha256(rederived_families[family]),
        }
        if sufficient["family"].get(family) != expected_sufficient:
            raise InstrumentError(
                f"R5c preflight {family} gene sufficient statistics differ"
            )
        replay = strict_family_inference(
            family,
            rederived_families[family],
            groups,
            n_draws=PRODUCTION_RANDOMISATION,
            seed=PRODUCTION_SEED,
            namespace=f"r5c-observed:{family}",
        )
        digest = canonical_sha256(replay)
        if (
            digest != family_record.get("inference_sha256")
            or canonical_sha256(family_record.get("inference")) != digest
        ):
            raise InstrumentError(f"R5c preflight {family} inference replay differs")
        replay_hashes[family] = digest
    try:
        clean_partner = np.asarray(
            partner_leg["native_conditions"]["clean_margin"], dtype=float
        )
        clean_pseudo = np.asarray(
            pseudo_leg["native_conditions"]["clean_margin"], dtype=float
        )
        if not np.array_equal(clean_partner, clean_pseudo):
            raise InstrumentError("R5c preflight native clean legs differ")
        native_replay = native_behavior_inference(
            clean_partner
            - np.asarray(
                partner_leg["native_conditions"]["m_minus_margin"], dtype=float
            ),
            clean_partner
            - np.asarray(
                pseudo_leg["native_conditions"]["m_minus_margin"], dtype=float
            ),
            groups,
            n_draws=PRODUCTION_RANDOMISATION,
            seed=PRODUCTION_SEED,
            namespace="r5c-native-entry-behavior",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InstrumentError(f"R5c preflight native replay failed: {exc}") from exc
    if canonical_sha256(native_replay) != canonical_sha256(
        measurement.get("native_behavior")
    ):
        raise InstrumentError("R5c preflight native inference replay differs")
    return {
        "status": "FROZEN_PREFLIGHT_REPLAY_PASS",
        "artifact_sha256": str(FROZEN_PREFLIGHT_ARTIFACT_SHA256),
        "family_inference_sha256": replay_hashes,
        "native_inference_sha256": canonical_sha256(native_replay),
        "primitive_family_rederivation_pass": True,
        "binding_suite_sha256": canonical_sha256(binding),
        "engineering_calibration_sha256": canonical_sha256(calibration),
        "diagnostic_record_validation": diagnostic_validation,
    }


def scientific_verdict(
    *,
    instrument_ok: bool,
    entry_behavior: str,
    c1: str,
    c2: str,
    c3_paths: Mapping[str, str],
    c4: str,
    c5: str,
    controls_active: bool,
    randoms_beaten: bool,
) -> str:
    """Keep healthy scientific negatives distinct from instrument failures."""

    if not instrument_ok:
        return "INSTRUMENT_UNRESOLVED"
    if entry_behavior == "NOT_REPLICATED":
        return "ENTRY_BEHAVIOR_NOT_REPLICATED"
    if entry_behavior == "NONSPECIFIC":
        return "ENTRY_BEHAVIOR_NONSPECIFIC"
    path_classes = [c1, c2, c4, c5, *c3_paths.values()]
    if "INSTRUMENT_UNRESOLVED" in path_classes:
        return "INSTRUMENT_UNRESOLVED"
    if any(value.startswith("SCIENTIFICALLY_UNRESOLVED") for value in path_classes):
        return "SCIENTIFICALLY_UNRESOLVED"
    if c1 != "SUPPORTED":
        return "L3_RESCUE_NOT_REPLICATED"
    if controls_active:
        return "NONSELECTIVE_OR_SHARED_DELIVERY"
    if c2 != "SUPPORTED":
        return (
            "WRITE_NOT_DISTINGUISHED_FROM_MATCHED_NULLS"
            if not randoms_beaten
            else "L3_WRITE_NOT_SUPPORTED"
        )
    if c4 != "SUPPORTED" or c5 != "SUPPORTED" or "SUPPORTED" not in set(c3_paths.values()):
        return (
            "NOT_DISTINGUISHED_FROM_MATCHED_NULLS"
            if not randoms_beaten
            else "TESTED_HEAD_SET_DISTINCT_DELIVERY_NOT_SUPPORTED"
        )
    if not randoms_beaten:
        return "NOT_DISTINGUISHED_FROM_MATCHED_NULLS"
    return "R5C_WRITE_AND_TESTED_HEAD_SET_DISTINCT_DELIVERY_SUPPORTED"


def _write_fresh_json(path: Path, record: Mapping[str, Any]) -> None:
    manifest_path = path.with_suffix(".manifest.json")
    if path.exists() or manifest_path.exists():
        raise SystemExit("R5c audit outputs are immutable; choose a fresh path")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    manifest = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r5c_model_free_registry_audit",
        sys.argv,
        repo=REPO,
        status=str(record.get("status")),
        config={
            "model_loaded": False,
            "annotation_read": False,
            "fasta_read": False,
            "chr8_chr9_opened": False,
            "instrument_fingerprint_sha256": record.get(
                "instrument_fingerprint_sha256"
            ),
        },
        seeds={"registry": PRODUCTION_SEED},
        notes=[
            "Model-free R5c registry/boundary/calibration audit only.",
            "No model, annotation, FASTA, cohort rows, activation, or chr8/9 was read.",
        ],
    )
    manifest.inputs = [
        {
            "path": str(input_path),
            "size_bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
        }
        for input_path in (
            PROTOCOL,
            RUNNER,
            STAT_CORE,
            MECHANISM_CORE,
            COHORT_CORE,
            HOOK_CORE,
            CIRCUIT_CORE,
            COUPLING_CORE,
            REVERSE_COUPLING_CORE,
            ADAPTER_CORE,
            PROVENANCE_CORE,
        )
    ]
    manifest.outputs = [
        {
            "path": str(path),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    manifest_bytes = (
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        with manifest_path.open("xb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise SystemExit("R5c audit outputs are immutable; choose a fresh path") from exc


def write_result_package(
    output: Path,
    result: Mapping[str, Any],
    *,
    input_paths: Sequence[str | Path],
    git_identity: Mapping[str, str],
) -> tuple[str, str]:
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() or manifest_path.exists():
        raise InstrumentError("R5c outputs are immutable; choose a fresh versioned path")
    output.parent.mkdir(parents=True, exist_ok=True)
    result_bytes = (json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n").encode()
    result_sha = hashlib.sha256(result_bytes).hexdigest()
    manifest = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r5c_acceptor_write_delivery",
        sys.argv,
        repo=REPO,
        status=str(result.get("status")),
        config={
            "stage": result.get("stage"),
            "bucket": result.get("bucket"),
            "instrument_fingerprint_sha256": result.get(
                "instrument_fingerprint_sha256"
            ),
            "family_row_registry_sha256": canonical_sha256(family_row_names()),
            "path_registry_sha256": canonical_sha256(
                {
                    family: {
                        path: [requirement.__dict__ for requirement in requirements]
                        for path, requirements in paths.items()
                    }
                    for family, paths in path_registry().items()
                }
            ),
            "runtime_bundle_sha256": result.get("runtime_bundle_sha256"),
            "execution_device": result.get("execution_device"),
        },
        seeds={"randomisation": PRODUCTION_SEED, "health_probe_selection": PRODUCTION_SEED},
        notes=[
            "Single masked acceptor G hard margin only; no masked-donor endpoint.",
            "Exact conditional c+ rescue, cross-fitted U_A N/S, and downstream N/S sandwich.",
            "One-, two-, three-head and residual-projector families remain separate.",
            "Boundary tests, TOST/IUT, Holm, and strict simultaneous CIs determine paths.",
            "Validation chromosomes chr11/16/19 only; chr8/9 remains closed.",
        ],
    )
    manifest.git_commit = git_identity["commit"]
    manifest.git_dirty = False
    manifest.inputs = [
        {
            "path": str(Path(path).expanduser().resolve()),
            "size_bytes": Path(path).expanduser().resolve().stat().st_size,
            "sha256": sha256_file(Path(path).expanduser().resolve()),
        }
        for path in input_paths
        if Path(path).expanduser().resolve().is_file()
    ]
    manifest.outputs = [
        {"path": str(output), "size_bytes": len(result_bytes), "sha256": result_sha}
    ]
    manifest_bytes = (
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode()
    try:
        with output.open("xb") as handle:
            handle.write(result_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        with manifest_path.open("xb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise InstrumentError("R5c output race detected; no overwrite permitted") from exc
    return result_sha, hashlib.sha256(manifest_bytes).hexdigest()


def expected_r5_binding(bucket: int) -> dict[str, Any]:
    if not _is_integer(bucket) or int(bucket) not in (0, 1):
        raise InstrumentError("R5 expected binding bucket must be literal 0 or 1")
    selected = R5_EXPECTED_CENSUS_IDENTITY["selected_by_bucket"][str(int(bucket))]
    return {
        "schema": R5_EXPECTED_CENSUS_IDENTITY["schema"],
        "bucket": int(bucket),
        "rows": selected["rows"],
        "genes": selected["genes"],
        "gene_ids_sha256": selected["gene_ids_sha256"],
        "ordered_rows_sha256": selected["ordered_rows_sha256"],
        "construction_sha256": selected["construction_sha256"],
        "full_census_identity_sha256": canonical_sha256(
            R5_EXPECTED_CENSUS_IDENTITY
        ),
    }


def validate_entry_target_binding(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Require R5a/R5b eligibility to refer to the fixed evaluation reservation."""

    expected = expected_r5_binding(1)
    if entry.get("cohort_sha256") != expected["ordered_rows_sha256"]:
        raise InstrumentError("R5a/R5b entry cohort is not the fixed bucket-1 reservation")
    if entry.get("construction_sha256") != expected["construction_sha256"]:
        raise InstrumentError("R5a/R5b entry construction is not the fixed bucket-1 construction")
    return expected


def _entry_artifact_hashes() -> dict[str, str]:
    return {
        "r3c": str(FROZEN_R3C_ENTRY_ARTIFACT_SHA256),
        "r5a": str(FROZEN_R5A_ENTRY_ARTIFACT_SHA256),
        "r5b": str(FROZEN_R5B_ENTRY_ARTIFACT_SHA256),
    }


def _artifact_paths(path: str | Path) -> tuple[Path, Path]:
    artifact = Path(path).expanduser().resolve()
    return artifact, artifact.with_suffix(".manifest.json")


def _snapshot_files(paths: Sequence[str | Path]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise InstrumentError(f"bound input is not a regular file: {path}")
        key = str(path)
        if key in snapshot:
            continue
        snapshot[key] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return snapshot


def _verify_file_snapshot(snapshot: Mapping[str, Mapping[str, Any]]) -> None:
    for raw, expected in snapshot.items():
        path = Path(raw)
        if (
            not path.is_file()
            or path.stat().st_size != expected.get("size_bytes")
            or sha256_file(path) != expected.get("sha256")
        ):
            raise InstrumentError(f"bound input changed during R5c execution: {path}")


def _checkpoint_file_paths(identity: Mapping[str, Any]) -> list[Path]:
    root = Path(str(identity["root"]))
    if root.is_file():
        return [root]
    return [root / str(record["relative_path"]) for record in identity["files"]]


def validate_preflight_context(
    record: Mapping[str, Any],
    *,
    entry_authorization: Mapping[str, Any],
    checkpoint_sha256: str,
    runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a replayed bucket-0 preflight to the current upstream/input context."""

    if record.get("status") != "PREFLIGHT_INSTRUMENT_PASS":
        raise InstrumentError("R5c preflight did not end in an instrument pass")
    entry = record.get("entry")
    if not isinstance(entry, Mapping):
        raise InstrumentError("R5c preflight lacks its upstream entry binding")
    if entry.get("artifact_sha256") != _entry_artifact_hashes():
        raise InstrumentError("R5c preflight upstream artifact pins differ")
    if entry.get("authorization") != dict(entry_authorization):
        raise InstrumentError("R5c preflight upstream authorization differs")
    checkpoint = record.get("checkpoint")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("checkpoint_aggregate_sha256") != checkpoint_sha256
    ):
        raise InstrumentError("R5c preflight checkpoint differs")
    if record.get("source_identity") != R5_EXPECTED_SOURCE_IDENTITY:
        raise InstrumentError("R5c preflight source identity differs")
    if record.get("runtime_identity") != stable_runtime_identity():
        raise InstrumentError("R5c preflight software/runtime identity differs")
    cohort = record.get("cohort")
    if (
        not isinstance(cohort, Mapping)
        or cohort.get("binding") != expected_r5_binding(0)
    ):
        raise InstrumentError("R5c preflight bucket-0 cohort binding differs")
    if record.get("r5b_runtime_contract") != dict(runtime_contract):
        raise InstrumentError("R5c preflight R5b runtime contract differs")
    if record.get("runtime_bundle_sha256") != runtime_contract.get(
        "runtime_bundle_sha256"
    ):
        raise InstrumentError("R5c preflight runtime bundle digest differs")
    return {
        "status": "FROZEN_PREFLIGHT_CONTEXT_PASS",
        "entry_authorization_sha256": canonical_sha256(entry_authorization),
        "runtime_contract_sha256": canonical_sha256(runtime_contract),
        "checkpoint_sha256": checkpoint_sha256,
        "cohort_binding_sha256": canonical_sha256(expected_r5_binding(0)),
    }


def _required_model_arguments(args: argparse.Namespace) -> None:
    required = {
        "r3c artifact": args.r3c_artifact,
        "R5a artifact": args.r5a_artifact,
        "R5b artifact": args.r5b_artifact,
        "R5b runtime bundle": args.r5b_runtime,
        "output": args.out,
    }
    if args.stage == "target":
        required["preflight artifact"] = args.preflight_artifact
    missing = [label for label, value in required.items() if value is None]
    if missing:
        raise InstrumentError(
            f"R5c {args.stage} requires: " + ", ".join(missing)
        )
    if (
        args.batch != PRODUCTION_BATCH
        or args.draws != PRODUCTION_RANDOMISATION
        or args.seed != PRODUCTION_SEED
        or args.calibration_trials != computed_calibration_trials()
    ):
        raise InstrumentError(
            "R5c production batch/randomisation/seed/calibration parameters are fixed"
        )


def run_authorized_model_stage(
    args: argparse.Namespace,
    *,
    authorization: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Execute an already-authorized preflight or target end to end."""

    if args.stage not in {"preflight", "target"}:
        raise InstrumentError("authorized model execution requires preflight or target")
    _required_model_arguments(args)
    output = Path(args.out).expanduser().resolve()
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise InstrumentError("R5c outputs are immutable; choose a fresh versioned path")

    git_before = require_clean_git()
    binding_payload = instrument_binding_payload()
    fingerprint = canonical_sha256(binding_payload)
    if fingerprint != authorization.get("FROZEN_INSTRUMENT_FINGERPRINT_SHA256"):
        raise InstrumentError("authorized R5c fingerprint changed before execution")

    r3c, r5a, r5b, entry_authorization = load_and_validate_entry_artifacts(
        r3c_path=args.r3c_artifact,
        r5a_path=args.r5a_artifact,
        r5b_path=args.r5b_artifact,
    )
    validate_entry_target_binding(entry_authorization)

    preflight_record: Mapping[str, Any] | None = None
    preflight_replay: Mapping[str, Any] | None = None
    preflight_context: Mapping[str, Any] | None = None
    if args.stage == "target":
        bucket0_contract = r5b_runtime_contract_for_bucket(
            r5b,
            bucket=0,
            cohort_binding=expected_r5_binding(0),
            entry_authorization=entry_authorization,
        )
        preflight_record, _ = _load_exact_json_artifact(
            args.preflight_artifact,
            expected_sha256=str(FROZEN_PREFLIGHT_ARTIFACT_SHA256),
            label="R5c preflight",
        )
        preflight_replay = validate_preflight_replay(
            preflight_record,
            fingerprint=fingerprint,
        )
        preflight_context = validate_preflight_context(
            preflight_record,
            entry_authorization=entry_authorization,
            checkpoint_sha256=str(entry_authorization["checkpoint_sha256"]),
            runtime_contract=bucket0_contract,
        )
        binding_suite = preflight_record["binding_suite"]
        engineering_calibration = preflight_record["engineering_calibration"]
    else:
        # These operate only on coherent synthetic primitives and the production
        # classifier.  They must pass before any model/data path is opened.
        binding_suite = run_constituent_binding_suite(
            n_draws=PRODUCTION_RANDOMISATION,
            seed=PRODUCTION_SEED,
        )
        engineering_calibration = run_engineering_calibration(
            n_draws=PRODUCTION_RANDOMISATION,
            seed=PRODUCTION_SEED,
            n_trials=computed_calibration_trials(),
        )
        validate_binding_and_calibration_records(
            binding_suite,
            engineering_calibration,
        )

    bucket = 0 if args.stage == "preflight" else 1
    rows, cohort = load_shared_r5_reservation(
        gtf=args.gtf,
        fasta=args.fasta,
        r3c_record=r3c,
        bucket=bucket,
    )
    if cohort["binding"] != expected_r5_binding(bucket):
        raise InstrumentError("reconstructed R5 cohort differs from its literal census")
    if (
        len(rows) != R5_ROWS_PER_BUCKET
        or len({str(row.example.gene_id) for row in rows}) != R5_ROWS_PER_BUCKET
        or {str(row.example.strand) for row in rows} != {"+", "-"}
        or any(str(row.example.chrom) not in R5_VALIDATION_CHROMOSOMES for row in rows)
    ):
        raise InstrumentError("R5c reservation is not 128 one-row-per-gene, both-strand rows")
    row_ids = tuple(str(row.example.example_id) for row in rows)
    runtime_contract = r5b_runtime_contract_for_bucket(
        r5b,
        bucket=bucket,
        cohort_binding=cohort["binding"],
        entry_authorization=entry_authorization,
    )
    runtime = load_r5b_runtime_bundle(
        args.r5b_runtime,
        r5b_record=runtime_contract,
        ordered_row_ids=row_ids,
    )
    runtime_validation = validate_runtime_fold_assignment(rows, runtime)

    checkpoint = checkpoint_identity(args.model)
    if checkpoint["checkpoint_aggregate_sha256"] != entry_authorization[
        "checkpoint_sha256"
    ]:
        raise InstrumentError("model checkpoint differs from the complete entry chain")
    if checkpoint["checkpoint_aggregate_sha256"] != runtime_contract[
        "checkpoint_sha256"
    ]:
        raise InstrumentError("model checkpoint differs from the R5b runtime contract")

    artifact_inputs: list[Path] = []
    for value in (args.r3c_artifact, args.r5a_artifact, args.r5b_artifact):
        artifact_inputs.extend(_artifact_paths(value))
    if args.stage == "target":
        artifact_inputs.extend(_artifact_paths(args.preflight_artifact))
    runtime_path = Path(args.r5b_runtime).expanduser().resolve()
    small_input_paths = artifact_inputs + [
        runtime_path,
        PROTOCOL,
        RUNNER,
        STAT_CORE,
        MECHANISM_CORE,
        COHORT_CORE,
        HOOK_CORE,
        CIRCUIT_CORE,
        COUPLING_CORE,
        REVERSE_COUPLING_CORE,
        ADAPTER_CORE,
        PROVENANCE_CORE,
    ]
    input_snapshot = _snapshot_files(small_input_paths)

    staging: tempfile.TemporaryDirectory[str] | None = None
    try:
        staging, staged_model = stage_checkpoint_bundle(args.model, checkpoint)
        device = pick_device(args.device)
        if (
            args.stage == "target"
            and preflight_record is not None
            and preflight_record.get("execution_device") != str(device)
        ):
            raise InstrumentError("R5c target execution device differs from preflight")
        adapter = load_adapter(str(staged_model), device, eager=False)
        adapter.model.eval()
        torch.set_grad_enabled(False)
        with torch.no_grad():
            measurement = run_model_measurements(
                adapter=adapter,
                rows=rows,
                runtime=runtime,
                batch=args.batch,
                n_draws=args.draws,
                seed=args.seed,
                binding_suite=binding_suite,
                engineering_calibration=engineering_calibration,
            )
    finally:
        if staging is not None:
            staging.cleanup()

    _verify_file_snapshot(input_snapshot)
    if checkpoint_identity(args.model) != checkpoint:
        raise InstrumentError("checkpoint source changed during R5c execution")
    if validate_r5_source_binding(args.gtf, args.fasta) != cohort["source_identity"]:
        raise InstrumentError("annotation/FASTA source changed during R5c execution")
    if require_clean_git() != git_before:
        raise InstrumentError("git identity changed during R5c execution")
    if canonical_sha256(instrument_binding_payload()) != fingerprint:
        raise InstrumentError("R5c instrument changed during execution")

    instrument_ok = measurement.get("instrument_ok") is True
    if args.stage == "preflight":
        status = (
            "PREFLIGHT_INSTRUMENT_PASS"
            if instrument_ok
            else "INSTRUMENT_UNRESOLVED"
        )
        tier = "DEVELOPMENT"
    else:
        status = str(measurement.get("verdict"))
        tier = "CONFIRMATORY"
    entry_record = {
        "artifact_sha256": _entry_artifact_hashes(),
        "authorization": dict(entry_authorization),
        "r3c_record_sha256": canonical_sha256(r3c),
        "r5a_record_sha256": canonical_sha256(r5a),
        "r5b_record_sha256": canonical_sha256(r5b),
    }
    registry_audit = model_free_registry_audit()
    if canonical_sha256(registry_audit) != binding_payload["registry_audit_sha256"]:
        raise InstrumentError("R5c registry audit changed before artifact construction")
    record = {
        "schema": SCHEMA,
        "experiment": "R5c_acceptor_write_delivery",
        "stage": args.stage,
        "bucket": bucket,
        "tier": tier,
        "status": status,
        "verdict": measurement.get("verdict"),
        "readout": "single masked acceptor G hard margin",
        "behavioral_endpoint": "masked_acceptor_G_margin_only",
        "masked_donor_endpoint_constructed": False,
        "sealed_chr8_chr9_opened": False,
        "opened_chromosomes": list(R5_VALIDATION_CHROMOSOMES),
        "instrument_fingerprint_sha256": fingerprint,
        "instrument_binding": binding_payload,
        STACK_BINDING_FIELD: binding_payload[STACK_BINDING_FIELD],
        STACK_FINGERPRINT_FIELD: binding_payload[STACK_FINGERPRINT_FIELD],
        "registry_audit": registry_audit,
        "literal_seals": dict(authorization),
        "git": git_before,
        "runtime_identity": stable_runtime_identity(),
        "execution_device": str(device),
        "checkpoint": checkpoint,
        "source_identity": cohort["source_identity"],
        "cohort": {
            "binding": cohort["binding"],
            "audit": cohort["audit"],
            "row_identity": [row.identity_record() for row in rows],
            "ordered_example_ids_sha256": canonical_sha256(row_ids),
        },
        "entry": entry_record,
        "r5b_runtime_contract": runtime_contract,
        "runtime_bundle_sha256": runtime.bundle_sha256,
        "runtime_validation": runtime_validation,
        "binding_suite": binding_suite,
        "engineering_calibration": engineering_calibration,
        "diagnostic_record_validation": validate_binding_and_calibration_records(
            binding_suite,
            engineering_calibration,
        ),
        "calibration_origin": (
            "RUN_ON_BUCKET0_BEFORE_MODEL_DATA_ACCESS"
            if args.stage == "preflight"
            else "INHERITED_FROM_LITERAL_BUCKET0_PREFLIGHT"
        ),
        "preflight_replay": preflight_replay,
        "preflight_context": preflight_context,
        "measurement": measurement,
        "scientific_outcomes_claim_eligible": bool(
            args.stage == "target"
            and instrument_ok
            and measurement.get("verdict")
            == "R5C_WRITE_AND_TESTED_HEAD_SET_DISTINCT_DELIVERY_SUPPORTED"
        ),
    }
    manifest_inputs: list[Path] = small_input_paths + [
        Path(args.gtf).expanduser().resolve(),
        Path(args.fasta).expanduser().resolve(),
        *_checkpoint_file_paths(checkpoint),
    ]
    result_sha, manifest_sha = write_result_package(
        output,
        record,
        input_paths=manifest_inputs,
        git_identity=git_before,
    )
    return record, {
        "output": str(output),
        "result_sha256": result_sha,
        "manifest": str(output.with_suffix(".manifest.json")),
        "manifest_sha256": manifest_sha,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "preflight", "target"), default="audit")
    parser.add_argument("--r3c-artifact")
    parser.add_argument("--r5a-artifact")
    parser.add_argument("--r5b-artifact")
    parser.add_argument("--r5b-runtime")
    parser.add_argument("--preflight-artifact")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch", type=int, default=PRODUCTION_BATCH)
    parser.add_argument("--draws", type=int, default=PRODUCTION_RANDOMISATION)
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument(
        "--calibration-trials", type=int, default=computed_calibration_trials()
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    # Authorization precedes Path resolution, stat, or reads of every user-supplied
    # model/data/runtime/upstream/preflight/output argument.
    authorization: Mapping[str, str] | None = None
    if args.stage in {"preflight", "target"}:
        try:
            authorization = (
                authorize_preflight()
                if args.stage == "preflight"
                else authorize_target()
            )
        except TargetSealError as exc:
            raise SystemExit(str(exc)) from exc

    if args.stage == "audit":
        audit = model_free_registry_audit()
        audit["instrument_binding_payload"] = instrument_binding_payload()
        audit["instrument_fingerprint_sha256"] = canonical_sha256(
            audit["instrument_binding_payload"]
        )
        if args.out:
            _write_fresh_json(Path(args.out).expanduser().resolve(), audit)
        else:
            print(json.dumps(audit, indent=2, sort_keys=True))
        return

    assert authorization is not None
    try:
        record, output = run_authorized_model_stage(
            args,
            authorization=authorization,
        )
    except ScientificStop as exc:
        raise SystemExit(f"R5c scientific stop: {exc}") from exc
    except InstrumentError as exc:
        raise SystemExit(f"R5c instrument failure: {exc}") from exc
    print(
        f"R5c {args.stage} status={record['status']} -> {output['output']} "
        f"(sha256={output['result_sha256']})"
    )


if __name__ == "__main__":
    main()
