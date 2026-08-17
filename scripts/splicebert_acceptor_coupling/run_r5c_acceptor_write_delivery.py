#!/usr/bin/env python
"""R5c: dense L3H1 write and masked-acceptor delivery capstone.

Preflight is an exact R1c validation-row instrument check.  The biological target is a
different stage: it consumes the exact authenticated 400-row sealed R1c chr8/chr9
handoff, the supported R5a target, and the terminal pre-unseal R5b optional-basis
disposition.  Dense writer and delivery evidence never depend on C2 availability.
The only behavioral readout constructed here is the hard masked-acceptor-G margin.

Only precomputable source/method pins are installed at the preflight-only freeze, so scored stages stop
before resolving user-supplied paths.  Target authorization and predecessor
authentication precede an irreversible target-attempt receipt; only after that
receipt may the R1c-tagged source examples, checkpoint, or output be inspected.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import io
import importlib.util
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

from gfmcd.circuits.hf_bert import (  # noqa: E402
    add_attention_output_delta,
    capture_head_outputs,
    patch_heads,
    project_head_context_to_residual,
)
from gfmcd.splicing.acceptor_coupling_mechanism import (  # noqa: E402
    PRIMARY_LOCAL_WRITER_SITE_ID,
    delivery_candidate_route_gaps,
    delivery_sandwich_estimands,
    energy_match_rows,
    l3_to_u_mediation_estimands,
    local_writer_delivery_gaps,
    local_writer_delivery_sandwich_estimands,
    local_writer_dense_estimands,
    local_writer_subspace_estimands,
    output_projected_head_contribution,
    replay_local_writer_delivery_sandwich_record,
    replay_local_writer_dense_record,
    replay_local_writer_subspace_record,
    replay_delivery_sandwich_record,
    replay_l3_to_u_mediation_record,
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
    R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA,
    R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
    R5_EXPECTED_SOURCE_IDENTITY,
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
    studentized_gene_bootstrap_tests,
    studentized_gene_bootstrap_tost,
)
from gfmcd.splicing.r5_target_ledger import (  # noqa: E402
    COMPLETION_SCHEMA,
    RECEIPT_SCHEMA,
    R5TargetSpentError,
    build_r5_biological_target_identity,
    complete_r5_target_attempt,
    reserve_r5_target_attempt,
)
from gfmcd.splicing.r5_stack_binding import (  # noqa: E402
    R5StackBindingError,
    SHARED_R1C_PACKAGE_PIN_NAMES,
    STACK_BINDING_FIELD,
    STACK_FINGERPRINT_FIELD,
    build_r5_stack_binding,
    normalized_literal_pin_source_sha256,
    r5_stack_fingerprint,
    validate_r5_runtime_policy,
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
R5B_RUNNER = REPO / "scripts/splicebert_acceptor_coupling/run_r5b_acceptor_side_state.py"

# Literal non-wildcard seals.  Target authorization checks all of these before it
# even resolves a user-supplied path.  A development artifact may never fill them.
FROZEN_PROTOCOL_SHA256: str | None = "98913630c476d848586f59bd72cd209c0d943ecd4b57231441642d5873eede3d"
FROZEN_INSTRUMENT_FINGERPRINT_SHA256: str | None = "44ff7aad74cb08d6b3de7753fb8ee0acc8c8cc2e3c3032b2adc40210315f3324"
FROZEN_R5A_PREFLIGHT_ARTIFACT_SHA256: str | None = None
FROZEN_R5A_PREFLIGHT_MANIFEST_SHA256: str | None = None
FROZEN_R5A_TARGET_ARTIFACT_SHA256: str | None = None
FROZEN_R5A_TARGET_MANIFEST_SHA256: str | None = None
FROZEN_R5B_DISPOSITION_ARTIFACT_SHA256: str | None = None
FROZEN_R5B_DISPOSITION_MANIFEST_SHA256: str | None = None
FROZEN_R5C_PREFLIGHT_ARTIFACT_SHA256: str | None = None
FROZEN_R5C_PREFLIGHT_MANIFEST_SHA256: str | None = None
FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256: str | None = None
FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256: str | None = None
FROZEN_R1C_TARGET_ARTIFACT_SHA256: str | None = None
FROZEN_R1C_TARGET_MANIFEST_SHA256: str | None = None
FROZEN_R1C_TARGET_ATTEMPT_SHA256: str | None = None
FROZEN_R1C_TARGET_COMPLETION_SHA256: str | None = None

SCHEMA = "gfmcd.r5c.acceptor_write_delivery/2"
AUDIT_SCHEMA = "gfmcd.r5c.model_free_registry_audit/1"
SESOI = 0.05
STAGE_ALPHA = 0.05
FAMILY_ALPHA = STAGE_ALPHA
PRODUCTION_RANDOMISATION = 4000
PRODUCTION_BOOTSTRAP_DRAWS = 10_000
ROBUST_TRIM_FRACTION = 0.10
PRODUCTION_SEED = 0
PRODUCTION_BATCH = 8
MIN_GENE_CLUSTERS = 30
MIN_GENES = MIN_GENE_CLUSTERS
R5C_EXACT_ROWS = 400
CALIBRATION_EPSILON = 0.01
CALIBRATION_POWER_FLOOR = 0.90
CALIBRATION_PLANT_SUCCESS_RATE = 0.95
ENGINEERING_DIAGNOSTIC_TRIALS = 8
ENGINEERING_DIAGNOSTIC_DRAWS = 399
SENSITIVITY_PLANT_EFFECT = 0.10
SENSITIVITY_ORIENTATIONS = 3
HEALTH_NONINFERIORITY_HARM_MARGIN = 0.10
HEALTH_ALPHA = 0.05
R1C_COMMON_LEG_ATOL = 1e-5
R1C_COMMON_LEG_RTOL = 1e-5

L3_READER = (3, 1)
WRITER_SITE_ID = PRIMARY_LOCAL_WRITER_SITE_ID
PRIMARY_WRITE_REQUIRES_LOW_RANK = False
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
EXPECTED_FAMILY_SIZES = {"C1": 9, "C2": 10, "C3": 72, "C4": 40, "C5": 20}
R5B_BASIS_CONTRACT_SCHEMA = "gfmcd.r5b.frozen_local_writer_basis/2"
R5B_DISPOSITION_SCHEMA = "gfmcd.r5b.local_writer_basis_disposition/2"
R5B_DISPOSITION_STATUSES = (
    "VALIDATED_BASIS",
    "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS",
    "BASIS_INSTRUMENT_UNRESOLVED",
)
R5B_RUNTIME_ARRAY_NAMES = (
    "frozen_basis",
    "random_bases",
    "gene_generic_axis",
    "readout_axis",
    "selected_rank",
)
R5_ENTRY_CONTRACT_SCHEMA = "gfmcd.r5.entry_contract/2"


def stage_pin_names(stage: str) -> tuple[str, ...]:
    """Return the literal package pins required before resolving stage paths."""

    common = (
        "FROZEN_PROTOCOL_SHA256",
        "FROZEN_INSTRUMENT_FINGERPRINT_SHA256",
        "FROZEN_R5B_DISPOSITION_ARTIFACT_SHA256",
        "FROZEN_R5B_DISPOSITION_MANIFEST_SHA256",
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256",
    )
    if stage == "preflight":
        return common + (
            "FROZEN_R5A_PREFLIGHT_ARTIFACT_SHA256",
            "FROZEN_R5A_PREFLIGHT_MANIFEST_SHA256",
        )
    if stage == "target":
        return common + (
            "FROZEN_R5A_PREFLIGHT_ARTIFACT_SHA256",
            "FROZEN_R5A_PREFLIGHT_MANIFEST_SHA256",
            "FROZEN_R5C_PREFLIGHT_ARTIFACT_SHA256",
            "FROZEN_R5C_PREFLIGHT_MANIFEST_SHA256",
            "FROZEN_R1C_TARGET_ARTIFACT_SHA256",
            "FROZEN_R1C_TARGET_MANIFEST_SHA256",
            "FROZEN_R1C_TARGET_ATTEMPT_SHA256",
            "FROZEN_R1C_TARGET_COMPLETION_SHA256",
            "FROZEN_R5A_TARGET_ARTIFACT_SHA256",
            "FROZEN_R5A_TARGET_MANIFEST_SHA256",
        )
    raise ValueError("stage pins exist only for preflight and target")


def build_r5c_target_identity(
    handoff: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Build the method-insensitive spend key for the exact R1c row/gene cohort."""

    if handoff.get("row_count") != 400:
        raise InstrumentError("R5c target identity requires the exact 400-row handoff")
    try:
        identity = build_r5_biological_target_identity(
            stage="R5c",
            checkpoint_sha256=checkpoint_sha256,
            target_gene_ids_sha256=str(
                handoff["sorted_unique_gene_ids_sha256"]
            ),
            ordered_rows_sha256=str(handoff["cohort_sha256"]),
            construction_sha256=str(handoff["capstone_construction_sha256"]),
            endpoint_policy="masked_acceptor_G_margin_only",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InstrumentError(f"R5c target identity is malformed: {exc}") from exc
    return identity


def fixed_scientific_gate_order() -> tuple[str, ...]:
    """Return the single fixed sequence that controls capstone interpretation."""

    return (
        "R1C_PAIRED_DONOR_BEHAVIOR",
        "R5A_DONOR_LOCAL_READER",
        "R5C_DENSE_LOCAL_WRITE",
        "R5C_PRIMARY_PAIR_DELIVERY",
        "R5C_LOW_RANK_STRENGTHENING",
        "R5C_DISTINCT_TESTED_SET_STRENGTHENING",
        "R5C_SECONDARY_H7_TRIPLE_STRENGTHENING",
    )


def target_call_graph_registry() -> dict[str, tuple[str, ...]]:
    """Audit the target-only cohort/provenance edges separately from preflight."""

    return {
        "target_only": (
            "load_r5c_preflight_package",
            "load_r1c_target_handoff",
            "load_r5a_stage_package:target",
            "load_r5b_disposition_package:pre_unseal_optional_C2",
            "reserve_r5_target_attempt",
            "load_r1c_verified_target_common_legs:after_R5c_receipt",
            "load_examples_unscored:test_chr8_chr9_after_receipt",
            "materialize_r5_capstone_rows",
            "replay_r1c_common_legs_before_interventions",
            "run_model_measurements:400_rows_repeated_genes_allowed",
            "complete_r5_target_attempt:after_publication",
        ),
        "forbidden_target_edges": (
            "load_shared_r5_reservation",
            "load_residual_preflight_rows",
            "build_r5_reservations",
            "validate_entry_chain",
            "load_and_validate_entry_artifacts",
            "r5b_runtime_contract_for_bucket",
            "load_r5b_runtime_bundle",
        ),
    }


class TargetSealError(RuntimeError):
    """Raised before any target-supplied path can be inspected."""


class InstrumentError(RuntimeError):
    """A failed hook, identity, health, provenance, or calibration invariant."""


class ScientificStop(RuntimeError):
    """A healthy but negative prerequisite outcome that validly closes R5c."""


OPTIONAL_C2_NOT_EVALUATED_DISPOSITIONS = frozenset(
    {
        "NOT_EVALUATED_NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS",
        "NOT_EVALUATED_R5B_BASIS_INSTRUMENT_UNRESOLVED",
    }
)


def optional_basis_execution_plan(
    *, status: str, runtime: Any | None
) -> dict[str, Any]:
    """Translate R5b's terminal disposition into the path-local C2 plan."""

    if status not in R5B_DISPOSITION_STATUSES:
        raise InstrumentError("R5b basis disposition status differs")
    should_execute = status == "VALIDATED_BASIS"
    if (runtime is not None) is not should_execute:
        raise InstrumentError("R5b optional-basis runtime presence differs from status")
    classifications = {
        "VALIDATED_BASIS": "PENDING_TARGET_STRENGTHENING",
        "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS": (
            "NOT_EVALUATED_NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS"
        ),
        "BASIS_INSTRUMENT_UNRESOLVED": (
            "NOT_EVALUATED_R5B_BASIS_INSTRUMENT_UNRESOLVED"
        ),
    }
    return {
        "execute_c2": should_execute,
        "classification": classifications[status],
        "dense_primary_unaffected": True,
    }


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


def frozen_basis_array_sha256(value: Any) -> str:
    """Reproduce R5b /2's dtype-preserving runtime-array digest."""

    array = np.ascontiguousarray(np.asarray(value))
    if not np.isfinite(array).all():
        raise InstrumentError("frozen R5b array contains non-finite values")
    digest = hashlib.sha256()
    digest.update(f"{array.dtype.str}|{array.ndim}|".encode("ascii"))
    digest.update(",".join(str(int(size)) for size in array.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


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


def matched_random_pairs() -> tuple[tuple[tuple[int, int], ...], ...]:
    """Layer-matched L4+L4 nulls for the all-L4 primary pair."""

    return random_ensemble_registry()["pairs_l4_l4"]


def matched_random_triples() -> tuple[tuple[tuple[int, int], ...], ...]:
    """Layer-matched L4+L4+L4 nulls for the all-L4 H7 triple."""

    return random_ensemble_registry()["triples_l4_l4_l4"]


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


def _writer_row(estimand: str, endpoint: str) -> str:
    if estimand not in {"S_writer", "N_writer"}:
        raise ValueError("dense writer estimand must be S_writer or N_writer")
    return (
        f"C1|site={WRITER_SITE_ID}|component={head_name(L3_READER)}|"
        f"estimand={estimand}|endpoint={endpoint}"
    )


def _u_row(estimand: str, endpoint: str) -> str:
    return f"C2|component=U_A|estimand={estimand}|endpoint={endpoint}"


def _u_random_mean_gap_row(estimand: str, endpoint: str) -> str:
    if endpoint not in {"partner", "specific"}:
        raise ValueError("low-rank random-mean gaps are partner or specificity")
    return (
        "C2|gap=U_A-mean_32_matched_random_bases|"
        f"estimand={estimand}|endpoint={endpoint}"
    )


def _sandwich_row(family: str, component: str, estimand: str, endpoint: str) -> str:
    return f"{family}|component={component}|estimand={estimand}|endpoint={endpoint}"


def _gap_row(
    family: str,
    target: str,
    control: str,
    estimand: str,
    endpoint: str,
) -> str:
    if endpoint not in {"partner", "specific"}:
        raise ValueError("delivery superiority gaps are partner or direct specificity")
    return f"{family}|gap={target}-{control}|estimand={estimand}|endpoint={endpoint}"


def _delivery_random_mean_id(family: str) -> str:
    if family == "C4":
        return "mean_8_layer_matched_random_pairs"
    if family == "C5":
        return "mean_8_layer_matched_random_triples"
    raise ValueError("random delivery means exist only for C4 and C5")


def family_row_names() -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}
    rows["C1"] = tuple(
        _writer_row(estimand, endpoint)
        for estimand in ("S_writer", "N_writer")
        for endpoint in ENDPOINTS
    ) + tuple(_direct_row(DIRECT_NEGATIVE, endpoint) for endpoint in ENDPOINTS)
    c2 = [
        _u_row(estimand, endpoint)
        for estimand in ("S_U", "N_U")
        for endpoint in ENDPOINTS
    ]
    c2.extend(
        _u_random_mean_gap_row(estimand, endpoint)
        for estimand in ("S_U", "N_U")
        for endpoint in ("partner", "specific")
    )
    rows["C2"] = tuple(c2)

    c3 = [
        _sandwich_row("C3", head_name(component), estimand, endpoint)
        for component in ACCEPTOR_CANDIDATES + DONOR_ROUTE_COMPARATORS
        for endpoint in ENDPOINTS
        for estimand in SANDWICH_ESTIMANDS
    ]
    c3.extend(
        _gap_row("C3", head_name(candidate), head_name(route), estimand, endpoint)
        for candidate in ACCEPTOR_CANDIDATES
        for route in DONOR_ROUTE_COMPARATORS
        for estimand in SANDWICH_ESTIMANDS
        for endpoint in ("partner", "specific")
    )
    rows["C3"] = tuple(c3)

    pair_components = (PRIMARY_ACCEPTOR_PAIR,) + DONOR_ROUTE_PAIRS
    c4 = [
        _sandwich_row("C4", ensemble_name(component), estimand, endpoint)
        for component in pair_components
        for endpoint in ENDPOINTS
        for estimand in SANDWICH_ESTIMANDS
    ]
    pair_target = ensemble_name(PRIMARY_ACCEPTOR_PAIR)
    c4.extend(
        _gap_row("C4", pair_target, ensemble_name(control), estimand, endpoint)
        for control in DONOR_ROUTE_PAIRS
        for estimand in SANDWICH_ESTIMANDS
        for endpoint in ("partner", "specific")
    )
    c4.extend(
        _gap_row(
            "C4",
            pair_target,
            _delivery_random_mean_id("C4"),
            estimand,
            endpoint,
        )
        for estimand in SANDWICH_ESTIMANDS
        for endpoint in ("partner", "specific")
    )
    rows["C4"] = tuple(c4)

    triple_components = (SECONDARY_ACCEPTOR_TRIPLE, DONOR_ROUTE_TRIPLE)
    c5 = [
        _sandwich_row("C5", ensemble_name(component), estimand, endpoint)
        for component in triple_components
        for endpoint in ENDPOINTS
        for estimand in SANDWICH_ESTIMANDS
    ]
    triple_target = ensemble_name(SECONDARY_ACCEPTOR_TRIPLE)
    c5.extend(
        _gap_row("C5", triple_target, ensemble_name(control), estimand, endpoint)
        for control in (DONOR_ROUTE_TRIPLE,)
        for estimand in SANDWICH_ESTIMANDS
        for endpoint in ("partner", "specific")
    )
    c5.extend(
        _gap_row(
            "C5",
            triple_target,
            _delivery_random_mean_id("C5"),
            estimand,
            endpoint,
        )
        for estimand in SANDWICH_ESTIMANDS
        for endpoint in ("partner", "specific")
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
            return SESOI
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
            _writer_row(estimand, endpoint),
            "EQUIVALENT_WITHIN_SESOI" if endpoint == "pseudo" else "MATERIAL_POSITIVE",
        )
        for estimand in ("S_writer", "N_writer")
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
    # R5b has already frozen and validated the optional basis against its random
    # subspaces.  R5c reports target random-gap geometry but does not spend a second
    # target-vetoing random family on this nested strengthening.

    c3_paths: dict[str, tuple[Requirement, ...]] = {}
    for candidate in ACCEPTOR_CANDIDATES:
        name = head_name(candidate)
        requirements = _target_sandwich_requirements("C3", name)
        for route in DONOR_ROUTE_COMPARATORS:
            requirements.extend(_equivalent_sandwich_requirements("C3", head_name(route)))
            requirements.extend(
                Requirement(
                    _gap_row("C3", name, head_name(route), estimand, endpoint),
                    "SUPERIOR_TO_CONTROL",
                )
                for estimand in SANDWICH_ESTIMANDS
                for endpoint in ("partner", "specific")
            )
        c3_paths[f"P_C3_{name}"] = tuple(requirements)

    pair_target = ensemble_name(PRIMARY_ACCEPTOR_PAIR)
    c4_basic = _target_sandwich_requirements("C4", pair_target)
    c4_basic.extend(
        Requirement(
            _gap_row(
                "C4",
                pair_target,
                _delivery_random_mean_id("C4"),
                estimand,
                endpoint,
            ),
            "SUPERIOR_TO_CONTROL",
        )
        for estimand in SANDWICH_ESTIMANDS
        for endpoint in ("partner", "specific")
    )
    c4_distinct = list(c4_basic)
    for route in DONOR_ROUTE_PAIRS:
        route_name = ensemble_name(route)
        c4_distinct.extend(_equivalent_sandwich_requirements("C4", route_name))
    for control in DONOR_ROUTE_PAIRS:
        c4_distinct.extend(
            Requirement(
                _gap_row(
                    "C4", pair_target, ensemble_name(control), estimand, endpoint
                ),
                "SUPERIOR_TO_CONTROL",
            )
            for estimand in SANDWICH_ESTIMANDS
            for endpoint in ("partner", "specific")
        )

    triple_target = ensemble_name(SECONDARY_ACCEPTOR_TRIPLE)
    triple_route = ensemble_name(DONOR_ROUTE_TRIPLE)
    c5 = _target_sandwich_requirements("C5", triple_target)
    c5.extend(_equivalent_sandwich_requirements("C5", triple_route))
    # Random triples remain recorded diagnostics.  The only decision-bearing random
    # veto is C4's intervention-size- and layer-matched pair family.
    for control in (DONOR_ROUTE_TRIPLE,):
        c5.extend(
            Requirement(
                _gap_row(
                    "C5", triple_target, ensemble_name(control), estimand, endpoint
                ),
                "SUPERIOR_TO_CONTROL",
            )
            for estimand in SANDWICH_ESTIMANDS
            for endpoint in ("partner", "specific")
        )

    return {
        "C1": {"P_C1_DENSE_LOCAL_WRITE": tuple(c1)},
        "C2": {"P_C2_LOW_RANK_STRENGTHENING": tuple(c2)},
        "C3": c3_paths,
        "C4": {
            "P_C4_PRIMARY_PAIR_DELIVERY": tuple(c4_basic),
            "P_C4_DISTINCT_WITHIN_TESTED_SET": tuple(c4_distinct),
        },
        "C5": {"P_C5_SECONDARY_TRIPLE": tuple(c5)},
    }


def _requirement_edges(requirement: Requirement) -> tuple[tuple[str, float, str], ...]:
    if requirement.decision == "MATERIAL_POSITIVE":
        return (("positive_edge", SESOI, "greater"),)
    if requirement.decision == "SUPERIOR_TO_CONTROL":
        return (("superiority_edge", SESOI, "greater"),)
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
        return lo > SESOI
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


def robust_family_inference(
    family_name: str,
    values: Any,
    groups: Sequence[str],
    *,
    n_draws: int = PRODUCTION_BOOTSTRAP_DRAWS,
    seed: int = PRODUCTION_SEED,
    namespace: str = "r5c-robust-observed",
    min_genes: int = MIN_GENES,
    health_ok: bool = True,
) -> dict[str, Any]:
    """Classify R5c paths with raw- and trimmed-gene bootstrap evidence.

    This is the decision-bearing capstone procedure.  It treats the unweighted gene
    mean and the prespecified 10%-trimmed gene mean as a conjunction, uses strict
    SESOI/TOST boundaries, and reports the median, quantiles, prevalence, and a
    direction-aware top-one-percent deletion estimate for every decision row.  The
    bootstrap is explicitly asymptotic; the historical sign-flip procedure remains an
    engineering diagnostic and never overrides this result.
    """

    if family_name not in FAMILY_NAMES:
        raise ValueError("unknown R5c family")
    names = family_row_names()[family_name]
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(names), len(groups)) or not np.isfinite(matrix).all():
        raise ValueError("family values must be finite [exact_rows, examples]")
    if not _is_integer(n_draws) or int(n_draws) < 1:
        raise ValueError("n_draws must be a positive integer")

    paths = path_registry()[family_name]
    by_row: dict[str, Requirement] = {}
    for requirements in paths.values():
        for requirement in requirements:
            prior = by_row.setdefault(requirement.row, requirement)
            if prior.decision != requirement.decision:
                raise RuntimeError("one family row has contradictory path requirements")
    row_index = {name: index for index, name in enumerate(names)}
    positive = [
        requirement
        for requirement in by_row.values()
        if requirement.decision != "EQUIVALENT_WITHIN_SESOI"
    ]
    equivalent = [
        requirement
        for requirement in by_row.values()
        if requirement.decision == "EQUIVALENT_WITHIN_SESOI"
    ]
    shared_namespace = f"{namespace}:{family_name}:shared-gene-resamples"
    positive_tests = (
        studentized_gene_bootstrap_tests(
            np.vstack([matrix[row_index[item.row]] for item in positive]),
            groups,
            [item.row for item in positive],
            boundaries=[item.boundary for item in positive],
            alternatives="greater",
            alpha=FAMILY_ALPHA,
            trim_fraction=ROBUST_TRIM_FRACTION,
            n_draws=int(n_draws),
            seed=int(seed),
            namespace=shared_namespace,
            min_genes=min_genes,
        )
        if positive
        else {}
    )
    equivalence_tests = (
        studentized_gene_bootstrap_tost(
            np.vstack([matrix[row_index[item.row]] for item in equivalent]),
            groups,
            [item.row for item in equivalent],
            sesoi=SESOI,
            alpha=FAMILY_ALPHA,
            trim_fraction=ROBUST_TRIM_FRACTION,
            n_draws=int(n_draws),
            seed=int(seed),
            namespace=shared_namespace,
            min_genes=min_genes,
        )
        if equivalent
        else {}
    )

    resample_hashes: set[str] = set()
    row_p: dict[str, float] = {}
    row_pass: dict[str, bool] = {}
    row_tests: dict[str, Any] = {}
    for row, requirement in by_row.items():
        if requirement.decision == "EQUIVALENT_WITHIN_SESOI":
            test = equivalence_tests[row]
            row_p[row] = float(test.p_value)
            row_pass[row] = bool(test.equivalence_pass)
            serialized = test.to_dict()
            resample_hashes.update(
                {
                    test.lower_edge.raw.shared_resample_indices_sha256,
                    test.lower_edge.trimmed.shared_resample_indices_sha256,
                    test.upper_edge.raw.shared_resample_indices_sha256,
                    test.upper_edge.trimmed.shared_resample_indices_sha256,
                }
            )
        else:
            test = positive_tests[row]
            row_p[row] = float(test.joint_p_value)
            row_pass[row] = bool(test.joint_boundary_pass)
            serialized = test.to_dict()
            resample_hashes.update(
                {
                    test.raw.shared_resample_indices_sha256,
                    test.trimmed.shared_resample_indices_sha256,
                }
            )
        row_tests[row] = {
            "expected": requirement.decision,
            "boundary": requirement.boundary
            if requirement.decision != "EQUIVALENT_WITHIN_SESOI"
            else [-SESOI, SESOI],
            "joint_p_value": row_p[row],
            "strict_raw_and_trimmed_pass": row_pass[row],
            "test": serialized,
        }
    if len(resample_hashes) != 1:
        raise InstrumentError("robust decision rows did not share one gene resample matrix")

    raw_path_p = {
        path: intersection_union_pvalue([row_p[item.row] for item in requirements])
        for path, requirements in paths.items()
    }
    # C3 contains three alternative singleton-head claims.  Every other family is a
    # singleton or nested fixed-sequence path and therefore is not Holm-penalized.
    adjusted = (
        holm_adjust_named(raw_path_p)
        if family_name == "C3"
        else dict(raw_path_p)
    )
    path_results: dict[str, Any] = {}
    for path, requirements in paths.items():
        all_rows_pass = all(row_pass[item.row] for item in requirements)
        p_pass = adjusted[path] < FAMILY_ALPHA
        if not health_ok:
            classification = "INSTRUMENT_UNRESOLVED"
        elif all_rows_pass and p_pass:
            classification = "SUPPORTED"
        else:
            classification = "NOT_SUPPORTED_AT_FROZEN_THRESHOLDS"
        path_results[path] = {
            "iut_p_value": raw_path_p[path],
            "multiplicity_adjusted_p": adjusted[path],
            "multiplicity_rule": (
                "Holm across three alternative singleton-head paths"
                if family_name == "C3"
                else "fixed-sequence singleton-or-nested path; no cross-stage alpha split"
            ),
            "p_pass": p_pass,
            "all_raw_and_trimmed_rows_pass": all_rows_pass,
            "classification": classification,
        }
    return {
        "schema": "gfmcd.r5c.robust_family_inference/2",
        "family": family_name,
        "alpha": FAMILY_ALPHA,
        "sesoi": SESOI,
        "trim_fraction": ROBUST_TRIM_FRACTION,
        "B": int(n_draws),
        "seed": int(seed),
        "namespace": shared_namespace,
        "inference_kind": "studentized_gene_bootstrap_asymptotic",
        "randomisation_unit": "gene",
        "row_names": list(names),
        "row_order_sha256": canonical_sha256(names),
        "shared_resample_indices_sha256": next(iter(resample_hashes)),
        "row_tests": row_tests,
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


def _attention_output_dense(adapter: Any, layer: int) -> torch.nn.Module:
    try:
        dense = adapter.encoder_layers[int(layer)].attention.output.dense
    except (AttributeError, IndexError, TypeError) as exc:
        raise InstrumentError(
            "checkpoint lacks the BERT attention.output.dense writer site"
        ) from exc
    if not isinstance(dense, torch.nn.Module) or not hasattr(dense, "weight"):
        raise InstrumentError("attention output projection is not a dense module")
    return dense


@contextmanager
def audited_add_attention_output_delta(
    adapter: Any,
    *,
    layer: int,
    token_mask: torch.Tensor,
    delta: Any,
    intervention_id: str,
    role: str,
    component_id: str,
    audit_sink: dict[str, Any],
):
    """Add a residual-width delta at one query after W_O and before residual add.

    The hook is on ``attention.output.dense`` itself, whose output is the projected
    attention write (including one shared bias).  We add only the independently
    reconstructed, bias-free L3H1 contribution delta at the masked-acceptor query.
    """

    if not all(isinstance(value, str) and value for value in (intervention_id, role, component_id)):
        raise ValueError("post-W_O intervention identity must be non-empty")
    if not torch.is_tensor(token_mask) or token_mask.ndim != 2:
        raise ValueError("token_mask must have shape [batch, sequence]")
    mask = token_mask.detach().to(dtype=torch.bool, device="cpu")
    if torch.any(torch.sum(mask.to(dtype=torch.int64), dim=1) != 1):
        raise ValueError("post-W_O writer requires exactly one query token per row")
    dense = _attention_output_dense(adapter, int(layer))
    requested = torch.as_tensor(
        delta, device=dense.weight.device, dtype=dense.weight.dtype
    )
    if requested.ndim != 2 or requested.shape[0] != mask.shape[0]:
        raise ValueError("post-W_O delta must have shape [batch,residual_width]")
    state: dict[str, Any] = {"before_count": 0, "after_count": 0}

    def observer(key: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not torch.is_tensor(output) or output.ndim != 3:
                raise InstrumentError("attention output projection did not emit [B,S,D]")
            if tuple(output.shape[:2]) != tuple(mask.shape):
                raise InstrumentError("post-W_O query mask differs from dense output")
            if output.shape[2] != requested.shape[1]:
                raise InstrumentError("post-W_O delta width differs from dense output")
            count_key = f"{key}_count"
            state[count_key] += 1
            if state[count_key] != 1:
                raise InstrumentError("attention output projection executed more than once")
            state[key] = output.detach().clone()

        return hook

    completed = False
    before_handle = dense.register_forward_hook(observer("before"))
    after_handle = None
    try:
        with add_attention_output_delta(
            adapter,
            int(layer),
            requested,
            token_mask=token_mask,
        ):
            after_handle = dense.register_forward_hook(observer("after"))
            yield
            completed = True
    finally:
        if after_handle is not None:
            after_handle.remove()
        before_handle.remove()
    if not completed:
        return
    if state["before_count"] != 1 or state["after_count"] != 1:
        raise InstrumentError("post-W_O writer hook did not execute exactly once")
    before = state["before"]
    after = state["after"]
    requested = requested.to(device=before.device, dtype=before.dtype)
    local_mask = mask.to(device=before.device)
    observed = after[local_mask] - before[local_mask]
    replay_error = float(torch.max(torch.abs(observed - requested)).detach().cpu())
    off_mask_equal = torch.equal(before[~local_mask], after[~local_mask])
    indices = [int(value) for value in torch.argmax(mask.to(torch.int64), dim=1)]
    record: dict[str, Any] = {
        "schema": "gfmcd.r5c.post_wo_delta_audit/1",
        "intervention_id": intervention_id,
        "role": role,
        "component_id": component_id,
        "layer": int(layer),
        "writer_site_id": WRITER_SITE_ID,
        "bias_included_in_component": False,
        "n_rows": int(mask.shape[0]),
        "sequence_tokens": int(mask.shape[1]),
        "requested_token_indices": indices,
        "requested_token_mask_sha256": _boolean_array_sha256(mask.numpy()),
        "requested_delta_sha256": _torch_array_sha256(requested),
        "observed_delta_sha256": _torch_array_sha256(observed),
        "before_dense_output_sha256": _torch_array_sha256(before),
        "after_dense_output_sha256": _torch_array_sha256(after),
        "delta_replay_max_abs_error": replay_error,
        "off_mask_exact": bool(off_mask_equal),
        "hook_count": 1,
    }
    record["audit_sha256"] = canonical_sha256(record)
    audit_sink.clear()
    audit_sink.update(record)


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


def local_writer_common_leg_record(
    leg_id: str,
    m_corrupt: Any,
    m_writer_restored: Any,
    *,
    writer_site_id: str,
    writer_intervention_id: str,
    m_corrupt_capture_sha256: str | None = None,
    m_writer_restored_capture_sha256: str | None = None,
) -> dict[str, str]:
    """Bind the common source leg used by every v2 delivery component."""

    if not isinstance(leg_id, str) or not leg_id:
        raise ValueError("leg_id must be non-empty")
    if writer_site_id != WRITER_SITE_ID:
        raise ValueError("delivery must use the frozen upstream local writer site")
    if not isinstance(writer_intervention_id, str) or not writer_intervention_id:
        raise ValueError("writer_intervention_id must be non-empty")
    corrupt = np.asarray(m_corrupt, dtype=float)
    restored = np.asarray(m_writer_restored, dtype=float)
    if (
        corrupt.ndim != 1
        or restored.shape != corrupt.shape
        or not np.isfinite(np.vstack((corrupt, restored))).all()
    ):
        raise ValueError("local-writer common legs must be finite aligned vectors")
    corrupt_hash = array_sha256(corrupt)
    restored_hash = array_sha256(restored)
    corrupt_capture = m_corrupt_capture_sha256 or canonical_sha256(
        ["margin-only-capture", leg_id, "corrupt", corrupt_hash]
    )
    restored_capture = m_writer_restored_capture_sha256 or canonical_sha256(
        ["margin-only-capture", leg_id, "writer-restored", restored_hash]
    )
    if any(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
        for value in (corrupt_capture, restored_capture)
    ):
        raise ValueError("local-writer capture identities must be SHA-256 digests")
    record = {
        "leg_id": leg_id,
        "writer_site_id": writer_site_id,
        "writer_intervention_id": writer_intervention_id,
        "m_corrupt_sha256": corrupt_hash,
        "m_writer_restored_sha256": restored_hash,
        "m_corrupt_capture_sha256": str(corrupt_capture),
        "m_writer_restored_capture_sha256": str(restored_capture),
    }
    return record | {"common_leg_pair_sha256": canonical_sha256(record)}


def validate_local_writer_common_legs(
    records: Sequence[Mapping[str, Any]],
    *,
    m_corrupt: Any | None = None,
    m_writer_restored: Any | None = None,
    capture_identity: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Require every delivery arm to share one exact local-writer source leg."""

    if not records:
        raise InstrumentError("at least one local-writer common leg is required")
    required = {
        "leg_id",
        "writer_site_id",
        "writer_intervention_id",
        "m_corrupt_sha256",
        "m_writer_restored_sha256",
        "m_corrupt_capture_sha256",
        "m_writer_restored_capture_sha256",
        "common_leg_pair_sha256",
    }
    if any(set(record) != required for record in records):
        raise InstrumentError("local-writer common-leg record schema differs")
    first = dict(records[0])
    if any(dict(record) != first for record in records[1:]):
        raise InstrumentError("candidate/comparator local-writer common legs differ")
    if first["writer_site_id"] != WRITER_SITE_ID:
        raise InstrumentError("delivery common leg is not at the upstream writer site")
    unsigned = {
        key: first[key] for key in required if key != "common_leg_pair_sha256"
    }
    if first["common_leg_pair_sha256"] != canonical_sha256(unsigned):
        raise InstrumentError("local-writer common-leg digest does not replay")
    hash_fields = {
        "m_corrupt_sha256",
        "m_writer_restored_sha256",
        "m_corrupt_capture_sha256",
        "m_writer_restored_capture_sha256",
        "common_leg_pair_sha256",
    }
    if any(re.fullmatch(r"[0-9a-f]{64}", first[key]) is None for key in hash_fields):
        raise InstrumentError("local-writer common-leg hashes are malformed")
    if m_corrupt is not None and first["m_corrupt_sha256"] != array_sha256(m_corrupt):
        raise InstrumentError("common corrupt capture differs from primitive cells")
    if m_writer_restored is not None and first[
        "m_writer_restored_sha256"
    ] != array_sha256(m_writer_restored):
        raise InstrumentError("common writer-restored capture differs from primitive cells")
    if capture_identity is not None:
        expected = {
            "m_corrupt_capture_sha256",
            "m_writer_restored_capture_sha256",
        }
        if set(capture_identity) != expected or any(
            first[key] != capture_identity[key] for key in expected
        ):
            raise InstrumentError("local-writer tensor capture identities differ")
    return first


def endpoint_vectors(partner: Any, pseudo: Any) -> dict[str, np.ndarray]:
    left = np.asarray(partner, dtype=float)
    right = np.asarray(pseudo, dtype=float)
    if left.ndim != 1 or right.shape != left.shape or not np.isfinite(np.vstack((left, right))).all():
        raise ValueError("partner/pseudo vectors must be finite and aligned")
    return {"partner": left, "pseudo": right, "specific": left - right}


def row_project(delta: Any, projectors: Any) -> np.ndarray:
    """Apply one broadcast projector or row-orthonormal basis per row."""

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


def local_writer_source_cache(
    head_cache: Mapping[int, torch.Tensor],
    *,
    query_token_indices: Sequence[int],
    replacement: Any,
) -> dict[int, torch.Tensor]:
    """Construct an L3H1-only source cache for a query-local writer patch."""

    layer, head = L3_READER
    if layer not in head_cache:
        raise InstrumentError("head cache lacks the frozen local-writer layer")
    original = head_cache[layer]
    if not torch.is_tensor(original) or original.ndim != 4:
        raise InstrumentError("local-writer head cache must have shape [B,H,S,D_head]")
    indices = tuple(query_token_indices)
    if (
        len(indices) != original.shape[0]
        or any(not _is_integer(index) for index in indices)
        or any(int(index) < 0 or int(index) >= original.shape[2] for index in indices)
    ):
        raise ValueError("query_token_indices must give one valid token per row")
    values = torch.as_tensor(replacement, device=original.device, dtype=original.dtype)
    if values.shape != (original.shape[0], original.shape[3]):
        raise ValueError("replacement must have shape [batch,D_head]")
    source = original.clone()
    batch = torch.arange(original.shape[0], device=original.device)
    tokens = torch.tensor(indices, device=original.device, dtype=torch.long)
    source[batch, head, tokens] = values
    return {layer: source}


def validate_broadcast_projectors(
    projectors: Any,
    *,
    selected_rank: int,
    expected_rows: int,
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
        raise InstrumentError("R5c needs one broadcast global operator per row")
    if matrices.shape[1] == matrices.shape[2]:
        representation = "square_projector"
        symmetry = float(np.max(np.abs(matrices - matrices.swapaxes(1, 2))))
        idempotence = float(np.max(np.abs(matrices @ matrices - matrices)))
        ranks = [int(np.linalg.matrix_rank(matrix, tol=atol)) for matrix in matrices]
    else:
        representation = "row_orthonormal_basis"
        if matrices.shape[1] != int(selected_rank) or matrices.shape[2] < matrices.shape[1]:
            raise InstrumentError("broadcast basis shape differs from selected rank")
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
        raise InstrumentError("broadcast projectors fail symmetry/idempotence/rank")
    return {
        "n_projectors": matrices.shape[0],
        "selected_rank": int(selected_rank),
        "representation": representation,
        "symmetry_max_abs": symmetry,
        "idempotence_max_abs": idempotence,
        "operator_array_sha256": array_sha256(matrices),
    }


def runtime_projector_capability_audit(
    runtime: "FrozenR5bGlobalBasis",
    *,
    expected_rows: int = R5C_EXACT_ROWS,
) -> dict[str, Any]:
    """Validate the one selected and 32 random global bases broadcast by R5c."""

    if not _is_integer(expected_rows) or int(expected_rows) < 1:
        raise InstrumentError("projector audit row count must be positive")
    indices = np.arange(int(expected_rows), dtype=np.int64)
    selected_bases = runtime.row_bases(indices)
    selected_audit = validate_broadcast_projectors(
        selected_bases,
        selected_rank=runtime.selected_rank,
        expected_rows=int(expected_rows),
    )
    random_bases = runtime.random_row_bases(indices)
    random_audits = []
    for draw, bases in enumerate(random_bases):
        audit = validate_broadcast_projectors(
            bases,
            selected_rank=runtime.selected_rank,
            expected_rows=int(expected_rows),
        )
        random_audits.append({"draw": draw, **audit})
    authenticated = dict(runtime.authenticated_runtime_array_sha256)
    observed_hashes = {
        "frozen_basis": frozen_basis_array_sha256(runtime.frozen_basis),
        "random_bases": frozen_basis_array_sha256(runtime.random_bases),
        "gene_generic_axis": frozen_basis_array_sha256(runtime.gene_generic_axis),
        "readout_axis": frozen_basis_array_sha256(runtime.readout_axis),
        "selected_rank": frozen_basis_array_sha256(
            np.asarray(runtime.selected_rank, dtype="<i8")
        ),
    }
    if observed_hashes != authenticated:
        raise InstrumentError("projector audit inputs differ from authenticated R5b arrays")
    record = {
        "schema": "gfmcd.r5c.global_projector_capability/2",
        "selected": selected_audit,
        "random": random_audits,
        "authenticated_runtime_array_sha256": authenticated,
        "observed_runtime_array_sha256": observed_hashes,
        "all_projectors_validated": 1 + len(random_audits),
        "audit_sha256": canonical_sha256(
            {"selected": selected_audit, "random": random_audits}
        ),
    }
    record.update(
        {
            "application": (
                "one authenticated global basis broadcast unchanged; no target folds"
            ),
            "expected_rows": int(expected_rows),
        }
    )
    return record


def _validate_orthonormal_basis_array(name: str, bases: np.ndarray, rank: int) -> None:
    if bases.shape[-2] != rank:
        raise InstrumentError(f"{name} rank differs from selected R5b rank")
    gram = np.einsum("...kd,...ld->...kl", bases, bases)
    expected = np.broadcast_to(np.eye(rank), gram.shape)
    if not np.allclose(gram, expected, rtol=0.0, atol=1e-7):
        raise InstrumentError(f"{name} rows are not orthonormal")


@dataclass(frozen=True)
class FrozenR5bGlobalBasis:
    """Exact union-fit R5b basis, broadcast unchanged to every capstone row."""

    selected_rank: int
    frozen_basis: np.ndarray
    random_bases: np.ndarray
    gene_generic_axis: np.ndarray
    readout_axis: np.ndarray
    bundle_sha256: str
    contract_sha256: str
    authenticated_runtime_array_sha256: Mapping[str, str]
    target_refit_performed: bool = False
    target_fold_assignment: None = None

    @property
    def width(self) -> int:
        return int(self.frozen_basis.shape[-1])

    def row_bases(self, indices: Sequence[int]) -> np.ndarray:
        count = len(tuple(indices))
        return np.broadcast_to(
            self.frozen_basis,
            (count, *self.frozen_basis.shape),
        ).copy()

    def random_row_bases(self, indices: Sequence[int]) -> tuple[np.ndarray, ...]:
        count = len(tuple(indices))
        return tuple(
            np.broadcast_to(
                self.random_bases[draw],
                (count, *self.random_bases[draw].shape),
            ).copy()
            for draw in range(32)
        )


def validate_frozen_r5b_basis_contract(
    contract: Mapping[str, Any],
    *,
    expected_stack_fingerprint: str,
) -> dict[str, Any]:
    """Validate the model-free portion of R5b's frozen global-basis /2 contract."""

    if contract.get("schema") != R5B_BASIS_CONTRACT_SCHEMA:
        raise InstrumentError("frozen R5b basis contract schema differs")
    payload = {key: value for key, value in contract.items() if key != "contract_sha256"}
    if contract.get("contract_sha256") != canonical_sha256(payload):
        raise InstrumentError("frozen R5b basis contract digest differs")
    if (
        contract.get("writer_site_id") != WRITER_SITE_ID
        or contract.get("path_role") != "nested_candidate_only"
        or contract.get("source_buckets") != [0, 1]
        or contract.get("fit_timing") != "after_bucket1_validation_locked"
        or contract.get("development_only_source") is not True
        or contract.get("representation")
        != "post_W_O_L3H1_additive_contribution_at_acceptor_query"
        or not _is_integer(contract.get("residual_width"))
        or int(contract.get("residual_width", 0)) < 1
        or re.fullmatch(
            r"[0-9a-f]{64}", str(contract.get("post_wo_writer_contract_sha256"))
        )
        is None
    ):
        raise InstrumentError("frozen R5b basis provenance differs")
    rank = contract.get("selected_rank")
    if not _is_integer(rank) or int(rank) < 1:
        raise InstrumentError("frozen R5b selected rank is malformed")
    if contract.get(STACK_FINGERPRINT_FIELD) != expected_stack_fingerprint:
        raise InstrumentError("frozen R5b basis joint-stack fingerprint differs")
    policy = contract.get("target_policy")
    expected_policy = {
        "refit": "forbidden",
        "rank_selection": "forbidden",
        "basis_mutation": "forbidden",
        "basis_application": "exact authenticated arrays only",
    }
    if policy != expected_policy:
        raise InstrumentError("frozen R5b target policy differs")
    bindings = contract.get("upstream_artifact_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "r5a_entry",
        "r5b_derivation",
        "r5b_validation",
    }:
        raise InstrumentError("frozen R5b upstream package registry differs")
    for name, binding in bindings.items():
        if not isinstance(binding, Mapping) or set(binding) != {
            "artifact_sha256",
            "manifest_sha256",
        }:
            raise InstrumentError(f"frozen R5b {name} package binding differs")
        if any(
            re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
            for value in binding.values()
        ):
            raise InstrumentError(f"frozen R5b {name} package digest is malformed")
    for field in ("checkpoint_sha256", "cohort_audit_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(contract.get(field))) is None:
            raise InstrumentError(f"frozen R5b {field} is malformed")
    hooks = contract.get("r5c_target_attempt_hooks")
    expected_hooks = {
        "owner_stage": "R5c biological target",
        "identity_builder": (
            "gfmcd.splicing.r5_target_ledger.build_r5_biological_target_identity"
        ),
        "reserve_hook": "gfmcd.splicing.r5_target_ledger.reserve_r5_target_attempt",
        "complete_hook": "gfmcd.splicing.r5_target_ledger.complete_r5_target_attempt",
        "reserve_after": "all predecessor and upstream packages authenticated",
        "reserve_before": "cohort/path/model/output inspection",
        "complete_after": "immutable result and manifest publication",
        "r5b_calls_target_ledger": False,
    }
    if hooks != expected_hooks:
        raise InstrumentError("frozen R5b target-ledger ownership differs")
    return {
        "schema": R5B_BASIS_CONTRACT_SCHEMA,
        "selected_rank": int(rank),
        "checkpoint_sha256": contract["checkpoint_sha256"],
        STACK_FINGERPRINT_FIELD: expected_stack_fingerprint,
        "target_refit_forbidden": True,
        "target_rank_selection_forbidden": True,
        "target_basis_mutation_forbidden": True,
        "contract_sha256": contract["contract_sha256"],
    }


def load_frozen_r5b_global_basis(
    path: str | Path,
    *,
    contract: Mapping[str, Any],
    expected_stack_fingerprint: str,
) -> FrozenR5bGlobalBasis:
    """Load exact global R5b arrays; no target fitting, rank choice, or folds."""

    validated = validate_frozen_r5b_basis_contract(
        contract,
        expected_stack_fingerprint=expected_stack_fingerprint,
    )
    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise InstrumentError(f"frozen R5b basis bundle cannot be read: {exc}") from exc
    bundle = contract.get("bundle")
    if not isinstance(bundle, Mapping) or (
        bundle.get("file_name") != source.name
        or bundle.get("size_bytes") != len(raw)
        or bundle.get("sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise InstrumentError("frozen R5b basis bundle bytes differ")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != set(R5B_RUNTIME_ARRAY_NAMES):
                raise InstrumentError("frozen R5b basis array registry differs")
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in R5B_RUNTIME_ARRAY_NAMES
            }
    except (OSError, ValueError) as exc:
        raise InstrumentError(f"frozen R5b basis bundle is invalid: {exc}") from exc
    records = contract.get("runtime_arrays")
    if not isinstance(records, Mapping) or set(records) != set(R5B_RUNTIME_ARRAY_NAMES):
        raise InstrumentError("frozen R5b runtime-array contract differs")
    observed_hashes: dict[str, str] = {}
    for name, value in arrays.items():
        expected = records[name]
        observed_hashes[name] = frozen_basis_array_sha256(value)
        if not isinstance(expected, Mapping) or (
            expected.get("shape") != list(value.shape)
            or expected.get("dtype") != value.dtype.str
            or expected.get("sha256") != observed_hashes[name]
        ):
            raise InstrumentError(f"frozen R5b runtime array differs: {name}")
    rank_raw = arrays["selected_rank"]
    if rank_raw.shape != () or not _is_integer(rank_raw.item()):
        raise InstrumentError("frozen R5b selected-rank array is malformed")
    rank = int(rank_raw.item())
    basis = arrays["frozen_basis"]
    random = arrays["random_bases"]
    generic = arrays["gene_generic_axis"]
    readout = arrays["readout_axis"]
    if (
        rank != validated["selected_rank"]
        or basis.ndim != 2
        or basis.shape[0] != rank
        or random.shape != (32, rank, basis.shape[1])
        or generic.shape != (basis.shape[1],)
        or readout.shape != (basis.shape[1],)
    ):
        raise InstrumentError("frozen R5b basis shapes differ")
    _validate_orthonormal_basis_array("frozen basis", basis, rank)
    _validate_orthonormal_basis_array("random bases", random, rank)
    if not np.isclose(np.linalg.norm(generic), 1.0, rtol=0.0, atol=1e-8):
        raise InstrumentError("frozen R5b gene-generic axis is not unit norm")
    if not np.isclose(np.linalg.norm(readout), 1.0, rtol=0.0, atol=1e-8):
        raise InstrumentError("frozen R5b readout axis is not unit norm")
    if contract.get("basis_id") != "sha256:" + observed_hashes["frozen_basis"]:
        raise InstrumentError("frozen R5b basis ID differs")
    if contract.get("random_basis_ids") != [
        "sha256:" + frozen_basis_array_sha256(random[draw])
        for draw in range(32)
    ]:
        raise InstrumentError("frozen R5b random-basis IDs differ")
    return FrozenR5bGlobalBasis(
        selected_rank=rank,
        frozen_basis=basis,
        random_bases=random,
        gene_generic_axis=generic,
        readout_axis=readout,
        bundle_sha256=hashlib.sha256(raw).hexdigest(),
        contract_sha256=str(contract["contract_sha256"]),
        authenticated_runtime_array_sha256=observed_hashes,
    )


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


def _capture_selected_local_subspace_pair(
    adapter: Any,
    task: Any,
    *,
    clean_encoded: Mapping[str, torch.Tensor],
    corrupt_encoded: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
    query_mask: torch.Tensor,
    layer: int,
    projected: np.ndarray,
    clean_margin: np.ndarray,
    corrupt_margin: np.ndarray,
    leg_id: str,
    chunk_identity: str,
    basis_id: str,
) -> tuple[ConditionCapture, ConditionCapture, dict[str, Any], tuple[dict[str, Any], ...]]:
    """Run the optional R5b-selected C2 pair only when a basis was validated."""

    selected_sufficiency_audit: dict[str, Any] = {}
    selected_sufficiency = _capture_condition(
        adapter,
        task,
        corrupt_encoded,
        readout_positions,
        contexts=(
            audited_add_attention_output_delta(
                adapter,
                layer=layer,
                token_mask=query_mask,
                delta=projected,
                intervention_id=f"{leg_id}:{chunk_identity}:selected_local_u_sufficiency",
                role="selected_local_writer_subspace_sufficiency",
                component_id="U_A",
                audit_sink=selected_sufficiency_audit,
            ),
        ),
    )
    selected_necessity_audit: dict[str, Any] = {}
    selected_necessity = _capture_condition(
        adapter,
        task,
        clean_encoded,
        readout_positions,
        contexts=(
            audited_add_attention_output_delta(
                adapter,
                layer=layer,
                token_mask=query_mask,
                delta=-projected,
                intervention_id=f"{leg_id}:{chunk_identity}:selected_local_u_necessity",
                role="selected_local_writer_subspace_necessity",
                component_id="U_A",
                audit_sink=selected_necessity_audit,
            ),
        ),
    )
    estimands = local_writer_subspace_estimands(
        m_clean=clean_margin,
        m_corrupt=corrupt_margin,
        m_corrupt_with_projected_writer=selected_sufficiency.margin,
        m_clean_without_projected_writer=selected_necessity.margin,
        leg_id=leg_id,
        writer_site_id=WRITER_SITE_ID,
        basis_id=basis_id,
    )
    return (
        selected_sufficiency,
        selected_necessity,
        estimands,
        (selected_sufficiency_audit, selected_necessity_audit),
    )


def execute_conditional_leg(
    adapter: Any,
    task: Any,
    *,
    clean_encoded: Mapping[str, torch.Tensor],
    corrupt_encoded: Mapping[str, torch.Tensor],
    readout_positions: Sequence[int],
    projectors: Any | None,
    random_projectors: Sequence[Any] | None,
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
    query_token_indices = tuple(
        int(position) + int(task.off) for position in readout_positions
    )
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
    include_optional_basis = projectors is not None
    random_projectors = () if random_projectors is None else random_projectors
    if include_optional_basis and len(random_projectors) != 32:
        raise InstrumentError("R5c requires exactly 32 selected-rank random projectors")
    if not include_optional_basis and len(random_projectors) != 0:
        raise InstrumentError(
            "optional random projectors cannot execute without a validated basis"
        )
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
    layer, head = L3_READER
    batch_indices = torch.arange(len(query_token_indices))
    token_indices = torch.tensor(query_token_indices, dtype=torch.long)
    clean_head_context_tensor = (
        clean.heads[layer][batch_indices, head, token_indices]
        .float()
    )
    corrupt_head_context_tensor = (
        minus.heads[layer][batch_indices, head, token_indices]
        .float()
    )
    clean_head_context = clean_head_context_tensor.detach().cpu().numpy()
    corrupt_head_context = corrupt_head_context_tensor.detach().cpu().numpy()
    output_dense = _attention_output_dense(adapter, layer)
    output_weight = output_dense.weight.detach().float().cpu().numpy()
    clean_writer = (
        project_head_context_to_residual(
            adapter, L3_READER, clean_head_context_tensor
        )
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    corrupt_writer = (
        project_head_context_to_residual(
            adapter, L3_READER, corrupt_head_context_tensor
        )
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    # Bind the torch causal bridge to the independent model-free W_O algebra.
    if not np.allclose(
        clean_writer,
        output_projected_head_contribution(
            clean_head_context,
            output_weight,
            head_index=head,
            num_heads=int(adapter.num_heads),
        ),
        rtol=1e-6,
        atol=1e-6,
    ) or not np.allclose(
        corrupt_writer,
        output_projected_head_contribution(
            corrupt_head_context,
            output_weight,
            head_index=head,
            num_heads=int(adapter.num_heads),
        ),
        rtol=1e-6,
        atol=1e-6,
    ):
        raise InstrumentError("torch and model-free post-W_O contribution bridges differ")
    writer_delta = clean_writer - corrupt_writer

    # Retain the historical pre-W_O patch only as an equivalence diagnostic.  Every
    # decision-bearing writer intervention below edits the true post-W_O residual write.
    pre_wo_equivalence_audit: dict[str, Any] = {}
    pre_wo_context = audited_patch_heads(
        adapter,
        clean.heads,
        [L3_READER],
        token_mask=query_mask,
        intervention_id=f"{leg_id}:{chunk_identity}:pre_wo_equivalence_diagnostic",
        role="pre_wo_writer_equivalence_diagnostic",
        component_id=head_name(L3_READER),
        audit_sink=pre_wo_equivalence_audit,
    )
    c_plus_pre_wo = _capture_condition(
        adapter,
        task,
        corrupt_encoded,
        readout_positions,
        contexts=(pre_wo_context,),
    )
    dense_sufficiency_audit: dict[str, Any] = {}
    c_plus = _capture_condition(
        adapter,
        task,
        corrupt_encoded,
        readout_positions,
        contexts=(
            audited_add_attention_output_delta(
                adapter,
                layer=layer,
                token_mask=query_mask,
                delta=writer_delta,
                intervention_id=f"{leg_id}:{chunk_identity}:dense_writer_sufficiency",
                role="dense_local_writer_sufficiency",
                component_id=head_name(L3_READER),
                audit_sink=dense_sufficiency_audit,
            ),
        ),
    )
    mutation_audits.extend((pre_wo_equivalence_audit, dense_sufficiency_audit))
    captures.extend((c_plus_pre_wo, c_plus))
    equivalence_margin_max_abs_error = float(
        np.max(np.abs(c_plus_pre_wo.margin - c_plus.margin))
    )
    equivalence_residual_max_abs_error = float(
        np.max(np.abs(c_plus_pre_wo.query_residual - c_plus.query_residual))
    )
    if (
        equivalence_margin_max_abs_error > 1e-5
        or equivalence_residual_max_abs_error > 1e-5
    ):
        raise InstrumentError(
            "post-W_O writer delta is not equivalent to the full clean-head patch"
        )
    dense_necessity_audit: dict[str, Any] = {}
    clean_with_corrupt_writer = _capture_condition(
        adapter,
        task,
        clean_encoded,
        readout_positions,
        contexts=(
            audited_add_attention_output_delta(
                adapter,
                layer=layer,
                token_mask=query_mask,
                delta=-writer_delta,
                intervention_id=(
                    f"{leg_id}:{chunk_identity}:dense_writer_necessity"
                ),
                role="dense_local_writer_necessity",
                component_id=head_name(L3_READER),
                audit_sink=dense_necessity_audit,
            ),
        ),
    )
    mutation_audits.append(dense_necessity_audit)
    captures.append(clean_with_corrupt_writer)
    dense_writer = local_writer_dense_estimands(
        m_clean=clean.margin,
        m_corrupt=minus.margin,
        m_corrupt_with_clean_writer=c_plus.margin,
        m_clean_with_corrupt_writer=clean_with_corrupt_writer.margin,
        leg_id=leg_id,
        writer_site_id=WRITER_SITE_ID,
    )
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

    projected: np.ndarray | None = None
    selected_sufficiency: ConditionCapture | None = None
    selected_necessity: ConditionCapture | None = None
    u_estimands: dict[str, Any] | None = None
    if include_optional_basis:
        projected = row_project(writer_delta, projectors)
        basis_id = f"row-indexed-local-writer:{array_sha256(projectors)}"
        (
            selected_sufficiency,
            selected_necessity,
            u_estimands,
            selected_audits,
        ) = _capture_selected_local_subspace_pair(
            adapter,
            task,
            clean_encoded=clean_encoded,
            corrupt_encoded=corrupt_encoded,
            readout_positions=readout_positions,
            query_mask=query_mask,
            layer=layer,
            projected=projected,
            clean_margin=clean.margin,
            corrupt_margin=minus.margin,
            leg_id=leg_id,
            chunk_identity=chunk_identity,
            basis_id=basis_id,
        )
        captures.extend((selected_sufficiency, selected_necessity))
        mutation_audits.extend(selected_audits)
    random_u: list[dict[str, Any]] = []
    for draw, random_set in enumerate(random_projectors):
        if projected is None:
            raise InstrumentError("random R5b projector executed without selected basis")
        raw_random_projected = row_project(writer_delta, random_set)
        random_projected = energy_match_rows(projected, raw_random_projected)
        energy_error = float(
            np.max(
                np.abs(
                    np.linalg.norm(random_projected, axis=1)
                    - np.linalg.norm(projected, axis=1)
                )
            )
        )
        random_sufficiency_audit: dict[str, Any] = {}
        random_sufficient = _capture_condition(
            adapter,
            task,
            corrupt_encoded,
            readout_positions,
            contexts=(
                audited_add_attention_output_delta(
                    adapter,
                    layer=layer,
                    token_mask=query_mask,
                    delta=random_projected,
                    intervention_id=(
                        f"{leg_id}:{chunk_identity}:random_local_u_{draw:02d}_sufficiency"
                    ),
                    role="random_local_writer_subspace_sufficiency",
                    component_id=f"random_projector_{draw:02d}",
                    audit_sink=random_sufficiency_audit,
                ),
            ),
        )
        random_necessity_audit: dict[str, Any] = {}
        random_without = _capture_condition(
            adapter,
            task,
            clean_encoded,
            readout_positions,
            contexts=(
                audited_add_attention_output_delta(
                    adapter,
                    layer=layer,
                    token_mask=query_mask,
                    delta=-random_projected,
                    intervention_id=(
                        f"{leg_id}:{chunk_identity}:random_local_u_{draw:02d}_necessity"
                    ),
                    role="random_local_writer_subspace_necessity",
                    component_id=f"random_projector_{draw:02d}",
                    audit_sink=random_necessity_audit,
                ),
            ),
        )
        captures.extend((random_sufficient, random_without))
        mutation_audits.extend((random_sufficiency_audit, random_necessity_audit))
        random_record = local_writer_subspace_estimands(
            m_clean=clean.margin,
            m_corrupt=minus.margin,
            m_corrupt_with_projected_writer=random_sufficient.margin,
            m_clean_without_projected_writer=random_without.margin,
            leg_id=leg_id,
            writer_site_id=WRITER_SITE_ID,
            basis_id=f"random-{draw:02d}:{array_sha256(random_set)}",
        )
        random_record.update(
            {
                "draw": draw,
                "projector_sha256": array_sha256(random_set),
                "per_row_energy_match_max_abs_error": energy_error,
                "redrawn": False,
                "sufficiency_capture_sha256": _condition_capture_identity(
                    random_sufficient
                )["capture_sha256"],
                "necessity_capture_sha256": _condition_capture_identity(
                    random_without
                )["capture_sha256"],
                "mutation_audit_sha256": {
                    "sufficiency": random_sufficiency_audit["audit_sha256"],
                    "necessity": random_necessity_audit["audit_sha256"],
                },
            }
        )
        random_u.append(random_record)
    minus_capture_identity = _condition_capture_identity(minus)
    c_plus_capture_identity = _condition_capture_identity(c_plus)
    common_capture_identity = {
        "m_corrupt_capture_sha256": minus_capture_identity["capture_sha256"],
        "m_writer_restored_capture_sha256": c_plus_capture_identity["capture_sha256"],
    }
    writer_intervention_id = "dense-clean-L3H1-output-into-corrupt-query"
    common = local_writer_common_leg_record(
        leg_id,
        minus.margin,
        c_plus.margin,
        writer_site_id=WRITER_SITE_ID,
        writer_intervention_id=writer_intervention_id,
        **common_capture_identity,
    )

    sandwich: dict[str, Any] = {}
    common_records: list[Mapping[str, Any]] = []
    for heads in downstream_registry:
        component_id = ensemble_name(heads)
        writer_restoration_audit: dict[str, Any] = {}
        delivery_necessity_audit: dict[str, Any] = {}
        necessity_contexts = (
            audited_add_attention_output_delta(
                adapter,
                layer=layer,
                token_mask=query_mask,
                delta=writer_delta,
                intervention_id=(
                    f"{leg_id}:{chunk_identity}:{component_id}:writer_restoration"
                ),
                role="delivery_writer_restoration",
                component_id=component_id,
                audit_sink=writer_restoration_audit,
            ),
            audited_patch_heads(
                adapter,
                minus.heads,
                tuple(heads),
                token_mask=query_mask,
                intervention_id=(
                    f"{leg_id}:{chunk_identity}:{component_id}:delivery_necessity"
                ),
                role="delivery_necessity",
                component_id=component_id,
                audit_sink=delivery_necessity_audit,
            ),
        )
        necessity = _capture_condition(
            adapter,
            task,
            corrupt_encoded,
            readout_positions,
            contexts=necessity_contexts,
        )
        delivery_sufficiency_audit: dict[str, Any] = {}
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
                        f"{leg_id}:{chunk_identity}:{component_id}:delivery_sufficiency"
                    ),
                    role="delivery_sufficiency",
                    component_id=component_id,
                    audit_sink=delivery_sufficiency_audit,
                ),
            ),
        )
        captures.extend((necessity, sufficiency))
        mutation_audits.extend(
            (
                writer_restoration_audit,
                delivery_necessity_audit,
                delivery_sufficiency_audit,
            )
        )
        record = local_writer_delivery_sandwich_estimands(
            m_corrupt=minus.margin,
            m_writer_restored=c_plus.margin,
            m_writer_restored_with_d_corrupt=necessity.margin,
            m_corrupt_with_d_from_writer_restored=sufficiency.margin,
            leg_id=leg_id,
            component_id=component_id,
            writer_site_id=WRITER_SITE_ID,
            writer_intervention_id=writer_intervention_id,
        )
        record["common_leg"] = dict(common)
        record["mutation_audit_sha256"] = {
            "writer_restoration": writer_restoration_audit["audit_sha256"],
            "necessity": delivery_necessity_audit["audit_sha256"],
            "sufficiency": delivery_sufficiency_audit["audit_sha256"],
        }
        record["condition_capture_sha256"] = {
            "necessity": _condition_capture_identity(necessity)["capture_sha256"],
            "sufficiency": _condition_capture_identity(sufficiency)["capture_sha256"],
        }
        sandwich[component_id] = record
        common_records.append(record["common_leg"])
    validate_local_writer_common_legs(
        common_records,
        m_corrupt=minus.margin,
        m_writer_restored=c_plus.margin,
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
            "decision_writer_site_id": WRITER_SITE_ID,
            "decision_writer_is_post_wo": True,
            "decision_writer_bias_included": False,
            "pre_wo_patch_role": "equivalence_diagnostic_only",
            "post_wo_pre_wo_equivalence": {
                "margin_max_abs_error": equivalence_margin_max_abs_error,
                "query_residual_max_abs_error": equivalence_residual_max_abs_error,
                "tolerance": 1e-5,
                "pass": True,
                "pre_wo_capture_sha256": _condition_capture_identity(
                    c_plus_pre_wo
                )["capture_sha256"],
                "post_wo_capture_sha256": _condition_capture_identity(c_plus)[
                    "capture_sha256"
                ],
            },
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
            "m_corrupt_margin": minus.margin,
            "m_writer_restored_margin": c_plus.margin,
            "m_clean_with_corrupt_writer_margin": clean_with_corrupt_writer.margin,
            "clean_margin_sha256": array_sha256(clean.margin),
            "m_corrupt_margin_sha256": array_sha256(minus.margin),
            "m_writer_restored_margin_sha256": array_sha256(c_plus.margin),
            "m_clean_with_corrupt_writer_margin_sha256": array_sha256(
                clean_with_corrupt_writer.margin
            ),
            "clean_writer_state_sha256": array_sha256(clean_writer),
            "corrupt_writer_state_sha256": array_sha256(corrupt_writer),
            "l3h1_pre_wo_clean_context_sha256": array_sha256(clean_head_context),
            "l3h1_pre_wo_corrupt_context_sha256": array_sha256(
                corrupt_head_context
            ),
            "l3_attention_output_projection_weight_sha256": array_sha256(
                output_weight
            ),
            "l3h1_post_wo_clean_contribution_sha256": array_sha256(clean_writer),
            "l3h1_post_wo_corrupt_contribution_sha256": array_sha256(
                corrupt_writer
            ),
            "l3h1_post_wo_delta_sha256": array_sha256(writer_delta),
            "l3h1_post_wo_bias_included": False,
            "selected_subspace_sufficiency_capture_sha256": (
                None
                if selected_sufficiency is None
                else _condition_capture_identity(selected_sufficiency)[
                    "capture_sha256"
                ]
            ),
            "selected_subspace_necessity_capture_sha256": (
                None
                if selected_necessity is None
                else _condition_capture_identity(selected_necessity)[
                    "capture_sha256"
                ]
            ),
            "clean_capture_identity": _condition_capture_identity(clean),
            "m_corrupt_capture_identity": minus_capture_identity,
            "m_writer_restored_capture_identity": c_plus_capture_identity,
            "m_clean_with_corrupt_writer_capture_identity": (
                _condition_capture_identity(clean_with_corrupt_writer)
            ),
        },
        "dense_writer": dense_writer,
        "local_writer_subspace": u_estimands,
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
        "random_local_writer_subspaces": random_u,
        "delivery": sandwich,
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


def _replay_dense_writer_envelope(record: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return replay_local_writer_dense_record(record)
    except (TypeError, ValueError) as exc:
        raise InstrumentError(f"dense local-writer replay failed: {exc}") from exc


def _replay_local_subspace_envelope(record: Mapping[str, Any]) -> dict[str, Any]:
    core_keys = {"leg_id", "writer_site_id", "basis_id", "cells", "estimands"}
    allowed_metadata = {
        "draw",
        "projector_sha256",
        "per_row_energy_match_max_abs_error",
        "redrawn",
        "sufficiency_capture_sha256",
        "necessity_capture_sha256",
        "mutation_audit_sha256",
    }
    if not isinstance(record, Mapping) or not core_keys.issubset(record) or not set(
        record
    ).issubset(core_keys | allowed_metadata):
        raise InstrumentError("local-writer subspace primitive envelope schema differs")
    try:
        return replay_local_writer_subspace_record(
            {key: record[key] for key in core_keys}
        )
    except (TypeError, ValueError) as exc:
        raise InstrumentError(f"local-writer subspace replay failed: {exc}") from exc


def _replay_delivery_envelope(record: Mapping[str, Any]) -> dict[str, Any]:
    core_keys = {
        "leg_id",
        "component_id",
        "writer_site_id",
        "writer_intervention_id",
        "cells",
        "estimands",
    }
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
        return replay_local_writer_delivery_sandwich_record(
            {key: record[key] for key in core_keys}
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
    replayed = _replay_delivery_envelope(leg["delivery"][component])
    return np.asarray(replayed["estimands"][f"{estimand}_d"], dtype=float)


def derive_family_matrices(
    partner_leg: Mapping[str, Any],
    pseudo_leg: Mapping[str, Any],
    *,
    include_optional_basis: bool = True,
) -> dict[str, np.ndarray]:
    """Derive the dense/delivery families and optional basis family from raw legs.

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
    replayed_dense_writer: dict[str, dict[str, Any]] = {}
    replayed_random: dict[str, list[dict[str, Any]]] = {}
    replayed_delivery: dict[str, dict[str, dict[str, Any]]] = {}
    replayed_direct_negative: dict[str, dict[str, Any]] = {}
    for leg in (partner_leg, pseudo_leg):
        if set(leg.get("delivery", {})) != expected_components:
            raise InstrumentError("source leg lacks the complete downstream registry")
        source_id = str(leg["leg_id"])
        canonical_delivery = {
            component: _replay_delivery_envelope(record)
            for component, record in leg["delivery"].items()
        }
        common = validate_local_writer_common_legs(
            [record["common_leg"] for record in leg["delivery"].values()],
            m_corrupt=next(iter(canonical_delivery.values()))["cells"]["m_corrupt"],
            m_writer_restored=next(iter(canonical_delivery.values()))["cells"][
                "m_writer_restored"
            ],
        )
        if "common_leg" in leg and dict(leg["common_leg"]) != common:
            raise InstrumentError("source-leg common capture differs from its sandwiches")
        replayed_dense_writer[source_id] = _replay_dense_writer_envelope(
            leg["dense_writer"]
        )
        if include_optional_basis:
            if len(leg.get("random_local_writer_subspaces", ())) != 32:
                raise InstrumentError(
                    "source leg lacks the exact random-projector registry"
                )
            replayed_u[source_id] = _replay_local_subspace_envelope(
                leg["local_writer_subspace"]
            )
            replayed_random[source_id] = [
                _replay_local_subspace_envelope(record)
                for record in leg["random_local_writer_subspaces"]
            ]
        elif (
            leg.get("local_writer_subspace") is not None
            or leg.get("random_local_writer_subspaces") not in ([], ())
        ):
            raise InstrumentError(
                "nonvalidated R5b disposition must not execute optional basis arms"
            )
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

    # C1: dense query-local L3H1 writer necessity/sufficiency and L0H2 control.
    c1: dict[str, np.ndarray] = {}
    for estimand in ("S_writer", "N_writer"):
        vectors = endpoint_vectors(
            replayed_dense_writer["partner"]["estimands"][estimand],
            replayed_dense_writer["pseudo"]["estimands"][estimand],
        )
        for endpoint in ENDPOINTS:
            add(c1, _writer_row(estimand, endpoint), vectors[endpoint])
    negative_vectors = endpoint_vectors(
        replayed_direct_negative["partner"]["estimands"]["direct_negative_rescue"],
        replayed_direct_negative["pseudo"]["estimands"]["direct_negative_rescue"],
    )
    for endpoint in ENDPOINTS:
        add(c1, _direct_row(DIRECT_NEGATIVE, endpoint), negative_vectors[endpoint])

    # C2: selected U_A and descriptive target random-gap geometry.  R5b's frozen
    # validation owns the decision-bearing basis-vs-random disposition.
    c2: dict[str, np.ndarray] = {}
    if include_optional_basis:
        for estimand in ("S_U", "N_U"):
            target = endpoint_vectors(
                replayed_u["partner"]["estimands"][estimand],
                replayed_u["pseudo"]["estimands"][estimand],
            )
            for endpoint in ENDPOINTS:
                add(c2, _u_row(estimand, endpoint), target[endpoint])
        random_by_draw: dict[tuple[int, str, str], np.ndarray] = {}
        for draw in range(32):
            p_record = partner_leg["random_local_writer_subspaces"][draw]
            q_record = pseudo_leg["random_local_writer_subspaces"][draw]
            if p_record.get("draw") != draw or q_record.get("draw") != draw:
                raise InstrumentError("random-projector order differs across source legs")
            if p_record.get("projector_sha256") != q_record.get("projector_sha256"):
                raise InstrumentError(
                    "random-projector identity differs across source legs"
                )
            for estimand in ("S_U", "N_U"):
                random_vectors = endpoint_vectors(
                    replayed_random["partner"][draw]["estimands"][estimand],
                    replayed_random["pseudo"][draw]["estimands"][estimand],
                )
                for endpoint in ("partner", "specific"):
                    random_by_draw[(draw, estimand, endpoint)] = random_vectors[
                        endpoint
                    ]
        for estimand in ("S_U", "N_U"):
            selected = endpoint_vectors(
                replayed_u["partner"]["estimands"][estimand],
                replayed_u["pseudo"]["estimands"][estimand],
            )
            for endpoint in ("partner", "specific"):
                random_mean = np.mean(
                    np.stack(
                        [
                            random_by_draw[(draw, estimand, endpoint)]
                            for draw in range(32)
                        ]
                    ),
                    axis=0,
                )
                add(
                    c2,
                    _u_random_mean_gap_row(estimand, endpoint),
                    selected[endpoint] - random_mean,
                )

    def direct_delivery_gap_vectors(
        target_name: str,
        control_name: str,
    ) -> dict[tuple[str, str], np.ndarray]:
        """Form target-minus-control contrasts before gene-level inference."""

        partner_gaps = local_writer_delivery_gaps(
            replayed_delivery["partner"][target_name],
            replayed_delivery["partner"][control_name],
        )
        pseudo_gaps = local_writer_delivery_gaps(
            replayed_delivery["pseudo"][target_name],
            replayed_delivery["pseudo"][control_name],
        )
        output: dict[tuple[str, str], np.ndarray] = {}
        for estimand in SANDWICH_ESTIMANDS:
            key = f"{estimand}_candidate_minus_comparator"
            partner_gap = np.asarray(partner_gaps[key], dtype=float)
            pseudo_gap = np.asarray(pseudo_gaps[key], dtype=float)
            output[(estimand, "partner")] = partner_gap
            output[(estimand, "specific")] = partner_gap - pseudo_gap
        return output

    def build_delivery_family(
        family: str,
        components: Sequence[Sequence[tuple[int, int]]],
        target: Sequence[tuple[int, int]],
        controls: Sequence[Sequence[tuple[int, int]]],
        random_controls: Sequence[Sequence[tuple[int, int]]],
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
            # Each source helper independently asserts its writer-restored common
            # leg. Specificity is then a direct paired contrast of the two gaps.
            gaps = direct_delivery_gap_vectors(target_name, control_name)
            for estimand in SANDWICH_ESTIMANDS:
                for endpoint in ("partner", "specific"):
                    add(
                        rows,
                        _gap_row(
                            family,
                            target_name,
                            control_name,
                            estimand,
                            endpoint,
                        ),
                        gaps[(estimand, endpoint)],
                    )
        random_gap_vectors = [
            direct_delivery_gap_vectors(target_name, ensemble_name(control))
            for control in random_controls
        ]
        if not random_gap_vectors:
            raise InstrumentError("delivery random-mean gate lacks matched controls")
        for estimand in SANDWICH_ESTIMANDS:
            for endpoint in ("partner", "specific"):
                add(
                    rows,
                    _gap_row(
                        family,
                        target_name,
                        _delivery_random_mean_id(family),
                        estimand,
                        endpoint,
                    ),
                    np.mean(
                        np.stack(
                            [gaps[(estimand, endpoint)] for gaps in random_gap_vectors]
                        ),
                        axis=0,
                    ),
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
            gaps = direct_delivery_gap_vectors(candidate_name, route_name)
            for estimand in SANDWICH_ESTIMANDS:
                for endpoint in ("partner", "specific"):
                    add(
                        c3,
                        _gap_row(
                            "C3",
                            candidate_name,
                            route_name,
                            estimand,
                            endpoint,
                        ),
                        gaps[(estimand, endpoint)],
                    )

    c4_components = (PRIMARY_ACCEPTOR_PAIR,) + DONOR_ROUTE_PAIRS
    c4 = build_delivery_family(
        "C4",
        c4_components,
        PRIMARY_ACCEPTOR_PAIR,
        DONOR_ROUTE_PAIRS,
        matched_random_pairs(),
    )
    c5_components = (SECONDARY_ACCEPTOR_TRIPLE, DONOR_ROUTE_TRIPLE)
    c5 = build_delivery_family(
        "C5",
        c5_components,
        SECONDARY_ACCEPTOR_TRIPLE,
        (DONOR_ROUTE_TRIPLE,),
        matched_random_triples(),
    )

    derived = {"C1": c1, "C3": c3, "C4": c4, "C5": c5}
    if include_optional_basis:
        derived = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    expected = family_row_names()
    for family in derived:
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
    component_ids = tuple(chunks[0].get("delivery", {}))
    if any(tuple(chunk.get("delivery", {})) != component_ids for chunk in chunks):
        raise InstrumentError("downstream component order differs across chunks")

    def merge_dense_writer(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        canonical_records = [_replay_dense_writer_envelope(record) for record in records]
        merged = {
            "leg_id": leg_id,
            "writer_site_id": WRITER_SITE_ID,
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
        return replay_local_writer_dense_record(merged)

    def merge_subspace(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        canonical_records = [_replay_local_subspace_envelope(record) for record in records]
        merged = {
            "leg_id": leg_id,
            "writer_site_id": WRITER_SITE_ID,
            "basis_id": canonical_sha256(
                [record["basis_id"] for record in canonical_records]
            ),
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
        return replay_local_writer_subspace_record(merged)

    dense_writer = merge_dense_writer([chunk["dense_writer"] for chunk in chunks])
    basis_available = chunks[0].get("local_writer_subspace") is not None
    if any(
        (chunk.get("local_writer_subspace") is not None) != basis_available
        for chunk in chunks
    ):
        raise InstrumentError("optional R5b basis availability differs across chunks")
    u_a = (
        merge_subspace([chunk["local_writer_subspace"] for chunk in chunks])
        if basis_available
        else None
    )
    random_u: list[dict[str, Any]] = []
    expected_random_draws = 32 if basis_available else 0
    if any(
        len(chunk.get("random_local_writer_subspaces", ()))
        != expected_random_draws
        for chunk in chunks
    ):
        raise InstrumentError("optional R5b random-projector count differs across chunks")
    for draw in range(expected_random_draws):
        records = [chunk["random_local_writer_subspaces"][draw] for chunk in chunks]
        if any(record.get("draw") != draw for record in records):
            raise InstrumentError("random-projector draw order differs across chunks")
        merged = merge_subspace(records)
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
        "m_corrupt_capture_sha256": canonical_sha256(
            [record["m_corrupt_capture_sha256"] for record in chunk_common_records]
        ),
        "m_writer_restored_capture_sha256": canonical_sha256(
            [
                record["m_writer_restored_capture_sha256"]
                for record in chunk_common_records
            ]
        ),
    }
    for component in component_ids:
        records = [chunk["delivery"][component] for chunk in chunks]
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
        primitive = replay_local_writer_delivery_sandwich_record(
            {
                "leg_id": leg_id,
                "component_id": component,
                "writer_site_id": WRITER_SITE_ID,
                "writer_intervention_id": records[0]["writer_intervention_id"],
                "cells": cells,
                "estimands": estimands,
            }
        )
        sandwich[component] = primitive | {
            "leg_id": leg_id,
            "component_id": component,
            "common_leg": local_writer_common_leg_record(
                leg_id,
                cells["m_corrupt"],
                cells["m_writer_restored"],
                writer_site_id=WRITER_SITE_ID,
                writer_intervention_id=primitive["writer_intervention_id"],
                **aggregate_capture_identity,
            ),
            "mutation_audit_sha256_by_chunk": {
                direction: [
                    record["mutation_audit_sha256"][direction]
                    for record in records
                ]
                for direction in (
                    "writer_restoration",
                    "necessity",
                    "sufficiency",
                )
            },
            "condition_capture_sha256": {
                direction: canonical_sha256(
                    [record["condition_capture_sha256"][direction] for record in records]
                )
                for direction in ("necessity", "sufficiency")
            },
        }
    common = validate_local_writer_common_legs(
        [record["common_leg"] for record in sandwich.values()],
        m_corrupt=next(iter(sandwich.values()))["cells"]["m_corrupt"],
        m_writer_restored=next(iter(sandwich.values()))["cells"][
            "m_writer_restored"
        ],
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
        "dense_writer": dense_writer,
        "local_writer_subspace": u_a,
        "direct_negative": {
            "component_id": head_name(DIRECT_NEGATIVE),
            "cells": direct_cells,
            "estimands": {"direct_negative_rescue": direct_estimand},
            "margin_sha256": array_sha256(direct_cells["m_direct_negative"]),
        },
        "random_local_writer_subspaces": random_u,
        "delivery": sandwich,
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
            for name in (
                "clean_margin",
                "m_corrupt_margin",
                "m_writer_restored_margin",
                "m_clean_with_corrupt_writer_margin",
            )
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


def production_health_arm_registry(
    *, include_optional_basis: bool
) -> dict[str, dict[str, str]]:
    """Return every decision-bearing production N/S arm and its own baseline."""

    if type(include_optional_basis) is not bool:
        raise TypeError("optional-basis health selection must be a literal boolean")
    registry: dict[str, dict[str, str]] = {
        "C1|L3H1.output_projected|necessity": {
            "constructor": "post_wo_writer_delta_subtraction",
            "baseline_state": "clean",
        },
        "C1|L3H1.output_projected|sufficiency": {
            "constructor": "post_wo_writer_delta_addition",
            "baseline_state": "corrupt",
        },
    }
    if include_optional_basis:
        for component in (
            "selected_global_basis",
            *(f"random_global_basis_{draw:02d}" for draw in range(32)),
        ):
            for direction in ("necessity", "sufficiency"):
                registry[f"C2|{component}|{direction}"] = {
                    "constructor": (
                        "post_wo_projected_writer_delta_subtraction"
                        if direction == "necessity"
                        else "post_wo_projected_writer_delta_addition"
                    ),
                    "baseline_state": (
                        "clean" if direction == "necessity" else "corrupt"
                    ),
                }
    for ensemble in complete_downstream_registry():
        component = ensemble_name(ensemble)
        registry[f"delivery|{component}|necessity"] = {
            "constructor": "post_wo_writer_restoration_plus_corrupt_head_patch",
            "baseline_state": "writer_restored",
        }
        registry[f"delivery|{component}|sufficiency"] = {
            "constructor": "writer_restored_head_patch_into_corrupt",
            "baseline_state": "corrupt",
        }
    return registry


def validate_production_health_record(
    record: Mapping[str, Any], *, include_optional_basis: bool
) -> dict[str, Any]:
    """Fail closed unless health covers the exact production arms and baselines."""

    expected = production_health_arm_registry(
        include_optional_basis=include_optional_basis
    )
    records = record.get("records")
    if (
        record.get("schema") != "gfmcd.r5c.paired_noop_health/3"
        or record.get("candidate_outcomes_used") is not False
        or record.get("optional_basis_executed") is not include_optional_basis
        or record.get("arm_registry") != expected
        or record.get("arm_registry_sha256") != canonical_sha256(expected)
        or not isinstance(records, Mapping)
        or set(records) != {"partner", "pseudo"}
    ):
        raise InstrumentError("R5c health identity or exact production arm coverage differs")
    all_pass = True
    for source in ("partner", "pseudo"):
        source_records = records.get(source)
        if not isinstance(source_records, Mapping) or set(source_records) != set(expected):
            raise InstrumentError("R5c health exact production arm coverage differs")
        for name, arm in expected.items():
            observed = source_records.get(name)
            if (
                not isinstance(observed, Mapping)
                or observed.get("constructor") != arm["constructor"]
                or observed.get("baseline_state") != arm["baseline_state"]
                or type(observed.get("strict_noninferiority_pass")) is not bool
            ):
                raise InstrumentError(
                    "R5c health production constructor or own baseline differs"
                )
            all_pass = all_pass and observed["strict_noninferiority_pass"]
    if record.get("all_arms_ok") is not all_pass:
        raise InstrumentError("R5c health all-arm disposition does not rederive")
    return {
        "status": "EXACT_PRODUCTION_HEALTH_REGISTRY_CONFIRMED",
        "arm_count_per_source": len(expected),
        "optional_basis_executed": include_optional_basis,
        "all_arms_ok": all_pass,
    }


def run_bounded_hook_health(
    adapter: Any,
    task: Any,
    rows: Sequence[Any],
    *,
    runtime: FrozenR5bGlobalBasis | None = None,
    max_examples: int = 16,
) -> dict[str, Any]:
    """Run every production N/S constructor on masked non-site probe outcomes."""

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
    downstream_registry = complete_downstream_registry()
    include_optional_basis = runtime is not None
    arm_registry = production_health_arm_registry(
        include_optional_basis=include_optional_basis
    )
    selected_bases = (
        None
        if runtime is None
        else runtime.row_bases(range(len(selected)))
    )
    random_bases = (
        ()
        if runtime is None
        else runtime.random_row_bases(range(len(selected)))
    )
    if include_optional_basis and len(random_bases) != 32:
        raise InstrumentError("R5c health requires exactly 32 random R5b projectors")

    def encoded_with_probes(sequences: Sequence[str]) -> Mapping[str, torch.Tensor]:
        encoded = task.encode(sequences, readouts)
        for row_index, positions in enumerate(probes):
            for position in positions:
                encoded["input_ids"][row_index, position + task.off] = task.mask_id
        return encoded

    def score(logits: torch.Tensor) -> np.ndarray:
        values: list[int] = []
        for row_index, (row, positions) in enumerate(zip(selected, probes)):
            for position in positions:
                token_logits = logits[row_index, position + task.off]
                predicted = max(
                    "ACGT", key=lambda base: float(token_logits[task.bid[base]].item())
                )
                values.append(int(predicted == row.clean_sequence[position]))
        return np.asarray(values, dtype=np.int8)

    def evaluate_arm(
        *,
        baseline: np.ndarray,
        intervention: np.ndarray,
        arm: Mapping[str, str],
    ) -> dict[str, Any]:
        if baseline.shape != intervention.shape or baseline.size != 2 * len(selected):
            raise InstrumentError("R5c paired health arm or own baseline is incomplete")
        harms = int(np.count_nonzero((baseline == 1) & (intervention == 0)))
        benefits = int(np.count_nonzero((baseline == 0) & (intervention == 1)))
        _lower, harm_upper = one_sided_binomial_bounds(
            harms,
            int(baseline.size),
            alpha=HEALTH_ALPHA,
        )
        return {
            **arm,
            "paired_probe_count": int(baseline.size),
            "baseline_correct": int(baseline.sum()),
            "intervention_correct": int(intervention.sum()),
            "paired_harms": harms,
            "paired_benefits": benefits,
            "paired_mean_accuracy_change": float(np.mean(intervention - baseline)),
            "harm_probability_one_sided_upper": harm_upper,
            "noninferiority_harm_margin": HEALTH_NONINFERIORITY_HARM_MARGIN,
            "strict_noninferiority_pass": bool(
                harm_upper < HEALTH_NONINFERIORITY_HARM_MARGIN
            ),
        }

    records: dict[str, dict[str, Any]] = {}
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
        query_token_indices = tuple(
            int(position) + int(task.off) for position in readouts
        )
        cache_device = clean_cache[L3_READER[0]].device
        batch_indices = torch.arange(len(query_token_indices), device=cache_device)
        token_indices = torch.tensor(
            query_token_indices,
            dtype=torch.long,
            device=cache_device,
        )
        clean_correct = score(clean_holder["output"].logits.float())
        for source, sequences in sources.items():
            corrupt_encoded = encoded_with_probes(sequences)
            corrupt_holder: dict[str, Any] = {}
            corrupt_cache = capture_head_outputs(
                adapter,
                lambda: corrupt_holder.setdefault(
                    "output", adapter.model(**corrupt_encoded)
                ),
            )
            corrupt_correct = score(corrupt_holder["output"].logits.float())
            layer, head = L3_READER
            clean_context = clean_cache[layer][
                batch_indices, head, token_indices
            ].float()
            corrupt_context = corrupt_cache[layer][
                batch_indices, head, token_indices
            ].float()
            writer_delta = (
                project_head_context_to_residual(adapter, L3_READER, clean_context)
                - project_head_context_to_residual(
                    adapter, L3_READER, corrupt_context
                )
            ).detach()
            writer_delta_array = writer_delta.float().cpu().numpy()

            writer_restored_holder: dict[str, Any] = {}
            writer_restored_audit: dict[str, Any] = {}
            with audited_add_attention_output_delta(
                adapter,
                layer=layer,
                token_mask=query_mask,
                delta=writer_delta,
                intervention_id=f"health:{source}:dense_writer_sufficiency",
                role="dense_local_writer_sufficiency",
                component_id=head_name(L3_READER),
                audit_sink=writer_restored_audit,
            ):
                writer_restored_cache = capture_head_outputs(
                    adapter,
                    lambda: writer_restored_holder.setdefault(
                        "output", adapter.model(**corrupt_encoded)
                    ),
                )
            writer_restored_correct = score(
                writer_restored_holder["output"].logits.float()
            )
            dense_necessity_audit: dict[str, Any] = {}
            with audited_add_attention_output_delta(
                adapter,
                layer=layer,
                token_mask=query_mask,
                delta=-writer_delta,
                intervention_id=f"health:{source}:dense_writer_necessity",
                role="dense_local_writer_necessity",
                component_id=head_name(L3_READER),
                audit_sink=dense_necessity_audit,
            ):
                dense_necessity_correct = score(
                    adapter.model(**clean_encoded).logits.float()
                )

            interventions: dict[str, np.ndarray] = {
                "C1|L3H1.output_projected|necessity": dense_necessity_correct,
                "C1|L3H1.output_projected|sufficiency": writer_restored_correct,
            }

            if selected_bases is not None:
                selected_projected = row_project(writer_delta_array, selected_bases)
                selected_sufficiency_audit: dict[str, Any] = {}
                with audited_add_attention_output_delta(
                    adapter,
                    layer=layer,
                    token_mask=query_mask,
                    delta=selected_projected,
                    intervention_id=f"health:{source}:selected_local_u_sufficiency",
                    role="selected_local_writer_subspace_sufficiency",
                    component_id="U_A",
                    audit_sink=selected_sufficiency_audit,
                ):
                    interventions["C2|selected_global_basis|sufficiency"] = score(
                        adapter.model(**corrupt_encoded).logits.float()
                    )
                selected_necessity_audit: dict[str, Any] = {}
                with audited_add_attention_output_delta(
                    adapter,
                    layer=layer,
                    token_mask=query_mask,
                    delta=-selected_projected,
                    intervention_id=f"health:{source}:selected_local_u_necessity",
                    role="selected_local_writer_subspace_necessity",
                    component_id="U_A",
                    audit_sink=selected_necessity_audit,
                ):
                    interventions["C2|selected_global_basis|necessity"] = score(
                        adapter.model(**clean_encoded).logits.float()
                    )
                for draw, random_set in enumerate(random_bases):
                    random_projected = energy_match_rows(
                        selected_projected,
                        row_project(writer_delta_array, random_set),
                    )
                    random_sufficiency_audit: dict[str, Any] = {}
                    with audited_add_attention_output_delta(
                        adapter,
                        layer=layer,
                        token_mask=query_mask,
                        delta=random_projected,
                        intervention_id=(
                            f"health:{source}:random_local_u_{draw:02d}_sufficiency"
                        ),
                        role="random_local_writer_subspace_sufficiency",
                        component_id=f"random_projector_{draw:02d}",
                        audit_sink=random_sufficiency_audit,
                    ):
                        interventions[
                            f"C2|random_global_basis_{draw:02d}|sufficiency"
                        ] = score(adapter.model(**corrupt_encoded).logits.float())
                    random_necessity_audit: dict[str, Any] = {}
                    with audited_add_attention_output_delta(
                        adapter,
                        layer=layer,
                        token_mask=query_mask,
                        delta=-random_projected,
                        intervention_id=(
                            f"health:{source}:random_local_u_{draw:02d}_necessity"
                        ),
                        role="random_local_writer_subspace_necessity",
                        component_id=f"random_projector_{draw:02d}",
                        audit_sink=random_necessity_audit,
                    ):
                        interventions[
                            f"C2|random_global_basis_{draw:02d}|necessity"
                        ] = score(adapter.model(**clean_encoded).logits.float())

            for ensemble in downstream_registry:
                component = ensemble_name(ensemble)
                writer_context_audit: dict[str, Any] = {}
                necessity_audit: dict[str, Any] = {}
                with ExitStack() as stack:
                    stack.enter_context(
                        audited_add_attention_output_delta(
                            adapter,
                            layer=layer,
                            token_mask=query_mask,
                            delta=writer_delta,
                            intervention_id=(
                                f"health:{source}:{component}:writer_restoration"
                            ),
                            role="delivery_writer_restoration",
                            component_id=component,
                            audit_sink=writer_context_audit,
                        )
                    )
                    stack.enter_context(
                        audited_patch_heads(
                            adapter,
                            corrupt_cache,
                            ensemble,
                            token_mask=query_mask,
                            intervention_id=(
                                f"health:{source}:{component}:delivery_necessity"
                            ),
                            role="delivery_necessity",
                            component_id=component,
                            audit_sink=necessity_audit,
                        )
                    )
                    interventions[f"delivery|{component}|necessity"] = score(
                        adapter.model(**corrupt_encoded).logits.float()
                    )
                sufficiency_audit: dict[str, Any] = {}
                with audited_patch_heads(
                    adapter,
                    writer_restored_cache,
                    ensemble,
                    token_mask=query_mask,
                    intervention_id=(
                        f"health:{source}:{component}:delivery_sufficiency"
                    ),
                    role="delivery_sufficiency",
                    component_id=component,
                    audit_sink=sufficiency_audit,
                ):
                    interventions[f"delivery|{component}|sufficiency"] = score(
                        adapter.model(**corrupt_encoded).logits.float()
                    )

            if set(interventions) != set(arm_registry):
                raise InstrumentError("R5c health exact production arm coverage differs")
            baseline_states = {
                "clean": clean_correct,
                "corrupt": corrupt_correct,
                "writer_restored": writer_restored_correct,
            }
            source_records: dict[str, Any] = {}
            for name, arm in arm_registry.items():
                source_records[name] = evaluate_arm(
                    baseline=baseline_states[arm["baseline_state"]],
                    intervention=interventions[name],
                    arm=arm,
                )
            records[source] = source_records
    result = {
        "schema": "gfmcd.r5c.paired_noop_health/3",
        "examples": len(selected),
        "probes_per_example": 2,
        "candidate_outcomes_used": False,
        "comparison": "paired intervention-minus-own-baseline correctness",
        "optional_basis_executed": include_optional_basis,
        "arm_registry": arm_registry,
        "arm_registry_sha256": canonical_sha256(arm_registry),
        "absolute_accuracy_floor_used": False,
        "inclusive_boundary_used": False,
        "alpha": HEALTH_ALPHA,
        "noninferiority_harm_margin": HEALTH_NONINFERIORITY_HARM_MARGIN,
        "records": records,
        "all_arms_ok": all(
            record["strict_noninferiority_pass"]
            for source_records in records.values()
            for record in source_records.values()
        ),
    }
    validate_production_health_record(
        result,
        include_optional_basis=include_optional_basis,
    )
    return result


def load_exact_json_package(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    expected_artifact_sha256: str,
    expected_manifest_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Load an exact clean JSON package using relocation-safe logical names."""

    artifact = Path(artifact_path).expanduser().resolve()
    manifest_file = Path(manifest_path).expanduser().resolve()
    for kind, expected in (
        ("artifact", expected_artifact_sha256),
        ("manifest", expected_manifest_sha256),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None:
            raise InstrumentError(f"{label} literal {kind} pin is malformed")
    try:
        artifact_sha256 = sha256_file(artifact)
        manifest_sha256 = sha256_file(manifest_file)
    except OSError as exc:
        raise InstrumentError(f"{label} artifact/manifest cannot be read: {exc}") from exc
    if artifact_sha256 != expected_artifact_sha256:
        raise InstrumentError(f"{label} artifact bytes differ from the literal pin")
    if manifest_sha256 != expected_manifest_sha256:
        raise InstrumentError(f"{label} manifest bytes differ from the literal pin")
    try:
        record = json.loads(artifact.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InstrumentError(f"{label} artifact/manifest is not valid JSON: {exc}") from exc
    if not isinstance(record, dict) or not isinstance(manifest, dict):
        raise InstrumentError(f"{label} artifact/manifest schema is not an object")
    outputs = manifest.get("outputs")
    matches = [
        item
        for item in outputs if isinstance(outputs, list) and isinstance(item, Mapping)
        if Path(str(item.get("path", ""))).name == artifact.name
        and item.get("sha256") == artifact_sha256
        and item.get("size_bytes") == artifact.stat().st_size
    ] if isinstance(outputs, list) else []
    if len(matches) != 1:
        raise InstrumentError(f"{label} manifest does not independently bind its result")
    if manifest.get("git_dirty") is not False:
        raise InstrumentError(f"{label} was not produced from a clean worktree")
    identity = {
        "artifact": {
            "logical_name": artifact.name,
            "size_bytes": artifact.stat().st_size,
            "sha256": artifact_sha256,
        },
        "manifest": {
            "logical_name": manifest_file.name,
            "size_bytes": manifest_file.stat().st_size,
            "sha256": manifest_sha256,
        },
    }
    return record, manifest, identity


def validate_r5a_stage_record(
    record: Mapping[str, Any],
    *,
    stage: str,
    expected_stack_fingerprint: str,
    capstone_handoff: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the R5a /2 predecessor without reopening R3c as evidence."""

    if stage not in {"preflight", "target"}:
        raise InstrumentError("R5a predecessor stage must be preflight or target")
    expected_status = (
        "PREFLIGHT_PASS"
        if stage == "preflight"
        else "L3H1_DONOR_LOCAL_READER_SUPPORTED"
    )
    contract = record.get("r5_entry_contract")
    endpoint = record.get("behavioral_endpoint")
    donor_endpoint = record.get(
        "masked_donor_endpoint_constructed",
        record.get("donor_readout_constructed"),
    )
    if (
        record.get("schema") != "gfmcd.r5a.acceptor_address_binding/2"
        or record.get("stage") != stage
        or record.get("status") != expected_status
        or record.get("all_instrument_gates_pass") is not True
        or endpoint != "masked_acceptor_G_margin_only"
        or donor_endpoint is not False
        or not isinstance(contract, Mapping)
        or contract.get("schema") != R5_ENTRY_CONTRACT_SCHEMA
        or contract.get("instrument_gates_pass") is not True
        or contract.get("verdict") != expected_status
        or contract.get("r3c_used_as_scientific_entry") is not False
        or contract.get("masked_donor_endpoint_constructed") is not False
    ):
        raise InstrumentError(f"R5a {stage} /2 contract differs")
    if stage == "target" and contract.get("claim_eligible") is not True:
        raise ScientificStop("R5a target reader was not supported")
    if re.fullmatch(r"[0-9a-f]{64}", str(expected_stack_fingerprint)) is None:
        raise InstrumentError("expected joint-stack fingerprint is malformed")
    if (
        record.get(STACK_FINGERPRINT_FIELD) != expected_stack_fingerprint
        or contract.get(STACK_FINGERPRINT_FIELD) != expected_stack_fingerprint
    ):
        raise InstrumentError("R5a and R5c joint-stack fingerprints differ")
    checkpoint_sha256 = contract.get("checkpoint_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(checkpoint_sha256)) is None:
        raise InstrumentError("R5a checkpoint binding is malformed")

    same_row_capstone_bound = False
    if stage == "target":
        if not isinstance(capstone_handoff, Mapping):
            raise InstrumentError("R5a target requires the authenticated R1c capstone handoff")
        expected_binding = {
            "capstone_cohort_schema": capstone_handoff.get("schema"),
            "capstone_row_count": capstone_handoff.get("row_count"),
            "capstone_cohort_sha256": capstone_handoff.get("cohort_sha256"),
            "capstone_ordered_row_ids_sha256": capstone_handoff.get(
                "ordered_row_ids_sha256"
            ),
            "capstone_ordered_gene_ids_sha256": capstone_handoff.get(
                "ordered_gene_ids_sha256"
            ),
            "capstone_sorted_unique_gene_ids_sha256": capstone_handoff.get(
                "sorted_unique_gene_ids_sha256"
            ),
            "capstone_construction_sha256": capstone_handoff.get(
                "capstone_construction_sha256"
            ),
        }
        if expected_binding["capstone_cohort_schema"] != R5_CAPSTONE_COHORT_HANDOFF_SCHEMA:
            raise InstrumentError("R5a target capstone handoff schema differs")
        if any(contract.get(key) != value for key, value in expected_binding.items()):
            raise InstrumentError("R5a target capstone handoff differs")
        same_row_capstone_bound = True
    elif capstone_handoff is not None:
        raise InstrumentError("R5a preflight must not consume the future capstone handoff")
    return {
        "stage": stage,
        "status": expected_status,
        "predecessor_entry_nonuse_authenticated": True,
        "same_row_capstone_bound": same_row_capstone_bound,
        "checkpoint_sha256": checkpoint_sha256,
        STACK_FINGERPRINT_FIELD: expected_stack_fingerprint,
        "entry_contract_sha256": canonical_sha256(contract),
    }


def load_r5a_stage_package(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    stage: str,
    authorization: Mapping[str, str],
    expected_stack_fingerprint: str,
    capstone_handoff: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    prefix = "R5A_PREFLIGHT" if stage == "preflight" else "R5A_TARGET"
    record, _manifest, identity = load_exact_json_package(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        expected_artifact_sha256=authorization[f"FROZEN_{prefix}_ARTIFACT_SHA256"],
        expected_manifest_sha256=authorization[f"FROZEN_{prefix}_MANIFEST_SHA256"],
        label=f"R5a {stage}",
    )
    validated = validate_r5a_stage_record(
        record,
        stage=stage,
        expected_stack_fingerprint=expected_stack_fingerprint,
        capstone_handoff=capstone_handoff,
    )
    return record, identity, validated


def _load_bound_r5b_disposition_validator() -> Any:
    """Lazy-load the stack-bound R5b offline validator without an import cycle."""

    module_name = "_gfmcd_bound_r5b_disposition_validator"
    spec = importlib.util.spec_from_file_location(module_name, R5B_RUNNER)
    if spec is None or spec.loader is None:
        raise InstrumentError("cannot load the bound R5b disposition validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise InstrumentError(f"R5b disposition validator import failed: {exc}") from exc
    validator = getattr(module, "validate_local_writer_basis_disposition", None)
    if not callable(validator):
        raise InstrumentError("bound R5b runner lacks its public disposition validator")
    return validator


def load_r5b_disposition_package(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    bundle_path: str | Path | None,
    derivation_artifact_path: str | Path,
    derivation_manifest_path: str | Path,
    validation_artifact_path: str | Path,
    validation_manifest_path: str | Path,
    authorization: Mapping[str, str],
    expected_stack_fingerprint: str,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, Any] | None,
    FrozenR5bGlobalBasis | None,
]:
    """Authenticate the terminal R5b disposition; make C2 strictly path-local."""

    record, _manifest, identity = load_exact_json_package(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        expected_artifact_sha256=authorization[
            "FROZEN_R5B_DISPOSITION_ARTIFACT_SHA256"
        ],
        expected_manifest_sha256=authorization[
            "FROZEN_R5B_DISPOSITION_MANIFEST_SHA256"
        ],
        label="frozen R5b basis disposition",
    )
    validator = _load_bound_r5b_disposition_validator()
    try:
        verified = validator(
            artifact_path,
            manifest_path,
            bundle_path=bundle_path,
            derivation_artifact_path=derivation_artifact_path,
            derivation_manifest_path=derivation_manifest_path,
            validation_artifact_path=validation_artifact_path,
            validation_manifest_path=validation_manifest_path,
        )
    except Exception as exc:
        raise InstrumentError(f"R5b disposition authentication failed: {exc}") from exc
    if (
        not isinstance(verified, Mapping)
        or record.get("schema") != R5B_DISPOSITION_SCHEMA
        or verified.get("authenticated") is not True
        or verified.get("record") != record
        or verified.get("artifact") != identity["artifact"]
        or verified.get("manifest") != identity["manifest"]
        or verified.get("status") not in R5B_DISPOSITION_STATUSES
        or verified.get("dense_primary_unaffected") is not True
    ):
        raise InstrumentError("R5b disposition verifier result differs")
    contract_value = verified.get("frozen_basis_runtime_contract")
    contract = dict(contract_value) if isinstance(contract_value, Mapping) else None
    runtime: FrozenR5bGlobalBasis | None = None
    if verified["status"] == "VALIDATED_BASIS":
        if contract is None or bundle_path is None:
            raise InstrumentError("validated R5b disposition lacks its runtime bundle")
        runtime = load_frozen_r5b_global_basis(
            bundle_path,
            contract=contract,
            expected_stack_fingerprint=expected_stack_fingerprint,
        )
    elif contract is not None or bundle_path is not None:
        raise InstrumentError("nonvalidated R5b disposition must not load a basis bundle")
    return dict(verified), identity, contract, runtime


def load_r1c_preflight_handoff(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    authorization: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate R1c's exact nonsealed 400-row validation handoff."""

    try:
        handoff = load_r1c_preflight_cohort_handoff(artifact_path, manifest_path)
        verified = validate_r5_preflight_cohort_handoff(handoff)
    except (R5CapstoneCohortError, OSError, TypeError, ValueError) as exc:
        raise InstrumentError(f"R1c preflight handoff authentication failed: {exc}") from exc
    if (
        verified.get("artifact", {}).get("sha256")
        != authorization["FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256"]
        or verified.get("manifest", {}).get("sha256")
        != authorization["FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256"]
        or verified.get("schema") != R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA
        or verified.get("verified") is not True
        or verified.get("row_count") != R5C_EXACT_ROWS
        or verified.get("completion_receipt_available") is not False
        or verified.get("scientific_entry_authorized") is not False
        or verified.get("residual_validation_cohort_used") is not False
    ):
        raise InstrumentError("R1c validation-preflight handoff disposition differs")
    return handoff, verified


def materialize_r1c_preflight_rows(
    *,
    gtf: str | Path,
    fasta: str | Path,
    handoff: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Materialize all authenticated val rows in R1c order with no filtering."""

    source_identity = validate_r5_source_binding(gtf, fasta)
    examples = load_examples_unscored(
        str(gtf),
        str(fasta),
        "val",
        None,
        tag="r1c-preflight-r1c-single-g-primary-family-v2",
        allowed_chromosomes=R5_VALIDATION_CHROMOSOMES,
    )
    try:
        rows, audit = materialize_r5_preflight_rows(examples, handoff)
    except (R5CapstoneCohortError, RuntimeError, TypeError, ValueError) as exc:
        raise InstrumentError(f"R5c preflight materialization failed: {exc}") from exc
    if (
        len(rows) != R5C_EXACT_ROWS
        or audit.get("handoff_order_preserved") is not True
        or audit.get("rows_filtered_or_reordered") is not False
    ):
        raise InstrumentError("R5c preflight materialization is not exact and complete")
    return rows, {
        "source_identity": source_identity,
        "audit": audit,
        "binding": {
            key: handoff[key]
            for key in (
                "schema",
                "row_count",
                "cohort_sha256",
                "ordered_row_ids_sha256",
                "ordered_gene_ids_sha256",
                "sorted_unique_gene_ids_sha256",
                "unique_gene_count",
                "ordered_partner_margin_levels_sha256",
                "source_package_sha256",
                "preflight_construction_sha256",
            )
        }
        | {"materialization": audit},
        "role": "AUTHENTICATED_R1C_NONSEALED_VALIDATION_400_ROW_PREFLIGHT",
        "r5c_capstone_authorized": False,
    }


def load_r1c_target_handoff(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    attempt_path: str | Path,
    completion_path: str | Path,
    preflight_artifact_path: str | Path,
    preflight_manifest_path: str | Path,
    authorization: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the shared exact six-file R1c verifier/handoff chain."""

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
    except (R5CapstoneCohortError, OSError, TypeError, ValueError) as exc:
        raise InstrumentError(f"R1c capstone handoff authentication failed: {exc}") from exc
    package_fields = {
        "artifact": "FROZEN_R1C_TARGET_ARTIFACT_SHA256",
        "manifest": "FROZEN_R1C_TARGET_MANIFEST_SHA256",
        "attempt": "FROZEN_R1C_TARGET_ATTEMPT_SHA256",
        "completion": "FROZEN_R1C_TARGET_COMPLETION_SHA256",
        "preflight_artifact": "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
        "preflight_manifest": "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256",
    }
    if any(
        not isinstance(verified.get(kind), Mapping)
        or verified[kind].get("sha256") != authorization[pin]
        for kind, pin in package_fields.items()
    ):
        raise InstrumentError("R1c handoff package differs from R5c's literal pins")
    if (
        verified.get("verified") is not True
        or verified.get("row_count") != 400
        or verified.get("residual_validation_bucket_1_used") is not False
        or verified.get("completion_receipt_available") is not True
    ):
        raise InstrumentError("R1c capstone handoff disposition differs")
    return handoff, verified


def load_r1c_target_common_leg_references(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    attempt_path: str | Path,
    completion_path: str | Path,
    preflight_artifact_path: str | Path,
    preflight_manifest_path: str | Path,
    handoff: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load R1c outcomes only after the caller has created its target receipt."""

    try:
        package = load_r1c_verified_target_common_legs(
            artifact_path,
            manifest_path,
            attempt_path,
            completion_path,
            preflight_artifact_path,
            preflight_manifest_path,
        )
    except (R5CapstoneCohortError, OSError, TypeError, ValueError) as exc:
        raise InstrumentError(
            f"verified R1c target common-leg load failed: {exc}"
        ) from exc
    references = r1c_verified_target_common_leg_references(package, handoff)
    return references, {
        "schema": package["schema"],
        "source_package_sha256": package["source_package_sha256"],
        "successor_attestation_sha256": package[
            "successor_attestation_sha256"
        ],
        "cohort_sha256": package["cohort_sha256"],
        "ordered_row_ids_sha256": package["ordered_row_ids_sha256"],
        "ordered_partner_margin_levels_sha256": package[
            "ordered_partner_margin_levels_sha256"
        ],
        "record_sha256": package["record_sha256"],
        "row_count": package["row_count"],
        "authorized_use": package["authorized_use"],
        "loaded_after_r5c_target_receipt": True,
        "used_for_cohort_selection": False,
    }


def validate_r5c_preflight_record(
    record: Mapping[str, Any],
    *,
    fingerprint: str,
    r1c_preflight_identity: Mapping[str, Any],
    r5a_preflight_identity: Mapping[str, Any],
    r5b_disposition_identity: Mapping[str, Any],
    r5b_basis_contract_sha256: str | None,
    r5b_status: str,
) -> dict[str, Any]:
    """Validate exact 400-row R1c validation-preflight instrument evidence."""

    cohort = record.get("cohort")
    inputs = record.get("preflight_inputs")
    measurement = record.get("measurement")
    if (
        record.get("schema") != SCHEMA
        or record.get("stage") != "preflight"
        or record.get("status") != "PREFLIGHT_INSTRUMENT_PASS"
        or record.get("instrument_fingerprint_sha256") != fingerprint
        or record.get("sealed_chr8_chr9_opened") is not False
        or record.get("scientific_outcomes_claim_eligible") is not False
        or not isinstance(cohort, Mapping)
        or cohort.get("role")
        != "AUTHENTICATED_R1C_NONSEALED_VALIDATION_400_ROW_PREFLIGHT"
        or cohort.get("row_count") != R5C_EXACT_ROWS
        or cohort.get("rows_filtered_or_reordered") is not False
        or cohort.get("r5c_capstone_authorized") is not False
        or not isinstance(inputs, Mapping)
        or inputs.get("r1c_validation_preflight") != r1c_preflight_identity
        or inputs.get("r5a_preflight") != r5a_preflight_identity
        or inputs.get("r5b_disposition") != r5b_disposition_identity
        or inputs.get("r5b_status") != r5b_status
        or inputs.get("r5b_basis_contract_sha256")
        != r5b_basis_contract_sha256
        or inputs.get("future_r1c_or_r5a_target_required") is not False
        or not isinstance(measurement, Mapping)
        or measurement.get("instrument_ok") is not True
        or record.get("binding_suite", {}).get("status") != "PASS"
        or record.get("engineering_calibration", {}).get("decision_authority")
        is not False
        or measurement.get("observed_centered_robust_sensitivity", {}).get(
            "status"
        )
        != "PASS"
    ):
        raise InstrumentError("R5c R1c-validation preflight contract differs")
    families = measurement.get("families")
    if not isinstance(families, Mapping) or tuple(families) != FAMILY_NAMES:
        raise InstrumentError("R5c preflight family registry differs")
    basis_plan = optional_basis_execution_plan(
        status=r5b_status,
        runtime=(object() if r5b_status == "VALIDATED_BASIS" else None),
    )
    if measurement.get("optional_r5b_basis") != {
        "disposition_status": r5b_status,
        "execution_plan": basis_plan,
    }:
        raise InstrumentError("R5c preflight optional-basis disposition differs")
    for family in FAMILY_NAMES:
        family_record = families[family]
        if not isinstance(family_record, Mapping):
            raise InstrumentError("R5c preflight family record is malformed")
        if family == "C2" and not basis_plan["execute_c2"]:
            if family_record != {
                "executed": False,
                "disposition": basis_plan["classification"],
                "dense_primary_unaffected": True,
            }:
                raise InstrumentError("R5c preflight C2 skip record differs")
            continue
        inference = family_record.get("inference", {})
        diagnostic = family_record.get("sign_flip_diagnostic", {})
        if (
            family_record.get("executed") is not True
            or
            inference.get("B") != PRODUCTION_BOOTSTRAP_DRAWS
            or inference.get("inference_kind")
            != "studentized_gene_bootstrap_asymptotic"
            or diagnostic.get("decision_authority") is not False
        ):
            raise InstrumentError("R5c preflight inference authority differs")
    return {
        "status": "FROZEN_R1C_VALIDATION_PREFLIGHT_PASS",
        "future_target_dependency_used": False,
        "r5c_capstone_cohort_opened": False,
        "r5b_disposition_frozen_pre_unseal": True,
        "r5b_status": r5b_status,
        "preflight_record_sha256": canonical_sha256(record),
    }


def validate_r5_source_binding(gtf: str | Path, fasta: str | Path) -> dict[str, Any]:
    """Require the comprehensive-GTF and hg38 source bytes used for the census."""

    try:
        return validate_expected_r5_source_identity(gtf, fasta)
    except (OSError, RuntimeError, ValueError) as exc:
        raise InstrumentError(f"R5 source binding failed: {exc}") from exc


CAPABILITY_GATES = (
    "single_masked_acceptor_G_readout",
    "no_masked_donor_endpoint",
    "full_mlm_logits_and_margin_replay",
    "dense_full_residual_replay",
    "l3_query_self_patch_noop",
    "all_registered_head_query_self_patch_noop",
    "local_writer_query_only_isolation",
    "common_legs_bit_identical",
    "optional_r5b_basis_execution_coherent",
    "optional_r5b_random_projectors_coherent",
    "ensemble_membership_and_patch_size_exact",
    "necessity_and_sufficiency_health_complete",
    "robust_authority_binding_suite_pass",
    "observed_centered_robust_sensitivity_pass",
    "candidate_outcomes_not_used_for_capability",
)


def validate_post_wo_delta_audit(
    record: Mapping[str, Any],
    *,
    expected_role: str | None = None,
    expected_component_id: str | None = None,
) -> dict[str, Any]:
    required = {
        "schema",
        "intervention_id",
        "role",
        "component_id",
        "layer",
        "writer_site_id",
        "bias_included_in_component",
        "n_rows",
        "sequence_tokens",
        "requested_token_indices",
        "requested_token_mask_sha256",
        "requested_delta_sha256",
        "observed_delta_sha256",
        "before_dense_output_sha256",
        "after_dense_output_sha256",
        "delta_replay_max_abs_error",
        "off_mask_exact",
        "hook_count",
        "audit_sha256",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != required
        or record.get("schema") != "gfmcd.r5c.post_wo_delta_audit/1"
        or record.get("writer_site_id") != WRITER_SITE_ID
        or record.get("bias_included_in_component") is not False
        or record.get("layer") != L3_READER[0]
    ):
        raise InstrumentError("post-W_O delta audit schema/site differs")
    unsigned = {key: record[key] for key in required if key != "audit_sha256"}
    if record["audit_sha256"] != canonical_sha256(unsigned):
        raise InstrumentError("post-W_O delta audit digest does not replay")
    if expected_role is not None and record.get("role") != expected_role:
        raise InstrumentError("post-W_O delta audit role differs")
    if expected_component_id is not None and record.get("component_id") != expected_component_id:
        raise InstrumentError("post-W_O delta audit component differs")
    if not all(
        isinstance(record.get(field), str) and record.get(field)
        for field in ("intervention_id", "role", "component_id")
    ):
        raise InstrumentError("post-W_O delta audit identity is incomplete")
    n_rows = record.get("n_rows")
    sequence_tokens = record.get("sequence_tokens")
    indices = record.get("requested_token_indices")
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
        raise InstrumentError("post-W_O delta audit token coordinates differ")
    reconstructed = np.zeros((int(n_rows), int(sequence_tokens)), dtype=np.uint8)
    reconstructed[np.arange(int(n_rows)), np.asarray(indices, dtype=int)] = 1
    if record.get("requested_token_mask_sha256") != _boolean_array_sha256(reconstructed):
        raise InstrumentError("post-W_O delta audit mask does not replay")
    hash_fields = (
        "requested_delta_sha256",
        "observed_delta_sha256",
        "before_dense_output_sha256",
        "after_dense_output_sha256",
    )
    if any(re.fullmatch(r"[0-9a-f]{64}", str(record.get(field))) is None for field in hash_fields):
        raise InstrumentError("post-W_O delta tensor digest is malformed")
    error = record.get("delta_replay_max_abs_error")
    if (
        isinstance(error, bool)
        or not isinstance(error, (int, float))
        or not np.isfinite(float(error))
        or float(error) > 1e-5
        or record.get("off_mask_exact") is not True
        or record.get("hook_count") != 1
    ):
        raise InstrumentError("post-W_O delta was not isolated to the exact query")
    return {
        "audit_sha256": str(record["audit_sha256"]),
        "requested_token_indices": [int(value) for value in indices],
        "writer_site_id": WRITER_SITE_ID,
        "bias_included_in_component": False,
    }


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
    optional_basis_executed = leg.get("local_writer_subspace") is not None
    expected_random_count = 32 if optional_basis_executed else 0
    if len(leg.get("random_local_writer_subspaces", ())) != expected_random_count:
        raise InstrumentError("source leg optional-basis primitive registry differs")
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
        head_singleton_roles = {
            "registered_clean_self_patch": (registered, "ALL_REGISTERED_HEADS"),
            "registered_minus_self_patch": (registered, "ALL_REGISTERED_HEADS"),
            "pre_wo_writer_equivalence_diagnostic": (
                (L3_READER,),
                head_name(L3_READER),
            ),
            "direct_negative": ((DIRECT_NEGATIVE,), head_name(DIRECT_NEGATIVE)),
        }
        post_wo_singleton_roles = {
            "dense_local_writer_sufficiency": head_name(L3_READER),
            "dense_local_writer_necessity": (
                head_name(L3_READER)
            ),
        }
        if optional_basis_executed:
            post_wo_singleton_roles.update(
                {
                    "selected_local_writer_subspace_sufficiency": "U_A",
                    "selected_local_writer_subspace_necessity": "U_A",
                }
            )
        query_indices: list[int] | None = None
        for role, (heads, component_id) in head_singleton_roles.items():
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
        for role, component_id in post_wo_singleton_roles.items():
            if len(by_role.get(role, ())) != 1:
                raise InstrumentError(f"mutation-audit chunk lacks exactly one {role}")
            validated = validate_post_wo_delta_audit(
                by_role[role][0],
                expected_role=role,
                expected_component_id=component_id,
            )
            if validated["requested_token_indices"] != query_indices:
                raise InstrumentError("post-W_O mutation used a wrong token within a chunk")
            all_digests.append(validated["audit_sha256"])
        if query_indices != expected_query_indices_by_chunk[chunk_index]:
            raise InstrumentError("mutation audit used a wrong masked-acceptor token")
        random_roles = (
            "random_local_writer_subspace_sufficiency",
            "random_local_writer_subspace_necessity",
        )
        for role in random_roles:
            records_for_role = by_role.get(role, ())
            if len(records_for_role) != expected_random_count:
                raise InstrumentError(f"mutation-audit {role} registry is incomplete")
            by_component = {
                str(record.get("component_id")): record
                for record in records_for_role
            }
            expected_random = {
                f"random_projector_{draw:02d}"
                for draw in range(expected_random_count)
            }
            if set(by_component) != expected_random:
                raise InstrumentError(f"mutation-audit {role} component registry differs")
            for component_id in sorted(expected_random):
                validated = validate_post_wo_delta_audit(
                    by_component[component_id],
                    expected_role=role,
                    expected_component_id=component_id,
                )
                if validated["requested_token_indices"] != query_indices:
                    raise InstrumentError("mutation audit used a wrong token within a chunk")
                all_digests.append(validated["audit_sha256"])

        delivery_roles = (
            "delivery_writer_restoration",
            "delivery_necessity",
            "delivery_sufficiency",
        )
        for role in delivery_roles:
            records_for_role = by_role.get(role, ())
            if len(records_for_role) != len(components):
                raise InstrumentError(f"mutation-audit {role} registry is incomplete")
            by_component = {str(record.get("component_id")): record for record in records_for_role}
            if set(by_component) != set(components):
                raise InstrumentError(f"mutation-audit {role} component registry differs")
            for component_id, ensemble in components.items():
                if role == "delivery_writer_restoration":
                    validated = validate_post_wo_delta_audit(
                        by_component[component_id],
                        expected_role=role,
                        expected_component_id=component_id,
                    )
                else:
                    validated = validate_head_mutation_audit(
                        by_component[component_id],
                        expected_heads=ensemble,
                        expected_role=role,
                        expected_component_id=component_id,
                    )
                if validated["requested_token_indices"] != query_indices:
                    raise InstrumentError("mutation audit used a wrong token within a chunk")
                all_digests.append(validated["audit_sha256"])
        expected_roles = (
            set(head_singleton_roles)
            | set(post_wo_singleton_roles)
            | (set(random_roles) if optional_basis_executed else set())
            | set(delivery_roles)
        )
        if set(by_role) != expected_roles:
            raise InstrumentError("mutation-audit chunk contains an undeclared role")

        for component_id in components:
            envelope = leg["delivery"][component_id]
            audit_lists = envelope.get("mutation_audit_sha256_by_chunk", {})
            if not isinstance(audit_lists, Mapping):
                raise InstrumentError("sandwich lacks independent mutation records")
            direction_map = {
                "writer_restoration": "delivery_writer_restoration",
                "necessity": "delivery_necessity",
                "sufficiency": "delivery_sufficiency",
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
    native = leg.get("native_conditions", {}).get("clean_margin", ())
    if total_rows != len(native) or total_rows < MIN_GENE_CLUSTERS:
        raise InstrumentError("mutation audits do not cover every supplied R5c row")
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
        "optional_r5b_basis_executed": optional_basis_executed,
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
    delivery = leg.get("delivery")
    chunks = leg.get("chunk_condition_hashes")
    hook_capability = leg.get("hook_capability")
    if (
        not isinstance(delivery, Mapping)
        or not isinstance(chunks, list)
        or not chunks
        or not isinstance(hook_capability, Mapping)
    ):
        raise InstrumentError("source leg lacks captured common-leg identities")
    canonical = {
        component: _replay_delivery_envelope(record)
        for component, record in delivery.items()
    }
    first = next(iter(canonical.values()))
    captured_records = [
        chunk[field]
        for chunk in chunks
        for field in (
            "m_corrupt_capture_identity",
            "m_writer_restored_capture_identity",
        )
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
        "m_corrupt_capture_sha256": canonical_sha256(
            [
                chunk["m_corrupt_capture_identity"]["capture_sha256"]
                for chunk in chunks
            ]
        ),
        "m_writer_restored_capture_sha256": canonical_sha256(
            [
                chunk["m_writer_restored_capture_identity"]["capture_sha256"]
                for chunk in chunks
            ]
        ),
    }
    common = validate_local_writer_common_legs(
        [record["common_leg"] for record in delivery.values()],
        m_corrupt=first["cells"]["m_corrupt"],
        m_writer_restored=first["cells"]["m_writer_restored"],
        capture_identity=capture_identity,
    )
    if dict(leg.get("common_leg", {})) != common:
        raise InstrumentError("source common-leg identity differs from captured tensors")
    return {
        "leg_id": leg.get("leg_id"),
        "common_leg_pair_sha256": common["common_leg_pair_sha256"],
        "captured_tensor_identity_sha256": canonical_sha256(capture_identity),
        "components": len(delivery),
    }


def validate_projector_capability_audit(
    record: Mapping[str, Any],
    *,
    expected_runtime_array_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    common_required = {
        "schema",
        "selected",
        "random",
        "authenticated_runtime_array_sha256",
        "observed_runtime_array_sha256",
        "all_projectors_validated",
        "audit_sha256",
    }
    required = common_required | {"application", "expected_rows"}
    if (
        not isinstance(record, Mapping)
        or set(record) != required
        or record.get("schema") != "gfmcd.r5c.global_projector_capability/2"
    ):
        raise InstrumentError("global-basis projector capability schema differs")
    expected_rows = record.get("expected_rows")
    if (
        not _is_integer(expected_rows)
        or int(expected_rows) < 1
        or (
            record.get("application")
            != "one authenticated global basis broadcast unchanged; no target folds"
        )
    ):
        raise InstrumentError("projector application policy differs")
    selected = record["selected"]
    random_records = record["random"]
    if not isinstance(selected, Mapping) or not isinstance(random_records, list) or len(random_records) != 32:
        raise InstrumentError("global-basis projector audit registry differs")
    if not _is_integer(selected.get("selected_rank")) or int(
        selected["selected_rank"]
    ) < 1:
        raise InstrumentError("global-basis projector selected rank is malformed")
    all_records = [selected, *random_records]
    for index, audit in enumerate(all_records):
        try:
            symmetry = float(audit.get("symmetry_max_abs"))
            idempotence = float(audit.get("idempotence_max_abs"))
        except (TypeError, ValueError):
            raise InstrumentError("global-basis projector diagnostic is malformed") from None
        if (
            audit.get("n_projectors") != int(expected_rows)
            or audit.get("selected_rank") != selected.get("selected_rank")
            or not np.isfinite([symmetry, idempotence]).all()
            or symmetry > 1e-7
            or idempotence > 1e-7
            or audit.get("representation")
            not in {"square_projector", "row_orthonormal_basis"}
            or not re.fullmatch(r"[0-9a-f]{64}", str(audit.get("operator_array_sha256")))
            or (index and audit.get("draw") != index - 1)
        ):
            raise InstrumentError("global-basis projector diagnostic differs")
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
        raise InstrumentError("global-basis projectors differ from authenticated arrays")
    if (
        record["all_projectors_validated"] != 33
        or record["audit_sha256"]
        != canonical_sha256({"selected": selected, "random": random_records})
    ):
        raise InstrumentError("global-basis projector audit digest differs")
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
    optional_basis = evidence.get("optional_r5b_basis")
    if not isinstance(optional_basis, Mapping):
        raise InstrumentError("capability lacks the optional R5b basis disposition")
    status = str(optional_basis.get("disposition_status"))
    executed = optional_basis.get("executed") is True
    plan = optional_basis_execution_plan(
        status=status,
        runtime=object() if executed else None,
    )
    if optional_basis.get("execution_plan") != plan:
        raise InstrumentError("optional R5b basis execution plan differs")
    projector_validation: Mapping[str, Any] | None = None
    projector_record = evidence.get("global_basis_projectors")
    if executed:
        projector_validation = validate_projector_capability_audit(
            projector_record if isinstance(projector_record, Mapping) else {},
            expected_runtime_array_sha256=expected_runtime_array_sha256,
        )
    elif projector_record is not None or expected_runtime_array_sha256 is not None:
        raise InstrumentError("nonexecuted R5b basis has projector/runtime evidence")
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
    health_validation = validate_production_health_record(
        health,
        include_optional_basis=executed,
    )
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
        "local_writer_query_only_isolation": all(
            item["query_only"] is True for item in mutations.values()
        ),
        "common_legs_bit_identical": all(
            item["components"] == len(complete_downstream_registry())
            for item in common.values()
        ),
        "optional_r5b_basis_execution_coherent": (
            all(
                item["optional_r5b_basis_executed"] is executed
                for item in mutations.values()
            )
            and (
                projector_validation is not None
                and projector_validation.get("validated") == 33
                if executed
                else projector_validation is None
            )
        ),
        "optional_r5b_random_projectors_coherent": (
            random_projectors.get("redrawn_count") == 0
            and random_projectors.get("draws_per_leg") == (32 if executed else 0)
            and (
                random_projectors.get("energy_match_max_abs_error", np.inf)
                <= 1e-7
                if executed
                else random_projectors.get("energy_match_max_abs_error") is None
            )
        ),
        "ensemble_membership_and_patch_size_exact": all(
            item["ensemble_membership_and_patch_size_exact"] is True
            for item in mutations.values()
        ),
        "necessity_and_sufficiency_health_complete": (
            health_validation["all_arms_ok"] is True
        ),
        "robust_authority_binding_suite_pass": diagnostics.get(
            "binding_suite_status"
        )
        == "PASS"
        and diagnostics.get("binding_decision_authority_exercised") is True,
        "observed_centered_robust_sensitivity_pass": diagnostics.get(
            "observed_sensitivity_status"
        )
        == "PASS",
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
    prespecified fixed sequence—not from this engineering simulation.  C3's three
    alternative singleton-head decompositions remain one Holm family.
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
    trials = ENGINEERING_DIAGNOSTIC_TRIALS
    return {
        "schema": "gfmcd.r5c.calibration_registry/1",
        "engineering_not_scientific_fwer_evidence": True,
        "formal_nominal_control": (
            "valid super-uniform arbitrary-boundary constituents; path IUT=max; "
            "fixed-sequence alpha=.05 for ordered scientific gates; Holm only across "
            "the three alternative C3 singleton-head decompositions"
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
        "diagnostic_draws": ENGINEERING_DIAGNOSTIC_DRAWS,
        "authorizes_target": False,
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
    target_effect: float = 0.20,
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
    if not np.isfinite(target_effect) or float(target_effect) <= SESOI:
        raise ValueError("synthetic target effect must strictly exceed the SESOI")

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
            initial = [float(target_effect) if target else 0.0, 0.0]
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

    def random_mean_members(control: str) -> tuple[str, ...]:
        if family == "C2" and control == "mean_32_matched_random_bases":
            return tuple(f"random_projector_{draw:02d}" for draw in range(32))
        if family == "C4" and control == _delivery_random_mean_id("C4"):
            return tuple(ensemble_name(item) for item in matched_random_pairs())
        if family == "C5" and control == _delivery_random_mean_id("C5"):
            return tuple(ensemble_name(item) for item in matched_random_triples())
        return ()

    # Material target primitives; all controls begin at equivalence.
    if family == "C1":
        for estimand in ("S_writer", "N_writer"):
            ensure(head_name(L3_READER), estimand, target=True)
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
            members = random_mean_members(control)
            controls = members or (control,)
            for member in controls:
                ensure(member, estimand)
            control_pseudo_mean = float(
                np.mean([raw_base[(member, estimand)][1] for member in controls])
            )
            if fields["endpoint"] == "partner":
                control_partner = raw_base[(target, estimand)][0] - boundary
            elif fields["endpoint"] == "specific":
                target_specific = (
                    raw_base[(target, estimand)][0]
                    - raw_base[(target, estimand)][1]
                )
                control_partner = (
                    target_specific
                    + control_pseudo_mean
                    - boundary
                )
            else:  # pragma: no cover - row constructor restricts the registry
                raise InstrumentError("unknown direct gap endpoint")
            for member in controls:
                set_base(member, estimand, 0, control_partner)

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
            target_values = ensure(target, estimand)
            members = random_mean_members(control)
            if members:
                control_values = [
                    np.mean(
                        np.stack([ensure(member, estimand)[endpoint] for member in members]),
                        axis=0,
                    )
                    for endpoint in (0, 1)
                ]
            else:
                control_values = ensure(control, estimand)
            if fields["endpoint"] == "partner":
                value = target_values[0] - control_values[0]
            elif fields["endpoint"] == "specific":
                value = (target_values[0] - target_values[1]) - (
                    control_values[0] - control_values[1]
                )
            else:  # pragma: no cover - row constructor restricts the registry
                raise InstrumentError("unknown direct gap endpoint")
        values.append(np.asarray(value, dtype=float))
    return np.vstack(values)


def _equal_gene_centered_rows(
    values: Any,
    groups: Sequence[str],
) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    labels = tuple(map(str, groups))
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise InstrumentError("observed-centered sensitivity rows are not aligned")
    ordered_genes = tuple(dict.fromkeys(labels))
    if len(ordered_genes) < MIN_GENE_CLUSTERS:
        raise InstrumentError("observed-centered sensitivity lacks gene clusters")
    gene_means = np.stack(
        [
            matrix[:, [index for index, label in enumerate(labels) if label == gene]].mean(
                axis=1
            )
            for gene in ordered_genes
        ],
        axis=1,
    )
    equal_gene_location = gene_means.mean(axis=1)
    return matrix - equal_gene_location[:, None]


def _robust_path_minimum_slack(
    family_result: Mapping[str, Any],
    requirements: Sequence[Requirement],
) -> float:
    slacks: list[float] = []
    for requirement in requirements:
        test = family_result["row_tests"][requirement.row]["test"]
        if requirement.decision in {"MATERIAL_POSITIVE", "SUPERIOR_TO_CONTROL"}:
            slacks.extend(
                float(test[estimator]["bound"]) - SESOI
                for estimator in ("raw", "trimmed")
            )
        else:
            for estimator in ("raw", "trimmed"):
                slacks.append(
                    float(test["lower_edge"][estimator]["bound"]) + SESOI
                )
                slacks.append(
                    SESOI - float(test["upper_edge"][estimator]["bound"])
                )
    return float(min(slacks))


def validate_observed_centered_robust_sensitivity(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        record.get("schema")
        != "gfmcd.r5c.observed_centered_robust_sensitivity/1"
        or record.get("status") != "PASS"
        or record.get("row_count") != 400
        or not _is_integer(record.get("gene_count"))
        or int(record["gene_count"]) < MIN_GENE_CLUSTERS
        or record.get("B") != PRODUCTION_BOOTSTRAP_DRAWS
        or record.get("trim_fraction") != ROBUST_TRIM_FRACTION
        or record.get("plant_effect") != SENSITIVITY_PLANT_EFFECT
        or record.get("orientations") != SENSITIVITY_ORIENTATIONS
        or record.get("orientation_successes") != SENSITIVITY_ORIENTATIONS
        or record.get("orientation_success_rate") != 1.0
        or record.get("minimum_boundary_slack", 0.0) <= 0.0
        or record.get("decision_authority")
        != "mean_and_10pct_trimmed_studentized_gene_cluster_bootstrap"
        or record.get("observed_target_outcomes_used") is not False
    ):
        raise InstrumentError("R5c observed-centered robust sensitivity differs")
    digests = record.get("orientation_inference_sha256")
    if (
        not isinstance(digests, list)
        or len(digests) != SENSITIVITY_ORIENTATIONS
        or any(re.fullmatch(r"[0-9a-f]{64}", str(value)) is None for value in digests)
    ):
        raise InstrumentError("R5c robust sensitivity inference digests differ")
    return {
        "status": "PASS",
        "record_sha256": canonical_sha256(record),
        "minimum_boundary_slack": float(record["minimum_boundary_slack"]),
    }


def run_observed_centered_robust_sensitivity(
    families: Mapping[str, Any],
    groups: Sequence[str],
    *,
    seed: int = PRODUCTION_SEED,
) -> dict[str, Any]:
    """Plant primary effects into preflight residuals at target size and B."""

    labels = tuple(map(str, groups))
    if len(labels) != 400:
        raise InstrumentError("robust sensitivity requires the target-shaped 400 rows")
    selected_families = ("C1", "C4")
    centered = {
        family: _equal_gene_centered_rows(families[family], labels)
        for family in selected_families
    }
    path_names = {
        "C1": "P_C1_DENSE_LOCAL_WRITE",
        "C4": "P_C4_PRIMARY_PAIR_DELIVERY",
    }
    orientation_records: list[dict[str, Any]] = []
    all_slacks: list[float] = []
    for orientation in range(SENSITIVITY_ORIENTATIONS):
        sign_by_gene = {
            gene: (
                1.0
                if int.from_bytes(
                    hashlib.sha256(
                        f"r5c-sensitivity|{orientation}|{gene}".encode()
                    ).digest()[:8],
                    "big",
                )
                % 2
                else -1.0
            )
            for gene in dict.fromkeys(labels)
        }
        signs = np.asarray([sign_by_gene[gene] for gene in labels], dtype=float)
        results: dict[str, Any] = {}
        slacks: dict[str, float] = {}
        for family in selected_families:
            planted = coherent_synthetic_family(
                family,
                n_examples=len(labels),
                target_effect=SENSITIVITY_PLANT_EFFECT,
            ) + centered[family] * signs[None, :]
            result = robust_family_inference(
                family,
                planted,
                labels,
                n_draws=PRODUCTION_BOOTSTRAP_DRAWS,
                seed=seed,
                namespace=f"r5c-observed-sensitivity:{orientation}:{family}",
            )
            path = path_names[family]
            if result["paths"][path]["classification"] != "SUPPORTED":
                raise InstrumentError(
                    f"R5c observed-centered sensitivity failed {family} orientation {orientation}"
                )
            slacks[family] = _robust_path_minimum_slack(
                result,
                path_registry()[family][path],
            )
            results[family] = canonical_sha256(result)
        minimum = min(slacks.values())
        if minimum <= 0.0:
            raise InstrumentError("R5c robust sensitivity has no strict resolution slack")
        all_slacks.append(minimum)
        orientation_records.append(
            {
                "orientation": orientation,
                "path_minimum_boundary_slack": slacks,
                "inference_sha256": canonical_sha256(results),
            }
        )
    record = {
        "schema": "gfmcd.r5c.observed_centered_robust_sensitivity/1",
        "status": "PASS",
        "row_count": len(labels),
        "gene_count": len(set(labels)),
        "B": PRODUCTION_BOOTSTRAP_DRAWS,
        "trim_fraction": ROBUST_TRIM_FRACTION,
        "plant_effect": SENSITIVITY_PLANT_EFFECT,
        "orientations": SENSITIVITY_ORIENTATIONS,
        "orientation_successes": SENSITIVITY_ORIENTATIONS,
        "orientation_success_rate": 1.0,
        "minimum_boundary_slack": min(all_slacks),
        "orientation_inference_sha256": [
            item["inference_sha256"] for item in orientation_records
        ],
        "orientation_records": orientation_records,
        "residual_source": "R1c_validation_preflight_rows_only",
        "observed_target_outcomes_used": False,
        "decision_authority": (
            "mean_and_10pct_trimmed_studentized_gene_cluster_bootstrap"
        ),
    }
    validate_observed_centered_robust_sensitivity(record)
    return record


def run_constituent_binding_suite(
    *, n_draws: int = PRODUCTION_BOOTSTRAP_DRAWS, seed: int = PRODUCTION_SEED
) -> dict[str, Any]:
    """Bind all algebraic faces and robust production inference archetypes."""

    if not _is_integer(n_draws) or int(n_draws) < 1:
        raise ValueError("robust binding draws must be positive")
    binding_genes = 64
    groups = [f"synthetic-gene-{index:03d}" for index in range(binding_genes)]
    registry = calibration_registry()
    cases = registry["coherent_face_registry"]
    face_hashes: list[str] = []
    archetypes: dict[tuple[str, str], dict[str, str]] = {}
    for case in cases:
        matrix = coherent_synthetic_family(
            case["family"],
            boundary_row=case["row"],
            boundary_edge=case["edge"],
            n_examples=binding_genes,
        )
        names = family_row_names()[case["family"]]
        row_index = names.index(case["row"])
        requirement = next(
            requirement
            for requirements in path_registry()[case["family"]].values()
            for requirement in requirements
            if requirement.row == case["row"]
        )
        expected_boundary = {
            edge: boundary
            for edge, boundary, _ in _requirement_edges(requirement)
        }[case["edge"]]
        if not np.allclose(
            matrix[row_index], expected_boundary, rtol=0.0, atol=1e-12
        ):
            raise InstrumentError("coherent primitive face did not regenerate its edge")
        face_hashes.append(array_sha256(matrix))
        archetypes.setdefault((requirement.decision, case["edge"]), case)

    # Run the actual raw+trimmed studentized-bootstrap path classifier at one
    # coherent representative of each structural boundary archetype.
    robust_archetypes: dict[str, Any] = {}
    for archetype, case in archetypes.items():
        matrix = coherent_synthetic_family(
            case["family"],
            boundary_row=case["row"],
            boundary_edge=case["edge"],
            n_examples=binding_genes,
        )
        result = robust_family_inference(
            case["family"],
            matrix,
            groups,
            n_draws=n_draws,
            seed=seed,
            namespace="r5c-robust-binding:" + ":".join(archetype),
        )
        requiring_paths = [
            path
            for path, requirements in path_registry()[case["family"]].items()
            if any(requirement.row == case["row"] for requirement in requirements)
        ]
        if any(result["paths"][path]["classification"] == "SUPPORTED" for path in requiring_paths):
            raise InstrumentError("a coherent boundary face was falsely supported")
        robust_archetypes["|".join(archetype)] = {
            "representative_row": case["row"],
            "representative_edge": case["edge"],
            "requiring_paths": requiring_paths,
            "inference_sha256": canonical_sha256(result),
        }

    plant_results: dict[str, Any] = {}
    for family in FAMILY_NAMES:
        result = robust_family_inference(
            family,
            coherent_synthetic_family(family, n_examples=binding_genes),
            groups,
            n_draws=n_draws,
            seed=seed,
            namespace=f"r5c-robust-binding:{family}:all-alternative",
        )
        if any(path["classification"] != "SUPPORTED" for path in result["paths"].values()):
            raise InstrumentError(f"{family} coherent all-alternative plant was not supported")
        plant_results[family] = canonical_sha256(result)
    return {
        "schema": "gfmcd.r5c.constituent_binding_suite/2",
        "robust_authority_classifier_reused": True,
        "inference_kind": "studentized_gene_bootstrap_asymptotic",
        "B": int(n_draws),
        "binding_gene_count": binding_genes,
        "literal_faces_checked": len(cases),
        "literal_face_matrices_sha256": canonical_sha256(face_hashes),
        "robust_boundary_archetypes": robust_archetypes,
        "robust_boundary_archetype_count": len(archetypes),
        "all_alternative_plant_inference_sha256": plant_results,
        "all_boundary_archetypes_rejected": True,
        "all_path_plants_supported": True,
        "decision_authority_exercised": True,
        "status": "PASS",
    }


def run_engineering_calibration(
    *,
    n_draws: int = ENGINEERING_DIAGNOSTIC_DRAWS,
    seed: int = PRODUCTION_SEED,
    n_trials: int | None = None,
) -> dict[str, Any]:
    """Run the prospective coherent-face Type-I/power implementation audit."""

    trials = ENGINEERING_DIAGNOSTIC_TRIALS if n_trials is None else int(n_trials)
    if trials < 1:
        raise ValueError("calibration trials must be positive")
    production = (
        trials == ENGINEERING_DIAGNOSTIC_TRIALS
        and n_draws == ENGINEERING_DIAGNOSTIC_DRAWS
    )
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
        "schema": "gfmcd.r5c.engineering_calibration/2",
        "frozen_diagnostic_configuration": production,
        "N_cal": trials,
        "n_draws_per_classifier": n_draws,
        "family_records": records,
        "all_family_gates_pass": all_pass,
        "engineering_not_scientific_fwer_evidence": True,
        "decision_authority": False,
        "authorizes_target": False,
        "status": "DIAGNOSTIC_PASS" if all_pass else "DIAGNOSTIC_FAIL",
    }


def validate_binding_and_calibration_records(
    binding: Mapping[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate every cheap, replayable invariant of the preflight diagnostics."""

    registry = calibration_registry()
    plants = binding.get("all_alternative_plant_inference_sha256")
    diagnostics = binding.get("robust_boundary_archetypes")
    if (
        not isinstance(plants, Mapping)
        or not isinstance(diagnostics, Mapping)
        or binding.get("schema") != "gfmcd.r5c.constituent_binding_suite/2"
        or binding.get("status") != "PASS"
        or binding.get("robust_authority_classifier_reused") is not True
        or binding.get("inference_kind")
        != "studentized_gene_bootstrap_asymptotic"
        or binding.get("B") != PRODUCTION_BOOTSTRAP_DRAWS
        or binding.get("binding_gene_count") != 64
        or binding.get("literal_faces_checked") != registry["literal_edge_count"]
        or binding.get("all_boundary_archetypes_rejected") is not True
        or binding.get("all_path_plants_supported") is not True
        or binding.get("decision_authority_exercised") is not True
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
        requirement = next(
            requirement
            for requirements in path_registry()[case["family"]].values()
            for requirement in requirements
            if requirement.row == case["row"]
        )
        key = "|".join((requirement.decision, case["edge"]))
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
        binding.get("robust_boundary_archetype_count") != len(expected_diagnostics)
        or set(diagnostics) != set(expected_diagnostics)
        or any(
            not isinstance(item, Mapping) for item in diagnostics.values()
        )
        or any(
            any(diagnostics[key].get(field) != value for field, value in expected.items())
            for key, expected in expected_diagnostics.items()
        )
    ):
        raise InstrumentError("R5c robust binding archetype registry differs")

    trials = ENGINEERING_DIAGNOSTIC_TRIALS
    family_records = calibration.get("family_records")
    if (
        calibration.get("schema") != "gfmcd.r5c.engineering_calibration/2"
        or calibration.get("status") not in {"DIAGNOSTIC_PASS", "DIAGNOSTIC_FAIL"}
        or calibration.get("frozen_diagnostic_configuration") is not True
        or calibration.get("N_cal") != trials
        or calibration.get("n_draws_per_classifier")
        != ENGINEERING_DIAGNOSTIC_DRAWS
        or calibration.get("engineering_not_scientific_fwer_evidence") is not True
        or calibration.get("decision_authority") is not False
        or calibration.get("authorizes_target") is not False
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
            or record.get("balanced_case_cycle_sha256") != expected_cycle_hash
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("false_positive_pattern_sha256"))
            )
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(record.get("plant_success_pattern_sha256"))
            )
        ):
            raise InstrumentError(f"R5c {family} calibration replay differs")
    all_pass = all(
        record.get("boundary_gate_ok") is True
        and record.get("plant_gate_ok") is True
        for record in family_records.values()
    )
    if (
        calibration.get("all_family_gates_pass") is not all_pass
        or calibration.get("status")
        != ("DIAGNOSTIC_PASS" if all_pass else "DIAGNOSTIC_FAIL")
    ):
        raise InstrumentError("R5c engineering diagnostic disposition differs")
    return {
        "status": "R5C_BINDING_AND_CALIBRATION_RECORDS_VALID",
        "binding_sha256": canonical_sha256(binding),
        "calibration_sha256": canonical_sha256(calibration),
        "engineering_diagnostic_authorizes_target": False,
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


def r1c_verified_target_common_leg_references(
    verified_common_legs: Mapping[str, Any],
    handoff: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Bind verified post-receipt R1c levels to the outcome-free handoff."""

    rows = verified_common_legs.get("rows")
    handoff_rows = handoff.get("rows")
    payload = {
        key: value
        for key, value in verified_common_legs.items()
        if key != "record_sha256"
    }
    if (
        verified_common_legs.get("schema")
        != R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA
        or verified_common_legs.get("record_sha256")
        != canonical_sha256(payload)
        or verified_common_legs.get("row_count") != R5C_EXACT_ROWS
        or verified_common_legs.get("model_outputs_present") is not True
        or verified_common_legs.get("may_be_used_for_cohort_selection") is not False
        or verified_common_legs.get("authorized_use")
        != "same-row R1c common-leg replay after the downstream target receipt only"
        or verified_common_legs.get(
            "caller_must_verify_downstream_target_receipt_before_load"
        )
        is not True
        or verified_common_legs.get("source_package_sha256")
        != handoff.get("source_package_sha256")
        or verified_common_legs.get("successor_attestation_sha256")
        != handoff.get("successor_attestation_sha256")
        or verified_common_legs.get("cohort_sha256")
        != handoff.get("cohort_sha256")
        or verified_common_legs.get("ordered_row_ids_sha256")
        != handoff.get("ordered_row_ids_sha256")
        or handoff.get("row_count") != R5C_EXACT_ROWS
        or not isinstance(rows, list)
        or len(rows) != R5C_EXACT_ROWS
        or not isinstance(handoff_rows, list)
        or len(handoff_rows) != R5C_EXACT_ROWS
    ):
        raise InstrumentError("verified R1c common-leg source contract differs")
    names = ("P_clean", "P_under_D", "P_under_N")
    references = {name: [] for name in names}
    for index, (row, identity) in enumerate(zip(rows, handoff_rows)):
        if not isinstance(row, Mapping) or not isinstance(identity, Mapping):
            raise InstrumentError("R1c common-leg row is malformed")
        if (
            set(row) != {"example_id", "gene_id", "margin_levels"}
            or row.get("example_id") != identity.get("example_id")
            or row.get("gene_id") != identity.get("gene_id")
        ):
            raise InstrumentError("R1c common-leg row order differs from the handoff")
        levels = row.get("margin_levels")
        if not isinstance(levels, Mapping) or set(levels) != set(names):
            raise InstrumentError(f"R1c common-leg row {index} lacks margin levels")
        for name in names:
            try:
                value = float(levels[name])
            except (KeyError, TypeError, ValueError):
                raise InstrumentError(
                    f"R1c common-leg row {index} has malformed {name}"
                ) from None
            if not np.isfinite(value):
                raise InstrumentError("R1c common-leg reference is non-finite")
            references[name].append(value)
    if canonical_sha256(rows) != verified_common_legs.get(
        "ordered_partner_margin_levels_sha256"
    ):
        raise InstrumentError("verified R1c common-leg digest differs")
    arrays = {
        name: np.asarray(values, dtype=float) for name, values in references.items()
    }
    return arrays


def r1c_preflight_common_leg_references(
    handoff: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Extract the authenticated common legs from the outcome-limited val handoff."""

    rows = handoff.get("rows")
    common = handoff.get("common_leg_rows")
    if (
        handoff.get("schema") != R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA
        or not isinstance(rows, list)
        or not isinstance(common, list)
        or len(rows) != R5C_EXACT_ROWS
        or len(common) != R5C_EXACT_ROWS
    ):
        raise InstrumentError("R1c preflight common-leg handoff differs")
    names = ("P_clean", "P_under_D", "P_under_N")
    output = {name: [] for name in names}
    for identity, record in zip(rows, common, strict=True):
        if (
            not isinstance(identity, Mapping)
            or not isinstance(record, Mapping)
            or record.get("example_id") != identity.get("example_id")
            or record.get("gene_id") != identity.get("gene_id")
            or set(record) != {"example_id", "gene_id", "margin_levels"}
            or not isinstance(record.get("margin_levels"), Mapping)
            or set(record["margin_levels"]) != set(names)
        ):
            raise InstrumentError("R1c preflight common-leg row order differs")
        for name in names:
            value = record["margin_levels"][name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
            ):
                raise InstrumentError("R1c preflight common-leg value is malformed")
            output[name].append(float(value))
    if canonical_sha256(common) != handoff.get(
        "ordered_partner_margin_levels_sha256"
    ):
        raise InstrumentError("R1c preflight common-leg digest differs")
    return {name: np.asarray(values, dtype=float) for name, values in output.items()}


def validate_r1c_common_leg_margin_arrays(
    references: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Require all same-row base forwards to reproduce R1c before interventions."""

    names = ("P_clean", "P_under_D", "P_under_N")
    if set(references) != set(names) or set(observed) != set(names):
        raise InstrumentError("R1c common-leg margin registry differs")
    diagnostics: dict[str, Any] = {}
    for name in names:
        expected = np.asarray(references[name], dtype=float)
        actual = np.asarray(observed[name], dtype=float)
        if (
            expected.ndim != 1
            or actual.shape != expected.shape
            or expected.size < 1
            or not np.isfinite(expected).all()
            or not np.isfinite(actual).all()
        ):
            raise InstrumentError("R1c same-row common-leg replay shape differs")
        absolute = np.abs(actual - expected)
        allowed = R1C_COMMON_LEG_ATOL + R1C_COMMON_LEG_RTOL * np.abs(expected)
        mismatches = np.flatnonzero(absolute > allowed)
        if mismatches.size:
            raise InstrumentError(
                "R1c same-row common-leg replay differs; no row dropping is allowed"
            )
        diagnostics[name] = {
            "rows": int(expected.size),
            "max_abs_error": float(absolute.max(initial=0.0)),
            "reference_sha256": array_sha256(expected),
            "observed_sha256": array_sha256(actual),
            "mismatch_count": 0,
        }
    return {
        "schema": "gfmcd.r5c.r1c_same_row_common_leg_replay/1",
        "atol": R1C_COMMON_LEG_ATOL,
        "rtol": R1C_COMMON_LEG_RTOL,
        "all_400_rows_required": True,
        "rows_dropped": 0,
        "diagnostics": diagnostics,
        "all_pass": True,
    }


def replay_r1c_common_legs_before_interventions(
    *,
    adapter: Any,
    rows: Sequence[Any],
    references: Mapping[str, Any],
    batch: int,
) -> dict[str, Any]:
    """Run only P clean/D/N base forwards, then fail before any hook intervention."""

    if len(rows) != 400:
        raise InstrumentError("R1c common-leg replay requires all 400 materialized rows")
    task = CouplingTask(adapter, batch=batch)
    collected = {name: [] for name in ("P_clean", "P_under_D", "P_under_N")}
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            local = rows[start : start + batch]
            readouts = [int(row.acceptor_g) for row in local]
            sequences = {
                "P_clean": [row.clean_sequence for row in local],
                "P_under_D": [row.paired_corrupt_sequence for row in local],
                "P_under_N": [row.pseudo_corrupt_sequence for row in local],
            }
            for name, values in sequences.items():
                encoded = task.encode(values, readouts)
                logits = adapter.model(**encoded).logits.float()
                collected[name].extend(task.margin(logits, readouts).cpu().tolist())
    observed = {name: np.asarray(values, dtype=float) for name, values in collected.items()}
    return validate_r1c_common_leg_margin_arrays(references, observed)


def run_model_measurements(
    *,
    adapter: Any,
    rows: Sequence[Any],
    runtime: FrozenR5bGlobalBasis | None,
    r5b_status: str,
    batch: int,
    n_draws: int,
    seed: int,
    binding_suite: Mapping[str, Any],
    engineering_calibration: Mapping[str, Any],
    frozen_preflight_sensitivity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute both source legs, exact families, health, and observed inference."""

    if len(rows) < MIN_GENES or batch < 1:
        raise InstrumentError(
            "R5c model execution needs enough gene-cluster rows and positive batch size"
        )
    task = CouplingTask(adapter, batch=batch)
    registry = complete_downstream_registry()
    basis_plan = optional_basis_execution_plan(
        status=r5b_status,
        runtime=runtime,
    )
    include_optional_basis = bool(basis_plan["execute_c2"])
    projector_capability = (
        runtime_projector_capability_audit(runtime, expected_rows=len(rows))
        if runtime is not None
        else None
    )
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
        selected_bases = None if runtime is None else runtime.row_bases(indices)
        random_bases = None if runtime is None else runtime.random_row_bases(indices)
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
    families = derive_family_matrices(
        partner_leg,
        pseudo_leg,
        include_optional_basis=include_optional_basis,
    )
    executed_families = tuple(families)
    groups = [str(row.example.gene_id) for row in rows]
    if len(set(groups)) < MIN_GENES:
        raise InstrumentError(
            "R5c inference lacks the minimum independent gene clusters"
        )
    if frozen_preflight_sensitivity is None:
        observed_sensitivity = run_observed_centered_robust_sensitivity(
            families,
            groups,
            seed=seed,
        )
    else:
        validate_observed_centered_robust_sensitivity(
            frozen_preflight_sensitivity
        )
        observed_sensitivity = dict(frozen_preflight_sensitivity)
    sign_flip_diagnostic = {
        family: strict_family_inference(
            family,
            families[family],
            groups,
            n_draws=n_draws,
            seed=seed,
            namespace=f"r5c-observed:{family}",
        )
        for family in executed_families
    }
    inference = {
        family: robust_family_inference(
            family,
            families[family],
            groups,
            n_draws=PRODUCTION_BOOTSTRAP_DRAWS,
            seed=seed,
            namespace=f"r5c-robust-observed:{family}",
        )
        for family in executed_families
    }
    clean_partner = partner_leg["native_conditions"]["clean_margin"]
    clean_pseudo = pseudo_leg["native_conditions"]["clean_margin"]
    if not np.array_equal(clean_partner, clean_pseudo):
        raise InstrumentError("partner and pseudo legs do not share the native clean margin")
    native = native_behavior_inference(
        clean_partner - partner_leg["native_conditions"]["m_corrupt_margin"],
        clean_partner - pseudo_leg["native_conditions"]["m_corrupt_margin"],
        groups,
        n_draws=n_draws,
        seed=seed,
        namespace="r5c-native-entry-behavior",
    )
    health = run_bounded_hook_health(adapter, task, rows, runtime=runtime)
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
        for item in leg["random_local_writer_subspaces"]
    ) and all(
        len(leg["random_local_writer_subspaces"])
        == (32 if include_optional_basis else 0)
        for leg in (partner_leg, pseudo_leg)
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
    random_energy_errors = [
        float(item["per_row_energy_match_max_abs_error"])
        for leg in (partner_leg, pseudo_leg)
        for item in leg["random_local_writer_subspaces"]
    ]
    random_projector_evidence = {
        "energy_match_max_abs_error": (
            max(random_energy_errors) if random_energy_errors else None
        ),
        "redrawn_count": sum(
            item.get("redrawn") is not False
            for leg in (partner_leg, pseudo_leg)
            for item in leg["random_local_writer_subspaces"]
        ),
        "draws_per_leg": 32 if include_optional_basis else 0,
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
        "optional_r5b_basis": {
            "disposition_status": r5b_status,
            "executed": include_optional_basis,
            "execution_plan": basis_plan,
        },
        "global_basis_projectors": projector_capability,
        "random_projectors": random_projector_evidence,
        "health": health,
        "diagnostics": {
            "binding_suite_status": binding_suite.get("status"),
            "binding_decision_authority_exercised": binding_suite.get(
                "decision_authority_exercised"
            ),
            "engineering_calibration_status": engineering_calibration.get("status"),
            "engineering_calibration_authorizes_target": engineering_calibration.get(
                "authorizes_target"
            ),
            "observed_sensitivity_status": observed_sensitivity.get("status"),
            "observed_sensitivity_sha256": canonical_sha256(observed_sensitivity),
        },
    }
    projector_validation = (
        validate_projector_capability_audit(
            projector_capability,
            expected_runtime_array_sha256=runtime.authenticated_runtime_array_sha256,
        )
        if runtime is not None and projector_capability is not None
        else None
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
        "local_writer_query_only_isolation": all(
            item["query_only"] for item in mutation_audit.values()
        ),
        "common_legs_bit_identical": all(
            item["components"] == len(registry)
            for item in common_leg_capture.values()
        ),
        "optional_r5b_basis_execution_coherent": (
            projector_validation is not None
            and projector_validation.get("validated") == 33
            if include_optional_basis
            else projector_validation is None
        ),
        "optional_r5b_random_projectors_coherent": energy_ok,
        "ensemble_membership_and_patch_size_exact": all(
            item["ensemble_membership_and_patch_size_exact"]
            for item in mutation_audit.values()
        ),
        "necessity_and_sufficiency_health_complete": (
            validate_production_health_record(
                health,
                include_optional_basis=include_optional_basis,
            )["all_arms_ok"]
            is True
        ),
        "robust_authority_binding_suite_pass": binding_suite.get("status") == "PASS"
        and binding_suite.get("decision_authority_exercised") is True,
        "observed_centered_robust_sensitivity_pass": observed_sensitivity.get(
            "status"
        )
        == "PASS",
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
            expected_runtime_array_sha256=(
                None
                if runtime is None
                else runtime.authenticated_runtime_array_sha256
            ),
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
        for family in executed_families
    }
    if not include_optional_basis:
        path_classes["C2"] = {
            "P_C2_LOW_RANK_STRENGTHENING": basis_plan["classification"]
        }
    controls_active = structured_controls_active(inference)
    verdict = scientific_verdict(
        instrument_ok=instrument_ok,
        entry_behavior="PASS",
        dense_write=path_classes["C1"]["P_C1_DENSE_LOCAL_WRITE"],
        low_rank=path_classes["C2"]["P_C2_LOW_RANK_STRENGTHENING"],
        c3_paths=path_classes["C3"],
        primary_delivery=path_classes["C4"]["P_C4_PRIMARY_PAIR_DELIVERY"],
        distinct_tested_set=path_classes["C4"][
            "P_C4_DISTINCT_WITHIN_TESTED_SET"
        ],
        secondary_triple=path_classes["C5"]["P_C5_SECONDARY_TRIPLE"],
        controls_active=controls_active,
    )
    return {
        "groups": groups,
        "ordered_gene_ids_sha256": canonical_sha256(groups),
        "native_behavior": native | {"decision_role": "diagnostic_only"},
        "partner_leg": _json_ready(partner_leg),
        "pseudo_leg": _json_ready(pseudo_leg),
        "families": {
            family: (
                {
                    "row_names": list(family_row_names()[family]),
                    "raw_values": families[family].tolist(),
                    "raw_values_sha256": array_sha256(families[family]),
                    "inference": inference[family],
                    "inference_sha256": canonical_sha256(inference[family]),
                    "sign_flip_diagnostic": sign_flip_diagnostic[family]
                    | {"decision_authority": False},
                    "sign_flip_diagnostic_sha256": canonical_sha256(
                        sign_flip_diagnostic[family]
                        | {"decision_authority": False}
                    ),
                    "executed": True,
                }
                if family in executed_families
                else {
                    "executed": False,
                    "disposition": basis_plan["classification"],
                    "dense_primary_unaffected": True,
                }
            )
            for family in FAMILY_NAMES
        },
        "gene_sufficient_statistics": {
            "one_row_per_gene": len(set(groups)) == len(groups),
            "repeated_genes_allowed": True,
            "n_rows": len(groups),
            "n_genes": len(set(groups)),
            "raw_gene_values_stored_in_families": True,
            "family": {
                family: {
                    "row_sum": np.sum(families[family], axis=1).tolist(),
                    "row_sum_squares": np.sum(
                        np.square(families[family]), axis=1
                    ).tolist(),
                    "raw_values_sha256": array_sha256(families[family]),
                }
                for family in executed_families
            },
        },
        "observed_centered_robust_sensitivity": observed_sensitivity,
        "capability": capability_record,
        "capability_status": capability_status,
        "instrument_ok": instrument_ok,
        "path_classifications": path_classes,
        "optional_r5b_basis": {
            "disposition_status": r5b_status,
            "execution_plan": basis_plan,
        },
        "controls_active": controls_active,
        "individual_random_controls_role": (
            "authenticated_primitives_and_descriptive_geometry_only"
        ),
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
        "status": "FROZEN_MODEL_FREE_REGISTRY_AUDIT_PASS",
        "model_loaded": False,
        "annotation_read": False,
        "fasta_read": False,
        "cohort_read": False,
        "chr8_chr9_opened": False,
        "readout": "single masked acceptor G hard margin",
        "masked_donor_endpoint_constructed": False,
        "scientific_gate_order": list(fixed_scientific_gate_order()),
        "scientific_entry": {
            "r1c": "authenticated sealed chr8/chr9 acceptor behavior",
            "r5a": "same-row supported donor-local reader target /2",
            "r5b_role": (
                "terminal pre-unseal optional-basis disposition; C2 only and never a "
                "dense-primary veto"
            ),
        },
        "preflight_cohort_role": (
            "exact_authenticated_R1c_validation_preflight_400_rows_chr11_chr16_chr19"
        ),
        "target_cohort_role": "authenticated_R1c_400_row_same_row_capstone",
        "cohort_contracts": {
            "preflight": {
                "schema": R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
                "row_count": R5C_EXACT_ROWS,
                "chromosomes": list(R5_VALIDATION_CHROMOSOMES),
                "source": "authenticated_R1c_validation_preflight_handoff",
                "rows_filtered_or_reordered": False,
            },
            "target": {
                "schema": R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
                "row_count": R5C_EXACT_ROWS,
                "chromosomes": ["chr8", "chr9"],
                "source": "authenticated_R1c_sealed_target_handoff_after_receipt",
                "rows_filtered_or_reordered": False,
            },
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
    pin_names = tuple(
        dict.fromkeys(stage_pin_names("preflight") + stage_pin_names("target"))
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
        "schema": "gfmcd.r5c.instrument_binding/2",
        STACK_BINDING_FIELD: stack,
        STACK_FINGERPRINT_FIELD: r5_stack_fingerprint(stack),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "normalized_runner_sha256": normalized_runner_sha256(),
        "stat_core_sha256": hashlib.sha256(STAT_CORE.read_bytes()).hexdigest(),
        "mechanism_core_sha256": hashlib.sha256(MECHANISM_CORE.read_bytes()).hexdigest(),
        "cohort_core_sha256": normalized_literal_pin_source_sha256(
            COHORT_CORE,
            permitted_pin_names=SHARED_R1C_PACKAGE_PIN_NAMES,
        ),
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
        "production_bootstrap_draws": PRODUCTION_BOOTSTRAP_DRAWS,
        "robust_trim_fraction": ROBUST_TRIM_FRACTION,
        "decision_inference": (
            "raw_and_10pct_trimmed_unweighted_gene_means; shared whole-gene "
            "studentized bootstrap; strict SESOI/TOST; asymptotic"
        ),
        "decision_authority": (
            "mean_and_10pct_trimmed_studentized_gene_cluster_bootstrap"
        ),
        "sign_flip_role": "diagnostic_only",
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


def _authorize_scored_stage(stage: str) -> dict[str, str]:
    pins = {name: globals()[name] for name in stage_pin_names(stage)}
    missing = [name for name, value in pins.items() if value is None]
    if missing:
        raise TargetSealError(
            f"R5c {stage} is hard closed: literal seals remain None ("
            + ", ".join(missing)
            + ")"
        )
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in pins.values()):
        raise TargetSealError(f"R5c {stage} seal has a malformed SHA-256")
    shared_r1c = {
        "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256": (
            SHARED_R1C_PREFLIGHT_ARTIFACT_SHA256
        ),
        "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256": (
            SHARED_R1C_PREFLIGHT_MANIFEST_SHA256
        ),
    }
    if any(pins[name] != value for name, value in shared_r1c.items()):
        raise TargetSealError(
            "R5c R1c preflight pins differ from the canonical shared R1c preflight pins"
        )
    if stage == "target":
        shared_r1c_target = {
            "FROZEN_R1C_TARGET_ARTIFACT_SHA256": SHARED_R1C_TARGET_ARTIFACT_SHA256,
            "FROZEN_R1C_TARGET_MANIFEST_SHA256": SHARED_R1C_TARGET_MANIFEST_SHA256,
            "FROZEN_R1C_TARGET_ATTEMPT_SHA256": SHARED_R1C_TARGET_ATTEMPT_SHA256,
            "FROZEN_R1C_TARGET_COMPLETION_SHA256": SHARED_R1C_TARGET_COMPLETION_SHA256,
        }
        if any(pins[name] != value for name, value in shared_r1c_target.items()):
            raise TargetSealError(
                "R5c R1c pins differ from the canonical shared capstone pins"
            )
    payload = instrument_binding_payload()
    observed_protocol = payload["protocol_sha256"]
    observed_fingerprint = canonical_sha256(payload)
    if observed_protocol != FROZEN_PROTOCOL_SHA256:
        raise TargetSealError("R5c protocol hash differs from the literal freeze")
    if observed_fingerprint != FROZEN_INSTRUMENT_FINGERPRINT_SHA256:
        raise TargetSealError("R5c instrument fingerprint differs from the literal freeze")
    return {name: str(value) for name, value in pins.items()}


def authorize_target() -> dict[str, str]:
    return _authorize_scored_stage("target")


def authorize_preflight() -> dict[str, str]:
    """Authorize exact R1c validation-row preflight without future target pins."""

    return _authorize_scored_stage("preflight")


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


def validate_exact_measurement_replay(
    record: Mapping[str, Any], *, fingerprint: str, expected_stage: str
) -> dict[str, Any]:
    """Offline-rederive capability, matrices, and inference from stored primitives."""

    if expected_stage not in {"preflight", "target"}:
        raise ValueError("R5c replay stage must be preflight or target")
    current_stack = build_r5_stack_binding(repo=REPO)
    try:
        validate_serialized_r5_stack(record, current=current_stack)
    except R5StackBindingError as exc:
        raise InstrumentError(
            f"R5c {expected_stage} joint stack differs: {exc}"
        ) from exc
    if (
        record.get("schema") != SCHEMA
        or record.get("stage") != expected_stage
        or record.get("instrument_fingerprint_sha256") != fingerprint
        or record.get("sealed_chr8_chr9_opened") is not (expected_stage == "target")
        or record.get("cohort", {}).get("role")
        != (
            "AUTHENTICATED_R1C_NONSEALED_VALIDATION_400_ROW_PREFLIGHT"
            if expected_stage == "preflight"
            else "AUTHENTICATED_R1C_400_ROW_SAME_ROW_CAPSTONE"
        )
        or record.get("cohort", {}).get("row_count") != R5C_EXACT_ROWS
        or record.get("cohort", {}).get("rows_filtered_or_reordered") is not False
    ):
        raise InstrumentError(f"R5c {expected_stage} identity differs")
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
        raise InstrumentError(f"R5c {expected_stage} instrument/registry binding differs")
    measurement = record.get("measurement")
    if not isinstance(measurement, Mapping):
        raise InstrumentError(f"R5c {expected_stage} lacks its measurement record")
    capability_status = measurement.get("capability_status")
    if (
        measurement.get("instrument_ok") is not True
        or not isinstance(capability_status, Mapping)
        or capability_status.get("status") != "R5C_CAPABILITY_CONFIRMED"
    ):
        raise InstrumentError(f"R5c {expected_stage} capability status is inconsistent")
    binding = record.get("binding_suite")
    calibration = record.get("engineering_calibration")
    if not isinstance(binding, Mapping) or not isinstance(calibration, Mapping):
        raise InstrumentError(f"R5c {expected_stage} lacks diagnostic records")
    diagnostic_validation = validate_binding_and_calibration_records(
        binding,
        calibration,
    )
    groups = measurement.get("groups")
    if (
        not isinstance(groups, list)
        or len(groups) != R5C_EXACT_ROWS
        or len(set(map(str, groups))) < MIN_GENE_CLUSTERS
        or canonical_sha256(groups) != measurement.get("ordered_gene_ids_sha256")
    ):
        raise InstrumentError(f"R5c {expected_stage} lacks exact inference groups")
    partner_leg = measurement.get("partner_leg")
    pseudo_leg = measurement.get("pseudo_leg")
    if not isinstance(partner_leg, Mapping) or not isinstance(pseudo_leg, Mapping):
        raise InstrumentError(f"R5c {expected_stage} lacks primitive partner/pseudo legs")
    optional_basis = measurement.get("optional_r5b_basis")
    top_basis = record.get("r5b_optional_basis")
    if not isinstance(optional_basis, Mapping) or not isinstance(top_basis, Mapping):
        raise InstrumentError(f"R5c {expected_stage} lacks its R5b basis disposition")
    r5b_status = str(optional_basis.get("disposition_status"))
    stored_basis_plan = optional_basis.get("execution_plan")
    if not isinstance(stored_basis_plan, Mapping):
        raise InstrumentError(f"R5c {expected_stage} R5b execution plan is malformed")
    basis_plan = optional_basis_execution_plan(
        status=r5b_status,
        runtime=(object() if stored_basis_plan.get("execute_c2") else None),
    )
    if (
        optional_basis.get("execution_plan") != basis_plan
        or top_basis.get("disposition_status") != r5b_status
        or top_basis.get("execution_plan") != basis_plan
    ):
        raise InstrumentError(f"R5c {expected_stage} R5b basis disposition differs")
    expected_runtime_hashes = top_basis.get("runtime_array_sha256")
    if basis_plan["execute_c2"]:
        if not isinstance(expected_runtime_hashes, Mapping):
            raise InstrumentError("executed R5b basis lacks authenticated runtime hashes")
        contract = top_basis.get("contract")
        if not isinstance(contract, Mapping):
            raise InstrumentError("executed R5b basis lacks its runtime contract")
        validate_frozen_r5b_basis_contract(
            contract,
            expected_stack_fingerprint=str(record.get(STACK_FINGERPRINT_FIELD)),
        )
        runtime_records = contract.get("runtime_arrays")
        if (
            not isinstance(runtime_records, Mapping)
            or expected_runtime_hashes
            != {
                name: value.get("sha256")
                for name, value in runtime_records.items()
                if isinstance(value, Mapping)
            }
            or top_basis.get("runtime_bundle_sha256")
            != contract.get("bundle", {}).get("sha256")
        ):
            raise InstrumentError("executed R5b runtime identity differs")
    elif expected_runtime_hashes is not None:
        raise InstrumentError("nonexecuted R5b basis has runtime hashes")
    elif any(
        top_basis.get(field) is not None
        for field in ("contract", "runtime_bundle_sha256")
    ):
        raise InstrumentError("nonexecuted R5b basis carries runtime material")
    validate_capability(
        measurement.get("capability", {}),
        partner_leg=partner_leg,
        pseudo_leg=pseudo_leg,
        expected_runtime_array_sha256=expected_runtime_hashes,
    )
    try:
        rederived_families = derive_family_matrices(
            partner_leg,
            pseudo_leg,
            include_optional_basis=bool(basis_plan["execute_c2"]),
        )
    except (KeyError, TypeError, ValueError, InstrumentError) as exc:
        raise InstrumentError(f"R5c {expected_stage} primitive replay failed: {exc}") from exc
    family_records = measurement.get("families")
    if (
        not isinstance(family_records, Mapping)
        or tuple(family_records) != FAMILY_NAMES
    ):
        raise InstrumentError(f"R5c {expected_stage} lacks its family records")
    sufficient = measurement.get("gene_sufficient_statistics")
    if (
        not isinstance(sufficient, Mapping)
        or sufficient.get("repeated_genes_allowed") is not True
        or sufficient.get("n_rows") != R5C_EXACT_ROWS
        or sufficient.get("n_genes") != len(set(map(str, groups)))
        or sufficient.get("raw_gene_values_stored_in_families") is not True
        or not isinstance(sufficient.get("family"), Mapping)
    ):
        raise InstrumentError(f"R5c {expected_stage} lacks exact gene sufficient statistics")
    replay_hashes: dict[str, str] = {}
    diagnostic_hashes: dict[str, str] = {}
    replayed_inference: dict[str, Mapping[str, Any]] = {}
    for family in FAMILY_NAMES:
        family_record = family_records.get(family)
        if not isinstance(family_record, Mapping):
            raise InstrumentError(f"R5c {expected_stage} lacks {family}")
        if family not in rederived_families:
            expected_skip = {
                "executed": False,
                "disposition": basis_plan["classification"],
                "dense_primary_unaffected": True,
            }
            if family != "C2" or dict(family_record) != expected_skip:
                raise InstrumentError(f"R5c {expected_stage} optional C2 disposition differs")
            continue
        values = np.asarray(family_record.get("raw_values"), dtype=float)
        if values.shape != (
            EXPECTED_FAMILY_SIZES[family],
            R5C_EXACT_ROWS,
        ) or array_sha256(values) != family_record.get("raw_values_sha256"):
            raise InstrumentError(f"R5c {expected_stage} {family} raw values differ")
        if (
            family_record.get("row_names") != list(family_row_names()[family])
            or family_record.get("executed") is not True
            or not np.array_equal(values, rederived_families[family])
        ):
            raise InstrumentError(
                f"R5c {expected_stage} {family} values do not rederive from primitive legs"
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
                f"R5c {expected_stage} {family} gene sufficient statistics differ"
            )
        replay = robust_family_inference(
            family,
            rederived_families[family],
            groups,
            n_draws=PRODUCTION_BOOTSTRAP_DRAWS,
            seed=PRODUCTION_SEED,
            namespace=f"r5c-robust-observed:{family}",
        )
        digest = canonical_sha256(replay)
        if (
            digest != family_record.get("inference_sha256")
            or canonical_sha256(family_record.get("inference")) != digest
        ):
            raise InstrumentError(f"R5c {expected_stage} {family} inference replay differs")
        replay_hashes[family] = digest
        replayed_inference[family] = replay
        diagnostic = strict_family_inference(
            family,
            rederived_families[family],
            groups,
            n_draws=PRODUCTION_RANDOMISATION,
            seed=PRODUCTION_SEED,
            namespace=f"r5c-observed:{family}",
        )
        diagnostic = diagnostic | {"decision_authority": False}
        diagnostic_digest = canonical_sha256(diagnostic)
        if (
            diagnostic_digest
            != family_record.get("sign_flip_diagnostic_sha256")
            or canonical_sha256(family_record.get("sign_flip_diagnostic"))
            != diagnostic_digest
        ):
            raise InstrumentError(
                f"R5c {expected_stage} {family} sign-flip diagnostic replay differs"
            )
        diagnostic_hashes[family] = diagnostic_digest
    try:
        clean_partner = np.asarray(
            partner_leg["native_conditions"]["clean_margin"], dtype=float
        )
        clean_pseudo = np.asarray(
            pseudo_leg["native_conditions"]["clean_margin"], dtype=float
        )
        if not np.array_equal(clean_partner, clean_pseudo):
            raise InstrumentError(f"R5c {expected_stage} native clean legs differ")
        native_replay = native_behavior_inference(
            clean_partner
            - np.asarray(
                    partner_leg["native_conditions"]["m_corrupt_margin"], dtype=float
            ),
            clean_partner
            - np.asarray(
                    pseudo_leg["native_conditions"]["m_corrupt_margin"], dtype=float
            ),
            groups,
            n_draws=PRODUCTION_RANDOMISATION,
            seed=PRODUCTION_SEED,
            namespace="r5c-native-entry-behavior",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise InstrumentError(f"R5c {expected_stage} native replay failed: {exc}") from exc
    native_stored = measurement.get("native_behavior")
    if isinstance(native_stored, Mapping):
        native_stored = {
            key: value
            for key, value in native_stored.items()
            if key != "decision_role"
        }
    if canonical_sha256(native_replay) != canonical_sha256(native_stored):
        raise InstrumentError(f"R5c {expected_stage} native inference replay differs")
    official_replay: dict[str, Any] = {}
    if expected_stage == "target":
        path_classes = {
            family: {
                path: result["classification"]
                for path, result in replayed_inference[family]["paths"].items()
            }
            for family in replayed_inference
        }
        if not basis_plan["execute_c2"]:
            path_classes["C2"] = {
                "P_C2_LOW_RANK_STRENGTHENING": basis_plan["classification"]
            }
        path_classes = {family: path_classes[family] for family in FAMILY_NAMES}
        controls_active = structured_controls_active(replayed_inference)
        verdict = scientific_verdict(
            instrument_ok=True,
            entry_behavior="PASS",
            dense_write=path_classes["C1"]["P_C1_DENSE_LOCAL_WRITE"],
            low_rank=path_classes["C2"]["P_C2_LOW_RANK_STRENGTHENING"],
            c3_paths=path_classes["C3"],
            primary_delivery=path_classes["C4"]["P_C4_PRIMARY_PAIR_DELIVERY"],
            distinct_tested_set=path_classes["C4"][
                "P_C4_DISTINCT_WITHIN_TESTED_SET"
            ],
            secondary_triple=path_classes["C5"]["P_C5_SECONDARY_TRIPLE"],
            controls_active=controls_active,
        )
        gate_dispositions = ordered_gate_dispositions(
            instrument_ok=True,
            dense_write=path_classes["C1"]["P_C1_DENSE_LOCAL_WRITE"],
            primary_delivery=path_classes["C4"]["P_C4_PRIMARY_PAIR_DELIVERY"],
            low_rank=path_classes["C2"]["P_C2_LOW_RANK_STRENGTHENING"],
            distinct_tested_set=path_classes["C4"][
                "P_C4_DISTINCT_WITHIN_TESTED_SET"
            ],
            secondary_triple=path_classes["C5"]["P_C5_SECONDARY_TRIPLE"],
        )
        claim_eligible = scientific_claim_eligible(
            status=verdict,
            instrument_ok=True,
        )
        if (
            measurement.get("path_classifications") != path_classes
            or measurement.get("controls_active") is not controls_active
            or measurement.get("verdict") != verdict
            or record.get("status") != verdict
            or record.get("fixed_scientific_gate_order")
            != list(fixed_scientific_gate_order())
            or record.get("fixed_scientific_gate_dispositions")
            != gate_dispositions
            or record.get("scientific_outcomes_claim_eligible")
            is not claim_eligible
        ):
            raise InstrumentError(
                "R5c target official scientific disposition does not replay"
            )
        official_replay = {
            "path_classifications": path_classes,
            "controls_active": controls_active,
            "official_disposition": verdict,
            "fixed_scientific_gate_dispositions": gate_dispositions,
            "scientific_outcomes_claim_eligible": claim_eligible,
        }
    return {
        "status": "EXACT_OFFLINE_MEASUREMENT_REPLAY_PASS",
        "stage": expected_stage,
        "family_inference_sha256": replay_hashes,
        "family_sign_flip_diagnostic_sha256": diagnostic_hashes,
        "native_inference_sha256": canonical_sha256(native_replay),
        "primitive_family_rederivation_pass": True,
        "binding_suite_sha256": canonical_sha256(binding),
        "engineering_calibration_sha256": canonical_sha256(calibration),
        "diagnostic_record_validation": diagnostic_validation,
        **official_replay,
    }


def validate_preflight_replay(
    record: Mapping[str, Any], *, fingerprint: str
) -> dict[str, Any]:
    replay = validate_exact_measurement_replay(
        record,
        fingerprint=fingerprint,
        expected_stage="preflight",
    )
    return replay | {
        "status": "FROZEN_PREFLIGHT_REPLAY_PASS",
        "artifact_sha256": str(FROZEN_R5C_PREFLIGHT_ARTIFACT_SHA256),
    }


def validate_target_replay(
    record: Mapping[str, Any], *, fingerprint: str
) -> dict[str, Any]:
    """Exact offline replay for a published target result, without model/data IO."""

    common = record.get("r1c_same_row_common_leg_replay")
    diagnostics = common.get("diagnostics") if isinstance(common, Mapping) else None
    if (
        not isinstance(diagnostics, Mapping)
        or set(diagnostics) != {"P_clean", "P_under_D", "P_under_N"}
        or common.get("all_400_rows_required") is not True
        or common.get("rows_dropped") != 0
        or common.get("all_pass") is not True
        or any(
            item.get("rows") != R5C_EXACT_ROWS or item.get("mismatch_count") != 0
            for item in diagnostics.values()
            if isinstance(item, Mapping)
        )
        or any(not isinstance(item, Mapping) for item in diagnostics.values())
    ):
        raise InstrumentError("R5c target lacks its exact R1c common-leg replay")
    upstream = record.get("upstream_chain")
    if (
        not isinstance(upstream, Mapping)
        or upstream.get("chain_sha256")
        != canonical_sha256(
            {key: value for key, value in upstream.items() if key != "chain_sha256"}
        )
    ):
        raise InstrumentError("R5c target upstream chain does not replay")
    return validate_exact_measurement_replay(
        record,
        fingerprint=fingerprint,
        expected_stage="target",
    ) | {"r1c_common_legs_replayed": True, "upstream_chain_replayed": True}


def _published_file_identity(path: Path) -> dict[str, Any]:
    return {
        "logical_name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_r5c_target_package(
    artifact_path: str | Path,
    manifest_path: str | Path,
    attempt_path: str | Path,
    completion_path: str | Path,
    *,
    fingerprint: str,
) -> dict[str, Any]:
    """Authenticate and scientifically replay one published R5c target package."""

    try:
        paths = {
            "artifact": Path(artifact_path).expanduser().resolve(strict=True),
            "manifest": Path(manifest_path).expanduser().resolve(strict=True),
            "attempt": Path(attempt_path).expanduser().resolve(strict=True),
            "completion": Path(completion_path).expanduser().resolve(strict=True),
        }
        identities = {
            name: _published_file_identity(path) for name, path in paths.items()
        }
        loaded = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name, path in paths.items()
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise InstrumentError(f"R5c target package cannot be read exactly: {exc}") from exc
    if any(not isinstance(value, dict) for value in loaded.values()):
        raise InstrumentError("R5c target package members must be JSON objects")
    artifact = loaded["artifact"]
    manifest = loaded["manifest"]
    attempt = loaded["attempt"]
    completion = loaded["completion"]

    outputs = manifest.get("outputs")
    matching_outputs = (
        [
            item
            for item in outputs
            if isinstance(item, Mapping)
            and Path(str(item.get("path", ""))).name
            == identities["artifact"]["logical_name"]
            and item.get("size_bytes") == identities["artifact"]["size_bytes"]
            and item.get("sha256") == identities["artifact"]["sha256"]
        ]
        if isinstance(outputs, list)
        else []
    )
    inputs = manifest.get("inputs")
    matching_attempts = (
        [
            item
            for item in inputs
            if isinstance(item, Mapping)
            and Path(str(item.get("path", ""))).name
            == identities["attempt"]["logical_name"]
            and item.get("size_bytes") == identities["attempt"]["size_bytes"]
            and item.get("sha256") == identities["attempt"]["sha256"]
        ]
        if isinstance(inputs, list)
        else []
    )
    if (
        manifest.get("experiment_id")
        != "splicebert_acceptor_coupling:r5c_acceptor_write_delivery"
        or manifest.get("git_dirty") is not False
        or len(matching_outputs) != 1
        or len(matching_attempts) != 1
    ):
        raise InstrumentError(
            "R5c target manifest does not bind the exact artifact and attempt"
        )

    expected_attempt_fields = {
        "schema",
        "created_at_utc",
        "biological_target_key",
        "identity",
        "attempt_context",
        "policy",
    }
    identity = attempt.get("identity")
    context = attempt.get("attempt_context")
    if (
        set(attempt) != expected_attempt_fields
        or attempt.get("schema") != RECEIPT_SCHEMA
        or not isinstance(attempt.get("created_at_utc"), str)
        or not attempt.get("created_at_utc")
        or not isinstance(identity, Mapping)
        or not isinstance(context, Mapping)
    ):
        raise InstrumentError("R5c target attempt receipt does not replay exactly")
    try:
        expected_identity = build_r5_biological_target_identity(
            stage="R5c",
            checkpoint_sha256=str(identity.get("checkpoint_sha256")),
            target_gene_ids_sha256=str(identity.get("target_gene_ids_sha256")),
            ordered_rows_sha256=str(identity.get("ordered_rows_sha256")),
            construction_sha256=str(identity.get("construction_sha256")),
            endpoint_policy="masked_acceptor_G_margin_only",
        )
    except ValueError as exc:
        raise InstrumentError("R5c target biological identity is malformed") from exc
    embedded_attempt = artifact.get("target_attempt_receipt")
    cohort = artifact.get("cohort")
    checkpoint = artifact.get("checkpoint")
    preflight = artifact.get("preflight_package")
    upstream = artifact.get("upstream_chain")
    if (
        dict(identity) != expected_identity
        or attempt.get("biological_target_key")
        != identity.get("biological_target_key")
        or attempt.get("policy")
        != (
            "create-only before target cohort/path/model/output inspection; failures "
            "remain spent; executable repackaging cannot change the ledger key"
        )
        or not isinstance(embedded_attempt, Mapping)
        or {
            key: value
            for key, value in embedded_attempt.items()
            if key != "receipt_path"
        }
        != attempt
        or Path(str(embedded_attempt.get("receipt_path", ""))).name
        != identities["attempt"]["logical_name"]
        or not isinstance(cohort, Mapping)
        or not isinstance(checkpoint, Mapping)
        or not isinstance(preflight, Mapping)
        or not isinstance(upstream, Mapping)
        or identity.get("checkpoint_sha256")
        != checkpoint.get("checkpoint_aggregate_sha256")
        or identity.get("target_gene_ids_sha256")
        != cohort.get("sorted_unique_gene_ids_sha256")
        or identity.get("ordered_rows_sha256") != cohort.get("cohort_sha256")
        or identity.get("construction_sha256")
        != cohort.get("capstone_construction_sha256")
        or context
        != {
            "r5_stack_fingerprint_sha256": artifact.get(
                STACK_FINGERPRINT_FIELD
            ),
            "stage_preflight_artifact_sha256": preflight.get(
                "artifact_sha256"
            ),
            "stage_preflight_manifest_sha256": preflight.get(
                "manifest_sha256"
            ),
            "upstream_package_chain_sha256": upstream.get("chain_sha256"),
        }
    ):
        raise InstrumentError("R5c target attempt and artifact identities differ")

    expected_completion_fields = {
        "schema",
        "created_at_utc",
        "biological_target_key",
        "receipt_sha256",
        "result",
        "manifest",
        "official_disposition",
        "status",
    }
    expected_result_identity = {
        key: identities["artifact"][key] for key in ("size_bytes", "sha256")
    }
    expected_manifest_identity = {
        key: identities["manifest"][key] for key in ("size_bytes", "sha256")
    }
    if (
        set(completion) != expected_completion_fields
        or completion.get("schema") != COMPLETION_SCHEMA
        or completion.get("status") != "TARGET_PUBLICATION_COMPLETE"
        or not isinstance(completion.get("created_at_utc"), str)
        or not completion.get("created_at_utc")
        or completion.get("biological_target_key")
        != attempt.get("biological_target_key")
        or completion.get("receipt_sha256") != identities["attempt"]["sha256"]
        or completion.get("result") != expected_result_identity
        or completion.get("manifest") != expected_manifest_identity
        or artifact.get("target_completion")
        != {"logical_name": identities["completion"]["logical_name"]}
    ):
        raise InstrumentError(
            "R5c target completion does not authenticate the exact publication"
        )

    replay = validate_target_replay(artifact, fingerprint=fingerprint)
    if completion.get("official_disposition") != replay.get(
        "official_disposition"
    ):
        raise InstrumentError(
            "R5c target completion official disposition does not replay"
        )
    return {
        "schema": "gfmcd.r5c.verified_target_package/1",
        "status": "EXACT_R5C_TARGET_PACKAGE_VERIFIED",
        "official_disposition": replay["official_disposition"],
        "scientific_outcomes_claim_eligible": replay[
            "scientific_outcomes_claim_eligible"
        ],
        "biological_target_key": attempt["biological_target_key"],
        "package_identities": identities,
        "offline_replay": replay,
    }


SUPPORTED_SCIENTIFIC_DISPOSITIONS = frozenset(
    {
        "R5C_DENSE_WRITE_AND_PRIMARY_PAIR_DELIVERY_SUPPORTED",
        "R5C_PRIMARY_PAIR_WITH_LOW_RANK_STRENGTHENING_SUPPORTED",
        "R5C_DENSE_WRITE_PRIMARY_PAIR_AND_DISTINCT_TESTED_SET_SUPPORTED",
        "R5C_DENSE_WRITE_PRIMARY_PAIR_DISTINCT_TESTED_SET_AND_H7_TRIPLE_SUPPORTED",
        "R5C_DENSE_WRITE_DISTINCT_TESTED_SET_AND_LOW_RANK_SUPPORTED",
        "R5C_DENSE_WRITE_DISTINCT_TESTED_SET_AND_NESTED_STRENGTHENINGS_SUPPORTED",
    }
)


def scientific_claim_eligible(*, status: str, instrument_ok: bool) -> bool:
    """License claims only for the closed set of supported R5c dispositions."""

    if type(instrument_ok) is not bool:
        raise TypeError("instrument_ok must be a literal boolean")
    return instrument_ok and status in SUPPORTED_SCIENTIFIC_DISPOSITIONS


def structured_controls_active(
    inference: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Recompute the decision-bearing structured-control activity flag."""

    return any(
        abs(
            float(
                inference[family]["row_tests"][row]["test"]["lower_edge"][
                    estimator
                ]["estimate"]
            )
        )
        >= SESOI
        for family in inference
        for row, requirement in {
            requirement.row: requirement
            for requirements in path_registry()[family].values()
            for requirement in requirements
        }.items()
        if requirement.decision == "EQUIVALENT_WITHIN_SESOI"
        for estimator in ("raw", "trimmed")
    )


def scientific_verdict(
    *,
    instrument_ok: bool,
    entry_behavior: str,
    dense_write: str,
    low_rank: str,
    c3_paths: Mapping[str, str],
    primary_delivery: str,
    distinct_tested_set: str,
    secondary_triple: str,
    controls_active: bool,
) -> str:
    """Return the strongest fixed-sequence claim without optional-path vetoes."""

    if not instrument_ok:
        return "INSTRUMENT_UNRESOLVED"
    if entry_behavior == "NOT_REPLICATED":
        return "ENTRY_BEHAVIOR_NOT_REPLICATED"
    if entry_behavior == "NONSPECIFIC":
        return "ENTRY_BEHAVIOR_NONSPECIFIC"
    primary_classes = [dense_write, primary_delivery]
    if "INSTRUMENT_UNRESOLVED" in primary_classes:
        return "INSTRUMENT_UNRESOLVED"
    if dense_write != "SUPPORTED":
        return "DENSE_LOCAL_WRITE_NOT_SUPPORTED"
    if primary_delivery != "SUPPORTED":
        return "PRIMARY_PAIR_DELIVERY_NOT_SUPPORTED"
    # Nested strengthenings cannot veto the primary result, but they are interpreted in
    # one ordered fixed sequence. A reached healthy negative leaves later gates untested;
    # an exact R5b-derived C2 skip is not a reached gate, so interpretation continues.
    c2_not_evaluated = low_rank in OPTIONAL_C2_NOT_EVALUATED_DISPOSITIONS
    if low_rank != "SUPPORTED" and not c2_not_evaluated:
        return "R5C_DENSE_WRITE_AND_PRIMARY_PAIR_DELIVERY_SUPPORTED"
    if distinct_tested_set != "SUPPORTED" or controls_active:
        return (
            "R5C_PRIMARY_PAIR_WITH_LOW_RANK_STRENGTHENING_SUPPORTED"
            if not c2_not_evaluated
            else "R5C_DENSE_WRITE_AND_PRIMARY_PAIR_DELIVERY_SUPPORTED"
        )
    if secondary_triple != "SUPPORTED":
        return (
            "R5C_DENSE_WRITE_DISTINCT_TESTED_SET_AND_LOW_RANK_SUPPORTED"
            if not c2_not_evaluated
            else "R5C_DENSE_WRITE_PRIMARY_PAIR_AND_DISTINCT_TESTED_SET_SUPPORTED"
        )
    return (
        "R5C_DENSE_WRITE_DISTINCT_TESTED_SET_AND_NESTED_STRENGTHENINGS_SUPPORTED"
        if not c2_not_evaluated
        else "R5C_DENSE_WRITE_PRIMARY_PAIR_DISTINCT_TESTED_SET_AND_H7_TRIPLE_SUPPORTED"
    )


def ordered_gate_dispositions(
    *,
    instrument_ok: bool,
    dense_write: str,
    primary_delivery: str,
    low_rank: str,
    distinct_tested_set: str,
    secondary_triple: str,
) -> dict[str, dict[str, Any]]:
    """Serialize scientific stops separately from instrument failures."""

    observed = {
        "R5C_DENSE_LOCAL_WRITE": dense_write,
        "R5C_PRIMARY_PAIR_DELIVERY": primary_delivery,
        "R5C_LOW_RANK_STRENGTHENING": low_rank,
        "R5C_DISTINCT_TESTED_SET_STRENGTHENING": distinct_tested_set,
        "R5C_SECONDARY_H7_TRIPLE_STRENGTHENING": secondary_triple,
    }
    output: dict[str, dict[str, Any]] = {
        "R1C_PAIRED_DONOR_BEHAVIOR": {
            "status": "SUPPORTED_BY_AUTHENTICATED_PREDECESSOR",
            "tested_here": False,
        },
        "R5A_DONOR_LOCAL_READER": {
            "status": "SUPPORTED_BY_AUTHENTICATED_PREDECESSOR",
            "tested_here": False,
        },
    }
    stopped: str | None = None
    for name in fixed_scientific_gate_order()[2:]:
        c2_not_evaluated = (
            name == "R5C_LOW_RANK_STRENGTHENING"
            and observed[name] in OPTIONAL_C2_NOT_EVALUATED_DISPOSITIONS
        )
        if stopped == "instrument":
            status = "NOT_TESTED_DUE_TO_PRIOR_INSTRUMENT_FAILURE"
        elif stopped == "scientific":
            status = "NOT_TESTED_DUE_TO_PRIOR_SCIENTIFIC_STOP"
        elif not instrument_ok:
            status = "INSTRUMENT_UNRESOLVED_AT_REACHED_GATE"
            stopped = "instrument"
        else:
            status = observed[name]
            if status != "SUPPORTED" and not c2_not_evaluated:
                stopped = (
                    "instrument"
                    if status == "INSTRUMENT_UNRESOLVED"
                    else "scientific"
                )
        output[name] = {
            "status": status,
            "tested_here": not c2_not_evaluated
            and status
            not in {
                "NOT_TESTED_DUE_TO_PRIOR_INSTRUMENT_FAILURE",
                "NOT_TESTED_DUE_TO_PRIOR_SCIENTIFIC_STOP",
            },
        }
    return output


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
    stage = str(result.get("stage"))
    cohort = result.get("cohort")
    cohort_role = cohort.get("role") if isinstance(cohort, Mapping) else None
    basis = result.get("r5b_optional_basis")
    basis_bundle_sha256 = (
        basis.get("runtime_bundle_sha256") if isinstance(basis, Mapping) else None
    )
    notes = [
        "Single masked acceptor G hard margin only; no masked-donor endpoint.",
        "Dense local L3H1 post-W_O write is primary; an R5b-validated basis is an optional C2 strengthening and is never refit or folded on target.",
        "Primary L4H11+L4H9 pair delivery is decision-bearing; distinct-tested-set and H7 arms are optional strengthenings.",
        "Mean and 10%-trimmed mean studentized gene-cluster bootstrap is the sole decision authority; sign flip is diagnostic only.",
        "Only C4's matched two-head random family can veto its primary pair; C3 singletons and C5 random triples are diagnostic/descriptive.",
    ]
    if stage == "preflight":
        notes.append(
            "The exact authenticated R1c validation-preflight 400 rows on chr11/16/19 are instrument preflight only; no R1c capstone or chr8/9 target was opened."
        )
    else:
        notes.extend(
            (
                "Target uses the exact authenticated 400-row R1c chr8/9 handoff in stored order; repeated genes are clustered and no row is dropped.",
                "The R1c P_clean/P_under_D/P_under_N common legs were replayed before writer or delivery interpretation.",
                "A biological target-attempt receipt precedes capstone/model/output inspection; completion follows immutable publication.",
            )
        )
    manifest = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r5c_acceptor_write_delivery",
        sys.argv,
        repo=REPO,
        status=str(result.get("status")),
        config={
            "stage": stage,
            "cohort_role": cohort_role,
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
            "r5b_optional_basis_bundle_sha256": basis_bundle_sha256,
            "execution_device": result.get("execution_device"),
            "decision_authority": result.get("decision_authority"),
        },
        seeds={"randomisation": PRODUCTION_SEED, "health_probe_selection": PRODUCTION_SEED},
        notes=notes,
    )
    manifest.git_commit = git_identity["commit"]
    manifest.git_dirty = False
    resolved_inputs = [Path(path).expanduser().resolve() for path in input_paths]
    missing_inputs = [path for path in resolved_inputs if not path.is_file()]
    if missing_inputs:
        raise InstrumentError(
            "R5c manifest input is missing: "
            + ", ".join(str(path) for path in missing_inputs)
        )
    manifest.inputs = [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in resolved_inputs
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


def _required_model_arguments(args: argparse.Namespace) -> None:
    common = {
        "frozen R5b disposition artifact": args.r5b_disposition_artifact,
        "frozen R5b disposition manifest": args.r5b_disposition_manifest,
        "R5b derivation artifact": args.r5b_derivation_artifact,
        "R5b derivation manifest": args.r5b_derivation_manifest,
        "R5b validation artifact": args.r5b_validation_artifact,
        "R5b validation manifest": args.r5b_validation_manifest,
        "output": args.out,
    }
    if args.stage == "preflight":
        required = common | {
            "R1c validation-preflight artifact": args.r1c_preflight_artifact,
            "R1c validation-preflight manifest": args.r1c_preflight_manifest,
            "R5a preflight artifact": args.r5a_preflight_artifact,
            "R5a preflight manifest": args.r5a_preflight_manifest,
        }
    elif args.stage == "target":
        required = common | {
            "R5c preflight artifact": args.r5c_preflight_artifact,
            "R5c preflight manifest": args.r5c_preflight_manifest,
            "R1c target artifact": args.r1c_artifact,
            "R1c target manifest": args.r1c_manifest,
            "R1c target attempt": args.r1c_attempt,
            "R1c target completion": args.r1c_completion,
            "R1c predecessor preflight artifact": args.r1c_preflight_artifact,
            "R1c predecessor preflight manifest": args.r1c_preflight_manifest,
            "R5a target artifact": args.r5a_target_artifact,
            "R5a target manifest": args.r5a_target_manifest,
        }
    else:
        raise InstrumentError("scored R5c stage must be preflight or target")
    missing = [label for label, value in required.items() if value is None]
    if missing:
        raise InstrumentError(f"R5c {args.stage} requires: " + ", ".join(missing))
    if (
        args.batch != PRODUCTION_BATCH
        or args.draws != PRODUCTION_RANDOMISATION
        or args.seed != PRODUCTION_SEED
        or args.calibration_trials != ENGINEERING_DIAGNOSTIC_TRIALS
    ):
        raise InstrumentError(
            "R5c production batch/randomisation/seed/calibration parameters are fixed"
        )


def _package_pair_matches_pins(
    identity: Mapping[str, Any],
    *,
    artifact_sha256: str,
    manifest_sha256: str,
) -> bool:
    return bool(
        isinstance(identity.get("artifact"), Mapping)
        and isinstance(identity.get("manifest"), Mapping)
        and identity["artifact"].get("sha256") == artifact_sha256
        and identity["manifest"].get("sha256") == manifest_sha256
    )


def load_r5c_preflight_package(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    authorization: Mapping[str, str],
    fingerprint: str,
    r5b_disposition_identity: Mapping[str, Any],
    r5b_basis_contract_sha256: str | None,
    r5b_status: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    record, _manifest, identity = load_exact_json_package(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        expected_artifact_sha256=authorization[
            "FROZEN_R5C_PREFLIGHT_ARTIFACT_SHA256"
        ],
        expected_manifest_sha256=authorization[
            "FROZEN_R5C_PREFLIGHT_MANIFEST_SHA256"
        ],
        label="R5c R1c-validation preflight",
    )
    inputs = record.get("preflight_inputs")
    if not isinstance(inputs, Mapping):
        raise InstrumentError("R5c preflight lacks exact predecessor packages")
    r1c_identity = inputs.get("r1c_validation_preflight")
    r5a_identity = inputs.get("r5a_preflight")
    stored_disposition_identity = inputs.get("r5b_disposition")
    if (
        not isinstance(r1c_identity, Mapping)
        or not _package_pair_matches_pins(
            r1c_identity,
            artifact_sha256=authorization[
                "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256"
            ],
            manifest_sha256=authorization[
                "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256"
            ],
        )
        or not isinstance(r5a_identity, Mapping)
        or not _package_pair_matches_pins(
            r5a_identity,
            artifact_sha256=authorization[
                "FROZEN_R5A_PREFLIGHT_ARTIFACT_SHA256"
            ],
            manifest_sha256=authorization[
                "FROZEN_R5A_PREFLIGHT_MANIFEST_SHA256"
            ],
        )
        or stored_disposition_identity != r5b_disposition_identity
        or not _package_pair_matches_pins(
            stored_disposition_identity,
            artifact_sha256=authorization[
                "FROZEN_R5B_DISPOSITION_ARTIFACT_SHA256"
            ],
            manifest_sha256=authorization[
                "FROZEN_R5B_DISPOSITION_MANIFEST_SHA256"
            ],
        )
        or inputs.get("r5b_status") != r5b_status
        or inputs.get("r5b_basis_contract_sha256")
        != r5b_basis_contract_sha256
    ):
        raise InstrumentError("R5c preflight predecessor package chain differs")
    validated = validate_r5c_preflight_record(
        record,
        fingerprint=fingerprint,
        r1c_preflight_identity=r1c_identity,
        r5a_preflight_identity=r5a_identity,
        r5b_disposition_identity=stored_disposition_identity,
        r5b_basis_contract_sha256=r5b_basis_contract_sha256,
        r5b_status=r5b_status,
    )
    replay = validate_preflight_replay(record, fingerprint=fingerprint)
    return record, identity, validated | {"offline_replay": replay}


def _target_upstream_chain(
    *,
    preflight_identity: Mapping[str, Any],
    r1c_verified: Mapping[str, Any],
    r5a_identity: Mapping[str, Any],
    r5a_validation: Mapping[str, Any],
    r5b_identity: Mapping[str, Any],
    r5b_disposition: Mapping[str, Any],
    r5b_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = {
        "schema": "gfmcd.r5c.target_upstream_chain/2",
        "temporal_contract": {
            "methods_preflight_and_r5b_disposition_frozen_before_r1c_unseal": True,
            "post_unseal_scientific_order": ["R1c_target", "R5a_target", "R5c_target"],
            "r5b_optional_basis_is_pre_unseal_instrument_not_scientific_stage": True,
            "target_refit_or_rank_selection": False,
        },
        "r5c_preflight": dict(preflight_identity),
        "r1c_target": {
            key: r1c_verified[key]
            for key in (
                "artifact",
                "manifest",
                "attempt",
                "attempt_id",
                "biological_target_key",
                "completion",
                "completion_id",
                "source_package_sha256",
                "cohort_sha256",
                "ordered_row_ids_sha256",
                "ordered_gene_ids_sha256",
                "sorted_unique_gene_ids_sha256",
                "capstone_construction_sha256",
            )
        },
        "r5a_target": {
            "package": dict(r5a_identity),
            "validation": dict(r5a_validation),
        },
        "r5b_pre_unseal_disposition": {
            "package": dict(r5b_identity),
            "status": r5b_disposition["status"],
            "path_local_c2_disposition": r5b_disposition[
                "path_local_c2_disposition"
            ],
            "runtime_contract_sha256": (
                None if r5b_contract is None else r5b_contract["contract_sha256"]
            ),
            "dense_primary_unaffected": True,
        },
    }
    return payload | {"chain_sha256": canonical_sha256(payload)}


def _target_capstone_summary(
    verified: Mapping[str, Any],
    materialization: Mapping[str, Any],
    rows: Sequence[Any],
) -> dict[str, Any]:
    return {
        "role": "AUTHENTICATED_R1C_400_ROW_SAME_ROW_CAPSTONE",
        "schema": verified["schema"],
        "row_count": verified["row_count"],
        "unique_gene_count": verified["unique_gene_count"],
        "cohort_sha256": verified["cohort_sha256"],
        "ordered_row_ids_sha256": verified["ordered_row_ids_sha256"],
        "ordered_gene_ids_sha256": verified["ordered_gene_ids_sha256"],
        "sorted_unique_gene_ids_sha256": verified[
            "sorted_unique_gene_ids_sha256"
        ],
        "source_legacy_cohort_sha256": verified[
            "source_legacy_cohort_sha256"
        ],
        "source_package_sha256": verified["source_package_sha256"],
        "capstone_construction_sha256": verified[
            "capstone_construction_sha256"
        ],
        "materialization": dict(materialization),
        "row_identity": [row.identity_record() for row in rows],
        "row_identity_sha256": canonical_sha256(
            [row.identity_record() for row in rows]
        ),
        "construction_sha256": canonical_sha256(
            [row.construction_record() for row in rows]
        ),
        "repeated_genes_allowed": True,
        "rows_filtered_or_reordered": False,
    }


def run_authorized_model_stage(
    args: argparse.Namespace,
    *,
    authorization: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute one sealed v2 stage with target-ledger ordering enforced."""

    if args.stage not in {"preflight", "target"}:
        raise InstrumentError("authorized model execution requires preflight or target")
    _required_model_arguments(args)
    try:
        runtime_policy = validate_r5_runtime_policy(device=args.device)
    except R5StackBindingError as exc:
        raise InstrumentError(f"R5c strict runtime policy failed: {exc}") from exc

    git_before = require_clean_git()
    binding_payload = instrument_binding_payload()
    fingerprint = canonical_sha256(binding_payload)
    if fingerprint != authorization.get("FROZEN_INSTRUMENT_FINGERPRINT_SHA256"):
        raise InstrumentError("authorized R5c fingerprint changed before execution")
    stack_fingerprint = binding_payload[STACK_FINGERPRINT_FIELD]
    current_stack = binding_payload[STACK_BINDING_FIELD]

    (
        r5b_disposition,
        r5b_identity,
        r5b_contract,
        runtime,
    ) = load_r5b_disposition_package(
        artifact_path=args.r5b_disposition_artifact,
        manifest_path=args.r5b_disposition_manifest,
        bundle_path=args.r5b_basis_bundle,
        derivation_artifact_path=args.r5b_derivation_artifact,
        derivation_manifest_path=args.r5b_derivation_manifest,
        validation_artifact_path=args.r5b_validation_artifact,
        validation_manifest_path=args.r5b_validation_manifest,
        authorization=authorization,
        expected_stack_fingerprint=stack_fingerprint,
    )
    r5b_status = str(r5b_disposition["status"])
    basis_plan = optional_basis_execution_plan(status=r5b_status, runtime=runtime)

    target_receipt: Mapping[str, Any] | None = None
    target_completion: Mapping[str, Any] | None = None
    preflight_record: Mapping[str, Any] | None = None
    preflight_validation: Mapping[str, Any] | None = None
    r1c_verified: Mapping[str, Any] | None = None
    r5a_record: Mapping[str, Any]
    r5a_identity: Mapping[str, Any]
    r5a_validation: Mapping[str, Any]
    upstream_chain: Mapping[str, Any] | None = None
    r1c_common_leg: Mapping[str, Any] | None = None
    r1c_common_leg_source: Mapping[str, Any] | None = None
    frozen_preflight_sensitivity: Mapping[str, Any] | None = None

    if args.stage == "preflight":
        r5a_record, r5a_identity, r5a_validation = load_r5a_stage_package(
            artifact_path=args.r5a_preflight_artifact,
            manifest_path=args.r5a_preflight_manifest,
            stage="preflight",
            authorization=authorization,
            expected_stack_fingerprint=stack_fingerprint,
        )
        if (
            r5b_contract is not None
            and r5a_validation["checkpoint_sha256"]
            != r5b_contract["checkpoint_sha256"]
        ):
            raise InstrumentError("R5a preflight and frozen R5b basis checkpoints differ")
        binding_suite = run_constituent_binding_suite(
            n_draws=PRODUCTION_BOOTSTRAP_DRAWS,
            seed=PRODUCTION_SEED,
        )
        engineering_calibration = run_engineering_calibration(
            n_draws=ENGINEERING_DIAGNOSTIC_DRAWS,
            seed=PRODUCTION_SEED,
            n_trials=ENGINEERING_DIAGNOSTIC_TRIALS,
        )
        validate_binding_and_calibration_records(
            binding_suite,
            engineering_calibration,
        )
        output = Path(args.out).expanduser().resolve()
        if output.exists() or output.with_suffix(".manifest.json").exists():
            raise InstrumentError("R5c outputs are immutable; choose a fresh versioned path")
        r1c_preflight_handoff, r1c_preflight_verified = load_r1c_preflight_handoff(
            artifact_path=args.r1c_preflight_artifact,
            manifest_path=args.r1c_preflight_manifest,
            authorization=authorization,
        )
        rows, cohort = materialize_r1c_preflight_rows(
            gtf=args.gtf,
            fasta=args.fasta,
            handoff=r1c_preflight_handoff,
        )
        common_leg_references = r1c_preflight_common_leg_references(
            r1c_preflight_handoff
        )
        r1c_common_leg_source = {
            "schema": r1c_preflight_handoff["schema"],
            "source_package_sha256": r1c_preflight_handoff[
                "source_package_sha256"
            ],
            "cohort_sha256": r1c_preflight_handoff["cohort_sha256"],
            "ordered_row_ids_sha256": r1c_preflight_handoff[
                "ordered_row_ids_sha256"
            ],
            "ordered_partner_margin_levels_sha256": r1c_preflight_handoff[
                "ordered_partner_margin_levels_sha256"
            ],
            "record_sha256": r1c_preflight_handoff["record_sha256"],
            "row_count": r1c_preflight_handoff["row_count"],
            "loaded_after_r5c_target_receipt": False,
            "used_for_cohort_selection": False,
        }
        preflight_inputs = {
            "r1c_validation_preflight": {
                key: value
                for key, value in r1c_preflight_verified.items()
                if key not in {"rows", "common_leg_rows"}
            },
            "r5a_preflight": dict(r5a_identity),
            "r5b_disposition": dict(r5b_identity),
            "r5b_status": r5b_status,
            "r5b_path_local_c2_disposition": r5b_disposition[
                "path_local_c2_disposition"
            ],
            "r5b_basis_contract_sha256": (
                None if r5b_contract is None else r5b_contract["contract_sha256"]
            ),
            "future_r1c_or_r5a_target_required": False,
        }
    else:
        preflight_record_raw, preflight_identity, preflight_validation = (
            load_r5c_preflight_package(
                artifact_path=args.r5c_preflight_artifact,
                manifest_path=args.r5c_preflight_manifest,
                authorization=authorization,
                fingerprint=fingerprint,
                r5b_disposition_identity=r5b_identity,
                r5b_basis_contract_sha256=(
                    None
                    if r5b_contract is None
                    else r5b_contract["contract_sha256"]
                ),
                r5b_status=r5b_status,
            )
        )
        preflight_record = preflight_record_raw
        binding_suite = preflight_record["binding_suite"]
        engineering_calibration = preflight_record["engineering_calibration"]
        frozen_preflight_sensitivity = preflight_record.get("measurement", {}).get(
            "observed_centered_robust_sensitivity"
        )
        if not isinstance(frozen_preflight_sensitivity, Mapping):
            raise InstrumentError("R5c preflight lacks robust sensitivity evidence")

        handoff, r1c_verified_loaded = load_r1c_target_handoff(
            artifact_path=args.r1c_artifact,
            manifest_path=args.r1c_manifest,
            attempt_path=args.r1c_attempt,
            completion_path=args.r1c_completion,
            preflight_artifact_path=args.r1c_preflight_artifact,
            preflight_manifest_path=args.r1c_preflight_manifest,
            authorization=authorization,
        )
        r1c_verified = r1c_verified_loaded
        r5a_record, r5a_identity, r5a_validation = load_r5a_stage_package(
            artifact_path=args.r5a_target_artifact,
            manifest_path=args.r5a_target_manifest,
            stage="target",
            authorization=authorization,
            expected_stack_fingerprint=stack_fingerprint,
            capstone_handoff=r1c_verified,
        )
        try:
            validate_serialized_r5_stack(r5a_record, current=current_stack)
        except R5StackBindingError as exc:
            raise InstrumentError(
                f"R5a target joint-stack package differs before receipt: {exc}"
            ) from exc
        if (
            r5b_contract is not None
            and r5a_validation["checkpoint_sha256"]
            != r5b_contract["checkpoint_sha256"]
        ):
            raise InstrumentError("R5a target and frozen pre-unseal R5b basis checkpoints differ")
        upstream_chain = _target_upstream_chain(
            preflight_identity=preflight_identity,
            r1c_verified=r1c_verified,
            r5a_identity=r5a_identity,
            r5a_validation=r5a_validation,
            r5b_identity=r5b_identity,
            r5b_disposition=r5b_disposition,
            r5b_contract=r5b_contract,
        )
        identity = build_r5c_target_identity(
            r1c_verified,
            checkpoint_sha256=r5a_validation["checkpoint_sha256"],
        )
        attempt_context = {
            "r5_stack_fingerprint_sha256": stack_fingerprint,
            "stage_preflight_artifact_sha256": authorization[
                "FROZEN_R5C_PREFLIGHT_ARTIFACT_SHA256"
            ],
            "stage_preflight_manifest_sha256": authorization[
                "FROZEN_R5C_PREFLIGHT_MANIFEST_SHA256"
            ],
            "upstream_package_chain_sha256": upstream_chain["chain_sha256"],
        }
        try:
            target_receipt = reserve_r5_target_attempt(
                repo=REPO,
                identity=identity,
                attempt_context=attempt_context,
            )
        except (R5TargetSpentError, OSError, RuntimeError, ValueError) as exc:
            raise InstrumentError(f"R5c biological target authorization failed: {exc}") from exc

        # The irreversible receipt now exists.  Only now may output/model/source paths,
        # R1c outcome rows, or chr8/9 source examples be inspected.
        output = Path(args.out).expanduser().resolve()
        if output.exists() or output.with_suffix(".manifest.json").exists():
            raise InstrumentError("R5c outputs are immutable; choose a fresh versioned path")
        common_leg_references, r1c_common_leg_source = (
            load_r1c_target_common_leg_references(
                artifact_path=args.r1c_artifact,
                manifest_path=args.r1c_manifest,
                attempt_path=args.r1c_attempt,
                completion_path=args.r1c_completion,
                preflight_artifact_path=args.r1c_preflight_artifact,
                preflight_manifest_path=args.r1c_preflight_manifest,
                handoff=r1c_verified,
            )
        )
        source_identity = validate_r5_source_binding(args.gtf, args.fasta)
        examples = load_examples_unscored(
            str(args.gtf),
            str(args.fasta),
            "test",
            None,
            tag="r1c-target-r1c-single-g-primary-family-v2",
            allowed_chromosomes=("chr8", "chr9"),
        )
        try:
            rows, materialization = materialize_r5_capstone_rows(examples, handoff)
        except (R5CapstoneCohortError, RuntimeError, TypeError, ValueError) as exc:
            raise InstrumentError(f"R5c capstone materialization failed: {exc}") from exc
        if (
            len(rows) != 400
            or materialization.get("handoff_order_preserved") is not True
            or materialization.get("rows_filtered_or_reordered") is not False
        ):
            raise InstrumentError("R5c capstone materialization is not exact and complete")
        cohort = {
            "source_identity": source_identity,
            "binding": _target_capstone_summary(
                r1c_verified,
                materialization,
                rows,
            ),
            "role": "AUTHENTICATED_R1C_400_ROW_SAME_ROW_CAPSTONE",
        }
        preflight_inputs = None

    try:
        validate_serialized_r5_stack(r5a_record, current=current_stack)
    except R5StackBindingError as exc:
        raise InstrumentError(f"R5a joint-stack package differs: {exc}") from exc

    checkpoint = checkpoint_identity(args.model)
    if (
        r5b_contract is not None
        and checkpoint["checkpoint_aggregate_sha256"]
        != r5b_contract["checkpoint_sha256"]
    ):
        raise InstrumentError("model checkpoint differs from the frozen R5b basis")
    if checkpoint["checkpoint_aggregate_sha256"] != r5a_validation["checkpoint_sha256"]:
        raise InstrumentError("model checkpoint differs from the R5a predecessor")

    package_paths: list[Path] = [
        Path(value).expanduser().resolve()
        for value in (
            args.r5b_disposition_artifact,
            args.r5b_disposition_manifest,
            args.r5b_derivation_artifact,
            args.r5b_derivation_manifest,
            args.r5b_validation_artifact,
            args.r5b_validation_manifest,
        )
    ]
    if args.r5b_basis_bundle is not None:
        package_paths.append(Path(args.r5b_basis_bundle).expanduser().resolve())
    if args.stage == "preflight":
        package_paths.extend(
            (
                Path(args.r5a_preflight_artifact).expanduser().resolve(),
                Path(args.r5a_preflight_manifest).expanduser().resolve(),
            )
        )
    else:
        package_paths.extend(
            Path(value).expanduser().resolve()
            for value in (
                args.r5c_preflight_artifact,
                args.r5c_preflight_manifest,
                args.r1c_artifact,
                args.r1c_manifest,
                args.r1c_attempt,
                args.r1c_completion,
                args.r1c_preflight_artifact,
                args.r1c_preflight_manifest,
                args.r5a_target_artifact,
                args.r5a_target_manifest,
                target_receipt["receipt_path"],
            )
        )
    small_input_paths = package_paths + [
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
        R5B_RUNNER,
    ]
    input_snapshot = _snapshot_files(small_input_paths)

    staging: tempfile.TemporaryDirectory[str] | None = None
    try:
        staging, staged_model = stage_checkpoint_bundle(args.model, checkpoint)
        device = pick_device(args.device)
        if str(device) != "mps":
            raise InstrumentError("R5c strict runtime selected a non-MPS device")
        adapter = load_adapter(str(staged_model), device, eager=False)
        adapter.model.eval()
        torch.set_grad_enabled(False)
        with torch.no_grad():
            assert common_leg_references is not None
            r1c_common_leg = replay_r1c_common_legs_before_interventions(
                adapter=adapter,
                rows=rows,
                references=common_leg_references,
                batch=args.batch,
            )
            measurement = run_model_measurements(
                adapter=adapter,
                rows=rows,
                runtime=runtime,
                r5b_status=r5b_status,
                batch=args.batch,
                n_draws=args.draws,
                seed=args.seed,
                binding_suite=binding_suite,
                engineering_calibration=engineering_calibration,
                frozen_preflight_sensitivity=frozen_preflight_sensitivity,
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
    status = (
        "PREFLIGHT_INSTRUMENT_PASS"
        if args.stage == "preflight" and instrument_ok
        else "INSTRUMENT_UNRESOLVED"
        if args.stage == "preflight"
        else str(measurement.get("verdict"))
    )
    path_classes = measurement.get("path_classifications", {})
    gate_dispositions = (
        None
        if args.stage == "preflight"
        else ordered_gate_dispositions(
            instrument_ok=instrument_ok,
            dense_write=path_classes["C1"]["P_C1_DENSE_LOCAL_WRITE"],
            primary_delivery=path_classes["C4"][
                "P_C4_PRIMARY_PAIR_DELIVERY"
            ],
            low_rank=path_classes["C2"]["P_C2_LOW_RANK_STRENGTHENING"],
            distinct_tested_set=path_classes["C4"][
                "P_C4_DISTINCT_WITHIN_TESTED_SET"
            ],
            secondary_triple=path_classes["C5"]["P_C5_SECONDARY_TRIPLE"],
        )
    )
    registry_audit = model_free_registry_audit()
    if canonical_sha256(registry_audit) != binding_payload["registry_audit_sha256"]:
        raise InstrumentError("R5c registry audit changed before artifact construction")

    if args.stage == "preflight":
        cohort_record = {
            "role": cohort["role"],
            "row_count": len(rows),
            "unique_gene_count": len(
                {str(row.example.gene_id) for row in rows}
            ),
            "rows_filtered_or_reordered": False,
            "r5c_capstone_authorized": False,
            "binding": cohort["binding"],
            "audit": cohort["audit"],
            "row_identity": [row.identity_record() for row in rows],
        }
    else:
        cohort_record = cohort["binding"]

    record = {
        "schema": SCHEMA,
        "experiment": "R5c_acceptor_write_delivery",
        "stage": args.stage,
        "status": status,
        "tier": "DEVELOPMENT_INSTRUMENT" if args.stage == "preflight" else "CONFIRMATORY",
        "behavioral_endpoint": "masked_acceptor_G_margin_only",
        "masked_donor_endpoint_constructed": False,
        "sealed_chr8_chr9_opened": args.stage == "target",
        "opened_chromosomes": (
            ["chr8", "chr9"]
            if args.stage == "target"
            else list(R5_VALIDATION_CHROMOSOMES)
        ),
        "instrument_fingerprint_sha256": fingerprint,
        "instrument_binding": binding_payload,
        STACK_BINDING_FIELD: binding_payload[STACK_BINDING_FIELD],
        STACK_FINGERPRINT_FIELD: stack_fingerprint,
        "registry_audit": registry_audit,
        "literal_seals": dict(authorization),
        "git": git_before,
        "runtime_identity": stable_runtime_identity(),
        "runtime_policy": runtime_policy,
        "execution_device": str(device),
        "checkpoint": checkpoint,
        "source_identity": cohort["source_identity"],
        "cohort": cohort_record,
        "preflight_inputs": preflight_inputs,
        "preflight_package": (
            None
            if args.stage == "preflight"
            else {
                "record_sha256": canonical_sha256(preflight_record),
                "validation": dict(preflight_validation),
                "artifact_sha256": authorization[
                    "FROZEN_R5C_PREFLIGHT_ARTIFACT_SHA256"
                ],
                "manifest_sha256": authorization[
                    "FROZEN_R5C_PREFLIGHT_MANIFEST_SHA256"
                ],
            }
        ),
        "r1c_capstone_handoff": (
            None
            if r1c_verified is None
            else {
                key: value
                for key, value in r1c_verified.items()
                if key != "rows"
            }
        ),
        "r1c_same_row_common_leg_replay": r1c_common_leg,
        "r1c_common_leg_source": r1c_common_leg_source,
        "r5a_predecessor": {
            "package": dict(r5a_identity),
            "validation": dict(r5a_validation),
        },
        "r5b_optional_basis": {
            "package": dict(r5b_identity),
            "disposition_status": r5b_status,
            "execution_plan": basis_plan,
            "contract": r5b_contract,
            "runtime_bundle_sha256": (
                None if runtime is None else runtime.bundle_sha256
            ),
            "runtime_array_sha256": (
                None
                if runtime is None
                else dict(runtime.authenticated_runtime_array_sha256)
            ),
            "target_refit_performed": False,
            "target_fold_assignment": None,
        },
        "upstream_chain": upstream_chain,
        "target_attempt_receipt": target_receipt,
        "target_completion": (
            None
            if target_receipt is None
            else {
                "logical_name": Path(
                    str(target_receipt["receipt_path"])
                ).with_suffix(".completion.json").name
            }
        ),
        "binding_suite": binding_suite,
        "engineering_calibration": engineering_calibration,
        "measurement": measurement,
        "fixed_scientific_gate_order": list(fixed_scientific_gate_order()),
        "fixed_scientific_gate_dispositions": gate_dispositions,
        "decision_authority": (
            "mean_and_10pct_trimmed_studentized_gene_cluster_bootstrap"
        ),
        "sign_flip_role": "diagnostic_only",
        "scientific_outcomes_claim_eligible": bool(
            args.stage == "target"
            and scientific_claim_eligible(
                status=status,
                instrument_ok=instrument_ok,
            )
        ),
    }

    fasta_path = Path(args.fasta).expanduser().resolve()
    manifest_inputs: list[Path] = small_input_paths + [
        Path(args.gtf).expanduser().resolve(),
        fasta_path,
        Path(str(fasta_path) + ".fai"),
        *_checkpoint_file_paths(checkpoint),
    ]
    result_sha, manifest_sha = write_result_package(
        output,
        record,
        input_paths=manifest_inputs,
        git_identity=git_before,
    )
    manifest_path = output.with_suffix(".manifest.json")
    if args.stage == "target":
        assert target_receipt is not None
        try:
            target_completion = complete_r5_target_attempt(
                receipt=target_receipt,
                result_identity={
                    "size_bytes": output.stat().st_size,
                    "sha256": result_sha,
                },
                manifest_identity={
                    "size_bytes": manifest_path.stat().st_size,
                    "sha256": manifest_sha,
                },
                official_disposition=status,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise InstrumentError(
                "R5c result was published but target completion failed; target remains spent: "
                + str(exc)
            ) from exc
    return record, {
        "output": str(output),
        "result_sha256": result_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "target_completion": target_completion,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "preflight", "target"), default="audit")
    parser.add_argument("--r5a-preflight-artifact")
    parser.add_argument("--r5a-preflight-manifest")
    parser.add_argument("--r5a-target-artifact")
    parser.add_argument("--r5a-target-manifest")
    parser.add_argument("--r5b-disposition-artifact")
    parser.add_argument("--r5b-disposition-manifest")
    parser.add_argument("--r5b-basis-bundle")
    parser.add_argument("--r5b-derivation-artifact")
    parser.add_argument("--r5b-derivation-manifest")
    parser.add_argument("--r5b-validation-artifact")
    parser.add_argument("--r5b-validation-manifest")
    parser.add_argument("--r5c-preflight-artifact")
    parser.add_argument("--r5c-preflight-manifest")
    parser.add_argument("--r1c-artifact")
    parser.add_argument("--r1c-manifest")
    parser.add_argument("--r1c-attempt")
    parser.add_argument("--r1c-completion")
    parser.add_argument("--r1c-preflight-artifact")
    parser.add_argument("--r1c-preflight-manifest")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gtf", default=DEFAULT_GTF)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch", type=int, default=PRODUCTION_BATCH)
    parser.add_argument("--draws", type=int, default=PRODUCTION_RANDOMISATION)
    parser.add_argument("--seed", type=int, default=PRODUCTION_SEED)
    parser.add_argument(
        "--calibration-trials", type=int, default=ENGINEERING_DIAGNOSTIC_TRIALS
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
