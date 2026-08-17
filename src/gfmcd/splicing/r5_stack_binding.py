"""Canonical prospective binding for the complete R5a--R5c mechanism stack.

The three R5 runners deliberately keep their own experiment-specific instrument
payloads.  This module adds the stronger invariant that was previously missing: every
stage also carries *one identical* stack payload and fingerprint.  The payload binds all
three runners and protocols, the R3d scientific-entry and R3c provenance runners, the full
repository-local Python import closure, the schema vocabulary, and the stable runtime.

Freeze-pin assignment values are the only normalized source text.  Consequently filling
reviewed literal pins does not create a self-referential hash, while any executable or
method change does.
"""

from __future__ import annotations

import ast
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Mapping, Sequence

from gfmcd.splicing.acceptor_mechanism_cohort import (
    R1C_TARGET_ATTEMPT_SCHEMA,
    R1C_TARGET_COMPLETION_SCHEMA,
    R1C_BIOLOGICAL_TARGET_IDENTITY_SCHEMA,
    R1C_TARGET_SCHEMA,
    R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
    R5_CAPSTONE_ROW_MATERIALIZATION_SCHEMA,
    R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA,
    R5_COHORT_SCHEMA,
    R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
    R5_PREFLIGHT_ROW_MATERIALIZATION_SCHEMA,
    R5_RESIDUAL_VALIDATION_BUCKET_ROLES,
    R5_RESIDUAL_VALIDATION_ROWS_PER_BUCKET,
    R5_RESIDUAL_VALIDATION_STAGE_CONSUMERS,
    R5A_R5C_PREFLIGHT_COHORT_ROLE,
    R5C_REQUIRED_COHORT_ROLE,
)
from gfmcd.splicing.r5_upstream_provenance import (
    PINNED_R5_REPOSITORY_RUNTIME_IDENTITY,
    PINNED_R5_SOURCE_IDENTITY,
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
    R5_EXCLUSION_REGISTRY_SCHEMA,
    R5UpstreamProvenanceError,
    UPSTREAM_PACKAGE_IDENTITIES,
    derive_r5_exclusion_registry,
    load_r3c_exclusion_package,
    load_r3d_entry_package,
    validate_r3d_target_record,
    validate_r5_exclusion_registry,
)


STACK_SCHEMA = "gfmcd.r5.stack_binding/2"
R3C_ENTRY_SCHEMA = "gfmcd.r3c.acceptor_head_confirmation/2"
# Compatibility name for the upstream artifact schema.  R3c is not an R5 entry gate.
R3C_PROVENANCE_SCHEMA = R3C_ENTRY_SCHEMA
R3C_INSTRUMENT_REVISION = "r3c-v2.1-monotone-health-early-preflight-authorization"
R3C_TARGET_HEALTH_SCHEMA = "gfmcd.r3c.target_path_health_authorization/1"
R3C_TARGET_HEALTH_RULE = (
    "effective target health = reviewed preflight health AND current health"
)
R3C_PATH_HEALTH_RULE = "preflight_health_ok AND current_health_ok"
R3C_PATH_NAMES = (
    "L3H1_reused_component",
    "L2H15_mean_component",
    "L1H12_mean_component",
    "L4H11_noising_component",
    "L4H9_noising_component",
    "L4H7_noising_component",
    "distinct_acceptor_route_candidate",
)
R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS = ("L2H15_mean_component",)
STACK_FINGERPRINT_FIELD = "r5_stack_fingerprint_sha256"
STACK_BINDING_FIELD = "r5_stack_binding"

R5_RUNNER_PATHS = (
    "scripts/splicebert_acceptor_coupling/run_r5a_acceptor_address_binding.py",
    "scripts/splicebert_acceptor_coupling/run_r5b_acceptor_side_state.py",
    "scripts/splicebert_acceptor_coupling/run_r5c_acceptor_write_delivery.py",
)
R5_PROTOCOL_PATHS = (
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5A_ACCEPTOR_ADDRESS_BINDING.md",
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5B_ACCEPTOR_SIDE_STATE.md",
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R5C_ACCEPTOR_WRITE_DELIVERY.md",
)
R3C_RUNNER_PATH = (
    "scripts/splicebert_acceptor_coupling/run_r3c_acceptor_head_confirmation.py"
)
R3D_RUNNER_PATH = (
    "scripts/splicebert_acceptor_coupling/run_r3d_acceptor_specificity_adjudication.py"
)
R1C_RUNNER_PATH = (
    "scripts/splicebert_acceptor_coupling/run_r1c_sealed_reverse_behavior.py"
)
R1C_PROTOCOL_PATH = (
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R1C_SEALED_REVERSE_BEHAVIOR_CONFIRMATION.md"
)
R1C_BEHAVIOR_SCHEMA = "gfmcd.r1c.reverse_behavior/3"
REPOSITORY_RUNTIME_PATHS = ("pyproject.toml", "uv.lock")
RUNTIME_PACKAGES = (
    "numpy",
    "scipy",
    "torch",
    "transformers",
    "tokenizers",
    "pyfaidx",
)
SHARED_R1C_PACKAGE_PIN_NAMES = (
    "FROZEN_R1C_PREFLIGHT_ARTIFACT_LOGICAL_NAME",
    "FROZEN_R1C_PREFLIGHT_ARTIFACT_SIZE_BYTES",
    "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
    "FROZEN_R1C_PREFLIGHT_MANIFEST_LOGICAL_NAME",
    "FROZEN_R1C_PREFLIGHT_MANIFEST_SIZE_BYTES",
    "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256",
    "FROZEN_R1C_TARGET_ARTIFACT_LOGICAL_NAME",
    "FROZEN_R1C_TARGET_ARTIFACT_SIZE_BYTES",
    "FROZEN_R1C_TARGET_ARTIFACT_SHA256",
    "FROZEN_R1C_TARGET_MANIFEST_LOGICAL_NAME",
    "FROZEN_R1C_TARGET_MANIFEST_SIZE_BYTES",
    "FROZEN_R1C_TARGET_MANIFEST_SHA256",
    "FROZEN_R1C_TARGET_ATTEMPT_LOGICAL_NAME",
    "FROZEN_R1C_TARGET_ATTEMPT_SIZE_BYTES",
    "FROZEN_R1C_TARGET_ATTEMPT_SHA256",
    "FROZEN_R1C_TARGET_COMPLETION_LOGICAL_NAME",
    "FROZEN_R1C_TARGET_COMPLETION_SIZE_BYTES",
    "FROZEN_R1C_TARGET_COMPLETION_SHA256",
)

# This is an explicit cross-stage contract, not a claim that every schema token is
# emitted in every run.  Literal schema strings discovered from the source closure are
# recorded separately so adding or removing an executable schema changes the binding.
STACK_CONTRACT = {
    "stage_order": [
        "R3d Path-A authenticated prior",
        "R1c nonsealed validation behavior preflight",
        "R5a authenticated R1c-validation preflight and instrument freeze",
        "R5b residual bucket-0 derivation, bucket-1 validation, and union-basis freeze",
        "R5c authenticated R1c-validation preflight and delivery-instrument freeze",
        "R1c sealed chr8/chr9 acceptor-only behavior target",
        "R5a authenticated capstone target",
        "R5c authenticated capstone target",
    ],
    "method_freeze_order_before_chr8_chr9_unseal": [
        "R1c nonsealed validation behavior preflight",
        "R5a authenticated R1c-validation preflight and instrument freeze",
        "R5b residual bucket-0 derivation, bucket-1 validation, and union-basis freeze",
        "R5c authenticated R1c-validation preflight and delivery-instrument freeze",
    ],
    "scientific_target_order_after_method_freeze": [
        "R1c sealed chr8/chr9 acceptor-only behavior target",
        "R5a authenticated capstone target",
        "R5c authenticated capstone target",
    ],
    "non_entry_provenance_inputs": ["exact committed R3c preflight"],
    "behavioral_endpoint": "single masked acceptor G hard margin only",
    "masked_donor_endpoint_constructed": False,
    "target_holdout_scope": {
        "licensed_term": "endpoint-outcome-held-out",
        "historical_chr8_chr9_sequence_materialization_disclosed": True,
        "globally_sequence_unseen": False,
        "known_prior_masked_acceptor_endpoint_scores_on_target_rows": 0,
    },
    "writer_site_semantics": {
        "causal_measurement_site": "post-WO attention output in residual space",
        "model_free_projection_primitive": (
            "gfmcd.splicing.acceptor_coupling_mechanism."
            "output_projected_head_contribution"
        ),
        "torch_projection_primitive": (
            "gfmcd.circuits.hf_bert.project_head_context_to_residual"
        ),
        "torch_intervention_primitive": (
            "gfmcd.circuits.hf_bert.add_attention_output_delta"
        ),
        "pre_WO_context_is_not_a_residual_writer_measurement": True,
    },
    "residual_validation_cohort": {
        "schema": R5_COHORT_SCHEMA,
        "authorized_stage_consumers": list(
            R5_RESIDUAL_VALIDATION_STAGE_CONSUMERS
        ),
        "bucket_roles": dict(R5_RESIDUAL_VALIDATION_BUCKET_ROLES),
        "rows_per_bucket": R5_RESIDUAL_VALIDATION_ROWS_PER_BUCKET,
        "r5a_r5c_preflight_authorized": False,
        "r5c_capstone_authorized": False,
    },
    "r5a_r5c_preflight_cohort": {
        "handoff_schema": R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
        "row_materialization_schema": R5_PREFLIGHT_ROW_MATERIALIZATION_SCHEMA,
        "required_role": R5A_R5C_PREFLIGHT_COHORT_ROLE,
        "source_schema": R1C_TARGET_SCHEMA,
        "source_stage": "preflight",
        "source_split": "val",
        "source_n": 400,
        "source_preconstruction_allowed_chromosomes": ["chr11", "chr16", "chr19"],
        "residual_validation_cohort_may_be_substituted": False,
        "completion_or_target_attempt_required": False,
        "scientific_entry_authorized": False,
        "preserved_outcome_fields": ["P_clean", "P_under_D", "P_under_N"],
        "semantic_verifier": "verify_successor_preflight_package",
        "prospective_source_package_pins": {
            "artifact": {
                "logical_role": "authenticated R1c validation-preflight artifact",
                "content_schema": R1C_TARGET_SCHEMA,
                "logical_name": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_ARTIFACT_LOGICAL_NAME",
                    "value": "<FROZEN_PIN>",
                },
                "size_bytes": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_ARTIFACT_SIZE_BYTES",
                    "value": "<FROZEN_PIN>",
                },
                "sha256": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
                    "value": "<FROZEN_PIN>",
                },
            },
            "manifest": {
                "logical_role": "sibling R1c validation-preflight ExperimentManifest",
                "logical_name": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_MANIFEST_LOGICAL_NAME",
                    "value": "<FROZEN_PIN>",
                },
                "size_bytes": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_MANIFEST_SIZE_BYTES",
                    "value": "<FROZEN_PIN>",
                },
                "sha256": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256",
                    "value": "<FROZEN_PIN>",
                },
            },
        },
    },
    "r5c_capstone_cohort": {
        "handoff_schema": R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
        "row_materialization_schema": R5_CAPSTONE_ROW_MATERIALIZATION_SCHEMA,
        "verified_common_leg_schema": R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA,
        "verified_common_leg_loader": "load_r1c_verified_target_common_legs",
        "verified_common_leg_use": (
            "same-row P_clean/P_under_D/P_under_N replay after each downstream "
            "target receipt; never cohort selection"
        ),
        "row_materialization_digest_field": (
            "materialization_construction_sha256"
        ),
        "source_examples_may_be_passed_only_after_target_ledger_authorization": True,
        "materializer_performs_io": False,
        "required_role": R5C_REQUIRED_COHORT_ROLE,
        "source_schema": R1C_TARGET_SCHEMA,
        "semantic_verifier": "verify_successor_target_package",
        "authenticated_input_count": 6,
        "may_reuse_residual_validation_bucket_1": False,
        "handoff_constructs_or_reads_chr8_chr9": False,
        "prospective_source_package_pins": {
            "artifact": {
                "logical_role": "authenticated R1c target artifact",
                "content_schema": R1C_TARGET_SCHEMA,
                "logical_name": {
                    "pin_symbol": "FROZEN_R1C_TARGET_ARTIFACT_LOGICAL_NAME",
                    "value": "<FROZEN_PIN>",
                },
                "size_bytes": {
                    "pin_symbol": "FROZEN_R1C_TARGET_ARTIFACT_SIZE_BYTES",
                    "value": "<FROZEN_PIN>",
                },
                "sha256": {
                    "pin_symbol": "FROZEN_R1C_TARGET_ARTIFACT_SHA256",
                    "value": "<FROZEN_PIN>",
                },
            },
            "manifest": {
                "logical_role": "sibling R1c ExperimentManifest",
                "logical_name": {
                    "pin_symbol": "FROZEN_R1C_TARGET_MANIFEST_LOGICAL_NAME",
                    "value": "<FROZEN_PIN>",
                },
                "size_bytes": {
                    "pin_symbol": "FROZEN_R1C_TARGET_MANIFEST_SIZE_BYTES",
                    "value": "<FROZEN_PIN>",
                },
                "sha256": {
                    "pin_symbol": "FROZEN_R1C_TARGET_MANIFEST_SHA256",
                    "value": "<FROZEN_PIN>",
                },
            },
            "attempt": {
                "logical_role": "create-only R1c biological-target attempt",
                "content_schema": R1C_TARGET_ATTEMPT_SCHEMA,
                "logical_name": {
                    "pin_symbol": "FROZEN_R1C_TARGET_ATTEMPT_LOGICAL_NAME",
                    "value": "<FROZEN_PIN>",
                },
                "size_bytes": {
                    "pin_symbol": "FROZEN_R1C_TARGET_ATTEMPT_SIZE_BYTES",
                    "value": "<FROZEN_PIN>",
                },
                "sha256": {
                    "pin_symbol": "FROZEN_R1C_TARGET_ATTEMPT_SHA256",
                    "value": "<FROZEN_PIN>",
                },
            },
            "completion": {
                "logical_role": "create-only R1c target package completion",
                "content_schema": R1C_TARGET_COMPLETION_SCHEMA,
                "logical_name": {
                    "pin_symbol": "FROZEN_R1C_TARGET_COMPLETION_LOGICAL_NAME",
                    "value": "<FROZEN_PIN>",
                },
                "size_bytes": {
                    "pin_symbol": "FROZEN_R1C_TARGET_COMPLETION_SIZE_BYTES",
                    "value": "<FROZEN_PIN>",
                },
                "sha256": {
                    "pin_symbol": "FROZEN_R1C_TARGET_COMPLETION_SHA256",
                    "value": "<FROZEN_PIN>",
                },
            },
            "preflight_artifact": {
                "logical_role": "exact authenticated predecessor R1c preflight artifact",
                "content_schema": R1C_TARGET_SCHEMA,
                "logical_name": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_ARTIFACT_LOGICAL_NAME",
                    "value": "<FROZEN_PIN>",
                },
                "size_bytes": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_ARTIFACT_SIZE_BYTES",
                    "value": "<FROZEN_PIN>",
                },
                "sha256": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
                    "value": "<FROZEN_PIN>",
                },
            },
            "preflight_manifest": {
                "logical_role": "exact authenticated predecessor R1c preflight manifest",
                "logical_name": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_MANIFEST_LOGICAL_NAME",
                    "value": "<FROZEN_PIN>",
                },
                "size_bytes": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_MANIFEST_SIZE_BYTES",
                    "value": "<FROZEN_PIN>",
                },
                "sha256": {
                    "pin_symbol": "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256",
                    "value": "<FROZEN_PIN>",
                },
            },
            "value_policy": (
                "runtime loader verifies installed literals; joint stack binds only "
                "pin symbols and logical roles to avoid post-target self-reference"
            ),
        },
    },
    "target_ledger_identity_policy": {
        "biological_key_schema": "gfmcd.r5.biological_target_identity/1",
        "biological_key_fields": [
            "schema",
            "program",
            "stage",
            "checkpoint_sha256",
            "target_gene_ids_sha256",
            "ordered_rows_sha256",
        ],
        "method_context_only_fields": [
            "construction_sha256",
            "endpoint_policy",
            "r5_stack_fingerprint_sha256",
            "stage_preflight_artifact_sha256",
            "stage_preflight_manifest_sha256",
            "upstream_package_chain_sha256",
        ],
        "method_or_package_changes_may_reset_spend": False,
    },
    "r5b_basis_disposition": {
        "schema": "gfmcd.r5b.local_writer_basis_disposition/2",
        "must_be_resolved_before_chr8_chr9_unseal": True,
        "artifact_and_manifest_required_for_every_status": True,
        "prospective_package_pins": {
            "artifact_sha256": {
                "pin_symbol": "FROZEN_R5B_DISPOSITION_ARTIFACT_SHA256",
                "value": "<FROZEN_PIN>",
            },
            "manifest_sha256": {
                "pin_symbol": "FROZEN_R5B_DISPOSITION_MANIFEST_SHA256",
                "value": "<FROZEN_PIN>",
            },
        },
        "statuses": {
            "VALIDATED_BASIS": {
                "runtime_bundle_required": True,
                "selected_basis_fields_required": True,
                "r5c_low_rank_C2_enabled": True,
                "r5c_dense_C1_C4_enabled": True,
            },
            "NO_VALIDATED_BASIS_AT_FROZEN_THRESHOLDS": {
                "runtime_bundle_required": False,
                "selected_basis_fields_required": False,
                "r5c_low_rank_C2_enabled": False,
                "r5c_dense_C1_C4_enabled": True,
                "blocks_r5c_primary_paths": False,
            },
            "BASIS_INSTRUMENT_UNRESOLVED": {
                "runtime_bundle_required": False,
                "selected_basis_fields_required": False,
                "r5c_low_rank_C2_enabled": False,
                "r5c_dense_C1_C4_enabled": True,
                "blocks_r5c_primary_paths": False,
            },
        },
        "r5c_target_scores_r5b_on_chr8_chr9": False,
    },
    # Preflight/method freeze never opens chr8/chr9.  Only the post-freeze target chain
    # may consume the authenticated R1c handoff under a stage-specific receipt.
    "sealed_chr8_chr9_opened": False,
    "sealed_chr8_chr9_access_scope": {
        "R1c_R5a_R5b_R5c_method_freeze": "not opened",
        "R5a_R5c_preflight_rows": (
            "authenticated R1c val chr11/16/19 cohort; never residual buckets"
        ),
        "R5b_method_rows": "residual bucket 0 derivation and bucket 1 validation",
        "R1c_behavior_target": "sealed target behavior read after all method freezes",
        "R5a_R5c_targets": (
            "authenticated handoff materialization only after each stage receipt"
        ),
        "R5b": "no chr8/chr9 target stage",
    },
    "scientific_entry": {
        "source": "exact committed R3d target artifact plus sibling manifest",
        "decision_path": "L3H1_reused_component",
        "required_classification": "SUPPORTED",
        "path_b_role": "candidate_generation_only",
        "path_b_can_gate_entry": False,
        "r3c_role": "exclusion_and_provenance_only",
    },
    "required_behavior_predecessor": {
        "schema": R1C_BEHAVIOR_SCHEMA,
        "runner_path": R1C_RUNNER_PATH,
        "protocol_path": R1C_PROTOCOL_PATH,
        "split": "chr8/chr9 sealed target",
        "endpoint": "single masked acceptor G hard margin only",
        "must_pass_before_r5a_target": True,
        "must_not_gate_r5a_preflight_or_r5b_basis_freeze": True,
        "attempt_schema": R1C_TARGET_ATTEMPT_SCHEMA,
        "completion_schema": R1C_TARGET_COMPLETION_SCHEMA,
        "biological_target_identity_schema": (
            R1C_BIOLOGICAL_TARGET_IDENTITY_SCHEMA
        ),
        "preflight_verifier": "verify_successor_preflight_package",
        "preflight_verifier_schema": "gfmcd.r1c.verified_preflight_handoff/1",
        "target_verifier": "verify_successor_target_package",
        "target_verifier_schema": "gfmcd.r1c.verified_target_handoff/1",
        "target_requires_exact_predecessor_preflight_pair": True,
    },
    "handoff_schemas": {
        "r3c_provenance": R3C_PROVENANCE_SCHEMA,
        "scientific_entry": R3D_ENTRY_SCHEMA,
        "R1c_behavior": R1C_BEHAVIOR_SCHEMA,
        "R1c_target_attempt": R1C_TARGET_ATTEMPT_SCHEMA,
        "R1c_target_completion": R1C_TARGET_COMPLETION_SCHEMA,
        "R1c_biological_target_identity": R1C_BIOLOGICAL_TARGET_IDENTITY_SCHEMA,
        "entry": R5_ENTRY_SCHEMA,
        "exclusion_registry": R5_EXCLUSION_REGISTRY_SCHEMA,
        "cohort": R5_COHORT_SCHEMA,
        "preflight_cohort_handoff": R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
        "preflight_row_materialization": R5_PREFLIGHT_ROW_MATERIALIZATION_SCHEMA,
        "capstone_cohort_handoff": R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
        "capstone_row_materialization": R5_CAPSTONE_ROW_MATERIALIZATION_SCHEMA,
        "verified_R1c_target_common_legs": R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA,
        "r5b_runtime": "gfmcd.r5b.r5c_runtime_contract/1",
    },
}

# Public executable registries whose contents decide scanned families, paths, controls,
# calibration cases, or handoff structure.  Their defining AST nodes are recorded below;
# this makes the joint payload explicitly auditable without importing the runners (which
# would create circular imports and could accidentally execute module setup).
EXECUTABLE_REGISTRY_SYMBOLS = {
    R5_RUNNER_PATHS[0]: (
        "family_row_names",
        "named_path_specs",
        "classifier_calibration",
        "PATH_ORDER",
        "FAMILY_ORDER",
    ),
    R5_RUNNER_PATHS[1]: (
        "causal_primitive_names",
        "causal_family_names",
        "causal_boundary_edge_registry",
        "causal_requirement_registry",
        "prospective_calibration_trial_count",
        "RANK_GRID",
    ),
    R5_RUNNER_PATHS[2]: (
        "family_row_names",
        "path_registry",
        "random_ensemble_registry",
        "calibration_registry",
        "CAPABILITY_GATES",
        "FAMILY_NAMES",
    ),
}

_SCHEMA_TOKEN = re.compile(
    r"[\"'](gfmcd\.(?:r1c|r3c|r3d|r5)[A-Za-z0-9_.-]*/\d+)[\"']"
)

R5_EXECUTION_POLICY = {
    "device": "mps",
    "required_absent_environment_variables": ["PYTORCH_ENABLE_MPS_FALLBACK"],
}


class R5StackBindingError(ValueError):
    """Raised when a serialized R5 stack identity is absent or differs."""


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_repo() -> Path:
    # src/gfmcd/splicing/r5_stack_binding.py -> repository root
    return Path(__file__).resolve().parents[3]


def stable_runtime_identity() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = package_version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "packages": packages,
    }


def normalize_freeze_pins(text: str) -> tuple[str, tuple[str, ...]]:
    """Normalize only RHS values of top-level freeze assignments.

    Replacing the value node instead of the whole assignment preserves annotations,
    whitespace, and comments byte-for-byte.  Multiline literal values are still replaced
    as one span.
    """

    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))
    replacements: list[tuple[int, int, str, str]] = []
    for node in tree.body:
        name: str | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
        if name is None or re.fullmatch(r"FROZEN_[A-Z0-9_]+", name) is None:
            continue
        value = node.value
        if value is None:
            raise R5StackBindingError(f"freeze pin {name} has no assignment value")
        start = line_offsets[value.lineno - 1] + value.col_offset
        end = line_offsets[value.end_lineno - 1] + value.end_col_offset
        replacements.append((start, end, "<FROZEN_PIN>", name))
    normalized = text
    for start, end, replacement, _name in reversed(replacements):
        normalized = normalized[:start] + replacement + normalized[end:]
    names = tuple(name for _start, _end, _replacement, name in replacements)
    return normalized, names


def normalized_literal_pin_source_sha256(
    path: Path,
    *,
    permitted_pin_names: Sequence[str],
) -> str:
    """Hash source while normalizing one explicit set of top-level literal pins.

    This stricter helper is for source modules that are also hashed directly by a
    prospective instrument.  Unlike the joint-stack normalizer, it cannot silently
    broaden its normalization surface: every permitted name must occur exactly once as
    a simple top-level assignment with a literal RHS.  All other source bytes remain
    binding.
    """

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise R5StackBindingError(
            f"could not parse literal-pin source {path}"
        ) from exc
    permitted = tuple(permitted_pin_names)
    if not permitted or len(set(permitted)) != len(permitted):
        raise R5StackBindingError("permitted literal-pin names must be unique and nonempty")
    permitted_set = set(permitted)
    lines = source.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))
    replacements: list[tuple[int, int, str]] = []
    counts = {name: 0 for name in permitted}
    for node in tree.body:
        name: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
            value = node.value
        elif isinstance(node, ast.Assign):
            relevant_targets = [
                target.id
                for target in node.targets
                if isinstance(target, ast.Name) and target.id in permitted_set
            ]
            if relevant_targets and len(node.targets) != 1:
                raise R5StackBindingError(
                    "permitted literal freeze pin must use one simple assignment: "
                    + relevant_targets[0]
                )
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                value = node.value
        if name not in permitted_set:
            continue
        counts[name] += 1
        if counts[name] != 1:
            raise R5StackBindingError(
                f"permitted literal freeze pin is assigned more than once: {name}"
            )
        if value is None:
            raise R5StackBindingError(
                f"permitted literal freeze pin has no assignment value: {name}"
            )
        try:
            ast.literal_eval(value)
        except (TypeError, ValueError, SyntaxError) as exc:
            raise R5StackBindingError(
                f"permitted freeze pin RHS is not literal: {name}"
            ) from exc
        start = line_offsets[value.lineno - 1] + value.col_offset
        end = line_offsets[value.end_lineno - 1] + value.end_col_offset
        replacements.append((start, end, "<FROZEN_LITERAL_PIN>"))
    missing = [name for name in permitted if counts[name] != 1]
    if missing:
        raise R5StackBindingError(
            "permitted literal freeze pin is missing: " + missing[0]
        )
    normalized = source
    for start, end, replacement in reversed(replacements):
        normalized = normalized[:start] + replacement + normalized[end:]
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalized_source_record(path: Path, *, repo: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    normalized, pins = normalize_freeze_pins(text)
    relative = str(path.resolve().relative_to(repo.resolve()))
    if relative in (
        *R5_RUNNER_PATHS,
        R1C_RUNNER_PATH,
        R3C_RUNNER_PATH,
        R3D_RUNNER_PATH,
    ) and not pins:
        raise R5StackBindingError(f"expected freeze pins in bound runner: {relative}")
    return {
        "relative_path": relative,
        # The exact on-disk size cannot enter the prospective fingerprint: replacing
        # ``None`` with a reviewed literal pin generally changes it.  Exact source sizes
        # and hashes are separately recorded in each execution manifest.
        "normalized_size_bytes": len(normalized.encode("utf-8")),
        "normalization": (
            "all and only top-level FROZEN_* assignment values"
            if pins
            else "exact bytes"
        ),
        "normalized_pin_names": list(pins),
        "sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
    }


def _module_path(module: str, *, repo: Path) -> Path | None:
    if not module.startswith("gfmcd"):
        return None
    base = repo / "src" / Path(*module.split("."))
    file_candidate = base.with_suffix(".py")
    package_candidate = base / "__init__.py"
    if file_candidate.is_file():
        return file_candidate.resolve()
    if package_candidate.is_file():
        return package_candidate.resolve()
    return None


def _parent_package_inits(path: Path, *, repo: Path) -> tuple[Path, ...]:
    src = (repo / "src").resolve()
    try:
        relative = path.resolve().relative_to(src)
    except ValueError:
        return ()
    parts = relative.parts[:-1]
    output: list[Path] = []
    for depth in range(1, len(parts) + 1):
        candidate = src.joinpath(*parts[:depth], "__init__.py")
        if candidate.is_file():
            output.append(candidate.resolve())
    return tuple(output)


def _local_imports(path: Path, *, repo: Path) -> tuple[Path, ...]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise R5StackBindingError(f"could not parse bound source {path}") from exc
    found: set[Path] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
            # ``from gfmcd.package import helper_module`` executes the submodule when
            # that name resolves to a local module.  Function/class aliases simply do
            # not resolve here and are harmless.
            modules.extend(
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name != "*"
            )
        for module in modules:
            candidate = _module_path(module, repo=repo)
            if candidate is not None:
                found.add(candidate)
                found.update(_parent_package_inits(candidate, repo=repo))
    return tuple(sorted(found))


def executable_registry_records(*, repo: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for relative, requested in EXECUTABLE_REGISTRY_SYMBOLS.items():
        path = repo / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        nodes: dict[str, ast.AST] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nodes[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        nodes[target.id] = node
        missing = [name for name in requested if name not in nodes]
        if missing:
            raise R5StackBindingError(
                f"registered executable symbol is missing from {relative}: {missing[0]}"
            )
        output[relative] = {
            name: {
                "node_type": type(nodes[name]).__name__,
                "ast_sha256": hashlib.sha256(
                    ast.dump(nodes[name], annotate_fields=True, include_attributes=False).encode(
                        "utf-8"
                    )
                ).hexdigest(),
            }
            for name in requested
        }
    return output


def local_python_closure(
    roots: Sequence[Path], *, repo: Path | None = None
) -> tuple[Path, ...]:
    root = (repo or _default_repo()).resolve()
    pending = [path.resolve() for path in roots]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not path.is_file():
            raise R5StackBindingError(f"bound source is missing: {path}")
        seen.add(path)
        for imported in _local_imports(path, repo=root):
            if imported not in seen:
                pending.append(imported)
    return tuple(sorted(seen, key=lambda path: str(path.relative_to(root))))


def build_r5_stack_binding(*, repo: Path | None = None) -> dict[str, Any]:
    root = (repo or _default_repo()).resolve()
    runner_roots = tuple(
        root / path
        for path in (
            *R5_RUNNER_PATHS,
            R3D_RUNNER_PATH,
            R3C_RUNNER_PATH,
            R1C_RUNNER_PATH,
        )
    )
    protocols = tuple(root / path for path in R5_PROTOCOL_PATHS)
    predecessor_protocols = (root / R1C_PROTOCOL_PATH,)
    runtime_files = tuple(root / path for path in REPOSITORY_RUNTIME_PATHS)
    missing = [
        str(path)
        for path in (*runner_roots, *protocols, *predecessor_protocols, *runtime_files)
        if not path.is_file()
    ]
    if missing:
        raise R5StackBindingError(f"R5 stack source is missing: {missing[0]}")

    closure = local_python_closure(runner_roots, repo=root)
    source_records = {
        str(path.relative_to(root)): normalized_source_record(path, repo=root)
        for path in closure
    }
    protocol_records = {
        str(path.relative_to(root)): {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in protocols
    }
    predecessor_protocol_records = {
        str(path.relative_to(root)): {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in predecessor_protocols
    }
    schema_tokens = sorted(
        {
            token
            for path in closure
            for token in _SCHEMA_TOKEN.findall(path.read_text(encoding="utf-8"))
        }
    )
    executable_registries = executable_registry_records(repo=root)
    repository_runtime_files = {
        str(path.relative_to(root)): {
            "relative_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in runtime_files
    }
    return {
        "schema": STACK_SCHEMA,
        "contract": STACK_CONTRACT,
        "runner_roots": list(R5_RUNNER_PATHS),
        "scientific_entry_runner": R3D_RUNNER_PATH,
        # Compatibility alias retained for existing runner payloads.
        "dynamic_entry_runner": R3D_RUNNER_PATH,
        "exclusion_provenance_runner": R3C_RUNNER_PATH,
        "required_behavior_predecessor_runner": R1C_RUNNER_PATH,
        "behavior_predecessor_source_present_and_bound": True,
        "protocols": protocol_records,
        "behavior_predecessor_protocol": {
            "required_path": R1C_PROTOCOL_PATH,
            "present_and_bound": True,
            "records": predecessor_protocol_records,
        },
        "local_python_closure": {
            "discovery": (
                "recursive AST traversal of repository-local gfmcd imports plus package "
                "initializers; R3d scientific-entry, R3c provenance, and R1c behavior "
                "runners are explicit roots"
            ),
            "n_files": len(source_records),
            "files": source_records,
            "ordered_paths_sha256": canonical_sha256(list(source_records)),
        },
        "schema_registry": schema_tokens,
        "schema_registry_sha256": canonical_sha256(schema_tokens),
        "executable_registries": executable_registries,
        "executable_registries_sha256": canonical_sha256(executable_registries),
        "upstream_packages": json.loads(json.dumps(UPSTREAM_PACKAGE_IDENTITIES)),
        "source_data_identity": json.loads(json.dumps(PINNED_R5_SOURCE_IDENTITY)),
        "repository_runtime_files": repository_runtime_files,
        "upstream_r3d_repository_runtime_identity": json.loads(
            json.dumps(PINNED_R5_REPOSITORY_RUNTIME_IDENTITY)
        ),
        "execution_policy": dict(R5_EXECUTION_POLICY),
        "runtime_identity": stable_runtime_identity(),
    }


def validate_r5_runtime_policy(
    *, device: str, environ: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Enforce the prospective strict-MPS execution policy before model access."""

    if str(device) != R5_EXECUTION_POLICY["device"]:
        raise R5StackBindingError("R5 execution device must be mps")
    environment = os.environ if environ is None else environ
    forbidden = R5_EXECUTION_POLICY["required_absent_environment_variables"]
    present = [name for name in forbidden if name in environment]
    if present:
        raise R5StackBindingError(
            "R5 strict MPS fallback variable must be absent: " + ", ".join(present)
        )
    return {
        "schema": STACK_SCHEMA,
        "verified": True,
        "device": "mps",
        "absent_environment_variables": list(forbidden),
    }


def r5_stack_fingerprint(binding: Mapping[str, Any] | None = None) -> str:
    return canonical_sha256(build_r5_stack_binding() if binding is None else dict(binding))


def validate_r5_stack_binding(
    binding: Mapping[str, Any],
    fingerprint: str,
    *,
    current: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observed = dict(binding)
    expected = dict(current) if current is not None else build_r5_stack_binding()
    if observed.get("schema") != STACK_SCHEMA:
        raise R5StackBindingError("R5 joint stack binding schema differs")
    if canonical_sha256(observed) != fingerprint:
        raise R5StackBindingError("R5 joint stack fingerprint does not bind its payload")
    expected_fingerprint = canonical_sha256(expected)
    if observed != expected or fingerprint != expected_fingerprint:
        raise R5StackBindingError("R5 joint stack differs from the current frozen stack")
    return {
        "schema": STACK_SCHEMA,
        "verified": True,
        STACK_FINGERPRINT_FIELD: expected_fingerprint,
    }


def validate_serialized_r5_stack(
    record: Mapping[str, Any], *, current: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    binding = record.get(STACK_BINDING_FIELD)
    fingerprint = record.get(STACK_FINGERPRINT_FIELD)
    if not isinstance(binding, Mapping) or not isinstance(fingerprint, str):
        raise R5StackBindingError("serialized R5 joint stack identity is missing")
    return validate_r5_stack_binding(binding, fingerprint, current=current)


def validate_r3c_target_health_contract(
    record: Mapping[str, Any],
    *,
    required_supported_path: str = "L3H1_reused_component",
) -> dict[str, Any]:
    """Validate a historical R3c health record for provenance replay only.

    This compatibility validator cannot authorize R5 scientific entry.  R5 v2 uses
    :func:`load_r3d_entry_package` and exact R3d Path A for that decision.
    """

    if required_supported_path not in R3C_PATH_NAMES:
        raise R5StackBindingError("required R3c path is outside the frozen registry")
    if record.get("instrument_revision") != R3C_INSTRUMENT_REVISION:
        raise R5StackBindingError("R3c monotone-health instrument revision differs")
    paths = record.get("paths")
    authorization = record.get("path_health_authorization")
    if not isinstance(paths, Mapping) or tuple(paths) != R3C_PATH_NAMES:
        raise R5StackBindingError("R3c target lacks the ordered seven-path registry")
    if not isinstance(authorization, Mapping):
        raise R5StackBindingError("R3c target health authorization is missing")
    if (
        authorization.get("schema") != R3C_TARGET_HEALTH_SCHEMA
        or authorization.get("rule") != R3C_TARGET_HEALTH_RULE
        or authorization.get("can_only_shrink_from_preflight") is not True
    ):
        raise R5StackBindingError("R3c target health authorization policy differs")

    def ordered_subset(field: str) -> list[str]:
        value = authorization.get(field)
        if not isinstance(value, list):
            raise R5StackBindingError(f"R3c target health {field} is missing")
        if value != [name for name in R3C_PATH_NAMES if name in value]:
            raise R5StackBindingError(
                f"R3c target health {field} is not an ordered unique subset"
            )
        return value

    preflight_authorized = ordered_subset("preflight_authorized_paths")
    preflight_suppressed = ordered_subset("preflight_suppressed_paths")
    current_healthy = ordered_subset("current_healthy_paths")
    effective_authorized = ordered_subset("effective_authorized_paths")
    effective_suppressed = ordered_subset("effective_suppressed_paths")
    if (
        preflight_suppressed
        != [name for name in R3C_PATH_NAMES if name not in preflight_authorized]
        or effective_authorized
        != [
            name
            for name in R3C_PATH_NAMES
            if name in preflight_authorized and name in current_healthy
        ]
        or effective_suppressed
        != [name for name in R3C_PATH_NAMES if name not in effective_authorized]
    ):
        raise R5StackBindingError("R3c target health sets do not implement intersection")
    if any(
        name in preflight_authorized
        or name not in preflight_suppressed
        or name in effective_authorized
        or name not in effective_suppressed
        for name in R3C_HISTORICAL_HEALTH_SUPPRESSED_PATHS
    ):
        raise R5StackBindingError(
            "R3c target revived a path suppressed by the reviewed v1 health ceiling"
        )

    for name in R3C_PATH_NAMES:
        path = paths[name]
        if not isinstance(path, Mapping) or not isinstance(path.get("health"), Mapping):
            raise R5StackBindingError(f"R3c target path health is missing for {name}")
        health = path["health"]
        preflight_ok = name in preflight_authorized
        current_ok = name in current_healthy
        effective_ok = preflight_ok and current_ok
        if (
            health.get("preflight_health_ok") is not preflight_ok
            or health.get("current_health_ok") is not current_ok
            or health.get("ok") is not effective_ok
            or health.get("authorization_rule") != R3C_PATH_HEALTH_RULE
            or (
                not effective_ok
                and path.get("classification") != "HEALTH_FAILED"
            )
        ):
            raise R5StackBindingError(
                f"R3c target path decision bypasses health authorization for {name}"
            )

    required = paths[required_supported_path]
    if (
        required_supported_path not in effective_authorized
        or required.get("classification") != "SUPPORTED"
    ):
        raise R5StackBindingError("required R3c path is not health-authorized support")
    family = record.get("family")
    calibration = (
        family.get("production_path_preflight")
        if isinstance(family, Mapping)
        else None
    )
    if (
        not isinstance(calibration, Mapping)
        or calibration.get("all_health_eligible_path_plants_powered") is not True
        or calibration.get("health_eligible_paths") != list(R3C_PATH_NAMES)
    ):
        raise R5StackBindingError(
            "R3c calibration does not power all seven health-enabled paths"
        )
    return {
        "schema": R3C_TARGET_HEALTH_SCHEMA,
        "verified": True,
        "scientific_entry_authorized": False,
        "required_supported_path": required_supported_path,
        "effective_authorized_paths": effective_authorized,
    }


def validate_r3c_target_manifest_contract(
    record: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Authenticate a historical R3c target manifest for provenance replay only."""

    config = manifest.get("config")
    construction = record.get("construction")
    git_identity = record.get("git_authorization_identity")
    if (
        manifest.get("experiment_id")
        != "splicebert_acceptor_coupling:r3c_acceptor_head_confirmation"
        or manifest.get("status") != "CONFIRMATORY"
        or manifest.get("git_dirty") is not False
        or not isinstance(config, Mapping)
        or not isinstance(construction, Mapping)
        or not isinstance(git_identity, Mapping)
        or config.get("stage") != "target"
        or config.get("gene_bucket") != 1
        or config.get("instrument_fingerprint_sha256")
        != record.get("instrument_fingerprint_sha256")
        or config.get("construction_sha256")
        != construction.get("ordered_rows_sha256")
        or config.get("input_snapshot_sha256")
        != record.get("input_snapshot_sha256")
        or manifest.get("git_commit") != git_identity.get("commit")
        or git_identity.get("dirty") is not False
    ):
        raise R5StackBindingError("R3c target sibling manifest context differs")
    return {
        "verified": True,
        "scientific_entry_authorized": False,
        "experiment_id": manifest["experiment_id"],
        "status": manifest["status"],
    }
