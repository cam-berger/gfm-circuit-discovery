"""Authenticated cohort construction for the R5 inverse-acceptor program.

R5b alone uses the outcome-blind two-bucket residual-validation reservation while
deriving and validating its frozen union basis.  R5a and R5c preflight instead consume
the exact authenticated, nonsealed 400-row R1c validation cohort; their targets consume
the separately authenticated sealed 400-row R1c chr8/chr9 cohort.  The three roles are
kept explicit so a residual bucket can never become a biological capstone by accident.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from gfmcd.splicing.coupling import (
    MOTIF_CLEAN_DONOR_ALT,
    CouplingExample,
    planted_donor_motifs,
    substitute,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout
from gfmcd.splicing.r5_upstream_provenance import (
    HISTORICAL_R3_GENE_COUNT,
    HISTORICAL_R3_GENE_IDS_SHA256,
    PINNED_R5_SOURCE_IDENTITY,
    R3C_FIXED_GENE_COUNT,
    R3C_FIXED_GENE_IDS_SHA256,
    R3D_REGISTRY_AND_TARGET_UNION_GENE_IDS_SHA256,
    R3D_REGISTRY_GENE_COUNT,
    R3D_REGISTRY_GENE_IDS_SHA256,
    R3D_TARGET_GENE_COUNT,
    R3D_TARGET_GENE_IDS_SHA256,
    R5_PRIOR_EXPOSURE_UNION_GENE_COUNT,
    R5_PRIOR_EXPOSURE_UNION_GENE_IDS_SHA256,
)


R5_SALT = "r5-acceptor-mechanism-v1"
R5_ADJACENT_SALT = "r5-adjacent-bank-v2"
R5_VALIDATION_CHROMOSOMES = ("chr11", "chr16", "chr19")
R5_DONOR_PSEUDO_CALIPER = 40
R5_RESIDUAL_VALIDATION_ROWS_PER_BUCKET = 128
# Compatibility name for existing R5a/R5b code.  It is not an R5c sample-size contract.
R5_ROWS_PER_BUCKET = R5_RESIDUAL_VALIDATION_ROWS_PER_BUCKET
R5_COHORT_SCHEMA = "gfmcd.r5.acceptor_mechanism_cohort/3"
R5_CAPSTONE_COHORT_HANDOFF_SCHEMA = "gfmcd.r5.capstone_cohort_handoff/1"
R5_CAPSTONE_ROW_MATERIALIZATION_SCHEMA = (
    "gfmcd.r5.capstone_row_materialization/1"
)
R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA = (
    "gfmcd.r5.verified_r1c_target_common_legs/1"
)
R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA = "gfmcd.r5.preflight_cohort_handoff/1"
R5_PREFLIGHT_ROW_MATERIALIZATION_SCHEMA = (
    "gfmcd.r5.preflight_row_materialization/1"
)
R1C_TARGET_SCHEMA = "gfmcd.r1c.reverse_behavior/3"
R1C_BIOLOGICAL_TARGET_IDENTITY_SCHEMA = (
    "gfmcd.r1c.biological_target_identity/1"
)
R1C_TARGET_ATTEMPT_SCHEMA = "gfmcd.r1c.target_attempt/2"
R1C_TARGET_COMPLETION_SCHEMA = "gfmcd.r1c.target_completion/1"
R1C_TARGET_ATTEMPT_STATUS = "AUTHORIZED_SEALED_OUTCOMES_MAY_BE_SPENT"
R1C_TARGET_COMPLETION_STATUS = "TARGET_PUBLICATION_COMPLETE"
R1C_CONFIRMED_BEHAVIOR_VERDICTS = (
    "CONFIRMED_PAIRED_DONOR_ACCEPTOR_BEHAVIOR",
    "CONFIRMED_DOWNSTREAM_ACCEPTOR_PREFERENCE",
)
R1C_PRIMARY_FAMILY_NAMES = ("Y_PD", "Y_PN", "S_P", "I")
R1C_CELL_NAMES = ("Y_PD", "Y_PN", "Y_XD", "Y_XN")
R1C_CONSTRUCTION_VERSION = "r1c-acceptor-readout-by-donor-edit-boundary-inference-v4"
R1C_COHORT_ORDER_VERSION = "r1c-single-g-primary-family-v2"
R1C_MATCHED_DECOY_DONOR_CALIPER = 40
R1C_DYAD_DISTANCE_CALIPER = 20
R1C_CLAIM_BOUNDARY = (
    "Behavior rung only. I is a downstream-versus-exon-mate preference contrast on "
    "outcome-blind distance-matched dyads; X cells are descriptive and need not be "
    "equivalent. No component, mediation, semantic, biological-mechanism, or "
    "circuit-completeness conclusion is licensed."
)
# Prospective literal pins.  Each package stays unset until its R1c stage exists and is
# reviewed; callers must fail before resolving any supplied path while a pin is absent.
FROZEN_R1C_PREFLIGHT_ARTIFACT_LOGICAL_NAME: str | None = None
FROZEN_R1C_PREFLIGHT_ARTIFACT_SIZE_BYTES: int | None = None
FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256: str | None = None
FROZEN_R1C_PREFLIGHT_MANIFEST_LOGICAL_NAME: str | None = None
FROZEN_R1C_PREFLIGHT_MANIFEST_SIZE_BYTES: int | None = None
FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256: str | None = None
FROZEN_R1C_TARGET_ARTIFACT_LOGICAL_NAME: str | None = None
FROZEN_R1C_TARGET_ARTIFACT_SIZE_BYTES: int | None = None
FROZEN_R1C_TARGET_ARTIFACT_SHA256: str | None = None
FROZEN_R1C_TARGET_MANIFEST_LOGICAL_NAME: str | None = None
FROZEN_R1C_TARGET_MANIFEST_SIZE_BYTES: int | None = None
FROZEN_R1C_TARGET_MANIFEST_SHA256: str | None = None
FROZEN_R1C_TARGET_ATTEMPT_LOGICAL_NAME: str | None = None
FROZEN_R1C_TARGET_ATTEMPT_SIZE_BYTES: int | None = None
FROZEN_R1C_TARGET_ATTEMPT_SHA256: str | None = None
FROZEN_R1C_TARGET_COMPLETION_LOGICAL_NAME: str | None = None
FROZEN_R1C_TARGET_COMPLETION_SIZE_BYTES: int | None = None
FROZEN_R1C_TARGET_COMPLETION_SHA256: str | None = None
# Compatibility aliases.  Installing reviewed values changes only the normalized
# ``FROZEN_*`` assignments above, never these source expressions or the joint stack.
R1C_TARGET_ARTIFACT_LOGICAL_NAME = FROZEN_R1C_TARGET_ARTIFACT_LOGICAL_NAME
R1C_TARGET_ARTIFACT_SIZE_BYTES = FROZEN_R1C_TARGET_ARTIFACT_SIZE_BYTES
R1C_TARGET_ARTIFACT_SHA256 = FROZEN_R1C_TARGET_ARTIFACT_SHA256
R1C_TARGET_MANIFEST_LOGICAL_NAME = FROZEN_R1C_TARGET_MANIFEST_LOGICAL_NAME
R1C_TARGET_MANIFEST_SIZE_BYTES = FROZEN_R1C_TARGET_MANIFEST_SIZE_BYTES
R1C_TARGET_MANIFEST_SHA256 = FROZEN_R1C_TARGET_MANIFEST_SHA256
R1C_TARGET_ATTEMPT_LOGICAL_NAME = FROZEN_R1C_TARGET_ATTEMPT_LOGICAL_NAME
R1C_TARGET_ATTEMPT_SIZE_BYTES = FROZEN_R1C_TARGET_ATTEMPT_SIZE_BYTES
R1C_TARGET_ATTEMPT_SHA256 = FROZEN_R1C_TARGET_ATTEMPT_SHA256
R1C_TARGET_COMPLETION_LOGICAL_NAME = FROZEN_R1C_TARGET_COMPLETION_LOGICAL_NAME
R1C_TARGET_COMPLETION_SIZE_BYTES = FROZEN_R1C_TARGET_COMPLETION_SIZE_BYTES
R1C_TARGET_COMPLETION_SHA256 = FROZEN_R1C_TARGET_COMPLETION_SHA256
R5_RESIDUAL_VALIDATION_COHORT_ROLE = "R5b_method_freeze_residual_validation"
R5_RESIDUAL_VALIDATION_STAGE_CONSUMERS = ("R5b",)
R5_RESIDUAL_VALIDATION_BUCKET_ROLES = {
    "0": "R5b_derivation",
    "1": "R5b_validation_and_union_refit",
}
R5A_R5C_PREFLIGHT_COHORT_ROLE = (
    "authenticated_R1c_nonsealed_validation_behavior_cohort"
)
R5C_REQUIRED_COHORT_ROLE = "authenticated_R1c_sealed_chr8_chr9_behavior_cohort"
R1C_ROW_POSITION_FIELDS = (
    "prior_real_acceptor",
    "paired_donor",
    "partner_acceptor",
    "matched_decoy_donor",
    "matched_decoy_donor_distance_mismatch",
    "dyad_distance_mismatch",
)
R5_PRIOR_EXPOSURE_PAIRWISE_INTERSECTIONS = {
    "historical_r3__r3c_fixed_cohorts": 0,
    "historical_r3__r3d_direct_exposure_registry": 0,
    "historical_r3__r3d_target": 0,
    "r3c_fixed_cohorts__r3d_direct_exposure_registry": 0,
    "r3c_fixed_cohorts__r3d_target": 0,
    "r3d_direct_exposure_registry__r3d_target": 0,
}
R5_EXPECTED_SOURCE_IDENTITY = {
    key: dict(value) for key, value in PINNED_R5_SOURCE_IDENTITY.items()
}
R5_EXPECTED_CENSUS_IDENTITY = {
    "schema": R5_COHORT_SCHEMA,
    "cohort_role": R5_RESIDUAL_VALIDATION_COHORT_ROLE,
    "authorized_stage_consumers": list(R5_RESIDUAL_VALIDATION_STAGE_CONSUMERS),
    "bucket_roles": dict(R5_RESIDUAL_VALIDATION_BUCKET_ROLES),
    "r5a_r5c_preflight_authorized": False,
    "r5c_capstone_authorized": False,
    "r5c_required_cohort_role": R5C_REQUIRED_COHORT_ROLE,
    "input_rows": 6064,
    "historical_r3_gene_count": HISTORICAL_R3_GENE_COUNT,
    "historical_r3_gene_ids_sha256": HISTORICAL_R3_GENE_IDS_SHA256,
    "r3c_gene_count": R3C_FIXED_GENE_COUNT,
    "r3c_gene_ids_sha256": R3C_FIXED_GENE_IDS_SHA256,
    "r3d_direct_exposure_registry_gene_count": R3D_REGISTRY_GENE_COUNT,
    "r3d_direct_exposure_registry_gene_ids_sha256": R3D_REGISTRY_GENE_IDS_SHA256,
    "r3d_target_gene_count": R3D_TARGET_GENE_COUNT,
    "r3d_target_gene_ids_sha256": R3D_TARGET_GENE_IDS_SHA256,
    "r3d_registry_and_target_union_gene_count": (
        R3D_REGISTRY_GENE_COUNT + R3D_TARGET_GENE_COUNT
    ),
    "r3d_registry_and_target_union_gene_ids_sha256": (
        R3D_REGISTRY_AND_TARGET_UNION_GENE_IDS_SHA256
    ),
    "prior_exposure_sets_pairwise_disjoint": True,
    "prior_exposure_pairwise_intersection_counts": dict(
        R5_PRIOR_EXPOSURE_PAIRWISE_INTERSECTIONS
    ),
    "prior_exposure_union_gene_count": R5_PRIOR_EXPOSURE_UNION_GENE_COUNT,
    "prior_exposure_union_gene_ids_sha256": (
        R5_PRIOR_EXPOSURE_UNION_GENE_IDS_SHA256
    ),
    # Backward-readable aliases now bind the complete four-source union.
    "combined_excluded_gene_count": R5_PRIOR_EXPOSURE_UNION_GENE_COUNT,
    "combined_excluded_gene_ids_sha256": R5_PRIOR_EXPOSURE_UNION_GENE_IDS_SHA256,
    "r3d_validation_row_overlap_by_source": {
        "r3d_direct_exposure_registry": 0,
        "r3d_target": 0,
    },
    "global_gene_freshness_claimed": False,
    "exclusions": {
        "excluded_gene": 5648,
        "pseudo_donor_outside_40nt_caliper": 50,
    },
    "eligible_rows": 366,
    "eligible_genes": 272,
    "available_genes_by_bucket": {"0": 142, "1": 130},
    "selected_by_bucket": {
        "0": {
            "rows": 128,
            "genes": 128,
            "both_strands": True,
            "gene_ids_sha256": (
                "af644dd66d8963a714f07a236fae6302a789044aeb1e52c0bb844e731328abe4"
            ),
            "ordered_rows_sha256": (
                "572289f8d2e8e6406e93596994b6bb5d3df04cd6571ff95bea85211ba4f358a9"
            ),
            "construction_sha256": (
                "3143a893894de702476145589652f3157001218ec24c1742f2b1cedd0eebe0fd"
            ),
        },
        "1": {
            "rows": 128,
            "genes": 128,
            "both_strands": True,
            "gene_ids_sha256": (
                "e6c3099298431c99b8afde686215b3287f3e0f63eedfe3b650db4d48fc13c84c"
            ),
            "ordered_rows_sha256": (
                "935d62284518b0ad72b398b2d7f41cb034015d63bb6617ca7e43a5b534ddbf95"
            ),
            "construction_sha256": (
                "7de99171bd678dada527b9411c350166cdc3a512020b4bdd18f1e28af5c08312"
            ),
        },
    },
}


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible object without whitespace or key-order ambiguity."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def r5_source_identity(gtf: str | Path, fasta: str | Path) -> dict[str, Any]:
    """Hash the exact annotation, reference, and FASTA index used by R5."""

    records: dict[str, Any] = {}
    fasta_path = Path(fasta).expanduser()
    values = (
        ("gtf", gtf),
        ("fasta", fasta_path),
        ("fasta_fai", Path(f"{fasta_path}.fai")),
    )
    for key, value in values:
        path = Path(value).expanduser().resolve(strict=True)
        records[key] = {
            "label": R5_EXPECTED_SOURCE_IDENTITY[key]["label"],
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    return records


def validate_expected_r5_source_identity(
    gtf: str | Path, fasta: str | Path
) -> dict[str, Any]:
    """Reject a source-compatible-looking census built from different source bytes."""

    observed = r5_source_identity(gtf, fasta)
    if observed != R5_EXPECTED_SOURCE_IDENTITY:
        raise RuntimeError(
            "R5 source identity drifted; the historical R3/R5 cohort requires the "
            f"pinned comprehensive GTF and hg38 bytes: observed={observed}, "
            f"expected={R5_EXPECTED_SOURCE_IDENTITY}"
        )
    return observed


class R5CapstoneCohortError(ValueError):
    """Raised when the prospective R1c-to-R5c cohort handoff differs."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _logical_file_identity(
    value: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if set(value) != {"logical_name", "size_bytes", "sha256"}:
        raise R5CapstoneCohortError(f"{label} identity fields differ")
    logical_name = value.get("logical_name")
    size_bytes = value.get("size_bytes")
    sha256 = value.get("sha256")
    if (
        not isinstance(logical_name, str)
        or not logical_name
        or Path(logical_name).name != logical_name
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes < 1
        or not _is_sha256(sha256)
    ):
        raise R5CapstoneCohortError(f"{label} identity is malformed")
    return {
        "logical_name": logical_name,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def r1c_serialized_row_identity(
    row: Mapping[str, Any],
    *,
    allowed_chromosomes: Sequence[str] = ("chr8", "chr9"),
) -> dict[str, Any]:
    """Extract one outcome-free R1c identity without reading sequence or genome bytes."""

    positions = row.get("positions")
    if not isinstance(positions, Mapping) or set(positions) != set(
        R1C_ROW_POSITION_FIELDS
    ):
        raise R5CapstoneCohortError("R1c serialized row coordinates differ")
    identity = {
        "example_id": row.get("example_id"),
        "gene_id": row.get("gene_id"),
        "chrom": row.get("chrom"),
        "strand": row.get("strand"),
        "positions": {name: positions[name] for name in R1C_ROW_POSITION_FIELDS},
    }
    if (
        not isinstance(identity["example_id"], str)
        or not identity["example_id"]
        or not isinstance(identity["gene_id"], str)
        or not identity["gene_id"]
        or identity["chrom"] not in set(map(str, allowed_chromosomes))
        or identity["strand"] not in {"+", "-"}
    ):
        raise R5CapstoneCohortError("R1c serialized row identity differs")
    for name in R1C_ROW_POSITION_FIELDS:
        coordinate = identity["positions"][name]
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, int)
            or coordinate < 0
        ):
            raise R5CapstoneCohortError(
                f"R1c serialized row coordinate {name} differs"
            )
    return identity


def _r1c_source_rows(
    record: Mapping[str, Any],
    *,
    allowed_chromosomes: Sequence[str] = ("chr8", "chr9"),
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = record.get("rows")
    if not isinstance(rows, list) or len(rows) != 400:
        raise R5CapstoneCohortError("R1c target must serialize exactly 400 rows")
    identities: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise R5CapstoneCohortError("R1c serialized row is malformed")
        identities.append(
            r1c_serialized_row_identity(
                row, allowed_chromosomes=allowed_chromosomes
            )
        )
    ordered_ids = [str(row["example_id"]) for row in identities]
    if len(set(ordered_ids)) != 400:
        raise R5CapstoneCohortError("R1c target row IDs must be unique")
    return identities, ordered_ids


def _r1c_ordered_partner_margin_levels(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Extract the completion-only common-leg replay identity without exporting it."""

    rows = record.get("rows")
    if not isinstance(rows, list) or len(rows) != 400:
        raise R5CapstoneCohortError("R1c target must serialize exactly 400 rows")
    output: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise R5CapstoneCohortError("R1c serialized row is malformed")
        levels = row.get("margin_levels")
        if not isinstance(levels, Mapping) or set(levels) != {
            "P_clean",
            "P_under_D",
            "P_under_N",
        }:
            raise R5CapstoneCohortError("R1c partner margin levels differ")
        normalized: dict[str, float] = {}
        for name in ("P_clean", "P_under_D", "P_under_N"):
            value = levels[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise R5CapstoneCohortError("R1c partner margin level is malformed")
            normalized[name] = float(value)
        output.append(
            {
                "example_id": row.get("example_id"),
                "gene_id": row.get("gene_id"),
                "margin_levels": normalized,
            }
        )
    return output


def build_r5_capstone_cohort_handoff(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    attempt: Mapping[str, Any],
    completion: Mapping[str, Any],
    *,
    artifact_identity: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    attempt_identity: Mapping[str, Any],
    completion_identity: Mapping[str, Any],
    preflight_artifact_identity: Mapping[str, Any],
    preflight_manifest_identity: Mapping[str, Any],
    successor_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Strip an authenticated R1c target to its exact outcome-free capstone rows.

    This semantic builder consumes only serialized JSON.  Production callers must use
    :func:`load_r1c_capstone_cohort_handoff`, which additionally requires reviewed literal
    artifact, manifest, attempt, and completion byte pins.
    """

    artifact = _logical_file_identity(artifact_identity, label="R1c target artifact")
    sibling_manifest = _logical_file_identity(
        manifest_identity, label="R1c target manifest"
    )
    target_attempt = _logical_file_identity(
        attempt_identity, label="R1c target-attempt receipt"
    )
    target_completion = _logical_file_identity(
        completion_identity, label="R1c target-completion receipt"
    )
    preflight_artifact = _logical_file_identity(
        preflight_artifact_identity, label="R1c preflight artifact"
    )
    preflight_manifest = _logical_file_identity(
        preflight_manifest_identity, label="R1c preflight manifest"
    )
    outputs = manifest.get("outputs")
    matching_outputs = (
        [
            item
            for item in outputs
            if isinstance(item, Mapping)
            and Path(str(item.get("path", ""))).name == artifact["logical_name"]
            and item.get("size_bytes") == artifact["size_bytes"]
            and item.get("sha256") == artifact["sha256"]
        ]
        if isinstance(outputs, list)
        else []
    )
    inputs = manifest.get("inputs")
    matching_attempt_inputs = (
        [
            item
            for item in inputs
            if isinstance(item, Mapping)
            and Path(str(item.get("path", ""))).name
            == target_attempt["logical_name"]
            and item.get("size_bytes") == target_attempt["size_bytes"]
            and item.get("sha256") == target_attempt["sha256"]
        ]
        if isinstance(inputs, list)
        else []
    )
    cohort_audit = record.get("cohort_audit")
    config = record.get("config")
    embedded_attempt = record.get("target_attempt")
    embedded_completion = record.get("target_completion")
    target_cohort_identity = record.get("target_cohort_identity")
    gates = record.get("gates")
    observed_paths = record.get("observed_decision_paths")
    gene_robustness = record.get("gene_robustness")
    instrument_fingerprint = record.get("instrument_fingerprint")
    attempt_payload = {key: value for key, value in attempt.items() if key != "attempt_id"}
    completion_payload = {
        key: value for key, value in completion.items() if key != "completion_id"
    }
    target_identity = attempt.get("target_identity")
    method_context = attempt.get("method_context")
    attested_packages = successor_attestation.get("package_identities")
    attested_preflight = successor_attestation.get("preflight_package")
    expected_target_identity_fields = {
        "schema",
        "program",
        "stage",
        "holdout_id",
        "target_chromosomes",
        "target_n",
        "checkpoint_sha256",
        "gtf_sha256",
        "fasta_sha256",
        "fasta_fai_sha256",
        "biological_target_key",
    }
    expected_attempt_fields = {
        "schema",
        "created_at_utc",
        "status",
        "biological_target_key",
        "target_identity",
        "method_context",
        "policy",
        "sealed_cohort_constructed_before_record",
        "model_loaded_before_record",
        "output_inspected_before_record",
        "attempt_id",
    }
    expected_method_context_fields = {
        "protocol_sha256",
        "instrument_fingerprint_sha256",
        "r5_stack_fingerprint_sha256",
        "checkpoint_bundle_sha256",
        "construction_contract_sha256",
        "endpoint_policy",
        "preflight_artifact",
        "preflight_manifest",
        "input_snapshot_sha256",
        "requested_artifact_logical_name",
        "requested_manifest_logical_name",
        "declared_completion_logical_name",
    }
    expected_embedded_attempt = (
        target_attempt
        | {
            "attempt_id": attempt.get("attempt_id"),
            "biological_target_key": attempt.get("biological_target_key"),
            "status": attempt.get("status"),
        }
    )
    if (
        record.get("schema") != R1C_TARGET_SCHEMA
        or record.get("stage") != "target"
        or record.get("split") != "test"
        or record.get("n") != 400
        or record.get("preconstruction_allowed_chromosomes") != ["chr8", "chr9"]
        or record.get("sealed_split_opened") is not True
        or record.get("all_gates_pass") is not True
        or record.get("verdict") not in set(R1C_CONFIRMED_BEHAVIOR_VERDICTS)
        or not isinstance(cohort_audit, Mapping)
        or cohort_audit.get("selected") != 400
        or cohort_audit.get("selection_reads_model_outputs") is not False
        or not isinstance(config, Mapping)
        or config.get("single_G_acceptor_endpoint") is not True
        or config.get("outcome_conditioning") is not None
        or config.get("construction_version") != R1C_CONSTRUCTION_VERSION
        or config.get("cohort_order_version") != R1C_COHORT_ORDER_VERSION
        or config.get("matched_decoy_donor_caliper")
        != R1C_MATCHED_DECOY_DONOR_CALIPER
        or config.get("dyad_distance_caliper") != R1C_DYAD_DISTANCE_CALIPER
        or config.get("primary_family_names") != list(R1C_PRIMARY_FAMILY_NAMES)
        or config.get("cell_names") != list(R1C_CELL_NAMES)
        or record.get("claim_boundary") != R1C_CLAIM_BOUNDARY
        or set(attempt) != expected_attempt_fields
        or attempt.get("schema") != R1C_TARGET_ATTEMPT_SCHEMA
        or attempt.get("status") != R1C_TARGET_ATTEMPT_STATUS
        or not isinstance(attempt.get("created_at_utc"), str)
        or not attempt.get("created_at_utc")
        or not isinstance(attempt.get("policy"), str)
        or not attempt.get("policy")
        or attempt.get("sealed_cohort_constructed_before_record") is not False
        or attempt.get("model_loaded_before_record") is not False
        or attempt.get("output_inspected_before_record") is not False
        or not _is_sha256(attempt.get("attempt_id"))
        or attempt.get("attempt_id") != canonical_sha256(attempt_payload)
        or not isinstance(embedded_attempt, Mapping)
        or dict(embedded_attempt) != expected_embedded_attempt
        or not isinstance(embedded_completion, Mapping)
        or dict(embedded_completion)
        != {"logical_name": target_completion["logical_name"]}
        or not isinstance(target_identity, Mapping)
        or set(target_identity) != expected_target_identity_fields
        or target_identity.get("schema")
        != R1C_BIOLOGICAL_TARGET_IDENTITY_SCHEMA
        or target_identity.get("program") != "r1c-sealed-reverse-behavior-v2"
        or target_identity.get("stage") != "R1c"
        or target_identity.get("holdout_id")
        != "r1c-canonical-internal-short-intron-chr8-chr9-v1"
        or target_identity.get("target_chromosomes") != ["chr8", "chr9"]
        or target_identity.get("target_n") != 400
        or not _is_sha256(target_identity.get("checkpoint_sha256"))
        or not _is_sha256(target_identity.get("gtf_sha256"))
        or not _is_sha256(target_identity.get("fasta_sha256"))
        or not _is_sha256(target_identity.get("fasta_fai_sha256"))
        or not _is_sha256(target_identity.get("biological_target_key"))
        or target_identity.get("biological_target_key")
        != canonical_sha256(
            {
                key: value
                for key, value in target_identity.items()
                if key != "biological_target_key"
            }
        )
        or attempt.get("biological_target_key")
        != target_identity.get("biological_target_key")
        or not isinstance(method_context, Mapping)
        or set(method_context) != expected_method_context_fields
        or any(
            not _is_sha256(method_context.get(name))
            for name in (
                "protocol_sha256",
                "instrument_fingerprint_sha256",
                "r5_stack_fingerprint_sha256",
                "checkpoint_bundle_sha256",
                "construction_contract_sha256",
                "input_snapshot_sha256",
            )
        )
        or method_context.get("checkpoint_bundle_sha256")
        != target_identity.get("checkpoint_sha256")
        or method_context.get("endpoint_policy")
        != (
            "single masked acceptor G hard margin; paired versus "
            "distance-matched donor GT->AC 2x2"
        )
        or method_context.get("requested_artifact_logical_name")
        != artifact["logical_name"]
        or method_context.get("requested_manifest_logical_name")
        != sibling_manifest["logical_name"]
        or method_context.get("declared_completion_logical_name")
        != target_completion["logical_name"]
        or not isinstance(method_context.get("preflight_artifact"), Mapping)
        or not isinstance(method_context.get("preflight_manifest"), Mapping)
        or dict(method_context["preflight_artifact"]) != preflight_artifact
        or dict(method_context["preflight_manifest"]) != preflight_manifest
        or not isinstance(instrument_fingerprint, Mapping)
        or instrument_fingerprint.get("fingerprint_sha256")
        != method_context.get("instrument_fingerprint_sha256")
        or not isinstance(instrument_fingerprint.get("checkpoint"), Mapping)
        or instrument_fingerprint["checkpoint"].get("bundle_sha256")
        != target_identity.get("checkpoint_sha256")
        or not isinstance(instrument_fingerprint.get("gtf"), Mapping)
        or instrument_fingerprint["gtf"].get("sha256")
        != target_identity.get("gtf_sha256")
        or not isinstance(instrument_fingerprint.get("fasta"), Mapping)
        or instrument_fingerprint["fasta"].get("sha256")
        != target_identity.get("fasta_sha256")
        or not isinstance(instrument_fingerprint.get("fasta_fai"), Mapping)
        or instrument_fingerprint["fasta_fai"].get("sha256")
        != target_identity.get("fasta_fai_sha256")
        or record.get("r5_stack_fingerprint_sha256")
        != method_context.get("r5_stack_fingerprint_sha256")
        or not isinstance(gates, Mapping)
        or gates.get("raw_trimmed_gene_robustness") is not True
        or not isinstance(observed_paths, Mapping)
        or set(observed_paths)
        != {"core_paired_donor", "stronger_downstream_preference"}
        or observed_paths.get("core_paired_donor") is not True
        or (
            record.get("verdict")
            == "CONFIRMED_DOWNSTREAM_ACCEPTOR_PREFERENCE"
            and observed_paths.get("stronger_downstream_preference") is not True
        )
        or not isinstance(gene_robustness, Mapping)
        or gene_robustness.get("schema")
        != "gfmcd.r1c.equal_gene_robustness/1"
        or gene_robustness.get("randomisation_unit") != "unweighted gene"
        or gene_robustness.get("trim_fraction") != 0.10
        or not isinstance(gene_robustness.get("n_draws"), int)
        or gene_robustness.get("n_draws") < 10_000
        or gene_robustness.get("minimum_draws") != 10_000
        or gene_robustness.get("all_required_robust") is not True
        or not isinstance(gene_robustness.get("paths"), Mapping)
        or gene_robustness["paths"].get("core_paired_donor")
        != {"pass": True}
        or (
            record.get("verdict")
            == "CONFIRMED_DOWNSTREAM_ACCEPTOR_PREFERENCE"
            and gene_robustness["paths"].get("stronger_downstream_preference")
            != {"pass": True}
        )
        or not isinstance(target_cohort_identity, Mapping)
        or manifest.get("git_dirty") is not False
        or len(matching_outputs) != 1
        or len(matching_attempt_inputs) != 1
        or successor_attestation.get("schema")
        != "gfmcd.r1c.verified_target_handoff/1"
        or not _is_sha256(successor_attestation.get("attestation_sha256"))
        or successor_attestation.get("attestation_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in successor_attestation.items()
                if key != "attestation_sha256"
            }
        )
        or not isinstance(attested_packages, Mapping)
        or dict(attested_packages)
        != {
            "artifact": artifact,
            "manifest": sibling_manifest,
            "attempt": target_attempt,
            "completion": target_completion,
        }
        or not isinstance(attested_preflight, Mapping)
        or dict(attested_preflight)
        != {
            "artifact": preflight_artifact,
            "manifest": preflight_manifest,
        }
    ):
        raise R5CapstoneCohortError("R1c target capstone source contract differs")
    boundary_inference = record.get("boundary_inference")
    path_tests = (
        boundary_inference.get("path_tests")
        if isinstance(boundary_inference, Mapping)
        else None
    )
    if not isinstance(path_tests, Mapping) or set(path_tests) != {
        "core_paired_donor",
        "stronger_downstream_preference",
    }:
        raise R5CapstoneCohortError("R1c target exact boundary paths differ")
    replayed_paths: dict[str, bool] = {}
    for name in ("core_paired_donor", "stronger_downstream_preference"):
        path = path_tests[name]
        decision = path.get("decision") if isinstance(path, Mapping) else None
        if decision not in {"DECLARED", "UNRESOLVED", "NOT_DECLARED"}:
            raise R5CapstoneCohortError("R1c target exact boundary path differs")
        replayed_paths[name] = decision == "DECLARED"
    if replayed_paths != dict(observed_paths) or (
        replayed_paths["stronger_downstream_preference"]
        and not replayed_paths["core_paired_donor"]
    ):
        raise R5CapstoneCohortError("R1c target exact boundary paths differ")

    endpoints = gene_robustness.get("endpoints")
    if not isinstance(endpoints, Mapping) or set(endpoints) != {
        "Y_PD",
        "Y_PN",
        "S_P",
        "I",
    }:
        raise R5CapstoneCohortError("R1c target equal-gene robustness differs")

    def robust_positive(name: str) -> bool:
        endpoint = endpoints.get(name)
        return bool(
            isinstance(endpoint, Mapping)
            and endpoint.get("joint_boundary_pass") is True
            and endpoint.get("trim_fraction") == 0.10
            and isinstance(endpoint.get("raw"), Mapping)
            and endpoint["raw"].get("estimator") == "unweighted_gene_mean"
            and isinstance(endpoint.get("trimmed"), Mapping)
            and endpoint["trimmed"].get("estimator")
            == "10%_trimmed_unweighted_gene_mean"
        )

    pseudo = endpoints.get("Y_PN")
    pseudo_edges_ok = bool(
        isinstance(pseudo, Mapping)
        and pseudo.get("equivalence_pass") is True
        and all(
            isinstance(pseudo.get(edge), Mapping)
            and pseudo[edge].get("joint_boundary_pass") is True
            and pseudo[edge].get("trim_fraction") == 0.10
            and isinstance(pseudo[edge].get("raw"), Mapping)
            and pseudo[edge]["raw"].get("estimator")
            == "unweighted_gene_mean"
            and isinstance(pseudo[edge].get("trimmed"), Mapping)
            and pseudo[edge]["trimmed"].get("estimator")
            == "10%_trimmed_unweighted_gene_mean"
            for edge in ("lower_edge", "upper_edge")
        )
    )
    if (
        not robust_positive("Y_PD")
        or not robust_positive("S_P")
        or not pseudo_edges_ok
        or (
            record.get("verdict")
            == "CONFIRMED_DOWNSTREAM_ACCEPTOR_PREFERENCE"
            and not robust_positive("I")
        )
    ):
        raise R5CapstoneCohortError("R1c target equal-gene robustness differs")
    ordered_partner_margin_levels = _r1c_ordered_partner_margin_levels(record)
    rows, ordered_ids = _r1c_source_rows(record)
    source_cohort_payload = [
        {
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "decoy_donor": row["positions"]["matched_decoy_donor"],
            "decoy_donor_distance_mismatch": row["positions"][
                "matched_decoy_donor_distance_mismatch"
            ],
            "dyad_distance_mismatch": row["positions"]["dyad_distance_mismatch"],
        }
        for row in rows
    ]
    source_cohort_sha256 = canonical_sha256(source_cohort_payload)
    cohort_sha256 = canonical_sha256(rows)
    ordered_ids_sha256 = canonical_sha256(ordered_ids)
    ordered_gene_ids = [str(row["gene_id"]) for row in rows]
    sorted_unique_gene_ids = sorted(set(ordered_gene_ids))
    ordered_gene_ids_sha256 = canonical_sha256(ordered_gene_ids)
    sorted_unique_gene_ids_sha256 = canonical_sha256(sorted_unique_gene_ids)
    source_construction_sha256 = canonical_sha256(
        {
            "construction_version": R1C_CONSTRUCTION_VERSION,
            "cohort_order_version": R1C_COHORT_ORDER_VERSION,
            "ordered_six_position_rows": rows,
        }
    )
    expected_source_cohort_identity = {
        "source_cohort_sha256": source_cohort_sha256,
        "ordered_example_ids_sha256": ordered_ids_sha256,
        "target_gene_ids_sha256": sorted_unique_gene_ids_sha256,
        "ordered_gene_ids_sha256": ordered_gene_ids_sha256,
        "ordered_six_position_rows_sha256": cohort_sha256,
        "construction_sha256": source_construction_sha256,
        "ordered_partner_margin_levels_sha256": canonical_sha256(
            ordered_partner_margin_levels
        ),
    }
    expected_completion_fields = {
        "schema",
        "created_at_utc",
        "status",
        "attempt_id",
        "biological_target_key",
        "method_fingerprint_sha256",
        "r5_stack_fingerprint_sha256",
        "cohort_digests",
        "artifact",
        "manifest",
        "attempt",
        "official_verdict",
        "all_gates_pass",
        "gates",
        "completion_id",
    }
    completion_artifact = completion.get("artifact")
    completion_manifest = completion.get("manifest")
    completion_attempt = completion.get("attempt")
    if (
        record.get("cohort_sha256") != source_cohort_sha256
        or cohort_audit.get("selected_cohort_sha256") != source_cohort_sha256
        or cohort_audit.get("selected_example_ids_sha256") != ordered_ids_sha256
        or dict(target_cohort_identity) != expected_source_cohort_identity
        or set(completion) != expected_completion_fields
        or completion.get("schema") != R1C_TARGET_COMPLETION_SCHEMA
        or completion.get("status") != R1C_TARGET_COMPLETION_STATUS
        or not isinstance(completion.get("created_at_utc"), str)
        or not completion.get("created_at_utc")
        or not _is_sha256(completion.get("completion_id"))
        or completion.get("completion_id") != canonical_sha256(completion_payload)
        or completion.get("attempt_id") != attempt.get("attempt_id")
        or completion.get("biological_target_key")
        != target_identity.get("biological_target_key")
        or completion.get("method_fingerprint_sha256")
        != method_context.get("instrument_fingerprint_sha256")
        or completion.get("r5_stack_fingerprint_sha256")
        != method_context.get("r5_stack_fingerprint_sha256")
        or completion.get("cohort_digests") != expected_source_cohort_identity
        or not isinstance(completion_artifact, Mapping)
        or _logical_file_identity(
            completion_artifact, label="R1c completion artifact"
        )
        != artifact
        or not isinstance(completion_manifest, Mapping)
        or _logical_file_identity(
            completion_manifest, label="R1c completion manifest"
        )
        != sibling_manifest
        or not isinstance(completion_attempt, Mapping)
        or _logical_file_identity(
            completion_attempt, label="R1c completion attempt"
        )
        != target_attempt
        or completion.get("official_verdict") != record.get("verdict")
        or completion.get("all_gates_pass") is not True
        or completion.get("gates") != gates
        or successor_attestation.get("attempt_id") != attempt.get("attempt_id")
        or successor_attestation.get("completion_id")
        != completion.get("completion_id")
        or successor_attestation.get("biological_target_key")
        != target_identity.get("biological_target_key")
        or successor_attestation.get("instrument_fingerprint_sha256")
        != method_context.get("instrument_fingerprint_sha256")
        or successor_attestation.get("r5_stack_fingerprint_sha256")
        != method_context.get("r5_stack_fingerprint_sha256")
        or successor_attestation.get("target_cohort_identity")
        != expected_source_cohort_identity
        or successor_attestation.get("observed_decision_paths")
        != dict(observed_paths)
        or successor_attestation.get("verdict") != record.get("verdict")
        or successor_attestation.get("all_gates_pass") is not True
        or successor_attestation.get("gates") != gates
    ):
        raise R5CapstoneCohortError("R1c target cohort digests do not rederive")
    # Validate the exact preflight logical identities only after the broad semantic gate,
    # keeping all malformed-package failures inside the capstone error boundary.
    _logical_file_identity(
        method_context["preflight_artifact"], label="R1c preflight artifact"
    )
    _logical_file_identity(
        method_context["preflight_manifest"], label="R1c preflight manifest"
    )
    attempt_package_identity = dict(expected_embedded_attempt)
    completion_package_identity = target_completion | {
        "completion_id": completion["completion_id"],
        "status": completion["status"],
    }
    source_package = {
        "artifact": artifact,
        "manifest": sibling_manifest,
        "attempt": attempt_package_identity,
        "completion": completion_package_identity,
        "preflight_artifact": preflight_artifact,
        "preflight_manifest": preflight_manifest,
    }
    source_package_sha256 = canonical_sha256(source_package)
    construction = {
        "source_schema": R1C_TARGET_SCHEMA,
        "construction_version": R1C_CONSTRUCTION_VERSION,
        "cohort_order_version": R1C_COHORT_ORDER_VERSION,
        "n": 400,
        "matched_decoy_donor_caliper": R1C_MATCHED_DECOY_DONOR_CALIPER,
        "dyad_distance_caliper": R1C_DYAD_DISTANCE_CALIPER,
        "preconstruction_allowed_chromosomes": ["chr8", "chr9"],
        "artifact": artifact,
        "manifest": sibling_manifest,
        "attempt": attempt_package_identity,
        "completion": completion_package_identity,
        "preflight_artifact": preflight_artifact,
        "preflight_manifest": preflight_manifest,
        "successor_attestation_sha256": successor_attestation[
            "attestation_sha256"
        ],
        "source_package_sha256": source_package_sha256,
        "full_serialized_row_identities_sha256": cohort_sha256,
    }
    capstone_construction_sha256 = canonical_sha256(construction)
    payload = {
        "schema": R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
        "cohort_role": R5C_REQUIRED_COHORT_ROLE,
        "source": {
            "artifact": artifact,
            "manifest": sibling_manifest,
            "attempt": attempt_package_identity,
            "completion": completion_package_identity,
            "preflight_artifact": preflight_artifact,
            "preflight_manifest": preflight_manifest,
            "verified_successor_schema": (
                "gfmcd.r1c.verified_target_handoff/1"
            ),
            "successor_attestation_sha256": successor_attestation[
                "attestation_sha256"
            ],
            "biological_target_key": target_identity["biological_target_key"],
            "schema": R1C_TARGET_SCHEMA,
            "stage": "target",
            "split": "test",
            "n": 400,
            "preconstruction_allowed_chromosomes": ["chr8", "chr9"],
            "verdict": record["verdict"],
            "source_cohort_sha256": source_cohort_sha256,
            "source_selected_example_ids_sha256": ordered_ids_sha256,
            "construction_version": R1C_CONSTRUCTION_VERSION,
            "cohort_order_version": R1C_COHORT_ORDER_VERSION,
            "matched_decoy_donor_caliper": R1C_MATCHED_DECOY_DONOR_CALIPER,
            "dyad_distance_caliper": R1C_DYAD_DISTANCE_CALIPER,
            "source_package_sha256": source_package_sha256,
            "completion_receipt_available": True,
        },
        "selection_reads_model_outputs": False,
        "residual_validation_bucket_1_used": False,
        "handoff_reads_genome": False,
        "handoff_reads_chr8_chr9_sequences": False,
        "rows": rows,
        "row_count": 400,
        "cohort_sha256": cohort_sha256,
        "ordered_row_ids_sha256": ordered_ids_sha256,
        "ordered_gene_ids_sha256": ordered_gene_ids_sha256,
        "sorted_unique_gene_ids_sha256": sorted_unique_gene_ids_sha256,
        "unique_gene_count": len(sorted_unique_gene_ids),
        "source_legacy_cohort_sha256": source_cohort_sha256,
        "source_package_sha256": source_package_sha256,
        "completion_receipt_available": True,
        "capstone_construction_sha256": capstone_construction_sha256,
    }
    return payload | {"record_sha256": canonical_sha256(payload)}


def validate_r5_capstone_cohort_handoff(
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """Reauthenticate a serialized capstone handoff without touching chr8/chr9 data."""

    expected_fields = {
        "schema",
        "cohort_role",
        "source",
        "selection_reads_model_outputs",
        "residual_validation_bucket_1_used",
        "handoff_reads_genome",
        "handoff_reads_chr8_chr9_sequences",
        "rows",
        "row_count",
        "cohort_sha256",
        "ordered_row_ids_sha256",
        "ordered_gene_ids_sha256",
        "sorted_unique_gene_ids_sha256",
        "unique_gene_count",
        "source_legacy_cohort_sha256",
        "source_package_sha256",
        "completion_receipt_available",
        "capstone_construction_sha256",
        "record_sha256",
    }
    if set(handoff) != expected_fields:
        raise R5CapstoneCohortError("R5 capstone handoff fields differ")
    payload = {key: value for key, value in handoff.items() if key != "record_sha256"}
    source = handoff.get("source")
    if (
        handoff.get("schema") != R5_CAPSTONE_COHORT_HANDOFF_SCHEMA
        or handoff.get("cohort_role") != R5C_REQUIRED_COHORT_ROLE
        or handoff.get("selection_reads_model_outputs") is not False
        or handoff.get("residual_validation_bucket_1_used") is not False
        or handoff.get("handoff_reads_genome") is not False
        or handoff.get("handoff_reads_chr8_chr9_sequences") is not False
        or handoff.get("completion_receipt_available") is not True
        or handoff.get("record_sha256") != canonical_sha256(payload)
        or not isinstance(source, Mapping)
        or set(source)
        != {
            "artifact",
            "manifest",
            "attempt",
            "completion",
            "preflight_artifact",
            "preflight_manifest",
            "verified_successor_schema",
            "successor_attestation_sha256",
            "biological_target_key",
            "schema",
            "stage",
            "split",
            "n",
            "preconstruction_allowed_chromosomes",
            "verdict",
            "source_cohort_sha256",
            "source_selected_example_ids_sha256",
            "construction_version",
            "cohort_order_version",
            "matched_decoy_donor_caliper",
            "dyad_distance_caliper",
            "source_package_sha256",
            "completion_receipt_available",
        }
        or source.get("schema") != R1C_TARGET_SCHEMA
        or source.get("stage") != "target"
        or source.get("split") != "test"
        or source.get("n") != 400
        or source.get("preconstruction_allowed_chromosomes") != ["chr8", "chr9"]
        or source.get("verdict") not in set(R1C_CONFIRMED_BEHAVIOR_VERDICTS)
        or source.get("construction_version") != R1C_CONSTRUCTION_VERSION
        or source.get("cohort_order_version") != R1C_COHORT_ORDER_VERSION
        or source.get("matched_decoy_donor_caliper")
        != R1C_MATCHED_DECOY_DONOR_CALIPER
        or source.get("dyad_distance_caliper") != R1C_DYAD_DISTANCE_CALIPER
        or source.get("completion_receipt_available") is not True
        or source.get("verified_successor_schema")
        != "gfmcd.r1c.verified_target_handoff/1"
        or not _is_sha256(source.get("successor_attestation_sha256"))
    ):
        raise R5CapstoneCohortError("R5 capstone handoff contract differs")
    if not isinstance(source, Mapping):
        raise R5CapstoneCohortError("R5 capstone source identity is missing")
    artifact = source.get("artifact")
    manifest = source.get("manifest")
    attempt = source.get("attempt")
    completion = source.get("completion")
    preflight_artifact_value = source.get("preflight_artifact")
    preflight_manifest_value = source.get("preflight_manifest")
    if (
        not isinstance(artifact, Mapping)
        or not isinstance(manifest, Mapping)
        or not isinstance(attempt, Mapping)
        or not isinstance(completion, Mapping)
        or not isinstance(preflight_artifact_value, Mapping)
        or not isinstance(preflight_manifest_value, Mapping)
    ):
        raise R5CapstoneCohortError("R5 capstone package identity is missing")
    artifact_identity = _logical_file_identity(artifact, label="R1c target artifact")
    manifest_identity = _logical_file_identity(manifest, label="R1c target manifest")
    if set(attempt) != {
        "logical_name",
        "size_bytes",
        "sha256",
        "attempt_id",
        "biological_target_key",
        "status",
    }:
        raise R5CapstoneCohortError("R5 capstone attempt identity fields differ")
    attempt_identity = _logical_file_identity(
        {key: attempt[key] for key in ("logical_name", "size_bytes", "sha256")},
        label="R1c target-attempt receipt",
    )
    attempt_id = attempt.get("attempt_id")
    biological_target_key = attempt.get("biological_target_key")
    if (
        not _is_sha256(attempt_id)
        or not _is_sha256(biological_target_key)
        or attempt.get("status") != R1C_TARGET_ATTEMPT_STATUS
        or source.get("biological_target_key") != biological_target_key
    ):
        raise R5CapstoneCohortError("R5 capstone attempt ID is malformed")
    attempt_package_identity = attempt_identity | {
        "attempt_id": attempt_id,
        "biological_target_key": biological_target_key,
        "status": R1C_TARGET_ATTEMPT_STATUS,
    }
    if set(completion) != {
        "logical_name",
        "size_bytes",
        "sha256",
        "completion_id",
        "status",
    }:
        raise R5CapstoneCohortError("R5 capstone completion identity fields differ")
    completion_identity = _logical_file_identity(
        {
            key: completion[key]
            for key in ("logical_name", "size_bytes", "sha256")
        },
        label="R1c target-completion receipt",
    )
    completion_id = completion.get("completion_id")
    if (
        not _is_sha256(completion_id)
        or completion.get("status") != R1C_TARGET_COMPLETION_STATUS
    ):
        raise R5CapstoneCohortError("R5 capstone completion ID is malformed")
    completion_package_identity = completion_identity | {
        "completion_id": completion_id,
        "status": R1C_TARGET_COMPLETION_STATUS,
    }
    preflight_artifact_identity = _logical_file_identity(
        preflight_artifact_value, label="R1c preflight artifact"
    )
    preflight_manifest_identity = _logical_file_identity(
        preflight_manifest_value, label="R1c preflight manifest"
    )
    source_package = {
        "artifact": artifact_identity,
        "manifest": manifest_identity,
        "attempt": attempt_package_identity,
        "completion": completion_package_identity,
        "preflight_artifact": preflight_artifact_identity,
        "preflight_manifest": preflight_manifest_identity,
    }
    source_package_sha256 = canonical_sha256(source_package)
    rows_value = handoff.get("rows")
    if not isinstance(rows_value, list) or len(rows_value) != 400:
        raise R5CapstoneCohortError("R5 capstone handoff must contain 400 rows")
    rows: list[dict[str, Any]] = []
    for row in rows_value:
        if not isinstance(row, Mapping):
            raise R5CapstoneCohortError("R5 capstone serialized row is malformed")
        identity = r1c_serialized_row_identity(row)
        if dict(row) != identity:
            raise R5CapstoneCohortError(
                "R5 capstone rows must contain outcome-free identities only"
            )
        rows.append(identity)
    ordered_ids = [str(row["example_id"]) for row in rows]
    ordered_gene_ids = [str(row["gene_id"]) for row in rows]
    sorted_unique_gene_ids = sorted(set(ordered_gene_ids))
    source_cohort_payload = [
        {
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "decoy_donor": row["positions"]["matched_decoy_donor"],
            "decoy_donor_distance_mismatch": row["positions"][
                "matched_decoy_donor_distance_mismatch"
            ],
            "dyad_distance_mismatch": row["positions"]["dyad_distance_mismatch"],
        }
        for row in rows
    ]
    construction = {
        "source_schema": R1C_TARGET_SCHEMA,
        "construction_version": R1C_CONSTRUCTION_VERSION,
        "cohort_order_version": R1C_COHORT_ORDER_VERSION,
        "n": 400,
        "matched_decoy_donor_caliper": R1C_MATCHED_DECOY_DONOR_CALIPER,
        "dyad_distance_caliper": R1C_DYAD_DISTANCE_CALIPER,
        "preconstruction_allowed_chromosomes": ["chr8", "chr9"],
        "artifact": artifact_identity,
        "manifest": manifest_identity,
        "attempt": attempt_package_identity,
        "completion": completion_package_identity,
        "preflight_artifact": preflight_artifact_identity,
        "preflight_manifest": preflight_manifest_identity,
        "successor_attestation_sha256": source[
            "successor_attestation_sha256"
        ],
        "source_package_sha256": source_package_sha256,
        "full_serialized_row_identities_sha256": canonical_sha256(rows),
    }
    if (
        len(set(ordered_ids)) != 400
        or handoff.get("row_count") != 400
        or handoff.get("cohort_sha256") != canonical_sha256(rows)
        or handoff.get("ordered_row_ids_sha256") != canonical_sha256(ordered_ids)
        or handoff.get("ordered_gene_ids_sha256")
        != canonical_sha256(ordered_gene_ids)
        or handoff.get("sorted_unique_gene_ids_sha256")
        != canonical_sha256(sorted_unique_gene_ids)
        or handoff.get("unique_gene_count") != len(sorted_unique_gene_ids)
        or handoff.get("source_legacy_cohort_sha256")
        != canonical_sha256(source_cohort_payload)
        or handoff.get("source_package_sha256") != source_package_sha256
        or source.get("source_package_sha256") != source_package_sha256
        or handoff.get("capstone_construction_sha256")
        != canonical_sha256(construction)
        or source.get("source_cohort_sha256")
        != canonical_sha256(source_cohort_payload)
        or source.get("source_selected_example_ids_sha256")
        != canonical_sha256(ordered_ids)
    ):
        raise R5CapstoneCohortError("R5 capstone row identity differs")
    return {
        "schema": R5_CAPSTONE_COHORT_HANDOFF_SCHEMA,
        "verified": True,
        "cohort_role": R5C_REQUIRED_COHORT_ROLE,
        "artifact": artifact_identity,
        "manifest": manifest_identity,
        "attempt": attempt_package_identity,
        "attempt_id": attempt_id,
        "biological_target_key": biological_target_key,
        "completion": completion_package_identity,
        "completion_id": completion_id,
        "preflight_artifact": preflight_artifact_identity,
        "preflight_manifest": preflight_manifest_identity,
        "verified_successor_schema": source["verified_successor_schema"],
        "successor_attestation_sha256": source[
            "successor_attestation_sha256"
        ],
        "rows": rows,
        "row_count": 400,
        "cohort_sha256": handoff["cohort_sha256"],
        "ordered_row_ids_sha256": handoff["ordered_row_ids_sha256"],
        "ordered_gene_ids_sha256": handoff["ordered_gene_ids_sha256"],
        "sorted_unique_gene_ids_sha256": handoff[
            "sorted_unique_gene_ids_sha256"
        ],
        "unique_gene_count": handoff["unique_gene_count"],
        "source_legacy_cohort_sha256": handoff[
            "source_legacy_cohort_sha256"
        ],
        "source_package_sha256": source_package_sha256,
        "completion_receipt_available": True,
        "capstone_construction_sha256": handoff[
            "capstone_construction_sha256"
        ],
        "residual_validation_bucket_1_used": False,
    }


def _r1c_target_pin_identity(kind: str) -> dict[str, Any]:
    values = {
        "artifact": (
            FROZEN_R1C_TARGET_ARTIFACT_LOGICAL_NAME,
            FROZEN_R1C_TARGET_ARTIFACT_SIZE_BYTES,
            FROZEN_R1C_TARGET_ARTIFACT_SHA256,
        ),
        "manifest": (
            FROZEN_R1C_TARGET_MANIFEST_LOGICAL_NAME,
            FROZEN_R1C_TARGET_MANIFEST_SIZE_BYTES,
            FROZEN_R1C_TARGET_MANIFEST_SHA256,
        ),
        "attempt": (
            FROZEN_R1C_TARGET_ATTEMPT_LOGICAL_NAME,
            FROZEN_R1C_TARGET_ATTEMPT_SIZE_BYTES,
            FROZEN_R1C_TARGET_ATTEMPT_SHA256,
        ),
        "completion": (
            FROZEN_R1C_TARGET_COMPLETION_LOGICAL_NAME,
            FROZEN_R1C_TARGET_COMPLETION_SIZE_BYTES,
            FROZEN_R1C_TARGET_COMPLETION_SHA256,
        ),
    }
    logical_name, size_bytes, sha256 = values[kind]
    if logical_name is None or size_bytes is None or sha256 is None:
        raise R5CapstoneCohortError(
            "R1c target artifact/manifest/attempt/completion pins remain unset; "
            "capstone is sealed"
        )
    return _logical_file_identity(
        {
            "logical_name": logical_name,
            "size_bytes": size_bytes,
            "sha256": sha256,
        },
        label=f"pinned R1c target {kind}",
    )


def _load_r1c_successor_verifier_module():
    """Load the bound R1c runner only when a model-free package replay is requested."""

    module_name = "_gfmcd_bound_r1c_successor_verifier"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    runner_path = (
        Path(__file__).resolve().parents[3]
        / "scripts/splicebert_acceptor_coupling/run_r1c_sealed_reverse_behavior.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise R5CapstoneCohortError("bound R1c successor verifier is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _invoke_r1c_successor_preflight_verifier(
    artifact_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    module = _load_r1c_successor_verifier_module()
    verifier = getattr(module, "verify_successor_preflight_package", None)
    if not callable(verifier):
        raise R5CapstoneCohortError("R1c preflight semantic verifier is unavailable")
    try:
        result = verifier(artifact_path, manifest_path)
    except Exception as exc:
        raise R5CapstoneCohortError(
            "R1c preflight semantic verifier rejected the package"
        ) from exc
    if not isinstance(result, Mapping):
        raise R5CapstoneCohortError("R1c preflight verifier result is malformed")
    return dict(result)


def _invoke_r1c_successor_target_verifier(
    artifact_path: str | Path,
    manifest_path: str | Path,
    attempt_path: str | Path,
    completion_path: str | Path,
    *,
    preflight_artifact_path: str | Path,
) -> dict[str, Any]:
    module = _load_r1c_successor_verifier_module()
    verifier = getattr(module, "verify_successor_target_package", None)
    if not callable(verifier):
        raise R5CapstoneCohortError("R1c target semantic verifier is unavailable")
    try:
        result = verifier(
            artifact_path,
            manifest_path,
            attempt_path,
            completion_path,
            preflight_artifact_path=preflight_artifact_path,
        )
    except Exception as exc:
        raise R5CapstoneCohortError(
            "R1c target semantic verifier rejected the package"
        ) from exc
    if not isinstance(result, Mapping):
        raise R5CapstoneCohortError("R1c target verifier result is malformed")
    return dict(result)


def load_r1c_capstone_cohort_handoff(
    artifact_path: str | Path,
    manifest_path: str | Path,
    attempt_path: str | Path,
    completion_path: str | Path,
    preflight_artifact_path: str | Path,
    preflight_manifest_path: str | Path,
) -> dict[str, Any]:
    """Load and replay the exact six-file R1c target/predecessor package."""

    artifact_pin = _r1c_target_pin_identity("artifact")
    manifest_pin = _r1c_target_pin_identity("manifest")
    attempt_pin = _r1c_target_pin_identity("attempt")
    completion_pin = _r1c_target_pin_identity("completion")
    preflight_artifact_pin = _r1c_preflight_pin_identity("artifact")
    preflight_manifest_pin = _r1c_preflight_pin_identity("manifest")

    def load_exact(path_value: str | Path, pin: Mapping[str, Any], label: str):
        path = Path(path_value).expanduser().resolve(strict=True)
        observed = {
            "logical_name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if observed != pin:
            raise R5CapstoneCohortError(f"{label} differs from its literal pin")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise R5CapstoneCohortError(f"{label} is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise R5CapstoneCohortError(f"{label} must contain a JSON object")
        return value, observed

    record, artifact_identity = load_exact(
        artifact_path, artifact_pin, "R1c target artifact"
    )
    manifest, manifest_identity = load_exact(
        manifest_path, manifest_pin, "R1c target manifest"
    )
    attempt, attempt_identity = load_exact(
        attempt_path, attempt_pin, "R1c target-attempt receipt"
    )
    completion, completion_identity = load_exact(
        completion_path, completion_pin, "R1c target-completion receipt"
    )
    preflight_record, preflight_artifact_identity = load_exact(
        preflight_artifact_path,
        preflight_artifact_pin,
        "R1c predecessor preflight artifact",
    )
    preflight_manifest, preflight_manifest_identity = load_exact(
        preflight_manifest_path,
        preflight_manifest_pin,
        "R1c predecessor preflight manifest",
    )
    expected_sibling = (
        Path(preflight_artifact_path)
        .expanduser()
        .resolve(strict=True)
        .with_suffix(".manifest.json")
    )
    if Path(preflight_manifest_path).expanduser().resolve(strict=True) != expected_sibling:
        raise R5CapstoneCohortError(
            "R1c successor verifier requires the exact sibling preflight manifest"
        )
    preflight_verified = _invoke_r1c_successor_preflight_verifier(
        preflight_artifact_path, preflight_manifest_path
    )
    preflight_attestation = preflight_verified.get("attestation")
    if (
        preflight_verified.get("record") != preflight_record
        or not isinstance(preflight_attestation, Mapping)
        or preflight_attestation.get("schema")
        != "gfmcd.r1c.verified_preflight_handoff/1"
        or preflight_attestation.get("artifact") != preflight_artifact_identity
        or preflight_attestation.get("manifest") != preflight_manifest_identity
    ):
        raise R5CapstoneCohortError(
            "R1c preflight verifier attestation differs from exact package bytes"
        )
    successor_attestation = _invoke_r1c_successor_target_verifier(
        artifact_path,
        manifest_path,
        attempt_path,
        completion_path,
        preflight_artifact_path=preflight_artifact_path,
    )
    handoff = build_r5_capstone_cohort_handoff(
        record,
        manifest,
        attempt,
        completion,
        artifact_identity=artifact_identity,
        manifest_identity=manifest_identity,
        attempt_identity=attempt_identity,
        completion_identity=completion_identity,
        preflight_artifact_identity=preflight_artifact_identity,
        preflight_manifest_identity=preflight_manifest_identity,
        successor_attestation=successor_attestation,
    )
    validate_r5_capstone_cohort_handoff(handoff)
    return handoff


def build_r1c_verified_target_common_legs(
    record: Mapping[str, Any],
    handoff: Mapping[str, Any],
    successor_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose only R1c's authenticated common-leg margins after cohort selection.

    The capstone cohort handoff deliberately contains no outcomes.  This second,
    explicitly outcome-bearing object exists only so a downstream target can replay
    the byte-identical R1c ``P_clean``, ``P_under_D``, and ``P_under_N`` legs after its
    own irreversible target receipt has been created.  It is never a cohort selector
    or a source of new estimands.
    """

    verified = validate_r5_capstone_cohort_handoff(handoff)
    if (
        not isinstance(successor_attestation, Mapping)
        or successor_attestation.get("schema")
        != "gfmcd.r1c.verified_target_handoff/1"
        or successor_attestation.get("attestation_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in successor_attestation.items()
                if key != "attestation_sha256"
            }
        )
        or successor_attestation.get("attestation_sha256")
        != verified["successor_attestation_sha256"]
        or successor_attestation.get("target_cohort_identity")
        != record.get("target_cohort_identity")
        or successor_attestation.get("all_gates_pass") is not True
        or successor_attestation.get("verdict")
        not in set(R1C_CONFIRMED_BEHAVIOR_VERDICTS)
    ):
        raise R5CapstoneCohortError(
            "R1c common-leg source lacks the exact successful target attestation"
        )

    common_rows = _r1c_ordered_partner_margin_levels(record)
    identity_rows = verified["rows"]
    if len(common_rows) != len(identity_rows) or any(
        common["example_id"] != identity["example_id"]
        or common["gene_id"] != identity["gene_id"]
        for common, identity in zip(common_rows, identity_rows, strict=True)
    ):
        raise R5CapstoneCohortError(
            "R1c common-leg row order differs from the outcome-free handoff"
        )
    common_digest = canonical_sha256(common_rows)
    target_identity = record.get("target_cohort_identity")
    if (
        not isinstance(target_identity, Mapping)
        or common_digest
        != target_identity.get("ordered_partner_margin_levels_sha256")
    ):
        raise R5CapstoneCohortError(
            "R1c common-leg margins differ from the completed target digest"
        )

    payload = {
        "schema": R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA,
        "source_package_sha256": verified["source_package_sha256"],
        "successor_attestation_sha256": successor_attestation[
            "attestation_sha256"
        ],
        "cohort_sha256": verified["cohort_sha256"],
        "ordered_row_ids_sha256": verified["ordered_row_ids_sha256"],
        "ordered_partner_margin_levels_sha256": common_digest,
        "row_count": len(common_rows),
        "rows": common_rows,
        "model_outputs_present": True,
        "may_be_used_for_cohort_selection": False,
        "authorized_use": (
            "same-row R1c common-leg replay after the downstream target receipt only"
        ),
        "caller_must_verify_downstream_target_receipt_before_load": True,
    }
    return payload | {"record_sha256": canonical_sha256(payload)}


def load_r1c_verified_target_common_legs(
    artifact_path: str | Path,
    manifest_path: str | Path,
    attempt_path: str | Path,
    completion_path: str | Path,
    preflight_artifact_path: str | Path,
    preflight_manifest_path: str | Path,
) -> dict[str, Any]:
    """Load semantically verified R1c target common legs from the six pinned files.

    Callers are responsible for creating their own irreversible target receipt before
    invoking this outcome-bearing loader.  The function reauthenticates the exact R1c
    package and returns no family decisions beyond its already-frozen behavior verdict.
    """

    handoff = load_r1c_capstone_cohort_handoff(
        artifact_path,
        manifest_path,
        attempt_path,
        completion_path,
        preflight_artifact_path,
        preflight_manifest_path,
    )
    artifact_resolved = Path(artifact_path).expanduser().resolve(strict=True)
    artifact_bytes = artifact_resolved.read_bytes()
    artifact_identity = {
        "logical_name": artifact_resolved.name,
        "size_bytes": len(artifact_bytes),
        "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
    }
    if artifact_identity != handoff["artifact"]:
        raise R5CapstoneCohortError(
            "R1c common-leg artifact changed after package authentication"
        )
    try:
        record = json.loads(artifact_bytes)
    except json.JSONDecodeError as exc:
        raise R5CapstoneCohortError(
            "R1c common-leg artifact is not valid JSON"
        ) from exc
    if not isinstance(record, Mapping):
        raise R5CapstoneCohortError(
            "R1c common-leg artifact must contain one JSON object"
        )
    successor_attestation = _invoke_r1c_successor_target_verifier(
        artifact_path,
        manifest_path,
        attempt_path,
        completion_path,
        preflight_artifact_path=preflight_artifact_path,
    )
    result = build_r1c_verified_target_common_legs(
        record, handoff, successor_attestation
    )
    current_artifact_identity = {
        "logical_name": artifact_resolved.name,
        "size_bytes": artifact_resolved.stat().st_size,
        "sha256": _file_sha256(artifact_resolved),
    }
    if current_artifact_identity != artifact_identity:
        raise R5CapstoneCohortError(
            "R1c common-leg artifact changed during semantic replay"
        )
    return result


def build_r5_preflight_cohort_handoff(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    artifact_identity: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    successor_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate R1c's nonsealed validation cohort for R5a/R5c preflight.

    Only the three partner common-leg margin levels survive from the outcome-bearing
    source rows.  Family decisions, descriptive cells, health outcomes, and scientific
    dispositions are deliberately not carried into the handoff.  This package is a
    method-preflight input and can never authorize scientific entry.
    """

    artifact = _logical_file_identity(
        artifact_identity, label="R1c preflight artifact"
    )
    sibling_manifest = _logical_file_identity(
        manifest_identity, label="R1c preflight manifest"
    )
    outputs = manifest.get("outputs")
    matching_outputs = (
        [
            item
            for item in outputs
            if isinstance(item, Mapping)
            and Path(str(item.get("path", ""))).name == artifact["logical_name"]
            and item.get("size_bytes") == artifact["size_bytes"]
            and item.get("sha256") == artifact["sha256"]
        ]
        if isinstance(outputs, list)
        else []
    )
    cohort_audit = record.get("cohort_audit")
    config = record.get("config")
    gates = record.get("gates")
    robustness = record.get("gene_robustness")
    fingerprint = record.get("instrument_fingerprint")
    target_cohort_identity = record.get("target_cohort_identity")
    if (
        record.get("schema") != R1C_TARGET_SCHEMA
        or record.get("experiment") != "R1c_corrected_reverse_behavior"
        or record.get("stage") != "preflight"
        or record.get("status") != "DEVELOPMENT"
        or record.get("evidence_rung") != "BEHAVIOR_ONLY"
        or record.get("split") != "val"
        or record.get("sealed_split_opened") is not False
        or record.get("n") != 400
        or record.get("preconstruction_allowed_chromosomes")
        != ["chr11", "chr16", "chr19"]
        or record.get("all_gates_pass") is not True
        or record.get("verdict") != "PREFLIGHT_GO"
        or record.get("target_attempt") is not None
        or record.get("target_completion") is not None
        or record.get("target_attempt_disposition") is not None
        or not isinstance(cohort_audit, Mapping)
        or cohort_audit.get("selected") != 400
        or cohort_audit.get("selection_reads_model_outputs") is not False
        or not isinstance(record.get("n_genes"), int)
        or isinstance(record.get("n_genes"), bool)
        or record.get("n_genes") < 30
        or cohort_audit.get("selected_genes") != record.get("n_genes")
        or not isinstance(config, Mapping)
        or config.get("single_G_acceptor_endpoint") is not True
        or config.get("outcome_conditioning") is not None
        or config.get("construction_version") != R1C_CONSTRUCTION_VERSION
        or config.get("cohort_order_version") != R1C_COHORT_ORDER_VERSION
        or config.get("matched_decoy_donor_caliper")
        != R1C_MATCHED_DECOY_DONOR_CALIPER
        or config.get("dyad_distance_caliper") != R1C_DYAD_DISTANCE_CALIPER
        or config.get("primary_family_names") != list(R1C_PRIMARY_FAMILY_NAMES)
        or config.get("cell_names") != list(R1C_CELL_NAMES)
        or record.get("claim_boundary") != R1C_CLAIM_BOUNDARY
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
        or gates.get("raw_trimmed_gene_robustness") is not True
        or not isinstance(robustness, Mapping)
        or robustness.get("schema") != "gfmcd.r1c.equal_gene_robustness/1"
        or robustness.get("randomisation_unit") != "unweighted gene"
        or robustness.get("trim_fraction") != 0.10
        or robustness.get("minimum_draws") != 10_000
        or not isinstance(robustness.get("n_draws"), int)
        or robustness.get("n_draws") < 10_000
        or robustness.get("all_required_robust") is not True
        or not isinstance(robustness.get("paths"), Mapping)
        or robustness["paths"].get("core_paired_donor") != {"pass": True}
        or not isinstance(fingerprint, Mapping)
        or not isinstance(fingerprint.get("fingerprint_sha256"), str)
        or not _is_sha256(fingerprint.get("fingerprint_sha256"))
        or not isinstance(fingerprint.get("checkpoint"), Mapping)
        or not _is_sha256(fingerprint["checkpoint"].get("bundle_sha256"))
        or not isinstance(fingerprint.get("gtf"), Mapping)
        or fingerprint["gtf"].get("sha256")
        != PINNED_R5_SOURCE_IDENTITY["gtf"]["sha256"]
        or not isinstance(fingerprint.get("fasta"), Mapping)
        or fingerprint["fasta"].get("sha256")
        != PINNED_R5_SOURCE_IDENTITY["fasta"]["sha256"]
        or not isinstance(fingerprint.get("fasta_fai"), Mapping)
        or fingerprint["fasta_fai"].get("sha256")
        != PINNED_R5_SOURCE_IDENTITY["fasta_fai"]["sha256"]
        or not isinstance(target_cohort_identity, Mapping)
        or manifest.get("git_dirty") is not False
        or not isinstance(manifest.get("config"), Mapping)
        or manifest["config"].get("stage") != "preflight"
        or len(matching_outputs) != 1
        or successor_attestation.get("schema")
        != "gfmcd.r1c.verified_preflight_handoff/1"
        or not _is_sha256(successor_attestation.get("attestation_sha256"))
        or successor_attestation.get("attestation_sha256")
        != canonical_sha256(
            {
                key: value
                for key, value in successor_attestation.items()
                if key != "attestation_sha256"
            }
        )
        or successor_attestation.get("artifact") != artifact
        or successor_attestation.get("manifest") != sibling_manifest
        or successor_attestation.get("n") != 400
        or successor_attestation.get("n_genes") != record.get("n_genes")
        or successor_attestation.get("verdict") != "PREFLIGHT_GO"
        or successor_attestation.get("all_gates_pass") is not True
        or successor_attestation.get("gates") != gates
    ):
        raise R5CapstoneCohortError("R1c preflight cohort source contract differs")

    rows, ordered_ids = _r1c_source_rows(
        record, allowed_chromosomes=("chr11", "chr16", "chr19")
    )
    common_leg_rows = _r1c_ordered_partner_margin_levels(record)
    ordered_gene_ids = [str(row["gene_id"]) for row in rows]
    sorted_unique_gene_ids = sorted(set(ordered_gene_ids))
    source_cohort_payload = [
        {
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "decoy_donor": row["positions"]["matched_decoy_donor"],
            "decoy_donor_distance_mismatch": row["positions"][
                "matched_decoy_donor_distance_mismatch"
            ],
            "dyad_distance_mismatch": row["positions"]["dyad_distance_mismatch"],
        }
        for row in rows
    ]
    source_cohort_sha256 = canonical_sha256(source_cohort_payload)
    ordered_ids_sha256 = canonical_sha256(ordered_ids)
    cohort_sha256 = canonical_sha256(rows)
    ordered_gene_ids_sha256 = canonical_sha256(ordered_gene_ids)
    sorted_unique_gene_ids_sha256 = canonical_sha256(sorted_unique_gene_ids)
    partner_margin_levels_sha256 = canonical_sha256(common_leg_rows)
    source_construction_sha256 = canonical_sha256(
        {
            "construction_version": R1C_CONSTRUCTION_VERSION,
            "cohort_order_version": R1C_COHORT_ORDER_VERSION,
            "ordered_six_position_rows": rows,
        }
    )
    expected_source_cohort_identity = {
        "source_cohort_sha256": source_cohort_sha256,
        "ordered_example_ids_sha256": ordered_ids_sha256,
        "target_gene_ids_sha256": sorted_unique_gene_ids_sha256,
        "ordered_gene_ids_sha256": ordered_gene_ids_sha256,
        "ordered_six_position_rows_sha256": cohort_sha256,
        "construction_sha256": source_construction_sha256,
        "ordered_partner_margin_levels_sha256": partner_margin_levels_sha256,
    }
    if (
        record.get("cohort_sha256") != source_cohort_sha256
        or cohort_audit.get("selected_cohort_sha256") != source_cohort_sha256
        or cohort_audit.get("selected_example_ids_sha256") != ordered_ids_sha256
        or dict(target_cohort_identity) != expected_source_cohort_identity
        or successor_attestation.get("target_cohort_identity")
        != expected_source_cohort_identity
    ):
        raise R5CapstoneCohortError("R1c preflight cohort digests do not rederive")

    source_package = {"artifact": artifact, "manifest": sibling_manifest}
    source_package_sha256 = canonical_sha256(source_package)
    construction = {
        "source_schema": R1C_TARGET_SCHEMA,
        "construction_version": R1C_CONSTRUCTION_VERSION,
        "cohort_order_version": R1C_COHORT_ORDER_VERSION,
        "n": 400,
        "matched_decoy_donor_caliper": R1C_MATCHED_DECOY_DONOR_CALIPER,
        "dyad_distance_caliper": R1C_DYAD_DISTANCE_CALIPER,
        "preconstruction_allowed_chromosomes": ["chr11", "chr16", "chr19"],
        "artifact": artifact,
        "manifest": sibling_manifest,
        "source_package_sha256": source_package_sha256,
        "full_serialized_row_identities_sha256": cohort_sha256,
        "ordered_partner_margin_levels_sha256": partner_margin_levels_sha256,
        "successor_attestation_sha256": successor_attestation[
            "attestation_sha256"
        ],
    }
    preflight_construction_sha256 = canonical_sha256(construction)
    payload = {
        "schema": R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
        "cohort_role": R5A_R5C_PREFLIGHT_COHORT_ROLE,
        "source": {
            "artifact": artifact,
            "manifest": sibling_manifest,
            "schema": R1C_TARGET_SCHEMA,
            "stage": "preflight",
            "split": "val",
            "n": 400,
            "preconstruction_allowed_chromosomes": ["chr11", "chr16", "chr19"],
            "verdict": "PREFLIGHT_GO",
            "source_cohort_sha256": source_cohort_sha256,
            "source_selected_example_ids_sha256": ordered_ids_sha256,
            "construction_version": R1C_CONSTRUCTION_VERSION,
            "cohort_order_version": R1C_COHORT_ORDER_VERSION,
            "matched_decoy_donor_caliper": R1C_MATCHED_DECOY_DONOR_CALIPER,
            "dyad_distance_caliper": R1C_DYAD_DISTANCE_CALIPER,
            "checkpoint_sha256": fingerprint["checkpoint"]["bundle_sha256"],
            "source_package_sha256": source_package_sha256,
            "verified_successor_schema": (
                "gfmcd.r1c.verified_preflight_handoff/1"
            ),
            "successor_attestation_sha256": successor_attestation[
                "attestation_sha256"
            ],
            "completion_receipt_available": False,
            "scientific_entry_authorized": False,
        },
        "selection_reads_model_outputs": False,
        "scientific_entry_authorized": False,
        "sealed_split_opened": False,
        "residual_validation_cohort_used": False,
        "handoff_reads_genome": False,
        "rows": rows,
        "common_leg_rows": common_leg_rows,
        "preserved_outcome_fields": ["P_clean", "P_under_D", "P_under_N"],
        "row_count": 400,
        "cohort_sha256": cohort_sha256,
        "ordered_row_ids_sha256": ordered_ids_sha256,
        "ordered_gene_ids_sha256": ordered_gene_ids_sha256,
        "sorted_unique_gene_ids_sha256": sorted_unique_gene_ids_sha256,
        "unique_gene_count": len(sorted_unique_gene_ids),
        "ordered_partner_margin_levels_sha256": partner_margin_levels_sha256,
        "source_legacy_cohort_sha256": source_cohort_sha256,
        "source_package_sha256": source_package_sha256,
        "completion_receipt_available": False,
        "preflight_construction_sha256": preflight_construction_sha256,
    }
    return payload | {"record_sha256": canonical_sha256(payload)}


def validate_r5_preflight_cohort_handoff(
    handoff: Mapping[str, Any],
) -> dict[str, Any]:
    """Reauthenticate a nonsealed R1c validation handoff without genome access."""

    expected_fields = {
        "schema",
        "cohort_role",
        "source",
        "selection_reads_model_outputs",
        "scientific_entry_authorized",
        "sealed_split_opened",
        "residual_validation_cohort_used",
        "handoff_reads_genome",
        "rows",
        "common_leg_rows",
        "preserved_outcome_fields",
        "row_count",
        "cohort_sha256",
        "ordered_row_ids_sha256",
        "ordered_gene_ids_sha256",
        "sorted_unique_gene_ids_sha256",
        "unique_gene_count",
        "ordered_partner_margin_levels_sha256",
        "source_legacy_cohort_sha256",
        "source_package_sha256",
        "completion_receipt_available",
        "preflight_construction_sha256",
        "record_sha256",
    }
    payload = {key: value for key, value in handoff.items() if key != "record_sha256"}
    source = handoff.get("source")
    if (
        set(handoff) != expected_fields
        or handoff.get("schema") != R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA
        or handoff.get("cohort_role") != R5A_R5C_PREFLIGHT_COHORT_ROLE
        or handoff.get("selection_reads_model_outputs") is not False
        or handoff.get("scientific_entry_authorized") is not False
        or handoff.get("sealed_split_opened") is not False
        or handoff.get("residual_validation_cohort_used") is not False
        or handoff.get("handoff_reads_genome") is not False
        or handoff.get("completion_receipt_available") is not False
        or handoff.get("preserved_outcome_fields")
        != ["P_clean", "P_under_D", "P_under_N"]
        or handoff.get("record_sha256") != canonical_sha256(payload)
        or not isinstance(source, Mapping)
        or set(source)
        != {
            "artifact",
            "manifest",
            "schema",
            "stage",
            "split",
            "n",
            "preconstruction_allowed_chromosomes",
            "verdict",
            "source_cohort_sha256",
            "source_selected_example_ids_sha256",
            "construction_version",
            "cohort_order_version",
            "matched_decoy_donor_caliper",
            "dyad_distance_caliper",
            "checkpoint_sha256",
            "source_package_sha256",
            "verified_successor_schema",
            "successor_attestation_sha256",
            "completion_receipt_available",
            "scientific_entry_authorized",
        }
        or source.get("schema") != R1C_TARGET_SCHEMA
        or source.get("stage") != "preflight"
        or source.get("split") != "val"
        or source.get("n") != 400
        or source.get("preconstruction_allowed_chromosomes")
        != ["chr11", "chr16", "chr19"]
        or source.get("verdict") != "PREFLIGHT_GO"
        or source.get("construction_version") != R1C_CONSTRUCTION_VERSION
        or source.get("cohort_order_version") != R1C_COHORT_ORDER_VERSION
        or source.get("matched_decoy_donor_caliper")
        != R1C_MATCHED_DECOY_DONOR_CALIPER
        or source.get("dyad_distance_caliper") != R1C_DYAD_DISTANCE_CALIPER
        or not _is_sha256(source.get("checkpoint_sha256"))
        or source.get("completion_receipt_available") is not False
        or source.get("scientific_entry_authorized") is not False
        or source.get("verified_successor_schema")
        != "gfmcd.r1c.verified_preflight_handoff/1"
        or not _is_sha256(source.get("successor_attestation_sha256"))
    ):
        raise R5CapstoneCohortError("R5 preflight handoff contract differs")

    artifact_value = source.get("artifact")
    manifest_value = source.get("manifest")
    if not isinstance(artifact_value, Mapping) or not isinstance(
        manifest_value, Mapping
    ):
        raise R5CapstoneCohortError("R5 preflight package identity is missing")
    artifact = _logical_file_identity(
        artifact_value, label="R1c preflight artifact"
    )
    sibling_manifest = _logical_file_identity(
        manifest_value, label="R1c preflight manifest"
    )
    source_package = {"artifact": artifact, "manifest": sibling_manifest}
    source_package_sha256 = canonical_sha256(source_package)
    rows_value = handoff.get("rows")
    common_value = handoff.get("common_leg_rows")
    if (
        not isinstance(rows_value, list)
        or len(rows_value) != 400
        or not isinstance(common_value, list)
        or len(common_value) != 400
    ):
        raise R5CapstoneCohortError("R5 preflight handoff must contain 400 rows")
    rows: list[dict[str, Any]] = []
    common_rows: list[dict[str, Any]] = []
    for index, (row, common) in enumerate(zip(rows_value, common_value, strict=True)):
        if not isinstance(row, Mapping) or not isinstance(common, Mapping):
            raise R5CapstoneCohortError("R5 preflight row is malformed")
        identity = r1c_serialized_row_identity(
            row, allowed_chromosomes=("chr11", "chr16", "chr19")
        )
        if dict(row) != identity or set(common) != {
            "example_id",
            "gene_id",
            "margin_levels",
        }:
            raise R5CapstoneCohortError(
                "R5 preflight handoff preserves non-common-leg outcomes"
            )
        levels = common.get("margin_levels")
        if (
            common.get("example_id") != identity["example_id"]
            or common.get("gene_id") != identity["gene_id"]
            or not isinstance(levels, Mapping)
            or set(levels) != {"P_clean", "P_under_D", "P_under_N"}
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in levels.values()
            )
        ):
            raise R5CapstoneCohortError(
                f"R5 preflight common-leg row {index} differs"
            )
        rows.append(identity)
        common_rows.append(
            {
                "example_id": common["example_id"],
                "gene_id": common["gene_id"],
                "margin_levels": {
                    name: float(levels[name])
                    for name in ("P_clean", "P_under_D", "P_under_N")
                },
            }
        )
    ordered_ids = [str(row["example_id"]) for row in rows]
    ordered_gene_ids = [str(row["gene_id"]) for row in rows]
    sorted_unique_gene_ids = sorted(set(ordered_gene_ids))
    source_cohort_payload = [
        {
            "example_id": row["example_id"],
            "gene_id": row["gene_id"],
            "decoy_donor": row["positions"]["matched_decoy_donor"],
            "decoy_donor_distance_mismatch": row["positions"][
                "matched_decoy_donor_distance_mismatch"
            ],
            "dyad_distance_mismatch": row["positions"]["dyad_distance_mismatch"],
        }
        for row in rows
    ]
    construction = {
        "source_schema": R1C_TARGET_SCHEMA,
        "construction_version": R1C_CONSTRUCTION_VERSION,
        "cohort_order_version": R1C_COHORT_ORDER_VERSION,
        "n": 400,
        "matched_decoy_donor_caliper": R1C_MATCHED_DECOY_DONOR_CALIPER,
        "dyad_distance_caliper": R1C_DYAD_DISTANCE_CALIPER,
        "preconstruction_allowed_chromosomes": ["chr11", "chr16", "chr19"],
        "artifact": artifact,
        "manifest": sibling_manifest,
        "source_package_sha256": source_package_sha256,
        "full_serialized_row_identities_sha256": canonical_sha256(rows),
        "ordered_partner_margin_levels_sha256": canonical_sha256(common_rows),
        "successor_attestation_sha256": source[
            "successor_attestation_sha256"
        ],
    }
    if (
        len(set(ordered_ids)) != 400
        or handoff.get("row_count") != 400
        or handoff.get("cohort_sha256") != canonical_sha256(rows)
        or handoff.get("ordered_row_ids_sha256") != canonical_sha256(ordered_ids)
        or handoff.get("ordered_gene_ids_sha256")
        != canonical_sha256(ordered_gene_ids)
        or handoff.get("sorted_unique_gene_ids_sha256")
        != canonical_sha256(sorted_unique_gene_ids)
        or handoff.get("unique_gene_count") != len(sorted_unique_gene_ids)
        or handoff.get("ordered_partner_margin_levels_sha256")
        != canonical_sha256(common_rows)
        or handoff.get("source_legacy_cohort_sha256")
        != canonical_sha256(source_cohort_payload)
        or handoff.get("source_package_sha256") != source_package_sha256
        or source.get("source_package_sha256") != source_package_sha256
        or source.get("source_cohort_sha256")
        != canonical_sha256(source_cohort_payload)
        or source.get("source_selected_example_ids_sha256")
        != canonical_sha256(ordered_ids)
        or handoff.get("preflight_construction_sha256")
        != canonical_sha256(construction)
    ):
        raise R5CapstoneCohortError("R5 preflight row identity differs")
    return {
        "schema": R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA,
        "verified": True,
        "cohort_role": R5A_R5C_PREFLIGHT_COHORT_ROLE,
        "artifact": artifact,
        "manifest": sibling_manifest,
        "rows": rows,
        "common_leg_rows": common_rows,
        "row_count": 400,
        "checkpoint_sha256": source["checkpoint_sha256"],
        "cohort_sha256": handoff["cohort_sha256"],
        "ordered_row_ids_sha256": handoff["ordered_row_ids_sha256"],
        "ordered_gene_ids_sha256": handoff["ordered_gene_ids_sha256"],
        "sorted_unique_gene_ids_sha256": handoff[
            "sorted_unique_gene_ids_sha256"
        ],
        "unique_gene_count": handoff["unique_gene_count"],
        "ordered_partner_margin_levels_sha256": handoff[
            "ordered_partner_margin_levels_sha256"
        ],
        "source_legacy_cohort_sha256": handoff[
            "source_legacy_cohort_sha256"
        ],
        "source_package_sha256": source_package_sha256,
        "verified_successor_schema": source["verified_successor_schema"],
        "successor_attestation_sha256": source[
            "successor_attestation_sha256"
        ],
        "completion_receipt_available": False,
        "scientific_entry_authorized": False,
        "preflight_construction_sha256": handoff[
            "preflight_construction_sha256"
        ],
        "residual_validation_cohort_used": False,
    }


def _r1c_preflight_pin_identity(kind: str) -> dict[str, Any]:
    values = {
        "artifact": (
            FROZEN_R1C_PREFLIGHT_ARTIFACT_LOGICAL_NAME,
            FROZEN_R1C_PREFLIGHT_ARTIFACT_SIZE_BYTES,
            FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256,
        ),
        "manifest": (
            FROZEN_R1C_PREFLIGHT_MANIFEST_LOGICAL_NAME,
            FROZEN_R1C_PREFLIGHT_MANIFEST_SIZE_BYTES,
            FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256,
        ),
    }
    logical_name, size_bytes, sha256 = values[kind]
    if logical_name is None or size_bytes is None or sha256 is None:
        raise R5CapstoneCohortError(
            "R1c preflight artifact/manifest pins remain unset; R5a/R5c "
            "preflight is sealed"
        )
    return _logical_file_identity(
        {
            "logical_name": logical_name,
            "size_bytes": size_bytes,
            "sha256": sha256,
        },
        label=f"pinned R1c preflight {kind}",
    )


def load_r1c_preflight_cohort_handoff(
    artifact_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    """Load the future exact two-file R1c validation-preflight package."""

    artifact_pin = _r1c_preflight_pin_identity("artifact")
    manifest_pin = _r1c_preflight_pin_identity("manifest")

    def load_exact(path_value: str | Path, pin: Mapping[str, Any], label: str):
        path = Path(path_value).expanduser().resolve(strict=True)
        observed = {
            "logical_name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        if observed != pin:
            raise R5CapstoneCohortError(f"{label} differs from its literal pin")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise R5CapstoneCohortError(f"{label} is not valid JSON") from exc
        if not isinstance(value, Mapping):
            raise R5CapstoneCohortError(f"{label} must contain a JSON object")
        return value, observed

    record, artifact_identity = load_exact(
        artifact_path, artifact_pin, "R1c preflight artifact"
    )
    manifest, manifest_identity = load_exact(
        manifest_path, manifest_pin, "R1c preflight manifest"
    )
    verified_successor = _invoke_r1c_successor_preflight_verifier(
        artifact_path, manifest_path
    )
    successor_attestation = verified_successor.get("attestation")
    if verified_successor.get("record") != record or not isinstance(
        successor_attestation, Mapping
    ):
        raise R5CapstoneCohortError(
            "R1c preflight verifier result differs from exact package bytes"
        )
    handoff = build_r5_preflight_cohort_handoff(
        record,
        manifest,
        artifact_identity=artifact_identity,
        manifest_identity=manifest_identity,
        successor_attestation=successor_attestation,
    )
    validate_r5_preflight_cohort_handoff(handoff)
    return handoff


def r5_gene_bucket(gene_id: str) -> int:
    digest = hashlib.sha256(f"{R5_SALT}|bucket|{gene_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 2


def r5_row_priority(example_id: str) -> tuple[str, str]:
    return (_sha256_text(f"{R5_SALT}|row|{example_id}"), str(example_id))


def r5_gene_priority(gene_id: str) -> tuple[str, str]:
    return (_sha256_text(f"{R5_SALT}|order|{gene_id}"), str(gene_id))


def _window(start: int) -> tuple[int, int, int, int]:
    return (start, start + 1, start + 2, start + 3)


def deterministic_adjacent_window(
    *,
    example_id: str,
    sequence: str,
    paired_donor: int,
    pseudo_donor: int,
    acceptor_g: int,
) -> tuple[int, int, int, int] | None:
    """Choose the frozen four-token adjacent control, or ``None`` if neither is valid."""

    side_bit = hashlib.sha256(
        f"{R5_ADJACENT_SALT}|{example_id}".encode()
    ).digest()[-1] & 1
    protected = (
        set(_window(paired_donor - 1))
        | set(_window(pseudo_donor - 1))
        | {int(acceptor_g)}
    )
    # Search a frozen local bank in increasing four-token shells.  The original two-window
    # design (shell 1 only) left just 61/58 genes after the predeclared exclusions.  The
    # eight-shell bank was chosen model-free before any R5 outcome and preserves the nearest
    # admissible control while recovering 142/130 genes on the validation census.
    for shell in range(1, 9):
        upstream = _window(paired_donor - (4 * shell + 1))
        downstream = _window(paired_donor + (4 * shell - 1))
        order = (
            (upstream, downstream)
            if side_bit == 0
            else (downstream, upstream)
        )
        for candidate in order:
            start, *_, end = candidate
            # The one-base-padded span must itself be in bounds because motif hygiene is
            # evaluated across both edges of the four-token control.
            if start < 1 or end + 1 >= len(sequence):
                continue
            if set(candidate) & protected:
                continue
            padded = sequence[start - 1 : end + 2]
            if "GT" in padded or "AG" in padded:
                continue
            return candidate
    return None


@dataclass(frozen=True)
class R5CohortRow:
    """One fully constructed, outcome-blind R5 row."""

    example: CouplingExample
    pseudo_donor: int
    adjacent_window: tuple[int, int, int, int] | None
    donor_alt: str = MOTIF_CLEAN_DONOR_ALT

    @property
    def acceptor_g(self) -> int:
        return acceptor_readout(self.example)

    @property
    def paired_window(self) -> tuple[int, int, int, int]:
        return _window(int(self.example.donor_pos) - 1)

    @property
    def pseudo_window(self) -> tuple[int, int, int, int]:
        return _window(int(self.pseudo_donor) - 1)

    @property
    def clean_sequence(self) -> str:
        return self.example.sequence

    @property
    def paired_corrupt_sequence(self) -> str:
        return substitute(
            self.example.sequence, int(self.example.donor_pos), self.donor_alt
        )

    @property
    def pseudo_corrupt_sequence(self) -> str:
        return substitute(self.example.sequence, int(self.pseudo_donor), self.donor_alt)

    def identity_record(self) -> dict[str, Any]:
        return {
            "example_id": str(self.example.example_id),
            "gene_id": str(self.example.gene_id),
            "chrom": str(self.example.chrom),
            "strand": str(self.example.strand),
            "paired_donor": int(self.example.donor_pos),
            "pseudo_donor": int(self.pseudo_donor),
            "acceptor_g": int(self.acceptor_g),
            "paired_window": list(self.paired_window),
            "pseudo_window": list(self.pseudo_window),
            "adjacent_window": (
                None
                if self.adjacent_window is None
                else list(self.adjacent_window)
            ),
            "donor_alt": self.donor_alt,
        }

    def construction_record(self) -> dict[str, Any]:
        return self.identity_record() | {
            "clean_sequence_sha256": _sha256_text(self.clean_sequence),
            "paired_corrupt_sequence_sha256": _sha256_text(
                self.paired_corrupt_sequence
            ),
            "pseudo_corrupt_sequence_sha256": _sha256_text(
                self.pseudo_corrupt_sequence
            ),
        }


def _normalise_chromosome(value: Any) -> str:
    text = str(value)
    return text if text.startswith("chr") else f"chr{text}"


def _r1c_matched_decoy_donor(example: CouplingExample) -> int | None:
    """Replay the exact model-free R1c donor comparator rule."""

    if example.next_acceptor_pos is None:
        return None
    partner = int(example.next_acceptor_pos)
    donor = int(example.donor_pos)
    target_distance = partner - donor
    candidates = [
        int(candidate)
        for candidate in example.other_gt
        if int(candidate) < partner - 10 and abs(int(candidate) - donor) > 10
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: abs((partner - candidate) - target_distance),
    )


def _materialize_r5_authenticated_rows(
    examples: Sequence[CouplingExample],
    handoff: Mapping[str, Any],
    *,
    cohort_kind: str,
) -> tuple[tuple[R5CohortRow, ...], dict[str, Any]]:
    """Materialize one authenticated R1c cohort without performing any I/O."""

    if cohort_kind == "target":
        verified = validate_r5_capstone_cohort_handoff(handoff)
        expected_split = "test"
        materialization_schema = R5_CAPSTONE_ROW_MATERIALIZATION_SCHEMA
        source_construction_field = "capstone_construction_sha256"
        ledger_required = True
    elif cohort_kind == "preflight":
        verified = validate_r5_preflight_cohort_handoff(handoff)
        expected_split = "val"
        materialization_schema = R5_PREFLIGHT_ROW_MATERIALIZATION_SCHEMA
        source_construction_field = "preflight_construction_sha256"
        ledger_required = False
    else:
        raise ValueError("cohort_kind must be 'preflight' or 'target'")
    input_by_id: dict[str, CouplingExample] = {}
    for example in examples:
        example_id = str(example.example_id)
        if not example_id or example_id in input_by_id:
            raise R5CapstoneCohortError(
                "capstone source examples must have unique example IDs"
            )
        input_by_id[example_id] = example

    authenticated_rows = verified["rows"]
    missing_ids = [
        str(identity["example_id"])
        for identity in authenticated_rows
        if str(identity["example_id"]) not in input_by_id
    ]
    if missing_ids:
        raise R5CapstoneCohortError(
            "capstone source examples are missing authenticated handoff IDs"
        )

    materialized: list[R5CohortRow] = []
    unavailable_adjacent_ids: list[str] = []
    for identity in authenticated_rows:
        example_id = str(identity["example_id"])
        example = input_by_id[example_id]
        positions = identity["positions"]
        prior_acceptor = int(example.acceptor_pos)
        donor = int(example.donor_pos)
        partner = (
            None
            if example.next_acceptor_pos is None
            else int(example.next_acceptor_pos)
        )
        pseudo = _r1c_matched_decoy_donor(example)
        if example.intron_length_down is None:
            dyad_mismatch = None
        else:
            dyad_mismatch = abs(
                int(example.exon_length) - int(example.intron_length_down)
            )
        donor_mismatch = None if pseudo is None else abs(pseudo - donor)
        derived_positions = {
            "prior_real_acceptor": prior_acceptor,
            "paired_donor": donor,
            "partner_acceptor": partner,
            "matched_decoy_donor": pseudo,
            "matched_decoy_donor_distance_mismatch": donor_mismatch,
            "dyad_distance_mismatch": dyad_mismatch,
        }
        if (
            str(example.gene_id) != identity["gene_id"]
            or _normalise_chromosome(example.chrom) != identity["chrom"]
            or str(example.strand) != identity["strand"]
            or str(example.split) != expected_split
        ):
            raise R5CapstoneCohortError(
                f"{example_id}: authenticated gene/chrom/strand/split identity differs"
            )
        if derived_positions != positions:
            raise R5CapstoneCohortError(
                f"{example_id}: authenticated R1c coordinates differ"
            )
        assert partner is not None and pseudo is not None
        sequence = str(example.sequence)
        if (
            donor < 1
            or donor + 2 >= len(sequence)
            or pseudo < 1
            or pseudo + 2 >= len(sequence)
            or prior_acceptor < 2
            or prior_acceptor > len(sequence)
            or partner < 2
            or partner > len(sequence)
            or sequence[donor : donor + 2] != "GT"
            or sequence[pseudo : pseudo + 2] != "GT"
            or sequence[prior_acceptor - 2 : prior_acceptor] != "AG"
            or sequence[partner - 2 : partner] != "AG"
        ):
            raise R5CapstoneCohortError(
                f"{example_id}: authenticated GT/AG motif contract differs"
            )
        if (
            donor_mismatch > R1C_MATCHED_DECOY_DONOR_CALIPER
            or dyad_mismatch is None
            or dyad_mismatch > R1C_DYAD_DISTANCE_CALIPER
        ):
            raise R5CapstoneCohortError(
                f"{example_id}: authenticated R1c caliper contract differs"
            )
        if planted_donor_motifs(sequence, donor, MOTIF_CLEAN_DONOR_ALT) or (
            planted_donor_motifs(sequence, pseudo, MOTIF_CLEAN_DONOR_ALT)
        ):
            raise R5CapstoneCohortError(
                f"{example_id}: GT-to-AC edit plants a splice motif"
            )
        adjacent = deterministic_adjacent_window(
            example_id=example_id,
            sequence=sequence,
            paired_donor=donor,
            pseudo_donor=pseudo,
            acceptor_g=partner - 1,
        )
        if adjacent is None:
            unavailable_adjacent_ids.append(example_id)
        materialized.append(
            R5CohortRow(
                example=example,
                pseudo_donor=pseudo,
                adjacent_window=adjacent,
                donor_alt=MOTIF_CLEAN_DONOR_ALT,
            )
        )

    row_identities = [row.identity_record() for row in materialized]
    row_constructions = [row.construction_record() for row in materialized]
    ordered_ids = [str(row.example.example_id) for row in materialized]
    ordered_gene_ids = [str(row.example.gene_id) for row in materialized]
    materialized_rows_sha256 = canonical_sha256(row_identities)
    materialized_constructions_sha256 = canonical_sha256(row_constructions)
    construction = {
        "schema": materialization_schema,
        "cohort_kind": cohort_kind,
        "handoff_record_sha256": handoff["record_sha256"],
        "handoff_cohort_sha256": handoff["cohort_sha256"],
        "handoff_source_construction_sha256": handoff[
            source_construction_field
        ],
        "source_package_sha256": handoff["source_package_sha256"],
        "row_count": 400,
        "ordered_row_ids_sha256": canonical_sha256(ordered_ids),
        "donor_alt": MOTIF_CLEAN_DONOR_ALT,
        "matched_decoy_donor_caliper": R1C_MATCHED_DECOY_DONOR_CALIPER,
        "dyad_distance_caliper": R1C_DYAD_DISTANCE_CALIPER,
        "adjacent_window_required": False,
        "materialized_rows_sha256": materialized_rows_sha256,
        "materialized_constructions_sha256": materialized_constructions_sha256,
    }
    payload = {
        "schema": materialization_schema,
        "cohort_kind": cohort_kind,
        "handoff_schema": handoff["schema"],
        "handoff_record_sha256": handoff["record_sha256"],
        "handoff_cohort_sha256": handoff["cohort_sha256"],
        "handoff_source_construction_sha256": handoff[
            source_construction_field
        ],
        "source_package_sha256": handoff["source_package_sha256"],
        "input_example_count": len(examples),
        "row_count": len(materialized),
        "unique_gene_count": len(set(ordered_gene_ids)),
        "ordered_row_ids_sha256": canonical_sha256(ordered_ids),
        "ordered_gene_ids_sha256": canonical_sha256(ordered_gene_ids),
        "materialized_rows_sha256": materialized_rows_sha256,
        "materialized_constructions_sha256": materialized_constructions_sha256,
        "materialization_construction_sha256": canonical_sha256(construction),
        "handoff_order_preserved": True,
        "rows_filtered_or_reordered": False,
        "mapping_uses_authenticated_example_ids_only": True,
        "selection_reads_model_outputs": False,
        "repeated_genes_allowed": True,
        "adjacent_window_required": False,
        "adjacent_window_available_count": (
            len(materialized) - len(unavailable_adjacent_ids)
        ),
        "adjacent_window_unavailable_count": len(unavailable_adjacent_ids),
        "adjacent_window_unavailable_example_ids_sha256": canonical_sha256(
            unavailable_adjacent_ids
        ),
        "materializer_performs_io": False,
        "materializer_loads_model": False,
        "caller_must_supply_examples_after_target_ledger_authorization": (
            ledger_required
        ),
        "target_ledger_authorization_verified_here": False,
    }
    return tuple(materialized), payload | {"record_sha256": canonical_sha256(payload)}


def materialize_r5_preflight_rows(
    examples: Sequence[CouplingExample],
    handoff: Mapping[str, Any],
) -> tuple[tuple[R5CohortRow, ...], dict[str, Any]]:
    """Map the exact nonsealed R1c validation rows for R5a/R5c preflight.

    ``examples`` is the caller-loaded unscored validation census.  The shared function
    performs no genome/model I/O and makes no scientific-entry decision.
    """

    return _materialize_r5_authenticated_rows(
        examples, handoff, cohort_kind="preflight"
    )


def materialize_r5_capstone_rows(
    examples: Sequence[CouplingExample],
    handoff: Mapping[str, Any],
) -> tuple[tuple[R5CohortRow, ...], dict[str, Any]]:
    """Map exact sealed R1c target rows after the caller spends its stage receipt.

    The function is pure and does not verify the external ledger itself.  It maps all
    400 IDs in authenticated order, replays all coordinate/motif invariants, and retains
    rows whose optional adjacent diagnostic is unavailable.
    """

    return _materialize_r5_authenticated_rows(
        examples, handoff, cohort_kind="target"
    )


def evaluate_r5_row(
    example: CouplingExample,
    pseudo_donor: int | None,
    *,
    excluded_gene_ids: set[str] | frozenset[str],
    donor_alt: str = MOTIF_CLEAN_DONOR_ALT,
    allowed_chromosomes: Sequence[str] = R5_VALIDATION_CHROMOSOMES,
    donor_pseudo_caliper: int = R5_DONOR_PSEUDO_CALIPER,
) -> tuple[R5CohortRow | None, str | None]:
    """Evaluate every outcome-blind eligibility rule and return one failure reason."""

    if str(example.chrom) not in set(map(str, allowed_chromosomes)):
        return None, "chromosome_not_allowed"
    if str(example.gene_id) in excluded_gene_ids:
        return None, "excluded_gene"
    if example.next_acceptor_pos is None:
        return None, "missing_intron_mate_acceptor"
    donor = int(example.donor_pos)
    q = acceptor_readout(example)
    if donor < 1 or donor + 2 >= len(example.sequence):
        return None, "paired_donor_out_of_bounds"
    if q < 1 or q >= len(example.sequence):
        return None, "acceptor_readout_out_of_bounds"
    if example.sequence[donor : donor + 2] != "GT":
        return None, "paired_donor_not_GT"
    if example.sequence[q - 1 : q + 1] != "AG":
        return None, "intron_mate_acceptor_not_AG"
    if pseudo_donor is None:
        return None, "missing_pseudo_donor"
    pseudo = int(pseudo_donor)
    if pseudo == donor:
        return None, "pseudo_donor_equals_paired_donor"
    if pseudo < 1 or pseudo + 2 >= len(example.sequence):
        return None, "pseudo_donor_out_of_bounds"
    if abs(pseudo - donor) > int(donor_pseudo_caliper):
        return None, "pseudo_donor_outside_40nt_caliper"
    if example.sequence[pseudo : pseudo + 2] != "GT":
        return None, "pseudo_donor_not_GT"
    if planted_donor_motifs(example.sequence, donor, donor_alt):
        return None, "paired_donor_edit_plants_splice_motif"
    if planted_donor_motifs(example.sequence, pseudo, donor_alt):
        return None, "pseudo_donor_edit_plants_splice_motif"
    adjacent = deterministic_adjacent_window(
        example_id=str(example.example_id),
        sequence=example.sequence,
        paired_donor=donor,
        pseudo_donor=pseudo,
        acceptor_g=q,
    )
    if adjacent is None:
        return None, "missing_admissible_adjacent_window"
    return (
        R5CohortRow(
            example=example,
            pseudo_donor=pseudo,
            adjacent_window=adjacent,
            donor_alt=donor_alt,
        ),
        None,
    )


def _pseudo_for(
    pseudo_donors: Mapping[str, int | None], example: CouplingExample
) -> int | None:
    value = pseudo_donors.get(str(example.example_id))
    return None if value is None else int(value)


def build_r5_reservations(
    examples: Sequence[CouplingExample],
    pseudo_donors: Mapping[str, int | None],
    *,
    historical_r3_gene_ids: Sequence[str],
    r3c_gene_ids: Sequence[str],
    r3d_registry_gene_ids: Sequence[str] = (),
    r3d_target_gene_ids: Sequence[str] = (),
    rows_per_bucket: int = R5_ROWS_PER_BUCKET,
    require_capacity: bool = True,
) -> tuple[dict[int, tuple[R5CohortRow, ...]], dict[str, Any]]:
    """Build two deterministic residual-validation reservations and their audit.

    Bucket 0 is R5b derivation; bucket 1 is R5b validation followed by the frozen union
    refit.  Neither bucket is an R5a/R5c preflight or biological-target cohort.
    """

    if isinstance(rows_per_bucket, bool) or not isinstance(rows_per_bucket, int):
        raise TypeError("rows_per_bucket must be an integer")
    if rows_per_bucket < 1:
        raise ValueError("rows_per_bucket must be positive")
    source_gene_ids = {
        "historical_r3": sorted(set(map(str, historical_r3_gene_ids))),
        "r3c_fixed_cohorts": sorted(set(map(str, r3c_gene_ids))),
        "r3d_direct_exposure_registry": sorted(
            set(map(str, r3d_registry_gene_ids))
        ),
        "r3d_target": sorted(set(map(str, r3d_target_gene_ids))),
    }
    source_names = list(source_gene_ids)
    pairwise_intersections: dict[str, int] = {}
    for index, left in enumerate(source_names):
        for right in source_names[index + 1 :]:
            pairwise_intersections[f"{left}__{right}"] = len(
                set(source_gene_ids[left]) & set(source_gene_ids[right])
            )
    if any(pairwise_intersections.values()):
        raise RuntimeError("R5 prior-exposure source registries must be pairwise disjoint")
    historical = source_gene_ids["historical_r3"]
    r3c = source_gene_ids["r3c_fixed_cohorts"]
    r3d_registry = source_gene_ids["r3d_direct_exposure_registry"]
    r3d_target = source_gene_ids["r3d_target"]
    source_sets = {
        name: frozenset(values) for name, values in source_gene_ids.items()
    }
    excluded = frozenset().union(*source_sets.values())
    validation_row_overlap = {
        "r3d_direct_exposure_registry": sum(
            str(example.gene_id) in source_sets["r3d_direct_exposure_registry"]
            for example in examples
        ),
        "r3d_target": sum(
            str(example.gene_id) in source_sets["r3d_target"]
            for example in examples
        ),
    }
    if any(validation_row_overlap.values()):
        raise RuntimeError(
            "R3d exposure source overlaps an R5 validation census row"
        )
    exclusions: Counter[str] = Counter()
    eligible_by_gene: dict[str, list[R5CohortRow]] = {}
    for example in examples:
        row, reason = evaluate_r5_row(
            example,
            _pseudo_for(pseudo_donors, example),
            excluded_gene_ids=excluded,
        )
        if row is None:
            exclusions[str(reason)] += 1
            continue
        eligible_by_gene.setdefault(str(example.gene_id), []).append(row)

    chosen_by_gene = {
        gene: min(rows, key=lambda row: r5_row_priority(row.example.example_id))
        for gene, rows in eligible_by_gene.items()
    }
    available: dict[int, list[R5CohortRow]] = {0: [], 1: []}
    for gene, row in chosen_by_gene.items():
        available[r5_gene_bucket(gene)].append(row)
    for bucket in (0, 1):
        available[bucket].sort(key=lambda row: r5_gene_priority(row.example.gene_id))

    if require_capacity:
        short = {
            bucket: len(rows)
            for bucket, rows in available.items()
            if len(rows) < rows_per_bucket
        }
        if short:
            raise RuntimeError(
                "R5 one-row-per-gene reservation capacity is insufficient: "
                + ", ".join(
                    f"bucket {bucket} has {count}, requires {rows_per_bucket}"
                    for bucket, count in sorted(short.items())
                )
            )

    selected = {
        bucket: tuple(available[bucket][:rows_per_bucket]) for bucket in (0, 1)
    }
    selected_audit: dict[str, Any] = {}
    for bucket in (0, 1):
        rows = selected[bucket]
        identity = [row.identity_record() for row in rows]
        constructions = [row.construction_record() for row in rows]
        genes = [str(row.example.gene_id) for row in rows]
        selected_audit[str(bucket)] = {
            "rows": len(rows),
            "genes": len(set(genes)),
            "both_strands": {str(row.example.strand) for row in rows} == {"+", "-"},
            "gene_ids": genes,
            "gene_ids_sha256": canonical_sha256(genes),
            "ordered_rows_sha256": canonical_sha256(identity),
            "construction_sha256": canonical_sha256(constructions),
        }
    audit = {
        "schema": R5_COHORT_SCHEMA,
        "cohort_role": R5_RESIDUAL_VALIDATION_COHORT_ROLE,
        "authorized_stage_consumers": list(
            R5_RESIDUAL_VALIDATION_STAGE_CONSUMERS
        ),
        "bucket_roles": dict(R5_RESIDUAL_VALIDATION_BUCKET_ROLES),
        "r5a_r5c_preflight_authorized": False,
        "r5c_capstone_authorized": False,
        "r5c_required_cohort_role": R5C_REQUIRED_COHORT_ROLE,
        "selection_reads_model_outputs": False,
        "allowed_chromosomes": list(R5_VALIDATION_CHROMOSOMES),
        "sealed_chr8_chr9_opened": False,
        "donor_pseudo_caliper": R5_DONOR_PSEUDO_CALIPER,
        "adjacent_window_policy": {
            "salt": R5_ADJACENT_SALT,
            "shells": 8,
            "tokens_per_window": 4,
            "maximum_absolute_offset_from_donor": 34,
            "selection": "nearest admissible shell; hash-fixed side order within shell",
        },
        "rows_per_bucket": rows_per_bucket,
        "input_rows": len(examples),
        "historical_r3_gene_count": len(historical),
        "historical_r3_gene_ids_sha256": canonical_sha256(historical),
        "r3c_gene_count": len(r3c),
        "r3c_gene_ids_sha256": canonical_sha256(r3c),
        "r3d_direct_exposure_registry_gene_count": len(r3d_registry),
        "r3d_direct_exposure_registry_gene_ids_sha256": canonical_sha256(
            r3d_registry
        ),
        "r3d_target_gene_count": len(r3d_target),
        "r3d_target_gene_ids_sha256": canonical_sha256(r3d_target),
        "r3d_registry_and_target_union_gene_count": len(
            set(r3d_registry) | set(r3d_target)
        ),
        "r3d_registry_and_target_union_gene_ids_sha256": canonical_sha256(
            sorted(set(r3d_registry) | set(r3d_target))
        ),
        "prior_exposure_sets_pairwise_disjoint": True,
        "prior_exposure_pairwise_intersection_counts": pairwise_intersections,
        "prior_exposure_union_gene_count": len(excluded),
        "prior_exposure_union_gene_ids_sha256": canonical_sha256(sorted(excluded)),
        "combined_excluded_gene_count": len(excluded),
        "combined_excluded_gene_ids_sha256": canonical_sha256(sorted(excluded)),
        "r3d_validation_row_overlap_by_source": validation_row_overlap,
        "global_gene_freshness_claimed": False,
        "exclusions": dict(sorted(exclusions.items())),
        "eligible_rows": sum(len(rows) for rows in eligible_by_gene.values()),
        "eligible_genes": len(chosen_by_gene),
        "available_genes_by_bucket": {
            str(bucket): len(available[bucket]) for bucket in (0, 1)
        },
        "selected_by_bucket": selected_audit,
    }
    return selected, audit


def validate_r5_reservation_audit(
    audit: Mapping[str, Any], *, rows_per_bucket: int = R5_ROWS_PER_BUCKET
) -> None:
    """Fail closed on the cohort properties needed by all three R5 stages."""

    if audit.get("schema") != R5_COHORT_SCHEMA:
        raise ValueError("R5 cohort audit schema differs")
    if (
        audit.get("cohort_role") != R5_RESIDUAL_VALIDATION_COHORT_ROLE
        or audit.get("authorized_stage_consumers")
        != list(R5_RESIDUAL_VALIDATION_STAGE_CONSUMERS)
        or audit.get("bucket_roles") != R5_RESIDUAL_VALIDATION_BUCKET_ROLES
        or audit.get("r5a_r5c_preflight_authorized") is not False
        or audit.get("r5c_capstone_authorized") is not False
        or audit.get("r5c_required_cohort_role") != R5C_REQUIRED_COHORT_ROLE
    ):
        raise ValueError(
            "R5b residual-validation cohort cannot authorize R5a/R5c stages"
        )
    if audit.get("selection_reads_model_outputs") is not False:
        raise ValueError("R5 cohort selection must be outcome-blind")
    if audit.get("sealed_chr8_chr9_opened") is not False:
        raise ValueError("R5 cohort must not open chr8/chr9")
    if audit.get("allowed_chromosomes") != list(R5_VALIDATION_CHROMOSOMES):
        raise ValueError("R5 cohort chromosome registry differs")
    if audit.get("rows_per_bucket") != rows_per_bucket:
        raise ValueError("R5 cohort reservation size differs")
    pairwise_counts = audit.get("prior_exposure_pairwise_intersection_counts")
    if (
        audit.get("prior_exposure_sets_pairwise_disjoint") is not True
        or not isinstance(pairwise_counts, Mapping)
        or set(pairwise_counts) != set(R5_PRIOR_EXPOSURE_PAIRWISE_INTERSECTIONS)
        or any(type(value) is not int or value != 0 for value in pairwise_counts.values())
    ):
        raise ValueError("R5 prior-exposure registries must be pairwise disjoint")
    r3d_overlap = audit.get("r3d_validation_row_overlap_by_source")
    if (
        not isinstance(r3d_overlap, Mapping)
        or set(r3d_overlap) != {
            "r3d_direct_exposure_registry",
            "r3d_target",
        }
        or any(type(value) is not int or value != 0 for value in r3d_overlap.values())
    ):
        raise ValueError("R5 validation rows must not overlap either R3d source")
    if audit.get("global_gene_freshness_claimed") is not False:
        raise ValueError("R5 cohort must not claim global gene freshness")
    if (
        audit.get("combined_excluded_gene_count")
        != audit.get("prior_exposure_union_gene_count")
        or audit.get("combined_excluded_gene_ids_sha256")
        != audit.get("prior_exposure_union_gene_ids_sha256")
        or audit.get("r3d_registry_and_target_union_gene_count")
        != audit.get("r3d_direct_exposure_registry_gene_count", 0)
        + audit.get("r3d_target_gene_count", 0)
        or audit.get("prior_exposure_union_gene_count")
        != sum(
            audit.get(key, 0)
            for key in (
                "historical_r3_gene_count",
                "r3c_gene_count",
                "r3d_direct_exposure_registry_gene_count",
                "r3d_target_gene_count",
            )
        )
    ):
        raise ValueError("R5 prior-exposure union identity is incoherent")
    for key in (
        "historical_r3_gene_ids_sha256",
        "r3c_gene_ids_sha256",
        "r3d_direct_exposure_registry_gene_ids_sha256",
        "r3d_target_gene_ids_sha256",
        "r3d_registry_and_target_union_gene_ids_sha256",
        "prior_exposure_union_gene_ids_sha256",
    ):
        value = audit.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"R5 cohort lacks {key}")
    if audit.get("adjacent_window_policy") != {
        "salt": R5_ADJACENT_SALT,
        "shells": 8,
        "tokens_per_window": 4,
        "maximum_absolute_offset_from_donor": 34,
        "selection": "nearest admissible shell; hash-fixed side order within shell",
    }:
        raise ValueError("R5 adjacent-window bank differs")
    selected = audit.get("selected_by_bucket")
    if not isinstance(selected, Mapping) or set(selected) != {"0", "1"}:
        raise ValueError("R5 cohort lacks both fixed reservations")
    for bucket in ("0", "1"):
        record = selected[bucket]
        if (
            record.get("rows") != rows_per_bucket
            or record.get("genes") != rows_per_bucket
            or record.get("both_strands") is not True
        ):
            raise ValueError(
                f"R5 bucket {bucket} must contain {rows_per_bucket} genes and both strands"
            )
        for key in (
            "gene_ids_sha256",
            "ordered_rows_sha256",
            "construction_sha256",
        ):
            value = record.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"R5 bucket {bucket} lacks {key}")


def r5_census_identity(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the literal model-free identity that every R5 runner must bind."""

    selected = audit["selected_by_bucket"]
    return {
        "schema": audit["schema"],
        "cohort_role": audit["cohort_role"],
        "authorized_stage_consumers": audit["authorized_stage_consumers"],
        "bucket_roles": audit["bucket_roles"],
        "r5a_r5c_preflight_authorized": audit[
            "r5a_r5c_preflight_authorized"
        ],
        "r5c_capstone_authorized": audit["r5c_capstone_authorized"],
        "r5c_required_cohort_role": audit["r5c_required_cohort_role"],
        "input_rows": audit["input_rows"],
        "historical_r3_gene_count": audit["historical_r3_gene_count"],
        "historical_r3_gene_ids_sha256": audit[
            "historical_r3_gene_ids_sha256"
        ],
        "r3c_gene_count": audit["r3c_gene_count"],
        "r3c_gene_ids_sha256": audit["r3c_gene_ids_sha256"],
        "r3d_direct_exposure_registry_gene_count": audit[
            "r3d_direct_exposure_registry_gene_count"
        ],
        "r3d_direct_exposure_registry_gene_ids_sha256": audit[
            "r3d_direct_exposure_registry_gene_ids_sha256"
        ],
        "r3d_target_gene_count": audit["r3d_target_gene_count"],
        "r3d_target_gene_ids_sha256": audit["r3d_target_gene_ids_sha256"],
        "r3d_registry_and_target_union_gene_count": audit[
            "r3d_registry_and_target_union_gene_count"
        ],
        "r3d_registry_and_target_union_gene_ids_sha256": audit[
            "r3d_registry_and_target_union_gene_ids_sha256"
        ],
        "prior_exposure_sets_pairwise_disjoint": audit[
            "prior_exposure_sets_pairwise_disjoint"
        ],
        "prior_exposure_pairwise_intersection_counts": audit[
            "prior_exposure_pairwise_intersection_counts"
        ],
        "prior_exposure_union_gene_count": audit[
            "prior_exposure_union_gene_count"
        ],
        "prior_exposure_union_gene_ids_sha256": audit[
            "prior_exposure_union_gene_ids_sha256"
        ],
        "combined_excluded_gene_count": audit["combined_excluded_gene_count"],
        "combined_excluded_gene_ids_sha256": audit[
            "combined_excluded_gene_ids_sha256"
        ],
        "r3d_validation_row_overlap_by_source": audit[
            "r3d_validation_row_overlap_by_source"
        ],
        "global_gene_freshness_claimed": audit["global_gene_freshness_claimed"],
        "exclusions": audit["exclusions"],
        "eligible_rows": audit["eligible_rows"],
        "eligible_genes": audit["eligible_genes"],
        "available_genes_by_bucket": audit["available_genes_by_bucket"],
        "selected_by_bucket": {
            bucket: {
                key: selected[bucket][key]
                for key in (
                    "rows",
                    "genes",
                    "both_strands",
                    "gene_ids_sha256",
                    "ordered_rows_sha256",
                    "construction_sha256",
                )
            }
            for bucket in ("0", "1")
        },
    }


def validate_expected_r5_census_identity(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Abort if the full validation census drifts from the pre-outcome R5 identity."""

    observed = r5_census_identity(audit)
    if observed != R5_EXPECTED_CENSUS_IDENTITY:
        raise RuntimeError(
            "R5 model-free census identity drifted: "
            f"observed={observed}, expected={R5_EXPECTED_CENSUS_IDENTITY}"
        )
    return observed


__all__ = [
    "FROZEN_R1C_PREFLIGHT_ARTIFACT_LOGICAL_NAME",
    "FROZEN_R1C_PREFLIGHT_ARTIFACT_SHA256",
    "FROZEN_R1C_PREFLIGHT_ARTIFACT_SIZE_BYTES",
    "FROZEN_R1C_PREFLIGHT_MANIFEST_LOGICAL_NAME",
    "FROZEN_R1C_PREFLIGHT_MANIFEST_SHA256",
    "FROZEN_R1C_PREFLIGHT_MANIFEST_SIZE_BYTES",
    "FROZEN_R1C_TARGET_ARTIFACT_LOGICAL_NAME",
    "FROZEN_R1C_TARGET_ARTIFACT_SHA256",
    "FROZEN_R1C_TARGET_ARTIFACT_SIZE_BYTES",
    "FROZEN_R1C_TARGET_ATTEMPT_LOGICAL_NAME",
    "FROZEN_R1C_TARGET_ATTEMPT_SHA256",
    "FROZEN_R1C_TARGET_ATTEMPT_SIZE_BYTES",
    "FROZEN_R1C_TARGET_COMPLETION_LOGICAL_NAME",
    "FROZEN_R1C_TARGET_COMPLETION_SHA256",
    "FROZEN_R1C_TARGET_COMPLETION_SIZE_BYTES",
    "FROZEN_R1C_TARGET_MANIFEST_LOGICAL_NAME",
    "FROZEN_R1C_TARGET_MANIFEST_SHA256",
    "FROZEN_R1C_TARGET_MANIFEST_SIZE_BYTES",
    "R1C_CELL_NAMES",
    "R1C_BIOLOGICAL_TARGET_IDENTITY_SCHEMA",
    "R1C_CLAIM_BOUNDARY",
    "R1C_COHORT_ORDER_VERSION",
    "R1C_CONFIRMED_BEHAVIOR_VERDICTS",
    "R1C_CONSTRUCTION_VERSION",
    "R1C_DYAD_DISTANCE_CALIPER",
    "R1C_MATCHED_DECOY_DONOR_CALIPER",
    "R1C_PRIMARY_FAMILY_NAMES",
    "R1C_ROW_POSITION_FIELDS",
    "R1C_TARGET_ARTIFACT_LOGICAL_NAME",
    "R1C_TARGET_ARTIFACT_SHA256",
    "R1C_TARGET_ARTIFACT_SIZE_BYTES",
    "R1C_TARGET_ATTEMPT_LOGICAL_NAME",
    "R1C_TARGET_ATTEMPT_SCHEMA",
    "R1C_TARGET_ATTEMPT_SHA256",
    "R1C_TARGET_ATTEMPT_SIZE_BYTES",
    "R1C_TARGET_ATTEMPT_STATUS",
    "R1C_TARGET_COMPLETION_LOGICAL_NAME",
    "R1C_TARGET_COMPLETION_SCHEMA",
    "R1C_TARGET_COMPLETION_SHA256",
    "R1C_TARGET_COMPLETION_SIZE_BYTES",
    "R1C_TARGET_COMPLETION_STATUS",
    "R1C_TARGET_MANIFEST_LOGICAL_NAME",
    "R1C_TARGET_MANIFEST_SHA256",
    "R1C_TARGET_MANIFEST_SIZE_BYTES",
    "R1C_TARGET_SCHEMA",
    "R5_ADJACENT_SALT",
    "R5_CAPSTONE_COHORT_HANDOFF_SCHEMA",
    "R5_CAPSTONE_ROW_MATERIALIZATION_SCHEMA",
    "R5_VERIFIED_R1C_COMMON_LEGS_SCHEMA",
    "R5CapstoneCohortError",
    "R5_COHORT_SCHEMA",
    "R5_DONOR_PSEUDO_CALIPER",
    "R5_EXPECTED_CENSUS_IDENTITY",
    "R5_EXPECTED_SOURCE_IDENTITY",
    "R5_PRIOR_EXPOSURE_PAIRWISE_INTERSECTIONS",
    "R5_PREFLIGHT_COHORT_HANDOFF_SCHEMA",
    "R5_PREFLIGHT_ROW_MATERIALIZATION_SCHEMA",
    "R5A_R5C_PREFLIGHT_COHORT_ROLE",
    "R5_RESIDUAL_VALIDATION_COHORT_ROLE",
    "R5_RESIDUAL_VALIDATION_ROWS_PER_BUCKET",
    "R5_RESIDUAL_VALIDATION_STAGE_CONSUMERS",
    "R5_RESIDUAL_VALIDATION_BUCKET_ROLES",
    "R5_ROWS_PER_BUCKET",
    "R5_SALT",
    "R5_VALIDATION_CHROMOSOMES",
    "R5C_REQUIRED_COHORT_ROLE",
    "R5CohortRow",
    "build_r5_capstone_cohort_handoff",
    "build_r5_preflight_cohort_handoff",
    "build_r1c_verified_target_common_legs",
    "build_r5_reservations",
    "canonical_sha256",
    "deterministic_adjacent_window",
    "evaluate_r5_row",
    "load_r1c_capstone_cohort_handoff",
    "load_r1c_preflight_cohort_handoff",
    "load_r1c_verified_target_common_legs",
    "materialize_r5_capstone_rows",
    "materialize_r5_preflight_rows",
    "r1c_serialized_row_identity",
    "r5_gene_bucket",
    "r5_census_identity",
    "r5_gene_priority",
    "r5_row_priority",
    "r5_source_identity",
    "validate_expected_r5_source_identity",
    "validate_r5_capstone_cohort_handoff",
    "validate_r5_preflight_cohort_handoff",
    "validate_r5_reservation_audit",
    "validate_expected_r5_census_identity",
]
