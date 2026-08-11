"""Stage gates and immutable seals for P3 register-relocation confirmation.

This module contains no model forward pass and no scientific outcome.  It binds the
audited Phase-II authorization to a separately capable Phase-III development instrument,
then produces the only package a reserve runner may consume.  The P3 authorization is
deliberately independent of the optional P10 branch.
"""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from gfmcd.provenance import sha256_file
from gfmcd.splicing.pbc_register_candidates import validate_candidate_screen
from gfmcd.splicing.pbc_register_mediation import validate_candidate_freeze
from gfmcd.splicing.pbc_register_relocation import (
    OPPOSITE_PARTIAL_SPECIFICITY_SEED,
    PARTIAL_SPECIFICITY_SEED,
)


PHASE2_ANALYSIS_SCHEMA = "gfmcd.pbc.carrier_factorial_analysis.v1"
PHASE2_AUDIT_SCHEMA = "gfmcd.pbc.carrier_factorial_audit.v1"
REGISTER_PREFLIGHT_SCHEMA = "gfmcd.pbc.p3_register_preflight.v1"
REGISTER_PREFLIGHT_ANALYSIS_SCHEMA = "gfmcd.pbc.p3_register_preflight_analysis.v1"
REGISTER_PREFLIGHT_AUDIT_SCHEMA = "gfmcd.pbc.p3_register_preflight_audit.v1"
REGISTER_TARGET_SEAL_SCHEMA = "gfmcd.pbc.p3_register_target_seal.v1"
REGISTER_MANIFEST_CHAIN_SCHEMA = "gfmcd.pbc.p3_register_manifest_chain.v1"
REGISTER_TARGET_SCHEMA = "gfmcd.pbc.p3_register_target.v1"
CANDIDATE_SCREEN_SCHEMA = "gfmcd.pbc.p3_register_candidate_screen.v1"
CANDIDATE_FREEZE_SCHEMA = "gfmcd.pbc.p3_register_candidate_freeze.v1"
CANDIDATE_FREEZE_AUDIT_SCHEMA = "gfmcd.pbc.p3_register_candidate_freeze_audit.v1"
PATH_MDE_SCHEMA = "gfmcd.pbc.p3_register_path_mde.v1"
MEDIATION_TARGET_SCHEMA = "gfmcd.pbc.p3_register_mediation_target.v1"

P3_BINS = tuple(range(339, 346))
P10_BINS = tuple(range(94, 101))
REGISTER_CONTROL_SEEDS = tuple(range(2026081101, 2026081132))
SPAN_NUCLEOTIDE = (392, 512)
SPAN_TOKEN = (393, 513)
DECOY_SPAN_NUCLEOTIDE = (152, 272)
DECOY_SPAN_TOKEN = (153, 273)
REGISTER_DIRECTION = "right"
DEVELOPMENT_BLOCKS = 120
RESERVE_BLOCKS = 384
RESERVE_CELLS = 3_456
SESOI = 0.05
MEDIATION_INFERENCE_SEED = 2026081006
MEDIATION_BOOTSTRAP_DRAWS = 20_000
MEDIATION_STATES = (
    "raw",
    "scrub",
    "restore",
    "attenuated_host",
    "sufficiency",
    "no_ramp_host",
)
MEDIATION_OPERATIONS = {
    "raw": "none",
    "scrub": "replace_a_r_with_same_state_a_0",
    "restore": "replace_with_same_state_a_r_from_intact_band",
    "attenuated_host": "none",
    "sufficiency": "inject_same_state_delta_a_r_minus_a_0",
    "no_ramp_host": "none",
}
REGISTER_PREREQUISITE_ARTIFACTS = (
    "phase2_target",
    "phase2_analysis",
    "phase2_audit",
    "register_preflight",
    "register_preflight_analysis",
    "register_preflight_audit",
    "candidate_screen",
    "candidate_freeze",
    "path_mde",
    "candidate_freeze_audit",
)


class RegisterStageError(RuntimeError):
    """A stage binding, authorization, or immutable-seal contract failed."""


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def file_record_matches(record: Mapping[str, Any]) -> bool:
    """Verify one absolute file record without trusting its recorded size or digest."""

    path = Path(str(record.get("path", ""))).expanduser()
    return bool(
        path.is_file()
        and type(record.get("size_bytes")) is int
        and path.stat().st_size == int(record["size_bytes"])
        and isinstance(record.get("sha256"), str)
        and sha256_file(path) == record["sha256"]
    )


def sibling_manifest_path(artifact_path: str | Path) -> Path:
    artifact = Path(artifact_path).expanduser().resolve()
    return artifact.with_name(artifact.stem + ".manifest.json")


def manifest_binds_path(
    manifest: Mapping[str, Any], path: str | Path, *, field: str
) -> bool:
    """Verify one input/output record in a sibling experiment manifest."""

    records = manifest.get(field)
    expected = Path(path).expanduser().resolve()
    return bool(
        isinstance(records, list)
        and any(
            isinstance(record, Mapping)
            and file_record_matches(record)
            and Path(str(record.get("path", ""))).expanduser().resolve() == expected
            for record in records
        )
    )


def validate_sibling_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    artifact_path: str | Path,
) -> dict[str, Any]:
    """Validate an artifact's exact sibling manifest and all recorded inputs."""

    manifest_file = Path(manifest_path).expanduser().resolve()
    artifact_file = Path(artifact_path).expanduser().resolve()
    inputs = manifest.get("inputs")
    created = None
    try:
        created = datetime.fromisoformat(str(manifest.get("created_at_utc", "")))
    except ValueError:
        pass
    commit = str(manifest.get("git_commit", ""))
    checks = {
        "exact_sibling_path": manifest_file == sibling_manifest_path(artifact_file),
        "manifest_file_present": manifest_file.is_file(),
        "artifact_output_bound": manifest_binds_path(
            manifest, artifact_file, field="outputs"
        ),
        "all_inputs_current": bool(
            isinstance(inputs, list)
            and inputs
            and all(
                isinstance(record, Mapping) and file_record_matches(record)
                for record in inputs
            )
        ),
        "chronology_timestamp_recorded": bool(
            created is not None and created.tzinfo is not None
        ),
        "git_revision_recorded": len(commit) == 40
        and all(character in "0123456789abcdef" for character in commit),
        "git_state_recorded": type(manifest.get("git_dirty")) is bool,
        "command_recorded": bool(
            isinstance(manifest.get("command"), list) and manifest.get("command")
        ),
        "stage_recorded": bool(
            isinstance(manifest.get("config"), Mapping)
            and isinstance(manifest["config"].get("stage"), str)
            and manifest["config"]["stage"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "created_at_utc": None if created is None else created.isoformat(),
        "git_commit": commit,
    }


def manifest_precedes(
    producer_manifest: Mapping[str, Any], consumer_manifest: Mapping[str, Any]
) -> bool:
    """Return whether two timezone-aware manifest timestamps are chronological."""

    try:
        producer = datetime.fromisoformat(str(producer_manifest["created_at_utc"]))
        consumer = datetime.fromisoformat(str(consumer_manifest["created_at_utc"]))
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        producer.tzinfo is not None
        and consumer.tzinfo is not None
        and producer <= consumer
    )


def _load_bound_json(record: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    if not file_record_matches(record):
        raise RegisterStageError(f"{label} binding is missing or changed")
    payload = json.loads(Path(str(record["path"])).expanduser().read_text())
    if not isinstance(payload, Mapping):
        raise RegisterStageError(f"{label} binding is not a JSON object")
    return payload


def validate_candidate_bundle_bindings(
    bindings: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the independently audited, pre-reserve candidate bundle."""

    required = (
        "candidate_screen",
        "candidate_freeze",
        "candidate_freeze_audit",
        "path_mde",
    )
    if any(label not in bindings for label in required):
        raise RegisterStageError("reserve seal lacks the complete candidate-freeze bundle")
    screen = _load_bound_json(bindings["candidate_screen"], "candidate screen")
    freeze = _load_bound_json(bindings["candidate_freeze"], "candidate freeze")
    audit = _load_bound_json(
        bindings["candidate_freeze_audit"], "candidate freeze audit"
    )
    path_mde = _load_bound_json(bindings["path_mde"], "candidate path MDE")
    if (
        screen.get("schema") != CANDIDATE_SCREEN_SCHEMA
        or screen.get("status") != "COMPLETE"
        or screen.get("stage") != "development"
        or int(screen.get("n_blocks", -1)) != 120
        or list(screen.get("token_support", [])) != list(SPAN_TOKEN)
    ):
        raise RegisterStageError("candidate screen is not the exact executed development census")
    try:
        validate_candidate_screen(screen, enforce_development_contract=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise RegisterStageError(
            "candidate screen is declarative rather than an executed 102-component record"
        ) from exc
    if (
        freeze.get("schema") != CANDIDATE_FREEZE_SCHEMA
        or freeze.get("status") != "COMPLETE"
        or freeze.get("stage") != "development"
        or list(freeze.get("token_support", [])) != list(SPAN_TOKEN)
    ):
        raise RegisterStageError("candidate freeze is incomplete or has the wrong support")
    try:
        validate_candidate_freeze(freeze)
    except (KeyError, TypeError, ValueError) as exc:
        raise RegisterStageError("candidate freeze does not materialize all matched sets") from exc
    freeze_sources = freeze.get("sources")
    bound_screen = (
        freeze_sources.get("candidate_screen")
        if isinstance(freeze_sources, Mapping)
        else None
    )
    if not isinstance(bound_screen, Mapping) or dict(bound_screen) != dict(
        bindings["candidate_screen"]
    ):
        raise RegisterStageError("candidate freeze is not derived from the bound raw screen")
    if (
        audit.get("schema") != CANDIDATE_FREEZE_AUDIT_SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("candidate_screen_sha256") != bindings["candidate_screen"]["sha256"]
        or audit.get("candidate_freeze_sha256") != bindings["candidate_freeze"]["sha256"]
        or audit.get("path_mde_sha256") != bindings["path_mde"]["sha256"]
    ):
        raise RegisterStageError("candidate freeze independent audit is absent or unbound")
    audit_checks = audit.get("checks")
    independence = audit.get("independence")
    recomputed = audit.get("recomputed")
    if not (
        isinstance(audit_checks, Mapping)
        and len(audit_checks) >= 15
        and all(value is True for value in audit_checks.values())
        and isinstance(independence, Mapping)
        and independence.get("imports_candidate_selection_logic") is False
        and independence.get(
            "recomputes_universe_ranking_prefix_matching_hashes_and_mde"
        )
        is True
        and isinstance(recomputed, Mapping)
        and int(recomputed.get("null_sets_total", -1)) == 93
        and audit.get("candidate_freeze_canonical_sha256")
        == canonical_json_sha256(freeze)
    ):
        raise RegisterStageError("candidate-freeze audit is a non-recomputed attestation")
    mde = path_mde.get("mde80")
    if (
        path_mde.get("schema") != PATH_MDE_SCHEMA
        or path_mde.get("status") != "COMPLETE"
        or path_mde.get("stage") != "development"
        or path_mde.get("passed") is not True
        or int(path_mde.get("reserve_blocks", -1)) != RESERVE_BLOCKS
        or not isinstance(mde, Mapping)
        or any(
            not math.isfinite(float(mde.get(name, math.inf)))
            or float(mde.get(name, math.inf)) > SESOI
            for name in ("D_path", "R_path", "Q_path")
        )
    ):
        raise RegisterStageError("candidate D/R/Q path instrument is not resolved at 0.05")
    selection = freeze.get("selection")
    if not isinstance(selection, Mapping) or not 1 <= int(
        selection.get("candidate_count", 0)
    ) <= 8:
        raise RegisterStageError("candidate freeze has no valid selected prefix")
    tensor_digest = str(freeze.get("patch_tensor_sha256", ""))
    if len(tensor_digest) != 64:
        raise RegisterStageError("candidate freeze lacks an executed patch tensor digest")
    return {
        "candidate_count": int(selection["candidate_count"]),
        "selection_class": str(selection.get("classification")),
        "patch_tensor_sha256": tensor_digest,
        "candidate_screen_sha256": bindings["candidate_screen"]["sha256"],
        "candidate_freeze_sha256": bindings["candidate_freeze"]["sha256"],
        "candidate_freeze_audit_sha256": bindings["candidate_freeze_audit"]["sha256"],
        "path_mde_sha256": bindings["path_mde"]["sha256"],
        "path_mde80": {
            name: float(mde[name]) for name in ("D_path", "R_path", "Q_path")
        },
    }


def deterministic_shard_paths(path: str | Path, shard_count: int) -> list[str]:
    resolved = Path(path).expanduser().resolve()
    count = int(shard_count)
    if count < 1 or count > RESERVE_BLOCKS:
        raise RegisterStageError("shard count must lie between one and reserve block count")
    if count == 1:
        return [str(resolved)]
    return [
        str(
            resolved.with_name(
                f"{resolved.stem}.shard-{index:03d}-of-{count:03d}{resolved.suffix}"
            )
        )
        for index in range(count)
    ]


def validate_phase2_authorization(
    analysis: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    analysis_sha256: str,
) -> dict[str, Any]:
    """Return the audited asymmetric Phase-II authorization or fail closed."""

    if (
        analysis.get("schema") != PHASE2_ANALYSIS_SCHEMA
        or analysis.get("status") != "COMPLETE"
    ):
        raise RegisterStageError("Phase-II analysis is incomplete or has the wrong schema")
    if audit.get("schema") != PHASE2_AUDIT_SCHEMA or audit.get("status") != "PASS":
        raise RegisterStageError("Phase-II independent audit did not pass")
    if audit.get("analysis_sha256") != analysis_sha256:
        raise RegisterStageError("Phase-II audit is not bound to this analysis")
    authorization = analysis.get("authorization")
    if not isinstance(authorization, Mapping):
        raise RegisterStageError("Phase-II analysis lacks its authorization record")
    p3 = authorization.get("authorize_p3_phase3")
    joint = authorization.get("authorize_joint_phase3")
    if type(p3) is not bool or type(joint) is not bool:
        raise RegisterStageError("Phase-II authorization flags must be explicit booleans")
    if joint and not p3:
        raise RegisterStageError("joint Phase III cannot be authorized without P3")

    jreg_direction = None
    if joint:
        endpoints = analysis.get("endpoints")
        if not isinstance(endpoints, Mapping) or not isinstance(endpoints.get("J"), Mapping):
            raise RegisterStageError("joint authorization lacks the corrected Phase-II J endpoint")
        estimate = float(endpoints["J"].get("estimate", math.nan))
        if not math.isfinite(estimate) or estimate == 0:
            raise RegisterStageError("joint authorization needs a finite nonzero J direction")
        jreg_direction = "positive" if estimate > 0 else "negative"
    return {
        "phase2_p3_authorized": bool(p3),
        "phase2_joint_authorized": bool(joint),
        "jreg_direction": jreg_direction,
    }


def projected_mde80(
    per_block: Sequence[Mapping[str, Any]],
    *,
    endpoint_names: Sequence[str],
    reserve_blocks: int = RESERVE_BLOCKS,
) -> dict[str, float]:
    """Project normal-approximation MDE80 from stratum-centered development blocks."""

    rows = tuple(per_block)
    names = tuple(str(name) for name in endpoint_names)
    if len(rows) < 6 or reserve_blocks < 6 or not names:
        raise ValueError("MDE projection needs blocks, endpoints, and a positive reserve")
    labels = np.asarray(
        [f"{row['chrom']}:{row['strand']}" for row in rows], dtype=object
    )
    matrix = np.asarray(
        [[float(row[name]) for name in names] for row in rows], dtype=float
    )
    if not np.isfinite(matrix).all():
        raise ValueError("MDE endpoints must be finite")
    centered = matrix.copy()
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if int(mask.sum()) < 2:
            raise ValueError(f"MDE stratum {label!r} needs at least two blocks")
        centered[mask] -= centered[mask].mean(axis=0)
    standard_deviation = centered.std(axis=0, ddof=1)
    multiplier = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    values = multiplier * standard_deviation / math.sqrt(int(reserve_blocks))
    return {name: float(value) for name, value in zip(names, values)}


def register_preflight_decision(
    *,
    development_analysis: Mapping[str, Any],
    construction_mechanics: Mapping[str, Any],
    health: Mapping[str, Any],
    resolution: Mapping[str, Any],
    fixture_calibration: Mapping[str, Any],
    phase2_authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the complete no-reserve capability gate for Phase-III P3 confirmation."""

    required_resolution = (
        "omega3_intact",
        "s_-1_intact",
        "s_0_intact",
        "s_1_intact",
        "D_path",
        "R_path",
        "Q_path",
    )
    resolution_rows: dict[str, dict[str, Any]] = {}
    for name in required_resolution:
        value = float(resolution.get(name, math.inf))
        resolution_rows[name] = {
            "mde80": value,
            "threshold": SESOI,
            "passed": bool(math.isfinite(value) and value <= SESOI),
        }
    mechanics_pass = bool(
        construction_mechanics.get("passed")
        and float(construction_mechanics.get("register_zero_no_hook_max_abs", math.inf))
        <= 1e-8
        and float(construction_mechanics.get("order_replay_max_abs", math.inf)) <= 1e-8
    )
    health_states = health.get("states")
    inherited_health = health.get("authorized_table_health")
    health_mode = (
        str(inherited_health.get("target_mode"))
        if isinstance(inherited_health, Mapping)
        else ""
    )
    expected_health_tables = (
        204 if health_mode == "REGISTER_JOINT" else 102
        if health_mode == "REGISTER_P3_ONLY"
        else -1
    )
    inherited_tables = (
        inherited_health.get("tables")
        if isinstance(inherited_health, Mapping)
        else None
    )
    health_pass = bool(
        health.get("schema") == "gfmcd.pbc.p3_register_table_health.v1"
        and health.get("stage") == "development"
        and isinstance(health_states, Mapping)
        and len(health_states) == expected_health_tables
        and all(
            isinstance(row, Mapping) and row.get("passed") is True
            for row in health_states.values()
        )
        and isinstance(inherited_health, Mapping)
        and inherited_health.get("schema")
        == "gfmcd.pbc.carrier_factorial_table_health.v1"
        and int(inherited_health.get("n_blocks", -1)) == DEVELOPMENT_BLOCKS
        and int(inherited_health.get("n_tables", -1)) == expected_health_tables
        and inherited_health.get("all_tables_pass") is True
        and isinstance(inherited_tables, Mapping)
        and set(inherited_tables) == set(health_states)
        and all(
            isinstance(row, Mapping)
            and row.get("pass") is True
            and isinstance(row.get("raw_blocks"), list)
            and len(row["raw_blocks"]) == DEVELOPMENT_BLOCKS
            for row in inherited_tables.values()
        )
        and health.get("all_required_states_passed") is True
        and float(health.get("minimum_p10_reader_retention", -math.inf)) >= 0.80
    )
    fixture_pass = bool(
        fixture_calibration.get("all_required_cases_passed") is True
        and float(fixture_calibration.get("familywise_false_positive_rate", math.inf))
        <= 0.06
        and float(fixture_calibration.get("corrected_power_at_0_075", -math.inf))
        >= 0.80
    )
    phase2_p3 = phase2_authorization.get("phase2_p3_authorized")
    phase2_joint = phase2_authorization.get("phase2_joint_authorized")
    if type(phase2_p3) is not bool or type(phase2_joint) is not bool:
        raise RegisterStageError("Phase-II authorization flags are absent")
    relocation_pass = development_analysis.get("status") in {
        "P3_REGISTER_RELOCATION_SUPPORTED",
        "P3_REGISTER_PARTIAL_CONCORDANCE_SUPPORTED",
    }
    mandatory = bool(
        phase2_p3
        and relocation_pass
        and mechanics_pass
        and health_pass
        and fixture_pass
        and all(row["passed"] for row in resolution_rows.values())
    )
    optional_jreg_mde = float(resolution.get("jreg", math.inf))
    joint = bool(
        mandatory
        and phase2_joint
        and math.isfinite(optional_jreg_mde)
        and optional_jreg_mde <= SESOI
        and health.get("joint_state_passed") is True
    )
    return {
        "schema": REGISTER_PREFLIGHT_ANALYSIS_SCHEMA,
        "stage": "development",
        "status": (
            "P3_REGISTER_PREFLIGHT_PASS"
            if mandatory
            else "P3_REGISTER_INSTRUMENT_UNRESOLVED"
        ),
        "authorization": {
            "authorize_p3_register_reserve": mandatory,
            "authorize_joint_register_reserve": joint,
            "p10_failure_does_not_veto_p3": True,
        },
        "gates": {
            "phase2_p3_authorized": bool(phase2_p3),
            "development_relocation": bool(relocation_pass),
            "construction_and_replay": mechanics_pass,
            "health_and_p10_retention": health_pass,
            "fixture_calibration": fixture_pass,
            "resolution": all(row["passed"] for row in resolution_rows.values()),
        },
        "resolution": resolution_rows,
        "optional_jreg": {
            "mde80": optional_jreg_mde,
            "passed": bool(math.isfinite(optional_jreg_mde) and optional_jreg_mde <= SESOI),
            "authorized": joint,
        },
    }


def make_register_target_seal(
    *,
    phase2_authorization: Mapping[str, Any],
    preflight_analysis: Mapping[str, Any],
    preflight_audit: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, Any]],
    include_p10: bool,
    p3_attenuation_dose: float,
    p10_attenuation_dose: float | None,
    output_path: str,
    shard_count: int = 1,
    relocation_analysis_path: str | None = None,
    relocation_audit_path: str | None = None,
    mediation_output_path: str | None = None,
    mediation_analysis_path: str | None = None,
    mediation_audit_path: str | None = None,
    mediation_shard_count: int | None = None,
    prerequisite_manifest_chain: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the deterministic reserve seal after every prerequisite has passed."""

    chain_checks = prerequisite_manifest_chain.get("checks")
    chain_artifacts = prerequisite_manifest_chain.get("artifacts")
    if not (
        prerequisite_manifest_chain.get("schema")
        == REGISTER_MANIFEST_CHAIN_SCHEMA
        and prerequisite_manifest_chain.get("status") == "PASS"
        and prerequisite_manifest_chain.get("stage") == "target"
        and isinstance(chain_checks, Mapping)
        and chain_checks
        and all(value is True for value in chain_checks.values())
        and isinstance(chain_artifacts, Mapping)
        and set(chain_artifacts) == set(REGISTER_PREREQUISITE_ARTIFACTS)
    ):
        raise RegisterStageError("prerequisite producer/audit manifest chain did not pass")
    authorization = preflight_analysis.get("authorization")
    if (
        preflight_analysis.get("schema") != REGISTER_PREFLIGHT_ANALYSIS_SCHEMA
        or preflight_analysis.get("status") != "P3_REGISTER_PREFLIGHT_PASS"
        or not isinstance(authorization, Mapping)
        or authorization.get("authorize_p3_register_reserve") is not True
    ):
        raise RegisterStageError("register development did not authorize the reserve")
    if (
        preflight_audit.get("schema") != REGISTER_PREFLIGHT_AUDIT_SCHEMA
        or preflight_audit.get("status") != "PASS"
    ):
        raise RegisterStageError("register preflight audit did not pass")
    if not bindings or not all(file_record_matches(record) for record in bindings.values()):
        raise RegisterStageError("one or more reserve-seal file bindings are invalid")
    candidate_bundle = validate_candidate_bundle_bindings(bindings)
    p3 = phase2_authorization.get("phase2_p3_authorized") is True
    joint = phase2_authorization.get("phase2_joint_authorized") is True
    if not p3:
        raise RegisterStageError("Phase II did not authorize P3 register confirmation")
    if include_p10 and (
        not joint or authorization.get("authorize_joint_register_reserve") is not True
    ):
        raise RegisterStageError("the optional P10 register arm is not authorized")
    p3_dose = float(p3_attenuation_dose)
    if not math.isfinite(p3_dose) or not 0 < p3_dose <= 1:
        raise RegisterStageError("the frozen Phase-II P3 attenuation dose is invalid")
    p10_dose = (
        None if p10_attenuation_dose is None else float(p10_attenuation_dose)
    )
    if include_p10 and (
        p10_dose is None or not math.isfinite(p10_dose) or not 0 < p10_dose <= 1
    ):
        raise RegisterStageError("joint register confirmation needs the frozen P10 dose")
    if not include_p10 and p10_dose is not None:
        raise RegisterStageError("P3-only seal must not import a P10 dose")
    jreg_direction = (
        phase2_authorization.get("jreg_direction") if include_p10 else None
    )
    shards = int(shard_count)
    resolved_output = Path(output_path).expanduser().resolve()
    shard_paths = deterministic_shard_paths(resolved_output, shards)
    relocation_analysis = Path(
        relocation_analysis_path
        or resolved_output.with_name(resolved_output.stem + ".analysis.json")
    ).expanduser().resolve()
    relocation_audit = Path(
        relocation_audit_path
        or resolved_output.with_name(resolved_output.stem + ".audit.json")
    ).expanduser().resolve()
    mediation_output = Path(
        mediation_output_path
        or resolved_output.with_name(resolved_output.stem + ".mediation.json")
    ).expanduser().resolve()
    mediation_analysis = Path(
        mediation_analysis_path
        or mediation_output.with_name(mediation_output.stem + ".analysis.json")
    ).expanduser().resolve()
    mediation_audit = Path(
        mediation_audit_path
        or mediation_output.with_name(mediation_output.stem + ".audit.json")
    ).expanduser().resolve()
    mediation_shards = int(mediation_shard_count or shards)
    mediation_shard_paths = deterministic_shard_paths(
        mediation_output, mediation_shards
    )
    all_outputs = {
        resolved_output,
        relocation_analysis,
        relocation_audit,
        mediation_output,
        mediation_analysis,
        mediation_audit,
        *(Path(value) for value in shard_paths),
        *(Path(value) for value in mediation_shard_paths),
    }
    expected_count = 6 + len(set(shard_paths)) + len(set(mediation_shard_paths))
    # The assembled path is intentionally also the single-shard path when count=1.
    expected_count -= int(shards == 1) + int(mediation_shards == 1)
    if len(all_outputs) != expected_count:
        raise RegisterStageError("relocation and mediation output paths overlap")
    if any(path.exists() for path in all_outputs):
        raise RegisterStageError("one or more frozen reserve output paths already exist")
    payload: dict[str, Any] = {
        "schema": REGISTER_TARGET_SEAL_SCHEMA,
        "status": "TARGET_AUTHORIZED",
        "stage": "target",
        "mode": "JOINT" if include_p10 else "P3_ONLY",
        "split_role": "phase3_reserve",
        "reserve_blocks": RESERVE_BLOCKS,
        "reserve_cells": RESERVE_CELLS,
        "output_path": str(resolved_output),
        "shard_count": shards,
        "shard_paths": shard_paths,
        "authorization": {
            "phase2_p3_authorized": True,
            "phase2_joint_authorized": bool(joint),
            "register_preflight_passed": True,
            "candidate_freeze_audited": True,
            "path_instrument_capable": True,
        },
        "design": {
            "p3_bins": list(P3_BINS),
            "p10_bins": list(P10_BINS),
            "registers": [-1, 0, 1],
            "direction": REGISTER_DIRECTION,
            "span_nucleotide": list(SPAN_NUCLEOTIDE),
            "span_token": list(SPAN_TOKEN),
            "decoy_span_nucleotide": list(DECOY_SPAN_NUCLEOTIDE),
            "decoy_span_token": list(DECOY_SPAN_TOKEN),
            "control_seeds": list(REGISTER_CONTROL_SEEDS),
            "p3_attenuation_dose": p3_dose,
            "p10_attenuation_dose": p10_dose,
            "jreg_direction": jreg_direction,
        },
        "candidate_bundle": candidate_bundle,
        "prerequisite_manifest_chain": {
            "schema": REGISTER_MANIFEST_CHAIN_SCHEMA,
            "stage": "target",
            "status": "PASS",
            "checks": dict(chain_checks),
            "artifacts": {
                str(label): dict(record)
                for label, record in sorted(chain_artifacts.items())
            },
        },
        "relocation_outputs": {
            "target": str(resolved_output),
            "analysis": str(relocation_analysis),
            "audit": str(relocation_audit),
            "inference_seed": MEDIATION_INFERENCE_SEED,
            "expected_partial_specificity_seed": PARTIAL_SPECIFICITY_SEED,
            "opposite_partial_specificity_seed": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
            "bootstrap_draws": MEDIATION_BOOTSTRAP_DRAWS,
            "reserve_blocks": RESERVE_BLOCKS,
            "reserve_cells": RESERVE_CELLS,
        },
        "mediation": {
            "schema": MEDIATION_TARGET_SCHEMA,
            "stage": "mediation",
            "target_output_path": str(mediation_output),
            "shard_count": mediation_shards,
            "shard_paths": mediation_shard_paths,
            "analysis_output_path": str(mediation_analysis),
            "audit_output_path": str(mediation_audit),
            "inference_seed": MEDIATION_INFERENCE_SEED,
            "bootstrap_draws": MEDIATION_BOOTSTRAP_DRAWS,
            "reserve_blocks": RESERVE_BLOCKS,
            "reserve_cells": RESERVE_CELLS,
            "token_support": list(SPAN_TOKEN),
            "states": list(MEDIATION_STATES),
            "same_state_operations": dict(MEDIATION_OPERATIONS),
            "hook_sites": {
                "head": "attention.self.context_pre_output_projection",
                "mlp": "encoder.layer.output.dense_pre_residual_add",
            },
            "p3_attenuation_order": [
                "construct_intact_position_table",
                "attenuate_complete_frozen_p3_band",
                "apply_requested_full_span_analytic_register_ramp",
                "apply_same_state_activation_intervention",
            ],
            "candidate_freeze_sha256": candidate_bundle[
                "candidate_freeze_sha256"
            ],
        },
        "bindings": {label: dict(record) for label, record in sorted(bindings.items())},
        "outcome_firewall": {
            "phase3_reserve_read_before_seal": False,
            "reserve_outcomes_used_for_design": False,
        },
    }
    payload["seal_sha256"] = canonical_json_sha256(payload)
    return payload


def verify_register_target_seal(seal: Mapping[str, Any]) -> None:
    """Recompute all static seal bindings before a target runner loads the checkpoint."""

    if (
        seal.get("schema") != REGISTER_TARGET_SEAL_SCHEMA
        or seal.get("status") != "TARGET_AUTHORIZED"
        or seal.get("stage") != "target"
    ):
        raise RegisterStageError("wrong or inactive P3 register target seal")
    body = dict(seal)
    expected = str(body.pop("seal_sha256", ""))
    if not expected or canonical_json_sha256(body) != expected:
        raise RegisterStageError("P3 register target seal hash mismatch")
    if (
        int(seal.get("reserve_blocks", -1)) != RESERVE_BLOCKS
        or int(seal.get("reserve_cells", -1)) != RESERVE_CELLS
    ):
        raise RegisterStageError("P3 register reserve size drifted from 384/3456")
    design = seal.get("design")
    expected_design = {
        "p3_bins": list(P3_BINS),
        "p10_bins": list(P10_BINS),
        "registers": [-1, 0, 1],
        "direction": REGISTER_DIRECTION,
        "span_nucleotide": list(SPAN_NUCLEOTIDE),
        "span_token": list(SPAN_TOKEN),
        "decoy_span_nucleotide": list(DECOY_SPAN_NUCLEOTIDE),
        "decoy_span_token": list(DECOY_SPAN_TOKEN),
        "control_seeds": list(REGISTER_CONTROL_SEEDS),
        "p3_attenuation_dose": None,
        "p10_attenuation_dose": None,
        "jreg_direction": None,
    }
    if not isinstance(design, Mapping):
        raise RegisterStageError("P3 register target seal lacks its design")
    comparable = dict(design)
    jreg_direction = comparable.pop("jreg_direction", None)
    p3_dose = comparable.pop("p3_attenuation_dose", None)
    p10_dose = comparable.pop("p10_attenuation_dose", None)
    expected_design.pop("jreg_direction")
    expected_design.pop("p3_attenuation_dose")
    expected_design.pop("p10_attenuation_dose")
    if comparable != expected_design:
        raise RegisterStageError("P3 register target design drifted from the protocol")
    mode = seal.get("mode")
    if mode not in {"P3_ONLY", "JOINT"}:
        raise RegisterStageError("P3 register target mode is invalid")
    if mode == "P3_ONLY" and jreg_direction is not None:
        raise RegisterStageError("P3-only seal must not import a Jreg direction")
    if mode == "JOINT" and jreg_direction not in {"positive", "negative"}:
        raise RegisterStageError("joint seal lacks the frozen Jreg direction")
    if not isinstance(p3_dose, (int, float)) or not 0 < float(p3_dose) <= 1:
        raise RegisterStageError("target seal lacks a valid frozen P3 attenuation dose")
    if mode == "P3_ONLY" and p10_dose is not None:
        raise RegisterStageError("P3-only seal must not contain a P10 attenuation dose")
    if mode == "JOINT" and (
        not isinstance(p10_dose, (int, float)) or not 0 < float(p10_dose) <= 1
    ):
        raise RegisterStageError("joint seal lacks a valid frozen P10 attenuation dose")
    shards = seal.get("shard_count")
    shard_paths = seal.get("shard_paths")
    if type(shards) is not int or not 1 <= shards <= RESERVE_BLOCKS:
        raise RegisterStageError("target seal has an invalid shard count")
    expected_paths = deterministic_shard_paths(str(seal.get("output_path", "")), shards)
    if shard_paths != expected_paths:
        raise RegisterStageError("target seal shard paths are not the deterministic frozen paths")
    relocation = seal.get("relocation_outputs")
    if not isinstance(relocation, Mapping) or relocation != {
        "target": str(Path(str(seal.get("output_path", ""))).resolve()),
        "analysis": str(Path(str(relocation.get("analysis", ""))).resolve()),
        "audit": str(Path(str(relocation.get("audit", ""))).resolve()),
        "inference_seed": MEDIATION_INFERENCE_SEED,
        "expected_partial_specificity_seed": PARTIAL_SPECIFICITY_SEED,
        "opposite_partial_specificity_seed": OPPOSITE_PARTIAL_SPECIFICITY_SEED,
        "bootstrap_draws": MEDIATION_BOOTSTRAP_DRAWS,
        "reserve_blocks": RESERVE_BLOCKS,
        "reserve_cells": RESERVE_CELLS,
    }:
        raise RegisterStageError("relocation output/inference bindings drifted")
    mediation = seal.get("mediation")
    if not isinstance(mediation, Mapping):
        raise RegisterStageError("unified reserve seal lacks mediation bindings")
    mediation_shards = mediation.get("shard_count")
    if type(mediation_shards) is not int or not 1 <= mediation_shards <= RESERVE_BLOCKS:
        raise RegisterStageError("mediation shard count is invalid")
    mediation_paths = mediation.get("shard_paths")
    expected_mediation = {
        "schema": MEDIATION_TARGET_SCHEMA,
        "stage": "mediation",
        "target_output_path": str(
            Path(str(mediation.get("target_output_path", ""))).resolve()
        ),
        "shard_count": mediation_shards,
        "shard_paths": deterministic_shard_paths(
            str(mediation.get("target_output_path", "")), int(mediation_shards)
        ),
        "analysis_output_path": str(
            Path(str(mediation.get("analysis_output_path", ""))).resolve()
        ),
        "audit_output_path": str(
            Path(str(mediation.get("audit_output_path", ""))).resolve()
        ),
        "inference_seed": MEDIATION_INFERENCE_SEED,
        "bootstrap_draws": MEDIATION_BOOTSTRAP_DRAWS,
        "reserve_blocks": RESERVE_BLOCKS,
        "reserve_cells": RESERVE_CELLS,
        "token_support": list(SPAN_TOKEN),
        "states": list(MEDIATION_STATES),
        "same_state_operations": dict(MEDIATION_OPERATIONS),
        "hook_sites": {
            "head": "attention.self.context_pre_output_projection",
            "mlp": "encoder.layer.output.dense_pre_residual_add",
        },
        "p3_attenuation_order": [
            "construct_intact_position_table",
            "attenuate_complete_frozen_p3_band",
            "apply_requested_full_span_analytic_register_ramp",
            "apply_same_state_activation_intervention",
        ],
        "candidate_freeze_sha256": str(
            mediation.get("candidate_freeze_sha256", "")
        ),
    }
    if dict(mediation) != expected_mediation or mediation_paths != expected_mediation["shard_paths"]:
        raise RegisterStageError("mediation paths, hooks, operations, or inference drifted")
    firewall = seal.get("outcome_firewall")
    if not isinstance(firewall, Mapping) or any(value is not False for value in firewall.values()):
        raise RegisterStageError("reserve outcome firewall is not intact")
    bindings = seal.get("bindings")
    if not isinstance(bindings, Mapping) or not bindings:
        raise RegisterStageError("P3 register target seal has no file bindings")
    if not all(
        isinstance(record, Mapping) and file_record_matches(record)
        for record in bindings.values()
    ):
        raise RegisterStageError("P3 register target binding hash mismatch")
    candidate = validate_candidate_bundle_bindings(bindings)
    if seal.get("candidate_bundle") != candidate:
        raise RegisterStageError("candidate bundle drifted from its audited bindings")
    if mediation.get("candidate_freeze_sha256") != candidate["candidate_freeze_sha256"]:
        raise RegisterStageError("mediation does not bind the audited candidate freeze")
    chain = seal.get("prerequisite_manifest_chain")
    chain_checks = chain.get("checks") if isinstance(chain, Mapping) else None
    chain_artifacts = chain.get("artifacts") if isinstance(chain, Mapping) else None
    if not (
        isinstance(chain, Mapping)
        and chain.get("schema") == REGISTER_MANIFEST_CHAIN_SCHEMA
        and chain.get("stage") == "target"
        and chain.get("status") == "PASS"
        and isinstance(chain_checks, Mapping)
        and chain_checks
        and all(value is True for value in chain_checks.values())
        and isinstance(chain_artifacts, Mapping)
        and set(chain_artifacts) == set(REGISTER_PREREQUISITE_ARTIFACTS)
    ):
        raise RegisterStageError("P3 register prerequisite manifest chain drifted")


__all__ = [
    "CANDIDATE_FREEZE_AUDIT_SCHEMA",
    "CANDIDATE_FREEZE_SCHEMA",
    "CANDIDATE_SCREEN_SCHEMA",
    "DECOY_SPAN_NUCLEOTIDE",
    "DECOY_SPAN_TOKEN",
    "MEDIATION_BOOTSTRAP_DRAWS",
    "MEDIATION_INFERENCE_SEED",
    "MEDIATION_OPERATIONS",
    "MEDIATION_STATES",
    "MEDIATION_TARGET_SCHEMA",
    "P10_BINS",
    "P3_BINS",
    "REGISTER_CONTROL_SEEDS",
    "REGISTER_DIRECTION",
    "REGISTER_PREFLIGHT_ANALYSIS_SCHEMA",
    "REGISTER_PREFLIGHT_AUDIT_SCHEMA",
    "REGISTER_PREFLIGHT_SCHEMA",
    "REGISTER_MANIFEST_CHAIN_SCHEMA",
    "REGISTER_PREREQUISITE_ARTIFACTS",
    "REGISTER_TARGET_SCHEMA",
    "REGISTER_TARGET_SEAL_SCHEMA",
    "RESERVE_BLOCKS",
    "RESERVE_CELLS",
    "SPAN_NUCLEOTIDE",
    "SPAN_TOKEN",
    "RegisterStageError",
    "canonical_json_sha256",
    "deterministic_shard_paths",
    "file_record_matches",
    "manifest_binds_path",
    "manifest_precedes",
    "make_register_target_seal",
    "projected_mde80",
    "register_preflight_decision",
    "sibling_manifest_path",
    "validate_phase2_authorization",
    "validate_candidate_bundle_bindings",
    "validate_sibling_manifest",
    "verify_register_target_seal",
]
