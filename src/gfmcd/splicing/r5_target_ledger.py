"""Irreversible biological-target receipts for the conditional R5 stages.

The ledger path is keyed by biological target identity rather than executable package
bytes.  Repackaging or changing code therefore cannot create a fresh attempt on an
already exposed cohort.  Exact stack, upstream, and preflight identities remain bound in
the receipt as attempt context.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


IDENTITY_SCHEMA = "gfmcd.r5.biological_target_identity/1"
RECEIPT_SCHEMA = "gfmcd.r5.target_attempt_receipt/1"
COMPLETION_SCHEMA = "gfmcd.r5.target_attempt_completion/1"
R5_STAGES = ("R5a", "R5b", "R5c")
R1C_IDENTITY_SCHEMA = "gfmcd.r1c.biological_target_identity/1"
R1C_RECEIPT_SCHEMA = "gfmcd.r1c.target_attempt/2"
R1C_COMPLETION_SCHEMA = "gfmcd.r1c.target_completion/1"
R1C_PROGRAM = "r1c-sealed-reverse-behavior-v2"


class R5TargetSpentError(RuntimeError):
    """The canonical target identity already has a receipt or completion."""


class R1CTargetSpentError(RuntimeError):
    """The exact sealed R1c checkpoint/cohort identity is already spent."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _biological_key_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the method-insensitive same-stage cohort spend identity.

    Construction, endpoint, stack, upstream, and preflight identities remain preserved
    in the receipt, but none may mint a fresh attempt after outcomes from the same
    checkpoint/cohort/stage have been exposed.
    """

    return {
        "schema": identity.get("schema"),
        "program": identity.get("program"),
        "stage": identity.get("stage"),
        "checkpoint_sha256": identity.get("checkpoint_sha256"),
        "target_gene_ids_sha256": identity.get("target_gene_ids_sha256"),
        "ordered_rows_sha256": identity.get("ordered_rows_sha256"),
    }


def _r1c_biological_key_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: identity.get(key)
        for key in (
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
        )
    }


def build_r1c_biological_target_identity(
    *,
    checkpoint_sha256: str,
    gtf_sha256: str,
    fasta_sha256: str,
    fasta_fai_sha256: str,
) -> dict[str, Any]:
    """Key one spend from pre-read source identities and the declared holdout."""

    stable = {
        "schema": R1C_IDENTITY_SCHEMA,
        "program": R1C_PROGRAM,
        "stage": "R1c",
        "holdout_id": "r1c-canonical-internal-short-intron-chr8-chr9-v1",
        "target_chromosomes": ["chr8", "chr9"],
        "target_n": 400,
        "checkpoint_sha256": _require_sha256(
            "checkpoint_sha256", checkpoint_sha256
        ),
        "gtf_sha256": _require_sha256(
            "gtf_sha256", gtf_sha256
        ),
        "fasta_sha256": _require_sha256(
            "fasta_sha256", fasta_sha256
        ),
        "fasta_fai_sha256": _require_sha256(
            "fasta_fai_sha256", fasta_fai_sha256
        ),
    }
    return stable | {
        "biological_target_key": _canonical_sha256(
            _r1c_biological_key_payload(stable)
        )
    }


def build_r5_biological_target_identity(
    *,
    stage: str,
    checkpoint_sha256: str,
    target_gene_ids_sha256: str,
    ordered_rows_sha256: str,
    construction_sha256: str,
    endpoint_policy: str,
) -> dict[str, Any]:
    """Build the stable key that prevents any same-stage target retry."""

    if stage not in R5_STAGES:
        raise ValueError(f"stage must be one of {R5_STAGES}")
    if not isinstance(endpoint_policy, str) or not endpoint_policy:
        raise ValueError("endpoint_policy must be a non-empty string")
    stable = {
        "schema": IDENTITY_SCHEMA,
        "program": "r5-acceptor-mechanism-capstone-v2",
        "stage": stage,
        "checkpoint_sha256": _require_sha256(
            "checkpoint_sha256", checkpoint_sha256
        ),
        "target_gene_ids_sha256": _require_sha256(
            "target_gene_ids_sha256", target_gene_ids_sha256
        ),
        "ordered_rows_sha256": _require_sha256(
            "ordered_rows_sha256", ordered_rows_sha256
        ),
        "construction_sha256": _require_sha256(
            "construction_sha256", construction_sha256
        ),
        "endpoint_policy": endpoint_policy,
        "masked_donor_endpoint_constructed": False,
    }
    return stable | {
        "biological_target_key": _canonical_sha256(_biological_key_payload(stable))
    }


def _git_common_dir(repo: Path) -> Path:
    resolved = repo.expanduser().resolve(strict=True)
    process = subprocess.run(
        ["git", "-C", str(resolved), "rev-parse", "--git-common-dir"],
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = process.stdout.strip()
    if not rendered:
        raise RuntimeError("git returned an empty common directory")
    candidate = Path(rendered)
    if not candidate.is_absolute():
        candidate = resolved / candidate
    return candidate.resolve(strict=True)


def _ledger_path(repo: Path, identity: Mapping[str, Any]) -> Path:
    stage = identity.get("stage")
    key = identity.get("biological_target_key")
    if stage not in R5_STAGES or not isinstance(key, str):
        raise ValueError("identity is missing its stage or biological target key")
    return _git_common_dir(repo) / "gfmcd_target_attempts" / "r5" / stage.lower() / f"{key}.json"


def _r1c_ledger_path(repo: Path, identity: Mapping[str, Any]) -> Path:
    key = identity.get("biological_target_key")
    if identity.get("stage") != "R1c" or not isinstance(key, str):
        raise ValueError("R1c identity is missing its stage or biological target key")
    return _git_common_dir(repo) / "gfmcd_target_attempts" / "r1c" / f"{key}.json"


def _exclusive_write(path: Path, payload: Mapping[str, Any], *, what: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise R5TargetSpentError(f"{what} already exists; biological target is already spent") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        # Never remove a created receipt: even a partial failure conservatively spends the
        # target and requires audit rather than retry.
        raise


def _validate_logical_identity(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"logical_name", "size_bytes", "sha256"}:
        raise ValueError(f"{name} must contain logical_name, size_bytes, and sha256")
    logical_name = value.get("logical_name")
    size = value.get("size_bytes")
    if (
        not isinstance(logical_name, str)
        or not logical_name
        or Path(logical_name).name != logical_name
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 1
    ):
        raise ValueError(f"{name} logical identity is malformed")
    return {
        "logical_name": logical_name,
        "size_bytes": int(size),
        "sha256": _require_sha256(name, value.get("sha256")),
    }


def reserve_r1c_target_attempt(
    *,
    repo: str | Path,
    identity: Mapping[str, Any],
    method_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Spend the exact sealed R1c cohort before cohort/model/output inspection."""

    expected_identity_fields = {
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
    expected_key = _canonical_sha256(_r1c_biological_key_payload(identity))
    if (
        set(identity) != expected_identity_fields
        or identity.get("schema") != R1C_IDENTITY_SCHEMA
        or identity.get("program") != R1C_PROGRAM
        or identity.get("stage") != "R1c"
        or identity.get("holdout_id")
        != "r1c-canonical-internal-short-intron-chr8-chr9-v1"
        or identity.get("target_chromosomes") != ["chr8", "chr9"]
        or identity.get("target_n") != 400
        or identity.get("biological_target_key") != expected_key
    ):
        raise ValueError("R1c biological target identity does not replay")
    for field in (
        "checkpoint_sha256",
        "gtf_sha256",
        "fasta_sha256",
        "fasta_fai_sha256",
    ):
        _require_sha256(field, identity.get(field))

    required_context = {
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
    if set(method_context) != required_context:
        raise ValueError("R1c method context fields differ")
    normalized_context = dict(method_context)
    for field in (
        "protocol_sha256",
        "instrument_fingerprint_sha256",
        "r5_stack_fingerprint_sha256",
        "checkpoint_bundle_sha256",
        "construction_contract_sha256",
        "input_snapshot_sha256",
    ):
        normalized_context[field] = _require_sha256(field, method_context.get(field))
    if normalized_context["checkpoint_bundle_sha256"] != identity.get(
        "checkpoint_sha256"
    ):
        raise ValueError("R1c checkpoint context differs from biological identity")
    endpoint_policy = method_context.get("endpoint_policy")
    if not isinstance(endpoint_policy, str) or not endpoint_policy:
        raise ValueError("R1c endpoint_policy must be a non-empty string")
    for field in ("preflight_artifact", "preflight_manifest"):
        value = method_context.get(field)
        if not isinstance(value, Mapping):
            raise ValueError(f"R1c {field} identity is missing")
        normalized_context[field] = _validate_logical_identity(field, value)
    for field in (
        "requested_artifact_logical_name",
        "requested_manifest_logical_name",
        "declared_completion_logical_name",
    ):
        value = method_context.get(field)
        if not isinstance(value, str) or not value or Path(value).name != value:
            raise ValueError(f"R1c {field} must be a bare logical name")

    path = _r1c_ledger_path(Path(repo), identity)
    if normalized_context["declared_completion_logical_name"] != path.with_suffix(
        ".completion.json"
    ).name:
        raise ValueError("R1c declared completion name differs from canonical ledger path")
    receipt_without_id = {
        "schema": R1C_RECEIPT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUTHORIZED_SEALED_OUTCOMES_MAY_BE_SPENT",
        "biological_target_key": expected_key,
        "target_identity": dict(identity),
        "method_context": normalized_context,
        "policy": (
            "create-only before target cohort/path/model/output inspection; failures "
            "remain spent; method, endpoint, construction, and code cannot reset spend"
        ),
        "sealed_cohort_constructed_before_record": False,
        "model_loaded_before_record": False,
        "output_inspected_before_record": False,
    }
    receipt = receipt_without_id | {
        "attempt_id": _canonical_sha256(receipt_without_id)
    }
    try:
        _exclusive_write(path, receipt, what="R1c target attempt receipt")
    except R5TargetSpentError as exc:
        raise R1CTargetSpentError(str(exc)) from exc
    return receipt | {"receipt_path": str(path)}


def _logical_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"publication path is not a regular file: {resolved}")
    return {
        "logical_name": resolved.name,
        "size_bytes": resolved.stat().st_size,
        "sha256": _file_sha256(resolved),
    }


def complete_r1c_target_attempt(
    *,
    receipt: Mapping[str, Any],
    artifact_path: str | Path,
    manifest_path: str | Path,
    cohort_digests: Mapping[str, Any],
    method_fingerprint_sha256: str,
    r5_stack_fingerprint_sha256: str,
    official_verdict: str,
    all_gates_pass: bool,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish the immutable R1c completion only after artifact and manifest exist."""

    receipt_path_raw = receipt.get("receipt_path")
    if not isinstance(receipt_path_raw, str):
        raise ValueError("R1c receipt must carry its canonical receipt_path")
    receipt_path = Path(receipt_path_raw).expanduser().resolve(strict=True)
    persisted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    supplied_receipt = {
        key: value for key, value in receipt.items() if key != "receipt_path"
    }
    if (
        persisted_receipt != supplied_receipt
        or persisted_receipt.get("schema") != R1C_RECEIPT_SCHEMA
        or persisted_receipt.get("attempt_id")
        != _canonical_sha256(
            {
                key: value
                for key, value in persisted_receipt.items()
                if key != "attempt_id"
            }
        )
    ):
        raise ValueError("R1c receipt bytes do not replay")

    artifact = Path(artifact_path).expanduser().resolve(strict=True)
    manifest = Path(manifest_path).expanduser().resolve(strict=True)
    artifact_identity = _logical_file_identity(artifact)
    manifest_identity = _logical_file_identity(manifest)
    attempt_base_identity = _logical_file_identity(receipt_path)
    attempt_package_identity = attempt_base_identity | {
        "attempt_id": persisted_receipt["attempt_id"],
        "biological_target_key": persisted_receipt["biological_target_key"],
        "status": persisted_receipt["status"],
    }
    context = persisted_receipt.get("method_context")
    target_identity = persisted_receipt.get("target_identity")
    if not isinstance(context, Mapping) or not isinstance(target_identity, Mapping):
        raise ValueError("R1c receipt context is missing")
    if artifact_identity["logical_name"] != context.get(
        "requested_artifact_logical_name"
    ) or manifest_identity["logical_name"] != context.get(
        "requested_manifest_logical_name"
    ):
        raise ValueError("R1c publication logical names differ from the receipt")
    if (
        _require_sha256("method_fingerprint_sha256", method_fingerprint_sha256)
        != context.get("instrument_fingerprint_sha256")
        or _require_sha256(
            "r5_stack_fingerprint_sha256", r5_stack_fingerprint_sha256
        )
        != context.get("r5_stack_fingerprint_sha256")
    ):
        raise ValueError("R1c completion method identity differs from the receipt")

    expected_digest_fields = {
        "source_cohort_sha256",
        "ordered_example_ids_sha256",
        "target_gene_ids_sha256",
        "ordered_gene_ids_sha256",
        "ordered_six_position_rows_sha256",
        "ordered_partner_margin_levels_sha256",
        "construction_sha256",
    }
    if set(cohort_digests) != expected_digest_fields:
        raise ValueError("R1c cohort digest fields differ")
    normalized_digests = {
        name: _require_sha256(name, value)
        for name, value in cohort_digests.items()
    }
    try:
        artifact_record = json.loads(artifact.read_text(encoding="utf-8"))
        manifest_record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("R1c publication files must contain JSON objects") from exc
    if (
        not isinstance(artifact_record, Mapping)
        or artifact_record.get("target_attempt") != attempt_package_identity
        or artifact_record.get("target_completion")
        != {"logical_name": context.get("declared_completion_logical_name")}
        or artifact_record.get("target_cohort_identity") != normalized_digests
    ):
        raise ValueError("R1c artifact publication package fields differ")
    outputs = manifest_record.get("outputs") if isinstance(manifest_record, Mapping) else None
    matching_outputs = (
        [
            item
            for item in outputs
            if isinstance(item, Mapping)
            and Path(str(item.get("path", ""))).name
            == artifact_identity["logical_name"]
            and item.get("size_bytes") == artifact_identity["size_bytes"]
            and item.get("sha256") == artifact_identity["sha256"]
        ]
        if isinstance(outputs, list)
        else []
    )
    if len(matching_outputs) != 1:
        raise ValueError("R1c manifest does not bind exactly one artifact output")
    if not isinstance(official_verdict, str) or not official_verdict:
        raise ValueError("R1c official_verdict must be a non-empty string")
    if (
        not isinstance(all_gates_pass, bool)
        or not isinstance(gates, Mapping)
        or not gates
        or any(not isinstance(value, bool) for value in gates.values())
        or all_gates_pass is not all(gates.values())
    ):
        raise ValueError("R1c completion gates do not replay")

    completion_path = receipt_path.with_suffix(".completion.json")
    completion_without_id = {
        "schema": R1C_COMPLETION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TARGET_PUBLICATION_COMPLETE",
        "attempt_id": persisted_receipt["attempt_id"],
        "biological_target_key": persisted_receipt["biological_target_key"],
        "method_fingerprint_sha256": method_fingerprint_sha256,
        "r5_stack_fingerprint_sha256": r5_stack_fingerprint_sha256,
        "cohort_digests": normalized_digests,
        "artifact": artifact_identity,
        "manifest": manifest_identity,
        "attempt": attempt_base_identity,
        "official_verdict": official_verdict,
        "all_gates_pass": all_gates_pass,
        "gates": dict(gates),
    }
    completion = completion_without_id | {
        "completion_id": _canonical_sha256(completion_without_id)
    }
    try:
        _exclusive_write(
            completion_path, completion, what="R1c target attempt completion"
        )
    except R5TargetSpentError as exc:
        raise R1CTargetSpentError(str(exc)) from exc
    return completion | {"completion_path": str(completion_path)}


def reserve_r5_target_attempt(
    *,
    repo: str | Path,
    identity: Mapping[str, Any],
    attempt_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the canonical receipt before target cohort/model/output inspection."""

    expected_fields = {
        "schema",
        "program",
        "stage",
        "checkpoint_sha256",
        "target_gene_ids_sha256",
        "ordered_rows_sha256",
        "construction_sha256",
        "endpoint_policy",
        "masked_donor_endpoint_constructed",
        "biological_target_key",
    }
    expected_key = _canonical_sha256(_biological_key_payload(identity))
    if (
        identity.get("schema") != IDENTITY_SCHEMA
        or identity.get("program") != "r5-acceptor-mechanism-capstone-v2"
        or identity.get("stage") not in R5_STAGES
        or identity.get("masked_donor_endpoint_constructed") is not False
        or not isinstance(identity.get("endpoint_policy"), str)
        or not identity.get("endpoint_policy")
        or identity.get("biological_target_key") != expected_key
        or set(identity) != expected_fields
    ):
        raise ValueError("biological target identity does not replay")
    for name in (
        "checkpoint_sha256",
        "target_gene_ids_sha256",
        "ordered_rows_sha256",
        "construction_sha256",
    ):
        _require_sha256(name, identity.get(name))
    required_context = {
        "r5_stack_fingerprint_sha256",
        "stage_preflight_artifact_sha256",
        "stage_preflight_manifest_sha256",
        "upstream_package_chain_sha256",
    }
    if set(attempt_context) != required_context:
        raise ValueError(
            f"attempt_context must contain exactly {tuple(sorted(required_context))}"
        )
    for key, value in attempt_context.items():
        _require_sha256(key, value)
    path = _ledger_path(Path(repo), identity)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "biological_target_key": expected_key,
        "identity": dict(identity),
        "attempt_context": dict(attempt_context),
        "policy": (
            "create-only before target cohort/path/model/output inspection; failures "
            "remain spent; executable repackaging cannot change the ledger key"
        ),
    }
    _exclusive_write(path, receipt, what="target attempt receipt")
    return receipt | {"receipt_path": str(path)}


def _validate_output_identity(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != {"size_bytes", "sha256"}:
        raise ValueError(f"{name} must contain size_bytes and sha256")
    size = value["size_bytes"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"{name}.size_bytes must be a positive integer")
    return {"size_bytes": int(size), "sha256": _require_sha256(name, value["sha256"])}


def complete_r5_target_attempt(
    *,
    receipt: Mapping[str, Any],
    result_identity: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    official_disposition: str,
) -> dict[str, Any]:
    """Bind immutable result and manifest identities to an existing receipt."""

    receipt_path_raw = receipt.get("receipt_path")
    if not isinstance(receipt_path_raw, str):
        raise ValueError("receipt must carry its canonical receipt_path")
    receipt_path = Path(receipt_path_raw).resolve(strict=True)
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    supplied = {key: value for key, value in receipt.items() if key != "receipt_path"}
    if persisted != supplied or persisted.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("receipt bytes do not match the supplied receipt")
    if not isinstance(official_disposition, str) or not official_disposition:
        raise ValueError("official_disposition must be a non-empty string")
    completion_path = receipt_path.with_suffix(".completion.json")
    completion = {
        "schema": COMPLETION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "biological_target_key": persisted["biological_target_key"],
        "receipt_sha256": _file_sha256(receipt_path),
        "result": _validate_output_identity("result_identity", result_identity),
        "manifest": _validate_output_identity("manifest_identity", manifest_identity),
        "official_disposition": official_disposition,
        "status": "TARGET_PUBLICATION_COMPLETE",
    }
    _exclusive_write(completion_path, completion, what="target attempt completion")
    return completion | {"completion_path": str(completion_path)}


__all__ = [
    "COMPLETION_SCHEMA",
    "IDENTITY_SCHEMA",
    "R1C_COMPLETION_SCHEMA",
    "R1C_IDENTITY_SCHEMA",
    "R1C_PROGRAM",
    "R1C_RECEIPT_SCHEMA",
    "R1CTargetSpentError",
    "RECEIPT_SCHEMA",
    "R5TargetSpentError",
    "build_r1c_biological_target_identity",
    "build_r5_biological_target_identity",
    "complete_r1c_target_attempt",
    "complete_r5_target_attempt",
    "reserve_r1c_target_attempt",
    "reserve_r5_target_attempt",
]
