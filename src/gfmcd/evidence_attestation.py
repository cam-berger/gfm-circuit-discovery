"""Verify compact evidence records whose raw packages live in Git history.

Large scientific artifacts are immutable evidence, but checking every raw calibration row
into the active tree makes ordinary code review unusable.  An evidence attestation keeps the
review-facing record compact while binding the original bytes to an annotated archive tag.

The public interface deliberately exposes one operation: verify the attestation.  Callers do
not need to know how annotated tags, historical blobs, active audit files, or legacy claim IDs
are represented internally.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping


ATTESTATION_SCHEMA = "gfmcd.evidence_attestation/1"
_SLUG = r"[a-z0-9]+(?:-[a-z0-9]+)*"
EVIDENCE_ID_RE = re.compile(rf"^{_SLUG}/{_SLUG}/{_SLUG}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}$")
LEGAL_RECORD_KINDS = {
    "finding",
    "limit",
    "instrument",
    "diagnostic",
    "development",
    "synthesis",
    "correction",
}
LEGAL_STATUSES = {
    "supported",
    "bounded",
    "inconclusive",
    "ineligible",
    "closed",
    "instrument-unresolved",
    "development",
    "retracted",
    "open",
}
LEGAL_EVIDENCE_RUNGS = {
    "behavior",
    "representation",
    "selection-qk",
    "routing-attention",
    "output-ov",
    "mediation",
    "directed-circuit",
    "biological-interpretation",
    "instrument",
    "not-applicable",
}
LEGAL_REVIEW_TIERS = {"confirmatory", "verification", "exploratory", "development"}
LEGAL_ALIAS_TYPES = {"claim", "run", "correction", "artifact", "protocol"}


class EvidenceAttestationError(ValueError):
    """The compact record is malformed or no longer matches its archived evidence."""


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    label: str,
    optional: set[str] | None = None,
) -> None:
    optional = optional or set()
    missing = required - value.keys()
    if missing:
        raise EvidenceAttestationError(
            f"{label} is missing required field {sorted(missing)[0]!r}"
        )
    unknown = value.keys() - required - optional
    if unknown:
        raise EvidenceAttestationError(
            f"{label} has unknown field {sorted(unknown)[0]!r}"
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvidenceAttestationError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceAttestationError(f"{label} must be a non-empty string")
    return value


def _validate_repo_path(value: Any, label: str) -> str:
    raw = _require_string(value, label)
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or raw != path.as_posix():
        raise EvidenceAttestationError(f"{label} must be a normalized repository-relative path")
    return raw


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=repo, text=True, stderr=subprocess.PIPE
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise EvidenceAttestationError(
            f"git {' '.join(args)} failed: {detail.strip()}"
        ) from exc


def _discover_repository(start: Path) -> Path:
    """Resolve a repository root from a file-adjacent directory or an explicit hint."""

    start = start.resolve()
    if not start.is_dir():
        raise EvidenceAttestationError(f"repository search path is not a directory: {start}")
    root = _git(start, "rev-parse", "--show-toplevel")
    return Path(root).resolve()


def _historical_blob_identity(repo: Path, commit: str, path: str) -> tuple[int, str]:
    spec = f"{commit}:{path}"
    try:
        process = subprocess.Popen(
            ["git", "cat-file", "blob", spec],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise EvidenceAttestationError(f"cannot read archived blob {spec}: {exc}") from exc

    assert process.stdout is not None
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    returncode = process.wait()
    if returncode:
        raise EvidenceAttestationError(
            f"cannot read archived blob {spec}: {stderr.strip() or f'exit {returncode}'}"
        )
    return size, digest.hexdigest()


def _historical_blob_bytes(repo: Path, commit: str, path: str) -> bytes:
    spec = f"{commit}:{path}"
    try:
        return subprocess.check_output(
            ["git", "cat-file", "blob", spec], cwd=repo, stderr=subprocess.PIPE
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        raise EvidenceAttestationError(
            f"cannot read archived blob {spec}: {str(detail).strip() or str(exc)}"
        ) from exc


def _local_file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
    except OSError as exc:
        raise EvidenceAttestationError(f"cannot read active file {path}: {exc}") from exc
    return size, digest.hexdigest()


def _parse_json_bytes(data: bytes, *, label: str, allow_nonfinite: bool = False) -> Any:
    def reject_nonfinite(value: str) -> Any:
        raise EvidenceAttestationError(f"{label} contains non-standard JSON constant {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceAttestationError(f"{label} has duplicate object key {key!r}")
            result[key] = value
        return result

    options: dict[str, Any] = {"object_pairs_hook": reject_duplicate_keys}
    if not allow_nonfinite:
        options["parse_constant"] = reject_nonfinite
    try:
        return json.loads(data, **options)
    except EvidenceAttestationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceAttestationError(f"cannot parse {label} as JSON: {exc}") from exc


def _resolve_json_pointer(document: Any, pointer: str, *, label: str) -> Any:
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise EvidenceAttestationError(f"{label} must be an RFC 6901 JSON Pointer")
    current = document
    for encoded in pointer[1:].split("/"):
        if re.search(r"~(?:[^01]|$)", encoded):
            raise EvidenceAttestationError(f"{label} has an invalid JSON Pointer escape")
        token = encoded.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(current, list):
                if not re.fullmatch(r"0|[1-9][0-9]*", token):
                    raise KeyError(token)
                current = current[int(token)]
            elif isinstance(current, Mapping):
                current = current[token]
            else:
                raise KeyError(token)
        except (KeyError, IndexError) as exc:
            raise EvidenceAttestationError(
                f"{label} does not resolve at token {token!r}"
            ) from exc
    return current


def _json_equal(observed: Any, expected: Any) -> bool:
    """Compare JSON values without Python's ``True == 1`` coercion."""

    if isinstance(observed, bool) or isinstance(expected, bool):
        return type(observed) is type(expected) and observed == expected
    if isinstance(observed, Mapping) and isinstance(expected, Mapping):
        return observed.keys() == expected.keys() and all(
            _json_equal(observed[key], expected[key]) for key in observed
        )
    if isinstance(observed, list) and isinstance(expected, list):
        return len(observed) == len(expected) and all(
            _json_equal(left, right) for left, right in zip(observed, expected)
        )
    if type(observed) is not type(expected):
        # JSON has one numeric type even though Python decodes it as int or float.
        if isinstance(observed, (int, float)) and isinstance(expected, (int, float)):
            return observed == expected
        return False
    return observed == expected


def _verify_file_record(
    record: Mapping[str, Any], *, observed_size: int, observed_sha256: str, label: str
) -> None:
    expected_size = record.get("size_bytes")
    expected_sha256 = record.get("sha256")
    if type(expected_size) is not int or expected_size < 0:
        raise EvidenceAttestationError(f"{label}.size_bytes must be a non-negative integer")
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256):
        raise EvidenceAttestationError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    if (observed_size, observed_sha256) != (expected_size, expected_sha256):
        raise EvidenceAttestationError(
            f"{label} identity mismatch: observed {observed_size}/{observed_sha256}, "
            f"expected {expected_size}/{expected_sha256}"
        )


def verify_evidence_attestation(
    path: str | Path, *, repo: str | Path | None = None
) -> dict[str, Any]:
    """Load and fully verify one compact evidence attestation.

    Verification is intentionally stronger than JSON-schema validation.  It resolves the
    annotated tag, checks its exact tag object and peeled commit, proves every execution commit
    is in the archived lineage, streams every historical blob through SHA-256, and authenticates
    each compact file retained in the active tree.
    """

    attestation_path = Path(path).resolve()
    repository = _discover_repository(
        Path(repo) if repo is not None else attestation_path.parent
    )
    try:
        document = _parse_json_bytes(
            attestation_path.read_bytes(), label=f"attestation {attestation_path}"
        )
    except OSError as exc:
        raise EvidenceAttestationError(f"cannot read {attestation_path}: {exc}") from exc
    root = _require_mapping(document, "attestation")
    _require_exact_fields(
        root,
        required={"schema", "attestation_id", "records", "archive", "active_files"},
        label="attestation",
    )
    if root.get("schema") != ATTESTATION_SCHEMA:
        raise EvidenceAttestationError(
            f"schema must be {ATTESTATION_SCHEMA!r}, got {root.get('schema')!r}"
        )

    attestation_id = _require_string(root.get("attestation_id"), "attestation_id")
    if not EVIDENCE_ID_RE.fullmatch(attestation_id):
        raise EvidenceAttestationError("attestation_id must be a descriptive slash-separated slug")
    study = attestation_id.split("/", 1)[0]

    records = root.get("records")
    if not isinstance(records, list) or not records:
        raise EvidenceAttestationError("records must be a non-empty array")
    evidence_ids: set[str] = set()
    claim_aliases: set[str] = set()
    source_checks: list[tuple[str, Mapping[str, Any]]] = []
    for index, raw_record in enumerate(records):
        record = _require_mapping(raw_record, f"records[{index}]")
        _require_exact_fields(
            record,
            required={
                "id",
                "title",
                "summary",
                "record_kind",
                "status",
                "evidence_rung",
                "review_tier",
                "claim_eligible",
                "scope",
                "first_missing_bridge",
                "legacy_aliases",
                "source_checks",
            },
            label=f"records[{index}]",
        )
        evidence_id = _require_string(record.get("id"), f"records[{index}].id")
        if not EVIDENCE_ID_RE.fullmatch(evidence_id) or not evidence_id.startswith(study + "/"):
            raise EvidenceAttestationError(
                f"records[{index}].id must be a descriptive slug in study {study!r}"
            )
        if evidence_id in evidence_ids:
            raise EvidenceAttestationError(f"duplicate evidence id {evidence_id!r}")
        evidence_ids.add(evidence_id)
        _require_string(record.get("title"), f"records[{index}].title")
        _require_string(record.get("summary"), f"records[{index}].summary")
        record_kind = record.get("record_kind")
        if record_kind not in LEGAL_RECORD_KINDS:
            raise EvidenceAttestationError(
                f"records[{index}].record_kind must be one of {sorted(LEGAL_RECORD_KINDS)}"
            )
        status = record.get("status")
        if status not in LEGAL_STATUSES:
            raise EvidenceAttestationError(
                f"records[{index}].status must be one of {sorted(LEGAL_STATUSES)}"
            )
        evidence_rung = record.get("evidence_rung")
        if evidence_rung not in LEGAL_EVIDENCE_RUNGS:
            raise EvidenceAttestationError(
                f"records[{index}].evidence_rung must be one of {sorted(LEGAL_EVIDENCE_RUNGS)}"
            )
        review_tier = record.get("review_tier")
        if review_tier not in LEGAL_REVIEW_TIERS:
            raise EvidenceAttestationError(
                f"records[{index}].review_tier must be one of {sorted(LEGAL_REVIEW_TIERS)}"
            )
        if not isinstance(record.get("claim_eligible"), bool):
            raise EvidenceAttestationError(f"records[{index}].claim_eligible must be a boolean")
        _require_string(record.get("scope"), f"records[{index}].scope")
        missing_bridge = record.get("first_missing_bridge")
        if missing_bridge is not None:
            _require_string(missing_bridge, f"records[{index}].first_missing_bridge")

        aliases = record.get("legacy_aliases")
        if not isinstance(aliases, list):
            raise EvidenceAttestationError(f"records[{index}].legacy_aliases must be an array")
        record_aliases: set[tuple[str, str]] = set()
        for alias_index, raw_alias in enumerate(aliases):
            alias_label = f"records[{index}].legacy_aliases[{alias_index}]"
            alias = _require_mapping(raw_alias, alias_label)
            _require_exact_fields(alias, required={"type", "id"}, label=alias_label)
            alias_type = alias.get("type")
            if alias_type not in LEGAL_ALIAS_TYPES:
                raise EvidenceAttestationError(
                    f"{alias_label}.type must be one of {sorted(LEGAL_ALIAS_TYPES)}"
                )
            alias_id = _require_string(alias.get("id"), f"{alias_label}.id")
            identity = (alias_type, alias_id)
            if identity in record_aliases:
                raise EvidenceAttestationError(f"duplicate legacy alias {identity!r}")
            record_aliases.add(identity)
            if alias_type == "claim":
                if alias_id in claim_aliases:
                    raise EvidenceAttestationError(f"duplicate legacy alias {identity!r}")
                claim_aliases.add(alias_id)

        checks = record.get("source_checks")
        if not isinstance(checks, list) or not checks:
            raise EvidenceAttestationError(
                f"records[{index}].source_checks must be a non-empty array"
            )
        for check_index, raw_check in enumerate(checks):
            check_label = f"records[{index}].source_checks[{check_index}]"
            check = _require_mapping(raw_check, check_label)
            _require_exact_fields(
                check,
                required={"archived_path", "pointer", "equals"},
                label=check_label,
            )
            source_checks.append((check_label, check))

    archive = _require_mapping(root.get("archive"), "archive")
    _require_exact_fields(
        archive,
        required={"tag", "tag_object", "commit", "execution_commits", "files"},
        label="archive",
    )
    tag = _require_string(archive.get("tag"), "archive.tag")
    if not tag.startswith("archive/"):
        raise EvidenceAttestationError("archive.tag must use the archive/ namespace")
    try:
        valid_tag = subprocess.run(
            ["git", "check-ref-format", f"refs/tags/{tag}"],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError as exc:
        raise EvidenceAttestationError(f"cannot validate archive tag {tag!r}: {exc}") from exc
    if not valid_tag:
        raise EvidenceAttestationError(f"archive.tag must be a valid Git tag name: {tag!r}")
    tag_object = _require_string(archive.get("tag_object"), "archive.tag_object")
    commit = _require_string(archive.get("commit"), "archive.commit")
    if not GIT_OBJECT_RE.fullmatch(tag_object) or not GIT_OBJECT_RE.fullmatch(commit):
        raise EvidenceAttestationError("archive Git identities must be full lowercase object IDs")
    observed_tag_type = _git(repository, "cat-file", "-t", f"refs/tags/{tag}")
    if observed_tag_type != "tag":
        raise EvidenceAttestationError(f"archive tag {tag!r} is not annotated")
    if _git(repository, "rev-parse", f"refs/tags/{tag}") != tag_object:
        raise EvidenceAttestationError(f"archive tag object for {tag!r} differs")
    if _git(repository, "rev-parse", f"refs/tags/{tag}^{{commit}}") != commit:
        raise EvidenceAttestationError(f"archive tag {tag!r} no longer peels to {commit}")

    execution_commits = archive.get("execution_commits")
    if not isinstance(execution_commits, list) or not execution_commits:
        raise EvidenceAttestationError("archive.execution_commits must be a non-empty array")
    for index, execution_commit in enumerate(execution_commits):
        if not isinstance(execution_commit, str) or not GIT_OBJECT_RE.fullmatch(execution_commit):
            raise EvidenceAttestationError(
                f"archive.execution_commits[{index}] must be a full Git commit ID"
            )
        try:
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", execution_commit, commit],
                cwd=repository,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise EvidenceAttestationError(
                f"execution commit {execution_commit} is not in archived lineage {commit}"
            ) from exc

    archived_files = archive.get("files")
    if not isinstance(archived_files, list) or not archived_files:
        raise EvidenceAttestationError("archive.files must be a non-empty array")
    seen_paths: set[str] = set()
    archived_json_modes: dict[str, str] = {}
    for index, raw_file in enumerate(archived_files):
        record = _require_mapping(raw_file, f"archive.files[{index}]")
        _require_exact_fields(
            record,
            required={"path", "size_bytes", "sha256"},
            optional={"json_mode"},
            label=f"archive.files[{index}]",
        )
        relative = _validate_repo_path(record.get("path"), f"archive.files[{index}].path")
        if relative in seen_paths:
            raise EvidenceAttestationError(f"duplicate archived path {relative!r}")
        seen_paths.add(relative)
        observed_size, observed_sha = _historical_blob_identity(repository, commit, relative)
        _verify_file_record(
            record,
            observed_size=observed_size,
            observed_sha256=observed_sha,
            label=f"archive.files[{index}]",
        )
        json_mode = record.get("json_mode", "strict")
        if json_mode not in {"strict", "historical-nonfinite"}:
            raise EvidenceAttestationError(
                f"archive.files[{index}].json_mode must be 'strict' or 'historical-nonfinite'"
            )
        archived_json_modes[relative] = json_mode

    parsed_sources: dict[str, Any] = {}
    for check_label, check in source_checks:
        relative = _validate_repo_path(check.get("archived_path"), f"{check_label}.archived_path")
        if relative not in seen_paths:
            raise EvidenceAttestationError(
                f"{check_label}.archived_path must name an authenticated archive file"
            )
        pointer = check.get("pointer")
        if not isinstance(pointer, str):
            raise EvidenceAttestationError(f"{check_label}.pointer must be a string")
        if "equals" not in check:
            raise EvidenceAttestationError(f"{check_label}.equals is required")
        if relative not in parsed_sources:
            data = _historical_blob_bytes(repository, commit, relative)
            parsed_sources[relative] = _parse_json_bytes(
                data,
                label=f"archived file {relative}",
                allow_nonfinite=archived_json_modes[relative] == "historical-nonfinite",
            )
        observed = _resolve_json_pointer(
            parsed_sources[relative], pointer, label=f"{check_label}.pointer"
        )
        if not _json_equal(observed, check["equals"]):
            raise EvidenceAttestationError(
                f"{check_label} mismatch: observed {observed!r}, expected {check['equals']!r}"
            )

    active_files = root.get("active_files")
    if not isinstance(active_files, list):
        raise EvidenceAttestationError("active_files must be an array")
    active_paths: set[str] = set()
    for index, raw_file in enumerate(active_files):
        record = _require_mapping(raw_file, f"active_files[{index}]")
        _require_exact_fields(
            record,
            required={"path", "size_bytes", "sha256"},
            label=f"active_files[{index}]",
        )
        relative = _validate_repo_path(record.get("path"), f"active_files[{index}].path")
        if relative in active_paths:
            raise EvidenceAttestationError(f"duplicate active path {relative!r}")
        active_paths.add(relative)
        local = repository / relative
        if not local.is_file():
            raise EvidenceAttestationError(f"active file is absent: {relative}")
        if local.is_symlink():
            raise EvidenceAttestationError(f"active file must not be a symbolic link: {relative}")
        size, digest = _local_file_identity(local)
        _verify_file_record(
            record,
            observed_size=size,
            observed_sha256=digest,
            label=f"active_files[{index}]",
        )

    return dict(root)


__all__ = [
    "ATTESTATION_SCHEMA",
    "EvidenceAttestationError",
    "verify_evidence_attestation",
]
