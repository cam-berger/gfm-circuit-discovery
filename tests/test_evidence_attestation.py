from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gfmcd.evidence_attestation import (
    EvidenceAttestationError,
    verify_evidence_attestation,
)


def _run(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _write(repo: Path, relative: str, data: bytes) -> dict[str, object]:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": relative,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _fixture(
    tmp_path: Path,
    *,
    archived_bytes: bytes = b'{"raw": true}\n',
    json_mode: str | None = None,
) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init", "-q")
    _run(repo, "config", "user.name", "Test")
    _run(repo, "config", "user.email", "test@example.com")
    archived = _write(repo, "raw/package.json", archived_bytes)
    if json_mode is not None:
        archived["json_mode"] = json_mode
    active = _write(repo, "results/audit.json", b'{"audit": true}\n')
    _run(repo, "add", ".")
    _run(repo, "commit", "-qm", "archive evidence")
    commit = _run(repo, "rev-parse", "HEAD")
    tag = "archive/test-evidence"
    _run(repo, "tag", "-a", tag, "-m", "test archive")
    tag_object = _run(repo, "rev-parse", f"refs/tags/{tag}")
    document = {
        "schema": "gfmcd.evidence_attestation/1",
        "attestation_id": "acceptor-coupling/tests/package",
        "records": [
            {
                "id": "acceptor-coupling/tests/result",
                "title": "Readable result title",
                "summary": "A compact test result.",
                "record_kind": "finding",
                "status": "bounded",
                "evidence_rung": "behavior",
                "review_tier": "confirmatory",
                "claim_eligible": True,
                "scope": "Synthetic fixture only.",
                "first_missing_bridge": "mediation",
                "legacy_aliases": [{"type": "claim", "id": "AC99"}],
                "source_checks": [
                    {
                        "archived_path": "raw/package.json",
                        "pointer": "/raw",
                        "equals": True,
                    }
                ],
            }
        ],
        "archive": {
            "tag": tag,
            "tag_object": tag_object,
            "commit": commit,
            "execution_commits": [commit],
            "files": [archived],
        },
        "active_files": [active],
    }
    attestation = repo / "results/attestation.json"
    attestation.write_text(json.dumps(document, indent=2) + "\n")
    return repo, attestation


def test_verifies_annotated_archive_and_active_audit(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    observed = verify_evidence_attestation(attestation, repo=repo)
    assert observed["attestation_id"] == "acceptor-coupling/tests/package"


def test_discovers_repository_from_attestation_location(tmp_path: Path) -> None:
    _repo, attestation = _fixture(tmp_path)

    observed = verify_evidence_attestation(attestation)

    assert observed["attestation_id"] == "acceptor-coupling/tests/package"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda doc: doc["archive"].update(commit="0" * 40), "no longer peels"),
        (lambda doc: doc["archive"]["files"][0].update(size_bytes=1), "identity mismatch"),
        (lambda doc: doc["records"][0].update(id="AC99"), "descriptive slug"),
        (
            lambda doc: doc["records"].append(dict(doc["records"][0])),
            "duplicate evidence id",
        ),
    ],
)
def test_fails_closed_on_identity_or_naming_drift(tmp_path: Path, mutate, message: str) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    mutate(document)
    attestation.write_text(json.dumps(document, indent=2) + "\n")
    with pytest.raises(EvidenceAttestationError, match=message):
        verify_evidence_attestation(attestation, repo=repo)


def test_fails_when_active_audit_changes(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    (repo / "results/audit.json").write_text('{"audit": "changed"}\n')
    with pytest.raises(EvidenceAttestationError, match="identity mismatch"):
        verify_evidence_attestation(attestation, repo=repo)


def test_attestation_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    text = attestation.read_text().replace(
        '"schema": "gfmcd.evidence_attestation/1",',
        '"schema": "gfmcd.evidence_attestation/1",\n  "schema": "gfmcd.evidence_attestation/1",',
        1,
    )
    attestation.write_text(text)

    with pytest.raises(EvidenceAttestationError, match="duplicate object key 'schema'"):
        verify_evidence_attestation(attestation, repo=repo)


def test_attestation_json_rejects_nonfinite_constants(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    text = attestation.read_text().replace('"equals": true', '"equals": NaN', 1)
    attestation.write_text(text)

    with pytest.raises(EvidenceAttestationError, match="non-standard JSON constant NaN"):
        verify_evidence_attestation(attestation, repo=repo)


def test_attestation_schema_rejects_unknown_fields(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    document["unreviewed_note"] = "typo"
    attestation.write_text(json.dumps(document, indent=2) + "\n")

    with pytest.raises(EvidenceAttestationError, match="attestation has unknown field"):
        verify_evidence_attestation(attestation, repo=repo)


def test_source_check_reports_json_pointer_mismatch(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    document["records"][0]["source_checks"][0]["equals"] = False
    attestation.write_text(json.dumps(document, indent=2) + "\n")

    with pytest.raises(EvidenceAttestationError, match=r"source_checks\[0\] mismatch"):
        verify_evidence_attestation(attestation, repo=repo)


def test_historical_nonfinite_json_can_support_a_finite_source_check(tmp_path: Path) -> None:
    repo, attestation = _fixture(
        tmp_path,
        archived_bytes=b'{"raw": true, "legacy_metric": NaN}\n',
        json_mode="historical-nonfinite",
    )

    observed = verify_evidence_attestation(attestation, repo=repo)

    assert observed["records"][0]["status"] == "bounded"


def test_strict_archived_json_rejects_historical_nonfinite_constants(tmp_path: Path) -> None:
    repo, attestation = _fixture(
        tmp_path,
        archived_bytes=b'{"raw": true, "legacy_metric": NaN}\n',
    )

    with pytest.raises(EvidenceAttestationError, match="non-standard JSON constant NaN"):
        verify_evidence_attestation(attestation, repo=repo)


def test_source_check_equality_distinguishes_json_booleans_from_numbers(
    tmp_path: Path,
) -> None:
    repo, attestation = _fixture(tmp_path, archived_bytes=b'{"raw": 1}\n')

    with pytest.raises(EvidenceAttestationError, match="mismatch"):
        verify_evidence_attestation(attestation, repo=repo)


def test_source_checks_run_only_after_archive_bytes_authenticate(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    document["archive"]["files"][0]["sha256"] = "0" * 64
    document["records"][0]["source_checks"][0]["pointer"] = "/absent"
    attestation.write_text(json.dumps(document, indent=2) + "\n")

    with pytest.raises(EvidenceAttestationError, match="identity mismatch"):
        verify_evidence_attestation(attestation, repo=repo)


def test_file_sizes_require_json_integers_not_booleans(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    document["archive"]["files"][0]["size_bytes"] = True
    attestation.write_text(json.dumps(document, indent=2) + "\n")

    with pytest.raises(EvidenceAttestationError, match="non-negative integer"):
        verify_evidence_attestation(attestation, repo=repo)


def test_typed_legacy_aliases_map_to_only_one_semantic_record(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    duplicate = json.loads(json.dumps(document["records"][0]))
    duplicate["id"] = "acceptor-coupling/tests/second-result"
    document["records"].append(duplicate)
    attestation.write_text(json.dumps(document, indent=2) + "\n")

    with pytest.raises(EvidenceAttestationError, match="duplicate legacy alias"):
        verify_evidence_attestation(attestation, repo=repo)


def test_run_and_correction_aliases_may_govern_multiple_records(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    shared_aliases = [
        {"type": "run", "id": "M6"},
        {"type": "correction", "id": "ACR20"},
    ]
    document["records"][0]["legacy_aliases"] = shared_aliases
    second = json.loads(json.dumps(document["records"][0]))
    second["id"] = "acceptor-coupling/tests/second-result"
    document["records"].append(second)
    attestation.write_text(json.dumps(document, indent=2) + "\n")

    observed = verify_evidence_attestation(attestation, repo=repo)

    assert len(observed["records"]) == 2


def test_descriptive_ids_have_exactly_three_segments(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    document["records"][0]["id"] = "acceptor-coupling/tests/extra/result"
    attestation.write_text(json.dumps(document, indent=2) + "\n")

    with pytest.raises(
        EvidenceAttestationError,
        match="descriptive slug",
    ):
        verify_evidence_attestation(attestation, repo=repo)


def test_archive_only_attestation_may_have_no_active_files(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    document["active_files"] = []
    attestation.write_text(json.dumps(document, indent=2) + "\n")

    observed = verify_evidence_attestation(attestation, repo=repo)

    assert observed["active_files"] == []


def test_active_file_records_must_be_unique(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    document["active_files"].append(dict(document["active_files"][0]))
    attestation.write_text(json.dumps(document, indent=2) + "\n")

    with pytest.raises(EvidenceAttestationError, match="duplicate active path"):
        verify_evidence_attestation(attestation, repo=repo)


def test_archive_tag_must_be_a_literal_valid_ref_name(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    document = json.loads(attestation.read_text())
    document["archive"]["tag"] = "archive/test^{commit}"
    attestation.write_text(json.dumps(document, indent=2) + "\n")

    with pytest.raises(EvidenceAttestationError, match="valid Git tag name"):
        verify_evidence_attestation(attestation, repo=repo)


def test_cli_verifies_attestation_and_reports_semantic_identity(tmp_path: Path) -> None:
    repo, attestation = _fixture(tmp_path)
    script = Path(__file__).parents[1] / "scripts" / "verify_archived_evidence.py"

    completed = subprocess.run(
        [sys.executable, str(script), str(attestation)],
        cwd=repo,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "VERIFIED acceptor-coupling/tests/package (1 record)" in completed.stdout
