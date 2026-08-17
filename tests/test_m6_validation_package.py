"""Model-free memorial for the controlling natural donor-class M6 package.

The validation reserve is spent.  These tests authenticate the first clean package and its
historical source without reopening the cohort, loading the checkpoint, or rerunning inference.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "results/splicebert_acceptor_coupling/m6_validation_confirmation.json"
MANIFEST = RESULT.with_name("m6_validation_confirmation.manifest.json")
PROTOCOL = (
    REPO
    / "studies/splicebert_acceptor_coupling/preregistrations/"
    "PREREGISTRATION_M6_VALIDATION_CONFIRMATION.md"
)
RUN_COMMIT = "42954a67947ed7d2748c57816661b917cf5d78e8"
RESULT_SHA = "80150013b084cb659c7e3a0e1739573fd1a12e0c95c8cc433067ec9929d4051a"
MANIFEST_SHA = "4e470908b93885133a817a12eee2b074c8008f3d0b3aafbfb4ac596ae27216df"
PROTOCOL_SHA = "da326e4da94b5daa916f59f7268001e290ea00ab5ef2eb8da255c4dd477bf46a"
RUNNER_SHA = "bdbf9c953b23c245734802890db88081ef8a3b1f58c6e4403004f4debaf36b08"
SUPERSEDED_RESULT_SHA = "726fafee99fd580dd9b05f243703a90b3b0f158bd9bccef6fae1650927c61c32"
SUPERSEDED_MANIFEST_SHA = "ba7a8d144ff96d0ff5733da82ded3770c9666f5cc6f7349d146d8c906ca4eb06"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _git_show_bytes(revision_path: str) -> bytes:
    return subprocess.run(
        ["git", "show", revision_path],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout


def test_first_clean_m6_package_has_exact_immutable_identities() -> None:
    assert RESULT.stat().st_size == 3_663
    assert MANIFEST.stat().st_size == 2_216
    assert _sha256(RESULT) == RESULT_SHA
    assert _sha256(MANIFEST) == MANIFEST_SHA
    assert _sha256(PROTOCOL) == PROTOCOL_SHA

    record = json.loads(RESULT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    output = manifest["outputs"]
    assert output == [
        {
            "path": output[0]["path"],
            "sha256": RESULT_SHA,
            "size_bytes": 3_663,
        }
    ]
    assert manifest["git_commit"] == RUN_COMMIT
    assert manifest["git_dirty"] is False
    assert manifest["config"]["protocol_sha256"] == PROTOCOL_SHA
    assert record["protocol_sha256"] == PROTOCOL_SHA
    assert manifest["status"] == "CONFIRMATORY"
    assert record["tier"] == "confirmatory"


def test_m6_scientific_fields_and_memorials_are_exact_but_scoped() -> None:
    record = json.loads(RESULT.read_text())
    manifest = json.loads(MANIFEST.read_text())

    assert record["split"] == "val"
    assert record["sealed_chr8_chr9_opened"] is False
    assert record["validation_reserve_spent"] is True
    assert record["n_pairs"] == 67
    assert record["n_genes"] == 56
    assert record["cohort"]["row_sha256"] == (
        "0dff7e6fd314d237407c926fd13e5784bfbbd9138b24b78ef74e288a922301d9"
    )
    assert record["cohort"]["gene_sha256"] == (
        "cb0fe02d5fda8ecbdcae454e2647e16fee43f259184de41f26c1541ced609b51"
    )
    assert record["primary"] == {
        "estimand": "mean(Y|GC-AG)/mean(Y|matched GT-AG), operator GG",
        "rho": 0.2570867642398911,
        "upper_bound": 0.5402253140142983,
        "null": 0.6,
        "verdict": "CONFIRMED",
        "train_reference": [0.242, 0.173, 0.326],
    }
    assert record["secondary"] == {
        "estimand": "L3H1 donor attention mass ratio at the readout token",
        "ratio": 0.37554981086464917,
        "upper_bound": 0.5808928386872912,
        "null": 0.7,
        "verdict": "CONFIRMED",
        "train_reference": 0.3644,
    }
    assert record["all_memorials_pass"] is True
    assert all(item["PASS"] for item in record["memorials"].values())

    # The observed checkpoint directory was not content-bound.  This is why the immutable
    # self-label above is downgraded to bounded verification in AC10/AC11.
    checkpoint = manifest["inputs"][0]
    assert checkpoint["missing"] is True
    assert "sha256" not in checkpoint


def test_historical_authorising_source_is_recoverable_and_exposes_the_hash_guard_gap() -> None:
    runner = _git_show_bytes(
        f"{RUN_COMMIT}:scripts/splicebert_acceptor_coupling/"
        "run_m6_validation_confirmation.py"
    )
    protocol = _git_show_bytes(
        f"{RUN_COMMIT}:studies/splicebert_acceptor_coupling/preregistrations/"
        "PREREGISTRATION_M6_VALIDATION_CONFIRMATION.md"
    )
    assert _sha256_bytes(runner) == RUNNER_SHA
    assert _sha256_bytes(protocol) == PROTOCOL_SHA
    assert protocol == PROTOCOL.read_bytes()

    source = runner.decode()
    assert "if a.protocol_sha256 is None:" in source
    assert "protocol_sha256\": a.protocol_sha256" in source
    assert "matching this document" not in source


def test_only_the_first_package_is_live_evidence() -> None:
    index = json.loads((REPO / "results/MANIFEST.json").read_text())
    rows = [row for row in index["rows"] if row.get("claim_id") == "AC10;AC11"]
    assert len(rows) == 2
    artifact_row = next(row for row in rows if row["evidence_level"])
    assert artifact_row["status"] == "canonical"
    assert artifact_row["evidence_level"] == "Behaviour"
    assert artifact_row["polarity"] == "bounded"
    assert artifact_row["artifact"] == (
        "results/splicebert_acceptor_coupling/m6_validation_confirmation.json"
    )

    m6_files = sorted(
        path.name
        for path in (REPO / "results/splicebert_acceptor_coupling").glob("m6*")
    )
    assert m6_files == [
        "m6_validation_confirmation.json",
        "m6_validation_confirmation.manifest.json",
    ]
    live_bytes = b"\n".join(
        path.read_bytes()
        for path in (REPO / "results/MANIFEST.csv", REPO / "results/MANIFEST.json", RESULT, MANIFEST)
    )
    assert SUPERSEDED_RESULT_SHA.encode() not in live_bytes
    assert SUPERSEDED_MANIFEST_SHA.encode() not in live_bytes
