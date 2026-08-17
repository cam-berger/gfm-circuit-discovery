from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

from gfmcd.evidence_attestation import verify_evidence_attestation


REPO = Path(__file__).resolve().parents[1]
TAG = "archive/acceptor-inverse-coupling-2026-08-17"
TAG_OBJECT = "d71568c158e402cd14b3c816ef11110a508e2026"
ARCHIVE_COMMIT = "16a8bd0b2e716c229488cdf89be76ac97b174eac"

R1C_ATTESTATION = (
    REPO
    / "results/splicebert_acceptor_coupling/attestations/"
    "frozen_handoff_schema_mismatch_v1.json"
)
M6_ATTESTATION = (
    REPO
    / "results/splicebert_acceptor_coupling/attestations/"
    "natural_gc_validation_v1.json"
)
M4D_ATTESTATION = (
    REPO
    / "results/splicebert_acceptor_coupling/attestations/development/"
    "natural_gc_qk_score_diagnostic_v1.json"
)

R1C_ARTIFACT = (
    "results/splicebert_acceptor_coupling/r1c_reverse_behavior_preflight_v4.json"
)
R1C_MANIFEST = (
    "results/splicebert_acceptor_coupling/"
    "r1c_reverse_behavior_preflight_v4.manifest.json"
)
R1C_CONSUMER = "src/gfmcd/splicing/acceptor_mechanism_cohort.py"
R1C_RUNNER = "scripts/splicebert_acceptor_coupling/run_r1c_sealed_reverse_behavior.py"
R1C_PROTOCOL = (
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R1C_SEALED_REVERSE_BEHAVIOR_CONFIRMATION.md"
)
R1C_EXECUTION = "9008ba3b923e5f8a272ba08bec0799444d44b24a"

M6_ARTIFACT = "results/splicebert_acceptor_coupling/m6_validation_confirmation.json"
M6_MANIFEST = (
    "results/splicebert_acceptor_coupling/m6_validation_confirmation.manifest.json"
)
M6_PROTOCOL = (
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "PREREGISTRATION_M6_VALIDATION_CONFIRMATION.md"
)
M6_RUNNER = "scripts/splicebert_acceptor_coupling/run_m6_validation_confirmation.py"
M6_EXECUTION = "42954a67947ed7d2748c57816661b917cf5d78e8"

M4D_ARTIFACT = (
    "results/splicebert_acceptor_coupling/development/m4d_score_decomposition.json"
)
M4D_MANIFEST = (
    "results/splicebert_acceptor_coupling/development/"
    "m4d_score_decomposition.manifest.json"
)
M4D_EXECUTION = "b79570d68a5cde9a03f33af516fb369a26267892"


def _git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "cat-file", "blob", f"{commit}:{path}"],
        cwd=REPO,
    )


def _file_identities(records: list[dict[str, object]]) -> dict[str, tuple[int, str]]:
    return {
        str(record["path"]): (int(record["size_bytes"]), str(record["sha256"]))
        for record in records
    }


def _source_checks(record: dict[str, object]) -> dict[tuple[str, str], object]:
    return {
        (str(check["archived_path"]), str(check["pointer"])): check["equals"]
        for check in record["source_checks"]
    }


def test_frozen_handoff_attestation_exposes_real_terminal_schema_mismatch() -> None:
    observed = verify_evidence_attestation(R1C_ATTESTATION, repo=REPO)

    assert observed["attestation_id"] == (
        "acceptor-coupling/instrument/frozen-handoff-schema-mismatch"
    )
    assert observed["archive"] | {"files": []} == {
        "tag": TAG,
        "tag_object": TAG_OBJECT,
        "commit": ARCHIVE_COMMIT,
        "execution_commits": [R1C_EXECUTION],
        "files": [],
    }
    assert observed["active_files"] == []
    assert _file_identities(observed["archive"]["files"]) == {
        R1C_ARTIFACT: (
            74_352_692,
            "1b27c4132c71a576c435b7b71f32831af9a44fbca6325931f35716ae4133cbdd",
        ),
        R1C_MANIFEST: (
            11_370,
            "7412f3159e82f6ab0bbaa95e2796f41aa038eef4233efa7f09ec65867e39c45c",
        ),
        R1C_RUNNER: (
            203_036,
            "28203db871fd2ffc1e80cbf71efb7a7079941fd4008373b89144f75750c49f3b",
        ),
        R1C_CONSUMER: (
            130_458,
            "7307568901810b25cc2a84a18dac874565bccd48fcd865224ee2eef5481b8b49",
        ),
        R1C_PROTOCOL: (
            34_008,
            "2a2ac9dec6fb52b551cd87f96f7e3f07d6bc29431539011ebfaca2ce8c08da76",
        ),
    }

    record = observed["records"][0]
    assert record["id"] == observed["attestation_id"]
    assert record["record_kind"] == "instrument"
    assert record["status"] == "instrument-unresolved"
    assert record["evidence_rung"] == "instrument"
    assert record["review_tier"] == "confirmatory"
    assert record["claim_eligible"] is False
    assert {
        (alias["type"], alias["id"]) for alias in record["legacy_aliases"]
    } >= {("claim", "AC9"), ("correction", "ACR19"), ("run", "R1c v4")}

    checks = _source_checks(record)
    assert checks[(R1C_ARTIFACT, "/verdict")] == "PREFLIGHT_GO"
    assert checks[(R1C_ARTIFACT, "/n")] == 400
    assert checks[(R1C_ARTIFACT, "/n_genes")] == 321
    assert len(checks[(R1C_ARTIFACT, "/gates")]) == 13
    assert all(checks[(R1C_ARTIFACT, "/gates")].values())
    producer_keys = set(checks[(R1C_ARTIFACT, "/rows/0/margin_levels")])

    consumer_source = _git_blob(ARCHIVE_COMMIT, R1C_CONSUMER)
    assert hashlib.sha256(consumer_source).hexdigest() == (
        "7307568901810b25cc2a84a18dac874565bccd48fcd865224ee2eef5481b8b49"
    )
    module = ast.parse(consumer_source)
    extractor = next(
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_r1c_ordered_partner_margin_levels"
    )
    string_sets = [
        {element.value for element in node.elts}
        for node in ast.walk(extractor)
        if isinstance(node, ast.Set)
        and all(isinstance(element, ast.Constant) and isinstance(element.value, str) for element in node.elts)
    ]
    consumer_keys = {"P_clean", "P_under_D", "P_under_N"}
    assert consumer_keys in string_sets
    assert producer_keys == consumer_keys | {"X_clean", "X_under_D", "X_under_N"}
    assert producer_keys != consumer_keys


def test_natural_gc_validation_attestation_binds_bounded_modifier_records() -> None:
    observed = verify_evidence_attestation(M6_ATTESTATION, repo=REPO)

    assert observed["attestation_id"] == "acceptor-coupling/natural-gc/validation-modifiers"
    assert observed["archive"]["tag"] == TAG
    assert observed["archive"]["tag_object"] == TAG_OBJECT
    assert observed["archive"]["commit"] == ARCHIVE_COMMIT
    assert observed["archive"]["execution_commits"] == [M6_EXECUTION]
    assert _file_identities(observed["archive"]["files"]) == {
        M6_ARTIFACT: (
            3_663,
            "80150013b084cb659c7e3a0e1739573fd1a12e0c95c8cc433067ec9929d4051a",
        ),
        M6_MANIFEST: (
            2_216,
            "4e470908b93885133a817a12eee2b074c8008f3d0b3aafbfb4ac596ae27216df",
        ),
        M6_PROTOCOL: (
            5_263,
            "da326e4da94b5daa916f59f7268001e290ea00ab5ef2eb8da255c4dd477bf46a",
        ),
    }
    assert _file_identities(observed["active_files"]) == {
        M6_PROTOCOL: (
            5_263,
            "da326e4da94b5daa916f59f7268001e290ea00ab5ef2eb8da255c4dd477bf46a",
        )
    }

    records = {record["id"]: record for record in observed["records"]}
    assert set(records) == {
        "acceptor-coupling/natural-gc/behavior-modifier",
        "acceptor-coupling/natural-gc/l3h1-attention-modifier",
    }
    behavior = records["acceptor-coupling/natural-gc/behavior-modifier"]
    attention = records["acceptor-coupling/natural-gc/l3h1-attention-modifier"]
    for record, rung, claim in (
        (behavior, "behavior", "AC10"),
        (attention, "routing-attention", "AC11"),
    ):
        assert record["record_kind"] == "finding"
        assert record["status"] == "bounded"
        assert record["evidence_rung"] == rung
        assert record["review_tier"] == "verification"
        assert record["claim_eligible"] is True
        assert {
            alias["id"]
            for alias in record["legacy_aliases"]
            if alias["type"] == "claim"
        } == {claim}

    behavior_checks = _source_checks(behavior)
    attention_checks = _source_checks(attention)
    primary = behavior_checks[(M6_ARTIFACT, "/primary")]
    secondary = attention_checks[(M6_ARTIFACT, "/secondary")]
    assert (primary["rho"], primary["upper_bound"], primary["null"]) == (
        0.2570867642398911,
        0.5402253140142983,
        0.6,
    )
    assert (secondary["ratio"], secondary["upper_bound"], secondary["null"]) == (
        0.37554981086464917,
        0.5808928386872912,
        0.7,
    )
    assert behavior_checks[(M6_MANIFEST, "/git_dirty")] is False
    assert behavior_checks[(M6_MANIFEST, "/inputs/0/missing")] is True

    active_protocol = (REPO / M6_PROTOCOL).read_bytes()
    assert active_protocol == _git_blob(M6_EXECUTION, M6_PROTOCOL)
    assert active_protocol == _git_blob(ARCHIVE_COMMIT, M6_PROTOCOL)
    assert hashlib.sha256(active_protocol).hexdigest() == (
        "da326e4da94b5daa916f59f7268001e290ea00ab5ef2eb8da255c4dd477bf46a"
    )
    execution_runner = _git_blob(M6_EXECUTION, M6_RUNNER)
    assert hashlib.sha256(execution_runner).hexdigest() == (
        "bdbf9c953b23c245734802890db88081ef8a3b1f58c6e4403004f4debaf36b08"
    )


def test_qk_score_attestation_is_nonclaim_development_diagnostic() -> None:
    observed = verify_evidence_attestation(M4D_ATTESTATION, repo=REPO)

    assert observed["attestation_id"] == "acceptor-coupling/natural-gc/qk-score-diagnostic"
    assert observed["archive"]["tag"] == TAG
    assert observed["archive"]["tag_object"] == TAG_OBJECT
    assert observed["archive"]["commit"] == ARCHIVE_COMMIT
    assert observed["archive"]["execution_commits"] == [M4D_EXECUTION]
    assert _file_identities(observed["archive"]["files"]) == {
        M4D_ARTIFACT: (
            15_313,
            "d1d960296e53be684909e129b6de3ae4ddc9b15e1a688c0813109d7db337f763",
        ),
        M4D_MANIFEST: (
            2_189,
            "879b262b93ab108e545f350b071a62cd715bb7012872455f31ce76a1562af165",
        ),
    }
    artifact_record = next(
        record for record in observed["archive"]["files"] if record["path"] == M4D_ARTIFACT
    )
    assert artifact_record["json_mode"] == "historical-nonfinite"
    assert observed["active_files"] == []

    record = observed["records"][0]
    assert record["id"] == observed["attestation_id"]
    assert record["record_kind"] == "diagnostic"
    assert record["status"] == "development"
    assert record["evidence_rung"] == "selection-qk"
    assert record["review_tier"] == "development"
    assert record["claim_eligible"] is False
    assert not any(alias["type"] == "claim" for alias in record["legacy_aliases"])
    assert {alias["id"] for alias in record["legacy_aliases"]} >= {"ACR23", "M4d"}

    checks = _source_checks(record)
    expected_differences = {
        "L0H14": (0.4290274113420788, 0.26835828929497346, 0.5881876510020364),
        "L0H5": (0.25021834648738345, 0.09456241375593062, 0.39673034174685123),
        "L0H3": (0.21133896622063805, 0.06436592265566, 0.35371369251122026),
        "L0H4": (0.10916772074013213, -0.09700238710530677, 0.310173669613979),
        "L1H6": (0.07126043385026667, -0.02362878693998345, 0.16951856180460226),
        "L1H13": (-0.008756685080273224, -0.11815828158611592, 0.09723623288392326),
        "L2H15": (-0.45538115572484833, -0.6976525417572375, -0.22232480222256784),
    }
    for head, expected in expected_differences.items():
        margin_shift = checks[(M4D_ARTIFACT, f"/probes/{head}/margin_shift")]
        assert (
            margin_shift["difference"],
            margin_shift["ci"][0],
            margin_shift["ci"][1],
        ) == expected
    assert checks[(M4D_MANIFEST, "/git_dirty")] is True
    assert checks[(M4D_MANIFEST, "/inputs/0/missing")] is True

    raw_artifact = _git_blob(ARCHIVE_COMMIT, M4D_ARTIFACT)
    assert b'"log_ratio": NaN' in raw_artifact
    historical = json.loads(raw_artifact)
    assert math.isnan(historical["probes"]["L0H3"]["decomposition"]["log_ratio"])
