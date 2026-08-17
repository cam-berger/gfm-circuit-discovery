from __future__ import annotations

import json
from pathlib import Path

from gfmcd.evidence_attestation import verify_evidence_attestation


REPO = Path(__file__).resolve().parents[1]
ATTESTATION = (
    REPO
    / "results/splicebert_acceptor_coupling/attestations/"
    "paired_donor_component_confirmation_v1.json"
)
AUDIT = (
    REPO
    / "results/splicebert_acceptor_coupling/"
    "r3c_acceptor_head_target_v2_1.audit.json"
)

TAG = "archive/acceptor-inverse-coupling-2026-08-17"
TAG_OBJECT = "d71568c158e402cd14b3c816ef11110a508e2026"
ARCHIVE_COMMIT = "16a8bd0b2e716c229488cdf89be76ac97b174eac"
PREFLIGHT_PATH = (
    "results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.json"
)
TARGET_PATH = (
    "results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.json"
)

ARCHIVED_IDENTITIES = {
    PREFLIGHT_PATH: (
        33_361_804,
        "c0c9306fe894ce73bfd981a14c85e2aec6660da33f01d9716cee4327b3ff2d3e",
    ),
    "results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.manifest.json": (
        9_620,
        "4a53f96b365c7bed227c8e0c995cee65a022620994e8898bab988b354fce41bb",
    ),
    TARGET_PATH: (
        33_412_449,
        "ebd4bfd8c10cc0ccb218b3f1ceca634605903d8afbb5617de312087100627cc6",
    ),
    "results/splicebert_acceptor_coupling/r3c_acceptor_head_target_v2_1.manifest.json": (
        10_319,
        "dc644901224227f03e683f7416587dccf3eedd700052b211089242b9b8d69cbe",
    ),
}

ACTIVE_IDENTITIES = {
    "studies/splicebert_acceptor_coupling/preregistrations/"
    "DRAFT_R3C_ACCEPTOR_HEAD_CONFIRMATION.md": (
        24_074,
        "c0239606079aa985fd173f31bee68ec154e162af54ede74ee5315033c8e22c38",
    ),
    "results/splicebert_acceptor_coupling/"
    "r3c_acceptor_head_target_v2_1.audit.json": (
        3_092,
        "0363d02dced48d0d1a8ba56f1862be4a0cbb0b82eb508f9aeb87b8c61b05edb3",
    ),
}

EXPECTED_RECORDS = {
    "acceptor-coupling/components/l1h12-mean-ablation": (
        "supported",
        True,
        "finding",
        "AC5",
        ("500-row", "304-gene", "L1H12", "seven-path"),
    ),
    "acceptor-coupling/components/l4-head-noising": (
        "supported",
        True,
        "finding",
        "AC6",
        ("L4H11", "L4H9", "L4H7", "seven-path"),
    ),
    "acceptor-coupling/components/route-hypotheses-inconclusive": (
        "inconclusive",
        False,
        "limit",
        "AC7",
        ("L3H1", "distinct-acceptor-route", "inconclusive"),
    ),
    "acceptor-coupling/components/l2h15-health-ineligible": (
        "ineligible",
        False,
        "limit",
        "AC8",
        ("L2H15", "monotone authorization", "health-failed"),
    ),
}

EXPECTED_CLASSIFICATIONS = {
    "L3H1_reused_component": "INCONCLUSIVE_METHOD_DISAGREEMENT",
    "L2H15_mean_component": "HEALTH_FAILED",
    "L1H12_mean_component": "SUPPORTED",
    "L4H11_noising_component": "SUPPORTED",
    "L4H9_noising_component": "SUPPORTED",
    "L4H7_noising_component": "SUPPORTED",
    "distinct_acceptor_route_candidate": "INCONCLUSIVE_METHOD_DISAGREEMENT",
}


def _file_identities(records: list[dict[str, object]]) -> dict[str, tuple[int, str]]:
    return {
        str(record["path"]): (int(record["size_bytes"]), str(record["sha256"]))
        for record in records
    }


def test_archived_r3c_package_and_semantic_records_verify() -> None:
    observed = verify_evidence_attestation(ATTESTATION, repo=REPO)

    assert observed["attestation_id"] == "acceptor-coupling/components/held-out-confirmation"
    assert observed["archive"]["tag"] == TAG
    assert observed["archive"]["tag_object"] == TAG_OBJECT
    assert observed["archive"]["commit"] == ARCHIVE_COMMIT
    assert observed["archive"]["execution_commits"] == [
        "bb22dff7e2f3ec4e3de73a8d45ad1740793b66f4",
        "8dde21ce043d7d9e4835e7ccc5851d13a3af4a36",
    ]
    assert _file_identities(observed["archive"]["files"]) == ARCHIVED_IDENTITIES
    assert _file_identities(observed["active_files"]) == ACTIVE_IDENTITIES

    records = {record["id"]: record for record in observed["records"]}
    assert set(records) == set(EXPECTED_RECORDS)
    for evidence_id, (
        status,
        claim_eligible,
        record_kind,
        legacy_claim,
        summary_terms,
    ) in EXPECTED_RECORDS.items():
        record = records[evidence_id]
        assert record["record_kind"] == record_kind
        assert record["status"] == status
        assert record["evidence_rung"] == "output-ov"
        assert record["review_tier"] == "confirmatory"
        assert record["claim_eligible"] is claim_eligible
        assert record["scope"]
        assert record["first_missing_bridge"]
        assert all(term in record["summary"] for term in summary_terms)
        assert {alias["id"] for alias in record["legacy_aliases"] if alias["type"] == "claim"} == {
            legacy_claim
        }
        assert {
            alias["type"]
            for alias in record["legacy_aliases"]
        } >= {"claim", "run", "protocol", "artifact"}
        assert all(
            check["archived_path"] in {PREFLIGHT_PATH, TARGET_PATH}
            for check in record["source_checks"]
        )

    preflight_checks = {
        check["pointer"]: check["equals"]
        for check in records["acceptor-coupling/components/l1h12-mean-ablation"][
            "source_checks"
        ]
        if check["archived_path"] == PREFLIGHT_PATH
    }
    assert preflight_checks == {
        "/stage": "preflight",
        "/status": "DEVELOPMENT",
        "/verdict": "TARGET_BINDABLE_PREFLIGHT_GO",
        "/git_authorization_identity/commit": (
            "bb22dff7e2f3ec4e3de73a8d45ad1740793b66f4"
        ),
    }


def test_compact_audit_preserves_cohort_classifications_and_license_boundary() -> None:
    audit = json.loads(AUDIT.read_text())

    assert audit["schema"] == "gfmcd.r3c.acceptor_head_confirmation.audit/1"
    assert audit["status"] == "AUDIT_PASS_MIXED_SCIENTIFIC_OUTCOME"
    assert (
        audit["audited_artifact"]["size_bytes"],
        audit["audited_artifact"]["sha256"],
    ) == ARCHIVED_IDENTITIES[TARGET_PATH]
    assert audit["bound_preflight"] == {
        "artifact_sha256": ARCHIVED_IDENTITIES[
            "results/splicebert_acceptor_coupling/r3c_acceptor_head_preflight_v2_1.json"
        ][1],
        "manifest_sha256": ARCHIVED_IDENTITIES[
            "results/splicebert_acceptor_coupling/"
            "r3c_acceptor_head_preflight_v2_1.manifest.json"
        ][1],
    }
    assert audit["instrument"]["clean_provenance_commit"] == (
        "8dde21ce043d7d9e4835e7ccc5851d13a3af4a36"
    )
    assert audit["cohort"] == {
        "stage": "target",
        "gene_bucket": 1,
        "rows": 500,
        "genes": 304,
        "chromosomes": ["chr11", "chr16", "chr19"],
        "historical_r3_gene_overlap": 0,
        "bucket_0_gene_overlap": 0,
        "sealed_chr8_chr9_opened": False,
        "masked_donor_endpoint_constructed": False,
    }
    assert audit["path_classifications"] == EXPECTED_CLASSIFICATIONS
    assert "L1H12" in audit["licensed_result"]
    assert all(head in audit["licensed_result"] for head in ("L4H11", "L4H9", "L4H7"))
    assert audit["not_licensed"] == [
        "L3H1 reuse",
        "L2H15 component support",
        "a distinct acceptor route",
        "address or payload identity",
        "write or delivery mediation",
        "sufficiency",
        "a directed or complete circuit",
        "a biological mechanism",
    ]
    assert audit["audit_execution"] == {
        "model_rescored": False,
        "target_cohort_reopened": False,
        "chr8_chr9_accessed": False,
        "independent_exact_replays": 2,
    }
