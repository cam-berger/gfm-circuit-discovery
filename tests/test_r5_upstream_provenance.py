from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil

import pytest

from gfmcd.splicing.r5_upstream_provenance import (
    R3C_PREFLIGHT_ARTIFACT_SHA256,
    R3C_PREFLIGHT_MANIFEST_SHA256,
    R3D_REGISTRY_ARTIFACT_SHA256,
    R3D_REGISTRY_MANIFEST_SHA256,
    R3D_TARGET_ARTIFACT_SHA256,
    R3D_TARGET_COMPLETION_SHA256,
    R3D_TARGET_MANIFEST_SHA256,
    R3D_TARGET_ATTEMPT_SHA256,
    R5_EXCLUSION_REGISTRY_SCHEMA,
    R5UpstreamProvenanceError,
    canonical_sha256,
    derive_r5_exclusion_registry,
    load_r3c_exclusion_package,
    load_r3d_entry_package,
    validate_r3d_target_record,
    validate_r5_exclusion_registry,
)


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results/splicebert_acceptor_coupling"
TARGET_ID = "b8ff2fedfd3add86c88d882a0b1c36788e2f41cafb583ff1179c4779d0d4f839"


def _relocate(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def _materialize_bound_receipts(target: Path, directory: Path) -> tuple[Path, Path]:
    target_record = json.loads(target.read_text())
    target_manifest = json.loads(target.with_suffix(".manifest.json").read_text())
    recorded_result_path = str(target_manifest["outputs"][0]["path"])
    recorded_manifest_path = str(Path(recorded_result_path).with_suffix(".manifest.json"))
    attempt_record = target_record["target_attempt"]["receipt"]
    attempt = directory / f"{TARGET_ID}.json"
    attempt.write_text(json.dumps(attempt_record, indent=2, sort_keys=True) + "\n")
    completion_record = {
        "attempt_id": "3f140f56ccd69b991a98199af4c9f96c10aedd9e8256906651482e8fc997394d",
        "attempt_sha256": R3D_TARGET_ATTEMPT_SHA256,
        "closed_at_utc": "2026-08-17T00:56:38.990133+00:00",
        "details": {
            "manifest": {
                "path": recorded_manifest_path,
                "sha256": R3D_TARGET_MANIFEST_SHA256,
                "size_bytes": 13_255,
            },
            "official_path_classifications": {
                "L3H1_reused_component": "SUPPORTED",
                "distinct_acceptor_route_candidate": "SUPPORTED",
            },
            "result": {
                "path": recorded_result_path,
                "sha256": R3D_TARGET_ARTIFACT_SHA256,
                "size_bytes": 10_267_233,
            },
            "verdict": "PROSPECTIVE_TRAIN_ENDPOINT_VERIFICATION_COMPLETE",
        },
        "schema": "gfmcd.r3d.target_attempt_completion/1",
        "status": "TARGET_PUBLICATION_COMPLETE",
    }
    completion = directory / f"{TARGET_ID}.completion.json"
    completion.write_text(
        json.dumps(completion_record, indent=2, sort_keys=True) + "\n"
    )
    return attempt, completion


def test_exact_upstream_packages_are_relocation_safe_and_path_a_is_the_only_entry_gate(
    tmp_path,
):
    target = _relocate(
        RESULTS / "r3d_acceptor_specificity_target_v1.json",
        tmp_path / "relocated/r3d_acceptor_specificity_target_v1.json",
    )
    target_manifest = _relocate(
        RESULTS / "r3d_acceptor_specificity_target_v1.manifest.json",
        tmp_path / "relocated/r3d_acceptor_specificity_target_v1.manifest.json",
    )
    registry = _relocate(
        RESULTS / "r3d_train_exposure_registry_v1.json",
        tmp_path / "relocated/r3d_train_exposure_registry_v1.json",
    )
    registry_manifest = _relocate(
        RESULTS / "r3d_train_exposure_registry_v1.manifest.json",
        tmp_path / "relocated/r3d_train_exposure_registry_v1.manifest.json",
    )
    attempt, completion = _materialize_bound_receipts(target, target.parent)
    package = load_r3d_entry_package(
        target,
        target_manifest,
        registry_path=registry,
        registry_manifest_path=registry_manifest,
        attempt_path=attempt,
        completion_path=completion,
    )
    assert package["artifact"]["sha256"] == R3D_TARGET_ARTIFACT_SHA256
    assert "path" not in package["artifact"]
    assert str(tmp_path) not in json.dumps(package, sort_keys=True)
    assert package["manifest"]["sha256"] == R3D_TARGET_MANIFEST_SHA256
    assert package["registry"]["artifact"]["sha256"] == (
        R3D_REGISTRY_ARTIFACT_SHA256
    )
    assert package["registry"]["manifest"]["sha256"] == (
        R3D_REGISTRY_MANIFEST_SHA256
    )
    assert package["entry"]["decision_path"] == "L3H1_reused_component"
    assert package["entry"]["classification"] == "SUPPORTED"
    assert package["entry"]["path_b_used_to_gate_entry"] is False
    assert package["target_gene_count"] == 3909
    assert package["target_gene_ids_sha256"].startswith("c41dbd02")
    assert package["checkpoint_and_data"]["fasta_index_sha256"].startswith(
        "3b425de2"
    )
    assert package["attempt"]["sha256"] == R3D_TARGET_ATTEMPT_SHA256
    assert package["completion"]["sha256"] == R3D_TARGET_COMPLETION_SHA256

    record = json.loads(target.read_text())
    path_b_changed = copy.deepcopy(record)
    path_b_changed["paths"]["distinct_acceptor_route_candidate"][
        "classification"
    ] = "INCONCLUSIVE"
    assert validate_r3d_target_record(path_b_changed)["classification"] == (
        "SUPPORTED"
    )
    path_a_changed = copy.deepcopy(record)
    path_a_changed["paths"]["L3H1_reused_component"][
        "classification"
    ] = "INCONCLUSIVE"
    with pytest.raises(R5UpstreamProvenanceError, match="Path A"):
        validate_r3d_target_record(path_a_changed)


def test_prior_exposure_registry_is_exact_four_way_disjoint_and_not_global_freshness():
    r3c = load_r3c_exclusion_package(
        RESULTS / "r3c_acceptor_head_preflight_v2_1.json",
        RESULTS / "r3c_acceptor_head_preflight_v2_1.manifest.json",
    )
    assert r3c["artifact"]["sha256"] == R3C_PREFLIGHT_ARTIFACT_SHA256
    assert r3c["manifest"]["sha256"] == R3C_PREFLIGHT_MANIFEST_SHA256
    r3d = load_r3d_entry_package(
        RESULTS / "r3d_acceptor_specificity_target_v1.json",
        RESULTS / "r3d_acceptor_specificity_target_v1.manifest.json",
        registry_path=RESULTS / "r3d_train_exposure_registry_v1.json",
        registry_manifest_path=(
            RESULTS / "r3d_train_exposure_registry_v1.manifest.json"
        ),
    )
    registry = derive_r5_exclusion_registry(r3c, r3d)
    assert registry["schema"] == R5_EXCLUSION_REGISTRY_SCHEMA
    validated = validate_r5_exclusion_registry(registry)
    assert validated["counts"] == {
        "historical_r3": 645,
        "r3c_fixed_cohorts": 588,
        "r3d_direct_exposure_registry": 1112,
        "r3d_target": 3909,
        "prior_exposure_union": 6254,
    }
    assert validated["prior_exposure_union_gene_ids_sha256"] == (
        "14f2cbb7d1aa4764da9825a481bdb1f2002e2d47022b0292f809494e163e9ed3"
    )
    assert validated["r3d_registry_and_target_union_gene_ids_sha256"] == (
        "2d64dab36515e9d6344f73164a91c668e15d50f15003cfdd3a5a4ecc87e9fd13"
    )
    assert validated["all_pairwise_intersections_zero"] is True
    assert registry["scope"]["global_gene_freshness_claimed"] is False
    assert len(validated["prior_exposure_gene_ids"]) == 6254

    tampered = copy.deepcopy(registry)
    tampered["gene_sets"]["r3d_target"]["gene_ids"][0] = tampered[
        "gene_sets"
    ]["historical_r3"]["gene_ids"][0]
    with pytest.raises(R5UpstreamProvenanceError):
        validate_r5_exclusion_registry(tampered)

    overclaim = copy.deepcopy(registry)
    overclaim["upstream_packages"]["r3d_target"]["entry"][
        "not_sealed_or_independent_confirmation"
    ] = False
    payload = {
        key: value for key, value in overclaim.items() if key != "record_sha256"
    }
    overclaim["record_sha256"] = canonical_sha256(payload)
    with pytest.raises(R5UpstreamProvenanceError, match="upstream package identity"):
        validate_r5_exclusion_registry(overclaim)
