from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from gfmcd.artifact_guard import ArtifactExistsError
from gfmcd.provenance import file_record
from gfmcd.splicing import positional_boundary_coherence as pbc


REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _load(
    "build_pbc_composition_balanced_cohort",
    REPO
    / "scripts/splicebert_positional_boundary_coherence/build_composition_balanced_cohort.py",
)
AUDIT = _load(
    "audit_pbc_composition_balanced_cohort",
    REPO
    / "scripts/splicebert_positional_boundary_coherence/audit_composition_balanced_cohort.py",
)
CARRIER_PREFLIGHT = _load(
    "run_pbc_carrier_factorial_preflight",
    REPO
    / "scripts/splicebert_positional_boundary_coherence/run_carrier_factorial_preflight.py",
)
REGISTER_PREFLIGHT = _load(
    "run_pbc_register_preflight",
    REPO / "scripts/splicebert_positional_boundary_coherence/run_p3_register_preflight.py",
)
HELPERS = _load(
    "pbc_test_helpers",
    REPO / "tests/test_positional_boundary_coherence.py",
)


@pytest.fixture(scope="module")
def runner_artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp("pbc-runner")
    candidates = directory / "candidates.json"
    blacklist = directory / "blacklist.json"
    variance = directory / "variance.json"
    cohort = directory / "cohort.json"
    candidates.write_text(
        json.dumps(HELPERS.synthetic_candidate_package(n_per_pool=40), sort_keys=True) + "\n"
    )
    blacklist.write_text('{"gene_ids": []}\n')
    variance.write_text(
        '{"alpha": 0.05, "power": 0.8, "sesoi": 0.05, "variance": 0.0}\n'
    )
    output, manifest, package = BUILD.build_artifact(
        candidates_path=candidates,
        blacklist_path=blacklist,
        frozen_variance_path=variance,
        output_path=cohort,
        block_counts={"development": 5, "confirmation": 5, "phase3_reserve": 5},
        minimum_confirmation_blocks=5,
        auc_n_boot=8,
        inner_folds=2,
        pseudo_family_size=1,
        minimum_stratum_blocks=0,
    )
    return {
        "directory": directory,
        "candidates": candidates,
        "blacklist": blacklist,
        "variance": variance,
        "cohort": output,
        "manifest": manifest,
        "package": package,
    }


def test_builder_writes_immutable_json_and_hashed_manifest(runner_artifacts):
    artifact = runner_artifacts
    assert artifact["cohort"].exists()
    manifest = json.loads(artifact["manifest"].read_text())
    assert manifest["status"] == artifact["package"]["status"]
    assert manifest["outputs"][0]["sha256"]
    assert artifact["package"]["model_outcomes_read"] is False
    assert artifact["package"]["stage"] == "construction"
    assert manifest["config"]["stage"] == "construction"
    assert artifact["package"]["target_execution_authorized"] is False
    with pytest.raises(ArtifactExistsError):
        BUILD.output_paths(artifact["cohort"])


def test_auditor_writes_separate_authorization_record(runner_artifacts):
    artifact = runner_artifacts
    audit_path = artifact["directory"] / "cohort.audit.json"
    output, manifest, audit = AUDIT.audit_artifact(
        cohort_path=artifact["cohort"],
        candidates_path=artifact["candidates"],
        blacklist_path=artifact["blacklist"],
        frozen_variance_path=artifact["variance"],
        output_path=audit_path,
    )
    assert output.exists() and manifest.exists()
    assert audit["passed"] is True
    assert audit["target_execution_authorized"] is False
    assert audit["status"] == "AUDIT_PASS_INSTRUMENT_UNRESOLVED"
    assert audit["stage"] == "construction"
    assert json.loads(manifest.read_text())["config"]["stage"] == "construction"
    with pytest.raises(ArtifactExistsError):
        AUDIT.output_paths(audit_path)


def test_runners_write_and_audit_compact_capacity_stop(tmp_path):
    candidates = tmp_path / "capacity_candidates.json"
    blacklist = tmp_path / "capacity_blacklist.json"
    variance = tmp_path / "capacity_variance.json"
    cohort = tmp_path / "capacity_stop.json"
    candidates.write_text(
        json.dumps(HELPERS.synthetic_candidate_package(n_per_pool=1), sort_keys=True) + "\n"
    )
    blacklist.write_text('{"gene_ids": []}\n')
    variance.write_text(
        '{"alpha": 0.05, "power": 0.8, "sesoi": 0.05, "variance": 0.0}\n'
    )
    output, manifest, package = BUILD.build_artifact(
        candidates_path=candidates,
        blacklist_path=blacklist,
        frozen_variance_path=variance,
        output_path=cohort,
        block_counts={"development": 1, "confirmation": 1, "phase3_reserve": 0},
        minimum_confirmation_blocks=1,
        auc_n_boot=2,
        inner_folds=2,
        pseudo_family_size=1,
        minimum_stratum_blocks=0,
    )
    assert package["schema"] == "gfmcd.positional_boundary_coherence.phase1_capacity_stop.v1"
    assert package["status"] == "PHASE1_INSTRUMENT_UNRESOLVED"
    assert package["blocks"] == [] and package["cells"] == []
    assert json.loads(manifest.read_text())["outputs"][0]["sha256"]

    audit_output, audit_manifest, audit = AUDIT.audit_artifact(
        cohort_path=output,
        candidates_path=candidates,
        blacklist_path=blacklist,
        frozen_variance_path=variance,
        output_path=tmp_path / "capacity_stop.audit.json",
    )
    assert audit_output.exists() and audit_manifest.exists()
    assert audit["status"] == "AUDIT_PASS_INSTRUMENT_UNRESOLVED"
    assert audit["passed"] is True
    assert audit["target_execution_authorized"] is False


def test_production_cli_writes_stop_then_exits_two_and_auditor_exits_zero(tmp_path):
    candidates = tmp_path / "candidate_census_v1.json"
    candidate_manifest = tmp_path / "candidate_census_v1.manifest.json"
    candidate_audit = tmp_path / "candidate_census_v1.audit.json"
    blacklist = tmp_path / "blacklist.json"
    variance = tmp_path / "variance.json"
    cohort = tmp_path / "capacity_stop.json"
    audit_path = tmp_path / "capacity_stop.audit.json"

    payload = HELPERS.synthetic_candidate_package(n_per_pool=1)
    payload["candidate_mode"] = "production"
    payload["stage"] = "construction"
    candidates.write_text(json.dumps(payload, sort_keys=True) + "\n")
    candidate_record = file_record(candidates)
    candidate_manifest.write_text(
        json.dumps(
            {
                "experiment_id": (
                    "splicebert-positional-boundary-coherence:candidate-census-v1"
                ),
                "status": "CENSUS_BUILT_AUDIT_PENDING",
                "config": {
                    "acceptor_pos": 512,
                    "algorithm_version": "pbc-real-census-v1",
                    "allowed_chromosomes": sorted(
                        [
                            *(f"chr{i}" for i in range(1, 8)),
                            *(f"chr{i}" for i in range(10, 23)),
                            "chrX",
                        ]
                    ),
                    "construct_length": 1022,
                    "donor_pos": 392,
                    "stage": "construction",
                },
                "git_commit": "addd7be93c321373118b02b82bc3d95eeb0bd20a",
                "git_dirty": True,
                "outputs": [candidate_record],
            },
            sort_keys=True,
        )
        + "\n"
    )
    candidate_audit.write_text(
        json.dumps(
            {
                "schema": pbc.CANDIDATE_AUDIT_SCHEMA,
                "stage": "construction",
                "status": "AUDIT_PASS",
                "passed": True,
                "candidate_census_sha256": pbc.sha256_payload(payload),
                "candidate_content_sha256": "b" * 64,
                "reconstructed_candidates_sha256": "c" * 64,
            },
            sort_keys=True,
        )
        + "\n"
    )
    blacklist.write_text('{"gene_ids": []}\n')
    variance.write_text(
        '{"alpha": 0.05, "power": 0.8, "sesoi": 0.05, "variance": 0.0}\n'
    )
    build_args = [
        "--candidates",
        str(candidates),
        "--candidate-audit",
        str(candidate_audit),
        "--candidate-manifest",
        str(candidate_manifest),
        "--blacklist",
        str(blacklist),
        "--frozen-variance",
        str(variance),
        "--out",
        str(cohort),
    ]
    with pytest.raises(SystemExit) as stopped:
        BUILD.main(build_args)
    assert stopped.value.code == 2
    package = json.loads(cohort.read_text())
    receipt = package["candidate_census_receipt"]
    assert receipt["candidate_file"] == {
        "name": "candidate_census_v1.json",
        "size_bytes": candidate_record["size_bytes"],
        "sha256": candidate_record["sha256"],
    }
    assert receipt["producer_manifest"]["git_commit"].startswith("addd7be9")
    assert package["clean_checkout_reexecutable"] is False
    assert package["target_execution_authorized"] is False

    AUDIT.main(
        [
            "--cohort",
            str(cohort),
            "--candidates",
            str(candidates),
            "--candidate-audit",
            str(candidate_audit),
            "--candidate-manifest",
            str(candidate_manifest),
            "--blacklist",
            str(blacklist),
            "--frozen-variance",
            str(variance),
            "--out",
            str(audit_path),
        ]
    )
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "AUDIT_PASS_INSTRUMENT_UNRESOLVED"
    assert audit["passed"] is True
    assert audit["target_execution_authorized"] is False
    assert audit["clean_checkout_reexecutable"] is False
    assert audit["rederived"]["candidate_census_receipt"] == receipt
    assert CARRIER_PREFLIGHT.construction_audit_authorizes_target(audit) is False
    with pytest.raises(RuntimeError, match="wrong PBC cohort schema"):
        CARRIER_PREFLIGHT.execute_preflight(
            cohort=package,
            candidate_package=payload,
            provider=None,
            split_role="development",
            shared_gate_source=None,
            construction_audit_pass=False,
            n_controls=CARRIER_PREFLIGHT.MIN_STRUCTURED_CONTROLS,
            control_seed=CARRIER_PREFLIGHT.STRUCTURED_CONTROL_BASE_SEED,
        )
    with pytest.raises(
        RuntimeError, match="wrong positional-boundary-coherence cohort schema"
    ):
        REGISTER_PREFLIGHT.development_cells(package)


@pytest.mark.parametrize("module", [BUILD, AUDIT])
def test_candidate_receipt_rejects_missing_semantics_and_forged_manifest(
    tmp_path, module
):
    candidates = tmp_path / "candidate_census_v1.json"
    producer_manifest = tmp_path / "candidate_census_v1.manifest.json"
    candidates.write_text('{"fixture": true}\n')
    candidate_record = file_record(candidates)
    valid_manifest = {
        "experiment_id": "splicebert-positional-boundary-coherence:candidate-census-v1",
        "status": "CENSUS_BUILT_AUDIT_PENDING",
        "config": {
            "acceptor_pos": 512,
            "algorithm_version": "pbc-real-census-v1",
            "allowed_chromosomes": sorted(
                [
                    *(f"chr{i}" for i in range(1, 8)),
                    *(f"chr{i}" for i in range(10, 23)),
                    "chrX",
                ]
            ),
            "construct_length": 1022,
            "donor_pos": 392,
            "stage": "construction",
        },
        "git_commit": pbc.CANONICAL_CANDIDATE_PRODUCER_REVISION,
        "git_dirty": True,
        "outputs": [candidate_record],
    }
    producer_manifest.write_text(json.dumps(valid_manifest) + "\n")
    missing_semantics = {"candidate_census_sha256": "a" * 64}
    with pytest.raises(ValueError, match="semantic census hashes"):
        module._candidate_census_receipt(
            candidates, producer_manifest, missing_semantics
        )

    complete_audit = {
        "candidate_census_sha256": "a" * 64,
        "candidate_content_sha256": "b" * 64,
        "reconstructed_candidates_sha256": "c" * 64,
    }
    forged = dict(valid_manifest)
    forged["experiment_id"] = "evil:forged"
    forged["status"] = "TARGET_POSITIVE"
    forged["git_commit"] = "z" * 40
    producer_manifest.write_text(json.dumps(forged) + "\n")
    with pytest.raises(ValueError, match="manifest identity"):
        module._candidate_census_receipt(candidates, producer_manifest, complete_audit)


def test_cli_defaults_pin_frozen_role_counts_and_calibration_family():
    args = BUILD.parse_args(
        [
            "--candidates",
            "candidates.json",
            "--candidate-audit",
            "candidates.audit.json",
            "--blacklist",
            "blacklist.json",
            "--frozen-variance",
            "variance.json",
            "--out",
            "cohort.json",
        ]
    )
    assert (args.development_blocks, args.confirmation_blocks, args.reserve_blocks) == (
        120,
        600,
        384,
    )
    assert args.pseudo_family_size == 99
    assert args.auc_bootstrap == 2000
    BUILD.validate_production_args(args)


def test_cli_rejects_smaller_confirmation_or_reserve_roles():
    args = BUILD.parse_args(
        [
            "--candidates",
            "candidates.json",
            "--candidate-audit",
            "candidates.audit.json",
            "--blacklist",
            "blacklist.json",
            "--frozen-variance",
            "variance.json",
            "--confirmation-blocks",
            "599",
            "--out",
            "cohort.json",
        ]
    )
    with pytest.raises(ValueError, match="exactly 600"):
        BUILD.validate_production_args(args)


def test_runners_are_model_free_by_construction():
    for path in (
        REPO
        / "scripts/splicebert_positional_boundary_coherence/build_composition_balanced_cohort.py",
        REPO
        / "scripts/splicebert_positional_boundary_coherence/audit_composition_balanced_cohort.py",
    ):
        source = path.read_text()
        assert "transformers" not in source
        assert "torch" not in source
        assert "_dataset" not in source
        assert "model_outcomes_read" in source or "No SpliceBERT checkpoint" in source
