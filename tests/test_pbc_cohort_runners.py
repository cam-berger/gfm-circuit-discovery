from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from gfmcd.artifact_guard import ArtifactExistsError


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
