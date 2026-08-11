"""The sibling-manifest contract for the DNABERT-2 promoter runners.

The study shipped ~104 artifacts with no provenance record, so these tests pin the properties that
gap is fixed by: the manifest lands on the repository-standard sibling path, it hashes what it
claims, it records the probe that defines the readout, and it cannot be defeated by a `Path` in the
config dict — which is how a manifest write would otherwise fail at the end of a long run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gfmcd.dnabert.manifest import sibling_manifest_path, write_run_manifest


def test_sibling_path_follows_the_repository_convention():
    # 247 artifacts under results/ use <stem>.manifest.json; match that, not the older
    # <stem-minus-suffix>_manifest.json spelling this study shipped by hand.
    assert sibling_manifest_path("a/b_summary.json") == Path("a/b_summary.manifest.json")
    assert sibling_manifest_path("a/b.csv") == Path("a/b.manifest.json")
    assert sibling_manifest_path(Path("a/b_per_example.csv")) == Path(
        "a/b_per_example.manifest.json"
    )


def _write(tmp_path: Path, name: str, text: str = "{}") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_manifest_records_outputs_with_real_hashes(tmp_path):
    summary = _write(tmp_path, "run_summary.json", '{"mean": 1}')
    detail = _write(tmp_path, "run_per_example.csv", "a,b\n1,2\n")

    written = write_run_manifest(
        summary,
        experiment_id="unit_test",
        config={"limit": 8},
        outputs=[detail],
    )

    assert written == sibling_manifest_path(summary)
    doc = json.loads(written.read_text())

    recorded = {Path(entry["path"]).name: entry for entry in doc["outputs"]}
    assert set(recorded) == {"run_summary.json", "run_per_example.csv"}
    for entry in recorded.values():
        assert len(entry["sha256"]) == 64
        assert entry["size_bytes"] > 0

    assert doc["experiment_id"] == "unit_test"
    assert doc["config"]["limit"] == 8
    assert isinstance(doc["command"], list) and doc["command"]
    assert doc["created_at_utc"]


def test_primary_output_is_never_recorded_twice(tmp_path):
    summary = _write(tmp_path, "run_summary.json")

    written = write_run_manifest(
        summary,
        experiment_id="unit_test",
        config={},
        # a runner that passes its primary path in both places must not double-count it
        outputs=[summary],
    )

    doc = json.loads(written.read_text())
    assert len(doc["outputs"]) == 1


def test_path_valued_config_is_serialisable(tmp_path):
    # 23 runners declare argparse options with type=Path and pass vars(args) straight through.
    summary = _write(tmp_path, "run_summary.json")

    written = write_run_manifest(
        summary,
        experiment_id="unit_test",
        config={
            "cohort": Path("results/x/cohort.csv"),
            "sizes": (5, 15),
            "nested": {"probe": Path("p.joblib")},
        },
    )

    doc = json.loads(written.read_text())
    assert doc["config"]["cohort"] == "results/x/cohort.csv"
    assert doc["config"]["sizes"] == [5, 15]
    assert doc["config"]["nested"]["probe"] == "p.joblib"


def test_helper_fields_never_clobber_a_runner_config_key(tmp_path):
    # Runners pass vars(args) straight through and many declare --cohort, so the split-provenance
    # marker must not land on that key. This is the bug this test was written for.
    summary = _write(tmp_path, "run_summary.json")

    written = write_run_manifest(
        summary,
        experiment_id="unit_test",
        config={"cohort": Path("results/x/cohort.csv")},
    )

    config = json.loads(written.read_text())["config"]
    assert config["cohort"] == "results/x/cohort.csv"
    assert "split_leakage_control" in config


def test_probe_is_hashed_because_it_defines_the_readout(tmp_path):
    # The probe hash is what distinguishes the canonical Layer-2 assay from the superseded Layer-3
    # replicate; recording the path alone would not.
    summary = _write(tmp_path, "run_summary.json")
    probe = _write(tmp_path, "selected_layer_2_probe.joblib", "not-really-a-probe")

    written = write_run_manifest(
        summary,
        experiment_id="unit_test",
        config={},
        probe_path=probe,
        selected_layer=2,
        model_name="quietflamingo/dnabert2-no-flashattention",
        model_revision="813031b2bf86d9e960a027e2734c908009f31601",
    )

    doc = json.loads(written.read_text())
    assert doc["config"]["selected_layer"] == 2
    assert len(doc["config"]["probe_sha256"]) == 64
    assert doc["config"]["model_revision"].startswith("813031b2")


def test_seed_is_recorded_where_the_repository_looks_for_it(tmp_path):
    summary = _write(tmp_path, "run_summary.json")

    written = write_run_manifest(
        summary, experiment_id="unit_test", config={}, seed=42
    )
    assert json.loads(written.read_text())["seeds"] == {"global": 42}

    written = write_run_manifest(
        summary, experiment_id="unit_test", config={}
    )
    assert json.loads(written.read_text())["seeds"] == {}


def test_status_defaults_to_exploratory_so_a_run_cannot_promote_itself(tmp_path):
    summary = _write(tmp_path, "run_summary.json")
    written = write_run_manifest(summary, experiment_id="unit_test", config={})
    assert json.loads(written.read_text())["status"] == "EXPLORATORY"


def test_missing_input_is_skipped_rather_than_faked(tmp_path):
    summary = _write(tmp_path, "run_summary.json")
    present = _write(tmp_path, "cohort.csv", "a\n1\n")

    written = write_run_manifest(
        summary,
        experiment_id="unit_test",
        config={},
        inputs=[present, tmp_path / "absent.csv"],
    )

    doc = json.loads(written.read_text())
    names = {Path(entry["path"]).name for entry in doc["inputs"]}
    assert "cohort.csv" in names
    assert "absent.csv" not in names


@pytest.mark.parametrize("stem", ["a_summary.json", "a_curve.csv", "a.json"])
def test_every_primary_shape_gets_exactly_one_manifest(tmp_path, stem):
    primary = _write(tmp_path, stem)
    written = write_run_manifest(primary, experiment_id="unit_test", config={})
    assert written.is_file()
    assert written.name.endswith(".manifest.json")
