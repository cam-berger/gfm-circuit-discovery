from __future__ import annotations

import json

import pytest

from gfmcd.artifact_guard import ArtifactExistsError
from gfmcd.reconstruction_atlas.artifacts import write_result_bundle
from gfmcd.reconstruction_atlas.statistics import (
    add_one_permutation_pvalue,
    holm_adjust,
    max_t_adjusted_pvalues,
)


def test_add_one_and_full_family_corrections() -> None:
    assert add_one_permutation_pvalue(3.0, [0.0, 1.0, 2.0]) == 0.25
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adjusted == {"a": 0.03, "b": 0.08, "c": 0.5}
    max_t = max_t_adjusted_pvalues(
        {"a": 3.0, "b": 1.0},
        [{"a": 0.0, "b": 0.0}, {"a": 2.0, "b": 4.0}],
    )
    assert max_t["a"] == pytest.approx(2 / 3)
    assert max_t["b"] == pytest.approx(2 / 3)


def test_result_and_manifest_are_immutable_and_hashed(tmp_path) -> None:
    source = tmp_path / "input.txt"
    source.write_text("data")
    out = tmp_path / "result.json"
    result, manifest = write_result_bundle(
        out,
        {"value": 1},
        experiment_id="test",
        status="EXPLORATORY",
        model_identity={"revision": "rev"},
        input_paths=(source,),
        parameters={"alpha": 1},
        seeds={"main": 7},
        command=("run",),
        repo=tmp_path,
    )
    record = json.loads(manifest.read_text())
    assert result.exists()
    assert record["checkpoint"]["revision"] == "rev"
    assert record["data"][0]["sha256"]
    assert record["outputs"][0]["sha256"]
    with pytest.raises(ArtifactExistsError):
        write_result_bundle(
            out,
            {"value": 2},
            experiment_id="test",
            status="EXPLORATORY",
            model_identity={"revision": "rev"},
            input_paths=(source,),
            parameters={},
            seeds={},
        )
