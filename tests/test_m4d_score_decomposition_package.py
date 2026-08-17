"""Model-free audit memorial for the M4d development package.

The package refines adaptive candidates; it does not create a claim.  These tests authenticate
the exact result package and the recovered post-run source snapshot, and make the reasons for
that evidence boundary executable without loading the model, genome, cohort, or bootstrap rows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
RESULT = (
    REPO
    / "results/splicebert_acceptor_coupling/development/m4d_score_decomposition.json"
)
MANIFEST = RESULT.with_name("m4d_score_decomposition.manifest.json")
SNAPSHOT = "773c9e8987e05e2605974004795f92f67425cd07"
EXECUTION_HEAD = "b79570d68a5cde9a03f33af516fb369a26267892"

RESULT_SHA = "d1d960296e53be684909e129b6de3ae4ddc9b15e1a688c0813109d7db337f763"
MANIFEST_SHA = "879b262b93ab108e545f350b071a62cd715bb7012872455f31ce76a1562af165"
RECOVERED_SNAPSHOT_SOURCE_SHA = {
    "src/gfmcd/circuits/hf_bert.py": (
        "62356945f6e29f66fb971301bed9261ce5189af4657137cdd143ae169148f41d"
    ),
    "src/gfmcd/splicing/score_margin.py": (
        "b30514d231a2ea7923afd2df1e854a1f6fda190ac5ae4d7cf2d74e3c0a94273f"
    ),
    "scripts/splicebert_acceptor_coupling/run_m4d_score_decomposition.py": (
        "b3570976fb8f28d48d30b06d2e4e1dc918a9b986d03764edeb2bb80d268dfb66"
    ),
    "tests/test_capture_attention_scores.py": (
        "c626dfce1835420030d308aa49cf405fb73984fa16ad4b68d8284053acd719e4"
    ),
    "tests/test_score_margin.py": (
        "ad4f03b68cfbddac25ad0279ee45769c1f1adb4f0dcd12996023f1474242256a"
    ),
}
FROZEN_HF_BERT_SHA = (
    "8c33b5a6226a75ae4480c9a3b914d1649351da3797039ced20efa2e7fbcdc120"
)


def _load_reviewed_runner():
    path = REPO / "scripts/splicebert_acceptor_coupling/run_m4d_score_decomposition.py"
    spec = importlib.util.spec_from_file_location("reviewed_m4d_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_show(path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{SNAPSHOT}:{path}"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout


def _reject_nonstandard_constant(token: str):
    raise ValueError(f"non-standard JSON constant: {token}")


def test_m4d_package_has_exact_historical_bytes_and_crosslinks() -> None:
    assert RESULT.stat().st_size == 15_313
    assert MANIFEST.stat().st_size == 2_189
    assert _sha256(RESULT.read_bytes()) == RESULT_SHA
    assert _sha256(MANIFEST.read_bytes()) == MANIFEST_SHA

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["outputs"] == [
        {
            "path": manifest["outputs"][0]["path"],
            "sha256": RESULT_SHA,
            "size_bytes": 15_313,
        }
    ]
    assert manifest["git_commit"] == EXECUTION_HEAD
    assert manifest["git_dirty"] is True
    assert manifest["status"] == "DEVELOPMENT"
    assert manifest["config"] == {"n_boot": 2000, "n_pairs": 181, "reader": "L3H1"}

    checkpoint = manifest["inputs"][0]
    assert checkpoint["missing"] is True
    assert "sha256" not in checkpoint
    assert manifest["inputs"][1]["sha256"] == (
        "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453"
    )
    assert manifest["inputs"][2]["sha256"] == (
        "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51"
    )


def test_post_run_snapshot_is_reachable_and_preserves_recovered_source_bytes() -> None:
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", SNAPSHOT, "HEAD"],
        cwd=REPO,
        check=False,
    ).returncode == 0
    assert subprocess.run(
        ["git", "rev-parse", f"{SNAPSHOT}^"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == EXECUTION_HEAD

    for path, expected in RECOVERED_SNAPSHOT_SOURCE_SHA.items():
        assert _sha256(_git_show(path)) == expected

    # Snapshotting after the run repairs retrievability, not execution-byte authentication: the
    # manifest names only the dirty parent and does not hash source files.  The reviewed tree keeps
    # the frozen R1c/R5 core byte-identical and relocates score capture to a new module.
    assert SNAPSHOT != EXECUTION_HEAD
    assert _sha256((REPO / "src/gfmcd/circuits/hf_bert.py").read_bytes()) == (
        FROZEN_HF_BERT_SHA
    )
    runner = (
        REPO
        / "scripts/splicebert_acceptor_coupling/run_m4d_score_decomposition.py"
    ).read_text()
    assert "gfmcd.circuits.attention_scores" in runner
    assert "eligible_donor_motifs" in runner
    assert "allow_nan=False" in runner
    assert "collapsed_baseline_shift_benchmark" in runner
    assert "curvature_ceiling" not in runner


def test_exact_development_result_is_numerically_scoped() -> None:
    record = json.loads(RESULT.read_text())
    assert record["schema"] == "gfmcd.m4d.score_decomposition/1"
    assert record["tier"] == "development"
    assert record["split"] == "train"
    assert record["sealed_chr8_chr9_opened"] is False
    assert record["validation_reserve_spent"] is False
    assert record["n_pairs"] == 181
    assert record["n_genes"] == 152
    assert record["reader"] == "L3H1"
    assert record["baseline"] == {
        "mass": {"GC": 0.17015116624461255, "GT": 0.4669067566393708},
        "margin": {"GC": -3.2416604159630436, "GT": -0.4279720262983662},
    }
    assert all(
        item.get("pass", True)
        for item in record["memorials"].values()
    )

    expected = {
        "L0H14": (0.4290274113420788, [0.26835828929497346, 0.5881876510020364]),
        "L0H5": (0.25021834648738345, [0.09456241375593062, 0.39673034174685123]),
        "L0H3": (0.21133896622063805, [0.06436592265566, 0.35371369251122026]),
        "L0H4": (0.10916772074013213, [-0.09700238710530677, 0.310173669613979]),
        "L1H6": (0.07126043385026667, [-0.02362878693998345, 0.16951856180460226]),
        "L1H13": (-0.008756685080273224, [-0.11815828158611592, 0.09723623288392326]),
        "L2H15": (-0.45538115572484833, [-0.6976525417572375, -0.22232480222256784]),
    }
    for head, (difference, interval) in expected.items():
        observed = record["probes"][head]["margin_shift"]
        assert observed["difference"] == pytest.approx(difference)
        assert observed["ci"] == pytest.approx(interval)


def test_artifact_itself_exposes_nonreplayable_and_nonstandard_boundaries() -> None:
    raw = RESULT.read_text()
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        json.loads(raw, parse_constant=_reject_nonstandard_constant)
    assert "NaN" in raw

    record = json.loads(raw)
    assert "cohort" not in record
    assert "rows" not in record
    assert "draws" not in record
    assert record["probes"]["L0H3"]["decomposition"]["verdict"] == "UNDEFINED"
    assert record["probes"]["L0H14"]["decomposition_ci"]["n_usable_draws"] == 1736
    assert record["probes"]["L0H5"]["decomposition_ci"]["n_usable_draws"] == 1811

    historical_runner = _git_show(
        "scripts/splicebert_acceptor_coupling/run_m4d_score_decomposition.py"
    ).decode()
    assert 'ap.add_argument("--batch", type=int, default=4)' in historical_runner
    assert "genes = [x.gene_id for x in gc]" in historical_runner
    assert 'edit_heads(ad, [(l, h)], mode="mean")' in historical_runner
    assert "multiplicity" not in historical_runner.lower()
    assert 'memorials["curvature_ceiling"]' in historical_runner


def test_reviewed_runner_clusters_both_arms_and_intervention_batches() -> None:
    paired_dependency_groups = _load_reviewed_runner().paired_dependency_groups
    treatment = [SimpleNamespace(gene_id=x) for x in ("A", "B", "C", "D")]
    comparator = [SimpleNamespace(gene_id=x) for x in ("X", "X", "Y", "C")]
    groups = paired_dependency_groups(treatment, comparator, batch_size=1)
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[2]
    assert len(set(groups)) == 2

    independent_treatment = [SimpleNamespace(gene_id=x) for x in ("A", "B", "C", "D")]
    independent_comparator = [SimpleNamespace(gene_id=x) for x in ("W", "X", "Y", "Z")]
    batch_groups = paired_dependency_groups(
        independent_treatment,
        independent_comparator,
        batch_size=2,
    )
    assert batch_groups[0] == batch_groups[1]
    assert batch_groups[2] == batch_groups[3]
    assert batch_groups[0] != batch_groups[2]

    with pytest.raises(ValueError, match="paired one-to-one"):
        paired_dependency_groups(treatment, comparator[:-1], batch_size=1)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        paired_dependency_groups(treatment, comparator, batch_size=0)


def test_m4d_has_no_live_claim_or_evidence_ladder_promotion() -> None:
    index = json.loads((REPO / "results/MANIFEST.json").read_text())
    rows = [
        row
        for row in index["rows"]
        if row["artifact"].endswith("m4d_score_decomposition.json")
    ]
    assert len(rows) == 1
    assert rows[0]["claim_id"] == ""
    assert rows[0]["status"] == "development"
    assert rows[0]["evidence_level"] == "QK"
    assert rows[0]["polarity"] == "na"

    claims = (REPO / "studies/splicebert_acceptor_coupling/CLAIMS.md").read_text()
    assert "| AC12 |" not in claims
    assert claims.count("| AC10 |") == 1
    assert claims.count("| AC11 |") == 1
    assert "ACR23" in claims
