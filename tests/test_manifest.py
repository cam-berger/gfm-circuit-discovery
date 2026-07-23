"""Memorial tests for the results master index (`results/MANIFEST.{csv,json}`, review fix F2).

The manifest is only worth trusting if it cannot drift from the tree it indexes, so this file
does two things:

  1. asserts the real repository is currently consistent, and
  2. proves the checker can FAIL — a deliberately broken manifest is built in a tmp tree for each
     of the five required checks, and `scripts/verify_manifest.py` must reject every one.

Without (2) the guard would be a memorial that cannot fail, which this project does not accept
(house rules R7/R21).

The `polarity` column gets the same treatment as `status`: an illegal value, a missing cell, a
CSV/JSON disagreement and a wrong `polarity_counts` summary each have a test that builds the
broken state on disk and requires a rejection. The real-repository tests additionally pin the
ledger-derived polarity of the rows the column exists for — C13's powered attention null, C14's
powered refutation of the nucleosome bridge, C15's sensitivity bound — so that silently
relabelling a null as positive evidence fails here.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VERIFIER = REPO / "scripts/verify_manifest.py"


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_manifest", VERIFIER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vm = _load_verifier()


# --------------------------------------------------------------------------------------
# tmp-tree fixtures: a minimal repository whose manifest we can break on purpose
# --------------------------------------------------------------------------------------

GOOD_ROW = {
    "claim_id": "C1",
    "status": "canonical",
    "evidence_level": "Behaviour",
    "polarity": "positive",
    "artifact": "results/a.json",
    "script": "scripts/run_a.py",
    "manifest": "results/a.manifest.json",
    "figure": "figures/f.png",
}


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "results").mkdir()
    (tmp_path / "results/a.json").write_text("{}\n")
    (tmp_path / "results/a.manifest.json").write_text("{}\n")
    (tmp_path / "results/b.json").write_text("{}\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run_a.py").write_text("")
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures/f.png").write_bytes(b"")
    (tmp_path / "figures/g.png").write_bytes(b"")
    return tmp_path


def _write_manifest(
    repo: Path,
    rows,
    *,
    json_rows=None,
    row_count=None,
    status_counts=None,
    polarity_counts=None,
    columns=None,
):
    """Write both serialisations; the keyword arguments let a test desynchronise them."""
    fieldnames = vm.COLUMNS if columns is None else columns
    with (repo / "results/MANIFEST.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    body = json_rows if json_rows is not None else rows
    counts: dict[str, int] = {}
    polarities: dict[str, int] = {}
    for row in body:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if "polarity" in row:
            polarities[row["polarity"]] = polarities.get(row["polarity"], 0) + 1
    doc = {
        "generated_by": "test fixture",
        "row_count": len(body) if row_count is None else row_count,
        "status_counts": counts if status_counts is None else status_counts,
        "polarity_counts": polarities if polarity_counts is None else polarity_counts,
        "columns": fieldnames,
        "rows": body,
    }
    (repo / "results/MANIFEST.json").write_text(json.dumps(doc, indent=2) + "\n")


def _check(repo: Path, expected: set[str]) -> list[str]:
    csv_rows = vm.read_csv_rows(repo / "results/MANIFEST.csv")
    json_doc = vm.read_json_doc(repo / "results/MANIFEST.json")
    return vm.check_manifest(repo, csv_rows, json_doc, expected)


# --------------------------------------------------------------------------------------
# the guard must PASS on a consistent manifest
# --------------------------------------------------------------------------------------


def test_consistent_manifest_reports_no_problems(fake_repo: Path):
    _write_manifest(fake_repo, [GOOD_ROW])
    assert _check(fake_repo, {"results/a.json"}) == []


# --------------------------------------------------------------------------------------
# ...and must FAIL on each of the five required checks
# --------------------------------------------------------------------------------------


def test_check_1_tracked_file_missing_from_the_manifest_fails(fake_repo: Path):
    _write_manifest(fake_repo, [GOOD_ROW])
    problems = _check(fake_repo, {"results/a.json", "results/b.json"})
    assert any("coverage" in p and "results/b.json" in p for p in problems), problems


def test_check_1_duplicate_row_fails(fake_repo: Path):
    _write_manifest(fake_repo, [GOOD_ROW, dict(GOOD_ROW)])
    problems = _check(fake_repo, {"results/a.json"})
    assert any("appears 2 times" in p for p in problems), problems


def test_check_1_row_for_a_file_that_is_not_there_fails(fake_repo: Path):
    row = dict(GOOD_ROW, artifact="results/ghost.json")
    _write_manifest(fake_repo, [GOOD_ROW, row])
    problems = _check(fake_repo, {"results/a.json"})
    assert any("untracked/absent" in p and "ghost" in p for p in problems), problems


def test_check_2_artifact_missing_from_disk_fails(fake_repo: Path):
    row = dict(GOOD_ROW, artifact="results/ghost.json")
    _write_manifest(fake_repo, [row])
    problems = _check(fake_repo, {"results/ghost.json"})
    assert any(p.startswith("artifact: does not exist on disk") for p in problems), problems


@pytest.mark.parametrize("column", ["script", "manifest", "figure"])
def test_check_3_dangling_reference_fails(fake_repo: Path, column: str):
    row = dict(GOOD_ROW)
    row[column] = "scripts/not_a_real_file.py"
    _write_manifest(fake_repo, [row])
    problems = _check(fake_repo, {"results/a.json"})
    assert any(p.startswith(f"{column}: ") and "missing path" in p for p in problems), problems


def test_check_3_dangling_entry_inside_a_list_fails(fake_repo: Path):
    # The figure column may hold a ';'-separated list; a bad entry anywhere must be caught.
    row = dict(GOOD_ROW, figure="figures/f.png;figures/nope.png")
    _write_manifest(fake_repo, [row])
    problems = _check(fake_repo, {"results/a.json"})
    assert any("figures/nope.png" in p for p in problems), problems
    assert not any("figures/f.png" in p for p in problems), problems


def test_check_4_illegal_status_fails(fake_repo: Path):
    _write_manifest(fake_repo, [dict(GOOD_ROW, status="probably-fine")])
    problems = _check(fake_repo, {"results/a.json"})
    assert any(p.startswith("status: ") for p in problems), problems


@pytest.mark.parametrize("bad", ["probably-fine", "POSITIVE", "weak-positive", "", "nul"])
def test_check_5_illegal_polarity_fails(fake_repo: Path, bad: str):
    """The load-bearing memorial: a row whose polarity is outside the vocabulary is rejected.

    `weak-positive` is the failure mode the column exists to prevent — a powered null or a
    refutation quietly re-described as a shade of supporting evidence.
    """
    _write_manifest(fake_repo, [dict(GOOD_ROW, polarity=bad)])
    problems = _check(fake_repo, {"results/a.json"})
    assert any(p.startswith("polarity: ") for p in problems), problems
    assert any(repr(bad) in p for p in problems), problems


def test_check_5_every_legal_polarity_is_accepted(fake_repo: Path):
    # The complement of the test above: the guard must not reject the real vocabulary, or
    # nulls and refutations could not be recorded at all.
    for value in vm.LEGAL_POLARITY:
        _write_manifest(fake_repo, [dict(GOOD_ROW, polarity=value)])
        assert _check(fake_repo, {"results/a.json"}) == [], value


def test_check_5_missing_polarity_cell_fails(fake_repo: Path):
    # A CSV row that is short of cells arrives from csv.DictReader with polarity=None.
    _write_manifest(fake_repo, [GOOD_ROW])
    csv_rows = vm.read_csv_rows(fake_repo / "results/MANIFEST.csv")
    csv_rows[0]["polarity"] = None
    json_doc = vm.read_json_doc(fake_repo / "results/MANIFEST.json")
    problems = vm.check_manifest(fake_repo, csv_rows, json_doc, {"results/a.json"})
    assert any(p.startswith("polarity: ") and "no 'polarity' cell" in p for p in problems), problems


def test_check_5_manifest_without_the_polarity_column_is_rejected(fake_repo: Path):
    # The whole point of the fix: an index that still has only the old seven columns is invalid.
    legacy = [c for c in vm.COLUMNS if c != "polarity"]
    row = {k: v for k, v in GOOD_ROW.items() if k != "polarity"}
    _write_manifest(fake_repo, [row], columns=legacy)
    with pytest.raises(SystemExit, match="header"):
        vm.read_csv_rows(fake_repo / "results/MANIFEST.csv")


def test_check_6_csv_and_json_drift_fails(fake_repo: Path):
    _write_manifest(
        fake_repo,
        [GOOD_ROW],
        json_rows=[dict(GOOD_ROW, status="development")],
    )
    problems = _check(fake_repo, {"results/a.json"})
    assert any(p.startswith("parity: MANIFEST.csv and MANIFEST.json disagree") for p in problems)


def test_check_6_polarity_drift_between_csv_and_json_fails(fake_repo: Path):
    _write_manifest(fake_repo, [GOOD_ROW], json_rows=[dict(GOOD_ROW, polarity="null")])
    problems = _check(fake_repo, {"results/a.json"})
    assert any(p.startswith("parity: MANIFEST.csv and MANIFEST.json disagree") for p in problems)


def test_check_6_json_row_without_polarity_fails(fake_repo: Path):
    stripped = {k: v for k, v in GOOD_ROW.items() if k != "polarity"}
    _write_manifest(fake_repo, [GOOD_ROW], json_rows=[stripped])
    problems = _check(fake_repo, {"results/a.json"})
    assert any("eight manifest columns" in p for p in problems), problems


def test_check_6_wrong_summary_counts_fail(fake_repo: Path):
    _write_manifest(fake_repo, [GOOD_ROW], row_count=99, status_counts={"canonical": 99})
    problems = _check(fake_repo, {"results/a.json"})
    assert any("row_count 99" in p for p in problems), problems
    assert any("status_counts" in p for p in problems), problems


def test_check_6_wrong_polarity_counts_fail(fake_repo: Path):
    _write_manifest(fake_repo, [GOOD_ROW], polarity_counts={"positive": 7})
    problems = _check(fake_repo, {"results/a.json"})
    assert any("polarity_counts" in p for p in problems), problems


def test_json_without_polarity_counts_is_rejected(fake_repo: Path):
    _write_manifest(fake_repo, [GOOD_ROW])
    doc = json.loads((fake_repo / "results/MANIFEST.json").read_text())
    doc.pop("polarity_counts")
    (fake_repo / "results/MANIFEST.json").write_text(json.dumps(doc, indent=2) + "\n")
    with pytest.raises(SystemExit, match="polarity_counts"):
        vm.read_json_doc(fake_repo / "results/MANIFEST.json")


def test_wrong_header_is_rejected(fake_repo: Path):
    (fake_repo / "results/MANIFEST.csv").write_text("artifact,status\nresults/a.json,canonical\n")
    with pytest.raises(SystemExit, match="header"):
        vm.read_csv_rows(fake_repo / "results/MANIFEST.csv")


def test_cli_exits_non_zero_on_a_broken_manifest(fake_repo: Path):
    _write_manifest(fake_repo, [dict(GOOD_ROW, status="probably-fine")])
    proc = subprocess.run(
        [sys.executable, str(VERIFIER), "--repo", str(fake_repo)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "FAIL" in (proc.stdout + proc.stderr)


def test_cli_exits_non_zero_on_an_illegal_polarity(fake_repo: Path):
    _write_manifest(fake_repo, [dict(GOOD_ROW, polarity="weak-positive")])
    proc = subprocess.run(
        [sys.executable, str(VERIFIER), "--repo", str(fake_repo)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "polarity" in (proc.stdout + proc.stderr)


# --------------------------------------------------------------------------------------
# the real repository
# --------------------------------------------------------------------------------------


def test_repository_manifest_is_consistent():
    csv_rows = vm.read_csv_rows(REPO / "results/MANIFEST.csv")
    json_doc = vm.read_json_doc(REPO / "results/MANIFEST.json")
    expected = vm.tracked_result_files(REPO)
    problems = vm.check_manifest(REPO, csv_rows, json_doc, expected)
    assert problems == [], "\n".join(problems[:20])


def test_repository_manifest_columns_are_the_review_specified_seven_plus_polarity():
    # The column set is fixed by docs/REVIEW_2026-07-22.md F2 and reused verbatim downstream;
    # `polarity` was added immediately after `evidence_level` by the audit fix.
    with (REPO / "results/MANIFEST.csv").open(newline="") as fh:
        header = next(csv.reader(fh))
    assert header == [
        "claim_id",
        "status",
        "evidence_level",
        "polarity",
        "artifact",
        "script",
        "manifest",
        "figure",
    ]


def test_repository_every_row_declares_a_legal_polarity():
    rows = vm.read_csv_rows(REPO / "results/MANIFEST.csv")
    illegal = sorted({r["polarity"] for r in rows} - set(vm.LEGAL_POLARITY))
    assert illegal == [], illegal
    assert all(r["polarity"] for r in rows), "every row must carry a polarity"


def test_repository_polarity_matches_the_claim_ledger_on_the_non_positive_rows():
    """Pin the rows the column exists for, straight from `studies/*/CLAIMS.md` wording.

    C13 is 'established: powered attention null', C14 is the powered refutation of the
    nucleosome bridge, C15 is 'bounded ... do not establish absence', C7/C10/C30/R22/HCA10 are
    ledger-'bounded', and IC1/HCA4 are nulls. If any of them is ever relabelled as supporting
    evidence at its rung, this fails.
    """
    expected = {
        "results/phase-lattice/human_carrier.json": "null",  # C13
        "results/phase-lattice/postsoftmax.json": "null",  # C11,C13
        "results/phase-lattice/band_mass.json": "null",  # C11,C13
        "results/phase-lattice/dyad_real.json": "refutation",  # C14
        "results/phase-lattice/dyad_wavelength.json": "refutation",  # C14
        "results/phase-lattice/dyad_fpr.json": "refutation",  # C14
        "results/phase-lattice/fig_report_data.json": "refutation",  # C8,C11,C13,C14
        "results/phase-lattice/corpus_spectrum.json": "bounded",  # C15
        "results/phase-lattice/splice_profile.json": "bounded",  # C15
        "results/phase-lattice/splice_phasing.json": "bounded",  # C15
        "results/phase-lattice/splice_long.json": "bounded",  # C15
        "results/phase-lattice/carrier_mediation.json": "bounded",  # C9,C10
        "results/splicebert_coupling/completeness_null.json": "bounded",  # C7
        "results/splicebert_coupling/faithfulness.json": "bounded",  # C7
        "results/splicebert_coupling/a9b_payload_semantics_composition.json": "bounded",  # C30
        "results/splicebert_coupling/a9c_matched_composition_reconcile.json": "bounded",  # C30;R22
        "results/splicebert_coupling/a9c_sweep_seeded_draws.json": "bounded",  # R22
        "results/splicebert_coupling/ic1_distal_intron_necessity_v8.json": "null",  # IC1
        "results/splicebert_head_context_atlas/specificity_pilot_v1.json": "null",  # HCA4
        "results/splicebert_head_context_atlas/group_head_rerank_valid_donor_v2.json": "bounded",
        "results/splicebert_coupling/sealed_test.json": "positive",  # C1–C4, the contrast
    }
    by_artifact = {r["artifact"]: r["polarity"] for r in vm.read_csv_rows(REPO / "results/MANIFEST.csv")}
    wrong = {
        path: (by_artifact.get(path), want)
        for path, want in expected.items()
        if by_artifact.get(path) != want
    }
    assert wrong == {}, wrong


def test_repository_no_null_or_refutation_row_is_recorded_as_positive():
    """The audit finding itself: no artifact whose only ledger claim is a null/refutation/bound
    may sit in the index labelled `positive`."""
    non_positive_claims = {"C7", "C10", "C13", "C14", "C15", "C30", "R22", "IC1", "HCA4", "HCA10"}
    offenders = []
    for row in vm.read_csv_rows(REPO / "results/MANIFEST.csv"):
        if row["status"] != "canonical":
            continue
        claims = {c.strip() for c in re.split(r"[;,/]", row["claim_id"]) if c.strip()}
        if claims and claims <= non_positive_claims and row["polarity"] == "positive":
            offenders.append((row["artifact"], row["claim_id"]))
    assert offenders == [], offenders


def test_repository_manifest_indexes_every_tracked_result_file():
    tracked = vm.tracked_result_files(REPO)
    listed = {r["artifact"] for r in vm.read_csv_rows(REPO / "results/MANIFEST.csv")}
    assert tracked - listed == set(), sorted(tracked - listed)[:20]
    assert listed - tracked == set(), sorted(listed - tracked)[:20]
