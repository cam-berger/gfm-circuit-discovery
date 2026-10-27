"""gfmcd.record: the census guard over legacy manifests, and write/bind/verify/resolve end to end.

The end-to-end tests build a throwaway git repository, so tracked-only behaviour is exercised for
real and nothing is written into this checkout.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gfmcd import record

REPO = Path(__file__).resolve().parents[1]
CENSUS = REPO / "tests/data/record_census.json"


# -- census guard ---------------------------------------------------------------------------------

def test_legacy_census_failures_are_exactly_the_ledger() -> None:
    ledger = json.loads(CENSUS.read_text())
    reports = record.verify_all(tracked_only=True)
    observed = {(r.manifest, c.path, c.sha256, c.status) for r in reports for c in r.failures}
    expected = {(f["manifest"], f["path"], f["sha256"], f["status"]) for f in ledger["failures"]}
    assert not observed - expected, "NEW unverifiable records:\n  " + "\n  ".join(
        map(str, sorted(observed - expected)))
    assert not expected - observed, (
        "These ledger records verify again; regenerate with "
        "`python3 scripts/verify_records.py --write-census`:\n  "
        + "\n  ".join(map(str, sorted(expected - observed))))
    verified = sum(r.counts["ok"] + r.counts["moved"] for r in reports)
    assert verified >= ledger["floor_verified"], f"only {verified} records verify; the guard eroded"


def test_records_written_by_this_module_verify_fully() -> None:
    listed = subprocess.run(["git", "ls-files", "*.manifest.json"], cwd=REPO, capture_output=True,
                            text=True, check=True).stdout.split()
    for name in listed:
        path = REPO / name
        if path.is_file() and json.loads(path.read_text()).get("schema") == record.SCHEMA:
            record.verify(path, tracked_only=True).raise_for_failures()


def test_sibling_clone_paths_map_into_the_repo_and_genomes_stay_external() -> None:
    clone = "/Users/someone/Research/projects/gfm-circuit-discovery-gtex-shift/results/phase-lattice/x.json"
    assert record.repo_relative(clone, REPO) == "results/phase-lattice/x.json"
    assert record.repo_relative("/Users/someone/Research/data/genomes/hg38.fa", REPO) is None
    assert record.repo_relative("results/a.json", REPO) == "results/a.json"


# -- end to end in a throwaway repository ---------------------------------------------------------

def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root,
                   check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "results").mkdir()
    runner = root / "scripts/run_example.py"
    runner.write_text("print('example runner')\n")
    (root / "results/input.json").write_text('{"x": 1}\n')
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "seed")
    monkeypatch.setattr(sys, "argv", [str(runner), "--flag"])
    monkeypatch.setenv("GFMCD_CACHE", str(tmp_path / "cache"))
    return root


def _commit(root: Path) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "update")


def test_write_records_relative_paths_code_and_external_inputs(repo: Path, tmp_path: Path) -> None:
    external = tmp_path / "genome.fa"
    external.write_text(">chr1\nACGT\n")
    out = record.write(repo / "results/out.json", {"value": 0.5}, experiment="example", repo=repo,
                       inputs=[repo / "results/input.json", external], seeds={"bootstrap": 7},
                       model={"name": "m", "revision": "r", "files": {"w.bin": "a" * 64}})
    manifest = json.loads(out.manifest.read_text())
    assert out.manifest == repo / "results/out.manifest.json"
    assert manifest["schema"] == record.SCHEMA and manifest["seeds"] == {"bootstrap": 7}
    assert manifest["outputs"][0]["path"] == "results/out.json"
    assert manifest["inputs"][0]["path"] == "results/input.json"
    assert manifest["inputs"][1] == {"name": "genome.fa", "external": True, "local_path": str(external.resolve()),
                                     "sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
                                     "size_bytes": external.stat().st_size}
    assert [c["path"] for c in manifest["code"]] == ["scripts/run_example.py"]
    assert manifest["model"]["files"] == {"w.bin": "a" * 64}
    assert json.loads(out.path.read_text()) == {"value": 0.5}


def test_write_is_exclusive_strict_and_leaves_nothing_on_refusal(repo: Path) -> None:
    target = repo / "results/out.json"
    record.write(target, {"a": 1}, experiment="e", repo=repo)
    with pytest.raises(FileExistsError):
        record.write(target, {"a": 2}, experiment="e", repo=repo)
    with pytest.raises(ValueError):
        record.write(repo / "results/nan.json", {"a": float("nan")}, experiment="e", repo=repo)
    assert not (repo / "results/nan.json").exists() and not (repo / "results/nan.manifest.json").exists()
    with pytest.raises(ValueError, match="bind"):
        record.write(repo / "results/table.csv", {"a": 1}, experiment="e", repo=repo)


def test_gzip_output_is_deterministic_and_resolvable_by_its_uncompressed_bytes(repo: Path) -> None:
    out = record.write(repo / "results/rows.json.gz", {"rows": list(range(50))}, experiment="e", repo=repo)
    raw = gzip.decompress(out.path.read_bytes())
    assert out.manifest == repo / "results/rows.manifest.json"
    assert out.uncompressed_sha256 == hashlib.sha256(raw).hexdigest()
    assert gzip.compress(raw, compresslevel=9, mtime=0) == out.path.read_bytes()
    _commit(repo)
    found = record.resolve(out.uncompressed_sha256, repo=repo, tracked_only=True)
    assert found.read_bytes() == raw and repo not in found.parents  # materialized outside the tree


def test_verify_classifies_ok_moved_drift_missing_and_external(repo: Path, tmp_path: Path) -> None:
    external = tmp_path / "data/genome.fa"
    external.parent.mkdir()
    external.write_text(">chr1\nACGT\n")
    a = record.write(repo / "results/a.json", {"a": 1}, experiment="e", repo=repo,
                     inputs=[repo / "results/input.json", external])
    b = record.write(repo / "results/b.json", {"b": 1}, experiment="e", repo=repo)
    c = record.write(repo / "results/c.json", {"c": 1}, experiment="e", repo=repo)
    _commit(repo)
    statuses = lambda m, **kw: {ch.path: ch.status for ch in record.verify(m, repo=repo, **kw).checks}  # noqa: E731
    assert statuses(a.manifest, tracked_only=True) == {
        "results/a.json": "ok", "results/input.json": "ok", "genome.fa": "external-unchecked",
        "scripts/run_example.py": "ok"}
    assert statuses(a.manifest, data_roots=[tmp_path / "data"])["genome.fa"] == "external-ok"
    _git(repo, "mv", "results/input.json", "results/moved.json")
    (repo / "results/b.json").write_text('{"b": 2}\n')
    (repo / "results/c.json").unlink()
    _commit(repo)
    assert statuses(a.manifest, tracked_only=True)["results/input.json"] == "moved"
    assert statuses(b.manifest, tracked_only=True)["results/b.json"] == "drift"
    report = record.verify(c.manifest, repo=repo, tracked_only=True)
    assert {ch.path: ch.status for ch in report.checks}["results/c.json"] == "missing"
    with pytest.raises(ValueError, match="unverifiable"):
        report.raise_for_failures()


def test_tracked_only_ignores_untracked_files_at_the_recorded_path(repo: Path) -> None:
    out = record.write(repo / "results/local.json", {"raw": True}, experiment="e", repo=repo)
    (repo / ".gitignore").write_text("results/local.json\n")
    _commit(repo)  # the manifest is tracked; the payload stays local, as raw CC payloads do
    local = {c.path: c.status for c in record.verify(out.manifest, repo=repo).checks}
    clean = {c.path: c.status for c in record.verify(out.manifest, repo=repo, tracked_only=True).checks}
    assert local["results/local.json"] == "ok" and clean["results/local.json"] == "missing"


def test_bind_records_an_existing_file_once(repo: Path) -> None:
    figure = repo / "results/figure.png"
    figure.write_bytes(b"\x89PNG fake")
    bound = record.bind(figure, experiment="fig", repo=repo, inputs=[repo / "results/input.json"])
    assert bound.manifest == repo / "results/figure.manifest.json"
    assert json.loads(bound.manifest.read_text())["outputs"][0]["path"] == "results/figure.png"
    with pytest.raises(FileExistsError):
        record.bind(figure, experiment="fig", repo=repo)
