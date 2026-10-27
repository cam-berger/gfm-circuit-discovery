"""gfmcd.pins: the repo-wide pin guard, the parser, and statuses end to end in a throwaway repo."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from gfmcd import pins

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "tests/data/pin_census.json"


def test_unverifiable_pins_are_exactly_the_ledger() -> None:
    ledger = json.loads(LEDGER.read_text())
    checks = pins.check()
    observed = {(c.source, c.path, c.sha256, c.status) for c in checks if c.status in ("drift", "missing")}
    expected = {(f["source"], f["path"], f["sha256"], f["status"]) for f in ledger["failures"]}
    assert not observed - expected, "NEW unverifiable pins (a pinned file changed or vanished):\n  " + \
        "\n  ".join(map(str, sorted(observed - expected)))
    assert not expected - observed, ("These ledger pins verify again; regenerate with "
                                     "`python3 scripts/verify_records.py --pins --write-census`:\n  "
                                     + "\n  ".join(map(str, sorted(expected - observed))))
    verified = sum(c.status in ("ok", "moved") for c in checks)
    assert verified >= ledger["floor_verified"], f"only {verified} pins verify; the guard eroded"


def test_parser_reads_records_and_path_keyed_maps_only() -> None:
    sha = "a" * 64
    document = {
        "code": {"src/x.py": sha, "config.json": sha, "note": "not a pin"},  # bare name skipped
        "inputs": [{"path": "/elsewhere/results/y.json", "sha256": sha}, {"path": "z", "sha256": "short"}],
        "package_sha256": sha,  # a key without "/" names no file
    }
    assert sorted(pins._pins(document)) == [("/elsewhere/results/y.json", sha), ("src/x.py", sha)]


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=root, check=True,
                   capture_output=True)


def test_statuses_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GFMCD_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "tests/data").mkdir(parents=True)
    files = {name: f"print({i})\n".encode() for i, name in enumerate(("a", "b", "c", "d"))}
    for name, data in files.items():
        (root / f"src/{name}.py").write_bytes(data)
    sha = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    archived = b"archived shard\n"
    with tarfile.open(root / "results/shards.tar.gz", "w:gz") as bundle:
        info = tarfile.TarInfo("shards/s1.json")
        info.size = len(archived)
        bundle.addfile(info, io.BytesIO(archived))
    freeze = {"files": {f"src/{n}.py": sha[n] for n in sha},
              "shard": {"path": "results/shards/s1.json", "sha256": hashlib.sha256(archived).hexdigest()}}
    (root / "results/freeze.json").write_text(json.dumps(freeze))
    (root / "results/run.manifest.json").write_text(json.dumps({"files": {"src/zzz.py": "f" * 64}}))
    (root / "tests/data/ledger.json").write_text(json.dumps({"files": {"src/zzz.py": "f" * 64}}))
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "seed")

    (root / "src/b.py").write_text("print('edited')\n")  # drift
    _git(root, "mv", "src/c.py", "src/c_moved.py")         # moved
    _git(root, "rm", "-q", "src/d.py")                     # missing
    _git(root, "commit", "-qam", "change")

    status = {c.path: (c.status, c.resolved) for c in pins.check(root)}
    assert set(status) == {"src/a.py", "src/b.py", "src/c.py", "src/d.py", "results/shards/s1.json"}
    assert status["src/a.py"][0] == "ok"
    assert status["src/b.py"][0] == "drift"
    assert status["src/c.py"] == ("moved", "src/c_moved.py")
    assert status["src/d.py"][0] == "missing"
    assert status["results/shards/s1.json"] == ("moved", "results/shards.tar.gz!shards/s1.json")
