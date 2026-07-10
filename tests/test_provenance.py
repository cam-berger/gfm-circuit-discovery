"""Provenance stamping + reference-manifest schema (TO_FIX.md §12, §14).

No network, no genome: validates the manifest schema and the stamp helpers only.
"""

from __future__ import annotations

import json

import pytest

from gfmcd import provenance
from gfmcd.paths import project_paths


def test_reference_manifest_schema():
    manifest = json.loads((project_paths().references / "manifest.json").read_text())
    assert manifest["assembly"]
    assert manifest["files"], "manifest lists no files"
    for name, spec in manifest["files"].items():
        for key in ("url", "download_name", "install_name", "sha256", "license"):
            assert key in spec, f"{name} missing {key}"
        assert spec["url"].startswith("http")
        # sha256 is null until pinned, but the key must exist (we never fabricate a checksum).
        assert spec["sha256"] is None or isinstance(spec["sha256"], str)


def test_stamp_attaches_trail_and_validates_status():
    out = provenance.stamp({"acc": 0.5}, status="PROVISIONAL", seed=0, script="x.py", timestamp="2026-07-10T00:00:00Z")
    p = out["_provenance"]
    assert p["status"] == "PROVISIONAL"
    assert p["seed"] == 0
    assert p["timestamp"] == "2026-07-10T00:00:00Z"
    assert "git_sha" in p and "versions" in p and "command" in p
    assert out["acc"] == 0.5  # original payload preserved


def test_stamp_rejects_unknown_status():
    with pytest.raises(ValueError):
        provenance.stamp({}, status="DONE", seed=1)


def test_run_manifest_writes_files(tmp_path):
    rm = provenance.RunManifest(run_id="r1", study="reading_frame", status="EXPLORATORY", seeds={"model": 0, "data": 1})
    p = rm.write(tmp_path / "r1")
    blob = json.loads(p.read_text())
    assert blob["run_id"] == "r1" and blob["seeds"] == {"model": 0, "data": 1}
    assert (tmp_path / "r1" / "command.txt").exists()


def test_cache_sidecar_roundtrip(tmp_path):
    npz = tmp_path / "w.npz"
    npz.write_bytes(b"\x00")
    side = provenance.write_cache_sidecar(npz, {"generator": "sample_windows", "seed": 4242, "shape": [8, 256]})
    meta = json.loads(side.read_text())
    assert meta["seed"] == 4242 and "git_sha" in meta
    assert side == provenance.sidecar_for(npz)
