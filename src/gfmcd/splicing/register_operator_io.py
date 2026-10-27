"""Immutable JSON receipts and source binding for the register-operator study."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from gfmcd.provenance import ExperimentManifest, file_record, sha256_file


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path: str | Path) -> dict:
    raw = Path(path).read_bytes()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def manifest_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix == ".gz":
        p = p.with_suffix("")
    return p.with_suffix(".manifest.json")


def guard_paths(path: str | Path) -> tuple[Path, Path]:
    out = Path(path)
    manifest = manifest_path(out)
    for p in (out, manifest):
        if p.exists() or p.is_symlink():
            raise FileExistsError(f"immutable output already exists: {p}")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out, manifest


def write_artifact(path: str | Path, value: dict, *, inputs: list[str | Path],
                   experiment: str, config: dict | None = None,
                   seeds: dict | None = None, status: str | None = None) -> dict:
    out, mp = guard_paths(path)
    raw = json_bytes(value)
    payload = gzip.compress(raw, mtime=0) if out.suffix == ".gz" else raw
    manifest = ExperimentManifest.create(
        experiment, sys.argv, repo=Path(__file__).resolve().parents[3],
        status=status or value.get("status", "COMPLETE"), config=config or {}, seeds=seeds or {},
        notes=["Actual input/source bytes are pinned; HEAD alone does not identify this dirty checkout.",
               f"uncompressed_json_sha256={hashlib.sha256(raw).hexdigest()}"])
    for p in inputs:
        if not Path(p).is_file():
            raise FileNotFoundError(p)
        manifest.add_input(p)
    # Both writes are exclusive. A partial pair is an abandoned attempt, never overwritten.
    with out.open("xb") as handle:
        handle.write(payload)
    manifest.add_output(out)
    with mp.open("x") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return {"output": file_record(out), "manifest": file_record(mp),
            "uncompressed_sha256": hashlib.sha256(raw).hexdigest()}


def verify_freeze(path: str | Path, repo: str | Path) -> dict:
    freeze = read_json(path)
    if freeze.get("status") != "FROZEN":
        raise ValueError("execution requires a frozen package")
    for name, expected in freeze["files"].items():
        p = Path(name)
        if not p.is_absolute():
            p = Path(repo) / p
        if not p.is_file() or sha256_file(p) != expected:
            raise ValueError(f"frozen file changed or missing: {p}")
    return freeze


def verify_artifact_manifest(path: str | Path, *, verify_inputs: bool = True) -> dict:
    out = Path(path).resolve(strict=True)
    manifest = read_json(manifest_path(out))
    matches = [r for r in manifest.get("outputs", []) if Path(r["path"]).resolve() == out]
    if len(matches) != 1 or matches[0].get("sha256") != sha256_file(out):
        raise ValueError(f"manifest does not bind this artifact: {out}")
    if verify_inputs:
        for record in manifest.get("inputs", []):
            p = Path(record["path"])
            if not p.is_file() or sha256_file(p) != record.get("sha256"):
                raise ValueError(f"artifact input has changed: {p}")
    return manifest


def require_manifest_input(manifest: dict, path: str | Path) -> None:
    """A PASS for another input must never authorize this package."""
    p = Path(path).resolve(strict=True)
    rows = [r for r in manifest.get("inputs", []) if Path(r["path"]).resolve() == p]
    if len(rows) != 1 or rows[0].get("sha256") != sha256_file(p):
        raise ValueError(f"audit does not bind requested input: {p}")
