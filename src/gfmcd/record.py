"""Write, verify, and resolve result records by content: one module for the result seam.

Every result crosses this seam, yet it had grown four writers and about twenty ad hoc readers.
Of 3,228 manifest records that carry a sha256, 1,551 verified at their recorded path; the rest had
moved, drifted, gone missing, or pointed at another machine, because records stored absolute paths.

    write(path, payload, ...)   result JSON (or .json.gz) plus sibling manifest, both exclusive
    bind(path, ...)             manifest for a file the caller just produced
    verify(manifest_path)       classify every record; never raises (verify_all: many, one index)
    resolve(sha256, hint)       find bytes by content: path, tracked file, archive, data roots

New manifests keep the keys ``provenance.ExperimentManifest`` writes (so existing readers work) and
add ``schema``, ``code`` (runner and loaded repo modules by sha256), ``tracked_modified`` and
``model``. In-repo paths are repo-relative; external files are named, hashed, and carry a
``local_path`` that verification never trusts. Standard library only, like the pinned
``provenance`` and ``artifact_guard`` it wraps, so a plain clone can verify records.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping

from gfmcd.artifact_guard import guard_output_path
from gfmcd.provenance import ExperimentManifest

SCHEMA = "gfmcd.record/1"
REPO = Path(__file__).resolve().parents[2]
FAILURES = ("drift", "missing")
_SHA256 = re.compile(r"[0-9a-f]{64}")


# -- public interface ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    path: Path
    manifest: Path
    sha256: str
    uncompressed_sha256: str | None = None


@dataclass(frozen=True)
class Check:
    path: str  # as recorded (a path, or an external file's name)
    sha256: str
    status: str  # ok | moved | drift | missing | external-ok | external-unchecked
    resolved: str | None = None


@dataclass(frozen=True)
class Report:
    manifest: str
    checks: tuple[Check, ...] = field(default_factory=tuple)

    @property
    def counts(self) -> Counter:
        return Counter(check.status for check in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if check.status in FAILURES)

    def raise_for_failures(self) -> None:
        if self.failures:
            listed = "\n  ".join(f"{c.status}: {c.path} ({c.sha256[:12]})" for c in self.failures)
            raise ValueError(f"{self.manifest} has unverifiable records:\n  {listed}")


def write(path: str | Path, payload: Any, *, experiment: str, inputs: Iterable[str | Path] = (),
          config: Mapping | None = None, seeds: Mapping | None = None, status: str = "COMPLETE",
          notes: Iterable[str] = (), model: Mapping | None = None, repo: str | Path | None = None) -> Record:
    """Write ``payload`` as strict JSON (gzip if the name ends in ``.gz``) plus its manifest."""
    out, manifest = _claim(path)
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    data = gzip.compress(raw, compresslevel=9, mtime=0) if out.suffix == ".gz" else raw
    record = _new_manifest(experiment, inputs, config, seeds, status, notes, model, repo)  # before writing
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        handle.write(data)
    return _seal(out, manifest, record, repo, uncompressed=raw if out.suffix == ".gz" else None)


def bind(path: str | Path, *, experiment: str, inputs: Iterable[str | Path] = (),
         config: Mapping | None = None, seeds: Mapping | None = None, status: str = "COMPLETE",
         notes: Iterable[str] = (), model: Mapping | None = None, repo: str | Path | None = None) -> Record:
    """Write the manifest for a file the caller has just produced (a figure, a table)."""
    out = Path(path)
    if not out.is_file():
        raise FileNotFoundError(out)
    manifest = guard_output_path(manifest_path(out))
    record = _new_manifest(experiment, inputs, config, seeds, status, notes, model, repo)
    uncompressed = gzip.decompress(out.read_bytes()) if out.suffix == ".gz" else None
    return _seal(out, manifest, record, repo, uncompressed=uncompressed)


def verify(manifest: str | Path, *, repo: str | Path | None = None, tracked_only: bool = False,
           data_roots: Iterable[str | Path] | None = None, _shared: "_Index | None" = None) -> Report:
    """Classify every sha256-bearing record in a manifest. ``tracked_only`` ignores files git does
    not track, so the answer does not depend on what happens to sit on this disk."""
    root = Path(repo or REPO).resolve()
    index = _shared or _Index(root, tracked_only, data_roots)
    document = json.loads(Path(manifest).read_text())
    checks = []
    for item in _records(document):
        recorded, sha = item.get("path") or item.get("name"), item["sha256"]
        relative = None if item.get("external") else repo_relative(recorded, root, exists=index.exists)
        if relative is None:
            found = index.external(_basename(recorded), sha)
            checks.append(Check(recorded, sha, "external-ok" if found else "external-unchecked",
                                str(found) if found else None))
            continue
        at_path = root / relative
        if index.usable(at_path) and index.sha(at_path) == sha:
            checks.append(Check(recorded, sha, "ok", relative))
            continue
        elsewhere = index.find(sha, item.get("size_bytes"))
        if elsewhere is not None:
            checks.append(Check(recorded, sha, "moved", _display(elsewhere, root)))
        else:
            checks.append(Check(recorded, sha, "drift" if index.usable(at_path) else "missing"))
    return Report(_display(Path(manifest).resolve(), root), tuple(checks))


def verify_all(manifests: Iterable[str | Path] | None = None, *, repo: str | Path | None = None,
               tracked_only: bool = True, data_roots: Iterable[str | Path] | None = None) -> list[Report]:
    """``verify`` over many manifests with one shared index; default: every tracked manifest."""
    root = Path(repo or REPO).resolve()
    index = _Index(root, tracked_only, data_roots)
    if manifests is None:
        manifests = sorted(p for p in index.tracked if p.name.endswith(".manifest.json") and p.is_file())
    return [verify(m, repo=root, _shared=index) for m in manifests]


def resolve(sha256: str, hint: str | Path | None = None, *, size_bytes: int | None = None,
            repo: str | Path | None = None, tracked_only: bool = False,
            data_roots: Iterable[str | Path] | None = None) -> Path:
    """Return a file holding these bytes: the hinted path, any tracked file, the decompressed
    contents of a tracked ``.gz`` (materialized in a cache, never in the tree), or a data root."""
    if not _SHA256.fullmatch(sha256):
        raise ValueError(f"not a sha256: {sha256!r}")
    root = Path(repo or REPO).resolve()
    index = _Index(root, tracked_only, data_roots)
    if hint is not None:
        relative = repo_relative(str(hint), root, exists=index.exists)
        if relative is not None and index.usable(root / relative) and index.sha(root / relative) == sha256:
            return root / relative
    found = index.find(sha256, size_bytes)
    if found is None and hint is not None:
        found = index.external(_basename(str(hint)), sha256)
    if found is None:
        raise FileNotFoundError(f"no committed file matches {hint or sha256} ({sha256[:12]})")
    return found


def manifest_path(artifact: str | Path) -> Path:
    """``x.json`` / ``x.json.gz`` / ``x.png`` -> ``x.manifest.json``."""
    p = Path(artifact)
    if p.suffix == ".gz":
        p = p.with_suffix("")
    return p.with_suffix(".manifest.json")


def repo_relative(path: str, repo: str | Path | None = None, *, exists=None) -> str | None:
    """Map a recorded path into the repository, or ``None`` if it names an external file.

    Absolute paths from any clone or worktree map by their tail starting at a repository
    top-level directory, provided that directory's next component exists here (``exists`` decides,
    on repo-relative strings; the default asks the filesystem).
    """
    root = Path(repo or REPO).resolve()
    exists = exists or (lambda relative: (root / relative).exists())
    if not path.startswith("/"):
        return path
    candidate = Path(path)
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        pass
    parts = candidate.parts
    for i in range(1, len(parts) - 1):
        if exists(parts[i]) and exists(f"{parts[i]}/{parts[i + 1]}"):
            return "/".join(parts[i:])
    return None


# -- implementation --------------------------------------------------------------------------------


def _basename(path: str) -> str:
    return path.rstrip("/").rsplit("/", 1)[-1]


def _claim(path: str | Path) -> tuple[Path, Path]:
    out = guard_output_path(Path(path))
    if out.suffix not in (".json", ".gz") or (out.suffix == ".gz" and not out.name.endswith(".json.gz")):
        raise ValueError("write() produces .json or .json.gz; use bind() for other files")
    return out, guard_output_path(manifest_path(out))


def _new_manifest(experiment, inputs, config, seeds, status, notes, model, repo) -> dict:
    root = Path(repo or REPO).resolve()
    base = ExperimentManifest.create(experiment, sys.argv, repo=root, status=status,
                                     config=dict(config or {}), seeds=dict(seeds or {}), notes=list(notes))
    document = base.to_dict()
    document["schema"] = SCHEMA
    document["inputs"] = [_file_record(Path(p), root) for p in inputs]
    document["code"] = _code_closure(root)
    document["tracked_modified"] = _git_lines(root, "diff", "--name-only", "HEAD")
    if model is not None:
        files = dict(model.get("files", {}))
        if not files or not all(_SHA256.fullmatch(str(v)) for v in files.values()):
            raise ValueError("model needs files={name: sha256}")
        document["model"] = {"name": model.get("name"), "revision": model.get("revision"), "files": files}
    return document


def _seal(out: Path, manifest: Path, document: dict, repo, *, uncompressed: bytes | None) -> Record:
    root = Path(repo or REPO).resolve()
    output = _file_record(out, root)
    inner = None
    if uncompressed is not None:
        inner = hashlib.sha256(uncompressed).hexdigest()
        output.update({"uncompressed_sha256": inner, "uncompressed_size_bytes": len(uncompressed)})
    document["outputs"] = [output]
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with manifest.open("x") as handle:  # a lone output without its manifest is an abandoned attempt
        handle.write(text)
    return Record(out, manifest, output["sha256"], inner)


def _file_record(path: Path, root: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    record = {"sha256": _hash_file(path), "size_bytes": path.stat().st_size}
    try:
        record["path"] = path.relative_to(root).as_posix()
    except ValueError:
        record.update({"name": path.name, "external": True, "local_path": str(path)})
    return record


def _code_closure(root: Path) -> list[dict]:
    files = set()
    runner = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else None
    if runner is not None and runner.is_file():
        files.add(runner)
    for module in list(sys.modules.values()):
        origin = getattr(module, "__file__", None)
        if origin:
            files.add(Path(origin).resolve())
    keep = []
    for f in files:
        try:
            relative = f.relative_to(root)
        except ValueError:
            continue
        if relative.parts[0] in ("scripts", "src") and f.suffix == ".py" and f.is_file():
            keep.append({"path": relative.as_posix(), "sha256": _hash_file(f)})
    return sorted(keep, key=lambda r: r["path"])


def _git_lines(root: Path, *args: str) -> list[str] | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return sorted(line for line in out.splitlines() if line)


def _records(document: Any):
    if isinstance(document, dict):
        sha = document.get("sha256")
        named = isinstance(document.get("path"), str) or (
            document.get("external") and isinstance(document.get("name"), str))
        if named and isinstance(sha, str) and _SHA256.fullmatch(sha):
            yield document
        for value in document.values():
            yield from _records(value)
    elif isinstance(document, list):
        for value in document:
            yield from _records(value)


def _display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


_HASHES: dict[tuple[str, int, int], str] = {}


def _hash_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        _HASHES[key] = digest.hexdigest()
    return _HASHES[key]


class _Index:
    """Content lookup over one repository; hashes lazily, by size first."""

    def __init__(self, root: Path, tracked_only: bool, data_roots) -> None:
        self.root, self.tracked_only = root, tracked_only
        listed = _git_lines(root, "ls-files")
        if listed is None:  # not a git checkout: everything present counts as committed
            listed = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
        self.tracked = {root / p for p in listed}
        self.tracked_paths = set(listed) | {
            "/".join(p.split("/")[:k]) for p in listed for k in range(1, p.count("/") + 1)}
        self.by_size: dict[int, list[Path]] = {}
        for p in sorted(self.tracked):
            if p.is_file():
                self.by_size.setdefault(p.stat().st_size, []).append(p)
        self.archives = sorted(p for p in self.tracked if p.suffix == ".gz" and p.is_file())
        self._inflated: dict[Path, str] = {}
        roots = data_roots if data_roots is not None else filter(
            None, os.environ.get("GFMCD_DATA_ROOTS", "").split(os.pathsep))
        self.data_roots = [Path(r) for r in roots]

    def exists(self, relative: str) -> bool:
        """Whether a repo-relative path exists, counting only tracked paths when tracked_only."""
        return relative in self.tracked_paths if self.tracked_only else (self.root / relative).exists()

    def usable(self, path: Path) -> bool:
        return path.is_file() and (not self.tracked_only or path in self.tracked)

    def sha(self, path: Path) -> str:
        return _hash_file(path)

    def find(self, sha256: str, size: int | None = None) -> Path | None:
        # Identical bytes have identical size, so a known size narrows the search to one bucket.
        sizes = [size] if isinstance(size, int) else sorted(self.by_size)
        for s in sizes:
            for p in self.by_size.get(s, ()):
                if _hash_file(p) == sha256:
                    return p
        return self._from_archive(sha256)

    def _from_archive(self, sha256: str) -> Path | None:
        for archive in self.archives:
            if archive not in self._inflated:
                digest = hashlib.sha256()
                with gzip.open(archive, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
                self._inflated[archive] = digest.hexdigest()
            if self._inflated[archive] == sha256:
                return _materialize(archive, sha256)
        return None

    def external(self, name: str, sha256: str) -> Path | None:
        for base in self.data_roots:
            for candidate in [base / name, *base.rglob(name)]:
                if candidate.is_file() and _hash_file(candidate) == sha256:
                    return candidate
        return None


def _materialize(archive: Path, sha256: str) -> Path:
    cache = Path(os.environ.get("GFMCD_CACHE", Path(tempfile.gettempdir()) / "gfmcd-record-cache"))
    target = cache / sha256
    if not (target.is_file() and _hash_file(target) == sha256):
        cache.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        with gzip.open(archive, "rb") as source, partial.open("wb") as sink:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                sink.write(chunk)
        partial.replace(target)
    return target
