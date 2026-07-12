"""Small, dependency-free helpers for experiment provenance.

The real-GFM studies need a stricter artifact trail than the original exploratory
scripts.  This module intentionally uses only the Python standard library so it
can be called before optional model or genome dependencies are installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of *path* without loading it all into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(repo: str | Path | None = None) -> str | None:
    """Return the current commit hash, or ``None`` outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo) if repo is not None else None,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_dirty(repo: str | Path | None = None) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo) if repo is not None else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def file_record(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    record: dict[str, Any] = {"path": str(p)}
    if p.exists() and p.is_file():
        record.update({"size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    else:
        record.update({"missing": True})
    return record


@dataclass
class ExperimentManifest:
    """Machine-readable record attached to every settled experiment artifact."""

    experiment_id: str
    command: list[str]
    status: str = "EXPLORATORY"
    config: dict[str, Any] = field(default_factory=dict)
    seeds: dict[str, int] = field(default_factory=dict)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    git_commit: str | None = None
    git_dirty: bool | None = None
    environment: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        experiment_id: str,
        command: Iterable[str],
        *,
        repo: str | Path | None = None,
        **kwargs: Any,
    ) -> "ExperimentManifest":
        env = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "pid": os.getpid(),
        }
        return cls(
            experiment_id=experiment_id,
            command=list(command),
            git_commit=git_commit(repo),
            git_dirty=git_dirty(repo),
            environment=env,
            **kwargs,
        )

    def add_input(self, path: str | Path) -> None:
        self.inputs.append(file_record(path))

    def add_output(self, path: str | Path) -> None:
        self.outputs.append(file_record(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return p
