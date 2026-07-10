"""Provenance stamping for results and run manifests (TO_FIX.md §1.4, §14).

A result is eligible for active prose only when it carries an artifact trail: the code version,
a seed, the command, and a status. `stamp()` attaches that trail to any result dict; `git_sha()`
and `git_dirty()` read the current checkout. Nothing here touches the network or an accelerator.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUSES = ("EXPLORATORY", "PROVISIONAL", "REPLICATED", "SETTLED", "WITHDRAWN", "RETRACTED", "BLOCKED")


def git_sha(short: bool = False) -> str | None:
    cmd = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except Exception:
        return None


def _versions() -> dict[str, str]:
    v = {"python": platform.python_version()}
    try:
        import torch

        v["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import numpy

        v["numpy"] = numpy.__version__
    except Exception:
        pass
    return v


def stamp(
    result: dict[str, Any],
    *,
    status: str,
    seed: int | dict[str, int] | None,
    script: str | None = None,
    checkpoint: str | None = None,
    timestamp: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Return `result` with a `_provenance` block attached.

    `timestamp` may be passed in for reproducibility; if omitted the current UTC time is used.
    `status` must be one of STATUSES (raises otherwise, so a typo cannot silently pass).
    """
    if status not in STATUSES:
        raise ValueError(f"status {status!r} not in {STATUSES}")
    result = dict(result)
    result["_provenance"] = {
        "status": status,
        "seed": seed,
        "script": script if script is not None else (Path(sys.argv[0]).name if sys.argv and sys.argv[0] else None),
        "checkpoint": checkpoint,
        "command": " ".join(sys.argv) if sys.argv else None,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "timestamp": timestamp if timestamp is not None else datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "os": platform.platform(),
        **extra,
    }
    return result


@dataclass
class RunManifest:
    run_id: str
    study: str
    status: str
    config_path: str | None = None
    model_source: str | None = None
    data_sources: list[str] | None = None
    seeds: dict[str, int] | None = None
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "study": self.study,
            "status": self.status,
            "git_commit": git_sha(),
            "config_path": self.config_path,
            "model_source": self.model_source,
            "data_sources": self.data_sources or [],
            "seeds": self.seeds or {},
            "command": " ".join(sys.argv) if sys.argv else None,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def write(self, run_dir: Path) -> Path:
        run_dir.mkdir(parents=True, exist_ok=True)
        p = run_dir / "manifest.json"
        p.write_text(json.dumps(self.to_dict(), indent=2))
        (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")
        return p


def sidecar_for(npz_path: Path) -> Path:
    """The `.meta.json` sidecar path for a cache file."""
    return npz_path.with_suffix(npz_path.suffix + ".meta.json")


def write_cache_sidecar(npz_path: Path, meta: dict[str, Any]) -> Path:
    """Record how a cache file was generated (generator, seed, split, git, shapes)."""
    meta = dict(meta)
    meta.setdefault("git_sha", git_sha())
    p = sidecar_for(npz_path)
    p.write_text(json.dumps(meta, indent=2))
    return p
