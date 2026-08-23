"""Figures must carry the provenance of the data they were drawn from.

A figure that cannot name (and hash) its source artifact is not evidence -- it is
decoration, and it will silently drift from the numbers the moment a run is re-done.
``save_figure`` refuses to write without at least one source artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Sequence


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def figure_provenance(sources: Sequence[str | Path]) -> dict:
    try:
        root_text = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
        ).stdout.strip()
        repo_root = Path(root_text).resolve() if root_text else None
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip() or None
    except Exception:
        repo_root = None
        commit = None

    def display_path(source: str | Path) -> str:
        path = Path(source)
        resolved = path.resolve()
        if repo_root is not None:
            try:
                return str(resolved.relative_to(repo_root))
            except ValueError:
                pass
        return str(path)

    return {
        "git_commit": commit,
        "sources": [
            {"path": display_path(source), "sha256": _sha256(Path(source))}
            for source in sources
        ],
    }


def save_figure(fig, out: str | Path, sources: Sequence[str | Path], *,
                formats: Sequence[str] = ("png", "pdf")) -> Path:
    if not sources:
        raise ValueError("a figure must declare the artifact(s) it was drawn from")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(out.with_suffix(f".{ext}"))
    (out.with_suffix(".provenance.json")).write_text(
        json.dumps(figure_provenance(sources), indent=2) + "\n")
    return out
