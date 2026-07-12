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
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "git_commit": commit,
        "sources": [{"path": str(Path(s)), "sha256": _sha256(Path(s))} for s in sources],
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
