"""Every sha256 pin outside result manifests, found and checked in one place.

Protocol freezes, confirmation packages, attempt and audit records, figure sidecars and
attestations pin code, protocol documents and data by sha256. Each had its own checker, or none:
the C36 package and the CC freezes were guarded, the Head-Context Atlas packages, the CC attempt and
design-audit records, and the attestations were not, so a pinned file could change unnoticed.

    inventory(repo)   every pin: ``{"path": ..., "sha256": ...}`` records and ``{path: sha256}``
                      maps in tracked JSON, except result manifests (gfmcd.record's census) and
                      tests/data (ledgers)
    check(repo)       classify each pin with gfmcd.record's statuses and resolver

Map keys count as paths only when they contain "/": bare names (a checkpoint's ``config.json``)
are model files, which ``gfmcd.models`` verifies. Python-literal pins inside scripts are checked
by their own scripts and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable

from gfmcd import record

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Pin:
    source: str  # the committed JSON that holds the pin
    path: str  # as recorded
    sha256: str


@dataclass(frozen=True)
class PinCheck:
    source: str
    path: str
    sha256: str
    status: str  # gfmcd.record's: ok | moved | drift | missing | external-ok | external-unchecked
    resolved: str | None = None


def sources(repo: str | Path | None = None) -> list[str]:
    """Tracked JSON that may carry pins: everything except result manifests and test ledgers."""
    root = Path(repo or record.REPO).resolve()
    listed = subprocess.run(["git", "ls-files", "*.json"], cwd=root, capture_output=True, text=True,
                            check=True).stdout.split()
    return sorted(f for f in listed
                  if not f.endswith(".manifest.json") and not f.startswith("tests/data/"))


def inventory(repo: str | Path | None = None) -> list[Pin]:
    root = Path(repo or record.REPO).resolve()
    pins: set[Pin] = set()
    for source in sources(root):
        try:
            document = json.loads((root / source).read_text())
        except (OSError, ValueError):
            continue
        pins.update(Pin(source, path, sha) for path, sha in _pins(document))
    return sorted(pins, key=lambda p: (p.source, p.path, p.sha256))


def check(repo: str | Path | None = None, *, tracked_only: bool = True,
          data_roots: Iterable[str | Path] | None = None) -> list[PinCheck]:
    root = Path(repo or record.REPO).resolve()
    pins = inventory(root)
    checks = record.classify(({"path": p.path, "sha256": p.sha256} for p in pins), repo=root,
                             tracked_only=tracked_only, data_roots=data_roots)
    return [PinCheck(p.source, p.path, p.sha256, c.status, c.resolved) for p, c in zip(pins, checks)]


def _pins(document: Any):
    if isinstance(document, dict):
        path, sha = document.get("path"), document.get("sha256")
        if isinstance(path, str) and isinstance(sha, str) and _SHA256.fullmatch(sha):
            yield path, sha
        for key, value in document.items():
            if isinstance(value, str) and "/" in key and _SHA256.fullmatch(value):
                yield key, value
            else:
                yield from _pins(value)
    elif isinstance(document, list):
        for value in document:
            yield from _pins(value)
