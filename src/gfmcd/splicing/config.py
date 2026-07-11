"""JSON configuration helpers for the SpliceBERT study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError("study config must be a JSON object")
    return data


def resolve_repo_path(repo: str | Path, value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(repo) / p
