from __future__ import annotations

from pathlib import Path

from gfmcd.figures.io import figure_provenance


REPO = Path(__file__).resolve().parents[1]


def test_figure_provenance_uses_portable_repository_relative_paths() -> None:
    record = figure_provenance([REPO / "README.md"])

    assert record["sources"][0]["path"] == "README.md"
    assert len(record["sources"][0]["sha256"]) == 64
