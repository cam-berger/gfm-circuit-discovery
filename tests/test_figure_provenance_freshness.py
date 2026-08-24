"""Every hashed figure sidecar must still describe the artifacts on disk.

`figure_provenance` records a sha256 per source at render time, but nothing re-read those
hashes afterwards: `verify_recorded_sources` runs only inside the two editable renderers,
and only when someone re-renders. So a figure could keep a stale sidecar indefinitely --
an artifact gets re-run, the figure is not regenerated, and the recorded hash quietly stops
matching the file it names. That is the same failure class as the hand-composited DNABERT-2
paper figure, which drifted from its own cited artifact because no generator and no sidecar
tied the two together.

These checks are cheap, hermetic, and run on every push. They do not open an artifact's
scientific content; they prove the figure set and the results tree still describe each other.

The active figure directory contains only manuscript figures, and every one must have a hashed
sidecar. The checks below prove that the figure set and results tree still describe each other.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sidecars() -> list[Path]:
    return sorted(FIGURES.rglob("*.provenance.json"))


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def test_sidecars_exist() -> None:
    assert _sidecars(), "no figure provenance sidecars found; the figure set lost its provenance"


def test_every_paper_figure_has_a_sidecar() -> None:
    rendered = sorted(FIGURES.glob("*.png"))
    assert rendered, "no manuscript figures found"
    missing = [path.name for path in rendered if not path.with_suffix(".provenance.json").exists()]
    assert not missing, f"manuscript figures missing provenance sidecars: {missing}"


@pytest.mark.parametrize("sidecar", _sidecars(), ids=lambda p: _rel(p))
def test_recorded_source_hashes_are_current(sidecar: Path) -> None:
    """A recorded hash that no longer matches means the figure is stale, not the artifact."""
    record = json.loads(sidecar.read_text())
    sources = record.get("sources")
    assert sources, f"{_rel(sidecar)} records no sources; a figure must name its artifacts"

    stale: list[str] = []
    for item in sources:
        assert isinstance(item, dict), (
            f"{_rel(sidecar)} uses the pre-hashing schema; new sidecars must record "
            f"{{path, sha256}} so drift is detectable"
        )
        source = REPO / item["path"]
        if not source.exists():
            stale.append(f"missing:  {item['path']}")
            continue
        observed = _sha256(source)
        if observed != item["sha256"]:
            stale.append(
                f"changed:  {item['path']}\n"
                f"            recorded {item['sha256']}\n"
                f"            on disk  {observed}"
            )

    assert not stale, (
        f"{_rel(sidecar)} no longer describes its sources -- regenerate the figure:\n  "
        + "\n  ".join(stale)
    )


@pytest.mark.parametrize("sidecar", _sidecars(), ids=lambda p: _rel(p))
def test_recorded_source_paths_are_repository_relative(sidecar: Path) -> None:
    """An absolute path in a sidecar records one machine's filesystem, not a provenance."""
    record = json.loads(sidecar.read_text())
    absolute = [
        item["path"] for item in record["sources"] if Path(item["path"]).is_absolute()
    ]
    assert not absolute, f"{_rel(sidecar)} records absolute paths: {absolute}"


@pytest.mark.parametrize("sidecar", _sidecars(), ids=lambda p: _rel(p))
def test_sidecar_has_a_rendered_figure_beside_it(sidecar: Path) -> None:
    """A provenance record with no figure beside it is an orphan, not evidence."""
    stem = Path(str(sidecar).removesuffix(".provenance.json"))
    rendered = [stem.with_suffix(ext) for ext in (".png", ".pdf", ".svg")]
    assert any(path.exists() for path in rendered), (
        f"{_rel(sidecar)} has no rendered figure beside it"
    )
