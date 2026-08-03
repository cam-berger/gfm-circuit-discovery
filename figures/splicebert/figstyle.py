"""The SpliceBERT figure style module of record: one import for every ``make_fig*.py``.

Before this module the figure set carried its style in three places -- the CVD-validated SVG
palette in ``scripts/splicebert/figstyle.py``, the Okabe-Ito matplotlib tokens in
``src/gfmcd/figures/style.py``, and a hand-written five-colour palette inside
``make_fig5_p3_phase_surgery_confirmation.py``.  Every generator now imports this file instead,
so fonts, colours, DPI and the PDF+PNG save path have a single definition.

THIS FILE CHANGED NO COLOUR, NO SIZE AND NO DPI.  It is a refactor: every regenerated figure is
byte-identical to the one it replaces.  Where the three palettes disagree (three different blues
are in use) the disagreement is preserved and named here rather than resolved, because resolving
it would restyle published panels; see ``docs/REVIEW_OPEN_QUESTIONS.md``.

THE SVG PALETTE IS RE-EXPORTED, NOT COPIED.  Its colours were computed, not chosen -- every pair
that appears together in a panel was checked for separation under simulated deuteranopia,
protanopia and tritanopia (floor dE 12).  Copying computed values into a second file would create
a second source of truth for them, so this module loads the one definition from
``scripts/splicebert/figstyle.py`` and re-exports it.  Read that module's docstring for the
validation record.

THE COLOUR THREAD, which runs through every figure and must not be broken:

    period-3 carrier  = BLUE (``PERIOD3``)        ~10.63 carrier = ORANGE (``CARRIER_1063``)
    donor-acceptor coupling = PURPLE               nucleosomal 10.298 = RED
    deleted / failed / not Level A = SLATE

Methods figures that describe the workflow rather than a carrier use no semantic hue at all, so
the thread stays reserved for results.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Sequence

REPO = Path(__file__).resolve().parents[2]

# Must be set before matplotlib is first imported; the confirmation figure already relied on it.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gfmcd-matplotlib")

if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _load(module_name: str, path: Path):
    """Import a file by path under an explicit module name.

    The name is explicit so that loading ``scripts/splicebert/figstyle.py`` from a module that is
    itself called ``figstyle`` cannot collide in ``sys.modules``.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:                      # pragma: no cover - unreachable
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_svg = _load("_gfmcd_svg_figstyle", REPO / "scripts/splicebert/figstyle.py")

# ---------------------------------------------------------------- the validated SVG palette
BLUE = _svg.BLUE          # period-3 carrier
ORANGE = _svg.ORANGE      # ~10.63 carrier
PURPLE = _svg.PURPLE      # donor-acceptor coupling (behaviour)
GREEN = _svg.GREEN        # intron
RED = _svg.RED            # nucleosomal 10.298
SLATE = _svg.SLATE        # deleted / failed / not Level A
MIST = _svg.MIST          # recessive context
GOLD = _svg.GOLD          # codon phase 1
MAGENTA = _svg.MAGENTA    # codon phase 2
PALEBLUE = _svg.PALEBLUE
SAND = _svg.SAND
OLIVE = _svg.OLIVE
ASH = _svg.ASH

INK, SEC, MUT, RULE, PANEL = _svg.INK, _svg.SEC, _svg.MUT, _svg.RULE, _svg.PANEL
PHASE = _svg.PHASE        # codon position 0 / 1 / 2
LAYER = _svg.LAYER        # sequential layer ramp
FONT = _svg.FONT

Canvas = _svg.Canvas
nice_p = _svg.nice_p

# ---------------------------------------------------------------- the colour thread, named
PERIOD3 = BLUE
CARRIER_1063 = ORANGE
COUPLING = PURPLE
NUCLEOSOMAL = RED
DELETED = SLATE

# ---------------------------------------------------------------- matplotlib layer
import matplotlib  # noqa: E402

matplotlib.use("Agg")

from gfmcd.figures import (  # noqa: E402
    COLORS, figure_provenance, grouped_bars_ci, interaction_plot, panel_label, ranked_dots,
)
from gfmcd.figures.style import RC  # noqa: E402
from gfmcd.figures.style import use_style as _use_style  # noqa: E402

#: Screen DPI and the raster DPI the project set for publication output.  Both come from
#: ``RC`` and are named here so a figure can ask for them without reaching into rcParams.
DPI_SCREEN = RC["figure.dpi"]
DPI_RASTER = RC["savefig.dpi"]

#: The confirmation figure was authored at 240 dpi with its own five colours.  Both are recorded
#: here, unchanged, so the figure has no private style; unifying them is a restyle, not a
#: refactor, and is left to the owner.
DPI_CONFIRMATION_PANEL = 240
CONFIRMATION = {
    "blue": "#3569a8",
    "red": "#c64b47",
    "teal": "#2a8f88",
    "ink": "#202633",
    "grey": "#87909e",
    "grid": "#d9dee7",
}

_RASTER_FORMATS = {"png", "jpg", "jpeg", "tif", "tiff"}


def use_style() -> None:
    """Apply the project matplotlib rcParams.  Delegates, so there is one RC definition."""
    _use_style()


def panel_header(axis, letter: str, title: str) -> None:
    """Draw one short, consistently aligned panel header.

    Point offsets keep the panel letter and title on the same baseline even
    when neighboring axes have different widths.
    """
    common = {
        "xy": (0, 1),
        "xycoords": "axes fraction",
        "textcoords": "offset points",
        "ha": "left",
        "va": "bottom",
        "fontweight": "bold",
        "color": INK,
        "annotation_clip": False,
    }
    axis.annotate(letter, xytext=(-30, 12), fontsize=11.5, **common)
    axis.annotate(title, xytext=(0, 12), fontsize=10, **common)


def save_raster_and_vector(fig, out: str | Path, *, formats: Sequence[str] = ("png", "pdf"),
                           dpi: float | None = None,
                           bbox_inches: str | None = None) -> list[Path]:
    """Write one figure as BOTH a raster and a vector file and return the paths written.

    ``dpi`` is applied only to raster formats, since it has no meaning for a vector page.  With
    both keywords left at ``None`` the call is exactly ``fig.savefig(path)``, i.e. whatever the
    active rcParams say -- which is how the existing figures were produced.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        kwargs: dict = {}
        if bbox_inches is not None:
            kwargs["bbox_inches"] = bbox_inches
        if dpi is not None and ext in _RASTER_FORMATS:
            kwargs["dpi"] = dpi
        path = out.with_suffix(f".{ext}")
        fig.savefig(path, **kwargs)
        written.append(path)
    return written


def save_figure(fig, out: str | Path, sources: Sequence[str | Path], *,
                formats: Sequence[str] = ("png", "pdf"), dpi: float | None = None,
                bbox_inches: str | None = None) -> Path:
    """Save PDF+PNG and the provenance sidecar that names and hashes the source artifacts.

    A figure that cannot name its sources is decoration, so an empty ``sources`` is refused --
    the same contract as ``gfmcd.figures.save_figure``, which this replaces at the call sites.
    """
    if not sources:
        raise ValueError("a figure must declare the artifact(s) it was drawn from")
    out = Path(out)
    save_raster_and_vector(fig, out, formats=formats, dpi=dpi, bbox_inches=bbox_inches)
    out.with_suffix(".provenance.json").write_text(
        json.dumps(figure_provenance(sources), indent=2) + "\n")
    return out


def style_axis(axis, *, panel: str, title: str) -> None:
    """The confirmation figure's panel styling: left-set bold panel letter, y-grid, open frame."""
    axis.set_title(f"{panel}  {title}", loc="left", fontsize=12, fontweight="bold", pad=10)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=CONFIRMATION["grid"], linewidth=0.7, alpha=0.8)
    axis.tick_params(labelsize=9)


__all__ = [
    "ASH", "BLUE", "CARRIER_1063", "COLORS", "CONFIRMATION", "COUPLING", "Canvas",
    "DELETED", "DPI_CONFIRMATION_PANEL", "DPI_RASTER", "DPI_SCREEN", "FONT", "GOLD", "GREEN",
    "INK", "LAYER", "MAGENTA", "MIST", "MUT", "NUCLEOSOMAL", "OLIVE", "ORANGE", "PALEBLUE",
    "PANEL", "PERIOD3", "PHASE", "PURPLE", "RC", "RED", "RULE", "SAND", "SEC", "SLATE",
    "figure_provenance", "grouped_bars_ci", "interaction_plot", "nice_p", "panel_label",
    "panel_header", "ranked_dots", "save_figure", "save_raster_and_vector", "style_axis",
    "use_style",
]
