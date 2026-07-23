#!/usr/bin/env python
"""2D companion to the 3D carrier-interaction surface: the same surface, readable off the page.

The 3D render (``phase_graphs_3D/carrier-wave-interaction-paper.svg``) is the right object for a
talk and the wrong one for a paper body -- a perspective surface cannot be read quantitatively.
This figure plots THE SAME FUNCTION, with the same constants, as a heat map with contours plus a
cut along the physical trajectory, so a reader can put a number on any point.

  A  The joint weight over the two carrier coordinates, contoured.  The white/orange line is the
     physical trajectory x = y -- the only line in the plane a real sequence position travels.
  B  The same surface read along that trajectory, factored.  exp(a + b) = exp(a) * exp(b), so the
     period-3 and ~10.63 contributions separate exactly; the product is what the model sees.

NO RECOMPUTATION.  ``joint_weight`` and its constants are imported from the 3D generator, so the
two figures cannot drift apart.  A memorial checks the long period against the recorded carrier
wavelength in ``fig_report_data.json`` and refuses to draw if they disagree.

WHAT THIS FIGURE IS NOT.  It is an analytic picture of the GEOMETRY of two orthogonal positional
carriers, not a measurement of attention.  The amplitudes are the ones set in the 3D generator.
No evidence-level claim attaches to it: the ~10.63 carrier's status is Level A (attention) with
OV/mediation and behaviour OPEN, and this drawing does not change that.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from figstyle import (  # noqa: E402
    CARRIER_1063, COLORS, INK, PERIOD3, panel_label, save_figure, use_style,
)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE_SRC = HERE / "phase_graphs_3D" / "generate_carrier_paper_figure.py"
REPORT = REPO / "results/phase-lattice/fig_report_data.json"
OUTBASE = HERE / "fig_carrier_heatmap"

TRAJECTORY = "#D55E00"   # the trajectory colour the 3D figure already uses; not a carrier hue


def _load_surface():
    spec = importlib.util.spec_from_file_location("_carrier_surface_3d", SURFACE_SRC)
    if spec is None or spec.loader is None:                      # pragma: no cover - unreachable
        raise ImportError(f"cannot load {SURFACE_SRC}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_carrier_surface_3d"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    for p in (SURFACE_SRC, REPORT):
        if not p.exists():
            raise SystemExit(f"missing {p}")
    surface = _load_surface()
    recorded = json.loads(REPORT.read_text())["wavelengths"]["carrier"]["T"]
    if round(recorded, 5) != surface.T_LONG:
        raise RuntimeError(
            f"the drawn long period {surface.T_LONG} does not match the recorded carrier "
            f"wavelength {recorded}; refusing to draw a surface the record does not support")

    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9),
                             gridspec_kw={"width_ratios": [1.0, 1.15]})

    # ---- A: the surface as a heat map ------------------------------------------------
    grid = np.linspace(-15, 15, 181)
    xx, yy = np.meshgrid(grid, grid)
    zz = surface.joint_weight(xx, yy)

    ax = axes[0]
    ax.grid(False)
    mesh = ax.pcolormesh(xx, yy, zz, cmap="viridis", shading="gouraud",
                         vmin=float(zz.min()), vmax=float(zz.max()))
    ax.contour(xx, yy, zz, levels=6, colors="white", linewidths=0.35, alpha=0.55)
    ax.plot(grid, grid, color="white", lw=2.6, solid_capstyle="round")
    ax.plot(grid, grid, color=TRAJECTORY, lw=1.4, solid_capstyle="round",
            label="physical trajectory $x=y$")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect("equal")
    ax.set_xlabel("period-3 coordinate, $x$ (nt)")
    ax.set_ylabel("10.63-nt coordinate, $y$ (nt)")
    ax.legend(loc="upper left", fontsize=6.5, framealpha=0.85, frameon=True)
    bar = fig.colorbar(mesh, ax=ax, shrink=0.9, pad=0.03, aspect=22)
    bar.set_label("unnormalized attention weight $e^{C_3+C_{10}}$", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    ax.set_title("The two-carrier surface, contoured", pad=8)
    panel_label(ax, "A")

    # ---- B: the same surface along the trajectory, factored --------------------------
    s = np.linspace(-15, 15, 1201)
    period3 = np.exp(surface.A_CODON * np.cos(2 * np.pi * s / 3))
    long_carrier = np.exp(surface.A_LONG * np.cos(2 * np.pi * s / surface.T_LONG))
    joint = surface.joint_weight(s, s)

    ax = axes[1]
    ax.plot(s, period3, color=PERIOD3, lw=1.1, label="period-3 factor")
    ax.plot(s, long_carrier, color=CARRIER_1063, lw=1.1,
            label=f"{surface.T_LONG:.2f}-nt factor")
    ax.plot(s, joint, color=INK, lw=1.6, label="product (what the model sees)")
    ax.axhline(1.0, color=COLORS["text"], lw=0.6, alpha=0.5)
    ax.set_xlim(-15, 15)
    ax.set_xlabel("position along $x=y$ (nt)")
    ax.set_ylabel("unnormalized attention weight")
    ax.legend(loc="upper center", ncol=3, fontsize=6.5, columnspacing=1.1,
              handlelength=1.6, bbox_to_anchor=(0.5, 1.02))
    ax.set_ylim(0, float(joint.max()) * 1.42)
    ax.set_title("Read along the trajectory: an exact factorisation", pad=8)
    panel_label(ax, "B")

    fig.tight_layout(w_pad=2.0)
    out = save_figure(fig, OUTBASE, sources=[SURFACE_SRC, REPORT])
    plt.close(fig)
    print(f"wrote {out}.png / .pdf  (+ provenance)")


if __name__ == "__main__":
    main()
