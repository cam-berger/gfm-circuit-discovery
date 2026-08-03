#!/usr/bin/env python
"""Full donor–acceptor mechanism and its held-out faithfulness boundary.

Panel A is a compact topology rendering of the structured causal core. Panel B
is drawn directly from the A12 held-out complement-faithfulness curve. Numeric
claims are kept out of the schematic unless they are themselves plotted.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from figstyle import (  # noqa: E402
    COUPLING,
    INK,
    MUT,
    PANEL,
    PERIOD3,
    RULE,
    SEC,
    panel_header,
    save_figure,
    use_style,
)

A10 = REPO / "results" / "splicebert_coupling" / "a10_generator_edges.json"
A12 = REPO / "results" / "splicebert_coupling" / "a12_faithfulness_size.json"
A3 = REPO / "results" / "splicebert_coupling" / "a3_l2h15_address_payload.json"
CONFIRMATION = REPO / "results" / "splicebert_coupling" / "confirmation.json"
DONOR_ANCHOR = REPO / "results" / "splicebert_coupling" / "donor_anchor.json"
READOUT_PATH = REPO / "results" / "splicebert_coupling" / "readout_path.json"
CARRIER_DOSE = REPO / "results" / "phase-lattice" / "carrier_dose.json"
OUT = HERE / "fig_full_circuit"

def _node(
    ax,
    center: tuple[float, float],
    size: tuple[float, float],
    title: str,
    subtitle: str,
    *,
    edge: str,
    fill: str = "white",
    linewidth: float = 1.2,
) -> dict[str, float]:
    cx, cy = center
    width, height = size
    left, bottom = cx - width / 2, cy - height / 2
    patch = FancyBboxPatch(
        (left, bottom),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=fill,
        edgecolor=edge,
        linewidth=linewidth,
        clip_on=True,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy + 0.025,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color=INK,
        zorder=4,
    )
    ax.text(
        cx,
        cy - 0.040,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.7,
        color=SEC,
        zorder=4,
    )
    return {
        "left": left,
        "right": left + width,
        "bottom": bottom,
        "top": bottom + height,
        "cx": cx,
        "cy": cy,
    }


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str,
    linewidth: float = 1.7,
    rad: float = 0.0,
    linestyle: str = "-",
) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=5,
        clip_on=True,
        zorder=2,
    )
    ax.add_patch(patch)


def _check_sources() -> tuple[dict, dict]:
    a10 = json.loads(A10.read_text())
    a12 = json.loads(A12.read_text())
    if not (
        a10.get("confirmatory")
        and a10.get("preflight_PASS")
        and a10.get("serial_chain_confirmed")
    ):
        raise RuntimeError("refusing to draw an unqualified generator topology")
    if not (a12.get("confirmatory") and a12.get("preflight_PASS")):
        raise RuntimeError("refusing to draw an unqualified faithfulness curve")
    return a10, a12


def main() -> None:
    a10, a12 = _check_sources()

    use_style()
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "lines.linewidth": 1.8,
        }
    )

    fig = plt.figure(figsize=(11.0, 3.45))
    ax = fig.add_axes((0.035, 0.10, 0.665, 0.78))
    faith = fig.add_axes((0.775, 0.10, 0.205, 0.78))

    # A — two model-internal streams converge, then move through the readout.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_header(ax, "A", "Circuit Architecture")

    seq = _node(
        ax, (0.095, 0.82), (0.135, 0.18), "Acceptor-local", "sequence state",
        edge=COUPLING, fill=PANEL,
    )
    heads = _node(
        ax, (0.265, 0.82), (0.155, 0.18), "L0H7 · H12 · H10", "payload heads",
        edge=COUPLING, fill=PANEL,
    )
    mlp01 = _node(
        ax, (0.425, 0.82), (0.125, 0.18), "MLP0 → MLP1", "serial backbone",
        edge=COUPLING, fill=PANEL,
    )

    p3 = _node(
        ax, (0.095, 0.18), (0.145, 0.18), "Period-3 code", "coordinate",
        edge=PERIOD3, fill="#F3F8FC",
    )
    l2 = _node(
        ax, (0.260, 0.18), (0.110, 0.18), "L2H15", "address",
        edge=PERIOD3, fill="#F3F8FC",
    )
    mlp2 = _node(
        ax, (0.500, 0.50), (0.100, 0.18), "MLP2", "convergence",
        edge=COUPLING, fill="white", linewidth=1.5,
    )
    l3 = _node(
        ax, (0.625, 0.50), (0.100, 0.18), "L3H1", "binds + writes",
        edge=COUPLING, fill="#F3EFF8", linewidth=1.7,
    )
    uinfo = _node(
        ax, (0.750, 0.50), (0.090, 0.18), r"$U_{\mathrm{info}}$", "donor state",
        edge=COUPLING, fill=PANEL,
    )
    transport = _node(
        ax, (0.860, 0.50), (0.095, 0.18), "L4H0/2 · L5H11", "transport",
        edge=MUT, fill="white",
    )
    readout = _node(
        ax, (0.960, 0.50), (0.065, 0.18), "MLM", "donor\nmargin",
        edge=MUT, fill="white",
    )

    _arrow(ax, (seq["right"], seq["cy"]), (heads["left"], heads["cy"]), color=COUPLING)
    _arrow(ax, (heads["right"], heads["cy"]), (mlp01["left"], mlp01["cy"]), color=COUPLING)
    _arrow(
        ax,
        (mlp01["right"], mlp01["cy"]),
        (mlp2["left"], mlp2["cy"] + 0.04),
        color=COUPLING,
        rad=-0.08,
    )
    _arrow(ax, (p3["right"], p3["cy"]), (l2["left"], l2["cy"]), color=PERIOD3)
    _arrow(
        ax,
        (l2["right"], l2["cy"]),
        (mlp2["left"], mlp2["cy"] - 0.04),
        color=PERIOD3,
        rad=0.08,
    )
    _arrow(ax, (mlp2["right"], mlp2["cy"]), (l3["left"], l3["cy"]), color=COUPLING, linewidth=2.0)
    _arrow(ax, (l3["right"], l3["cy"]), (uinfo["left"], uinfo["cy"]), color=COUPLING)
    _arrow(ax, (uinfo["right"], uinfo["cy"]), (transport["left"], transport["cy"]), color=COUPLING)
    _arrow(ax, (transport["right"], transport["cy"]), (readout["left"], readout["cy"]),
           color=COUPLING, linewidth=2.0)

    # B — the topology is structured, but complement faithfulness has no sparse knee.
    curve = a12["curve_H"]
    sizes = np.asarray(sorted(int(size) for size in curve))
    values = np.asarray([curve[str(size)]["F_H"] for size in sizes])
    lows = np.asarray([curve[str(size)]["ci"][0] for size in sizes])
    highs = np.asarray([curve[str(size)]["ci"][1] for size in sizes])
    faith.fill_between(sizes, lows, highs, color=COUPLING, alpha=0.12, linewidth=0)
    faith.plot(sizes, values, color=COUPLING, marker="o", markersize=3.5)
    faith.axhline(0.9, color=MUT, linestyle="--", linewidth=1.0)
    faith.axvline(72, color=MUT, linestyle=":", linewidth=1.0)
    faith.scatter([72], [curve["72"]["F_H"]], s=36, color=COUPLING,
                  edgecolor="white", linewidth=0.7, zorder=4)
    faith.text(
        69,
        0.965,
        "90% region\n64–72 nodes",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=SEC,
    )
    faith.set_xlim(0, 106)
    faith.set_ylim(-0.04, 1.12)
    faith.set_xlabel("Retained nodes (of 103)")
    faith.set_ylabel("Complement faithfulness")
    faith.spines[["top", "right"]].set_visible(False)
    faith.grid(axis="y", color=RULE, linewidth=0.65)
    faith.grid(axis="x", visible=False)
    panel_header(faith, "B", "Circuit Faithfulness")

    save_figure(
        fig,
        OUT,
        sources=[A10, A12, A3, CONFIRMATION, DONOR_ANCHOR, READOUT_PATH, CARRIER_DOSE],
        formats=("png", "pdf"),
        dpi=450,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
