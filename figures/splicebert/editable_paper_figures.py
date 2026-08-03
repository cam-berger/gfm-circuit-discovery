"""Publication-width renderers used by ``notebooks/edit_neurips_paper_figures.ipynb``.

The notebook owns the editable configuration.  This module keeps the rendering,
artifact validation, provenance, QA, and safe-promotion machinery in importable
functions so that a notebook restart cannot leave part of a renderer undefined.

No scientific quantity is recomputed here.  Both figures read the immutable
artifacts used by the paper and refuse to render if the recorded instrument gates
or source hashes fail.  Appearance and layout remain deliberately editable.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from figstyle import (  # noqa: E402
    COUPLING,
    GREEN,
    INK,
    MUT,
    PANEL,
    PERIOD3,
    RULE,
    SEC,
)


CORE_PROVENANCE = HERE / "fig_full_circuit.provenance.json"
PHASE_PROVENANCE = HERE / "fig5_p3_phase_surgery_confirmation.provenance.json"

CORE_A10 = REPO / "results/splicebert_coupling/a10_generator_edges.json"
CORE_A12 = REPO / "results/splicebert_coupling/a12_faithfulness_size.json"

PHASE_ANALYSIS = (
    REPO
    / "results/phase-lattice/p3_phase_surgery_confirmation_v1/confirmation_analysis.json"
)
PHASE_PACKAGE = (
    REPO / "results/phase-lattice/p3_phase_surgery_confirmation_package_v1.json"
)

DEFAULT_EXPORT_DIR = HERE / "workshop_edits"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo_relative(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _resolve_recorded_source(recorded: str) -> Path:
    """Resolve an absolute provenance path after a repository move when possible."""
    path = Path(recorded)
    if path.exists():
        return path
    for anchor in ("results", "studies", "data", "figures"):
        if anchor in path.parts:
            suffix = Path(*path.parts[path.parts.index(anchor) :])
            candidate = REPO / suffix
            if candidate.exists():
                return candidate
    return path


def verify_recorded_sources(provenance_path: str | Path) -> list[Path]:
    """Verify every input hash in an existing figure-provenance record."""
    provenance_path = Path(provenance_path)
    record = json.loads(provenance_path.read_text())
    sources: list[Path] = []
    failures: list[str] = []
    for item in record.get("sources", []):
        path = _resolve_recorded_source(item["path"])
        sources.append(path)
        if not path.exists():
            failures.append(f"missing: {path}")
            continue
        observed = sha256_file(path)
        if observed != item["sha256"]:
            failures.append(
                f"hash mismatch: {_repo_relative(path)}\n"
                f"  recorded {item['sha256']}\n  observed {observed}"
            )
    if failures:
        joined = "\n".join(failures)
        raise RuntimeError(f"refusing to draw from changed sources:\n{joined}")
    if not sources:
        raise RuntimeError(f"no sources recorded in {provenance_path}")
    return sources


def load_core_data() -> dict[str, Any]:
    """Load the structured-core topology gate and held-out faithfulness curve."""
    sources = verify_recorded_sources(CORE_PROVENANCE)
    a10 = json.loads(CORE_A10.read_text())
    a12 = json.loads(CORE_A12.read_text())
    if not (
        a10.get("confirmatory")
        and a10.get("preflight_PASS")
        and a10.get("serial_chain_confirmed")
    ):
        raise RuntimeError("structured-core topology is not qualified for rendering")
    if not (a12.get("confirmatory") and a12.get("preflight_PASS")):
        raise RuntimeError("faithfulness curve is not qualified for rendering")

    curve = a12["curve_H"]
    sizes = np.asarray(sorted(int(size) for size in curve))
    return {
        "sizes": sizes,
        "faithfulness": np.asarray([curve[str(size)]["F_H"] for size in sizes]),
        "ci_low": np.asarray([curve[str(size)]["ci"][0] for size in sizes]),
        "ci_high": np.asarray([curve[str(size)]["ci"][1] for size in sizes]),
        "crossing_90": {"lower": 64, "upper": 72, "sampled": curve["72"]["F_H"]},
        "n_nodes": 103,
        "sources": sources,
    }


def load_phase_data() -> dict[str, Any]:
    """Load the independently adjudicated period-3 surgery package."""
    sources = verify_recorded_sources(PHASE_PROVENANCE)
    analysis = json.loads(PHASE_ANALYSIS.read_text())
    if analysis.get("schema") != "gfmcd.p3_phase_surgery.confirmation_analysis.v1":
        raise RuntimeError("unexpected period-3 confirmation schema")
    if analysis.get("status") != "CONFIRMATORY_ADJUDICATED":
        raise RuntimeError("period-3 analysis has not been independently adjudicated")
    if not analysis.get("all_instrument_gates_pass"):
        raise RuntimeError("period-3 instrument gates failed")
    if analysis.get("package_sha256") != sha256_file(PHASE_PACKAGE):
        raise RuntimeError("period-3 analysis/package hash mismatch")

    primary = analysis["primary"]
    direction = analysis["direction_specificity"]
    context = analysis["matched_context"]
    curve = primary["figure_curve"]
    amounts = list(range(2, 11))
    return {
        "amounts": np.asarray(amounts),
        "curve": {
            name: {
                "mean": np.asarray(
                    [curve[name][str(amount)]["mean_coupling"] for amount in amounts]
                ),
                "se": np.asarray(
                    [curve[name][str(amount)]["se_across_genes"] for amount in amounts]
                ),
            }
            for name in ("cds", "upstream_intron")
        },
        "primary": {
            "mean": primary["mean"],
            "ci": primary["ci"],
            "sesoi": 0.05,
            "n_genes": primary["n_genes"],
            "mde80": primary["mde80"],
        },
        "direction": {
            "p3_mean": direction["p3_mean"],
            "controls": np.asarray(list(direction["control_means"].values())),
            "paired_excess_mean": direction["p3_minus_mean_control"]["mean"],
            "paired_excess_ci": direction["p3_minus_mean_control"]["ci"],
            "rank_p": direction["add_one_rank_p_one_sided"],
            "n_genes": direction["n_genes"],
        },
        "context": {
            "differences": np.asarray(
                [row["cds_psi"] - row["non_cds_psi"] for row in context["per_pair"]]
            ),
            "delta_mean": context["cds_minus_non_cds"]["mean"],
            "delta_ci": context["cds_minus_non_cds"]["ci"],
            "n_pairs": context["n_pairs"],
            "n_same_gene_pairs": context["n_same_gene_pairs"],
        },
        "sources": sources,
    }


def default_style() -> dict[str, Any]:
    """Return a fresh, notebook-editable style dictionary."""
    return {
        "font_family": "sans-serif",
        "font_sans": ["Arial", "Helvetica", "DejaVu Sans"],
        "base_font_pt": 7.2,
        "title_font_pt": 8.0,
        "panel_letter_pt": 8.4,
        "annotation_font_pt": 6.4,
        "node_title_pt": 7.1,
        "node_subtitle_pt": 6.2,
        "axis_linewidth": 0.65,
        "data_linewidth": 1.35,
        "marker_size": 3.8,
        "grid_linewidth": 0.45,
        "semantic": {
            "payload": COUPLING,
            "coordinate": PERIOD3,
            "coupling": COUPLING,
            "intron": GREEN,
            "neutral": MUT,
            "ink": INK,
            "secondary": SEC,
            "rule": RULE,
            "panel": PANEL,
            "white": "#FFFFFF",
            "payload_fill": "#F5F0F8",
            "coordinate_fill": "#EEF5FA",
            "neutral_fill": "#F5F6F7",
        },
    }


def default_core_layout() -> dict[str, Any]:
    """Return a one-column, vertically reflowed circuit layout."""
    return {
        "figsize": (3.25, 4.65),
        "height_ratios": (2.45, 1.45),
        "panel_gap": 0.08,
        "nodes": [
            {
                "id": "payload",
                "xy": (0.25, 0.86),
                "wh": (0.43, 0.18),
                "title": "PAYLOAD",
                "subtitle": "acceptor-local sequence\nL0H7/H12/H10 → MLP0 → MLP1",
                "role": "payload",
            },
            {
                "id": "address",
                "xy": (0.75, 0.86),
                "wh": (0.43, 0.18),
                "title": "ADDRESS",
                "subtitle": "period-3 coordinate\n→ L2H15",
                "role": "coordinate",
            },
            {
                "id": "mlp2",
                "xy": (0.50, 0.64),
                "wh": (0.34, 0.14),
                "title": "MLP2",
                "subtitle": "convergence",
                "role": "payload",
            },
            {
                "id": "l3h1",
                "xy": (0.50, 0.46),
                "wh": (0.34, 0.14),
                "title": "L3H1",
                "subtitle": "binds + writes",
                "role": "payload",
                "emphasis": True,
            },
            {
                "id": "uinfo",
                "xy": (0.50, 0.285),
                "wh": (0.34, 0.14),
                "title": r"$U_{\mathrm{info}}$",
                "subtitle": "rank-8 donor state",
                "role": "payload",
            },
            {
                "id": "readout",
                "xy": (0.50, 0.105),
                "wh": (0.72, 0.14),
                "title": "L4H0 / L4H2 / L5H11 → MLM",
                "subtitle": "transport to masked-donor margin",
                "role": "neutral",
            },
        ],
        "edges": [
            {"from": "payload", "to": "mlp2", "role": "payload", "rad": -0.08},
            {"from": "address", "to": "mlp2", "role": "coordinate", "rad": 0.08},
            {"from": "mlp2", "to": "l3h1", "role": "payload"},
            {"from": "l3h1", "to": "uinfo", "role": "payload"},
            {"from": "uinfo", "to": "readout", "role": "payload"},
        ],
        "faithfulness_xlim": (0, 104),
        "faithfulness_ylim": (-0.03, 1.06),
        "faithfulness_xticks": (0, 25, 50, 75, 100),
    }


def default_phase_layout() -> dict[str, Any]:
    """Return a compact 2×2 layout at the actual NeurIPS column width."""
    return {
        "figsize": (3.25, 4.25),
        "titles": {
            "A": "Register",
            "B": "Primary effect",
            "C": "Controls",
            "D": "Coding boundary",
        },
        "primary_xlim": (-0.065, 0.205),
        "controls_ylim": (-0.018, 0.165),
        "context_xlim": (-0.42, 0.44),
        "line_label_mode": "direct",
        "legend_location": "lower right",
        "register_offsets": (3, 6, 9),
    }


def _rc_params(style: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "font.family": style["font_family"],
        "font.sans-serif": style["font_sans"],
        "font.size": style["base_font_pt"],
        "axes.titlesize": style["title_font_pt"],
        "axes.labelsize": style["base_font_pt"],
        "axes.linewidth": style["axis_linewidth"],
        "xtick.labelsize": style["base_font_pt"] - 0.4,
        "ytick.labelsize": style["base_font_pt"] - 0.4,
        "xtick.major.width": style["axis_linewidth"],
        "ytick.major.width": style["axis_linewidth"],
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.fontsize": style["annotation_font_pt"],
        "lines.linewidth": style["data_linewidth"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }


def palette_preview(style: Mapping[str, Any] | None = None):
    """Draw the semantic palette used by both figures."""
    style = deepcopy(style or default_style())
    semantic = style["semantic"]
    roles = ("payload", "coordinate", "intron", "neutral")
    with mpl.rc_context(_rc_params(style)):
        fig, ax = plt.subplots(figsize=(3.25, 0.55), constrained_layout=True)
        ax.set_xlim(0, len(roles))
        ax.set_ylim(0, 1)
        ax.axis("off")
        for index, role in enumerate(roles):
            ax.add_patch(
                FancyBboxPatch(
                    (index + 0.06, 0.17),
                    0.88,
                    0.60,
                    boxstyle="round,pad=0.01,rounding_size=0.04",
                    facecolor=semantic[role],
                    edgecolor="none",
                )
            )
            text_color = "white" if role != "neutral" else semantic["ink"]
            ax.text(
                index + 0.5,
                0.47,
                role,
                ha="center",
                va="center",
                color=text_color,
                fontsize=style["annotation_font_pt"],
                fontweight="bold",
            )
    return fig


def _panel_title(ax, letter: str, title: str, style: Mapping[str, Any]) -> None:
    ax.set_title(
        rf"$\bf{{{letter}}}$  {title}",
        loc="left",
        fontsize=style["title_font_pt"],
        color=style["semantic"]["ink"],
        pad=5,
    )


def _open_axis(ax, style: Mapping[str, Any], *, grid_axis: str = "y") -> None:
    semantic = style["semantic"]
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(
        axis=grid_axis,
        color=semantic["rule"],
        linewidth=style["grid_linewidth"],
        zorder=0,
    )
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    ax.tick_params(colors=semantic["ink"], pad=1.5)
    ax.xaxis.label.set_color(semantic["ink"])
    ax.yaxis.label.set_color(semantic["ink"])


def _node_colors(node: Mapping[str, Any], style: Mapping[str, Any]) -> tuple[str, str]:
    semantic = style["semantic"]
    role = node["role"]
    edge = semantic[role]
    fill = semantic.get(f"{role}_fill", semantic["white"])
    return edge, fill


def _draw_node(ax, node: Mapping[str, Any], style: Mapping[str, Any]) -> None:
    x, y = node["xy"]
    width, height = node["wh"]
    edge, fill = _node_colors(node, style)
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        transform=ax.transAxes,
        boxstyle="round,pad=0.006,rounding_size=0.018",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.25 if node.get("emphasis") else 0.85,
        clip_on=False,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y + height * 0.19,
        node["title"],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=style["node_title_pt"],
        fontweight="bold",
        color=style["semantic"]["ink"],
        zorder=4,
    )
    ax.text(
        x,
        y - height * 0.19,
        node["subtitle"],
        transform=ax.transAxes,
        ha="center",
        va="center",
        multialignment="center",
        linespacing=1.15,
        fontsize=style["node_subtitle_pt"],
        color=style["semantic"]["secondary"],
        zorder=4,
    )


def _box_anchor(source: Mapping[str, Any], target: Mapping[str, Any]) -> tuple[float, float]:
    """Intersect the source box with the ray from source to target."""
    sx, sy = source["xy"]
    tx, ty = target["xy"]
    dx, dy = tx - sx, ty - sy
    half_w, half_h = source["wh"][0] / 2, source["wh"][1] / 2
    candidates = []
    if dx:
        candidates.append(half_w / abs(dx))
    if dy:
        candidates.append(half_h / abs(dy))
    scale = min(candidates) if candidates else 0.0
    return sx + dx * scale, sy + dy * scale


def _draw_edge(
    ax,
    edge: Mapping[str, Any],
    nodes: Mapping[str, Mapping[str, Any]],
    style: Mapping[str, Any],
) -> None:
    source, target = nodes[edge["from"]], nodes[edge["to"]]
    start = _box_anchor(source, target)
    end = _box_anchor(target, source)
    patch = FancyArrowPatch(
        start,
        end,
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=1.15 if edge.get("emphasis") else 0.95,
        color=style["semantic"][edge["role"]],
        connectionstyle=f"arc3,rad={edge.get('rad', 0.0)}",
        shrinkA=2,
        shrinkB=3,
        clip_on=False,
        zorder=2,
    )
    ax.add_patch(patch)


def render_core_figure(
    data: Mapping[str, Any],
    *,
    style: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
):
    """Render the structured core at its final one-column physical size."""
    style = deepcopy(style or default_style())
    layout = deepcopy(layout or default_core_layout())
    semantic = style["semantic"]

    with mpl.rc_context(_rc_params(style)):
        fig = plt.figure(figsize=layout["figsize"], constrained_layout=True)
        grid = fig.add_gridspec(
            2,
            1,
            height_ratios=layout["height_ratios"],
            hspace=layout["panel_gap"],
        )
        circuit = fig.add_subplot(grid[0])
        faith = fig.add_subplot(grid[1])

        circuit.set_xlim(0, 1)
        circuit.set_ylim(0, 1)
        circuit.axis("off")
        _panel_title(circuit, "A", "Structured causal core", style)
        nodes = {node["id"]: node for node in layout["nodes"]}
        for edge in layout["edges"]:
            _draw_edge(circuit, edge, nodes, style)
        for node in layout["nodes"]:
            _draw_node(circuit, node, style)

        x = np.asarray(data["sizes"])
        faith.fill_between(
            x,
            data["ci_low"],
            data["ci_high"],
            color=semantic["coupling"],
            alpha=0.13,
            linewidth=0,
            zorder=1,
        )
        faith.plot(
            x,
            data["faithfulness"],
            color=semantic["coupling"],
            marker="o",
            markersize=style["marker_size"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
        faith.axhline(0.9, color=semantic["neutral"], linestyle="--", linewidth=0.75)
        faith.axvline(72, color=semantic["neutral"], linestyle=":", linewidth=0.75)
        faith.scatter(
            [72],
            [data["crossing_90"]["sampled"]],
            s=26,
            color=semantic["coupling"],
            edgecolor="white",
            linewidth=0.55,
            zorder=4,
        )
        faith.annotate(
            "90% region\n64–72 nodes",
            xy=(72, data["crossing_90"]["sampled"]),
            xytext=(45, 1.015),
            textcoords="data",
            ha="left",
            va="top",
            fontsize=style["annotation_font_pt"],
            color=semantic["secondary"],
            arrowprops={
                "arrowstyle": "-",
                "color": semantic["neutral"],
                "linewidth": 0.55,
            },
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
        )
        faith.set_xlim(*layout["faithfulness_xlim"])
        faith.set_ylim(*layout["faithfulness_ylim"])
        faith.set_xticks(layout["faithfulness_xticks"])
        faith.set_xlabel(f"Retained nodes (of {data['n_nodes']})", labelpad=2)
        faith.set_ylabel("Complement\nfaithfulness", labelpad=2)
        _panel_title(faith, "B", "Distributed faithfulness boundary", style)
        _open_axis(faith, style)
    return fig


def render_phase_figure(
    data: Mapping[str, Any],
    *,
    style: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
):
    """Render the four-panel confirmation at one-column physical size."""
    style = deepcopy(style or default_style())
    layout = deepcopy(layout or default_phase_layout())
    semantic = style["semantic"]

    with mpl.rc_context(_rc_params(style)):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=layout["figsize"],
            constrained_layout=True,
        )

        ax = axes[0, 0]
        x = np.asarray(data["amounts"])
        curves_drawn: dict[str, np.ndarray] = {}
        for name, color, label, marker in (
            ("cds", semantic["coordinate"], "CDS", "o"),
            ("upstream_intron", semantic["intron"], "Upstream intron", "s"),
        ):
            mean = np.asarray(data["curve"][name]["mean"])
            se = np.asarray(data["curve"][name]["se"])
            curves_drawn[name] = mean
            ax.plot(
                x,
                mean,
                marker=marker,
                markersize=style["marker_size"],
                color=color,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=label,
                zorder=3,
            )
            ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.12, linewidth=0)
        for amount in layout["register_offsets"]:
            ax.axvspan(amount - 0.13, amount + 0.13, color=semantic["coordinate"], alpha=0.08)
        ax.set_xticks((2, 4, 6, 8, 10))
        ax.set_xlabel("Local shift (nt)", labelpad=2)
        ax.set_ylabel("Coupling\n(logit units)", labelpad=2)
        if layout.get("line_label_mode", "direct") == "legend":
            ax.legend(
                frameon=False,
                loc=layout["legend_location"],
                handlelength=1.0,
                handletextpad=0.35,
                borderaxespad=0.2,
            )
        else:
            for name, label, color, offset, vertical in (
                ("cds", "CDS", semantic["coordinate"], 0.006, "bottom"),
                ("upstream_intron", "intron", semantic["intron"], -0.005, "top"),
            ):
                ax.text(
                    x[-1] - 0.05,
                    curves_drawn[name][-1] + offset,
                    label,
                    ha="right",
                    va=vertical,
                    fontsize=style["annotation_font_pt"],
                    color=color,
                    fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.3},
                    zorder=5,
                )
        _panel_title(ax, "A", layout["titles"]["A"], style)
        _open_axis(ax, style)

        ax = axes[0, 1]
        primary = data["primary"]
        ax.axvspan(-primary["sesoi"], primary["sesoi"], color=semantic["rule"], alpha=0.55)
        ax.axvline(0, color=semantic["ink"], linewidth=0.7)
        ax.axvline(primary["sesoi"], color=semantic["neutral"], linestyle="--", linewidth=0.75)
        ax.errorbar(
            primary["mean"],
            0,
            xerr=[
                [primary["mean"] - primary["ci"][0]],
                [primary["ci"][1] - primary["mean"]],
            ],
            fmt="o",
            color=semantic["coordinate"],
            markersize=5.2,
            linewidth=1.45,
            capsize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=3,
        )
        ax.set_yticks([])
        ax.set_xlim(*layout["primary_xlim"])
        ax.set_ylim(-0.55, 0.55)
        ax.set_xlabel(r"CDS $-$ intron contrast", labelpad=2)
        ax.text(
            0.98,
            0.91,
            f"{primary['mean']:+.3f}\n[{primary['ci'][0]:.3f}, {primary['ci'][1]:.3f}]",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=style["annotation_font_pt"],
            color=semantic["coordinate"],
            fontweight="bold",
        )
        ax.text(
            0.03,
            0.08,
            "SESOI ±0.05",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=style["annotation_font_pt"],
            color=semantic["secondary"],
        )
        _panel_title(ax, "B", layout["titles"]["B"], style)
        _open_axis(ax, style, grid_axis="x")

        ax = axes[1, 0]
        controls = np.asarray(data["direction"]["controls"])
        jitter = np.linspace(-0.12, 0.12, len(controls))
        ax.scatter(
            jitter,
            controls,
            s=15,
            color=semantic["neutral"],
            alpha=0.8,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        ax.scatter(
            [1],
            [data["direction"]["p3_mean"]],
            s=34,
            color=semantic["coordinate"],
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        ax.axhline(0, color=semantic["ink"], linewidth=0.65)
        ax.set_xticks((0, 1), ("Controls\n(n=19)", "Learned\nperiod-3"))
        ax.set_xlim(-0.38, 1.38)
        ax.set_ylim(*layout["controls_ylim"])
        ax.set_ylabel(r"Phase contrast, $\Psi_C$", labelpad=2)
        ax.text(
            0.03,
            0.92,
            f"excess {data['direction']['paired_excess_mean']:+.3f}\n"
            f"[{data['direction']['paired_excess_ci'][0]:.3f}, "
            f"{data['direction']['paired_excess_ci'][1]:.3f}]; "
            f"$p={data['direction']['rank_p']:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=style["annotation_font_pt"],
            color=semantic["secondary"],
        )
        _panel_title(ax, "C", layout["titles"]["C"], style)
        _open_axis(ax, style)

        ax = axes[1, 1]
        differences = np.asarray(data["context"]["differences"])
        # The vertical coordinate is only deterministic jitter, not a measured axis.
        # Keep it compact so the estimate and the explicit boundary label have clear air.
        y = np.linspace(-0.08, 0.08, len(differences))
        ax.scatter(
            differences,
            y,
            s=13,
            color=semantic["neutral"],
            alpha=0.62,
            edgecolor="white",
            linewidth=0.25,
            zorder=2,
        )
        mean = data["context"]["delta_mean"]
        ci = data["context"]["delta_ci"]
        ax.errorbar(
            mean,
            0.29,
            xerr=[[mean - ci[0]], [ci[1] - mean]],
            fmt="o",
            color=semantic["coordinate"],
            markersize=4.8,
            linewidth=1.4,
            capsize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=4,
        )
        ax.axvline(0, color=semantic["ink"], linewidth=0.65)
        ax.set_yticks([])
        ax.set_xlim(*layout["context_xlim"])
        ax.set_ylim(-0.30, 0.46)
        ax.set_xlabel(r"CDS $-$ non-CDS contrast", labelpad=2)
        ax.text(
            0.98,
            0.04,
            f"{mean:+.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]\ninconclusive",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=style["annotation_font_pt"],
            color=semantic["secondary"],
        )
        _panel_title(ax, "D", layout["titles"]["D"], style)
        _open_axis(ax, style, grid_axis="x")
    return fig


def audit_figure(fig, *, name: str, minimum_text_pt: float = 6.0) -> dict[str, Any]:
    """Return and print a compact physical-size and legibility audit."""
    sizes = [
        float(text.get_fontsize())
        for text in fig.findobj(mpl.text.Text)
        if text.get_text().strip()
    ]
    width, height = (float(value) for value in fig.get_size_inches())
    below = sorted({round(size, 2) for size in sizes if size < minimum_text_pt})
    report = {
        "name": name,
        "width_in": width,
        "height_in": height,
        "minimum_text_pt": min(sizes) if sizes else None,
        "text_sizes_below_threshold": below,
        "target": "single-column" if width <= 3.5 else "double-column",
    }
    status = "PASS" if not below else "REVIEW"
    print(
        f"{status} {name}: {width:.2f} × {height:.2f} in; "
        f"minimum text {report['minimum_text_pt']:.1f} pt; target {report['target']}"
    )
    if below:
        print(f"  text below {minimum_text_pt:.1f} pt: {below}")
    return report


def export_figure(
    fig,
    *,
    name: str,
    sources: Iterable[str | Path],
    claim_boundary: str,
    style: Mapping[str, Any],
    layout: Mapping[str, Any],
    output_dir: str | Path = DEFAULT_EXPORT_DIR,
    dpi: int = 600,
) -> dict[str, Path]:
    """Write PDF, SVG, PNG, and a source-hashed provenance sidecar."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / name
    outputs: dict[str, Path] = {}
    for extension in ("pdf", "svg", "png"):
        path = base.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        outputs[extension] = path

    record = {
        "schema": "gfmcd.figure_edit_provenance.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "renderer": _repo_relative(__file__),
        "sources": [
            {"path": _repo_relative(path), "sha256": sha256_file(path)}
            for path in sources
        ],
        "outputs": [
            {"path": _repo_relative(path), "sha256": sha256_file(path)}
            for path in outputs.values()
        ],
        "claim_boundary": claim_boundary,
        "style": style,
        "layout": layout,
    }
    provenance_path = base.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    outputs["provenance"] = provenance_path
    print("Exported:")
    for path in outputs.values():
        print(f"  {_repo_relative(path)}")
    return outputs


def promote_to_manuscript(
    *,
    core_outputs: Mapping[str, Path],
    phase_outputs: Mapping[str, Path],
    confirmation: str,
) -> Path:
    """Back up and replace the paper figures after explicit notebook confirmation.

    The exact confirmation token prevents an accidental Run-All from overwriting the
    manuscript assets.  PDF, PNG, and provenance are backed up before replacement.
    """
    if confirmation != "PROMOTE REVIEWED FIGURES":
        raise RuntimeError(
            "promotion disabled; pass confirmation='PROMOTE REVIEWED FIGURES' only "
            "after visually reviewing the PDF and PNG exports"
        )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = REPO / "output/figure_backups" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=False)
    groups = (
        (core_outputs, HERE / "fig_full_circuit"),
        (phase_outputs, HERE / "fig5_p3_phase_surgery_confirmation"),
    )
    for outputs, target_base in groups:
        for key, suffix in (("pdf", ".pdf"), ("png", ".png"), ("provenance", ".provenance.json")):
            target = target_base.with_suffix(suffix)
            if target.exists():
                shutil.copy2(target, backup_dir / target.name)
            shutil.copy2(outputs[key], target)
    print(f"Promoted reviewed figures; previous assets backed up to {_repo_relative(backup_dir)}")
    return backup_dir


__all__ = [
    "DEFAULT_EXPORT_DIR",
    "PHASE_ANALYSIS",
    "PHASE_PACKAGE",
    "audit_figure",
    "default_core_layout",
    "default_phase_layout",
    "default_style",
    "export_figure",
    "load_core_data",
    "load_phase_data",
    "palette_preview",
    "promote_to_manuscript",
    "render_core_figure",
    "render_phase_figure",
    "sha256_file",
    "verify_recorded_sources",
]
