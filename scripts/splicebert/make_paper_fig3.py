#!/usr/bin/env python3
"""Render the Submission 39 supplementary figure (paper Fig. 3) from recorded artifacts.

Follows the published design (Helvetica Neue, blue/green series, purple accents) with
its corrections applied: no Psi_C bracket at a single shift, "reconstruction" rather than
"binding" for the address x payload factorial (R23), and the panel-A curve labelled as the
period-3 cosine through the measured means. Because a period-3 sequence sampled at integer
shifts has exactly one harmonic, that cosine passes through every measured mean; values
between integer shifts are interpolation, not measurements.

    python scripts/splicebert/make_paper_fig3.py --out PAPER/fig_3.png

Writes PNG (600 dpi), vector PDF and a provenance sidecar. No model is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402

SOURCES = {
    "phase": ("results/phase-lattice/p3_phase_surgery_confirmation_v1/confirmation_analysis.json",
              "77959c53c9dd33d4ae3a591b19ba7e8c581a5f327080a1a53218056be82746e9"),
    "factorial": ("results/splicebert_coupling/a2_address_payload.json",
                  "40f992cd5328b05ea3615cd1c4c1c73e67f82ff404b56d999392b86da7cf5042"),
    "faithfulness": ("results/splicebert_coupling/a12_faithfulness_size.json",
                     "6023774734367d79e9ecca60e756471dc68f53cd32c242f488c3fc5a7df127bc"),
    "spectrum": ("results/phase-lattice/fig_phase_data.json",
                 "91d7f51e113fecd7d8972d6b3a1f87278de9c3efe22fbc600f82e9584ebe9316"),
}

BLUE = "#0B6FB8"
PURPLE = "#6A4A9C"
GREEN = "#00845F"
SHADE = "#E1E7EC"
INK = "#16232E"
MUTED = "#5B6670"
GRID = "#E6E9ED"
BAR_GREY = "#9AA3AB"
DIVIDER = "#E3E5EA"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> dict:
    out = {}
    for key, (relative, expected) in SOURCES.items():
        path = REPO / relative
        if sha256(path) != expected:
            raise ValueError(f"source identity changed; review before rendering: {relative}")
        out[key] = json.loads(path.read_text())
    return out


def period3_cosine(shifts: np.ndarray, values: np.ndarray):
    """Least-squares c + a*cos(2*pi*s/3) + b*sin(2*pi*s/3); exact for a period-3 sequence."""
    w = 2 * np.pi / 3
    design = np.column_stack([np.ones_like(shifts), np.cos(w * shifts), np.sin(w * shifts)])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = float(np.abs(design @ coef - values).max())
    return (lambda s: coef[0] + coef[1] * np.cos(w * s) + coef[2] * np.sin(w * s)), residual


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#AEB5BC")
    ax.tick_params(colors=MUTED, labelsize=10.5, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def render(data: dict):
    plt.rcParams.update({
        "font.family": "Helvetica Neue", "font.size": 11, "text.color": INK,
        "axes.labelcolor": MUTED, "axes.labelsize": 11.5, "svg.hashsalt": "gfmcd-paper-fig3",
        "pdf.fonttype": 42,
    })
    # Same aspect and panel geometry as the published figure (6312 x 3424 px).
    fig = plt.figure(figsize=(13, 7.05))
    axes = [[fig.add_axes([0.075, 0.687, 0.397, 0.202]), fig.add_axes([0.64, 0.592, 0.1985, 0.219])],
            [fig.add_axes([0.075, 0.180, 0.397, 0.190]), fig.add_axes([0.5875, 0.165, 0.3825, 0.205])]]
    for x0, y0, title in ((0.008, 0.985, "A  Register Dependence"),
                          (0.508, 0.985, "B  Address × Payload Reconstruction"),
                          (0.008, 0.487, "C  Faithfulness"),
                          (0.508, 0.487, "D  Period-3 Singular-Energy Spectrum")):
        fig.text(x0, y0, title, fontsize=17, fontweight="bold", color=INK, va="top")
    fig.add_artist(Line2D([0.5, 0.5], [0.515, 0.985], color=DIVIDER, linewidth=1))
    fig.add_artist(Line2D([0.5, 0.5], [0.015, 0.485], color=DIVIDER, linewidth=1))
    fig.add_artist(Line2D([0.008, 0.992], [0.5, 0.5], color=DIVIDER, linewidth=1))
    values = {}

    # A: measured means +-1 gene SE; the curve is the exact period-3 cosine through them.
    ax = axes[0][0]
    style(ax)
    primary = data["phase"]["primary"]
    curve = primary["figure_curve"]
    shifts = np.array(sorted(map(int, curve["cds"])))
    dense = np.linspace(shifts.min(), shifts.max(), 400)
    for shift in shifts:
        if curve["cds"][str(shift)]["register"] == "in":
            ax.axvspan(shift - 0.16, shift + 0.16, color=SHADE, zorder=0)
    for key, color, marker, line, face in (("upstream_intron", GREEN, "s", "--", "white"),
                                           ("cds", BLUE, "o", "-", BLUE)):
        means = np.array([curve[key][str(s)]["mean_coupling"] for s in shifts])
        ses = np.array([curve[key][str(s)]["se_across_genes"] for s in shifts])
        fit, residual = period3_cosine(shifts.astype(float), means)
        values[f"A_{key}_cosine_max_residual"] = residual
        ax.plot(dense, fit(dense), line, color=color, linewidth=2.1, zorder=2)
        ax.errorbar(shifts, means, yerr=ses, fmt="none", ecolor=color, elinewidth=1.1,
                    capsize=0, zorder=3)
        ax.plot(shifts, means, marker, color=color, markerfacecolor=face, markersize=5.5,
                markeredgewidth=1.3, linestyle="none", zorder=4)
    ax.set(xlim=(2, 10), ylim=(0.55, 0.80), xticks=shifts, yticks=np.arange(0.55, 0.801, 0.05),
           xlabel="Local shift (nt)", ylabel="Coupling (logit units)")
    ax.yaxis.set_major_formatter("{x:.2f}")
    for label in ax.get_xticklabels():
        if int(label.get_text()) % 3 == 0:
            label.set_color(PURPLE)
            label.set_fontweight("bold")
    lo, hi = primary["ci"]
    ax.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", markersize=5, linewidth=2, label="CDS block"),
        Line2D([], [], color=GREEN, marker="s", markerfacecolor="white", markersize=5,
               linestyle="--", linewidth=2, label="Upstream intron"),
    ], loc="center", bbox_to_anchor=(0.2735, 0.594), bbox_transform=fig.transFigure, ncol=2,
        frameon=False, fontsize=11.5, handlelength=2.4, columnspacing=3)
    for text, color in zip(ax.get_legend().get_texts(), (BLUE, GREEN)):
        text.set_color(color)
        text.set_fontweight("bold")
    fig.text(0.2735, 0.556, rf"$\Psi_{{\mathrm{{C}}}}$ = {primary['mean']:.3f} [{lo:.3f}, {hi:.3f}]",
             ha="center", va="center", color=PURPLE, fontsize=11.5, fontweight="bold")

    # B: normalized recovery with recorded 95% gene-bootstrap CIs.
    ax = axes[0][1]
    factorial = data["factorial"]
    cells = {(0, 0): "baseline", (0, 1): "payload_only", (1, 0): "address_only", (1, 1): "joint"}
    top = max(factorial["recoveries"][k]["recovery"] for k in cells.values())
    pale = np.array([241, 245, 249]) / 255
    blue = np.array(matplotlib.colors.to_rgb(BLUE))
    for (row, col), key in cells.items():
        record = factorial["recoveries"][key]
        value = record["recovery"]
        joint = key == "joint"
        fill = blue if joint else pale + (np.array([205, 225, 242]) / 255 - pale) * value / 0.06
        ax.add_patch(Rectangle((col + 0.03, row + 0.03), 0.94, 0.94, facecolor=fill,
                               edgecolor=PURPLE if joint else "#D5DCE3",
                               linewidth=2.6 if joint else 1.0))
        ink = "white" if joint else INK
        ax.text(col + 0.5, row + 0.43, f"{value:.3f}", ha="center", va="center", color=ink,
                fontsize=24, fontweight="bold")
        note = "reference" if key == "baseline" else f"[{record['ci'][0]:.3f}, {record['ci'][1]:.3f}]"
        ax.text(col + 0.5, row + 0.74, note.replace("-", "−"), ha="center", va="center",
                color="#DCE8F4" if joint else MUTED, fontsize=11)
    ax.set(xlim=(0, 2), ylim=(2, 0))
    ax.axis("off")
    ax.text(1, -0.40, "Payload — L3H1 values", ha="center", va="center", fontsize=12.5,
            fontweight="bold", color=INK)
    for col, name in enumerate(("Corrupt", "Clean")):
        ax.text(col + 0.5, -0.15, name, ha="center", va="center", fontsize=11.5, color=MUTED)
    for row, name in enumerate(("Attention\ncorrupt", "Attention\nclean")):
        ax.text(-0.08, row + 0.5, name, ha="right", va="center", fontsize=11.5, color=MUTED)
    ax.text(-0.96, 1, "Address — attention", rotation=90, ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=INK)
    ax.text(2.12, 1.0, f"Interaction\n{factorial['interaction_AxV']:+.3f}", ha="left",
            va="center", color=PURPLE, fontsize=15, fontweight="bold", linespacing=1.15)

    # C: all measured sizes, recorded CIs; 72 is the first measured size at or above 90%.
    ax = axes[1][0]
    style(ax)
    points = data["faithfulness"]["curve_H"]
    sizes = np.array(sorted(map(int, points)))
    means = np.array([points[str(s)]["F_H"] for s in sizes])
    bands = np.array([points[str(s)]["ci"] for s in sizes])
    first = int(sizes[np.argmax(means >= 0.9)])
    values["C_first_size_at_or_above_90"] = first
    ax.fill_between(sizes, bands[:, 0], bands[:, 1], color=BLUE, alpha=0.16, linewidth=0)
    ax.plot(sizes, means, "-", color=BLUE, linewidth=2.1)
    ax.plot(sizes, means, "o", color=BLUE, markerfacecolor="white", markersize=5,
            markeredgewidth=1.3)
    ax.axhline(0.9, color=PURPLE, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.axvline(first, color=PURPLE, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.7)
    ax.plot(first, points[str(first)]["F_H"], "o", color=PURPLE, markersize=8, zorder=5)
    ax.text(103, 0.86, "90%", ha="right", va="top", color=PURPLE, fontsize=12,
            fontweight="bold")
    ax.set(xlim=(0, 106), ylim=(-0.03, 1.13), xticks=[0, 20, 40, 60, first, 90, 103],
           yticks=np.arange(0, 1.01, 0.25),
           xlabel="Nodes retained, by fixed held-out importance rank", ylabel="Faithfulness")
    ax.yaxis.set_major_formatter("{x:.2f}")
    for label in ax.get_xticklabels():
        if label.get_text() == str(first):
            label.set_color(PURPLE)
            label.set_fontweight("bold")
    ax.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", markerfacecolor="white", linestyle="none",
               markersize=5.5, label=f"{len(sizes)} measured circuit sizes"),
        Patch(facecolor=BLUE, alpha=0.16, label="Gene-bootstrap 95% CI"),
    ], loc="center", bbox_to_anchor=(0.2735, 0.05), bbox_transform=fig.transFigure, ncol=2,
        frameon=False, fontsize=11.5, columnspacing=3)
    for text, color in zip(ax.get_legend().get_texts(), (BLUE, MUTED)):
        text.set_color(color)
        text.set_fontweight("bold")

    # D: energy shares of the band-limited period-3 component (bins 339-345; rank <= 14).
    ax = axes[1][1]
    style(ax)
    share = np.array(data["spectrum"]["svd"])
    components = np.arange(1, len(share) + 1)
    ax.bar(components, share, width=0.6,
           color=[BLUE if c <= 2 else BAR_GREY for c in components])
    pair = share[0] + share[1]
    values["D_leading_pair_share"] = float(pair)
    ax.plot([0.7, 0.7, 2.3, 2.3], [0.525, 0.545, 0.545, 0.525], color=BLUE, linewidth=1.4,
            clip_on=False)
    backing = {"facecolor": "white", "edgecolor": "none", "pad": 1.5}
    ax.text(3.2, 0.505, f"Leading pair = {100 * pair:.1f}% of period-3 energy", color=BLUE,
            fontsize=12.5, fontweight="bold", va="center", bbox=backing)
    ax.text(3.2, 0.425, f"{share[0]:.3f} ≈ {share[1]:.3f}: a cosine/sine pair", color=MUTED,
            fontsize=11.5, va="center", bbox=backing)
    ax.set(xlim=(0.4, len(share) + 0.6), ylim=(0, 0.55), xticks=components,
           yticks=np.arange(0, 0.51, 0.1),
           xlabel="Singular component of the period-3 carrier", ylabel="Share of energy")
    for label in ax.get_xticklabels()[:2]:
        label.set_color(BLUE)
        label.set_fontweight("bold")
    return fig, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="PNG path; PDF written beside it")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = [args.out.with_suffix(ext) for ext in (".png", ".pdf", ".provenance.json")]
    for path in outputs:
        guard_output_path(path, overwrite=args.overwrite)
    fig, values = render(load())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[0], dpi=args.dpi, facecolor="white")
    fig.savefig(outputs[1], facecolor="white", metadata={"CreationDate": None})
    plt.close(fig)

    sources = [{"path": relative, "sha256": sha256(REPO / relative)} for relative, _ in SOURCES.values()]
    sources.append({"path": "scripts/splicebert/make_paper_fig3.py", "sha256": sha256(Path(__file__))})
    run = lambda *cmd: subprocess.run(cmd, cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa: E731
    provenance = {
        "git_commit": run("git", "rev-parse", "HEAD"),
        "git_dirty": bool(run("git", "status", "--porcelain")),
        "command": ["python", *sys.argv],
        "sources": sources,
        "environment": {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
                        "numpy": np.__version__},
        "rendering": {
            "A": "Measured means ±1 gene SE. Curves are the least-squares period-3 cosine through the "
                 "means, exact for a period-3 sequence; between integer shifts it is interpolation.",
            "B": "Normalized recovery with recorded 95% gene-bootstrap CIs; interaction has no recorded CI.",
            "C": "All measured sizes with recorded CIs; the marked size is the first measured at or above 90%.",
            "D": "Energy shares S^2/sum(S^2) of the SVD of the band-limited (bins 339-345) period-3 "
                 "component; only the first ten are recorded.",
        },
        "values": values,
    }
    outputs[2].write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
