#!/usr/bin/env python3
"""Render the four-panel mechanism figure from recorded aggregate artifacts.

No model, genome, or new statistical analysis is used. The only derived scientific
number is the linear 90% crossing of C33's recorded means. Error bars in panel A
are recorded gene standard errors; B/C intervals are recorded gene-bootstrap CIs.
Panel D shows the two singular values of a single Fourier bin, whose rank is at
most two by construction. The source does not contain a ten-component spectrum.

    uv run --locked python scripts/splicebert/make_mechanism_figure.py --overwrite

Writes figures/fig_3.{png,svg,provenance.json}; existing outputs require --overwrite.
Scientific status and provenance limitations remain controlled by the claim ledger.
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
from matplotlib.patches import Rectangle
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.figures.io import figure_provenance  # noqa: E402
from gfmcd.figures.style import COLORS  # noqa: E402

SOURCES = (
    ("results/phase-lattice/p3_phase_surgery_confirmation_v1/confirmation_analysis.json",
     "77959c53c9dd33d4ae3a591b19ba7e8c581a5f327080a1a53218056be82746e9"),
    ("results/splicebert_coupling/a2_address_payload.json",
     "40f992cd5328b05ea3615cd1c4c1c73e67f82ff404b56d999392b86da7cf5042"),
    ("results/splicebert_coupling/a12_faithfulness_size.json",
     "6023774734367d79e9ecca60e756471dc68f53cd32c242f488c3fc5a7df127bc"),
    ("results/phase-lattice/fig_report_data.json",
     "3bca5b20be7c51a158aa3214a288bf49f1bdd76636f412f6728e9ea9bd835057"),
)


def load_sources() -> list[dict]:
    """Refuse changed inputs rather than silently renewing the figure's claims."""
    documents = []
    for relative, expected in SOURCES:
        raw = (REPO / relative).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Source identity changed; review the claim before rendering: {relative}")
        documents.append(json.loads(raw))
    return documents


def first_crossing(curve: dict, threshold: float) -> tuple[float, int, int]:
    """Return the first linear crossing and its two measured node counts."""
    sizes = sorted(map(int, curve))
    for left, right in zip(sizes, sizes[1:]):
        low, high = curve[str(left)]["F_H"], curve[str(right)]["F_H"]
        if low < threshold <= high:
            return left + (threshold - low) * (right - left) / (high - low), left, right
    raise ValueError(f"No measured interval crosses faithfulness {threshold}")


def _caption(fig, ax, text: str) -> None:
    box = ax.get_position()
    fig.text(box.x0, box.y0 - 0.065, text, va="top", ha="left",
             fontsize=9.1, color="#46525B", linespacing=1.5)


def render(documents: list[dict]):
    phase, factorial, faithfulness, report = documents
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": COLORS["text"], "axes.labelcolor": COLORS["text"],
        "axes.edgecolor": "#AAB2B8", "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True,
        "axes.grid": False, "xtick.color": "#46525B", "ytick.color": "#46525B",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.transparent": False,
        "svg.hashsalt": "gfmcd-mechanism-figure",
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.4))
    fig.subplots_adjust(left=0.080, right=0.972, top=0.855, bottom=0.225,
                        wspace=0.36, hspace=0.80)
    fig.text(0.080, 0.965, "Positional geometry and causal computation", fontsize=21,
             fontweight="bold", va="top")
    fig.text(0.080, 0.923, "SpliceBERT.1024nt  ·  Recorded model-internal evidence", fontsize=11,
             color="#596872", va="top")
    titles = ("A  Phase-aligned use of the period-3 carrier",
              "B  L3H1 address–payload binding",
              "C  Static-ranking faithfulness",
              "D  Period-3 Fourier plane")
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=20)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)

    # A: the source records one SE per point, not pointwise confidence intervals.
    ax = axes[0, 0]
    curve = phase["primary"]["figure_curve"]
    shifts = np.array(sorted(map(int, curve["cds"])))
    for shift in shifts:
        if curve["cds"][str(shift)]["register"] == "in":
            ax.axvspan(shift - 0.14, shift + 0.14, color="#EEF1F4", zorder=0)
    for key, label, color, marker, style in (
        ("upstream_intron", "Upstream intron", COLORS["accent"], "s", "--"),
        ("cds", "CDS block", COLORS["signal"], "o", "-"),
    ):
        values = [curve[key][str(shift)] for shift in shifts]
        ax.errorbar(shifts, [v["mean_coupling"] for v in values],
                    yerr=[v["se_across_genes"] for v in values],
                    marker=marker, linestyle=style, color=color, linewidth=1.5,
                    markersize=4.5, markerfacecolor="white", capsize=2.5, label=label)
    ax.set(xlim=(1.8, 10.2), ylim=(0.55, 0.83), xticks=shifts,
           xlabel="Local positional-carrier shift (nt)", ylabel="Coupling (raw logit units)")
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=8.7)
    primary = phase["primary"]
    lo, hi = primary["ci"]
    _caption(fig, ax,
             rf"$\Psi_C$ = {primary['mean']:+.3f} [{lo:.3f}, {hi:.3f}] (95% CI); {primary['n_genes']} genes."
             "\nPoints ±1 gene SE; shaded shifts preserve carrier register."
             "\nSequence is fixed. Coding specificity remains unresolved.")

    # B: reconstruct–inject recovery is normalized to the clean/corrupt effect.
    ax = axes[0, 1]
    ax.grid(False)
    for (row, col), key in {
        (0, 0): "baseline", (0, 1): "payload_only",
        (1, 0): "address_only", (1, 1): "joint",
    }.items():
        record = factorial["recoveries"][key]
        value = record["recovery"]
        color = plt.colormaps["Blues"](0.12 + 0.73 * value / 0.4)
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=color, edgecolor="white", linewidth=3))
        ink = "white" if key == "joint" else COLORS["text"]
        ax.text(col + 0.5, row + 0.42, f"{value:.3f}", ha="center", va="center",
                color=ink, fontsize=23, fontweight="bold")
        interval = ("Reference" if key == "baseline" else
                    f"[{record['ci'][0]:.3f}, {record['ci'][1]:.3f}]")
        ax.text(col + 0.5, row + 0.70, interval, ha="center", va="center", color=ink, fontsize=10)
    ax.set(xlim=(0, 2), ylim=(2, 0), xticks=[0.5, 1.5], yticks=[0.5, 1.5])
    ax.set_xticklabels(["Corrupt values", "Clean values"])
    ax.set_yticklabels(["Corrupt\nattention", "Clean\nattention"])
    ax.set_xlabel("Payload: L3H1 values")
    ax.set_ylabel("Address: L3H1 attention")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    _caption(fig, ax,
             f"Fractional recovery (dimensionless), with 95% gene-bootstrap CIs."
             f"\nInteraction = {factorial['interaction_AxV']:+.3f}; its CI is not recorded."
             f"\n{factorial['n_genes']} validation genes; the zero reference is definitional.")

    # C: straight segments through all 17 recorded means and CI limits.
    ax = axes[1, 0]
    points = faithfulness["curve_H"]
    sizes = np.array(sorted(map(int, points)))
    values = np.array([points[str(size)]["F_H"] for size in sizes])
    intervals = np.array([points[str(size)]["ci"] for size in sizes])
    ax.fill_between(sizes, intervals[:, 0], intervals[:, 1], color=COLORS["signal"], alpha=0.14)
    ax.plot(sizes, values, "o-", color=COLORS["signal"], markersize=4, linewidth=1.8,
            markerfacecolor="white")
    crossing, left, right = first_crossing(points, 0.90)
    ax.axhline(0.9, color="#707B84", linestyle="--", linewidth=1)
    ax.axhline(1, color="#CCD2D7", linewidth=0.8)
    ax.plot(crossing, 0.9, "o", markersize=7, markerfacecolor="white",
            markeredgecolor=COLORS["ablated"], markeredgewidth=1.7)
    ax.annotate(f"90% at {crossing:.1f} nodes\n(linear interpolation)", xy=(crossing, 0.9),
                xytext=(43, 0.33), fontsize=10, ha="left", color=COLORS["ablated"],
                arrowprops={"arrowstyle": "-", "color": COLORS["ablated"], "linewidth": 1.1})
    ax.set(xlim=(0, 106), ylim=(-0.045, 1.13), xticks=[0, 20, 40, 60, 80, 103],
           xlabel="Nodes retained from a fixed importance ranking",
           ylabel="Faithfulness (recovered coupling fraction)")
    _caption(fig, ax,
             f"{len(sizes)} measured sizes; band: 95% gene-bootstrap CIs."
             f"\nFirst sampled point ≥90%: {right}/103; crossing bracket ({left}, {right}]."
             "\nGradual curve under this ranking; minimality is not established.")

    # D: do not fabricate a full scree spectrum from a two-row Fourier basis.
    ax = axes[1, 1]
    plane = report["planes"]["period3"]
    sigma = np.asarray(plane["sigma"])
    ax.bar([1, 2], sigma, color=[COLORS["signal"], "#62A8CB"], width=0.58)
    for pos, value in enumerate(sigma, start=1):
        ax.text(pos, value + 5, f"{value:.1f}", ha="center", va="bottom", fontsize=12)
    ax.set(xlim=(0.35, 2.65), ylim=(0, 225), xticks=[1, 2],
           xlabel="Singular component of the Fourier-bin basis",
           ylabel="Singular value of [Re F; Im F]")
    _caption(fig, ax,
             rf"Period {plane['period']:g} nt (bin {plane['bin']}); circularity $\sigma_2/\sigma_1$ = {plane['circularity']:.3f}."
             "\nRank ≤2 follows from the single-frequency decomposition."
             "\nThis geometry alone does not establish causal use or circuit size.")

    fig.text(0.080, 0.045,
             "Source artifacts and rendering conventions are hashed in fig_3.provenance.json.",
             fontsize=9, color="#596872")
    fig.text(0.080, 0.025,
             "A–C show scoped model interventions; D shows representation geometry. No biological reading-frame claim is established.",
             fontsize=9, color="#596872")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "figures/fig_3.png")
    parser.add_argument("--overwrite", action="store_true", help="replace generated figure files only")
    args = parser.parse_args()
    outputs = [args.out.with_suffix(ext) for ext in (".png", ".svg", ".provenance.json")]
    for path in outputs:
        guard_output_path(path, overwrite=args.overwrite)
    documents = load_sources()
    fig = render(documents)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path in outputs[:2]:
        metadata = {"Date": None} if path.suffix == ".svg" else {}
        fig.savefig(path, dpi=240, facecolor="white", transparent=False, metadata=metadata)
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)

    provenance = figure_provenance(
        [REPO / relative for relative, _ in SOURCES]
        + [Path(__file__), REPO / "src/gfmcd/figures/style.py"]
    )
    provenance["renderer"] = {
        "path": str(Path(__file__).resolve().relative_to(REPO)),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    provenance["command"] = ["python", *sys.argv]
    provenance["git_dirty"] = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip())
    provenance["environment"] = {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
                                 "numpy": np.__version__}
    provenance["rendering"] = {
        "A": "Recorded means ±1 gene SE; straight segments. Psi_C annotation uses its recorded 95% CI.",
        "B": "Dimensionless recovery fractions and recorded 95% gene-bootstrap CIs; interaction has no recorded CI.",
        "C": "All 17 measured sizes; straight segments and recorded 95% gene-bootstrap CIs. The 90% crossing is linearly interpolated.",
        "D": "Only the two recorded singular values of the single period-3 Fourier bin; no unrecorded spectrum or energy-share claim.",
    }
    provenance["limitations"] = [
        "A: C36 omits required strand/donor-phase sensitivities and has a frozen code-pin mismatch (R46); coding specificity remains unresolved.",
        "B/C: original manifests record dirty producer trees and empty output lists; this render does not repair historical provenance.",
        "D: fig_report_data.json has no sibling result manifest; single-frequency rank is structural, not causal evidence.",
    ]
    outputs[2].write_text(json.dumps(provenance, indent=2) + "\n")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
