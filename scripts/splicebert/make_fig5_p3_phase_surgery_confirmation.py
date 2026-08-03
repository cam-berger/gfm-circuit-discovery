#!/usr/bin/env python
"""Figure 5: nucleotide-invariant phase surgery.

Reads only the immutable independent confirmation analysis and frozen package.
All coupling quantities are raw logit differences and are labelled as logit
units. The figure separates present-day use, direction specificity, and the
unresolved coding-context boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gfmcd-matplotlib")

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO / "figures" / "splicebert"
sys.path.insert(0, str(FIGURE_DIR))
sys.path.insert(0, str(REPO / "src"))

from figstyle import (  # noqa: E402
    GREEN,
    INK,
    MUT,
    PERIOD3,
    RULE,
    SEC,
    panel_header,
    save_raster_and_vector,
    use_style,
)
from gfmcd.provenance import sha256_file  # noqa: E402

ANALYSIS = REPO / "results/phase-lattice/p3_phase_surgery_confirmation_v1/confirmation_analysis.json"
PACKAGE = REPO / "results/phase-lattice/p3_phase_surgery_confirmation_package_v1.json"
FIGDATA = REPO / "results/phase-lattice/p3_phase_surgery_confirmation_v1/figure_data.json"
OUTBASE = FIGURE_DIR / "fig5_p3_phase_surgery_confirmation"


def build_figure_data(analysis: dict) -> dict:
    if analysis.get("schema") != "gfmcd.p3_phase_surgery.confirmation_analysis.v1":
        raise ValueError("wrong P3 confirmation-analysis schema")
    if analysis.get("status") != "CONFIRMATORY_ADJUDICATED":
        raise RuntimeError("P3 confirmation has not been adjudicated")
    if not analysis.get("all_instrument_gates_pass"):
        raise RuntimeError("refusing to plot a biological result with failed instrument gates")

    primary = analysis["primary"]
    direction = analysis["direction_specificity"]
    context = analysis["matched_context"]
    curve = primary["figure_curve"]
    amounts = list(range(2, 11))
    return {
        "schema": "gfmcd.p3_phase_surgery.figure_data.v1",
        "source_analysis_sha256": None,
        "amounts": amounts,
        "register": ["in" if amount % 3 == 0 else "off" for amount in amounts],
        "curve": {
            name: {
                "mean": [curve[name][str(amount)]["mean_coupling"] for amount in amounts],
                "se": [curve[name][str(amount)]["se_across_genes"] for amount in amounts],
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
            "control_means": direction["control_means"],
            "paired_excess_mean": direction["p3_minus_mean_control"]["mean"],
            "paired_excess_ci": direction["p3_minus_mean_control"]["ci"],
            "rank_p": direction["add_one_rank_p_one_sided"],
            "n_genes": direction["n_genes"],
        },
        "context": {
            "cds_psi": [row["cds_psi"] for row in context["per_pair"]],
            "non_cds_psi": [row["non_cds_psi"] for row in context["per_pair"]],
            "delta_mean": context["cds_minus_non_cds"]["mean"],
            "delta_ci": context["cds_minus_non_cds"]["ci"],
            "n_pairs": context["n_pairs"],
            "n_same_gene_pairs": context["n_same_gene_pairs"],
            "verdict": "inconclusive",
        },
    }

def _open_axis(ax, *, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=RULE, linewidth=0.65)
    other = "x" if grid_axis == "y" else "y"
    ax.grid(axis=other, visible=False)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text())
    package = json.loads(PACKAGE.read_text())
    data = build_figure_data(analysis)
    analysis_sha = sha256_file(ANALYSIS)
    package_sha = sha256_file(PACKAGE)
    if analysis["package_sha256"] != package_sha:
        raise RuntimeError("analysis/package hash mismatch")
    data["source_analysis_sha256"] = analysis_sha
    data["source_package_sha256"] = package_sha

    use_style()
    plt.rcParams.update(
        {
            "font.size": 8.2,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 7.4,
            "lines.linewidth": 1.8,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.9, 5.65), constrained_layout=True)

    # A — descriptive shift curve; inference is the paired contrast in B.
    ax = axes[0, 0]
    x = np.asarray(data["amounts"])
    for name, color, label, marker in (
        ("cds", PERIOD3, "CDS block", "o"),
        ("upstream_intron", GREEN, "Upstream-intron block", "s"),
    ):
        mean = np.asarray(data["curve"][name]["mean"])
        se = np.asarray(data["curve"][name]["se"])
        ax.plot(
            x,
            mean,
            marker=marker,
            markersize=4.2,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
            zorder=3,
        )
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.11, linewidth=0)
    for amount in (3, 6, 9):
        ax.axvspan(amount - 0.16, amount + 0.16, color=PERIOD3, alpha=0.08, linewidth=0)
    ax.set_xticks(x)
    ax.set_xlabel("Local shift (nt); shaded offsets preserve register")
    ax.set_ylabel("Donor–acceptor coupling (logit units)")
    ax.legend(frameon=False, loc="lower right", handletextpad=0.45)
    panel_header(ax, "A", "Register Dependence")
    _open_axis(ax)

    # B — primary estimate and the frozen smallest effect size of interest.
    ax = axes[0, 1]
    primary = data["primary"]
    ax.axvspan(-primary["sesoi"], primary["sesoi"], color=RULE, alpha=0.55, zorder=0)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.axvline(primary["sesoi"], color=MUT, linestyle="--", linewidth=1.0)
    ax.errorbar(
        primary["mean"],
        0,
        xerr=[
            [primary["mean"] - primary["ci"][0]],
            [primary["ci"][1] - primary["mean"]],
        ],
        fmt="o",
        color=PERIOD3,
        markersize=7,
        linewidth=2.0,
        capsize=4,
        markeredgecolor="white",
        markeredgewidth=0.7,
    )
    ax.set_yticks([])
    ax.set_xlim(-0.06, 0.20)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xlabel(r"Phase-aligned contrast, $\Psi_C$ (logit units)")
    ax.text(
        primary["mean"],
        0.18,
        f"{primary['mean']:.3f} [{primary['ci'][0]:.3f}, {primary['ci'][1]:.3f}]",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=PERIOD3,
        fontweight="bold",
    )
    ax.text(
        primary["sesoi"] + 0.003,
        -0.43,
        "Effect threshold 0.05",
        ha="left",
        va="bottom",
        fontsize=6.9,
        color=SEC,
    )
    panel_header(ax, "B", "Phase-Aligned Effect")
    _open_axis(ax, grid_axis="x")

    # C — collapse the arbitrary control index to a direct learned-vs-control comparison.
    ax = axes[1, 0]
    controls = np.asarray(list(data["direction"]["control_means"].values()))
    jitter = np.linspace(-0.12, 0.12, len(controls))
    ax.scatter(
        jitter,
        controls,
        s=25,
        color=MUT,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.scatter(
        [1],
        [data["direction"]["p3_mean"]],
        s=55,
        color=PERIOD3,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    ax.axhline(0, color=INK, linewidth=0.85)
    ax.set_xticks([0, 1], ["19 matched\ncontrols", "Learned period-3\ndirection"])
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(-0.015, 0.155)
    ax.set_ylabel(r"Phase-aligned contrast, $\Psi_C$ (logit units)")
    ax.text(
        0.98,
        0.16,
        f"Paired excess {data['direction']['paired_excess_mean']:.3f}\n"
        f"[{data['direction']['paired_excess_ci'][0]:.3f}, "
        f"{data['direction']['paired_excess_ci'][1]:.3f}]; "
        f"$p={data['direction']['rank_p']:.2f}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.1,
        color=SEC,
    )
    panel_header(ax, "C", "Direction Specificity")
    _open_axis(ax)

    # D — pairwise differences show heterogeneity without implying a 2D relationship.
    ax = axes[1, 1]
    differences = (
        np.asarray(data["context"]["cds_psi"])
        - np.asarray(data["context"]["non_cds_psi"])
    )
    y = np.linspace(-0.18, 0.18, len(differences))
    ax.scatter(
        differences,
        y,
        s=21,
        color=MUT,
        alpha=0.68,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    context = data["context"]
    ax.errorbar(
        context["delta_mean"],
        0.32,
        xerr=[
            [context["delta_mean"] - context["delta_ci"][0]],
            [context["delta_ci"][1] - context["delta_mean"]],
        ],
        fmt="o",
        color=PERIOD3,
        markersize=6.5,
        linewidth=2.0,
        capsize=4,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=4,
    )
    ax.axvline(0, color=INK, linewidth=0.85)
    ax.set_yticks([])
    ax.set_ylim(-0.30, 0.47)
    ax.set_xlabel(r"CDS $-$ non-CDS contrast (logit units)")
    ax.text(
        context["delta_mean"],
        0.405,
        f"{context['delta_mean']:.3f} "
        f"[{context['delta_ci'][0]:.3f}, {context['delta_ci'][1]:.3f}]",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=PERIOD3,
        fontweight="bold",
    )
    panel_header(ax, "D", "Coding Enrichment")
    _open_axis(ax, grid_axis="x")

    FIGDATA.parent.mkdir(parents=True, exist_ok=True)
    FIGDATA.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    save_raster_and_vector(
        fig,
        OUTBASE,
        formats=("png", "pdf"),
        dpi=450,
        bbox_inches="tight",
    )
    plt.close(fig)

    provenance = {
        "schema": "gfmcd.figure_provenance.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": {"path": str(Path(__file__)), "sha256": sha256_file(__file__)},
        "sources": [
            {"path": str(ANALYSIS), "sha256": analysis_sha},
            {"path": str(PACKAGE), "sha256": package_sha},
        ],
        "outputs": [{"path": str(FIGDATA), "sha256": sha256_file(FIGDATA)}]
        + [
            {
                "path": str(OUTBASE.with_suffix(suffix)),
                "sha256": sha256_file(OUTBASE.with_suffix(suffix)),
            }
            for suffix in (".png", ".pdf")
        ],
        "claim_boundary": (
            "Confirms present-day phase-aligned period-3 use in CDS blocks; "
            "coding enrichment and origin remain unresolved."
        ),
    }
    OUTBASE.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
