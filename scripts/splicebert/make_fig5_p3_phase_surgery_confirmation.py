#!/usr/bin/env python
"""Figure 5: clean P3 phase-surgery confirmation.

Reads only the immutable independent confirmation analysis and its frozen package.  It refuses failed
instrument gates and writes the exact plotted data plus source hashes beside the figure.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gfmcd-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import sha256_file  # noqa: E402


ANALYSIS = REPO / "results/phase-lattice/p3_phase_surgery_confirmation_v1/confirmation_analysis.json"
PACKAGE = REPO / "results/phase-lattice/p3_phase_surgery_confirmation_package_v1.json"
OUTBASE = REPO / "figures/splicebert/fig5_p3_phase_surgery_confirmation"
FIGDATA = REPO / "results/phase-lattice/p3_phase_surgery_confirmation_v1/figure_data.json"


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


def style_axis(axis, *, panel: str, title: str) -> None:
    axis.set_title(f"{panel}  {title}", loc="left", fontsize=12, fontweight="bold", pad=10)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#d9dee7", linewidth=0.7, alpha=0.8)
    axis.tick_params(labelsize=9)


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

    blue, red, teal, ink, grey = "#3569a8", "#c64b47", "#2a8f88", "#202633", "#87909e"
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0), constrained_layout=True)
    fig.suptitle(
        "Nucleotide-invariant surgery confirms phase-aligned use of SpliceBERT's P3 direction",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    x = np.asarray(data["amounts"])
    for name, color, label in (
        ("cds", red, "CDS block"),
        ("upstream_intron", blue, "Upstream-intron block"),
    ):
        mean = np.asarray(data["curve"][name]["mean"])
        se = np.asarray(data["curve"][name]["se"])
        ax.plot(x, mean, marker="o", linewidth=2.0, color=color, label=label)
        ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.14, linewidth=0)
    for amount in (3, 6, 9):
        ax.axvspan(amount - 0.18, amount + 0.18, color=teal, alpha=0.12, linewidth=0)
    ax.set_xticks(x)
    ax.set_xlabel("Local P3 shift r (nt); teal = r mod 3 = 0")
    ax.set_ylabel("Donor–acceptor coupling (nats)")
    ax.legend(frameon=False, fontsize=9)
    style_axis(ax, panel="A", title="Only off-register CDS surgery lowers coupling")

    ax = axes[0, 1]
    primary = data["primary"]
    ax.axvspan(-primary["sesoi"], primary["sesoi"], color="#eef1f5", zorder=0)
    ax.axvline(0, color=ink, linewidth=1)
    ax.errorbar(
        primary["mean"], 0, xerr=[[primary["mean"] - primary["ci"][0]],
                                  [primary["ci"][1] - primary["mean"]]],
        fmt="o", color=red, markersize=8, linewidth=2.4, capsize=5,
    )
    ax.axvline(primary["sesoi"], color=grey, linestyle="--", linewidth=1.2)
    ax.set_yticks([])
    ax.set_xlim(-0.06, 0.20)
    ax.set_xlabel("ΨC = modulo recovery(CDS) − modulo recovery(intron), nats")
    ax.text(
        0.63, 0.66,
        f"{primary['mean']:.3f}  [{primary['ci'][0]:.3f}, {primary['ci'][1]:.3f}]",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10,
        fontweight="bold", color=red,
    )
    ax.text(primary["sesoi"] + 0.003, -0.17, "SESOI 0.05", fontsize=8.5, color=grey)
    style_axis(ax, panel="B", title=f"Material primary effect across {primary['n_genes']} untouched genes")

    ax = axes[1, 0]
    controls = data["direction"]["control_means"]
    labels = sorted(controls)
    values = [controls[label] for label in labels]
    ax.scatter(range(1, 20), values, s=35, color=grey, alpha=0.9, label="19 geometry-matched controls")
    ax.scatter([20], [data["direction"]["p3_mean"]], s=80, color=red,
               edgecolor="white", linewidth=0.8, label="learned P3 direction")
    ax.axhline(0, color=ink, linewidth=1)
    ax.set_xticks([1, 5, 10, 15, 19, 20], ["1", "5", "10", "15", "19", "P3"])
    ax.set_ylabel("Mean ΨC (nats)")
    ax.set_xlabel("Pre-seeded hidden-axis component")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.text(
        0.98, 0.18,
        f"paired P3−control = {data['direction']['paired_excess_mean']:.3f}\n"
        f"95% CI [{data['direction']['paired_excess_ci'][0]:.3f}, "
        f"{data['direction']['paired_excess_ci'][1]:.3f}]\nrank p={data['direction']['rank_p']:.2f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
    )
    style_axis(ax, panel="C", title="The effect is specific to the learned hidden direction")

    ax = axes[1, 1]
    cds = np.asarray(data["context"]["cds_psi"])
    non = np.asarray(data["context"]["non_cds_psi"])
    lo = min(float(cds.min()), float(non.min())) - 0.05
    hi = max(float(cds.max()), float(non.max())) + 0.05
    ax.plot([lo, hi], [lo, hi], color=grey, linestyle="--", linewidth=1)
    ax.scatter(non, cds, s=28, color=teal, alpha=0.72, edgecolor="none")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Matched non-CDS-exon ΨC (nats)")
    ax.set_ylabel("Matched CDS ΨC (nats)")
    ax.text(
        0.04, 0.96,
        f"CDS−non-CDS = {data['context']['delta_mean']:.3f}\n"
        f"95% CI [{data['context']['delta_ci'][0]:.3f}, {data['context']['delta_ci'][1]:.3f}]\n"
        "coding enrichment: inconclusive",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#d9dee7", "alpha": 0.9, "pad": 5},
    )
    style_axis(ax, panel="D", title="Matched coding-context enrichment is unresolved")

    FIGDATA.parent.mkdir(parents=True, exist_ok=True)
    FIGDATA.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    fig.savefig(OUTBASE.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(OUTBASE.with_suffix(".pdf"), bbox_inches="tight")
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
            "Confirms present-day phase-aligned P3 use in CDS blocks; coding enrichment and origin unresolved."
        ),
    }
    OUTBASE.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {OUTBASE.with_suffix('.png')}, .pdf, figure data, and provenance")


if __name__ == "__main__":
    main()
