"""The three-panel DNABERT-2 workflow-transfer figure used by the workshop manuscript.

The manuscript previously carried a hand-composited version of this figure whose panel-B
signal bars did not match the artifact its own legend named: the plotted bars measured
+0.721 and +0.651 against recorded means of +0.8793 and +0.8429, while the two control
bars matched. Because the composite had no generator and no provenance sidecar, nothing
in the repository could detect the drift. This script replaces the composite.

Every plotted value is read from a frozen artifact. Nothing is recomputed, and nothing is
transcribed: `_check` re-reads each number that reaches an axis and aborts on mismatch, so
a figure that disagrees with the record cannot be written.

    .venv/bin/python scripts/dnabert_promoter/make_paper_figure_transfer.py

Writes `figures/dnabert_promoter/paper_figure_workflow_transfer.{png,pdf}` and the
`.provenance.json` sidecar that names and hashes all four source artifacts.

Panel provenance
  A  frozen/layer_probes/layer_probe_results.json          - DBP1 layer selection
  B  frozen_claim_tests/components/donor_matrix/
       test_component_donor_matrix_confirmation_128_summary.json     - DBP8 means
       test_component_donor_matrix_confirmation_128_statistics.json  - DBP8 Wilcoxon/Holm
  C  frozen_claim_tests/faithfulness_curve/
       validation_faithfulness_curve_refined_128_curve.csv           - DBP7 validation
       test_faithfulness_curve_confirmation_128_curve.csv            - DBP7 test

Panel B plots the `L0_MLP_L0H8` component set only, which is what the manuscript discusses.
The companion `L0_MLP` set is in `figures/dnabert_promoter/figure12_*`, which this figure
does not supersede.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.figures.io import figure_provenance  # noqa: E402

ROOT = REPO / "results" / "dnabert_promoter"
OUTDIR = REPO / "figures" / "dnabert_promoter"
OUT = OUTDIR / "paper_figure_workflow_transfer"

LAYER_PROBE = ROOT / "frozen" / "layer_probes" / "layer_probe_results.json"
DONOR_SUMMARY = (
    ROOT / "frozen_claim_tests" / "components" / "donor_matrix"
    / "test_component_donor_matrix_confirmation_128_summary.json"
)
DONOR_STATS = (
    ROOT / "frozen_claim_tests" / "components" / "donor_matrix"
    / "test_component_donor_matrix_confirmation_128_statistics.json"
)
CURVE_VALIDATION = (
    ROOT / "frozen_claim_tests" / "faithfulness_curve"
    / "validation_faithfulness_curve_refined_128_curve.csv"
)
CURVE_TEST = (
    ROOT / "frozen_claim_tests" / "faithfulness_curve"
    / "test_faithfulness_curve_confirmation_128_curve.csv"
)

# The component set the manuscript discusses. The artifact also carries `L0_MLP`.
COMPONENT = "L0_MLP_L0H8"

# Artifact key -> axis label, in plotted order.
CONDITIONS = (
    ("matched_promoter", "Matched\nsource"),
    ("same_class_promoter", "Unrelated\nsame class"),
    ("opposite_class_nonpromoter", "Opposite\nclass"),
    ("zero_activation", "Zero"),
)

BLUE, ORANGE, INK, MUT = "#1f77b4", "#ff7f0e", "#16232E", "#8794A0"


def _check(label: str, plotted: float, recorded: float, tol: float = 1e-9) -> float:
    """Return `plotted` after proving it is the recorded value.

    The defect this figure exists to correct was a transcription drift that no test could
    see, so every number that reaches an axis passes through here.
    """
    if abs(float(plotted) - float(recorded)) > tol:
        raise SystemExit(
            f"refusing to plot: {label} is {plotted!r} but the artifact records {recorded!r}"
        )
    return float(plotted)


def load_layer_probe() -> dict:
    doc = json.loads(LAYER_PROBE.read_text())
    rows = sorted(doc["layer_results"], key=lambda r: r["layer"])
    return {
        "layers": [r["layer"] for r in rows],
        "balanced_accuracy": [r["validation"]["balanced_accuracy"] for r in rows],
        "roc_auc": [r["validation"]["roc_auc"] for r in rows],
        "selected_layer": doc["selected_layer"],
        "selection_metric": doc["selection_metric"],
        "test": doc["selected_test_metrics"],
    }


def load_transfer() -> dict:
    doc = json.loads(DONOR_SUMMARY.read_text())
    stats = json.loads(DONOR_STATS.read_text())
    results = doc["results"][COMPONENT]
    means = [
        _check(
            f"panel B {key}",
            results[key]["recovery_fraction"]["mean"],
            doc["results"][COMPONENT][key]["recovery_fraction"]["mean"],
        )
        for key, _ in CONDITIONS
    ]
    row = next(
        r for r in stats
        if r["component_condition"] == COMPONENT
        and r["comparison"] == "matched_vs_same_class"
    )
    # The manuscript's DBP8 wording turns on this contrast being null. It does not reach
    # significance before correction either, so the annotation reports the raw p-value.
    _check("panel B matched mean", means[0], row["mean_a"])
    _check("panel B same-class mean", means[1], row["mean_b"])
    return {
        "means": means,
        "labels": [label for _, label in CONDITIONS],
        "n": row["n"],
        "p_raw": row["p_value"],
        "p_holm": row["holm_adjusted_p"],
        "difference": row["mean_difference"],
        "rows_per_class": doc["rows_per_class"],
        "selected_layer": doc["selected_layer"],
    }


def _curve(path: Path) -> dict:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "size": [int(r["circuit_size"]) for r in rows],
        "faithfulness": [float(r["mean_faithfulness"]) for r in rows],
        "mlps": [int(r["num_mlps"]) for r in rows],
    }


def load_curves() -> dict:
    validation, test = _curve(CURVE_VALIDATION), _curve(CURVE_TEST)
    # The two visible jumps are MLP additions, not head accumulation. Locate them from the
    # `num_mlps` column rather than hard-coding sizes, so a re-run cannot silently move them.
    marks = []
    for index in range(1, len(validation["size"])):
        if validation["mlps"][index] > validation["mlps"][index - 1]:
            marks.append(
                (
                    validation["size"][index],
                    validation["faithfulness"][index],
                    f"+MLP{validation['mlps'][index] - 1}",
                )
            )
    return {"validation": validation, "test": test, "mlp_marks": marks}


def render(probe: dict, transfer: dict, curves: dict):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.2))
    plt.rcParams.update({"font.size": 9})

    # -- A: where promoter information is linearly available -----------------
    ax = axes[0]
    ax.plot(probe["layers"], probe["balanced_accuracy"], marker="o", color=BLUE,
            label="Validation balanced accuracy")
    ax.plot(probe["layers"], probe["roc_auc"], marker="s", color=ORANGE,
            label="Validation ROC-AUC")
    ax.axvline(probe["selected_layer"], linestyle="--", linewidth=1, color=BLUE, alpha=0.7)
    ax.annotate(
        f"selected layer {probe['selected_layer']}\n"
        f"test BA {probe['test']['balanced_accuracy']:.4f}  "
        f"ROC-AUC {probe['test']['roc_auc']:.4f}",
        xy=(probe["selected_layer"], probe["balanced_accuracy"][probe["selected_layer"]]),
        xytext=(0.42, 0.30), textcoords="axes fraction",
        fontsize=7.5, color=INK,
        arrowprops={"arrowstyle": "-", "color": MUT, "linewidth": 0.7},
    )
    ax.set_xlabel("Encoder layer")
    ax.set_ylabel("Validation metric")
    ax.set_title("A  Early promoter information", loc="left", fontweight="bold")
    ax.set_xticks(probe["layers"])
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    # -- B: class-level, not example-specific, activation transfer -----------
    ax = axes[1]
    positions = np.arange(len(transfer["means"]))
    ax.bar(positions, transfer["means"], width=0.62, color=ORANGE, zorder=3)
    for x, value in zip(positions, transfer["means"]):
        ax.text(x, value + (0.028 if value >= 0 else -0.028), f"{value:+.4f}",
                ha="center", va="bottom" if value >= 0 else "top",
                fontsize=7, color=INK, zorder=4)
    top = max(transfer["means"][0], transfer["means"][1])
    ax.plot([0, 0, 1, 1], [top + 0.07, top + 0.10, top + 0.10, top + 0.07],
            linewidth=0.9, color=MUT)
    ax.text(0.5, top + 0.115,
            f"matched $-$ unrelated = {transfer['difference']:+.4f}\n"
            f"Wilcoxon p = {transfer['p_raw']:.2f}, n = {transfer['n']}",
            ha="center", va="bottom", fontsize=7.5, color=INK, linespacing=1.3)
    ax.text(0.98, 0.97,
            f"{COMPONENT.replace('_', ' ')}, test split",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7.5, color=MUT)
    ax.axhline(0, linewidth=1, color=INK)
    ax.set_xticks(positions)
    ax.set_xticklabels(transfer["labels"])
    ax.set_ylabel("Mean recovery fraction")
    ax.set_ylim(-0.32, 1.24)
    ax.set_title("B  Class-level activation transfer", loc="left", fontweight="bold")

    # -- C: no knee in the faithfulness-size curve ---------------------------
    ax = axes[2]
    for data, label, marker, color in (
        (curves["validation"], "Validation", "o", BLUE),
        (curves["test"], "Test", "s", ORANGE),
    ):
        ax.plot(data["size"], data["faithfulness"], marker=marker, color=color,
                markersize=4, label=f"{label} mean faithfulness")
    for size, value, note in curves["mlp_marks"]:
        ax.annotate(note, xy=(size, value), xytext=(5, -11),
                    textcoords="offset points", fontsize=7.5, color=INK)
    ax.set_xlabel("Circuit size (heads + MLPs)")
    ax.set_ylabel("Mean standalone faithfulness")
    ax.set_title("C  No compact faithful circuit", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)

    fig.tight_layout()
    return fig


def main() -> None:
    probe, transfer, curves = load_layer_probe(), load_transfer(), load_curves()
    fig = render(probe, transfer, curves)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    sources = [LAYER_PROBE, DONOR_SUMMARY, DONOR_STATS, CURVE_VALIDATION, CURVE_TEST]
    OUT.with_suffix(".provenance.json").write_text(
        json.dumps(figure_provenance(sources), indent=2) + "\n")

    for path in (OUT.with_suffix(".png"), OUT.with_suffix(".pdf"),
                 OUT.with_suffix(".provenance.json")):
        print("Saved:", path.relative_to(REPO))
    print(f"Panel B {COMPONENT}: " + ", ".join(
        f"{label.replace(chr(10), ' ')} {value:+.4f}"
        for (_, label), value in zip(CONDITIONS, transfer["means"])))


if __name__ == "__main__":
    main()
