from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/dnabert_promoter")

# Published figures belong in figures/<family>/, not under results/: `results/**/figures/` is
# gitignored so that the published set lives in exactly one place, next to its captions.
OUTDIR = Path("figures/dnabert_promoter")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Figures 4-8 below are built from the SUPERSEDED encoder-Layer-3 replicate under
# results/dnabert_promoter/causal/layer3_*, not from the canonical Layer-2 suite. They therefore do
# not depict the numbers in studies/dnabert_promoter/CLAIMS.md (DBP5-DBP8) - see DBPR7 and each
# figure's .CAPTION.md. Figures 1-3 are canonical.


def save_both(fig: plt.Figure, stem: str) -> None:
    png_path = OUTDIR / f"{stem}.png"
    pdf_path = OUTDIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", png_path)
    print("Saved:", pdf_path)


# ---------------------------------------------------------------------
# Figure 1: Frozen layer-probe validation trajectory
# ---------------------------------------------------------------------
layer_probe_path = (
    ROOT / "frozen" / "layer_probes" / "layer_probe_results.json"
)
layer_probe = json.loads(layer_probe_path.read_text())

layer_results = sorted(
    layer_probe["layer_results"],
    key=lambda item: item["layer"],
)

layers = [item["layer"] for item in layer_results]
val_ba = [
    item["validation"]["balanced_accuracy"]
    for item in layer_results
]
val_roc = [
    item["validation"]["roc_auc"]
    for item in layer_results
]
selected_layer = layer_probe["selected_layer"]
selected_test = layer_probe["selected_test_metrics"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(
    layers,
    val_ba,
    marker="o",
    label="Validation balanced accuracy",
)
ax.plot(
    layers,
    val_roc,
    marker="s",
    label="Validation ROC-AUC",
)
ax.axvline(
    selected_layer,
    linestyle="--",
    label=f"Selected layer = {selected_layer}",
)
ax.set_xlabel("Layer")
ax.set_ylabel("Metric")
ax.set_title(
    "Frozen DNABERT layerwise probe trajectory\n"
    f"Selected Layer {selected_layer} test BA="
    f"{selected_test['balanced_accuracy']:.3f}, "
    f"test ROC-AUC={selected_test['roc_auc']:.3f}"
)
ax.set_xticks(layers)
ax.legend()
save_both(fig, "figure1_frozen_layer_probe_validation")


# ---------------------------------------------------------------------
# Figure 2: Validation all-layer corruption trajectory
# Figure 3: Test all-layer corruption trajectory
# ---------------------------------------------------------------------
def plot_corruption_trajectory(csv_path: Path, stem: str, title: str) -> None:
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["layer"],
        df["clean_balanced_accuracy"],
        marker="o",
        label="Clean",
    )
    ax.plot(
        df["layer"],
        df["mono_balanced_accuracy"],
        marker="s",
        label="Mononucleotide shuffle",
    )
    ax.plot(
        df["layer"],
        df["dinucleotide_balanced_accuracy"],
        marker="^",
        label="Dinucleotide shuffle",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(title)
    ax.set_xticks(df["layer"])
    ax.legend()
    save_both(fig, stem)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["layer"],
        df["mono_balanced_accuracy_drop"],
        marker="s",
        label="Mono BA drop",
    )
    ax.plot(
        df["layer"],
        df["dinucleotide_balanced_accuracy_drop"],
        marker="^",
        label="Dinucleotide BA drop",
    )
    peak_layer = int(
        df.loc[
            df["mono_balanced_accuracy_drop"].idxmax(),
            "layer",
        ]
    )
    ax.axvline(
        peak_layer,
        linestyle="--",
        label=f"Peak mono-drop layer = {peak_layer}",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Balanced accuracy drop")
    ax.set_title(title + " — corruption damage")
    ax.set_xticks(df["layer"])
    ax.legend()
    save_both(fig, stem + "_drops")


plot_corruption_trajectory(
    ROOT
    / "behavioral"
    / "all_layer_corruption"
    / "validation_full_trajectory.csv",
    "figure2_validation_corruption_trajectory",
    "Frozen DNABERT validation corruption trajectory",
)

plot_corruption_trajectory(
    ROOT
    / "behavioral"
    / "all_layer_corruption"
    / "test_full_trajectory.csv",
    "figure3_test_corruption_trajectory",
    "Frozen DNABERT test corruption trajectory",
)


# ---------------------------------------------------------------------
# Figure 4: Layer-3 head heatmap (validation)
# ---------------------------------------------------------------------
head_summary_path = (
    ROOT
    / "causal"
    / "layer3_head_ablation"
    / "validation_layer3_head_256_summary.json"
)
head_summary = json.loads(head_summary_path.read_text())

heads = head_summary["heads"]
max_layer = max(item["layer"] for item in heads)
max_head = max(item["head"] for item in heads)

heat = np.full((max_layer + 1, max_head + 1), np.nan)
for item in heads:
    heat[item["layer"], item["head"]] = item["promoter_mean_damage"]

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(heat, aspect="auto", cmap="coolwarm")
ax.set_xlabel("Head")
ax.set_ylabel("Layer")
ax.set_title(
    "Frozen DNABERT Layer-3 head ablation map\n"
    "(promoter mean damage; more negative = more promoter-supporting)"
)
ax.set_xticks(range(max_head + 1))
ax.set_yticks(range(max_layer + 1))

# Highlight the locked six-head set
primary_heads = {
    (0, 8),
    (1, 8),
    (2, 3),
    (2, 8),
    (2, 9),
    (3, 11),
}
for layer, head in primary_heads:
    ax.text(
        head,
        layer,
        "★",
        ha="center",
        va="center",
        fontsize=14,
    )

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Promoter mean damage")
save_both(fig, "figure4_layer3_head_heatmap")


# ---------------------------------------------------------------------
# Figure 5 + 6: Grouped ablation (validation and test)
# ---------------------------------------------------------------------
def plot_grouped_ablation(summary_path: Path, stem: str, title: str) -> None:
    data = json.loads(summary_path.read_text())
    groups = data["groups"]

    names = [item["name"] for item in groups]
    damages = [item["promoter_mean_damage"] for item in groups]
    fractions = [item["promoter_fraction_damaged"] for item in groups]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, damages)
    ax.set_ylabel("Promoter mean damage")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    save_both(fig, stem)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, fractions)
    ax.set_ylabel("Fraction of promoters damaged")
    ax.set_title(title + " — fraction damaged")
    ax.tick_params(axis="x", rotation=45)
    save_both(fig, stem + "_fractions")


plot_grouped_ablation(
    ROOT
    / "causal"
    / "layer3_grouped_head_ablation"
    / "validation_layer3_grouped_256_summary.json",
    "figure5_validation_grouped_ablation",
    "Frozen DNABERT Layer-3 grouped ablation (validation)",
)

plot_grouped_ablation(
    ROOT
    / "causal"
    / "layer3_grouped_head_ablation"
    / "test_layer3_grouped_256_summary.json",
    "figure6_test_grouped_ablation",
    "Frozen DNABERT Layer-3 grouped ablation (test)",
)


# ---------------------------------------------------------------------
# Figure 7 + 8: Donor matrix summary (validation and test)
# ---------------------------------------------------------------------
def plot_donor_matrix(summary_path: Path, stem: str, title: str) -> None:
    data = json.loads(summary_path.read_text())
    condition = data["results"]["PRIMARY_SIX_HEADS"]

    donor_types = [
        "matched_promoter",
        "same_class_promoter",
        "opposite_class_nonpromoter",
        "zero_activation",
        "self_corrupt",
    ]

    means = [
        condition[name]["recovery_fraction"]["mean"]
        for name in donor_types
    ]
    labels = [
        "Matched promoter",
        "Same-class promoter",
        "Opposite-class non-promoter",
        "Zero activation",
        "Self-corrupt",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, means)
    ax.axhline(0.0, linestyle="--")
    ax.set_ylabel("Mean recovery fraction")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    save_both(fig, stem)


plot_donor_matrix(
    ROOT
    / "causal"
    / "layer3_primary_head_donor_matrix"
    / "validation_layer3_primary_donor_matrix_128_summary.json",
    "figure7_validation_donor_matrix",
    "Frozen DNABERT Layer-3 donor matrix (validation)",
)

plot_donor_matrix(
    ROOT
    / "causal"
    / "layer3_primary_head_donor_matrix"
    / "test_layer3_primary_donor_matrix_128_summary.json",
    "figure8_test_donor_matrix",
    "Frozen DNABERT Layer-3 donor matrix (test)",
)


# ---------------------------------------------------------------------
# Figure 9: standalone faithfulness versus circuit size
#
# The study's central negative result: there is no knee. The five-node
# candidate circuit reproduces under a fifth of full-model behaviour, and
# faithfulness only reaches 1.0 once all three early MLPs are included.
# Canonical Layer-2 assay (DBP7).
# ---------------------------------------------------------------------
def plot_faithfulness_curve(
    validation_csv: Path,
    test_csv: Path,
    stem: str,
) -> None:
    validation = pd.read_csv(validation_csv)
    test = pd.read_csv(test_csv)

    fig, ax = plt.subplots(figsize=(8, 5))
    for frame, label, marker in (
        (validation, "Validation", "o"),
        (test, "Test", "s"),
    ):
        ax.plot(
            frame["circuit_size"],
            frame["mean_faithfulness"],
            marker=marker,
            label=f"{label} mean faithfulness",
        )

    # The two jumps are MLP additions, not head additions. Mark them so the
    # curve cannot be read as gradual head accumulation.
    for size, note in (
        (16, "+MLP1"),
        (39, "+MLP2"),
    ):
        row = validation.loc[validation["circuit_size"] == size]
        if row.empty:
            continue
        ax.annotate(
            note,
            xy=(size, float(row["mean_faithfulness"].iloc[0])),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=9,
        )

    base = validation.loc[validation["circuit_size"] == 5]
    if not base.empty:
        ax.axhline(
            float(base["mean_faithfulness"].iloc[0]),
            linestyle=":",
            linewidth=1,
            label="Five-node candidate circuit",
        )

    ax.set_xlabel("Circuit size (heads + MLPs)")
    ax.set_ylabel("Mean standalone faithfulness")
    ax.set_title(
        "Frozen DNABERT standalone faithfulness versus circuit size\n"
        "No knee: compactness is refuted at the sizes tested (DBP7)"
    )
    ax.legend()
    save_both(fig, stem)


plot_faithfulness_curve(
    ROOT
    / "frozen_claim_tests"
    / "faithfulness_curve"
    / "validation_faithfulness_curve_refined_128_curve.csv",
    ROOT
    / "frozen_claim_tests"
    / "faithfulness_curve"
    / "test_faithfulness_curve_confirmation_128_curve.csv",
    "figure9_faithfulness_versus_circuit_size",
)


# ---------------------------------------------------------------------
# Figure 10: candidate-circuit necessity and delivery
#
# Conditional effects under the same total-effect denominator. They are NOT
# additive and must not be read as a percentage decomposition (DBP6).
# ---------------------------------------------------------------------
def plot_circuit_accounting(
    validation_path: Path,
    test_path: Path,
    stem: str,
) -> None:
    splits = {
        "Validation": json.loads(validation_path.read_text()),
        "Test": json.loads(test_path.read_text()),
    }
    metrics = ("necessity_fraction", "delivery_fraction")
    labels = ("Necessity\n(knockout)", "Delivery\n(clean patch)")

    positions = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for offset, (split, doc) in zip((-width / 2, width / 2), splits.items()):
        means = [doc[metric]["mean"] for metric in metrics]
        # std of the per-example distribution, not a standard error
        spreads = [doc[metric]["std"] for metric in metrics]
        ax.bar(
            positions + offset,
            means,
            width,
            yerr=spreads,
            capsize=4,
            label=f"{split} (n={doc['rows']})",
        )

    ax.axhline(1.0, linestyle="--", linewidth=1, color="grey")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fraction of total corruption effect")
    ax.set_title(
        "Frozen DNABERT candidate circuit: L0H8, L1H8, L2H9, L2H3, MLP0\n"
        "Conditional effects under a shared denominator - not additive (DBP6)"
    )
    ax.legend()
    save_both(fig, stem)


plot_circuit_accounting(
    ROOT
    / "frozen_claim_tests"
    / "circuit_accounting"
    / "validation_candidate_circuit_128_summary.json",
    ROOT
    / "frozen_claim_tests"
    / "circuit_accounting"
    / "test_candidate_circuit_confirmation_128_summary.json",
    "figure10_candidate_circuit_accounting",
)


# ---------------------------------------------------------------------
# Figure 11: the canonical Layer-2 activation-transplant matrix
#
# The contrast that matters is matched versus same-class: it is null, so the
# transplant carries class-level rather than example-specific information.
# This is the figure that figures 7 and 8 were mistaken for (DBP8).
# ---------------------------------------------------------------------
def plot_component_donor_matrix(
    summary_path: Path,
    stem: str,
    title: str,
) -> None:
    doc = json.loads(summary_path.read_text())
    conditions = (
        "matched_promoter",
        "same_class_promoter",
        "opposite_class_nonpromoter",
        "zero_activation",
    )
    condition_labels = (
        "Matched\nsource",
        "Unrelated\nsame class",
        "Opposite\nclass",
        "Zero",
    )
    components = sorted(doc["results"])

    positions = np.arange(len(conditions))
    width = 0.8 / max(len(components), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, component in enumerate(components):
        offset = (index - (len(components) - 1) / 2) * width
        means = [
            doc["results"][component][condition]["recovery_fraction"]["mean"]
            for condition in conditions
        ]
        ax.bar(positions + offset, means, width, label=component)

    ax.axhline(0.0, linewidth=1, color="black")
    ax.set_xticks(positions)
    ax.set_xticklabels(condition_labels)
    ax.set_ylabel("Mean recovery fraction")
    ax.set_title(title)
    ax.legend(title="Component set")
    save_both(fig, stem)


plot_component_donor_matrix(
    ROOT
    / "frozen_claim_tests"
    / "components"
    / "donor_matrix"
    / "validation_component_donor_matrix_128_summary.json",
    "figure11_validation_component_transplant_matrix",
    "Frozen DNABERT Layer-2 activation transplant (validation)\n"
    "Matched vs unrelated same-class is null: class-level transfer (DBP8)",
)

plot_component_donor_matrix(
    ROOT
    / "frozen_claim_tests"
    / "components"
    / "donor_matrix"
    / "test_component_donor_matrix_confirmation_128_summary.json",
    "figure12_test_component_transplant_matrix",
    "Frozen DNABERT Layer-2 activation transplant (test confirmation)\n"
    "Matched vs unrelated same-class is null: class-level transfer (DBP8)",
)


print("\nAll frozen DNABERT figures saved to:", OUTDIR)
