from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("results/dnabert_promoter")
OUTDIR = ROOT / "figures" / "frozen"
OUTDIR.mkdir(parents=True, exist_ok=True)


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

print("\nAll frozen DNABERT figures saved to:", OUTDIR)
