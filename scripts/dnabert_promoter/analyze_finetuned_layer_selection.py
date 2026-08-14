"""Resolve the fine-tuned layer-probe sweep into a band, not a point.

`selected_endpoint.json` reports Layer 10 as the validation argmax. That argmax leads the runner-up
by `0.000998` balanced-accuracy on a single seed, which is smaller than the sampling error of the
dev split itself, so "Layer 10" names the winner of a coin flip rather than a localized endpoint.
This runner reads the committed per-layer metrics and reports every layer that is not separated
from the argmax at 1 and 2 standard errors.

It touches no model and downloads nothing: the layer sweep is already committed, and this is
arithmetic on it. That is deliberate — the seeded repeat that would settle the question needs a GPU
run, and this artifact is the honest bound available without one (`DBPR11`).

The standard error is the balanced-accuracy sampling error under a per-class binomial, using the
layer's own balanced accuracy as the proxy for both class recalls:

    SE(BA) = 0.5 * sqrt( r(1-r)/n_0 + r(1-r)/n_1 )

The split is near-balanced (2974 / 2946), so the proxy is tight; the alternative is to solve the
two class recalls from accuracy and balanced accuracy, which divides by `n_0 - n_1 = 28` and is
numerically worthless. The approximation is recorded in the artifact rather than hidden.

This is an unpaired bound. Per-layer per-example predictions are not committed, so a paired test —
which would be tighter — cannot be run from the tree. The band is therefore conservative in the
direction that matters: it may include layers a paired test would separate, and will not exclude
layers a paired test would keep.

    uv run python scripts/dnabert_promoter/analyze_finetuned_layer_selection.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from gfmcd.dnabert.manifest import write_run_manifest


DEFAULT_METRICS = Path(
    "results/dnabert_promoter/finetuned/"
    "layer_localization/probes/layer_probe_metrics.csv"
)

DEFAULT_DATASET_SUMMARY = Path(
    "results/dnabert_promoter/finetuned/"
    "shared/gue_prom_300_all_dataset_summary.json"
)

DEFAULT_OUTPUT = Path(
    "results/dnabert_promoter/finetuned/"
    "layer_localization/layer_selection_band.json"
)

SELECTION_SPLIT = "dev"
SELECTION_METRIC = "dev_balanced_accuracy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS,
    )

    parser.add_argument(
        "--dataset-summary",
        type=Path,
        default=DEFAULT_DATASET_SUMMARY,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def balanced_accuracy_se(
    balanced_accuracy: float,
    n_negative: int,
    n_positive: int,
) -> float:
    """Sampling SE of balanced accuracy, using BA as the proxy for both class recalls."""
    variance = (
        balanced_accuracy * (1.0 - balanced_accuracy) / n_negative
        + balanced_accuracy * (1.0 - balanced_accuracy) / n_positive
    )
    return 0.5 * math.sqrt(variance)


def main() -> None:
    args = parse_args()

    with args.metrics.open() as handle:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]

    summary = json.loads(args.dataset_summary.read_text())
    label_counts = summary[SELECTION_SPLIT]["label_counts"]
    n_negative = int(label_counts["0"])
    n_positive = int(label_counts["1"])

    argmax = max(rows, key=lambda row: row[SELECTION_METRIC])
    best_layer = int(argmax["layer"])
    best_value = argmax[SELECTION_METRIC]

    standard_error = balanced_accuracy_se(best_value, n_negative, n_positive)

    runner_up = max(
        (row for row in rows if int(row["layer"]) != best_layer),
        key=lambda row: row[SELECTION_METRIC],
    )
    margin = best_value - runner_up[SELECTION_METRIC]

    def band(multiplier: float) -> list[int]:
        threshold = best_value - multiplier * standard_error
        return sorted(
            int(row["layer"])
            for row in rows
            if row[SELECTION_METRIC] >= threshold
        )

    band_1se = band(1.0)
    band_2se = band(2.0)

    test_argmax = max(rows, key=lambda row: row["test_balanced_accuracy"])

    payload = {
        "selection_metric": SELECTION_METRIC,
        "selection_split": SELECTION_SPLIT,
        "selection_split_label_counts": {
            "0": n_negative,
            "1": n_positive,
        },
        "argmax_layer": best_layer,
        "argmax_value": best_value,
        "runner_up_layer": int(runner_up["layer"]),
        "runner_up_value": runner_up[SELECTION_METRIC],
        "margin_over_runner_up": margin,
        "standard_error": standard_error,
        "standard_error_method": (
            "per-class binomial; balanced accuracy used as the proxy for both class "
            "recalls; unpaired, because per-layer per-example predictions are not committed"
        ),
        "margin_in_standard_errors": margin / standard_error,
        "layers_within_1_se_of_argmax": band_1se,
        "layers_within_2_se_of_argmax": band_2se,
        "test_argmax_layer": int(test_argmax["layer"]),
        "test_argmax_value": test_argmax["test_balanced_accuracy"],
        "argmax_layer_test_value": argmax["test_balanced_accuracy"],
        "reading": (
            "The validation argmax is not separated from its runner-up. Report the "
            "layer choice as the band, and describe Layer "
            f"{best_layer} only as the layer the stated selection rule returned on one seed."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print("Saved:", args.out)

    manifest_path = write_run_manifest(
        args.out,
        experiment_id="dnabert_promoter_finetuned_layer_selection_band",
        config=vars(args),
        inputs=[args.metrics, args.dataset_summary],
        status="BOUNDED",
        notes=[
            "Arithmetic on committed metrics; no model is loaded and no split is re-read.",
            "Unpaired bound. A seeded probe repeat is still owed and is gated on a GPU run.",
        ],
    )
    print("Saved:", manifest_path)

    print(
        f"argmax L{best_layer} leads L{int(runner_up['layer'])} by "
        f"{margin:.6f} = {margin / standard_error:.2f} SE; "
        f"1-SE band {band_1se}"
    )


if __name__ == "__main__":
    main()
