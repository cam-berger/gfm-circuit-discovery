"""Compute full test metrics and bootstrap intervals for the selected frozen probe."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from gfmcd.dnabert.manifest import write_run_manifest

REPRESENTATION_PATH = Path(
    "results/dnabert_promoter/frozen/representations/test.npz"
)
MODEL_PATH = Path(
    "results/dnabert_promoter/frozen/"
    "layer_probes/selected_layer_2_probe.joblib"
)
OUTPUT_DIR = Path(
    "results/dnabert_promoter/frozen/layer_probes"
)

LAYER = 2
SEED = 42
N_BOOTSTRAP = 2000


def calculate_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    tn, fp, _, _ = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp)

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(labels, probabilities)
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "mcc": float(
            matthews_corrcoef(labels, predictions)
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(REPRESENTATION_PATH) as payload:
        features = payload["features"].copy()
        labels = payload["labels"].astype(int).copy()

    if features.shape != (5865, 12, 768):
        raise ValueError(
            f"Unexpected feature shape: {features.shape}"
        )

    model = joblib.load(MODEL_PATH)

    probabilities = model.predict_proba(
        features[:, LAYER, :]
    )[:, 1]

    predictions = model.predict(
        features[:, LAYER, :]
    ).astype(int)

    predictions_path = (
        OUTPUT_DIR
        / "selected_layer_2_test_predictions.csv"
    )

    pd.DataFrame(
        {
            "example_index": np.arange(len(labels)),
            "label": labels,
            "prediction": predictions,
            "probability_promoter": probabilities,
            "correct": predictions == labels,
        }
    ).to_csv(
        predictions_path,
        index=False,
    )

    metrics = calculate_metrics(
        labels,
        predictions,
        probabilities,
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)

    rng = np.random.default_rng(SEED)

    bootstrap_values = {
        name: np.empty(
            N_BOOTSTRAP,
            dtype=np.float64,
        )
        for name in metrics
    }

    for replicate in range(N_BOOTSTRAP):
        sampled_indices = np.concatenate(
            [
                rng.choice(
                    negative_indices,
                    size=len(negative_indices),
                    replace=True,
                ),
                rng.choice(
                    positive_indices,
                    size=len(positive_indices),
                    replace=True,
                ),
            ]
        )

        sample_metrics = calculate_metrics(
            labels[sampled_indices],
            predictions[sampled_indices],
            probabilities[sampled_indices],
        )

        for name, value in sample_metrics.items():
            bootstrap_values[name][replicate] = value

        if (
            replicate == 0
            or (replicate + 1) % 250 == 0
            or replicate + 1 == N_BOOTSTRAP
        ):
            print(
                f"Completed {replicate + 1}/"
                f"{N_BOOTSTRAP}"
            )

    intervals = {
        name: {
            "lower_95": float(
                np.quantile(values, 0.025)
            ),
            "upper_95": float(
                np.quantile(values, 0.975)
            ),
        }
        for name, values in bootstrap_values.items()
    }

    summary = {
        "model": (
            "quietflamingo/"
            "dnabert2-no-flashattention"
        ),
        "endpoint": "encoder_layer_2_output",
        "layer_index": LAYER,
        "classifier": (
            "standardized_logistic_regression"
        ),
        "test_examples": int(len(labels)),
        "label_counts": {
            "non_promoter": int(
                np.sum(labels == 0)
            ),
            "promoter": int(
                np.sum(labels == 1)
            ),
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "metrics": metrics,
        "bootstrap": {
            "method": (
                "stratified nonparametric bootstrap"
            ),
            "seed": SEED,
            "replicates": N_BOOTSTRAP,
            "confidence_level": 0.95,
            "intervals": intervals,
        },
    }

    summary_path = (
        OUTPUT_DIR
        / "selected_layer_2_test_statistics.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n===== TEST METRICS =====")

    for name, value in metrics.items():
        interval = intervals[name]

        print(
            f"{name}: {value:.6f} "
            f"[{interval['lower_95']:.6f}, "
            f"{interval['upper_95']:.6f}]"
        )

    print("Saved:", predictions_path)
    print("Saved:", summary_path)

    # Sibling provenance record. Written after the outputs exist so their hashes are real.
    manifest_path = write_run_manifest(
        summary_path,
        experiment_id="dnabert_promoter_frozen_test_statistics",
        config={},
        outputs=[predictions_path],
        inputs=[PROBE_PATH],
        seed=SEED,
        probe_path=PROBE_PATH,
    )
    print("Saved:", manifest_path)

    print(
        "FROZEN TEST STATISTICAL "
        "VALIDATION SUCCEEDED"
    )


if __name__ == "__main__":
    main()
