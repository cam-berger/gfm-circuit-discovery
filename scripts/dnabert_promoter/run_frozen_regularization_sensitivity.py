"""Test frozen DNABERT layer selection across logistic-regression strengths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPRESENTATION_DIR = Path(
    "results/dnabert_promoter/frozen/representations"
)
OUTPUT_DIR = Path(
    "results/dnabert_promoter/frozen/layer_probes"
)
OUTPUT_PATH = (
    OUTPUT_DIR
    / "regularization_sensitivity.json"
)

C_VALUES = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]

EXPECTED_LAYERS = 12
EXPECTED_HIDDEN_SIZE = 768


def load_split(
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    path = REPRESENTATION_DIR / f"{split}.npz"

    with np.load(path) as payload:
        features = payload["features"].copy()
        labels = payload["labels"].astype(int).copy()

    expected_shape = (
        len(labels),
        EXPECTED_LAYERS,
        EXPECTED_HIDDEN_SIZE,
    )

    if features.shape != expected_shape:
        raise ValueError(
            f"Unexpected {split} feature shape: "
            f"{features.shape}; expected "
            f"{expected_shape}"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"{split} contains non-finite features"
        )

    return features, labels


def evaluate(
    model,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(
        features
    )[:, 1]

    return {
        "accuracy": float(
            accuracy_score(
                labels,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities,
            )
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_features, train_labels = load_split(
        "train"
    )
    validation_features, validation_labels = (
        load_split("validation")
    )

    results = []

    for c_value in C_VALUES:
        print(
            f"\n===== C={c_value} ====="
        )

        layer_results = []

        for layer in range(EXPECTED_LAYERS):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    max_iter=3000,
                    solver="lbfgs",
                ),
            )

            model.fit(
                train_features[:, layer, :],
                train_labels,
            )

            metrics = evaluate(
                model,
                validation_features[
                    :, layer, :
                ],
                validation_labels,
            )

            layer_results.append(
                {
                    "layer": layer,
                    "validation": metrics,
                }
            )

            print(
                f"Layer {layer:2d} | "
                f"BA="
                f"{metrics['balanced_accuracy']:.6f} | "
                f"AUC="
                f"{metrics['roc_auc']:.6f}"
            )

        ranked_layers = sorted(
            layer_results,
            key=lambda item: (
                item["validation"][
                    "balanced_accuracy"
                ],
                item["validation"]["roc_auc"],
            ),
            reverse=True,
        )

        best = ranked_layers[0]

        results.append(
            {
                "C": c_value,
                "selected_layer": int(
                    best["layer"]
                ),
                "selected_validation_metrics":
                    best["validation"],
                "layer_results":
                    layer_results,
                "ranked_layers": [
                    {
                        "rank": rank,
                        **item,
                    }
                    for rank, item in enumerate(
                        ranked_layers,
                        start=1,
                    )
                ],
            }
        )

        print(
            "Selected layer:",
            best["layer"],
        )

    layer_win_counts = {
        str(layer): 0
        for layer in range(
            EXPECTED_LAYERS
        )
    }

    for result in results:
        layer_win_counts[
            str(result["selected_layer"])
        ] += 1

    summary = {
        "classifier":
            "standardized_logistic_regression",
        "selection_metric":
            "validation_balanced_accuracy",
        "tie_breaker":
            "validation_roc_auc",
        "C_values": C_VALUES,
        "num_layers": EXPECTED_LAYERS,
        "results": results,
        "layer_win_counts":
            layer_win_counts,
        "layer_2_selected_count": int(
            layer_win_counts["2"]
        ),
        "layer_2_selected_fraction": float(
            layer_win_counts["2"]
            / len(C_VALUES)
        ),
        "test_split_used": False,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n===== REGULARIZATION "
        "SENSITIVITY SUMMARY ====="
    )

    for result in results:
        metrics = result[
            "selected_validation_metrics"
        ]

        print(
            f"C={result['C']:<7} | "
            f"selected_layer="
            f"{result['selected_layer']} | "
            f"BA="
            f"{metrics['balanced_accuracy']:.6f} | "
            f"AUC="
            f"{metrics['roc_auc']:.6f}"
        )

    print(
        "Layer 2 selected:",
        f"{summary['layer_2_selected_count']}"
        f"/{len(C_VALUES)}",
    )
    print("Saved:", OUTPUT_PATH)
    print(
        "FROZEN REGULARIZATION "
        "SENSITIVITY SUCCEEDED"
    )


if __name__ == "__main__":
    main()
