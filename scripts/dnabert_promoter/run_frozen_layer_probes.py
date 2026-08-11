"""Train and evaluate one linear probe per frozen DNABERT layer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
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

ALL_PROBE_DIR = OUTPUT_DIR / "all_layer_probes"

SEED = 42
EXPECTED_LAYERS = 12
EXPECTED_HIDDEN_SIZE = 768


def load_split(
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = REPRESENTATION_DIR / f"{split}.npz"

    with np.load(path) as payload:
        features = payload["features"].copy()
        labels = payload["labels"].copy()
        sequence_sha256 = payload[
            "sequence_sha256"
        ].copy()

    return features, labels, sequence_sha256


def validate_features(
    split: str,
    features: np.ndarray,
    labels: np.ndarray,
    sequence_sha256: np.ndarray,
) -> None:
    if features.ndim != 3:
        raise ValueError(
            f"{split} features must have shape "
            "[rows, layers, hidden], got "
            f"{features.shape}"
        )

    if features.shape[1:] != (
        EXPECTED_LAYERS,
        EXPECTED_HIDDEN_SIZE,
    ):
        raise ValueError(
            f"Unexpected {split} feature shape: "
            f"{features.shape}"
        )

    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"{split} feature/label row mismatch: "
            f"{features.shape[0]} vs "
            f"{labels.shape[0]}"
        )

    if features.shape[0] != sequence_sha256.shape[0]:
        raise ValueError(
            f"{split} feature/hash row mismatch"
        )

    if not np.isfinite(features).all():
        raise ValueError(
            f"{split} contains non-finite features"
        )

    unique_labels = sorted(
        np.unique(labels).tolist()
    )

    if unique_labels != [0, 1]:
        raise ValueError(
            f"Unexpected {split} labels: "
            f"{unique_labels}"
        )


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
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
        "pr_auc": float(
            average_precision_score(
                labels,
                probabilities,
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0,
            )
        ),
    }


def evaluate(
    model,
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[
    dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    predictions = model.predict(
        features
    ).astype(int)

    probabilities = model.predict_proba(
        features
    )[:, 1]

    margins = model.decision_function(
        features
    )

    metrics = compute_metrics(
        labels,
        predictions,
        probabilities,
    )

    return (
        metrics,
        predictions,
        probabilities,
        margins,
    )


def save_test_predictions(
    *,
    layer: int,
    labels: np.ndarray,
    sequence_sha256: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    margins: np.ndarray,
) -> Path:
    output_path = (
        OUTPUT_DIR
        / f"layer_{layer}_test_predictions.csv"
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_index",
                "sequence_sha256",
                "label",
                "prediction",
                "probability",
                "margin",
            ],
        )

        writer.writeheader()

        for row_index in range(
            labels.shape[0]
        ):
            writer.writerow(
                {
                    "row_index": row_index,
                    "sequence_sha256": str(
                        sequence_sha256[
                            row_index
                        ]
                    ),
                    "label": int(
                        labels[row_index]
                    ),
                    "prediction": int(
                        predictions[row_index]
                    ),
                    "probability": float(
                        probabilities[
                            row_index
                        ]
                    ),
                    "margin": float(
                        margins[row_index]
                    ),
                }
            )

    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ALL_PROBE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_features,
        train_labels,
        train_hashes,
    ) = load_split("train")

    (
        validation_features,
        validation_labels,
        validation_hashes,
    ) = load_split("validation")

    (
        test_features,
        test_labels,
        test_hashes,
    ) = load_split("test")

    for (
        split,
        features,
        labels,
        hashes,
    ) in (
        (
            "train",
            train_features,
            train_labels,
            train_hashes,
        ),
        (
            "validation",
            validation_features,
            validation_labels,
            validation_hashes,
        ),
        (
            "test",
            test_features,
            test_labels,
            test_hashes,
        ),
    ):
        validate_features(
            split,
            features,
            labels,
            hashes,
        )

    layer_results: list[dict] = []
    fitted_models: dict[int, object] = {}

    for layer in range(
        EXPECTED_LAYERS
    ):
        print(
            f"\n===== Layer {layer} ====="
        )

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=SEED,
                solver="lbfgs",
            ),
        )

        model.fit(
            train_features[
                :,
                layer,
                :,
            ],
            train_labels,
        )

        (
            validation_metrics,
            _,
            _,
            _,
        ) = evaluate(
            model,
            validation_features[
                :,
                layer,
                :,
            ],
            validation_labels,
        )

        (
            test_metrics,
            test_predictions,
            test_probabilities,
            test_margins,
        ) = evaluate(
            model,
            test_features[
                :,
                layer,
                :,
            ],
            test_labels,
        )

        model_path = (
            ALL_PROBE_DIR
            / f"layer_{layer}_probe.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        prediction_path = (
            save_test_predictions(
                layer=layer,
                labels=test_labels,
                sequence_sha256=(
                    test_hashes
                ),
                predictions=(
                    test_predictions
                ),
                probabilities=(
                    test_probabilities
                ),
                margins=test_margins,
            )
        )

        layer_results.append(
            {
                "layer": layer,
                "validation": (
                    validation_metrics
                ),
                "test": test_metrics,
                "model_file": str(
                    model_path
                ),
                "test_predictions_file": (
                    str(prediction_path)
                ),
            }
        )

        fitted_models[layer] = model

        print(
            "Validation:",
            validation_metrics,
        )

        print(
            "Test:",
            test_metrics,
        )

    # Layer selection remains based only on validation.
    ranked_layers = sorted(
        layer_results,
        key=lambda result: (
            result["validation"][
                "balanced_accuracy"
            ],
            result["validation"][
                "roc_auc"
            ],
        ),
        reverse=True,
    )

    selected_result = ranked_layers[0]
    selected_layer = int(
        selected_result["layer"]
    )

    selected_model = (
        fitted_models[selected_layer]
    )

    results = {
        "seed": SEED,
        "classifier": (
            "standardized_logistic_regression"
        ),
        "selection_metric": (
            "validation_balanced_accuracy"
        ),
        "tie_breaker": (
            "validation_roc_auc"
        ),
        "test_usage": (
            "descriptive evaluation only; "
            "not used for layer selection"
        ),
        "representation_scope": (
            "transformer layer outputs 0-11; "
            "embedding output not included"
        ),
        "num_layers": EXPECTED_LAYERS,
        "hidden_size": (
            EXPECTED_HIDDEN_SIZE
        ),
        "train_rows": int(
            train_features.shape[0]
        ),
        "validation_rows": int(
            validation_features.shape[0]
        ),
        "test_rows": int(
            test_features.shape[0]
        ),
        "layer_results": layer_results,
        "ranked_layers": [
            {
                "rank": rank,
                **result,
            }
            for rank, result in enumerate(
                ranked_layers,
                start=1,
            )
        ],
        "selected_layer": (
            selected_layer
        ),
        "selected_validation_metrics": (
            selected_result[
                "validation"
            ]
        ),
        "selected_test_metrics": (
            selected_result["test"]
        ),
    }

    results_path = (
        OUTPUT_DIR
        / "all_layer_probe_results.json"
    )

    results_path.write_text(
        json.dumps(
            results,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    trajectory_path = (
        OUTPUT_DIR
        / "all_layer_probe_trajectory.csv"
    )

    trajectory_rows = []

    for result in layer_results:
        layer = int(result["layer"])
        validation = result[
            "validation"
        ]
        test = result["test"]

        trajectory_rows.append(
            {
                "layer": layer,
                "validation_accuracy": (
                    validation["accuracy"]
                ),
                "validation_balanced_accuracy": (
                    validation[
                        "balanced_accuracy"
                    ]
                ),
                "validation_roc_auc": (
                    validation["roc_auc"]
                ),
                "validation_pr_auc": (
                    validation["pr_auc"]
                ),
                "test_accuracy": (
                    test["accuracy"]
                ),
                "test_balanced_accuracy": (
                    test[
                        "balanced_accuracy"
                    ]
                ),
                "test_roc_auc": (
                    test["roc_auc"]
                ),
                "test_pr_auc": (
                    test["pr_auc"]
                ),
                "test_f1": test["f1"],
                "test_precision": (
                    test["precision"]
                ),
                "test_recall": (
                    test["recall"]
                ),
                "selected_by_validation": (
                    layer == selected_layer
                ),
            }
        )

    with trajectory_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                trajectory_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            trajectory_rows
        )

    # Preserve the canonical selected-probe path
    # used by the existing mechanistic experiments.
    selected_model_path = (
        OUTPUT_DIR
        / (
            f"selected_layer_"
            f"{selected_layer}_probe.joblib"
        )
    )

    joblib.dump(
        selected_model,
        selected_model_path,
    )

    print(
        "\n===== Selected layer ====="
    )
    print(
        "Layer:",
        selected_layer,
    )
    print(
        "Validation:",
        selected_result[
            "validation"
        ],
    )
    print(
        "Test:",
        selected_result["test"],
    )
    print(
        "Saved results:",
        results_path,
    )
    print(
        "Saved trajectory:",
        trajectory_path,
    )
    print(
        "Saved all probes:",
        ALL_PROBE_DIR,
    )
    print(
        "Canonical selected probe:",
        selected_model_path,
    )
    print(
        "ALL-LAYER FROZEN PROBES SUCCEEDED"
    )


if __name__ == "__main__":
    main()
