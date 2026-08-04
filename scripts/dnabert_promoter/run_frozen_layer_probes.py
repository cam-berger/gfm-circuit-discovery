"""Train and evaluate one linear probe per frozen DNABERT layer."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
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

SEED = 42


def load_split(split: str) -> tuple[np.ndarray, np.ndarray]:
    path = REPRESENTATION_DIR / f"{split}.npz"

    with np.load(path) as payload:
        features = payload["features"].copy()
        labels = payload["labels"].copy()

    return features, labels


def compute_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
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
    }


def evaluate(
    model,
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    return compute_metrics(
        labels,
        predictions,
        probabilities,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_features, train_labels = load_split("train")
    validation_features, validation_labels = load_split("validation")
    test_features, test_labels = load_split("test")

    expected_layers = 12
    expected_hidden_size = 768

    for name, features in (
        ("train", train_features),
        ("validation", validation_features),
        ("test", test_features),
    ):
        if features.ndim != 3:
            raise ValueError(
                f"{name} features must have shape "
                "[rows, layers, hidden], got "
                f"{features.shape}"
            )

        if features.shape[1:] != (
            expected_layers,
            expected_hidden_size,
        ):
            raise ValueError(
                f"Unexpected {name} feature shape: "
                f"{features.shape}"
            )

        if not np.isfinite(features).all():
            raise ValueError(
                f"{name} contains non-finite features"
            )

    layer_results: list[dict] = []
    fitted_models: dict[int, object] = {}

    for layer in range(expected_layers):
        print(f"\n===== Layer {layer} =====")

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=SEED,
                solver="lbfgs",
            ),
        )

        model.fit(
            train_features[:, layer, :],
            train_labels,
        )

        validation_metrics = evaluate(
            model,
            validation_features[:, layer, :],
            validation_labels,
        )

        layer_results.append(
            {
                "layer": layer,
                "validation": validation_metrics,
            }
        )

        fitted_models[layer] = model

        print("Validation:", validation_metrics)

    ranked_layers = sorted(
        layer_results,
        key=lambda result: (
            result["validation"]["balanced_accuracy"],
            result["validation"]["roc_auc"],
        ),
        reverse=True,
    )

    selected_result = ranked_layers[0]
    selected_layer = int(selected_result["layer"])
    selected_model = fitted_models[selected_layer]

    test_metrics = evaluate(
        selected_model,
        test_features[:, selected_layer, :],
        test_labels,
    )

    results = {
        "seed": SEED,
        "classifier": "standardized_logistic_regression",
        "selection_metric": "validation_balanced_accuracy",
        "tie_breaker": "validation_roc_auc",
        "num_layers": expected_layers,
        "hidden_size": expected_hidden_size,
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
        "selected_layer": selected_layer,
        "selected_validation_metrics": selected_result[
            "validation"
        ],
        "selected_test_metrics": test_metrics,
    }

    results_path = (
        OUTPUT_DIR / "layer_probe_results.json"
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

    model_path = (
        OUTPUT_DIR
        / f"selected_layer_{selected_layer}_probe.joblib"
    )

    joblib.dump(selected_model, model_path)

    print("\n===== Selected layer =====")
    print("Layer:", selected_layer)
    print(
        "Validation:",
        selected_result["validation"],
    )
    print("Test:", test_metrics)
    print("Saved results:", results_path)
    print("Saved model:", model_path)
    print("FROZEN LAYER PROBES SUCCEEDED")


if __name__ == "__main__":
    main()
