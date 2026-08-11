"""Run non-neural baselines on the clean promoter cohort."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("results/dnabert_promoter/shared/clean")
OUTPUT_DIR = Path("results/dnabert_promoter/frozen/baselines")
SEED = 42


def load_split(split: str) -> tuple[list[str], np.ndarray]:
    path = DATA_DIR / f"{split}.csv"

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    sequences = [row["sequence"] for row in rows]
    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int64,
    )

    return sequences, labels


def gc_features(sequences: list[str]) -> np.ndarray:
    rows = []

    for sequence in sequences:
        length = len(sequence)
        gc = sum(base in "GC" for base in sequence) / length
        g = sequence.count("G") / length
        c = sequence.count("C") / length
        a = sequence.count("A") / length
        t = sequence.count("T") / length

        rows.append([gc, a, c, g, t])

    return np.asarray(rows, dtype=np.float64)


def metrics(
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


def evaluate_classifier(
    model,
    train_x,
    train_y,
    validation_x,
    validation_y,
    test_x,
    test_y,
) -> dict:
    model.fit(train_x, train_y)

    validation_predictions = model.predict(validation_x)
    validation_probabilities = model.predict_proba(validation_x)[:, 1]

    test_predictions = model.predict(test_x)
    test_probabilities = model.predict_proba(test_x)[:, 1]

    return {
        "validation": metrics(
            validation_y,
            validation_predictions,
            validation_probabilities,
        ),
        "test": metrics(
            test_y,
            test_predictions,
            test_probabilities,
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_sequences, train_labels = load_split("train")
    validation_sequences, validation_labels = load_split(
        "validation"
    )
    test_sequences, test_labels = load_split("test")

    majority_label = Counter(train_labels.tolist()).most_common(1)[0][0]

    majority_results = {}

    for split_name, labels in (
        ("validation", validation_labels),
        ("test", test_labels),
    ):
        predictions = np.full_like(labels, majority_label)
        probabilities = np.full(
            labels.shape,
            float(majority_label),
            dtype=np.float64,
        )

        majority_results[split_name] = {
            "accuracy": float(
                accuracy_score(labels, predictions)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(labels, predictions)
            ),
            "roc_auc": 0.5,
        }

    gc_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=SEED,
        ),
    )

    gc_results = evaluate_classifier(
        gc_model,
        gc_features(train_sequences),
        train_labels,
        gc_features(validation_sequences),
        validation_labels,
        gc_features(test_sequences),
        test_labels,
    )

    kmer_model = make_pipeline(
        CountVectorizer(
            analyzer="char",
            ngram_range=(3, 6),
            lowercase=False,
            min_df=2,
            max_features=100000,
        ),
        LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            n_jobs=-1,
        ),
    )

    kmer_results = evaluate_classifier(
        kmer_model,
        train_sequences,
        train_labels,
        validation_sequences,
        validation_labels,
        test_sequences,
        test_labels,
    )

    results = {
        "seed": SEED,
        "data_dir": str(DATA_DIR),
        "majority": majority_results,
        "gc_composition": gc_results,
        "kmer_3_to_6": kmer_results,
    }

    output_path = OUTPUT_DIR / "sequence_baselines.json"
    output_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for name, result in results.items():
        if not isinstance(result, dict) or "validation" not in result:
            continue

        print(f"\n===== {name} =====")
        print("Validation:", result["validation"])
        print("Test:", result["test"])

    print("\nSaved:", output_path)
    print("SEQUENCE BASELINES SUCCEEDED")


if __name__ == "__main__":
    main()
