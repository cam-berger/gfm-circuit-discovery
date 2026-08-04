"""Run missing control experiments for the frozen DNABERT promoter study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
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
REPRESENTATION_DIR = Path(
    "results/dnabert_promoter/frozen/representations"
)
OUTPUT_DIR = Path(
    "results/dnabert_promoter/frozen/baseline_controls"
)

SELECTED_LAYER = 2
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--shuffle-repetitions",
        type=int,
        default=20,
    )

    return parser.parse_args()


def load_sequences(
    split: str,
) -> tuple[list[str], np.ndarray]:
    path = DATA_DIR / f"{split}.csv"

    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    sequences = [row["sequence"] for row in rows]
    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int64,
    )

    return sequences, labels


def load_layer_features(
    split: str,
    layer: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = REPRESENTATION_DIR / f"{split}.npz"

    with np.load(path) as payload:
        features = payload["features"][:, layer, :].copy()
        labels = payload["labels"].copy()

    return features, labels


def calculate_metrics(
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


def evaluate_model(
    model,
    train_x,
    train_y: np.ndarray,
    validation_x,
    validation_y: np.ndarray,
    test_x,
    test_y: np.ndarray,
) -> dict:
    model.fit(train_x, train_y)

    validation_predictions = model.predict(validation_x)
    validation_probabilities = model.predict_proba(
        validation_x
    )[:, 1]

    test_predictions = model.predict(test_x)
    test_probabilities = model.predict_proba(
        test_x
    )[:, 1]

    return {
        "validation": calculate_metrics(
            validation_y,
            validation_predictions,
            validation_probabilities,
        ),
        "test": calculate_metrics(
            test_y,
            test_predictions,
            test_probabilities,
        ),
    }


def gc_features(
    sequences: list[str],
) -> np.ndarray:
    rows = []

    for sequence in sequences:
        length = len(sequence)

        rows.append(
            [
                sequence.count("A") / length,
                sequence.count("C") / length,
                sequence.count("G") / length,
                sequence.count("T") / length,
                sum(base in "GC" for base in sequence) / length,
            ]
        )

    return np.asarray(rows, dtype=np.float64)


def run_shuffled_label_controls(
    repetitions: int,
) -> dict:
    train_x, train_y = load_layer_features(
        "train",
        SELECTED_LAYER,
    )

    validation_x, validation_y = load_layer_features(
        "validation",
        SELECTED_LAYER,
    )

    runs = []

    for repetition in range(repetitions):
        run_seed = SEED + repetition
        rng = np.random.default_rng(run_seed)

        shuffled_train_y = rng.permutation(train_y)

        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                random_state=run_seed,
                solver="lbfgs",
            ),
        )

        model.fit(train_x, shuffled_train_y)

        predictions = model.predict(validation_x)
        probabilities = model.predict_proba(
            validation_x
        )[:, 1]

        run_metrics = calculate_metrics(
            validation_y,
            predictions,
            probabilities,
        )

        runs.append(
            {
                "repetition": repetition,
                "seed": run_seed,
                **run_metrics,
            }
        )

        print(
            f"Shuffle {repetition + 1:02d}/{repetitions}: "
            f"balanced_accuracy="
            f"{run_metrics['balanced_accuracy']:.4f}, "
            f"roc_auc={run_metrics['roc_auc']:.4f}"
        )

    balanced_accuracies = np.asarray(
        [run["balanced_accuracy"] for run in runs]
    )

    roc_aucs = np.asarray(
        [run["roc_auc"] for run in runs]
    )

    return {
        "selected_layer": SELECTED_LAYER,
        "repetitions": repetitions,
        "runs": runs,
        "summary": {
            "mean_balanced_accuracy": float(
                balanced_accuracies.mean()
            ),
            "std_balanced_accuracy": float(
                balanced_accuracies.std(ddof=1)
            ),
            "minimum_balanced_accuracy": float(
                balanced_accuracies.min()
            ),
            "maximum_balanced_accuracy": float(
                balanced_accuracies.max()
            ),
            "mean_roc_auc": float(
                roc_aucs.mean()
            ),
            "std_roc_auc": float(
                roc_aucs.std(ddof=1)
            ),
            "minimum_roc_auc": float(
                roc_aucs.min()
            ),
            "maximum_roc_auc": float(
                roc_aucs.max()
            ),
        },
    }


def run_exact_kmer_baseline(
    k: int,
    train_sequences: list[str],
    train_labels: np.ndarray,
    validation_sequences: list[str],
    validation_labels: np.ndarray,
    test_sequences: list[str],
    test_labels: np.ndarray,
) -> dict:
    model = make_pipeline(
        CountVectorizer(
            analyzer="char",
            ngram_range=(k, k),
            lowercase=False,
        ),
        LogisticRegression(
            max_iter=2000,
            random_state=SEED,
            solver="lbfgs",
        ),
    )

    return evaluate_model(
        model,
        train_sequences,
        train_labels,
        validation_sequences,
        validation_labels,
        test_sequences,
        test_labels,
    )


def run_gc_plus_3mer(
    train_sequences: list[str],
    train_labels: np.ndarray,
    validation_sequences: list[str],
    validation_labels: np.ndarray,
    test_sequences: list[str],
    test_labels: np.ndarray,
) -> dict:
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(3, 3),
        lowercase=False,
    )

    train_kmers = vectorizer.fit_transform(
        train_sequences
    )

    validation_kmers = vectorizer.transform(
        validation_sequences
    )

    test_kmers = vectorizer.transform(
        test_sequences
    )

    train_gc = csr_matrix(
        gc_features(train_sequences)
    )

    validation_gc = csr_matrix(
        gc_features(validation_sequences)
    )

    test_gc = csr_matrix(
        gc_features(test_sequences)
    )

    train_x = hstack(
        [train_kmers, train_gc],
        format="csr",
    )

    validation_x = hstack(
        [validation_kmers, validation_gc],
        format="csr",
    )

    test_x = hstack(
        [test_kmers, test_gc],
        format="csr",
    )

    model = LogisticRegression(
        max_iter=2000,
        random_state=SEED,
        solver="lbfgs",
    )

    results = evaluate_model(
        model,
        train_x,
        train_labels,
        validation_x,
        validation_labels,
        test_x,
        test_labels,
    )

    results["feature_count"] = int(
        train_x.shape[1]
    )

    return results


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_sequences, train_labels = load_sequences(
        "train"
    )

    validation_sequences, validation_labels = load_sequences(
        "validation"
    )

    test_sequences, test_labels = load_sequences(
        "test"
    )

    print("===== Shuffled-label Layer 2 controls =====")

    shuffled_results = run_shuffled_label_controls(
        args.shuffle_repetitions
    )

    kmer_results = {}

    for k in (1, 2, 3):
        print(f"\n===== Exact {k}-mer baseline =====")

        result = run_exact_kmer_baseline(
            k,
            train_sequences,
            train_labels,
            validation_sequences,
            validation_labels,
            test_sequences,
            test_labels,
        )

        kmer_results[f"{k}mer"] = result

        print("Validation:", result["validation"])
        print("Test:", result["test"])

    print("\n===== GC + exact 3-mer baseline =====")

    gc_plus_3mer = run_gc_plus_3mer(
        train_sequences,
        train_labels,
        validation_sequences,
        validation_labels,
        test_sequences,
        test_labels,
    )

    print(
        "Validation:",
        gc_plus_3mer["validation"],
    )

    print(
        "Test:",
        gc_plus_3mer["test"],
    )

    results = {
        "seed": SEED,
        "selected_layer": SELECTED_LAYER,
        "shuffled_label_control": shuffled_results,
        "exact_kmer_baselines": kmer_results,
        "gc_plus_exact_3mer": gc_plus_3mer,
    }

    output_path = (
        OUTPUT_DIR
        / "missing_baseline_controls.json"
    )

    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n===== Shuffled-label summary =====")
    print(
        shuffled_results["summary"]
    )

    print("\nSaved:", output_path)
    print("MISSING BASELINE CONTROLS SUCCEEDED")


if __name__ == "__main__":
    main()
