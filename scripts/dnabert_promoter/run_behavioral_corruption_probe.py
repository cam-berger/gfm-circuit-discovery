"""Evaluate clean and composition-preserving corruptions with the Layer 2 probe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from gfmcd.adapters.dnabert import DnaBertAdapter


MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"
MODEL_REVISION = "813031b2bf86d9e960a027e2734c908009f31601"

PROBE_PATH = Path(
    "results/dnabert_promoter/frozen/"
    "layer_probes/selected_layer_2_probe.joblib"
)

OUTPUT_DIR = Path(
    "results/dnabert_promoter/behavioral/probe_results"
)

SELECTED_LAYER = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-name",
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    return parser.parse_args()


def load_cohort(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pooled_layer_features(
    adapter: DnaBertAdapter,
    sequences: list[str],
    *,
    layer: int,
    batch_size: int,
) -> np.ndarray:
    feature_batches = []

    for start in range(0, len(sequences), batch_size):
        end = min(start + batch_size, len(sequences))
        batch = sequences[start:end]

        tokenized = adapter.tokenize(batch)

        with torch.inference_mode():
            output = adapter.forward(
                tokenized,
                output_hidden_states=True,
                output_attentions=False,
            )

            if output.hidden_states is None:
                raise RuntimeError(
                    "adapter did not return hidden states"
                )

            biological_mask = (
                adapter.biological_token_mask(tokenized)
            )

            hidden = output.hidden_states[layer]

            mask = biological_mask.to(
                dtype=hidden.dtype,
            ).unsqueeze(-1)

            denominator = mask.sum(dim=1).clamp_min(1.0)

            pooled = (
                hidden * mask
            ).sum(dim=1) / denominator

        feature_batches.append(
            pooled.cpu().to(torch.float32).numpy()
        )

        print(
            f"\rProcessed {end:,}/{len(sequences):,}",
            end="",
            flush=True,
        )

    print()

    return np.concatenate(feature_batches, axis=0)


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions,
            )
        ),
        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),
    }


def summarize_changes(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "fraction_negative": float(
            np.mean(values < 0)
        ),
        "fraction_positive": float(
            np.mean(values > 0)
        ),
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_cohort(args.cohort)

    labels = np.asarray(
        [int(row["label"]) for row in rows],
        dtype=np.int64,
    )

    sequence_sets = {
        "clean": [
            row["clean_sequence"]
            for row in rows
        ],
        "mononucleotide": [
            row["mono_sequence"]
            for row in rows
        ],
        "dinucleotide": [
            row["dinucleotide_sequence"]
            for row in rows
        ],
    }

    print("Loading model...")

    adapter = DnaBertAdapter.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=args.device,
    )

    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)

    adapter.model.eval()

    probe = joblib.load(PROBE_PATH)

    features = {}
    logits = {}
    probabilities = {}
    predictions = {}
    metrics = {}

    for condition, sequences in sequence_sets.items():
        print(f"\n===== {condition} =====")

        condition_features = pooled_layer_features(
            adapter,
            sequences,
            layer=SELECTED_LAYER,
            batch_size=args.batch_size,
        )

        condition_probabilities = probe.predict_proba(
            condition_features
        )[:, 1]

        condition_logits = probe.decision_function(
            condition_features
        )

        condition_predictions = probe.predict(
            condition_features
        )

        features[condition] = condition_features
        logits[condition] = condition_logits
        probabilities[condition] = condition_probabilities
        predictions[condition] = condition_predictions

        metrics[condition] = classification_metrics(
            labels,
            condition_predictions,
            condition_probabilities,
        )

        print("Metrics:", metrics[condition])

    mono_logit_change = (
        logits["mononucleotide"]
        - logits["clean"]
    )

    di_logit_change = (
        logits["dinucleotide"]
        - logits["clean"]
    )

    mono_probability_change = (
        probabilities["mononucleotide"]
        - probabilities["clean"]
    )

    di_probability_change = (
        probabilities["dinucleotide"]
        - probabilities["clean"]
    )

    per_example_rows = []

    for index, row in enumerate(rows):
        per_example_rows.append(
            {
                "source_index": row["source_index"],
                "label": int(row["label"]),
                "clean_logit": float(
                    logits["clean"][index]
                ),
                "mono_logit": float(
                    logits["mononucleotide"][index]
                ),
                "dinucleotide_logit": float(
                    logits["dinucleotide"][index]
                ),
                "mono_logit_change": float(
                    mono_logit_change[index]
                ),
                "dinucleotide_logit_change": float(
                    di_logit_change[index]
                ),
                "clean_probability": float(
                    probabilities["clean"][index]
                ),
                "mono_probability": float(
                    probabilities["mononucleotide"][index]
                ),
                "dinucleotide_probability": float(
                    probabilities["dinucleotide"][index]
                ),
                "mono_probability_change": float(
                    mono_probability_change[index]
                ),
                "dinucleotide_probability_change": float(
                    di_probability_change[index]
                ),
                "mono_changed_fraction": float(
                    row["mono_changed_fraction"]
                ),
                "dinucleotide_changed_fraction": float(
                    row["dinucleotide_changed_fraction"]
                ),
            }
        )

    summary = {
        "cohort": str(args.cohort),
        "rows": len(rows),
        "selected_layer": SELECTED_LAYER,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "metrics": metrics,
        "logit_changes": {
            "mononucleotide_minus_clean": (
                summarize_changes(
                    mono_logit_change
                )
            ),
            "dinucleotide_minus_clean": (
                summarize_changes(
                    di_logit_change
                )
            ),
        },
        "probability_changes": {
            "mononucleotide_minus_clean": (
                summarize_changes(
                    mono_probability_change
                )
            ),
            "dinucleotide_minus_clean": (
                summarize_changes(
                    di_probability_change
                )
            ),
        },
        "class_specific_logit_changes": {
            "label_0": {
                "mononucleotide_minus_clean": (
                    summarize_changes(
                        mono_logit_change[
                            labels == 0
                        ]
                    )
                ),
                "dinucleotide_minus_clean": (
                    summarize_changes(
                        di_logit_change[
                            labels == 0
                        ]
                    )
                ),
            },
            "label_1": {
                "mononucleotide_minus_clean": (
                    summarize_changes(
                        mono_logit_change[
                            labels == 1
                        ]
                    )
                ),
                "dinucleotide_minus_clean": (
                    summarize_changes(
                        di_logit_change[
                            labels == 1
                        ]
                    )
                ),
            },
        },
    }

    summary_path = OUTPUT_DIR / (
        f"{args.output_name}_summary.json"
    )

    detail_path = OUTPUT_DIR / (
        f"{args.output_name}_per_example.csv"
    )

    feature_path = OUTPUT_DIR / (
        f"{args.output_name}_features.npz"
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

    with detail_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                per_example_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(per_example_rows)

    np.savez_compressed(
        feature_path,
        clean_features=features["clean"],
        mononucleotide_features=(
            features["mononucleotide"]
        ),
        dinucleotide_features=(
            features["dinucleotide"]
        ),
        labels=labels,
    )

    print("\n===== Behavioral summary =====")
    print(json.dumps(summary, indent=2))
    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print("Saved:", feature_path)
    print("BEHAVIORAL CORRUPTION PROBE SUCCEEDED")


if __name__ == "__main__":
    main()
