"""Measure clean and corruption behavior across all frozen DNABERT layers."""

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
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from gfmcd.adapters.dnabert import DnaBertAdapter
from gfmcd.dnabert.manifest import write_run_manifest


MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"
MODEL_REVISION = "813031b2bf86d9e960a027e2734c908009f31601"

PROBE_DIR = Path(
    "results/dnabert_promoter/frozen/"
    "layer_probes/all_layer_probes"
)

OUTPUT_DIR = Path(
    "results/dnabert_promoter/behavioral/"
    "all_layer_corruption"
)

NUM_LAYERS = 12


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


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def extract_all_layers(
    adapter: DnaBertAdapter,
    sequences: list[str],
    *,
    batch_size: int,
) -> np.ndarray:
    batches = []

    for start in range(
        0,
        len(sequences),
        batch_size,
    ):
        end = min(
            start + batch_size,
            len(sequences),
        )

        tokenized = adapter.tokenize(
            sequences[start:end]
        )

        with torch.inference_mode():
            output = adapter.forward(
                tokenized,
                output_hidden_states=True,
                output_attentions=False,
            )

        if output.hidden_states is None:
            raise RuntimeError(
                "Model returned no hidden states"
            )

        if len(output.hidden_states) != NUM_LAYERS:
            raise RuntimeError(
                "Unexpected hidden-state count: "
                f"{len(output.hidden_states)}"
            )

        biological_mask = (
            adapter.biological_token_mask(
                tokenized
            )
        )

        pooled_layers = []

        for hidden in output.hidden_states:
            mask = biological_mask.to(
                hidden.dtype
            ).unsqueeze(-1)

            pooled = (
                (hidden * mask).sum(dim=1)
                / mask.sum(dim=1).clamp_min(1.0)
            )

            pooled_layers.append(
                pooled.cpu().float().numpy()
            )

        batch_features = np.stack(
            pooled_layers,
            axis=1,
        )

        batches.append(batch_features)

        print(
            f"\rProcessed {end:,}/{len(sequences):,}",
            end="",
            flush=True,
        )

    print()

    return np.concatenate(
        batches,
        axis=0,
    )


def metrics(
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
    }


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(
            values.std(ddof=1)
        ),
        "median": float(
            np.median(values)
        ),
        "minimum": float(
            values.min()
        ),
        "maximum": float(
            values.max()
        ),
        "fraction_positive": float(
            np.mean(values > 0)
        ),
        "fraction_negative": float(
            np.mean(values < 0)
        ),
    }


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_rows(args.cohort)

    labels = np.asarray(
        [
            int(row["label"])
            for row in rows
        ],
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

    print("Rows:", len(rows))
    print("Loading frozen DNABERT...")

    adapter = DnaBertAdapter.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=args.device,
    )

    adapter.model.eval()

    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)

    features = {}

    for condition, sequences in sequence_sets.items():
        print(f"\n===== {condition} =====")

        features[condition] = extract_all_layers(
            adapter,
            sequences,
            batch_size=args.batch_size,
        )

        expected_shape = (
            len(rows),
            NUM_LAYERS,
            adapter.hidden_size,
        )

        if features[condition].shape != expected_shape:
            raise RuntimeError(
                f"Unexpected {condition} shape: "
                f"{features[condition].shape}; "
                f"expected {expected_shape}"
            )

    trajectory_rows = []
    layer_summaries = []

    for layer in range(NUM_LAYERS):
        print(f"\nScoring Layer {layer}")

        probe_path = (
            PROBE_DIR
            / f"layer_{layer}_probe.joblib"
        )

        probe = joblib.load(probe_path)

        layer_logits = {}
        layer_probabilities = {}
        layer_predictions = {}
        layer_metrics = {}

        for condition in sequence_sets:
            layer_features = features[
                condition
            ][:, layer, :]

            layer_logits[condition] = (
                probe.decision_function(
                    layer_features
                )
            )

            layer_probabilities[condition] = (
                probe.predict_proba(
                    layer_features
                )[:, 1]
            )

            layer_predictions[condition] = (
                probe.predict(
                    layer_features
                )
            )

            layer_metrics[condition] = metrics(
                labels,
                layer_predictions[
                    condition
                ],
                layer_probabilities[
                    condition
                ],
            )

        mono_change = (
            layer_logits["mononucleotide"]
            - layer_logits["clean"]
        )

        di_change = (
            layer_logits["dinucleotide"]
            - layer_logits["clean"]
        )

        promoter_mask = labels == 1
        nonpromoter_mask = labels == 0

        layer_summary = {
            "layer": layer,
            "probe_file": str(
                probe_path
            ),
            "metrics": layer_metrics,
            "mono_logit_change": {
                "overall": summarize(
                    mono_change
                ),
                "promoter": summarize(
                    mono_change[
                        promoter_mask
                    ]
                ),
                "nonpromoter": summarize(
                    mono_change[
                        nonpromoter_mask
                    ]
                ),
            },
            "dinucleotide_logit_change": {
                "overall": summarize(
                    di_change
                ),
                "promoter": summarize(
                    di_change[
                        promoter_mask
                    ]
                ),
                "nonpromoter": summarize(
                    di_change[
                        nonpromoter_mask
                    ]
                ),
            },
        }

        layer_summaries.append(
            layer_summary
        )

        trajectory_rows.append(
            {
                "layer": layer,
                "clean_balanced_accuracy": (
                    layer_metrics["clean"][
                        "balanced_accuracy"
                    ]
                ),
                "mono_balanced_accuracy": (
                    layer_metrics[
                        "mononucleotide"
                    ][
                        "balanced_accuracy"
                    ]
                ),
                "dinucleotide_balanced_accuracy": (
                    layer_metrics[
                        "dinucleotide"
                    ][
                        "balanced_accuracy"
                    ]
                ),
                "clean_roc_auc": (
                    layer_metrics["clean"][
                        "roc_auc"
                    ]
                ),
                "mono_roc_auc": (
                    layer_metrics[
                        "mononucleotide"
                    ][
                        "roc_auc"
                    ]
                ),
                "dinucleotide_roc_auc": (
                    layer_metrics[
                        "dinucleotide"
                    ][
                        "roc_auc"
                    ]
                ),
                "mono_promoter_mean_logit_change": float(
                    mono_change[
                        promoter_mask
                    ].mean()
                ),
                "mono_nonpromoter_mean_logit_change": float(
                    mono_change[
                        nonpromoter_mask
                    ].mean()
                ),
                "dinucleotide_promoter_mean_logit_change": float(
                    di_change[
                        promoter_mask
                    ].mean()
                ),
                "dinucleotide_nonpromoter_mean_logit_change": float(
                    di_change[
                        nonpromoter_mask
                    ].mean()
                ),
                "mono_balanced_accuracy_drop": float(
                    layer_metrics["clean"][
                        "balanced_accuracy"
                    ]
                    - layer_metrics[
                        "mononucleotide"
                    ][
                        "balanced_accuracy"
                    ]
                ),
                "dinucleotide_balanced_accuracy_drop": float(
                    layer_metrics["clean"][
                        "balanced_accuracy"
                    ]
                    - layer_metrics[
                        "dinucleotide"
                    ][
                        "balanced_accuracy"
                    ]
                ),
            }
        )

        print(
            "Clean BA:",
            f"{layer_metrics['clean']['balanced_accuracy']:.6f}",
        )

        print(
            "Mono BA:",
            f"{layer_metrics['mononucleotide']['balanced_accuracy']:.6f}",
        )

        print(
            "Di BA:",
            f"{layer_metrics['dinucleotide']['balanced_accuracy']:.6f}",
        )

    csv_path = (
        OUTPUT_DIR
        / f"{args.output_name}_trajectory.csv"
    )

    json_path = (
        OUTPUT_DIR
        / f"{args.output_name}_summary.json"
    )

    with csv_path.open(
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

    summary = {
        "cohort": str(
            args.cohort
        ),
        "rows": len(rows),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "num_layers": NUM_LAYERS,
        "hidden_state_mapping": (
            "output.hidden_states[layer]"
        ),
        "layer_summaries": (
            layer_summaries
        ),
    }

    json_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nSaved:", csv_path)
    print("Saved:", json_path)

    # Sibling provenance record. Written after the outputs exist so their hashes are real.
    manifest_path = write_run_manifest(
        json_path,
        experiment_id="dnabert_promoter_all_layer_corruption",
        config=vars(args),
        outputs=[csv_path],
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
    )
    print("Saved:", manifest_path)

    print(
        "ALL-LAYER CORRUPTION TRAJECTORY SUCCEEDED"
    )


if __name__ == "__main__":
    main()
