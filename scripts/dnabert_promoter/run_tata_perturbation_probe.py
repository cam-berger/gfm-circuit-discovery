"""Score controlled TATA-like motif perturbations with the fixed Layer-2 probe."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.stats import wilcoxon

from gfmcd.adapters.dnabert import DnaBertAdapter

from run_head_restoration_pilot import (
    MODEL_NAME,
    MODEL_REVISION,
    PROBE_PATH,
)

SELECTED_LAYER = 2

DEFAULT_INPUT_DIR = Path(
    "results/dnabert_promoter/"
    "frozen_claim_tests/biological_features/"
    "tata_perturbations"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/dnabert_promoter/"
    "frozen_claim_tests/biological_features/"
    "tata_perturbations/probe_results"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        required=True,
    )

    parser.add_argument(
        "--cohort",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-name",
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    return parser.parse_args()


def masked_mean(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(
        hidden.dtype
    )

    denominator = mask.sum(dim=1).clamp_min(
        1.0
    )

    return (hidden * mask).sum(
        dim=1
    ) / denominator


def score_sequences(
    adapter: DnaBertAdapter,
    probe,
    sequences: list[str],
    batch_size: int,
) -> np.ndarray:
    logits = []

    for start in range(
        0,
        len(sequences),
        batch_size,
    ):
        batch = sequences[
            start:start + batch_size
        ]

        tokenized = adapter.tokenize(batch)

        with torch.inference_mode():
            output = adapter.forward(
                tokenized,
                output_hidden_states=True,
                output_attentions=False,
            )

        if output.hidden_states is None:
            raise RuntimeError(
                "Model did not return hidden states"
            )

        hidden = output.hidden_states[
            SELECTED_LAYER
        ]

        pooled = masked_mean(
            hidden,
            tokenized.attention_mask,
        )

        batch_logits = probe.decision_function(
            pooled.detach().cpu().numpy()
        )

        logits.extend(
            np.asarray(
                batch_logits,
                dtype=float,
            ).tolist()
        )

        print(
            f"\rProcessed "
            f"{min(start + batch_size, len(sequences))}"
            f"/{len(sequences)}",
            end="",
            flush=True,
        )

    print()

    return np.asarray(
        logits,
        dtype=float,
    )


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": (
            float(values.std(ddof=1))
            if values.size > 1
            else 0.0
        ),
        "median": float(
            np.median(values)
        ),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "fraction_positive": float(
            np.mean(values > 0)
        ),
        "fraction_negative": float(
            np.mean(values < 0)
        ),
    }


def paired_test(
    first: np.ndarray,
    second: np.ndarray,
    alternative: str,
) -> dict[str, float]:
    delta = first - second

    result = wilcoxon(
        delta,
        alternative=alternative,
        zero_method="wilcox",
    )

    return {
        "n": int(delta.size),
        "mean_first": float(
            first.mean()
        ),
        "mean_second": float(
            second.mean()
        ),
        "mean_difference": float(
            delta.mean()
        ),
        "median_difference": float(
            np.median(delta)
        ),
        "fraction_first_greater": float(
            np.mean(delta > 0)
        ),
        "wilcoxon_statistic": float(
            result.statistic
        ),
        "p_value": float(
            result.pvalue
        ),
    }


def main() -> None:
    args = parse_args()

    cohort_path = (
        args.cohort
        if args.cohort is not None
        else DEFAULT_INPUT_DIR
        / (
            f"{args.split}_"
            "tata_perturbation_cohort.csv"
        )
    )

    output_name = (
        args.output_name
        or f"{args.split}_tata_perturbation"
    )

    DEFAULT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with cohort_path.open(
        encoding="utf-8"
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if not rows:
        raise RuntimeError(
            "Perturbation cohort is empty"
        )

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

    probe = joblib.load(PROBE_PATH)

    sequence_columns = {
        "clean": "clean_sequence",
        "motif_destroyed": (
            "motif_destroyed_sequence"
        ),
        "matched_nonmotif": (
            "matched_nonmotif_sequence"
        ),
        "identity": "identity_sequence",
    }

    scores = {}

    for condition, column in (
        sequence_columns.items()
    ):
        print(f"\nScoring {condition}...")

        scores[condition] = score_sequences(
            adapter,
            probe,
            [
                row[column]
                for row in rows
            ],
            args.batch_size,
        )

    clean = scores["clean"]
    motif = scores["motif_destroyed"]
    control = scores["matched_nonmotif"]
    identity = scores["identity"]

    motif_damage = clean - motif
    control_damage = clean - control

    excess_motif_damage = (
        motif_damage - control_damage
    )

    identity_difference = (
        identity - clean
    )

    result_rows = []

    for index, row in enumerate(rows):
        result_rows.append(
            {
                "row_index": index,
                "split": args.split,
                "source_index": (
                    row["source_index"]
                ),
                "motif": row["motif"],
                "motif_start": (
                    row["motif_start"]
                ),
                "motif_edit_positions": (
                    row[
                        "motif_edit_positions"
                    ]
                ),
                "control_edit_positions": (
                    row[
                        "control_edit_positions"
                    ]
                ),
                "clean_logit": clean[index],
                "motif_destroyed_logit": (
                    motif[index]
                ),
                "matched_nonmotif_logit": (
                    control[index]
                ),
                "identity_logit": (
                    identity[index]
                ),
                "motif_damage": (
                    motif_damage[index]
                ),
                "control_damage": (
                    control_damage[index]
                ),
                "excess_motif_damage": (
                    excess_motif_damage[
                        index
                    ]
                ),
                "identity_difference": (
                    identity_difference[
                        index
                    ]
                ),
            }
        )

    summary = {
        "split": args.split,
        "cohort": str(cohort_path),
        "rows": len(rows),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": (
            SELECTED_LAYER
        ),
        "clean_logit": summarize(clean),
        "motif_destroyed_logit": (
            summarize(motif)
        ),
        "matched_nonmotif_logit": (
            summarize(control)
        ),
        "motif_damage": summarize(
            motif_damage
        ),
        "control_damage": summarize(
            control_damage
        ),
        "excess_motif_damage": (
            summarize(
                excess_motif_damage
            )
        ),
        "tests": {
            "motif_damage_greater_than_zero": (
                paired_test(
                    motif_damage,
                    np.zeros_like(
                        motif_damage
                    ),
                    "greater",
                )
            ),
            "control_damage_greater_than_zero": (
                paired_test(
                    control_damage,
                    np.zeros_like(
                        control_damage
                    ),
                    "greater",
                )
            ),
            "motif_damage_greater_than_control": (
                paired_test(
                    motif_damage,
                    control_damage,
                    "greater",
                )
            ),
        },
        "identity_max_absolute_difference": (
            float(
                np.max(
                    np.abs(
                        identity_difference
                    )
                )
            )
        ),
    }

    per_example_path = (
        DEFAULT_OUTPUT_DIR
        / f"{output_name}_per_example.csv"
    )

    summary_path = (
        DEFAULT_OUTPUT_DIR
        / f"{output_name}_summary.json"
    )

    with per_example_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                result_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(result_rows)

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "\n===== TATA perturbation summary ====="
    )
    print(
        "Mean motif damage:",
        f"{motif_damage.mean():.6f}",
    )
    print(
        "Mean matched-control damage:",
        f"{control_damage.mean():.6f}",
    )
    print(
        "Mean excess motif damage:",
        f"{excess_motif_damage.mean():.6f}",
    )
    print(
        "Motif > control p-value:",
        summary["tests"][
            "motif_damage_greater_than_control"
        ]["p_value"],
    )
    print(
        "Fraction motif more damaging:",
        f"{np.mean(excess_motif_damage > 0):.3f}",
    )
    print(
        "Identity maximum difference:",
        summary[
            "identity_max_absolute_difference"
        ],
    )

    print("\nSaved:", per_example_path)
    print("Saved:", summary_path)
    print(
        "TATA PERTURBATION PROBE SUCCEEDED"
    )


if __name__ == "__main__":
    main()
