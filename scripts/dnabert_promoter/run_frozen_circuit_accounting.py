"""Measure delivery and necessity for the frozen DNABERT Layer-2 circuit.

Candidate circuit:
- Layer-0 MLP output
- L0H8
- L1H8
- L2H9
- L2H3

Readout:
- masked-mean encoder Layer-2 representation
- canonical frozen Layer-2 logistic-regression probe
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

from gfmcd.adapters.dnabert import DnaBertAdapter

from run_combined_mlp_head_restoration_controls import (
    HEAD_GROUP,
    MLP_LAYER,
    capture_components_and_logit,
    load_promoters,
    run_combined_patch,
    summarize,
)

from run_head_restoration_pilot import (
    DEFAULT_COHORT,
    MODEL_NAME,
    MODEL_REVISION,
    PROBE_PATH,
    SELECTED_LAYER,
    assert_adapter_tokenization_matches,
    get_corrupt_sequence,
    tokenize_with_offsets,
)
from gfmcd.dnabert.manifest import write_run_manifest


OUTPUT_DIR = Path(
    "results/dnabert_promoter/frozen_claim_tests/"
    "circuit_accounting"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--output-name",
        default="validation_candidate_circuit_128",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--minimum-total-effect",
        type=float,
        default=1e-8,
        help=(
            "Examples with absolute clean-minus-corrupt effect "
            "below this threshold receive NaN normalized metrics."
        ),
    )

    return parser.parse_args()


def finite_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "minimum": float("nan"),
            "maximum": float("nan"),
            "fraction_positive": float("nan"),
            "fraction_negative": float("nan"),
        }

    result = summarize(finite)
    result["n"] = int(finite.size)
    return result


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_promoters(
        args.cohort,
        args.limit,
    )

    print("Rows:", len(rows))
    print("Model:", MODEL_NAME)
    print("Model revision:", MODEL_REVISION)
    print("Selected encoder layer:", SELECTED_LAYER)
    print("MLP layer:", MLP_LAYER)
    print("Circuit heads:", HEAD_GROUP)

    adapter = DnaBertAdapter.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=args.device,
    )

    adapter.model.eval()

    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)

    probe = joblib.load(PROBE_PATH)

    clean_sequences = [
        row["clean_sequence"]
        for row in rows
    ]

    corrupt_sequences = [
        get_corrupt_sequence(row)
        for row in rows
    ]

    clean_offsets: list[list[tuple[int, int]]] = []
    corrupt_offsets: list[list[tuple[int, int]]] = []

    print("\nPreparing token-offset mappings...")

    for index in range(len(rows)):
        clean_ids, clean_offset_row = tokenize_with_offsets(
            adapter,
            clean_sequences[index],
        )

        corrupt_ids, corrupt_offset_row = tokenize_with_offsets(
            adapter,
            corrupt_sequences[index],
        )

        assert_adapter_tokenization_matches(
            adapter,
            clean_sequences[index],
            clean_ids,
        )

        assert_adapter_tokenization_matches(
            adapter,
            corrupt_sequences[index],
            corrupt_ids,
        )

        clean_offsets.append(clean_offset_row)
        corrupt_offsets.append(corrupt_offset_row)

    clean_attention_outputs = []
    corrupt_attention_outputs = []
    clean_mlp_outputs = []
    corrupt_mlp_outputs = []
    clean_logits = []
    corrupt_logits = []

    print("Capturing clean and corrupt activations...")

    for index in range(len(rows)):
        (
            clean_attention,
            clean_mlp,
            clean_logit,
        ) = capture_components_and_logit(
            adapter,
            probe,
            clean_sequences[index],
        )

        (
            corrupt_attention,
            corrupt_mlp,
            corrupt_logit,
        ) = capture_components_and_logit(
            adapter,
            probe,
            corrupt_sequences[index],
        )

        clean_attention_outputs.append(clean_attention)
        corrupt_attention_outputs.append(corrupt_attention)
        clean_mlp_outputs.append(clean_mlp)
        corrupt_mlp_outputs.append(corrupt_mlp)
        clean_logits.append(clean_logit)
        corrupt_logits.append(corrupt_logit)

        print(
            f"\rCaptured {index + 1}/{len(rows)}",
            end="",
            flush=True,
        )

    print("\nRunning circuit interventions...")

    result_rows: list[dict[str, object]] = []

    for index, row in enumerate(rows):
        clean_logit = clean_logits[index]
        corrupt_logit = corrupt_logits[index]
        total_effect = clean_logit - corrupt_logit

        # Delivery:
        # corrupt recipient receives clean candidate-circuit activations.
        delivery_logit = run_combined_patch(
            adapter,
            probe,
            corrupt_sequences[index],
            donor_attention_outputs=(
                clean_attention_outputs[index]
            ),
            donor_mlp_output=clean_mlp_outputs[index],
            donor_offsets=clean_offsets[index],
            recipient_offsets=corrupt_offsets[index],
        )

        # Necessity:
        # clean recipient receives corrupt candidate-circuit activations.
        knockout_logit = run_combined_patch(
            adapter,
            probe,
            clean_sequences[index],
            donor_attention_outputs=(
                corrupt_attention_outputs[index]
            ),
            donor_mlp_output=corrupt_mlp_outputs[index],
            donor_offsets=corrupt_offsets[index],
            recipient_offsets=clean_offsets[index],
        )

        # Self-patching controls.
        self_clean_logit = run_combined_patch(
            adapter,
            probe,
            clean_sequences[index],
            donor_attention_outputs=(
                clean_attention_outputs[index]
            ),
            donor_mlp_output=clean_mlp_outputs[index],
            donor_offsets=clean_offsets[index],
            recipient_offsets=clean_offsets[index],
        )

        self_corrupt_logit = run_combined_patch(
            adapter,
            probe,
            corrupt_sequences[index],
            donor_attention_outputs=(
                corrupt_attention_outputs[index]
            ),
            donor_mlp_output=corrupt_mlp_outputs[index],
            donor_offsets=corrupt_offsets[index],
            recipient_offsets=corrupt_offsets[index],
        )

        delivered_amount = (
            delivery_logit - corrupt_logit
        )

        destroyed_amount = (
            clean_logit - knockout_logit
        )

        if abs(total_effect) > args.minimum_total_effect:
            delivery_fraction = (
                delivered_amount / total_effect
            )

            necessity_fraction = (
                destroyed_amount / total_effect
            )
        else:
            delivery_fraction = float("nan")
            necessity_fraction = float("nan")

        result_rows.append(
            {
                "recipient_index": index,
                "source_index": row["source_index"],
                "clean_logit": clean_logit,
                "corrupt_logit": corrupt_logit,
                "total_effect": total_effect,
                "delivery_logit": delivery_logit,
                "delivered_amount": delivered_amount,
                "delivery_fraction": delivery_fraction,
                "knockout_logit": knockout_logit,
                "destroyed_amount": destroyed_amount,
                "necessity_fraction": necessity_fraction,
                "self_clean_logit": self_clean_logit,
                "self_clean_minus_baseline": (
                    self_clean_logit - clean_logit
                ),
                "self_corrupt_logit": self_corrupt_logit,
                "self_corrupt_minus_baseline": (
                    self_corrupt_logit - corrupt_logit
                ),
            }
        )

        print(
            f"{index + 1:03d}/{len(rows)} "
            f"TE={total_effect:+.3f} "
            f"D={delivery_fraction:+.3f} "
            f"N={necessity_fraction:+.3f}"
        )

    total_effects = np.asarray(
        [row["total_effect"] for row in result_rows],
        dtype=np.float64,
    )

    delivery_amounts = np.asarray(
        [row["delivered_amount"] for row in result_rows],
        dtype=np.float64,
    )

    delivery_fractions = np.asarray(
        [row["delivery_fraction"] for row in result_rows],
        dtype=np.float64,
    )

    destroyed_amounts = np.asarray(
        [row["destroyed_amount"] for row in result_rows],
        dtype=np.float64,
    )

    necessity_fractions = np.asarray(
        [row["necessity_fraction"] for row in result_rows],
        dtype=np.float64,
    )

    self_clean_changes = np.asarray(
        [
            row["self_clean_minus_baseline"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    self_corrupt_changes = np.asarray(
        [
            row["self_corrupt_minus_baseline"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    valid_effect_mask = (
        np.abs(total_effects)
        > args.minimum_total_effect
    )

    summary = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": SELECTED_LAYER,
        "probe_path": str(PROBE_PATH),
        "cohort": str(args.cohort),
        "rows": len(rows),
        "valid_normalized_rows": int(
            valid_effect_mask.sum()
        ),
        "minimum_total_effect": (
            args.minimum_total_effect
        ),
        "candidate_circuit": {
            "mlp_layers": [MLP_LAYER],
            "heads": [
                {
                    "layer": layer,
                    "head": head,
                }
                for layer, head in HEAD_GROUP
            ],
        },
        "definitions": {
            "total_effect": (
                "clean_logit - corrupt_logit"
            ),
            "delivery_fraction": (
                "(delivery_logit - corrupt_logit) "
                "/ total_effect"
            ),
            "necessity_fraction": (
                "(clean_logit - knockout_logit) "
                "/ total_effect"
            ),
            "delivery_intervention": (
                "Clean candidate-circuit activations "
                "patched into the corrupt run."
            ),
            "necessity_intervention": (
                "Corrupt candidate-circuit activations "
                "patched into the clean run."
            ),
        },
        "total_effect": finite_summary(
            total_effects
        ),
        "delivered_amount": finite_summary(
            delivery_amounts
        ),
        "delivery_fraction": finite_summary(
            delivery_fractions
        ),
        "destroyed_amount": finite_summary(
            destroyed_amounts
        ),
        "necessity_fraction": finite_summary(
            necessity_fractions
        ),
        "fraction_positive_total_effect": float(
            np.mean(total_effects > 0)
        ),
        "fraction_positive_delivery": float(
            np.mean(delivery_amounts > 0)
        ),
        "fraction_positive_necessity": float(
            np.mean(destroyed_amounts > 0)
        ),
        "self_clean_max_absolute_difference": float(
            np.max(np.abs(self_clean_changes))
        ),
        "self_corrupt_max_absolute_difference": float(
            np.max(np.abs(self_corrupt_changes))
        ),
    }

    summary_path = OUTPUT_DIR / (
        f"{args.output_name}_summary.json"
    )

    detail_path = OUTPUT_DIR / (
        f"{args.output_name}_per_example.csv"
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
            fieldnames=list(result_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(result_rows)

    print("\n===== Frozen circuit accounting =====")
    print(
        "Mean delivery fraction:",
        summary["delivery_fraction"]["mean"],
    )
    print(
        "Mean necessity fraction:",
        summary["necessity_fraction"]["mean"],
    )
    print(
        "Self-clean max absolute difference:",
        summary[
            "self_clean_max_absolute_difference"
        ],
    )
    print(
        "Self-corrupt max absolute difference:",
        summary[
            "self_corrupt_max_absolute_difference"
        ],
    )
    print("\nSaved:", summary_path)
    print("Saved:", detail_path)

    # Sibling provenance record. Written after the outputs exist so their hashes are real.
    manifest_path = write_run_manifest(
        summary_path,
        experiment_id="dnabert_promoter_circuit_accounting",
        config=vars(args),
        outputs=[detail_path],
        inputs=[PROBE_PATH],
        probe_path=PROBE_PATH,
        selected_layer=SELECTED_LAYER,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
    )
    print("Saved:", manifest_path)

    print("FROZEN CIRCUIT ACCOUNTING SUCCEEDED")


if __name__ == "__main__":
    main()
