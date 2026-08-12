"""Compare matched, mismatched, and self-corrupt activation restoration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from gfmcd.adapters.dnabert import DnaBertAdapter

from run_head_restoration_pilot import (
    MODEL_NAME,
    MODEL_REVISION,
    PROBE_PATH,
    DEFAULT_COHORT,
    OUTPUT_DIR,
    NUM_HEADS,
    tokenize_with_offsets,
    assert_adapter_tokenization_matches,
    capture_clean_head_output,
    run_sequence,
    run_patched_corrupt,
    get_corrupt_sequence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT,
    )

    parser.add_argument(
        "--output-name",
        default="validation_L0H8_restoration_controls_16",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--head",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--donor-offset",
        type=int,
        default=1,
        help=(
            "Cyclic offset used to choose the "
            "mismatched promoter donor"
        ),
    )

    return parser.parse_args()


def load_promoter_rows(
    path: Path,
    limit: int,
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    promoter_rows = [
        row
        for row in rows
        if int(row["label"]) == 1
    ]

    if len(promoter_rows) < limit:
        raise ValueError(
            f"Requested {limit} promoters but only "
            f"{len(promoter_rows)} are available"
        )

    return promoter_rows[:limit]


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("Cannot summarize empty array")

    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1))
        if values.size > 1
        else 0.0,
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "fraction_positive": float(
            np.mean(values > 0)
        ),
        "fraction_negative": float(
            np.mean(values < 0)
        ),
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not 0 <= args.head < NUM_HEADS:
        raise ValueError(
            f"--head must be between 0 and {NUM_HEADS - 1}"
        )

    rows = load_promoter_rows(
        args.cohort,
        args.limit,
    )

    print("Promoter rows:", len(rows))
    print(
        f"Testing Layer {args.layer} Head {args.head}"
    )

    print("\nLoading DNABERT...")

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

    clean_ids = []
    clean_offsets = []
    corrupt_ids = []
    corrupt_offsets = []

    print("\nPreparing token offsets...")

    for index in range(len(rows)):
        clean_id_row, clean_offset_row = (
            tokenize_with_offsets(
                adapter,
                clean_sequences[index],
            )
        )

        corrupt_id_row, corrupt_offset_row = (
            tokenize_with_offsets(
                adapter,
                corrupt_sequences[index],
            )
        )

        assert_adapter_tokenization_matches(
            adapter,
            clean_sequences[index],
            clean_id_row,
        )

        assert_adapter_tokenization_matches(
            adapter,
            corrupt_sequences[index],
            corrupt_id_row,
        )

        clean_ids.append(clean_id_row)
        clean_offsets.append(clean_offset_row)
        corrupt_ids.append(corrupt_id_row)
        corrupt_offsets.append(corrupt_offset_row)

    print("Capturing clean and corrupt activations...")

    clean_activations = []
    corrupt_activations = []
    clean_logits = []
    corrupt_logits = []

    for index in range(len(rows)):
        clean_activation, clean_feature = (
            capture_clean_head_output(
                adapter,
                clean_sequences[index],
                layer=args.layer,
            )
        )

        corrupt_activation, corrupt_feature = (
            capture_clean_head_output(
                adapter,
                corrupt_sequences[index],
                layer=args.layer,
            )
        )

        clean_logit = float(
            probe.decision_function(
                clean_feature.detach()
                .cpu()
                .numpy()
            )[0]
        )

        corrupt_logit = float(
            probe.decision_function(
                corrupt_feature.detach()
                .cpu()
                .numpy()
            )[0]
        )

        clean_activations.append(
            clean_activation
        )

        corrupt_activations.append(
            corrupt_activation
        )

        clean_logits.append(clean_logit)
        corrupt_logits.append(corrupt_logit)

        print(
            f"\rCaptured {index + 1}/{len(rows)}",
            end="",
            flush=True,
        )

    print()

    result_rows = []

    for recipient_index, row in enumerate(rows):
        donor_index = (
            recipient_index + args.donor_offset
        ) % len(rows)

        if donor_index == recipient_index:
            raise ValueError(
                "--donor-offset must not map recipients "
                "to themselves"
            )

        clean_logit = clean_logits[
            recipient_index
        ]

        corrupt_logit = corrupt_logits[
            recipient_index
        ]

        matched_logit = run_patched_corrupt(
            adapter,
            probe,
            corrupt_sequences[recipient_index],
            layer=args.layer,
            head=args.head,
            clean_attention_output=(
                clean_activations[
                    recipient_index
                ]
            ),
            clean_offsets=(
                clean_offsets[
                    recipient_index
                ]
            ),
            corrupt_offsets=(
                corrupt_offsets[
                    recipient_index
                ]
            ),
        )

        mismatched_logit = run_patched_corrupt(
            adapter,
            probe,
            corrupt_sequences[recipient_index],
            layer=args.layer,
            head=args.head,
            clean_attention_output=(
                clean_activations[
                    donor_index
                ]
            ),
            clean_offsets=(
                clean_offsets[
                    donor_index
                ]
            ),
            corrupt_offsets=(
                corrupt_offsets[
                    recipient_index
                ]
            ),
        )

        self_corrupt_logit = run_patched_corrupt(
            adapter,
            probe,
            corrupt_sequences[recipient_index],
            layer=args.layer,
            head=args.head,
            clean_attention_output=(
                corrupt_activations[
                    recipient_index
                ]
            ),
            clean_offsets=(
                corrupt_offsets[
                    recipient_index
                ]
            ),
            corrupt_offsets=(
                corrupt_offsets[
                    recipient_index
                ]
            ),
        )

        total_damage = (
            clean_logit - corrupt_logit
        )

        matched_restored = (
            matched_logit - corrupt_logit
        )

        mismatched_restored = (
            mismatched_logit - corrupt_logit
        )

        self_corrupt_change = (
            self_corrupt_logit
            - corrupt_logit
        )

        matched_recovery = (
            matched_restored / total_damage
            if abs(total_damage) > 1e-8
            else float("nan")
        )

        mismatched_recovery = (
            mismatched_restored / total_damage
            if abs(total_damage) > 1e-8
            else float("nan")
        )

        matched_advantage = (
            matched_restored
            - mismatched_restored
        )

        result_rows.append(
            {
                "recipient_index": recipient_index,
                "recipient_source_index": row[
                    "source_index"
                ],
                "mismatched_donor_index": (
                    donor_index
                ),
                "mismatched_donor_source_index": (
                    rows[donor_index][
                        "source_index"
                    ]
                ),
                "clean_logit": clean_logit,
                "corrupt_logit": corrupt_logit,
                "total_corruption_damage": (
                    total_damage
                ),
                "matched_logit": matched_logit,
                "matched_restored_amount": (
                    matched_restored
                ),
                "matched_recovery_fraction": (
                    matched_recovery
                ),
                "mismatched_logit": (
                    mismatched_logit
                ),
                "mismatched_restored_amount": (
                    mismatched_restored
                ),
                "mismatched_recovery_fraction": (
                    mismatched_recovery
                ),
                "matched_minus_mismatched": (
                    matched_advantage
                ),
                "self_corrupt_logit": (
                    self_corrupt_logit
                ),
                "self_corrupt_minus_baseline": (
                    self_corrupt_change
                ),
            }
        )

        print(
            f"{recipient_index + 1:02d}/{len(rows)} "
            f"matched={matched_restored:+.3f} "
            f"mismatched={mismatched_restored:+.3f} "
            f"advantage={matched_advantage:+.3f} "
            f"self={self_corrupt_change:+.6f}"
        )

    matched = np.asarray(
        [
            row["matched_restored_amount"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    mismatched = np.asarray(
        [
            row["mismatched_restored_amount"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    advantage = matched - mismatched

    self_corrupt = np.asarray(
        [
            row["self_corrupt_minus_baseline"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    matched_recovery = np.asarray(
        [
            row["matched_recovery_fraction"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    mismatched_recovery = np.asarray(
        [
            row["mismatched_recovery_fraction"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    summary = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "patched_layer": args.layer,
        "patched_head": args.head,
        "rows": len(rows),
        "mismatched_donor_rule": (
            f"cyclic promoter offset "
            f"{args.donor_offset}"
        ),
        "mismatched_donor_offset": (
            args.donor_offset
        ),
        "matched_restored_amount": summarize(
            matched
        ),
        "mismatched_restored_amount": summarize(
            mismatched
        ),
        "matched_minus_mismatched": summarize(
            advantage
        ),
        "matched_recovery_fraction": summarize(
            matched_recovery[
                np.isfinite(matched_recovery)
            ]
        ),
        "mismatched_recovery_fraction": summarize(
            mismatched_recovery[
                np.isfinite(mismatched_recovery)
            ]
        ),
        "fraction_matched_better_than_mismatched": (
            float(np.mean(advantage > 0))
        ),
        "self_corrupt_minus_baseline": summarize(
            self_corrupt
        ),
        "self_corrupt_max_absolute_difference": (
            float(np.max(np.abs(self_corrupt)))
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
            fieldnames=list(
                result_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(result_rows)

    print("\n===== Donor-control summary =====")
    print(
        "Matched mean restoration:",
        f"{matched.mean():.6f}",
    )
    print(
        "Mismatched mean restoration:",
        f"{mismatched.mean():.6f}",
    )
    print(
        "Matched advantage:",
        f"{advantage.mean():.6f}",
    )
    print(
        "Matched better than mismatched:",
        f"{100 * np.mean(advantage > 0):.1f}%",
    )
    print(
        "Self-corrupt maximum absolute change:",
        f"{np.max(np.abs(self_corrupt)):.8f}",
    )

    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print("HEAD RESTORATION CONTROLS SUCCEEDED")


if __name__ == "__main__":
    main()
