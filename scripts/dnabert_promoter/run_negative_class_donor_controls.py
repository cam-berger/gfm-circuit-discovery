"""Compare promoter and non-promoter donors in combined restoration."""

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
    tokenize_with_offsets,
    assert_adapter_tokenization_matches,
    get_corrupt_sequence,
)

from run_combined_mlp_head_restoration_controls import (
    HEAD_GROUP,
    MLP_LAYER,
    SELECTED_LAYER,
    capture_components_and_logit,
    run_combined_patch,
    summarize,
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
        default="validation_negative_class_donors_16",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--promoter-donor-offset",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--nonpromoter-offset",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    return parser.parse_args()


def load_class_rows(
    path: Path,
    limit: int,
) -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    promoters = [
        row for row in rows
        if int(row["label"]) == 1
    ]

    nonpromoters = [
        row for row in rows
        if int(row["label"]) == 0
    ]

    if len(promoters) < limit:
        raise ValueError(
            f"Only {len(promoters)} promoters available"
        )

    if len(nonpromoters) < limit:
        raise ValueError(
            f"Only {len(nonpromoters)} non-promoters available"
        )

    return promoters[:limit], nonpromoters[:limit]


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    promoters, nonpromoters = load_class_rows(
        args.cohort,
        args.limit,
    )

    if (
        args.promoter_donor_offset
        % len(promoters)
        == 0
    ):
        raise ValueError(
            "Promoter donor offset maps recipients "
            "to themselves"
        )

    print("Promoter recipients:", len(promoters))
    print("Non-promoter donors:", len(nonpromoters))
    print("MLP component: Layer 0 mlp.wo")
    print("Head group:", HEAD_GROUP)
    print(
        "Promoter donor offset:",
        args.promoter_donor_offset,
    )
    print(
        "Non-promoter offset:",
        args.nonpromoter_offset,
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

    promoter_clean_sequences = [
        row["clean_sequence"]
        for row in promoters
    ]

    promoter_corrupt_sequences = [
        get_corrupt_sequence(row)
        for row in promoters
    ]

    nonpromoter_sequences = [
        row["clean_sequence"]
        for row in nonpromoters
    ]

    promoter_clean_offsets = []
    promoter_corrupt_offsets = []
    nonpromoter_offsets = []

    print("\nPreparing token offsets...")

    for index in range(len(promoters)):
        clean_ids, clean_offsets = (
            tokenize_with_offsets(
                adapter,
                promoter_clean_sequences[index],
            )
        )

        corrupt_ids, corrupt_offsets = (
            tokenize_with_offsets(
                adapter,
                promoter_corrupt_sequences[index],
            )
        )

        negative_ids, negative_offsets = (
            tokenize_with_offsets(
                adapter,
                nonpromoter_sequences[index],
            )
        )

        assert_adapter_tokenization_matches(
            adapter,
            promoter_clean_sequences[index],
            clean_ids,
        )

        assert_adapter_tokenization_matches(
            adapter,
            promoter_corrupt_sequences[index],
            corrupt_ids,
        )

        assert_adapter_tokenization_matches(
            adapter,
            nonpromoter_sequences[index],
            negative_ids,
        )

        promoter_clean_offsets.append(
            clean_offsets
        )

        promoter_corrupt_offsets.append(
            corrupt_offsets
        )

        nonpromoter_offsets.append(
            negative_offsets
        )

    promoter_clean_attention = []
    promoter_corrupt_attention = []
    nonpromoter_attention = []

    promoter_clean_mlp = []
    promoter_corrupt_mlp = []
    nonpromoter_mlp = []

    promoter_clean_logits = []
    promoter_corrupt_logits = []
    nonpromoter_logits = []

    print("Capturing activations...")

    for index in range(len(promoters)):
        (
            clean_attention,
            clean_mlp,
            clean_logit,
        ) = capture_components_and_logit(
            adapter,
            probe,
            promoter_clean_sequences[index],
        )

        (
            corrupt_attention,
            corrupt_mlp,
            corrupt_logit,
        ) = capture_components_and_logit(
            adapter,
            probe,
            promoter_corrupt_sequences[index],
        )

        (
            negative_attention,
            negative_mlp,
            negative_logit,
        ) = capture_components_and_logit(
            adapter,
            probe,
            nonpromoter_sequences[index],
        )

        promoter_clean_attention.append(
            clean_attention
        )

        promoter_corrupt_attention.append(
            corrupt_attention
        )

        nonpromoter_attention.append(
            negative_attention
        )

        promoter_clean_mlp.append(clean_mlp)
        promoter_corrupt_mlp.append(corrupt_mlp)
        nonpromoter_mlp.append(negative_mlp)

        promoter_clean_logits.append(clean_logit)
        promoter_corrupt_logits.append(
            corrupt_logit
        )
        nonpromoter_logits.append(
            negative_logit
        )

        print(
            f"\rCaptured {index + 1}/{len(promoters)}",
            end="",
            flush=True,
        )

    print()

    result_rows = []

    for recipient_index, promoter_row in enumerate(
        promoters
    ):
        mismatched_promoter_index = (
            recipient_index
            + args.promoter_donor_offset
        ) % len(promoters)

        nonpromoter_index = (
            recipient_index
            + args.nonpromoter_offset
        ) % len(nonpromoters)

        recipient_sequence = (
            promoter_corrupt_sequences[
                recipient_index
            ]
        )

        recipient_offsets = (
            promoter_corrupt_offsets[
                recipient_index
            ]
        )

        matched_logit = run_combined_patch(
            adapter,
            probe,
            recipient_sequence,
            donor_attention_outputs=(
                promoter_clean_attention[
                    recipient_index
                ]
            ),
            donor_mlp_output=(
                promoter_clean_mlp[
                    recipient_index
                ]
            ),
            donor_offsets=(
                promoter_clean_offsets[
                    recipient_index
                ]
            ),
            recipient_offsets=recipient_offsets,
        )

        mismatched_promoter_logit = (
            run_combined_patch(
                adapter,
                probe,
                recipient_sequence,
                donor_attention_outputs=(
                    promoter_clean_attention[
                        mismatched_promoter_index
                    ]
                ),
                donor_mlp_output=(
                    promoter_clean_mlp[
                        mismatched_promoter_index
                    ]
                ),
                donor_offsets=(
                    promoter_clean_offsets[
                        mismatched_promoter_index
                    ]
                ),
                recipient_offsets=recipient_offsets,
            )
        )

        nonpromoter_logit = run_combined_patch(
            adapter,
            probe,
            recipient_sequence,
            donor_attention_outputs=(
                nonpromoter_attention[
                    nonpromoter_index
                ]
            ),
            donor_mlp_output=(
                nonpromoter_mlp[
                    nonpromoter_index
                ]
            ),
            donor_offsets=(
                nonpromoter_offsets[
                    nonpromoter_index
                ]
            ),
            recipient_offsets=recipient_offsets,
        )

        self_corrupt_logit = run_combined_patch(
            adapter,
            probe,
            recipient_sequence,
            donor_attention_outputs=(
                promoter_corrupt_attention[
                    recipient_index
                ]
            ),
            donor_mlp_output=(
                promoter_corrupt_mlp[
                    recipient_index
                ]
            ),
            donor_offsets=recipient_offsets,
            recipient_offsets=recipient_offsets,
        )

        clean_logit = promoter_clean_logits[
            recipient_index
        ]

        corrupt_logit = promoter_corrupt_logits[
            recipient_index
        ]

        total_damage = clean_logit - corrupt_logit

        matched_restored = (
            matched_logit - corrupt_logit
        )

        mismatched_promoter_restored = (
            mismatched_promoter_logit
            - corrupt_logit
        )

        nonpromoter_restored = (
            nonpromoter_logit - corrupt_logit
        )

        self_change = (
            self_corrupt_logit - corrupt_logit
        )

        matched_recovery = (
            matched_restored / total_damage
            if abs(total_damage) > 1e-8
            else float("nan")
        )

        mismatched_promoter_recovery = (
            mismatched_promoter_restored
            / total_damage
            if abs(total_damage) > 1e-8
            else float("nan")
        )

        nonpromoter_recovery = (
            nonpromoter_restored / total_damage
            if abs(total_damage) > 1e-8
            else float("nan")
        )

        promoter_class_advantage = (
            mismatched_promoter_restored
            - nonpromoter_restored
        )

        matched_vs_nonpromoter = (
            matched_restored
            - nonpromoter_restored
        )

        result_rows.append(
            {
                "recipient_index": recipient_index,
                "recipient_source_index": (
                    promoter_row["source_index"]
                ),
                "mismatched_promoter_index": (
                    mismatched_promoter_index
                ),
                "mismatched_promoter_source_index": (
                    promoters[
                        mismatched_promoter_index
                    ]["source_index"]
                ),
                "nonpromoter_index": (
                    nonpromoter_index
                ),
                "nonpromoter_source_index": (
                    nonpromoters[
                        nonpromoter_index
                    ]["source_index"]
                ),
                "clean_logit": clean_logit,
                "corrupt_logit": corrupt_logit,
                "total_corruption_damage": (
                    total_damage
                ),
                "matched_promoter_logit": (
                    matched_logit
                ),
                "matched_promoter_restoration": (
                    matched_restored
                ),
                "matched_promoter_recovery": (
                    matched_recovery
                ),
                "mismatched_promoter_logit": (
                    mismatched_promoter_logit
                ),
                "mismatched_promoter_restoration": (
                    mismatched_promoter_restored
                ),
                "mismatched_promoter_recovery": (
                    mismatched_promoter_recovery
                ),
                "nonpromoter_donor_clean_logit": (
                    nonpromoter_logits[
                        nonpromoter_index
                    ]
                ),
                "nonpromoter_patch_logit": (
                    nonpromoter_logit
                ),
                "nonpromoter_restoration": (
                    nonpromoter_restored
                ),
                "nonpromoter_recovery": (
                    nonpromoter_recovery
                ),
                "mismatched_promoter_minus_nonpromoter": (
                    promoter_class_advantage
                ),
                "matched_promoter_minus_nonpromoter": (
                    matched_vs_nonpromoter
                ),
                "self_corrupt_logit": (
                    self_corrupt_logit
                ),
                "self_corrupt_minus_baseline": (
                    self_change
                ),
            }
        )

        print(
            f"{recipient_index + 1:02d}/"
            f"{len(promoters)} "
            f"matched={matched_restored:+.3f} "
            f"promoter={mismatched_promoter_restored:+.3f} "
            f"nonpromoter={nonpromoter_restored:+.3f} "
            f"class_adv={promoter_class_advantage:+.3f} "
            f"self={self_change:+.8f}"
        )

    matched = np.asarray(
        [
            row["matched_promoter_restoration"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    mismatched_promoter = np.asarray(
        [
            row["mismatched_promoter_restoration"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    nonpromoter = np.asarray(
        [
            row["nonpromoter_restoration"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    class_advantage = (
        mismatched_promoter - nonpromoter
    )

    matched_vs_nonpromoter = (
        matched - nonpromoter
    )

    self_change = np.asarray(
        [
            row["self_corrupt_minus_baseline"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    matched_recovery = np.asarray(
        [
            row["matched_promoter_recovery"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    mismatched_promoter_recovery = np.asarray(
        [
            row["mismatched_promoter_recovery"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    nonpromoter_recovery = np.asarray(
        [
            row["nonpromoter_recovery"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    summary = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": SELECTED_LAYER,
        "patched_mlp_layer": MLP_LAYER,
        "patched_head_group": [
            {
                "layer": layer,
                "head": head,
            }
            for layer, head in HEAD_GROUP
        ],
        "rows": len(promoters),
        "promoter_donor_offset": (
            args.promoter_donor_offset
        ),
        "nonpromoter_offset": (
            args.nonpromoter_offset
        ),
        "matched_promoter_restoration": (
            summarize(matched)
        ),
        "mismatched_promoter_restoration": (
            summarize(mismatched_promoter)
        ),
        "nonpromoter_restoration": (
            summarize(nonpromoter)
        ),
        "mismatched_promoter_minus_nonpromoter": (
            summarize(class_advantage)
        ),
        "matched_promoter_minus_nonpromoter": (
            summarize(matched_vs_nonpromoter)
        ),
        "matched_promoter_recovery": (
            summarize(
                matched_recovery[
                    np.isfinite(matched_recovery)
                ]
            )
        ),
        "mismatched_promoter_recovery": (
            summarize(
                mismatched_promoter_recovery[
                    np.isfinite(
                        mismatched_promoter_recovery
                    )
                ]
            )
        ),
        "nonpromoter_recovery": (
            summarize(
                nonpromoter_recovery[
                    np.isfinite(
                        nonpromoter_recovery
                    )
                ]
            )
        ),
        "fraction_promoter_better_than_nonpromoter": (
            float(np.mean(class_advantage > 0))
        ),
        "fraction_matched_better_than_nonpromoter": (
            float(
                np.mean(
                    matched_vs_nonpromoter > 0
                )
            )
        ),
        "self_corrupt_max_absolute_difference": (
            float(np.max(np.abs(self_change)))
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

    print("\n===== Negative-class donor summary =====")
    print(
        "Matched promoter mean restoration:",
        f"{matched.mean():.6f}",
    )
    print(
        "Mismatched promoter mean restoration:",
        f"{mismatched_promoter.mean():.6f}",
    )
    print(
        "Non-promoter mean restoration:",
        f"{nonpromoter.mean():.6f}",
    )
    print(
        "Promoter-class mean advantage:",
        f"{class_advantage.mean():.6f}",
    )
    print(
        "Promoter donor better than non-promoter:",
        f"{100 * np.mean(class_advantage > 0):.1f}%",
    )
    print(
        "Matched recovery fraction:",
        f"{np.nanmean(matched_recovery):.6f}",
    )
    print(
        "Mismatched promoter recovery:",
        f"{np.nanmean(mismatched_promoter_recovery):.6f}",
    )
    print(
        "Non-promoter recovery:",
        f"{np.nanmean(nonpromoter_recovery):.6f}",
    )
    print(
        "Self-corrupt maximum absolute change:",
        f"{np.max(np.abs(self_change)):.8f}",
    )

    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print(
        "NEGATIVE-CLASS DONOR CONTROLS SUCCEEDED"
    )


if __name__ == "__main__":
    main()
