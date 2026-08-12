"""Select a balanced high-signal cohort for causal localization."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_BEHAVIOR_PATH = Path(
    "results/dnabert_promoter/behavioral/"
    "probe_results/validation_full_per_example.csv"
)

DEFAULT_COHORT_PATH = Path(
    "results/dnabert_promoter/behavioral/"
    "corruption_cohorts/validation_full.csv"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/dnabert_promoter/causal/localization_cohorts"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--behavior-results",
        type=Path,
        default=DEFAULT_BEHAVIOR_PATH,
    )

    parser.add_argument(
        "--corruption-cohort",
        type=Path,
        default=DEFAULT_COHORT_PATH,
    )

    parser.add_argument(
        "--promoters",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--non-promoters",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--output-name",
        default="validation_localization_256.csv",
    )

    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    behavior_rows = read_csv(args.behavior_results)
    cohort_rows = read_csv(args.corruption_cohort)

    cohort_by_source = {
        row["source_index"]: row
        for row in cohort_rows
    }

    promoter_candidates = []
    non_promoter_candidates = []

    for row in behavior_rows:
        source_index = row["source_index"]
        cohort_row = cohort_by_source[source_index]

        combined = {
            **cohort_row,
            "clean_logit": row["clean_logit"],
            "mono_logit": row["mono_logit"],
            "dinucleotide_logit": row["dinucleotide_logit"],
            "mono_logit_change": row["mono_logit_change"],
            "dinucleotide_logit_change": row["dinucleotide_logit_change"],
            "clean_probability": row["clean_probability"],
            "mono_probability": row["mono_probability"],
            "dinucleotide_probability": row["dinucleotide_probability"],
        }

        label = int(row["label"])
        clean_logit = float(row["clean_logit"])
        mono_change = float(row["mono_logit_change"])

        if label == 1 and clean_logit > 0 and mono_change < 0:
            combined["selection_score"] = str(-mono_change)
            combined["selection_reason"] = (
                "promoter_clean_positive_mono_damage"
            )
            promoter_candidates.append(combined)

        elif label == 0 and clean_logit < 0:
            combined["selection_score"] = str(-clean_logit)
            combined["selection_reason"] = (
                "non_promoter_clean_negative_control"
            )
            non_promoter_candidates.append(combined)

    promoter_candidates.sort(
        key=lambda row: float(row["selection_score"]),
        reverse=True,
    )

    non_promoter_candidates.sort(
        key=lambda row: float(row["selection_score"]),
        reverse=True,
    )

    selected_promoters = promoter_candidates[: args.promoters]
    selected_non_promoters = non_promoter_candidates[
        : args.non_promoters
    ]

    if len(selected_promoters) != args.promoters:
        raise RuntimeError(
            f"Only found {len(selected_promoters)} eligible promoters"
        )

    if len(selected_non_promoters) != args.non_promoters:
        raise RuntimeError(
            f"Only found {len(selected_non_promoters)} "
            "eligible non-promoters"
        )

    selected = selected_promoters + selected_non_promoters

    for index, row in enumerate(selected):
        row["localization_index"] = str(index)

    output_path = DEFAULT_OUTPUT_DIR / args.output_name

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(selected[0].keys()),
        )
        writer.writeheader()
        writer.writerows(selected)

    promoter_changes = [
        float(row["mono_logit_change"])
        for row in selected_promoters
    ]

    non_promoter_changes = [
        float(row["mono_logit_change"])
        for row in selected_non_promoters
    ]

    summary = {
        "rows": len(selected),
        "promoters": len(selected_promoters),
        "non_promoters": len(selected_non_promoters),
        "selection_corruption": "mononucleotide",
        "mean_selected_promoter_logit_change": (
            sum(promoter_changes) / len(promoter_changes)
        ),
        "mean_selected_non_promoter_logit_change": (
            sum(non_promoter_changes) / len(non_promoter_changes)
        ),
        "output_file": str(output_path),
    }

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("LOCALIZATION COHORT SUCCEEDED")


if __name__ == "__main__":
    main()
