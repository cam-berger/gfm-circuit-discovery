"""Build clean, mono-shuffled, and dinucleotide-shuffled promoter cohorts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gfmcd.dnabert.perturb import (
    changed_fraction,
    dinucleotide_counts,
    dinucleotide_shuffle,
    mononucleotide_shuffle,
    nucleotide_counts,
)


DATA_DIR = Path("results/dnabert_promoter/shared/clean")
OUTPUT_DIR = Path(
    "results/dnabert_promoter/behavioral/corruption_cohorts"
)

BASE_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="validation",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--output-name",
        default=None,
    )

    return parser.parse_args()


def load_rows(split: str) -> list[dict]:
    path = DATA_DIR / f"{split}.csv"

    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.split)

    if args.limit is not None:
        rows = rows[: args.limit]

    output_rows = []

    for index, row in enumerate(rows):
        clean = row["sequence"]

        mono_seed = BASE_SEED + index
        di_seed = BASE_SEED + 1_000_000 + index

        mono = mononucleotide_shuffle(
            clean,
            seed=mono_seed,
        )

        di = dinucleotide_shuffle(
            clean,
            seed=di_seed,
        )

        if nucleotide_counts(mono) != nucleotide_counts(clean):
            raise RuntimeError(
                f"mononucleotide preservation failed at row {index}"
            )

        if nucleotide_counts(di) != nucleotide_counts(clean):
            raise RuntimeError(
                f"dinucleotide nucleotide-count preservation "
                f"failed at row {index}"
            )

        if dinucleotide_counts(di) != dinucleotide_counts(clean):
            raise RuntimeError(
                f"dinucleotide preservation failed at row {index}"
            )

        output_rows.append(
            {
                "split": args.split,
                "source_index": row["source_index"],
                "label": row["label"],
                "clean_sequence": clean,
                "mono_sequence": mono,
                "dinucleotide_sequence": di,
                "clean_sha256": row["sequence_sha256"],
                "mono_seed": mono_seed,
                "dinucleotide_seed": di_seed,
                "mono_changed_fraction": changed_fraction(
                    clean,
                    mono,
                ),
                "dinucleotide_changed_fraction": changed_fraction(
                    clean,
                    di,
                ),
            }
        )

    output_name = args.output_name

    if output_name is None:
        suffix = (
            f"_{args.limit}"
            if args.limit is not None
            else "_full"
        )
        output_name = f"{args.split}{suffix}.csv"

    output_path = OUTPUT_DIR / output_name

    fieldnames = list(output_rows[0].keys())

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(output_rows)

    mono_changes = [
        float(row["mono_changed_fraction"])
        for row in output_rows
    ]

    di_changes = [
        float(row["dinucleotide_changed_fraction"])
        for row in output_rows
    ]

    summary = {
        "split": args.split,
        "rows": len(output_rows),
        "base_seed": BASE_SEED,
        "output_file": str(output_path),
        "mean_mono_changed_fraction": (
            sum(mono_changes) / len(mono_changes)
        ),
        "mean_dinucleotide_changed_fraction": (
            sum(di_changes) / len(di_changes)
        ),
        "minimum_mono_changed_fraction": min(mono_changes),
        "minimum_dinucleotide_changed_fraction": min(di_changes),
    }

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print("BEHAVIORAL CORRUPTION COHORT SUCCEEDED")


if __name__ == "__main__":
    main()
