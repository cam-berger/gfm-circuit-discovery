"""Build and validate the fixed GUE prom_core_all cohort."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from datasets import load_dataset

from gfmcd.dnabert.dataset import (
    build_records,
    exact_duplicate_report,
    records_sha256,
    reverse_complement_overlap_report,
    split_summary,
)


DATASET_NAME = "leannmlindsey/GUE"
DATASET_CONFIG = "prom_core_all"
OUTPUT_DIR = Path("results/dnabert_promoter/shared")

SOURCE_TO_OUTPUT_SPLIT = {
    "train": "train",
    "dev": "validation",
    "test": "test",
}


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def write_records_csv(path: Path, records) -> None:
    fieldnames = [
        "split",
        "source_index",
        "sequence",
        "label",
        "length",
        "gc_fraction",
        "sequence_sha256",
        "reverse_complement_sha256",
    ]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(record.to_dict())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading {DATASET_NAME}/{DATASET_CONFIG}"
    )

    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
    )

    records_by_split = {}

    for source_split, output_split in SOURCE_TO_OUTPUT_SPLIT.items():
        split = dataset[source_split]

        records = build_records(
            split["sequence"],
            split["label"],
            split=output_split,
        )

        records_by_split[output_split] = records

        output_path = OUTPUT_DIR / f"{output_split}.csv"
        write_records_csv(output_path, records)

        print(
            f"Wrote {output_split}: "
            f"{len(records):,} records → {output_path}"
        )

    summaries = {
        split: split_summary(records)
        for split, records in records_by_split.items()
    }

    hashes = {
        split: records_sha256(records)
        for split, records in records_by_split.items()
    }

    duplicate_report = exact_duplicate_report(
        records_by_split
    )

    reverse_complement_report = (
        reverse_complement_overlap_report(
            records_by_split
        )
    )

    dataset_summary = {
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "source_split_mapping": SOURCE_TO_OUTPUT_SPLIT,
        "splits": summaries,
        "total_rows": sum(
            len(records)
            for records in records_by_split.values()
        ),
    }

    split_manifest = {
        "dataset_name": DATASET_NAME,
        "dataset_config": DATASET_CONFIG,
        "splits": {
            split: {
                "rows": len(records),
                "records_sha256": hashes[split],
                "output_file": f"{split}.csv",
            }
            for split, records in records_by_split.items()
        },
    }

    leakage_report = {
        "exact_duplicates": duplicate_report,
        "reverse_complement_overlaps": reverse_complement_report,
    }

    write_json(
        OUTPUT_DIR / "dataset_summary.json",
        dataset_summary,
    )

    write_json(
        OUTPUT_DIR / "dataset_sha256.json",
        hashes,
    )

    write_json(
        OUTPUT_DIR / "split_manifest.json",
        split_manifest,
    )

    write_json(
        OUTPUT_DIR / "leakage_report.json",
        leakage_report,
    )

    print("\n===== Dataset summary =====")

    for split, summary in summaries.items():
        print(
            f"{split}: "
            f"rows={summary['rows']:,}, "
            f"labels={summary['label_counts']}, "
            f"lengths={summary['unique_lengths']}, "
            f"mean_gc={summary['mean_gc_fraction']:.4f}"
        )

    print("\n===== Exact duplicates =====")
    print(
        "Duplicate groups:",
        duplicate_report["duplicate_group_count"],
    )
    print(
        "Cross-split duplicate groups:",
        duplicate_report[
            "cross_split_duplicate_group_count"
        ],
    )
    print(
        "Conflicting-label duplicate groups:",
        duplicate_report[
            "conflicting_label_duplicate_group_count"
        ],
    )

    print("\n===== Reverse-complement overlaps =====")
    print(
        "Overlap pairs:",
        reverse_complement_report[
            "overlap_pair_count"
        ],
    )
    print(
        "Cross-split overlap pairs:",
        reverse_complement_report[
            "cross_split_overlap_pair_count"
        ],
    )
    print(
        "Conflicting-label overlap pairs:",
        reverse_complement_report[
            "conflicting_label_overlap_pair_count"
        ],
    )

    print("\nOutput directory:", OUTPUT_DIR)
    print("PROMOTER COHORT BUILD SUCCEEDED")


if __name__ == "__main__":
    main()
