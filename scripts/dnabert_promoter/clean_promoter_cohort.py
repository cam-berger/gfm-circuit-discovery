"""Create leakage-controlled splits from the raw promoter cohort."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


INPUT_DIR = Path("results/dnabert_promoter/shared")
OUTPUT_DIR = Path("results/dnabert_promoter/shared/clean")

SPLIT_ORDER = ("train", "validation", "test")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty split: {path}")

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = {
        split: read_csv(INPUT_DIR / f"{split}.csv")
        for split in SPLIT_ORDER
    }

    labels_by_hash: dict[str, set[int]] = defaultdict(set)

    for rows in raw.values():
        for row in rows:
            labels_by_hash[row["sequence_sha256"]].add(
                int(row["label"])
            )

    conflicting_hashes = {
        sequence_hash
        for sequence_hash, labels in labels_by_hash.items()
        if len(labels) > 1
    }

    cleaned: dict[str, list[dict]] = {}
    globally_seen: set[str] = set()

    report = {
        "policy": {
            "remove_conflicting_label_hashes_globally": True,
            "deduplicate_within_split": True,
            "precedence": list(SPLIT_ORDER),
            "remove_later_split_hashes_seen_earlier": True,
        },
        "conflicting_hash_count": len(conflicting_hashes),
        "splits": {},
    }

    for split in SPLIT_ORDER:
        rows = raw[split]
        within_seen: set[str] = set()
        retained: list[dict] = []

        removed_conflicting = 0
        removed_within_split_duplicate = 0
        removed_cross_split_duplicate = 0

        for row in rows:
            sequence_hash = row["sequence_sha256"]

            if sequence_hash in conflicting_hashes:
                removed_conflicting += 1
                continue

            if sequence_hash in within_seen:
                removed_within_split_duplicate += 1
                continue

            if sequence_hash in globally_seen:
                removed_cross_split_duplicate += 1
                continue

            retained.append(row)
            within_seen.add(sequence_hash)

        globally_seen.update(within_seen)
        cleaned[split] = retained

        report["splits"][split] = {
            "input_rows": len(rows),
            "output_rows": len(retained),
            "removed_conflicting": removed_conflicting,
            "removed_within_split_duplicate": (
                removed_within_split_duplicate
            ),
            "removed_cross_split_duplicate": (
                removed_cross_split_duplicate
            ),
            "label_counts": {
                "0": sum(int(row["label"]) == 0 for row in retained),
                "1": sum(int(row["label"]) == 1 for row in retained),
            },
        }

        write_csv(
            OUTPUT_DIR / f"{split}.csv",
            retained,
        )

    retained_hashes = {
        split: {
            row["sequence_sha256"]
            for row in rows
        }
        for split, rows in cleaned.items()
    }

    assert retained_hashes["train"].isdisjoint(
        retained_hashes["validation"]
    )
    assert retained_hashes["train"].isdisjoint(
        retained_hashes["test"]
    )
    assert retained_hashes["validation"].isdisjoint(
        retained_hashes["test"]
    )

    write_json(
        OUTPUT_DIR / "cleaning_report.json",
        report,
    )

    print("Conflicting hashes removed:", len(conflicting_hashes))

    for split in SPLIT_ORDER:
        info = report["splits"][split]
        print(
            f"{split}: "
            f"{info['input_rows']:,} → {info['output_rows']:,}; "
            f"conflicting={info['removed_conflicting']}, "
            f"within_split={info['removed_within_split_duplicate']}, "
            f"cross_split={info['removed_cross_split_duplicate']}, "
            f"labels={info['label_counts']}"
        )

    print("CLEAN PROMOTER COHORT SUCCEEDED")


if __name__ == "__main__":
    main()
