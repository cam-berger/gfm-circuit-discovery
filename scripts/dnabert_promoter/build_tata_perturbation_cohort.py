"""Build controlled TATA-like motif perturbation cohorts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

PATTERN = re.compile(r"TATA[AT][AT]")
MOTIF_LENGTH = 6
EDIT_OFFSETS = (2, 3)

DEFAULT_INPUT_DIR = Path(
    "results/dnabert_promoter/"
    "frozen_claim_tests/biological_features"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/dnabert_promoter/"
    "frozen_claim_tests/biological_features/"
    "tata_perturbations"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        required=True,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-name",
        default=None,
    )

    return parser.parse_args()


def sha256(sequence: str) -> str:
    return hashlib.sha256(
        sequence.encode("utf-8")
    ).hexdigest()


def mutate_base(base: str) -> str:
    mapping = {
        "T": "G",
        "A": "C",
    }

    if base not in mapping:
        raise ValueError(
            f"Unexpected motif base for intervention: {base}"
        )

    return mapping[base]


def apply_edits(
    sequence: str,
    positions: tuple[int, int],
) -> tuple[str, str]:
    chars = list(sequence)
    changes = []

    for position in positions:
        old = chars[position]
        new = mutate_base(old)

        chars[position] = new
        changes.append(
            f"{position}:{old}>{new}"
        )

    return "".join(chars), ",".join(changes)


def creates_tata_like(sequence: str) -> bool:
    return PATTERN.search(sequence) is not None


def choose_control_positions(
    sequence: str,
    motif_start: int,
) -> tuple[int, int] | None:
    motif_positions = set(
        range(
            motif_start,
            motif_start + MOTIF_LENGTH,
        )
    )

    # Search from the downstream end so the control is
    # spatially separated from the early motif.
    for first in range(
        len(sequence) - 2,
        -1,
        -1,
    ):
        second = first + 1

        if second >= len(sequence):
            continue

        positions = (first, second)

        if any(
            position in motif_positions
            for position in positions
        ):
            continue

        if sequence[first] != "T":
            continue

        if sequence[second] != "A":
            continue

        edited, _ = apply_edits(
            sequence,
            positions,
        )

        # The clean sequence has exactly one TATA-like motif.
        # The control must preserve it and create no additional one.
        matches = list(
            PATTERN.finditer(edited)
        )

        if len(matches) != 1:
            continue

        if matches[0].start() != motif_start:
            continue

        return positions

    return None


def main() -> None:
    args = parse_args()

    input_path = (
        args.input
        if args.input is not None
        else DEFAULT_INPUT_DIR
        / f"{args.split}_tata_like_eligible.csv"
    )

    output_name = (
        args.output_name
        or f"{args.split}_tata_perturbation_cohort.csv"
    )

    DEFAULT_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        DEFAULT_OUTPUT_DIR / output_name
    )

    with input_path.open(
        encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    output_rows = []
    exclusions = []

    for row in rows:
        sequence = row["sequence"].upper()
        motif_start = int(
            row["motif_start"]
        )

        motif_positions = tuple(
            motif_start + offset
            for offset in EDIT_OFFSETS
        )

        motif_destroyed, motif_changes = (
            apply_edits(
                sequence,
                motif_positions,
            )
        )

        if creates_tata_like(
            motif_destroyed
        ):
            exclusions.append(
                {
                    "source_index": (
                        row["source_index"]
                    ),
                    "reason": (
                        "motif_destruction_left_"
                        "tata_like_match"
                    ),
                }
            )
            continue

        control_positions = (
            choose_control_positions(
                sequence,
                motif_start,
            )
        )

        if control_positions is None:
            exclusions.append(
                {
                    "source_index": (
                        row["source_index"]
                    ),
                    "reason": (
                        "no_matched_TA_control"
                    ),
                }
            )
            continue

        control_sequence, control_changes = (
            apply_edits(
                sequence,
                control_positions,
            )
        )

        if not PATTERN.search(
            control_sequence
        ):
            raise RuntimeError(
                "Control edit destroyed the "
                "original TATA-like motif"
            )

        output_rows.append(
            {
                "split": args.split,
                "source_index": (
                    row["source_index"]
                ),
                "label": 1,
                "clean_sequence": sequence,
                "motif_destroyed_sequence": (
                    motif_destroyed
                ),
                "matched_nonmotif_sequence": (
                    control_sequence
                ),
                "identity_sequence": sequence,
                "motif": row["motif"],
                "motif_start": motif_start,
                "motif_end": (
                    motif_start
                    + MOTIF_LENGTH
                ),
                "motif_edit_positions": (
                    ",".join(
                        str(value)
                        for value in motif_positions
                    )
                ),
                "motif_changes": (
                    motif_changes
                ),
                "control_edit_positions": (
                    ",".join(
                        str(value)
                        for value in control_positions
                    )
                ),
                "control_changes": (
                    control_changes
                ),
                "edit_count": 2,
                "clean_gc_count": (
                    sequence.count("G")
                    + sequence.count("C")
                ),
                "motif_destroyed_gc_count": (
                    motif_destroyed.count("G")
                    + motif_destroyed.count("C")
                ),
                "control_gc_count": (
                    control_sequence.count("G")
                    + control_sequence.count("C")
                ),
                "clean_sha256": sha256(
                    sequence
                ),
                "motif_destroyed_sha256": (
                    sha256(motif_destroyed)
                ),
                "matched_nonmotif_sha256": (
                    sha256(control_sequence)
                ),
            }
        )

    if not output_rows:
        raise RuntimeError(
            "No eligible perturbation rows"
        )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                output_rows[0].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "split": args.split,
        "input_rows": len(rows),
        "included_rows": len(
            output_rows
        ),
        "excluded_rows": len(
            exclusions
        ),
        "intervention": {
            "motif_pattern": (
                "TATA[AT][AT]"
            ),
            "motif_edit_offsets": (
                list(EDIT_OFFSETS)
            ),
            "substitutions": {
                "T": "G",
                "A": "C",
            },
            "edit_count": 2,
        },
        "output_file": str(
            output_path
        ),
        "exclusions": exclusions,
    }

    summary_path = (
        output_path.with_suffix(
            ".json"
        )
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

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "TATA PERTURBATION COHORT "
        "SUCCEEDED"
    )


if __name__ == "__main__":
    main()
