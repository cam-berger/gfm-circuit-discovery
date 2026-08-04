"""Inspect DNABERT token alignment between clean and corrupted sequences."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from gfmcd.adapters.dnabert import DnaBertAdapter


MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"
MODEL_REVISION = "813031b2bf86d9e960a027e2734c908009f31601"

DEFAULT_COHORT = Path(
    "results/dnabert_promoter/causal/"
    "localization_cohorts/validation_localization_256.csv"
)

OUTPUT_DIR = Path(
    "results/dnabert_promoter/causal/token_alignment"
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
        default="validation_localization_256",
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def get_sequence_column(
    row: dict[str, str],
    candidates: tuple[str, ...],
) -> str:
    for column in candidates:
        value = row.get(column)

        if value:
            return value

    raise KeyError(
        "Could not find any sequence column among "
        f"{candidates}. Available columns: {sorted(row)}"
    )


def tokenize_single(
    adapter: DnaBertAdapter,
    sequence: str,
) -> dict[str, object]:
    tokenized = adapter.tokenize([sequence])

    input_ids = tokenized.input_ids[0]
    attention_mask = tokenized.attention_mask[0]

    valid_ids = input_ids[
        attention_mask.to(dtype=bool)
    ].detach().cpu().tolist()

    tokens = adapter.tokenizer.convert_ids_to_tokens(
        valid_ids
    )

    return {
        "ids": valid_ids,
        "tokens": tokens,
        "length": len(valid_ids),
    }


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.cohort)

    print("Rows:", len(rows))
    print("Loading DNABERT tokenizer...")

    adapter = DnaBertAdapter.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device=args.device,
    )

    output_rows = []
    length_differences = []
    exact_id_matches = 0
    equal_token_lengths = 0

    for index, row in enumerate(rows):
        clean_sequence = get_sequence_column(
            row,
            (
                "clean_sequence",
                "sequence",
            ),
        )

        corrupt_sequence = get_sequence_column(
            row,
            (
                "mono_corrupt_sequence",
                "mononucleotide_corrupt_sequence",
                "corrupt_sequence",
                "mono_sequence",
            ),
        )

        clean = tokenize_single(
            adapter,
            clean_sequence,
        )

        corrupt = tokenize_single(
            adapter,
            corrupt_sequence,
        )

        same_length = (
            clean["length"] == corrupt["length"]
        )

        exact_ids = (
            clean["ids"] == corrupt["ids"]
        )

        if same_length:
            equal_token_lengths += 1

        if exact_ids:
            exact_id_matches += 1

        difference = (
            int(corrupt["length"])
            - int(clean["length"])
        )

        length_differences.append(difference)

        output_rows.append(
            {
                "row_index": index,
                "localization_index": row.get(
                    "localization_index",
                    "",
                ),
                "source_index": row.get(
                    "source_index",
                    "",
                ),
                "label": int(row["label"]),
                "clean_token_count": clean["length"],
                "corrupt_token_count": corrupt["length"],
                "token_count_difference": difference,
                "same_token_count": same_length,
                "exact_token_id_match": exact_ids,
                "clean_tokens": json.dumps(
                    clean["tokens"]
                ),
                "corrupt_tokens": json.dumps(
                    corrupt["tokens"]
                ),
            }
        )

        print(
            f"\rProcessed {index + 1}/{len(rows)}",
            end="",
            flush=True,
        )

    print()

    differences = np.asarray(
        length_differences,
        dtype=np.int64,
    )

    label_counts = Counter(
        int(row["label"])
        for row in rows
    )

    summary = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "rows": len(rows),
        "label_counts": dict(label_counts),
        "equal_token_length_count": (
            equal_token_lengths
        ),
        "equal_token_length_fraction": float(
            equal_token_lengths / len(rows)
        ),
        "different_token_length_count": (
            len(rows) - equal_token_lengths
        ),
        "exact_token_id_match_count": (
            exact_id_matches
        ),
        "token_count_difference": {
            "mean": float(differences.mean()),
            "median": float(
                np.median(differences)
            ),
            "minimum": int(differences.min()),
            "maximum": int(differences.max()),
            "mean_absolute": float(
                np.mean(np.abs(differences))
            ),
        },
        "difference_histogram": {
            str(key): int(value)
            for key, value in sorted(
                Counter(
                    differences.tolist()
                ).items()
            )
        },
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
                output_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(output_rows)

    print("\n===== Token-alignment summary =====")
    print(
        "Same token count:",
        f"{equal_token_lengths}/{len(rows)}",
        f"({100 * equal_token_lengths / len(rows):.1f}%)",
    )
    print(
        "Different token count:",
        len(rows) - equal_token_lengths,
    )
    print(
        "Mean absolute token-count difference:",
        f"{np.mean(np.abs(differences)):.3f}",
    )
    print(
        "Difference range:",
        int(differences.min()),
        "to",
        int(differences.max()),
    )
    print(
        "Exact token-ID matches:",
        exact_id_matches,
    )

    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print("TOKEN ALIGNMENT INSPECTION SUCCEEDED")


if __name__ == "__main__":
    main()
