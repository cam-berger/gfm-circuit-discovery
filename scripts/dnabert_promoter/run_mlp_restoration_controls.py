"""Restore clean Layer 0 MLP output into corrupted DNABERT runs."""

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
    HIDDEN_SIZE,
    SELECTED_LAYER,
    tokenize_with_offsets,
    assert_adapter_tokenization_matches,
    build_overlap_mapping,
    get_corrupt_sequence,
)


PATCHED_LAYER = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT,
    )
    parser.add_argument(
        "--output-name",
        default="validation_L0_mlp_controls_16",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--donor-offset",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    return parser.parse_args()


def load_promoters(
    path: Path,
    limit: int,
) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    promoters = [
        row
        for row in rows
        if int(row["label"]) == 1
    ]

    if len(promoters) < limit:
        raise ValueError(
            f"Requested {limit} promoters but only "
            f"{len(promoters)} are available"
        )

    return promoters[:limit]


def mlp_output_module(
    adapter: DnaBertAdapter,
):
    return (
        adapter.model.encoder.layer[PATCHED_LAYER]
        .mlp.wo
    )


def pool_selected_layer(
    adapter: DnaBertAdapter,
    tokenized,
) -> torch.Tensor:
    output = adapter.forward(
        tokenized,
        output_hidden_states=True,
        output_attentions=False,
    )

    if output.hidden_states is None:
        raise RuntimeError("No hidden states returned")

    hidden = output.hidden_states[SELECTED_LAYER]

    biological_mask = adapter.biological_token_mask(
        tokenized
    ).to(
        device=hidden.device,
        dtype=hidden.dtype,
    )

    mask = biological_mask.unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)

    return (hidden * mask).sum(dim=1) / denominator


def capture_mlp_and_logit(
    adapter: DnaBertAdapter,
    probe,
    sequence: str,
) -> tuple[torch.Tensor, float]:
    captured: dict[str, torch.Tensor] = {}

    def capture_hook(
        _module,
        _inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        captured["output"] = output.detach().clone()
        return output

    handle = mlp_output_module(
        adapter
    ).register_forward_hook(capture_hook)

    try:
        tokenized = adapter.tokenize([sequence])

        with torch.inference_mode():
            feature = pool_selected_layer(
                adapter,
                tokenized,
            )
    finally:
        handle.remove()

    if "output" not in captured:
        raise RuntimeError(
            "Layer 0 MLP output was not captured"
        )

    logit = float(
        probe.decision_function(
            feature.detach().cpu().numpy()
        )[0]
    )

    return captured["output"], logit


def make_mlp_patch_hook(
    *,
    donor_output: torch.Tensor,
    donor_offsets: list[tuple[int, int]],
    recipient_offsets: list[tuple[int, int]],
):
    mapping = build_overlap_mapping(
        donor_offsets,
        recipient_offsets,
    )

    def hook(
        _module,
        _inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if output.ndim != 2:
            raise ValueError(
                f"Expected packed MLP output, got "
                f"{tuple(output.shape)}"
            )

        if output.shape[-1] != HIDDEN_SIZE:
            raise ValueError(
                f"Expected hidden size {HIDDEN_SIZE}, "
                f"got {output.shape[-1]}"
            )

        if output.shape[0] != len(
            recipient_offsets
        ):
            raise ValueError(
                "Recipient offsets do not match "
                "MLP output length"
            )

        if donor_output.shape[0] != len(
            donor_offsets
        ):
            raise ValueError(
                "Donor offsets do not match "
                "captured MLP output length"
            )

        patched = output.clone()

        for recipient_token, contributors in enumerate(
            mapping
        ):
            if not contributors:
                continue

            restored = torch.zeros(
                HIDDEN_SIZE,
                dtype=output.dtype,
                device=output.device,
            )

            for donor_token, weight in contributors:
                restored += (
                    float(weight)
                    * donor_output[
                        donor_token
                    ].to(
                        device=output.device,
                        dtype=output.dtype,
                    )
                )

            patched[recipient_token] = restored

        return patched

    return hook


def run_mlp_patch(
    adapter: DnaBertAdapter,
    probe,
    sequence: str,
    *,
    donor_output: torch.Tensor,
    donor_offsets: list[tuple[int, int]],
    recipient_offsets: list[tuple[int, int]],
) -> float:
    handle = mlp_output_module(
        adapter
    ).register_forward_hook(
        make_mlp_patch_hook(
            donor_output=donor_output,
            donor_offsets=donor_offsets,
            recipient_offsets=recipient_offsets,
        )
    )

    try:
        tokenized = adapter.tokenize([sequence])

        with torch.inference_mode():
            feature = pool_selected_layer(
                adapter,
                tokenized,
            )
    finally:
        handle.remove()

    return float(
        probe.decision_function(
            feature.detach().cpu().numpy()
        )[0]
    )


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("Cannot summarize empty array")

    return {
        "mean": float(values.mean()),
        "std": (
            float(values.std(ddof=1))
            if values.size > 1
            else 0.0
        ),
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

    rows = load_promoters(
        args.cohort,
        args.limit,
    )

    if args.donor_offset % len(rows) == 0:
        raise ValueError(
            "--donor-offset maps recipients to themselves"
        )

    print("Promoter rows:", len(rows))
    print("Restoring Layer 0 MLP output")
    print("Donor offset:", args.donor_offset)

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

    clean_offsets = []
    corrupt_offsets = []

    print("\nPreparing offsets...")

    for index in range(len(rows)):
        clean_ids, clean_offset_row = (
            tokenize_with_offsets(
                adapter,
                clean_sequences[index],
            )
        )

        corrupt_ids, corrupt_offset_row = (
            tokenize_with_offsets(
                adapter,
                corrupt_sequences[index],
            )
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

    clean_outputs = []
    corrupt_outputs = []
    clean_logits = []
    corrupt_logits = []

    print("Capturing Layer 0 MLP outputs...")

    for index in range(len(rows)):
        clean_output, clean_logit = (
            capture_mlp_and_logit(
                adapter,
                probe,
                clean_sequences[index],
            )
        )

        corrupt_output, corrupt_logit = (
            capture_mlp_and_logit(
                adapter,
                probe,
                corrupt_sequences[index],
            )
        )

        clean_outputs.append(clean_output)
        corrupt_outputs.append(corrupt_output)
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

        matched_logit = run_mlp_patch(
            adapter,
            probe,
            corrupt_sequences[recipient_index],
            donor_output=clean_outputs[
                recipient_index
            ],
            donor_offsets=clean_offsets[
                recipient_index
            ],
            recipient_offsets=corrupt_offsets[
                recipient_index
            ],
        )

        mismatched_logit = run_mlp_patch(
            adapter,
            probe,
            corrupt_sequences[recipient_index],
            donor_output=clean_outputs[
                donor_index
            ],
            donor_offsets=clean_offsets[
                donor_index
            ],
            recipient_offsets=corrupt_offsets[
                recipient_index
            ],
        )

        self_corrupt_logit = run_mlp_patch(
            adapter,
            probe,
            corrupt_sequences[recipient_index],
            donor_output=corrupt_outputs[
                recipient_index
            ],
            donor_offsets=corrupt_offsets[
                recipient_index
            ],
            recipient_offsets=corrupt_offsets[
                recipient_index
            ],
        )

        clean_logit = clean_logits[recipient_index]
        corrupt_logit = corrupt_logits[
            recipient_index
        ]

        total_damage = clean_logit - corrupt_logit
        matched_restored = (
            matched_logit - corrupt_logit
        )
        mismatched_restored = (
            mismatched_logit - corrupt_logit
        )
        advantage = (
            matched_restored - mismatched_restored
        )
        self_change = (
            self_corrupt_logit - corrupt_logit
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

        result_rows.append(
            {
                "recipient_index": recipient_index,
                "recipient_source_index": row[
                    "source_index"
                ],
                "mismatched_donor_index": donor_index,
                "mismatched_donor_source_index": (
                    rows[donor_index]["source_index"]
                ),
                "clean_logit": clean_logit,
                "corrupt_logit": corrupt_logit,
                "total_corruption_damage": total_damage,
                "matched_logit": matched_logit,
                "matched_restored_amount": (
                    matched_restored
                ),
                "matched_recovery_fraction": (
                    matched_recovery
                ),
                "mismatched_logit": mismatched_logit,
                "mismatched_restored_amount": (
                    mismatched_restored
                ),
                "mismatched_recovery_fraction": (
                    mismatched_recovery
                ),
                "matched_minus_mismatched": advantage,
                "self_corrupt_logit": (
                    self_corrupt_logit
                ),
                "self_corrupt_minus_baseline": (
                    self_change
                ),
            }
        )

        print(
            f"{recipient_index + 1:02d}/{len(rows)} "
            f"matched={matched_restored:+.3f} "
            f"mismatched={mismatched_restored:+.3f} "
            f"advantage={advantage:+.3f} "
            f"self={self_change:+.8f}"
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

    self_change = np.asarray(
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
        "selected_layer": SELECTED_LAYER,
        "patched_layer": PATCHED_LAYER,
        "patched_component": "mlp.wo",
        "rows": len(rows),
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
            self_change
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

    print("\n===== Layer 0 MLP restoration summary =====")
    print(
        "Matched mean restoration:",
        f"{matched.mean():.6f}",
    )
    print(
        "Mismatched mean restoration:",
        f"{mismatched.mean():.6f}",
    )
    print(
        "Matched mean advantage:",
        f"{advantage.mean():.6f}",
    )
    print(
        "Matched better than mismatched:",
        f"{100 * np.mean(advantage > 0):.1f}%",
    )
    print(
        "Mean matched recovery fraction:",
        f"{np.nanmean(matched_recovery):.6f}",
    )
    print(
        "Self-corrupt maximum absolute change:",
        f"{np.max(np.abs(self_change)):.8f}",
    )

    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print("MLP RESTORATION CONTROLS SUCCEEDED")


if __name__ == "__main__":
    main()
