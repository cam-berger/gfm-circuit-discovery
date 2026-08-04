"""Restore a clean attention-head output into corrupted DNABERT runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from gfmcd.adapters.dnabert import DnaBertAdapter


MODEL_NAME = "quietflamingo/dnabert2-no-flashattention"
MODEL_REVISION = "813031b2bf86d9e960a027e2734c908009f31601"

PROBE_PATH = Path(
    "results/dnabert_promoter/frozen/"
    "layer_probes/selected_layer_2_probe.joblib"
)

DEFAULT_COHORT = Path(
    "results/dnabert_promoter/causal/"
    "localization_cohorts/validation_localization_256.csv"
)

OUTPUT_DIR = Path(
    "results/dnabert_promoter/causal/"
    "activation_restoration"
)

SELECTED_LAYER = 2
HIDDEN_SIZE = 768
NUM_HEADS = 12
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cohort",
        type=Path,
        default=DEFAULT_COHORT,
    )

    parser.add_argument(
        "--output-name",
        default="validation_L0H8_restoration_pilot_16",
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


def get_corrupt_sequence(
    row: dict[str, str],
) -> str:
    candidates = (
        "mono_sequence",
        "mono_corrupt_sequence",
        "mononucleotide_corrupt_sequence",
        "corrupt_sequence",
    )

    for column in candidates:
        value = row.get(column)

        if value:
            return value

    raise KeyError(
        "Could not locate mononucleotide-corrupt sequence"
    )


def tokenize_with_offsets(
    adapter: DnaBertAdapter,
    sequence: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = adapter.tokenizer(
        sequence,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=False,
    )

    token_ids = list(encoded["input_ids"])
    offsets = [
        (int(start), int(end))
        for start, end in encoded["offset_mapping"]
    ]

    return token_ids, offsets


def assert_adapter_tokenization_matches(
    adapter: DnaBertAdapter,
    sequence: str,
    expected_ids: list[int],
) -> None:
    tokenized = adapter.tokenize([sequence])

    valid_mask = tokenized.attention_mask[0].to(
        dtype=torch.bool
    )

    actual_ids = (
        tokenized.input_ids[0][valid_mask]
        .detach()
        .cpu()
        .tolist()
    )

    if actual_ids != expected_ids:
        raise RuntimeError(
            "Offset tokenizer output does not match "
            "adapter tokenization"
        )


def build_overlap_mapping(
    clean_offsets: list[tuple[int, int]],
    corrupt_offsets: list[tuple[int, int]],
) -> list[list[tuple[int, float]]]:
    """Map each corrupt token to overlapping clean tokens."""

    mapping: list[list[tuple[int, float]]] = []

    for corrupt_start, corrupt_end in corrupt_offsets:
        if corrupt_end <= corrupt_start:
            mapping.append([])
            continue

        overlaps: list[tuple[int, int]] = []

        for clean_index, (
            clean_start,
            clean_end,
        ) in enumerate(clean_offsets):
            if clean_end <= clean_start:
                continue

            overlap = max(
                0,
                min(corrupt_end, clean_end)
                - max(corrupt_start, clean_start),
            )

            if overlap > 0:
                overlaps.append(
                    (clean_index, overlap)
                )

        total_overlap = sum(
            overlap
            for _, overlap in overlaps
        )

        if total_overlap == 0:
            mapping.append([])
            continue

        mapping.append(
            [
                (
                    clean_index,
                    overlap / total_overlap,
                )
                for clean_index, overlap in overlaps
            ]
        )

    return mapping


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
        raise RuntimeError(
            "Model returned no hidden states"
        )

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


def run_sequence(
    adapter: DnaBertAdapter,
    probe,
    sequence: str,
) -> float:
    tokenized = adapter.tokenize([sequence])

    with torch.inference_mode():
        feature = pool_selected_layer(
            adapter,
            tokenized,
        )

    logit = probe.decision_function(
        feature.detach().cpu().numpy()
    )

    return float(logit[0])


def capture_clean_head_output(
    adapter: DnaBertAdapter,
    sequence: str,
    *,
    layer: int,
) -> tuple[torch.Tensor, float]:
    module = (
        adapter.model.encoder.layer[layer]
        .attention.self
    )

    captured: dict[str, torch.Tensor] = {}

    def capture_hook(
        _module,
        _inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        captured["output"] = (
            output.detach().clone()
        )

        return output

    handle = module.register_forward_hook(
        capture_hook
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

    if "output" not in captured:
        raise RuntimeError(
            "Clean head output was not captured"
        )

    return captured["output"], feature


def run_patched_corrupt(
    adapter: DnaBertAdapter,
    probe,
    corrupt_sequence: str,
    *,
    layer: int,
    head: int,
    clean_attention_output: torch.Tensor,
    clean_offsets: list[tuple[int, int]],
    corrupt_offsets: list[tuple[int, int]],
) -> float:
    module = (
        adapter.model.encoder.layer[layer]
        .attention.self
    )

    start = head * HEAD_DIM
    end = start + HEAD_DIM

    mapping = build_overlap_mapping(
        clean_offsets,
        corrupt_offsets,
    )

    def restoration_hook(
        _module,
        _inputs,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if output.ndim != 2:
            raise ValueError(
                "Expected packed attention output"
            )

        if output.shape[-1] != HIDDEN_SIZE:
            raise ValueError(
                "Unexpected attention hidden size"
            )

        if output.shape[0] != len(corrupt_offsets):
            raise ValueError(
                "Corrupt offset count does not match "
                "attention token count"
            )

        if clean_attention_output.shape[0] != len(
            clean_offsets
        ):
            raise ValueError(
                "Clean offset count does not match "
                "captured attention token count"
            )

        patched = output.clone()

        for corrupt_index, contributors in enumerate(
            mapping
        ):
            if not contributors:
                continue

            restored_slice = torch.zeros(
                HEAD_DIM,
                dtype=output.dtype,
                device=output.device,
            )

            for clean_index, weight in contributors:
                restored_slice = (
                    restored_slice
                    + float(weight)
                    * clean_attention_output[
                        clean_index,
                        start:end,
                    ].to(
                        device=output.device,
                        dtype=output.dtype,
                    )
                )

            patched[
                corrupt_index,
                start:end,
            ] = restored_slice

        return patched

    handle = module.register_forward_hook(
        restoration_hook
    )

    try:
        patched_logit = run_sequence(
            adapter,
            probe,
            corrupt_sequence,
        )

    finally:
        handle.remove()

    return patched_logit


def summarize(
    values: np.ndarray,
) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1))
        if values.size > 1
        else 0.0,
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
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
        f"Restoring Layer {args.layer} Head {args.head}"
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

    result_rows = []

    for index, row in enumerate(rows):
        clean_sequence = row["clean_sequence"]
        corrupt_sequence = get_corrupt_sequence(row)

        clean_ids, clean_offsets = (
            tokenize_with_offsets(
                adapter,
                clean_sequence,
            )
        )

        corrupt_ids, corrupt_offsets = (
            tokenize_with_offsets(
                adapter,
                corrupt_sequence,
            )
        )

        assert_adapter_tokenization_matches(
            adapter,
            clean_sequence,
            clean_ids,
        )

        assert_adapter_tokenization_matches(
            adapter,
            corrupt_sequence,
            corrupt_ids,
        )

        clean_attention_output, clean_feature = (
            capture_clean_head_output(
                adapter,
                clean_sequence,
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

        corrupt_logit = run_sequence(
            adapter,
            probe,
            corrupt_sequence,
        )

        patched_logit = run_patched_corrupt(
            adapter,
            probe,
            corrupt_sequence,
            layer=args.layer,
            head=args.head,
            clean_attention_output=(
                clean_attention_output
            ),
            clean_offsets=clean_offsets,
            corrupt_offsets=corrupt_offsets,
        )

        total_damage = clean_logit - corrupt_logit
        restored_amount = (
            patched_logit - corrupt_logit
        )

        recovery_fraction = (
            restored_amount / total_damage
            if abs(total_damage) > 1e-8
            else float("nan")
        )

        result_rows.append(
            {
                "localization_index": row[
                    "localization_index"
                ],
                "source_index": row["source_index"],
                "clean_token_count": len(
                    clean_offsets
                ),
                "corrupt_token_count": len(
                    corrupt_offsets
                ),
                "clean_logit": clean_logit,
                "corrupt_logit": corrupt_logit,
                "patched_logit": patched_logit,
                "total_corruption_damage": (
                    total_damage
                ),
                "restored_amount": restored_amount,
                "recovery_fraction": (
                    recovery_fraction
                ),
            }
        )

        print(
            f"{index + 1:02d}/{len(rows)} "
            f"clean={clean_logit:+.3f} "
            f"corrupt={corrupt_logit:+.3f} "
            f"patched={patched_logit:+.3f} "
            f"recovery={recovery_fraction:+.3f}"
        )

    damage = np.asarray(
        [
            row["total_corruption_damage"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    restored = np.asarray(
        [
            row["restored_amount"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    recovery = np.asarray(
        [
            row["recovery_fraction"]
            for row in result_rows
        ],
        dtype=np.float64,
    )

    finite_recovery = recovery[
        np.isfinite(recovery)
    ]

    summary = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "selected_layer": SELECTED_LAYER,
        "patched_layer": args.layer,
        "patched_head": args.head,
        "rows": len(rows),
        "total_corruption_damage": summarize(
            damage
        ),
        "restored_amount": summarize(
            restored
        ),
        "recovery_fraction": summarize(
            finite_recovery
        ),
        "fraction_positive_restoration": float(
            np.mean(restored > 0)
        ),
        "fraction_negative_restoration": float(
            np.mean(restored < 0)
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

    print("\n===== Restoration summary =====")
    print(
        "Mean corruption damage:",
        f"{damage.mean():.6f}",
    )
    print(
        "Mean restored amount:",
        f"{restored.mean():.6f}",
    )
    print(
        "Mean recovery fraction:",
        f"{finite_recovery.mean():.6f}",
    )
    print(
        "Positive restoration:",
        f"{100 * np.mean(restored > 0):.1f}%",
    )

    print("\nSaved:", summary_path)
    print("Saved:", detail_path)
    print("HEAD RESTORATION PILOT SUCCEEDED")


if __name__ == "__main__":
    main()
