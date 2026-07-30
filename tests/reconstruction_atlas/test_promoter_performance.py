from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gfmcd.reconstruction_atlas import promoter_source_execution
from gfmcd.reconstruction_atlas.promoter_performance import (
    summarize_masked_classification,
)


def test_masked_classification_counts_flips_and_metric_changes() -> None:
    rows = [
        {
            "split": "development",
            "pair_id": "a",
            "target_clean_margin": 2.0,
            "target_masked_margin": -1.0,
            "control_clean_margin": -2.0,
            "control_masked_margin": -1.0,
        },
        {
            "split": "development",
            "pair_id": "b",
            "target_clean_margin": 1.0,
            "target_masked_margin": 0.5,
            "control_clean_margin": -0.5,
            "control_masked_margin": 0.25,
        },
    ]
    result = summarize_masked_classification(rows, bootstrap_draws=100, seed=7)
    assert result["clean"]["accuracy"] == 1.0
    assert result["masked"]["accuracy"] == 0.5
    assert result["changes"]["accuracy"] == -0.5
    assert result["label_transitions"]["target"]["promoter_to_non_promoter"] == 1
    assert result["label_transitions"]["control"]["non_promoter_to_promoter"] == 1
    assert result["bootstrap"]["draws"] == 100


def test_masked_classification_rejects_duplicate_or_nonfinite_rows() -> None:
    row = {
        "split": "development",
        "pair_id": "a",
        "target_clean_margin": 1.0,
        "target_masked_margin": 0.0,
        "control_clean_margin": -1.0,
        "control_masked_margin": -0.5,
    }
    with pytest.raises(ValueError):
        summarize_masked_classification([row, row], bootstrap_draws=10, seed=1)
    with pytest.raises(ValueError):
        summarize_masked_classification(
            [{**row, "target_clean_margin": float("nan")}],
            bootstrap_draws=10,
            seed=1,
        )


def test_score_equal_dose_masked_tile_records_decision_margins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    offsets = torch.tensor(
        [
            [[0, 0], [0, 5], [5, 10], [0, 0]],
            [[0, 0], [0, 5], [5, 10], [0, 0]],
        ]
    )
    batch = SimpleNamespace(
        n_pairs=1,
        pair_ids=("pair-a",),
        offset_mapping=offsets,
        token_counts=(4, 4),
    )
    monkeypatch.setattr(
        promoter_source_execution,
        "tokenize_promoter_pairs",
        lambda _bundle, _pairs: batch,
    )
    monkeypatch.setattr(
        promoter_source_execution,
        "promoter_margin_tensor",
        lambda _bundle, _batch: torch.tensor([2.0, -2.0]),
    )
    monkeypatch.setattr(
        promoter_source_execution,
        "explicit_token_mask_effects_for_batch",
        lambda *_args, **_kwargs: {
            "target_loss": [0.5],
            "control_loss": [0.25],
            "target_specific_loss": [0.25],
        },
    )
    rows = promoter_source_execution.score_equal_dose_masked_tile(
        SimpleNamespace(),
        [("pair-a", "A" * 300, "C" * 300)],
        tile_index=0,
        tile_width_nt=10,
        token_dose=2,
        maximum_cover_nt=12,
        maximum_cover_difference_nt=0,
        batch_pairs=1,
    )
    assert len(rows) == 1
    assert rows[0]["target_clean_margin"] == 2.0
    assert rows[0]["target_masked_margin"] == 1.5
    assert rows[0]["control_clean_margin"] == -2.0
    assert rows[0]["control_masked_margin"] == -2.25
