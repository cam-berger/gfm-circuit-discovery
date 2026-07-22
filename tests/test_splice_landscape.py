from __future__ import annotations

import pytest

from gfmcd.splicing.splice_landscape import (
    compare_landscapes,
    overlapping_window_starts,
    summarize_landscape,
)


def test_overlapping_window_starts_cover_only_affected_windows():
    assert overlapping_window_starts(20, 8, 12, window_length=5) == tuple(range(4, 12))
    assert overlapping_window_starts(20, 0, 2, window_length=5) == (0, 1)
    assert overlapping_window_starts(20, 18, 20, window_length=5) == (14, 15)


def test_landscape_summary_and_creation_flag_use_native_or_strong_floor():
    scorer = lambda window: float(window.count("G"))
    native = "A" * 20
    variant = native[:8] + "GGGG" + native[12:]
    summary = summarize_landscape(
        native, 8, 12, window_length=5, scorer=scorer, strong_threshold=3.0
    )
    assert summary.max_score == 0.0
    assert summary.n_strong == 0
    change = compare_landscapes(
        native, variant, 8, 12, window_length=5, scorer=scorer, strong_threshold=3.0
    )
    assert change.delta_max == 4.0
    assert change.delta_n_strong > 0
    assert change.new_competitive_site


def test_landscape_rejects_nonfinite_scores_and_length_changes():
    with pytest.raises(ValueError, match="non-finite"):
        summarize_landscape("A" * 20, 8, 12, window_length=5, scorer=lambda _w: float("nan"))
    with pytest.raises(ValueError, match="equal-length"):
        compare_landscapes(
            "A" * 20, "A" * 19, 8, 12, window_length=5, scorer=lambda w: 0.0
        )


def test_maxent_scorers_refuse_to_score_noncanonical_core_motifs():
    try:
        from gfmcd.splicing.splice_landscape import maxent_scorers
        score3, score5 = maxent_scorers()
    except RuntimeError:
        pytest.skip("optional maxentpy unavailable")
    assert score3("C" * 23) == -100.0
    assert score5("C" * 9) == -100.0
    assert score3("C" * 18 + "AG" + "C" * 3) > -100.0
    assert score5("CAGGTAAGT") > -100.0
