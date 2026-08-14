from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing import positional_boundary_coherence as pbc


def _nested_fixture() -> tuple[list[dict[str, float]], list[bool], list[str], list[str]]:
    rng = np.random.default_rng(404)
    n_blocks, n_features = 15, 12
    strata_names = ["chr1|+", "chr1|-", "chr2|+", "chr2|-", "chr3|+"]
    latent = rng.normal(size=(n_blocks, n_features))
    rows: list[dict[str, float]] = []
    labels: list[bool] = []
    groups: list[str] = []
    strata: list[str] = []
    for block in range(n_blocks):
        for donor_phase in range(3):
            for acceptor_phase in range(3):
                values = (
                    latent[block]
                    + 0.15 * rng.normal(size=n_features)
                    + 0.02 * (donor_phase - acceptor_phase)
                )
                rows.append(
                    {f"f{index:02d}": float(value) for index, value in enumerate(values)}
                )
                labels.append(donor_phase == acceptor_phase)
                groups.append(f"b{block:02d}")
                strata.append(strata_names[block % len(strata_names)])
    return rows, labels, groups, strata


def _calibration_cells(n_blocks: int = 10) -> list[dict[str, object]]:
    rng = np.random.default_rng(707)
    cells: list[dict[str, object]] = []
    for block in range(n_blocks):
        chrom = f"chr{1 + block % 3}"
        strand = "+" if block % 2 else "-"
        latent = rng.normal(size=8)
        for donor_phase in range(3):
            for acceptor_phase in range(3):
                values = latent + 0.1 * rng.normal(size=8)
                cells.append(
                    {
                        "block_id": f"b{block:02d}",
                        "chrom": chrom,
                        "strand": strand,
                        "donor_phase": donor_phase,
                        "acceptor_phase": acceptor_phase,
                        "phase_match": donor_phase == acceptor_phase,
                        "covariates": {
                            f"f{index:02d}": float(value)
                            for index, value in enumerate(values)
                        },
                    }
                )
    return cells


def test_nested_auc_matches_preoptimization_golden_and_parallel_replay(monkeypatch):
    rows, labels, groups, strata = _nested_fixture()
    monkeypatch.setattr(pbc, "_AUC_OUTER_WORKERS", 1)
    sequential = pbc.nested_crossfit_auc(
        rows,
        labels,
        groups,
        strata,
        seed=771,
        n_boot=37,
        inner_folds=3,
    )
    # liblinear probabilities can differ in their last floating-point bits across
    # CPU/BLAS backends, so their raw-byte digest is a same-environment replay
    # receipt rather than a portable numerical golden. The estimand, interval,
    # selected hyperparameters, and sequential/parallel replay remain pinned below.
    prediction_sha256 = sequential["prediction_sha256"]
    assert len(prediction_sha256) == 64
    assert set(prediction_sha256) <= set("0123456789abcdef")
    assert sequential["estimate"] == 0.4950617283950617
    assert sequential["ci_low"] == pytest.approx(0.4823950617283951, abs=1e-15)
    assert sequential["ci_high"] == pytest.approx(0.5130123456790123, abs=1e-15)
    assert [row["selected_c"] for row in sequential["folds"]] == [
        0.0001,
        0.001,
        10.0,
        1.0,
        0.1,
    ]

    monkeypatch.setattr(pbc, "_AUC_OUTER_WORKERS", 5)
    parallel = pbc.nested_crossfit_auc(
        rows,
        labels,
        groups,
        strata,
        seed=771,
        n_boot=37,
        inner_folds=3,
    )
    assert parallel == sequential


def test_vectorized_whole_block_bootstrap_matches_repeated_row_reference():
    rng = np.random.default_rng(5)
    rows: list[dict[str, float]] = []
    labels: list[bool] = []
    groups: list[str] = []
    strata: list[str] = []
    predictions: list[float] = []
    for block in range(18):
        for row in range(9):
            rows.append({"x": float(rng.normal())})
            labels.append(row < 3)
            groups.append(f"b{block:02d}")
            strata.append(f"chr{block % 3 + 1}|{'+' if block % 2 else '-'}")
            predictions.append(round(float(rng.random()), 2))  # exercise tied scores
    design = pbc._prepare_nested_auc_design(rows, groups, strata)
    label_array = np.asarray(labels, dtype=np.int8)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    counts = pbc._draw_group_bootstrap_counts(design, n_boot=211, seed=123)
    optimized = pbc._auc_from_group_counts(
        label_array, prediction_array, design.row_group_codes, counts
    )
    reference = pbc._reference_group_bootstrap_auc(
        design, label_array, prediction_array, n_boot=211, seed=123
    )
    assert np.max(np.abs(optimized - reference)) <= 2e-16
    for threshold in pbc._AUC_FROZEN_GATE_THRESHOLDS:
        old_low, old_high = np.quantile(reference, (0.025, 0.975))
        new_low, new_high = np.quantile(optimized, (0.025, 0.975))
        assert (old_low <= threshold <= old_high) == (
            new_low <= threshold <= new_high
        )
        assert (old_high <= threshold) == (new_high <= threshold)
        assert (old_low >= threshold) == (new_low >= threshold)


def test_bootstrap_replays_reference_at_a_frozen_threshold(monkeypatch):
    rows, labels, groups, strata = _nested_fixture()
    design = pbc._prepare_nested_auc_design(rows, groups, strata)
    label_array = np.asarray(labels, dtype=np.int8)
    predictions = np.full(len(labels), 0.5, dtype=np.float64)
    calls = 0
    reference = pbc._reference_group_bootstrap_auc

    def counted_reference(*args, **kwargs):
        nonlocal calls
        calls += 1
        return reference(*args, **kwargs)

    monkeypatch.setattr(pbc, "_reference_group_bootstrap_auc", counted_reference)
    result = pbc._bootstrap_auc(
        design, label_array, predictions, n_boot=31, seed=19
    )
    assert calls == 1
    assert np.array_equal(result, np.full(31, 0.5))


def test_calibration_reuses_base_feature_matrix_for_every_pseudo(monkeypatch):
    cells = _calibration_cells()
    feature_rows = [dict(cell["covariates"]) for cell in cells]
    groups = [str(cell["block_id"]) for cell in cells]
    strata = [f"{cell['chrom']}|{cell['strand']}" for cell in cells]
    design = pbc._prepare_nested_auc_design(feature_rows, groups, strata)
    calls = 0
    prepare = pbc._prepare_nested_auc_design

    def counted_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return prepare(*args, **kwargs)

    monkeypatch.setattr(pbc, "_prepare_nested_auc_design", counted_prepare)
    result = pbc.calibration_hooks(
        cells,
        seed=701,
        outer_folds=5,
        inner_folds=2,
        auc_n_boot=11,
        auc_upper_threshold=0.55,
        pseudo_family_size=3,
        _auc_design=design,
    )
    assert calls == 1  # only the planted-label feature adds a new matrix
    assert result["pseudo_label_family"]["n"] == 3
    assert len(result["pseudo_label_family"]["assignments"]) == 3
