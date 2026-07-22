from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/splicebert/match_p3_phase_surgery_contexts.py"
)
SPEC = importlib.util.spec_from_file_location("match_p3_phase_surgery_contexts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_exact_strand_match_never_crosses_strand_and_is_one_to_one():
    left = [
        {"gene_id": f"l{i}", "strand": strand, "covariates": np.array([float(i), 0.0])}
        for i, strand in enumerate(("+", "+", "-"))
    ]
    right = [
        {"gene_id": f"r{i}", "strand": strand, "covariates": np.array([float(i), 0.0])}
        for i, strand in enumerate(("+", "-", "-"))
    ]
    pairs = MODULE.exact_strand_match(left, right)
    assert len(pairs) == 2
    assert all(a["strand"] == b["strand"] for a, b, _distance in pairs)
    assert len({a["gene_id"] for a, _b, _d in pairs}) == 2
    assert len({b["gene_id"] for _a, b, _d in pairs}) == 2


def test_block_gc_and_smd_are_finite():
    assert MODULE.block_gc("ACGT" * 10, [[0, 4], [4, 8]]) == 0.5
    left = np.array([[0.0, 1.0], [2.0, 3.0], [1.0, 2.0]])
    right = left + 0.2
    assert np.isfinite(MODULE.standardized_mean_difference(left, right)).all()


def test_balance_pruning_is_deterministic_and_respects_minimum():
    pairs = []
    for index in range(12):
        left = {"gene_id": f"l{index}", "covariates": np.array([float(index), 0.0])}
        right = {
            "gene_id": f"r{index}",
            "covariates": np.array([float(index) + (20.0 if index == 11 else 0.0), 0.0]),
        }
        pairs.append((left, right, float(index)))
    kept, dropped = MODULE.prune_pairs_for_balance(pairs, max_abs_smd=0.25, min_pairs=8)
    assert len(kept) >= 8
    assert dropped
    left = np.vstack([pair[0]["covariates"] for pair in kept])
    right = np.vstack([pair[1]["covariates"] for pair in kept])
    assert np.max(np.abs(MODULE.standardized_mean_difference(left, right))) <= 0.25


def test_within_gene_matches_are_taken_before_cross_gene_assignment():
    left = [
        {"gene_id": "shared", "strand": "+", "covariates": np.array([0.0, 0.0])},
        {"gene_id": "left", "strand": "+", "covariates": np.array([2.0, 0.0])},
    ]
    right = [
        {"gene_id": "shared", "strand": "+", "covariates": np.array([3.0, 0.0])},
        {"gene_id": "right", "strand": "+", "covariates": np.array([2.0, 0.0])},
    ]
    pairs = MODULE.within_gene_then_strand_match(left, right)
    assert any(a["gene_id"] == b["gene_id"] == "shared" for a, b, _distance in pairs)
