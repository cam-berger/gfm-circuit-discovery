from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert_head_context_atlas/analyze_atlas_discovery.py"
SPEC = importlib.util.spec_from_file_location("atlas_analysis", SCRIPT)
atlas = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(atlas)


def test_bh_adjust_is_monotone_in_rank_and_order_preserving() -> None:
    p = np.asarray([0.04, 0.001, 0.03, 0.9])
    q = atlas.bh_adjust(p)
    assert np.allclose(q, [0.05333333333333334, 0.004, 0.05333333333333334, 0.9])
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_gene_cluster_stats_collapse_within_gene_dependence() -> None:
    values = np.asarray([[1.0], [1.2], [-1.0], [-0.8]])
    stats_clustered = atlas.gene_cluster_stats(values, ["g1", "g1", "g2", "g2"])
    stats_unique = atlas.gene_cluster_stats(values, ["g1", "g2", "g3", "g4"])
    assert np.isclose(stats_clustered["point"][0], 0.1)
    assert stats_clustered["se"][0] > stats_unique["se"][0]
    assert stats_clustered["n_genes"] == 2


def test_dyadic_stats_reduce_to_observation_meat_for_unique_endpoints() -> None:
    values = np.asarray([[-1.0], [1.0]])
    stats = atlas.dyadic_stats(
        values,
        case_genes=["a1", "a2"],
        control_genes=["b1", "b2"],
        weights=np.ones(2),
    )
    # Sum incident squares=4, subtract one observation diagonal=2, finite correction=4/3.
    expected = np.sqrt(2 * (4 / 3)) / 2
    assert np.isclose(stats["se"][0], expected)
    assert stats["n_endpoint_genes"] == 4
    assert stats["n_effective"] == 2


def test_dyadic_stats_reject_same_gene_endpoint() -> None:
    try:
        atlas.dyadic_stats(
            np.asarray([[1.0]]), ["g"], ["g"], np.ones(1)
        )
    except ValueError as exc:
        assert "same endpoint" in str(exc)
    else:
        raise AssertionError("same-gene dyad should fail")


def test_dyadic_stats_reject_repeated_endpoint_dyad() -> None:
    try:
        atlas.dyadic_stats(
            np.asarray([[1.0], [2.0]]),
            ["a", "a"],
            ["b", "b"],
            np.ones(2),
        )
    except ValueError as exc:
        assert "repeated endpoint-gene dyad" in str(exc)
    else:
        raise AssertionError("repeated endpoint dyad should fail closed")


def test_classification_respects_material_and_equivalence_bounds() -> None:
    assert atlas.classify(0.08, 0.06, 0.10, 0.01, 0.05) == "material_support"
    assert atlas.classify(-0.08, -0.10, -0.06, 0.01, 0.05) == "material_suppress"
    assert atlas.classify(0.02, 0.01, 0.03, 0.01, 0.05) == "detected_subfloor"
    assert atlas.classify(0.0, -0.04, 0.04, 1.0, 0.05) == "bounded_small"
    assert atlas.classify(0.0, -0.08, 0.08, 1.0, 0.05) == "inconclusive"


def _general(component: str, score: float, eligible: bool = True) -> dict:
    return {
        "component_id": component,
        "estimand": "A",
        "score_bound_excess": score,
        "general_shortlist_eligible": eligible,
    }


def _context(component: str, contrast: str, role: str, score: float) -> dict:
    return {
        "component_id": component,
        "contrast": contrast,
        "target_role": role,
        "estimand": "A",
        "score_bound_excess": score,
        "context_associated": True,
        "n_effective_pairs": 100.0,
        "mde80": 0.02,
    }


def test_shortlist_caps_components_and_contrasts_and_preserves_roles() -> None:
    general = [_general(f"L0H{i}", 1.0 - i / 10) for i in range(6)]
    context = [
        _context("L1H0", "x|donor", "donor", 1.0),
        _context("L1H0", "y|donor", "donor", 0.9),
        _context("L1H0", "z|donor", "donor", 0.8),
        _context("L1H0", "w|donor", "donor", 0.7),
        _context("L2H0", "a|acceptor", "acceptor", 0.6),
        _context("L2H1", "a|acceptor", "acceptor", 0.5),
        _context("L2H2", "a|acceptor", "acceptor", 0.4),
        _context("L3H0", "b|acceptor", "acceptor", 0.3),
        _context("L3H1", "c|donor", "donor", 0.2),
    ]
    result = atlas.build_shortlist(general, context, tau=0.05, min_context_neff=80)
    assert len(result["general"]["selected"]) == 4
    selected = result["context"]["selected"]
    assert len(selected) <= 8
    assert {row["target_role"] for row in selected} == {"donor", "acceptor"}
    assert max(
        sum(row["component_id"] == component for row in selected)
        for component in {row["component_id"] for row in selected}
    ) <= 3
    assert max(
        sum(row["contrast"] == contrast for row in selected)
        for contrast in {row["contrast"] for row in selected}
    ) <= 2
