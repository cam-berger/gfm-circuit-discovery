"""gfmcd.inference: parity with every copy it replaces, and the behaviour it adds.

Parity targets are pinned or protocol-cited code that must not change: `circuit.gboot`,
`circuit.gboot_ratio`, `circuit.ratio_of_means`, `boundary_grammar.holm_adjust`, five Holm copies in
runners, and the Head-Context Atlas `gene_clustered_boot`. Runner functions are lifted from their
source with `ast`, so no runner module (or its model imports) is executed.
"""

from __future__ import annotations

import ast
from collections import defaultdict
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from gfmcd import inference
from gfmcd.inference import GroupedSample, holm
from gfmcd.splicing import boundary_grammar, circuit, grouped_stats

REPO = Path(__file__).resolve().parents[1]


def _lift(path: str, name: str):
    """Compile one top-level function from a runner without importing the runner."""
    tree = ast.parse((REPO / path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"np": np, "math": math, "defaultdict": defaultdict, "Sequence": Sequence}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def _fixture(seed: int, shift: float):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"ENSG{i:05d}" for i in rng.permutation(40)], rng.integers(1, 6, 40))
    numerator = rng.normal(0.4, 0.5, groups.size)
    denominator = rng.normal(shift, 0.6, groups.size)
    numerator[3], denominator[5] = np.nan, np.inf
    return numerator, denominator, groups


# Captured from grouped_stats before it became an adapter (same fixtures, same calls).
GOLDEN = {
    "mean": {"estimate": 0.34071397319729635, "ci_low": 0.2771151949451697,
             "ci_high": 0.4029760824215635, "n": 131, "n_groups": 40, "n_boot": 3000},
    "ratio": {"estimate": 0.3857360929374503, "ci_low": 0.3060271827940961,
              "ci_high": 0.48919914855176966, "n": 130, "n_groups": 40, "n_boot": 3000},
    "signflip": 0.24187906046976512,
    "ratio_weak": {"estimate": 5.1108259012193535, "ci_low": -31.204465432976253,
                   "ci_high": 49.935075047794165, "n": 114, "n_groups": 40, "n_boot": 3000},
}


def test_grouped_stats_adapters_return_their_pre_module_outputs() -> None:
    num, den, groups = _fixture(7, 0.9)
    weak_num, weak_den, weak_groups = _fixture(8, 0.08)
    assert grouped_stats.grouped_bootstrap(num, groups, n_boot=3000, seed=11) == GOLDEN["mean"]
    assert grouped_stats.grouped_bootstrap_ratio(num, den, groups, n_boot=3000, seed=11) == GOLDEN["ratio"]
    assert grouped_stats.grouped_signflip_pvalue(num - 0.3, groups, n_perm=2000, seed=11) == GOLDEN["signflip"]
    assert grouped_stats.grouped_bootstrap_ratio(
        weak_num, weak_den, weak_groups, n_boot=3000, seed=11) == GOLDEN["ratio_weak"]


def test_first_seen_order_reproduces_circuit_gboot_exactly() -> None:
    num, _, groups = _fixture(7, 0.9)
    keep = np.isfinite(num)
    interval = GroupedSample(num, groups, order="first_seen").mean(n_boot=2000, seed=0)
    assert (interval.low, interval.high) == circuit.gboot(num[keep].tolist(), groups[keep].tolist())
    # Sorted order is a different (equally valid) draw-to-gene mapping: the source of the
    # [0.044, 0.348] vs [0.035, 0.358] disagreement between the two pinned copies.
    sorted_interval = GroupedSample(num, groups).mean(n_boot=2000, seed=0)
    assert (sorted_interval.low, sorted_interval.high) != (interval.low, interval.high)


def test_ratio_matches_the_pinned_ratio_helpers_for_positive_denominators() -> None:
    num, den, groups = _fixture(7, 0.9)
    keep = np.isfinite(num) & np.isfinite(den)
    interval = GroupedSample(num, groups, order="first_seen").ratio(den, n_boot=2000, seed=0)
    low, high = circuit.gboot_ratio(num[keep], den[keep], groups[keep])
    assert interval.low == pytest.approx(low, rel=0, abs=1e-12)
    assert interval.high == pytest.approx(high, rel=0, abs=1e-12)
    assert interval.estimate == pytest.approx(circuit.ratio_of_means(num[keep], den[keep]), abs=1e-12)
    assert interval.sign_changes == 0


def test_ratio_refuses_a_bootstrap_that_passes_through_infinity() -> None:
    num, den, groups = _fixture(8, 0.08)
    with pytest.raises(ValueError, match="change sign"):
        GroupedSample(num, groups).ratio(den, n_boot=3000, seed=11)
    lenient = GroupedSample(num, groups).ratio(den, n_boot=3000, seed=11, strict_sign=False)
    assert lenient.sign_changes > 0
    assert lenient.legacy_dict() == GOLDEN["ratio_weak"]


def test_negative_denominator_keeps_its_sign() -> None:
    groups = np.repeat(np.arange(20), 3)
    den = np.full(groups.size, -1.0)
    num = 0.59 * den + np.random.default_rng(1).normal(0, 0.01, groups.size)
    interval = GroupedSample(num, groups).ratio(den, n_boot=500, seed=0)
    assert interval.estimate == pytest.approx(0.59, abs=0.01)
    assert 0.5 < interval.low <= interval.high < 0.7 and interval.sign_changes == 0


def test_gene_weighting_reproduces_the_atlas_gene_clustered_bootstrap() -> None:
    gene_clustered_boot = _lift(
        "scripts/splicebert_head_context_atlas/behavioral_pilot.py", "gene_clustered_boot")
    num, _, groups = _fixture(7, 0.9)
    atlas = gene_clustered_boot(list(zip(groups.tolist(), num.tolist())), sesoi=0.1, n_boot=4000, seed=0)
    interval = GroupedSample(num, groups, weighting="gene").mean(n_boot=4000, seed=0)
    assert interval.estimate == atlas["mean"]
    assert [interval.low, interval.high] == atlas["ci"]
    assert interval.n == atlas["n"] and interval.n_groups == atlas["n_genes"]
    # The two estimands genuinely differ on unbalanced genes: name the weighting.
    assert GroupedSample(num, groups).mean(n_boot=10, seed=0).estimate != interval.estimate


HOLM_COPIES = [
    ("scripts/splicebert/audit_bg_boundary.py", "_holm"),
    ("scripts/splicebert_head_context_atlas/analyze_atlas_mechanism_validation.py", "holm_adjust"),
    ("scripts/splicebert_head_context_atlas/validate_atlas_mechanism_analysis.py", "holm"),
    ("scripts/dnabert_promoter/run_component_donor_matrix.py", "holm_adjust"),
]


@pytest.mark.parametrize("seed", range(5))
def test_holm_matches_every_copy_including_ties(seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = rng.choice(np.round(rng.uniform(0, 0.3, 8), 3), size=15)  # forced ties
    expected = holm(p)
    np.testing.assert_array_equal(expected, boundary_grammar.holm_adjust(p))
    for path, name in HOLM_COPIES:
        np.testing.assert_array_equal(np.asarray(_lift(path, name)(p.tolist()), dtype=float), expected,
                                      err_msg=f"{path}:{name}")
    reject = _lift("scripts/splicebert_acceptor_coupling/run_r3_paired_component_scan.py", "holm")(
        {f"k{i}": float(v) for i, v in enumerate(p)}, alpha=0.05)
    assert reject == {f"k{i}": bool(v <= 0.05) for i, v in enumerate(expected)}


def test_holm_refuses_invalid_p_values() -> None:
    for bad in ([0.1, np.nan], [1.2], [[0.1]]):
        with pytest.raises(ValueError):
            holm(bad)


def test_interface_requires_draws_and_seed_and_reports_drops() -> None:
    num, _, groups = _fixture(7, 0.9)
    with pytest.raises(TypeError):
        GroupedSample(num, groups).mean()  # type: ignore[call-arg]
    interval = GroupedSample(num, groups).mean(n_boot=50, seed=0)
    assert interval.n_dropped == 1 and interval.n + interval.n_dropped == num.size
    assert interval.method == "gene-grouped percentile bootstrap; row-weighted mean; sorted groups; PCG64"
    assert list(interval.as_dict())[:6] == list(GOLDEN["mean"])
    with pytest.raises(ValueError):
        GroupedSample(num, groups, weighting="exon")  # type: ignore[arg-type]


def test_sorted_order_does_not_depend_on_row_order() -> None:
    num, den, groups = _fixture(7, 0.9)
    permutation = np.random.default_rng(3).permutation(num.size)
    a = GroupedSample(num, groups, weighting="gene").mean(n_boot=500, seed=2)
    b = GroupedSample(num[permutation], groups[permutation], weighting="gene").mean(n_boot=500, seed=2)
    assert (a.low, a.high) == pytest.approx((b.low, b.high), abs=1e-12)


def test_resampling_chunks_do_not_change_the_stream(monkeypatch) -> None:
    num, _, groups = _fixture(7, 0.9)
    whole = GroupedSample(num, groups).mean(n_boot=300, seed=5)
    monkeypatch.setattr(inference, "_CHUNK", 7)
    assert GroupedSample(num, groups).mean(n_boot=300, seed=5) == whole
