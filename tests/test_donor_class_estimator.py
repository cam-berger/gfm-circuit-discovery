"""Two-sample gene-clustered ratio for the donor-class contrast.

The treatment (GC-AG) and comparator (GT-AG) arms live on DIFFERENT rows, so the paired
estimator in acceptor_payload does not apply. And 140 of the 189 treatment genes also carry a
comparator intron, so the two arms are correlated through the gene: resampling them
independently would break that correlation and misstate the interval. The estimator therefore
resamples over the UNION of genes, taking each drawn gene's rows in both arms together.
"""

from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing.donor_class import gboot_two_sample_ratio


def test_identical_arms_give_a_ratio_of_exactly_one():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    g = ["g0", "g0", "g1", "g1"]
    est, lo, hi = gboot_two_sample_ratio(v, g, v, g, n_draws=200, seed=0, min_genes=2)
    assert est == pytest.approx(1.0)
    assert lo <= 1.0 <= hi


def test_planted_ratio_is_recovered():
    rng = np.random.default_rng(0)
    genes = [f"g{i//2}" for i in range(120)]
    base = np.abs(rng.normal(1.0, 0.1, 120)) + 0.5
    est, lo, hi = gboot_two_sample_ratio(
        base * 0.4, genes, base, genes, n_draws=400, seed=0, min_genes=30
    )
    assert est == pytest.approx(0.4, abs=0.02)
    assert hi < 1.0


def test_a_zero_treatment_arm_reads_as_ratio_zero():
    """rho ~ 0 is the scientifically decisive outcome and must be representable."""
    genes = [f"g{i}" for i in range(40)]
    base = np.full(40, 1.0)
    est, lo, hi = gboot_two_sample_ratio(
        np.zeros(40), genes, base, genes, n_draws=200, seed=0, min_genes=30
    )
    assert est == pytest.approx(0.0)
    assert hi < 0.1


def test_the_resampling_unit_is_the_gene():
    """One gene carrying every row must collapse the interval to a point."""
    v_t = np.linspace(0.2, 0.6, 20)
    v_c = np.linspace(0.5, 1.5, 20)
    g = ["g0"] * 20
    est, lo, hi = gboot_two_sample_ratio(v_t, g, v_c, g, n_draws=50, seed=0, min_genes=1)
    assert lo == pytest.approx(est)
    assert hi == pytest.approx(est)


def test_shared_genes_keep_the_arms_correlated():
    """Genes appearing in both arms must be drawn together, not independently.

    With strong gene-level co-variation the correlation-preserving interval is materially
    narrower than one built by resampling the arms independently.
    """
    rng = np.random.default_rng(3)
    n_genes = 80
    genes, v_t, v_c = [], [], []
    for i in range(n_genes):
        level = rng.normal(1.0, 0.6)  # gene-level effect shared by both arms
        genes.append(f"g{i}")
        v_t.append(0.5 * level)
        v_c.append(1.0 * level)
    v_t, v_c = np.asarray(v_t), np.asarray(v_c)

    _, lo, hi = gboot_two_sample_ratio(
        v_t, genes, v_c, genes, n_draws=400, seed=0, min_genes=30
    )
    paired = hi - lo

    rng2 = np.random.default_rng(0)
    draws = []
    for _ in range(400):
        it = rng2.integers(0, n_genes, n_genes)
        ic = rng2.integers(0, n_genes, n_genes)
        draws.append(v_t[it].sum() / v_c[ic].sum())
    independent = float(np.quantile(draws, 0.975) - np.quantile(draws, 0.025))

    assert paired < 0.5 * independent


def test_is_deterministic_under_a_fixed_seed():
    genes = [f"g{i//2}" for i in range(60)]
    v_t, v_c = np.full(60, 0.4), np.full(60, 0.9)
    a = gboot_two_sample_ratio(v_t, genes, v_c, genes, n_draws=100, seed=5, min_genes=30)
    b = gboot_two_sample_ratio(v_t, genes, v_c, genes, n_draws=100, seed=5, min_genes=30)
    assert a == b


def test_below_the_gene_floor_raises():
    g = [f"g{i}" for i in range(5)]
    with pytest.raises(ValueError):
        gboot_two_sample_ratio(
            np.ones(5), g, np.ones(5), g, n_draws=10, seed=0, min_genes=30
        )


def test_non_positive_comparator_sum_raises():
    g = [f"g{i}" for i in range(40)]
    with pytest.raises(ValueError):
        gboot_two_sample_ratio(
            np.ones(40), g, np.zeros(40), g, n_draws=10, seed=0, min_genes=30
        )


def test_unequal_arm_sizes_give_the_ratio_of_MEANS_not_of_sums():
    """The comparator arm is k:1, so its total is inflated by k.

    Ratio of sums would divide rho by k and understate the treatment arm k-fold. This is a
    regression guard against exactly that: the arms below are 40 vs 120 rows, so sums give
    0.167 while the correct per-intron ratio is 0.5.
    """
    gt = [f"g{i}" for i in range(40)]
    gc = [f"g{i//3}" for i in range(120)]
    est, lo, hi = gboot_two_sample_ratio(
        np.full(40, 0.5), gt, np.full(120, 1.0), gc, n_draws=100, seed=0, min_genes=30
    )
    assert est == pytest.approx(0.5)
    assert est != pytest.approx((40 * 0.5) / (120 * 1.0))
    assert lo <= est <= hi


def test_k_to_one_matching_does_not_change_rho():
    """Duplicating every comparator must leave the per-intron ratio untouched."""
    g_t = [f"g{i}" for i in range(40)]
    v_t = np.full(40, 0.3)
    one = gboot_two_sample_ratio(v_t, g_t, np.full(40, 1.0), g_t,
                                 n_draws=50, seed=0, min_genes=30)[0]
    three = gboot_two_sample_ratio(v_t, g_t, np.full(120, 1.0),
                                   [f"g{i//3}" for i in range(120)],
                                   n_draws=50, seed=0, min_genes=30)[0]
    assert one == pytest.approx(three)
