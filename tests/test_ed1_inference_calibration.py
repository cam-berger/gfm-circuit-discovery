"""ED1 inference layer must be CALIBRATED, not merely dyadic-looking.

A decision rule whose false-positive rate is not ~alpha is not a decision rule.  The first
ED1 implementation was a Bayesian-bootstrap weighted mean with Exp(1) gene weights averaged
over a pair's two genes; ``Var((w_a+w_b)/2) = 0.5`` for ``w ~ Exp(1)``, so it reproduced
half the sampling variance, made intervals ~sqrt(2) too narrow, and rejected true nulls at
**FPR ~0.19 at every n** (flat in n -- a bias, not a small-sample effect).  It was caught by
simulation before any confirmatory data existed.  This test keeps it caught.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gfmcd.splicing.exon_intron import dyadic_bootstrap, twoway_wild_p  # noqa: E402

N_PAIRS, N_GENES, N_SIM, N_BOOT = 800, 700, 150, 250
ALPHA = 0.05


def _dyadic_sample(rng, effect=0.0, gene_sd=0.5):
    """Pairs loading on two genes each, with gene-level random effects and heavy tails —
    the dependence structure ED1's matched pairs actually have."""
    ga = rng.integers(0, N_GENES, N_PAIRS)
    gb = rng.integers(0, N_GENES, N_PAIRS)
    g = rng.normal(0, gene_sd, N_GENES)
    v = rng.standard_t(2, N_PAIRS) * 0.5 + 0.5 * (g[ga] + g[gb]) + effect
    return v, [f"g{i}" for i in ga], [f"g{i}" for i in gb]


def _fpr(fn):
    rng = np.random.default_rng(20260725)
    return sum(fn(*_dyadic_sample(rng), s) for s in range(N_SIM)) / N_SIM


def _boot_rejects(v, ga, gb, s):
    _, lo, hi = dyadic_bootstrap(v, ga, gb, n_boot=N_BOOT, seed=s, alpha=ALPHA)
    return not (lo <= 0 <= hi)


def _wild_rejects(v, ga, gb, s):
    return twoway_wild_p(v, ga, gb, n_perm=250, seed=s) <= ALPHA


def test_dyadic_bootstrap_false_positive_rate_is_near_alpha():
    fpr = _fpr(_boot_rejects)
    # Monte-Carlo SE at alpha=0.05, N_SIM=150 is ~0.018; the broken scheme sat at ~0.19.
    assert 0.01 <= fpr <= 0.12, f"dyadic bootstrap FPR={fpr:.3f}, expected ~{ALPHA}"


def test_twoway_wild_false_positive_rate_is_near_alpha():
    fpr = _fpr(_wild_rejects)
    assert 0.01 <= fpr <= 0.12, f"two-way wild FPR={fpr:.3f}, expected ~{ALPHA}"


def test_dyadic_bootstrap_has_power_against_a_real_effect():
    rng = np.random.default_rng(7)
    hits = sum(_boot_rejects(*_dyadic_sample(rng, effect=0.30), s) for s in range(60))
    assert hits / 60 >= 0.8, f"power={hits/60:.2f} at effect 0.30 — estimator is too blunt"


def test_interval_is_not_degenerate():
    """A CI that always contains zero would pass the FPR test while testing nothing."""
    rng = np.random.default_rng(3)
    v, ga, gb = _dyadic_sample(rng, effect=1.0)
    m, lo, hi = dyadic_bootstrap(v, ga, gb, n_boot=N_BOOT, seed=0)
    assert lo > 0, f"CI [{lo:.3f},{hi:.3f}] contains 0 at a unit-sized true effect"
    assert hi > lo and np.isfinite(m)


@pytest.mark.parametrize("n_boot", [200, 400])
def test_dyadic_bootstrap_is_seed_reproducible(n_boot):
    rng = np.random.default_rng(11)
    v, ga, gb = _dyadic_sample(rng)
    a = dyadic_bootstrap(v, ga, gb, n_boot=n_boot, seed=5)
    b = dyadic_bootstrap(v, ga, gb, n_boot=n_boot, seed=5)
    assert a == b
