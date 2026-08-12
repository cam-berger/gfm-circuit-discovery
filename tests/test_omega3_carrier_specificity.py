"""Memorial: Omega3 is exactly zero under any r-invariant readout.

This is the property that makes the successor design's primary endpoint carrier-specific BY
CONSTRUCTION rather than by argument, so it deserves a check that can fail rather than a
paragraph of prose.

The design holds the token sequence IDENTICAL across the carrier worlds r in {-1, 0, +1};
only the model's learned period-3 positional coordinate is moved. The rival hypothesis --
that the model reads raw period-3 sequence energy with no positional coordinate at all --
therefore predicts a readout that CANNOT depend on r. This file proves that such a readout
gives Omega3 == 0 exactly, so any resolved nonzero Omega3 excludes the rival arithmetically.

The confound the rival would exploit is real and measured: on 109 authentic gene-disjoint
donor/acceptor pairs, phase-aligned matched joins carry about 1.28x the period-3 power of
mismatched joins, present in 70.6% of pairs. That contaminates Psi_design (which equals S_0)
and is excluded from Omega3 by the identity below.
"""

from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing.pbc_register_relocation import preferred_delta

REGISTERS = (-1, 0, 1)


def orientation_contrast(g_r, klass: int) -> float:
    """T_r(c) = G(c, r) - mean over the other two classes at the same r."""
    others = [d for d in (0, 1, 2) if d != klass]
    return float(g_r[klass] - np.mean([g_r[d] for d in others]))


def omega3(g_by_r) -> float:
    """Omega3 = mean over r of S_r, with S_r = T_r(preferred_delta(r))."""
    return float(np.mean([orientation_contrast(g_by_r[r], preferred_delta(r)) for r in REGISTERS]))


def test_orientation_contrasts_sum_to_zero_at_every_register():
    """The identity behind everything else. Fails if T_r is defined against the WHOLE mean
    (including class c) instead of against the other two classes only."""
    rng = np.random.default_rng(20260811)
    for _ in range(2000):
        g = rng.normal(size=3)
        total = sum(orientation_contrast(g, c) for c in (0, 1, 2))
        assert abs(total) < 1e-12


def test_omega3_is_exactly_zero_for_any_r_invariant_readout():
    """THE memorial. A readout that ignores r -- the raw-periodicity rival, which sees
    identical tokens in every carrier world -- must give exactly zero.

    Fails if the three S_r were ever redefined to select the SAME class at every r, because
    then the sum no longer telescopes and an r-invariant readout would leak through."""
    rng = np.random.default_rng(20260812)
    worst = 0.0
    for _ in range(2000):
        shared = rng.normal(size=3)
        worst = max(worst, abs(omega3({r: shared for r in REGISTERS})))
    assert worst < 1e-12, f"an r-invariant readout leaked into Omega3: {worst:.3e}"


def test_the_three_registers_select_three_distinct_classes():
    """Why the telescoping works. If two registers selected the same class the identity
    would break, and the previous test would stop being a real constraint."""
    assert sorted(preferred_delta(r) for r in REGISTERS) == [0, 1, 2]


def test_omega3_is_nonzero_for_a_planted_r_dependent_ridge():
    """Sensitivity. Without this the zero-result test could be satisfied by a function that
    returns 0.0 for every input."""
    rng = np.random.default_rng(20260813)
    values = []
    for _ in range(2000):
        g_by_r = {}
        for r in REGISTERS:
            base = rng.normal(size=3)
            base[preferred_delta(r)] += 0.5
            g_by_r[r] = base
        values.append(omega3(g_by_r))
    assert np.mean(values) > 0.25
    assert np.mean(np.array(values) > 0) > 0.6


def test_psi_design_is_the_r_zero_member_not_an_independent_quantity():
    """Psi_design == S_0. It is diagnostic only, and it is the member of the family that the
    raw-periodicity confound contaminates."""
    rng = np.random.default_rng(20260814)
    g = rng.normal(size=3)
    assert preferred_delta(0) == 0
    assert orientation_contrast(g, 0) == pytest.approx(
        omega3({-1: np.zeros(3), 0: g, 1: np.zeros(3)}) * 3.0
    )
