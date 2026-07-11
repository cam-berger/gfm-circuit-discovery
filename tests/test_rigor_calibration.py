"""Memorial tests for RRR findings 1-2 (calibration + multiplicity).

These turn the audit argument into executable facts:
 - the max-null p-value machinery is uniform on pure noise (else selection inflates);
 - a planted signal is still detected (else the audit is vacuously conservative);
 - the finite-sample p-value never returns 0 (which would break FDR);
 - BH-FDR controls discoveries and required_null_draws sizing is honest.
"""
from __future__ import annotations

import numpy as np

from gfmcd.rigor import (
    audit_calibration,
    benjamini_hochberg,
    empirical_upper_p_value,
    max_null_calibrate,
    required_null_draws,
)


def test_max_null_uniform_on_noise():
    """Pure noise through the real selection path -> uniform max-p, ~0 FDR discoveries."""
    rows = audit_calibration(
        lambda rng, size: rng.standard_normal(size),
        lambda rng, size: (lambda r: r.standard_normal(size)),
        search_sizes=(16, 48), n_searches=80, n_null=199, random_seed=0)
    for r in rows:
        assert r["uniform_ok"], f"max-p not uniform at size {r['search_size']}"
        assert r["fpr_ok"], f"FPR exceeds alpha at size {r['search_size']}"
        assert r["discoveries_at_fdr"] <= 0.02, "discoveries on pure noise"


def test_planted_signal_is_detected():
    """A strong planted effect must clear the same null the noise cannot."""
    rng = np.random.default_rng(0)
    observed = np.concatenate([[6.0], rng.standard_normal(15)])  # one real hit among 16
    cal = max_null_calibrate(observed, lambda r: r.standard_normal(16), n_null=499, random_seed=1)
    assert cal.p[0] < 0.05 and cal.q[0] < 0.10, "planted signal not recovered -> audit is vacuous"
    assert cal.p[1:].min() >= 0.02, "noise candidate flagged -> anti-conservative"


def test_empirical_p_never_zero():
    """(1 + #{null>=obs})/(n+1) has a floor of 1/(n+1); plain #/n could be 0 and break FDR."""
    p = empirical_upper_p_value(1e9, np.zeros(100))
    assert p == 1 / 101


def test_bh_and_required_draws():
    q = benjamini_hochberg([0.001, 0.2, 0.5, 0.9])
    assert q[0] < 0.05 and (np.diff(np.sort(q)) >= -1e-12).all()
    # to let one true hit pass BH among 96 tests at q=0.1 you need >= ~960 null draws
    assert required_null_draws(96, q=0.1, expected_hits=1) == 960
