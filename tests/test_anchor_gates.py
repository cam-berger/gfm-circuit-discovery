import numpy as np

from gfmcd.splicing.anchor_gates import (
    alive_margin,
    nec_fraction,
    null_recoveries,
    suff_recovery,
)
from gfmcd.splicing.subspace import top_k_subspace


def _linear_margins_fn(direction, floor=0.0):
    """A fake unembed whose donor-margin is (h . dir_hat) + floor."""
    d = direction / np.linalg.norm(direction)
    return lambda H: np.asarray(H, float) @ d + floor


def test_lowrank_coupling_passes_suff_nec_and_beats_null():
    rng = np.random.default_rng(0)
    D, N = 64, 120
    d = rng.standard_normal(D)
    d /= np.linalg.norm(d)
    H_co = rng.standard_normal((N, D))
    H_cl = H_co + 1.5 * np.outer(rng.standard_normal(N) + 3.0, d)  # coupling ALONG d, rank-1
    mf = _linear_margins_fn(d)
    m_cl, m_co = mf(H_cl), mf(H_co)
    U = top_k_subspace(H_cl - H_co, 1)
    suff = suff_recovery(mf, H_cl, H_co, m_cl, m_co, U)
    assert suff > 0.95  # PASS branch
    assert nec_fraction(mf, H_cl, H_co, m_cl, m_co, U) > 0.95
    nulls = null_recoveries(mf, H_cl, H_co, m_cl, m_co, 1, 200, rng)
    assert suff > nulls.max()  # read direction beats every random one


def test_suff_fails_when_coupling_lives_in_the_svd_tail():
    # The coupling is a small POSITIVE signal in dim 0; dims 1.. carry huge unrelated
    # variance, so the top-8 SVD subspace of Delta-h misses the readout direction.
    rng = np.random.default_rng(1)
    D, N = 64, 200
    H_co = rng.standard_normal((N, D))
    dh = 8.0 * rng.standard_normal((N, D))  # large unrelated moves in dims 1..D-1
    dh[:, 0] = np.abs(rng.standard_normal(N)) + 1.0  # small positive coupling in dim 0
    H_cl = H_co + dh
    mf = lambda H: np.asarray(H, float)[:, 0]  # readout reads ONLY dim 0
    m_cl, m_co = mf(H_cl), mf(H_co)
    U = top_k_subspace(dh, 8)
    assert suff_recovery(mf, H_cl, H_co, m_cl, m_co, U) < 0.30  # FAIL branch reachable


def test_alive_collapses_for_reader_but_not_for_copier():
    rng = np.random.default_rng(2)
    D = 64
    d = rng.standard_normal(D)
    reader = _linear_margins_fn(d, floor=0.0)
    h_baseline = np.zeros(D)  # reader -> margin 0 == floor
    assert abs(alive_margin(reader, h_baseline, 10)) < 1e-9  # collapses to floor
    copier = lambda H: np.full(len(H), 5.0)  # ignores h
    assert alive_margin(copier, h_baseline, 10) == 5.0  # does NOT collapse -> C-Alive catches it
