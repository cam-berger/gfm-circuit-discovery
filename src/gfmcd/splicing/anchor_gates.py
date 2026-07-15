"""The four donor-anchor causal tests.

``margins_fn(H)->np[N]`` applies the unembed + ``margin_hard`` to residuals ``H`` [N,D].
Sufficiency and necessity are exact duals (one-way span substitutions, never swaps);
``null_recoveries`` is the size-matched specificity null; ``alive_margin`` is the R8 check.

Prereg: ``studies/splicebert_partner_selection/PREREGISTRATION_DONOR_ANCHOR.md``.
"""
from __future__ import annotations

import numpy as np

from gfmcd.splicing.circuit import ratio_of_means
from gfmcd.splicing.subspace import patch_component, random_subspace


def suff_recovery(margins_fn, H_cl, H_co, m_cl, m_co, U) -> float:
    """Patch only span(U) from clean into corrupt; fraction of coupling recovered."""
    m = margins_fn(patch_component(H_co, H_cl, U))
    return ratio_of_means(m - m_co, m_cl - m_co)


def nec_fraction(margins_fn, H_cl, H_co, m_cl, m_co, U) -> float:
    """Replace clean's span(U) with the corrupt run's; fraction of coupling destroyed."""
    m = margins_fn(patch_component(H_cl, H_co, U))
    return ratio_of_means(m_cl - m, m_cl - m_co)


def null_recoveries(margins_fn, H_cl, H_co, m_cl, m_co, k, n_random, rng) -> np.ndarray:
    """Sufficiency recovery for ``n_random`` random rank-k subspaces (size-matched null)."""
    D = H_cl.shape[1]
    return np.array([
        suff_recovery(margins_fn, H_cl, H_co, m_cl, m_co, random_subspace(D, k, rng))
        for _ in range(n_random)
    ])


def alive_margin(margins_fn, h_baseline, N) -> float:
    """Mean donor margin when every donor residual is replaced by the no-info baseline."""
    return float(np.mean(margins_fn(np.tile(np.asarray(h_baseline, float), (N, 1)))))
