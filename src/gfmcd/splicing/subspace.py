"""Subspace primitives for the donor-anchor causal tests (numpy, model-free).

``U``/``V`` are orthonormal ``[k, D]`` (rows are basis vectors); ``rows``/``h`` are ``[N, D]``.
Prereg: ``studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_DONOR_ANCHOR.md``.
"""
from __future__ import annotations

import numpy as np


def top_k_subspace(rows, k: int) -> np.ndarray:
    """Top-k right singular vectors of ``rows`` [N,D] -> orthonormal U [k,D]."""
    _, _, Vh = np.linalg.svd(np.asarray(rows, float), full_matrices=False)
    return Vh[:k]


def project(h, U) -> np.ndarray:
    """Component of ``h`` [N,D] in span(U) [k,D] (U orthonormal rows)."""
    h = np.asarray(h, float)
    return (h @ U.T) @ U


def project_out(h, U) -> np.ndarray:
    return np.asarray(h, float) - project(h, U)


def patch_component(h_target, h_source, U) -> np.ndarray:
    """Replace h_target's span(U) component with h_source's."""
    return project_out(h_target, U) + project(h_source, U)


def subspace_overlap(U, V) -> float:
    """Mean squared cosine of principal angles between span(U), span(V) [k,D] orthonormal."""
    s = np.linalg.svd(np.asarray(U, float) @ np.asarray(V, float).T, compute_uv=False)
    return float(np.mean(np.clip(s, 0.0, 1.0) ** 2))


def random_subspace(D: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random orthonormal U [k,D]."""
    Q, _ = np.linalg.qr(rng.standard_normal((D, k)))
    return Q.T


def energy_spectrum(rows) -> np.ndarray:
    """Cumulative fraction of squared-singular-value energy of ``rows`` [N,D]."""
    s = np.linalg.svd(np.asarray(rows, float), compute_uv=False)
    e = np.cumsum(s ** 2)
    return e / e[-1]
