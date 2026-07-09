"""Synthetic superposition with known atoms — the standard identifiability testbed.

Test B needs activations whose ground-truth atoms are recoverable in principle, so that a
low MCC indicts the dictionary rather than the data. That means independently sampled
supports. Activations lifted from the planted model do *not* qualify: its phase feature is
a deterministic function of the other two, so its residual code takes only 45 distinct
values and no dictionary learner can separate atoms from their co-occurring combinations.
See `tests/test_b_dictionary.py::test_planted_code_is_not_identifiable`.
"""

from __future__ import annotations

import torch


def random_directions(n_atoms: int, d: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(n_atoms, d, generator=g)
    return dirs / dirs.norm(dim=-1, keepdim=True)


def sample_sparse_code(
    n_samples: int,
    n_atoms: int,
    k: int,
    seed: int = 0,
    mag_range: tuple[float, float] = (0.5, 1.5),
) -> torch.Tensor:
    """Exactly-k-sparse codes with independent supports and varying magnitudes."""
    g = torch.Generator().manual_seed(seed)
    code = torch.zeros(n_samples, n_atoms)

    support = torch.argsort(torch.rand(n_samples, n_atoms, generator=g), dim=-1)[:, :k]
    lo, hi = mag_range
    mags = torch.rand(n_samples, k, generator=g) * (hi - lo) + lo
    code.scatter_(1, support, mags)
    return code
