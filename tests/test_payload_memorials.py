"""Memorial / invariant tests for donor-graph experiment 6 (A2b payload localization).

Pure torch/numpy (no model); the load-bearing memorials (value_linearity, U_info projector
properties on real activations) run live in the runner's preflight.  Here we check the analytic
kernel: the projector built from an orthonormal basis is a symmetric idempotent projector, the
delivered value edit is linear, and the head-output reconstruction is the intended sum.
"""
from __future__ import annotations

import numpy as np
import torch

from gfmcd.splicing.payload import deliver, delivered_energy, head_output, projector
from gfmcd.splicing.subspace import random_subspace, top_k_subspace


def test_projector_is_symmetric_idempotent():
    rng = np.random.default_rng(0)
    U = random_subspace(16, 4, rng)                  # [4,16] orthonormal rows
    P = projector(U, device="cpu")
    assert P.shape == (16, 16)
    assert torch.allclose(P, P.T, atol=1e-6)                     # symmetric
    assert torch.allclose(P @ P, P, atol=1e-5)                  # idempotent
    # complement is orthogonal to P
    I = torch.eye(16)
    assert torch.allclose(P @ (I - P), torch.zeros(16, 16), atol=1e-5)


def test_projector_matches_subspace_project():
    # projector(U) applied as x @ P must equal subspace.project (rows @ Uᵀ) @ U
    rng = np.random.default_rng(1)
    U = random_subspace(12, 3, rng)
    P = projector(U, device="cpu")
    x = torch.randn(5, 12)
    ref = torch.tensor((x.numpy() @ U.T) @ U, dtype=torch.float32)
    assert torch.allclose(x @ P, ref, atol=1e-5)


def test_decomposition_sums_to_delta():
    rng = np.random.default_rng(2)
    U = random_subspace(20, 6, rng)
    P = projector(U, device="cpu")
    I = torch.eye(20)
    dr = torch.randn(4, 3, 20)                        # [B,K,hidden]
    assert torch.allclose(dr @ P + dr @ (I - P), dr, atol=1e-5)


def test_deliver_is_linear_value_edit():
    torch.manual_seed(0)
    Wv = torch.randn(8, 20)                           # [Dh, hidden]
    dr = torch.randn(4, 3, 20)
    dV = deliver(Wv, dr)
    assert dV.shape == (4, 3, 8)
    assert torch.allclose(dV, dr @ Wv.T, atol=1e-6)
    # linearity: deliver(a+b) == deliver(a)+deliver(b)
    dr2 = torch.randn(4, 3, 20)
    assert torch.allclose(deliver(Wv, dr + dr2), dV + deliver(Wv, dr2), atol=1e-5)


def test_head_output_is_base_plus_weighted_window():
    torch.manual_seed(1)
    base = torch.randn(4, 8)
    a = torch.randn(4, 3)
    dV = torch.randn(4, 3, 8)
    got = head_output(base, a, dV)
    exp = base + torch.einsum("bk,bkd->bd", a, dV)
    assert torch.allclose(got, exp, atol=1e-6)
    # zero window edit -> base (the address_floor payload)
    assert torch.allclose(head_output(base, a, torch.zeros_like(dV)), base, atol=1e-6)


def test_delivered_energy_is_frobenius_over_window():
    dV = torch.randn(4, 3, 8)
    e = delivered_energy(dV)
    assert e.shape == (4,)
    assert torch.allclose(e, dV.flatten(1).norm(dim=1), atol=1e-6)


def test_top_k_subspace_is_orthonormal():
    rng = np.random.default_rng(3)
    rows = rng.standard_normal((50, 24))
    U = top_k_subspace(rows, 8)                       # [8,24]
    assert U.shape == (8, 24)
    assert np.allclose(U @ U.T, np.eye(8), atol=1e-6)            # orthonormal rows
