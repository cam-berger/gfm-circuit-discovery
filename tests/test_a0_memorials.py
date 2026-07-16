"""Memorial / invariant tests for donor-graph experiment 4 (A0 targeting provenance).

A failure here means the reconstruction arithmetic is wrong, not that a refactor broke a unit.
These are torch/numpy-only (no genome, no model); the model-dependent memorials
(positionwise reconstruction, stage-7 == clean, QK-reconstruction) run live in the runner's
preflight, exactly as the acceptor-address swap-identity check does.

Guards:
* the stage partition is complete and ordered (``clamp_after``);
* the three-curve endpoint identity ``S_K(7) = S_Q(7) = S_J(7)`` holds by construction (the tell
  that would fail if the curves secretly shared a query/key tensor);
* the mean-resample reference writes the role-mean at tracked positions and nothing elsewhere,
  carrying no example-specific a+/a(other) distinction (D-A0-1);
* the pure QK window readouts (per-token logits, LSE attraction) match hand-computation.
"""
from __future__ import annotations

import numpy as np
import torch

from gfmcd.splicing.provenance import (
    COMPONENTS, N_STAGES, attraction, clamp_after, gather_at, scatter_reference,
    shared_window_reference, three_curves, window_logits,
)


def test_clamp_after_partitions_the_stages():
    assert COMPONENTS[0] == "embed"
    assert N_STAGES == 7
    assert clamp_after(0) == COMPONENTS                 # stage 0: everything is reference
    assert clamp_after(7) == ()                         # stage 7: nothing clamped == clean
    assert clamp_after(3) == COMPONENTS[3:]             # clean through component 3, reference after
    # every component appears exactly once, embeddings first, then (layer, kind) in depth order
    blocks = [c for c in COMPONENTS if c != "embed"]
    assert blocks == [(0, "attn"), (0, "mlp"), (1, "attn"), (1, "mlp"), (2, "attn"), (2, "mlp")]


def test_three_curve_endpoint_identity():
    # At stage 7 the reconstructed query/key ARE the true full query/key, so all three curves
    # must coincide.  A divergence here would mean the curves are not sharing q_F/k_F correctly.
    torch.manual_seed(0)
    B, S, H, Dh = 3, 20, 4, 32
    q = torch.randn(B, S, H * Dh)
    k = torch.randn(B, S, H * Dh)
    donor = torch.tensor([5, 6, 7])
    wplus = torch.tensor([[10, 11, 12, 13]] * B)
    wother = torch.tensor([[2, 3, 4, 5]] * B)
    c = three_curves(q, k, q, k, donor, wplus, wother, Dh)
    assert np.allclose(c["S_K"], c["S_J"], atol=1e-6)
    assert np.allclose(c["S_Q"], c["S_J"], atol=1e-6)


def test_three_curves_isolate_key_and_query_sides():
    # S_K varies only the keys (query pinned at q_F); S_Q varies only the query (keys pinned at
    # k_F).  Feeding a stage query/key that differ from the final ones must move S_K via keys and
    # S_Q via query, and only S_J via both -- the whole point of the factorization.
    torch.manual_seed(1)
    B, S, H, Dh = 2, 16, 2, 32
    q_F, k_F = torch.randn(B, S, H * Dh), torch.randn(B, S, H * Dh)
    q_s, k_s = torch.randn(B, S, H * Dh), torch.randn(B, S, H * Dh)
    donor = torch.tensor([3, 4])
    wplus = torch.tensor([[8, 9, 10, 11]] * B)
    wother = torch.tensor([[1, 2, 3, 4]] * B)
    c = three_curves(q_s, k_s, q_F, k_F, donor, wplus, wother, Dh)
    # S_K uses (q_F, k_s); S_Q uses (q_s, k_F); S_J uses (q_s, k_s).  Independent randoms -> the
    # three are genuinely different quantities.
    assert not np.allclose(c["S_K"], c["S_J"])
    assert not np.allclose(c["S_Q"], c["S_J"])


def test_gather_at_selects_positions():
    t = torch.arange(2 * 5 * 3, dtype=torch.float).reshape(2, 5, 3)
    pos = torch.tensor([[0, 4], [2, 1]])
    got = gather_at(t, pos)
    assert torch.equal(got[0, 0], t[0, 0])
    assert torch.equal(got[0, 1], t[0, 4])
    assert torch.equal(got[1, 0], t[1, 2])
    assert torch.equal(got[1, 1], t[1, 1])


def test_scatter_reference_writes_role_mean_only_at_tracked_positions():
    device = "cpu"
    H = 3
    ncol = 4
    means = {c: torch.randn(ncol, H) for c in COMPONENTS}
    tracked = torch.tensor([[1, 4, 7, 9], [0, 2, 5, 8]])       # [B=2, P=4]
    ref = scatter_reference(means, tracked, seq_len=12, device=device)
    for c in COMPONENTS:
        t = ref[c]
        assert t.shape == (2, 12, H)
        # role-mean present at each tracked position, identical across examples (population mean)
        for b in range(2):
            for col in range(ncol):
                assert torch.allclose(t[b, tracked[b, col]], means[c][col])
        # everything else is zero (untracked positions are never read by a clamp)
        mask = torch.ones(2, 12, dtype=torch.bool)
        for b in range(2):
            mask[b, tracked[b]] = False
        assert torch.count_nonzero(t[mask]) == 0


def test_scatter_reference_carries_no_a_plus_vs_a_other_distinction():
    # The reference at the a+ column and at the a(other) column are BOTH population means, so a
    # reconstructed Delta_k at an all-reference stage is contributed by the (empty) clean prefix,
    # not by the reference -- the property D-A0-1 relies on.  Here: identical means -> zero diff.
    H = 2
    shared = torch.randn(1, H)
    means = {c: shared.repeat(2, 1) for c in COMPONENTS}       # col 0 == col 1 (a+ == a_other)
    tracked = torch.tensor([[3, 6]])
    ref = scatter_reference(means, tracked, seq_len=8, device="cpu")
    for c in COMPONENTS:
        assert torch.allclose(ref[c][0, 3], ref[c][0, 6])


def test_shared_window_reference_makes_a_plus_and_decoy_identical():
    # A1a neutralisation: writing the SAME per-offset reference to both windows makes the a+ and
    # decoy key windows identical at matched offsets -- the property `shared_ref_neutralises`
    # checks live on the model, tested here on the pure scatter.
    K, H = 4, 3
    refc = torch.randn(K, H)
    wplus = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    wother = torch.tensor([[4, 5, 6, 7], [26, 27, 28, 29]])
    t = shared_window_reference(refc, wplus, wother, seq_len=32, device="cpu")
    assert t.shape == (2, 32, H)
    for b in range(2):
        for d in range(K):
            assert torch.allclose(t[b, wplus[b, d]], refc[d])
            assert torch.allclose(t[b, wother[b, d]], refc[d])
            assert torch.allclose(t[b, wplus[b, d]], t[b, wother[b, d]])   # identical -> no discrim
    mask = torch.ones(2, 32, dtype=torch.bool)
    for b in range(2):
        mask[b, wplus[b]] = False
        mask[b, wother[b]] = False
    assert torch.count_nonzero(t[mask]) == 0


def test_window_logits_match_hand_computation():
    torch.manual_seed(2)
    B, S, H, Dh = 2, 12, 3, 32
    q = torch.randn(B, S, H * Dh)
    k = torch.randn(B, S, H * Dh)
    donor = torch.tensor([4, 7])
    win = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    got = window_logits(q, k, donor, win, Dh)                  # head H3 = 1 -> slice [32:64]
    for b in range(B):
        qh = q[b, donor[b], 32:64]
        for j in range(4):
            kh = k[b, win[b, j], 32:64]
            assert abs(float(got[b, j]) - float(qh @ kh) / (Dh ** 0.5)) < 1e-4


def test_attraction_is_lse_over_window_minus_background():
    torch.manual_seed(3)
    B, S, H, Dh = 2, 14, 2, 32
    q = torch.randn(B, S, H * Dh)
    k = torch.randn(B, S, H * Dh)
    donor = torch.tensor([3, 5])
    wplus = torch.tensor([[9, 10, 11, 12]] * B)
    wother = torch.tensor([[1, 2, 3, 4]] * B)
    wbg = torch.tensor([[6, 7, 8, 9]] * B)
    a = attraction(q, k, donor, wplus, wother, wbg, Dh)
    lse_p = torch.logsumexp(window_logits(q, k, donor, wplus, Dh), dim=-1)
    lse_bg = torch.logsumexp(window_logits(q, k, donor, wbg, Dh), dim=-1)
    assert np.allclose(a["a_plus_vs_bg"], (lse_p - lse_bg).numpy(), atol=1e-5)
