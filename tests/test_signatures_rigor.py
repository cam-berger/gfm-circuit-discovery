"""Memorial tests for RRR track-2 findings F3 (depth-selection null) and F4 (held-out probe).

Model-free: they exercise the statistical machinery directly with synthetic residuals, so they
document WHY the old behavior was wrong without needing a trained model.
"""
from __future__ import annotations

import numpy as np
import torch

from gfmcd.signatures import _depth_max_null_p, _layer_effects, fit_probe, probe_accuracy


# ---- F4: in-sample probe accuracy is inflated; held-out is honest ----
def test_insample_probe_inflates_vs_heldout():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, 8, generator=g)
    y = torch.randint(0, 3, (300,), generator=g)  # RANDOM labels: no real signal, chance = 1/3
    probe, mu, sd = fit_probe(x, y, steps=300)
    in_sample = probe_accuracy(probe, mu, sd, x, y)          # reads on the SAME data it fit
    tr, te = torch.arange(210), torch.arange(210, 300)
    p2, m2, s2 = fit_probe(x[tr], y[tr], steps=300)
    held_out = probe_accuracy(p2, m2, s2, x[te], y[te])      # reads on unseen data
    assert in_sample > held_out + 0.05, "in-sample probe should overfit random labels"
    assert abs(held_out - 1 / 3) < 0.17, "held-out accuracy on random labels must sit near chance"


# ---- F3: max-over-depth needs a null; a real effect clears it, pure noise does not ----
def _resid(rng, N, S, D, phase=None, signal=False):
    base = torch.from_numpy(rng.standard_normal((N, S, D)).astype("float32")) * 0.3
    if signal and phase is not None:
        oneh = np.eye(3, D, dtype="float32")[phase]  # one-hot phase in the first 3 dims
        base += torch.from_numpy(oneh) * 3.0
    return base


def _masks(N, S):
    phase = np.random.default_rng(1).integers(0, 3, (N, S))
    mask = np.ones((N, S), dtype=bool)
    far = np.tile(np.arange(S), (N, 1)) >= S // 2
    return phase, mask, far, far.copy()


def test_depth_null_flags_signal_and_clears_noise():
    N, S, D = 60, 12, 8
    phase, mask, far, far_ctl = _masks(N, S)
    rng = np.random.default_rng(7)

    # SIGNAL: 3 layers, one carries phase; ablation (R1) destroys it at far positions -> real drop.
    layers = ["L0", "L1", "L2"]
    R0 = {n: _resid(rng, N, S, D, phase, signal=(n == "L1")) for n in layers}
    R1 = {n: R0[n].clone() for n in layers}
    R1["L1"][torch.from_numpy(far)] = torch.from_numpy(rng.standard_normal((int(far.sum()), D)).astype("float32"))
    R2 = {n: R0[n].clone() for n in layers}  # control: no change -> control drop ~ 0
    obs, p_sig, _ = _depth_max_null_p(R0, R1, R2, mask, far, far_ctl, phase, n_null=60, steps=150, seed=0)
    assert obs > 0.3 and p_sig < 0.05, f"real depth effect must clear the null (obs={obs:.2f}, p={p_sig})"

    # NO SIGNAL: all-noise residuals -> the argmax over 3 layers is inflated, but the permutation
    # null is inflated the same way, so the observed peak is NOT significant.
    Rn0 = {n: _resid(rng, N, S, D) for n in layers}
    Rn1 = {n: _resid(rng, N, S, D) for n in layers}
    Rn2 = {n: _resid(rng, N, S, D) for n in layers}
    _, p_noise, _ = _depth_max_null_p(Rn0, Rn1, Rn2, mask, far, far_ctl, phase, n_null=60, steps=150, seed=0)
    assert p_noise > 0.1, f"pure-noise depth-argmax must not be significant (p={p_noise})"
