"""Tripwires for the codon-phase-lattice experiment.  One per way it can lie to you.

These need no genome and no model.  They are algebra.  That is the point: the mistakes they
guard against are algebraic, and an algebraic mistake cannot be caught by looking at data.
"""
from __future__ import annotations

import numpy as np
import pytest


def _phase_correlated_values(n: int, dim: int, rng, *, period3: float = 1.0) -> np.ndarray:
    """Value vectors along a sequence, with period-3 autocorrelation -- i.e. CODING DNA.

    v_j = period3 * s[j % 3] + noise.  The base at j resembles the base at j+3 far more than the
    base at j+1.  That is not a fact about any model.  It is a fact about the genome.
    """
    s = rng.normal(size=(3, dim))
    return period3 * s[np.arange(n) % 3] + rng.normal(size=(n, dim)) * 0.5


def _roll_output(a: dict[int, float], v: np.ndarray, i: int, delta: int) -> np.ndarray:
    """out_i^delta = sum_k a_k v_{i+k+delta}   -- roll the routing, keep the values."""
    return sum(w * v[i + k + delta] for k, w in a.items())


def test_a_phase_pure_head_sawtooths_with_no_model_at_all():
    """THE MEMORIAL FOR THE SELECTION EFFECT.  Read this before reporting any sawtooth.

    Roll a PHASE-PURE attention profile off its phase and it takes a bigger hit than the same
    profile rolled by 3 -- for no reason whatsoever except that the sequence it reads is
    period-3 autocorrelated.

    There is no model here.  No weights.  No training.  No codon computation.  Only

        out_i = sum_k a_k v_{i+k}

    with `a` supported on k = 0 (mod 3), and `v` period-3 autocorrelated.  The sawtooth appears
    anyway, exactly as the algebra says it must:

        E|| out^d - out^0 ||^2 = sum_t w(t) [ 2C(t) - C(t+d) - C(t-d) ]

    C = autocovariance of v;  w(t) = sum_k a_k a_{k+t} = autocorrelation of the head's OWN
    attention profile.  A phase-pure head has w supported on t = 0 (mod 3) -- precisely the lags
    where C is periodic.  Phase purity MECHANICALLY amplifies the sawtooth.

    The 42 "lattice" heads were SELECTED for phase purity.  So a raw damage sawtooth, larger for
    them than for random heads, larger in exons than in introns, is GUARANTEED -- and is a
    picture of the genome, not of SpliceBERT.

    If you are about to report a raw sawtooth as evidence that the lattice is functional:
    don't.  Divide by the perturbation actually delivered.  See run_phase_shift.py.
    """
    rng = np.random.default_rng(0)
    v = _phase_correlated_values(400, 32, rng)
    a = {k: 1 / 5 for k in (3, 6, 9, 12, 15)}          # 100% of its mass on phase 0 (mod 3)

    pert = {}
    for d in range(1, 7):
        pert[d] = float(np.mean([
            np.linalg.norm(_roll_output(a, v, i, d) - _roll_output(a, v, i, 0))
            for i in range(50, 350)
        ]))

    in_phase = np.mean([pert[3], pert[6]])
    off_phase = np.mean([pert[1], pert[2], pert[4], pert[5]])

    assert off_phase > 1.5 * in_phase, (
        f"the selection effect did not reproduce (off {off_phase:.3f} vs in {in_phase:.3f}) -- "
        "if this test ever fails, the memorial is stale, but do NOT conclude the confound is gone"
    )


def test_the_sawtooth_survives_even_on_WHITE_NOISE_values():
    """THE SECOND, WORSE MEMORIAL.  The sawtooth does not even need the genome.

    This test was originally written to assert the OPPOSITE -- that killing the sequence's
    period-3 structure kills the sawtooth.  IT FAILED, and it was right to.  With i.i.d. random
    value vectors -- no codons, no autocorrelation, no biology of any kind -- a stride-3 head
    STILL sawtooths (off-phase 1.77 vs in-phase 0.95).

    The mechanism is set overlap, and it is embarrassingly simple.  A comb on offsets
    {3, 6, 9, 12, 15}:

        rolled by 3 -> reads {6, 9, 12, 15, 18}   -- FOUR of its five old positions
        rolled by 1 -> reads {4, 7, 10, 13, 16}   -- NONE of them

    Rolling a stride-3 comb by a multiple of 3 slides it ONTO ITSELF.  It keeps reading most of
    the same tokens, so its output barely moves.  Rolling it by 1 moves it to a disjoint set.
    No sequence statistics are involved at any point.

    The exact statement (see the next test): for white-noise values,

        E|| out^d - out^0 ||^2  =  2 sigma^2 [ w(0) - w(d) ]

    -- the perturbation size is a function of the head's OWN attention-profile autocorrelation
    w, and of NOTHING ELSE.

    So: ANY stride-3 head sawtooths, on ANY input, coding or not, trained or not.  A raw damage
    sawtooth is not evidence about SpliceBERT, about codons, or about the genome.  It is
    evidence that the head has a stride-3 comb -- which is the criterion by which it was
    SELECTED into the lattice in the first place.  That is circular, and it is R7's shape: a
    result guaranteed by construction.

    NORMALISE BY THE PERTURBATION DELIVERED, OR REPORT NOTHING.
    """
    rng = np.random.default_rng(0)
    v = _phase_correlated_values(400, 32, rng, period3=0.0)      # WHITE NOISE. No codons.
    a = {k: 1 / 5 for k in (3, 6, 9, 12, 15)}

    pert = {d: float(np.mean([
        np.linalg.norm(_roll_output(a, v, i, d) - _roll_output(a, v, i, 0))
        for i in range(50, 350)
    ])) for d in range(1, 7)}

    in_phase = np.mean([pert[3], pert[6]])
    off_phase = np.mean([pert[1], pert[2], pert[4], pert[5]])

    assert off_phase > 1.5 * in_phase, (
        "the comb-overlap sawtooth did not reproduce on white noise -- if this ever fails, do "
        "NOT conclude the confound is gone; conclude the memorial is broken"
    )


def test_perturbation_size_is_the_profile_autocorrelation_and_nothing_else():
    """The identity behind the memorial above.  This is the whole confound, in one line.

        E|| out^d - out^0 ||^2  =  2 sigma^2 [ w(0) - w(d) ],     w(t) = sum_k a_k a_{k+t}

    for i.i.d. values with variance sigma^2 per component.  The sequence has been integrated
    out completely: the ONLY thing that shapes damage-vs-delta is the head's own comb.

    A stride-3 comb has w(t) nonzero ONLY at t = 0 (mod 3).  Therefore w(d) is large at
    d = 3, 6 and ZERO at d = 1, 2, 4, 5 -- so the perturbation is SMALL in phase and LARGE off
    phase.  Guaranteed.  Forever.  For every phase-pure head in every model ever trained.
    """
    rng = np.random.default_rng(7)
    dim, n = 64, 4000
    v = rng.normal(size=(n, dim))                       # i.i.d., sigma^2 = 1 per component
    a = {k: 1 / 5 for k in (3, 6, 9, 12, 15)}

    def w(t):                                          # profile autocorrelation
        return sum(a[k] * a.get(k + t, 0.0) for k in a)

    for d in (1, 2, 3, 4, 5, 6):
        emp = float(np.mean([
            np.linalg.norm(_roll_output(a, v, i, d) - _roll_output(a, v, i, 0)) ** 2
            for i in range(40, n - 40)
        ]))
        pred = 2 * dim * (w(0) - w(d))                 # sigma^2 = 1, summed over `dim` comps
        assert abs(emp - pred) / pred < 0.06, (
            f"delta={d}: empirical {emp:.2f} vs predicted {pred:.2f} -- the confound algebra "
            "in run_phase_shift.py's docstring is WRONG and every conclusion drawn from it "
            "must be revisited"
        )


def _pert_curve(a, v, deltas=range(1, 10)):
    return {d: float(np.mean([
        np.linalg.norm(_roll_output(a, v, i, d) - _roll_output(a, v, i, 0))
        for i in range(60, len(v) - 60)
    ])) for d in deltas}


def _sawtooth(pert):
    inp = np.mean([pert[d] for d in pert if d % 3 == 0])
    off = np.mean([pert[d] for d in pert if d % 3])
    return off / inp


def test_the_copier_control_is_valid_ONLY_IF_the_values_are_not_period3():
    """WHY THE COPIER IS THE CONTROL -- AND THE CONDITION UNDER WHICH IT IS NOT.

    This test was written asserting a COPIER's perturbation is flat in delta.  IT FAILED
    (ratio 2.33), and the failure is the point.

    The general identity is

        E|| out^d - out^0 ||^2  =  sum_t w(t) [ 2C(t) - C(t+d) - C(t-d) ]

    For a single-tooth head, w(t) = delta_{t,0}, so it collapses to

        E|| out^d - out^0 ||^2  =  2 [ C(0) - C(d) ]

    -- a function of the VALUE-VECTOR AUTOCOVARIANCE C alone.  So:

      * If C is white, C(d) = 0 for all d != 0  ->  the copier's perturbation is FLAT.
        It cannot manufacture a phase sawtooth.  IT IS A VALID CONTROL.
      * If C is itself period-3, C(d) is LARGE at d = 0 (mod 3)  ->  the copier's perturbation
        is SMALL in phase.  IT SAWTOOTHS TOO, and the control is VOID.

    THE VALIDITY OF THE ENTIRE COMB-SHIFT DESIGN THEREFORE RESTS ON AN EMPIRICAL FACT ABOUT
    SPLICEBERT, NOT ON ALGEBRA: are its value vectors period-3 autocorrelated?

    MEASURED: they are not.  The 24 copiers' perturbation ratio (off-phase / in-phase) is
    1.033, flat to 3%, while the 6 combs sit at 2.057.  `run_comb_shift.py` reports this as a
    MANIPULATION CHECK on every run and screams if the copiers stop being flat.

    Do not delete that check.  It is not decoration -- it is the assumption the experiment
    stands on, and it is one the model could revoke at any time (a different model, a different
    layer, a different sequence distribution).
    """
    rng = np.random.default_rng(3)
    copier = {-1: 1.0}

    white = _phase_correlated_values(600, 32, rng, period3=0.0)
    assert abs(_sawtooth(_pert_curve(copier, white)) - 1.0) < 0.10, (
        "a copier is NOT flat even on white values -- the identity 2[C(0) - C(d)] is wrong "
        "and the whole confound model must be rebuilt"
    )

    coding = _phase_correlated_values(600, 32, rng, period3=1.0)
    assert _sawtooth(_pert_curve(copier, coding)) > 1.5, (
        "a copier failed to sawtooth on STRONGLY period-3 values -- if this ever passes, the "
        "caveat above is wrong, but do not therefore assume the control is unconditionally safe"
    )

    # And the comb sawtooths on BOTH -- which is why only the comb/copier CONTRAST identifies
    # anything, and only when the copiers are measured to be flat.
    comb = {k: 1 / 5 for k in (3, 6, 9, 12, 15)}
    assert _sawtooth(_pert_curve(comb, white)) > 1.5


def test_the_damage_design_matrix_is_full_rank():
    """b (the phase effect) must be identifiable from a (the displacement slope).

    damage(d) = c + a*d + b*1[d mod 3 != 0]   on d = 1..6.

    If the mod-3 indicator were a linear function of d, b would be unidentifiable and the fit
    would silently return an arbitrary split of the same signal between a and b.  It is not:
    """
    D = np.array([[1.0, d, float(d % 3 != 0)] for d in range(1, 7)])
    assert np.linalg.matrix_rank(D) == 3

    # and the dips at 3 AND 6 are what no monotone displacement effect can fake
    monotone = D @ np.array([0.0, 1.0, 0.0])          # pure displacement, no phase
    assert monotone[2] > monotone[1] and monotone[5] > monotone[4], (
        "a pure displacement effect must be INCREASING at d=3 and d=6; if it can dip there, "
        "the whole identification argument is void"
    )


def test_roll_by_zero_is_the_identity():
    """delta = 0 runs the FULL roll path and must change nothing.

    Not a tautology: the path rolls, zeroes the wrap-around, and renormalises the row.  Any of
    those can be wrong (wrong axis, off-by-one in the nucleotide block, renormalising a row that
    already summed to 1 but with a stale denominator).  delta=0 is the only cell where we know
    the answer in advance, which is exactly what makes it the calibration.
    """
    rng = np.random.default_rng(0)
    w = rng.random((2, 4, 16, 20))
    w = w / w.sum(-1, keepdims=True)
    lo, hi = 1, 19                                     # nucleotide columns; 0=CLS, 19=SEP

    out = w.copy()
    blk = np.roll(out[:, [1, 2], :, lo:hi], 0, axis=-1)   # delta = 0
    out[:, [1, 2], :, lo:hi] = blk
    out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)

    assert np.allclose(out, w, atol=1e-12)


@pytest.mark.parametrize("delta", [1, 2, 3, -1, -2, -3])
def test_roll_conserves_probability_and_drops_the_wrap(delta):
    """A rolled row must still be a convex combination, and must NOT wrap around the sequence.

    Wrap-around would make position 1 read position N -- a 1000nt jump reported as a 1nt shift.
    """
    rng = np.random.default_rng(1)
    w = rng.random((1, 1, 8, 20))
    w = w / w.sum(-1, keepdims=True)
    lo, hi = 1, 19

    out = w.copy()
    blk = np.roll(out[:, [0], :, lo:hi], delta, axis=-1)
    if delta > 0:
        blk[..., :delta] = 0.0
    else:
        blk[..., delta:] = 0.0
    out[:, [0], :, lo:hi] = blk
    out = out / np.maximum(out.sum(-1, keepdims=True), 1e-12)

    assert np.allclose(out.sum(-1), 1.0, atol=1e-9)
    assert (out >= 0).all()
