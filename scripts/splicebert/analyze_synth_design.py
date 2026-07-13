#!/usr/bin/env python
"""PAPER MATH FOR THE SYNTHETIC ARM.  Zero GPU.  Run this BEFORE any training -- that is D26's rule.

D26 said: "compute H(composition) on paper before training."  This does that, and four more things
the pilot never checked.  Every number below is analytic or a seconds-of-CPU simulation of the
CORPUS ONLY -- no model, no MPS, no training.

    1. HEADROOM.  Exact, with the clip the generator actually applies.  Which depths can teach
       a model anything at all?
    2. THE CLIP.  make_corpus does  pW = clip(0.59 + depth*cos/2, 0.03, 0.97).  At pW0=0.59 the
       cosine's upper excursion CLIPS above depth 0.76.  "depth 1.0" is not depth 1.0.
    3. COHERENCE LENGTH -- the thing that actually kills the low-depth arms.  T ~ U(c-0.3, c+0.3)
       is a BAND, not a tone.  Two positions d apart drift out of phase by 2*pi*d*dT/T^2.  That
       caps how many periods a model can coherently integrate, which sets the phase-estimation
       noise, which is the real floor.  The entropy headroom is only the CEILING.
    4. THE GENERATOR IS A CONTROL THAT CAN FAIL, AND IT WAS NEVER RUN.  If make_corpus does not
       actually put the band in the sequence, every downstream null is an artifact.  Check it.
    5. THE QUANTIZATION NULL.  The corpus band is ~2.7 bins WIDE.  "Quantized" means P compresses
       it to ~1 bin.  So the experiment needs a number for what FAITHFUL representation would
       look like -- otherwise "sharp" has no referent.  Compute it.

Also: the effective table length is the CROP LENGTH, not max_position_embeddings.  Rows past the
crop are never trained.  So the table-length control must vary the CROP.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/splicebert"))
from run_synth_quantization import make_corpus  # noqa: E402  -- the REAL generator, not a copy

LN2 = math.log(2.0)
RULE = "=" * 94


def Hb(p):
    p = np.clip(np.asarray(p, float), 1e-15, 1 - 1e-15)
    return -(p * np.log(p) + (1 - p) * np.log1p(-p))


# ---------------------------------------------------------------- 1 & 2: headroom, with the clip
def headroom(pW0: float, depth: float, n: int = 200_000) -> dict:
    """EXACT achievable-loss gap on the W/S channel, applying the generator's own clip.

    Loss with no positional info : Hb(mean pW) + ln2
    Loss with the phase KNOWN    : E_theta[Hb(pW(theta))] + ln2
    The ln2 is the A-vs-T / G-vs-C coin flip.  It is IRREDUCIBLE BY CONSTRUCTION -- no model, at
    any size, can ever price it.  On this channel it is ~50% of the loss, thrown away.
    """
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    raw = pW0 + depth * np.cos(th) / 2.0
    pW = np.clip(raw, 0.03, 0.97)                       # <- exactly what make_corpus does
    clipped = float((raw != pW).mean())
    pbar = float(pW.mean())                             # the clip SHIFTS the mean composition too
    eff_depth = float(pW.max() - pW.min())              # peak-to-trough, after the clip
    no_pos = Hb(pbar) + LN2
    with_pos = float(Hb(pW).mean()) + LN2
    return dict(depth=depth, pbar=pbar, eff_depth=eff_depth, clipped_frac=clipped,
                no_pos=float(no_pos), with_pos=with_pos, gap=float(no_pos - with_pos),
                pct=float(100 * (no_pos - with_pos) / no_pos))


# ---------------------------------------------------------------- 3: coherence & phase noise
def coherence(centre: float, halfwidth: float) -> float:
    """Distance over which an unknown T within the band drifts 1 radian out of phase.

    phase(d) = 2*pi*d/T.  d(phase)/dT = -2*pi*d/T^2.  A wavelength error dT costs
    2*pi*d*dT/T^2 radians at distance d.  Set that to 1 rad:
    """
    return centre ** 2 / (2 * math.pi * halfwidth)


def phase_noise(depth: float, pbar: float, n_eff: float) -> dict:
    """Fisher info for the phase from n_eff Bernoulli(pW) samples, and what the resulting
    phase error does to the signal the model can actually exploit.

    I_phi = n * (depth/2)^2 * E[sin^2] / (pbar(1-pbar)) = n * depth^2 / (8 * pbar(1-pbar))
    A phase error ~N(0, sigma) attenuates a coherent cosine by E[cos(delta)] = exp(-sigma^2/2).
    """
    I = n_eff * depth ** 2 / (8 * pbar * (1 - pbar))
    sigma = float("inf") if I <= 0 else math.sqrt(1.0 / I)
    atten = math.exp(-min(sigma, 50) ** 2 / 2) if I > 0 else 0.0
    return dict(sigma_rad=sigma, sigma_deg=math.degrees(sigma), atten=atten)


# ---------------------------------------------------------------- 4 & 5: the generator, measured
def corpus_band(centre: float, depth: float, L: int, seed: int = 1,
                n: int = 2_000_000) -> dict:
    """Measure the band the generator ACTUALLY produces, in the table's own Fourier bins.

    This is the positive control the pilot never ran.  It also gives the FAITHFUL-representation
    null: if P simply mirrored the corpus's band, Q_win would be the corpus's own peak share.
    """
    seq = make_corpus(n, centre, depth, seed)
    isW = ((seq == 0) | (seq == 3)).astype(np.float64)      # A or T
    isW -= isW.mean()
    nwin = n // L
    X = isW[: nwin * L].reshape(nwin, L)
    pw = (np.abs(np.fft.rfft(X, axis=1)) ** 2).mean(0)       # power, averaged over windows
    pw[0] = 0.0
    band = np.arange(43, 54)
    q = pw[band] / pw[band].sum()
    win = int(band[int(np.argmax(q))])
    med = float(np.median(pw[1:]))
    # mode-domain band edges implied by T ~ U(centre-0.3, centre+0.3)
    m_lo, m_hi = L / (centre + 0.30), L / (centre - 0.30)
    return dict(centre=centre, win_bin=win, q_win=float(q.max()),
                q=dict(zip(band.tolist(), np.round(q, 4).tolist())),
                peak_over_median=float(pw[win] / med),
                mode_lo=m_lo, mode_hi=m_hi, mode_width_bins=m_hi - m_lo,
                mode_centre=L / centre)


def main() -> None:
    print(RULE)
    print("1+2.  HEADROOM ON THE W/S CHANNEL -- exact, with the generator's clip")
    print(RULE)
    for pW0, tag in ((0.59, "pW0=0.59  (what the generator uses: gc=0.41)"),
                     (0.50, "pW0=0.50  (BALANCED -- no clip until depth 1.0)")):
        print(f"\n  {tag}")
        print(f"  {'depth':>6} {'eff':>6} {'clip%':>6} {'no-pos':>8} {'oracle':>8} "
              f"{'headroom':>9} {'% of loss':>10}")
        for d in (0.02, 0.05, 0.10, 0.25, 0.50, 0.76, 1.00):
            h = headroom(pW0, d)
            print(f"  {d:>6.0%} {h['eff_depth']:>6.2f} {100*h['clipped_frac']:>5.1f}% "
                  f"{h['no_pos']:>8.4f} {h['with_pos']:>8.4f} {h['gap']:>9.4f} {h['pct']:>9.2f}%")
    print("\n  ln2 = 0.6931 nats of every loss above is the A-vs-T / G-vs-C coin flip.")
    print("  It is IRREDUCIBLE.  ~50% of the channel is spent on noise the model cannot price.")

    print("\n" + RULE)
    print("3.  COHERENCE LENGTH -- the floor the entropy table does NOT show")
    print(RULE)
    for c in (10.45, 10.67, 10.85):
        d_coh = coherence(c, 0.30)
        print(f"  centre {c}:  T ~ U({c-0.3:.2f}, {c+0.3:.2f})  ->  1-radian coherence length "
              f"= T^2/(2*pi*dT) = {d_coh:5.1f} nt  ({d_coh/c:.1f} periods)")
    d_coh = coherence(10.67, 0.30)
    print(f"\n  So a model at position i can coherently pool only ~+/-{d_coh/2:.0f} nt "
          f"(~{d_coh:.0f} nt total), NOT the 150-300 nt block.")
    print(f"  Blocks are 150-300 nt, but the BAND ITSELF decoheres in {d_coh:.0f} nt.\n")
    print(f"  {'depth':>6} {'sigma(phi)':>12} {'attenuation':>12}   verdict")
    for d in (0.02, 0.05, 0.10, 0.25, 0.50, 0.76, 1.00):
        pn = phase_noise(d, 0.5, d_coh)
        v = ("phase is NOISE" if pn['sigma_rad'] > 1.2 else
             "marginal" if pn['sigma_rad'] > 0.6 else "phase is resolved")
        print(f"  {d:>6.0%} {pn['sigma_deg']:>11.0f}d {pn['atten']:>12.3f}   {v}")
    print("\n  The entropy headroom is the CEILING.  This is the FLOOR -- and it bites first.")

    print("\n" + RULE)
    print("4.  THE GENERATOR, MEASURED (the positive control the pilot never ran)")
    print(RULE)
    print("  If the band is not IN the sequence, every downstream null is an artifact.\n")
    faithful = {}
    for c in (10.45, 10.67, 10.85):
        b = corpus_band(c, 0.50, 512)
        faithful[c] = b
        print(f"  centre {c}:  corpus band spans modes {b['mode_lo']:.2f}-{b['mode_hi']:.2f} "
              f"({b['mode_width_bins']:.2f} bins wide), centred {b['mode_centre']:.2f}")
        print(f"     measured: winning bin {b['win_bin']}, peak/median {b['peak_over_median']:.1f}x, "
              f"Q_win {b['q_win']:.3f}")
        print(f"     Q: {b['q']}\n")

    print(RULE)
    print("5.  THE QUANTIZATION NULL -- what 'sharp' has to beat")
    print(RULE)
    qw = faithful[10.67]['q_win']
    print(f"  The corpus band at 10.67 is {faithful[10.67]['mode_width_bins']:.2f} BINS WIDE and its")
    print(f"  own peak holds only Q_win = {qw:.3f} of the 43-53 band.")
    print(f"\n  FAITHFUL representation  ->  P mirrors the corpus:      Q_win ~ {qw:.2f}")
    print(f"  QUANTIZED representation ->  P collapses it to one bin:  Q_win >> {qw:.2f}")
    print(f"  SpliceBERT's REAL carrier holds 76% of its band in one bin  ->  Q_win = 0.76")
    print(f"\n  ==> THE EXPERIMENT'S DECISIVE STATISTIC IS Q_win, AND ITS NULL IS {qw:.2f}, NOT 1/11.")
    print(f"      A flat band would give 1/11 = 0.091 (what the surviving CONTROL run gave: 0.101).")

    print("\n" + RULE)
    print("6.  BIN PREDICTIONS -- and the table-length control")
    print(RULE)
    print("  NOTE: the effective table length is the CROP LENGTH.  Rows past the crop are never")
    print("  trained, so allocating 512 rows and cropping 496 gives an effective L of 496.")
    print("  ==> the table-length control must vary the CROP, not max_position_embeddings.\n")
    print(f"  {'crop L':>7} {'lambda':>8} {'L/lambda':>9} {'-> bin':>7} {'realised':>9} {'detune':>7}")
    for L in (500, 512, 521):
        for lam in (10.45, 10.6667, 10.85):
            m = L / lam
            b = int(round(m))
            print(f"  {L:>7} {lam:>8.4f} {m:>9.3f} {b:>7} {L/b:>9.4f} {abs(m-b):>7.2f}")
        print()
    print("  THE DISCRIMINATOR: hold lambda = 10.6667 and move the crop.")
    for L in (500, 512, 521):
        m = 512 / 48 if False else L / 10.6667
        b = int(round(m))
        print(f"     crop {L}: bin {b:>2}  (realised {L/b:.4f} nt)")
    print("     Bin NUMBER moves 47 -> 48 -> 49 while the PHYSICAL wavelength stays ~10.67.")
    print("     If the bin instead stays at 48, the carrier is ARCHITECTURAL, not physical,")
    print("     and the whole chromatin story dies.  This control CAN fail.")

    print("\n" + RULE)
    print("VERDICT ON THE PLANNED GRID")
    print(RULE)
    h10 = headroom(0.59, 0.10)
    pn10 = phase_noise(0.10, 0.5, d_coh)
    print(f"  The plan says: 'initially at 10% depth, then 2% and 5%'.")
    print(f"     at 10%:  headroom {h10['gap']:.4f} nats = {h10['pct']:.2f}% of the loss,")
    print(f"              and the phase is unresolvable (sigma = {pn10['sigma_deg']:.0f} deg).")
    print(f"     2% and 5% are strictly worse.")
    print(f"  ==> ALL THREE PLANNED DEPTHS ARE BELOW THE INSTRUMENT'S FLOOR.")
    print(f"      4 corpora x 5 seeds x 3 depths = 60 runs, all returning the same null.")


if __name__ == "__main__":
    main()
