#!/usr/bin/env python
"""ASSAY 2 — THE DYAD-ANCHORED MATCHED FILTER, AND ITS POWER CURVE.

Assay 1 is CLOSED.  Its only reportable result is methodological:

    at 120 Mb, the unanchored POWER assay can exclude only large, globally incoherent effects --
    NOT the expected few-percent nucleosomal signal.

p >= 0.39 there is NOT evidence against chromatin.  The calibrated sensitivity limit is.

THE ANCHORED ESTIMATOR (the user's specification, verbatim)
----------------------------------------------------------
    C_m = SUM over dyads a  SUM over k in [-K, K]   h(k) * [x_m(a+k) - mu_m(a,k)] * e^{-2*pi*i*k/10.6667}

  * NORMALISE ONLY AFTER EVERY DYAD HAS BEEN ACCUMULATED.  The signal then adds LINEARLY and the
    noise as sqrt(N), so SNR ~ a*sqrt(N) -- a LINEAR detector.  Assay 1 pooled power and was
    QUADRATIC (SNR ~ a^2*sqrt(N)), which is why 2% needed 625x more data there and needs only
    25x here.  That single difference is the whole reason to anchor.
  * K = 75 (the nucleosome core), a SMOOTH taper h (Hann) rather than a rectangular boundary,
    so a slow landmark-locked trend cannot leak into the 10.67 bin.
  * WW primary; SS, RR, YY as separate secondary channels.  Counterphase channels never pooled.
  * No exact Fourier bin is needed: over 151 nt, evaluating the complex exponential directly at
    10.6667 IS the matched filter.  Zero-padding would prettify the grid, not add resolution.

WHAT THIS RUN DOES -- AND DOES NOT
----------------------------------
It does NOT touch a real nucleosome map (we do not have one).  It BUILDS the instrument and
measures its POWER CURVE on simulated dyads placed in real genomic sequence, so that we know --
BEFORE spending effort acquiring the Voong et al. 2016 mouse-ESC chemical map -- whether ~10^6
dyads can actually reach the acceptance criteria.  Acquiring data for an underpowered assay is
the same mistake as R16, one step earlier.

ACCEPTANCE CRITERIA, FIXED BEFORE LOOKING AT ANY REAL RESULT
    * false-positive rate near 5%
    * detection probability >= 80% at 2% modulation
    * correct phase recovery
    * usable sensitivity across wavelengths 10.3-10.9
    * stability across held-out chromosomes

DYAD PRECISION IS THE HIDDEN KILLER.  Gaussian positional jitter sigma attenuates the amplitude by
    exp[-0.5*(2*pi*sigma/10.67)^2]
    -> 84% at 1 bp, 50% at 2 bp, 21% at 3 bp.
A nominal million dyads with sloppy positioning is an underpowered million.  This run measures
the attenuation empirically and checks it against that formula.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from pyfaidx import Fasta

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.circuit import DEFAULT_FASTA  # noqa: E402

T0 = 10.6667                       # the preregistered wavelength
K = 75                             # nucleosome core: +-75 nt
OUT = REPO / "results/phase-lattice/dyad_assay.json"
CH = {"WW": ("AT", "AT"), "SS": ("GC", "GC"), "RR": ("AG", "AG"), "YY": ("CT", "CT")}

k = np.arange(-K, K + 1)
H = np.hanning(2 * K + 1)          # smooth taper -- no rectangular boundary
E = np.exp(-2j * np.pi * k / T0)   # the matched filter
FILT = H * E                       # h(k) * e^{-i omega k}


def dinuc(win: np.ndarray, ch: str) -> np.ndarray:
    """win: (N, 2K+2) uint8.  Returns (N, 2K+1) float 0/1 dinucleotide indicator."""
    s1, s2 = CH[ch]
    m1 = np.isin(win[:, :-1], [ord(c) for c in s1])
    m2 = np.isin(win[:, 1:], [ord(c) for c in s2])
    return (m1 & m2).astype(np.float32)


def C_of(win: np.ndarray, ch: str) -> complex:
    """The anchored accumulator.  mu is the PER-DYAD window mean (removes composition);
    the smooth taper then keeps any residual slow trend out of the 10.67 bin.
    NORMALISED ONLY AT THE END."""
    x = dinuc(win, ch)
    x = x - x.mean(1, keepdims=True)                 # mu_m(a, .)
    return complex((x * FILT[None, :]).sum())        # sum over k AND over dyads


def inject(win: np.ndarray, depth: float, T: float, phase: float, rng) -> np.ndarray:
    """A WW tone at a FIXED phase relative to the dyad -- the biological model."""
    if depth <= 0:
        return win
    kk = np.arange(win.shape[1]) - (K + 1)
    p = depth * (1 + np.cos(2 * np.pi * kk / T + phase)) / 2
    hit = rng.random(win.shape) < p[None, :]
    out = win.copy()
    out[hit] = rng.choice([ord("A"), ord("T")], hit.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyads", type=int, default=100_000)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--mb", type=float, default=40.0)
    args = ap.parse_args()

    fa = Fasta(str(DEFAULT_FASTA), as_raw=True)
    rng = np.random.default_rng(0)
    chroms = ["chr1", "chr2", "chr3", "chr4"]
    per = int(args.mb * 1e6 / len(chroms))
    blocks = []
    for c in chroms:
        n = len(fa[c])
        s = int(rng.integers(5_000_000, n - per - 5_000_000))
        b = np.frombuffer(fa[c][s:s + per].upper().encode(), dtype=np.uint8)
        blocks.append((c, b))
    print(f"{args.mb:.0f} Mb of real sequence from {chroms}")
    print(f"{args.dyads:,} simulated dyads/replicate, {args.reps} replicates, K = {K}\n")

    def draw(n, seed, jitter=0.0):
        """Extract windows around n random dyads; `jitter` is the ASSAY's positional error."""
        r = np.random.default_rng(seed)
        outs, chrs = [], []
        for ci, (c, b) in enumerate(blocks):
            m = n // len(blocks)
            a = r.integers(K + 10, len(b) - K - 10, size=m)
            if jitter:
                a = np.clip(a + np.round(r.normal(0, jitter, m)).astype(int),
                            K + 10, len(b) - K - 10)
            idx = a[:, None] + np.arange(-K, K + 2)[None, :]
            w = b[idx]
            ok = np.isin(w, [ord(x) for x in "ACGT"]).all(1)
            outs.append(w[ok])
            chrs.append(np.full(ok.sum(), ci))
        return np.concatenate(outs), np.concatenate(chrs)

    rec: dict = {"T0": T0, "K": K, "dyads": args.dyads, "reps": args.reps}

    # -------------------------------------------------- the power curve
    print("=" * 96)
    print("POWER CURVE.  Detection = |C| above the 95th percentile of the depth-0 null.")
    print("Each cell: DETECTION PROBABILITY over independent replicates (fresh dyads, fresh noise).")
    print("=" * 96)
    JIT = [0.0, 1.0, 2.0]
    DEPTHS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]

    # the null first -- and the injection is applied to a *clean* draw each time
    null = []
    for r in range(args.reps * 4):
        w, _ = draw(args.dyads, 1000 + r)
        null.append(abs(C_of(w, "WW")))
    thr = float(np.percentile(null, 95))
    print(f"\nnull |C| : median {np.median(null):.1f}, 95th pct {thr:.1f}  "
          f"(n = {len(null)} replicates)\n")

    print(f"{'depth':>7s} | " + " | ".join(f"jitter {j:.0f} bp" for j in JIT))
    print("-" * 60)
    rec["power"] = {}
    for depth in DEPTHS:
        row = []
        for j in JIT:
            hits, amps, phs = 0, [], []
            for r in range(args.reps):
                rr = np.random.default_rng(7000 + r * 13 + int(depth * 1e4) + int(j * 100))
                T = float(rr.uniform(10.3, 10.9)) if depth > 0 else T0
                ph = float(rr.uniform(0, 2 * np.pi))
                w, _ = draw(args.dyads, 2000 + r)              # clean windows
                if depth > 0:
                    w = inject(w, depth, T, ph, rr)
                # the ASSAY's dyad error: shift the window, i.e. re-extract with jitter
                if j > 0:
                    sh = np.round(rr.normal(0, j, len(w))).astype(int)
                    sh = np.clip(sh, -8, 8)
                    idx = (np.arange(w.shape[1])[None, :] + sh[:, None])
                    idx = np.clip(idx, 0, w.shape[1] - 1)
                    w = np.take_along_axis(w, idx, axis=1)
                C = C_of(w, "WW")
                amps.append(abs(C)); phs.append(np.angle(C))
                hits += abs(C) > thr
            row.append(hits / args.reps)
            rec["power"][f"{depth}_{j}"] = dict(detect=hits / args.reps,
                                                mean_abs=float(np.mean(amps)))
        lab = "0 (FPR)" if depth == 0 else f"{depth*100:.1f}%"
        print(f"{lab:>7s} | " + " | ".join(f"{v*100:9.0f}%  " for v in row))
    print("-" * 60)

    # -------------------------------------------------- jitter attenuation vs the formula
    print("\nJITTER ATTENUATION -- measured vs  exp[-0.5*(2*pi*sigma/10.67)^2]\n")
    print(f"{'sigma':>6s} {'predicted':>11s} {'measured':>10s}")
    a0 = rec["power"]["0.1_0.0"]["mean_abs"]
    rec["jitter"] = {}
    for j in JIT:
        pred = float(np.exp(-0.5 * (2 * np.pi * j / T0) ** 2))
        meas = rec["power"][f"0.1_{j}"]["mean_abs"] / a0
        rec["jitter"][f"{j}"] = dict(predicted=pred, measured=float(meas))
        print(f"{j:6.0f} {pred:11.2f} {meas:10.2f}")

    # -------------------------------------------------- how many dyads do we NEED?
    print("\n" + "=" * 96)
    print("ACCEPTANCE CRITERIA (fixed before any real data):  FPR ~5%,  >=80% power at 2%")
    print("=" * 96)
    fpr = rec["power"]["0.0_0.0"]["detect"]
    p2 = rec["power"]["0.02_0.0"]["detect"]
    print(f"\n  at {args.dyads:,} perfect dyads:  FPR = {fpr*100:.0f}%,  power at 2% = {p2*100:.0f}%")
    # SNR ~ a*sqrt(N): the amplitude scales linearly with N, the null |C| as sqrt(N)
    snr2 = rec["power"]["0.02_0.0"]["mean_abs"] / np.median(null)
    need = args.dyads * (2.5 / max(snr2, 1e-9)) ** 2       # ~2.5 sigma for 80% power at alpha .05
    print(f"  measured SNR at 2% = {snr2:.2f}  ->  need ~{need:,.0f} dyads for 80% power at 2%")
    for j in (1.0, 2.0):
        att = rec["jitter"][f"{j}"]["measured"]
        print(f"     with {j:.0f} bp dyad jitter (attenuation {att:.2f}): "
              f"~{need/max(att,1e-9)**2:,.0f} dyads")
    rec["need_dyads_2pct"] = float(need)

    # ---------------------------------------------------- the SPLIT-HALF PHASE PROTOCOL
    # |C| has a POSITIVE NOISE BIAS -- estimating the phase and testing it on the same data turns
    # that bias into another precise-looking result.  So: estimate the phase on HALF the
    # chromosomes, FREEZE it, and test the SIGNED complex projection Re(C * e^{-i phi}) on the
    # other half.  Under the null this is symmetric about zero, with NO magnitude bias.
    print("\n" + "=" * 96)
    print("SPLIT-HALF PHASE PROTOCOL  (|C| is positively biased; a signed projection is not)")
    print("  estimate phi on chr1+chr2 -> FREEZE -> test Re(C * e^{-i phi}) on chr3+chr4")
    print("=" * 96)
    print(f"\n{'depth':>7s} {'phi (train)':>12s} {'signed proj (TEST)':>20s} {'null sd':>9s} "
          f"{'z':>7s}")
    print("-" * 62)
    rec["splithalf"] = {}
    nulls = []
    for r in range(48):
        w, c = draw(args.dyads, 5000 + r)
        te = c >= 2
        nulls.append(float(np.real(C_of(w[te], "WW"))))
    sd = float(np.std(nulls))
    for depth in (0.0, 0.01, 0.02, 0.05):
        rr = np.random.default_rng(999 + int(depth * 1e4))
        T = float(rr.uniform(10.3, 10.9)) if depth > 0 else T0
        ph = float(rr.uniform(0, 2 * np.pi))
        w, c = draw(args.dyads, 4242)
        if depth > 0:
            w = inject(w, depth, T, ph, rr)
        tr, te = c < 2, c >= 2                       # chr1+2 train, chr3+4 test
        phi = np.angle(C_of(w[tr], "WW"))            # estimated on TRAIN, then FROZEN
        proj = float(np.real(C_of(w[te], "WW") * np.exp(-1j * phi)))
        z = proj / sd
        rec["splithalf"][f"{depth}"] = dict(phi=float(phi), proj=proj, z=z)
        lab = "0 (null)" if depth == 0 else f"{depth*100:.1f}%"
        print(f"{lab:>7s} {np.degrees(phi):+11.0f}° {proj:20.0f} {sd:9.0f} {z:7.1f}")
    print("-" * 62)
    print("  z on the HELD-OUT half, with the phase frozen from the other half.  No magnitude bias.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
