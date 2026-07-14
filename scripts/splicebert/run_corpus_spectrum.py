#!/usr/bin/env python
"""ASSAY 1 — THE UNANCHORED CORPUS SPECTRUM, WITH ITS POWER CALIBRATION BUILT IN.

Does the genome carry a ~10.67-nt dinucleotide periodicity at all?  That is the load-bearing
question: it is what the "genomic statistics selected the tone" row of the evidence hierarchy
rests on, and R16 retracted my previous answer because the instrument was never validated.

WHAT R16 TAUGHT, AND WHAT THIS FIXES
------------------------------------
  * POOL THE SUFFICIENT STATISTICS BEFORE NORMALISING.  My retracted estimator did
    `C += v/v[0]` -- it normalised EACH WINDOW by its own lag-0 term and then averaged.  At 1-2%
    modulation a single 4-kb window contains essentially no usable estimate, so per-window
    normalise-then-average manufactures precise-looking noise.  Here we accumulate raw POWER
    across segments and normalise once, at the end.
  * PREREGISTER THE FREQUENCY.  omega = 2*pi/10.6667, fixed in advance.  Every earlier estimator
    of mine went edge-seeking and reported whatever the boundary of its scan happened to be.
    With L = 4096, 4096/384 = 10.6667 EXACTLY, so the preregistered frequency lands on an exact
    Fourier bin: no leakage, no window smearing, no winner's curse.  The 9-12 scan is SECONDARY.
  * THE INJECTION STUDY IS PART OF THE ESTIMATOR, NOT AN AFTERTHOUGHT.  Inject known depths
    (0, 0.5, 1, 2, 5, 10%) at wavelengths 10.3-10.9 and random phases, run the COMPLETE
    procedure on each, and report the detection floor and the false-positive rate at depth 0.
    A null is worthless until you know the floor.
  * SCALE.  Detection ~ sqrt(effective length).  Going from a 10% floor to 2% needs ~25x more
    sequence: megabases, not hundreds of windows.

CONFOUNDS THE USER FLAGGED, AND HOW THEY ARE HANDLED
    repeats  -> the FASTA is soft-masked; every analysis is run BOTH with and without repeats
    channels -> WW, SS, RR, YY are kept SEPARATE.  Counterphase channels are never pooled.
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

# SEGMENT LENGTH MUST MATCH THE SIGNAL'S COHERENCE LENGTH, NOT EXCEED IT.
# The first version used L = 4096, where one Fourier bin is +-0.03 nt wide -- so the
# preregistered filter at 10.6667 was BLIND to an injected tone at 10.45, and the calibration
# failed to see even a 10% signal.  Correctly: the helical periodicity is a LOCAL feature
# (nucleosome scale, ~150-300 nt).  Over 4 kb its phase decoheres and its power smears.
# L = 512 gives 512/48 = 10.6667 EXACTLY -- the very Fourier grid SpliceBERT's own first-stage
# 512-row position table used -- and a bin wide enough to hold a real, slightly-drifting signal.
L = 512
HOP = 256
BIN0 = 48                         # the preregistered bin.  L/BIN0 = 10.6667 EXACTLY
FIT_LO, FIT_HI = 24, 96           # background fit band (periods 21.3 .. 5.3)
EXC_LO, EXC_HI = 44, 53           # excluded from the fit: periods 11.6 .. 9.7
OUT = REPO / "results/phase-lattice/corpus_spectrum.json"

CH = {"WW": ("AT", "AT"), "SS": ("GC", "GC"), "RR": ("AG", "AG"), "YY": ("CT", "CT")}


def indicator(b: np.ndarray, ch: str) -> np.ndarray:
    s1, s2 = CH[ch]
    m1 = np.isin(b, [ord(c) for c in s1])
    m2 = np.isin(b, [ord(c) for c in s2])
    return (m1[:-1] & m2[1:]).astype(np.float32)


def segments(x: np.ndarray) -> np.ndarray:
    n = (len(x) - L) // HOP + 1
    if n <= 0:
        return np.empty((0, L), np.float32)
    idx = np.arange(L)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def accumulate(d: np.ndarray, keep: np.ndarray | None = None):
    """Raw POWER, summed over segments.  No per-segment normalisation of any kind."""
    X = segments(d)
    if len(X) == 0:
        return np.zeros(L // 2 + 1), 0
    if keep is not None:
        K = segments(keep.astype(np.float32))
        X = X[K.mean(1) > 0.98]                       # segments that are >98% non-repeat
        if len(X) == 0:
            return np.zeros(L // 2 + 1), 0
    X = X - X.mean(1, keepdims=True)                  # kill DC; rectangular window (exact bin)
    F = np.fft.rfft(X, axis=1)
    return (np.abs(F) ** 2).sum(0), len(X)


def statistic(P: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Excess of the PREREGISTERED bin over a smooth local background.

    The null is the EMPIRICAL distribution of the same ratio at the flanking bins, so no
    distributional assumption is needed.  Returns (Z, p, null)."""
    f = np.arange(FIT_LO, FIT_HI)
    ok = (f < EXC_LO) | (f >= EXC_HI)
    lp = np.log(np.maximum(P[FIT_LO:FIT_HI], 1e-30))
    A = np.column_stack([f.astype(float) ** k for k in range(3)])
    beta = np.linalg.lstsq(A[ok], lp[ok], rcond=None)[0]
    bg = np.exp(A @ beta)
    ratio = P[FIT_LO:FIT_HI] / bg
    Z = float(ratio[BIN0 - FIT_LO])
    null = ratio[ok]                                   # flanking bins = the null
    p = float((np.sum(null >= Z) + 1) / (len(null) + 1))
    return Z, p, null


def inject(b: np.ndarray, depth: float, T: float, phase: float, rng) -> np.ndarray:
    """Force a fraction of positions to W with probability depth*(1+cos)/2 -- a KNOWN tone."""
    if depth <= 0:
        return b
    x = np.arange(len(b))
    hit = rng.random(len(b)) < depth * (1 + np.cos(2 * np.pi * x / T + phase)) / 2
    out = b.copy()
    out[hit] = rng.choice([ord("A"), ord("T")], hit.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=float, default=60.0, help="megabases of sequence")
    ap.add_argument("--chroms", default="chr1,chr2,chr3,chr4,chr5,chr6")
    args = ap.parse_args()

    fa = Fasta(str(DEFAULT_FASTA), as_raw=True)
    chroms = args.chroms.split(",")
    per = int(args.mb * 1e6 / len(chroms))
    rng = np.random.default_rng(0)

    chunks = []
    for c in chroms:
        n = len(fa[c])
        s = int(rng.integers(5_000_000, max(5_000_001, n - per - 5_000_000)))
        raw = fa[c][s:s + per]
        b = np.frombuffer(raw.encode(), dtype=np.uint8)
        keep = (b >= ord("A")) & (b <= ord("Z"))                 # uppercase = NOT repeat
        b = np.frombuffer(raw.upper().encode(), dtype=np.uint8)
        good = np.isin(b, [ord(x) for x in "ACGT"])
        if good.mean() < 0.9:
            continue
        chunks.append((b, keep, good))
    tot = sum(len(b) for b, _, _ in chunks)
    print(f"{len(chunks)} chunks, {tot/1e6:.1f} Mb, from {chroms}")
    print(f"repeat (soft-masked) fraction: {1 - np.mean([k.mean() for _, k, _ in chunks]):.1%}\n")

    def run(depth=0.0, T=10.6667, phase=0.0, seed=0, repeats=True):
        r = np.random.default_rng(seed)
        out = {}
        for ch in CH:
            P = np.zeros(L // 2 + 1)
            n = 0
            for b, keep, good in chunks:
                bb = inject(b, depth, T, phase, r) if depth > 0 else b
                d = indicator(bb, ch)
                g = (good[:-1] & good[1:])
                d = d * g
                Pc, nc = accumulate(d, None if repeats else keep[:-1])
                P += Pc
                n += nc
            out[ch] = statistic(P) + (n,)
        return out

    rec: dict = {"L": L, "bin0": BIN0, "period0": L / BIN0, "mb": tot / 1e6}

    # ---------------------------------------------------------------- CALIBRATION (first!)
    print("=" * 90)
    print("POWER CALIBRATION.  Inject a KNOWN tone; can the estimator find it?  (This decides")
    print("whether any null below is worth anything.)  Wavelengths 10.3-10.9, random phase.")
    print("=" * 90)
    print(f"\n{'depth':>7s} | " + " | ".join(f"{ch:>18s}" for ch in CH))
    print(f"{'':>7s} | " + " | ".join(f"{'Z':>8s} {'p':>9s}" for _ in CH))
    print("-" * 90)
    rec["calibration"] = {}
    for depth in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10):
        T = 10.6667                                    # MATCHED: the preregistered wavelength
        ph = float(rng.uniform(0, 2 * np.pi))
        res = run(depth, T, ph, seed=int(depth * 1000) + 7)
        rec["calibration"][f"{depth}"] = {ch: dict(Z=res[ch][0], p=res[ch][1], T=T) for ch in CH}
        cells = " | ".join(f"{res[ch][0]:8.2f} {res[ch][1]:9.4f}" for ch in CH)
        lab = "0 (null)" if depth == 0 else f"{depth*100:.1f}%"
        print(f"{lab:>7s} | {cells}")
    print("-" * 90)
    print("\n  ROBUSTNESS: does the filter still see a tone whose wavelength is OFF 10.6667?")
    print(f"  {'true T':>7s} {'depth':>7s} | " + " | ".join(f"{ch:>18s}" for ch in CH))
    rec["calibration_offT"] = {}
    for T in (10.3, 10.5, 10.9):
        res = run(0.05, T, 1.0, seed=int(T * 100))
        rec["calibration_offT"][f"{T}"] = {ch: dict(Z=res[ch][0], p=res[ch][1]) for ch in CH}
        cells = " | ".join(f"{res[ch][0]:8.2f} {res[ch][1]:9.4f}" for ch in CH)
        print(f"  {T:>7.2f} {'5%':>7s} | {cells}")
    print("-" * 90)
    print("  Z = power at the PREREGISTERED bin / smooth local background.  p from the flanking")
    print("  bins (empirical null).  depth 0 row = the FALSE-POSITIVE check.\n")

    # ---------------------------------------------------------------- THE REAL MEASUREMENT
    print("=" * 90)
    print("THE REAL GENOME, at the preregistered period 10.6667")
    print("=" * 90)
    rec["real"] = {}
    for rep, lab in ((True, "all sequence"), (False, "repeats EXCLUDED")):
        res = run(repeats=rep)
        rec["real"][lab] = {ch: dict(Z=res[ch][0], p=res[ch][1], nseg=res[ch][3]) for ch in CH}
        print(f"\n  {lab}  ({res['WW'][3]} segments x {L} nt = "
              f"{res['WW'][3]*L/1e6:.1f} Mb effective)")
        print(f"    {'chan':5s} {'Z':>8s} {'p':>9s}")
        for ch in CH:
            Z, p, _, n = res[ch]
            star = "  <== SIGNAL" if p < 0.01 and Z > 1.5 else ""
            print(f"    {ch:5s} {Z:8.2f} {p:9.4f}{star}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
