#!/usr/bin/env python
"""THE CONTROL I FAILED TO RUN — is the splice-phasing estimator SPECIFIC, or is it R16 again?

WHAT WENT WRONG.  run_splice_phasing.py calibrated by INJECTION: put a 5% tone of known wavelength
into the windows, check it comes back.  It did, on all 8 arm x channel combinations, to <0.06 nt.
That measured SENSITIVITY.  It never measured SPECIFICITY -- whether the estimator returns a NULL
when there is no tone at all.

And the discovery scan then produced this:

    18 of 32 peaks landed EXACTLY on a scan boundary (T = 9.000 or T = 11.500).

A matched filter whose maximum is pinned to the edge of its own scan range is not finding a tone.
It is climbing a TREND.  This is R16's first failure mode, verbatim -- "scan-edge peaks at my scan
boundary".  Junction-centred windows carry a systematic composition gradient (exon->intron, the
polypyrimidine tract) that the shifted null windows do not, and subtracting the per-window MEAN
removes DC but NOT a trend.  A tapered filter's response to a residual trend RISES as T grows,
which is exactly what pins peaks to the upper edge.

THREE TESTS, AND THE ESTIMATOR MUST PASS ALL THREE BEFORE ANY SPLICE RESULT IS BELIEVED
    1. NULL SCAN.  Run the IDENTICAL discovery scan on landmark-destroyed windows.  If the null
       also edge-pins and reaches comparable z, the estimator is broken and every number in
       splice_phasing.json is an artefact.
    2. EDGE TEST.  Widen the scan to [7, 16].  If the peaks MOVE to the new edges, they were
       never tones -- they track the boundary, not the biology.
    3. DETREND.  Replace the per-window MEAN subtraction with a per-window LINEAR (then
       quadratic) detrend, and re-run.  A real tone survives detrending.  A trend artefact dies.

Only channels that (a) peak in the INTERIOR of a widened scan, (b) beat the null scan, and
(c) survive detrending, are allowed to be reported at all.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/splicebert"))

from run_splice_phasing import (ARMS, CH, W, DISCOVERY, TEST, FASTA,   # noqa: E402
                                junctions, windows, COMP)

OUT = REPO / "results/phase-lattice/splice_specificity.json"


def coeffs_dt(win, arm, ch, T, detrend=1):
    """Same estimator, but with a per-window POLYNOMIAL detrend of order `detrend`.
    detrend=0 reproduces the original (mean only).  detrend=1 removes a linear trend."""
    lo, hi = ARMS[arm]
    k = np.arange(lo, hi + 1)
    h = np.hanning(len(k))
    filt = h * np.exp(-2j * np.pi * k / T)
    s1, s2 = CH[ch]
    i = W + k
    x = (np.isin(win[:, i], [ord(c) for c in s1])
         & np.isin(win[:, i + 1], [ord(c) for c in s2])).astype(np.float64)
    ok = np.isin(win[:, i[0] - 1:i[-1] + 2], [65, 67, 71, 84]).all(1)
    # per-window polynomial detrend: project OUT the polynomial basis, not just the mean
    B = np.vstack([k.astype(float) ** d for d in range(detrend + 1)]).T      # (nk, deg+1)
    Q, _ = np.linalg.qr(B)                                                    # orthonormal basis
    x = x - (x @ Q) @ Q.T
    c = x @ filt
    c[~ok] = 0
    return c


def scan(win, arm, ch, Ts, detrend):
    return np.array([abs(complex(coeffs_dt(win, arm, ch, float(T), detrend).sum())) for T in Ts])


def main() -> None:
    by = junctions()
    fa = Fasta(str(FASTA), as_raw=True)
    rng = np.random.default_rng(0)
    real, null = {}, {}
    for c in DISCOVERY:
        s = np.frombuffer(fa[c][:].upper().encode(), dtype=np.uint8)
        for which, tag in ((0, "donor"), (1, "acceptor")):
            a = np.array([j[which] for j in by[c]])
            st = np.array([1 if j[2] == "+" else 0 for j in by[c]], np.uint8)
            keep = (a > W + 10) & (a < len(s) - W - 10)
            a, st = a[keep], st[keep]
            real.setdefault(tag, []).append(windows(s, a, st))
            off = rng.choice([-1, 1], len(a)) * rng.integers(500, 5000, len(a))
            b = np.clip(a + off, W + 10, len(s) - W - 11)
            null.setdefault(tag, []).append(windows(s, b, st))
    real = {k: np.concatenate(v) for k, v in real.items()}
    null = {k: np.concatenate(v) for k, v in null.items()}
    print(f"discovery: {len(real['donor']):,} donors, {len(real['acceptor']):,} acceptors\n")

    WIDE = np.arange(7.0, 16.01, 0.05)
    rec = {}
    print("=" * 104)
    print("TEST 1+2+3 — widened scan [7, 16], REAL vs landmark-destroyed NULL, with detrending")
    print("=" * 104)
    print("  A peak at 7.00 or 16.00 is a SCAN-EDGE ARTEFACT, not a tone.")
    print("  If the NULL peaks in the same place with comparable amplitude, the arm is dead.\n")
    hdr = (f"  {'arm':16s} {'ch':>3} | {'mean-only':>18} | {'LINEAR detrend':>18} | "
           f"{'NULL (lin)':>18} | verdict")
    print(hdr)
    print("  " + "-" * 100)
    keep = []
    for arm in ARMS:
        tag = "donor" if arm.startswith("donor") else "acceptor"
        for ch in ("GG", "CC", "RR", "YY", "AG", "CT", "WW", "SS"):
            v0 = scan(real[tag], arm, ch, WIDE, 0)
            v1 = scan(real[tag], arm, ch, WIDE, 1)
            vn = scan(null[tag], arm, ch, WIDE, 1)
            T0, T1, Tn = (float(WIDE[v.argmax()]) for v in (v0, v1, vn))
            edge = T1 <= 7.05 or T1 >= 15.95
            beats = v1.max() > 3 * vn.max()
            live = (not edge) and beats
            if live:
                keep.append((arm, ch, T1, float(v1.max() / max(vn.max(), 1))))
            verdict = ("EDGE ARTEFACT" if edge else
                       ("null matches it" if not beats else f"LIVE  {v1.max()/max(vn.max(),1):.1f}x null"))
            rec[f"{arm}|{ch}"] = dict(T_meanonly=T0, T_linear=T1, T_null=Tn,
                                      amp=float(v1.max()), amp_null=float(vn.max()),
                                      edge=bool(edge), live=bool(live))
            print(f"  {arm:16s} {ch:>3} | {T0:>8.2f} {v0.max():>9,.0f} | "
                  f"{T1:>8.2f} {v1.max():>9,.0f} | {Tn:>8.2f} {vn.max():>9,.0f} | {verdict}")

    print("\n" + "=" * 104)
    n_edge = sum(1 for v in rec.values() if v["edge"])
    print(f"  scan-edge artefacts: {n_edge} of {len(rec)}")
    print(f"  channels that SURVIVE (interior peak AND >3x the null): {len(keep)}")
    if keep:
        for arm, ch, T, r in sorted(keep, key=lambda x: -x[3]):
            near = ("~ the CARRIER (10.63)" if abs(T - 10.63) < 0.25 else
                    "~ the NUCLEOSOMAL pitch (10.30)" if abs(T - 10.30) < 0.25 else "")
            print(f"     {arm:16s} {ch:>3}   T = {T:6.2f}   {r:5.1f}x null   {near}")
    else:
        print("     NONE.  The splice-junction phasing does not survive its own specificity test.")
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
