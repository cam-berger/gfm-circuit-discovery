#!/usr/bin/env python
"""THE SPLICE-JUNCTION PROFILE — the sufficient statistic, and the specificity test I owed.

TWO THINGS WERE WRONG WITH run_splice_phasing.py.

(1) IT WAS ~1000x SLOWER THAN NECESSARY, and that hid the problem.
    C(T) = SUM_j SUM_k x_j(k) h(k) e^{-2pi i k/T}
         = SUM_k [ SUM_j x_j(k) ] h(k) e^{-2pi i k/T}
         = profile(k) @ filt(T)
    The sum over junctions is a CONSTANT of T.  So the sufficient statistic is the AGGREGATE
    PROFILE -- the mean detrended dinucleotide density vs offset -- computed ONCE.  Scanning
    wavelengths is then a dot product with a 51-vector.  (This is what Kogan actually computed.)
    It also makes the artefact VISIBLE: you can just LOOK at the profile and see a trend.

(2) THE CALIBRATION MEASURED SENSITIVITY, NEVER SPECIFICITY.
    Injecting a 5% tone and recovering it proves the estimator can find a tone that IS there.
    It says NOTHING about whether it returns a null when one is NOT.  And the discovery scan then
    put 18 of 32 peaks EXACTLY on a scan boundary (T = 9.000 or 11.500) -- a filter whose maximum
    is pinned to the edge of its own scan range is climbing a TREND, not finding a tone.
    That is R16's first failure mode, verbatim.

    The cause: subtracting the per-window MEAN removes DC but NOT a trend, and junction-centred
    windows have a large one (exon->intron composition step, the polypyrimidine tract) that the
    shifted null windows do not.

WHAT THIS DOES
    * builds the profile over a WIDE offset range (|k| <= 200), once, for real and null anchors
    * per-window POLYNOMIAL detrend (order 0/1/2) -- a real tone survives; a trend dies
    * scans T in [7, 16] -- WIDER than before, so an edge-pinned peak is exposed as one
    * the NULL gets the IDENTICAL treatment.  If it peaks in the same place, the arm is dead.
    * reports the profile itself, so the oscillation (or the trend) can be SEEN, not inferred

NOTHING is anchored to Kogan's 10.58 or 10.38, or to SpliceBERT's 10.630.
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
from run_splice_phasing import CH, W, DISCOVERY, TEST, FASTA, junctions, windows  # noqa: E402

OUT = REPO / "results/phase-lattice/splice_profile.json"
ARMS = {"donor_exon": (-70, -20), "donor_intron": (20, 70),
        "acceptor_intron": (-70, -20), "acceptor_exon": (20, 70)}
ACGT = [65, 67, 71, 84]


def profile(win: np.ndarray, ch: str, lo: int, hi: int, deg: int):
    """Mean per-window-DETRENDED dinucleotide density vs offset, + its standard error.

    The detrend is applied WITHIN each window before averaging -- so a per-window composition
    gradient is removed at source and cannot survive into the aggregate.
    """
    k = np.arange(lo, hi + 1)
    i = W + k
    s1, s2 = CH[ch]
    x = (np.isin(win[:, i], [ord(c) for c in s1])
         & np.isin(win[:, i + 1], [ord(c) for c in s2])).astype(np.float64)
    ok = np.isin(win[:, i[0] - 1:i[-1] + 2], ACGT).all(1)
    x = x[ok]
    B = np.vstack([k.astype(float) ** d for d in range(deg + 1)]).T
    Q, _ = np.linalg.qr(B)
    x = x - (x @ Q) @ Q.T                                    # per-window polynomial detrend
    n = len(x)
    return x.mean(0), x.std(0) / math.sqrt(n), n


def scan(prof, lo, hi, Ts):
    k = np.arange(lo, hi + 1)
    h = np.hanning(len(k))
    return np.array([abs(complex((prof * h * np.exp(-2j * np.pi * k / T)).sum())) for T in Ts])


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
            real.setdefault("_", None)
            null.setdefault(tag, []).append(
                windows(s, np.clip(a + off, W + 10, len(s) - W - 11), st))
    real = {k: np.concatenate(v) for k, v in real.items() if k != "_"}
    null = {k: np.concatenate(v) for k, v in null.items()}
    print(f"discovery: {len(real['donor']):,} donors, {len(real['acceptor']):,} acceptors\n")

    Ts = np.arange(7.0, 16.01, 0.02)
    rec = {}
    print("=" * 106)
    print("SPECIFICITY — widened scan [7,16], per-window DETREND, and the landmark-destroyed NULL")
    print("=" * 106)
    print("  A peak at 7.00 or 16.00 is a SCAN-EDGE ARTEFACT.  If the NULL peaks with it, the arm is dead.\n")
    print(f"  {'arm':16s} {'ch':>3} | {'deg0 (mean)':>13} | {'deg1 (linear)':>13} | "
          f"{'deg2 (quad)':>13} | {'NULL deg1':>13} | verdict")
    print("  " + "-" * 102)
    live = []
    for arm, (lo, hi) in ARMS.items():
        tag = "donor" if arm.startswith("donor") else "acceptor"
        for ch in ("GG", "CC", "RR", "YY", "AG", "CT", "WW", "SS"):
            out = {}
            for deg in (0, 1, 2):
                p, se, n = profile(real[tag], ch, lo, hi, deg)
                v = scan(p, lo, hi, Ts)
                out[deg] = (float(Ts[v.argmax()]), float(v.max()), p, se, n)
            pn, sen, _ = profile(null[tag], ch, lo, hi, 1)
            vn = scan(pn, lo, hi, Ts)
            Tn, An = float(Ts[vn.argmax()]), float(vn.max())

            T1, A1 = out[1][0], out[1][1]
            edge = T1 <= 7.05 or T1 >= 15.95
            # amplitude in units of the profile's own noise: sum(h)*se is the 1-sigma scale
            k = np.arange(lo, hi + 1)
            sigma = float(np.sqrt((np.hanning(len(k)) ** 2 * out[1][3] ** 2).sum() / 2))
            snr = A1 / max(sigma, 1e-12)
            ok = (not edge) and A1 > 3 * An and snr > 5
            if ok:
                live.append((arm, ch, T1, snr, A1 / max(An, 1e-12)))
            verdict = ("EDGE ARTEFACT" if edge else
                       "null matches it" if A1 <= 3 * An else
                       f"weak (snr {snr:.1f})" if snr <= 5 else f"LIVE  snr {snr:.0f}")
            rec[f"{arm}|{ch}"] = dict(
                T_deg0=out[0][0], T_deg1=T1, T_deg2=out[2][0], amp_deg1=A1,
                T_null=Tn, amp_null=An, snr=snr, edge=bool(edge), live=bool(ok),
                profile_deg1=out[1][2].tolist(), n=int(out[1][4]))
            print(f"  {arm:16s} {ch:>3} | {out[0][0]:>6.2f} {out[0][1]:>6.3f} | "
                  f"{T1:>6.2f} {A1:>6.3f} | {out[2][0]:>6.2f} {out[2][1]:>6.3f} | "
                  f"{Tn:>6.2f} {An:>6.3f} | {verdict}")

    print("\n" + "=" * 106)
    print(f"  scan-edge artefacts: {sum(1 for v in rec.values() if v['edge'])} of {len(rec)}")
    print(f"  SURVIVORS (interior peak, >3x null, snr > 5): {len(live)}")
    print()
    if live:
        for arm, ch, T, snr, r in sorted(live, key=lambda x: -x[3]):
            near = ("<== the CARRIER (10.63)" if abs(T - 10.63) < 0.20 else
                    "<== the NUCLEOSOMAL pitch (10.30)" if abs(T - 10.30) < 0.20 else "")
            print(f"     {arm:16s} {ch:>3}   T = {T:6.2f}   snr {snr:5.1f}   {r:5.1f}x null  {near}")
    else:
        print("     NONE.  The splice-junction phasing does not survive its own specificity test.")
    print("\n  DECISION TABLE")
    print("    sharp near 10.6-10.65 ....... candidate source for SpliceBERT's carrier")
    print("    near 10.1-10.3 .............. the human-only carrier / chromatin pitch instead")
    print("    broad, no stable phase ...... cannot explain a SHARP carrier")
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
