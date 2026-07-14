#!/usr/bin/env python
"""THE SPLICE TEST, WITH ENOUGH FREQUENCY RESOLUTION TO MEAN ANYTHING.

WHY THE FIRST TWO ATTEMPTS COULD NOT HAVE WORKED -- and it is arithmetic, not bad luck.

A Hann window of N points resolves wavelength to  dT ~ 2 T^2 / N.
    N =  51 (the Kogan-like 20-70 window):  dT ~ 4.0 nt at T = 10   <- USELESS over [7, 16]
    N = 201 (an intron arm, 25-225):        dT ~ 1.0 nt
    N = 401:                                dT ~ 0.5 nt

With dT ~ 4 nt the whole scan range is about two resolution elements, so the argmax is free to
wander -- and it did: the "survivors" scattered from 7.70 to 15.78, 16 of 32 peaks pinned to a
scan EDGE, and GG and CC (the two PRIMARY channels, which Kogan says are COUNTERPHASE AT THE SAME
WAVELENGTH) peaked at 12.40 and 10.24.  That is a flat spectrum with a wandering maximum.

IT ALSO EXPLAINS KOGAN'S OWN NUMBER.  They used 42-nt windows -- the same resolution wall.
"10.58 +/- 0.35" is the spread of nine noisy periodogram maxima BECAUSE a 42-nt window cannot
locate a wavelength better than that.  It was never a measurement of one wavelength.

THE HARD LIMIT, AND IT IS BIOLOGICAL
    INTRONS are long (median 1-2 kb) -> their arms CAN be extended to ~200+ nt.  Resolution ~1 nt.
    EXONS are short (median ~120 nt) -> their arms CANNOT.  The window would run out of the exon.
    ==> THE EXON ARMS ARE INTRINSICALLY UNABLE TO RESOLVE THIS.  No amount of data fixes it.
        Any "10.24" or "10.32" from an exon arm carries +/- 4 nt and must never be quoted.

So this script tests the INTRON arms, at full length, on introns long enough to hold the window.
    * donor_intron    k in [25, 225]   introns >= 300 nt
    * acceptor_intron k in [-225, -25] introns >= 300 nt
    * per-window quadratic detrend (the polypyrimidine tract is a trend, and trends leak)
    * the landmark-destroyed NULL gets the IDENTICAL treatment
    * scan [7, 16]; a peak at an edge is still an artefact and is still reported as one
Nothing is anchored to Kogan's 10.58/10.38, or to SpliceBERT's 10.630.
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
from run_splice_phasing import CH, GTF, FASTA, COMP, DISCOVERY, TEST   # noqa: E402
import gzip                                                            # noqa: E402

OUT = REPO / "results/phase-lattice/splice_long.json"
WIN = 260
MINLEN = 300                      # intron must hold the window
ARMS = {"donor_intron": (25, 225), "acceptor_intron": (-225, -25)}
ACGT = [65, 67, 71, 84]


def long_introns():
    tx = {}
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t")
            if f[2] != "exon":
                continue
            tid = f[8].split('transcript_id "')[1].split('"')[0]
            tx.setdefault(tid, []).append((f[0], int(f[3]) - 1, int(f[4]), f[6]))
    seen, by = set(), {}
    for ex in tx.values():
        c, st = ex[0][0], ex[0][3]
        ex = sorted(ex, key=lambda e: e[1])
        for i in range(len(ex) - 1):
            a, b = ex[i][2], ex[i + 1][1]
            if b - a < MINLEN:                     # <-- the intron must be long enough
                continue
            key = (c, a, b, st)
            if key in seen:
                continue
            seen.add(key)
            d, q = (a, b) if st == "+" else (b - 1, a - 1)
            by.setdefault(c, []).append((d, q, st))
    return by


def wins(seq, anchors, strands):
    idx = anchors[:, None] + np.arange(-WIN, WIN + 1)[None, :]
    out = seq[idx]
    neg = strands == 0
    out[neg] = COMP[out[neg][:, ::-1]]
    return out


def profile(win, ch, lo, hi, deg=2):
    k = np.arange(lo, hi + 1)
    i = WIN + k
    s1, s2 = CH[ch]
    x = (np.isin(win[:, i], [ord(c) for c in s1])
         & np.isin(win[:, i + 1], [ord(c) for c in s2])).astype(np.float64)
    ok = np.isin(win[:, i[0] - 1:i[-1] + 2], ACGT).all(1)
    x = x[ok]
    B = np.vstack([k.astype(float) ** d for d in range(deg + 1)]).T
    Q, _ = np.linalg.qr(B)
    x = x - (x @ Q) @ Q.T
    return x.mean(0), x.std(0) / math.sqrt(len(x)), len(x)


def scan(p, lo, hi, Ts):
    k = np.arange(lo, hi + 1)
    h = np.hanning(len(k))
    return np.array([abs(complex((p * h * np.exp(-2j * np.pi * k / T)).sum())) for T in Ts])


def main() -> None:
    by = long_introns()
    fa = Fasta(str(FASTA), as_raw=True)
    rng = np.random.default_rng(0)
    real, null = {}, {}
    for c in DISCOVERY + TEST:
        if c not in by:
            continue
        s = np.frombuffer(fa[c][:].upper().encode(), dtype=np.uint8)
        grp = "D" if c in DISCOVERY else "T"
        for which, tag in ((0, "donor"), (1, "acceptor")):
            a = np.array([j[which] for j in by[c]])
            st = np.array([1 if j[2] == "+" else 0 for j in by[c]], np.uint8)
            keep = (a > WIN + 10) & (a < len(s) - WIN - 10)
            a, st = a[keep], st[keep]
            real.setdefault((grp, tag), []).append(wins(s, a, st))
            off = rng.choice([-1, 1], len(a)) * rng.integers(1000, 8000, len(a))
            null.setdefault((grp, tag), []).append(
                wins(s, np.clip(a + off, WIN + 10, len(s) - WIN - 11), st))
    real = {k: np.concatenate(v) for k, v in real.items()}
    null = {k: np.concatenate(v) for k, v in null.items()}
    nD, nT = len(real[("D", "donor")]), len(real[("T", "donor")])
    print(f"introns >= {MINLEN} nt:  {nD:,} discovery, {nT:,} held-out")
    for arm, (lo, hi) in ARMS.items():
        n = hi - lo + 1
        print(f"  {arm:16s} k in [{lo:>4}, {hi:>4}]  N = {n}  ->  resolution dT ~ "
              f"{2*10.5**2/n:.2f} nt at T = 10.5")
    print()

    Ts = np.arange(7.0, 16.005, 0.01)
    rec = {}
    print("=" * 100)
    print("DISCOVERY (chr1-8) — scan, FREEZE T and phi, then SIGNED test on held-out (chr9-22,X)")
    print("=" * 100)
    print(f"\n  {'arm':16s} {'ch':>3} | {'T':>7} {'amp':>7} {'null T':>7} {'null':>7} | "
          f"{'z (held-out)':>12} {'phase agr':>10} | verdict")
    print("  " + "-" * 96)
    for arm, (lo, hi) in ARMS.items():
        tag = "donor" if arm.startswith("donor") else "acceptor"
        for ch in ("GG", "CC", "RR", "YY", "AG", "CT", "WW", "SS"):
            pD, seD, _ = profile(real[("D", tag)], ch, lo, hi)
            pN, seN, _ = profile(null[("D", tag)], ch, lo, hi)
            v, vn = scan(pD, lo, hi, Ts), scan(pN, lo, hi, Ts)
            i = int(v.argmax())
            T_f = float(Ts[i])
            edge = T_f <= 7.02 or T_f >= 15.98
            k = np.arange(lo, hi + 1)
            h = np.hanning(len(k))
            # FREEZE phase on discovery
            cD = complex((pD * h * np.exp(-2j * np.pi * k / T_f)).sum())
            phi = math.atan2(cD.imag, cD.real)
            # SIGNED test on held-out chromosomes
            pT, seT, nT_ = profile(real[("T", tag)], ch, lo, hi)
            cT = complex((pT * h * np.exp(-2j * np.pi * k / T_f)).sum())
            stat = float((cT * np.exp(-1j * phi)).real)
            sig = float(np.sqrt((h ** 2 * seT ** 2).sum() / 2))
            z = stat / sig
            dphi = ((math.degrees(math.atan2(cT.imag, cT.real)) - math.degrees(phi) + 180) % 360) - 180
            beats = v.max() > 3 * vn.max()
            live = (not edge) and beats and abs(z) > 5
            near = ("  <== CARRIER 10.63" if abs(T_f - 10.63) < 0.20 else
                    "  <== NUCLEOSOMAL 10.30" if abs(T_f - 10.30) < 0.20 else "")
            verdict = ("EDGE ARTEFACT" if edge else "null matches" if not beats else
                       f"LIVE{near}" if live else f"weak (z={z:.1f})")
            rec[f"{arm}|{ch}"] = dict(T=T_f, amp=float(v.max()), T_null=float(Ts[vn.argmax()]),
                                      amp_null=float(vn.max()), z=z, phase_agreement_deg=dphi,
                                      edge=bool(edge), live=bool(live),
                                      profile=pD.tolist(), k_lo=lo, k_hi=hi)
            print(f"  {arm:16s} {ch:>3} | {T_f:>7.3f} {v.max():>7.4f} {Ts[vn.argmax()]:>7.2f} "
                  f"{vn.max():>7.4f} | {z:>+12.1f} {dphi:>+9.1f}d | {verdict}")

    live = [(k, v) for k, v in rec.items() if v["live"]]
    print("\n" + "=" * 100)
    print(f"  SURVIVORS (interior peak, >3x null, |z| > 5 on HELD-OUT chromosomes): {len(live)}")
    for k, v in sorted(live, key=lambda x: -abs(x[1]["z"])):
        print(f"     {k:24s}  T = {v['T']:6.3f}   z = {v['z']:+7.1f}   "
              f"phase agrees {v['phase_agreement_deg']:+.0f} deg")
    if not live:
        print("     NONE.")
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
