#!/usr/bin/env python
"""EXP 2 — THE REAL DYAD-ANCHORED SOURCE ASSAY.  Voong chemical-cleavage map, mouse ESC, mm9.

    Q2 (the user's):  "Does the proposed biological source exist?"
    Decisive result:  "A powered 10.67 signal with reproducible phase relative to dyads."

THE INSTRUMENT IS FROZEN AND IS IMPORTED, NOT REIMPLEMENTED.
    FILT = hanning(151) * exp(-2*pi*i*k/10.6667),  k in [-75, +75]
    c_j  = sum_k h(k) * [x_j(k) - mu_j] * e^{-i omega k}        <- per-window complex coefficient
    C    = sum_j c_j                                            <- ACCUMULATE, THEN normalise
It comes from run_dyad_fpr.py, which measured its FPR at 4.6% [4.1, 5.2] over 4,800 null
replicates with no jitter dependence (chi2(3)=5.00, p~0.54).  Nothing here may change it.

THE PRIMARY TEST IS THE FROZEN-PHASE SPLIT-HALF, AND THAT IS NOT A STYLISTIC CHOICE
----------------------------------------------------------------------------------
|C| is POSITIVELY BIASED -- it is a magnitude, so noise alone makes it grow.  Estimating the
phase and testing it on the same data turns that bias into a precise-looking result.  That is
exactly how R16 produced four different artifacts in a row.  So:

    DISCOVERY  chr1 + chr2   ->  phi = arg(C_disc)          then FREEZE phi
    TEST       chr3 + chr4   ->  Re(C_test * e^{-i phi})    a SIGNED statistic, null mean ZERO

The signed projection cannot be inflated by noise: under the null it is symmetric about 0.
D25 verified this on simulation (null z = -0.3; 1% -> 4.8, 2% -> 12.0, 5% -> 33.9).

THE NULL IS THE SAME ONE THE FPR WAS MEASURED ON: random positions in the same real sequence,
same chromosomes, same N.  Not a shuffle, not a rotation -- the identical procedure.

WHAT A POSITIVE HERE DOES AND DOES NOT BUY  (D25, unchanged)
    POSITIVE, POWERED  -> chromatin is a PLAUSIBLE SOURCE.  Nothing more.
    NULL, POWERED      -> the chromatin-source hypothesis is DISFAVOURED.
    NULL, UNDERPOWERED -> NO CONCLUSION.  (Report the sensitivity limit, as with Assay 1.)
It does NOT show SpliceBERT represents it.  That is Exp 3, the carrier x dyad-score interaction.

DATA TRAPS -- see data/NUCLEOSOME_MAP.md
    * the map is mm9, NOT mm10 (verified: chr1's last dyad at 197,195,276 vs mm9's 197,195,432)
    * its chromosomes are RENUMBERED: 20 = chrX, 21 = chrY, 22 = chrM.  chrM is excluded.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import os
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/splicebert"))

from run_dyad_fpr import FILT, K, T0                      # noqa: E402  THE FROZEN INSTRUMENT

MAP = Path(os.environ.get("GFMCD_DYAD_MAP", "/nonexistent/GSM2183909_unique.map_95pc.txt.gz"))
FASTA = Path(os.environ.get("GFMCD_MM9_FASTA", "/nonexistent/mm9.fa"))
OUT = REPO / "results/phase-lattice/dyad_real.json"

CHROM = {**{str(i): f"chr{i}" for i in range(1, 20)}, "20": "chrX", "21": "chrY", "22": "chrM"}
EXCLUDE = {"chrM", "chrY"}                  # 16 kb circular, high copy; and Y is unmappable
DISCOVERY = ["chr1", "chr2"]                # phase estimated here, then FROZEN
TEST = ["chr3", "chr4"]                     # signed projection tested here
CH = {"WW": "AT", "SS": "GC", "RR": "AG", "YY": "CT"}
ACGT = [ord(c) for c in "ACGT"]


def load_dyads() -> dict[str, np.ndarray]:
    by: dict[str, list] = {}
    with gzip.open(MAP, "rt") as fh:
        for line in fh:
            a, b = line.split()
            c = CHROM.get(a)
            if c and c not in EXCLUDE:
                by.setdefault(c, []).append(int(b))
    return {c: np.sort(np.asarray(v, dtype=np.int64)) for c, v in by.items()}


def coeffs(seq: np.ndarray, pos: np.ndarray, ch: str, T: float = T0) -> np.ndarray:
    """Per-window complex coefficient.  IDENTICAL to run_dyad_fpr.coeffs, generalised to a
    channel and a wavelength.  At ch='WW', T=T0 it is bit-identical to the frozen estimator."""
    filt = FILT if (ch == "WW" and T == T0) else (
        np.hanning(2 * K + 1) * np.exp(-2j * np.pi * np.arange(-K, K + 1) / T))
    S = np.array([ord(c) for c in CH[ch]])
    pos = np.clip(pos, K + 10, len(seq) - K - 12)
    out = np.empty(len(pos), np.complex128)
    for s in range(0, len(pos), 50_000):
        p = pos[s:s + 50_000]
        w = seq[p[:, None] + np.arange(-K, K + 2)[None, :]]
        ok = np.isin(w, ACGT).all(1)                       # drop windows touching N / repeats
        x = (np.isin(w[:, :-1], S) & np.isin(w[:, 1:], S)).astype(np.float32)
        x -= x.mean(1, keepdims=True)                      # mu_m(a, .)  -- kills composition
        c = x @ filt
        c[~ok] = 0                                         # a dropped window contributes NOTHING
        out[s:s + 50_000] = c
    return out


def amplitude(C: complex, n: int, mean_ww: float) -> tuple[float, float]:
    """A pure tone x(k) = xbar + A*cos(wk+phi) gives c = (A/2)*e^{i phi}*sum(h).
    So A = 2|C| / (n * sum(h)), and the DEPTH is A / xbar."""
    A = 2 * abs(C) / (n * np.hanning(2 * K + 1).sum())
    return float(A), float(A / mean_ww) if mean_ww else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulls", type=int, default=2000)
    ap.add_argument("--pool-mult", type=int, default=3, help="null pool = pool_mult x n_test")
    args = ap.parse_args()

    print("loading the Voong unique dyad map (mm9)...", flush=True)
    dy = load_dyads()
    tot = sum(len(v) for v in dy.values())
    print(f"  {tot:,} dyads over {len(dy)} chromosomes (chrM, chrY excluded)")
    for c in DISCOVERY + TEST:
        print(f"    {c}: {len(dy[c]):,}")

    fa = Fasta(str(FASTA), as_raw=True)
    rec: dict = {"map": str(MAP), "build": "mm9", "T0": T0, "K": K,
                 "discovery": DISCOVERY, "test": TEST, "n_dyads_total": tot}

    seqs, cs, nulls, mean_ww = {}, {}, {}, {}
    rng = np.random.default_rng(0)
    for c in DISCOVERY + TEST:
        print(f"\n[{c}] loading sequence...", flush=True)
        seqs[c] = np.frombuffer(fa[c][:].upper().encode(), dtype=np.uint8)
        p = dy[c]
        print(f"       {len(p):,} dyads -> coefficients", flush=True)
        cs[c] = coeffs(seqs[c], p, "WW")
        # the mean WW frequency, for converting |C| into a modulation DEPTH
        w = seqs[c][p[:200_000, None] + np.arange(-K, K + 2)[None, :]]
        xx = (np.isin(w[:, :-1], [ord("A"), ord("T")])
              & np.isin(w[:, 1:], [ord("A"), ord("T")]))
        mean_ww[c] = float(xx.mean())
        # THE NULL: random positions in the SAME sequence.  Identical procedure to the FPR run.
        q = rng.integers(K + 10, len(seqs[c]) - K - 12, size=len(p) * args.pool_mult)
        nulls[c] = coeffs(seqs[c], q, "WW")
        print(f"       |C| real = {abs(cs[c].sum()):,.0f}   mean WW = {mean_ww[c]:.4f}", flush=True)

    # ---------------------------------------------------------------- PRIMARY
    C_disc = complex(cs[DISCOVERY[0]].sum() + cs[DISCOVERY[1]].sum())
    n_disc = sum(len(cs[c]) for c in DISCOVERY)
    phi = math.atan2(C_disc.imag, C_disc.real)             # <-- FROZEN HERE, used nowhere else
    C_test = complex(cs[TEST[0]].sum() + cs[TEST[1]].sum())
    n_test = sum(len(cs[c]) for c in TEST)
    stat = float((C_test * np.exp(-1j * phi)).real)        # the SIGNED projection

    # NULL for the signed projection: random positions in the SAME sequence, same N, SAME frozen phi.
    #
    # ! Do NOT materialise (nulls x n_test) indices.  n_test is ~1.3e6, so 2000 replicates would
    #   be a 20 GB int64 array.  The statistic is a SUM of n_test i.i.d. draws of Re(c*e^-i.phi),
    #   so the CLT is exact to many decimals at this n:  N(n*mean(v), sqrt(n*var(v))).
    #   A small batch of explicit replicates is then run to CONFIRM the CLT, not to replace it.
    r = np.random.default_rng(7)
    pool = np.concatenate([nulls[c] for c in TEST])
    v = (pool * np.exp(-1j * phi)).real                     # the projected per-window null coeffs
    mu, sd = float(n_test * v.mean()), float(math.sqrt(n_test * v.var()))
    z = (stat - mu) / sd

    nrep = min(args.nulls, 400)                            # memory-safe: one replicate at a time
    boot = np.empty(nrep)
    for i in range(nrep):
        boot[i] = v[r.integers(0, len(v), size=n_test)].sum()
    mu_b, sd_b = float(boot.mean()), float(boot.std())
    p_emp = float(((boot >= stat).sum() + 1) / (nrep + 1))
    print(f"\n  CLT null:  mean {mu:,.0f}  sd {sd:,.0f}")
    print(f"  bootstrap: mean {mu_b:,.0f}  sd {sd_b:,.0f}  ({nrep} reps)  "
          f"-> CLT/boot sd ratio {sd/sd_b:.3f}")

    mw = float(np.mean([mean_ww[c] for c in TEST]))
    A, depth = amplitude(C_test, n_test, mw)
    phi_test = math.atan2(C_test.imag, C_test.real)
    dphi = math.degrees((phi_test - phi + math.pi) % (2 * math.pi) - math.pi)

    print("\n" + "=" * 78)
    print("PRIMARY — frozen-phase split-half, WW at T = 10.6667")
    print("=" * 78)
    print(f"  DISCOVERY {'+'.join(DISCOVERY)}: {n_disc:,} dyads, |C| = {abs(C_disc):,.0f}")
    print(f"     phase phi = {math.degrees(phi):+.1f} deg   <-- FROZEN")
    print(f"  TEST      {'+'.join(TEST)}: {n_test:,} dyads, |C| = {abs(C_test):,.0f}")
    print(f"     signed projection Re(C * e^-i.phi) = {stat:,.0f}")
    print(f"     null: mean {mu:,.0f}, sd {sd:,.0f}   ({args.nulls} replicates)")
    print(f"\n     z = {z:+.2f}      empirical p = {p_emp:.4f}")
    print(f"\n  independent phase on the TEST half = {math.degrees(phi_test):+.1f} deg")
    print(f"     phase agreement across the split = {dphi:+.1f} deg")
    print(f"  amplitude A = {A:.5f}   modulation DEPTH = {depth*100:.2f}% of mean WW ({mw:.4f})")

    rec["primary"] = dict(n_disc=n_disc, n_test=n_test, C_disc=[C_disc.real, C_disc.imag],
                          C_test=[C_test.real, C_test.imag], phi_deg=math.degrees(phi),
                          phi_test_deg=math.degrees(phi_test), phase_agreement_deg=dphi,
                          stat=stat, null_mean=mu, null_sd=sd, z=z, p=p_emp,
                          amplitude=A, depth=depth, mean_ww=mw, nulls=args.nulls)

    # ---------------------------------------------------------------- SECONDARY: channels
    print("\n" + "=" * 78)
    print("SECONDARY — the other dinucleotide channels (same frozen phase protocol)")
    print("=" * 78)
    rec["channels"] = {}
    for ch in ("SS", "RR", "YY"):
        cd = sum(complex(coeffs(seqs[c], dy[c], ch).sum()) for c in DISCOVERY)
        ct = sum(complex(coeffs(seqs[c], dy[c], ch).sum()) for c in TEST)
        ph = math.atan2(cd.imag, cd.real)
        st = float((ct * np.exp(-1j * ph)).real)
        zz = (st - mu) / sd                                # same null scale (same N, same seq)
        rec["channels"][ch] = dict(phi_deg=math.degrees(ph), stat=st, z_approx=zz)
        print(f"  {ch}: phi = {math.degrees(ph):+7.1f} deg   signed = {st:>12,.0f}   "
              f"z ~ {zz:+.1f}")
    print("  (WW and SS are counterphase by construction; they are NEVER pooled.)")

    # ---------------------------------------------------------------- SECONDARY: wavelength scan
    print("\n" + "=" * 78)
    print("SECONDARY — wavelength profile 10.3-10.9 (is 10.67 where the peak actually is?)")
    print("=" * 78)
    rec["wavelength"] = {}
    for T in (10.2, 10.3, 10.4, 10.5, 10.6, T0, 10.7, 10.8, 10.9, 11.0):
        cd = sum(complex(coeffs(seqs[c], dy[c], "WW", T).sum()) for c in DISCOVERY)
        ct = sum(complex(coeffs(seqs[c], dy[c], "WW", T).sum()) for c in TEST)
        ph = math.atan2(cd.imag, cd.real)
        st = float((ct * np.exp(-1j * ph)).real)
        rec["wavelength"][f"{T:.4f}"] = dict(stat=st, phi_deg=math.degrees(ph))
        bar = "#" * max(0, int(40 * st / max(stat, 1)))
        print(f"  T = {T:6.4f}  signed = {st:>13,.0f}  {bar}")

    # ------------------------------------------------- THE CONTROL THAT CAN FAIL: anchor offset
    # Move the anchor off the dyad by d bp.  If this is a genuine 10.67-nt tone PHASE-LOCKED to the
    # dyad, then two things must BOTH happen, and neither is something an artifact would produce:
    #   (a) while the window still covers the core (|d| < ~75), |C| stays LARGE; once the window
    #       leaves the nucleosome (|d| >> 75), |C| COLLAPSES to the random-position null;
    #   (b) the PHASE rotates LINEARLY with d, at exactly 360/10.6667 = 33.75 deg per bp.
    # (b) is the strong one.  A composition artefact, a mappability artefact, or a spacing alias
    # has no reason whatever to rotate at precisely the carrier's own frequency.
    print("\n" + "=" * 78)
    print("CONTROL THAT CAN FAIL — move the anchor off the dyad")
    print("=" * 78)
    print(f"  predicted phase rotation: {360/T0:.2f} deg per bp;  |C| must die outside the core\n")
    print(f"  {'offset':>7} {'|C|':>12} {'vs null':>8} {'phase':>9} {'d(phase)/d(bp)':>15}")
    rec["offset_control"] = {}
    ref = None
    null_mag = float(abs(np.concatenate([nulls[c] for c in TEST]).sum()))
    for d in (-500, -150, -100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100, 150, 500):
        Cd = sum(complex(coeffs(seqs[c], dy[c] + d, "WW").sum()) for c in TEST)
        ph = math.degrees(math.atan2(Cd.imag, Cd.real))
        if d == 0:
            ref = ph
        slope = ""
        if ref is not None and d != 0:
            dd = ((ph - ref + 180) % 360 - 180) / d
            slope = f"{dd:+.2f} deg/bp"
        rec["offset_control"][str(d)] = dict(absC=abs(Cd), phase_deg=ph)
        print(f"  {d:>+7} {abs(Cd):>12,.0f} {abs(Cd)/max(null_mag,1):>7.1f}x {ph:>+8.1f} {slope:>15}")
    print(f"\n  (the random-position null at this N has |C| ~ {null_mag:,.0f})")

    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")
    print("\n" + "=" * 78)
    print("REMEMBER: a positive here is PROVENANCE PLAUSIBILITY ONLY.")
    print("Model USE requires Exp 3 (carrier x dyad-score interaction).  Do not conflate them.")
    print("=" * 78)


if __name__ == "__main__":
    main()
