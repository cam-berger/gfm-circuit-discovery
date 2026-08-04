#!/usr/bin/env python
"""THE KOGAN TEST — is SpliceBERT's 10.63 carrier the SPLICE-JUNCTION sequence phasing?

THE MECHANISTIC POINT, WHICH IS THE WHOLE REASON THIS IS PLAUSIBLE
-----------------------------------------------------------------
A nucleosome dyad is INVISIBLE in sequence.  A splice junction is NOT: the model can find GT/AG
and use it as a PHASE ORIGIN for a relative positional wave.  That is the ingredient the dyad
hypothesis lacks -- and it is exactly what the phase-reset synthetic (D29) was missing, which is
why the table never moved there.  A landmark the model can SEE is what makes a learned carrier
mechanically possible.

    biology at dyads .................. 10.298   (D28, measured -- NOT the carrier)
    SpliceBERT's carrier .............. 10.630   (measured sub-bin, two grids)
    Kogan splice-junction RR/YY ....... "10.58 +/- 0.35"

⚠️ DO NOT REPRODUCE KOGAN LITERALLY.  Their 10.58 +/- 0.35 is the MEAN AND DISPERSION OF NINE
PERIODOGRAM MAXIMA -- not a confidence interval on one common wavelength.  They then FIXED the
period at 10.38 to estimate phase, and only a subset of their individual channel fits reached
significance.  So: nothing here is anchored to 10.58, to 10.38, or to 10.63.  We scan, we freeze
on discovery chromosomes, we test on held-out ones.

DESIGN
------
CHANNELS.  GG and CC are PRIMARY (Kogan: in human and mouse the strongest contributors were
specifically GG/CC; AA/TT were weak).  RR, YY, AG, CT, WW, SS are secondary.

FOUR ARMS, NEVER POOLED.  Signals on opposite sides of a junction can differ or CANCEL:
    donor_exon      k in [-70, -20]   relative to the first intron base (the G of GT)
    donor_intron    k in [+20, +70]
    acceptor_intron k in [-70, -20]   relative to the first exon base
    acceptor_exon   k in [+20, +70]
Every sequence is oriented 5'->3' in TRANSCRIPT direction (reverse-complemented on the - strand).
|k| < 20 is excluded so the GT/AG motif itself cannot leak into the estimate.

⚠️ THE ACCEPTOR-INTRON ARM IS THE DANGEROUS ONE.  The polypyrimidine tract sits right inside its
window and creates a strong composition GRADIENT.  A monotone trend in a tapered window is exactly
what produced R16's four successive artefacts.  The dyad calibration DOES NOT TRANSFER.  So this
script calibrates ITSELF, on real splice neighbourhoods, BEFORE it reports anything:
    * FPR from a landmark-destroyed null (junctions shifted 0.5-5 kb: same local sequence,
      same isochore, same composition -- no junction)
    * injection recovery: can it find a tone of known wavelength and phase that IS there?
    * a trend-only check: the acceptor-intron arm must return the null when no tone is present.

PROTOCOL (frozen before any result is examined)
    DISCOVERY  chr1-chr8    -> scan T in [9.0, 11.5], find the peak, read the phase.  FREEZE BOTH.
    TEST       chr9-chr22,X -> the SIGNED projection Re(C * e^{-i phi_frozen}) at T_frozen.
    The signed statistic has a null mean of ZERO.  |C| is positively biased; that is R16's trap.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

GTF = Path(os.environ.get("GFMCD_GTF", "/nonexistent/gencode.gtf.gz"))
FASTA = Path(os.environ.get("GFMCD_FASTA", "/nonexistent/hg38.fa"))
OUT = REPO / "results/phase-lattice/splice_phasing.json"

W = 220                                              # half-width extracted per junction
ARMS = {"donor_exon":      (-70, -20),
        "donor_intron":    (20, 70),
        "acceptor_intron": (-70, -20),
        "acceptor_exon":   (20, 70)}
CH = {"GG": ("G", "G"), "CC": ("C", "C"),            # PRIMARY (Kogan: strongest in human/mouse)
      "RR": ("AG", "AG"), "YY": ("CT", "CT"),        # secondary
      "AG": ("A", "G"), "CT": ("C", "T"),
      "WW": ("AT", "AT"), "SS": ("GC", "GC")}
DISCOVERY = [f"chr{i}" for i in range(1, 9)]
TEST = [f"chr{i}" for i in range(9, 23)] + ["chrX"]
COMP = np.zeros(256, np.uint8)
for a, b in zip(b"ACGTN", b"TGCAN"):
    COMP[a] = b


def junctions() -> dict[str, list]:
    """Unique, canonical GT-AG introns from GENCODE basic.  Returns per-chrom lists of
    (donor_anchor, acceptor_anchor, strand) in GENOMIC coords, transcript-oriented anchors."""
    tx: dict[str, list] = {}
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
            a, b = ex[i][2], ex[i + 1][1]              # intron [a, b) in genomic coords
            if b - a < 40:
                continue
            key = (c, a, b, st)
            if key in seen:
                continue
            seen.add(key)
            # transcript-oriented anchors: donor = first intron base, acceptor = first exon base
            d, q = (a, b) if st == "+" else (b - 1, a - 1)
            by.setdefault(c, []).append((d, q, st))
    return by


def windows(seq: np.ndarray, anchors: np.ndarray, strands: np.ndarray) -> np.ndarray:
    """(N, 2W+1) uint8, TRANSCRIPT-oriented, anchor at index W."""
    idx = anchors[:, None] + np.arange(-W, W + 1)[None, :]
    out = seq[idx]
    neg = strands == 0
    out[neg] = COMP[out[neg][:, ::-1]]
    return out


def coeffs(win: np.ndarray, arm: str, ch: str, T: float) -> np.ndarray:
    """Per-junction complex coefficient.  Per-window mean removed (kills composition), Hann taper
    (no rectangular boundary, so a monotone trend cannot leak into the band)."""
    lo, hi = ARMS[arm]
    k = np.arange(lo, hi + 1)
    h = np.hanning(len(k))
    filt = h * np.exp(-2j * np.pi * k / T)
    s1, s2 = CH[ch]
    i = W + k
    x = (np.isin(win[:, i], [ord(c) for c in s1])
         & np.isin(win[:, i + 1], [ord(c) for c in s2])).astype(np.float32)
    ok = np.isin(win[:, i[0] - 1:i[-1] + 2], [65, 67, 71, 84]).all(1)
    x -= x.mean(1, keepdims=True)
    c = x @ filt
    c[~ok] = 0
    return c


def scan(win_d, arm, ch, Ts):
    return np.array([abs(complex(coeffs(win_d, arm, ch, float(T)).sum())) for T in Ts])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulls", type=int, default=400)
    ap.add_argument("--calibrate-only", action="store_true")
    args = ap.parse_args()

    print("parsing GENCODE v50 basic ...", flush=True)
    by = junctions()
    fa = Fasta(str(FASTA), as_raw=True)
    rng = np.random.default_rng(0)

    win, nul = {}, {}
    for grp, chroms in (("D", DISCOVERY), ("T", TEST)):
        ws, ns = [], []
        for c in chroms:
            if c not in by:
                continue
            s = np.frombuffer(fa[c][:].upper().encode(), dtype=np.uint8)
            for which in (0, 1):                       # 0 = donor, 1 = acceptor
                a = np.array([j[which] for j in by[c]])
                st = np.array([1 if j[2] == "+" else 0 for j in by[c]], np.uint8)
                keep = (a > W + 10) & (a < len(s) - W - 10)
                a, st = a[keep], st[keep]
                ws.append((which, windows(s, a, st)))
                # NULL: destroy the landmark, keep the neighbourhood.  Shift 0.5-5 kb.
                off = rng.choice([-1, 1], len(a)) * rng.integers(500, 5000, len(a))
                b = np.clip(a + off, W + 10, len(s) - W - 11)
                ns.append((which, windows(s, b, st)))
        for which, tag in ((0, "donor"), (1, "acceptor")):
            win[(grp, tag)] = np.concatenate([w for k, w in ws if k == which])
            nul[(grp, tag)] = np.concatenate([w for k, w in ns if k == which])
        print(f"  {grp}: {len(win[(grp,'donor')]):,} donors, "
              f"{len(win[(grp,'acceptor')]):,} acceptors", flush=True)

    rec = {"n_discovery": len(win[("D", "donor")]), "n_test": len(win[("T", "donor")]),
           "arms": {k: list(v) for k, v in ARMS.items()}}

    # ================================================================ CALIBRATION, BEFORE ANYTHING
    print("\n" + "=" * 86)
    print("CALIBRATION ON REAL SPLICE NEIGHBOURHOODS — the dyad calibration does NOT transfer")
    print("=" * 86)
    T_CAL = 10.5
    rec["calibration"] = {}
    print(f"\n  {'arm':16s} {'chan':>4} {'|C| real':>11} {'|C| null':>10} {'ratio':>7}  "
          f"{'INJECT 5%':>10} {'recovered T':>12}")
    for arm in ARMS:
        tag = "donor" if arm.startswith("donor") else "acceptor"
        for ch in ("GG", "CC"):
            real = abs(complex(coeffs(win[("D", tag)], arm, ch, T_CAL).sum()))
            null = abs(complex(coeffs(nul[("D", tag)], arm, ch, T_CAL).sum()))
            # INJECTION: put a tone of KNOWN wavelength (10.5) and phase (0) into the NULL windows
            lo, hi = ARMS[arm]
            k = np.arange(lo, hi + 1)
            wj = nul[("D", tag)].copy()
            p = 0.05 * (1 + np.cos(2 * np.pi * k / 10.5)) / 2
            hit = rng.random((len(wj), len(k))) < p[None, :]
            sub = wj[:, W + k]
            sub[hit] = ord(ch[0])
            wj[:, W + k] = sub
            Ts = np.arange(9.0, 11.52, 0.02)
            v = scan(wj, arm, ch, Ts)
            Trec = float(Ts[v.argmax()])
            rec["calibration"][f"{arm}|{ch}"] = dict(real=real, null=null, ratio=real / max(null, 1),
                                                     injected_T=10.5, recovered_T=Trec)
            print(f"  {arm:16s} {ch:>4} {real:>11,.0f} {null:>10,.0f} {real/max(null,1):>7.1f}x  "
                  f"{'ok' if abs(Trec-10.5) < 0.25 else 'FAIL':>10} {Trec:>12.2f}")
    bad = [k for k, v in rec["calibration"].items() if abs(v["recovered_T"] - 10.5) > 0.25]
    print(f"\n  injection recovery: {'PASS' if not bad else 'FAIL on ' + ', '.join(bad)}")
    print("  (a 5% tone of known wavelength must be recovered to <0.25 nt, or the arm is unusable)")
    if args.calibrate_only:
        OUT.write_text(json.dumps(rec, indent=1))
        return

    # ================================================================ DISCOVERY -> FREEZE -> TEST
    Ts = np.arange(9.0, 11.52, 0.02)
    print("\n" + "=" * 86)
    print("DISCOVERY (chr1-8): scan, then FREEZE wavelength and phase.  Nothing is anchored to Kogan.")
    print("=" * 86)
    rec["result"] = {}
    for arm in ARMS:
        tag = "donor" if arm.startswith("donor") else "acceptor"
        print(f"\n  --- {arm} ---")
        for ch in ("GG", "CC", "RR", "YY", "WW", "SS", "AG", "CT"):
            v = scan(win[("D", tag)], arm, ch, Ts)
            i = int(v.argmax())
            if 0 < i < len(v) - 1:
                y0, y1, y2 = v[i - 1], v[i], v[i + 1]
                off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            else:
                off = 0.0
            T_f = float(Ts[i] + off * 0.02)                       # FROZEN
            cd = complex(coeffs(win[("D", tag)], arm, ch, T_f).sum())
            phi = math.atan2(cd.imag, cd.real)                    # FROZEN
            # TEST half, held-out chromosomes
            ct = complex(coeffs(win[("T", tag)], arm, ch, T_f).sum())
            stat = float((ct * np.exp(-1j * phi)).real)
            cn = coeffs(nul[("T", tag)], arm, ch, T_f)
            vv = (cn * np.exp(-1j * phi)).real
            n = len(cn)
            mu, sd = float(n * vv.mean()), float(math.sqrt(n * vv.var()))
            z = (stat - mu) / sd if sd else float("nan")
            pt = math.degrees(math.atan2(ct.imag, ct.real))
            dphi = ((pt - math.degrees(phi) + 180) % 360) - 180
            rec["result"][f"{arm}|{ch}"] = dict(T=T_f, phi_deg=math.degrees(phi), z=z,
                                                phase_agreement_deg=dphi, stat=stat)
            star = " <--" if abs(z) > 8 and ch in ("GG", "CC") else ""
            print(f"    {ch:>3}  T = {T_f:6.3f}   phi = {math.degrees(phi):+7.1f} deg   "
                  f"z = {z:+8.1f}   phase agrees to {dphi:+6.1f} deg{star}")

    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")
    print("\n" + "=" * 86)
    print("READ AGAINST THE DECISION TABLE — and note the carrier is 10.630, not 10.67")
    print("  sharp near 10.6-10.65 .......... strong candidate source for the carrier")
    print("  near 10.1-10.3 ................. the human-only carrier / chromatin pitch instead")
    print("  GG,CC counterphase but DIFFERENT T -> 'Kogan 10.58' hides multiple objects")
    print("  broad 9-11, no stable phase .... cannot explain a SHARP carrier")
    print("=" * 86)


if __name__ == "__main__":
    main()
