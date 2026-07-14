#!/usr/bin/env python
"""WHAT WAVELENGTH IS THE BIOLOGY ACTUALLY AT?  The single number that decides the chromatin story.

Exp 2 found the source, powerfully: WW phase-locked to mouse nucleosome dyads, depth 4.87%,
z = +249, phase reproducible across chromosomes to 0.4 deg, confined to the core and dead at
+/-500 bp.  But the coarse scan peaked at ~10.3, NOT at SpliceBERT's 10.6667.

That difference is the whole ballgame, so it gets measured three independent ways:

  1. FINE MATCHED-FILTER SCAN.  |C(T)| on a dense grid, frozen-phase split-half at every T.
     The matched filter is unbiased, so its peak IS the tone's wavelength.

  2. PHASE ROTATION WITH ANCHOR OFFSET.  Move the anchor d bp off the dyad; a tone at T rotates
     the measured phase by exactly 360/T degrees per bp.  Fit the slope by UNWRAPPING properly
     (the coarse run's naive mod-360 division was wrong for d > 5 bp).  This estimates T from the
     phase, not the magnitude -- a completely independent statistic.

  3. GENIC RESTRICTION.  The user's spec said "dyads overlapping the transcribed regions
     represented in SpliceBERT's training data."  SpliceBERT saw pre-mRNA, not whole genome.
     If the genic wavelength differs from the genome-wide one, THAT is what SpliceBERT could see.

WHY IT MATTERS.  In a 512-row table:
     10.30 -> mode 49.71   (nearly EXACTLY between bins 49 and 50 -- maximally off-grid)
     10.45 -> mode 49.00
     10.67 -> mode 48.00   <- SpliceBERT
So a grid-snap of a 10.3 tone lands on bin 49 or 50.  It does NOT land on 48.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/splicebert"))

from run_dyad_real import load_dyads, coeffs, FASTA, DISCOVERY, TEST   # noqa: E402
from run_dyad_fpr import K                                             # noqa: E402

OUT = REPO / "results/phase-lattice/dyad_wavelength.json"


def signed_at(seqs, dy, chroms_d, chroms_t, T, ch="WW", sel=None):
    """Frozen-phase split-half at wavelength T.  sel: optional per-chrom boolean mask."""
    cd = sum(complex(coeffs(seqs[c], dy[c] if sel is None else dy[c][sel[c]], ch, T).sum())
             for c in chroms_d)
    ct = sum(complex(coeffs(seqs[c], dy[c] if sel is None else dy[c][sel[c]], ch, T).sum())
             for c in chroms_t)
    ph = math.atan2(cd.imag, cd.real)
    return float((ct * np.exp(-1j * ph)).real), ph


def main() -> None:
    dy = load_dyads()
    fa = Fasta(str(FASTA), as_raw=True)
    seqs = {c: np.frombuffer(fa[c][:].upper().encode(), dtype=np.uint8)
            for c in DISCOVERY + TEST}
    rec = {}

    # ---------------------------------------------------------------- 1. fine scan
    print("=" * 82)
    print("1. FINE MATCHED-FILTER SCAN  (frozen-phase split-half at every T)")
    print("=" * 82)
    Ts = np.round(np.arange(9.8, 11.45, 0.05), 3)
    vals = []
    for T in Ts:
        s, _ = signed_at(seqs, dy, DISCOVERY, TEST, float(T))
        vals.append(s)
        print(f"  T = {T:6.2f}   {s:>12,.0f}  {'#' * max(0, int(56 * s / 1.1e6))}", flush=True)
    vals = np.array(vals)
    i = int(vals.argmax())
    # parabolic interpolation of the peak -- the filter is smooth, so this is exact to ~0.01 nt
    if 0 < i < len(vals) - 1:
        y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
        off = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    else:
        off = 0.0
    T_peak = float(Ts[i] + off * 0.05)
    print(f"\n  PEAK at T = {T_peak:.3f} nt   (grid max {Ts[i]}, parabolic offset {off:+.2f} bins)")
    print(f"  in a 512-row table that is mode {512/T_peak:.2f}")
    print(f"  SpliceBERT's carrier is T = 10.6667 = mode 48.00   -> signal there is "
          f"{100*np.interp(10.6667, Ts, vals)/vals[i]:.0f}% of peak")
    rec["fine_scan"] = dict(T=Ts.tolist(), signed=vals.tolist(), T_peak=T_peak,
                            mode512=512 / T_peak)

    # ---------------------------------------------------------------- 2. phase rotation
    print("\n" + "=" * 82)
    print("2. PHASE ROTATION WITH ANCHOR OFFSET  (an INDEPENDENT estimate of T, from phase)")
    print("=" * 82)
    print("   a tone at T rotates the measured phase by exactly 360/T deg per bp\n")
    ds = np.arange(-40, 41, 2)
    phs = []
    for d in ds:
        cd = sum(complex(coeffs(seqs[c], dy[c] + int(d), "WW").sum()) for c in DISCOVERY)
        phs.append(math.atan2(cd.imag, cd.real))
    phs = np.unwrap(np.array(phs))                    # UNWRAP -- the coarse run failed to
    A = np.vstack([ds, np.ones_like(ds)]).T
    slope, icpt = np.linalg.lstsq(A, phs, rcond=None)[0]
    deg_per_bp = math.degrees(slope)
    T_phase = 360.0 / abs(deg_per_bp)
    resid = phs - (A @ [slope, icpt])
    print(f"   fitted rotation = {deg_per_bp:+.3f} deg/bp   (rms resid {math.degrees(resid.std()):.2f} deg)")
    print(f"   => T = 360/|{deg_per_bp:.3f}| = {T_phase:.3f} nt")
    print(f"\n   scan says   T = {T_peak:.3f}")
    print(f"   phase says  T = {T_phase:.3f}")
    print(f"   AGREE to {abs(T_peak-T_phase):.3f} nt" if abs(T_peak - T_phase) < 0.15
          else f"   ⚠️ DISAGREE by {abs(T_peak-T_phase):.3f} nt")
    rec["phase_rotation"] = dict(deg_per_bp=deg_per_bp, T=T_phase,
                                 rms_resid_deg=math.degrees(resid.std()))

    # ---------------------------------------------------------------- 3. genic restriction
    print("\n" + "=" * 82)
    print("3. GENIC RESTRICTION  — SpliceBERT saw pre-mRNA, not whole genome")
    print("=" * 82)
    gtf = Path("/Users/cberg/Research/data/nucleosome/mm9_refGene.txt.gz")
    if not gtf.exists():
        print(f"   mm9 refGene not present at {gtf} — skipping.")
        print("   (fetch: https://hgdownload.soe.ucsc.edu/goldenPath/mm9/database/refGene.txt.gz)")
    else:
        import gzip
        iv: dict[str, list] = {}
        with gzip.open(gtf, "rt") as fh:
            for line in fh:
                f = line.rstrip("\n").split("\t")
                c, s, e = f[2], int(f[4]), int(f[5])       # chrom, txStart, txEnd
                if c in seqs:
                    iv.setdefault(c, []).append((s, e))
        sel = {}
        for c in DISCOVERY + TEST:
            mark = np.zeros(len(seqs[c]), dtype=bool)
            for s, e in iv.get(c, []):
                mark[s:e] = True
            sel[c] = mark[np.clip(dy[c], 0, len(mark) - 1)]
            print(f"   {c}: {sel[c].sum():>8,} of {len(dy[c]):>8,} dyads are transcribed "
                  f"({100*sel[c].mean():.1f}%)")
        gvals = []
        Tg = np.round(np.arange(9.8, 11.45, 0.05), 3)
        for T in Tg:
            s, _ = signed_at(seqs, dy, DISCOVERY, TEST, float(T), sel=sel)
            gvals.append(s)
        gvals = np.array(gvals)
        j = int(gvals.argmax())
        if 0 < j < len(gvals) - 1:
            y0, y1, y2 = gvals[j - 1], gvals[j], gvals[j + 1]
            offg = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
        else:
            offg = 0.0
        Tg_peak = float(Tg[j] + offg * 0.05)
        print(f"\n   GENIC peak at T = {Tg_peak:.3f} nt  (mode {512/Tg_peak:.2f} in a 512 table)")
        print(f"   genome-wide peak  T = {T_peak:.3f}")
        print(f"   shift = {Tg_peak - T_peak:+.3f} nt")
        rec["genic"] = dict(T=Tg.tolist(), signed=gvals.tolist(), T_peak=Tg_peak,
                            mode512=512 / Tg_peak)

    print("\n" + "=" * 82)
    print("WHAT BIN WOULD A 512-ROW TABLE PUT THIS IN?")
    print("=" * 82)
    for lbl, T in [("biology (this run)", T_peak), ("SpliceBERT's carrier", 10.6667)]:
        m = 512 / T
        print(f"   {lbl:22s} T = {T:6.3f}  ->  mode {m:6.2f}  ->  nearest bin {round(m)}")
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
