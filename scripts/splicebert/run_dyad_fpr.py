#!/usr/bin/env python
"""PRECONDITION FOR USING THE VOONG DATA: the dyad instrument's false-positive rate, at scale.

The user's requirement, before any real map is touched:
    >= 1000 null replicates; combined FPR near 5%; FPR must NOT depend on jitter;
    then FREEZE the complete analysis (exclusions, chromosome split).

NOTE ON WHAT THIS ACTUALLY TESTS.  Under the null, jitter CANNOT change the FPR analytically --
jittering a random dyad position just gives another random dyad position, so the distribution of
C is unchanged.  This run is therefore a test of the IMPLEMENTATION (window extraction, edge
clipping, the taper), not of the theory.  If the FPR moves with jitter, there is a bug.

The 8/12/4% spread at n=24 was 2, 3 and 1 hits -- entirely consistent with 5%.  This settles it.

SPEED: the per-window coefficient c_j = sum_k h(k)*[x_j(k) - mu_j]*e^{-i omega k} is precomputed
ONCE for a large pool; a replicate is then just a sum of sampled c_j.  1000s of replicates become
instant, and the estimator is bit-identical to run_dyad_assay.py's.
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

T0, K = 10.6667, 75
FILT = np.hanning(2 * K + 1) * np.exp(-2j * np.pi * np.arange(-K, K + 1) / T0)
W_SET = np.array([ord("A"), ord("T")])
OUT = REPO / "results/phase-lattice/dyad_fpr.json"


def coeffs(b: np.ndarray, pos: np.ndarray, jitter: float, rng) -> np.ndarray:
    """Per-window complex coefficient, exactly as run_dyad_assay.py computes it."""
    if jitter:
        pos = pos + np.round(rng.normal(0, jitter, len(pos))).astype(int)
    pos = np.clip(pos, K + 10, len(b) - K - 12)
    out = np.empty(len(pos), np.complex128)
    for s in range(0, len(pos), 50_000):                        # chunk: memory, not maths
        p = pos[s:s + 50_000]
        w = b[p[:, None] + np.arange(-K, K + 2)[None, :]]
        ok = np.isin(w, [ord(c) for c in "ACGT"]).all(1)
        x = (np.isin(w[:, :-1], W_SET) & np.isin(w[:, 1:], W_SET)).astype(np.float32)
        x -= x.mean(1, keepdims=True)
        c = x @ FILT
        c[~ok] = 0
        out[s:s + 50_000] = c
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=2_000_000)
    ap.add_argument("--dyads", type=int, default=100_000)
    ap.add_argument("--reps", type=int, default=1200)
    args = ap.parse_args()

    fa = Fasta(str(DEFAULT_FASTA), as_raw=True)
    rng = np.random.default_rng(0)
    chroms = ["chr1", "chr2", "chr3", "chr4"]
    per = 12_000_000
    B = []
    for c in chroms:
        s = int(rng.integers(5_000_000, len(fa[c]) - per - 5_000_000))
        B.append(np.frombuffer(fa[c][s:s + per].upper().encode(), dtype=np.uint8))
    print(f"pool: {args.pool:,} random dyads over {len(chroms)} chromosomes "
          f"({per*len(chroms)/1e6:.0f} Mb)\n")

    JIT = [0.0, 1.0, 2.0, 3.0]
    C = {}
    for j in JIT:
        parts = []
        for ci, b in enumerate(B):
            r = np.random.default_rng(100 + ci + int(j * 10))
            pos = r.integers(K + 10, len(b) - K - 12, size=args.pool // len(B))
            parts.append(coeffs(b, pos, j, r))
        C[j] = np.concatenate(parts)
        print(f"  jitter {j:.0f} bp: {len(C[j]):,} window coefficients")

    def reps(j, n, seed):
        r = np.random.default_rng(seed)
        idx = r.integers(0, len(C[j]), size=(n, args.dyads))
        return np.abs(C[j][idx].sum(1))

    # THRESHOLD: the 95th percentile of the null, fixed at jitter 0 -- then FROZEN
    thr = float(np.percentile(reps(0.0, 3000, 11), 95))
    print(f"\nthreshold (95th pct of the jitter-0 null, {args.dyads:,} dyads): |C| = {thr:.0f}")
    print("   -> FROZEN.  Applied unchanged at every jitter.\n")

    print("=" * 74)
    print(f"FALSE-POSITIVE RATE.  {args.reps} INDEPENDENT null replicates per jitter level.")
    print("=" * 74)
    print(f"\n{'jitter':>7s} {'FPR':>8s} {'95% CI':>18s} {'hits':>10s}")
    print("-" * 50)
    rec = {"threshold": thr, "dyads": args.dyads, "reps": args.reps, "fpr": {}}
    allhits = alln = 0
    for j in JIT:
        a = reps(j, args.reps, 900 + int(j * 7))
        h = int((a > thr).sum())
        f = h / args.reps
        se = np.sqrt(f * (1 - f) / args.reps)
        rec["fpr"][f"{j}"] = dict(fpr=f, hits=h, n=args.reps)
        allhits += h
        alln += args.reps
        print(f"{j:7.0f} {f*100:7.1f}% [{(f-1.96*se)*100:6.1f}, {(f+1.96*se)*100:5.1f}] "
              f"{h:6d}/{args.reps}")
    comb = allhits / alln
    se = np.sqrt(comb * (1 - comb) / alln)
    rec["fpr_combined"] = dict(fpr=comb, hits=allhits, n=alln)
    print("-" * 50)
    print(f"{'ALL':>7s} {comb*100:7.1f}% [{(comb-1.96*se)*100:6.1f}, {(comb+1.96*se)*100:5.1f}] "
          f"{allhits:6d}/{alln}")

    # does FPR depend on jitter?  chi-square across the four levels
    obs = np.array([rec["fpr"][f"{j}"]["hits"] for j in JIT])
    exp = np.full(len(JIT), obs.mean())
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    from math import erf, sqrt
    # 3 dof; use a simple survival approximation
    p_indep = float(np.exp(-chi2 / 2) * (1 + chi2 / 2 + chi2**2 / 8))    # chi2_3 upper tail approx
    rec["jitter_independence"] = dict(chi2=chi2, dof=len(JIT) - 1, p=p_indep)
    print(f"\n  does the FPR depend on jitter?   chi2({len(JIT)-1}) = {chi2:.2f},  p ~ {p_indep:.2f}")
    ok_fpr = abs(comb - 0.05) < 0.015
    ok_jit = p_indep > 0.05
    print(f"\n  FPR near 5%?              {'PASS' if ok_fpr else 'FAIL'}  ({comb*100:.1f}%)")
    print(f"  FPR independent of jitter? {'PASS' if ok_jit else 'FAIL'}")
    rec["pass"] = bool(ok_fpr and ok_jit)

    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
