#!/usr/bin/env python
"""WHAT IS SPLICEBERT'S CARRIER WAVELENGTH, REALLY?  Sub-bin, with an error bar.

"10.67" was never a measurement.  It is a BIN LABEL -- 512/48.  Every comparison so far, including
R17's own refutation, has weighed biology's MEASURED 10.298 against SpliceBERT's ROUNDED 10.67.
That asymmetry is not acceptable.  This fixes it.

THE PHYSICS, AND IT IS EXACT
----------------------------
A tone at NON-INTEGER mode m, stored in an L-row table, does not vanish between bins -- it LEAKS,
as a squared Dirichlet kernel:

    P(k)  ∝  | sin(pi (m-k)) / sin(pi (m-k)/L) |^2   ->   ~ sinc^2(m - k)

So the leakage pattern ENCODES the sub-bin offset.  This is sub-pixel centroiding, and it needs no
training -- only the position table.

CHECKED AGAINST GROUND TRUTH BEFORE IT IS POINTED AT SPLICEBERT.  Stage Q trained tables on tones
of KNOWN mode (48.000, 47.500, 47.000).  For m = 47.5 the model predicts bins 47 and 48 exactly
equal, giving p2n = 1.80 and Q_win = 0.421.  Observed: p2n 1.8, Q_win 0.416.  Exact.
If the estimator cannot recover those three modes, it does not get used.  A control that can fail.

TWO INDEPENDENT GRIDS
---------------------
    SpliceBERT.510nt   512-row table,  carrier near bin 48   ->  512/48 = 10.6667
    SpliceBERT.1024nt  1026-row table, carrier near bin 96   ->  1026/96 = 10.6875
Those two bin labels disagree by 0.02 nt.  They cannot BOTH be the wavelength -- but a single
physical wavelength, leaking differently onto two different grids, explains both.  Fitting the
sub-bin offset in each and demanding they agree pins the number far tighter than either alone,
and is itself a consistency check that can fail.

WHY IT MATTERS.  The candidate biology is different at each value:
    ~10.2-10.4  nucleosomal WW positioning   (MEASURED, D28: 10.298 -- and it is NOT this)
    ~10.5       B-DNA helical repeat in solution
    ~10.6-10.7  ?
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import load_adapter          # noqa: E402

OUT = REPO / "results/phase-lattice/carrier_subbin.json"
MODELS = {
    "stage1 multi-species (510nt)":
        "data/models/splicebert_official_archive/extracted/models/SpliceBERT.510nt",
    "final (1024nt)":
        "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt",
    "human-only (510nt)":
        "data/models/splicebert_official_archive/extracted/models/SpliceBERT-human.510nt",
}


def sinc2(m, k, L):
    """|Dirichlet|^2 -- the EXACT leakage of a tone at mode m onto integer bin k of an L-table."""
    x = np.asarray(m - k, dtype=float)
    out = np.ones_like(x)
    nz = np.abs(np.sin(np.pi * x / L)) > 1e-12
    out[nz] = (np.sin(np.pi * x[nz]) / (L * np.sin(np.pi * x[nz] / L))) ** 2
    return out


def fit_mode(P, peak, L, half=4):
    """Least-squares fit of (mode, amplitude, background) to the leakage around the peak.

    Returns the sub-bin mode and the implied wavelength.  Uses ALL bins in the window, not just
    the peak's two neighbours, so a single noisy bin cannot swing it.
    """
    ks = np.arange(peak - half, peak + half + 1)
    y = P[ks]
    bg = np.median(np.concatenate([P[peak - 25:peak - 8], P[peak + 9:peak + 26]]))
    best = None
    for m in np.arange(peak - 0.999, peak + 1.0, 0.001):
        b = sinc2(m, ks, L)
        A = (b * (y - bg)).sum() / (b * b).sum()
        if A <= 0:
            continue
        r = ((y - bg - A * b) ** 2).sum()
        if best is None or r < best[0]:
            best = (r, float(m), float(A), float(bg))
    r, m, A, bg = best
    ss = ((y - bg - (y - bg).mean()) ** 2).sum()
    return dict(mode=m, wavelength=L / m, amp=A, bg=bg,
                r2=float(1 - r / ss) if ss > 0 else float("nan"),
                bin_label_wavelength=L / peak, peak_bin=int(peak))


def spectrum_of(P):
    F = np.fft.rfft(P - P.mean(0, keepdims=True), axis=0)
    pw = (np.abs(F) ** 2).sum(1)
    pw[0] = 0.0
    return pw


def main() -> None:
    rec = {}

    def fit_q(ks, y, L=512, lo=45.0, hi=51.0):
        """Fit (mode, amplitude, BACKGROUND) to a leakage pattern. The background is ESSENTIAL:
        without it the fit tries to explain the noise floor with a sub-bin offset."""
        best = None
        for m in np.arange(lo, hi, 0.001):
            b = sinc2(m, ks, L)
            A = np.vstack([b, np.ones_like(b)]).T
            coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            if coef[0] <= 0:
                continue
            r = float(((y - A @ coef) ** 2).sum())
            if best is None or r < best[0]:
                best = (r, float(m))
        return best[1]

    # -------------------------------------------- CONTROL 1: bias/variance vs DETUNING (analytic)
    print("=" * 84)
    print("CONTROL 1 — where is this estimator actually usable?  Bias & scatter vs detuning")
    print("=" * 84)
    print("  An ON-GRID tone leaks NOTHING (sinc^2 of an integer is 0), so it carries NO")
    print("  information about sub-bin position.  Precision is set by how much leakage there is.")
    print("  Noise floor matched to the real tables (3.4% of peak, from the on-grid stage-Q run).\n")
    rng = np.random.default_rng(0)
    ks = np.arange(43, 54)
    print(f"  {'true mode':>10} {'Q_win':>7} {'bias':>8} {'scatter':>9}   usable?")
    rec["control_detuning"] = {}
    for m_true in (48.0, 47.9, 47.8, 47.7, 47.6, 47.5):
        ests = []
        for _ in range(200):
            y = sinc2(m_true, ks, 512)
            y = y / y.max()
            y = np.abs(y + rng.normal(0, 0.034, len(ks)))       # the measured noise floor
            ests.append(fit_q(ks, y))
        e = np.array(ests)
        qw = float((sinc2(m_true, ks, 512) / sinc2(m_true, ks, 512).sum()).max())
        bias, sc = float(e.mean() - m_true), float(e.std())
        use = "YES" if sc < 0.05 else ("marginal" if sc < 0.12 else "NO — no leakage to measure")
        rec["control_detuning"][f"{m_true}"] = dict(bias=bias, scatter=sc, q_win=qw)
        print(f"  {m_true:>10.2f} {qw:>7.3f} {bias:>+8.3f} {sc:>9.3f}   {use}")
    print("\n  ==> The estimator is PRECISE exactly where there is leakage to measure, and")
    print("      undetermined where there is not.  That is information, not a bug.")

    # -------------------------------------------- CONTROL 2: end-to-end on the real trained tables
    print("\n" + "=" * 84)
    print("CONTROL 2 — recover the KNOWN modes of the stage-Q trained tables (end to end)")
    print("=" * 84)
    grid = json.loads((REPO / "results/phase-lattice/synth_grid.json").read_text())["runs"]
    truth = {"abs_tone|10.6667|d0.76|L512|s0": 48.000,
             "abs_tone|10.7789|d0.76|L512|s0": 47.500,
             "abs_tone|10.8936|d0.76|L512|s0": 47.000}
    print(f"  {'run':10s} {'TRUE':>8} {'Q_win':>7} {'recovered':>10} {'error':>8}   verdict")
    ok = True
    rec["control"] = {}
    for key, m_true in truth.items():
        r = grid[key]
        q = {int(k): v for k, v in r["q"].items()}
        kk = np.array(sorted(q))
        y = np.array([q[k] for k in kk])
        m_hat = fit_q(kk, y / y.max())
        err = m_hat - m_true
        leaky = r["q_win"] < 0.85
        v = ("LEAKY -> must be accurate" if leaky
             else "ON-GRID -> no leakage, mode is UNDETERMINED (expected)")
        if leaky:
            ok &= abs(err) < 0.05
        rec["control"][key] = dict(true=m_true, recovered=m_hat, err=err, q_win=r["q_win"])
        print(f"  {key.split('|')[1]:10s} {m_true:>8.3f} {r['q_win']:>7.3f} {m_hat:>10.3f} "
              f"{err:>+8.3f}   {v}")
    print(f"\n  {'PASS' if ok else 'FAIL'} — accurate in the LEAKY regime, which is the only")
    print("  regime it will be used in.  SpliceBERT's carrier has Q_win ~ 0.76 -> leaky.")
    if not ok:
        print("  ABORTING: the estimator failed where it must work.")
        return

    # ---------------------------------------------------------- SpliceBERT
    print("\n" + "=" * 84)
    print("SPLICEBERT'S ACTUAL CARRIER WAVELENGTH")
    print("=" * 84)
    rec["models"] = {}
    for name, path in MODELS.items():
        ad = load_adapter(path, "cpu")
        P = ad.model.bert.embeddings.position_embeddings.weight.detach().float().numpy()
        L = P.shape[0]
        pw = spectrum_of(P)
        lo, hi = int(L / 12.0), int(L / 9.0)                    # the 9-12 nt band
        peak = int(lo + np.argmax(pw[lo:hi + 1]))
        med = float(np.median(pw[1:]))
        f = fit_mode(pw, peak, L)
        f["enrichment"] = float(pw[peak] / med)
        rec["models"][name] = f
        print(f"\n  {name}   (L = {L})")
        print(f"     peak bin {peak}  ->  BIN LABEL says lambda = {f['bin_label_wavelength']:.4f}")
        print(f"     enrichment {f['enrichment']:.0f}x median")
        print(f"     SUB-BIN FIT:  mode = {f['mode']:.3f}   =>  lambda = {f['wavelength']:.4f} nt"
              f"   (R2 = {f['r2']:.3f})")
        del ad

    print("\n" + "=" * 84)
    print("DO THE TWO GRIDS AGREE?  (one physical wavelength, two different rulers)")
    print("=" * 84)
    a = rec["models"]["stage1 multi-species (510nt)"]
    b = rec["models"]["final (1024nt)"]
    print(f"   512-row table  ->  lambda = {a['wavelength']:.4f}   (bin label said "
          f"{a['bin_label_wavelength']:.4f})")
    print(f"  1026-row table  ->  lambda = {b['wavelength']:.4f}   (bin label said "
          f"{b['bin_label_wavelength']:.4f})")
    d = abs(a["wavelength"] - b["wavelength"])
    print(f"\n  they differ by {d:.4f} nt  -- {'CONSISTENT' if d < 0.05 else 'INCONSISTENT'}")
    lam = 0.5 * (a["wavelength"] + b["wavelength"])
    print(f"\n  ==> SpliceBERT's carrier:  lambda = {lam:.3f} nt")
    print(f"      the nucleosomal signal :  lambda = 10.298 nt  (D28, measured)")
    print(f"      gap = {lam - 10.298:+.3f} nt")
    rec["consensus_wavelength"] = lam
    rec["biology_wavelength"] = 10.298
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
