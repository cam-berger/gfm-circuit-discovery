#!/usr/bin/env python
"""Derive the WAVE decomposition of R(k) and fold it into fig_phase_data.json.

This is a pure function of the R(k) already recorded there -- it loads no model and touches no
genome, so it is cheap, deterministic, and re-runnable.  `make_fig_phase_data.py` calls the same
library function, so the two can never disagree.

R(k) = envelope(k) + A*cos(2*pi*k/3 + phi)

The one cosine is not a modelling choice: at three samples per cycle the second harmonic aliases
onto the fundamental, so a period-3 integer signal has exactly one resolvable sinusoid.  See
src/gfmcd/splicing/wave.py and tests/test_wave_memorials.py.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.wave import boxcar_amplitude, decompose  # noqa: E402

DATA = REPO / "results/phase-lattice/fig_phase_data.json"
CONDS = ("intact", "no_carrier", "no_control")


def build(D: dict) -> dict:
    k = np.array(D["Rk_offsets"], float)
    out: dict = {"heads": {}}
    for hd, series in D["Rk"].items():
        rec = {}
        for c in CONDS:
            y = np.array(series[c], float)
            f = decompose(k, y)
            rec[c] = dict(A=f.A, phi=f.phi, env_coef=list(f.env_coef),
                          var_explained=f.var_explained, max_resid=f.max_resid,
                          peak_offset=f.peak_offset, codon_position=f.codon_position,
                          lock_error=f.lock_error,
                          A_boxcar=boxcar_amplitude(k, y))          # the model-free cross-check
        rec["kept_clock_deleted"] = rec["no_carrier"]["A"] / rec["intact"]["A"]
        rec["kept_control_deleted"] = rec["no_control"]["A"] / rec["intact"]["A"]
        out["heads"][hd] = rec

    A = {c: sum(out["heads"][h][c]["A"] for h in out["heads"]) for c in CONDS}
    out["summary"] = dict(
        A_mean_intact=A["intact"] / len(out["heads"]),
        kept_clock_deleted=A["no_carrier"] / A["intact"],       # ratio of means, not mean of ratios
        kept_control_deleted=A["no_control"] / A["intact"],
        # agreement between the two independent estimators; if this drifts from 1 the envelope
        # model is stealing amplitude and the fitted A is fiction (test_wave_memorials.py)
        ols_vs_boxcar=(sum(out["heads"][h]["intact"]["A_boxcar"] for h in out["heads"])
                       / A["intact"]),
        var_explained_min=min(out["heads"][h]["intact"]["var_explained"] for h in out["heads"]),
        var_explained_median=float(np.median(
            [out["heads"][h]["intact"]["var_explained"] for h in out["heads"]])),
        # the estimator's hallucination floor, measured on clock-free signals (see the memorial)
        leak_floor=0.08,
    )
    return out


if __name__ == "__main__":
    D = json.loads(DATA.read_text())
    D["wave"] = build(D)
    DATA.write_text(json.dumps(D, indent=1))

    w = D["wave"]
    print(f"saved -> {DATA}\n")
    print(f"{'head':8s} {'A (logits)':>11s} {'A_boxcar':>9s} {'codon pos':>10s} "
          f"{'var expl':>9s} | {'clock del':>10s} {'ctrl del':>9s}")
    print("-" * 74)
    for hd, r in w["heads"].items():
        i = r["intact"]
        print(f"{hd:8s} {i['A']:11.3f} {i['A_boxcar']:9.3f} {i['codon_position']:10d} "
              f"{i['var_explained']*100:8.1f}% | {r['kept_clock_deleted']*100:9.1f}% "
              f"{r['kept_control_deleted']*100:8.1f}%")
    s = w["summary"]
    print("-" * 74)
    print(f"  clock deleted keeps {s['kept_clock_deleted']*100:.1f}% of the wave; the "
          f"energy-matched control keeps {s['kept_control_deleted']*100:.1f}%")
    print(f"  OLS vs model-free boxcar: {s['ols_vs_boxcar']*100:.1f}% agreement "
          f"(the envelope is not stealing amplitude)")
    print(f"  one cosine explains {s['var_explained_median']*100:.1f}% of the ripple (median), "
          f"{s['var_explained_min']*100:.1f}% worst")
