#!/usr/bin/env python
"""THE DATA PACKAGE FOR THE FOUR PUBLICATION FIGURES.

One JSON in, four figures out.  The figures read NOTHING but this file, so the artwork
cannot drift from the result.

Two things are RECOMPUTED here rather than copied, and both change a number:

  1. EXACT ADD-ONE PERMUTATION p-VALUES.  run_postsoftmax.py stored p = (nul >= |c|).mean()
     over 400 permutations, so a head that no permutation beat was recorded as p = 0.0000.
     That is not a p-value; 0 hits in 400 draws bounds p, it does not measure it.  The exact
     add-one estimate is p = (hits + 1) / (nperm + 1), floor 1/401 = 0.002494.  BH-FDR at
     q = 0.05 is then re-applied ACROSS THE WHOLE SCANNED HEAD FAMILY (the declared
     correction), because the floor is not below BH's first threshold and the answer is
     therefore not guaranteed to survive.  IT DOES NOT, IN ONE MODEL -- see below.

  2. THE POSITION-TABLE SPECTRA, for all three checkpoints, on the convention already used by
     run_carrier_geometry.py:
         F = rfft(P - P.mean(0), axis=0)      <- FULL table, mean-centred rows
         power_rel[k] = |F[k]|^2.sum() / median(|F[1:]|^2.sum(1))
         period[k]    = N / k
         circularity  = sigma2/sigma1 of [Re F[k]; Im F[k]]   (1 = circle, 0 = line)
         overlap(a,b) = mean squared cosine of the principal angles between the two subspaces
     Every recomputed value is asserted against the recorded result before anything is
     written.  A mismatch is a HARD FAILURE, not a warning.

WHICH CHECKPOINT IS WHICH -- this matters and the figures say so:
    SpliceBERT.1024nt        final, multi-species.  The period-3 circuit lives here
                             (combs, L2H15, L3H1, dose, mediation).
    SpliceBERT.510nt         stage-1, multi-species.  The 8 Level-A heads live here.
    SpliceBERT-human.510nt   stage-1, HUMAN ONLY.  The matched control for the above.
The 510nt pair is used for the Level-A contrast precisely because the two checkpoints differ
ONLY in training data -- that is the whole point of the comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results/phase-lattice"
OUT = RES / "fig_report_data.json"

MODELS = {
    "final_1024_multispecies": "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt",
    "stage1_510_multispecies": "data/models/splicebert_official_archive/extracted/models/SpliceBERT.510nt",
    "stage1_510_human_only":   "data/models/splicebert_official_archive/extracted/models/SpliceBERT-human.510nt",
}
NPERM, FDR = 400, 0.05

# Preregistered QK readers for the human-only carrier (frozen before Level A was measured).
PREREG = ["L1H3", "L1H2", "L1H8", "L1H14", "L1H11", "L1H7", "L1H5", "L1H10",
          "L1H9", "L1H13", "L1H15", "L2H8", "L2H1", "L2H11"]

J = lambda n: json.loads((RES / f"{n}.json").read_text())


# ---------------------------------------------------------------- spectra & geometry
def table(path: str) -> np.ndarray:
    m = AutoModel.from_pretrained(str(REPO / path), torch_dtype=torch.float32)
    return m.embeddings.position_embeddings.weight.detach().float().cpu().numpy()


def spectrum(P: np.ndarray) -> dict:
    N = P.shape[0]
    F = np.fft.rfft(P - P.mean(0, keepdims=True), axis=0)
    pw = (np.abs(F) ** 2).sum(1)
    pw[0] = 0.0
    med = float(np.median(pw[1:]))

    def circ(k: int) -> float:
        s = np.linalg.svd(np.stack([F[k].real, F[k].imag]), compute_uv=False)
        return float(s[1] / s[0])

    return dict(N=N, F=F, pw=pw, med=med, circ=circ,
                rel=(pw / med), period=np.array([np.inf] + [N / k for k in range(1, len(pw))]))


def subspace(F: np.ndarray, bins: list[int]) -> np.ndarray:
    M = np.concatenate([np.stack([F[b].real, F[b].imag]) for b in bins], 0)
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[s > s[0] * 1e-6]


def overlap(F: np.ndarray, a: list[int], b: list[int]) -> float:
    return float((np.linalg.svd(subspace(F, a) @ subspace(F, b).T, compute_uv=False) ** 2).mean())


def plane(P: np.ndarray, F: np.ndarray, k: int) -> dict:
    """The trajectory that BIN k alone traces in its own 2-D plane.

    Bin k contributes  P_k(p) = (2/N) [ Re F[k] cos(2 pi k p / N)  -  Im F[k] sin(2 pi k p / N) ],
    which sweeps an ELLIPSE in the plane spanned by Re F[k] and Im F[k] as position p advances.
    Circularity = sigma2/sigma1 of that plane's basis:

        1.0 -> a CIRCLE.  Every phase in [0, 2pi) is realised in SOME embedding direction, so
               the bin HAS NO PHASE OF ITS OWN.  It is a rotating coordinate system, and a head
               CHOOSES a phase from it via its QK bilinear form.  It does not read one off.
        0.0 -> a LINE.  A standing wave: one phase, shared by every dimension.

    We reconstruct bin k rather than projecting the raw rows, because the raw rows carry every
    other bin too -- the scatter would be other components, not this carrier's geometry.
    """
    N = P.shape[0]
    M = np.stack([F[k].real, F[k].imag])                    # 2 x d
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    th = 2 * np.pi * k * np.arange(N) / N
    Pk = (2 / N) * (np.outer(np.cos(th), F[k].real) - np.outer(np.sin(th), F[k].imag))
    xy = Pk @ Vt[:2].T                                      # N x 2
    return dict(xy=[[float(a), float(b)] for a, b in xy],
                circularity=float(s[1] / s[0]),
                sigma=[float(s[0]), float(s[1])],
                bin=k, period=N / k)


def check(name: str, got: float, want: float, tol: float) -> None:
    ok = abs(got - want) <= tol
    print(f"   {'OK  ' if ok else 'FAIL'} {name:52s} got {got:12.4f}  recorded {want:12.4f}")
    if not ok:
        sys.exit(f"\nSELF-CHECK FAILED: {name}. The package does not reproduce the record. STOP.")


def main() -> None:
    print("SELF-CHECK: recomputed spectra vs the recorded result")
    S, PT = {}, {}
    for tag, path in MODELS.items():
        PT[tag] = table(path)
        S[tag] = spectrum(PT[tag])
        print(f"   {tag:26s} table {PT[tag].shape}")

    f1024, f510m, f510h = (S["final_1024_multispecies"], S["stage1_510_multispecies"],
                           S["stage1_510_human_only"])
    cp = {b["bin"]: b for b in J("carrier_phase")["bins"]}
    hc, cg, cs = J("human_carrier"), J("carrier_geometry"), J("carrier_subbin")

    check("1024nt bin 342 power/med", f1024["rel"][342], cp[342]["power_rel"], 0.5)
    check("1024nt bin 342 circularity", f1024["circ"](342), cp[342]["circularity"], 2e-3)
    check("1024nt bin 96  power/med", f1024["rel"][96], cp[96]["power_rel"], 0.5)
    check("1024nt bin 96  circularity", f1024["circ"](96), cp[96]["circularity"], 2e-3)
    check("1024nt overlap(p3, carrier)", overlap(f1024["F"], [342], [96]),
          cg["negctl_p3_vs_p10"], 1e-3)
    check("1024nt overlap(p3, its sidebands)", overlap(f1024["F"], [342], [341, 343]),
          cg["calib_p3_carrier_vs_sidebands"], 1e-3)
    check("510nt multi  bin 48 power/med", f510m["rel"][48], hc["multispecies"]["enrichment"], 0.5)
    check("510nt multi  bin 48 circularity", f510m["circ"](48), hc["multispecies"]["circularity"], 2e-3)
    check("510nt human  bin 50 power/med", f510h["rel"][50], hc["human_only"]["enrichment"], 0.5)
    check("510nt human  bin 50 circularity", f510h["circ"](50), hc["human_only"]["circularity"], 2e-3)
    print("   -> the package reproduces the record.\n")

    # ------------------------------------------------ exact add-one p + BH, per model
    print("EXACT ADD-ONE PERMUTATION p-VALUES  (floor 1/401 = %.6f)" % (1 / (NPERM + 1)))
    ps = J("postsoftmax")
    levelA = {}
    for name in ("multispecies", "human_only"):
        H = ps["models"][name]["heads"]
        n = len(H)
        rows = []
        for x in H:
            hits = int(round(x["p"] * NPERM))
            rows.append(dict(head=x["head"], layer=int(x["head"].split("H")[0][1:]),
                             m_T=x["m_T"], phase_deg=x["phase_deg"],
                             perm_hits=hits, n_perm=NPERM,
                             p_exact=(hits + 1) / (NPERM + 1),
                             kept_carrier=x["kept_carrier"], kept_reciprocal=x["kept_reciprocal"],
                             kept_low_leak=x["kept_low_leak"], kept_shuffled=x["kept_shuffled"]))
        rows.sort(key=lambda r: r["p_exact"])
        kmax = 0
        for i, r in enumerate(rows, 1):
            r["bh_crit"] = FDR * i / n
            if r["p_exact"] <= r["bh_crit"]:
                kmax = i
        sig = {r["head"] for r in rows[:kmax]}
        for r in rows:
            r["significant"] = r["head"] in sig
            r["LEVEL_A"] = bool(r["significant"] and r["kept_carrier"] < 0.50
                                and r["kept_reciprocal"] > 0.70 and r["kept_low_leak"] > 0.70)
        A = [r["head"] for r in rows if r["LEVEL_A"]]
        levelA[name] = rows
        was = sum(1 for x in H if x["LEVEL_A"])
        print(f"   {name:14s} n={n:3d} heads   BH rejects {kmax:2d}   LEVEL A: {len(A)} "
              f"(raw-p run said {was})   {A}")
        # The minimum family size that CAN reach significance at this permutation budget.
        need = int(np.ceil((1 / (NPERM + 1)) * n / FDR))
        print(f"                  -> at 400 permutations, BH needs >= {need} heads AT THE FLOOR "
              f"before any can be called significant.")
    print()

    # ------------------------------------------------ assemble
    dose, med, sur = J("carrier_dose"), J("carrier_mediation"), J("carrier_surgery")
    fp, dw, sl = J("fig_phase_data"), J("dyad_wavelength"), J("dyad_real")
    cph = J("carrier_phase")

    p3_bins, car_bins = [339, 340, 341, 342, 343, 344, 345], [94, 95, 96, 97, 98, 99, 100]
    rec = {
        "_README": "Generated by make_fig_report_data.py. The canonical phase-lattice figures read "
                   "ONLY this file. Every number is traceable to results/phase-lattice/*.json "
                   "or is recomputed here with a self-check against them.",
        "checkpoints": {
            "final_1024_multispecies": "SpliceBERT.1024nt - final, multi-species. Period-3 circuit.",
            "stage1_510_multispecies": "SpliceBERT.510nt - stage 1, multi-species. The 8 Level-A heads.",
            "stage1_510_human_only":   "SpliceBERT-human.510nt - stage 1, HUMAN ONLY. Matched control.",
            "note": "The Level-A contrast uses the 510nt PAIR because they differ only in training data.",
        },
        "status": {
            "period3_clock":        "ESTABLISHED - P, QK, attention, OV-mediation, behaviour.",
            "donor_acceptor":       "ESTABLISHED - mechanistic routing circuit.",
            "carrier_10_63":        "ESTABLISHED at Level A (attention). OV and behaviour: OPEN.",
            "human_only_10_2":      "NULL at Level A. Preregistered readers: 0 of 14. CLOSED at QK.",
            "multiplicity":         f"BH-FDR q={FDR} across the full scanned head family, per model.",
            "permutations":         f"{NPERM}; exact add-one p, floor {1/(NPERM+1):.6f}. Never 0.",
            "preregistered":        PREREG,
            "exploratory":          "The 8 multispecies Level-A heads were found by an unpreregistered "
                                    "scan of all heads clearing the 2% band-mass floor; they are "
                                    "multiplicity-corrected across that family.",
        },
        # ---- FIGURE 2
        "spectra": {
            tag: dict(N=int(S[tag]["N"]),
                      rel=[float(x) for x in S[tag]["rel"][1:]],           # k = 1..N/2, DC dropped
                      period=[float(x) for x in S[tag]["period"][1:]])
            for tag in MODELS
        },
        "top_bins_1024": cph["bins"],
        "planes": {
            "period3":  plane(PT["final_1024_multispecies"], f1024["F"], 342),
            "carrier":  plane(PT["final_1024_multispecies"], f1024["F"], 96),
        },
        "overlaps": {
            "p3_vs_carrier":       overlap(f1024["F"], [342], [96]),
            "p3_vs_p3_sidebands":  overlap(f1024["F"], [342], [341, 343]),
            "carrier_vs_sidebands": overlap(f1024["F"], [96], [97, 98]),
            "matrix_bins": [341, 342, 343, 95, 96, 97],
            "matrix": [[overlap(f1024["F"], [a], [b]) for b in (341, 342, 343, 95, 96, 97)]
                       for a in (341, 342, 343, 95, 96, 97)],
        },
        "subbin": cs["models"] | {"consensus_wavelength": cs["consensus_wavelength"],
                                  "biology_wavelength": cs["biology_wavelength"]},
        # ---- FIGURE 3
        "combs": ["L0H0", "L0H11", "L1H10", "L1H12", "L2H2", "L2H15"],
        "Rk": fp["Rk"], "Rk_offsets": fp["Rk_offsets"], "phase_mass": fp["phase_mass"],
        "head_phases": cph["heads"], "worst_phase_err_deg": cph["worst_phase_err_deg"],
        "dose": fp["dose"], "alphas": dose["alphas"], "chance_ll": dose["chance"],
        "dom_phase": J("dual_reader")["dom"],
        "surgery": dict(damage=sur["period3"]["damage"], kept=sur["period3"]["kept"],
                        p_did=sur["p_did"], p_exon=sur["p_exon"],
                        n_controls=len(sur["controls"]),
                        controls=[dict(bins=c["bins"], damage=c["damage"], kept=c["kept"])
                                  for c in sur["controls"]]),
        # THE NULLS ARE SIZE-MATCHED, AND POOLING THEM WOULD BE A CATEGORY ERROR.
        # 24 nulls restore ONE random head (max rescue 0.042); 24 restore SIX random heads
        # (median 0.063, but max 0.939 -- a random six-head set CAN nearly fully rescue).  Each
        # named set is therefore scored ONLY against the null of its own size, with an exact
        # add-one p.  "ALL 16 layer-2 heads" has NO size-matched null and so gets no p-value:
        # it is reported, and explicitly marked unsupported, rather than silently compared.
        "mediation": dict(
            intact=med["coupling_intact"], killed=med["coupling_killed"], supplied=med["supplied"],
            named={k: dict(rescue=v["rescue"], lo=v["lo"], hi=v["hi"], heads=v["heads"],
                           n_heads=len(v["heads"]),
                           p_add_one=(
                               (sum(1 for kk, vv in med["rescue"].items()
                                    if kk.startswith("__null") and len(vv["heads"]) == len(v["heads"])
                                    and vv["rescue"] >= v["rescue"]) + 1) /
                               (sum(1 for kk, vv in med["rescue"].items()
                                    if kk.startswith("__null") and len(vv["heads"]) == len(v["heads"])) + 1)
                               if any(kk.startswith("__null") and len(vv["heads"]) == len(v["heads"])
                                      for kk, vv in med["rescue"].items()) else None))
                   for k, v in med["rescue"].items() if not k.startswith("__null")},
            nulls_by_size={
                str(n): sorted(v["rescue"] for k, v in med["rescue"].items()
                               if k.startswith("__null") and len(v["heads"]) == n)
                for n in sorted({len(v["heads"]) for k, v in med["rescue"].items()
                                 if k.startswith("__null")})}),
        "bands": dict(period3=p3_bins, control=car_bins),
        # ---- FIGURE 4
        "levelA": levelA,
        "levelA_frozen": ps["frozen"],
        # measured by measure_band_mass.py, which proves its inputs by reproducing m_T to 1e-6
        "band_mass": J("band_mass"),
        "preflight": J("preflight_postsoftmax")["checks"],
        "wavelengths": {
            "nucleosomal":     dict(T=dw["fine_scan"]["T_peak"], z=sl["primary"]["z"],
                                    depth=sl["primary"]["depth"],
                                    corroborate=[dw["fine_scan"]["T_peak"],
                                                 dw["phase_rotation"]["T"], dw["genic"]["T_peak"]]),
            "carrier":         dict(T=cs["consensus_wavelength"],
                                    fits=[cs["models"]["stage1 multi-species (510nt)"]["wavelength"],
                                          cs["models"]["final (1024nt)"]["wavelength"]]),
            "human_only":      dict(T=cs["models"]["human-only (510nt)"]["wavelength"],
                                    bin_label=10.24, level_A=False),
            "kogan_unresolvable": dict(lo=7.0, hi=16.0, note="51-nt window: dT ~ 4 nt at T = 10."),
        },
    }
    OUT.write_text(json.dumps(rec, indent=1))
    print(f"saved -> {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
