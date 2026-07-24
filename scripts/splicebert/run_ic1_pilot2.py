#!/usr/bin/env python
"""IC1 pilot v2 (ACC_UP=70): seal the thresholds the freeze-audit flagged as unsealed --
 - sham-acceptor A' construction + ΔC_sham distribution (the dispersion comparator + material-gate control)
 - dispersion EXCESS-over-sham + its between-intron MDE (dispersion co-primary null licensing)
 - no_created_regulatory_signal band: Δ(max cryptic MaxEnt score3/score5) + PPT poly-Y run-length change
 - on-path (through-model) positive control: plant a strong cryptic 3'SS in W, require ΔC_plant ≥ SESOI
 - homology near-duplicate rate among eligible distal windows (gene->cluster collapse sanity)
Disjoint TRAIN split. Linear f. Reuses run_ic1_pilot's shuffler + f_linear.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from maxentpy import maxent                                                       # noqa: E402
from maxentpy.maxent import load_matrix3                                          # noqa: E402
try:
    from maxentpy.maxent import load_matrix5
    M5 = load_matrix5(); HAVE5 = True
except Exception:
    M5 = None; HAVE5 = False

from gfmcd.provenance import ExperimentManifest, sha256_file                      # noqa: E402
from gfmcd.splicing.circuit import (                                             # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, pick_device, eligible, _dataset)
from gfmcd.splicing.coupling import substitute                                    # noqa: E402
from run_ic1_pilot import anchored_shuffle, f_linear, window, DON_DN, hamming     # noqa: E402

M3 = load_matrix3()
ACC_UP = 70
N_SEED = 16
PLANT = "T" * 18 + "AGG"          # strong cryptic 3'SS (PPT + AG); positive-control payload (21 nt)


def max_score3(s):
    best = -99.0
    for i in range(len(s) - 22):
        w = s[i:i + 23].upper()
        if set(w) <= set("ACGT"):
            best = max(best, maxent.score3(w, matrix=M3))
    return best if best > -98 else float("nan")


def max_score5(s):
    if not HAVE5:
        return float("nan")
    best = -99.0
    for i in range(len(s) - 8):
        w = s[i:i + 9].upper()
        if set(w) <= set("ACGT"):
            best = max(best, maxent.score5(w, matrix=M5))
    return best if best > -98 else float("nan")


def max_polyY(s):
    best = cur = 0
    for c in s.upper():
        cur = cur + 1 if c in "CT" else 0
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pilot", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/development/ic1_pilot2.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = pick_device(args.device); ad = load_adapter(args.model, dev); task = CouplingTask(ad, batch=args.batch)

    train = eligible(_dataset(args.gtf, args.fasta), "train")
    seen, dedup = set(), []
    for x in train:
        if x.gene_id not in seen:
            seen.add(x.gene_id); dedup.append(x)

    def elig70(x):
        lo, hi = window(x, ACC_UP)
        return 0 <= lo and hi < len(x.sequence) and (hi - lo + 1) >= 30

    pool = [x for x in dedup if elig70(x)]
    # dose floor + take n_pilot
    rng = np.random.default_rng(0)
    pilot = []
    for x in pool:
        lo, hi = window(x, ACC_UP); W = x.sequence[lo:hi + 1]
        sh = [anchored_shuffle(W, np.random.default_rng((hash(x.gene_id) % 99991) * 31 + s)) for s in range(N_SEED)]
        ham = [hamming(W, s) for s in sh]
        if np.median(ham) >= 8 and np.median(ham) / len(W) >= 0.25 and len(set(sh)) == N_SEED:
            pilot.append((x, lo, hi, W, sh))
        if len(pilot) >= args.n_pilot:
            break
    print(f"pool(ACC_UP=70)={len(pool)}  dose-passing pilot={len(pilot)}")

    # ---- homology near-dup rate (sampled pairwise identity on W) ----
    Ws = [x.sequence[window(x, ACC_UP)[0]:window(x, ACC_UP)[1] + 1] for x in pool[:800]]
    def ident(a, b):
        n = min(len(a), len(b)); return sum(1 for i in range(n) if a[i] == b[i]) / max(len(a), len(b))
    ndup = 0
    for i in range(len(Ws)):
        for j in range(i + 1, min(i + 40, len(Ws))):
            if ident(Ws[i], Ws[j]) >= 0.8:
                ndup += 1; break
    homology_ndup_rate = ndup / max(len(Ws), 1)

    # ---- build all forwards: A+, A-, A_sham on native + each shuffled seed; + plant A+/A- ----
    seqs, dons, tags = [], [], []
    def add(seq, d, tag):
        seqs.append(seq); dons.append(d); tags.append(tag)
    crypt = []
    for gi, (x, lo, hi, W, sh) in enumerate(pilot):
        a, d0 = x.next_acceptor_pos, x.donor_pos
        Am = substitute(x.sequence, a - 2, "CC")            # partner corrupt (A-)
        Ash = substitute(x.sequence, a - 10, "AA")          # sham: PPT purine-swap (non-partner)
        add(x.sequence, d0, (gi, "I", "+")); add(Am, d0, (gi, "I", "-")); add(Ash, d0, (gi, "I", "s"))
        for s, Wsh in enumerate(sh):
            sp = x.sequence[:lo] + Wsh + x.sequence[hi + 1:]
            add(sp, d0, (gi, s, "+")); add(substitute(sp, a - 2, "CC"), d0, (gi, s, "-"))
            add(substitute(sp, a - 10, "AA"), d0, (gi, s, "s"))
        # on-path positive control: plant strong cryptic 3'SS mid-W
        mid = lo + (len(W) - len(PLANT)) // 2
        pl = x.sequence[:mid] + PLANT + x.sequence[mid + len(PLANT):]
        add(pl, d0, (gi, "PL", "+")); add(substitute(pl, a - 2, "CC"), d0, (gi, "PL", "-"))
        # cryptic-motif deltas (no model): native vs shuffled W
        d3 = [max_score3(Wsh) - max_score3(W) for Wsh in sh]
        d5 = [max_score5(Wsh) - max_score5(W) for Wsh in sh]
        dpy = [max_polyY(Wsh) - max_polyY(W) for Wsh in sh]
        crypt.append((np.nanmax(np.abs(d3)), np.nanmax(np.abs(d5)) if HAVE5 else np.nan,
                      float(np.max(np.abs(dpy)))))

    fv = f_linear(ad, task, seqs, dons, args.batch)
    by = {}
    for (gi, s, w), v in zip(tags, fv):
        by.setdefault(gi, {})[(s, w)] = v

    dc, dc_sham, disp, disp_sham, excess, dc_plant, fshift_ratio = [], [], [], [], [], [], []
    for gi in range(len(pilot)):
        b = by[gi]
        base = b[("I", "+")] - b[("I", "-")]
        base_s = b[("I", "+")] - b[("I", "s")]
        dcs = [(b[(s, "+")] - b[(s, "-")]) - base for s in range(N_SEED)]
        dss = [(b[(s, "+")] - b[(s, "s")]) - base_s for s in range(N_SEED)]
        dc.append(float(np.mean(dcs))); dc_sham.append(float(np.mean(dss)))
        disp.append(float(np.std(dcs))); disp_sham.append(float(np.std(dss)))
        excess.append(float(np.std(dcs) - np.std(dss)))
        dc_plant.append(((b[("PL", "+")] - b[("PL", "-")]) - base))
        # sham f-shift comparability: |f(A_sham)-f(A+)| vs |f(A-)-f(A+)|
        fshift_ratio.append(abs(b[("I", "s")] - b[("I", "+")]) / (abs(b[("I", "-")] - b[("I", "+")]) + 1e-9))

    zc = 2.8016
    n_elig = len(pool)
    sd_excess = float(np.std(excess, ddof=1))
    mde_disp_census = zc * sd_excess / np.sqrt(max(n_elig, 1))
    d3s = np.array([c[0] for c in crypt]); d5s = np.array([c[1] for c in crypt]); dpys = np.array([c[2] for c in crypt])

    rep = {
        "acc_up": ACC_UP, "n_seed": N_SEED, "n_eligible_genes_pool": n_elig, "n_pilot_postdose": len(pilot),
        "homology_ndup_rate_ge80pct": homology_ndup_rate,
        "sham": {"construction": "substitute(seq, a-10, 'AA')  (PPT purine-swap, non-partner)",
                 "dC_sham_mean": float(np.mean(dc_sham)), "dC_sham_absmean": float(np.mean(np.abs(dc_sham))),
                 "dC_sham_p95_abs": float(np.percentile(np.abs(dc_sham), 95)),
                 "fshift_ratio_med": float(np.median(fshift_ratio)),
                 "PASS_band_note": "seal ΔC_sham CI ⊆ [-0.05,+0.05]; fshift_ratio_med shows comparability"},
        "dispersion": {"disp_shuffle_mean": float(np.mean(disp)), "disp_sham_mean": float(np.mean(disp_sham)),
                       "excess_mean": float(np.mean(excess)), "excess_sd_between": sd_excess,
                       "mde_disp_census": float(mde_disp_census), "meets_0.05": bool(mde_disp_census <= 0.05)},
        "on_path_positive": {"plant": PLANT, "dC_plant_median": float(np.median(dc_plant)),
                             "dC_plant_absmedian": float(np.median(np.abs(dc_plant))),
                             "frac_ge_0.05": float(np.mean(np.abs(dc_plant) >= 0.05)),
                             "PASS": bool(np.median(np.abs(dc_plant)) >= 0.05)},
        "created_motif_band": {"d_max_score3_p95": float(np.nanpercentile(d3s, 95)),
                               "d_max_score5_p95": float(np.nanpercentile(d5s, 95)) if HAVE5 else None,
                               "d_polyY_runlen_p95": float(np.nanpercentile(dpys, 95)),
                               "have_score5": HAVE5,
                               "note": "seal these p95 as |Δ|-within-band cutoffs for no_created_regulatory_signal"},
    }
    out = REPO / args.out
    out.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    man = ExperimentManifest.create("splicebert_coupling:ic1_pilot2", sys.argv, status="PILOT",
        config={"acc_up": ACC_UP, "n_seed": N_SEED, "n_pilot": len(pilot)}, seeds={"base": 0},
        notes=["IC1 pilot v2: sham, dispersion-MDE, on-path positive control, created-motif band, homology."])
    man.add_input(str(ckpt / "pytorch_model.bin")); man.write(out.parent / "ic1_pilot2.manifest.json")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
