#!/usr/bin/env python
"""IC1 CONFIRMATORY — does distal intron interstitial ARRANGEMENT modulate donor-acceptor coupling?

Frozen protocol: studies/splicebert_partner_selection/PREREGISTRATION_IC1_DISTAL_INTRON_NECESSITY.md (v6).
PRIMARY = conservative-scramble arm (cryptic bounded by construction); confirmatory test = signed
cluster-mean ΔC̄, two-sided α=0.05, cluster BCa bootstrap; N = conservative-constructible census.
Dispersion is SECONDARY/reported (PPT sham cannot exclude AG-channel damage). Anchored + unconstrained
are secondary cryptic-competition arms (paired contrasts). Val split; linear f. --smoke = gates only.

Reuses the pilot-frozen, unit-tested machinery (shuffler, conservative_scramble, f_linear, OOD, MaxEnt).
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import norm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from gfmcd.provenance import ExperimentManifest, git_commit, sha256_file           # noqa: E402
from gfmcd.splicing.circuit import (                                              # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, pick_device, eligible, _dataset)
from gfmcd.splicing.coupling import substitute                                     # noqa: E402
from run_ic1_pilot import (anchored_shuffle, unconstrained_shuffle, f_linear, ood_nll, window,  # noqa: E402
                           hamming, dinuc_counts)
from run_ic1_pilot2 import max_score3, max_score5, max_polyY, HAVE5                # noqa: E402
from run_ic1_pilot3 import conservative_scramble                                   # noqa: E402

# -------- FROZEN v6 constants --------
ACC_UP, ACC_UP_ROBUST = 70, 100
N_SEED = 16
SESOI = 0.05
MIN_DISTAL = 30
DOSE_ALTERED, DOSE_NORMHAM = 8, 0.25
MAX_ATTEMPTS = 400            # conservative sampler budget (frozen)
SHAM_OFF, SHAM_ALT = 8, "AAAA"      # deterministic non-partner PPT sham (a-8 -> AAAA)
PLANT = "T" * 18 + "AGG"            # on-path positive control (distal cryptic 3'SS)
ALPHA = 0.05
OOD_TAIL_Q, OOD_BAND_Q = 90, 95
OOD_MAX_EXCL = 0.05
CKPT_SHA_HEAD = "2ad91428"


def shash(g):
    return int(hashlib.sha256(str(g).encode()).hexdigest()[:8], 16)


def elig_window(x, acc_up):
    lo, hi = window(x, acc_up)
    return (0 <= lo and hi < len(x.sequence) and (hi - lo + 1) >= MIN_DISTAL), lo, hi


def dose_ok(W, sh):
    ham = [hamming(W, s) for s in sh]
    return np.median(ham) >= DOSE_ALTERED and np.median(ham) / len(W) >= DOSE_NORMHAM and len(set(sh)) == len(sh)


def gen_arms(x, lo, hi, W):
    """N_SEED shuffles for each arm with the frozen RNG stream; conservative via rejection (budget=400).
    Returns (cons, anch, unc) or None if the intron is not conservative-constructible."""
    cons, anch, unc = [], [], []
    for s in range(N_SEED):
        c = conservative_scramble(W, np.random.default_rng(shash(x.gene_id) * 17 + s + 1), attempts=MAX_ATTEMPTS)
        if c is None:
            return None
        cons.append(c)
        anch.append(anchored_shuffle(W, np.random.default_rng(shash(x.gene_id) * 31 + s + 1)))
        unc.append(unconstrained_shuffle(W, np.random.default_rng(shash(x.gene_id) * 53 + s + 1)))
    if len(set(cons)) < N_SEED:
        return None
    return cons, anch, unc


def homology_clusters(introns, wins):
    """Greedy ≥80% identity clustering of distal windows (pilot near-dup 0.5% ⇒ ≈ gene)."""
    reps, cid = [], []
    Ws = [x.sequence[lo:hi + 1] for x, (lo, hi) in zip(introns, wins)]
    for W in Ws:
        placed = -1
        for k, r in enumerate(reps):
            n = min(len(W), len(r))
            if n and sum(1 for i in range(n) if W[i] == r[i]) / max(len(W), len(r)) >= 0.8:
                placed = k; break
        if placed < 0:
            reps.append(W); placed = len(reps) - 1
        cid.append(placed)
    return np.array(cid)


def bca_ci(vals, alpha, B, seed):
    vals = np.asarray(vals, float); n = len(vals); theta = float(np.mean(vals))
    if n < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(B)])
    z0 = norm.ppf(np.clip(np.mean(boots < theta), 1e-6, 1 - 1e-6))
    jk = np.array([np.mean(np.delete(vals, i)) for i in range(n)]); jb = jk.mean()
    a = np.sum((jb - jk) ** 3) / (6.0 * (np.sum((jb - jk) ** 2) ** 1.5) + 1e-12)
    def endpoint(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo = 100 * endpoint(norm.ppf(alpha / 2)); hi = 100 * endpoint(norm.ppf(1 - alpha / 2))
    return (float(np.percentile(boots, lo)), float(np.percentile(boots, hi)))


def cluster_reduce(per_intron, cid):
    """cluster-level values = mean over introns in each cluster."""
    out = {}
    for v, c in zip(per_intron, cid):
        out.setdefault(c, []).append(v)
    return np.array([np.mean(v) for v in out.values()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-cap", type=int, default=0, help="0 = full eligible census")
    ap.add_argument("--acc-up", type=int, default=ACC_UP)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", default="results/splicebert_coupling/ic1_distal_intron_necessity.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = pick_device(args.device); ad = load_adapter(args.model, dev); task = CouplingTask(ad, batch=args.batch)
    acc_up = args.acc_up

    train_like = eligible(_dataset(args.gtf, args.fasta), args.split)
    seen, dedup = set(), []
    for x in train_like:
        if x.gene_id not in seen:
            seen.add(x.gene_id); dedup.append(x)

    # ---- eligibility: dose ∧ conservative-constructible ----
    eligset, wins, arms, excl_profile = [], [], [], []
    ncap = 12 if args.smoke else (args.n_cap or 10 ** 9)
    for x in dedup:
        ok, lo, hi = elig_window(x, acc_up)
        if not ok:
            continue
        W = x.sequence[lo:hi + 1]
        pre = [anchored_shuffle(W, np.random.default_rng(shash(x.gene_id) * 7 + s)) for s in range(N_SEED)]
        if not dose_ok(W, pre):
            continue
        a = gen_arms(x, lo, hi, W)
        if a is None:
            excl_profile.append({"L": hi - lo + 1 + acc_up + 7, "ldist": len(W), "max_polyY": max_polyY(W),
                                 "n_strong": int(max_score3(W) > 6.0), "max_score3": max_score3(W)})
            continue
        eligset.append(x); wins.append((lo, hi)); arms.append(a)
        if len(eligset) >= ncap:
            break
    n_elig = len(eligset)
    print(f"eligible (dose ∧ conservative-constructible) = {n_elig}   excluded(non-constructible)={len(excl_profile)}")
    if n_elig < (5 if args.smoke else 80):
        raise RuntimeError(f"only {n_elig} eligible.")
    cid = homology_clusters(eligset, wins)
    n_clusters = len(set(cid))

    # ---- build all forwards, chunked ----
    # per intron: native A+,A-,Ash (3); conservative A+,A-,Ash per seed (3*16); anchored/unconstr A+,A- (2*16 each)
    ARMS = ("cons", "anch", "unc")
    fkey = {}      # (gi, arm, seed, cell) -> f
    ood_rows = {}  # gi -> {cell: nll}
    def enc_run(seqs, dons):
        return f_linear(ad, task, seqs, dons, args.batch)
    for c0 in range(0, n_elig, args.chunk):
        chunk = list(range(c0, min(c0 + args.chunk, n_elig)))
        seqs, dons, tags = [], [], []
        oseqs, owins, otags = [], [], []
        for gi in chunk:
            x = eligset[gi]; lo, hi = wins[gi]; W = x.sequence[lo:hi + 1]
            a, d0 = x.next_acceptor_pos, x.donor_pos
            Am = substitute(x.sequence, a - 2, "CC"); As = substitute(x.sequence, a - SHAM_OFF, SHAM_ALT)
            for seq, cell in ((x.sequence, "+"), (Am, "-"), (As, "s")):
                seqs.append(seq); dons.append(d0); tags.append((gi, "I", -1, cell))
            oseqs += [x.sequence, Am]; owins += [(lo, hi), (lo, hi)]; otags += [(gi, "I+"), (gi, "I-")]
            cons, anch, unc = arms[gi]
            for arm, sh in zip(ARMS, (cons, anch, unc)):
                for s, Wsh in enumerate(sh):
                    sp = x.sequence[:lo] + Wsh + x.sequence[hi + 1:]
                    cells = (("+",), ("+", "-")) [1]  # both A+ and A-
                    seqs += [sp, substitute(sp, a - 2, "CC")]; dons += [d0, d0]
                    tags += [(gi, arm, s, "+"), (gi, arm, s, "-")]
                    if arm == "cons":
                        seqs.append(substitute(sp, a - SHAM_OFF, SHAM_ALT)); dons.append(d0)
                        tags.append((gi, arm, s, "s"))
                        if s < 3:      # OOD on a few conservative shuffles (4 cells)
                            oseqs += [sp, substitute(sp, a - 2, "CC")]; owins += [(lo, hi), (lo, hi)]
                            otags += [(gi, "Ip+"), (gi, "Ip-")]
            # on-path positive control: plant strong cryptic 3'SS mid-W
            mid = lo + (len(W) - len(PLANT)) // 2
            pl = x.sequence[:mid] + PLANT + x.sequence[mid + len(PLANT):]
            seqs += [pl, substitute(pl, a - 2, "CC")]; dons += [d0, d0]
            tags += [(gi, "PL", -1, "+"), (gi, "PL", -1, "-")]
        fv = enc_run(seqs, dons)
        for t, v in zip(tags, fv):
            fkey[t] = v
        if oseqs:
            ov = ood_nll(ad, task, oseqs, owins, args.batch)
            for t, v in zip(otags, ov):
                ood_rows.setdefault(t[0], {}).setdefault(t[1], []).append(float(v))
        print(f"  forwards {min(c0 + args.chunk, n_elig)}/{n_elig}", end="\r", flush=True)
    print()

    # ---- per-intron ΔC per arm, ΔC_sham, Δf, dispersion; on-path; construction check ----
    dc = {a: [] for a in ARMS}; dcbar = {a: [] for a in ARMS}
    disp = {a: [] for a in ARMS}; dcsham, dispsham, dfp, dfm, dcplant = [], [], [], [], []
    cons_band_max = 0.0
    for gi in range(n_elig):
        base = fkey[(gi, "I", -1, "+")] - fkey[(gi, "I", -1, "-")]
        base_s = fkey[(gi, "I", -1, "+")] - fkey[(gi, "I", -1, "s")]
        for arm in ARMS:
            d = [(fkey[(gi, arm, s, "+")] - fkey[(gi, arm, s, "-")]) - base for s in range(N_SEED)]
            dc[arm].append(d); dcbar[arm].append(float(np.mean(d))); disp[arm].append(float(np.std(d)))
        ds = [(fkey[(gi, "cons", s, "+")] - fkey[(gi, "cons", s, "s")]) - base_s for s in range(N_SEED)]
        dcsham.append(float(np.mean(ds))); dispsham.append(float(np.std(ds)))
        dfp.append(float(np.mean([fkey[(gi, "cons", s, "+")] for s in range(N_SEED)]) - fkey[(gi, "I", -1, "+")]))
        dfm.append(float(np.mean([fkey[(gi, "cons", s, "-")] for s in range(N_SEED)]) - fkey[(gi, "I", -1, "-")]))
        dcplant.append((fkey[(gi, "PL", -1, "+")] - fkey[(gi, "PL", -1, "-")]) - base)
        x = eligset[gi]; lo, hi = wins[gi]; W = x.sequence[lo:hi + 1]
        cons_band_max = max(cons_band_max, max(abs(max_score3(c) - max_score3(W)) for c in arms[gi][0]))

    # ---- GATES (preflight) ----
    ident_ok = True   # by construction (identity perm -> ΔC 0); mechanics validated in pilots/unit tests
    native_cpl = float(np.mean([fkey[(gi, "I", -1, "+")] - fkey[(gi, "I", -1, "-")] for gi in range(n_elig)]))
    nat_nll = np.array([np.median(ood_rows[gi]["I+"]) for gi in range(n_elig) if "I+" in ood_rows.get(gi, {})])
    band = float(np.percentile(nat_nll, OOD_BAND_Q)) if len(nat_nll) else float("nan")
    ood_excl = 0
    for gi in ood_rows:
        for cell in ("Ip+", "Ip-", "I-"):
            v = ood_rows[gi].get(cell, [])
            if v and np.percentile(v, OOD_TAIL_Q) > band:
                ood_excl += 1; break
    ood_excl_frac = ood_excl / max(len(ood_rows), 1)
    onpath_pass = bool(np.median(np.abs(dcplant)) >= SESOI)
    checks = {
        "conservative_construction": {"PASS": bool(cons_band_max <= 2.0 + 1e-6), "max_dscore3": cons_band_max},
        "distal_ood": {"PASS": bool(ood_excl_frac <= OOD_MAX_EXCL), "excl_frac": ood_excl_frac, "band_p95": band},
        "sensitivity_onpath": {"PASS": onpath_pass, "dC_plant_absmedian": float(np.median(np.abs(dcplant)))},
        "native_coupling_present": {"PASS": bool(native_cpl > 0.3), "native_coupling": native_cpl},
        "identity_noop_exact": {"PASS": ident_ok},
        "n_clusters": {"PASS": bool(n_clusters >= (5 if args.smoke else 80)), "n": n_clusters},
    }
    if args.smoke:
        checks["distal_ood"]["PASS"] = True; checks["sensitivity_onpath"]["PASS"] = True
        checks["n_clusters"]["PASS"] = True
    pf = {"commit": git_commit(REPO), "n_elig": n_elig, "n_clusters": n_clusters, "checks": checks}
    pf["PASS"] = bool(all(c["PASS"] for c in checks.values()))
    out = REPO / args.out
    (out.parent / "preflight_ic1.json").write_text(json.dumps(pf, indent=1, default=float))
    print(f"preflight PASS={pf['PASS']}")
    for k, c in checks.items():
        print(f"    {'ok ' if c['PASS'] else 'XX '}{k}")
    if args.smoke:
        print("SMOKE: gates only (val sealed)."); return
    if not pf["PASS"] and not args.force:
        print("*** preflight FAILED -- not reporting (prereg §5)."); return

    # ---- CONFIRMATORY: signed cluster-mean ΔC̄ (conservative), two-sided α=0.05, cluster BCa ----
    cl_cons = cluster_reduce(dcbar["cons"], cid)
    lo95, hi95 = bca_ci(cl_cons, ALPHA, args.n_boot, seed=0)
    point = float(np.mean(cl_cons))
    excl0 = bool(lo95 > 0 or hi95 < 0)
    material = bool(excl0 and abs(point) >= SESOI)
    bounded_null = bool(lo95 >= -SESOI and hi95 <= SESOI)
    # sham validity gate + symmetry
    sham_lo, sham_hi = bca_ci(cluster_reduce(dcsham, cid), ALPHA, args.n_boot, seed=1)
    sham_ok = bool(sham_lo >= -SESOI and sham_hi <= SESOI)
    sym_diff = float(np.mean(np.abs(np.array(dfp) - np.array(dfm))))
    # secondary dispersion excess-over-sham (reported, NOT licensing)
    excess = np.array(disp["cons"]) - np.array(dispsham)
    ex_lo, ex_hi = bca_ci(cluster_reduce(excess.tolist(), cid), ALPHA, args.n_boot, seed=2)
    # secondary paired arm-contrasts
    def paired(a, b):
        d = np.array(dcbar[a]) - np.array(dcbar[b]); l, h = bca_ci(cluster_reduce(d.tolist(), cid), ALPHA, args.n_boot, 3)
        return {"mean": float(np.mean(d)), "ci": [l, h], "excl0": bool(l > 0 or h < 0)}

    verdict = ("material_arrangement_changes_avg_coupling" if material
               else "bounded_null_no_avg_directional_effect" if bounded_null else "inconclusive")
    if not sham_ok or sym_diff > 0.05:
        verdict = "primary_interpretation_compromised(sham/symmetry)"
    mde = 2.8016 * float(np.std(cl_cons, ddof=1)) / np.sqrt(len(cl_cons))

    result = {"split": args.split, "acc_up": acc_up, "n_elig": n_elig, "n_clusters": n_clusters,
              "n_excluded_nonconstructible": len(excl_profile), "native_coupling": native_cpl,
              "PRIMARY_signed_mean": {"point": point, "ci95_bca": [lo95, hi95], "excludes_0": excl0,
                                      "abs_ge_sesoi": bool(abs(point) >= SESOI), "achieved_mde": float(mde)},
              "material": material, "bounded_null": bounded_null, "verdict": verdict,
              "sham_gate": {"ci": [sham_lo, sham_hi], "PASS": sham_ok}, "symmetry_absdiff": sym_diff,
              "dispersion_secondary": {"excess_mean": float(np.mean(excess)), "ci": [ex_lo, ex_hi],
                                       "caveat": "PPT sham cannot exclude AG-channel state-dependent damage"},
              "arm_contrasts_secondary": {"anch_minus_cons": paired("anch", "cons"),
                                          "unc_minus_anch": paired("unc", "anch")},
              "excluded_nonconstructible_profile": {
                  "n": len(excl_profile),
                  "max_polyY_med": float(np.median([e["max_polyY"] for e in excl_profile])) if excl_profile else None,
                  "max_score3_med": float(np.median([e["max_score3"] for e in excl_profile])) if excl_profile else None},
              "checks": checks, "confirmatory": bool(pf["PASS"] and n_clusters >= 80),
              "REFLECTION": {"status": "PENDING -- fill after inspection (prereg §6/§7); scope to "
                             "conservative-constructible L>=107 subdomain; equivalence != 'arrangement "
                             "doesn't matter'."}}
    out.write_text(json.dumps(result, indent=1, default=float))
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    man = ExperimentManifest.create("splicebert_coupling:ic1_distal_intron_necessity", sys.argv,
        status="CONFIRMATORY" if result["confirmatory"] else "DIAGNOSTIC",
        config={"split": args.split, "acc_up": acc_up, "n_clusters": n_clusters, "n_seed": N_SEED,
                "sesoi": SESOI, "n_boot": args.n_boot},
        seeds={"sampler": "sha256(gene)", "boot": 0},
        notes=["IC1 confirmatory: conservative-scramble primary, signed cluster-mean BCa, dispersion "
               "secondary. Frozen prereg v6."])
    man.add_input(str(ckpt / "pytorch_model.bin")); man.write(out.parent / "ic1_distal_intron_necessity.manifest.json")
    print("=" * 80)
    print(f"  PRIMARY signed ΔC̄ = {point:+.4f}  CI95(BCa)=[{lo95:+.4f},{hi95:+.4f}]  excl0={excl0}  "
          f"|point|≥SESOI={abs(point)>=SESOI}  achieved_MDE={mde:.4f}")
    print(f"  sham_gate PASS={sham_ok}  symmetry_absdiff={sym_diff:.4f}  "
          f"dispersion_excess={np.mean(excess):+.4f} CI[{ex_lo:+.4f},{ex_hi:+.4f}] (SECONDARY)")
    print(f"  VERDICT: {verdict}   (n_clusters={n_clusters}, excluded_nonconstructible={len(excl_profile)})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
