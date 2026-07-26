#!/usr/bin/env python
"""ED1 -- is donor->acceptor coupling ARCHITECTURE-CONDITIONED (exon vs intron definition)?

Protocol of record:
  studies/splicebert_partner_selection/preregistrations/
      PREREGISTRATION_ED1_ARCHITECTURE_BOUNDARY_PAIRING.md
Design red-team (must-fixes M1-M9 applied):
  studies/splicebert_partner_selection/redteam/REDTEAM_SYNTHESIS_ED1.md

    upstream acceptor -- EXON -- donor -- INTRON -- downstream acceptor
          (a)                    (d)                    (na)

PRIMARY (intersection-union; BOTH must reject in the predicted direction):
    E_switch = mean over matched pairs of  d C_exon    (LONG - SHORT downstream intron)
    dd       = E_switch - I_switch                     (architecture x direction)
`dd` alone cannot separate a switch from cross-intron distance decay -- L_down IS the
cross-intron separation -- so `E_switch`, whose separation (L_exon) is matched to +-10nt,
is inside the primary rather than beside it.

Usage
    --smoke              memorials + decision-layer calibration on tiny TRAIN n; val sealed
    --split train        pilot: seals SESOI / sd_between / the native-coupling floor
    --split val          confirmatory
    --from-scores PATH   re-analyse an existing score artifact without re-scoring
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

np.seterr(all="ignore")
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, CouplingConfig, CouplingScorer, build_coupling_dataset, substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402
from gfmcd.splicing.exon_intron import (  # noqa: E402
    ARCH_LONG_MIN, ARCH_SHORT_MAX, DENSITY_KEYS, EXACT_KEYS, ShamPolicy,
    architecture_cell, axis_spec, covariate_row, density_spec, dyadic_bootstrap, eligible,
    dyadic_bootstrap_p, gc_fraction, match_pairs, sham_positions, signflip_p,
    twoway_wild_p,
)

# ---- SEALED constants (prereg §5/§8; scale re-derived on the LINEAR observable, M9) -----
SESOI = 0.10                     # nats, donor margin -- overridden by --sesoi at seal time
NATIVE_COUPLING_FLOOR = 0.30     # C_intron on down-SHORT must clear this (TRAIN-sealed)
BALANCE_MAX_SMD = 0.10
N_BOOT = 2000
ALPHA = 0.05
SHAM = ShamPolicy(min_sep=25, max_log2_ratio=1.0, k=1)

#: seven states; value = (which sites to destroy). "u"/"d" are the shams.
STATES = ("N", "U-", "D-", "UD-", "u-", "d-", "D-u-")


# ========================= construction ==================================================

def build_states(x, p_up_sham, p_dn_sham, k):
    """The seven sequences for alt-index ``k``.  ``None`` where a sham is unavailable."""
    alt = ACCEPTOR_ALTS[k]
    a, na = x.acceptor_pos, x.next_acceptor_pos
    s = x.sequence
    out = {"N": s, "U-": substitute(s, a - 2, alt), "D-": substitute(s, na - 2, alt)}
    out["UD-"] = substitute(out["U-"], na - 2, alt)
    out["u-"] = substitute(s, p_up_sham - 2, alt) if p_up_sham else None
    out["d-"] = substitute(s, p_dn_sham - 2, alt) if p_dn_sham else None
    out["D-u-"] = substitute(out["D-"], p_up_sham - 2, alt) if p_up_sham else None
    return out


def check_variant(native, variant, pos, label):
    """MEMORIAL `variant_construction_exact` -- on-path, can fail.

    Exactly 2 bases differ, at exactly [pos-2, pos), length preserved, and the target no
    longer reads AG.  Catches any off-by-one in the genomic->window position mapping.
    """
    if len(variant) != len(native):
        return f"{label}: length changed"
    diff = [i for i, (u, v) in enumerate(zip(native, variant)) if u != v]
    if diff != [pos - 2, pos - 1]:
        return f"{label}: changed {diff[:6]} expected {[pos-2, pos-1]}"
    if variant[pos - 2:pos] == "AG":
        return f"{label}: target still reads AG"
    return None


# ========================= scoring =======================================================

def score_split(args, cfg):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    tok = AutoTokenizer.from_pretrained(mp)
    scorer = CouplingScorer(AutoModelForMaskedLM.from_pretrained(mp).to(dev), tok,
                            device=dev, batch_size=args.batch)

    chroms = {"train": None, "val": list(cfg.val_chromosomes),
              "test": list(cfg.test_chromosomes)}[args.split]
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg,
                               allowed_chromosomes=set(chroms) if chroms else None)
    X = [x for x in X if x.split == args.split and eligible(x)]
    X.sort(key=lambda x: x.example_id)                      # deterministic, hashseed-free
    if args.limit and args.limit < len(X):
        # Deterministic STRIDE, not a head slice: example_id is chrom-ordered, so the
        # first N units are genomically clustered and yield almost no caliper matches.
        X = X[::len(X) // args.limit][:args.limit]
    print(f"[{args.split}] {len(X):,} eligible units  device={dev}", flush=True)

    pwm = pickle.load(open(args.pwm, "rb"))
    from gfmcd.splicing.baselines import site_window
    a_sc = lambda s, p: pwm.acceptor_pwm.score(site_window(s, p, "acceptor"))  # noqa: E731
    d_sc = lambda s, p: pwm.donor_pwm.score(site_window(s, p, "donor"))  # noqa: E731

    memo = {"variant_construction_exact": [], "readout_anchor": [], "identity_noop": []}
    rows, t0 = [], time.time()
    for n, x in enumerate(X):
        if n % 200 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(X)}  {el:.0f}s  eta {el/max(n,1)*(len(X)-n):.0f}s",
                  end="\r", flush=True)
        pu = sham_positions(x, "exon", SHAM)
        pd = sham_positions(x, "intron", SHAM)
        p_up, p_dn = (pu[0] if pu else None), (pd[0] if pd else None)

        # MEMORIAL readout_anchor: the masked index must carry the donor G (on-path).
        ids = tok(" ".join(x.sequence), return_tensors="pt")["input_ids"][0]
        if tok.convert_ids_to_tokens(int(ids[x.donor_pos + 1])).upper() != "G":
            memo["readout_anchor"].append(x.example_id)
        if x.sequence[x.donor_pos:x.donor_pos + 2] != "GT":
            memo["readout_anchor"].append(x.example_id + ":GT")

        seqs, tags = [], []
        for k in range(len(ACCEPTOR_ALTS)):
            st = build_states(x, p_up, p_dn, k)
            for name, s in st.items():
                if s is None:
                    continue
                if name != "N":
                    parent, tpos = {
                        "U-": (x.sequence, x.acceptor_pos),
                        "D-": (x.sequence, x.next_acceptor_pos),
                        "UD-": (st["U-"], x.next_acceptor_pos),
                        "u-": (x.sequence, p_up),
                        "d-": (x.sequence, p_dn),
                        "D-u-": (st["D-"], p_up),
                    }[name]
                    err = check_variant(parent, s, tpos, f"{x.example_id}/{name}/k{k}")
                    if err:
                        memo["variant_construction_exact"].append(err)
                if k > 0 and name == "N":
                    continue                                # native scored once
                seqs.append(s)
                tags.append((name, k))
        # MEMORIAL identity_noop_exact (labelled FORCED): AG->AG must be a no-op.
        seqs.append(substitute(x.sequence, x.acceptor_pos - 2, "AG"))
        tags.append(("noop", 0))

        pos = [x.donor_pos] * len(seqs)
        f_lin, f_max = scorer.margins_both(seqs, pos, "G")

        acc: dict = {}
        for (name, k), vl, vm in zip(tags, f_lin, f_max):
            acc.setdefault(name, []).append((vl, vm))
        f = {nm: (float(np.mean([v[0] for v in vs])), float(np.mean([v[1] for v in vs])))
             for nm, vs in acc.items()}
        if abs(f["noop"][0] - f["N"][0]) > 0:
            memo["identity_noop"].append(x.example_id)

        rows.append(dict(
            example_id=x.example_id, gene=x.gene_id, chrom=x.chrom, strand=x.strand,
            cell=architecture_cell(x), gene_type=x.gene_type,
            L_exon=x.exon_length, L_down=x.intron_length_down, L_up=x.intron_length_up,
            donor_pos=x.donor_pos, acc_pos=x.acceptor_pos, next_acc=x.next_acceptor_pos,
            s_acc=a_sc(x.sequence, x.acceptor_pos), s_next=a_sc(x.sequence, x.next_acceptor_pos),
            s_don=d_sc(x.sequence, x.donor_pos),
            gc_exon=gc_fraction(x.sequence[x.acceptor_pos:x.donor_pos]),
            gc_intron=gc_fraction(x.sequence[x.donor_pos:x.next_acceptor_pos]),
            n_iso_don=x.n_isoforms_donor, n_iso_acc=x.n_isoforms_acceptor,
            n_real_acc=len(x.real_acc), n_real_don=len(x.real_don),
            p_up_sham=p_up, p_dn_sham=p_dn, f={k: v for k, v in f.items() if k != "noop"},
        ))
    print()

    # MEMORIAL batch_invariance -- rescore N for a sample in a differently-composed batch.
    sub = rows[::max(1, len(rows) // 64)][:64]
    ids2 = {r["example_id"] for r in sub}
    seqs2 = [x.sequence for x in X if x.example_id in ids2]
    pos2 = [x.donor_pos for x in X if x.example_id in ids2]
    got = scorer._margin(seqs2, pos2, "G", reduce="mean")
    want = [r["f"]["N"][0] for r in rows if r["example_id"] in ids2]
    memo["batch_invariance_max_abs"] = float(np.max(np.abs(np.asarray(got) - np.asarray(want)))) \
        if len(want) == len(got) and want else float("nan")
    return rows, memo


# ========================= estimands =====================================================

def per_example(r, which=0):
    """Per-example estimands (``which=0`` linear observable, ``1`` max-margin)."""
    f = {k: v[which] for k, v in r["f"].items()}
    g = lambda k: f.get(k, float("nan"))  # noqa: E731
    out = dict(
        C_exon=f["N"] - g("U-"), C_intron=f["N"] - g("D-"),
        C_exon_sham=f["N"] - g("u-"), C_intron_sham=f["N"] - g("d-"),
        clean=f["N"],
    )
    out["S_exon"] = out["C_exon"] - out["C_exon_sham"]
    out["S_intron"] = out["C_intron"] - out["C_intron_sham"]
    out["G_partner"] = (g("D-") - g("UD-")) - (f["N"] - g("U-"))
    out["G_sham"] = (g("D-") - g("D-u-")) - (f["N"] - g("u-"))
    out["G_specific"] = out["G_partner"] - out["G_sham"]
    # MEMORIAL did_regrouping_identity (labelled FORCED by linear-f algebra)
    out["_regroup_resid"] = abs(out["G_partner"] - ((g("D-") - f["N"]) - (g("UD-") - g("U-"))))
    return out


def smd(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / s) if s > 0 else 0.0


def build_matched(rows, axis, density_matched=False):
    """Frozen outcome-blind matching.

    ``density_matched=False`` is the PRIMARY spec and estimates the TOTAL architecture
    effect; in-window site density is a descendant of the exposure, so it is reported as a
    diagnostic, not adjusted away.  ``density_matched=True`` is the D1 secondary and
    estimates the CONTROLLED DIRECT effect holding site density fixed.
    """
    arch = [dict(r, up_band=r["cell"][0], dn_band=r["cell"][1])
            for r in rows if r["cell"] is not None]
    if density_matched:
        cov, cal, ex = density_spec(axis)
    else:
        cov, cal = axis_spec(axis)
        ex = EXACT_KEYS[axis]
    key = "dn_band" if axis == "down" else "up_band"
    treated = [r[key] == "L" for r in arch]
    if axis == "up" and sum(treated) > len(treated) - sum(treated):
        treated = [not t for t in treated]        # process the smaller arm (max pairs)
    pairs = match_pairs(arch, treated, covariates=cov, calipers=cal, exact=ex)
    bal = {k: smd([covariate_row(arch[j])[k] for _, j in pairs],
                  [covariate_row(arch[i])[k] for i, _ in pairs]) for k in cov}
    # density reported separately: it is an exposure property, NOT part of the gate
    bal_density = {k: smd([arch[j][k] for _, j in pairs], [arch[i][k] for i, _ in pairs])
                   for k in DENSITY_KEYS}
    worst = max(abs(v) for v in bal.values()) if bal else float("nan")
    return arch, pairs, dict(bal, **{f"_density_{k}": v for k, v in bal_density.items()}), worst


def contrast(arch, pairs, est, field, treated_is_L=True, seed=0):
    """Matched-pair mean of (treated - control) for ``field``, with dyadic CI."""
    d, ga, gb = [], [], []
    for i, j in pairs:
        vi, vj = est[arch[i]["example_id"]][field], est[arch[j]["example_id"]][field]
        d.append((vi - vj) if treated_is_L else (vj - vi))
        ga.append(arch[i]["gene"])
        gb.append(arch[j]["gene"])
    m, lo, hi = dyadic_bootstrap(d, ga, gb, n_boot=N_BOOT, seed=seed, alpha=ALPHA)
    return dict(mean=m, lo=lo, hi=hi, n_pairs=len(d),
                signflip_p=signflip_p(d, seed=seed), diffs=d, gene_a=ga, gene_b=gb)


# ========================= calibration ===================================================

def calibrate(arch, pairs, est, sesoi, seed=0):
    """Planted-positive (sensitivity), permuted-label (specificity), planted-confound."""
    rng = np.random.default_rng(seed)
    base = {r["example_id"]: est[r["example_id"]]["C_exon"] for r in arch}
    gid = {r["example_id"]: r["gene"] for r in arch}
    pid = [(arch[i]["example_id"], arch[j]["example_id"]) for i, j in pairs]

    def reject(diffs, ga, gb, s):
        _, lo, hi = dyadic_bootstrap(diffs, ga, gb, n_boot=400, seed=s, alpha=ALPHA)
        return not (lo <= 0 <= hi)

    ga = [gid[t] for t, _ in pid]
    gb = [gid[c] for _, c in pid]
    raw = np.array([base[t] - base[c] for t, c in pid], float)

    # 1. SENSITIVITY: plant exactly +SESOI on a NULL-CENTRED cohort with the real
    #    dispersion.  Adding SESOI to `raw` as it stands would test detection of
    #    (true effect + SESOI), which is not the quantity the power claim is about.
    centred = raw - np.nanmean(raw)
    hits = sum(reject(centred + sesoi, ga, gb, 1000 + b) for b in range(args_nsim()))
    sens = hits / args_nsim()

    # 2. SPECIFICITY: within-pair sign flip -- the exact randomization null of a matched
    #    design.  NOT re-centred: subtracting the flipped mean would force the statistic to
    #    exactly zero, so the gate could never fire and would validate nothing (R21).
    fp = 0
    fp_wild = 0
    null_p = []
    for b in range(args_nsim()):
        s = rng.choice([-1.0, 1.0], size=len(centred))
        fp += reject(centred * s, ga, gb, 2000 + b)
        # frozen fallback (prereg §5 inference-selection rule): two-way wild sign-flip
        fp_wild += twoway_wild_p(centred * s, ga, gb, n_perm=400, seed=4000 + b) <= ALPHA
        # Amendment A1: the null distribution of the bootstrap p itself, so the critical
        # value can be CALIBRATED to this design's sign-flip null rather than assumed.
        null_p.append(dyadic_bootstrap_p(centred * s, ga, gb, n_boot=400, seed=6000 + b))
    spec_fpr = fp / args_nsim()
    spec_fpr_wild = fp_wild / args_nsim()
    p_star = float(np.percentile(null_p, 100 * ALPHA))

    # 3. CONFOUND-SPECIFICITY: an outcome driven PURELY by gc_intron.  Beta is set so the
    #    UNMATCHED estimator rejects; the MATCHED one must not.
    gc = {r["example_id"]: r["gc_intron"] for r in arch}
    dn_l = [r for r in arch if r["dn_band"] == "L"]
    dn_s = [r for r in arch if r["dn_band"] == "S"]
    gap = np.nanmean([gc[r["example_id"]] for r in dn_s]) - \
        np.nanmean([gc[r["example_id"]] for r in dn_l])
    beta = (3.0 * sesoi / gap) if abs(gap) > 1e-9 else 0.0
    noise = np.nanstd(raw) / np.sqrt(2) if np.isfinite(np.nanstd(raw)) else 1.0
    synth = {e: beta * gc[e] + rng.normal(0, noise) for e in gc}
    unmatched = np.nanmean([synth[r["example_id"]] for r in dn_l]) - \
        np.nanmean([synth[r["example_id"]] for r in dn_s])
    conf_hits = 0
    for b in range(args_nsim()):
        sy = {e: beta * gc[e] + rng.normal(0, noise) for e in gc}
        conf_hits += reject(np.array([sy[t] - sy[c] for t, c in pid]), ga, gb, 3000 + b)
    return dict(
        sensitivity_decision=dict(power=sens, target=0.80, PASS=sens >= 0.80,
                                  planted=sesoi),
        permuted_label_fpr=dict(
            fpr=spec_fpr, target=0.05, method="dyadic_multiplier_bootstrap",
            # Amendment A1 (outcome-blind, TRAIN): a heavy-tailed statistic leaves the
            # nominal interval slightly anti-conservative.  Over-rejection INFLATES a
            # positive claim but makes a NULL conclusion conservative, so the gate is
            # graded rather than binary: <=0.05 clean; <=0.10 usable with the calibrated
            # threshold p* applied to any positive verdict; >0.10 stops the run.
            PASS=spec_fpr <= 0.10, clean=spec_fpr <= 0.05,
            calibrated_p_star=p_star,
            note="positive verdicts must additionally clear p <= p_star when fpr > 0.05"),
        permuted_label_fpr_wild=dict(fpr=spec_fpr_wild, target=0.05,
                                     PASS=spec_fpr_wild <= 0.05,
                                     method="twoway_wild_signflip",
                                     note="frozen fallback (prereg 5); reported, gates only if the bootstrap fails"),
        planted_confound_null=dict(rate=conf_hits / args_nsim(), target=0.10,
                                   PASS=conf_hits / args_nsim() <= 0.10,
                                   beta=float(beta), unmatched_effect=float(unmatched),
                                   gc_gap=float(gap)),
    )


_NSIM = 100


def args_nsim():
    return _NSIM


# ========================= analysis ======================================================

def analyse(rows, memo, sesoi, floor, which=0, seed=0, calibrate_gates=True):
    est = {r["example_id"]: per_example(r, which) for r in rows}
    arch, pairs, bal, worst = build_matched(rows, "down")
    idx = {r["example_id"]: r for r in arch}

    E = contrast(arch, pairs, est, "C_exon", seed=seed)
    I = contrast(arch, pairs, est, "C_intron", seed=seed)
    dd_d = [e - i for e, i in zip(E["diffs"], I["diffs"])]
    m, lo, hi = dyadic_bootstrap(dd_d, E["gene_a"], E["gene_b"], n_boot=N_BOOT, seed=seed)
    DD = dict(mean=m, lo=lo, hi=hi, n_pairs=len(dd_d), signflip_p=signflip_p(dd_d, seed=seed))

    # ---- memorials -------------------------------------------------------------------
    cs = [est[r["example_id"]]["C_intron"] for r in arch if r["dn_band"] == "S"]
    ncp = float(np.nanmean(cs)) if cs else float("nan")
    mem = {
        "variant_construction_exact": dict(
            n_fail=len(memo["variant_construction_exact"]),
            examples=memo["variant_construction_exact"][:5],
            PASS=not memo["variant_construction_exact"], forced=False),
        "readout_anchor": dict(n_fail=len(memo["readout_anchor"]),
                               PASS=not memo["readout_anchor"], forced=False),
        "batch_invariance": dict(max_abs=memo.get("batch_invariance_max_abs"),
                                 PASS=(memo.get("batch_invariance_max_abs", 1) < 1e-4),
                                 forced=False),
        "native_coupling_present": dict(C_intron_down_short=ncp, floor=floor,
                                        PASS=bool(ncp >= floor), forced=False),
        # evaluated on the ADJUSTMENT SET only; the `_density_*` entries are exposure
        # descendants, reported as diagnostics and deliberately NOT gated (M4')
        "balance_achieved": dict(worst_smd=worst, table=bal,
                                 PASS=bool(worst < BALANCE_MAX_SMD), forced=False),
        "identity_noop_exact": dict(n_fail=len(memo["identity_noop"]),
                                    PASS=not memo["identity_noop"], forced=True,
                                    note="FORCED by construction: determinism/plumbing only (R21)"),
        "did_regrouping_identity": dict(
            max_resid=float(np.nanmax([est[e]["_regroup_resid"] for e in est])),
            PASS=bool(np.nanmax([est[e]["_regroup_resid"] for e in est]) < 1e-9),
            forced=True, note="FORCED by linear-f algebra; validates nothing on its own"),
    }
    cal = calibrate(arch, pairs, est, sesoi, seed=seed) if calibrate_gates else {}
    gates_pass = (all(v["PASS"] for v in mem.values())
                  and all(v["PASS"] for k, v in cal.items()
                          if k != "permuted_label_fpr_wild"))

    # ---- verdict (prereg §6; IUT) ----------------------------------------------------
    # Amendment A1: when the permuted-label FPR exceeds the nominal 0.05 (heavy tails),
    # a POSITIVE verdict must additionally clear the null-calibrated threshold p*.  A NULL
    # verdict needs no such correction -- an over-rejecting test that still fails to reject
    # makes the null conclusion conservative, not fragile.
    fpr_block = cal.get("permuted_label_fpr", {})
    p_star = fpr_block.get("calibrated_p_star")
    need_cal = bool(fpr_block) and not fpr_block.get("clean", True)
    dd_p = dyadic_bootstrap_p([e - i for e, i in zip(E["diffs"], I["diffs"])],
                              E["gene_a"], E["gene_b"], n_boot=N_BOOT, seed=seed)
    e_p = dyadic_bootstrap_p(E["diffs"], E["gene_a"], E["gene_b"], n_boot=N_BOOT, seed=seed)
    DD["boot_p"], E["boot_p"] = dd_p, e_p
    DD["calibrated_p_star"] = E["calibrated_p_star"] = p_star

    dd_sig = not (DD["lo"] <= 0 <= DD["hi"])
    e_sig = not (E["lo"] <= 0 <= E["hi"])
    if need_cal and p_star is not None and np.isfinite(p_star):
        dd_sig = dd_sig and dd_p <= p_star
        e_sig = e_sig and e_p <= p_star
    if not gates_pass:
        verdict = "GATES_FAILED -- no interpretation"
    elif dd_sig and DD["mean"] < 0:
        verdict = "REVERSE_SWITCH (falsifies the predicted direction)"
    elif dd_sig and e_sig and DD["mean"] > 0 and E["mean"] > 0 and abs(DD["mean"]) >= sesoi:
        verdict = "ARCHITECTURE_CONDITIONED_BOUNDARY_PAIRING"
    elif dd_sig and e_sig and DD["mean"] > 0 and E["mean"] > 0:
        verdict = "SMALL_BUT_DETECTABLE (sub-SESOI; not a mode switch)"
    elif dd_sig and DD["mean"] > 0 and not e_sig:
        verdict = "NOT_LICENSED_AS_SWITCH (carried by the distance-entangled cross-intron half)"
    elif (not dd_sig) and DD["lo"] >= -sesoi and DD["hi"] <= sesoi:
        verdict = "BOUNDED_NULL (no switch >= SESOI; analogy bounded)"
    else:
        verdict = "INCONCLUSIVE (underpowered)"

    # ---- secondaries ------------------------------------------------------------------
    sec = {}
    for fld in ("S_exon", "S_intron", "G_partner", "G_sham", "G_specific"):
        c = contrast(arch, pairs, est, fld, seed=seed)
        sec[f"switch_{fld}"] = {k: c[k] for k in ("mean", "lo", "hi", "n_pairs")}
    sec["dd_sham"] = _dd(arch, pairs, est, "C_exon_sham", "C_intron_sham", seed)
    sec["cells"] = {c: {f: _ci([est[r["example_id"]][f] for r in arch if r["cell"] == c],
                               [r["gene"] for r in arch if r["cell"] == c], seed)
                        for f in ("C_exon", "C_intron", "S_exon", "S_intron",
                                  "G_partner", "G_specific")}
                    for c in ("SS", "SL", "LS", "LL")}
    by_up: dict = {}
    for n, (i, j) in enumerate(pairs):
        by_up.setdefault(idx[arch[i]["example_id"]]["up_band"], []).append(n)
    sec["primary_by_upstream_band"] = {
        b: dict(n=len(ix),
                E_switch=float(np.mean([E["diffs"][k] for k in ix])),
                dd=float(np.mean([dd_d[k] for k in ix]))) for b, ix in by_up.items()}

    # D1 — controlled DIRECT effect holding in-window site density fixed (red-team M4').
    # This is what adjudicates "partner-specific switch" vs "generic dilution by competing
    # sites"; the primary above deliberately leaves density free because it descends from
    # the exposure.
    a1, p1, b1, w1 = build_matched(rows, "down", density_matched=True)
    E1 = contrast(a1, p1, est, "C_exon", seed=seed)
    I1 = contrast(a1, p1, est, "C_intron", seed=seed)
    dd1 = [e - i for e, i in zip(E1["diffs"], I1["diffs"])]
    m1, lo1, hi1 = dyadic_bootstrap(dd1, E1["gene_a"], E1["gene_b"], n_boot=N_BOOT, seed=seed)
    sec["density_matched_direct_effect"] = dict(
        n_pairs=len(p1), worst_smd=w1, balanced=bool(w1 < BALANCE_MAX_SMD),
        n_genes=len({r["gene"] for i, j in p1 for r in (a1[i], a1[j])}),
        E_switch={k: E1[k] for k in ("mean", "lo", "hi", "signflip_p")},
        dd=dict(mean=m1, lo=lo1, hi=hi1),
        note="controlled direct effect (site density held fixed); density descends from "
             "the exposure, so this is NOT the total effect")

    # S2 upstream axis
    a2, p2, b2, w2 = build_matched(rows, "up")
    sec["upstream_axis"] = dict(
        n_pairs=len(p2), worst_smd=w2,
        C_exon=contrast(a2, p2, est, "C_exon",
                        treated_is_L=(a2[p2[0][0]]["up_band"] == "L") if p2 else True,
                        seed=seed))
    for k in ("diffs", "gene_a", "gene_b"):
        sec["upstream_axis"]["C_exon"].pop(k, None)

    # S4 distance profile — C_exon vs L_exon and C_intron vs L_down over the FULL scored
    # range including the 201-299 buffer.  Descriptive only: is there a transition near the
    # classical 200-250 nt in the cross-intron curve that the cross-exon curve lacks?
    BINS = [(20, 60), (60, 100), (100, 140), (140, 200), (200, 250), (250, 300),
            (300, 400), (400, 550), (550, 900)]
    prof = {}
    for lo_, hi_ in BINS:
        ex = [est[r["example_id"]]["C_exon"] for r in rows if lo_ <= r["L_exon"] < hi_]
        it = [est[r["example_id"]]["C_intron"] for r in rows if lo_ <= (r["L_down"] or -1) < hi_]
        gx = [r["gene"] for r in rows if lo_ <= r["L_exon"] < hi_]
        gi = [r["gene"] for r in rows if lo_ <= (r["L_down"] or -1) < hi_]
        prof[f"{lo_}-{hi_}"] = dict(
            C_exon=_ci(ex, gx, seed) if len(ex) >= 25 else None,
            C_intron=_ci(it, gi, seed) if len(it) >= 25 else None)
    sec["distance_profile"] = prof

    # S6 within-gene robustness — genes contributing BOTH a down-SHORT and a down-LONG
    # unit, so gene-level expression / isochore context is held exactly fixed.
    per_gene: dict = {}
    for r in arch:
        per_gene.setdefault(r["gene"], {"S": [], "L": []})[r["dn_band"]].append(
            est[r["example_id"]])
    wg_e, wg_d, wg_g = [], [], []
    for g, v in per_gene.items():
        if not (v["S"] and v["L"]):
            continue
        de = np.nanmean([x["C_exon"] for x in v["L"]]) - np.nanmean([x["C_exon"] for x in v["S"]])
        di = np.nanmean([x["C_intron"] for x in v["L"]]) - np.nanmean([x["C_intron"] for x in v["S"]])
        wg_e.append(de)
        wg_d.append(de - di)
        wg_g.append(g)
    sec["within_gene"] = dict(
        n_genes=len(wg_g),
        E_switch=dict(zip(("mean", "lo", "hi"), dyadic_bootstrap(wg_e, wg_g, wg_g,
                                                                 n_boot=N_BOOT, seed=seed))),
        dd=dict(zip(("mean", "lo", "hi"), dyadic_bootstrap(wg_d, wg_g, wg_g,
                                                           n_boot=N_BOOT, seed=seed))),
        note="same gene contributes both arms; not covariate-matched, so it trades "
             "matching for exact gene-context control")

    # S7 sampling stability over 3 disjoint gene blocks
    genes = sorted({r["gene"] for r in arch})
    blk = {g: int(hashlib.sha256(g.encode()).hexdigest(), 16) % 3 for g in genes}
    sec["stability_blocks"] = {}
    for b in range(3):
        ix = [n for n, (i, _) in enumerate(pairs) if blk[arch[i]["gene"]] == b]
        if len(ix) > 30:
            sec["stability_blocks"][str(b)] = dict(
                n=len(ix), E_switch=float(np.mean([E["diffs"][k] for k in ix])),
                dd=float(np.mean([dd_d[k] for k in ix])))
    sds = [v["dd"] for v in sec["stability_blocks"].values()]
    ses = [v["E_switch"] for v in sec["stability_blocks"].values()]
    sec["stability_sd_dd"] = float(np.std(sds, ddof=1)) if len(sds) > 1 else float("nan")
    # E_switch is the statistic the verdict actually turns on (dd can be 6x SESOI, which
    # makes an absolute SD trigger fire trivially), so its stability is reported too.
    sec["stability_sd_E_switch"] = float(np.std(ses, ddof=1)) if len(ses) > 1 else float("nan")
    sec["stability_trigger"] = bool(sec["stability_sd_E_switch"] > 0.5 * sesoi)

    # S3/M3 tail concentration + robustness
    def _wins(v):
        v = np.asarray(v, float)
        if v.size < 20:
            return float("nan")
        lo_, hi_ = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
        return float(np.nanmean(np.clip(v, lo_, hi_)))

    ed = np.array(E["diffs"], float)
    if ed.size >= 20 and np.nansum(ed) != 0:
        q = np.nanpercentile(np.abs(ed), 90)
        sec["tail_share_top_decile"] = float(np.nansum(ed[np.abs(ed) >= q]) / np.nansum(ed))
    else:
        sec["tail_share_top_decile"] = float("nan")
    sec["E_switch_winsorized"] = _wins(ed)
    sec["dd_winsorized"] = _wins(dd_d)
    sec["within_gene_pairs"] = int(sum(1 for i, j in pairs
                                       if arch[i]["gene"] == arch[j]["gene"]))
    sec["n_genes_matched"] = len({r["gene"] for i, j in pairs
                                  for r in (arch[i], arch[j])})
    sec["achieved_mde"] = float(2.8 * np.nanstd(ed, ddof=1) / np.sqrt(len(ed))) \
        if ed.size > 1 else float("nan")
    by_gene: dict = {}
    for r in arch:
        by_gene.setdefault(r["gene"], []).append(est[r["example_id"]]["C_exon"])
    gm = [np.nanmean(v) for v in by_gene.values()]
    sec["sd_between_C_exon"] = float(np.nanstd(gm, ddof=1)) if len(gm) > 1 else float("nan")

    return dict(
        observable=("linear", "max")[which], sesoi=sesoi, verdict=verdict,
        gates_pass=bool(gates_pass),
        primary=dict(dd=DD, E_switch={k: E[k] for k in ("mean", "lo", "hi", "n_pairs", "signflip_p")},
                     I_switch={k: I[k] for k in ("mean", "lo", "hi", "n_pairs", "signflip_p")},
                     IUT="licensed only if BOTH dd and E_switch reject with the predicted sign"),
        memorials=mem, calibration=cal, secondary=sec,
        cohort=dict(n_units=len(rows), n_in_2x2=len(arch), n_pairs=len(pairs),
                    cells=dict(Counter(r["cell"] for r in rows))))


def _ci(vals, genes, seed):
    m, lo, hi = dyadic_bootstrap(vals, genes, genes, n_boot=800, seed=seed)
    return dict(mean=m, lo=lo, hi=hi, n=int(np.sum(~np.isnan(np.asarray(vals, float)))))


def _dd(arch, pairs, est, fa, fb, seed):
    a = contrast(arch, pairs, est, fa, seed=seed)
    b = contrast(arch, pairs, est, fb, seed=seed)
    d = [x - y for x, y in zip(a["diffs"], b["diffs"])]
    m, lo, hi = dyadic_bootstrap(d, a["gene_a"], a["gene_b"], n_boot=N_BOOT, seed=seed)
    return dict(mean=m, lo=lo, hi=hi)


# ========================= main ==========================================================

def main() -> None:
    global _NSIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sesoi", type=float, default=SESOI)
    ap.add_argument("--floor", type=float, default=NATIVE_COUPLING_FLOOR)
    ap.add_argument("--nsim", type=int, default=100)
    ap.add_argument("--from-scores", default=None)
    ap.add_argument("--gtf", default="/Users/cberg/Research/data/genomes/gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--pwm", default="/Users/cberg/Research/projects/gfm-circuit-discovery/"
                                     "colab_results/splicebert_run/baselines/local_pair_baseline.pkl")
    ap.add_argument("--model", default="/Users/cberg/Research/projects/gfm-circuit-discovery/data/"
                                       "models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--overwrite-record", action="store_true",
                    help="replace an existing artifact of record (results are immutable by default)")
    args = ap.parse_args()
    _NSIM = args.nsim

    if args.smoke:
        args.split, args.limit = "train", args.limit or 40
        print("=== SMOKE: memorials + decision-layer calibration on TRAIN; VAL SEALED ===")
    if args.split == "test":
        raise SystemExit("chr8/chr9 are SEALED for this study; refusing to score them.")

    cfg = CouplingConfig()
    if args.from_scores:
        blob = json.loads(Path(args.from_scores).read_text())
        rows, memo = blob["rows"], blob["memorials_raw"]
    else:
        rows, memo = score_split(args, cfg)
        tag = args.tag or ("smoke" if args.smoke else args.split)
        sp = REPO / f"results/splicebert_coupling/development/ed1_scores_{tag}.json"
        sp = guard_output_path(sp, overwrite=args.overwrite_record)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(dict(rows=rows, memorials_raw=memo), indent=1))
        print(f"scores -> {sp}")

    res = analyse(rows, memo, args.sesoi, args.floor, which=0)
    if not args.smoke:
        # compute ONCE -- a dict comprehension over the keys would re-run the whole
        # analysis (matching included) per key
        alt = analyse(rows, memo, args.sesoi, args.floor, which=1, calibrate_gates=False)
        res["secondary"]["max_margin_readout"] = {
            k: alt["primary"][k] for k in ("dd", "E_switch")}
    res["provenance"] = dict(
        split=args.split, smoke=bool(args.smoke), limit=args.limit,
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        model=args.model, gtf=args.gtf, sham=SHAM.__dict__,
        arch_bands=dict(short_max=ARCH_SHORT_MAX, long_min=ARCH_LONG_MIN),
        sesoi=args.sesoi, native_coupling_floor=args.floor, n_boot=N_BOOT, alpha=ALPHA,
        prereg="studies/splicebert_partner_selection/preregistrations/"
               "PREREGISTRATION_ED1_ARCHITECTURE_BOUNDARY_PAIRING.md")

    report(res, smoke=args.smoke)
    tag = args.tag or ("smoke" if args.smoke else args.split)
    # An explicit --tag always writes to development/, so a re-analysis can never silently
    # overwrite the confirmatory artifact of record.
    canonical = args.split == "val" and not args.smoke and not args.tag
    out = REPO / ("results/splicebert_coupling/ed1_architecture_boundary_pairing.json"
                  if canonical else
                  f"results/splicebert_coupling/development/ed1_{tag}.json")
    out = guard_output_path(out, overwrite=args.overwrite_record)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))

    from gfmcd.provenance import ExperimentManifest
    status = ("EXPLORATORY" if args.smoke else
              "PILOT" if args.split == "train" else "CONFIRMATORY")
    man = ExperimentManifest.create(
        "splicebert_coupling:ed1_architecture_boundary_pairing",
        [sys.argv[0], *sys.argv[1:]], repo=REPO, status=status,
        config=res["provenance"], seeds={"boot": 0, "matching": "deterministic (no RNG)"},
        notes=[
            "ED1: architecture x partner-direction interaction (exon vs intron definition). "
            "PRIMARY = intersection-union on {dd, E_switch}; dd alone cannot separate a switch "
            "from cross-intron distance decay because L_down IS the cross-intron separation.",
            "Inference: two-way dyadic multiplier bootstrap with ADDITIVE mean-zero gene "
            "multipliers (the Exp(1) weighted-mean variant over-rejects at FPR~0.19; see "
            "redteam B4). Site density is a descendant of the exposure and is NOT adjusted in "
            "the primary (redteam M4'); the D1 secondary holds it fixed.",
            f"verdict={res['verdict']}"])
    ck = Path(args.model) / "pytorch_model.bin"
    if ck.exists():
        man.add_input(str(ck))
    man.write(out.parent / f"{out.stem}.manifest.json")
    print(f"\nsaved -> {out}\nmanifest -> {out.parent / (out.stem + '.manifest.json')}")


def report(res, smoke=False):
    p, m, c, s = res["primary"], res["memorials"], res["calibration"], res["secondary"]
    print("\n" + "=" * 78)
    print(f"ED1  observable={res['observable']}  SESOI={res['sesoi']}  "
          f"units={res['cohort']['n_units']}  pairs={res['cohort']['n_pairs']}  "
          f"genes={s.get('n_genes_matched')}")
    print("-" * 78)
    print("MEMORIALS")
    for k, v in m.items():
        print(f"  {'PASS' if v['PASS'] else 'FAIL'}  {k:28s}"
              f"{'  [FORCED]' if v.get('forced') else ''}  "
              f"{ {kk: vv for kk, vv in v.items() if kk not in ('PASS','forced','note','table','examples')} }")
    print("CALIBRATION")
    for k, v in c.items():
        print(f"  {'PASS' if v['PASS'] else 'FAIL'}  {k:28s}  {v}")
    if smoke:
        print("=" * 78)
        print("SMOKE: no primary verdict computed.")
        return
    print("-" * 78)
    print("PRIMARY (intersection-union: BOTH must reject with the predicted sign)")
    for nm, v in (("dd  (arch x direction)", p["dd"]), ("E_switch (distance-clean)", p["E_switch"]),
                  ("I_switch (dist-entangled)", p["I_switch"])):
        sig = "*" if not (v["lo"] <= 0 <= v["hi"]) else " "
        bp = f"  boot_p={v['boot_p']:.4f}" if v.get("boot_p") is not None else ""
        print(f"  {sig} {nm:26s} {v['mean']:+.4f}  [{v['lo']:+.4f}, {v['hi']:+.4f}]"
              f"   signflip p={v['signflip_p']:.4f}{bp}")
    ps = p["dd"].get("calibrated_p_star")
    if ps is not None:
        print(f"  null-calibrated critical p* = {ps:.4f}"
              f"  (applied to POSITIVE verdicts only; nominal alpha={ALPHA})")
    print(f"  achieved MDE={s['achieved_mde']:.4f}   sd_between(C_exon)={s['sd_between_C_exon']:.4f}")
    print(f"  stability across 3 gene blocks: SD(E_switch)={s['stability_sd_E_switch']:.4f}"
          f"  SD(dd)={s['stability_sd_dd']:.4f}"
          f"{'   <-- INSTABILITY TRIGGER' if s['stability_trigger'] else ''}")
    print(f"  tail: top-decile share of E_switch = {s['tail_share_top_decile']:.3f}"
          f"   winsorized E_switch={s['E_switch_winsorized']:+.4f}  dd={s['dd_winsorized']:+.4f}")
    print(f"  specificity: dd_sham = {s['dd_sham']['mean']:+.4f} "
          f"[{s['dd_sham']['lo']:+.4f}, {s['dd_sham']['hi']:+.4f}]")
    dmd = s["density_matched_direct_effect"]
    print(f"  site-density rival (D1, controlled DIRECT effect, n={dmd['n_pairs']} pairs,"
          f" balanced={dmd['balanced']}):")
    print(f"      E_switch={dmd['E_switch']['mean']:+.4f} "
          f"[{dmd['E_switch']['lo']:+.4f},{dmd['E_switch']['hi']:+.4f}]   "
          f"dd={dmd['dd']['mean']:+.4f} [{dmd['dd']['lo']:+.4f},{dmd['dd']['hi']:+.4f}]")
    ua = s["upstream_axis"]["C_exon"]
    print(f"  upstream axis (clean, no distance entanglement): C_exon switch="
          f"{ua['mean']:+.4f} [{ua['lo']:+.4f},{ua['hi']:+.4f}]  n={s['upstream_axis']['n_pairs']}")
    for f in ("G_partner", "G_sham", "G_specific"):
        v = s[f"switch_{f}"]
        print(f"  arch-dependence of {f:11s} {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]")
    wg = s["within_gene"]
    print(f"  within-gene ({wg['n_genes']} genes contributing BOTH arms): "
          f"E_switch={wg['E_switch']['mean']:+.4f} "
          f"[{wg['E_switch']['lo']:+.4f},{wg['E_switch']['hi']:+.4f}]")
    print("-" * 78)
    print("DISTANCE PROFILE (S4, descriptive)")
    print(f"  {'separation':14s} {'C_exon vs L_exon':>24s} {'C_intron vs L_down':>24s}")
    for k, v in s["distance_profile"].items():
        f = lambda c: (f"{c['mean']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]" if c else "--")  # noqa: E731
        print(f"  {k:14s} {f(v['C_exon']):>24s} {f(v['C_intron']):>24s}")
    print("-" * 78)
    print("2x2 CELLS (mean [CI])")
    print(f"  {'cell':6s} {'C_exon':>22s} {'C_intron':>22s}")
    for cell in ("SS", "SL", "LS", "LL"):
        cc = s["cells"].get(cell, {})
        f = lambda k: (f"{cc[k]['mean']:+.3f} [{cc[k]['lo']:+.3f},{cc[k]['hi']:+.3f}]"  # noqa: E731
                       if k in cc else "--")
        print(f"  {cell:6s} {f('C_exon'):>22s} {f('C_intron'):>22s}   n={cc.get('C_exon',{}).get('n','-')}")
    print("-" * 78)
    print(f"VERDICT: {res['verdict']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
