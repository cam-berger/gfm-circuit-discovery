"""Caduceus-PS T1 harness: constructs -> margins -> estimands -> stats -> frozen decision (prereg v2).

Model-agnostic: takes any object with `donor_margin(seqs, positions)` + `mlm_health(seqs)` (CaduceusScorer on
CUDA, or MockCaduceus locally). Consumes a pre-built construct set (built LOCALLY by build_cohort.py so the GPU
only does forward passes). Implements: the 4 IUT gates (existence C̄, partner-specificity, donor-specificity,
ρ_partner≥τ), Fieller ratio CIs, paired gene-cluster bootstrap, joint-power certification, the long-range
transmission certification, and the §6 decision (QUALIFIES / DOES-NOT-QUALIFY / INSTRUMENT-LIMIT / UNDER-POWERED).

A "construct" (dict): {seq, pos, tag, ex_id, gene, chrom, split, delta?}. tag ∈
  {clean, partner_cc, donor_dead, nonpartner_cc, nondonor(other_gt)_cc, score3_cc, permuted_cc, nonsite,
   lr_block@<d>}. Each example contributes one value per tag; estimands are within-example differences.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np

Z = 1.959964


# ---------- margins (the ONLY model-dependent step; batched by equal length) ----------
def run_margins(scorer, constructs):
    """Return {construct_index: donor_margin}. All seqs are 1024 nt => one equal-length batch stream."""
    seqs = [c["seq"] for c in constructs]; pos = [c["pos"] for c in constructs]
    m = scorer.donor_margin(seqs, pos)
    return np.asarray(m, float)


# ---------- gene-clustered inference ----------
def gene_means(values, genes):
    """mean per gene -> array of gene means (the clustering unit)."""
    d = defaultdict(list)
    for v, g in zip(values, genes):
        d[g].append(v)
    keys = sorted(d)
    return np.array([np.mean(d[k]) for k in keys]), keys


def bca_ci(x, B=4000, alpha=0.05, seed=0):
    """BCa CI of the mean of gene-level values x."""
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.array([np.mean(rng.choice(x, n, replace=True)) for _ in range(B)])
    th = x.mean()
    z0 = _z(np.mean(boot < th))
    jk = np.array([np.mean(np.delete(x, i)) for i in range(n)])
    jbar = jk.mean(); num = ((jbar - jk) ** 3).sum(); den = 6 * (((jbar - jk) ** 2).sum() ** 1.5)
    a = num / den if den else 0.0
    lo = _cdf(z0 + (z0 + _q(alpha / 2)) / (1 - a * (z0 + _q(alpha / 2))))
    hi = _cdf(z0 + (z0 + _q(1 - alpha / 2)) / (1 - a * (z0 + _q(1 - alpha / 2))))
    return (float(np.quantile(boot, lo)), float(np.quantile(boot, hi)))


def _q(p): from scipy.stats import norm; return float(norm.ppf(p))
def _z(p): from scipy.stats import norm; return float(norm.ppf(min(max(p, 1e-6), 1 - 1e-6)))
def _cdf(z): from scipy.stats import norm; return float(norm.cdf(z))


def paired_gene_bootstrap_ratio(num_per_ex, den_per_ex, genes, B=4000, alpha=0.05, seed=0):
    """Ratio-of-gene-means CI (Fieller + percentile), resampling GENES once and recomputing BOTH means.
    num_per_ex/den_per_ex are per-example arrays sharing the SAME gene labels (paired)."""
    num = np.asarray(num_per_ex, float); den = np.asarray(den_per_ex, float); genes = np.asarray(genes)
    gk = sorted(set(genes.tolist()))
    gidx = {g: np.where(genes == g)[0] for g in gk}
    gm_num = np.array([num[gidx[g]].mean() for g in gk]); gm_den = np.array([den[gidx[g]].mean() for g in gk])
    point = gm_num.mean() / gm_den.mean()
    rng = np.random.default_rng(seed); n = len(gk); reps = []
    for _ in range(B):
        take = rng.integers(0, n, n)
        reps.append(gm_num[take].mean() / gm_den[take].mean())
    reps = np.array(reps)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    den_lo = np.quantile([gm_den[rng.integers(0, n, n)].mean() for _ in range(B)], alpha / 2)
    # Fieller honesty: if the denominator CI approaches 0 the ratio CI is unstable -> flag
    return {"point": float(point), "ci": (float(lo), float(hi)), "den_lo": float(den_lo),
            "den_clear_of_zero": bool(den_lo > 0)}


def joint_power(anticipated, n_boot=4000, seed=0):
    """Joint power of the IUT conjunction: P(ALL arms CI-exclude-0 | a PRE-REGISTERED anticipated meaningful
    effect and each arm's SE). anticipated: dict arm->(effect, se). Effects are the anticipated MEANINGFUL
    magnitudes (NOT the observed confirm estimates), so a true null is not self-certified as under-powered."""
    rng = np.random.default_rng(seed); arms = list(anticipated)
    hits = 0
    for _ in range(n_boot):
        ok = True
        for a in arms:
            eff, se = anticipated[a]
            if se <= 0 or not (rng.normal(eff, se) - Z * se > 0):
                ok = False; break
        hits += ok
    return hits / n_boot


# ---------- the frozen decision ----------
def decide(ex_vals, genes, splits, tau, *, lr_curve=None, mlm_health=None, both_strands_ok=None,
           rho_anticipated=0.15, lr_frac=0.1, min_joint_power=0.8, seed=0):
    """ex_vals: dict tag -> per-example donor margin array (aligned to `genes`,`splits`).
    Runs on the CONFIRMATORY split. Returns the full decision record."""
    conf = np.array([s == "confirm" for s in splits])
    g = np.asarray(genes)[conf]
    def pe(tag):  # per-example margins on confirm
        return np.asarray(ex_vals[tag], float)[conf]

    clean = pe("clean")
    C = clean - pe("partner_cc")                       # coupling at the TRUE donor
    A_donor = clean - pe("donor_dead")                 # local recognition anchor
    C_np = clean - pe("nonpartner_cc")                 # vs distance-matched non-partner corruption
    C_nd = pe("clean_other") - pe("partner_cc_other")  # SAME partner corruption read at ANOTHER real donor GT (other_gt)
    partner_spec = C - C_np                            # partner specificity contrast (per example)
    donor_spec = C - C_nd                              # donor specificity contrast (per example)

    gm_C, _ = gene_means(C, g)
    gm_ps, _ = gene_means(partner_spec, g)
    gm_ds, _ = gene_means(donor_spec, g)
    gm_A, _ = gene_means(A_donor, g)

    ci_C = bca_ci(gm_C, seed=seed)
    ci_ps = bca_ci(gm_ps, seed=seed + 1)
    ci_ds = bca_ci(gm_ds, seed=seed + 2)
    # ρ_partner = mean(partner_spec) / mean(A_donor), paired gene bootstrap
    rho_p = paired_gene_bootstrap_ratio(partner_spec, A_donor, g, seed=seed + 3)

    n_genes = len(set(g.tolist()))
    # gates (existence = CI-excludes-0; the tautological ρ-CI gate is DROPPED, B4)
    gate_exist = ci_C[0] > 0
    gate_partner = ci_ps[0] > 0
    gate_donor = ci_ds[0] > 0
    gate_floor = rho_p["point"] >= tau                 # meaningfulness on ρ_partner (C1)
    anchor_ok = bca_ci(gm_A, seed=seed + 4)[0] > 0

    # long-range certification (E1/§4): VISIBILITY (near block large, CI-excludes-0) AND mid-range 60-150
    # CI-excludes-0 AND non-negligible vs near (>= lr_frac * near) — else instrument-range-limit.
    lr_ok = None
    if lr_curve is not None:
        pt = {d: (v[0] + v[1]) / 2 for d, v in lr_curve.items()}
        near_d = 10 if 10 in lr_curve else (30 if 30 in lr_curve else None)
        mids = [d for d in (60, 100, 150) if d in lr_curve]
        if near_d is not None and mids:
            lr_ok = bool(lr_curve[near_d][0] > 0 and pt[near_d] > 0
                         and all(lr_curve[d][0] > 0 for d in mids)
                         and all(pt[d] >= lr_frac * pt[near_d] for d in mids))

    # joint power (B1): at a PRE-REGISTERED anticipated MEANINGFUL effect (rho_anticipated * A_donor) on the
    # realized gene-n SEs — NOT the observed confirm effect (so a true null is not auto-labelled under-powered).
    se_C = (ci_C[1] - ci_C[0]) / (2 * Z); se_p = (ci_ps[1] - ci_ps[0]) / (2 * Z); se_d = (ci_ds[1] - ci_ds[0]) / (2 * Z)
    anticipated = rho_anticipated * float(np.mean(gm_A))
    jp = joint_power({"exist": (anticipated, se_C), "partner": (anticipated, se_p), "donor": (anticipated, se_d)}, seed=seed + 5)

    memorials_ok = anchor_ok and (mlm_health is None or mlm_health > 0) and (both_strands_ok is not False)
    verdict = _verdict(gate_exist, gate_partner, gate_donor, gate_floor, memorials_ok, lr_ok, jp, min_joint_power)
    return {
        "n_genes_confirm": n_genes,
        "C_mean": float(np.mean(gm_C)), "C_ci": ci_C,
        "partner_spec_mean": float(np.mean(gm_ps)), "partner_spec_ci": ci_ps,
        "donor_spec_mean": float(np.mean(gm_ds)), "donor_spec_ci": ci_ds,
        "A_donor_mean": float(np.mean(gm_A)),
        "rho_partner": rho_p["point"], "rho_partner_ci": rho_p["ci"], "tau": tau,
        "gates": {"exist": bool(gate_exist), "partner": bool(gate_partner), "donor": bool(gate_donor),
                  "floor_rho_ge_tau": bool(gate_floor), "anchor": bool(anchor_ok)},
        "long_range_certified": lr_ok, "joint_power": jp, "mlm_health": mlm_health,
        "VERDICT": verdict,
    }


def _verdict(exist, partner, donor, floor, memorials_ok, lr_ok, jp, min_jp):
    if not memorials_ok:
        return "INSTRUMENT-LIMIT (memorial/anchor/strand failed) — no biological conclusion"
    gates_pass = exist and partner and donor and floor
    if gates_pass and lr_ok is True:                                   # §6(f): long-range certification required
        return "QUALIFIES-FOR-PHASE-3 (behaviour-only; mechanism unproven)"
    if lr_ok is False:
        return "INSTRUMENT-RANGE-LIMIT (downstream->donor channel dead/negligible @60-150nt) — no biological claim"
    if gates_pass and lr_ok is None:
        return "INDETERMINATE — gates pass but long-range channel uncertified (missing transmission constructs)"
    if jp < min_jp:
        return f"UNDER-POWERED (joint power {jp:.2f} < {min_jp} at anticipated effect) — inconclusive, widen cohort"
    return "DOES-NOT-QUALIFY — honest architecture/objective null (controls pass, channel certified, powered)"
