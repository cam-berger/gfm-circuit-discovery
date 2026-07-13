#!/usr/bin/env python
"""THE PRIMARY TEST -- and the reason the simple ratio is not it.

`run_comb_shift.py` reports b_norm, the mod-3 sawtooth in  damage / ||dout||.  That statistic
silently assumes DAMAGE IS LINEAR IN PERTURBATION SIZE.  It is not necessarily, and if it is
convex the whole result is an artefact:

    damage ~ ||dout||^gamma   =>   damage / ||dout|| ~ ||dout||^(gamma - 1)

For a COMB, ||dout|| off-phase is ~2x its in-phase value (comb self-overlap -- guaranteed, see
the memorial tests).  So if gamma > 1, then damage/||dout|| is LARGER off-phase FOR THAT REASON
ALONE, with no phase computation anywhere.  And a COPIER, whose ||dout|| is FLAT in delta, shows
nothing -- reproducing "b_norm(COMB) >> b_norm(COPIER)" from pure convexity.

    The control would have fired, the algebra would have looked clean, and the answer would
    have been wrong.  This is the same shape as R7 and R14: a statistic that cannot help but
    produce the expected pattern.

SO: ESTIMATE THE EXPONENT INSTEAD OF ASSUMING IT.

    log D_hd  =  a_h  +  gamma * log N_hd  +  sum_k  b_k * offphase(d) * 1[kind(h) = k]  +  eps

    a_h     head fixed effect  -- each head's own damage scale and downstream leverage
    gamma   the damage-perturbation exponent, ESTIMATED FROM THE DATA (not assumed = 1)
    b_k     THE PHASE EFFECT for head-kind k: extra log-damage from reading OFF phase
            AT MATCHED PERTURBATION SIZE.

b_COMB - b_COPIER is the claim.  It asks: at the same perturbation magnitude, and after the
damage-vs-perturbation curve has been fitted rather than assumed, is a comb hurt MORE by an
off-phase read than a copier is?

    gamma ~ 1  -> the simple ratio was fine after all.
    gamma > 1  -> it was inflating the combs, and this correction is load-bearing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
KINDS = ("COMB", "COPIER", "DIFFUSE")


def design(rows, region, cls, drop_selfhit=True):
    """Build (y, X, head_index, colnames) for the log-log model.

    DROP THE SELF-HIT CELLS.  Self-attention is restored after the roll, so if a tooth is
    carried onto the query's OWN column (offset 0) its mass is overwritten and lost -- the head
    is left attending to nothing.  That is a huge, delta-specific perturbation, and for a copier
    at offset -3 it lands on delta = 3, an IN-PHASE delta.  Such (head, delta) cells are
    excluded, uniformly for every head kind, rather than left to contaminate the control.
    """
    heads = [r for r in rows if r[region]["healthy"]]
    hid = {(r["layer"], r["head"]): i for i, r in enumerate(heads)}
    y, rowsX, hrow = [], [], []
    dropped = 0
    for r in heads:
        h = hid[(r["layer"], r["head"])]
        teeth = cls[f"L{r['layer']}H{r['head']}"]["teeth"]
        for d_s, D in r[region]["damage"].items():
            d = int(d_s)
            N = r[region]["norm"][d_s]
            if drop_selfhit and any(t + d == 0 for t in teeth):
                dropped += 1
                continue
            if D <= 0 or N <= 0:          # log-log needs positive damage AND positive perturbation
                continue
            fe = [0.0] * len(heads)
            fe[h] = 1.0
            off = float(d % 3 != 0)
            kk = [off * float(r["kind"] == k) for k in KINDS]
            rowsX.append(fe + [np.log(N)] + kk)
            y.append(np.log(D))
            hrow.append(h)
    names = [f"head{i}" for i in range(len(heads))] + ["gamma"] + [f"b_{k}" for k in KINDS]
    return (np.array(y), np.array(rowsX), np.array(hrow), names, heads, dropped)


def fit(y, X):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/phase-lattice/comb_shift.json")
    ap.add_argument("--boot", type=int, default=4000)
    args = ap.parse_args()

    R = json.loads((REPO / args.inp).read_text())
    rows = R["heads"]
    print("=" * 96)
    print("PRIMARY TEST -- phase effect at MATCHED perturbation, with the damage-perturbation")
    print("exponent ESTIMATED, not assumed.")
    print(f"  n_exons={R['n_exons']}  combs={R['n_comb']}  copiers={R['n_copier']}  "
          f"diffuse={R['n_diffuse']}")

    cls = R["classification"]
    out: dict = {}
    for region in ("exon", "intron"):
        y, X, hrow, names, heads, dropped = design(rows, region, cls)
        nh = len(heads)
        if dropped:
            print(f"  [{region}] dropped {dropped} (head,delta) cells where a tooth lands on the "
                  f"query's own column (self-hit)")
        coef = fit(y, X)
        gamma = float(coef[nh])
        b = {k: float(coef[nh + 1 + i]) for i, k in enumerate(KINDS)}

        # bootstrap over HEADS (the unit the claim is about); refit inside every replicate
        rng = np.random.default_rng(0)
        bs: dict = {k: [] for k in KINDS}
        bs["diff"] = []
        bs["gamma"] = []
        for _ in range(args.boot):
            pick = rng.choice(nh, nh, replace=True)
            sel = np.concatenate([np.flatnonzero(hrow == h) for h in pick])
            try:
                c = fit(y[sel], X[sel])
            except np.linalg.LinAlgError:
                continue
            bs["gamma"].append(float(c[nh]))
            bb = {k: float(c[nh + 1 + i]) for i, k in enumerate(KINDS)}
            for k in KINDS:
                bs[k].append(bb[k])
            bs["diff"].append(bb["COMB"] - bb["COPIER"])

        def ci(v):
            return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))

        gl, gh = ci(bs["gamma"])
        print(f"\n  [{region}]  n = {len(y)} (head x delta) points, {nh} heads")
        print(f"    gamma (damage ~ ||dout||^gamma) = {gamma:.3f}  95% CI [{gl:.3f}, {gh:.3f}]")
        if gl > 1.0:
            print("      => CONVEX. The simple damage/||dout|| ratio WAS inflating the combs.")
            print("         This correction is load-bearing.")
        elif gh < 1.0:
            print("      => CONCAVE (saturating). The ratio was DEFLATING the combs.")
        else:
            print("      => ~linear. The simple ratio was defensible; this confirms it.")

        print(f"\n    phase effect b (extra log-damage from reading OFF phase, "
              f"at MATCHED perturbation):")
        for k in KINDS:
            lo, hi = ci(bs[k])
            n_k = sum(1 for h in heads if h["kind"] == k)
            star = "  <-- excludes 0" if lo > 0 or hi < 0 else ""
            print(f"      b_{k:8s} = {b[k]:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]   "
                  f"n={n_k}{star}")

        dlo, dhi = ci(bs["diff"])
        d = b["COMB"] - b["COPIER"]
        p = float((1 + np.sum(np.array(bs["diff"]) <= 0)) / (1 + len(bs["diff"])))
        print(f"\n    THE CLAIM:  b_COMB - b_COPIER = {d:+.4f}  95% CI [{dlo:+.4f}, {dhi:+.4f}]"
              f"   boot p(<=0) = {p:.4f}")

        # ---- IDENTIFICATION DIAGNOSTIC ------------------------------------------------
        # gamma is identified almost entirely from the COMBS (the only heads whose ||dout||
        # moves with delta).  But within a comb, ||dout|| is ITSELF bigger off-phase -- so
        # log N and the off-phase indicator are correlated and gamma can trade off against
        # b_COMB.  If they are near-collinear, b_COMB is not trustworthy and we say so.
        cm = [i for i, h in enumerate(hrow) if heads[h]["kind"] == "COMB"]
        if cm:
            Z = np.column_stack([X[cm, nh], X[cm, nh + 1]])       # [log N, offphase*1(comb)]
            Zc = Z - Z.mean(0)
            r = float(np.corrcoef(Zc[:, 0], Zc[:, 1])[0, 1])
            cond = float(np.linalg.cond(np.column_stack([np.ones(len(cm)), Z])))
            print(f"\n    identification: within COMBS, corr(log||dout||, off-phase) = {r:+.3f}"
                  f"   cond = {cond:.1f}")
            if abs(r) > 0.95:
                print("      *** NEAR-COLLINEAR. gamma and b_COMB are not separately identified.")
                print("          Fall back to the fixed-gamma statistic below and say so. ***")

        # ---- ASSUMPTION-LIGHT FALLBACK (gamma fixed at 1) -----------------------------
        # s_h = [mean log D off - mean log D in] - [mean log N off - mean log N in]
        # i.e. the log-damage sawtooth MINUS the log-perturbation sawtooth. This is exactly
        # damage/||dout|| in logs, needs no fit, and cannot be destabilised by collinearity.
        s: dict = {k: [] for k in KINDS}
        for r_ in heads:
            teeth = cls[f"L{r_['layer']}H{r_['head']}"]["teeth"]
            D_, N_ = r_[region]["damage"], r_[region]["norm"]
            ok_d = [int(k) for k in D_ if D_[k] > 0 and N_[k] > 0
                    and not any(t + int(k) == 0 for t in teeth)]
            ip = [dd for dd in ok_d if dd % 3 == 0]
            op = [dd for dd in ok_d if dd % 3]
            if not ip or not op:
                continue
            lD = np.mean([np.log(D_[str(dd)]) for dd in op]) - np.mean([np.log(D_[str(dd)]) for dd in ip])
            lN = np.mean([np.log(N_[str(dd)]) for dd in op]) - np.mean([np.log(N_[str(dd)]) for dd in ip])
            s[r_["kind"]].append(float(lD - lN))
        print(f"\n    fixed-gamma=1 fallback  (log-damage sawtooth MINUS log-perturbation sawtooth):")
        for k in KINDS:
            if s[k]:
                print(f"      {k:8s} = {np.mean(s[k]):+.4f}  (n={len(s[k])})")
        s_diff = (np.mean(s["COMB"]) - np.mean(s["COPIER"])) if s["COMB"] and s["COPIER"] else float("nan")
        if s["COMB"] and s["COPIER"]:
            pool = np.array(s["COMB"] + s["COPIER"])
            nA = len(s["COMB"])
            pr = np.array([np.mean(x[:nA]) - np.mean(x[nA:])
                           for x in (rng.permutation(pool) for _ in range(20000))])
            pp = float((1 + np.sum(pr >= s_diff)) / (1 + len(pr)))
            print(f"      COMB - COPIER = {s_diff:+.4f}   exact permutation p = {pp:.4f}")

        out[region] = dict(gamma=gamma, gamma_ci=[gl, gh], b=b,
                           b_ci={k: list(ci(bs[k])) for k in KINDS},
                           diff=d, diff_ci=[dlo, dhi], p=p, n_heads=nh, n_points=int(len(y)),
                           fixed_gamma={k: (float(np.mean(s[k])) if s[k] else None) for k in KINDS},
                           fixed_gamma_diff=float(s_diff) if s_diff == s_diff else None)

    print("\n" + "=" * 96)
    e, i = out["exon"], out["intron"]
    print("VERDICT")
    if e["diff_ci"][0] > 0 and i["diff_ci"][0] <= 0:
        print("  The comb is USED, and specifically on CODING sequence.")
        print(f"    exon:   b_COMB - b_COPIER = {e['diff']:+.4f} [{e['diff_ci'][0]:+.4f}, "
              f"{e['diff_ci'][1]:+.4f}]  (excludes 0)")
        print(f"    intron: b_COMB - b_COPIER = {i['diff']:+.4f} [{i['diff_ci'][0]:+.4f}, "
              f"{i['diff_ci'][1]:+.4f}]  (does not)")
        print("  At MATCHED perturbation size, with the damage curve FITTED, a comb rolled off")
        print("  phase is hurt more than a copier is -- and only where codons exist.")
        print("  A copier cannot produce this: its perturbation is FLAT in delta, so no")
        print("  comb-overlap artefact is available to it. And it scores the SAME on the phase")
        print("  metric the combs were selected by.")
    elif e["diff_ci"][0] > 0:
        print("  The comb is USED -- but the effect is NOT exon-specific (it is present in")
        print("  introns too). That reads as an INTERNAL POSITIONAL PROTOCOL, not a codon reader.")
    else:
        print("  NO EFFECT. Once the damage-perturbation curve is fitted rather than assumed,")
        print("  a comb rolled off phase is hurt no more than a copier is.")
        print("  The combs are real STRUCTURE with no demonstrated FUNCTION -- and unlike R13,")
        print("  this test had the power to say so: every arm kept the behaviour.")

    p = REPO / "results/phase-lattice/comb_shift_primary.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
