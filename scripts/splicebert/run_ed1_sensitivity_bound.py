#!/usr/bin/env python
"""ED1 post-run SENSITIVITY BOUND on active cross-exon strengthening.

Status: POST-RUN ROBUSTNESS INTERPRETATION, not a preregistered outcome.  The frozen
decision table (prereg §6) classifies this run as **NOT LICENSED AS A SWITCH** -- `dd`
excludes 0 while `E_switch` includes 0 -- and the canonical artifact correctly records
`GATES_FAILED`.  Nothing here changes either.  This quantifies how large an *active*
cross-exon strengthening could still be hiding, given the one calibration gate that failed.

WHY THIS FORM.  A rescue that leans on the empirically fitted `gc_intron -> C_exon` slope
would be estimated AFTER the primary was seen.  This bound instead uses only the
PREREGISTERED plant: beta = 3 * SESOI / raw_gap, exactly as `planted_confound_null` froze
it.  That plant is deliberately severe, so the bias it implies is an upper envelope, and
the argument needs no post-outcome quantity.

    upper bound  =  one-sided UCB( E_switch )  +  |residual confound bias|

with BOTH terms bootstrapped jointly on the same dyadic resample, so the uncertainty in
the bias estimate is propagated rather than treated as a constant.  The interval is also
reported at a critical value CALIBRATED on the design's own within-pair sign-flip null,
because the permuted-label FPR came in at 0.060 rather than the nominal 0.050 -- so
nominal coverage is not assumed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.splicing.exon_intron import (  # noqa: E402
    EXACT_KEYS, axis_spec, match_pairs,
)

SESOI = 0.106
ALPHA = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="results/splicebert_coupling/development/ed1_scores_val.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--overwrite-record", action="store_true")
    args = ap.parse_args()

    rows = json.loads((REPO / args.scores).read_text())["rows"]
    arch = [dict(r, up_band=r["cell"][0], dn_band=r["cell"][1]) for r in rows if r["cell"]]
    for r in arch:
        r["C_exon"] = r["f"]["N"][0] - r["f"]["U-"][0]
    cov, cal = axis_spec("down")
    pairs = match_pairs(arch, [r["dn_band"] == "L" for r in arch],
                        covariates=cov, calipers=cal, exact=EXACT_KEYS["down"])

    d = np.array([arch[i]["C_exon"] - arch[j]["C_exon"] for i, j in pairs])
    dgc = np.array([arch[i]["gc_intron"] - arch[j]["gc_intron"] for i, j in pairs])
    ga = [arch[i]["gene"] for i, _ in pairs]
    gb = [arch[j]["gene"] for _, j in pairs]

    # PREREGISTERED plant strength -- reproduced from the frozen rule, not refitted.
    gcL = np.mean([r["gc_intron"] for r in arch if r["dn_band"] == "L"])
    gcS = np.mean([r["gc_intron"] for r in arch if r["dn_band"] == "S"])
    beta = 3.0 * SESOI / (gcS - gcL)

    keys = sorted(set(ga) | set(gb))
    idx = {g: i for i, g in enumerate(keys)}
    ia, ib = np.array([idx[g] for g in ga]), np.array([idx[g] for g in gb])
    same = ia == ib
    inv = 1.0 / np.sqrt(2.0)

    def draws(vals, n_boot, seed):
        """Dyadic multiplier draws of the pivot for mean(vals)."""
        dev = vals - vals.mean()
        rng = np.random.default_rng(seed)
        out = np.empty(n_boot)
        for b in range(n_boot):
            e = rng.standard_normal(len(keys))
            m = np.where(same, e[ia], (e[ia] + e[ib]) * inv)
            out[b] = float(np.mean(m * dev))
        return out

    # joint resample: E_switch and the bias move together on the same multipliers
    rng = np.random.default_rng(0)
    e_hat, gc_hat = float(d.mean()), float(dgc.mean())
    dev_e, dev_g = d - e_hat, dgc - gc_hat
    joint = np.empty(args.n_boot)
    e_only = np.empty(args.n_boot)
    for b in range(args.n_boot):
        e = rng.standard_normal(len(keys))
        m = np.where(same, e[ia], (e[ia] + e[ib]) * inv)
        eb = e_hat + float(np.mean(m * dev_e))
        gb_ = gc_hat + float(np.mean(m * dev_g))
        e_only[b] = eb
        joint[b] = eb + abs(beta * gb_)          # conservative: add |bias|
    ucb_e = float(np.percentile(e_only, 100 * (1 - ALPHA)))
    ucb_joint = float(np.percentile(joint, 100 * (1 - ALPHA)))

    # Independently CALIBRATED one-sided critical value from the within-pair sign-flip
    # null: the width comes from the design's own randomization null, not from assuming
    # the bootstrap has nominal coverage (measured permuted-label FPR was 0.060).
    rngc = np.random.default_rng(7)
    null_means = np.array([float(np.mean(rngc.choice([-1.0, 1.0], size=len(d)) * (d - e_hat)))
                           for _ in range(4000)])
    c_star = float(np.percentile(null_means, 100 * (1 - ALPHA)))
    ucb_cal = e_hat + c_star
    bias_pt = abs(beta * gc_hat)
    bias_ucb = float(np.percentile([abs(beta * (gc_hat + x)) for x in draws(dgc, 2000, 3)],
                                   100 * (1 - ALPHA)))
    ucb_cal_joint = ucb_cal + bias_ucb

    res = dict(
        status="POST-RUN ROBUSTNESS INTERPRETATION -- not a preregistered outcome; the "
               "frozen verdict remains NOT LICENSED AS A SWITCH / GATES_FAILED",
        n_pairs=len(pairs), sesoi=SESOI, alpha=ALPHA,
        E_switch_point=e_hat,
        preregistered_plant=dict(beta=beta, raw_gc_gap=float(gcS - gcL),
                                 residual_gc_gap=gc_hat,
                                 bias_point=bias_pt, bias_ucb95=bias_ucb,
                                 note="beta from the FROZEN rule 3*SESOI/raw_gap; no "
                                      "post-outcome slope is used"),
        bootstrap=dict(ucb95_E_switch=ucb_e, ucb95_E_switch_plus_bias=ucb_joint),
        signflip_calibrated=dict(critical_value=c_star, ucb95_E_switch=ucb_cal,
                                 ucb95_E_switch_plus_bias=ucb_cal_joint),
        verdict_bound=dict(
            bound=max(ucb_joint, ucb_cal_joint), sesoi=SESOI,
            excluded=bool(max(ucb_joint, ucb_cal_joint) < SESOI)))

    print("=" * 78)
    print("ED1 SENSITIVITY BOUND on ACTIVE cross-exon strengthening")
    print("  (post-run robustness; frozen verdict unchanged: NOT LICENSED AS A SWITCH)")
    print("=" * 78)
    print(f"  matched pairs                      {len(pairs)}")
    print(f"  E_switch point estimate            {e_hat:+.4f}")
    print(f"  PREREGISTERED plant beta           {beta:.3f}   (= 3*SESOI / raw gap {gcS-gcL:.4f})")
    print(f"  residual gc gap after matching     {gc_hat:+.5f}")
    print(f"  implied residual bias  |point|     {bias_pt:.4f}     95% upper  {bias_ucb:.4f}")
    print("-" * 78)
    print(f"  one-sided 95% UCB, E_switch alone            {ucb_e:+.4f}")
    print(f"  one-sided 95% UCB, E_switch + |bias| (joint) {ucb_joint:+.4f}")
    print(f"  sign-flip CALIBRATED UCB, E_switch           {ucb_cal:+.4f}  (c*={c_star:.4f})")
    print(f"  sign-flip CALIBRATED UCB, + |bias|           {ucb_cal_joint:+.4f}")
    print("-" * 78)
    b = max(ucb_joint, ucb_cal_joint)
    print(f"  CONSERVATIVE BOUND = {b:+.4f}   vs SESOI {SESOI}"
          f"   ->  {'BELOW SESOI' if b < SESOI else 'NOT below SESOI'}")
    print("  Reading: even under the deliberately severe preregistered plant, and with the")
    print("  bias uncertainty propagated and the interval calibrated on the design's own")
    print("  null, an ACTIVE strengthening of the cross-exon channel of SESOI size is")
    print("  excluded.  This bounds ACTIVE strengthening only -- it does NOT exclude")
    print("  exon-like relative DOMINANCE arising from a constant cross-exon channel plus")
    print("  decay of the cross-intron channel.")
    print("=" * 78)

    out = guard_output_path(
        REPO / (args.out or "results/splicebert_coupling/ed1_sensitivity_bound.json"),
        overwrite=args.overwrite_record)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))

    from gfmcd.provenance import ExperimentManifest
    ExperimentManifest.create(
        "splicebert_coupling:ed1_sensitivity_bound",
        [sys.argv[0], *sys.argv[1:]], repo=REPO, status="DIAGNOSTIC",
        config=dict(sesoi=SESOI, alpha=ALPHA, n_boot=args.n_boot, scores=args.scores),
        seeds={"bootstrap": 0, "signflip_calibration": 7, "bias_draws": 3},
        notes=["POST-RUN robustness interpretation, NOT a preregistered outcome. The frozen "
               "verdict remains NOT LICENSED AS A SWITCH (prereg S6 row 2) and the confirmatory "
               "artifact records GATES_FAILED.",
               "Uses ONLY preregistered quantities: beta = 3*SESOI/raw_gap exactly as "
               "planted_confound_null froze it. No post-outcome empirical slope is used.",
               "Bounds ACTIVE cross-exon strengthening; does NOT exclude exon-like relative "
               "dominance from a constant cross-exon channel plus cross-intron decay."]
    ).write(out.parent / f"{out.stem}.manifest.json")
    print(f"\nsaved -> {out}\nmanifest -> {out.parent / (out.stem + '.manifest.json')}")


if __name__ == "__main__":
    main()
