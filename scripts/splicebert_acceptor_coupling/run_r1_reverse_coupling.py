#!/usr/bin/env python
"""R1 — reverse-direction coupling gate: mask the acceptor, corrupt the partner donor.

Executes `studies/splicebert_acceptor_coupling/preregistrations/
PREREGISTRATION_R1_REVERSE_COUPLING_GATE.md`, FROZEN 2026-08-05.  Every threshold, verdict
class and control definition below is imported from that document, not invented here.

    uv run python scripts/splicebert_acceptor_coupling/run_r1_reverse_coupling.py --smoke
    uv run python scripts/splicebert_acceptor_coupling/run_r1_reverse_coupling.py \
        --n 1000 --out results/splicebert_acceptor_coupling/r1_reverse_coupling.json

VERIFICATION TIER, UNCONDITIONALLY.  Validation split only; chr8/9 is not opened.  The design
was established by two disclosed val pilots (prereg §7) and the decision rule was corrected
after the second (§7.1), so this protocol cannot emit CONFIRMATORY and does not try to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot, gboot_ratio,
    load_adapter, load_examples, matched_decoy_acceptor, matched_decoy_donor, pick_device,
    substitute,
)
from gfmcd.splicing.coupling import (  # noqa: E402
    MOTIF_CLEAN_DONOR_ALT, planted_donor_motifs, stable_rng,
)
from gfmcd.splicing.reverse_coupling import (  # noqa: E402
    FORWARD_ACCEPTOR_ALT, acceptor_readout, destroy_donor, donor_alt_admissible,
    margins_at, matched_operators,
)

# ---- FROZEN CONSTANTS (prereg §2-§6).  Not revisable by any number this script produces. --
SESOI = 0.05
DECOY_TOL = 40                     # matched_decoy_acceptor tolerance, prereg §3
N_BOOT = 2000
SEED = 0
AR1_ACC_VAL_MEAN_DELTA = 0.9909    # instrument anchor, prereg §5.1
ANCHOR_TOL = 0.15
HEALTH_FLOOR = 1.0                 # prereg §5.4
MIN_OP_N = 50                      # prereg §4
OPERATORS = ("AC", "AT", "CC", "CT", "TC", "TT", "CG", "TG", "AA")
PLANT_POS_MULT = 3                 # prereg §5.7
PLANT_POS_SEEDS = 20
PLANT_POS_MIN = 18
FPR_SEEDS = 99                     # prereg §5.8
FPR_MAX = 0.05
# C2's forward counterparts, quoted for comparison only (sealed_test.json).
C2_FORWARD_NONPARTNER_AG = 0.007
C2_FORWARD_OTHER_DONOR = 0.005


def ci(v, g):
    lo, hi = gboot(np.asarray(v), g, n=N_BOOT, seed=SEED)
    return dict(mean=float(np.mean(v)), ci=[float(lo), float(hi)], n=int(len(v)))


def mde80(v, g):
    """Minimum effect detectable at 80% power, from the gene-clustered bootstrap SD."""
    lo, hi = gboot(np.asarray(v), g, n=N_BOOT, seed=SEED)
    return float(2.802 * (hi - lo) / (2 * 1.96))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--smoke", action="store_true",
                    help="memorials only; endpoints withheld from stdout")
    ap.add_argument("--out", default=None, help="output path; REQUIRED for a settled run")
    args = ap.parse_args()

    if args.split != "val":
        raise SystemExit("prereg §2 fixes the split to val; chr8/9 is sealed and not opened here")
    if args.out is None and not args.smoke:
        raise SystemExit("--out is required (no default: a runner must not overwrite a registered artifact)")

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=False)
    task = CouplingTask(ad, batch=args.batch)
    n_req = 24 if args.smoke else args.n
    X, _ = load_examples(args.gtf, args.fasta, args.split, n_req, task=task,
                         tag=f"r1-reverse-{args.split}", min_effect=None)

    # ---- paired cohort: exons with BOTH matched comparators (C2's idiom, prereg §3) ------
    dd = [matched_decoy_donor(x) for x in X]
    da = [matched_decoy_acceptor(x, DECOY_TOL) for x in X]
    keep = [i for i in range(len(X)) if dd[i] is not None and da[i] is not None]
    P = [X[i] for i in keep]
    DD = [dd[i] for i in keep]
    DA = [da[i] for i in keep]
    g = [x.gene_id for x in P]
    if not P:
        raise SystemExit("no exon has both matched comparators; the instrument cannot run")

    seq = [x.sequence for x in P]
    a_pos = [acceptor_readout(x) for x in P]
    d_pos = [x.donor_pos for x in P]
    dec_a = [p - 1 for p in DA]

    corr_d = [destroy_donor(x, MOTIF_CLEAN_DONOR_ALT) for x in P]
    corr_dd = [substitute(x.sequence, p, MOTIF_CLEAN_DONOR_ALT) for x, p in zip(P, DD)]
    corr_fwd = [substitute(x.sequence, x.next_acceptor_pos - 2, FORWARD_ACCEPTOR_ALT) for x in P]

    a_clean = margins_at(task, a_pos, seq)
    own = a_clean - margins_at(task, a_pos, corr_d)
    o_clean = margins_at(task, dec_a, seq)
    readout_ctrl = o_clean - margins_at(task, dec_a, corr_d)
    partner_ctrl = a_clean - margins_at(task, a_pos, corr_dd)
    d_clean = margins_at(task, d_pos, seq)
    fwd = d_clean - margins_at(task, d_pos, corr_fwd)

    diff_readout = own - readout_ctrl
    diff_partner = own - partner_ctrl

    # ---- memorials (prereg §5) -----------------------------------------------------------
    M: dict[str, dict] = {}

    def mem(name, ok, forced=False, **detail):
        M[name] = dict(PASS=bool(ok), forced=bool(forced), **detail)

    mem("instrument_anchor_forward_YD",
        abs(float(fwd.mean()) - AR1_ACC_VAL_MEAN_DELTA) <= ANCHOR_TOL,
        observed=float(fwd.mean()), reference=AR1_ACC_VAL_MEAN_DELTA, tol=ANCHOR_TOL)

    plants = sum(1 for x in P if not donor_alt_admissible(x, MOTIF_CLEAN_DONOR_ALT))
    mem("construction_zero_planted_motifs", plants == 0, n_plants=plants, n=len(P))

    mem("edit_size_matched", matched_operators(MOTIF_CLEAN_DONOR_ALT, FORWARD_ACCEPTOR_ALT),
        donor_alt=MOTIF_CLEAN_DONOR_ALT, acceptor_alt=FORWARD_ACCEPTOR_ALT)

    mem("readout_health_both_sites",
        float(a_clean.mean()) > HEALTH_FLOOR and float(d_clean.mean()) > HEALTH_FLOOR,
        acceptor=float(a_clean.mean()), donor=float(d_clean.mean()), floor=HEALTH_FLOOR)

    mem("comparator_retention_published", True, forced=True,
        retained=len(P), eligible=len(X), frac=round(len(P) / max(len(X), 1), 4))

    cc_stratum = [i for i, x in enumerate(P) if donor_alt_admissible(x, "CC")]
    mem("selection_audit_vs_DONA_stratum", len(cc_stratum) > 0,
        n_cc_admissible=len(cc_stratum), frac=round(len(cc_stratum) / len(P), 4),
        Y_A_on_stratum=(float(own[cc_stratum].mean()) if cc_stratum else None),
        Y_A_full_cohort=float(own.mean()))

    # planted positive: inject 3x SESOI into the readout control, must breach the band
    hits = 0
    for s in range(PLANT_POS_SEEDS):
        rng = stable_rng(f"r1-plant-{s}", SEED)
        idx = [rng.randrange(len(readout_ctrl)) for _ in range(len(readout_ctrl))]
        planted = readout_ctrl[idx] + PLANT_POS_MULT * SESOI
        lo, _ = gboot(planted, [g[i] for i in idx], n=500, seed=s)
        hits += int(lo >= SESOI)
    mem("planted_positive", hits >= PLANT_POS_MIN, hits=hits, of=PLANT_POS_SEEDS,
        required=PLANT_POS_MIN, planted_effect=PLANT_POS_MULT * SESOI)

    # permuted-label FPR: swap partner/decoy labels within gene; rate of positive calls
    pos = 0
    for s in range(FPR_SEEDS):
        rng = stable_rng(f"r1-fpr-{s}", SEED)
        flip = np.array([1 if rng.random() < 0.5 else -1 for _ in range(len(own))])
        permuted = diff_partner * flip
        lo, _ = gboot(permuted, g, n=500, seed=s)
        pos += int(lo > SESOI)
    fpr = pos / FPR_SEEDS
    mem("permuted_label_fpr", fpr <= FPR_MAX, rate=fpr, max=FPR_MAX, seeds=FPR_SEEDS)

    all_pass = all(v["PASS"] for v in M.values())

    # ---- operator sweep (prereg §4; reported, cannot produce a GO) -----------------------
    ops = {}
    if not args.smoke:
        for z in OPERATORS:
            adm = [i for i, x in enumerate(P) if donor_alt_admissible(x, z)]
            if len(adm) < MIN_OP_N:
                ops[z] = dict(n=len(adm), status="INSUFFICIENT_SUPPORT")
                continue
            sub = [P[i] for i in adm]
            v = a_clean[adm] - margins_at(task, [a_pos[i] for i in adm],
                                          [destroy_donor(x, z) for x in sub])
            ref = own[adm]
            lo, hi = gboot_ratio(v, ref, [g[i] for i in adm], n=N_BOOT, seed=SEED)
            ops[z] = dict(n=len(adm), admissible_frac=round(len(adm) / len(P), 3),
                          mean=float(v.mean()), rho=float(v.sum() / ref.sum()),
                          rho_ci=[float(lo), float(hi)],
                          retains_T_at_d_plus_1=(z[1] == "T"))

    # ---- verdict (prereg §6, frozen order) ----------------------------------------------
    e_own = ci(own, g)
    e_read = ci(readout_ctrl, g)
    e_part = ci(partner_ctrl, g)
    e_dr = ci(diff_readout, g)
    e_dp = ci(diff_partner, g)
    mde = max(mde80(diff_readout, g), mde80(diff_partner, g))

    def material(e):
        lo, hi = e["ci"]
        excludes_zero = lo > 0 or hi < 0
        nearer = min(abs(lo), abs(hi)) if excludes_zero else 0.0
        return excludes_zero and nearer >= SESOI

    if not all_pass:
        verdict = "INCONCLUSIVE"
    elif e_own["ci"][0] <= SESOI:
        verdict = "REVERSE_INERT"
    elif mde > SESOI:
        verdict = "INSTRUMENT_UNRESOLVED"
    elif material(e_read) or material(e_part):
        verdict = "NO_GO — NONSPECIFIC"
    elif e_dr["ci"][0] > SESOI and e_dp["ci"][0] > SESOI:
        verdict = "GO"
    else:
        verdict = "INSTRUMENT_UNRESOLVED"

    # the DRAFTED rule's outcome, retained so both readings are recoverable (prereg §7.1)
    def within_band(e):
        return abs(e["ci"][0]) <= SESOI and abs(e["ci"][1]) <= SESOI
    drafted = verdict
    if verdict == "GO" and not (within_band(e_read) and within_band(e_part)):
        drafted = "INSTRUMENT_UNRESOLVED"

    sel_lo, sel_hi = gboot_ratio(own, np.maximum(partner_ctrl, 0) + 1e-9, g, n=N_BOOT, seed=SEED)

    R = dict(
        experiment="R1_reverse_coupling_gate",
        prereg="studies/splicebert_acceptor_coupling/preregistrations/"
               "PREREGISTRATION_R1_REVERSE_COUPLING_GATE.md",
        tier="VERIFICATION", confirmatory=False, split=args.split, sealed_split_opened=False,
        smoke=args.smoke,
        cohort=dict(n_eligible=len(X), n_paired=len(P), n_genes=len(set(g)),
                    device=dev, donor_alt=MOTIF_CLEAN_DONOR_ALT,
                    forward_acceptor_alt=FORWARD_ACCEPTOR_ALT, decoy_tol=DECOY_TOL),
        levels=dict(donor_clean=float(d_clean.mean()), acceptor_clean=float(a_clean.mean()),
                    decoy_acceptor_clean=float(o_clean.mean())),
        endpoints=dict(
            Y_A_reverse=e_own, Y_D_forward_anchor=ci(fwd, g),
            control_readout_decoy_acceptor=e_read,
            control_partner_decoy_GT=e_part,
            paired_diff_vs_readout=e_dr, paired_diff_vs_partner=e_dp,
            selectivity_ratio=dict(
                point=float(own.sum() / max(float(np.maximum(partner_ctrl, 0).sum()), 1e-9)),
                ci=[float(sel_lo), float(sel_hi)]),
            mde80_max=mde),
        forward_comparison=dict(
            c2_nonpartner_ag=C2_FORWARD_NONPARTNER_AG, c2_other_donor=C2_FORWARD_OTHER_DONOR,
            note="C2's forward controls, quoted from sealed_test.json for comparison only; "
                 "not recomputed and not evidence produced here"),
        operators=ops,
        memorials=M, memorials_all_pass=all_pass,
        verdict=verdict, verdict_under_drafted_rule=drafted,
        SESOI=SESOI,
        REFLECTION=dict(
            intent_check="Does destroying the partner donor move the masked-acceptor margin, "
                         "partner- and site-specifically, as C2 established for the mirror?",
            scope="Behaviour rung only. No mechanistic decomposition. Validation split; chr8/9 "
                  "sealed and not opened. One checkpoint. Nothing about splicing or biology.",
            could_not_have_failed=[
                "comparator_retention_published is FORCED — it records a number, it cannot fail",
                "the readout-specificity control reads at a spurious decoy AG whose clean margin "
                 "is far below a real acceptor's, so it sits near floor by construction; this "
                 "weakness is inherited from C2's own idiom and applies to both directions",
            ],
            disclosure="Decision rule corrected after pilot 2 (prereg §7.1); "
                       "verdict_under_drafted_rule records the pre-correction outcome.",
        ),
    )

    if args.smoke:
        print("SMOKE — memorials only, endpoints withheld")
        for k, v in M.items():
            print(f"  {'PASS' if v['PASS'] else 'FAIL'}  {k}{' [FORCED]' if v['forced'] else ''}")
        print(f"\nmemorials_all_pass={all_pass}  n_paired={len(P)}  n_genes={len(set(g))}")
        return

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(R, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r1_reverse_coupling", sys.argv,
        config=R["cohort"], seeds=dict(bootstrap=SEED), status="VERIFICATION",
        notes=["Validation split only; sealed chr8/9 not opened.",
               "Decision rule corrected after a disclosed pilot; see prereg §7.1.",
               f"Verdict {verdict}; under the drafted rule {drafted}."])
    for f in (args.model, args.gtf, args.fasta):
        try:
            man.add_input(f)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f}")
    man.add_output(p)
    man.write(p.with_suffix(".manifest.json"))

    print("=" * 78)
    for k, v in M.items():
        print(f"  {'PASS' if v['PASS'] else 'FAIL'}  {k}{' [FORCED]' if v['forced'] else ''}")
    print(f"\n  Y_A reverse        {e_own['mean']:+.4f}  [{e_own['ci'][0]:+.4f}, {e_own['ci'][1]:+.4f}]")
    print(f"  Y_D forward anchor {fwd.mean():+.4f}")
    print(f"  ctrl readout       {e_read['mean']:+.4f}  [{e_read['ci'][0]:+.4f}, {e_read['ci'][1]:+.4f}]")
    print(f"  ctrl partner       {e_part['mean']:+.4f}  [{e_part['ci'][0]:+.4f}, {e_part['ci'][1]:+.4f}]")
    print(f"  paired vs readout  {e_dr['mean']:+.4f}  [{e_dr['ci'][0]:+.4f}, {e_dr['ci'][1]:+.4f}]")
    print(f"  paired vs partner  {e_dp['mean']:+.4f}  [{e_dp['ci'][0]:+.4f}, {e_dp['ci'][1]:+.4f}]")
    print(f"\n  VERDICT {verdict}   (drafted rule: {drafted})")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
