#!/usr/bin/env python
"""R1b — reverse-direction coupling gate: corrected power gate + three-draw verdict stability.

Executes `studies/splicebert_acceptor_coupling/preregistrations/
PREREGISTRATION_R1b_REVERSE_COUPLING_GATE.md`, FROZEN 2026-08-05.

Differs from R1 in exactly two frozen places:
  §6.3  MDE80 is gated on the CONTROL ARMS (where the null under test lives), not on the paired
        differences (a heavy-tailed effect of size ~1.0 whose bootstrap SD scales with the effect;
        R1's gate was unpassable by construction -- CORRECTIONS.md ACR1).
  §5.9  Three seeded draws; the verdict class must be identical on all three.  R1's decoy-GT control
        excluded zero on one draw and included it on another, so specificity had been shown on one
        cohort selection rather than shown to be stable.

R1's clause 3 is computed alongside as `verdict_under_R1_rule` so both protocols stay comparable.

    uv run python scripts/splicebert_acceptor_coupling/run_r1b_reverse_coupling.py --smoke
    uv run python scripts/splicebert_acceptor_coupling/run_r1b_reverse_coupling.py \
        --n 1000 --out results/splicebert_acceptor_coupling/r1b_reverse_coupling.json

VERIFICATION TIER, UNCONDITIONALLY.  Validation split only; chr8/9 is not opened.
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
    MOTIF_CLEAN_DONOR_ALT, stable_rng,
)
from gfmcd.splicing.reverse_coupling import (  # noqa: E402
    FORWARD_ACCEPTOR_ALT, acceptor_readout, destroy_donor, donor_alt_admissible,
    margins_at, matched_operators,
)

# ---- FROZEN CONSTANTS (prereg §2-§6).  Not revisable by any number this script produces. --
SESOI = 0.05
DECOY_TOL = 40
N_BOOT = 2000
SEED = 0
AR1_ACC_VAL_MEAN_DELTA = 0.9909
ANCHOR_TOL = 0.15
HEALTH_FLOOR = 1.0
MIN_OP_N = 50
OPERATORS = ("AC", "AT", "CC", "CT", "TC", "TT", "CG", "TG", "AA")
PLANT_POS_MULT = 3
PLANT_POS_SEEDS = 20
PLANT_POS_MIN = 18
FPR_SEEDS = 99
FPR_MAX = 0.05
DRAW_TAGS = ("r1b-d0", "r1b-d1", "r1b-d2")          # prereg §2.1, frozen
#: Memorials every draw runs.  §5.9's stability comparison is computed on THESE only -- the two
#: calibration arms below calibrate the decision layer rather than the cohort and run once, on
#: draw 0, so including them would make draw 0's verdict class incomparable to the others'.
SHARED_MEMORIALS = (
    "instrument_anchor_forward_YD", "construction_zero_planted_motifs", "edit_size_matched",
    "readout_health_both_sites", "comparator_retention_published",
    "selection_audit_vs_DONA_stratum",
)
C2_FORWARD_NONPARTNER_AG = 0.007
C2_FORWARD_OTHER_DONOR = 0.005


def ci(v, g):
    lo, hi = gboot(np.asarray(v), g, n=N_BOOT, seed=SEED)
    return dict(mean=float(np.mean(v)), ci=[float(lo), float(hi)], n=int(len(v)))


def mde80(v, g):
    """Minimum effect detectable at 80% power, from the gene-clustered bootstrap SD."""
    lo, hi = gboot(np.asarray(v), g, n=N_BOOT, seed=SEED)
    return float(2.8016 * (hi - lo) / (2 * 1.96))


def material(e):
    """Materially non-null: CI excludes zero AND its nearer edge reaches SESOI (prereg §6.4)."""
    lo, hi = e["ci"]
    if not (lo > 0 or hi < 0):
        return False
    return min(abs(lo), abs(hi)) >= SESOI


def classify(mem_ok, e_own, e_read, e_part, e_dr, e_dp, mde_ctrl, mde_paired, *, rule="R1b"):
    """The frozen decision rule (prereg §6).  `rule='R1'` restores R1's clause 3 for comparison."""
    if not mem_ok:
        return "INCONCLUSIVE"
    if e_own["ci"][0] <= SESOI:
        return "REVERSE_INERT"
    if (mde_ctrl if rule == "R1b" else mde_paired) > SESOI:
        return "INSTRUMENT_UNRESOLVED"
    if material(e_read) or material(e_part):
        return "NO_GO — NONSPECIFIC"
    if e_dr["ci"][0] > SESOI and e_dp["ci"][0] > SESOI:
        return "GO"
    return "INSTRUMENT_UNRESOLVED"


def measure_draw(task, ad, args, tag, n_req, *, with_operators: bool):
    """One seeded draw: cohort, all four arms, endpoints, and the per-draw memorial inputs."""
    X, _ = load_examples(args.gtf, args.fasta, "val", n_req, task=task, tag=tag, min_effect=None)
    dd = [matched_decoy_donor(x) for x in X]
    da = [matched_decoy_acceptor(x, DECOY_TOL) for x in X]
    keep = [i for i in range(len(X)) if dd[i] is not None and da[i] is not None]
    P, DD, DA = [X[i] for i in keep], [dd[i] for i in keep], [da[i] for i in keep]
    if not P:
        raise SystemExit(f"draw {tag}: no exon has both matched comparators")
    g = [x.gene_id for x in P]

    seq = [x.sequence for x in P]
    a_pos = [acceptor_readout(x) for x in P]
    d_pos = [x.donor_pos for x in P]
    dec_a = [p - 1 for p in DA]

    a_clean = margins_at(task, a_pos, seq)
    own = a_clean - margins_at(task, a_pos, [destroy_donor(x, MOTIF_CLEAN_DONOR_ALT) for x in P])
    o_clean = margins_at(task, dec_a, seq)
    readout_ctrl = o_clean - margins_at(
        task, dec_a, [destroy_donor(x, MOTIF_CLEAN_DONOR_ALT) for x in P])
    partner_ctrl = a_clean - margins_at(
        task, a_pos, [substitute(x.sequence, p, MOTIF_CLEAN_DONOR_ALT) for x, p in zip(P, DD)])
    d_clean = margins_at(task, d_pos, seq)
    fwd = d_clean - margins_at(
        task, d_pos,
        [substitute(x.sequence, x.next_acceptor_pos - 2, FORWARD_ACCEPTOR_ALT) for x in P])

    ops = {}
    if with_operators:
        for z in OPERATORS:
            adm = [i for i, x in enumerate(P) if donor_alt_admissible(x, z)]
            if len(adm) < MIN_OP_N:
                ops[z] = dict(n=len(adm), status="INSUFFICIENT_SUPPORT")
                continue
            v = a_clean[adm] - margins_at(task, [a_pos[i] for i in adm],
                                          [destroy_donor(P[i], z) for i in adm])
            ref = own[adm]
            lo, hi = gboot_ratio(v, ref, [g[i] for i in adm], n=N_BOOT, seed=SEED)
            ops[z] = dict(n=len(adm), admissible_frac=round(len(adm) / len(P), 3),
                          mean=float(v.mean()), rho=float(v.sum() / ref.sum()),
                          rho_ci=[float(lo), float(hi)], retains_T_at_d_plus_1=(z[1] == "T"))

    sel_lo, sel_hi = gboot_ratio(own, np.maximum(partner_ctrl, 0) + 1e-9, g, n=N_BOOT, seed=SEED)
    return dict(
        tag=tag, P=P, g=g, own=own, readout_ctrl=readout_ctrl, partner_ctrl=partner_ctrl, fwd=fwd,
        a_clean=a_clean, d_clean=d_clean, o_clean=o_clean, n_eligible=len(X), operators=ops,
        selectivity=dict(point=float(own.sum() / max(float(np.maximum(partner_ctrl, 0).sum()), 1e-9)),
                         ci=[float(sel_lo), float(sel_hi)]))


def draw_memorials(D, *, full: bool):
    """§5 memorials for one draw.  `full=False` skips the two expensive calibration arms."""
    P, g, own = D["P"], D["g"], D["own"]
    M: dict[str, dict] = {}

    def mem(name, ok, forced=False, **detail):
        M[name] = dict(PASS=bool(ok), forced=bool(forced), **detail)

    mem("instrument_anchor_forward_YD",
        abs(float(D["fwd"].mean()) - AR1_ACC_VAL_MEAN_DELTA) <= ANCHOR_TOL,
        observed=float(D["fwd"].mean()), reference=AR1_ACC_VAL_MEAN_DELTA, tol=ANCHOR_TOL)
    plants = sum(1 for x in P if not donor_alt_admissible(x, MOTIF_CLEAN_DONOR_ALT))
    mem("construction_zero_planted_motifs", plants == 0, n_plants=plants, n=len(P))
    mem("edit_size_matched", matched_operators(MOTIF_CLEAN_DONOR_ALT, FORWARD_ACCEPTOR_ALT),
        donor_alt=MOTIF_CLEAN_DONOR_ALT, acceptor_alt=FORWARD_ACCEPTOR_ALT)
    mem("readout_health_both_sites",
        float(D["a_clean"].mean()) > HEALTH_FLOOR and float(D["d_clean"].mean()) > HEALTH_FLOOR,
        acceptor=float(D["a_clean"].mean()), donor=float(D["d_clean"].mean()), floor=HEALTH_FLOOR)
    mem("comparator_retention_published", True, forced=True,
        retained=len(P), eligible=D["n_eligible"], frac=round(len(P) / max(D["n_eligible"], 1), 4))
    cc = [i for i, x in enumerate(P) if donor_alt_admissible(x, "CC")]
    mem("selection_audit_vs_DONA_stratum", len(cc) > 0, n_cc_admissible=len(cc),
        frac=round(len(cc) / len(P), 4),
        Y_A_on_stratum=(float(own[cc].mean()) if cc else None),
        Y_A_full_cohort=float(own.mean()))

    if not full:
        return M

    hits = 0
    for s in range(PLANT_POS_SEEDS):
        rng = stable_rng(f"r1b-plant-{s}", SEED)
        idx = [rng.randrange(len(D["readout_ctrl"])) for _ in range(len(D["readout_ctrl"]))]
        lo, _ = gboot(D["readout_ctrl"][idx] + PLANT_POS_MULT * SESOI,
                      [g[i] for i in idx], n=500, seed=s)
        hits += int(lo >= SESOI)
    mem("planted_positive", hits >= PLANT_POS_MIN, hits=hits, of=PLANT_POS_SEEDS,
        required=PLANT_POS_MIN, planted_effect=PLANT_POS_MULT * SESOI)

    diff_partner = own - D["partner_ctrl"]
    pos = 0
    for s in range(FPR_SEEDS):
        rng = stable_rng(f"r1b-fpr-{s}", SEED)
        flip = np.array([1 if rng.random() < 0.5 else -1 for _ in range(len(own))])
        lo, _ = gboot(diff_partner * flip, g, n=500, seed=s)
        pos += int(lo > SESOI)
    mem("permuted_label_fpr", pos / FPR_SEEDS <= FPR_MAX, rate=pos / FPR_SEEDS,
        max=FPR_MAX, seeds=FPR_SEEDS)
    return M


def draw_endpoints(D):
    g = D["g"]
    e_own, e_read = ci(D["own"], g), ci(D["readout_ctrl"], g)
    e_part = ci(D["partner_ctrl"], g)
    e_dr, e_dp = ci(D["own"] - D["readout_ctrl"], g), ci(D["own"] - D["partner_ctrl"], g)
    return dict(
        Y_A_reverse=e_own, Y_D_forward_anchor=ci(D["fwd"], g),
        control_readout_decoy_acceptor=e_read, control_partner_decoy_GT=e_part,
        paired_diff_vs_readout=e_dr, paired_diff_vs_partner=e_dp,
        selectivity_ratio=D["selectivity"],
        mde80_controls_max=max(mde80(D["readout_ctrl"], g), mde80(D["partner_ctrl"], g)),
        mde80_paired_max=max(mde80(D["own"] - D["readout_ctrl"], g),
                             mde80(D["own"] - D["partner_ctrl"], g)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.out is None and not args.smoke:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=False)
    task = CouplingTask(ad, batch=args.batch)
    n_req = 24 if args.smoke else args.n

    draws, per_draw = [], []
    for i, tag in enumerate(DRAW_TAGS):
        D = measure_draw(task, ad, args, tag, n_req, with_operators=(i == 0 and not args.smoke))
        M = draw_memorials(D, full=(i == 0))
        E = draw_endpoints(D)
        ok = all(v["PASS"] for v in M.values())
        # §5.9 compares the DECISION LAYER across draws, so it must be computed on the memorial
        # set every draw actually runs.  The two calibration arms (planted positive, permuted-label
        # FPR) calibrate the decision layer itself rather than the cohort, run once on draw 0 per
        # E11's precedent, and would otherwise make draw 0's class incomparable to the others'.
        shared_ok = all(v["PASS"] for k, v in M.items() if k in SHARED_MEMORIALS)
        args_cls = (E["Y_A_reverse"], E["control_readout_decoy_acceptor"],
                    E["control_partner_decoy_GT"], E["paired_diff_vs_readout"],
                    E["paired_diff_vs_partner"], E["mde80_controls_max"], E["mde80_paired_max"])
        v_r1b = classify(ok, *args_cls, rule="R1b")
        v_r1 = classify(ok, *args_cls, rule="R1")
        v_shared = classify(shared_ok, *args_cls, rule="R1b")
        draws.append(D)
        per_draw.append(dict(tag=tag, n_paired=len(D["P"]), n_genes=len(set(D["g"])),
                             memorials=M, memorials_all_pass=ok,
                             shared_memorials_pass=shared_ok, endpoints=E,
                             verdict=v_r1b, verdict_under_R1_rule=v_r1,
                             verdict_on_shared_memorials=v_shared,
                             levels=dict(donor_clean=float(D["d_clean"].mean()),
                                         acceptor_clean=float(D["a_clean"].mean()),
                                         decoy_acceptor_clean=float(D["o_clean"].mean()))))
        print(f"  draw {tag}: n={len(D['P'])} genes={len(set(D['g']))} "
              f"Y_A={E['Y_A_reverse']['mean']:+.4f} verdict={v_r1b} "
              f"(shared-basis {v_shared})", flush=True)

    # ---- §5.9 draw stability, then the primary verdict from draw 0 ----------------------
    verdicts = [d["verdict_on_shared_memorials"] for d in per_draw]
    stable = len(set(verdicts)) == 1
    per_draw[0]["memorials"]["draw_stability"] = dict(
        PASS=bool(stable), forced=False, verdicts=verdicts, tags=list(DRAW_TAGS))
    all_ok = per_draw[0]["memorials_all_pass"] and stable
    per_draw[0]["memorials_all_pass"] = all_ok

    E0 = per_draw[0]["endpoints"]
    verdict = classify(all_ok, E0["Y_A_reverse"], E0["control_readout_decoy_acceptor"],
                       E0["control_partner_decoy_GT"], E0["paired_diff_vs_readout"],
                       E0["paired_diff_vs_partner"], E0["mde80_controls_max"],
                       E0["mde80_paired_max"], rule="R1b")
    verdict_r1 = classify(all_ok, E0["Y_A_reverse"], E0["control_readout_decoy_acceptor"],
                          E0["control_partner_decoy_GT"], E0["paired_diff_vs_readout"],
                          E0["paired_diff_vs_partner"], E0["mde80_controls_max"],
                          E0["mde80_paired_max"], rule="R1")

    def across(name):
        vals = [d["endpoints"][name]["mean"] for d in per_draw]
        return dict(values=[round(v, 4) for v in vals], range=[min(vals), max(vals)],
                    sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    NOTE="across-draw RANGE and SD, NOT a confidence interval")

    R = dict(
        experiment="R1b_reverse_coupling_gate",
        prereg="studies/splicebert_acceptor_coupling/preregistrations/"
               "PREREGISTRATION_R1b_REVERSE_COUPLING_GATE.md",
        supersedes_gate="R1 (verdict INSTRUMENT_UNRESOLVED; see CORRECTIONS.md ACR1)",
        tier="VERIFICATION", confirmatory=False, split="val", sealed_split_opened=False,
        smoke=args.smoke,
        config=dict(device=dev, donor_alt=MOTIF_CLEAN_DONOR_ALT,
                    forward_acceptor_alt=FORWARD_ACCEPTOR_ALT, decoy_tol=DECOY_TOL,
                    draw_tags=list(DRAW_TAGS), SESOI=SESOI),
        primary_draw=DRAW_TAGS[0], draws=per_draw,
        across_draws={k: across(k) for k in
                      ("Y_A_reverse", "control_readout_decoy_acceptor",
                       "control_partner_decoy_GT", "paired_diff_vs_readout",
                       "paired_diff_vs_partner")},
        operators=per_draw[0].get("endpoints") and draws[0]["operators"],
        forward_comparison=dict(c2_nonpartner_ag=C2_FORWARD_NONPARTNER_AG,
                                c2_other_donor=C2_FORWARD_OTHER_DONOR,
                                note="C2's forward controls, quoted from sealed_test.json for "
                                     "comparison only; not recomputed and not evidence here"),
        verdict=verdict, verdict_under_R1_rule=verdict_r1, draw_stability=stable,
        REFLECTION=dict(
            intent_check="Does destroying the partner donor move the masked-acceptor margin, "
                         "partner- and site-specifically, stably across cohort selection?",
            scope="Behaviour rung only. Validation split; chr8/9 sealed and not opened. Three "
                  "draws are RESAMPLED FROM THE SAME POOL -- they measure cohort-selection "
                  "stability, not independent replication, and do not widen the population.",
            could_not_have_failed=[
                "comparator_retention_published is FORCED -- it records a number (R7/R21)",
                "the readout-specificity control reads at a spurious decoy AG whose clean margin is "
                "far below a real acceptor's, so it sits near floor by construction; inherited from "
                "C2's own idiom and applies to both directions equally",
                "the z[1] operator split is no longer a prediction -- it was met twice before this "
                "run and is carried as a replication check only",
            ],
            disclosure="Every R1 endpoint was seen before this was frozen, on the same val pool "
                       "this gate resamples. The MDE80 relocation was written knowing clause 3 was "
                       "what blocked GO; prereg §7 states the case without mitigation, and "
                       "verdict_under_R1_rule preserves the other reading.",
            identifies_nothing="A GO authorises designing the mechanistic battery. It says nothing "
                               "about which components mediate the reverse direction, and in "
                               "particular nothing about whether L3H1 transfers.",
        ),
    )

    if args.smoke:
        print("\nSMOKE — memorials only, endpoints withheld")
        for k, v in per_draw[0]["memorials"].items():
            print(f"  {'PASS' if v['PASS'] else 'FAIL'}  {k}{' [FORCED]' if v['forced'] else ''}")
        print(f"\nmemorials_all_pass={all_ok}  stable={stable}  verdicts={verdicts}")
        return

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(R, indent=2, default=float))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r1b_reverse_coupling", sys.argv,
        config=R["config"], seeds=dict(bootstrap=SEED), status="VERIFICATION",
        notes=["Validation split only; sealed chr8/9 not opened.",
               "Supersedes the R1 gate's verdict; see CORRECTIONS.md ACR1.",
               f"Verdict {verdict}; under R1's clause 3 {verdict_r1}; draw-stable={stable}."])
    for f in (args.model, args.gtf, args.fasta):
        try:
            man.add_input(f)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f}")
    man.add_output(p)
    man.write(p.with_suffix(".manifest.json"))

    print("\n" + "=" * 78)
    for k, v in per_draw[0]["memorials"].items():
        print(f"  {'PASS' if v['PASS'] else 'FAIL'}  {k}{' [FORCED]' if v['forced'] else ''}")
    for nm, key in (("Y_A reverse", "Y_A_reverse"),
                    ("ctrl readout", "control_readout_decoy_acceptor"),
                    ("ctrl partner", "control_partner_decoy_GT"),
                    ("paired vs readout", "paired_diff_vs_readout"),
                    ("paired vs partner", "paired_diff_vs_partner")):
        e = E0[key]
        print(f"\n  {nm:<20}{e['mean']:+.4f}  [{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}]", end="")
    print(f"\n\n  MDE80 controls {E0['mde80_controls_max']:.4f}  (paired {E0['mde80_paired_max']:.4f})")
    print(f"  draws {verdicts}  stable={stable}")
    print(f"\n  VERDICT {verdict}   (under R1's clause 3: {verdict_r1})")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
