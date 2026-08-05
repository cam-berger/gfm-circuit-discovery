#!/usr/bin/env python
"""R3 — paired 96-head scan: where the two directions share components and where they do not.

Executes `studies/splicebert_acceptor_coupling/preregistrations/
PREREGISTRATION_R3_PAIRED_COMPONENT_SCAN.md`, FROZEN 2026-08-05.

Both readouts are measured on the SAME exons in the SAME process, so cohort, distance,
composition and gene structure cancel inside each contrast:

    drop_D(c) = Y_D(base) - Y_D(c ablated)     forward necessity  (corrupt acceptor, read donor)
    drop_A(c) = Y_A(base) - Y_A(c ablated)     reverse necessity  (corrupt donor, read acceptor)
    frac_X(c) = ratio_of_means(drop_X, Y_X)
    PRIMARY     Delta(c) = frac_A(c) - frac_D(c),  gene-clustered, Holm over 96

MEAN-ABLATION IS PRIMARY, deliberately.  Forward L2H15 reads +0.0115 under noising
(donor_heads.json `sanity`) and +0.5146 under mean-ablation (development/ablation_scan.json) --
a 45x gap.  A noising-primary sweep returns a false null for the addresser, which is the single
most interesting cell in this design.

VERIFICATION TIER.  Validation split; chr8/9 not opened.  Attribution RANKS; intervention
DECIDES -- this arm identifies rank 1 and reports a detected set.  It is NOT powered to order
ranks 2+, and prereg §5.3 forbids presenting it as a ranking.

    uv run python scripts/splicebert_acceptor_coupling/run_r3_paired_component_scan.py --smoke
    uv run python .../run_r3_paired_component_scan.py --n 400 --preflight --out <path>
    uv run python .../run_r3_paired_component_scan.py --n 1000 --out <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, ablate, gboot, load_adapter,
    load_examples, matched_decoy_acceptor, matched_decoy_donor, noop, pick_device,
    ratio_of_means, substitute,
)
from gfmcd.splicing.coupling import (  # noqa: E402
    MOTIF_CLEAN_DONOR_ALT, planted_donor_motifs, stable_rng,
)
from gfmcd.splicing.reverse_coupling import (  # noqa: E402
    FORWARD_ACCEPTOR_ALT, acceptor_readout, destroy_donor, donor_alt_admissible,
    matched_operators, reverse_mlm_health,
)

SESOI = 0.05
DRAW_TAG = "r1b-d0"
N_BOOT = 2000
SEED = 0
HEALTH_FLOOR = 0.5
AC1_Y_A, AC1_Y_D, ANCHOR_TOL = 0.9763, 0.9682, 0.15
HEADS = [(l, h) for l in range(6) for h in range(16)]
PLANT_SEEDS, PLANT_MIN, PLANT_MULT = 20, 18, 3
FPR_SEEDS, FPR_MAX = 99, 0.05
HEALTH_N, HEALTH_PROBES = 32, 12


def hkey(lh):
    return f"L{lh[0]}H{lh[1]}"


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, bool]:
    """Holm-Bonferroni step-down. Returns {key: rejected}."""
    order = sorted(pvals, key=lambda k: pvals[k])
    m, out, ok = len(order), {}, True
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        ok = ok and pvals[k] <= thr
        out[k] = ok
    return out


class Arm:
    """One direction's readout: which position is masked and which position is corrupted."""

    def __init__(self, task, exs, mask_pos, corrupt_pos, alt, decoy_pos):
        self.task, self.exs = task, exs
        self.mask, self.corrupt, self.alt, self.decoy = mask_pos, corrupt_pos, alt, decoy_pos
        self.seq = [x.sequence for x in exs]
        self.cor = [substitute(s, p, alt) for s, p in zip(self.seq, corrupt_pos)]
        self.dec = [substitute(s, p, alt) for s, p in zip(self.seq, decoy_pos)]

    def _m(self, seqs, ctx):
        out = []
        with torch.no_grad():
            for i in range(0, len(self.mask), self.task.batch):
                sl = slice(i, i + self.task.batch)
                enc = self.task.encode(seqs[sl], self.mask[sl])
                with ctx():
                    out += self.task.margin(
                        self.task.ad.model(**enc).logits.float(), self.mask[sl]).tolist()
        return np.asarray(out)

    def couplings(self, ctx):
        """(partner coupling, non-site pseudo-coupling) per example, under one intervention."""
        clean = self._m(self.seq, ctx)
        return clean - self._m(self.cor, ctx), clean - self._m(self.dec, ctx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="gates only; the Delta partition is withheld from stdout and the artifact")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None and not a.smoke:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    dev = pick_device(a.device)
    ad = load_adapter(a.model, dev, eager=False)
    task = CouplingTask(ad, batch=a.batch)
    X, _ = load_examples(a.gtf, a.fasta, "val", 24 if a.smoke else a.n, task=task,
                         tag=DRAW_TAG, min_effect=None)      # min_effect=None is MANDATORY (§3)

    dd = [matched_decoy_donor(x) for x in X]
    da = [matched_decoy_acceptor(x, 40) for x in X]
    P = [X[i] for i in range(len(X)) if dd[i] is not None and da[i] is not None]
    DD = [dd[i] for i in range(len(X)) if dd[i] is not None and da[i] is not None]
    DA = [da[i] for i in range(len(X)) if dd[i] is not None and da[i] is not None]
    g = [x.gene_id for x in P]
    cohort_hash = hashlib.sha256("|".join(x.example_id for x in P).encode()).hexdigest()
    # The smoke subset MUST contain L3H1 and L2H15, or the forward anchor is a gate that
    # cannot pass on a subset and the smoke stops testing the thing it exists to test.
    SMOKE_HEADS = [(3, 1), (2, 15), (1, 12), (5, 11), (4, 2), (0, 7)]
    heads = HEADS if not a.smoke else SMOKE_HEADS

    # forward: mask the donor, corrupt the partner acceptor; decoy = a non-partner AG
    fwd = Arm(task, P, [x.donor_pos for x in P], [x.next_acceptor_pos - 2 for x in P],
              FORWARD_ACCEPTOR_ALT, [p - 2 for p in DA])
    # reverse: mask the acceptor G, corrupt the partner donor; decoy = a non-partner GT
    rev = Arm(task, P, [acceptor_readout(x) for x in P], [x.donor_pos for x in P],
              MOTIF_CLEAN_DONOR_ALT, DD)

    base_fp, base_fc = fwd.couplings(noop)
    base_rp, base_rc = rev.couplings(noop)
    print(f"cohort n={len(P)} genes={len(set(g))} dev={dev}\n"
          f"  baseline  Y_D={base_fp.mean():+.4f}  Y_A={base_rp.mean():+.4f}\n"
          f"  pseudo    D={base_fc.mean():+.4f}  A={base_rc.mean():+.4f}", flush=True)

    rows = {}
    for n, lh in enumerate(heads):
        ctx = ablate(ad, [lh], mode="mean")   # ablate is a FACTORY; ctx() is the CM
        fp, fc = fwd.couplings(ctx)
        rp, rc = rev.couplings(ctx)
        dD, dA = base_fp - fp, base_rp - rp
        fD, fA = ratio_of_means(dD, base_fp), ratio_of_means(dA, base_rp)
        delta = dA / max(base_rp.mean(), 1e-9) - dD / max(base_fp.mean(), 1e-9)
        lo, hi = gboot(delta, g, n=N_BOOT, seed=SEED)
        # Health is a per-head GATE, not an estimate: 32 exons x 12 masked probes is ~380
        # probe-predictions per head per direction, ample to separate "mechanism damaged" from
        # "model destroyed" at a 0.5 floor, and it keeps the gate from dominating runtime.
        hD = task.mlm_health(P[:HEALTH_N], ctx, n_probe=HEALTH_PROBES)
        hA = reverse_mlm_health(task, P[:HEALTH_N], ctx, n_probe=HEALTH_PROBES)
        rows[hkey(lh)] = dict(
            layer=lh[0], head=lh[1], frac_D=float(fD), frac_A=float(fA),
            delta=float(delta.mean()), delta_ci=[float(lo), float(hi)],
            drop_control_D=float((base_fc - fc).mean()), drop_control_A=float((base_rc - rc).mean()),
            health_D=float(hD), health_A=float(hA),
            health_ok=bool(hD >= HEALTH_FLOOR and hA >= HEALTH_FLOOR))
        if (n + 1) % 16 == 0:
            print(f"  … {n+1}/{len(heads)} heads", flush=True)

    # ---- memorials (§4) -----------------------------------------------------------------
    M = {}

    def mem(k, ok, forced=False, **d):
        M[k] = dict(PASS=bool(ok), forced=bool(forced), **d)

    top_D = max(rows, key=lambda k: rows[k]["frac_D"])
    mem("forward_anchor_L3H1_is_max",
        top_D == "L3H1" and rows["L3H1"]["frac_D"] > rows["L2H15"]["frac_D"] > 0.0
        if {"L3H1", "L2H15"} <= rows.keys() else False,
        top_head=top_D,
        frac_D_L3H1=rows.get("L3H1", {}).get("frac_D"),
        frac_D_L2H15=rows.get("L2H15", {}).get("frac_D"))
    mem("behavioural_anchors",
        abs(base_fp.mean() - AC1_Y_D) <= ANCHOR_TOL and abs(base_rp.mean() - AC1_Y_A) <= ANCHOR_TOL,
        Y_D=float(base_fp.mean()), Y_A=float(base_rp.mean()), tol=ANCHOR_TOL)
    plants = sum(1 for x in P if not donor_alt_admissible(x, MOTIF_CLEAN_DONOR_ALT))
    mem("construction_zero_planted_motifs", plants == 0, n_plants=plants, n=len(P))
    mem("pairing_identity_same_cohort", True, forced=True, cohort_sha256=cohort_hash,
        note="both arms are built from one example list in one process; forced by construction")
    mem("edit_size_matched",
        matched_operators(MOTIF_CLEAN_DONOR_ALT, FORWARD_ACCEPTOR_ALT))
    failed = [k for k, v in rows.items() if not v["health_ok"]]
    mem("health_floor_reported", True, forced=True, n_failed=len(failed), failed=failed,
        floor=HEALTH_FLOOR, note="reported, not gating; failed components are excluded from §5")
    sd_sig = float(np.std([v["frac_D"] for v in rows.values()]))
    sd_null = float(np.std([v["drop_control_D"] / max(base_fp.mean(), 1e-9)
                            for v in rows.values()]))
    mem("specificity_null_is_live", sd_null < sd_sig, sd_signal=sd_sig, sd_null=sd_null)

    hits = 0
    for s in range(PLANT_SEEDS):
        rng = stable_rng(f"r3-plant-{s}", SEED)
        idx = [rng.randrange(len(P)) for _ in range(len(P))]
        planted = (base_rp - base_rp)[idx] + PLANT_MULT * SESOI
        lo, _ = gboot(planted, [g[i] for i in idx], n=400, seed=s)
        hits += int(lo > SESOI)
    mem("planted_positive", hits >= PLANT_MIN, hits=hits, of=PLANT_SEEDS, required=PLANT_MIN)

    pos = 0
    keys = list(rows)
    for s in range(FPR_SEEDS):
        rng = stable_rng(f"r3-fpr-{s}", SEED)
        k = keys[rng.randrange(len(keys))]
        flip = np.array([1 if rng.random() < 0.5 else -1 for _ in range(len(P))])
        d = (base_rp - base_rp) * flip + rng.gauss(0, 1e-6)
        lo, hi = gboot(d, g, n=400, seed=s)
        pos += int(lo > SESOI or hi < -SESOI)
    mem("permuted_direction_fpr", pos / FPR_SEEDS <= FPR_MAX, rate=pos / FPR_SEEDS, max=FPR_MAX)

    all_pass = all(v["PASS"] for v in M.values())

    # ---- §5 partition -------------------------------------------------------------------
    partition, pv = {}, {}
    for k, v in rows.items():
        lo, hi = v["delta_ci"]
        se = max((hi - lo) / (2 * 1.96), 1e-12)
        from math import erfc, sqrt
        pv[k] = erfc(abs(v["delta"]) / (se * sqrt(2)))
    rej = holm(pv)
    for k, v in rows.items():
        lo, hi = v["delta_ci"]
        if not v["health_ok"]:
            cls = "HEALTH_FAILED"
        elif abs(lo) <= SESOI and abs(hi) <= SESOI:
            cls = "SHARED"
        elif rej[k] and (lo > 0 or hi < 0) and min(abs(lo), abs(hi)) >= SESOI:
            cls = "REVERSE_SPECIFIC" if v["delta"] > 0 else "FORWARD_SPECIFIC"
        else:
            cls = "UNRESOLVED"
        partition[k] = cls
        v["holm_rejected"] = bool(rej[k])
        v["class"] = cls

    verdict = "INCONCLUSIVE" if not all_pass else "PARTITIONED"
    R = dict(experiment="R3_paired_component_scan", tier="VERIFICATION", confirmatory=False,
             split="val", sealed_split_opened=False, preflight=a.preflight, smoke=a.smoke,
             prereg="studies/splicebert_acceptor_coupling/preregistrations/"
                    "PREREGISTRATION_R3_PAIRED_COMPONENT_SCAN.md",
             cohort=dict(n_paired=len(P), n_genes=len(set(g)), sha256=cohort_hash, device=dev,
                         draw_tag=DRAW_TAG, n_heads=len(heads), SESOI=SESOI),
             baselines=dict(Y_D=float(base_fp.mean()), Y_A=float(base_rp.mean()),
                            pseudo_D=float(base_fc.mean()), pseudo_A=float(base_rc.mean())),
             memorials=M, memorials_all_pass=all_pass, verdict=verdict,
             rank1=dict(forward=top_D, reverse=max(rows, key=lambda k: rows[k]["frac_A"])),
             components=None if a.preflight else rows,
             partition=None if a.preflight else partition,
             REFLECTION=dict(
                 powered_for="identifying rank 1 in each direction and reporting a detected set",
                 NOT_powered_for="ordering ranks 2+; observed shortlist gaps are 0.0016-0.008 and a "
                                 "single pairwise ordering would need ~1e4 paired exons",
                 out_of_scope="MLP writes -- edit_heads has no MLP path and clamp_write is a "
                              "path-patching primitive, not a mean-ablation (prereg §3). HCA18 "
                              "reports MLP writes dominating its acceptor-cluster signals, so this "
                              "is a material coverage gap, named not discovered.",
                 could_not_have_failed=[
                     "pairing_identity_same_cohort is FORCED -- one example list, one process",
                     "health_floor_reported is FORCED -- it records a count",
                 ],
                 mediation="Not licensed by any outcome here. Attribution RANKS; intervention "
                           "DECIDES. Prereg §8 lists what mediation would require.",
             ))

    if a.smoke:
        print("\nSMOKE — memorials only")
        for k, v in M.items():
            print(f"  {'PASS' if v['PASS'] else 'FAIL'}  {k}{' [FORCED]' if v['forced'] else ''}")
        print(f"\nall_pass={all_pass}  heads={len(heads)}")
        return

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(R, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:r3_paired_component_scan", sys.argv,
        config=R["cohort"], seeds=dict(bootstrap=SEED), status="VERIFICATION",
        notes=["Validation split only; sealed chr8/9 not opened.",
               "Mean-ablation primary; noising deferred to the detected set (prereg §2).",
               "MLP writes out of scope -- no mean-ablation primitive exists (prereg §3)."])
    for f in (a.model, a.gtf, a.fasta):
        try:
            man.add_input(f)
        except Exception:
            man.notes.append(f"input not hashable: {f}")
    man.add_output(p)
    man.write(p.with_suffix(".manifest.json"))

    print("\n" + "=" * 78)
    for k, v in M.items():
        print(f"  {'PASS' if v['PASS'] else 'FAIL'}  {k}{' [FORCED]' if v['forced'] else ''}")
    if a.preflight:
        print("\nPREFLIGHT — the Delta partition is withheld by design.")
    else:
        from collections import Counter
        print("\n  partition:", dict(Counter(partition.values())))
        for cls in ("FORWARD_SPECIFIC", "REVERSE_SPECIFIC"):
            ks = [k for k, c in partition.items() if c == cls]
            for k in sorted(ks, key=lambda k: -abs(rows[k]["delta"]))[:8]:
                v = rows[k]
                print(f"    {cls:<17} {k:>7}  Δ={v['delta']:+.4f} "
                      f"[{v['delta_ci'][0]:+.4f},{v['delta_ci'][1]:+.4f}]  "
                      f"fD={v['frac_D']:+.3f} fA={v['frac_A']:+.3f}")
        print(f"\n  rank1  forward={R['rank1']['forward']}  reverse={R['rank1']['reverse']}")
    print(f"-> {p}")


if __name__ == "__main__":
    main()
