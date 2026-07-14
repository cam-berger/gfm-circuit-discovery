#!/usr/bin/env python
"""THE DECISIVE TEST: at MATCHED perturbation size, does phase still matter?

WHERE WE ARE.  All six combs show an enormous period-3 damage sawtooth.  L2H15 -- the coupling
circuit's ADDRESSER -- loses 0.1136 of log-prob rolled by 1 and 0.0001 rolled by 3.  L1H12 goes
NEGATIVE in phase (rolling it by 3, 6 or 9 slightly HELPS the model).  The copiers show nothing
like it.

AND IT IS NOT YET EVIDENCE OF ANYTHING, because the PERTURBATION sawtooths too:

    L2H15   damage   0.1136 (off)  vs  0.0001 (in)   ->  ~1000x
            ||dout|| 1.83   (off)  vs  0.22   (in)   ->  ~8.4x

Rolling a stride-3 comb by 3 slides it ONTO ITSELF -- it keeps reading most of the same tokens,
so its output barely moves.  That is set arithmetic (tests/test_phase_shift_memorials.py proves
it on white noise).  So within a comb, PHASE AND PERTURBATION SIZE ARE PHYSICALLY ENTANGLED
(corr(log||dout||, off-phase) = +0.83).  Whether a 1000x damage ratio exceeds what an 8.4x
perturbation ratio buys you depends entirely on the exponent gamma -- and gamma cannot be
estimated from the combs without circularity, nor safely borrowed from the copiers, whose
||dout|| barely moves at all.

THE WAY OUT IS INSIDE THE INTERVENTION.

Roll a comb IN PHASE by a LARGE delta -- 12, 15, 18, 21, 24 -- and it slides off its own support
entirely.  L2H15's teeth span -22..-4; rolled by 21 it reads +
-1..+17, sharing NOTHING with where
it started.  So a LARGE IN-PHASE roll delivers an OFF-PHASE-SIZED PERTURBATION.

    Now the two can be compared AT MATCHED ||dout||, WITHIN THE SAME HEAD.

    If damage depends only on perturbation SIZE:
        the in-phase and off-phase points fall on ONE curve.  Phase is irrelevant; the sawtooth
        was comb self-overlap all along.  The comb is inert routing.

    If the model specifically depends on the comb reading IN PHASE:
        at the same ||dout||, in-phase rolls do LESS damage.  The in-phase curve lies BELOW the
        off-phase curve, and the gap is the phase effect -- purged of perturbation size, purged
        of gamma, purged of the selection effect, estimated WITHIN each head.

This is falsifiable in both directions and needs no exponent, no normalisation, and no
cross-head extrapolation.  Every cell can be zero.

CONTROL: the same sweep on the 24 COPIERS.  A copier has no self-overlap (w(d) = 0 for all
d != 0), so its perturbation is flat and its in-phase and off-phase points sit at the SAME
||dout|| already.  Any separation there is the genome's period-3 base statistics, not the comb's
-- and it is the baseline the combs must beat.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.circuits.hf_bert import capture_head_outputs  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, load_examples,
    pick_device, roll_attention,
)

# delta out to 24: in-phase rolls (3,6,...,24) eventually slide a comb entirely off its own
# support, so their ||dout|| climbs into the off-phase range.  That overlap is the experiment.
DELTAS = tuple(range(1, 25))
CHANCE = float(np.log(0.25))
MASK_RATE = 1 / 7
MIN_LEN = 60


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--shift", default="results/phase-lattice/comb_shift.json")
    ap.add_argument("--out", default="results/phase-lattice/matched_perturbation.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off

    S = json.loads((REPO / args.shift).read_text())
    cls = S["classification"]
    COMB = [tuple(map(int, k[1:].split("H"))) for k, v in cls.items() if v["kind"] == "COMB"]
    COPIER = [tuple(map(int, k[1:].split("H"))) for k, v in cls.items() if v["kind"] == "COPIER"]
    HEADS = sorted(COMB) + sorted(COPIER)
    print(f"{len(COMB)} combs + {len(COPIER)} copiers   deltas 1..{max(DELTAS)}   device={dev}")

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n * 4, task=task,
                         tag=f"matched-{args.split}", min_effect=None)
    X = [x for x in X if x.gene_type == "protein_coding" and x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN][:args.n]
    SEQLEN = len(X[0].sequence)
    print(f"{len(X)} protein-coding exons   "
          f"est. {len(HEADS)*len(DELTAS)*len(X)*0.035/60:.0f} min\n")

    mrng = np.random.default_rng(args.seed)
    MASKS = [mrng.random(len(x.sequence)) < MASK_RATE for x in X]

    def region(x):
        return (x.acceptor_pos + 6, x.donor_pos - 6)

    XE = [x for x in X if region(x)[1] - region(x)[0] >= 24]
    dmg: dict = defaultdict(lambda: defaultdict(list))
    nrm: dict = defaultdict(lambda: defaultdict(list))

    nb = (len(X) + task.batch - 1) // task.batch
    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            enc = ad.tokenizer([" ".join(x.sequence) for x in ch], return_tensors="pt",
                               padding=True).to(ad.device)
            pos = []
            for j, gi in enumerate(range(i, i + len(ch))):
                ps = np.flatnonzero(MASKS[gi]).tolist()
                for p in ps:
                    enc["input_ids"][j, p + OFF] = task.mask_id
                pos.append(ps)

            def lp_of(lg):
                ll = torch.log_softmax(lg.float(), dim=-1)
                out = []
                for j, x in enumerate(ch):
                    idx = torch.tensor([p + OFF for p in pos[j]], device=ll.device)
                    tgt = torch.tensor([task.bid[x.sequence[p]] for p in pos[j]],
                                       device=ll.device)
                    v = np.full(len(x.sequence), np.nan)
                    for p, q in zip(pos[j], ll[j, idx, tgt].cpu().numpy()):
                        v[p] = q
                    out.append(v)
                return out

            lg0 = None

            def base():
                nonlocal lg0
                lg0 = ad.model(**enc).logits
                return lg0
            H0 = capture_head_outputs(ad, base)
            lp0 = lp_of(lg0)

            for (l, h) in HEADS:
                for d in DELTAS:
                    lgd = None

                    def rolled():
                        nonlocal lgd
                        with roll_attention(ad, [(l, h)], d, seq_len=SEQLEN, off=OFF):
                            lgd = ad.model(**enc).logits
                        return lgd
                    Hd = capture_head_outputs(ad, rolled)
                    lpd = lp_of(lgd)
                    dn = (H0[l][:, h] - Hd[l][:, h]).norm(dim=-1).cpu().numpy()
                    for j, x in enumerate(ch):
                        a, b = region(x)
                        if b - a < 24:
                            continue
                        dmg[(l, h)][d].append(
                            float(np.nanmean(lp0[j][a:b]) - np.nanmean(lpd[j][a:b])))
                        nrm[(l, h)][d].append(float(np.mean(dn[j, a + OFF:b + OFF])))
            print(f"  batch {i//task.batch+1}/{nb}", end="\r", flush=True)
    print()

    # ---------------------------------------------------------------- the matched comparison
    rows = []
    for (l, h) in HEADS:
        kind = cls[f"L{l}H{h}"]["kind"]
        D = {d: float(np.mean(dmg[(l, h)][d])) for d in DELTAS}
        N = {d: float(np.mean(nrm[(l, h)][d])) for d in DELTAS}
        inp = [d for d in DELTAS if d % 3 == 0]
        offp = [d for d in DELTAS if d % 3]
        nin = [N[d] for d in inp]
        noff = [N[d] for d in offp]
        overlap_lo, overlap_hi = max(min(nin), min(noff)), min(max(nin), max(noff))

        # PAIRED, WITHIN-HEAD: for each in-phase delta, interpolate the off-phase damage curve
        # (damage as a function of ||dout||) at the SAME ||dout||, and take the gap.
        oo = sorted(zip(noff, [D[d] for d in offp]))
        ox = np.array([a for a, _ in oo])
        oy = np.array([b for _, b in oo])
        gaps = []
        for d in inp:
            if not (ox.min() <= N[d] <= ox.max()):
                continue                                   # no matched off-phase point: skip
            pred = float(np.interp(N[d], ox, oy))
            gaps.append(dict(delta=d, n=N[d], d_in=D[d], d_off_pred=pred, gap=pred - D[d]))
        rows.append(dict(layer=l, head=h, kind=kind, damage=D, norm=N,
                         n_in_range=[float(min(nin)), float(max(nin))],
                         n_off_range=[float(min(noff)), float(max(noff))],
                         overlap=[float(overlap_lo), float(overlap_hi)],
                         matched=gaps,
                         mean_gap=float(np.mean([g["gap"] for g in gaps])) if gaps else None))

    print("=" * 100)
    print("MATCHED-PERTURBATION TEST")
    print("  gap = (off-phase damage interpolated at the SAME ||dout||) - (in-phase damage)")
    print("  gap > 0  =>  at equal perturbation, an IN-phase roll does LESS damage. Phase matters.")
    print("  gap ~ 0  =>  damage depends only on perturbation size. The sawtooth was self-overlap.")
    for kind in ("COMB", "COPIER"):
        hs = [r for r in rows if r["kind"] == kind]
        print(f"\n--- {kind} ---")
        print(f"  {'head':8s} {'||dout|| in':>16s} {'||dout|| off':>16s} {'#matched':>9s} "
              f"{'mean gap':>10s}")
        for r in sorted(hs, key=lambda x: (x["layer"], x["head"]))[:26]:
            g = f"{r['mean_gap']:+10.5f}" if r["mean_gap"] is not None else "        --"
            print(f"  L{r['layer']}H{r['head']:<5d} "
                  f"[{r['n_in_range'][0]:5.2f},{r['n_in_range'][1]:5.2f}] "
                  f"  [{r['n_off_range'][0]:5.2f},{r['n_off_range'][1]:5.2f}] "
                  f"{len(r['matched']):9d} {g}")

    cg = [r["mean_gap"] for r in rows if r["kind"] == "COMB" and r["mean_gap"] is not None]
    pg = [r["mean_gap"] for r in rows if r["kind"] == "COPIER" and r["mean_gap"] is not None]
    print("\n" + "-" * 100)
    stats: dict = {}
    if cg and pg:
        diff = float(np.mean(cg) - np.mean(pg))
        pool = np.array(cg + pg)
        nA = len(cg)
        rg = np.random.default_rng(0)
        perm = np.array([np.mean(x[:nA]) - np.mean(x[nA:])
                         for x in (rg.permutation(pool) for _ in range(20000))])
        p = float((1 + np.sum(perm >= diff)) / (1 + len(perm)))
        print(f"  COMB   mean gap = {np.mean(cg):+.5f}   (n={len(cg)} heads)")
        print(f"  COPIER mean gap = {np.mean(pg):+.5f}   (n={len(pg)} heads)")
        print(f"\n  COMB - COPIER = {diff:+.5f}   exact permutation p = {p:.4f}")
        stats = dict(comb_gap=float(np.mean(cg)), copier_gap=float(np.mean(pg)),
                     diff=diff, p=p, n_comb=len(cg), n_copier=len(pg))
        if p < 0.05 and diff > 0:
            print("\n  => AT MATCHED PERTURBATION, IN-PHASE READING IS SPECIFICALLY PROTECTED.")
            print("     The comb is not merely a big lever; the model depends on WHICH PHASE it")
            print("     reads, beyond how much its output moves. THE COMB IS USED.")
        else:
            print("\n  => NO PHASE EFFECT AT MATCHED PERTURBATION.  The damage sawtooth is fully")
            print("     explained by how much a roll moves the head's output -- which for a")
            print("     stride-3 comb is small in phase BY SET ARITHMETIC.  The combs are real")
            print("     STRUCTURE with no demonstrated FUNCTION.")

    out = REPO / args.out
    out.write_text(json.dumps({"n": len(XE), "deltas": list(DELTAS), "stats": stats,
                               "heads": rows}, indent=1))
    ExperimentManifest.create(
        "phase_lattice:matched_perturbation", sys.argv,
        config={"n": len(XE), "deltas": list(DELTAS)},
        notes=["Breaks the phase/perturbation entanglement WITHIN each head by rolling in-phase "
               "far enough (delta up to 24) that the comb slides off its own support and its "
               "in-phase perturbation reaches off-phase magnitudes.",
               "Needs no exponent, no normalisation, no cross-head extrapolation."],
    ).write(out.parent / "matched_perturbation.manifest.json")
    print("=" * 100)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
