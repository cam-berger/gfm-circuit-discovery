#!/usr/bin/env python
"""THE CODON-PHASE LATTICE, part 3: is the lattice USED?  (replaces the void R13 test)

WHY THE OBVIOUS EXPERIMENT DOES NOT WORK
----------------------------------------
Ablate the 42 lattice heads, see if codon-periodic prediction dies.  It does not -- and that
answer is WORTHLESS, because the ablation drives SpliceBERT to CHANCE-level nucleotide
prediction (log-prob -1.377; chance = ln(1/4) = -1.386; intact -1.120).  It removes 97% of the
model's entire headroom over chance.  The metric was read off a model that had lost the
behaviour.  That is R13, and it is the third time this project has measured a mechanism on a
model that could no longer perform the task (R8, R12).

    AN INTERVENTION THAT DESTROYS THE BEHAVIOUR CANNOT MEASURE THE MECHANISM.

THE INTERVENTION HERE: ROLL THE ROUTING, KEEP THE VALUES
-------------------------------------------------------
Roll one head's post-softmax attention along the KEY axis by `delta`.  The head still collects
REAL value vectors with ITS OWN weight profile -- just aimed `delta` nt off.  Weights, values,
every other head and every MLP are untouched.  This is the clean dissociation of *where a head
looks* (QK) from *what it collects* (OV) -- and the shuffle test that produced the original
"the lattice reads no bases" claim only ever constrained the former.

  delta = 3, 6  -> the head still reads its own phase, from different positions.  IN PHASE.
  delta = 1,2,4,5 -> the head reads a DIFFERENT phase.  OFF PHASE.

Prediction if the lattice is used: a SAWTOOTH in damage -- dips at delta = 3 and 6.

    damage(delta) = a*delta + b*1[delta mod 3 != 0] + c        (design matrix is full rank)

`a` absorbs any monotone displacement effect; `b` is the phase effect.  No monotone confound
can produce dips at BOTH 3 and 6.

*** AND THAT SAWTOOTH IS WORTHLESS ON ITS OWN.  READ THIS BEFORE READING ANY NUMBER. ***
-------------------------------------------------------------------------------------------
The perturbation SIZE is itself phase-dependent, by construction, for a reason that has nothing
to do with the model:

    out_i^d - out_i^0 = sum_k a_k ( v_{i+k+d} - v_{i+k} )
    E|| . ||^2        = sum_t w(t) [ 2C(t) - C(t+d) - C(t-d) ]

C = autocovariance of the value vectors along the sequence; w(t) = sum_k a_k a_{k+t} = the
autocorrelation of the head's OWN attention profile.  CODING DNA IS PERIOD-3 AUTOCORRELATED, so
C is large at t = 0 mod 3.  For a phase-pure head w is supported on t = 0 mod 3 -- exactly those
lags.  So rolling a phase-pure head OFF phase delivers a BIGGER perturbation than rolling it by
3, *whether or not anything downstream uses phase at all*.

The lattice heads were SELECTED for phase purity.  So raw damage MUST sawtooth for them, in
exons, and not for random heads or in introns -- reproducing every naive gate with no codon
computation anywhere in the model.  A pilot confirms it: rolling 42 RANDOM NON-LATTICE heads
gives a clean exon sawtooth (off-phase 0.087 vs in-phase 0.032) and no intron sawtooth.  A raw
sawtooth is a picture of the GENOME, not of SpliceBERT.

THE FIX: DIVIDE BY THE PERTURBATION ACTUALLY DELIVERED.
    damage ~ grad(L) . delta_out     =>     damage / ||delta_out|| ~ ||grad|| cos(theta)
The normalised quantity asks whether the model is hurt MORE THAN THE SIZE OF THE PERTURBATION
EXPLAINS -- i.e. whether off-phase reads are *specifically* harmful, not merely *bigger*.
Every headline number in this script is normalised.  The raw sawtooth is reported too, and
labelled as expected-by-construction.

AND THE SECOND FIX: ONE HEAD AT A TIME.
Rolling all 42 lattice heads FLOORS the model -- below chance (-1.495 at delta=1, vs chance
-1.386).  It is not damaged, it is anti-predictive: it predicts the base at i+1.  Damage
saturates, the dynamic range is gone, and we would get a false negative that looks exactly like
a true one.  So every head is rolled ALONE.  The model stays healthy (gate below), and every
head becomes one point in the graph that actually decides this:

    THE MONEY GRAPH:  normalised sawtooth amplitude  b_h  vs  the head's PHASE PURITY,
                      across all 96 heads, controlling for layer.

    slope > 0  ->  the more phase-pure a head, the more the model specifically depends on it
                   reading IN phase.  The lattice is USED.
    slope ~ 0  ->  every head that reads DNA shows this; it is a property of the genome's
                   autocorrelation and of nothing SpliceBERT learned.  The lattice is INERT
                   routing -- and this time we would have the power to say so.
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

DELTAS = (0, 1, 2, 3, 4, 5, 6)
OFFPHASE = tuple(d for d in DELTAS if d and d % 3)      # 1 2 4 5
INPHASE = tuple(d for d in DELTAS if d and not d % 3)   # 3 6
CHANCE = float(np.log(0.25))
MASK_STRIDE = 7           # coprime with 3: the mask pattern cannot manufacture a phase signal
MIN_LEN = 60
HEALTH_FLOOR = 0.5        # an arm must keep >= 50% of intact headroom over chance
MIN_PERT = 0.02           # a head delivering no perturbation cannot be normalised by it

# design matrix for damage(d) = c + a*d + b*1[d % 3 != 0], on d = 1..6.  Rank 3 (verified).
_D = np.array([[1.0, d, float(d % 3 != 0)] for d in DELTAS if d])


def fit_b(dmg: dict[int, float]) -> tuple[float, float]:
    """(b, a) -- the phase effect and the displacement slope."""
    y = np.array([dmg[d] for d in DELTAS if d])
    coef, *_ = np.linalg.lstsq(_D, y, rcond=None)
    return float(coef[2]), float(coef[1])


def gboot_b(per_gene: dict, genes: list[str], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Gene-grouped bootstrap of b.  Resample GENES, re-aggregate, RE-FIT.  Never resample rows."""
    rng = np.random.default_rng(seed)
    uq = sorted(set(genes))
    out = []
    for _ in range(n):
        pick = rng.choice(len(uq), size=len(uq), replace=True)
        gs = [uq[i] for i in pick]
        dmg = {}
        for d in DELTAS:
            if d == 0:
                continue
            num = sum(per_gene[g][d][0] for g in gs)
            den = sum(per_gene[g][d][1] for g in gs)
            dmg[d] = num / den if den > 1e-12 else 0.0
        out.append(fit_b(dmg)[0])
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--mask-offset", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--heads", default="all", help="'all' (96) or 'lattice'")
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--inventory", default="results/phase-lattice/phase_inventory.json")
    ap.add_argument("--out", default="results/phase-lattice/phase_shift.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off
    L, H = ad.num_layers, ad.num_heads

    inv = json.loads((REPO / args.inventory).read_text())
    purity = {(r["layer"], r["head"]): r["strength"] for r in inv["heads"]}
    lattice = {(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]}
    HEADS = (sorted(lattice) if args.heads == "lattice"
             else [(l, h) for l in range(L) for h in range(H)])

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n * 3, task=task,
                         tag=f"phaseshift-{args.split}", min_effect=None)
    # PROTEIN-CODING only.  The old script called the exon interior "exon (coding)" and never
    # checked: gtf.py:113 hard-filters to feature == "exon" and never reads a CDS line, so
    # nothing in the pipeline knows coding from UTR.  ~91% of internal exons of protein-coding
    # transcripts are pure CDS (measured on chr8/9), so this filter leaves ~9% UTR dilution --
    # an attenuation toward the null, stated rather than hidden.
    X = [x for x in X
         if x.gene_type == "protein_coding"
         and x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN][:args.n]
    genes = [x.gene_id for x in X]
    SEQLEN = len(X[0].sequence)
    print(f"{len(X)} protein-coding exons ({len(set(genes))} genes)   {len(HEADS)} heads x "
          f"{len(DELTAS)} deltas   device={dev}")
    print(f"est. {len(HEADS) * (len(DELTAS)-1) * len(X) * 0.035 / 60:.0f} min\n")

    def regions(x):
        return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
                "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}

    # WHICH examples contribute to which region, decided ONCE.  The skip rule (b - a >= 24)
    # depends only on the example and the region, never on the head or delta -- so the gene
    # list for the bootstrap is fixed per region and lines up with every accumulator.  (An
    # earlier draft rebuilt it as `ex_genes[:len(...)]`, which silently mis-pairs genes with
    # damages the moment ONE example is short.)
    REG_EX = {r: [x for x in X if regions(x)[r][1] - regions(x)[r][0] >= 24]
              for r in ("exon", "intron")}
    REG_GENES = {r: [x.gene_id for x in REG_EX[r]] for r in REG_EX}
    for r, xs in REG_EX.items():
        print(f"  {r}: {len(xs)}/{len(X)} examples have >= 24nt of region")

    # accumulators:  acc[head][region][delta] -> per-example lists
    dmg_l: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nrm_l: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ll_l: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gene_of: list[str] = []

    def masked_enc(ch):
        enc = ad.tokenizer([" ".join(x.sequence) for x in ch], return_tensors="pt",
                           padding=True).to(ad.device)
        pos = []
        for j, x in enumerate(ch):
            ps = [p for p in range(len(x.sequence)) if p % MASK_STRIDE == args.mask_offset]
            for p in ps:
                enc["input_ids"][j, p + OFF] = task.mask_id
            pos.append(ps)
        return enc, pos

    def logprobs(lg, ch, pos):
        ll = torch.log_softmax(lg.float(), dim=-1)
        out = []
        for j, x in enumerate(ch):
            idx = torch.tensor([p + OFF for p in pos[j]], device=ll.device)
            tgt = torch.tensor([task.bid[x.sequence[p]] for p in pos[j]], device=ll.device)
            v = np.full(len(x.sequence), np.nan)
            for p, q in zip(pos[j], ll[j, idx, tgt].cpu().numpy()):
                v[p] = q
            out.append(v)
        return out

    done = 0
    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            enc, pos = masked_enc(ch)

            cache: dict = {}
            lg0 = None

            def base():
                nonlocal lg0
                lg0 = ad.model(**enc).logits
                return lg0
            H0 = capture_head_outputs(ad, base)
            lp0 = logprobs(lg0, ch, pos)

            for (l, h) in HEADS:
                for d in DELTAS:
                    if d == 0:
                        for j, x in enumerate(ch):
                            for r, (a, b) in regions(x).items():
                                if b - a >= 24:
                                    ll_l[(l, h)][r][0].append(float(np.nanmean(lp0[j][a:b])))
                        continue
                    lgd = None

                    def rolled():
                        nonlocal lgd
                        with roll_attention(ad, [(l, h)], d, seq_len=SEQLEN, off=OFF):
                            lgd = ad.model(**enc).logits
                        return lgd
                    Hd = capture_head_outputs(ad, rolled)
                    lpd = logprobs(lgd, ch, pos)
                    dn = (H0[l][:, h] - Hd[l][:, h]).norm(dim=-1).cpu().numpy()   # [B,S]

                    for j, x in enumerate(ch):
                        for r, (a, b) in regions(x).items():
                            if b - a < 24:
                                continue
                            dmg_l[(l, h)][r][d].append(
                                float(np.nanmean(lp0[j][a:b]) - np.nanmean(lpd[j][a:b])))
                            nrm_l[(l, h)][r][d].append(float(np.mean(dn[j, a + OFF:b + OFF])))
                            ll_l[(l, h)][r][d].append(float(np.nanmean(lpd[j][a:b])))
                done += 1
                print(f"  head {done}/{len(HEADS) * ((len(X) + task.batch - 1) // task.batch)}"
                      f"  L{l}H{h}", end="\r", flush=True)
    print()

    rows = []
    for (l, h) in HEADS:
        rec: dict = {"layer": l, "head": h, "purity": purity[(l, h)],
                     "lattice": (l, h) in lattice}
        for r in ("exon", "intron"):
            dmg = {d: float(np.mean(dmg_l[(l, h)][r][d])) for d in DELTAS if d}
            nrm = {d: float(np.mean(nrm_l[(l, h)][r][d])) for d in DELTAS if d}
            nd = {d: (dmg[d] / nrm[d] if nrm[d] > 1e-9 else 0.0) for d in dmg}
            b_raw, a_raw = fit_b(dmg)
            b_nrm, a_nrm = fit_b(nd)

            # gene-grouped bootstrap of the NORMALISED b: (sum damage, sum norm) per gene
            gl = REG_GENES[r]
            assert len(gl) == len(dmg_l[(l, h)][r][1]), (
                f"gene list ({len(gl)}) does not line up with the damage rows "
                f"({len(dmg_l[(l, h)][r][1])}) for L{l}H{h}/{r} -- the bootstrap would pair "
                "each damage with the WRONG gene"
            )
            pg: dict = defaultdict(lambda: {d: [0.0, 0.0] for d in DELTAS if d})
            for d in DELTAS:
                if not d:
                    continue
                for g, dv, nv in zip(gl, dmg_l[(l, h)][r][d], nrm_l[(l, h)][r][d]):
                    pg[g][d][0] += dv
                    pg[g][d][1] += nv
            lo, hi = gboot_b(pg, gl)

            hr = {d: float(np.mean(ll_l[(l, h)][r][d])) - CHANCE for d in DELTAS}
            kept = min(hr[d] for d in DELTAS if d) / max(hr[0], 1e-9)
            # A head that delivers no perturbation cannot be normalised BY that perturbation:
            # damage/||dout|| is 0/0 and the ratio is noise amplified without bound.  Such heads
            # are EXCLUDED, not quietly divided by epsilon.
            pert = float(np.mean(list(nrm.values())))
            rec[r] = dict(damage=dmg, norm=nrm, ndamage=nd,
                          b_raw=b_raw, a_raw=a_raw, b_norm=b_nrm, b_lo=lo, b_hi=hi,
                          headroom=hr, kept=float(kept), pert=pert,
                          healthy=bool(kept >= HEALTH_FLOOR and pert >= MIN_PERT))
        rec["did_norm"] = rec["exon"]["b_norm"] - rec["intron"]["b_norm"]
        rows.append(rec)

    # ---------------------------------------------------------------- report
    print("=" * 96)
    print("PHASE-SHIFT INTERVENTION -- is the codon-phase lattice USED?")
    print(f"  {len(X)} protein-coding exons, {len(set(genes))} genes, split={args.split}")
    unhealthy = [r for r in rows if not r["exon"]["healthy"]]
    print(f"\nGATE 2 (health): {len(rows)-len(unhealthy)}/{len(rows)} heads keep >= "
          f"{HEALTH_FLOOR:.0%} of intact headroom at every delta.")
    if unhealthy:
        print("  UNHEALTHY (excluded from the fit, not silently kept): "
              + ", ".join(f"L{r['layer']}H{r['head']}({r['exon']['kept']:.0%})"
                          for r in unhealthy[:8]))
    ok = [r for r in rows if r["exon"]["healthy"] and r["intron"]["healthy"]]

    lat = [r for r in ok if r["lattice"]]
    non = [r for r in ok if not r["lattice"]]

    def m(rs, k, r="exon"):
        return float(np.mean([x[r][k] for x in rs])) if rs else float("nan")

    print("\n  RAW sawtooth b (EXPECTED BY CONSTRUCTION -- not evidence of anything):")
    print(f"    lattice heads  b_raw = {m(lat,'b_raw'):+.4f}    non-lattice  {m(non,'b_raw'):+.4f}")
    print("\n  NORMALISED sawtooth b (damage per unit of perturbation delivered) -- THE RESULT:")
    print(f"    lattice heads  b_norm = {m(lat,'b_norm'):+.5f}   non-lattice  "
          f"{m(non,'b_norm'):+.5f}")
    print(f"    intron         b_norm = {m(lat,'b_norm','intron'):+.5f}   non-lattice  "
          f"{m(non,'b_norm','intron'):+.5f}")

    # THE MONEY TEST: does b_norm rise with phase purity, controlling for layer?
    Y = np.array([r["exon"]["b_norm"] for r in ok])
    P = np.array([r["purity"] for r in ok])
    Lay = np.array([r["layer"] for r in ok])
    Dm = np.column_stack([np.ones(len(ok)), P] + [(Lay == k).astype(float) for k in range(1, L)])
    coef, *_ = np.linalg.lstsq(Dm, Y, rcond=None)
    slope = float(coef[1])
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        idx = rng.choice(len(ok), size=len(ok), replace=True)
        try:
            c, *_ = np.linalg.lstsq(Dm[idx], Y[idx], rcond=None)
            bs.append(float(c[1]))
        except np.linalg.LinAlgError:
            pass
    slo, shi = float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))

    print("\n" + "-" * 96)
    print("  THE MONEY TEST:  b_norm(exon)  ~  phase purity  +  layer fixed effects")
    print(f"    slope on purity = {slope:+.5f}   95% CI [{slo:+.5f}, {shi:+.5f}]   n={len(ok)} heads")
    if slo > 0:
        print("    => THE LATTICE IS USED.  The more phase-pure a head, the more the model")
        print("       specifically depends on it reading IN phase -- beyond the size of the")
        print("       perturbation that rolling it off-phase delivers.")
    elif shi < 0:
        print("    => INVERTED.  Phase-pure heads are LESS phase-sensitive once perturbation")
        print("       size is accounted for.  Report as-is; do not spin.")
    else:
        print("    => NO EFFECT.  Normalised phase sensitivity does NOT rise with phase purity.")
        print("       The sawtooth is a property of the GENOME's period-3 autocorrelation, seen")
        print("       through any head that reads DNA -- not of anything SpliceBERT learned to")
        print("       do with it.  The lattice is a positional habit, and THIS TIME the test had")
        print("       the power to say so (contrast R13).")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n_exons": len(X), "n_genes": len(set(genes)),
        "deltas": list(DELTAS), "chance": CHANCE, "mask_offset": args.mask_offset,
        "purity_slope": slope, "purity_slope_ci": [slo, shi], "n_heads_healthy": len(ok),
        "heads": rows,
    }, indent=1))
    ExperimentManifest.create(
        "phase_lattice:phase_shift", sys.argv,
        config={"split": args.split, "n": len(X), "deltas": list(DELTAS),
                "heads": len(HEADS), "mask_offset": args.mask_offset},
        notes=[
            "Replaces the VOID ablation test (R13): that one measured period-3 enrichment on a "
            "model that ablation had driven to chance-level prediction.",
            "ONE HEAD AT A TIME: rolling all 42 lattice heads floors the model BELOW chance "
            "(-1.495 vs ln(1/4) = -1.386), which destroys the dynamic range.",
            "THE RAW SAWTOOTH IS EXPECTED BY CONSTRUCTION -- coding DNA is period-3 "
            "autocorrelated and the heads were SELECTED for phase purity. Only the "
            "PERTURBATION-NORMALISED sawtooth, and its slope against phase purity, are evidence.",
        ],
    ).write(out.parent / "phase_shift.manifest.json")
    print("=" * 96)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
