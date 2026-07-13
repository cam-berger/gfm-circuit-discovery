#!/usr/bin/env python
"""IS THE CODON COMB USED?   comb heads vs sharp copiers, matched on the metric that lied.

TWO RETRACTIONS PRECEDE THIS SCRIPT.  Read them or you will re-run them.

R13 -- the ablation test of the lattice is VOID.  Ablating the 42 "lattice" heads drove
SpliceBERT to CHANCE-level nucleotide prediction (log-prob -1.377; chance = ln(1/4) = -1.386;
intact -1.120): 97% of its entire headroom.  The period-3 statistic was read off a model that
had lost the behaviour, and the "null" passed because all 15 random 42-head controls were
equally dead.
    >>> AN INTERVENTION THAT DESTROYS THE BEHAVIOUR CANNOT MEASURE THE MECHANISM. <<<

R14 -- there is no 42-head lattice.  The inventory scored heads by the fraction of local
attention mass on one residue class (mod 3).  THAT STATISTIC IS MAXIMISED BY A HEAD THAT ATTENDS
TO EXACTLY ONE POSITION: one offset has one remainder, so its "phase purity" is 1.000 by
arithmetic.  corr(purity, effective support) = -0.767.  All ten of the highest-purity heads are
single-offset COPIERS (L0H4 reads offset -1; -1 mod 3 = 2; purity 1.000).  24 of the 42 have
effective support < 2.  The "three phases covered, balanced 14/16/12" was a set of sharp heads
whose offsets happen to spread over three residue classes -- which is what phase-INDIFFERENT
sharp heads do.

WHAT SURVIVES: SEVEN GENUINE COMBS.
    L1H12  offsets -23,-20,-17,-14,-11   (gaps 3,3,3,3)   eff. support 21.2
    L1H10  offsets  -7, -4, -1, +2, +5   (gaps 3,3,3,3)   eff. support 16.8
    L2H15  offsets -16,-13,-10, -7, -4   (gaps 3,3,3,3)   eff. support 27.3
    L0H11  offsets  +6, +9,+12,+15,+18   (gaps 3,3,3,3)   eff. support 21.0
    L0H0   offsets -20,-17,-14,-11, -8   (gaps 3,3,3,3)   eff. support 36.8
    L2H2   offsets -12, -9, +6, +9,+12                    eff. support 35.8
    L0H13  offsets -18,-15,-12, -9, -6   (gaps 3,3,3,3)   eff. support 22.9
L2H15 is the coupling circuit's ADDRESSER, found by an ablation scan that knew nothing about
periodicity.  L1H12 is its other upstream head.  (L1H12 is NOT independent evidence -- the whole
lattice thread started from probing it.  L2H15's convergence is.)

THE INTERVENTION.  Roll one head's post-softmax attention along the KEY axis by delta.  Values,
weights, every other head, every MLP: untouched.  The head still averages REAL value vectors
with ITS OWN profile -- just aimed delta nt off.  Its read of ITSELF is restored exactly (see
circuit.py: otherwise a tooth slides onto the query's own [MASK] at one phase class, which
inflates the sawtooth for free).
    delta = 3, 6, 9      -> the comb still reads its own phase.  IN PHASE.
    delta = 1,2,4,5,7,8  -> it reads a different phase.          OFF PHASE.

*** THE RAW SAWTOOTH IS GUARANTEED AND MEANS NOTHING. ***
    E|| out^d - out^0 ||^2  =  2 sigma^2 [ w(0) - w(d) ],   w(t) = sum_k a_k a_{k+t}
A stride-3 comb rolled by 3 slides ONTO ITSELF -- it keeps reading most of the same tokens, so
its output barely moves.  Rolled by 1 it moves to a disjoint set.  This holds on WHITE NOISE
(tests/test_phase_shift_memorials.py).  No genome, no model, no codons required.  The combs were
SELECTED for exactly the property that produces it.

SO THE ONLY REPORTABLE STATISTIC IS DAMAGE PER UNIT OF PERTURBATION DELIVERED:
    damage ~ grad(L) . dout    =>    damage / ||dout|| ~ ||grad|| cos(theta)
"is the model hurt MORE THAN THE SIZE OF THE PERTURBATION EXPLAINS?"

AND THE CONTROL THE OLD DESIGN COULD NOT HAVE HAD -- THE COPIERS.
A single-tooth head has w(t) = delta_{t,0}, so w(d) = 0 for EVERY d != 0:
    ITS PERTURBATION SIZE IS FLAT IN delta.  IT CANNOT PRODUCE A COMB-OVERLAP SAWTOOTH.
Yet it scores purity 1.000 -- identical to the combs -- on the metric that built the lattice.

    COMBS vs COPIERS: matched on the nominal statistic, OPPOSITE in structure.

If the combs show a NORMALISED phase sawtooth and the copiers do not, no selection effect
explains it.  If both do, or neither does, the comb is inert routing and we say so.
The copiers' FLAT perturbation curve is also the manipulation check: if it is not flat, the
confound algebra above is wrong and every conclusion here must be revisited.
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

SPAN = 24
DELTAS = tuple(range(0, 10))                                  # 0..9 -> THREE in-phase points
INPHASE = tuple(d for d in DELTAS if d and d % 3 == 0)        # 3 6 9
OFFPHASE = tuple(d for d in DELTAS if d and d % 3)            # 1 2 4 5 7 8
CHANCE = float(np.log(0.25))
MASK_RATE = 1 / 7        # BERNOULLI, not stride: a stride-7 mask collides with a comb by CRT
MIN_LEN = 60
HEALTH_FLOOR = 0.5
MIN_PERT = 0.02

# damage(d) = c + a1*d + a2*d^2 + b*1[d mod 3 != 0].  QUADRATIC trend: the phase-blind damage
# curve is visibly non-linear, and with a LINEAR trend `b` becomes a curvature detector rather
# than a phase detector.  delta = 1..9 gives 9 points for 4 parameters.
_D = np.array([[1.0, d, d * d, float(d % 3 != 0)] for d in DELTAS if d])
assert np.linalg.matrix_rank(_D) == 4, "b is not identifiable from the trend -- ABORT"


def fit_b(dmg: dict[int, float]) -> float:
    y = np.array([dmg[d] for d in DELTAS if d])
    coef, *_ = np.linalg.lstsq(_D, y, rcond=None)
    return float(coef[3])


def classify(prof: np.ndarray) -> dict:
    """What KIND of head is this, structurally?  Never mind the metric that lied."""
    p = prof.copy()
    p[SPAN] = 0.0                                             # drop self
    tot = p.sum()
    if tot <= 0:
        return dict(kind="dead", eff=0.0, purity=0.0, teeth=[], span=0)
    q = p / tot
    ent = -np.sum(q[q > 0] * np.log(q[q > 0]))
    eff = float(np.exp(ent))                                  # effective # of offsets attended
    offs = np.arange(-SPAN, SPAN + 1)
    phase_mass = [float(q[(offs % 3 == r) & (offs != 0)].sum()) for r in range(3)]
    purity = max(phase_mass)
    teeth = [int(o) for o in offs[q >= 0.03]]                 # offsets carrying >= 3% of mass
    dom = int(np.argmax(phase_mass))

    # A COMB means what it says: several teeth, ALL congruent mod 3.
    # An 80%-of-mass version of this test admits L2H14 (teeth -16,-15,-14) and L4H13 (+2,+3,+4)
    # -- ADJACENT offsets, residues 2/0/1, the exact opposite of a stride-3 comb.  Congruence is
    # required of every tooth, not of most of the mass.  (`%` in Python is non-negative, so
    # -16 % 3 == 2 as intended.)
    congruent = bool(teeth) and all(o % 3 == teeth[0] % 3 for o in teeth)
    span = int(max(teeth) - min(teeth)) if teeth else 0

    if eff >= 3.0 and len(teeth) >= 3 and congruent and span >= 6 and purity >= 0.60:
        kind = "COMB"
    elif eff < 2.0:
        kind = "COPIER"                                       # one offset. purity 1.0 for free.
    elif eff >= 10.0 and purity < 0.50:
        kind = "DIFFUSE"
    else:
        kind = "other"
    # a continuous version, so the headline does not hang on a threshold:
    comb_score = float(purity * np.log(max(eff, 1.0)))
    return dict(kind=kind, eff=eff, purity=float(purity), teeth=teeth, span=span,
                congruent=congruent, comb_score=comb_score, dom_phase=dom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/phase-lattice/comb_shift.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    OFF, L, H = task.off, ad.num_layers, ad.num_heads

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n * 4, task=task,
                         tag=f"combshift-{args.split}", min_effect=None)
    X = [x for x in X
         if x.gene_type == "protein_coding"                   # gtf.py:113 never reads a CDS line
         and x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN][:args.n]
    SEQLEN = len(X[0].sequence)
    print(f"{len(X)} protein-coding exons, {len(set(x.gene_id for x in X))} genes   device={dev}")

    # ---------------------------------------------------------------- classify every head
    print("\nclassifying all 96 heads by ATTENTION PROFILE (not by the metric that lied)...")
    prof = np.zeros((L, H, 2 * SPAN + 1))
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for i in range(0, min(len(X), 8), task.batch):
            ch = X[i:i + task.batch]
            enc = task.encode([x.sequence for x in ch], [x.donor_pos for x in ch])
            A = ad.model(**enc, output_attentions=True).attentions
            S = enc["input_ids"].shape[1]
            qs = rng.integers(SPAN + 4, S - SPAN - 4, size=32)
            for l in range(L):
                Al = A[l].float().cpu().numpy()
                for h in range(H):
                    for b in range(len(ch)):
                        for p in qs:
                            prof[l, h] += Al[b, h, p, p - SPAN:p + SPAN + 1]

    cls = {(l, h): classify(prof[l, h]) for l in range(L) for h in range(H)}
    COMB = [k for k, v in cls.items() if v["kind"] == "COMB"]
    COPIER = [k for k, v in cls.items() if v["kind"] == "COPIER"]
    DIFFUSE = [k for k, v in cls.items() if v["kind"] == "DIFFUSE"]
    HEADS = COMB + COPIER + DIFFUSE

    print(f"\n  COMB    {len(COMB):2d}  spread over >=3 offsets, all congruent mod 3")
    for k in sorted(COMB):
        c = cls[k]
        print(f"    L{k[0]}H{k[1]:<3d} purity {c['purity']:.3f}  eff {c['eff']:5.1f}  "
              f"teeth {c['teeth'][:7]}")
    print(f"\n  COPIER  {len(COPIER):2d}  ONE offset -- purity 1.000 by arithmetic. "
          f"w(d)=0 for all d != 0 => FLAT perturbation. THE CONTROL.")
    print(f"  DIFFUSE {len(DIFFUSE):2d}  spread, no phase preference")
    print(f"\n  {len(HEADS)} heads x {len(DELTAS)-1} deltas x {len(X)} exons  "
          f"=> est. {len(HEADS)*(len(DELTAS)-1)*len(X)*0.035/60:.0f} min\n")

    def regions(x):
        return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
                "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}

    REG_EX = {r: [x for x in X if regions(x)[r][1] - regions(x)[r][0] >= 24]
              for r in ("exon", "intron")}
    REG_GENES = {r: [x.gene_id for x in REG_EX[r]] for r in REG_EX}

    # BERNOULLI mask, FIXED per example and IDENTICAL across every arm (so damage is paired).
    # A stride-7 mask is NOT safe here: a read at offset k is masked iff k + delta = 0 (mod 7),
    # which is independent of the query -> a SYSTEMATIC, delta-dependent count of [MASK] reads
    # that aliases into the mod-3 indicator (for L1H12: 1 masked read at every in-phase delta,
    # 0.5 on average off-phase).  Bernoulli gives P(read masked) = 1/7 for every k and delta.
    mrng = np.random.default_rng(args.seed)
    MASKS = [mrng.random(len(x.sequence)) < MASK_RATE for x in X]
    for m in MASKS:
        if not m.any():
            m[0] = True

    dmg_l: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    nrm_l: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    ll_l: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    def masked_enc(ch, ids):
        enc = ad.tokenizer([" ".join(x.sequence) for x in ch], return_tensors="pt",
                           padding=True).to(ad.device)
        pos = []
        for j, gi in enumerate(ids):
            ps = np.flatnonzero(MASKS[gi]).tolist()
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

    nb = (len(X) + task.batch - 1) // task.batch
    step = 0
    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            ids = list(range(i, min(i + task.batch, len(X))))
            enc, pos = masked_enc(ch, ids)

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
                    dn = (H0[l][:, h] - Hd[l][:, h]).norm(dim=-1).cpu().numpy()

                    for j, x in enumerate(ch):
                        for r, (a, b) in regions(x).items():
                            if b - a < 24:
                                continue
                            dmg_l[(l, h)][r][d].append(
                                float(np.nanmean(lp0[j][a:b]) - np.nanmean(lpd[j][a:b])))
                            nrm_l[(l, h)][r][d].append(float(np.mean(dn[j, a + OFF:b + OFF])))
                            ll_l[(l, h)][r][d].append(float(np.nanmean(lpd[j][a:b])))
            step += 1
            print(f"  batch {step}/{nb}", end="\r", flush=True)
    print()

    rows = []
    for (l, h) in HEADS:
        c = cls[(l, h)]
        rec = dict(layer=l, head=h, kind=c["kind"], purity=c["purity"], eff=c["eff"],
                   teeth=c["teeth"])
        for r in ("exon", "intron"):
            dmg = {d: float(np.mean(dmg_l[(l, h)][r][d])) for d in DELTAS if d}
            nrm = {d: float(np.mean(nrm_l[(l, h)][r][d])) for d in DELTAS if d}
            nd = {d: (dmg[d] / nrm[d] if nrm[d] > 1e-9 else 0.0) for d in dmg}
            hr = {d: float(np.mean(ll_l[(l, h)][r][d])) - CHANCE for d in DELTAS}
            kept = min(hr[d] for d in DELTAS if d) / max(hr[0], 1e-9)
            pert = float(np.mean(list(nrm.values())))
            # the perturbation sawtooth itself -- the MANIPULATION CHECK, not a result
            pin = float(np.mean([nrm[d] for d in INPHASE]))
            pof = float(np.mean([nrm[d] for d in OFFPHASE]))
            rec[r] = dict(damage=dmg, norm=nrm, ndamage=nd,
                          b_raw=fit_b(dmg), b_norm=fit_b(nd),
                          pert_in=pin, pert_off=pof, pert_ratio=pof / max(pin, 1e-9),
                          kept=float(kept), pert=pert,
                          healthy=bool(kept >= HEALTH_FLOOR and pert >= MIN_PERT))
        rows.append(rec)

    def sel(kind, r="exon", key="b_norm"):
        return [x[r][key] for x in rows if x["kind"] == kind and x[r]["healthy"]]

    print("=" * 98)
    print("COMB-SHIFT -- is the codon comb USED?")
    nh = sum(1 for x in rows if not x["exon"]["healthy"])
    print(f"\nGATE health: {len(rows)-nh}/{len(rows)} heads keep >= {HEALTH_FLOOR:.0%} of intact "
          f"headroom at every delta." + (f"  EXCLUDED: {nh}" if nh else ""))

    print("\nMANIPULATION CHECK -- perturbation size ||dout||, off-phase / in-phase.")
    print("  The algebra says: COMBS sawtooth (ratio > 1); COPIERS are FLAT (ratio ~ 1.00),")
    print("  because a single-tooth head has w(d) = 0 for every d != 0.")
    for k in ("COMB", "COPIER", "DIFFUSE"):
        v = [x["exon"]["pert_ratio"] for x in rows if x["kind"] == k and x["exon"]["healthy"]]
        if v:
            print(f"    {k:8s} n={len(v):2d}   ||dout|| off/in = {np.mean(v):.3f} "
                  f"[{np.min(v):.2f}, {np.max(v):.2f}]")
    cr = [x["exon"]["pert_ratio"] for x in rows if x["kind"] == "COPIER" and x["exon"]["healthy"]]
    if cr and abs(np.mean(cr) - 1.0) > 0.10:
        print(f"    *** COPIERS ARE NOT FLAT ({np.mean(cr):.3f}). The confound algebra is WRONG")
        print("        and every conclusion below must be revisited. ***")

    print("\nRAW sawtooth b  (GUARANTEED for a comb by self-overlap -- NOT evidence):")
    for k in ("COMB", "COPIER", "DIFFUSE"):
        v = sel(k, "exon", "b_raw")
        if v:
            print(f"    {k:8s} b_raw = {np.mean(v):+.4f}")

    print("\nNORMALISED sawtooth b = damage per unit perturbation  -- THE RESULT:")
    print(f"    {'':8s} {'exon':>18s} {'intron':>18s}")
    for k in ("COMB", "COPIER", "DIFFUSE"):
        ve, vi = sel(k, "exon"), sel(k, "intron")
        if ve:
            print(f"    {k:8s} {np.mean(ve):+18.5f} {np.mean(vi):+18.5f}   n={len(ve)}")

    ve, vc = sel("COMB", "exon"), sel("COPIER", "exon")
    print("\n" + "-" * 98)
    stats: dict = {}
    if ve and vc:
        rngb = np.random.default_rng(0)
        diff = float(np.mean(ve) - np.mean(vc))

        # EXACT LABEL PERMUTATION.  With ~7 combs a bootstrap on 7 points is thin; permuting the
        # COMB/COPIER label over the pooled heads is exact and assumes nothing.
        pool = np.array(ve + vc)
        nA = len(ve)
        perm = []
        for _ in range(20000):
            idx = rngb.permutation(len(pool))
            perm.append(pool[idx[:nA]].mean() - pool[idx[nA:]].mean())
        perm = np.array(perm)
        p_add1 = float((1 + np.sum(perm >= diff)) / (1 + len(perm)))
        lo, hi = (float(np.quantile(perm, 0.025)), float(np.quantile(perm, 0.975)))

        print(f"  THE TEST:  b_norm(COMB) - b_norm(COPIER) = {diff:+.5f}")
        print(f"    exact label permutation (20k):  null 95% band [{lo:+.5f}, {hi:+.5f}]   "
              f"add-one p = {p_add1:.4f}")
        print(f"    matched on the metric that lied: purity COMB "
              f"{np.mean([x['purity'] for x in rows if x['kind']=='COMB']):.3f} vs COPIER "
              f"{np.mean([x['purity'] for x in rows if x['kind']=='COPIER']):.3f}")

        # and the threshold-free version: regress b_norm on a CONTINUOUS comb score, all heads,
        # with layer fixed effects -- so the headline does not hang on where "COMB" was cut.
        okh = [x for x in rows if x["exon"]["healthy"]]
        Y = np.array([x["exon"]["b_norm"] for x in okh])
        Cs = np.array([cls[(x["layer"], x["head"])]["comb_score"] for x in okh])
        Lay = np.array([x["layer"] for x in okh])
        Dm = np.column_stack([np.ones(len(okh)), Cs]
                             + [(Lay == k).astype(float) for k in range(1, L)])
        co, *_ = np.linalg.lstsq(Dm, Y, rcond=None)
        bsl = []
        for _ in range(4000):
            ii = rngb.choice(len(okh), len(okh), replace=True)
            try:
                c2, *_ = np.linalg.lstsq(Dm[ii], Y[ii], rcond=None)
                bsl.append(float(c2[1]))
            except np.linalg.LinAlgError:
                pass
        slo, shi = float(np.quantile(bsl, 0.025)), float(np.quantile(bsl, 0.975))
        print(f"\n    threshold-free: b_norm ~ comb_score + layer, all {len(okh)} healthy heads")
        print(f"      slope = {float(co[1]):+.5f}   95% CI [{slo:+.5f}, {shi:+.5f}]")

        stats = dict(diff=diff, perm_lo=lo, perm_hi=hi, p_perm=p_add1,
                     slope=float(co[1]), slope_ci=[slo, shi],
                     comb_exon=float(np.mean(ve)), copier_exon=float(np.mean(vc)))
        print("\n  *** THIS SCRIPT DOES NOT RETURN A VERDICT, AND MUST NOT. ***")
        print("  b_norm = damage / ||dout|| ASSUMES damage is LINEAR in perturbation size.")
        print("  It is not: the fitted exponent is gamma = 1.80 [1.29, 2.32], and a comb's")
        print("  off-phase perturbation is ~3.5x its in-phase one BY SET ARITHMETIC.  Convexity")
        print("  alone therefore contributes ~(gamma-1)*ln(3.5) = +1.0 of spurious log-sawtooth --")
        print("  which is most of what the number above is made of.  An earlier version of this")
        print("  script printed 'THE COMB IS USED' here, from exactly that inflated statistic.")
        print("\n  The estimator that fits gamma instead of assuming it:  analyse_comb_shift.py")
        print("  The test that needs no gamma at all:                    run_matched_perturbation.py")
        print("  Read the verdict from THOSE.  Not from here.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n_exons": len(X), "deltas": list(DELTAS), "chance": CHANCE,
        "mask_rate": MASK_RATE, "seed": args.seed,
        "n_comb": len(COMB), "n_copier": len(COPIER), "n_diffuse": len(DIFFUSE),
        "stats": stats,
        "classification": {f"L{l}H{h}": cls[(l, h)] for (l, h) in HEADS},
        "heads": rows,
    }, indent=1))
    ExperimentManifest.create(
        "phase_lattice:comb_shift", sys.argv,
        config={"split": args.split, "n": len(X), "deltas": list(DELTAS),
                "comb": [list(k) for k in COMB], "copier": [list(k) for k in COPIER]},
        notes=[
            "Replaces the VOID ablation test (R13) and the retracted 42-head lattice (R14).",
            "COMBS vs COPIERS are MATCHED on the phase-purity metric (both ~1.0) and OPPOSITE "
            "in structure. A copier's perturbation is FLAT in delta (w(d)=0), so it CANNOT "
            "produce the comb-overlap sawtooth -- which makes it the control the old design "
            "could not have had.",
            "Only the PERTURBATION-NORMALISED sawtooth is evidence. The raw one is guaranteed.",
            "Self-attention is restored after the roll: otherwise a tooth slides onto the "
            "query's own [MASK] at one phase class, inflating the sawtooth for free.",
            "Bernoulli masking, not stride-7: a stride-7 mask collides with a comb by CRT.",
        ],
    ).write(out.parent / "comb_shift.manifest.json")
    print("=" * 98)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
