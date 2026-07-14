#!/usr/bin/env python
"""IS THE PERIOD-3 CARRIER USED?  Delete it from the position embeddings and find out.

WHERE THE PHASE ACTUALLY LIVES  (established on the weights alone -- no data, no forward passes)
------------------------------------------------------------------------------------------------
The comb heads are content-blind, so their attention logit is the position-position term:

    score(i,j) ~= (W_Q p_i) . (W_K p_j)  =  p_i^T (W_Q^T W_K) p_j

Computing R(k) = mean_i score(i, i+k) from the WEIGHTS ALONE gives a perfect period-3 comb:

    L2H15:  +0.07 +0.07 +0.48 | +0.07 +0.07 +0.48 | +0.07 +0.07 +0.48 | ...

So the heads do not CREATE the phase.  They READ it.  It is stored in the position-embedding
matrix P [1026 x 512], as a PERIOD-3 CARRIER:

  * 1026 = 3 x 342, so period 3 is an exact Fourier bin.
  * bins 341/342/343 carry 220x / 289x / 224x the median bin power.  The band 339-345 carries
    847x.  It is the ONLY high-frequency structure in P's top-10 modes -- every other strong
    mode is a smooth, low-frequency component (periods 1026, 513, 256, 171, 128, 103, 51, 22).
  * it is a CARRIER WITH SLOW PHASE MODULATION: the mod-3 contrast has constant magnitude along
    the whole sequence (1.02-1.11) while its DIRECTION slowly precesses (cos with the global
    direction wanders 0.38 -> 0.78).  That modulation is what puts sidebands at 342 +- 1, and it
    is why deleting bin 342 alone removes only 34% of the band.

  * DELETE THE BAND (7 of 513 bins) AND EVERY COMB IN THE MODEL COLLAPSES:
        comb amplitude retained  =   3.4%
    DELETE AN ENERGY-MATCHED BAND ELSEWHERE AND THEY ARE UNTOUCHED:
        comb amplitude retained  =  99.8%

THIS SCRIPT ASKS THE ONLY QUESTION LEFT: IS THE CARRIER *USED*?
--------------------------------------------------------------
Delete it, and measure what the model can no longer predict.

Why this is the right intervention, where rolling attention was not:
  * NO comb self-overlap.  Rolling a stride-3 comb by 3 slides it onto itself, so its output
    barely moves -- a sawtooth guaranteed by set arithmetic, on white noise, with no model at
    all (tests/test_phase_shift_memorials.py).  There is no such artefact here.
  * NO selection effect.  We are not intervening on heads that were CHOSEN for phase purity.
  * NO perturbation/phase entanglement.  Rolling a comb off-phase necessarily delivers a bigger
    perturbation, and the two cannot be separated within a head (corr = +0.83).  A Fourier band
    is deleted once, globally, and the ENERGY-MATCHED CONTROL BANDS subtract the rest.

THE DESIGN (a 2x2 with an empirical null)
    treatment : delete P's period-3 band (339-345)
    control   : delete each of N energy-matched bands elsewhere in the spectrum
    regions   : CDS-exon interior  vs  intron interior
    readout   : mean log-prob of the TRUE base at masked positions (Bernoulli 1/7)

    damage(arm, region) = LL(intact, region) - LL(arm, region)
    DiD(arm)            = damage(arm, exon) - damage(arm, intron)

    DiD(period-3) >> DiD(control bands)  ->  the carrier is used FOR CODING SEQUENCE.
    DiD(period-3) ~  DiD(control bands)  ->  it is a positional habit with no codon role.
    Both are informative.  Every cell can be zero.

AND THE GATE THAT R13 DID NOT HAVE.  Deleting a Fourier band could simply wreck the model, in
which case the readout measures nothing (that is exactly how the 42-head ablation produced a
void null: it drove SpliceBERT to chance, log-prob -1.377 vs ln(1/4) = -1.386).  MLM health is
reported for EVERY arm, next to every number, and an arm below the floor is not interpreted.
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

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot, load_adapter, load_examples,
    pick_device,
)

BAND = list(range(339, 346))          # the period-3 carrier + its modulation sidebands
CHANCE = float(np.log(0.25))
MASK_RATE = 1 / 7
MIN_LEN = 60
HEALTH_FLOOR = 0.5                    # keep >= 50% of intact headroom over chance, or don't read it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=384)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--controls", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/phase-lattice/carrier_surgery.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off

    emb = ad.model.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    P = P0.float().cpu().numpy()
    N, _ = P.shape
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    pw = (np.abs(F) ** 2).sum(1)
    W = len(BAND)
    target = float(pw[BAND].sum())

    def make(bins):
        G = F.copy()
        for b in bins:
            G[b] = 0
        return torch.tensor(np.fft.irfft(G, n=N, axis=0) + mu,
                            dtype=P0.dtype, device=P0.device)

    # ENERGY-MATCHED CONTROL BANDS: same width, total power within 25% of the period-3 band,
    # nowhere near period 3 or its harmonics.  These are the null: "deleting SOMETHING from P".
    cand = []
    for lo in range(3, len(pw) - W):
        b = list(range(lo, lo + W))
        per = N / np.mean(b)
        if abs(per - 3.0) < 0.6 or abs(per - 1.5) < 0.3 or abs(per - 6.0) < 0.6:
            continue
        e = float(pw[b].sum())
        if 0.75 * target <= e <= 1.25 * target:
            cand.append((b, e))
    rng = np.random.default_rng(args.seed)
    if len(cand) > args.controls:
        idx = rng.choice(len(cand), args.controls, replace=False)
        cand = [cand[i] for i in idx]
    print(f"period-3 band {BAND[0]}-{BAND[-1]}  power {target:.0f}")
    print(f"{len(cand)} energy-matched control bands (width {W}, power within 25%)\n")

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n * 4, task=task,
                         tag=f"carrier-{args.split}", min_effect=None)
    X = [x for x in X if x.gene_type == "protein_coding" and x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN][:args.n]
    print(f"{len(X)} protein-coding exons, {len(set(x.gene_id for x in X))} genes   device={dev}")
    print(f"est. {(2+len(cand))*len(X)*0.035/60:.0f} min\n")

    mrng = np.random.default_rng(args.seed)
    MASKS = [mrng.random(len(x.sequence)) < MASK_RATE for x in X]

    def regions(x):
        return {"exon": (x.acceptor_pos + 6, x.donor_pos - 6),
                "intron": (x.donor_pos + 8, x.next_acceptor_pos - 8)}

    def evaluate() -> dict:
        """mean log-prob of the true base, per example, per region."""
        out: dict = {"exon": [], "intron": []}
        genes: dict = {"exon": [], "intron": []}
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
                ll = torch.log_softmax(ad.model(**enc).logits.float(), dim=-1)
                for j, x in enumerate(ch):
                    idx = torch.tensor([p + OFF for p in pos[j]], device=ll.device)
                    tgt = torch.tensor([task.bid[x.sequence[p]] for p in pos[j]],
                                       device=ll.device)
                    v = np.full(len(x.sequence), np.nan)
                    for p, q in zip(pos[j], ll[j, idx, tgt].cpu().numpy()):
                        v[p] = q
                    for r, (a, b) in regions(x).items():
                        if b - a >= 24:
                            out[r].append(float(np.nanmean(v[a:b])))
                            genes[r].append(x.gene_id)
        return {"ll": out, "genes": genes}

    ARMS: list[tuple[str, list[int]]] = [("intact", []), ("period3", BAND)]
    ARMS += [(f"ctrl{i}", b) for i, (b, _) in enumerate(cand)]

    res: dict = {}
    for k, (name, bins) in enumerate(ARMS):
        emb.weight.data = P0.clone() if not bins else make(bins)
        r = evaluate()
        res[name] = r
        m = {reg: float(np.mean(r["ll"][reg])) for reg in ("exon", "intron")}
        print(f"  [{k+1}/{len(ARMS)}] {name:9s} exon {m['exon']:+.4f}  intron {m['intron']:+.4f}",
              end="\r", flush=True)
    emb.weight.data = P0.clone()
    print()

    base = {r: float(np.mean(res["intact"]["ll"][r])) for r in ("exon", "intron")}
    hr0 = {r: base[r] - CHANCE for r in base}
    genes = res["intact"]["genes"]

    def summarise(name):
        ll = {r: float(np.mean(res[name]["ll"][r])) for r in ("exon", "intron")}
        dmg = {r: base[r] - ll[r] for r in ll}
        kept = {r: (ll[r] - CHANCE) / hr0[r] for r in ll}
        return dict(ll=ll, damage=dmg, kept=kept, did=dmg["exon"] - dmg["intron"],
                    healthy=bool(min(kept.values()) >= HEALTH_FLOOR))

    S = {n: summarise(n) for n, _ in ARMS if n != "intact"}
    p3 = S["period3"]
    ctrls = [S[n] for n in S if n.startswith("ctrl")]

    print("=" * 96)
    print("CARRIER SURGERY -- delete P's period-3 band and see what the model can no longer do")
    print(f"  intact:  exon {base['exon']:+.4f}   intron {base['intron']:+.4f}"
          f"   [chance {CHANCE:+.4f}]")
    print(f"  headroom over chance:  exon {hr0['exon']:.4f}   intron {hr0['intron']:.4f}")

    print(f"\nGATE (health) -- an arm that has lost the behaviour cannot measure the mechanism:")
    print(f"  period-3 arm keeps {p3['kept']['exon']:.1%} of exon headroom, "
          f"{p3['kept']['intron']:.1%} of intron headroom  "
          f"{'OK' if p3['healthy'] else '*** BELOW FLOOR -- DO NOT INTERPRET ***'}")
    ch = [c for c in ctrls if c["healthy"]]
    print(f"  {len(ch)}/{len(ctrls)} control arms healthy")

    print(f"\n  {'arm':22s} {'exon dmg':>9s} {'intron dmg':>11s} {'DiD':>9s}")
    print(f"  {'PERIOD-3 (339-345)':22s} {p3['damage']['exon']:9.4f} "
          f"{p3['damage']['intron']:11.4f} {p3['did']:9.4f}   <== TREATMENT")
    cd = np.array([c["did"] for c in ctrls])
    ce = np.array([c["damage"]["exon"] for c in ctrls])
    ci_ = np.array([c["damage"]["intron"] for c in ctrls])
    print(f"  {'controls (mean)':22s} {ce.mean():9.4f} {ci_.mean():11.4f} {cd.mean():9.4f}")
    print(f"  {'controls (max)':22s} {ce.max():9.4f} {ci_.max():11.4f} {cd.max():9.4f}")

    p_did = float((1 + np.sum(cd >= p3["did"])) / (1 + len(cd)))
    p_exon = float((1 + np.sum(ce >= p3["damage"]["exon"])) / (1 + len(ce)))
    print(f"\n  add-one empirical p (DiD  vs {len(cd)} energy-matched control bands) = {p_did:.4f}")
    print(f"  add-one empirical p (exon damage)                                    = {p_exon:.4f}")

    # gene-grouped CI on the DiD, paired within example
    de = np.array(res["intact"]["ll"]["exon"]) - np.array(res["period3"]["ll"]["exon"])
    di = np.array(res["intact"]["ll"]["intron"]) - np.array(res["period3"]["ll"]["intron"])
    lo_e, hi_e = gboot(de, genes["exon"])
    lo_i, hi_i = gboot(di, genes["intron"])
    print(f"\n  exon damage   {de.mean():+.4f}  [{lo_e:+.4f}, {hi_e:+.4f}]  (gene-grouped)")
    print(f"  intron damage {di.mean():+.4f}  [{lo_i:+.4f}, {hi_i:+.4f}]")

    print("\n" + "-" * 96)
    if not p3["healthy"]:
        print("  VOID -- deleting the band destroyed the behaviour. This measures nothing (cf. R13).")
    elif p_did < 0.05 and p3["did"] > 0:
        print("  => THE PERIOD-3 CARRIER IS USED, AND SPECIFICALLY FOR CODING SEQUENCE.")
        print("     Deleting 7 of 513 Fourier bins from the position embeddings costs the model")
        print(f"     {p3['damage']['exon']:.4f} nats in exons and only {p3['damage']['intron']:.4f} in introns --")
        print(f"     a gap {len(cd)} energy-matched control bands do not reproduce (p = {p_did:.4f}).")
        print("     The model built a period-3 positional code and uses it to read codons.")
    elif p_exon < 0.05:
        print("  => The carrier is USED, but NOT specifically for coding sequence (the DiD is")
        print("     not an outlier). It reads as a general positional resource, not a codon reader.")
    else:
        print("  => NOT USED.  Deleting the period-3 carrier costs the model no more, in exons")
        print("     specifically, than deleting an energy-matched band anywhere else in P.")
        print("     The carrier is real, striking STRUCTURE with no demonstrated FUNCTION --")
        print("     and unlike R13, the model survived the intervention, so this null has power.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n": len(X), "band": BAND, "band_power": target, "chance": CHANCE,
        "intact": base, "headroom": hr0,
        "period3": p3, "controls": [dict(bins=b, **S[f"ctrl{i}"])
                                    for i, (b, _) in enumerate(cand)],
        "p_did": p_did, "p_exon": p_exon,
        "exon_damage_ci": [lo_e, hi_e], "intron_damage_ci": [lo_i, hi_i],
    }, indent=1))
    ExperimentManifest.create(
        "phase_lattice:carrier_surgery", sys.argv,
        config={"n": len(X), "band": BAND, "controls": len(cand)},
        notes=["The comb heads READ the phase; they do not create it. R(k) computed from the "
               "WEIGHTS ALONE is a perfect period-3 comb, and deleting P's period-3 band "
               "(7 of 513 bins) collapses every comb to 3.4% of amplitude while an "
               "energy-matched band leaves them at 99.8%.",
               "This intervention has no comb self-overlap artefact, no selection effect, and "
               "no perturbation/phase entanglement -- the three things that made the "
               "attention-roll design uninterpretable.",
               "MLM health is gated: an arm that lost the behaviour is not interpreted (R13)."],
    ).write(out.parent / "carrier_surgery.manifest.json")
    print("=" * 96)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
