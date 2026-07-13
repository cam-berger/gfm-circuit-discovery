#!/usr/bin/env python
"""THE CODON-PHASE LATTICE, part 2: is it FUNCTIONAL, or just there?

Part 1 established the structure.  42 of 96 heads put >60% of their local attention on a
single phase (mod 3) -- seven of them at 1.000.  All 42 are RELATIVE ("attend to offsets ≡ r
mod 3 FROM ME"), all three phases are covered (14 / 16 / 12), and the whole thing is absent
from a randomly-initialised model.  It was learned from masked-nucleotide pretraining alone.

But a lattice you can see is not a lattice the model uses.  The attention pattern is
content-blind (it survives sequence shuffling), so on the evidence so far this could be an
inert positional habit.  This script asks whether it does any work.

THE HANDLE.  Coding DNA is 3-periodic -- the wobble base of a codon is far less constrained
than the first two.  So if SpliceBERT's predictions are codon-aware, its per-nucleotide
masked-LM confidence should itself be 3-PERIODIC along a coding exon.  That is measurable
with no annotation at all: take the log-probability of the true base at every position in an
exon and look for power at period 3.

    P3  =  (Fourier power at period 3) / (total non-DC power)     -- bounded, phase-agnostic

THE TEST.
    1. Is the model's confidence 3-periodic inside CODING EXONS?      (if not, stop)
    2. Is it 3-periodic inside INTRONS?                               (it should NOT be)
    3. ABLATE the 42 lattice heads.  Does P3 collapse?
    4. Ablate 42 RANDOM heads.  Does P3 collapse just as much?        <- THE CONTROL

Step 4 is what makes this an experiment.  Deleting 42 of 96 heads damages the model whatever
you delete, so "P3 fell" proves nothing on its own -- exactly the trap that produced R12 in
the coupling study.  The claim requires that the lattice ablation kill the PERIODICITY
specifically, beyond what a size-matched random ablation does, and beyond its damage to
overall accuracy.  Both are reported.

Masking is done in 7 interleaved passes (positions ≡ o mod 7), which covers every position
exactly once and is COPRIME with 3 -- so the masking pattern cannot itself manufacture a
period-3 signal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, ablate, load_adapter,
    load_examples, noop, pick_device,
)

MASK_STRIDE = 7          # coprime with 3: the mask pattern cannot create a period-3 artefact
MIN_LEN = 60


def p3_power(a: np.ndarray) -> float:
    """ENRICHMENT of Fourier power at period 3 over the typical bin.  1.0 = no periodicity.

    NOT the fraction of total power.  The fraction is LENGTH-CONFOUNDED: for a length-M
    segment there are M//2 non-DC bins, so a flat spectrum already puts 1/(M//2) of its power
    at period 3.  Exons here are shorter than introns, so a fraction-based statistic
    manufactures a coding/intron ratio out of nothing but the length difference -- which is
    exactly what the first version of this script did (it printed a 1.96x 'coding-specific'
    periodicity that was an artefact).

    Dividing by the MEDIAN bin power gives a length-independent enrichment, and a robust one:
    the median is not dragged around by whatever else is in the spectrum.
    """
    a = np.asarray(a, dtype=float)
    if len(a) < 15:
        return float("nan")
    a = a - a.mean()
    P = np.abs(np.fft.rfft(a)) ** 2
    f = np.fft.rfftfreq(len(a))
    med = np.median(P[1:])
    if med < 1e-12:
        return float("nan")
    i3 = int(np.argmin(np.abs(f - 1.0 / 3.0)))
    return float(P[i3] / med)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--n-null", type=int, default=15)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--inventory", default="results/phase-lattice/phase_inventory.json")
    ap.add_argument("--out", default="results/phase-lattice/phase_function.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off

    inv = json.loads((REPO / args.inventory).read_text())
    LAT = [(r["layer"], r["head"]) for r in inv["heads"] if r["lattice"]]
    NON = [(l, h) for l in range(ad.num_layers) for h in range(ad.num_heads)
           if (l, h) not in set(LAT)]
    print(f"lattice heads: {len(LAT)}   non-lattice: {len(NON)}   device={dev}")

    X, _ = load_examples(args.gtf, args.fasta, "val", args.n, task=task, tag="phasefn",
                         min_effect=None)
    X = [x for x in X if x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN]
    print(f"{len(X)} exons with >= {MIN_LEN}nt of exon AND {MIN_LEN}nt of intron\n")

    # regions, per example: the EXON interior (coding) and the INTRON interior (non-coding)
    def regions(x):
        e = (x.acceptor_pos + 6, x.donor_pos - 6)                 # exon interior
        i = (x.donor_pos + 8, x.next_acceptor_pos - 8)            # intron interior
        return {"exon (coding)": e, "intron": i}

    def profiles(ctx):
        """Per-position log-prob of the TRUE base, for every example and both regions."""
        out = {k: [] for k in ("exon (coding)", "intron")}
        with torch.no_grad():
            for i in range(0, len(X), task.batch):
                ch = X[i:i + task.batch]
                lp = [np.full(len(x.sequence), np.nan) for x in ch]
                for o in range(MASK_STRIDE):
                    enc = ad.tokenizer([" ".join(x.sequence) for x in ch],
                                       return_tensors="pt", padding=True).to(ad.device)
                    pos = []
                    for j, x in enumerate(ch):
                        ps = [p for p in range(len(x.sequence)) if p % MASK_STRIDE == o]
                        for p in ps:
                            enc["input_ids"][j, p + OFF] = task.mask_id
                        pos.append(ps)
                    with ctx():
                        lg = ad.model(**enc).logits.float()
                    ll = torch.log_softmax(lg, dim=-1)
                    for j, x in enumerate(ch):
                        idx = torch.tensor([p + OFF for p in pos[j]], device=ll.device)
                        tgt = torch.tensor([task.bid[x.sequence[p]] for p in pos[j]],
                                           device=ll.device)
                        vals = ll[j, idx, tgt].cpu().numpy()
                        for p, v in zip(pos[j], vals):
                            lp[j][p] = v
                for j, x in enumerate(ch):
                    for name, (a, b) in regions(x).items():
                        if b - a >= 24:
                            out[name].append(lp[j][a:b])
        return out

    def score(ctx):
        prof = profiles(ctx)
        res = {}
        for name, arrs in prof.items():
            p3 = [p3_power(a) for a in arrs if not np.isnan(a).any()]
            acc = [np.mean(a) for a in arrs if not np.isnan(a).any()]
            res[name] = dict(p3=float(np.nanmean(p3)), p3_sd=float(np.nanstd(p3)),
                             logprob=float(np.mean(acc)), n=len(p3))
        return res

    print("=" * 84)
    print("1+2  IS THE MODEL'S CONFIDENCE 3-PERIODIC, AND IS IT SPECIFIC TO CODING?")
    print("     statistic = period-3 power / MEDIAN bin power.  1.0 means NO periodicity.")
    intact = score(noop)
    for name, r in intact.items():
        print(f"  {name:16s}  P3 enrichment = {r['p3']:.2f}x  +- {r['p3_sd']:.2f}"
              f"   mean log-prob {r['logprob']:+.3f}   n={r['n']}")
    ratio = intact["exon (coding)"]["p3"] / max(intact["intron"]["p3"], 1e-9)
    print(f"  => coding / intron = {ratio:.2f}x")
    if intact["exon (coding)"]["p3"] < 1.5:
        print("     THE MODEL'S CONFIDENCE IS NOT MEANINGFULLY 3-PERIODIC IN CODING EXONS.")
        print("     There is no codon periodicity in the predictions for the lattice to carry,")
        print("     so the ablation below cannot show one.  Report the structure, not a use.")
    elif ratio < 1.3:
        print("     Periodic, but NOT coding-specific -- so it is not codon structure.")
    else:
        print("     The model's confidence IS 3-periodic, and specifically in CODING sequence.")

    print("\n" + "=" * 84)
    print(f"3+4  DOES THE LATTICE CARRY IT?  (ablate {len(LAT)} lattice heads, vs "
          f"{args.n_null} size-matched random sets)")
    lat = score(ablate(ad, LAT))
    print(f"  {'condition':28s} {'P3 (exon)':>10s} {'P3 (intron)':>12s} {'log-prob':>10s}")
    print(f"  {'intact':28s} {intact['exon (coding)']['p3']:10.4f} "
          f"{intact['intron']['p3']:12.4f} {intact['exon (coding)']['logprob']:10.3f}")
    print(f"  {'ablate LATTICE (42)':28s} {lat['exon (coding)']['p3']:10.4f} "
          f"{lat['intron']['p3']:12.4f} {lat['exon (coding)']['logprob']:10.3f}")

    rng = random.Random(0)
    nulls = []
    for i in range(args.n_null):
        S = rng.sample([(l, h) for l in range(ad.num_layers) for h in range(ad.num_heads)],
                       len(LAT))
        r = score(ablate(ad, S))
        nulls.append(r)
        print(f"  {'random ' + str(i):28s} {r['exon (coding)']['p3']:10.4f} "
              f"{r['intron']['p3']:12.4f} {r['exon (coding)']['logprob']:10.3f}", end="\r")
    np3 = np.array([r["exon (coding)"]["p3"] for r in nulls])
    nlp = np.array([r["exon (coding)"]["logprob"] for r in nulls])
    print(f"  {'random 42-head (mean)':28s} {np3.mean():10.4f} "
          f"{np.mean([r['intron']['p3'] for r in nulls]):12.4f} {nlp.mean():10.3f}")
    print(f"  {'random 42-head (sd)':28s} {np3.std():10.4f} {'':12s} {nlp.std():10.3f}")

    p = (1 + int((np3 <= lat["exon (coding)"]["p3"]).sum())) / (len(np3) + 1)
    print(f"\n  add-one p (random sets that killed P3 at least as much): {p:.3f}"
          f"   [{int((np3 <= lat['exon (coding)']['p3']).sum())}/{len(np3)}]")
    # is the periodicity hit HARDER than general competence?
    d_p3 = 1 - lat["exon (coding)"]["p3"] / max(intact["exon (coding)"]["p3"], 1e-9)
    d_lp = lat["exon (coding)"]["logprob"] - intact["exon (coding)"]["logprob"]
    r_p3 = 1 - np3.mean() / max(intact["exon (coding)"]["p3"], 1e-9)
    r_lp = nlp.mean() - intact["exon (coding)"]["logprob"]
    print(f"\n  LATTICE ablation: P3 {-100*d_p3:+6.1f}%   log-prob {d_lp:+.3f} nats")
    print(f"  RANDOM  ablation: P3 {-100*r_p3:+6.1f}%   log-prob {r_lp:+.3f} nats")
    if p <= 0.05 and d_p3 > r_p3 + 0.10:
        print("\n  => THE LATTICE CARRIES THE CODON PERIODICITY.  Deleting it destroys the")
        print("     3-periodic structure of the model's predictions FAR beyond what deleting")
        print("     the same number of other heads does.  It is functional, not decorative.")
    else:
        print("\n  => NOT ESTABLISHED.  A size-matched random ablation does as much damage;")
        print("     the P3 collapse is collateral, not specific.  (This is the R12 trap, and")
        print("     the control is what catches it.)")

    R = dict(n=len(X), lattice=[list(h) for h in LAT], intact=intact, lattice_ablated=lat,
             nulls=[r["exon (coding)"] for r in nulls], p_add_one=float(p),
             coding_intron_ratio=float(ratio))
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "phase_lattice:function", sys.argv,
        config={"n": len(X), "n_lattice": len(LAT), "n_null": args.n_null,
                "mask_stride": MASK_STRIDE},
        notes=["Mask stride 7 is COPRIME with 3, so the masking cannot manufacture period-3.",
               "The size-matched random ablation is the control that makes this an experiment; "
               "without it, 'P3 fell' is just 'we deleted 42 heads'."],
    ).write(out.parent / "phase_function.manifest.json")
    print("=" * 84)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
