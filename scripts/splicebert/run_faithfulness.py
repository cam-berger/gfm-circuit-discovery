#!/usr/bin/env python
"""FAITHFULNESS + COMPLETENESS of the C = {L3H1, L2H15} circuit.

SCOPE — this runner is CURRENT and controls C7. It is not the A6/A12 faithfulness work.
    C7 (this file) is the early two-node {L3H1, L2H15} faithfulness/completeness measurement.
    The later, load-bearing faithfulness claims are C27 (`run_a6_circuit_faithfulness.py`, F=0.005,
    circuit necessary but NON-FAITHFUL) and C33 (`run_a12_faithfulness_size.py`, irreducibly
    distributed: ~72/103 nodes for 90%, no knee). Cite C27/C33 — not C7 — for "how faithful is the
    circuit"; note also R21, which retracted the node-cover reading of the A6/A7/A8/A10 memorials.

So far we have shown that the circuit's components are NECESSARY (ablation) and that the
proposed path MEDIATES the effect (K/V patching, competing-acceptor swaps).  Neither shows
that the circuit EXPLAINS the behaviour.  "These components matter" and "these components
are the mechanism" are different claims, and only the second deserves the word circuit.

Four measurements, each answering a question the others cannot:

  A. UNCONDITIONAL vs CONDITIONAL coupling.
     The mediation scripts select examples with baseline coupling >= 0.3 because a recovery
     FRACTION needs a denominator.  That conditions on the outcome.  The behavioural claim
     must be made on the unfiltered set, and the two numbers must both be on the record.

  B. NECESSITY.  Ablate C.  How much coupling is lost?  Also ablate the 94-head COMPLEMENT:
     if the behaviour survives without every other head, C is where the work happens.

  C. SUFFICIENCY.  Patch C's clean head outputs into the corrupt run.  How much is restored?
     Compare against patching the complement, and against patching all 96 (the ceiling).
     Necessity and sufficiency DISSOCIATE here -- L2H15 is necessary and restores nothing --
     so reporting only one of them would misdescribe the circuit.

  D. CIRCUIT-SIZE CURVES.  Order heads by their val ablation effect (a FROZEN order, read
     from ablation_scan.json -- not recomputed here).  Then sweep k:
         necessity(k)   : ablate the top-k          -> how fast does the coupling die?
         sufficiency(k) : ablate everything EXCEPT the top-k -> how fast does it come back?
     The knee of the sufficiency curve is the circuit's size.  MLM health is reported at
     every point, because "keep 2 heads, ablate 94" wrecks the language model and a
     coupling measured in a wrecked model needs that caveat attached, not buried.

Mean-ablation is applied identically to the clean and corrupt runs, so damage common to
both cancels in the DIFFERENCE.  That is what makes a coupling measurable at all in a
94-head-ablated model -- and it is also why MLM health must be reported next to it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    ADDRESSER, CIRCUIT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask,
    ablate, capture_head_outputs, complement, gboot, gboot_ratio, load_adapter,
    load_examples, noop, patch_heads, pick_device, ratio_of_means,
)

SWEEP = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]


def denoise(task, exs, heads):
    """Patch clean head outputs into the corrupt run.  Returns (numerator, denominator)."""
    ad = task.ad
    num: list[float] = []
    dnm: list[float] = []
    with torch.no_grad():
        for i in range(0, len(exs), task.batch):
            cl, co, dn = task.pair(exs[i:i + task.batch])
            cc = capture_head_outputs(ad, lambda: ad.model(**cl))
            mc = task.margin(ad.model(**cl).logits.float(), dn)
            mk = task.margin(ad.model(**co).logits.float(), dn)
            with patch_heads(ad, cc, heads):
                mp = task.margin(ad.model(**co).logits.float(), dn)
            num += (mp - mk).tolist()
            dnm += (mc - mk).tolist()
    return np.asarray(num), np.asarray(dnm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-curve", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--scan", default="results/splicebert_coupling/development/ablation_scan.json")
    ap.add_argument("--out", default="results/splicebert_coupling/faithfulness.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev)
    task = CouplingTask(ad, batch=args.batch)
    R: dict = {"split": args.split}

    # FROZEN head order -- read, never recomputed.  Recomputing it here would re-select the
    # circuit on the same data used to score it.
    scan = json.loads((REPO / args.scan).read_text())
    order = [(h["layer"], h["head"]) for h in scan["heads"]]
    if len(order) != ad.num_layers * ad.num_heads:
        raise SystemExit(f"ablation_scan.json has {len(order)} heads, expected 96")
    if tuple(order[:2]) != (READER, ADDRESSER):
        raise SystemExit(f"frozen order does not start with the circuit: {order[:2]}")
    print(f"frozen head order from {args.scan}: top-4 = {order[:4]}   device={dev}")

    CIRC = list(CIRCUIT)
    COMP = complement(ad, CIRC)
    L3_LAYER = READER[0]

    # ---- A. unconditional vs conditional ------------------------------------------------
    UNC, unc_eff = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                                 tag=f"faith-unc-{args.split}", min_effect=None)
    ug = [x.gene_id for x in UNC]
    ulo, uhi = gboot(unc_eff, ug)
    frac_pos = float(np.mean(unc_eff > 0.3))
    print(f"\nA. COUPLING on {len(UNC)} UNCONDITIONAL {args.split} exons")
    print(f"     mean {unc_eff.mean():+.3f}  [{ulo:+.3f},{uhi:+.3f}]   median {np.median(unc_eff):+.3f}")
    print(f"     {100*frac_pos:.0f}% of exons exceed the 0.3 conditioning threshold")
    print("     (the mediation numbers below are CONDITIONAL on that threshold -- a ratio")
    print("      needs a denominator.  These two means are not interchangeable.)")
    R["unconditional"] = dict(n=len(UNC), mean=float(unc_eff.mean()), ci=[ulo, uhi],
                              median=float(np.median(unc_eff)), frac_above_threshold=frac_pos)

    X, eff = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                           tag=f"faith-{args.split}", min_effect=0.3)
    genes = [x.gene_id for x in X]
    clo, chi = gboot(eff, genes)
    print(f"     conditional (>=0.3): n={len(X)}  mean {eff.mean():+.3f} [{clo:+.3f},{chi:+.3f}]")
    R["conditional"] = dict(n=len(X), mean=float(eff.mean()), ci=[clo, chi])

    CURVE = X[:args.n_curve]
    cgenes = [x.gene_id for x in CURVE]

    # ---- B. necessity -------------------------------------------------------------------
    base = task.coupling(X)
    h_base = task.mlm_health(CURVE)
    print(f"\nB. NECESSITY   baseline coupling {base.mean():+.3f}   MLM health {h_base:.4f}")
    nec: dict = {}
    for name, heads in [("L3H1", [READER]), ("L2H15", [ADDRESSER]),
                        ("circuit C = {L3H1, L2H15}", CIRC),
                        ("complement (94 heads)", COMP)]:
        c = task.coupling(X, ablate(ad, heads))
        h = task.mlm_health(CURVE, ablate(ad, heads))
        lo, hi = gboot(base - c, genes)
        nec[name] = dict(coupling=float(c.mean()), lost=float((base - c).mean()),
                         lost_ci=[lo, hi], frac_lost=float((base - c).mean() / base.mean()),
                         mlm_health=h, n_heads=len(heads))
        print(f"     ablate {name:26s} coupling {c.mean():+.3f}  "
              f"lost {100*(base-c).mean()/base.mean():5.1f}%  health {h:.4f}")
    R["baseline"] = dict(coupling=float(base.mean()), mlm_health=h_base)
    R["necessity"] = nec

    # ---- B2. COMPLETENESS, done in a way that can actually succeed ----------------------
    # "Keep only C, ablate the other 94" is very nearly a vacuous test: L3H1 reads a residual
    # stream that the layer-0..2 heads BUILD, so deleting them deletes its input and the
    # coupling dies for a reason that has nothing to do with whether C is the circuit.  A
    # circuit is a path THROUGH a network, not an island in it.  The test that can succeed
    # keeps C together with its upstream cone and ablates everything else.
    UPSTREAM = [(l, h) for l in range(L3_LAYER) for h in range(ad.num_heads)]
    L3_OTHERS = [(L3_LAYER, h) for h in range(ad.num_heads) if h != READER[1]]
    DOWNSTREAM = [(l, h) for l in range(L3_LAYER + 1, ad.num_layers)
                  for h in range(ad.num_heads)]
    print("\nB2. COMPLETENESS  (ablate everything OUTSIDE the circuit, keeping its inputs)")
    comp2: dict = {}
    for name, heads in [
        ("layer-3 heads except L3H1 (15)", L3_OTHERS),
        ("all layer-4+5 heads (32)", DOWNSTREAM),
        ("L3-others + L4 + L5 (47)  <- keep C + upstream cone", L3_OTHERS + DOWNSTREAM),
    ]:
        c = task.coupling(X, ablate(ad, heads))
        h = task.mlm_health(CURVE, ablate(ad, heads))
        lo, hi = gboot(c, genes)
        comp2[name] = dict(coupling=float(c.mean()), ci=[lo, hi],
                           frac_retained=float(c.mean() / base.mean()),
                           mlm_health=h, n_heads=len(heads))
        print(f"     ablate {name:44s} coupling {c.mean():+.3f}  "
              f"({100*c.mean()/base.mean():5.1f}% retained)  health {h:.4f}")
    R["completeness"] = comp2
    print("     If the coupling SURVIVES here, nothing outside C and its input cone is")
    print("     needed -- that is the completeness claim.  Compare against the 'keep ONLY")
    print("     top-k' column below, which deletes the input cone and therefore cannot.")

    # ---- C. sufficiency -----------------------------------------------------------------
    print("\nC. SUFFICIENCY  (patch clean head outputs into the CORRUPT run)")
    suf: dict = {}
    for name, heads in [("L3H1", [READER]), ("L2H15", [ADDRESSER]),
                        ("circuit C", CIRC), ("complement (94 heads)", COMP),
                        ("all 96 heads (ceiling)", COMP + CIRC)]:
        num, dnm = denoise(task, CURVE, heads)
        r = ratio_of_means(num, dnm)
        lo, hi = gboot_ratio(num, dnm, cgenes)
        suf[name] = dict(recovery=r, lo=lo, hi=hi, n_heads=len(heads))
        print(f"     patch {name:24s} recovery {r:6.3f}  [{lo:+.3f},{hi:+.3f}]")
    R["sufficiency"] = suf
    print("     NOTE 1: necessity and sufficiency DISSOCIATE.  L2H15 is necessary (ablation)")
    print("     and restores nothing (denoise): it is machinery L3H1 needs, not a carrier")
    print("     of the partner signal.  Reporting either alone misdescribes the circuit.")
    print("     NOTE 2: the COMPLEMENT also recovers a lot, and the two recoveries sum well")
    print("     past 1.0.  That is not a rival route -- patching the layer-0..2 heads clean")
    print("     repairs L3H1's own INPUT, so the unpatched L3H1 then recomputes correctly.")
    print("     Head-denoising a large set silently reconstructs the clean run; it cannot")
    print("     establish that a set is the ONLY route.  Necessity + the matched null do.")

    # ---- D. circuit-size curves ---------------------------------------------------------
    print(f"\nD. CIRCUIT-SIZE CURVES  (n={len(CURVE)}, frozen head order)")
    print(f"     {'k':>3s}  {'ablate top-k':>13s}  {'keep ONLY top-k':>16s}  {'health(keep-k)':>14s}")
    curve = []
    for k in SWEEP:
        top = order[:k]
        rest = [h for h in order if h not in set(top)]
        c_nec = task.coupling(CURVE, ablate(ad, top)) if k else task.coupling(CURVE)
        c_suf = task.coupling(CURVE, ablate(ad, rest)) if rest else task.coupling(CURVE)
        h_suf = task.mlm_health(CURVE, ablate(ad, rest)) if rest else h_base
        curve.append(dict(k=k, ablate_topk=float(c_nec.mean()),
                          keep_only_topk=float(c_suf.mean()), mlm_health_keep=h_suf))
        print(f"     {k:3d}  {c_nec.mean():13.3f}  {c_suf.mean():16.3f}  {h_suf:14.4f}")
    R["curve"] = curve
    R["curve_baseline"] = float(task.coupling(CURVE).mean())
    print("     ablate-top-k  -> how fast the coupling DIES     (necessity)")
    print("     keep-only-k   -> how fast it COMES BACK          (completeness)")
    print("     read the knee of the second curve WITH the health column beside it.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:faithfulness", sys.argv,
        config={"split": args.split, "n": len(X), "n_curve": len(CURVE),
                "circuit": [list(h) for h in CIRC], "frozen_order_from": args.scan},
        notes=["Head order is READ from ablation_scan.json, never recomputed here.",
               "Unconditional coupling is the behavioural claim; the >=0.3 filter is only "
               "for ratio denominators and every conditional number is labelled.",
               "MLM health accompanies every ablation; a coupling measured in a wrecked "
               "model is not evidence."],
    ).write(out.parent / "faithfulness.manifest.json")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
