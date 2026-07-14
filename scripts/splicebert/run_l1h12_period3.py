#!/usr/bin/env python
"""L1H12 attends on a PERIOD-3 COMB.  What kind of head is that?

From the donor, its attention lands on relative offsets ≡ 1 (mod 3) and essentially nowhere
else: 0.0177 vs 0.0007 vs 0.0006 across the three phases -- a 30.5x ratio, with period-3 the
strongest non-DC Fourier component.  It reads every third nucleotide.  Codon periodicity is
the obvious hypothesis, and it is not the only one.  Three cheap tests separate them.

TEST A -- IS THE COMB ANCHORED TO THE DONOR, OR TO ANY QUERY?
    Measure the comb from the donor, from the acceptor, and from random non-site positions.
      same comb everywhere  -> a RELATIVE-OFFSET head: "attend to every 3rd position from me".
                               Nothing to do with the donor specifically.
      only at the donor     -> something about the 5'ss triggers it.

TEST B -- IS IT SEQUENCE-DRIVEN, OR PURELY POSITIONAL?
    Re-run on SHUFFLED sequence (same length, same base composition, no motifs).  BERT's
    position embeddings are absolute, and learned absolute embeddings can encode relative
    offset in their inner products -- so a period-3 comb can be produced with no reference to
    the bases at all.
      comb survives shuffling -> POSITIONAL.  It is a property of the position embeddings,
                                 not a reading-frame detector.  (This is the boring answer,
                                 and it must stay reachable.)
      comb collapses          -> SEQUENCE-DRIVEN.  Then, and only then, is "codon structure"
                                 on the table.

TEST C -- DOES IT CARE ABOUT CODING?
    A reading-frame feature should matter more where there IS a reading frame.  Split L1H12's
    contribution to the coupling by gene_type (protein_coding vs everything else).  We already
    store gene_type, so this costs one extra pass.
      bigger in protein_coding -> consistent with a frame/codon role
      no difference            -> it is not about coding

None of this is confirmatory.  "L1H12 is a positional artefact of the embeddings" is a
perfectly good answer and the tests are built so it can win.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, ablate, gboot, load_adapter,
    load_examples, pick_device,
)

CAND = (1, 12)
L1, H1 = CAND
SPAN = 21          # offsets -SPAN..+SPAN


def comb_from(A, q, span=SPAN):
    """Attention by relative offset from query position q.  Returns [2*span+1]."""
    S = A.shape[-1]
    out = np.zeros(2 * span + 1)
    for k in range(-span, span + 1):
        j = q + k
        if 0 <= j < S:
            out[k + span] = A[q, j].item()
    return out


def phase_ratio(profile, span=SPAN, exclude_self=True):
    """Mean attention by (offset mod 3), and the best/worst ratio."""
    by = {0: [], 1: [], 2: []}
    for k in range(-span, span + 1):
        if exclude_self and k == 0:
            continue
        by[k % 3].append(profile[k + span])
    m = {r: float(np.mean(v)) for r, v in by.items()}
    hi, lo = max(m.values()), min(m.values())
    return m, hi / max(lo, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/l1h12_period3.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"l1p3-{args.split}", min_effect=0.3)
    genes = [x.gene_id for x in X]
    print(f"{len(X)} {args.split} exons   device={dev}\n")
    R: dict = {"split": args.split, "n": len(X)}

    # ================= TEST A + B  (one pass each) ================================
    rng = random.Random(0)
    combs = {k: np.zeros(2 * SPAN + 1) for k in
             ("donor", "acceptor", "random-pos", "donor@SHUFFLED", "random-pos@SHUFFLED")}
    cnt = Counter()

    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            dn = [x.donor_pos for x in ch]
            real = task.encode([x.sequence for x in ch], dn)
            # SHUFFLED: same length, same base composition, motifs destroyed
            sh = []
            for x in ch:
                s = list(x.sequence)
                rng.shuffle(s)
                sh.append("".join(s))
            shuf = task.encode(sh, dn)

            A_real = ad.model(**real, output_attentions=True).attentions[L1][:, H1]
            A_shuf = ad.model(**shuf, output_attentions=True).attentions[L1][:, H1]
            if float(A_real.abs().sum()) == 0.0:
                raise RuntimeError("all-zero attention -- adapter must be eager")

            for j, x in enumerate(ch):
                d = x.donor_pos + OFF
                a = x.next_acceptor_pos + OFF
                r = rng.randrange(OFF + SPAN + 5, A_real.shape[-1] - SPAN - 5)
                combs["donor"] += comb_from(A_real[j], d)
                combs["acceptor"] += comb_from(A_real[j], a)
                combs["random-pos"] += comb_from(A_real[j], r)
                combs["donor@SHUFFLED"] += comb_from(A_shuf[j], d)
                combs["random-pos@SHUFFLED"] += comb_from(A_shuf[j], r)
                cnt["n"] += 1
            print(f"  attention {min(i+task.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()
    for k in combs:
        combs[k] /= cnt["n"]

    print("=" * 80)
    print("TEST A/B  IS THE COMB QUERY-RELATIVE, AND IS IT SEQUENCE-DRIVEN?")
    print(f"  {'query':22s} {'phase 0':>9s} {'phase 1':>9s} {'phase 2':>9s} {'ratio':>8s}")
    R["combs"] = {}
    for k, prof in combs.items():
        m, ratio = phase_ratio(prof)
        R["combs"][k] = dict(phase={str(r): m[r] for r in m}, ratio=float(ratio),
                             profile={str(o - SPAN): float(prof[o]) for o in range(2*SPAN+1)})
        print(f"  {k:22s} {m[0]:9.4f} {m[1]:9.4f} {m[2]:9.4f} {ratio:7.1f}x")

    rd = R["combs"]["donor"]["ratio"]
    rr = R["combs"]["random-pos"]["ratio"]
    rs = R["combs"]["donor@SHUFFLED"]["ratio"]
    print()
    if rr > 5:
        print("  A: the comb appears from a RANDOM position too -- it is a RELATIVE-OFFSET")
        print("     head ('attend to every 3rd position from me'), not a donor-specific one.")
    else:
        print("  A: the comb is specific to the donor query.")
    if rs > 0.5 * rd:
        print("  B: the comb SURVIVES sequence shuffling -- it is POSITIONAL, produced by the")
        print("     position embeddings, and is NOT reading the bases.  The codon hypothesis")
        print("     does not survive this: a period-3 comb over shuffled DNA reads no codons.")
    else:
        print("  B: the comb COLLAPSES on shuffled sequence -- it is SEQUENCE-DRIVEN.")
        print("     Codon/reading-frame structure is now a live hypothesis.")

    # ================= TEST C  -- does it care about coding? ======================
    print("\n" + "=" * 80)
    print("TEST C  DOES L1H12's CONTRIBUTION DEPEND ON THERE BEING A READING FRAME?")
    base = task.coupling(X)
    abl = task.coupling(X, ablate(ad, [CAND]))
    drop = base - abl
    coding = np.array([x.gene_type == "protein_coding" for x in X])
    print(f"  gene types: {Counter(x.gene_type for x in X).most_common(4)}")
    R["coding"] = {}
    for label, mask in [("protein_coding", coding), ("non-coding", ~coding)]:
        if mask.sum() < 5:
            print(f"  {label:16s} n={int(mask.sum())} -- too few to compare")
            R["coding"][label] = dict(n=int(mask.sum()))
            continue
        g = [genes[i] for i in np.where(mask)[0]]
        lo, hi = gboot(drop[mask], g)
        R["coding"][label] = dict(n=int(mask.sum()), baseline=float(base[mask].mean()),
                                  drop=float(drop[mask].mean()), ci=[lo, hi],
                                  frac=float(drop[mask].mean() / base[mask].mean()))
        print(f"  {label:16s} n={int(mask.sum()):3d}   baseline {base[mask].mean():+.3f}"
              f"   L1H12 drop {drop[mask].mean():+.3f} [{lo:+.3f},{hi:+.3f}]"
              f"   ({100*drop[mask].mean()/base[mask].mean():.1f}%)")
    if coding.sum() >= 5 and (~coding).sum() >= 5:
        diff = drop[coding].mean() - drop[~coding].mean()
        print(f"  difference (coding - noncoding): {diff:+.3f}")
        print("  A reading-frame feature should matter MORE where there is a frame.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:l1h12_period3", sys.argv,
        config={"split": args.split, "n": len(X), "head": list(CAND)},
        notes=["EXPLORATORY.  'It is a positional artefact of the embeddings' is a reachable "
               "answer and the shuffled-sequence arm is what makes it reachable."],
    ).write(out.parent / "l1h12_period3.manifest.json")
    print("=" * 80)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
