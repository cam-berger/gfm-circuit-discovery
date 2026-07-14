#!/usr/bin/env python
"""THE CODON-PHASE LATTICE, part 1: what kind of object is it?

Established already:
  * 30 of 96 heads put >90% of their local attention mass on a single phase (mod 3);
    ten of them at 1.000.  Chance is 0.333.
  * A RANDOMLY-INITIALISED SpliceBERT has NONE of it (mean 0.344 = chance, 0/96 above 0.90).
    So the lattice is LEARNED, from masked-nucleotide pretraining alone.  Not architecture,
    not tokenizer, not init.
  * It SURVIVES sequence shuffling -> the attention PATTERN reads no bases.  It is a lattice,
    not (yet) a codon reader.

Two structural questions decide what it is.

  Q1  RELATIVE or ABSOLUTE?
      relative:  "attend to every 3rd position FROM ME"        -> offset k ≡ r (mod 3)
      absolute:  "attend to token indices ≡ c (mod 3)"         -> k ≡ c - p (mod 3)
      The two are distinguishable: under ABSOLUTE, the dominant RELATIVE phase must SHIFT with
      the query's own index mod 3.  Under RELATIVE it must not move.
      This matters: an absolute lattice is a global frame the whole model shares.  A relative
      lattice is a local stride each position applies to itself -- much weaker, and it cannot
      by itself represent a reading frame.

  Q2  ARE ALL THREE PHASES COVERED?
      A usable codon representation needs the three phases kept APART.  If every lattice head
      picks the same phase, the model has a stride, not a frame.  If the heads partition into
      three groups, it has three channels -- which is what a frame needs.

Both are cheap.  Neither can prove the lattice is USED; that is part 2.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, load_examples,
    pick_device,
)

SPAN = 24


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="sequences")
    ap.add_argument("--queries", type=int, default=24, help="query positions per sequence")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/phase-lattice/phase_inventory.json")
    args = ap.parse_args()

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    L, H = ad.num_layers, ad.num_heads

    X, _ = load_examples(args.gtf, args.fasta, "val", args.n, task=task, tag="phase-inv",
                         min_effect=None)
    print(f"{len(X)} real sequences   device={dev}\n")

    # mass[(l,h)][qphase][rel_phase] -- attention mass by the query's OWN index mod 3
    mass = defaultdict(lambda: np.zeros((3, 3)))
    rng = np.random.default_rng(0)

    with torch.no_grad():
        for i in range(0, len(X), task.batch):
            ch = X[i:i + task.batch]
            enc = task.encode([x.sequence for x in ch], [x.donor_pos for x in ch])
            A = ad.model(**enc, output_attentions=True).attentions   # tuple[L] of [B,H,S,S]
            S = enc["input_ids"].shape[1]
            qs = rng.integers(SPAN + 4, S - SPAN - 4, size=args.queries)
            for l in range(L):
                Al = A[l].float().cpu().numpy()
                for h in range(H):
                    for b in range(len(ch)):
                        for p in qs:
                            row = Al[b, h, p]
                            for k in range(-SPAN, SPAN + 1):
                                if k == 0:
                                    continue
                                mass[(l, h)][p % 3, k % 3] += row[p + k]
            print(f"  {min(i+task.batch, len(X))}/{len(X)}", end="\r", flush=True)
    print()

    rows = []
    for (l, h), M in mass.items():
        Mn = M / np.maximum(M.sum(axis=1, keepdims=True), 1e-12)   # normalise per query-phase
        dom = Mn.argmax(axis=1)              # dominant RELATIVE phase, for q ≡ 0,1,2 (mod 3)
        strength = float(Mn.max(axis=1).mean())
        # ABSOLUTE lattice: attended index ≡ c (mod 3)  =>  rel phase = (c - p) mod 3
        abs_c = [(int(dom[p]) + p) % 3 for p in range(3)]
        is_abs = len(set(abs_c)) == 1                    # the implied absolute phase agrees
        is_rel = len(set(dom.tolist())) == 1             # the relative phase never moves
        rows.append(dict(layer=l, head=h, strength=strength,
                         dominant_rel_phase=[int(v) for v in dom],
                         implied_abs_phase=abs_c,
                         absolute=bool(is_abs and not is_rel),
                         relative=bool(is_rel),
                         lattice=bool(strength > 0.60)))

    lat = [r for r in rows if r["lattice"]]
    print("=" * 84)
    print(f"PHASE INVENTORY   {len(lat)} of {L*H} heads are lattice heads "
          f"(>60% of local mass on one phase; chance = 33%)")

    print("\nQ1  RELATIVE or ABSOLUTE?")
    n_abs = sum(1 for r in lat if r["absolute"])
    n_rel = sum(1 for r in lat if r["relative"])
    print(f"  ABSOLUTE (attends token indices ≡ c mod 3, so the relative phase SHIFTS "
          f"with the query): {n_abs}/{len(lat)}")
    print(f"  RELATIVE (attends offsets ≡ r mod 3, the relative phase never moves): "
          f"{n_rel}/{len(lat)}")
    if n_abs > 0.7 * len(lat):
        print("  => an ABSOLUTE lattice.  The whole model shares ONE global phase frame.")
        print("     That is the substrate a reading frame needs.")
    elif n_rel > 0.7 * len(lat):
        print("  => a RELATIVE lattice: each position applies the same stride to itself.")
        print("     This CANNOT by itself represent a reading frame -- there is no shared")
        print("     origin.  It is a triplet-stride aggregator.")
    else:
        print("  => mixed; report both populations.")

    print("\nQ2  ARE THE THREE PHASES COVERED?")
    if n_abs >= n_rel:
        c = Counter(r["implied_abs_phase"][0] for r in lat if r["absolute"])
        print(f"  absolute phase c chosen by the lattice heads: {dict(sorted(c.items()))}")
    c2 = Counter(r["dominant_rel_phase"][0] for r in lat)
    print(f"  relative phase r (at query ≡ 0 mod 3):            {dict(sorted(c2.items()))}")
    if len(c2) == 3:
        print("  => all THREE phases are represented: the model keeps the codon positions")
        print("     APART, in separate channels.  That is a frame, not a stride.")
    else:
        print(f"  => only {len(c2)} of 3 phases used.  The model has a stride, not a frame.")

    print("\nby layer:")
    for l in range(L):
        hs = [r for r in lat if r["layer"] == l]
        print(f"  L{l}: {len(hs):2d}/16 lattice heads   "
              f"{' '.join(f'H{r[chr(104)+chr(101)+chr(97)+chr(100)]}' for r in sorted(hs, key=lambda r: r['head']))}")

    print("\nstrongest 12:")
    for r in sorted(rows, key=lambda r: -r["strength"])[:12]:
        kind = "ABS" if r["absolute"] else ("REL" if r["relative"] else "mix")
        print(f"  L{r['layer']}H{r['head']:<2d}  mass {r['strength']:.3f}  {kind}  "
              f"rel-phase by query-phase {r['dominant_rel_phase']}")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_seq": len(X), "span": SPAN, "heads": rows,
                               "n_lattice": len(lat), "n_absolute": n_abs,
                               "n_relative": n_rel}, indent=1))
    ExperimentManifest.create(
        "phase_lattice:inventory", sys.argv,
        config={"n": len(X), "queries": args.queries, "span": SPAN},
        notes=["ABSOLUTE vs RELATIVE is the load-bearing distinction: only an absolute lattice "
               "gives the model a shared phase origin, which is what a reading frame requires."],
    ).write(out.parent / "phase_inventory.manifest.json")
    print("=" * 84)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
