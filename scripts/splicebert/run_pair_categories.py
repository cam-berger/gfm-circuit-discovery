#!/usr/bin/env python
"""PANEL A: causal analogue of the SpliceBERT attention figure.

The original SpliceBERT paper reports that *attention* is elevated between donor and
acceptor sites of the SAME INTRON, especially in intermediate layers.  That is a
correlation.  This script asks the causal question over the same pair categories:

    for each relationship, how much does DESTROYING one site degrade the model's
    native MLM recognition of the other?

and it asks it twice -- with the model intact, and with L3H1 ablated.

Categories (mirroring the paper's pair taxonomy):
    same-intron donor-acceptor   perturb A' (next exon's acceptor)  -> measure donor D
    same-exon   donor-acceptor   perturb A  (this exon's acceptor)  -> measure donor D
    unpaired    donor-acceptor   perturb a real acceptor, not a partner of D -> measure D
    donor-donor                  perturb another real donor         -> measure D
    acceptor-acceptor            perturb another real acceptor      -> measure acceptor A
    distance-matched random      perturb a random NON-site at the same distance -> measure D

Expected message: same-intron coupling is strong in the intact model and SELECTIVELY
collapses under L3H1 ablation, while the control relationships move much less.  That is
what the attention analysis could not show -- which association the model actually USES.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.adapters.splicebert import SpliceBertAdapter  # noqa: E402
from gfmcd.circuits.hf_bert import edit_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.coupling import (  # noqa: E402
    ACCEPTOR_ALTS, DONOR_ALTS, CouplingConfig, build_coupling_dataset, perturb_random,
    stable_rng, substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402

LAYER, HEAD = 3, 1

CATEGORIES = [
    "same-intron\ndonor-acceptor",
    "same-exon\ndonor-acceptor",
    "unpaired\ndonor-acceptor",
    "donor-donor",
    "acceptor-acceptor",
    "distance-matched\nrandom",
]


@contextmanager
def _noop():
    yield


def gboot(v, g, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for a, b in zip(v, g):
        if not np.isnan(a):
            by.setdefault(b, []).append(a)
    k = list(by)
    if not k:
        return (np.nan, np.nan)
    m = np.empty(n)
    for i in range(n):
        p = rng.choice(len(k), len(k), replace=True)
        m[i] = np.mean([x for j in p for x in by[k[j]]])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default="/Users/cberg/Research/data/genomes/gencode.v50.basic.annotation.gtf.gz")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--model", default="data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")
    ap.add_argument("--out", default="figures/splicebert/data/pair_categories.json")
    args = ap.parse_args()

    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = args.device or ("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    mp = args.model if Path(args.model).is_absolute() else str(REPO / args.model)
    ad = SpliceBertAdapter(AutoModelForMaskedLM.from_pretrained(mp),
                           AutoTokenizer.from_pretrained(mp), device=dev)
    tok = ad.tokenizer
    MASK = tok.mask_token_id
    BID = {b: tok.convert_tokens_to_ids(b) for b in "ACGT"}
    OFF = ad.prefix_tokens

    def margin(seqs, pos, base, ctx):
        """logit(base) - max(other) at pos, under an intervention context."""
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), args.batch):
                S, P = seqs[i:i + args.batch], pos[i:i + args.batch]
                enc = tok([" ".join(s) for s in S], return_tensors="pt",
                          padding=True).to(ad.device)
                for j, p in enumerate(P):
                    enc["input_ids"][j, p + OFF] = MASK
                with ctx():
                    lg = ad.model(**enc).logits.float()
                for j, p in enumerate(P):
                    v = lg[j, p + OFF]
                    o = max(v[BID[b]].item() for b in "ACGT" if b != base)
                    out.append(v[BID[base]].item() - o)
        return np.asarray(out)

    cfg = CouplingConfig()
    X = build_coupling_dataset(args.gtf, PyfaidxGenome(args.fasta), cfg)
    X = [x for x in X if x.split == args.split and x.next_acceptor_pos is not None
         and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
         and 60 <= (x.next_acceptor_pos - x.donor_pos) <= 150]
    r = stable_rng("cat", cfg.seed)
    r.shuffle(X)
    X = X[: args.n]
    print(f"{len(X)} exons ({args.split}, held-out)   device={dev}")

    # Build, per category, a list of (perturbed_seq, measure_pos, measure_base, clean_seq, gene)
    jobs = {c: [] for c in CATEGORIES}
    for x in X:
        s, a, d, na = x.sequence, x.acceptor_pos, x.donor_pos, x.next_acceptor_pos
        rr = stable_rng(x.example_id, 0)

        jobs[CATEGORIES[0]].append((substitute(s, na - 2, ACCEPTOR_ALTS[0]), d, "G", s, x.gene_id))
        jobs[CATEGORIES[1]].append((substitute(s, a - 2, ACCEPTOR_ALTS[0]), d, "G", s, x.gene_id))

        # unpaired donor-acceptor: a REAL acceptor that is neither partner
        up = [p for p in x.real_acc if abs(p - a) > 10 and abs(p - na) > 10]
        if up:
            p = up[rr.randrange(len(up))]
            jobs[CATEGORIES[2]].append((substitute(s, p - 2, ACCEPTOR_ALTS[0]), d, "G", s, x.gene_id))

        # donor-donor: another REAL donor
        dd = [p for p in x.real_don if abs(p - d) > 10]
        if dd:
            p = dd[rr.randrange(len(dd))]
            jobs[CATEGORIES[3]].append((substitute(s, p, DONOR_ALTS[0]), d, "G", s, x.gene_id))

        # acceptor-acceptor: perturb another REAL acceptor, MEASURE this exon's acceptor
        aa = [p for p in x.real_acc if abs(p - a) > 10]
        if aa:
            p = aa[rr.randrange(len(aa))]
            jobs[CATEGORIES[4]].append((substitute(s, p - 2, ACCEPTOR_ALTS[0]), a - 1, "G", s, x.gene_id))

        # distance-matched random NON-site, same |offset| from the donor as A'
        off = na - d
        cand = [q for q in (d - off, d + off // 2) if 12 < q < len(s) - 12
                and s[q - 2:q] != "AG" and s[q:q + 2] not in ("GT", "GC")]
        if cand:
            q = cand[rr.randrange(len(cand))]
            jobs[CATEGORIES[5]].append((perturb_random(s, q), d, "G", s, x.gene_id))

    ABL = lambda: edit_heads(ad, [(LAYER, HEAD)], mode="mean")  # noqa: E731
    res = {}
    for cat, J in jobs.items():
        if not J:
            continue
        pert, pos, base, clean, genes = zip(*J)
        row = {}
        for cond, ctx in [("intact", _noop), ("L3H1 ablated", ABL)]:
            c = margin(list(clean), list(pos), base[0], ctx)
            k = margin(list(pert), list(pos), base[0], ctx)
            eff = c - k                                    # causal coupling
            lo, hi = gboot(eff, list(genes))
            row[cond] = dict(mean=float(eff.mean()), lo=lo, hi=hi, n=len(eff))
        res[cat] = row
        print(f"  {cat.replace(chr(10),' '):34s} intact {row['intact']['mean']:+.3f}"
              f"   L3H1-ablated {row['L3H1 ablated']['mean']:+.3f}   (n={row['intact']['n']})")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "categories": res}, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:pair_categories", sys.argv,
        config={"n": len(X), "split": args.split, "head": [LAYER, HEAD]},
        notes=["Causal analogue of the SpliceBERT attention pair-category figure.",
               "Perturbations substitute real bases; never [MASK]."],
    ).write(out.parent / "pair_categories.manifest.json")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
