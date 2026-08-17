#!/usr/bin/env python
"""M4c -- do the class-specific heads act by shaping what L3H1 can address?

M4b produced a dissociation that needs an explanation. The six family-wise class-specific
heads place essentially ZERO attention on the donor (mass ~0.0000), so their effect on the
coupling cannot be donor retrieval. Yet the donor address is real, large, and strongly
class-dependent -- and it lives almost entirely in L3H1 (GC 0.170 against GT 0.467), a head
whose own ablation profile is class-agnostic.

The obvious bridge: a layer-0/1 head does not read the donor, it helps BUILD the donor's
representation, and L3H1's query then matches against it. If so, removing one of those heads
should degrade what L3H1 finds:

    ablate a class-specific head  ->  L3H1's donor attention mass FALLS
    and, if the minor class has less margin, it falls fractionally FURTHER on GC

This measures exactly that: ablate each head in turn and re-read L3H1's donor attention mass
in both arms. Inert heads and shared load-bearing heads are included as contrasts, so a
general "ablating anything perturbs attention" effect would show up rather than be assumed.

Descriptive plus intervention on attention geometry, not on the margin. Development tier,
train split; chr8/9 never read and the validation reserve is not spent.

RECORDED OUTCOME (181 pairs / 152 genes; baseline L3H1 donor mass GC 0.1702, GT 0.4669):

    probe   kind                  GC frac   GT frac   diff      95% CI
    L0H14   class-specific         +0.383    +0.022   +0.361  [+0.255,+0.468]
    L0H4    class-specific         +0.300    +0.090   +0.210  [+0.099,+0.325]
    L0H3    class-specific         +0.199    -0.008   +0.206  [+0.121,+0.297]
    L0H5    class-specific         +0.219    +0.025   +0.194  [+0.097,+0.287]
    L1H6    class-specific         +0.100    +0.031   +0.069  [+0.011,+0.125]
    L1H13   class-specific         +0.072    +0.037   +0.035  [-0.025,+0.098] ns
    L1H12   shared load-bearing    +0.351    +0.232   +0.119  [-0.005,+0.249] ns
    L2H15   shared load-bearing    +0.529    +0.387   +0.142  [-0.011,+0.290] ns
    L0H0    contrast, inert        +0.005    +0.010   -0.005  ns
    L5H7    contrast, inert, late  +0.000    +0.000   +0.000

The prediction holds. Removing a class-specific early head costs the minor class 20-38% of
L3H1's donor attention against 0-9% for canonical; five of six differentials are significant.
The shared heads degrade addressing for BOTH classes with no significant differential, so
this is not a generic "ablating anything perturbs attention" effect. L5H7 sits downstream of
layer 3 and moves it by exactly +0.000 -- an architectural sanity check the design predicts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.circuits.hf_bert import edit_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _dataset,
    load_adapter, pick_device,
)
from gfmcd.splicing.coupling import planted_donor_motifs  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    COMPARATOR_MOTIF, MATCHED_OPERATOR, TREATMENT_MOTIF, assert_seal_intact,
    eligible_donor_motifs, flank_and_support_key, match_comparators, train_chromosomes,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout  # noqa: E402

TAG = "m4c-upstream-of-address-v1"
READER_LAYER, READER_HEAD = 3, 1     # the head whose donor attention we re-read
SEED, N_BOOT = 0, 2000

PROBES: list[tuple[int, int, str]] = [
    (0, 4, "class-specific"), (0, 5, "class-specific"), (0, 14, "class-specific"),
    (0, 3, "class-specific"), (1, 13, "class-specific"), (1, 6, "class-specific"),
    (1, 12, "shared load-bearing"), (2, 15, "shared load-bearing"),
    (0, 0, "contrast, inert"), (5, 7, "contrast, inert, late"),
]


def gboot_mean(v, genes, *, n_draws=N_BOOT, seed=SEED):
    by: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        by.setdefault(str(g), []).append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.integers(0, len(keys), len(keys))
        draws[i] = np.mean(v[[r for j in pick for r in by[keys[j]]]])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(np.mean(v)), float(lo), float(hi)


def donor_mass(ad, task, rows, off, batch, ctx_factory):
    """L3H1 attention mass from the readout token onto the donor dinucleotide."""
    vals = []
    for i in range(0, len(rows), batch):
        b = rows[i:i + batch]
        pos = [acceptor_readout(x) for x in b]
        enc = task.encode([x.sequence for x in b], pos)
        with torch.no_grad(), ctx_factory():
            out = ad.model(**enc, output_attentions=True)
        A = out.attentions[READER_LAYER]
        for j, x in enumerate(b):
            r, d = pos[j] + off, x.donor_pos + off
            vals.append(float(A[j, READER_HEAD, r, d:d + 2].sum()))
        del out
    return np.asarray(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    ad = load_adapter(a.model, pick_device(a.device), eager=True)
    task = CouplingTask(ad, batch=a.batch)
    off = ad.prefix_tokens

    D = _dataset(a.gtf, a.fasta, allowed_chromosomes=train_chromosomes())
    assert_seal_intact(D)
    pool = lambda m: [x for x in eligible_donor_motifs(D, "train", donor_motifs=(m,))  # noqa: E731
                      if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)]
    coh = match_comparators(pool(TREATMENT_MOTIF), pool(COMPARATOR_MOTIF), k=1,
                            key_fn=flank_and_support_key)
    gc, gt = coh.treatment, coh.comparators
    genes = [x.gene_id for x in gc]
    print(f"pairs={len(gc)} genes={len(set(genes))} reader=L{READER_LAYER}H{READER_HEAD}",
          flush=True)

    from contextlib import nullcontext
    base_gc = donor_mass(ad, task, gc, off, a.batch, nullcontext)
    base_gt = donor_mass(ad, task, gt, off, a.batch, nullcontext)
    b_gc, b_gt = base_gc.mean(), base_gt.mean()
    print(f"baseline L3H1 donor mass: GC {b_gc:.4f}  GT {b_gt:.4f}  "
          f"ratio {b_gc/b_gt:.4f}", flush=True)

    res = {}
    print()
    print(f"{'probe':>7} {'kind':>20} {'GC frac':>9} {'GT frac':>9} {'diff':>9} {'95% CI':>20}")
    for L, H, kind in PROBES:
        f = lambda l=L, h=H: edit_heads(ad, [(l, h)], mode="mean")  # noqa: E731
        mg = donor_mass(ad, task, gc, off, a.batch, f)
        mt = donor_mass(ad, task, gt, off, a.batch, f)
        fg = (b_gc - mg.mean()) / b_gc
        ft = (b_gt - mt.mean()) / b_gt
        per_pair = ((base_gc - mg) / b_gc) - ((base_gt - mt) / b_gt)
        e, lo, hi = gboot_mean(per_pair, genes)
        res[f"L{L}H{H}"] = {
            "kind": kind,
            "GC": {"mass": float(mg.mean()), "frac_drop": float(fg)},
            "GT": {"mass": float(mt.mean()), "frac_drop": float(ft)},
            "frac_drop_difference": e, "ci": [lo, hi],
        }
        print(f"L{L}H{H:<2} {kind:>20} {fg:>+9.3f} {ft:>+9.3f} {e:>+9.3f} "
              f"[{lo:>+.3f},{hi:>+.3f}]", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "gfmcd.m4c.upstream_of_address/1",
        "tier": "development", "split": "train", "attention_impl": "eager",
        "sealed_chr8_chr9_opened": False, "validation_reserve_spent": False,
        "reader": f"L{READER_LAYER}H{READER_HEAD}",
        "n_pairs": len(gc), "n_genes": len(set(genes)),
        "baseline_donor_mass": {"GC": float(b_gc), "GT": float(b_gt),
                                "ratio": float(b_gc / b_gt)},
        "probes": res,
        "licenses": ("Development-tier test of whether early class-specific heads shape the "
                     "donor representation that L3H1 addresses. Attention geometry only; no "
                     "margin, route, or biological claim follows."),
    }, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:m4c_upstream_of_address", sys.argv,
        config={"n_pairs": len(gc), "reader": f"L{READER_LAYER}H{READER_HEAD}"},
        seeds=dict(bootstrap=SEED), status="DEVELOPMENT",
        notes=["Train split only; chr8/9 not opened; validation reserve not spent."])
    for f_ in (a.model, a.gtf, a.fasta):
        try:
            man.add_input(f_)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f_}")
    man.add_output(out)
    man.write(out.with_suffix(".manifest.json"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
