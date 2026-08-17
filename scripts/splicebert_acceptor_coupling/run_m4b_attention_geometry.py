#!/usr/bin/env python
"""M4b -- does the acceptor LOOK AT its donor less when the donor is the minor class?

The causal swap in M4 asks whether the address MATTERS. This asks the prior, simpler
question directly and without any intervention: for each head, how much attention does the
masked-acceptor readout token place on the partner donor, and does it differ by donor class?

    address is class-dependent   the model attends to a GC donor less; the weaker coupling
                                 is partly a failure to look
    address is class-invariant   the model looks just as hard at a GC donor and the deficit
                                 lies in what it finds, or in what survives the trip

Two measures per head, both on matched pairs:
    donor_mass      sum of attention from the readout token onto the donor dinucleotide
    profile_cosine  cosine similarity between the two classes' donor-relative attention
                    profiles from the readout token, over a window around the donor

Purely descriptive: no ablation, no patching, no claim of mediation. Development tier,
train split; chr8/9 never read and the validation reserve is not spent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _dataset, eligible,
    load_adapter, pick_device,
)
from gfmcd.splicing.coupling import planted_donor_motifs  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    COMPARATOR_MOTIF, MATCHED_OPERATOR, TREATMENT_MOTIF, assert_seal_intact,
    flank_and_support_key, match_comparators, train_chromosomes,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout  # noqa: E402

TAG = "m4b-attention-geometry-v1"
WINDOW = 32          # +/- positions around the donor for the profile comparison
SEED, N_BOOT = 0, 2000


def gboot_paired(diff, genes, *, n_draws=N_BOOT, seed=SEED):
    """Gene-clustered CI for a paired per-example difference."""
    by: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        by.setdefault(str(g), []).append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)
    est = float(np.mean(diff))
    draws = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.integers(0, len(keys), len(keys))
        idx = [r for j in pick for r in by[keys[j]]]
        draws[i] = np.mean(diff[idx])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return est, float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    ad = load_adapter(a.model, pick_device(a.device), eager=True)
    task = CouplingTask(ad, batch=a.batch)
    off, L, H = ad.prefix_tokens, ad.num_layers, ad.num_heads

    D = _dataset(a.gtf, a.fasta, allowed_chromosomes=train_chromosomes())
    assert_seal_intact(D)
    pool = lambda m: [x for x in eligible(D, "train", donor_motifs=(m,))  # noqa: E731
                      if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)]
    coh = match_comparators(pool(TREATMENT_MOTIF), pool(COMPARATOR_MOTIF), k=1,
                            key_fn=flank_and_support_key)
    gc, gt = coh.treatment, coh.comparators
    genes = [x.gene_id for x in gc]
    print(f"pairs={len(gc)} genes={len(set(genes))}", flush=True)

    mass = {arm: np.zeros((L, H, len(gc))) for arm in ("GC", "GT")}
    prof = {arm: np.zeros((L, H, len(gc), 2 * WINDOW + 1)) for arm in ("GC", "GT")}

    for arm, rows in (("GC", gc), ("GT", gt)):
        for i in range(0, len(rows), a.batch):
            b = rows[i:i + a.batch]
            pos = [acceptor_readout(x) for x in b]
            enc = task.encode([x.sequence for x in b], pos)
            with torch.no_grad():
                out = ad.model(**enc, output_attentions=True)
            for li in range(L):
                A = out.attentions[li]                      # [B,H,S,S]
                for j, x in enumerate(b):
                    r = pos[j] + off                        # readout token
                    d = x.donor_pos + off                   # donor token
                    mass[arm][li, :, i + j] = A[j, :, r, d:d + 2].sum(-1).cpu().numpy()
                    lo, hi = d - WINDOW, d + WINDOW + 1
                    prof[arm][li, :, i + j, :] = A[j, :, r, lo:hi].cpu().numpy()
            del out
        print(f"  {arm} arm done", flush=True)

    print()
    print(f"{'head':>7} {'GC mass':>9} {'GT mass':>9} {'diff':>9} "
          f"{'95% CI':>20} {'cos(profile)':>13}")
    res = {}
    flagged = []
    for li in range(L):
        for h in range(H):
            dgc, dgt = mass["GC"][li, h], mass["GT"][li, h]
            e, lo, hi = gboot_paired(dgc - dgt, genes)
            pg, pt = prof["GC"][li, h], prof["GT"][li, h]
            cs = float(np.mean(np.sum(pg * pt, 1)
                               / (np.linalg.norm(pg, axis=1) * np.linalg.norm(pt, axis=1)
                                  + 1e-12)))
            k = f"L{li}H{h}"
            res[k] = {"GC_donor_mass": float(dgc.mean()), "GT_donor_mass": float(dgt.mean()),
                      "diff": e, "ci": [lo, hi], "profile_cosine": cs}
            if lo > 0 or hi < 0:
                flagged.append((abs(e), k, e, lo, hi, cs, float(dgc.mean()),
                                float(dgt.mean())))
    for _, k, e, lo, hi, cs, mg, mt in sorted(flagged, reverse=True)[:12]:
        print(f"{k:>7} {mg:>9.4f} {mt:>9.4f} {e:>+9.4f} "
              f"[{lo:>+.4f},{hi:>+.4f}] {cs:>13.4f}")
    print(f"\nheads whose donor-attention differs by class (CI excludes 0): "
          f"{len(flagged)} of {L*H}")
    cosines = np.array([v["profile_cosine"] for v in res.values()])
    print(f"donor-relative attention profile cosine, all heads: "
          f"median {np.median(cosines):.4f}  min {cosines.min():.4f}")

    out_p = Path(a.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps({
        "schema": "gfmcd.m4b.attention_geometry/1",
        "tier": "development", "split": "train", "attention_impl": "eager",
        "sealed_chr8_chr9_opened": False, "validation_reserve_spent": False,
        "n_pairs": len(gc), "n_genes": len(set(genes)), "window": WINDOW,
        "per_head": res,
        "n_heads_class_dependent_attention": len(flagged),
        "licenses": ("Descriptive attention geometry only. Establishes whether the readout "
                     "attends to the donor differently by class; no mediation, route, or "
                     "biological claim follows."),
    }, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:m4b_attention_geometry", sys.argv,
        config={"n_pairs": len(gc), "window": WINDOW}, seeds=dict(bootstrap=SEED),
        status="DEVELOPMENT",
        notes=["Train split only; chr8/9 not opened; validation reserve not spent.",
               "Descriptive: no intervention of any kind."])
    for f_ in (a.model, a.gtf, a.fasta):
        try:
            man.add_input(f_)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f_}")
    man.add_output(out_p)
    man.write(out_p.with_suffix(".manifest.json"))
    print(f"\n-> {out_p}")


if __name__ == "__main__":
    main()
