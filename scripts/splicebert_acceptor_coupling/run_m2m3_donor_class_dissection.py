#!/usr/bin/env python
"""M2/M3 -- where does the minor class's weakness live?

M1 found the masked-acceptor readout depends on a real ``GC-AG`` donor at ~0.22 of a
flank-matched ``GT-AG`` donor, an order of magnitude above a non-partner decoy. That is a
graded channel, but M1 cannot say WHICH stage is attenuated.

This assay ablates a fixed head set and re-measures the coupling separately on each arm.
The decisive quantity is the SHAPE of the two per-head profiles, not their heights:

    profiles correlate    -> one circuit at lower gain (the minor class engages the same
                             components, just less)
    profiles diverge      -> the minor class runs a different route

Fractional effects are used precisely because the arms differ ~5x in baseline; comparing raw
drops would rediscover M1 rather than localise it.

Development tier, train split. chr8/9 never read; the validation reserve is not spent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.circuits.hf_bert import edit_heads, matched_random_head_sets  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _dataset, eligible,
    load_adapter, pick_device,
)
from gfmcd.splicing.coupling import planted_donor_motifs, substitute  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    assert_seal_intact, train_chromosomes,
    COMPARATOR_MOTIF, MATCHED_OPERATOR, TREATMENT_MOTIF, donor_motif,
    gboot_two_sample_ratio, match_comparators, matching_balance,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout  # noqa: E402

CURRENT_TAG = "m2m3-donor-class-dissection-v1"
SEED = 0
N_BOOT = 2000
MIN_GENES = 30
#: Any size-1 selected set fixes the matched size; excluded from the sampled universe.
LAYER_FOR_SIZE, HEAD_FOR_SIZE = 3, 1

#: Fixed, pre-declared head set. Every entry has a prior standing in the record, so none is
#: chosen by looking at this assay's outcome.
HEAD_SET: list[tuple[int, int, str]] = [
    (3, 1, "L3H1 donor-arc binder (C22/C35); AC7 reuse inconclusive"),
    (2, 15, "L2H15 AC8 health-failed, diagnostic only"),
    (1, 12, "L1H12 AC5 supported under mean ablation"),
    (4, 11, "L4H11 AC6 supported under noising"),
    (4, 9, "L4H9 AC6 supported under noising"),
    (4, 7, "L4H7 AC6 supported under noising"),
    (4, 0, "L4H0 donor-route comparator, equivalent on the acceptor endpoint"),
    (4, 2, "L4H2 donor-route comparator"),
    (5, 11, "L5H11 donor-route comparator"),
    (0, 7, "L0H7 donor-side payload assembly (C29)"),
]


class Arm:
    """One donor class: clean and donor-destroyed sequences at a fixed acceptor readout."""

    def __init__(self, task, rows):
        self.task = task
        self.rows = rows
        self.pos = [acceptor_readout(x) for x in rows]
        self.clean = [x.sequence for x in rows]
        self.dead = [substitute(x.sequence, x.donor_pos, MATCHED_OPERATOR) for x in rows]
        self.genes = [x.gene_id for x in rows]

    def _m(self, seqs, ctx):
        out = []
        with torch.no_grad():
            for i in range(0, len(self.pos), self.task.batch):
                sl = slice(i, i + self.task.batch)
                enc = self.task.encode(seqs[sl], self.pos[sl])
                with ctx():
                    out += self.task.margin(
                        self.task.ad.model(**enc).logits.float(), self.pos[sl]).tolist()
        return np.asarray(out)

    def coupling(self, ctx):
        return self._m(self.clean, ctx) - self._m(self.dead, ctx)


def _noop_ctx():
    from contextlib import nullcontext
    return nullcontext()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--mode", default="mean", choices=("mean", "zero"))
    ap.add_argument("--n-random-sets", type=int, default=5)
    ap.add_argument("--all-heads", action="store_true",
                    help="scan every head; the pre-declared set cannot tell a "
                         "head-specific asymmetry from a layer-wide one")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    dev = pick_device(a.device)
    ad = load_adapter(a.model, dev, eager=False)
    task = CouplingTask(ad, batch=a.batch)

    D = _dataset(a.gtf, a.fasta, allowed_chromosomes=train_chromosomes())
    n_checked = assert_seal_intact(D)

    def pool(motif):
        return [x for x in eligible(D, "train", donor_motifs=(motif,))
                if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)]

    gc_rows = pool(TREATMENT_MOTIF)
    if a.smoke:
        gc_rows = gc_rows[:6]
    cohort = match_comparators(gc_rows, pool(COMPARATOR_MOTIF), k=a.k)
    treat, comp = Arm(task, cohort.treatment), Arm(task, cohort.comparators)

    if a.all_heads:
        declared = {(l, h): n for l, h, n in HEAD_SET}
        heads = [(l, h, declared.get((l, h), "unscreened"))
                 for l in range(ad.num_layers) for h in range(ad.num_heads)]
    else:
        heads = HEAD_SET[:3] if a.smoke else HEAD_SET

    base_t, base_c = treat.coupling(_noop_ctx), comp.coupling(_noop_ctx)
    b_t, b_c = float(base_t.mean()), float(base_c.mean())
    # The gene floor is a production guard; smoke exercises mechanics on a deliberately
    # tiny cohort, so it relaxes the floor rather than disabling it for real runs.
    rho0 = gboot_two_sample_ratio(base_t, treat.genes, base_c, comp.genes,
                                  n_draws=N_BOOT if not a.smoke else 100, seed=SEED,
                                  min_genes=1 if a.smoke else MIN_GENES)

    per_head = {}
    for layer, head, note in heads:
        ctx = lambda l=layer, h=head: edit_heads(ad, [(l, h)], mode=a.mode)
        yt, yc = treat.coupling(ctx), comp.coupling(ctx)
        mt, mc = float(yt.mean()), float(yc.mean())
        per_head[f"L{layer}H{head}"] = {
            "note": note,
            "treatment_GC": {"mean_Y": mt, "abs_drop": b_t - mt,
                             "frac_drop": (b_t - mt) / b_t if b_t else None},
            "comparator_GT": {"mean_Y": mc, "abs_drop": b_c - mc,
                              "frac_drop": (b_c - mc) / b_c if b_c else None},
        }
        print(f"  L{layer}H{head}: GC frac {(b_t-mt)/b_t:+.3f}   GT frac {(b_c-mc)/b_c:+.3f}",
              flush=True)

    ft = np.array([per_head[k]["treatment_GC"]["frac_drop"] for k in per_head])
    fc = np.array([per_head[k]["comparator_GT"]["frac_drop"] for k in per_head])
    pear = float(np.corrcoef(ft, fc)[0, 1]) if len(ft) > 2 else None
    order_t = np.argsort(np.argsort(-ft))
    order_c = np.argsort(np.argsort(-fc))
    spear = float(np.corrcoef(order_t, order_c)[0, 1]) if len(ft) > 2 else None

    # specificity control: same-size random head sets must not reproduce the profile
    rnd = []
    if not a.smoke:
        # No fallback. The real helper DE-DUPLICATES and enumerates small universes; its own
        # docstring records that sampling with replacement once overstated evidence 10.4x.
        # A bare except here would silently swap that guarantee for the very defect it fixes.
        sets = matched_random_head_sets(
            ad, [(LAYER_FOR_SIZE, HEAD_FOR_SIZE)], n=a.n_random_sets, seed=SEED,
            exclude_selected=True,
        )
        for s in sets:
            ctx = lambda ss=s: edit_heads(ad, list(ss), mode=a.mode)
            yt, yc = treat.coupling(ctx), comp.coupling(ctx)
            rnd.append({"heads": [[int(l), int(h)] for l, h in s],
                        "treatment_frac_drop": (b_t - float(yt.mean())) / b_t,
                        "comparator_frac_drop": (b_c - float(yc.mean())) / b_c})

    rows_dump = [
        {"example_id": x.example_id, "gene_id": x.gene_id, "arm": arm,
         "Y": float(y), "gap": int(x.next_acceptor_pos - x.donor_pos),
         "n_isoforms_donor": x.n_isoforms_donor,
         "n_isoforms_acceptor": x.n_isoforms_acceptor,
         "exon_length": x.exon_length, "gene_type": x.gene_type}
        for arm, A, Y in (("GC", treat, base_t), ("GT", comp, base_c))
        for x, y in zip(A.rows, Y)
    ]

    result = {
        "schema": "gfmcd.m2m3.donor_class_dissection/1",
        "experiment": "m2m3_donor_class_dissection",
        "tier": "development",
        "split": "train",
        "sealed_chr8_chr9_opened": False,
        "seal_enforced_by": "allowed_chromosomes=train_chromosomes() at dataset build",
        "validation_reserve_spent": False,
        "ablation_mode": a.mode,
        "operator": MATCHED_OPERATOR,
        "cohort": {"tag": CURRENT_TAG, "k": a.k,
                   "n_treatment": len(cohort.treatment),
                   "n_comparator": len(cohort.comparators),
                   "n_treatment_genes": len(set(treat.genes)),
                   "n_comparator_genes": len(set(comp.genes)),
                   "balance": matching_balance(cohort)},
        "baseline": {"treatment_GC_mean_Y": b_t, "comparator_GT_mean_Y": b_c,
                     "rho": rho0[0], "rho_ci": [rho0[1], rho0[2]]},
        "per_head": per_head,
        "profile_agreement": {
            "pearson_frac_drop": pear,
            "spearman_rank": spear,
            "interpretation": ("high agreement => one circuit at lower gain; "
                               "low agreement => a distinct route for the minor class"),
        },
        "matched_random_controls": rnd,
        "rows": rows_dump,
        "licenses": (
            "Development-tier localisation of where the minor donor class's weaker acceptor "
            "coupling sits across a pre-declared head set. Ablation is attribution, not "
            "mediation: no route, delivery, or biological claim follows."
        ),
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:m2m3_donor_class_dissection", sys.argv,
        config=result["cohort"], seeds=dict(bootstrap=SEED), status="DEVELOPMENT",
        notes=["Train split only; chr8/9 not opened; validation reserve not spent.",
               "Ablation attribution only; licenses no mediation or route claim."])
    for f in (a.model, a.gtf, a.fasta):
        try:
            man.add_input(f)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f}")
    man.add_output(out)
    man.write(out.with_suffix(".manifest.json"))
    print(f"\nbaseline rho={rho0[0]:.4f} [{rho0[1]:.4f},{rho0[2]:.4f}]  "
          f"pearson={pear}  spearman={spear} -> {out}")


if __name__ == "__main__":
    main()
