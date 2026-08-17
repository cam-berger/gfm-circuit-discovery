#!/usr/bin/env python
"""M5 -- exploratory MLP-write scan of the natural donor-class difference.

M1 found the masked-acceptor readout depends on a real ``GC-AG`` donor at ~0.22 of a
flank-matched ``GT-AG`` donor, an order of magnitude above a non-partner decoy. That is a
graded channel, but M1 cannot say WHICH stage is attenuated.

This assay mean- or zero-ablates each of the six MLP writes and re-measures coupling separately
on each arm.  The descriptive quantity is the shape of the two per-MLP profiles, not their heights:

    profiles correlate    -> one circuit at lower gain (the minor class engages the same
                             components, just less)
    profiles diverge      -> the minor class runs a different route

Fractional effects are used precisely because the arms differ ~5x in baseline; comparing raw
drops would rediscover M1 rather than localise it.

Development tier, train split.  This is an adaptive scan with no matched-random MLP family and
licenses no component or route claim.  The historical execution did not request chr8/9 or the
validation cohort; M6 subsequently spent its separate validation reserve.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.circuits.hf_bert import edit_mlp_writes  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _dataset,
    load_adapter, pick_device,
)
from gfmcd.splicing.coupling import planted_donor_motifs, substitute  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    assert_seal_intact, train_chromosomes,
    COMPARATOR_MOTIF, MATCHED_OPERATOR, TREATMENT_MOTIF, donor_motif,
    eligible_donor_motifs, gboot_two_sample_ratio, match_comparators, matching_balance,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout  # noqa: E402

CURRENT_TAG = "m5-mlp-donor-class-v1"
SEED = 0
N_BOOT = 2000
MIN_GENES = 30
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
    ap.add_argument("--dump-rows", action="store_true",
                    help="store per-row Y under every ablation; required for a per-MLP CI "
                         "and for a class-label permutation null")
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
        return [x for x in eligible_donor_motifs(D, "train", donor_motifs=(motif,))
                if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)]

    gc_rows = pool(TREATMENT_MOTIF)
    if a.smoke:
        gc_rows = gc_rows[:6]
    cohort = match_comparators(gc_rows, pool(COMPARATOR_MOTIF), k=a.k)
    treat, comp = Arm(task, cohort.treatment), Arm(task, cohort.comparators)

    layers = list(range(ad.num_layers))

    base_t, base_c = treat.coupling(_noop_ctx), comp.coupling(_noop_ctx)
    b_t, b_c = float(base_t.mean()), float(base_c.mean())
    rho0 = gboot_two_sample_ratio(base_t, treat.genes, base_c, comp.genes,
                                  n_draws=N_BOOT, seed=SEED, min_genes=MIN_GENES)

    per_mlp = {}
    for L in layers:
        ctx = lambda l=L: edit_mlp_writes(ad, [l], mode=a.mode)
        yt, yc = treat.coupling(ctx), comp.coupling(ctx)
        mt, mc = float(yt.mean()), float(yc.mean())
        per_mlp[f"MLP{L}"] = {
            "note": f"layer {L} MLP write",
            "treatment_GC": {"mean_Y": mt, "abs_drop": b_t - mt,
                             "frac_drop": (b_t - mt) / b_t if b_t else None},
            "comparator_GT": {"mean_Y": mc, "abs_drop": b_c - mc,
                              "frac_drop": (b_c - mc) / b_c if b_c else None},
        }
        if a.dump_rows:
            per_mlp[f"MLP{L}"]["rows"] = {
                "treatment_Y": [float(v) for v in yt],
                "comparator_Y": [float(v) for v in yc],
            }
        print(f"  MLP{L}: GC frac {(b_t-mt)/b_t:+.3f}   GT frac {(b_c-mc)/b_c:+.3f}   "
              f"absGC {b_t-mt:+.4f} absGT {b_c-mc:+.4f}", flush=True)

    ft = np.array([per_mlp[k]["treatment_GC"]["frac_drop"] for k in per_mlp])
    fc = np.array([per_mlp[k]["comparator_GT"]["frac_drop"] for k in per_mlp])
    pear = float(np.corrcoef(ft, fc)[0, 1]) if len(ft) > 2 else None
    order_t = np.argsort(np.argsort(-ft))
    order_c = np.argsort(np.argsort(-fc))
    spear = float(np.corrcoef(order_t, order_c)[0, 1]) if len(ft) > 2 else None

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
        "schema": "gfmcd.m5.mlp_donor_class/1",
        "experiment": "m5_mlp_donor_class",
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
        "baseline_rows": ({"treatment_Y": [float(v) for v in base_t],
                           "comparator_Y": [float(v) for v in base_c],
                           "treatment_genes": treat.genes,
                           "comparator_genes": comp.genes} if a.dump_rows else None),
        "baseline": {"treatment_GC_mean_Y": b_t, "comparator_GT_mean_Y": b_c,
                     "rho": rho0[0], "rho_ci": [rho0[1], rho0[2]]},
        "per_mlp": per_mlp,
        "profile_agreement": {
            "pearson_frac_drop": pear,
            "spearman_rank": spear,
            "interpretation": ("high agreement => one circuit at lower gain; "
                               "low agreement => a candidate layer-specific asymmetry"),
        },
        "matched_random_controls": None,
        "rows": rows_dump,
        "licenses": (
            "Adaptive development-tier MLP-write attribution. No matched-random MLP family, "
            "component necessity, route, delivery, or biological claim follows."
        ),
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:m5_mlp_donor_class", sys.argv,
        config=result["cohort"], seeds=dict(bootstrap=SEED), status="DEVELOPMENT",
        notes=["Train split only; this run does not request chr8/9 or validation rows.",
               "Adaptive MLP attribution only; no matched-random family or mechanism claim."])
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
