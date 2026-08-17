"""Item 2 -- is the graded donor scale a SITE property or an emission of the acceptor readout?

M1: destroying a real GC-AG donor moves the masked-ACCEPTOR margin only 0.24x as much as
destroying a matched GT-AG donor. Two readings are still open:

  SITE     the model's representation of a GC donor is itself weak, and the acceptor result
           is simply downstream of that.
  EMISSION the model represents a GC donor about as well as a GT donor, but transmits much
           less of it down the long-range channel.

The discriminating measurement needs no intervention at all: mask the donor's OWN G and read
the margin. Position d carries G in BOTH classes, so the prediction task is identical across
arms and the comparison is clean -- the classes differ only at d+1 (T vs C).

    site_margin  = margin(G | donor position masked)

  site margins comparable across arms   -> EMISSION: attenuation is in transmission
  GC site margin much lower             -> SITE: the acceptor result is downstream

Also measured, as a secondary: the margin at d+1 for each arm's own true base, which asks how
confidently the model predicts the class-defining base itself.

Pure measurement. No ablation, no patching. chr8/9 never read.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gfmcd.splicing.circuit import (  # noqa: E402
    CouplingTask, _dataset, gboot_ratio, load_adapter, pick_device,
)
from gfmcd.splicing.coupling import planted_donor_motifs  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    COMPARATOR_MOTIF, MATCHED_OPERATOR, TREATMENT_MOTIF, assert_seal_intact,
    eligible_donor_motifs, flank_and_support_key, gboot_two_sample_ratio,
    match_comparators, train_chromosomes,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout, margins_at  # noqa: E402

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--gtf", default=DEFAULT_GTF)
_ap.add_argument("--fasta", default=DEFAULT_FASTA)
_ap.add_argument("--model", default=DEFAULT_MODEL)
_ap.add_argument("--device", default=None)
_ap.add_argument("--out", default=None)
_a = _ap.parse_args()
if _a.out is None:
    raise SystemExit("--out is required (a runner must not default to a registered artifact)")
GTF, FASTA, MODEL, OUT = _a.gtf, _a.fasta, _a.model, _a.out

ad = load_adapter(MODEL, pick_device(_a.device), eager=False)
task = CouplingTask(ad, batch=16)

D = _dataset(GTF, FASTA, allowed_chromosomes=train_chromosomes())
assert_seal_intact(D)


def pool(m):
    return [x for x in eligible_donor_motifs(D, "train", donor_motifs=(m,))
            if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)]


coh = match_comparators(pool(TREATMENT_MOTIF), pool(COMPARATOR_MOTIF), k=1,
                        key_fn=flank_and_support_key)
gc, gt = coh.treatment, coh.comparators
print(f"matched pairs: {len(gc)} GC / {len({x.gene_id for x in gc})} genes")

# ---- primary: the shared G at the donor position, an identical task in both arms
site_gc = margins_at(task, [x.donor_pos for x in gc], [x.sequence for x in gc])
site_gt = margins_at(task, [x.donor_pos for x in gt], [x.sequence for x in gt])

# ---- the acceptor-readout baseline on the SAME rows, for the direct comparison
acc_gc = margins_at(task, [acceptor_readout(x) for x in gc], [x.sequence for x in gc])
acc_gt = margins_at(task, [acceptor_readout(x) for x in gt], [x.sequence for x in gt])

g_gc = [x.gene_id for x in gc]
g_gt = [x.gene_id for x in gt]

site_rho = gboot_two_sample_ratio(site_gc, g_gc, site_gt, g_gt,
                                  n_draws=2000, seed=0, min_genes=30)
acc_rho = gboot_two_sample_ratio(acc_gc, g_gc, acc_gt, g_gt,
                                 n_draws=2000, seed=0, min_genes=30)

print()
print("=" * 74)
print("Donor SITE margin (mask the donor's own G) -- identical task in both arms")
print("=" * 74)
print(f"  GC arm mean {site_gc.mean():+.4f}    GT arm mean {site_gt.mean():+.4f}")
print(f"  site ratio  rho = {site_rho[0]:.4f} [{site_rho[1]:.4f}, {site_rho[2]:.4f}]")
print()
print("Clean masked-ACCEPTOR margin on the same rows (a level check, not the coupling)")
print(f"  GC arm mean {acc_gc.mean():+.4f}    GT arm mean {acc_gt.mean():+.4f}")
print(f"  ratio       rho = {acc_rho[0]:.4f} [{acc_rho[1]:.4f}, {acc_rho[2]:.4f}]")
print()
print("For reference, M1's COUPLING ratio on this cohort: rho = 0.242 [0.173, 0.326]")
print()
if site_rho[1] > 0.80:
    verdict = "EMISSION"
    why = ("the model represents a GC donor about as well as a GT donor at the site itself, "
           "so the acceptor attenuation is in TRANSMISSION, not in the site representation")
elif site_rho[2] < 0.50:
    verdict = "SITE"
    why = ("the GC donor site is itself weakly represented; the acceptor result is largely "
           "downstream of that and is not evidence about long-range transmission")
else:
    verdict = "MIXED_OR_UNRESOLVED"
    why = ("the site ratio sits between the two clean readings; part of the acceptor "
           "attenuation is inherited from the site and part is not")
print(f"VERDICT: {verdict}")
print(f"  {why}")

res = {
    "schema": "gfmcd.item2.forward_site_ratio/1",
    "tier": "development", "split": "train",
    "sealed_chr8_chr9_opened": False, "validation_reserve_spent": False,
    "n_pairs": len(gc), "n_genes": len(set(g_gc)),
    "site_margin": {"GC_mean": float(site_gc.mean()), "GT_mean": float(site_gt.mean()),
                    "rho": site_rho[0], "ci": [site_rho[1], site_rho[2]]},
    "clean_acceptor_margin": {"GC_mean": float(acc_gc.mean()), "GT_mean": float(acc_gt.mean()),
                              "rho": acc_rho[0], "ci": [acc_rho[1], acc_rho[2]]},
    "m1_coupling_rho_reference": [0.242, 0.173, 0.326],
    "verdict": verdict, "verdict_reason": why,
    "licenses": ("A development-tier comparison of how well the model represents the two donor "
                 "classes at the donor site itself. No component, route, or mediation claim."),
}
_out = Path(OUT)
_out.parent.mkdir(parents=True, exist_ok=True)
_out.write_text(json.dumps(res, indent=2))
_man = ExperimentManifest.create(
    "splicebert_acceptor_coupling:m1c_site_vs_emission", sys.argv,
    config={"n_pairs": len(gc), "n_genes": len(set(g_gc))}, seeds=dict(bootstrap=0),
    status="DEVELOPMENT",
    notes=["Train split only; chr8/9 not opened; validation reserve not spent.",
           "Pure measurement: no ablation, no patching."])
for _f in (MODEL, GTF, FASTA):
    try:
        _man.add_input(_f)
    except Exception:
        _man.notes.append(f"input not hashable in this checkout: {_f}")
_man.add_output(_out)
_man.write(_out.with_suffix(".manifest.json"))
print(f"\nwrote {OUT}")
