#!/usr/bin/env python
"""M6 -- the held-out confirmation. Spends the validation reserve, once.

Everything in the M-arc so far is development tier on train. This is the only held-out test,
and the cohort is small and non-renewable: 67 matched pairs across 56 genes, all that survives
the GG operator filter on chr11/16/19. It is powered to replicate a large effect once and
nothing subtler -- projected gene-clustered half-width about 0.25.

TWO endpoints, frozen before the split was read, Bonferroni at alpha = 0.025 each:

  PRIMARY    rho = mean(Y_GC) / mean(Y_GT), the donor-class coupling ratio.
             H0: rho >= 0.60   (no substantial attenuation)
             Reject if the 97.5th bootstrap percentile of rho falls below 0.60.
             Train estimate 0.242 [0.173, 0.326], so 0.60 sits well above the development
             interval and is a conservative target rather than a tuned one.

  SECONDARY  L3H1 donor attention ratio = mass(GC) / mass(GT) from the readout token.
             H0: ratio >= 0.70.  Train estimate 0.170/0.467 = 0.364.

DISCLOSURE. Both thresholds were chosen after seeing train. That is what development data is
for, and it is why they are set conservatively rather than at the observed values -- but it
means these are confirmations of a pre-specified direction and magnitude, not discoveries.

chr8/9 is never requested: the dataset is built with allowed_chromosomes restricted to the
validation split, so the sealed rows are not materialised, let alone read.
"""

from __future__ import annotations

import argparse
import hashlib
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
from gfmcd.splicing.coupling import CouplingConfig, planted_donor_motifs, substitute  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    COMPARATOR_MOTIF, MATCHED_OPERATOR, TREATMENT_MOTIF, donor_motif,
    flank_and_support_key, match_comparators, matching_balance,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout, edit_size, margins_at  # noqa: E402

TAG = "m6-validation-confirmation-v1"
ALPHA = 0.025                 # Bonferroni over exactly two endpoints
RHO_NULL = 0.60               # primary H0: rho >= RHO_NULL
ATTN_NULL = 0.70              # secondary H0: attention ratio >= ATTN_NULL
READER_LAYER, READER_HEAD = 3, 1
SEED, N_BOOT, MIN_GENES = 0, 4000, 30

#: Pinned by the frozen protocol, from an outcome-blind census run before any model call.
EXPECT_PAIRS, EXPECT_GENES = 67, 56
EXPECT_ROW_SHA = "0dff7e6fd314d237407c926fd13e5784bfbbd9138b24b78ef74e288a922301d9"
EXPECT_GENE_SHA = "cb0fe02d5fda8ecbdcae454e2647e16fee43f259184de41f26c1541ced609b51"


def _sha(items) -> str:
    return hashlib.sha256(json.dumps(sorted(items)).encode()).hexdigest()


def upper_bound(draws, alpha=ALPHA):
    """One-sided upper confidence bound: the (1-alpha) bootstrap percentile."""
    return float(np.quantile(draws, 1.0 - alpha))


def gboot_ratio_draws(num, gnum, den, gden, *, n_draws=N_BOOT, seed=SEED):
    """Gene-clustered bootstrap draws of mean(num)/mean(den) over the gene union."""
    tn: dict[str, list[int]] = {}
    for i, g in enumerate(gnum):
        tn.setdefault(str(g), []).append(i)
    dn: dict[str, list[int]] = {}
    for i, g in enumerate(gden):
        dn.setdefault(str(g), []).append(i)
    keys = sorted(set(tn) | set(dn))
    if len(keys) < MIN_GENES:
        raise ValueError(f"at least {MIN_GENES} gene clusters required; got {len(keys)}")
    num, den = np.asarray(num, float), np.asarray(den, float)
    ts = np.array([num[tn[k]].sum() if k in tn else 0.0 for k in keys])
    tc = np.array([float(len(tn[k])) if k in tn else 0.0 for k in keys])
    ds = np.array([den[dn[k]].sum() if k in dn else 0.0 for k in keys])
    dc = np.array([float(len(dn[k])) if k in dn else 0.0 for k in keys])
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for i in range(n_draws):
        p = rng.integers(0, len(keys), len(keys))
        a, b = tc[p].sum(), dc[p].sum()
        out[i] = (ts[p].sum() / a) / (ds[p].sum() / b) if a > 0 and b > 0 and ds[p].sum() > 0 \
            else np.nan
    return out[np.isfinite(out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--protocol-sha256", default=None,
                    help="SHA-256 of the frozen protocol authorising this run")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    # Authorisation before any caller-supplied path is inspected (ACR17).
    if a.out is None:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")
    if a.protocol_sha256 is None:
        raise SystemExit("--protocol-sha256 is required: this run spends the validation reserve "
                         "and must be authorised by a frozen protocol")

    cfg = CouplingConfig()
    val_chroms = tuple(cfg.val_chromosomes)
    ad = load_adapter(a.model, pick_device(a.device), eager=True)
    task = CouplingTask(ad, batch=a.batch)
    off = ad.prefix_tokens

    # chr8/9 is never requested, so it is never materialised from FASTA.
    D = _dataset(a.gtf, a.fasta, allowed_chromosomes=val_chroms)
    leaked = sorted({x.chrom for x in D if x.chrom in set(cfg.test_chromosomes)})
    if leaked:
        raise AssertionError(f"SEAL LEAK: {leaked}")

    pool = lambda m: [x for x in eligible(D, "val", donor_motifs=(m,))  # noqa: E731
                      if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)]
    coh = match_comparators(pool(TREATMENT_MOTIF), pool(COMPARATOR_MOTIF), k=1,
                            key_fn=flank_and_support_key)
    gc, gt = coh.treatment, coh.comparators
    g_gc = [x.gene_id for x in gc]
    g_gt = [x.gene_id for x in gt]
    row_sha, gene_sha = _sha(x.example_id for x in gc), _sha(set(g_gc))

    def Y(rows, alt=MATCHED_OPERATOR):
        pos = [acceptor_readout(x) for x in rows]
        return (margins_at(task, pos, [x.sequence for x in rows])
                - margins_at(task, pos, [substitute(x.sequence, x.donor_pos, alt)
                                         for x in rows]))

    y_gc, y_gt = Y(gc), Y(gt)
    draws = gboot_ratio_draws(y_gc, g_gc, y_gt, g_gt)
    rho = float(y_gc.mean() / y_gt.mean())
    rho_hi = upper_bound(draws)
    primary = "CONFIRMED" if rho_hi < RHO_NULL else "NOT_CONFIRMED"

    def attn_mass(rows):
        vals = []
        for i in range(0, len(rows), a.batch):
            b = rows[i:i + a.batch]
            pos = [acceptor_readout(x) for x in b]
            enc = task.encode([x.sequence for x in b], pos)
            with torch.no_grad():
                out = ad.model(**enc, output_attentions=True)
            A = out.attentions[READER_LAYER]
            for j, x in enumerate(b):
                r, d = pos[j] + off, x.donor_pos + off
                vals.append(float(A[j, READER_HEAD, r, d:d + 2].sum()))
            del out
        return np.asarray(vals)

    m_gc, m_gt = attn_mass(gc), attn_mass(gt)
    adraws = gboot_ratio_draws(m_gc, g_gc, m_gt, g_gt)
    aratio = float(m_gc.mean() / m_gt.mean())
    a_hi = upper_bound(adraws)
    secondary = "CONFIRMED" if a_hi < ATTN_NULL else "NOT_CONFIRMED"

    bal = matching_balance(coh)
    balanced = all(abs(bal["treatment"][p][b] - bal["comparator"][p][b]) < 1e-9
                   for p in bal["treatment"] for b in "ACGT")
    memorials = {
        "cohort_matches_the_frozen_census": {
            "PASS": (len(gc) == EXPECT_PAIRS and len(set(g_gc)) == EXPECT_GENES
                     and row_sha == EXPECT_ROW_SHA and gene_sha == EXPECT_GENE_SHA),
            "observed": {"pairs": len(gc), "genes": len(set(g_gc)),
                         "row_sha256": row_sha, "gene_sha256": gene_sha},
            "expected": {"pairs": EXPECT_PAIRS, "genes": EXPECT_GENES,
                         "row_sha256": EXPECT_ROW_SHA, "gene_sha256": EXPECT_GENE_SHA},
            "note": "the split may not have moved between freezing and spending",
        },
        "seal_never_materialised": {
            "PASS": not leaked, "allowed_chromosomes": list(val_chroms)},
        "native_motifs_as_declared": {
            "PASS": ({donor_motif(x) for x in gc} == {TREATMENT_MOTIF}
                     and {donor_motif(x) for x in gt} == {COMPARATOR_MOTIF})},
        "edit_size_matched": {
            "PASS": ({edit_size(donor_motif(x), MATCHED_OPERATOR) for x in gc}
                     == {edit_size(donor_motif(x), MATCHED_OPERATOR) for x in gt} == {1})},
        "matching_exactly_balanced": {"PASS": bool(balanced)},
    }
    all_pass = all(m["PASS"] for m in memorials.values())
    if not all_pass:
        primary = secondary = "MEMORIALS_FAILED"

    print(f"pairs {len(gc)} / genes {len(set(g_gc))}   memorials_pass={all_pass}")
    print(f"PRIMARY   rho = {rho:.4f}   upper bound {rho_hi:.4f}  vs H0 >= {RHO_NULL}"
          f"   -> {primary}")
    print(f"SECONDARY L3H1 attention ratio = {aratio:.4f}   upper bound {a_hi:.4f}"
          f"  vs H0 >= {ATTN_NULL}   -> {secondary}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "gfmcd.m6.validation_confirmation/1",
        "tier": "confirmatory", "split": "val",
        "sealed_chr8_chr9_opened": False,
        "seal_enforced_by": "allowed_chromosomes=val at dataset build",
        "validation_reserve_spent": True,
        "protocol_sha256": a.protocol_sha256,
        "alpha_per_endpoint": ALPHA,
        "n_pairs": len(gc), "n_genes": len(set(g_gc)),
        "cohort": {"tag": TAG, "row_sha256": row_sha, "gene_sha256": gene_sha,
                   "balance": bal},
        "primary": {"estimand": "mean(Y|GC-AG)/mean(Y|matched GT-AG), operator GG",
                    "rho": rho, "upper_bound": rho_hi, "null": RHO_NULL,
                    "verdict": primary,
                    "train_reference": [0.242, 0.173, 0.326]},
        "secondary": {"estimand": "L3H1 donor attention mass ratio at the readout token",
                      "ratio": aratio, "upper_bound": a_hi, "null": ATTN_NULL,
                      "verdict": secondary, "train_reference": 0.3644},
        "memorials": memorials, "all_memorials_pass": all_pass,
        "licenses": ("A held-out confirmation that this checkpoint's masked-acceptor readout "
                     "depends substantially less on a minor-class donor, and that L3H1 attends "
                     "to it substantially less. Quasi-observational; no component, mediation, "
                     "route, or biological claim follows."),
    }, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:m6_validation_confirmation", sys.argv,
        config={"n_pairs": len(gc), "protocol_sha256": a.protocol_sha256},
        seeds=dict(bootstrap=SEED), status="CONFIRMATORY",
        notes=["Validation split; the reserve is SPENT by this run.",
               "chr8/9 never requested at dataset build."])
    for f_ in (a.model, a.gtf, a.fasta):
        try:
            man.add_input(f_)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f_}")
    man.add_output(out)
    man.write(out.with_suffix(".manifest.json"))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
