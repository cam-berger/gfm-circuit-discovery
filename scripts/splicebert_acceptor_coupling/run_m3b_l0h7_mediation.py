#!/usr/bin/env python
"""M3b -- does L0H7 MEDIATE the donor-class difference, or merely correlate with it?

M2/M3 found the two donor classes engage the same head profile (Pearson 0.981) with one
exception: L0H7, donor-side payload assembly, is +0.139 on the minor ``GC`` class against
+0.011 on canonical ``GT``. That is an ABLATION result -- attribution, which ranks. It does
not show L0H7 carries the class signal.

This is the transport test. For each matched (GC, GT) pair, L0H7's output at the DONOR
dinucleotide tokens is transplanted from one example into the other, in both the clean and
the donor-destroyed pass, and the coupling is recomputed. Sequences are all 1024 nt but
donor_pos takes 544 distinct values, so the source is realigned position-by-position rather
than rolled.

    transport fraction  T = (Y_recipient_patched - Y_recipient) / (Y_source - Y_recipient)

    T ~ 0   L0H7 does not carry the class signal; its ablation effect is not mediation
    T ~ 1   L0H7 fully carries it

Controls: exact self-patch identity; a different L0 head; and an UNMATCHED donor, which
separates example-specific content from a generic class axis -- the distinction that turned
C41 on the donor side.

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

from gfmcd.circuits.hf_bert import capture_head_outputs, patch_heads  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, _dataset, eligible,
    load_adapter, pick_device,
)
from gfmcd.splicing.coupling import planted_donor_motifs, substitute  # noqa: E402
from gfmcd.splicing.donor_class import (  # noqa: E402
    COMPARATOR_MOTIF, MATCHED_OPERATOR, TREATMENT_MOTIF, flank_and_support_key,
    match_comparators,
)
from gfmcd.splicing.reverse_coupling import acceptor_readout  # noqa: E402

TAG = "m3b-l0h7-mediation-v1"
LAYER, HEAD = 0, 7
CONTROL_HEAD = 5           # a different layer-0 head, same layer and size
DONOR_TOKEN_SPAN = 2       # the donor dinucleotide itself
SEED = 0
N_BOOT = 2000


def _capture(task, rows, seqs):
    """L0 head outputs for one batch, plus the margin of that same pass."""
    pos = [acceptor_readout(x) for x in rows]
    enc = task.encode(seqs, pos)
    out = {}

    def fwd():
        with torch.no_grad():
            out["logits"] = task.ad.model(**enc).logits.float()

    cache = capture_head_outputs(task.ad, fwd)
    return cache, task.margin(out["logits"], pos).cpu().numpy()


def _aligned_source(cache_layer, src_rows, rec_rows, off):
    """Source tensor holding SRC's donor-token vectors at REC's donor-token indices.

    donor_pos differs between paired examples, so a positional copy is required; rolling
    would wrap unrelated context into the patched window.
    """
    src = torch.zeros_like(cache_layer)
    mask = torch.zeros(cache_layer.shape[0], cache_layer.shape[2], dtype=torch.bool)
    for j, (s, r) in enumerate(zip(src_rows, rec_rows)):
        for t in range(DONOR_TOKEN_SPAN):
            si, ri = s.donor_pos + off + t, r.donor_pos + off + t
            src[j, :, ri, :] = cache_layer[j, :, si, :]
            mask[j, ri] = True
    return src, mask


def _y_patched(task, rec_rows, src_rows, head, off, batch):
    """Coupling for the recipient arm with `head`'s donor-token output taken from source."""
    ys = []
    for i in range(0, len(rec_rows), batch):
        rec, src = rec_rows[i:i + batch], src_rows[i:i + batch]
        vals = []
        for seq_of in (lambda x: x.sequence,
                       lambda x: substitute(x.sequence, x.donor_pos, MATCHED_OPERATOR)):
            cache, _ = _capture(task, src, [seq_of(x) for x in src])
            aligned, mask = _aligned_source(cache[LAYER], src, rec, off)
            pos = [acceptor_readout(x) for x in rec]
            enc = task.encode([seq_of(x) for x in rec], pos)
            with torch.no_grad(), patch_heads(task.ad, {LAYER: aligned}, [(LAYER, head)],
                                              token_mask=mask):
                vals.append(task.margin(task.ad.model(**enc).logits.float(), pos)
                            .cpu().numpy())
        ys.append(vals[0] - vals[1])
    return np.concatenate(ys)


def _y_plain(task, rows, batch):
    ys = []
    for i in range(0, len(rows), batch):
        b = rows[i:i + batch]
        pos = [acceptor_readout(x) for x in b]
        with torch.no_grad():
            c = task.margin(task.ad.model(**task.encode([x.sequence for x in b], pos))
                            .logits.float(), pos).cpu().numpy()
            d = task.margin(task.ad.model(**task.encode(
                [substitute(x.sequence, x.donor_pos, MATCHED_OPERATOR) for x in b], pos))
                .logits.float(), pos).cpu().numpy()
        ys.append(c - d)
    return np.concatenate(ys)


def gboot_transport(y_rec, y_rec_patched, y_src, genes, *, n_draws=N_BOOT, seed=SEED):
    """Gene-clustered CI for T = (patched - recipient) / (source - recipient)."""
    by: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        by.setdefault(str(g), []).append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)

    def T(idx):
        num = y_rec_patched[idx].mean() - y_rec[idx].mean()
        den = y_src[idx].mean() - y_rec[idx].mean()
        return num / den if abs(den) > 1e-9 else np.nan

    est = T(np.arange(len(y_rec)))
    draws = []
    for _ in range(n_draws):
        picked = rng.integers(0, len(keys), len(keys))
        idx = np.array([r for j in picked for r in by[keys[j]]])
        draws.append(T(idx))
    draws = np.array([d for d in draws if np.isfinite(d)])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(est), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        raise SystemExit("--out is required (a runner must not default to a registered artifact)")

    ad = load_adapter(a.model, pick_device(a.device), eager=False)
    task = CouplingTask(ad, batch=a.batch)
    off = ad.prefix_tokens

    D = _dataset(a.gtf, a.fasta)
    pool = lambda m: [x for x in eligible(D, "train", donor_motifs=(m,))  # noqa: E731
                      if not planted_donor_motifs(x.sequence, x.donor_pos, MATCHED_OPERATOR)]
    gc_rows = pool(TREATMENT_MOTIF)
    if a.smoke:
        gc_rows = gc_rows[:8]
    coh = match_comparators(gc_rows, pool(COMPARATOR_MOTIF), k=1,
                            key_fn=flank_and_support_key)
    gc, gt = coh.treatment, coh.comparators
    genes = [x.gene_id for x in gc]

    y_gc, y_gt = _y_plain(task, gc, a.batch), _y_plain(task, gt, a.batch)

    # identity memorial: patching an example from ITSELF must be an exact no-op
    y_self = _y_patched(task, gc, gc, HEAD, off, a.batch)
    self_err = float(np.max(np.abs(y_self - y_gc)))

    # forward transport: give the GC example the GT partner's L0H7 donor-token output
    y_gc_from_gt = _y_patched(task, gc, gt, HEAD, off, a.batch)
    # reverse transport
    y_gt_from_gc = _y_patched(task, gt, gc, HEAD, off, a.batch)
    # control head, same layer
    y_gc_ctrl = _y_patched(task, gc, gt, CONTROL_HEAD, off, a.batch)
    # unmatched donor: example-specific content vs a generic class axis
    shuffled = list(gt[1:]) + [gt[0]]
    y_gc_unmatched = _y_patched(task, gc, shuffled, HEAD, off, a.batch)

    res = {}
    for name, rec, patched, src in (
        ("forward_GC_gets_GT", y_gc, y_gc_from_gt, y_gt),
        ("reverse_GT_gets_GC", y_gt, y_gt_from_gc, y_gc),
        ("control_head_L0H%d" % CONTROL_HEAD, y_gc, y_gc_ctrl, y_gt),
        ("unmatched_source", y_gc, y_gc_unmatched, y_gt),
    ):
        est, lo, hi = gboot_transport(rec, patched, src, genes)
        res[name] = {"recipient_mean_Y": float(rec.mean()),
                     "patched_mean_Y": float(patched.mean()),
                     "source_mean_Y": float(src.mean()),
                     "transport_fraction": est, "ci": [lo, hi]}
        print(f"  {name:26s} Y {rec.mean():.4f} -> {patched.mean():.4f} "
              f"(source {src.mean():.4f})   T={est:+.3f} [{lo:+.3f},{hi:+.3f}]", flush=True)

    memorials = {
        "self_patch_is_exact_noop": {
            "PASS": self_err < 1e-5, "max_abs_err": self_err,
            "note": "patching an example from itself must reproduce its own coupling exactly",
        },
        "patched_span_is_the_donor_dinucleotide": {
            "PASS": DONOR_TOKEN_SPAN == 2, "span": DONOR_TOKEN_SPAN,
            "forced": True,
            "note": "architecture-forced constant, not evidence",
        },
        "arms_are_paired_one_to_one": {
            "PASS": len(gc) == len(gt), "n": len(gc),
        },
    }
    all_pass = all(m["PASS"] for m in memorials.values())

    result = {
        "schema": "gfmcd.m3b.l0h7_mediation/1",
        "experiment": "m3b_l0h7_mediation",
        "tier": "development", "split": "train",
        "sealed_chr8_chr9_opened": False, "validation_reserve_spent": False,
        "mediator": f"L{LAYER}H{HEAD}", "control_head": f"L{LAYER}H{CONTROL_HEAD}",
        "operator": MATCHED_OPERATOR, "donor_token_span": DONOR_TOKEN_SPAN,
        "cohort": {"tag": TAG, "n_pairs": len(gc), "n_genes": len(set(genes))},
        "transport": res,
        "memorials": memorials, "all_memorials_pass": all_pass,
        "licenses": (
            "A development-tier transport test for whether L0H7's donor-token output carries "
            "the donor-class signal. Transport is mediation evidence for THIS head at THIS "
            "span only; it licenses no route, no delivery, and no biological claim."
        ),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    man = ExperimentManifest.create(
        "splicebert_acceptor_coupling:m3b_l0h7_mediation", sys.argv,
        config=result["cohort"], seeds=dict(bootstrap=SEED), status="DEVELOPMENT",
        notes=["Train split only; chr8/9 not opened; validation reserve not spent."])
    for f in (a.model, a.gtf, a.fasta):
        try:
            man.add_input(f)
        except Exception:
            man.notes.append(f"input not hashable in this checkout: {f}")
    man.add_output(out)
    man.write(out.with_suffix(".manifest.json"))
    print(f"\nself-patch max abs err {self_err:.2e}  memorials_pass={all_pass} -> {out}")


if __name__ == "__main__":
    main()
