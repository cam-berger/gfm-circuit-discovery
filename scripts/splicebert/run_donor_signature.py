#!/usr/bin/env python
"""THE CAUSAL SIGNATURE OF U_info -- what does it encode?  (donor-graph 1e, core)

1a-1c mapped WHERE the acceptor-dependent signal flows (L3H1 -> MLP3/MLP4 -> L4/L5
transporter heads -> U_info -> donor logit).  1e asks WHAT U_info encodes, refusing to
characterise it by AG->CC corruption alone.  Because C16 showed U_info carries the coupling,
the coupling's signature IS U_info's signature.

Core conditions (annotation-free):
  - partner vs non-partner acceptor  -> paired-intron consistency vs generic splice evidence
  - donor vs non-site G              -> donor-identity confidence vs a global G shift
  - intron length mod 3              -> raw cross-site phase alignment?
  - strand                           -> symmetry sanity
Cited: period-3 dependence (C8).  Deferred to the period-3 bridge: coding-vs-noncoding and
annotated splice phase (need CDS annotation).

Prereg (FROZEN): studies/splicebert_partner_selection/preregistrations/PREREGISTRATION_DONOR_SIGNATURE.md
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot,
    load_adapter, load_examples, matched_decoy_acceptor, pick_device,
)
from gfmcd.splicing.coupling import substitute  # noqa: E402
from gfmcd.splicing.phase import example_phase, load_cds_frames  # noqa: E402


def pick_nonsite_g(x, min_gap: int = 30) -> int | None:
    d, a = x.donor_pos, x.next_acceptor_pos
    cands = [p for p in range(len(x.sequence))
             if x.sequence[p] == "G" and abs(p - d) >= min_gap
             and (a is None or abs(p - a) >= min_gap)]
    if not cands:
        return None
    mid = len(x.sequence) // 2
    return min(cands, key=lambda p: abs(p - mid))


def ci_str(lohi):
    return f"[{lohi[0]:+.3f},{lohi[1]:+.3f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/donor_signature.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    dev = pick_device(args.device)
    ad = load_adapter(args.model, dev, eager=False)     # margins only; no attention needed
    task = CouplingTask(ad, batch=args.batch)

    X, _ = load_examples(args.gtf, args.fasta, args.split, args.n, task=task,
                         tag=f"signature-{args.split}", min_effect=None)
    genes = np.array([x.gene_id for x in X])
    print(f"{len(X)} unconditional {args.split} exons   device={dev}")

    # splice-phase / coding annotation (the CDS bridge)
    val_chroms = {c if c.startswith("chr") else f"chr{c}" for c in {x.chrom for x in X}}
    cds = load_cds_frames(args.gtf, allowed_chromosomes=val_chroms)
    ph = [example_phase(x, cds) for x in X]
    coding = np.array([p is not None for p in ph])
    phase = np.array([p if p is not None else -1 for p in ph])

    def margins_at(seqs, positions):
        """Masked-G margin at `positions` for each sequence (mask -> logit(G)-max(A,C,T))."""
        out = []
        for i in range(0, len(seqs), args.batch):
            s, p = seqs[i:i + args.batch], positions[i:i + args.batch]
            enc = task.encode(s, p)
            out += task.margin(ad.model(**enc).logits.float(), p).tolist()
        return np.asarray(out)

    dn = [x.donor_pos for x in X]
    clean_seq = [x.sequence for x in X]
    partner_seq = [substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT) for x in X]

    # baseline coupling at the donor
    m_clean_don = margins_at(clean_seq, dn)
    m_partner_don = margins_at(partner_seq, dn)
    coup = m_clean_don - m_partner_don

    # --- partner vs non-partner acceptor ------------------------------------------------------
    decoys = [matched_decoy_acceptor(x) for x in X]
    has_dec = np.array([d is not None for d in decoys])
    decoy_seq = [substitute(x.sequence, d - 2, CORRUPT_ALT) if d is not None else x.sequence
                 for x, d in zip(X, decoys)]
    m_decoy_don = margins_at(decoy_seq, dn)
    nonpartner_coup = m_clean_don - m_decoy_don            # ~0 if partner-specific

    # --- donor vs non-site G ------------------------------------------------------------------
    ns = [pick_nonsite_g(x) for x in X]
    has_ns = np.array([p is not None for p in ns])
    ns_pos = [p if p is not None else x.donor_pos for x, p in zip(X, ns)]   # placeholder pos
    m_clean_ns = margins_at(clean_seq, ns_pos)
    m_partner_ns = margins_at(partner_seq, ns_pos)
    nonsite_coup = m_clean_ns - m_partner_ns              # ~0 if the corruption is donor-specific

    # --- assemble signature -------------------------------------------------------------------
    partner_ci = gboot(coup[has_dec], genes[has_dec])
    nonpartner_ci = gboot(nonpartner_coup[has_dec], genes[has_dec])
    donor_ci = gboot(coup, genes)
    nonsite_ci = gboot(nonsite_coup[has_ns], genes[has_ns])

    strand = np.array([x.strand for x in X])
    strand_groups = {s: gboot(coup[strand == s], genes[strand == s])
                     for s in ("+", "-") if (strand == s).any()}
    strand_means = {s: float(coup[strand == s].mean()) for s in strand_groups}

    mod3 = np.array([(x.intron_length_down % 3) if x.intron_length_down is not None else -1
                     for x in X])
    mod3_groups = {int(k): {"mean": float(coup[mod3 == k].mean()), "n": int((mod3 == k).sum()),
                            "ci": list(gboot(coup[mod3 == k], genes[mod3 == k]))}
                   for k in (0, 1, 2) if (mod3 == k).any()}

    # coding-vs-noncoding (exon is / is not a clean CDS) and annotated splice phase 0/1/2
    coding_ci = gboot(coup[coding], genes[coding])
    noncoding_ci = gboot(coup[~coding], genes[~coding]) if (~coding).any() else None
    phase_groups = {int(k): {"mean": float(coup[phase == k].mean()), "n": int((phase == k).sum()),
                             "ci": list(gboot(coup[phase == k], genes[phase == k]))}
                    for k in (0, 1, 2) if (phase == k).any()}

    s_partner = partner_ci[0] > nonpartner_ci[1]
    s_donor = donor_ci[0] > nonsite_ci[1]
    s_strand = (len(strand_groups) < 2) or (
        strand_groups["+"][0] <= strand_groups["-"][1] and strand_groups["-"][0] <= strand_groups["+"][1])

    verdict = {"S_partner": bool(s_partner), "S_donor": bool(s_donor), "S_strand": bool(s_strand)}
    verdict["overall"] = all(verdict.values())

    R = {
        "split": args.split, "n": len(X), "device": dev,
        "partner_vs_nonpartner": {
            "partner_mean": float(coup[has_dec].mean()), "partner_ci": list(partner_ci),
            "nonpartner_mean": float(nonpartner_coup[has_dec].mean()), "nonpartner_ci": list(nonpartner_ci),
            "n_with_decoy": int(has_dec.sum())},
        "donor_vs_nonsite": {
            "donor_mean": float(coup.mean()), "donor_ci": list(donor_ci),
            "nonsite_mean": float(nonsite_coup[has_ns].mean()), "nonsite_ci": list(nonsite_ci),
            "n_with_nonsite": int(has_ns.sum())},
        "by_strand": {s: {"mean": strand_means[s], "ci": list(strand_groups[s]),
                          "n": int((strand == s).sum())} for s in strand_groups},
        "by_intron_mod3": mod3_groups,
        "by_coding": {"coding_mean": float(coup[coding].mean()), "coding_ci": list(coding_ci),
                      "n_coding": int(coding.sum()),
                      "noncoding_mean": float(coup[~coding].mean()) if (~coding).any() else None,
                      "noncoding_ci": list(noncoding_ci) if noncoding_ci else None,
                      "n_noncoding": int((~coding).sum())},
        "by_phase": phase_groups,
        "verdict": {k: bool(v) for k, v in verdict.items()},
        "REFLECTION": {"status": "PENDING -- fill after inspection (prereg section 5)"},
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(R, indent=1))
    ExperimentManifest.create(
        "splicebert_coupling:donor_signature", sys.argv,
        config={"split": args.split, "n": len(X)},
        notes=["1e core: the causal signature of U_info (the coupling's signature).",
               "Coding-vs-noncoding and annotated splice phase deferred to the period-3 bridge.",
               "Period-3 dependence cited from C8, not re-run here."],
    ).write(out.parent / "donor_signature.manifest.json")

    print("\n" + "=" * 78)
    print(f"S-Partner:  partner {coup[has_dec].mean():+.3f} {ci_str(partner_ci)}  vs "
          f"non-partner {nonpartner_coup[has_dec].mean():+.3f} {ci_str(nonpartner_ci)}  "
          f"(n={int(has_dec.sum())})")
    print(f"S-Donor:    donor {coup.mean():+.3f} {ci_str(donor_ci)}  vs "
          f"non-site G {nonsite_coup[has_ns].mean():+.3f} {ci_str(nonsite_ci)}  "
          f"(n={int(has_ns.sum())})")
    print("by strand:  " + "   ".join(
        f"{s} {strand_means[s]:+.3f} {ci_str(strand_groups[s])} (n={int((strand==s).sum())})"
        for s in strand_groups))
    print("by intron mod 3:  " + "   ".join(
        f"{k}: {v['mean']:+.3f} {ci_str(v['ci'])} (n={v['n']})" for k, v in mod3_groups.items()))
    print(f"by coding:  coding {coup[coding].mean():+.3f} {ci_str(coding_ci)} (n={int(coding.sum())})"
          + (f"   noncoding {coup[~coding].mean():+.3f} {ci_str(noncoding_ci)} (n={int((~coding).sum())})"
             if noncoding_ci else "   (no noncoding exons)"))
    print("by phase:  " + "   ".join(
        f"{k}: {v['mean']:+.3f} {ci_str(v['ci'])} (n={v['n']})" for k, v in phase_groups.items()))
    print("\nVERDICT:")
    for k, v in verdict.items():
        print(f"  {k:10s} {'PASS' if v else 'FAIL'}")
    print("=" * 78)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
