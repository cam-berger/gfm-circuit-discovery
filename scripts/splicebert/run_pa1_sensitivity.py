#!/usr/bin/env python
"""PA1 sensitivity ladder -- Level 0 (query field) + Level 1 (final-residual shifted-write plant).

Draft protocol: DRAFT_PA1_DONOR_POSITION_SCAN.md SS12B/SS12C (NOT FROZEN).  Discovery split only.
EXPLORATORY, outcome-aware.  No verdict, no sealed data.

LEVEL 0 -- QUERY FIELD (proposed insertion ahead of the transplant ladder).
    A head's output lands at its QUERY position.  So "the acceptor-dependent write stays at the
    donor" is PARTLY ARCHITECTURE-FORCED whenever the querying position is the donor.  The
    discriminating measurement is therefore: across query positions d+k, how much does L3H1 attend to
    the acceptor window, and does that FIELD move when the acceptor moves?
        field fixed at d  -> retrieve-and-return: queries are pinned, retrieval target moves
        field moves       -> the whole candidate-donor query field tracks the acceptor

LEVEL 1 -- FINAL-RESIDUAL PLANT (instrument check; NEAR-FORCED, labelled).
    Transplant the acceptor-dependent final-residual delta from d+k to d+k+delta and re-apply the
    MLM head.  Because the MLM head is POSITION-SHARED, a translated residual delta yields a
    translated logit delta close to by construction -- this validates the scoring + displacement
    implementation ONLY.  Failure => the instrument is broken.  Success is NOT evidence for anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, READER, CouplingTask,
    load_adapter, load_examples, pick_device, substitute,
)
from gfmcd.splicing.unembed import donor_logits  # noqa: E402
from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from run_p3_shift_diagnostic import insertion_variant  # noqa: E402
from run_pa1_pilot import KS, MASK_HALF, ProfileScorer, anchor_delta_rho, displacement, gt_margin  # noqa: E402

L3, H3 = READER
ACC_OFFS = (-3, -2, -1, 0)      # the established four-token acceptor window (C22/C24)


def masked_ids(ad, seqs, centers, half=MASK_HALF):
    enc = ad.tokenizer([" ".join(s) for s in seqs], return_tensors="pt",
                       padding=True).to(ad.device)
    for j, d in enumerate(centers):
        enc["input_ids"][j, d - half + ad.prefix_tokens: d + half + ad.prefix_tokens + 1] = \
            ad.tokenizer.mask_token_id
    return enc


# ========================================================================================
# LEVEL 0 -- which query positions read the acceptor, and do they move?
# ========================================================================================
def query_field(ad, seqs, donors, acceptors, batch=1):
    """[N, len(KS)] L3H1 attention mass from query d+k to the four-token acceptor window."""
    out = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            S = seqs[i:i + batch]
            D = donors[i:i + batch]
            A = acceptors[i:i + batch]
            enc = masked_ids(ad, S, D)
            att = ad.model(**enc, output_attentions=True).attentions[L3][:, H3]  # [B,S,S]
            for j in range(len(S)):
                keys = [A[j] + o + ad.prefix_tokens for o in ACC_OFFS]
                row = []
                for k in KS:
                    q = D[j] + k + ad.prefix_tokens
                    row.append(float(att[j, q, keys].sum()))
                out.append(np.array(row))
            del att
    return np.asarray(out)


# ========================================================================================
# LEVEL 1 -- final-residual shifted-write plant (NEAR-FORCED instrument check)
# ========================================================================================
def residual_plant(ad, seqs, corr, donors, delta, half=MASK_HALF, batch=4):
    """P from planting the A+ minus A- final-residual delta at d+k+delta on the A- baseline."""
    bid = {b: ad.tokenizer.convert_tokens_to_ids(b) for b in "ACGT"}
    cls = ad.model.cls
    Pp = []
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            S, C, D = seqs[i:i + batch], corr[i:i + batch], donors[i:i + batch]
            Hp = ad.model(**masked_ids(ad, S, D), output_hidden_states=True).hidden_states[-1]
            Hm = ad.model(**masked_ids(ad, C, D), output_hidden_states=True).hidden_states[-1]
            Hplant = Hm.clone()
            for j, d in enumerate(D):
                lo = d - half + ad.prefix_tokens
                hi = d + half + ad.prefix_tokens
                Hplant[j, lo + delta: hi + delta + 1] += (Hp[j, lo:hi + 1] - Hm[j, lo:hi + 1])
            Lp = donor_logits(cls, Hplant).float()
            Lm = donor_logits(cls, Hm).float()
            for j, d in enumerate(D):
                lo = d - half + ad.prefix_tokens
                blk_p = Lp[j, lo: lo + 2 * half + 1]
                blk_m = Lm[j, lo: lo + 2 * half + 1]
                dp = gt_margin(blk_p.unsqueeze(0), bid)[0]
                dm = gt_margin(blk_m.unsqueeze(0), bid)[0]
                Pp.append(np.array([(dp[k + half] - dm[k + half]).item() for k in KS]))
    return np.asarray(Pp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)  # computed below; a literal record
                    # path here would default a re-run onto the artifact of record
    ap.add_argument("--overwrite-record", action="store_true",
                    help="replace an existing artifact of record "
                         "(results are immutable by default)")
    a = ap.parse_args()

    dev = pick_device(a.device)
    ad = load_adapter(a.model, dev, eager=True)      # eager REQUIRED to read attention
    task = CouplingTask(ad, batch=4)
    X, _ = load_examples(a.gtf, a.fasta, a.split, a.n, task=task,
                         tag="pa1_pilot", min_effect=None)
    seqs = [x.sequence for x in X]
    dn = [x.donor_pos for x in X]
    acc = [x.next_acceptor_pos for x in X]
    corr = [substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT) for x in X]
    print(f"[cohort] n={len(X)} genes={len(set(x.gene_id for x in X))} device={dev}")

    res = {
        "status": "EXPLORATORY sensitivity ladder L0+L1 -- discovery only, NO verdict",
        "protocol": "DRAFT_PA1_DONOR_POSITION_SCAN.md SS12B/SS12C (NOT FROZEN)",
        "config": {"split": a.split, "n": len(X), "mask_half": MASK_HALF, "ks": list(KS),
                   "reader": [L3, H3], "acc_offsets": list(ACC_OFFS), "device": dev},
    }

    # ---- LEVEL 0 -----------------------------------------------------------------------
    QF_nat = query_field(ad, seqs, dn, acc)
    l0 = {"native_mean_field": np.round(QF_nat.mean(0), 4).tolist(),
          "native_argmax_k": int(KS[int(QF_nat.mean(0).argmax())])}
    for d in (1, 3, 6):
        V, keep = [], []
        for i, x in enumerate(X):
            try:
                V.append(insertion_variant(x, "intron_insert", d))
                keep.append(i)
            except Exception:
                pass
        QF_e = query_field(ad, [v.sequence for v in V],
                           [X[i].donor_pos for i in keep],       # donor FIXED
                           [v.next_acceptor_pos for v in V])     # acceptor MOVED -> track it
        mn = QF_e.mean(0)
        # does the FIELD over k translate?  compare against the native field, shifted.
        dh = [displacement(QF_nat[i], QF_e[j]) for j, i in enumerate(keep)]
        dr = [anchor_delta_rho(QF_nat[i], QF_e[j], d) for j, i in enumerate(keep)]
        dr = [v for v in dr if v is not None]
        l0[f"delta_{d}"] = {
            "n": len(keep),
            "edited_mean_field": np.round(mn, 4).tolist(),
            "edited_argmax_k": int(KS[int(mn.argmax())]),
            "field_dhat_mean": float(np.nanmean(dh)),
            "field_drho_mean": float(np.mean(dr)) if dr else None,
            "total_acc_mass_native": float(QF_nat[keep].sum(1).mean()),
            "total_acc_mass_edited": float(QF_e.sum(1).mean()),
        }
    res["LEVEL0_query_field"] = l0
    res["LEVEL0_note"] = (
        "A head writes at its QUERY position. If the query field stays pinned at the donor while "
        "acceptor attention mass is retained, the fixed write location is ARCHITECTURE-EXPECTED, "
        "not an independent discovery -- and the L3H1 transplant (Level 2) measures CAPACITY, not "
        "the endogenous algorithm's choice.")

    # ---- LEVEL 1 -----------------------------------------------------------------------
    sc = ProfileScorer(ad, batch=4)
    P_nat = sc.profile(seqs, dn) - sc.profile(corr, dn)
    l1 = {}
    for d in (0, 1, 2, 3):
        Pp = residual_plant(ad, seqs, corr, dn, d)
        dh = np.array([displacement(P_nat[i], Pp[i]) for i in range(len(X))])
        dr = [anchor_delta_rho(P_nat[i], Pp[i], d) for i in range(len(X))]
        dr = [v for v in dr if v is not None]
        l1[str(d)] = {"mean_dhat": float(np.nanmean(dh)),
                      "frac_within_0p5": float(np.nanmean(np.abs(dh - d) <= 0.5)),
                      "mean_drho": float(np.mean(dr)) if dr else None,
                      "PASS": bool(np.nanmean(np.abs(dh - d) <= 0.5) >= 0.80)}
    res["LEVEL1_final_residual_plant"] = l1
    res["LEVEL1_note"] = (
        "NEAR-ARCHITECTURE-FORCED: the MLM head is position-shared, so a translated residual delta "
        "yields a translated logit delta close to by construction. Validates the scoring + "
        "displacement IMPLEMENTATION only. Failure => instrument broken. Success is NOT evidence.")

    out = guard_output_path(REPO / (a.out or "results/splicebert_coupling/pa1_sensitivity.json"),
                            overwrite=a.overwrite_record)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    m = ExperimentManifest.create(f"splicebert:{out.stem}", sys.argv, repo=REPO,
                                  status="EXPLORATORY", config=res["config"],
                                  notes=[res["status"], res["LEVEL0_note"], res["LEVEL1_note"]])
    m.add_output(out)
    m.write(out.with_suffix(".manifest.json"))
    print(json.dumps({k: v for k, v in res.items() if k.startswith("LEVEL")}, indent=2)[:3500])
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
