#!/usr/bin/env python
"""PA1 step 4 -- competing-algorithm test on a FRESH GENE-DISJOINT discovery cohort.

Draft protocol: DRAFT_PA1_DONOR_POSITION_SCAN.md SS12B/SS12C/SS12D (NOT FROZEN).
EXPLORATORY, discovery split only. Sealed chr8/9 NEVER opened.

TWO COMPETING ALGORITHMS
    acceptor-anchor : move acceptor by d  ->  the acceptor-dependent donor profile MOVES by d
    donor-query     : move acceptor by d  ->  attention retargets, the profile STAYS at the donor

PRIMARY STATISTIC: drho = z(rho_acceptor-locked) - z(rho_donor-locked) on identical common support.
    drho > 0 favours "the profile followed the shift"; drho < 0 favours "the profile stayed".
`dhat` is RETIRED as a gate/primary (SS12C: multimodal correlation surface); not computed here.

ARMS (all read with the SAME asymmetric mask, all vs the same native reference)
    T1     intron insert at midway        donor FIXED, acceptor +d      predict drho < 0
    T2     intron insert at gap//3        donor FIXED, acceptor +d      predict drho < 0  (2nd site)
    RECIP  exon insert at d-20 AND intron delete at midway
                                          donor +d, acceptor FIXED     predict drho > 0
    S2     insert 30nt AFTER the acceptor  nothing moves                predict drho <= 0 (sham)

RECIP IS THE SEQUENCE-LEVEL POSITIVE CONTROL.  The donor's 5'SS consensus moves WITH it but stays
inside the mask, so the ONLY channel by which the model can know the donor moved is the surviving
context outside the mask -- precisely the localiser the verification hypothesis posits.  If RECIP
cannot move the profile, the whole assay is insensitive and NOTHING is concluded from T1/T2.

DECISION (frozen before this run)
    RECIP drho > 0 CI excl 0  AND  T1 (and T2) drho < 0 CI excl 0
        -> verification-over-localization SUPPORTED at exploratory tier
    RECIP drho CI includes 0
        -> INSENSITIVE; T1/T2 uninterpretable; no conclusion
    RECIP > 0 and T1 > 0
        -> acceptor-anchor supported (would overturn SS12B's reading)
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

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot,
    load_adapter, load_examples, pick_device, substitute,
)
from run_p3_shift_diagnostic import choose_insertion, motif_counts  # noqa: E402
from run_pa1_pilot import anchor_delta_rho, gt_margin  # noqa: E402

# asymmetric mask: extends to d+10 so a donor consensus displaced by up to +3 stays HIDDEN
HALF_LO, HALF_HI = 7, 10
KS = tuple(range(-6, 7))
DELTAS = (1, 2, 3)
ACC_PROTECT = 40          # relaxed from IC1's 70 so the midway edit is constructible at gap>=86
MIN_GAP = 86
MIN_EXON = 25


class Scorer:
    """Factorized-GT-margin profile under an ASYMMETRIC mask [d-HALF_LO, d+HALF_HI]."""

    def __init__(self, ad, batch=6):
        self.ad, self.batch = ad, batch
        self.mask_id = ad.tokenizer.mask_token_id
        self.bid = {b: ad.tokenizer.convert_tokens_to_ids(b) for b in "ACGT"}
        self.off = ad.prefix_tokens

    def profile(self, seqs, centers):
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch):
                S, C = seqs[i:i + self.batch], centers[i:i + self.batch]
                enc = self.ad.tokenizer([" ".join(s) for s in S], return_tensors="pt",
                                        padding=True).to(self.ad.device)
                for j, d in enumerate(C):
                    enc["input_ids"][j, d - HALF_LO + self.off: d + HALF_HI + self.off + 1] = \
                        self.mask_id
                lg = self.ad.model(**enc).logits.float()
                for j, d in enumerate(C):
                    lo = d - HALF_LO + self.off
                    blk = lg[j, lo: lo + HALF_LO + HALF_HI + 1]
                    D = gt_margin(blk.unsqueeze(0), self.bid)[0]
                    out.append(np.array([D[k + HALF_LO].item() for k in KS]))
        return np.asarray(out)


# ---------------------------------------------------------------------------------------
# Variant builders
# ---------------------------------------------------------------------------------------
def _no_new_motif(orig: str, edited: str, lo: int, hi: int, hi2: int) -> bool:
    b, a = motif_counts(orig[lo:hi]), motif_counts(edited[lo:hi2])
    return not any(a[m] > b[m] for m in ("AG", "GT", "GC"))


def intron_insert(x, size, where="mid"):
    """Donor FIXED, acceptor +size. Length held by dropping far-right bases."""
    gap = x.next_acceptor_pos - x.donor_pos
    pos = x.donor_pos + (gap // 2 if where == "mid" else gap // 3)
    ins = choose_insertion(x.sequence, pos, size)
    seq = x.sequence[:pos] + ins + x.sequence[pos:len(x.sequence) - size]
    if not _no_new_motif(x.sequence, seq, max(0, pos - 18), pos + 18, pos + 18 + size):
        raise ValueError("motif created")
    return seq, x.donor_pos, x.next_acceptor_pos + size


def sham_downstream(x, size):
    """Nothing moves: edit 30 nt AFTER the partner acceptor."""
    pos = x.next_acceptor_pos + 30
    if pos >= len(x.sequence) - 40:
        raise ValueError("out of range")
    ins = choose_insertion(x.sequence, pos, size)
    seq = x.sequence[:pos] + ins + x.sequence[pos:len(x.sequence) - size]
    if not _no_new_motif(x.sequence, seq, max(0, pos - 18), pos + 18, pos + 18 + size):
        raise ValueError("motif created")
    return seq, x.donor_pos, x.next_acceptor_pos


def reciprocal(x, size):
    """Donor +size, acceptor FIXED, length preserved.

    Insert ``size`` in the upstream EXON at d-20 (outside the mask) and delete ``size`` from the
    intron midway between donor and acceptor.  Positions in [ins_pos, del_pos) shift by +size;
    positions >= del_pos+size are unchanged.  So the donor (and its whole local context) moves
    +size while the acceptor stays put.  The donor's consensus moves WITH it and remains inside
    the mask -> the only surviving cue is context.
    """
    gap = x.next_acceptor_pos - x.donor_pos
    ins_pos = x.donor_pos - 20
    del_pos = x.donor_pos + gap // 2
    if ins_pos < 12 or ins_pos > x.donor_pos - HALF_LO - 1:
        raise ValueError("insert inside mask / out of range")
    if del_pos + size > x.next_acceptor_pos - ACC_PROTECT:
        raise ValueError("delete inside protected acceptor zone")
    ins = choose_insertion(x.sequence, ins_pos, size)
    seq = (x.sequence[:ins_pos] + ins + x.sequence[ins_pos:del_pos]
           + x.sequence[del_pos + size:])
    if len(seq) != len(x.sequence):
        raise ValueError("length drift")
    if not _no_new_motif(x.sequence, seq, max(0, ins_pos - 18), ins_pos + 18, ins_pos + 18 + size):
        raise ValueError("motif created")
    return seq, x.donor_pos + size, x.next_acceptor_pos


BUILDERS = {
    "T1_acceptor_moves_mid": lambda x, s: intron_insert(x, s, "mid"),
    "T2_acceptor_moves_third": lambda x, s: intron_insert(x, s, "third"),
    "RECIP_donor_moves": reciprocal,
    "S2_sham_downstream": sham_downstream,
}
PREDICTION = {
    "T1_acceptor_moves_mid": "drho < 0 under donor-query",
    "T2_acceptor_moves_third": "drho < 0 under donor-query",
    "RECIP_donor_moves": "drho > 0 -- POSITIVE CONTROL; if this fails the assay is insensitive",
    "S2_sham_downstream": "drho <= 0 (nothing moved)",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--pool", type=int, default=600)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)  # computed below; a literal record
                    # path here would default a re-run onto the artifact of record
    ap.add_argument("--overwrite-record", action="store_true",
                    help="replace an existing artifact of record "
                         "(results are immutable by default)")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    dev = pick_device(a.device)
    ad = load_adapter(a.model, dev)
    task = CouplingTask(ad, batch=6)

    # ---- FRESH GENE-DISJOINT COHORT -----------------------------------------------------
    old, _ = load_examples(a.gtf, a.fasta, "val", 48, task=task, tag="pa1_pilot", min_effect=None)
    used = {x.gene_id for x in old}
    pool, _ = load_examples(a.gtf, a.fasta, "val", a.pool, task=task,
                            tag="pa1_step4_fresh", min_effect=None)
    X = [x for x in pool
         if x.gene_id not in used
         and (x.next_acceptor_pos - x.donor_pos) >= MIN_GAP
         and x.exon_length >= MIN_EXON
         and x.donor_pos >= 32][:(8 if a.smoke else a.n)]
    genes = np.array([x.gene_id for x in X])
    assert not (set(genes) & used), "cohort is NOT gene-disjoint"
    print(f"[cohort] n={len(X)} genes={len(set(genes))} disjoint_from_pilot=True "
          f"(pilot genes excluded: {len(used)}) device={dev}")

    sc = Scorer(ad, batch=6)
    seqs = [x.sequence for x in X]
    dn = [x.donor_pos for x in X]
    corr = [substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT) for x in X]
    P_nat = sc.profile(seqs, dn) - sc.profile(corr, dn)
    k0 = KS.index(0)
    other = np.delete(P_nat, k0, axis=1).mean(1)
    plo, phi = gboot(P_nat[:, k0] - other, genes)

    res = {
        "status": "EXPLORATORY -- fresh gene-disjoint discovery cohort; sealed split NOT opened",
        "protocol": "DRAFT_PA1_DONOR_POSITION_SCAN.md SS12D (NOT FROZEN)",
        "config": {"n": len(X), "n_genes": int(len(set(genes))), "mask": [HALF_LO, HALF_HI],
                   "ks": list(KS), "deltas": list(DELTAS), "acc_protect": ACC_PROTECT,
                   "min_gap": MIN_GAP, "device": dev, "gene_disjoint_from": "pa1_pilot (48 genes)"},
        "primary_statistic": "drho (dhat RETIRED per SS12C)",
        "native_profile": {
            "P_at_0": float(P_nat[:, k0].mean()),
            "peakedness": [float((P_nat[:, k0] - other).mean()), float(plo), float(phi)],
        },
        "arms": {},
    }

    for name, build in BUILDERS.items():
        res["arms"][name] = {"prediction": PREDICTION[name], "by_delta": {}}
        for d in DELTAS:
            S, C, keep = [], [], []
            for i, x in enumerate(X):
                try:
                    s, nd_, na_ = build(x, d)
                    S.append(s)
                    C.append(nd_)
                    keep.append((i, na_))
                except Exception:
                    pass
            if len(keep) < 10:
                res["arms"][name]["by_delta"][str(d)] = {"n": len(keep), "note": "insufficient"}
                continue
            # readout centre = the ORIGINAL donor index (absolute frame) for every arm
            cen = [X[i].donor_pos for i, _ in keep]
            Sc = [substitute(s, na - 2, CORRUPT_ALT) for s, (_, na) in zip(S, keep)]
            Pe = sc.profile(S, cen) - sc.profile(Sc, cen)
            dr = np.array([anchor_delta_rho(P_nat[i], Pe[j], d, ks=KS)
                           for j, (i, _) in enumerate(keep)], dtype=float)
            ok = ~np.isnan(dr)
            g = np.array([genes[i] for i, _ in keep])[ok]
            lo, hi = gboot(dr[ok], g)
            dP = float(np.median(np.abs(Pe - P_nat[[i for i, _ in keep]]).max(1)))
            res["arms"][name]["by_delta"][str(d)] = {
                "n": int(ok.sum()), "n_genes": int(len(set(g))),
                "drho_mean": float(dr[ok].mean()), "drho_ci95": [float(lo), float(hi)],
                "excludes_0": bool(lo > 0 or hi < 0),
                "sign": "positive" if dr[ok].mean() > 0 else "negative",
                "median_abs_dP": dP,
                "dP_over_peakedness": float(dP / (P_nat[:, k0] - other).mean()),
            }
            print(f"  {name:<26} d={d}  n={int(ok.sum()):>3}  drho={dr[ok].mean():+.3f} "
                  f"[{lo:+.3f},{hi:+.3f}]  excl0={bool(lo > 0 or hi < 0)}  |dP|={dP:.3f}")

    # ---- DECISION -----------------------------------------------------------------------
    def cell(nm, d):
        return res["arms"][nm]["by_delta"].get(str(d), {})

    rc = [cell("RECIP_donor_moves", d) for d in DELTAS]
    t1 = [cell("T1_acceptor_moves_mid", d) for d in DELTAS]
    t2 = [cell("T2_acceptor_moves_third", d) for d in DELTAS]
    recip_ok = any(c.get("excludes_0") and c.get("drho_mean", 0) > 0 for c in rc)
    t_neg = all(c.get("excludes_0") and c.get("drho_mean", 0) < 0 for c in t1 if c.get("n"))
    t2_neg = all(c.get("excludes_0") and c.get("drho_mean", 0) < 0 for c in t2 if c.get("n"))

    if not recip_ok:
        verdict = ("INSENSITIVE -- the reciprocal positive control did not move the profile. "
                   "T1/T2 are UNINTERPRETABLE; no conclusion about either algorithm.")
    elif t_neg and t2_neg:
        verdict = ("VERIFICATION-OVER-LOCALIZATION SUPPORTED (exploratory tier). The profile CAN be "
                   "moved by relocating the donor's surviving context (RECIP, drho>0), and does NOT "
                   "move when only the acceptor moves (T1 and T2, drho<0).")
    elif any(c.get("excludes_0") and c.get("drho_mean", 0) > 0 for c in t1):
        verdict = ("ACCEPTOR-ANCHOR SUPPORTED -- would OVERTURN SS12B. Re-audit before acting.")
    else:
        verdict = ("MIXED -- positive control passes but the target arms do not agree; "
                   "no algorithm call.")
    res["verdict"] = {"statement": verdict, "recip_positive_control_pass": bool(recip_ok),
                      "T1_all_negative_excl0": bool(t_neg), "T2_all_negative_excl0": bool(t2_neg),
                      "tier": "EXPLORATORY (discovery split; outcome-aware arc; NOT confirmatory)"}

    out = guard_output_path(REPO / (a.out or "results/splicebert_coupling/pa1_step4.json"),
                            overwrite=a.overwrite_record)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    m = ExperimentManifest.create(f"splicebert:{out.stem}", sys.argv, repo=REPO,
                                  status="EXPLORATORY", config=res["config"],
                                  notes=[res["status"], verdict])
    m.add_output(out)
    m.write(out.with_suffix(".manifest.json"))
    print("\nNATIVE peakedness:", res["native_profile"]["peakedness"])
    print("VERDICT:", verdict)
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
