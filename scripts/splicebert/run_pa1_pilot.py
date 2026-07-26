#!/usr/bin/env python
"""PA1 PILOT -- feasibility gates for the donor-position scan.

Draft protocol: studies/splicebert_partner_selection/preregistrations/DRAFT_PA1_DONOR_POSITION_SCAN.md
(v2, NOT FROZEN).  This runner answers the SS9 pilot questions ONLY.  It emits no addressing verdict
and reads no sealed split.

Gates implemented (SS3 memorial IDs):
  M1  standalone single-mask cell reproduces CouplingTask.coupling exactly
  M2  native A+ window-mask profile localizes the true donor near k=0        <- go/no-go
  M3  |P| profile magnitude exceeds the delta=0 replicate noise floor        <- go/no-go
  M7  delta=0 cell is bit-identical to native  => drho(0)=0, dhat=0
  M8  mask-width stability across w in {6,7,8}
  M9  MLM health under the long mask
  M10 estimator resolution (re-index by tau => dhat=tau).  FORCED by re-indexing -- validates the
      estimator, NOT the intervention.  Labelled as such in the artifact.
  PP  planted positive: S1 window-translation read in ABSOLUTE coordinates must displace by exactly
      delta.  Model-mediated (not forced) -- this is the real sensitivity gate.

Readout (SS1.1) is the FACTORIZED GT margin from one masked forward, a pseudo-likelihood score built
from independent per-position marginals -- never reported as log P(GT).

--smoke runs the arithmetic/mechanical gates on <=8 examples.
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

from gfmcd.provenance import ExperimentManifest, git_commit  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    CORRUPT_ALT, DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot,
    load_adapter, load_examples, pick_device, substitute,
)
from gfmcd.splicing.dataset import PyfaidxGenome  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "splicebert"))
from run_p3_shift_diagnostic import translation_variant  # noqa: E402

# --- frozen assay geometry (SS1.0) ------------------------------------------------------
MASK_HALF = 7                      # mask block [d-7, d+7]
KS = tuple(range(-6, 7))           # candidate offsets; candidate k uses (d+k, d+k+1)
PLANTED_DELTAS = (1, 2, 3)         # planted-positive translation sizes


# ========================================================================================
# Factorized GT margin  (SS1.1)
# ========================================================================================
def gt_margin(logits: torch.Tensor, bid: dict[str, int]) -> torch.Tensor:
    """D over candidate positions.  ``logits`` [B, P, V] restricted to the masked block.

    D(k) = l_k(G) + l_{k+1}(T) - max_{xy != GT} [ l_k(x) + l_{k+1}(y) ]

    The max factorizes exactly:
        max( max_{x!=G} l_k(x) + max_y l_{k+1}(y),  l_k(G) + max_{y!=T} l_{k+1}(y) )
    """
    g, t = bid["G"], bid["T"]
    acgt = torch.stack([logits[..., bid[b]] for b in "ACGT"], -1)     # [B, P, 4]
    lg, lt = acgt[..., 2], acgt[..., 3]                               # G is index 2, T index 3
    best_all = acgt.max(-1).values
    best_not_g = torch.stack([acgt[..., i] for i in (0, 1, 3)], -1).max(-1).values
    best_not_t = torch.stack([acgt[..., i] for i in (0, 1, 2)], -1).max(-1).values
    # candidate k pairs position p with p+1
    hit = lg[:, :-1] + lt[:, 1:]
    alt = torch.maximum(best_not_g[:, :-1] + best_all[:, 1:],
                        lg[:, :-1] + best_not_t[:, 1:])
    assert g == bid["G"] and t == bid["T"]
    return hit - alt                                                  # [B, P-1]


class ProfileScorer:
    """Window-mask donor-position profile.  One forward pass -> the whole profile."""

    def __init__(self, adapter, batch: int = 8, half: int = MASK_HALF):
        self.ad = adapter
        self.batch = batch
        self.half = half
        self.mask_id = adapter.tokenizer.mask_token_id
        self.bid = {b: adapter.tokenizer.convert_tokens_to_ids(b) for b in "ACGT"}
        self.off = adapter.prefix_tokens

    def profile(self, seqs: list[str], centers: list[int], ks=KS) -> np.ndarray:
        """[N, len(ks)] factorized-GT margin at offsets ``ks`` around each center."""
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), self.batch):
                S, C = seqs[i:i + self.batch], centers[i:i + self.batch]
                enc = self.ad.tokenizer([" ".join(s) for s in S], return_tensors="pt",
                                        padding=True).to(self.ad.device)
                for j, d in enumerate(C):
                    lo, hi = d - self.half, d + self.half
                    enc["input_ids"][j, lo + self.off: hi + self.off + 1] = self.mask_id
                logits = self.ad.model(**enc).logits.float()
                for j, d in enumerate(C):
                    lo = d - self.half + self.off
                    blk = logits[j, lo: lo + 2 * self.half + 1]        # [2h+1, V]
                    D = gt_margin(blk.unsqueeze(0), self.bid)[0]       # [2h]
                    # D index m corresponds to offset (m - half)
                    out.append(np.array([D[k + self.half].item() for k in ks]))
        return np.asarray(out)


# ========================================================================================
# Anchor comparison  (SS1.3)
# ========================================================================================
def _fisher(r: float) -> float:
    r = float(np.clip(r, -0.999999, 0.999999))
    return float(np.arctanh(r))


def anchor_delta_rho(p_native: np.ndarray, p_edit: np.ndarray, delta: int, ks=KS):
    """drho = z(rho_acceptor-locked) - z(rho_donor-locked) on IDENTICAL common support."""
    idx = {k: i for i, k in enumerate(ks)}
    support = [k for k in ks if (k - delta) in idx]
    if len(support) < 5:
        return None
    e = np.array([p_edit[idx[k]] for k in support])
    nd = np.array([p_native[idx[k]] for k in support])
    na = np.array([p_native[idx[k - delta]] for k in support])
    if e.std() < 1e-9 or nd.std() < 1e-9 or na.std() < 1e-9:
        return None
    return _fisher(np.corrcoef(e, na)[0, 1]) - _fisher(np.corrcoef(e, nd)[0, 1])


def displacement(p_native: np.ndarray, p_edit: np.ndarray, ks=KS, taus=range(-6, 7)):
    """dhat: parabolic interpolation about the argmax of correlation-vs-tau."""
    idx = {k: i for i, k in enumerate(ks)}
    rs, ts = [], []
    for t in taus:
        sup = [k for k in ks if (k - t) in idx]
        if len(sup) < 5:
            continue
        e = np.array([p_edit[idx[k]] for k in sup])
        n = np.array([p_native[idx[k - t]] for k in sup])
        if e.std() < 1e-9 or n.std() < 1e-9:
            continue
        rs.append(np.corrcoef(e, n)[0, 1])
        ts.append(t)
    if not rs:
        return float("nan")
    rs, ts = np.asarray(rs), np.asarray(ts)
    j = int(rs.argmax())
    if 0 < j < len(rs) - 1:
        y0, y1, y2 = rs[j - 1], rs[j], rs[j + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            return float(ts[j] + 0.5 * (y0 - y2) / den)
    return float(ts[j])


def _m10(P: np.ndarray, ks, taus=(-2, -1, 1, 2)) -> dict:
    """Forced re-index estimator check on an inner grid (shifted slice stays in the block)."""
    ki = tuple(k for k in ks if abs(k) <= max(ks) - 2)
    out = {}
    for tau in taus:
        if not all((k - tau) in ks for k in ki) or len(ki) < 5:
            out[str(tau)] = {"mean_abs_err": float("nan"), "n_grid": len(ki)}
            continue
        Pi = np.array([[P[i, ks.index(k)] for k in ki] for i in range(len(P))])
        sh = np.array([[P[i, ks.index(k - tau)] for k in ki] for i in range(len(P))])
        e = [abs(displacement(Pi[i], sh[i], ks=ki) - tau) for i in range(len(P))]
        out[str(tau)] = {"mean_abs_err": float(np.nanmean(e)), "n_grid": len(ki)}
    return out


# ========================================================================================
# Main
# ========================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--gtf", default=DEFAULT_GTF)
    ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/splicebert_coupling/pa1_pilot.json")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.n = min(a.n, 8)
    dev = pick_device(a.device)
    ad = load_adapter(a.model, dev)
    task = CouplingTask(ad, batch=a.batch)
    scorer = ProfileScorer(ad, batch=a.batch)

    X, coup = load_examples(a.gtf, a.fasta, a.split, a.n, task=task,
                            tag="pa1_pilot", min_effect=None)
    seqs = [x.sequence for x in X]
    dn = [x.donor_pos for x in X]
    genes = np.array([x.gene_id for x in X])
    print(f"[cohort] n={len(X)} genes={len(set(genes))} split={a.split} device={dev}")

    res: dict = {
        "status": "PILOT -- feasibility gates only; NO addressing verdict",
        "protocol": "DRAFT_PA1_DONOR_POSITION_SCAN.md v2 (NOT FROZEN)",
        "config": {"split": a.split, "n": len(X), "n_genes": int(len(set(genes))),
                   "mask_half": MASK_HALF, "ks": list(KS), "device": dev, "smoke": a.smoke,
                   "model": a.model},
        "gates": {},
    }

    # ---- M1: standalone single-mask cell reproduces C1 ---------------------------------
    single = task.coupling(X)
    res["gates"]["M1_single_mask_reproduces_C1"] = {
        "max_abs_diff": float(np.max(np.abs(single - coup))),
        "coupling_mean": float(single.mean()),
        "PASS": bool(np.max(np.abs(single - coup)) < 1e-4),
        "note": "exact arithmetic anchor; the window-mask P(0) is a DIFFERENT quantity",
    }

    # ---- native profiles ---------------------------------------------------------------
    corr_seqs = [substitute(x.sequence, x.next_acceptor_pos - 2, CORRUPT_ALT) for x in X]
    Dplus = scorer.profile(seqs, dn)
    Dminus = scorer.profile(corr_seqs, dn)
    P_native = Dplus - Dminus
    k0 = KS.index(0)

    # ---- M2: does the native A+ profile localize the true donor? -----------------------
    peak_k = np.array([KS[int(r.argmax())] for r in Dplus])
    res["gates"]["M2_native_profile_localizes_donor"] = {
        "frac_argmax_within1_of_0": float(np.mean(np.abs(peak_k) <= 1)),
        "frac_argmax_exact_0": float(np.mean(peak_k == 0)),
        "peak_k_hist": {int(k): int((peak_k == k).sum()) for k in KS if (peak_k == k).any()},
        "mean_D_at_0": float(Dplus[:, k0].mean()),
        "PASS": bool(np.mean(np.abs(peak_k) <= 1) >= 0.70),
        "note": "GO/NO-GO: if the window-masked model cannot find its own donor, the assay is dead",
    }

    # ---- M3: VACUOUS AS WRITTEN -- DISABLED (see prereg SS12B defect 5) -----------------
    # The original check re-ran DETERMINISTIC inference on identical inputs, so the "replicate
    # noise floor" was exactly 0.0 and `span > noise` could not fail.  A memorial that cannot
    # fail validates nothing.  Replacement (independent-cohort / independent-seed stability) is
    # part of the agreed redesign and is NOT implemented here.
    span = P_native.max(1) - P_native.min(1)
    res["gates"]["M3_profile_magnitude_DISABLED"] = {
        "median_span": float(np.median(span)),
        "P_at_0_mean": float(P_native[:, k0].mean()),
        "PASS": None,
        "note": "DISABLED: original form was forced (deterministic re-run => noise floor 0.0). "
                "Reported as a descriptive statistic only; gates nothing.",
    }

    # ---- M7: delta=0 identity ----------------------------------------------------------
    d0 = [anchor_delta_rho(P_native[i], P_native[i], 0) for i in range(len(X))]
    dh0 = [displacement(P_native[i], P_native[i]) for i in range(len(X))]
    res["gates"]["M7_zero_identity"] = {
        "max_abs_drho": float(np.nanmax(np.abs([v for v in d0 if v is not None]))),
        "max_abs_dhat": float(np.nanmax(np.abs(dh0))),
        "PASS": bool(np.nanmax(np.abs(dh0)) < 1e-6),
    }

    # ---- M10: estimator resolution (FORCED by re-indexing) -----------------------------
    # Evaluated on an INNER grid so the tau-shifted slice stays inside the SAME mask block.
    m10 = _m10(P_native, KS)
    res["gates"]["M10_estimator_resolution_FORCED"] = {
        "by_tau": m10,
        "note": "FORCED by re-indexing -- validates the ESTIMATOR only, not the intervention",
        "PASS": all(v["mean_abs_err"] < 0.5 for v in m10.values()),
    }

    if not a.smoke:
        genome = PyfaidxGenome(a.fasta)

        # ---- WIDTH SWEEP: the amplitude/resolution trade-off ---------------------------
        sweep = {}
        for w in (2, 3, 4, 5, 7):
            ks = tuple(k for k in KS if abs(k) <= w - 1)
            sc = ProfileScorer(ad, batch=a.batch, half=w)
            Dw = sc.profile(seqs, dn, ks=ks)
            Pw = Dw - sc.profile(corr_seqs, dn, ks=ks)
            j0 = ks.index(0)
            pk = np.array([ks[int(r.argmax())] for r in Dw])
            other = np.delete(Pw, j0, axis=1).mean(1)
            lo, hi = gboot(Pw[:, j0], genes)
            plo, phi = gboot(Pw[:, j0] - other, genes)
            sweep[str(w)] = {
                "n_candidates": len(ks),
                "D_argmax_eq_0": float(np.mean(pk == 0)),
                "D_argmax_within1": float(np.mean(np.abs(pk) <= 1)),
                "P_at_0": [float(Pw[:, j0].mean()), float(lo), float(hi)],
                "peakedness_P0_minus_mean": [float((Pw[:, j0] - other).mean()),
                                             float(plo), float(phi)],
                "M10_estimator": _m10(Pw, ks),
            }
        res["gates"]["WIDTH_SWEEP"] = sweep

        # ---- PP: planted positive -- S1 window translation in ABSOLUTE coordinates -----
        pp = {}
        for w in (5, 7):
            ks = tuple(k for k in KS if abs(k) <= w - 1)
            sc = ProfileScorer(ad, batch=a.batch, half=w)
            Pn = sc.profile(seqs, dn, ks=ks) - sc.profile(corr_seqs, dn, ks=ks)
            for d in PLANTED_DELTAS:
                V, keep = [], []
                for i, x in enumerate(X):
                    try:
                        V.append(translation_variant(x, d, genome))
                        keep.append(i)
                    except Exception:
                        pass
                if not keep:
                    continue
                vc = [substitute(v.sequence, v.next_acceptor_pos - 2, CORRUPT_ALT) for v in V]
                cen = [X[i].donor_pos for i in keep]   # ABSOLUTE coords: original donor index
                Pv = (sc.profile([v.sequence for v in V], cen, ks=ks)
                      - sc.profile(vc, cen, ks=ks))
                dh = np.array([displacement(Pn[i], Pv[j], ks=ks)
                               for j, i in enumerate(keep)])
                frac = float(np.nanmean(np.abs(dh - d) <= 0.5))
                pp[f"w{w}_d{d}"] = {
                    "n": len(keep), "true_delta": d, "mean_dhat": float(np.nanmean(dh)),
                    "attenuation_dhat_over_delta": float(np.nanmean(dh) / d),
                    "frac_within_0p5": frac, "PASS": bool(frac >= 0.80),
                }
        res["gates"]["PP_planted_positive_model_mediated"] = {
            "by_cell": pp,
            "threshold": "frac_within_0p5 >= 0.80 at |delta|=1, else STOP (SS4)",
            "note": "S1 translation read in ABSOLUTE coords; model-mediated, NOT a re-index identity",
            "PASS": all(v["PASS"] for k, v in pp.items() if k.endswith("_d1")),
        }

        # ---- M9: MLM health --------------------------------------------------------
        res["gates"]["M9_mlm_health"] = {
            "health": float(task.mlm_health(X)),
            "note": "reported; the frozen >=0.5-headroom gate applies to the confirmatory run",
        }

    # ---- variance / MDE ----------------------------------------------------------------
    if not a.smoke:
        lo, hi = gboot(P_native[:, k0], genes)
        res["precision"] = {
            "P_at_0_mean": float(P_native[:, k0].mean()),
            "P_at_0_ci95_gene_clustered": [float(lo), float(hi)],
            "half_width": float((hi - lo) / 2),
        }

    res["gates"]["all_pass"] = all(
        v.get("PASS", True) for v in res["gates"].values() if isinstance(v, dict))

    if not a.smoke:
        pp_ok = res["gates"]["PP_planted_positive_model_mediated"]["PASS"]
        res["verdict"] = {
            "proceed_to_confirmatory": False,
            "rule": "confirmatory not licensed; this prereg is NOT frozen",
            "statement": (
                "STOP -- confirmatory not run. NOTE: the original PP-based reason is WITHDRAWN "
                "(prereg SS12B). S1 window_translation moves the donor-acceptor pair TOGETHER across "
                "the absolute PE table, while the target holds the donor and moves only the acceptor; "
                "SpliceBERT is not translation-equivariant, so S1 is NOT a known-positive for the "
                "target and its failure does not bound the target's detectability. The stop stands "
                "because the design needs the SS12B redesign, not because the hypothesis is untestable."),
            "withdrawn_claim": (
                "'The assay cannot resolve a one-nucleotide translation' -- not licensed; the gate "
                "used the SECONDARY statistic (dhat, which has competing correlation peaks) rather "
                "than the primary (drho)."),
        }

    out = REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    manifest = ExperimentManifest.create(
        f"splicebert:{out.stem}", sys.argv, repo=REPO, status="EXPLORATORY",
        config=res["config"],
        notes=[res["status"], res["protocol"],
               "Feasibility gates only. No addressing verdict. Discovery split; test sealed."])
    manifest.add_output(out)
    manifest.write(out.with_suffix(".manifest.json"))
    print(json.dumps(res["gates"], indent=2)[:4000])
    print(f"\n[written] {out}")


if __name__ == "__main__":
    main()
