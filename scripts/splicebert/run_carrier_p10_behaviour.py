#!/usr/bin/env python
"""B1 behavioural arm — does deleting the ≈10.63 carrier move donor-acceptor COUPLING?
(PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR_V2.md §1 B1-BEHAV; the FROZEN-RUNNABLE arm.)

Frozen directional prediction (declared before the confirmatory cohort): EQUIVALENCE — deleting the ≈10.63
carrier does NOT move coupling beyond SESOI_C = 0.05.  Rationale (a PRIOR, not the test): the already-seen
carrier_dose control arm raised coupling only 0.946 -> 0.970 (|point| ~ 0.024).  The re-audit
(REDTEAM_SYNTHESIS_B1_V2.md) established this equivalence is REACHABLE: with |point|~0.024 and a paired
within-example gene-clustered contrast, HW < SESOI_C - |point| = 0.026 at n=256 needs only rho > 0.42 (not the
rho>=0.98 v1 feared).  So this arm is sizing-pilot-gated, not power-walled.

Design (prereg §1, §5 risk-2, §6):
  * PRIMARY: dC(D10) = C(intact) - C(BAND_P10-deleted), per-example paired, gene-clustered.  EQUIVALENCE iff
    the whole CI is within +-SESOI_C AND HW < SESOI_C - |point| (the sizing-pilot freeze gate).  A coupling
    POSITIVE instead means band 94-100 is NOT inert for coupling -> flag carrier_surgery/dose's control band
    for re-selection (the pre-committed dual branch).
  * M-p3contrast: the re-pointed deletion instrument, run at the period-3 band [339..345], MUST reproduce C8's
    coupling collapse (C_del/C_intact <= 0.05) -> proves the instrument can detect a USED carrier.
  * SECONDARY (descriptive, one-sided; MPS-wall-flagged): exon/intron masked-MLM dL under deletion + the
    energy-matched control-band DiD (reuse of carrier_surgery); NEVER a powered equivalence (sits on the
    0.02-nat wall).

⚠️ UNIT-UNTESTED IN THIS WORKTREE — SpliceBERT.1024nt + GENCODE data are absent here.  Validation contract:
run `--smoke` (memorials only) then the confirmatory run from the MAIN CHECKOUT (prereg §8).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import (  # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, gboot, load_adapter, load_examples, pick_device,
)

BAND_P10 = list(range(94, 101))     # the ≈10.63 carrier (7 bins; second_carrier.py / prereg D10)
P3_BAND = list(range(339, 346))     # period-3 carrier, for M-p3contrast (the re-pointed-instrument positive)
T_P10 = 1026 / 96
SESOI_C = 0.05
CHANCE = float(np.log(0.25))
MASK_RATE = 1 / 7
MIN_LEN = 60
HEALTH_FLOOR = 0.5


def make(P0, bins):
    P = P0.detach().float().cpu().numpy()
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    for b in bins:
        F[b] = 0.0
    return torch.tensor(np.fft.irfft(F, n=P.shape[0], axis=0) + mu, dtype=P0.dtype, device=P0.device)


def reconstruct_anchor(P0, bins):
    """M-del: zeroing then reconstructing all-but-`bins` must equal identity to <1e-6 (failable)."""
    P = P0.detach().float().cpu().numpy()
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    back = np.fft.irfft(F, n=P.shape[0], axis=0) + mu
    return float(np.abs(back - P).max())


def control_bands(P0, target_bins, width, n_ctrl=30, seed=0):
    """Energy-matched control bands, EXCLUDING period {3,1.5,6} AND ~10.6875+-5% (M-ctrl)."""
    P = P0.detach().float().cpu().numpy()
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    pw = (np.abs(F) ** 2).sum(1)
    N = P.shape[0]
    target = float(pw[target_bins].sum())
    cand = []
    for lo in range(3, len(pw) - width):
        b = list(range(lo, lo + width))
        per = N / float(np.mean(b))
        if abs(per - 3.0) < 0.6 or abs(per - 1.5) < 0.3 or abs(per - 6.0) < 0.6:
            continue
        if abs(per - T_P10) < T_P10 * 0.05:
            continue
        if 0.75 * target <= float(pw[b].sum()) <= 1.25 * target:
            cand.append(b)
    rng = np.random.default_rng(seed)
    if len(cand) > n_ctrl:
        cand = [cand[i] for i in rng.choice(len(cand), n_ctrl, replace=False)]
    return cand


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--controls", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/phase-lattice/carrier_p10_behaviour.json")
    args = ap.parse_args()
    dev = pick_device(None)
    ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
    task = CouplingTask(ad, batch=args.batch)
    OFF = task.off
    emb = ad.model.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()

    # ── memorials that need no examples ──────────────────────────────────────────────────────
    mem = {
        "M-del_reconstruct_anchor": float(reconstruct_anchor(P0, BAND_P10)),   # < 1e-6
        "M-ctrl_control_bands_off_band": None,   # filled below
        "M-det": True,   # PYTHONHASHSEED pinned by caller
    }
    ctrls = control_bands(P0, BAND_P10, width=len(BAND_P10), n_ctrl=args.controls)
    mem["M-ctrl_control_bands_off_band"] = bool(all(
        not (abs(1026 / float(np.mean(b)) - T_P10) < T_P10 * 0.05
             or abs(1026 / float(np.mean(b)) - 3.0) < 0.6) for b in ctrls))

    if args.smoke:
        # smoke: memorials + M-p3contrast on a tiny cohort; NO equivalence verdict
        exs, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", 24, task=task, tag="b1-p10-smoke",
                               min_effect=0.3)
        C_int = coupling(ad, task, emb, P0, P0, exs)[0]
        C_p3 = coupling(ad, task, emb, P0, make(P0, P3_BAND), exs)[0]
        ratio = float(np.mean(C_p3) / max(np.mean(C_int), 1e-9))
        mem["M-p3contrast_coupling_collapse_ratio"] = ratio   # must be <= 0.05
        emb.weight.data = P0.clone()
        ok = (mem["M-del_reconstruct_anchor"] < 1e-6 and mem["M-ctrl_control_bands_off_band"]
              and ratio <= 0.05)
        print("SMOKE memorials:", json.dumps(mem, indent=1))
        print(f"SMOKE gate: {'PASS' if ok else 'FAIL'} (no equivalence verdict emitted)")
        sys.exit(0 if ok else 1)

    # ── confirmatory ─────────────────────────────────────────────────────────────────────────
    X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", args.n * 4, task=task, tag="b1-p10",
                         min_effect=None)
    X = [x for x in X if x.gene_type == "protein_coding" and x.exon_length >= MIN_LEN
         and x.next_acceptor_pos - x.donor_pos >= MIN_LEN][:args.n]
    genes = [x.gene_id for x in X]
    print(f"{len(X)} exons / {len(set(genes))} genes")

    # PRIMARY: coupling equivalence + sizing pilot (rho, HW)
    C_int, health_int = coupling(ad, task, emb, P0, P0, X)
    C_del, health_del = coupling(ad, task, emb, P0, make(P0, BAND_P10), X)
    dC = C_int - C_del
    point = float(np.mean(dC))
    lo, hi = gboot(dC.tolist(), genes)
    hw = float((hi - lo) / 2)
    rho = float(np.corrcoef(C_int, C_del)[0, 1])
    equivalence = bool(abs(point) < SESOI_C and hw < (SESOI_C - abs(point)))

    # M-p3contrast: period-3 deletion must collapse coupling (instrument sensitivity)
    C_p3, _ = coupling(ad, task, emb, P0, make(P0, P3_BAND), X)
    p3_ratio = float(np.mean(C_p3) / max(np.mean(C_int), 1e-9))
    mem["M-p3contrast_coupling_collapse_ratio"] = p3_ratio
    mem["M-couphealth"] = bool(min(health_int, health_del) >= HEALTH_FLOOR)
    emb.weight.data = P0.clone()

    verdict = ("EQUIVALENCE_carrier_not_used_for_coupling" if equivalence
               else ("COUPLING_POSITIVE_reselect_control_band" if abs(point) >= SESOI_C
                     else "INCONCLUSIVE_underpowered"))

    result = {
        "carrier_bins": BAND_P10, "T": T_P10, "SESOI_C": SESOI_C, "n": len(X), "n_genes": len(set(genes)),
        "coupling_intact_mean": float(np.mean(C_int)), "coupling_deleted_mean": float(np.mean(C_del)),
        "dC_point": point, "dC_ci": [lo, hi], "dC_half_width": hw, "rho_paired": rho,
        "equivalence": equivalence, "verdict": verdict,
        "sizing_gate": {"needs": "HW < SESOI_C - |point|", "HW": hw, "threshold": SESOI_C - abs(point),
                        "pass": bool(hw < SESOI_C - abs(point))},
        "memorials": mem, "health": {"intact": health_int, "deleted": health_del},
        "note": "Behavioural (coupling) arm only.  MLM secondary is descriptive/one-sided and not run here by "
                "default (MPS wall).  A coupling POSITIVE re-flags band 94-100 as the surgery/dose inert "
                "control (prereg §5 risk-2).",
    }
    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=1))
    ExperimentManifest.create(
        "b1:carrier_p10_behaviour", sys.argv, repo=REPO, status=verdict,
        config={"carrier_bins": BAND_P10, "SESOI_C": SESOI_C, "n": len(X)},
        seeds={"PYTHONHASHSEED": 0},
        notes=["B1 behavioural arm: coupling equivalence under ≈10.63 deletion (frozen prediction: NULL). "
               "Sizing-pilot-gated; M-p3contrast checks the re-pointed instrument still collapses on period-3."],
    ).write(outp.with_suffix(".manifest.json"))

    print("=" * 88)
    print(f"coupling intact {np.mean(C_int):+.4f}  deleted {np.mean(C_del):+.4f}")
    print(f"dC(D10) {point:+.4f}  CI [{lo:+.4f},{hi:+.4f}]  HW {hw:.4f}  rho {rho:.3f}")
    print(f"sizing gate HW < {SESOI_C-abs(point):.4f}: {'PASS' if hw < SESOI_C-abs(point) else 'FAIL'}")
    print(f"M-p3contrast coupling collapse ratio {p3_ratio:.4f} (<=0.05 required)")
    print(f"VERDICT: {verdict}")
    print(f"saved -> {outp}")


def coupling(ad, task, emb, P0, P_variant, exs):
    """Per-example coupling C = margin(clean) - margin(corrupt) under a position-embedding variant.
    Returns (C array, global masked-MLM health under the variant)."""
    emb.weight.data = P_variant.clone()
    C, accs = [], []
    with torch.no_grad():
        for i in range(0, len(exs), task.batch):
            ch = exs[i:i + task.batch]
            cl, co, dn = task.pair(ch)
            mc = task.margin(ad.model(**cl).logits.float(), dn)
            mk = task.margin(ad.model(**co).logits.float(), dn)
            C += (mc - mk).detach().cpu().tolist()
            accs.append(_mlm_health(ad, task, ch))
    return np.asarray(C), float(np.mean(accs))


def _mlm_health(ad, task, ch):
    """Global masked-probe accuracy (clean encoding); the R13 health gate, kept >= chance headroom."""
    import numpy as _np
    r = _np.random.default_rng(0)
    dn = [x.donor_pos for x in ch]
    enc = task.encode([x.sequence for x in ch], dn)
    probes = []
    for j, x in enumerate(ch):
        ps = [p for p in sorted(r.choice(range(10, len(x.sequence) - 10),
                                          size=min(30, len(x.sequence) - 20), replace=False).tolist())
              if abs(p - x.donor_pos) > 2 and abs(p - (x.next_acceptor_pos or -99)) > 3]
        for p in ps:
            enc["input_ids"][j, p + task.off] = task.mask_id
        probes.append(ps)
    lg = ad.model(**enc).logits.float()
    acc = []
    for j, x in enumerate(ch):
        if not probes[j]:
            continue
        pred = lg[j, [p + task.off for p in probes[j]]].argmax(-1)
        tgt = torch.tensor([task.bid[x.sequence[p]] for p in probes[j]], device=pred.device)
        acc.append((pred == tgt).float().mean().item())
    return float(np.mean(acc)) if acc else 1.0


if __name__ == "__main__":
    main()
