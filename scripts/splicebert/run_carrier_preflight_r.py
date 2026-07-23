#!/usr/bin/env python
"""B1 Preflight R — does any head on the SpliceBERT.1024nt checkpoint reach post-softmax Level-A
at the ≈10.63 carrier band?  (PREREGISTRATION_B1_CARRIER_OV_BEHAVIOR_V2.md §2, the decisive fork.)

WHY THIS IS THE FORK.  C11's eight ≈10.63 Level-A readers were established on the 510nt checkpoints
(run_postsoftmax.py MODELS_A: multispecies = SpliceBERT.510nt, bins [46,47,48], T=10.61).  B1's OV +
behaviour machinery (U_info, coupling, carrier_surgery/dose, the C35 A-vs-V decomposition) lives ONLY on
SpliceBERT.1024nt, whose ≈10.63 QK readers (second_carrier.json) are DISJOINT and pre-softmax only.  So the
OV arm cannot assume C11's head list; it must re-establish the reader set ON-MODEL at 1024nt.  This script
IS that establishment:

    R_1024 = { heads reaching Level-A at BAND_P10 = [95,96,97] (peak bin 96, period 1026/96 = 10.6875) }

    R_1024 non-empty  -> the OV arm is live; run_carrier_ov_av.py loops over these heads.
    R_1024 = EMPTY    -> the OV arm is NON-IDENTIFIABLE by construction (Endpoint B); B1 collapses to the
                         behavioural-null arm alone (run_carrier_p10_behaviour.py).  This is the modal
                         outcome the prereg pre-commits to, and it is a legitimate close, not a failure.

REUSE.  The bounded post-softmax estimator (m_T, triangle-inequality-bounded in [0,1]), the per-head
permutation null, the mandatory deletion audit, and the FROZEN Level-A criteria are imported VERBATIM from
run_postsoftmax.py — not reimplemented (the R14/R15 lessons live in that instrument).  The only net-new
content is (a) the 1024nt model entry with BAND_P10 + a 1024nt-native energy-matched reciprocal band + the
low-leak band, and (b) writing R_1024.

⚠️ UNIT-UNTESTED IN THIS WORKTREE — the SpliceBERT.1024nt checkpoint and hg38.fa are absent here.  This
driver's validation contract is running it from the MAIN CHECKOUT (where data/models + genomes live); the
reused postsoftmax preflight gate is the mechanics check.  See PREREGISTRATION_B1_..._V2.md §8.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/splicebert"))

from gfmcd.provenance import ExperimentManifest  # noqa: E402
from gfmcd.splicing.circuit import load_adapter, pick_device  # noqa: E402
# Reuse the VALIDATED post-softmax instrument verbatim (R14/R15 hardened):
from run_postsoftmax import (  # noqa: E402
    m_of, profiles, deletion_audit, sha, LEVELA, FROZEN, LOWLEAK, M1024,
)
from transformers import AutoTokenizer  # noqa: E402

# ── the 1024nt ≈10.63 carrier (net-new; the only re-pointed parameters) ──────────────────────────
# The full ≈10.63 carrier is the 7-bin band [94..100] (second_carrier.py BAND_P10; the prereg D10), analogous
# to the 7-bin period-3 band [339..345]; peak bin 96 sets the period.  (postsoftmax used a tight 3-bin carrier
# on the DIFFERENT 510nt table; on 1024nt the carrier is 7 bins.)
BAND_P10 = list(range(94, 101))  # the ≈10.63 carrier band on the 1026-row table
T_P10 = 1026 / 96                 # = 10.6875, from the peak bin 96
OUT = REPO / "results/phase-lattice/carrier_preflight_r.json"


def energy_matched_reciprocal(P: np.ndarray, carrier_bins, width=3, seed=0):
    """A 1024nt-native reciprocal band: same width, energy within 25% of the carrier band, period NOT
    near {3, 10.6875, harmonics}.  Deleting it must NOT collapse a genuine ≈10.63 reader's m_T (specificity).
    Deterministic (nearest-energy pick)."""
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    pw = (np.abs(F) ** 2).sum(1)
    N = P.shape[0]
    target = float(pw[carrier_bins].sum())
    cand = []
    for lo in range(3, len(pw) - width):
        b = list(range(lo, lo + width))
        per = N / float(np.mean(b))
        if abs(per - 3.0) < 0.6 or abs(per - 1.5) < 0.3 or abs(per - 6.0) < 0.6:
            continue
        if abs(per - T_P10) < T_P10 * 0.05:          # off the ≈10.63 band itself (M-ctrl analogue)
            continue
        e = float(pw[b].sum())
        if 0.75 * target <= e <= 1.25 * target:
            cand.append((abs(e - target), b))
    if not cand:
        raise SystemExit("no energy-matched reciprocal band found off {3, 10.6} — investigate the spectrum")
    cand.sort()
    return cand[0][1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="sequences for the attention profiles")
    ap.add_argument("--seqlen", type=int, default=508)
    ap.add_argument("--nperm", type=int, default=400)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fasta", default="/Users/cberg/Research/data/genomes/hg38.fa")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    dev = pick_device(None)

    ad = load_adapter(str(REPO / M1024), dev, eager=True)
    m = ad.model
    tok = AutoTokenizer.from_pretrained(str(REPO / M1024))
    emb = m.bert.embeddings.position_embeddings
    P0t = emb.weight.data.clone()
    P0 = P0t.detach().float().cpu().numpy()
    h0 = sha(P0t)

    recip = energy_matched_reciprocal(P0, BAND_P10, width=len(BAND_P10))
    conds = {"carrier": BAND_P10, "reciprocal": recip, "low_leak": LOWLEAK}
    print(f"1024nt ≈10.63 carrier bins {BAND_P10} (T={T_P10:.4f})  reciprocal {recip}  low_leak {LOWLEAK}")

    # sequences: real human genome, same domain/estimator as postsoftmax.measure()
    from pyfaidx import Fasta
    rng = np.random.default_rng(7)
    fa = Fasta(args.fasta, as_raw=True)
    s = np.frombuffer(fa["chr8"][40_000_000:40_400_000].upper().encode(), dtype=np.uint8)
    ok = np.isin(s, [65, 67, 71, 84])
    L = args.seqlen
    n = 8 if args.smoke else args.n
    st = [i for i in rng.integers(0, len(s) - L - 12, 4 * n) if ok[i:i + L].all()][:n]
    B2I = {65: 6, 67: 7, 71: 8, 84: 9}
    body = [[B2I[int(b)] for b in s[i:i + L]] for i in st]
    ids = torch.tensor([[tok.cls_token_id] + b + [tok.sep_token_id] for b in body], device=dev)

    prof = {}
    prof["intact"], rowmax = profiles(m, ids)
    aud = {}
    for cn, bins in conds.items():
        a, P1 = deletion_audit(P0, bins, {k: v for k, v in conds.items() if k != cn})
        aud[cn] = a
        emb.weight.data = torch.tensor(P1, dtype=P0t.dtype, device=P0t.device)
        prof[cn], _ = profiles(m, ids)
    emb.weight.data = P0t.clone()
    assert sha(emb.weight.data) == h0, "CHECKPOINT MUTATED"

    # memorials (mechanics; each can fail)
    mem = {
        "M-read_row_sum": bool(rowmax < FROZEN["attn_row_sum_tol"]),
        "M-del_carrier_audit": bool(aud["carrier"]["PASS"]),
        "M-del_target_retained": float(aud["carrier"]["target_retained"]),
        "M-ctrl_reciprocal_off_band": bool(abs(1026 / float(np.mean(recip)) - T_P10) >= T_P10 * 0.05),
        "M-checkpoint_unmutated": bool(sha(emb.weight.data) == h0),
    }

    rows = []
    for h, pr in prof["intact"].items():
        # m_of uses offsets ±[6,60] around centre index 60 (profiles returns ±60), at period T
        from run_postsoftmax import BAND as OFFBAND
        e = pr[60 + OFFBAND]
        if e.sum() / max(pr.sum(), 1e-30) < FROZEN["band_mass_floor"]:
            continue
        c, r = m_of(e, T_P10)
        b = e / r
        nul = np.array([abs(m_of(b * rng.permutation(r), T_P10)[0]) for _ in range(args.nperm)])
        pv = float((nul >= abs(c)).mean())
        kept = {cn: abs(m_of(prof[cn][h][60 + OFFBAND], T_P10)[0]) / max(abs(c), 1e-12)
                for cn in list(conds) + ["intact"]}
        rows.append(dict(head=h, m_T=float(abs(c)), phase_deg=float(math.degrees(np.angle(c))), p=pv,
                         **{f"kept_{k}": float(v) for k, v in kept.items()}))

    # Benjamini-Hochberg across heads, then the FROZEN Level-A criteria (imported thresholds)
    rows.sort(key=lambda x: x["p"])
    nrows = len(rows)
    sig = set()
    for i, x in enumerate(rows, 1):
        if nrows and x["p"] <= LEVELA["fdr"] * i / nrows:
            sig = {y["head"] for y in rows[:i]}
    for x in rows:
        x["significant"] = x["head"] in sig
        x["LEVEL_A"] = bool(x["significant"]
                            and x["kept_carrier"] < LEVELA["collapse_max"]
                            and x["kept_reciprocal"] > LEVELA["survive_min"]
                            and x["kept_low_leak"] > LEVELA["survive_min"])
    R_1024 = [x["head"] for x in rows if x["LEVEL_A"]]

    result = {
        "carrier_bins": BAND_P10, "T": T_P10, "reciprocal_bins": recip, "low_leak_bins": LOWLEAK,
        "checkpoint": "SpliceBERT.1024nt sha 2ad91428..0149d", "smoke": args.smoke,
        "n_seq": len(body), "n_heads_scanned": nrows,
        "memorials": mem, "deletion_audit": aud,
        "R_1024": R_1024, "heads": rows,
        "fork": ("OV_ARM_LIVE" if R_1024 else "OV_ARM_ENDPOINT_B_behavioural_null_only"),
        "note": "L0-reader restriction (prereg must-fix #7) is applied downstream in run_carrier_ov_av.py, "
                "not here; this establishes the full Level-A set.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    ExperimentManifest.create(
        "b1:carrier_preflight_r", sys.argv, repo=REPO,
        status=result["fork"], config={"carrier_bins": BAND_P10, "T": T_P10, "reciprocal": recip},
        seeds={"PYTHONHASHSEED": 0, "profile_seed": 7},
        notes=["Re-points run_postsoftmax.py's validated Level-A instrument to SpliceBERT.1024nt / ≈10.63. "
               "Empty R_1024 => B1 OV arm Endpoint B (behavioural null alone) — a pre-committed close."],
    ).write(Path(args.out).with_suffix(".manifest.json"))

    print("=" * 96)
    print(f"memorials: " + "  ".join(f"{k}={v}" for k, v in mem.items()))
    print(f"heads scanned (mass floor): {nrows}   Level-A: {len(R_1024)}   {R_1024 if R_1024 else '(none)'}")
    print(f"FORK: {result['fork']}")
    print("=" * 96)
    print(f"saved -> {args.out}")
    if not R_1024:
        print("R_1024 = ∅  ->  OV arm is non-identifiable (Endpoint B).  Run only the behavioural arm.")


if __name__ == "__main__":
    main()
