#!/usr/bin/env python
"""IC1 pilot v4 — CALIBRATE + DEMONSTRATE the CORRECTED distal-OOD gate (outcome-blind).

The v6 gate (per-intron/per-cell exclusion vs a p95 sequence-level band, flat 5% cutoff) is
miscalibrated: it compares a compound exclusion rate to the 5% tail that defines the band, so it fails
even in-distribution perturbations (confirmatory excl_frac 7.45%). Corrected gate (Cam):
  - INCREMENTAL, WITHIN-STATE:  ΔNLL_s = NLL(I',A_s) - NLL(I,A_s), s in {+,-}  (acceptor corruption cancels).
  - NONINFERIORITY to the NATURAL null: p_shuffled_in_band >= p_natural_in_band - δ_OOD, per state,
    with band_s = p95 of NATURAL-to-NATURAL ΔNLL_s and δ_OOD fixed from natural-resampling variability.
Demonstrate the corrected gate: passes natural(out-of-sample)/identity, passes the conservative pilot,
FAILS a planted genuinely-OOD perturbation; same cell aggregation for natural & shuffled; gene-clustered
uncertainty (per-gene in-band rate first). Disjoint TRAIN split.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts" / "splicebert"))

from gfmcd.provenance import ExperimentManifest, sha256_file                       # noqa: E402
from gfmcd.splicing.circuit import (                                              # noqa: E402
    DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, load_adapter, pick_device, eligible, _dataset)
from gfmcd.splicing.coupling import substitute                                     # noqa: E402
from run_ic1_pilot import anchored_shuffle, ood_nll, window, hamming               # noqa: E402
from run_ic1_pilot3 import conservative_scramble                                   # noqa: E402
import hashlib                                                                     # noqa: E402

ACC_UP = 70; N_SEED = 16; K_NAT = 3; MIN_DISTAL = 30


def shash(g):
    return int(hashlib.sha256(str(g).encode()).hexdigest()[:8], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pilot", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gtf", default=DEFAULT_GTF); ap.add_argument("--fasta", default=DEFAULT_FASTA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="results/splicebert_coupling/development/ic1_pilot4_ood.json")
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = pick_device(args.device); ad = load_adapter(args.model, dev); task = CouplingTask(ad, batch=args.batch)

    train = eligible(_dataset(args.gtf, args.fasta), "train")
    seen, dd = set(), []
    for x in train:
        if x.gene_id not in seen:
            seen.add(x.gene_id); dd.append(x)

    def elig(x):
        lo, hi = window(x, ACC_UP)
        return 0 <= lo and hi < len(x.sequence) and (hi - lo + 1) >= MIN_DISTAL

    pool = [x for x in dd if elig(x)]
    # dose ∧ conservative-constructible; collect n_pilot
    pilot = []
    for x in pool:
        lo, hi = window(x, ACC_UP); W = x.sequence[lo:hi + 1]
        pre = [anchored_shuffle(W, np.random.default_rng(shash(x.gene_id) * 7 + s)) for s in range(N_SEED)]
        ham = [hamming(W, s) for s in pre]
        if not (np.median(ham) >= 8 and np.median(ham) / len(W) >= 0.25 and len(set(pre)) == N_SEED):
            continue
        cons = [conservative_scramble(W, np.random.default_rng(shash(x.gene_id) * 17 + s + 1), 400) for s in range(N_SEED)]
        if any(c is None for c in cons):
            continue
        pilot.append((x, lo, hi, W, cons))
        if len(pilot) >= args.n_pilot:
            break
    # a length-indexed pool of natural W (for natural-to-natural swaps), from OTHER genes
    natpool = [(x.sequence[window(x, ACC_UP)[0]:window(x, ACC_UP)[1] + 1], x.gene_id) for x in pool]

    # build NLL forwards for every perturbation × acceptor-state
    seqs, wins, tags = [], [], []
    def add(x, lo, hi, Wp, kind, sidx):
        a = x.next_acceptor_pos
        base = x.sequence[:lo] + Wp + x.sequence[hi + 1:]
        for cell, sq in (("+", base), ("-", substitute(base, a - 2, "CC"))):
            seqs.append(sq); wins.append((lo, hi)); tags.append((x.gene_id, kind, sidx, cell))
    rng = np.random.default_rng(0)
    for gi, (x, lo, hi, W, cons) in enumerate(pilot):
        L = len(W)
        add(x, lo, hi, W, "native", -1)                       # I (identity reference)
        # natural-to-natural swaps (calibration + out-of-sample): K_NAT+K_NAT draws
        cands = [w for (w, g) in natpool if g != x.gene_id and len(w) >= L]
        picks = [cands[k] for k in rng.integers(0, len(cands), 2 * K_NAT)]
        for k, w in enumerate(picks):
            add(x, lo, hi, w[:L], "nat", k)
        for s, c in enumerate(cons):                          # conservative shuffles
            add(x, lo, hi, c, "cons", s)
        add(x, lo, hi, ("GC" * (L // 2 + 1))[:L], "ood_gc", 0)    # planted OOD: alternating GC
        add(x, lo, hi, W[::-1], "ood_rev", 0)                     # planted OOD: reversed native window

    nll = ood_nll(ad, task, seqs, wins, args.batch)
    D = {}
    for (g, kind, sidx, cell), v in zip(tags, nll):
        D.setdefault(g, {}).setdefault(cell, {}).setdefault(kind, {})[sidx] = float(v)

    # incremental within-state ΔNLL_s and per-gene in-band rates
    def dnll(g, cell, kind, sidx):
        return D[g][cell][kind][sidx] - D[g][cell]["native"][-1]

    # band_s from natural CALIBRATION draws (first K_NAT), pooled; natural TEST = last K_NAT (out-of-sample)
    bands = {}
    for cell in ("+", "-"):
        cal = [dnll(g, cell, "nat", k) for g in D for k in range(K_NAT)]
        bands[cell] = float(np.percentile(cal, 95))
    def per_gene_inband(kind, ks, cell):
        rates = []
        for g in D:
            hits = [dnll(g, cell, kind, k) <= bands[cell] for k in ks if k in D[g][cell].get(kind, {})]
            if hits:
                rates.append(np.mean(hits))
        return np.array(rates)
    report = {"bands": bands, "n_pilot": len(pilot)}
    for cell in ("+", "-"):
        natr = per_gene_inband("nat", range(K_NAT, 2 * K_NAT), cell)     # out-of-sample natural
        consr = per_gene_inband("cons", range(N_SEED), cell)
        oodgc = per_gene_inband("ood_gc", [0], cell)
        oodrev = per_gene_inband("ood_rev", [0], cell)
        identr = np.array([1.0 if dnll(g, cell, "native", -1) <= bands[cell] else 0.0 for g in D])  # ΔNLL=0
        report[cell] = {"natural_inband_mean": float(np.mean(natr)), "natural_inband_sd": float(np.std(natr)),
                        "conservative_inband_mean": float(np.mean(consr)), "conservative_inband_sd": float(np.std(consr)),
                        "identity_inband_mean": float(np.mean(identr)),
                        "planted_ood_gc_inband_mean": float(np.mean(oodgc)),
                        "planted_ood_rev_inband_mean": float(np.mean(oodrev))}
    # δ_OOD fixed from natural-RATE resampling variability = gene-clustered SEM (NOT the per-gene SD, and
    # NOT tuned to any observed shuffle rate).
    ng = report["n_pilot"]
    delta = float(2 * np.mean([report[c]["natural_inband_sd"] / np.sqrt(max(ng, 1)) for c in ("+", "-")]))
    delta = max(0.03, round(delta, 3))
    report["delta_OOD"] = delta
    report["thresholds"] = {c: report[c]["natural_inband_mean"] - delta for c in ("+", "-")}
    demo = {"conservative_passes": all(report[c]["conservative_inband_mean"] >= report["thresholds"][c] for c in "+-"),
            "natural_passes": all(report[c]["natural_inband_mean"] >= report["thresholds"][c] for c in "+-"),
            "identity_passes": all(report[c]["identity_inband_mean"] >= report["thresholds"][c] for c in "+-"),
            "planted_ood_gc_FAILS": all(report[c]["planted_ood_gc_inband_mean"] < report["thresholds"][c] for c in "+-"),
            "planted_ood_rev_FAILS": all(report[c]["planted_ood_rev_inband_mean"] < report["thresholds"][c] for c in "+-")}
    demo["ALL_DEMONSTRATIONS_PASS"] = bool(demo["conservative_passes"] and demo["natural_passes"]
                                           and demo["identity_passes"]
                                           and (demo["planted_ood_gc_FAILS"] or demo["planted_ood_rev_FAILS"]))
    report["demonstration"] = demo
    out = REPO / args.out; out.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, indent=1, default=float))
    ckpt = Path(args.model); ckpt = ckpt if ckpt.is_absolute() else REPO / ckpt
    man = ExperimentManifest.create("splicebert_coupling:ic1_pilot4_ood", sys.argv, status="PILOT",
        config={"acc_up": ACC_UP, "n_seed": N_SEED, "k_nat": K_NAT}, seeds={"base": 0},
        notes=["IC1 corrected-OOD-gate calibration + demonstration (outcome-blind)."])
    man.add_input(str(ckpt / "pytorch_model.bin")); man.write(out.parent / "ic1_pilot4_ood.manifest.json")
    print(f"saved -> {out}   ALL_DEMOS_PASS={demo['ALL_DEMONSTRATIONS_PASS']}")


if __name__ == "__main__":
    main()
