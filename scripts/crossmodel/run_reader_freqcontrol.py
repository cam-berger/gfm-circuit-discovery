#!/usr/bin/env python
"""FREQUENCY-CONTROL DIAGNOSTIC for the cross-model reader scan (Phase 2, step 4 addendum).

WHY THIS IS MANDATORY.  run_reader_and_surgery.py reported 16 RNA-FM heads "significant" at
the =10.63 period under a per-head permutation null -- even though P has NO =10.63 carrier
(powered null, carrier_pspectrum.json).  This is the exact artifact the SpliceBERT
post-softmax methods memorialize: a per-head permutation null over a head's OWN residual
still over-calls a SMOOTH, NON-PERIODIC attention decay as "coherent", because permuting the
residual destroys the slow trend that a low-degree baseline could not fully absorb.

THE CONTROL.  The claim "16 heads read 10.63" is only meaningful if the significant-head
count at 10.63 is ELEVATED above the count at arbitrary control periods.  Here we sweep the
bounded statistic + BH-FDR across a dense period grid.  If 10.6875 is not a peak, the
apparent readers are an instrument artifact and the correct claim is: no head reads the
carrier (consistent with its absence in P).

This also replaces the degenerate top-1 masked-accuracy surgery channel (floored at 0 on
random OOD sequence for these ncRNA models) with a CONTINUOUS logit-shift channel: mean L2
change of the masked-position logit vector when a band is deleted from P.  Because P has no
carrier at the would-be bands, the expectation is a shift indistinguishable from matched
control bands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.crossmodel import carrier_spectrum as cs  # noqa: E402
from gfmcd.crossmodel import reader_scan as rs  # noqa: E402

MODELS = {"rnafm": "multimolecule/rnafm", "rnabert": "multimolecule/rnabert"}
PERIOD_3 = 3.0
PERIOD_1063 = 1026.0 / 96.0
SEED = 0
NSEQ = 12
SEQLEN = 200
SPAN = 60
NPERM = 200


def sha_bytes(b): return hashlib.sha256(b).hexdigest()


def load_eager(hf_id):
    from multimolecule import RnaFmForMaskedLM, RnaBertForMaskedLM  # noqa: F401
    from gfmcd.adapters.splicebert import SpliceBertAdapter
    return SpliceBertAdapter.from_pretrained(
        hf_id, tokenizer_mode="raw", device="cpu", model_class="masked_lm",
        attn_implementation="eager")


def n_sig(prof, T, band, w, nperm=NPERM):
    pvals = []
    for h, p in prof.items():
        pf = p[SPAN + band]
        _, pv, _ = rs.head_permutation_fpr(pf, T, band, w,
                                           np.random.default_rng(hash((h, round(T, 4))) % 2**31),
                                           nperm=nperm)
        pvals.append(pv)
    pvals = np.array(pvals)
    m = len(pvals); order = np.argsort(pvals)
    thr = 0.05 * np.arange(1, m + 1) / m
    passed = pvals[order] <= thr
    return (int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0,
            float(np.median(pvals)), float(np.max([abs(rs.m_of(prof[h][SPAN+band], T, band, w)[0]) for h in prof])))


def pe_module(model):
    for name, m in model.named_modules():
        if name.endswith("position_embeddings") and hasattr(m, "weight"):
            return m
    raise RuntimeError("no position_embeddings")


def logit_shift(ad, seqs, bins):
    """Mean L2 change of masked-position logits when the given P-bins are deleted."""
    pe = pe_module(ad.model)
    orig = pe.weight.data.clone()
    P0 = orig.detach().float().cpu().numpy().astype(np.float64)
    b = ad.tokenize(seqs)
    rng = np.random.default_rng(123)
    mask_id = ad.tokenizer.mask_token_id
    # fixed mask set across conditions
    masks = []
    for row in range(b.input_ids.shape[0]):
        n = b.nucleotide_lengths[row]
        pos = np.arange(ad.prefix_tokens, ad.prefix_tokens + n)
        k = max(1, int(0.15 * len(pos)))
        masks.append(rng.choice(pos, size=k, replace=False))

    def logits_for(Pw):
        pe.weight.data = torch.tensor(Pw, dtype=orig.dtype, device=orig.device)
        outs = []
        for row in range(b.input_ids.shape[0]):
            mids = b.input_ids[row:row + 1].clone()
            mids[0, masks[row]] = mask_id
            with torch.no_grad():
                o = ad.model(input_ids=mids, attention_mask=b.attention_mask[row:row + 1],
                             return_dict=True)
            outs.append(o.logits[0, masks[row]].float().cpu().numpy())
        return np.concatenate(outs, 0)
    try:
        L0 = logits_for(P0)
        L1 = logits_for(rs.strip_fft(P0, bins))
        return float(np.sqrt(((L1 - L0) ** 2).sum(1)).mean())
    finally:
        pe.weight.data = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/origin-crossmodel/canonical"))
    ap.add_argument("--models", default="rnafm,rnabert")
    args = ap.parse_args()
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    git_rev = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()
    band, w = rs.band_and_window(6, SPAN)
    rng = np.random.default_rng(SEED)

    # dense control grid over periods 5..24, plus the two named periods
    grid = sorted(set(np.round(np.linspace(5.0, 24.0, 20), 3).tolist()
                      + [PERIOD_3, round(PERIOD_1063, 4)]))
    results = {}
    for tag in args.models.split(","):
        ad = load_eager(MODELS[tag])
        N = ad.model.config.max_position_embeddings
        Luse = min(SEQLEN, ad.max_nucleotides)
        seqs = ["".join(rng.choice(list("ACGT"), Luse)) for _ in range(NSEQ)]
        b = ad.tokenize(seqs)
        prof, rowmax = rs.profiles(ad.model, b.input_ids, span=SPAN,
                                   attention_mask=b.attention_mask)
        print(f"\n===== {tag}  {len(prof)} heads =====")
        sweep = []
        for T in grid:
            ns, medp, maxm = n_sig(prof, T, band, w)
            sweep.append({"period": float(T), "n_sig_fdr05": ns, "median_p": medp, "max_m_T": maxm})
        # is either named period a peak above the control grid?
        counts = np.array([s["n_sig_fdr05"] for s in sweep])
        gridT = np.array([s["period"] for s in sweep])
        ctrl_mask = np.abs(gridT - PERIOD_1063) > 1.0
        ctrl_mask &= np.abs(gridT - PERIOD_3) > 1.0
        ctrl_counts = counts[ctrl_mask]
        def peak_verdict(T):
            i = int(np.argmin(np.abs(gridT - T)))
            c = counts[i]
            return {"period": float(T), "n_sig_here": int(c),
                    "control_grid_mean": float(ctrl_counts.mean()),
                    "control_grid_max": int(ctrl_counts.max()),
                    "is_peak_above_controls": bool(c > ctrl_counts.max())}
        v3 = peak_verdict(PERIOD_3)
        v10 = peak_verdict(PERIOD_1063)
        print(f"  period-3  : n_sig={v3['n_sig_here']} vs control grid "
              f"mean {v3['control_grid_mean']:.1f} max {v3['control_grid_max']} -> "
              f"peak={v3['is_peak_above_controls']}")
        print(f"  =10.63    : n_sig={v10['n_sig_here']} vs control grid "
              f"mean {v10['control_grid_mean']:.1f} max {v10['control_grid_max']} -> "
              f"peak={v10['is_peak_above_controls']}")

        # continuous surgery channel
        P0 = cs.get_position_table(ad.model)
        def binsfor(T, hw=3):
            bb = int(round(N / T)); return [x for x in range(bb-hw, bb+hw+1) if 0 < x < N//2+1]
        surg = {}
        for name, T in [("period3_wouldbe", PERIOD_3), ("carrier_10p63_wouldbe", PERIOD_1063)]:
            surg[name] = {"bins": binsfor(T), "logit_l2_shift": logit_shift(ad, seqs, binsfor(T))}
        ctrl = []
        nb = len(binsfor(PERIOD_3))
        for _ in range(15):
            cb = sorted(rng.choice(np.arange(20, N//2), size=nb, replace=False).tolist())
            ctrl.append(logit_shift(ad, seqs, cb))
        for name in surg:
            z = (surg[name]["logit_l2_shift"] - np.mean(ctrl)) / (np.std(ctrl) + 1e-12)
            surg[name]["z_vs_control_bands"] = float(z)
            print(f"  surgery {name:22s}: logit_L2_shift={surg[name]['logit_l2_shift']:.4f} "
                  f"(ctrl {np.mean(ctrl):.4f}+-{np.std(ctrl):.4f}, z={z:+.2f})")
        results[tag] = {
            "model": {"hf_id": MODELS[tag], "P_shape": list(P0.shape),
                      "P_weight_sha256": sha_bytes(P0.astype(np.float64).tobytes()),
                      "num_heads_total": len(prof)},
            "frequency_sweep": sweep,
            "period3_verdict": v3,
            "carrier_10p63_verdict": v10,
            "surgery_logit_shift": {"bands": surg,
                                    "control_bands": {"n": 15, "mean": float(np.mean(ctrl)),
                                                      "std": float(np.std(ctrl))}},
        }
        del ad

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "assay": "crossmodel_reader_freqcontrol",
        "produced_utc": stamp,
        "note": "Frequency control for the reader scan: the apparent =10.63 readers in RNA-FM "
                "are an envelope/permutation-null artifact unless the sig-head count at 10.6875 "
                "exceeds the control-period grid. Plus a continuous logit-shift surgery channel "
                "(top-1 accuracy is degenerate on random OOD sequence for these ncRNA models).",
        "period3": PERIOD_3, "carrier_10p63_period": PERIOD_1063,
        "seed": SEED, "nseq": NSEQ, "seqlen": SEQLEN, "span": SPAN, "nperm": NPERM,
        "results": results,
    }
    out = outdir / "reader_freqcontrol.json"
    out.write_text(json.dumps(payload, indent=1))
    (outdir / "reader_freqcontrol.manifest.json").write_text(json.dumps({
        "artifact": "reader_freqcontrol.json", "assay": "crossmodel_reader_freqcontrol",
        "produced_utc": stamp, "git_rev": git_rev,
        "environment": "gfmcd-mm (conda): python3.11 torch transformers multimolecule0.2.0",
        "models": {t: {"hf_id": results[t]["model"]["hf_id"],
                       "P_weight_sha256": results[t]["model"]["P_weight_sha256"]} for t in results},
        "params": {"nseq": NSEQ, "seqlen": SEQLEN, "span": SPAN, "nperm": NPERM},
        "seed": SEED, "command": "python scripts/crossmodel/run_reader_freqcontrol.py",
    }, indent=1))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
