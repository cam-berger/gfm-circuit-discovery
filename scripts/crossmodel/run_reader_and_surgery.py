#!/usr/bin/env python
"""READER SCAN (A-column) + WOULD-BE-BAND SURGERY for RNA-FM / RNA-BERT (Phase 2, step 4).

CONTEXT.  The P-spectrum assay (run_carrier_pspectrum.py) returned a POWERED NULL: neither
the period-3 nor the =10.63 carrier exists in either model's position table P.  The task's
step 4 says "IF a carrier is present in P: run the reader scan and the causal surgery."  It
is not.  So the causal P-band surgery has no carrier band to ablate.

This script does the two things that ARE still informative and honest:

  1. ALL-HEADS POST-SOFTMAX READER SCAN.  Measures the bounded coherence m_T (in [0,1]) at
     the two carrier periods, for every head, with a per-head permutation null.  If P has no
     carrier, no head should read one -- this confirms the absence propagates P -> A (it does
     not merely fail to appear in P while hiding in attention).  A head calling significant
     here would be a genuine surprise worth chasing.

  2. WOULD-BE-BAND DELETION as a bounded control, NOT a carrier ablation.  Delete the rfft
     bins where the carriers WOULD sit (period-3 and =10.63) and measure the effect on the
     model's native masked-base prediction, against MATCHED control bands (random same-count
     bins) and a non-outcome health channel (mean logit magnitude).  Because those bins hold
     no coherent carrier, the expectation is a small effect indistinguishable from control
     bands -- i.e. there is nothing there to be causal.  Reported as an effect size with a
     control distribution, never as "the carrier does nothing" (there is no carrier).

CLAIM SCOPE.  The donor-acceptor coupling behavioural arm does NOT transfer (ncRNA has no
introns / splice sites).  The ceiling here is: structural absence in P (done), absence in
post-softmax attention (this scan), and a generic masked-LM-prediction control surgery.
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
SEQLEN = 200          # long enough for span=60 with margin, short enough for CPU
SPAN = 60
NPERM = 200


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_eager(hf_id):
    from multimolecule import RnaFmForMaskedLM, RnaBertForMaskedLM  # noqa: F401
    from gfmcd.adapters.splicebert import SpliceBertAdapter
    return SpliceBertAdapter.from_pretrained(
        hf_id, tokenizer_mode="raw", device="cpu", model_class="masked_lm",
        attn_implementation="eager")


def bins_for_period(N, period, halfwidth=3):
    b = int(round(N / period))
    return [x for x in range(b - halfwidth, b + halfwidth + 1) if 0 < x < N // 2 + 1]


def masked_prediction_health(ad, seqs, P_override=None):
    """Native masked-LM: mask ~15% of positions, score mean top-1 accuracy of the true base
    and mean max-logit magnitude (a non-outcome health channel).  Optionally with a modified
    position table (P_override) installed for the duration.
    """
    pmod = cs.get_position_table  # noqa
    mod = ad.model
    # locate the position_embeddings module object
    pe = None
    for name, m in mod.named_modules():
        if name.endswith("position_embeddings") and hasattr(m, "weight"):
            pe = m
    assert pe is not None
    orig = pe.weight.data.clone()
    try:
        if P_override is not None:
            pe.weight.data = torch.tensor(P_override, dtype=orig.dtype, device=orig.device)
        rng = np.random.default_rng(123)
        b = ad.tokenize(seqs)
        ids = b.input_ids.clone()
        mask_id = ad.tokenizer.mask_token_id
        # choose maskable positions (non-special): tokens after prefix, before eos
        acc, mags = [], []
        for row in range(ids.shape[0]):
            n = b.nucleotide_lengths[row]
            pos = np.arange(ad.prefix_tokens, ad.prefix_tokens + n)
            k = max(1, int(0.15 * len(pos)))
            chosen = rng.choice(pos, size=k, replace=False)
            true = ids[row, chosen].clone()
            mids = ids[row:row + 1].clone()
            mids[0, chosen] = mask_id
            with torch.no_grad():
                out = mod(input_ids=mids, attention_mask=b.attention_mask[row:row + 1],
                          return_dict=True)
            logits = out.logits[0, chosen]           # (k, vocab)
            pred = logits.argmax(-1)
            acc.append(float((pred == true).float().mean()))
            mags.append(float(logits.max(-1).values.mean()))
        return {"masked_top1_acc": float(np.mean(acc)),
                "mean_max_logit": float(np.mean(mags))}
    finally:
        pe.weight.data = orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/origin-crossmodel/canonical"))
    ap.add_argument("--models", default="rnafm,rnabert")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    git_rev = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()
    band, w = rs.band_and_window(6, SPAN)
    rng = np.random.default_rng(SEED)

    results = {}
    for tag in args.models.split(","):
        hf_id = MODELS[tag]
        ad = load_eager(hf_id)
        N = ad.model.config.max_position_embeddings
        Luse = min(SEQLEN, ad.max_nucleotides)
        seqs = ["".join(rng.choice(list("ACGT"), Luse)) for _ in range(NSEQ)]
        b = ad.tokenize(seqs)
        prof, rowmax = rs.profiles(ad.model, b.input_ids, span=SPAN,
                                   attention_mask=b.attention_mask)
        print(f"\n===== {tag}  {len(prof)} heads, row-sum err {rowmax:.1e} =====")

        # --- reader scan: m_T + per-head permutation p at each carrier period ---
        scan = {}
        for cname, T in [("period3", PERIOD_3), ("carrier_10p63", PERIOD_1063)]:
            rows = []
            for hlabel, p in prof.items():
                pf = p[SPAN + band]
                bmass = rs.band_mass_fraction(p, band, SPAN)
                mval, pval, _ = rs.head_permutation_fpr(pf, T, band, w, rng, nperm=NPERM)
                rows.append({"head": hlabel, "m_T": float(mval), "p_perm": float(pval),
                             "band_mass": float(bmass)})
            pvals = np.array([r["p_perm"] for r in rows])
            # BH-FDR at 0.05
            order = np.argsort(pvals)
            m = len(pvals)
            thr = 0.05 * (np.arange(1, m + 1)) / m
            passed = pvals[order] <= thr
            n_sig = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
            sig_heads = [rows[i]["head"] for i in order[:n_sig]]
            top = sorted(rows, key=lambda r: -r["m_T"])[:5]
            scan[cname] = {
                "period": T,
                "n_heads": m,
                "n_sig_fdr05": n_sig,
                "sig_heads": sig_heads,
                "expected_false_positives_at_0.05": round(0.05 * m, 1),
                "top5_by_m_T": top,
                "all_heads": rows,
            }
            print(f"  {cname:14s} T={T:7.4f}: {n_sig} heads sig @FDR.05 "
                  f"(chance ~{0.05*m:.1f}); top m_T={top[0]['m_T']:.3f} ({top[0]['head']})")

        # --- would-be-band deletion control surgery ---
        P0 = cs.get_position_table(ad.model)
        base_health = masked_prediction_health(ad, seqs, None)
        surg = {"baseline": base_health, "deletions": {}}
        # target would-be bands
        target_bands = {
            "period3_wouldbe": bins_for_period(N, PERIOD_3),
            "carrier_10p63_wouldbe": bins_for_period(N, PERIOD_1063),
        }
        for name, tb in target_bands.items():
            Pd = rs.strip_fft(P0, tb)
            h = masked_prediction_health(ad, seqs, Pd)
            surg["deletions"][name] = {
                "bins": tb,
                "health": h,
                "d_acc": h["masked_top1_acc"] - base_health["masked_top1_acc"],
            }
        # matched control bands: random same-count high-freq bins, 20 draws -> null of d_acc
        ctrl_dacc = []
        nbins = len(target_bands["period3_wouldbe"])
        hi = N // 2
        for _ in range(20):
            cb = sorted(rng.choice(np.arange(20, hi), size=nbins, replace=False).tolist())
            Pd = rs.strip_fft(P0, cb)
            h = masked_prediction_health(ad, seqs, Pd)
            ctrl_dacc.append(h["masked_top1_acc"] - base_health["masked_top1_acc"])
        surg["matched_control_bands"] = {
            "n": 20, "d_acc_mean": float(np.mean(ctrl_dacc)),
            "d_acc_std": float(np.std(ctrl_dacc)),
            "d_acc_min": float(np.min(ctrl_dacc)), "d_acc_max": float(np.max(ctrl_dacc)),
        }
        for name in target_bands:
            dacc = surg["deletions"][name]["d_acc"]
            z = (dacc - np.mean(ctrl_dacc)) / (np.std(ctrl_dacc) + 1e-12)
            surg["deletions"][name]["z_vs_control_bands"] = float(z)
            print(f"  surgery {name:24s}: d_acc={dacc:+.4f}  "
                  f"(control bands {np.mean(ctrl_dacc):+.4f}+-{np.std(ctrl_dacc):.4f}, z={z:+.2f})")

        results[tag] = {
            "model": {"hf_id": hf_id, "P_shape": list(P0.shape),
                      "P_weight_sha256": sha_bytes(P0.astype(np.float64).tobytes()),
                      "num_heads_total": len(prof)},
            "reader_scan": scan,
            "surgery_control": surg,
        }
        del ad

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "assay": "crossmodel_reader_and_surgery",
        "produced_utc": stamp,
        "note": "Carrier absent in P (powered null); this is the A-column confirmation + a "
                "would-be-band deletion CONTROL, not a carrier ablation.",
        "period3": PERIOD_3, "carrier_10p63_period": PERIOD_1063,
        "seed": SEED, "nseq": NSEQ, "seqlen": SEQLEN, "span": SPAN, "nperm": NPERM,
        "results": results,
    }
    out = outdir / "reader_and_surgery.json"
    out.write_text(json.dumps(payload, indent=1))
    manifest = {
        "artifact": "reader_and_surgery.json",
        "assay": "crossmodel_reader_and_surgery",
        "produced_utc": stamp, "git_rev": git_rev,
        "environment": "gfmcd-mm (conda): python3.11 torch transformers multimolecule0.2.0",
        "models": {t: {"hf_id": results[t]["model"]["hf_id"],
                       "P_weight_sha256": results[t]["model"]["P_weight_sha256"],
                       "P_shape": results[t]["model"]["P_shape"]} for t in results},
        "params": {"nseq": NSEQ, "seqlen": SEQLEN, "span": SPAN, "nperm": NPERM,
                   "period3": PERIOD_3, "carrier_10p63_period": PERIOD_1063},
        "seed": SEED,
        "command": "python scripts/crossmodel/run_reader_and_surgery.py",
        "attention": "eager (output_attentions real); post-softmax diagonal-band profiles",
    }
    (outdir / "reader_and_surgery.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
