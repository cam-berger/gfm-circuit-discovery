#!/usr/bin/env python
"""P-SPECTRUM CARRIER DETECTION for RNA-FM and RNA-BERT (Phase 2, step 3).

Ports the SpliceBERT positional-carrier P-spectrum inventory to two ncRNA-trained single-nt
transformer MLMs.  The two carriers under test:

    period-3  carrier -- codon-scale.  In SpliceBERT's 1026-row table it is bins 339-345
                         (bin 342 = 1026/3) at ~289x the median bin power.
    =10.63    carrier -- a distinct high-frequency mode; SpliceBERT bin 96 of 1026
                         (period 1026/96 = 10.6875 nt).

CODON-ORIGIN TEST.  Both RNA-FM and RNA-BERT are trained on ncRNA, which is largely
ORF-free.  If SpliceBERT's period-3 carrier is a CODON artifact (built because the training
corpus is full of reading frames), it should be WEAK or ABSENT in these two models.  This is
the complementary origin datapoint to Phase 1's human-only sibling contrast.  BOTH presence
(convergence) and absence (idiosyncrasy) are reportable.

TWO ESTIMATORS, VALIDATED FIRST.
  * exact-bin FFT band ratio -- valid when the period lands on an integer rfft bin.  RNA-FM:
    1026/3 = 342 is EXACT, so period-3 is read this way (like SpliceBERT's final table).
  * leakage-aware joint least-squares -- valid at any real period.  RNA-BERT: 440/3 = 146.67
    is between bins, so period-3 (and =10.63 in both models) is read this way.  Validated to
    machine-precision amplitude recovery and 0.01-nt period localization (see tests).

INJECTION-CALIBRATED NOISE FLOOR + FPR.  A band amplitude means nothing without the
distribution of the same statistic where no carrier exists.  For each model and target
period we:
  (a) build a phase-scrambled null P (each dim's rfft phases randomized -> same per-dim power
      spectrum, no coherent positional tone) and measure the depth-0 false-positive rate of
      the rank test over many scrambles;
  (b) inject a KNOWN rank-structured tone at the target period at depths 0..X into scrambled
      P and report the detection floor (smallest depth reliably called) + amplitude recovery.

Everything here is a claim about P only.  A concentration in P is NOT attention and NOT
causal -- those are separate assays (reader scan; surgery).
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.crossmodel import carrier_spectrum as cs  # noqa: E402

MODELS = {
    "rnafm": "multimolecule/rnafm",
    "rnabert": "multimolecule/rnabert",
}

# The two named carriers.  period-3 is exact.  =10.63 is defined by SpliceBERT bin 96 of 1026.
PERIOD_3 = 3.0
PERIOD_1063 = 1026.0 / 96.0     # = 10.6875 nt  -- the =10.63 carrier's period
NUCLEOSOMAL = 10.298            # DISTINCT DNA signal; kept separate, never conflated

ENVDEG = 6
N_SCRAMBLE = 200
INJECT_DEPTHS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
SEED = 0


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_P(hf_id: str) -> tuple[np.ndarray, dict]:
    from multimolecule import RnaFmForMaskedLM, RnaBertForMaskedLM  # noqa: F401  (register)
    from gfmcd.adapters.splicebert import SpliceBertAdapter

    ad = SpliceBertAdapter.from_pretrained(
        hf_id, tokenizer_mode="raw", device="cpu", model_class="masked_lm"
    )
    P = cs.get_position_table(ad.model)
    md = ad.metadata()
    info = {
        "hf_id": hf_id,
        "P_shape": list(P.shape),
        "P_weight_sha256": sha_bytes(P.astype(np.float64).tobytes()),
        "config_sha256": md.config_sha256,
        "num_layers": ad.num_layers,
        "num_heads": ad.num_heads,
        "hidden_size": ad.hidden_size,
    }
    return P, info


def phase_scramble(P: np.ndarray, rng) -> np.ndarray:
    """Randomize the rfft PHASE of each dim independently, keep per-dim magnitude spectrum.

    Destroys any positional COHERENCE across dims (a carrier is a phase-locked tone shared
    by many dims) while preserving each dim's marginal power spectrum -- the correct null for
    'is there a coherent positional tone at period T'.
    """
    N, D = P.shape
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    mag = np.abs(F)
    nb = F.shape[0]
    out = np.empty_like(F)
    for d in range(D):
        ph = rng.uniform(0, 2 * np.pi, nb)
        ph[0] = 0.0
        if N % 2 == 0:
            ph[-1] = 0.0  # Nyquist real
        out[:, d] = mag[:, d] * np.exp(1j * ph)
    return np.fft.irfft(out, n=N, axis=0) + mu


def inject_tone(P: np.ndarray, T: float, depth: float, rng) -> np.ndarray:
    """Add a rank-structured, phase-locked period-T tone of relative depth ``depth``.

    Amplitude per dim is depth * (that dim's RMS), phases shared structure but with a small
    per-dim jitter so it is a realistic (not perfectly rank-1) carrier.
    """
    if depth <= 0:
        return P
    N, D = P.shape
    k = np.arange(N)
    rms = P.std(0)
    base_phase = rng.uniform(0, 2 * np.pi)
    jitter = rng.uniform(-0.3, 0.3, D)
    tone = depth * rms[None, :] * np.cos(2 * np.pi * k[:, None] / T + base_phase + jitter[None, :])
    return P + tone


def calibrate(P: np.ndarray, T: float, rng) -> dict:
    """Injection calibration on THIS model's P: depth-0 FPR + detection floor + recovery."""
    # depth-0 FPR: fraction of phase-scrambled nulls whose target band power exceeds the
    # 95th percentile of their OWN control distribution (i.e. the rank test fires by chance).
    fp = 0
    scr_bandpowers = []
    for i in range(N_SCRAMBLE):
        Pn = phase_scramble(P, rng)
        t = cs.carrier_test(Pn, T, envdeg=ENVDEG)
        scr_bandpowers.append(t["band_power"])
        if t["p_rank"] <= 0.05:
            fp += 1
    fpr = fp / N_SCRAMBLE
    scr_bandpowers = np.array(scr_bandpowers)

    # detection: inject at each depth into a FRESH scramble, does the rank test fire?
    det = {}
    for depth in INJECT_DEPTHS:
        hits, recov = 0, []
        for j in range(30):
            Pn = phase_scramble(P, rng)
            Pi = inject_tone(Pn, T, depth, rng)
            t = cs.carrier_test(Pi, T, envdeg=ENVDEG)
            if t["p_rank"] <= 0.05:
                hits += 1
            recov.append(t["band_power"])
        det[f"{depth}"] = {
            "detect_rate": hits / 30,
            "median_band_power": float(np.median(recov)),
        }
    # smallest depth with >=0.9 detection
    floor = None
    for depth in INJECT_DEPTHS:
        if depth > 0 and det[f"{depth}"]["detect_rate"] >= 0.9:
            floor = depth
            break
    return {
        "depth0_fpr": fpr,
        "scramble_band_power_median": float(np.median(scr_bandpowers)),
        "scramble_band_power_p95": float(np.percentile(scr_bandpowers, 95)),
        "detection_curve": det,
        "detection_floor_depth": floor,
        "n_scramble": N_SCRAMBLE,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/origin-crossmodel/canonical"))
    ap.add_argument("--models", default="rnafm,rnabert")
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    git_rev = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"]).decode().strip()

    rng = np.random.default_rng(SEED)
    results = {}
    for tag in args.models.split(","):
        hf_id = MODELS[tag]
        P, info = load_P(hf_id)
        N = P.shape[0]
        print(f"\n===== {tag}  P={P.shape} =====")

        rec = {"model": info, "carriers": {}}
        for cname, T in [("period3", PERIOD_3), ("carrier_10p63", PERIOD_1063)]:
            bin_f = N / T
            exact = abs(bin_f - round(bin_f)) < 1e-6
            la = cs.carrier_test(P, T, envdeg=ENVDEG)
            entry = {
                "period": T,
                "bin_float": bin_f,
                "bin_is_exact": bool(exact),
                "leakage_aware": la,
            }
            if exact:
                entry["exact_bin_fft"] = cs.exact_bin_band_ratio(P, T, halfwidth=3)
            entry["calibration"] = calibrate(P, T, rng)
            rec["carriers"][cname] = entry
            fpr = entry["calibration"]["depth0_fpr"]
            floor = entry["calibration"]["detection_floor_depth"]
            print(f"  {cname:14s} T={T:8.4f} bin={bin_f:8.3f} exact={exact}  "
                  f"Z={la['Z_over_median_control']:7.2f} p={la['p_rank']:.4f}  "
                  f"FPR0={fpr:.3f} floor={floor}")
        results[tag] = rec

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "assay": "crossmodel_carrier_pspectrum",
        "produced_utc": stamp,
        "period3": PERIOD_3,
        "carrier_10p63_period": PERIOD_1063,
        "nucleosomal_distinct_nt": NUCLEOSOMAL,
        "envdeg": ENVDEG,
        "seed": SEED,
        "results": results,
    }
    out = outdir / "carrier_pspectrum.json"
    out.write_text(json.dumps(payload, indent=1))
    manifest = {
        "artifact": "carrier_pspectrum.json",
        "assay": "crossmodel_carrier_pspectrum",
        "produced_utc": stamp,
        "git_rev": git_rev,
        "environment": "gfmcd-mm (conda): python3.11 torch transformers multimolecule0.2.0",
        "models": {t: {"hf_id": results[t]["model"]["hf_id"],
                       "config_sha256": results[t]["model"]["config_sha256"],
                       "P_weight_sha256": results[t]["model"]["P_weight_sha256"],
                       "P_shape": results[t]["model"]["P_shape"]} for t in results},
        "params": {"envdeg": ENVDEG, "n_scramble": N_SCRAMBLE, "inject_depths": INJECT_DEPTHS,
                   "period3": PERIOD_3, "carrier_10p63_period": PERIOD_1063},
        "seed": SEED,
        "command": "python scripts/crossmodel/run_carrier_pspectrum.py",
        "estimator": "leakage-aware joint LSQ (envdeg=6) + exact-bin FFT where integer; null = phase-scramble + control-period rank test",
    }
    (outdir / "carrier_pspectrum.manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
