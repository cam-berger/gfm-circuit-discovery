#!/usr/bin/env python
"""ORIGIN-SIBLINGS STEPS 2+3 — leakage-aware P-spectrum inventory + corpus contrast.

WHY A LEAKAGE-AWARE ESTIMATOR
-----------------------------
The siblings' position table P has N=512 rows. 512 is NOT divisible by 3, so period-3 is not an
exact Fourier bin (it lands at 512/3 = 170.67 -> energy smears across bins 170/171/172). A naive
single-bin FFT UNDERCOUNTS period-3 in the 510 models and can manufacture a false "absent" verdict.

FIX: measure power by DIRECT PROJECTION onto the cos/sin basis at the exact target frequency
(period_power below), which is exact at any N and needs no integer bin. This is the same estimator
family the repo's run_carrier_provenance.py used; here it is packaged with an injection-calibrated
noise floor, a permutation p-value from the high-frequency competition band, and a sensitivity
curve that quantifies the minimum detectable carrier depth (so an "absent" verdict carries a
ceiling on any hidden carrier).

VALIDATION GATE: the estimator must reproduce the KNOWN final-checkpoint anchors before it is
trusted on the 510s:
  * period-3 ~ 289x median (crude single-bin FFT bin 342 = 288.9x; MECHANISM bins 339-345)
  * ≈10.63 carrier at bin 96, rank <= 7 (run_second_carrier: "546x, rank 7 of 513")

NOMENCLATURE (enforced): "≈10.63 carrier" (period ~10.667 nt). The DNA nucleosomal signal
≈10.298 nt is DISTINCT and already refuted as the carrier's correlate — it is NOT measured here.
period-3 != ≈10.63.

CONTRASTS
  (A) CORPUS (sharp): stage1_510 (72-species) vs human_510 (human-only) — same stage, same N=512,
      same architecture, IDENTICAL leakage geometry; only the training corpus differs.
  (B) STAGE (coarse, confounded): stage1_510 vs final_1024 — stage AND max-length change together.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from transformers import AutoModelForMaskedLM  # noqa: E402

from gfmcd.provenance import ExperimentManifest  # noqa: E402

BASE = REPO / "data/models/splicebert_official_archive/extracted/models"
CKPTS = {
    "final_1024": BASE / "SpliceBERT.1024nt",
    "stage1_510": BASE / "SpliceBERT.510nt",
    "human_510": BASE / "SpliceBERT-human.510nt",
}
CORPUS = {"final_1024": "72-species", "stage1_510": "72-species", "human_510": "human-only"}
OUT = REPO / "results/origin-siblings/canonical/sibling_pspectrum.json"

PERIOD_P3 = 3.0
PERIOD_C10 = 10.6667          # ≈10.63 carrier; NOT the nucleosomal ≈10.298 nt DNA signal
HF_LO, HF_HI = 4.0, 20.0      # high-frequency competition band for the empirical noise floor
GUARD = 3.0                   # exclude +-GUARD/N around each target frequency from the null
DEPTHS = (0.0, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2)
N_PHASE = 8
SEED = 20260723


def load_P(path: Path) -> np.ndarray:
    m = AutoModelForMaskedLM.from_pretrained(str(path))
    P = m.bert.embeddings.position_embeddings.weight.detach().float().numpy()
    del m
    return P


def period_power(Pt: np.ndarray, f: float) -> float:
    """Power at frequency f (period 1/f), summed over columns. Exact at any N (no integer bin)."""
    N = Pt.shape[0]
    i = np.arange(N)
    C = np.cos(2 * np.pi * f * i) @ Pt
    S = np.sin(2 * np.pi * f * i) @ Pt
    return float((C * C).sum() + (S * S).sum())


def full_sweep(Pt: np.ndarray):
    N = Pt.shape[0]
    fs = np.arange(1, N // 2) / N
    pw = np.array([period_power(Pt, f) for f in fs])
    return fs, pw, float(np.median(pw))


def enrich(Pt: np.ndarray, period: float) -> float:
    _, _, med = full_sweep(Pt)
    return period_power(Pt, 1.0 / period) / med


def hf_stats(Pt: np.ndarray, period: float):
    """Enrichment at the target period + permutation p-value & thresholds from the HF band."""
    fs, pw, med = full_sweep(Pt)
    tgt = period_power(Pt, 1.0 / period) / med
    per = 1.0 / fs
    mask = (per >= HF_LO) & (per <= HF_HI)
    for ep in (PERIOD_P3, PERIOD_C10):
        mask &= (np.abs(fs - 1.0 / ep) > GUARD / Pt.shape[0])
    null = pw[mask] / med
    p = (np.sum(null >= tgt) + 1) / (null.size + 1)
    return {
        "enrichment": float(tgt),
        "perm_p": float(p),
        "n_hf_bins": int(null.size),
        "hf_null_p50": float(np.median(null)),
        "hf_null_p95": float(np.percentile(null, 95)),
        "hf_null_p99": float(np.percentile(null, 99)),
        "hf_null_max": float(null.max()),
        "detected_p99": bool(tgt > np.percentile(null, 99)),
    }


def inject_tone(A: np.ndarray, period: float, depth: float, rng) -> np.ndarray:
    """Distributed carrier: each column gets depth*col_std*cos(2pi i/period + phi_col)."""
    N = A.shape[0]
    i = np.arange(N)[:, None]
    phi = rng.uniform(0, 2 * np.pi, size=(1, A.shape[1]))
    tone = depth * A.std(0, keepdims=True) * np.cos(2 * np.pi * i / period + phi)
    return A + tone


def injection_curve(A: np.ndarray, period: float, p99: float):
    """depth -> mean/sd enrichment (N_PHASE random-phase draws); detection floor vs own HF p99."""
    curve = {}
    for d in DEPTHS:
        es = []
        for s in range(N_PHASE):
            rng = np.random.default_rng(SEED + int(d * 1000) * 97 + s)
            Ai = inject_tone(A, period, d, rng)
            Pt = Ai - Ai.mean(0, keepdims=True)
            es.append(enrich(Pt, period))
        curve[f"{d}"] = {"mean": float(np.mean(es)), "sd": float(np.std(es))}
    floor = next((d for d in DEPTHS if curve[f"{d}"]["mean"] > p99), None)
    return curve, floor


def single_bin_readout(P: np.ndarray, period: float):
    """Crude single-bin FFT readout (for anchor validation & legibility vs MECHANISM bins)."""
    Pt = P - P.mean(0, keepdims=True)
    F = np.fft.rfft(Pt, axis=0)
    binp = (np.abs(F) ** 2).sum(1)
    med = float(np.median(binp[1:]))
    b = int(round(P.shape[0] / period))
    rank = int((binp[1:] > binp[b]).sum()) + 1
    return {"bin": b, "period_at_bin": float(P.shape[0] / b),
            "pow_over_med": float(binp[b] / med), "rank": rank}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    Ps = {k: load_P(p) for k, p in CKPTS.items()}

    results = {}
    for k, P in Ps.items():
        Pt = P - P.mean(0, keepdims=True)
        p3 = hf_stats(Pt, PERIOD_P3)
        c10 = hf_stats(Pt, PERIOD_C10)
        p3_curve, p3_floor = injection_curve(P, PERIOD_P3, p3["hf_null_p99"])
        c10_curve, c10_floor = injection_curve(P, PERIOD_C10, c10["hf_null_p99"])
        results[k] = {
            "corpus": CORPUS[k], "N": int(P.shape[0]),
            "period3": {**p3, "single_bin": single_bin_readout(P, PERIOD_P3),
                        "injection_curve": p3_curve, "detection_floor_depth": p3_floor},
            "carrier_10_63": {**c10, "single_bin": single_bin_readout(P, PERIOD_C10),
                              "injection_curve": c10_curve, "detection_floor_depth": c10_floor},
        }

    # -------- VALIDATION GATE on the final checkpoint (must reproduce known anchors) --------
    fin = results["final_1024"]
    val = {
        "p3_enrichment": fin["period3"]["enrichment"],
        "p3_single_bin_pow_over_med": fin["period3"]["single_bin"]["pow_over_med"],
        "p3_single_bin_bin": fin["period3"]["single_bin"]["bin"],
        "c10_single_bin_bin": fin["carrier_10_63"]["single_bin"]["bin"],
        "c10_single_bin_rank": fin["carrier_10_63"]["single_bin"]["rank"],
    }
    assert 270 <= val["p3_enrichment"] <= 310, f"p3 enrich {val['p3_enrichment']} off anchor ~289"
    assert 280 <= val["p3_single_bin_pow_over_med"] <= 300, "crude p3 off anchor 288.9x"
    assert val["p3_single_bin_bin"] == 342, f"p3 bin {val['p3_single_bin_bin']} != 342"
    assert val["c10_single_bin_bin"] == 96, f"10.63 bin {val['c10_single_bin_bin']} != 96"
    assert val["c10_single_bin_rank"] <= 7, f"10.63 rank {val['c10_single_bin_rank']} > 7"
    validation_passed = True

    # -------- CORPUS CONTRAST (the load-bearing test): stage1_510 vs human_510 --------
    s, h = results["stage1_510"], results["human_510"]
    contrast = {
        "period3": {
            "stage1_72sp": s["period3"]["enrichment"],
            "human_only": h["period3"]["enrichment"],
            "ratio_72sp_over_human": s["period3"]["enrichment"] / h["period3"]["enrichment"],
            "human_detected_p99": h["period3"]["detected_p99"],
            "human_perm_p": h["period3"]["perm_p"],
            "human_detection_floor_depth": h["period3"]["detection_floor_depth"],
        },
        "carrier_10_63": {
            "stage1_72sp": s["carrier_10_63"]["enrichment"],
            "human_only": h["carrier_10_63"]["enrichment"],
            "ratio_72sp_over_human": (s["carrier_10_63"]["enrichment"]
                                      / h["carrier_10_63"]["enrichment"]),
            "human_detected_p99": h["carrier_10_63"]["detected_p99"],
            "human_perm_p": h["carrier_10_63"]["perm_p"],
            "human_detection_floor_depth": h["carrier_10_63"]["detection_floor_depth"],
        },
    }

    record = {
        "experiment": "origin-siblings-pspectrum-v1",
        "estimator": {
            "method": "leakage-free direct cos/sin projection (exact at any N)",
            "period3_nt": PERIOD_P3, "carrier_10_63_nt": PERIOD_C10,
            "hf_band_periods": [HF_LO, HF_HI], "guard_bins": GUARD,
            "injection_depths": list(DEPTHS), "n_phase_draws": N_PHASE, "seed": SEED,
            "note_nucleosomal": "≈10.298 nt DNA signal is DISTINCT and NOT measured here",
        },
        "validation_final_anchors": val,
        "validation_passed": validation_passed,
        "per_checkpoint": results,
        "corpus_contrast_stage1_vs_human": contrast,
    }
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True))
    print(f"wrote {OUT.relative_to(REPO)}")

    man = ExperimentManifest.create(
        "origin-siblings-pspectrum-v1",
        ["scripts/splicebert/run_sibling_pspectrum.py"],
        repo=REPO, status="CANONICAL",
        config={"period3_nt": PERIOD_P3, "carrier_10_63_nt": PERIOD_C10,
                "hf_band": [HF_LO, HF_HI], "seed": SEED},
        seeds={"master": SEED},
    )
    for p in CKPTS.values():
        man.add_input(p / "pytorch_model.bin")
    man.add_output(OUT)
    OUT.with_suffix(".manifest.json").write_text(json.dumps(man.to_dict(), indent=2, sort_keys=True))
    print(f"wrote {OUT.with_suffix('.manifest.json').relative_to(REPO)}")

    # ---- console summary ----
    print("\n=== VALIDATION (final anchors) passed:", validation_passed, "===")
    print(f"  p3 enrich={val['p3_enrichment']:.1f}x  crude bin342={val['p3_single_bin_pow_over_med']:.1f}x"
          f"  |  10.63 bin96 rank={val['c10_single_bin_rank']}")
    print("\n=== CORPUS CONTRAST (stage1 72-sp vs human-only, both N=512) ===")
    for lab, key in [("period-3", "period3"), ("≈10.63", "carrier_10_63")]:
        c = contrast[key]
        print(f"  {lab:8s}: 72sp={c['stage1_72sp']:8.1f}x  human={c['human_only']:6.1f}x  "
              f"ratio={c['ratio_72sp_over_human']:6.1f}x  human_detected={c['human_detected_p99']}"
              f"  human_p={c['human_perm_p']:.3g}  floor_depth={c['human_detection_floor_depth']}")


if __name__ == "__main__":
    main()
