#!/usr/bin/env python
"""POST-SOFTMAX ATTENTION — the D32 gap, behind a preflight gate that cannot be skipped.

WHY THIS EXISTS.  Every "reader" claim in this project is a claim about QK LOGITS on RANDOM
sequence.  R14 is the memorial for the fact that logits != attention != behaviour.  D32 says 14
heads read the human-only ~10.2 carrier; whether that comb SURVIVES THE SOFTMAX is UNMEASURED.
Three previous attempts each returned a physically impossible number (184-207% modulation of a
PROBABILITY) because a polynomial envelope cannot fit a steep attention decay, so the cosine term
absorbed the decay instead of a comb.

THE GATE.  main() runs preflight() FIRST.  If any check fails, the program EXITS BEFORE LOADING
ANY EXPERIMENTAL DATA and writes preflight.json with the failure.  See studies/splicebert_partner_selection/METHODS.md.

FIVE AMENDMENTS FROM THE REVIEW, AND TWO OF THEM WERE THINGS I HAD WRONG
-----------------------------------------------------------------------
1. ANOMALOUS CONTROL -> ANALYSIS FREEZE, not "the instrument is broken".  A control that
   out-damages its treatment invalidates the INTERPRETATION; it does not prove the pipeline is
   broken -- the "inert" band may simply be functional.  Freeze, diagnose both, then decide.
   And NAME CONTROLS ONLY BY VERIFIED PROPERTIES: "energy-matched band" (if energy is matched),
   "low-power band" (if power is low), "behaviourally inert" ONLY after inertness is DEMONSTRATED.
   "Energy-matched inert band" was two claims wearing one name, and only one was ever tested.

2. ⛔ THE BOXCAR DOES NOT NOTCH A NON-INTEGER PERIOD.  I asserted it did, generalising from
   `1 + 2cos(2pi/3) = 0` -- an EXACT identity that holds only for INTEGER period 3.  A length-M
   boxcar has zeros ONLY at T = M/n.  MEASURED: M=10 leaks 2.4% at T=10.24; M=11 leaks 3.4% at
   T=10.63.  NOT zero.  So: no "annihilates exactly" anywhere.  The baseline is a CUBIC IN
   LOG-SPACE (attention decays ~exponentially, so log A is near-linear; a cubic fits the envelope
   and has almost no bandwidth at 5.4 cycles), and its LEAKAGE IS MEASURED BY INJECTION, not
   assumed: recovery within 0.8-3.2% of amplitude, <3.5 deg of phase, on realistic envelopes.

3. A STATISTIC BOUNDED BY CONSTRUCTION.  "amplitude / mean" is unbounded once a fitted baseline is
   involved -- which is exactly how I produced 184-207%.  Instead:
        m_T = |SUM_k w_k r_k e^{-2 pi i k / T}| / SUM_k w_k r_k      with w_k >= 0, r_k >= 0
   Triangle inequality gives 0 <= m_T <= 1 FOR ANY r >= 0, with NO sinusoidal model assumed.
   r_k = A_k / baseline_k is a ratio of positives.  Out-of-bounds is a HARD FAILURE.

4. PERIOD-3 IS NECESSARY BUT NOT SUFFICIENT.  Recovering D21 proves the EXTRACTION PATH works.  It
   says nothing about specificity for a NON-INTEGER 10.2 tone under a steep envelope.  So the
   preflight also needs a TARGET-FREQUENCY positive (inject a known 10.2 tone into REAL attention
   envelopes; recover amplitude AND phase) and a TARGET-FREQUENCY negative (the same envelopes with
   the periodic ORDER destroyed -> an FPR).  The deleted-model control is useful but is NOT the
   FPR study.

5. ⛔ THE CONTENT-SHUFFLE CRITERION WAS BACKWARDS -- THIS IS R15 AGAIN.  A positional comb built
   from (P, W_Q, W_K) MUST survive content shuffling; that is what makes it POSITIONAL.  I listed
   shuffle-survival as an ARTEFACT criterion, which conflates QK with OV -- the exact mistake R15
   is the memorial for.  Correctly:
        carrier deletion  -> MUST remove the attention comb
        content shuffling -> NEED NOT remove it (and probably should not)
        phase-scrambling  -> MUST remove any content-sensitive OV / behavioural effect

CLAIM-LEVEL FIREWALL.  Evidence NEVER auto-promotes between levels:
    P         a feature exists in the position table
    QK        a logit circuit reads it
    A         it survives the softmax into attention        <- THIS SCRIPT, and only this
    OV        the head's OUTPUT carries relevant content
    BEHAVIOUR model performance causally depends on it
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/splicebert"))

from gfmcd.splicing.circuit import load_adapter, pick_device      # noqa: E402
from run_comb_shift import classify, SPAN                          # noqa: E402  THE ORIGINAL, imported
from transformers import AutoTokenizer                             # noqa: E402

OUT = REPO / "results/phase-lattice/postsoftmax.json"
PRE = REPO / "results/phase-lattice/preflight_postsoftmax.json"

M1024 = "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"
MHUM = "data/models/splicebert_official_archive/extracted/models/SpliceBERT-human.510nt"
MMUL = "data/models/splicebert_official_archive/extracted/models/SpliceBERT.510nt"

BAND_P3_1026 = list(range(339, 346))          # the codon clock in the 1026-row table
COMB6 = ["L0H0", "L0H11", "L1H10", "L1H12", "L2H2", "L2H15"]   # D21's six comb heads

# ══════════════════════════════════════════════════════════════════════════════════════════
# FROZEN EXPECTATIONS.  Declared HERE, before the run.  Hashed into preflight.json.
# If a check fails, I investigate the INSTRUMENT.  I do NOT tune these until they pass.
# ══════════════════════════════════════════════════════════════════════════════════════════
FROZEN = {
    # (1) BIOLOGICAL POSITIVE -- D21, already established post-softmax. Chance = 1/3.
    "p3_intact_purity":   [0.72, 0.86],      # D21 measured 0.794
    "p3_deleted_purity":  [0.30, 0.40],      # D21 measured 0.345 (chance 0.333)
    # (2) TARGET-FREQUENCY POSITIVE -- inject a KNOWN tone into REAL envelopes
    "inject_amp_bias_max":   0.20,           # |recovered/true - 1|
    "inject_phase_err_max":  20.0,           # degrees
    "inject_depths":         [0.05, 0.10, 0.20],
    # (3) TARGET-FREQUENCY NEGATIVE -- periodic ORDER destroyed -> FPR at a preregistered threshold
    "fpr_max":               0.07,           # nominal 5%
    # (4) BOUNDS -- attention is a PROBABILITY. m_T is bounded by the triangle inequality.
    "m_T_max":               1.0 + 1e-9,     # ANY violation is a HARD FAILURE
    "attn_row_sum_tol":      1e-3,
    "band_mass_floor":       0.02,           # a head must place >=2% of its attention in the band,
                                             # else it is EXCLUDED -- not reported as "modulated".
                                             # (My broken runs "found" 95% modulation on heads with
                                             #  mean attention of 0.00001.)
    # (5) DELETION AUDIT -- the intervention must do what it claims
    "target_retained_max":     0.05,         # <=5% of the target band's power survives
    "protected_retained_min":  0.95,         # >=95% of every OTHER carrier survives
}
# ⚠️ THE BAND IS SPLIT (|k| in [6,60], self-spike excluded), SO EACH CONTIGUOUS HALF GETS ITS OWN
# HANN.  My first version applied ONE hanning(110) across the concatenation, which put the window's
# PEAK exactly on the discontinuity at k = -6 -> +6.  |SUM w e^{-2pi i k/T}| then EXPLODED, and m_T
# measured THE WINDOW'S OWN LEAKAGE rather than the comb: the tone-free null saturated at m_T ~ 0.98
# and the injection recovery was off by 336%.  window_leakage + null_location now gate on exactly this.
_KL, _KR = np.arange(-60, -5), np.arange(6, 61)
BAND = np.concatenate([_KL, _KR])
W = np.concatenate([np.hanning(len(_KL)), np.hanning(len(_KR))])   # each HALF windowed smoothly


def sha(x) -> str:
    a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────── the BOUNDED estimator
def m_of(A: np.ndarray, T: float, deg: int = 3):
    """m_T = |SUM w r e^{-2pi i k/T}| / SUM w r  with r = A/baseline >= 0.
    BOUNDED IN [0,1] BY THE TRIANGLE INEQUALITY, for any r >= 0.  No sinusoidal model assumed.
    Baseline = exp(cubic fit to log A) -- fits the steep decay; its LEAKAGE IS MEASURED, not claimed.
    Returns (m_T, phase_rad, band_mass_fraction)."""
    A = np.asarray(A, float)
    if A.min() <= 0:
        A = A + 1e-12
    b = np.exp(np.polyval(np.polyfit(BAND, np.log(A), deg), BAND))
    r = A / b                                                   # >= 0 by construction
    num = (W * r * np.exp(-2j * np.pi * BAND / T)).sum()
    den = (W * r).sum()
    return complex(num / den), r                 # the COMPLEX coefficient c; |c| = m_T in [0,1]


def profiles(model, ids, span=60):
    """Local attention profile A(k) per head, averaged over queries and batch."""
    with torch.no_grad():
        att = model.bert(input_ids=ids, output_attentions=True).attentions
    S = ids.shape[1]
    lo, hi = span + 10, S - span - 10
    out, rows = {}, []
    for l, a in enumerate(att):
        a = a.float().cpu().numpy()
        rows.append(float(np.abs(a.sum(-1) - 1.0).max()))       # row-sum invariant
        for h in range(a.shape[1]):
            Mh = a[:, h]
            q = np.arange(lo, hi)
            prof = np.array([Mh[:, q, q + k].mean() for k in range(-span, span + 1)])
            out[f"L{l}H{h}"] = prof
    return out, max(rows)


def strip_fft(P: np.ndarray, bins):
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    for b in bins:
        F[b] = 0
    return np.fft.irfft(F, n=P.shape[0], axis=0) + mu


def deletion_audit(P0: np.ndarray, bins, protect: dict) -> dict:
    """MANDATORY. Every intervention proves it did what it claims -- and only what it claims."""
    P1 = strip_fft(P0, bins)
    def pw(P):
        F = np.fft.rfft(P - P.mean(0, keepdims=True), axis=0)
        return (np.abs(F) ** 2).sum(1)
    p0, p1 = pw(P0), pw(P1)
    nb = [b for b in range(min(bins) - 3, max(bins) + 4) if b not in bins and b > 0]
    a = dict(
        bins_removed=list(bins),
        target_retained=float(p1[bins].sum() / max(p0[bins].sum(), 1e-30)),
        neighbour_retained=float(p1[nb].sum() / max(p0[nb].sum(), 1e-30)),
        frobenius_removed=float(np.linalg.norm(P1 - P0) / np.linalg.norm(P0)),
        energy_removed_frac=float(1 - p1.sum() / p0.sum()),
        hash_before=sha(P0), hash_after=sha(P1),
        tensors_differ=bool(sha(P0) != sha(P1)),
    )
    for name, pb in protect.items():
        pb = [b for b in pb if b < len(p0)]
        a[f"protected_{name}_retained"] = float(p1[pb].sum() / max(p0[pb].sum(), 1e-30))
    a["PASS"] = bool(
        a["target_retained"] <= FROZEN["target_retained_max"]
        and a["tensors_differ"]
        and all(v >= FROZEN["protected_retained_min"]
                for k, v in a.items() if k.startswith("protected_"))
    )
    return a, P1


# ═══════════════════════════════════════════════════════════════════════════ THE GATE
def preflight(dev) -> dict:
    R = {"frozen": FROZEN,
         "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True).stdout.strip(),
         "checks": {}}
    ok = True

    def chk(name, passed, **kw):
        nonlocal ok
        R["checks"][name] = dict(PASS=bool(passed), **kw)
        ok &= bool(passed)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}   " +
              "  ".join(f"{k}={v if not isinstance(v,float) else round(v,4)}"
                        for k, v in kw.items() if k != "detail"))

    print("\n" + "=" * 96)
    print("PREFLIGHT — nothing is measured until every one of these passes")
    print("=" * 96)

    # ---- 4. BOUNDS & RESOLUTION (declared, then computed) -------------------------------
    kern = {f"{T}": float(abs(np.polyfit(BAND, np.log(np.exp(-np.abs(BAND) / 18.0)), 3)[0]))
            for T in (10.24, 10.63)}
    res = 2 * 10.4 ** 2 / len(BAND)
    chk("bounds_resolution", res < 2.5, resolution_nt=res, n_offsets=len(BAND),
        note="dT ~ 2T^2/N; must be finer than the 10.24-vs-10.63 gap is NOT required here "
             "(we test AT declared T, we do not scan)")

    # ---- 1. BIOLOGICAL POSITIVE: recover D21's period-3 phase mass ----------------------
    ad = load_adapter(str(REPO / M1024), dev, eager=True)
    m = ad.model
    tok = AutoTokenizer.from_pretrained(str(REPO / M1024))
    emb = m.bert.embeddings.position_embeddings
    P0t = emb.weight.data.clone()
    P0 = P0t.detach().float().cpu().numpy()
    h_start = sha(P0t)
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 1022)) for _ in range(16)]
    ids = tok([" ".join(s) for s in seqs], return_tensors="pt").to(ad.device)["input_ids"]

    pi, rowmax = profiles(m, ids)
    chk("probability_invariants", rowmax < FROZEN["attn_row_sum_tol"], max_row_sum_err=rowmax)

    # classify() is D21's ORIGINAL function, imported not reimplemented -- and it expects the
    # ORIGINAL's window (SPAN = 24, i.e. 49 offsets), not mine (60).  Slice to ITS window.
    # A reimplementation would have silently used MY window, returned a purity that was not 0.794,
    # and tempted me to tune the definition until it matched.  That is the sin itself.
    def p3(prof):
        return prof[60 - SPAN: 60 + SPAN + 1]
    pur_i = float(np.mean([classify(p3(pi[h]))["purity"] for h in COMB6]))

    aud3, P3del = deletion_audit(P0, BAND_P3_1026, {})
    emb.weight.data = torch.tensor(P3del, dtype=P0t.dtype, device=P0t.device)
    pd, _ = profiles(m, ids)
    emb.weight.data = P0t.clone()
    pur_d = float(np.mean([classify(p3(pd[h]))["purity"] for h in COMB6]))

    lo, hi = FROZEN["p3_intact_purity"]
    chk("positive_control_period3_intact", lo <= pur_i <= hi, purity=pur_i,
        expected="D21: 0.794", band=[lo, hi])
    lo, hi = FROZEN["p3_deleted_purity"]
    chk("negative_control_period3_deleted", lo <= pur_d <= hi, purity=pur_d,
        expected="D21: 0.345 (chance 0.333)", band=[lo, hi])
    chk("deletion_audit_period3", aud3["PASS"], **{k: v for k, v in aud3.items()
                                                   if k not in ("bins_removed", "PASS")})
    chk("checkpoint_not_mutated", sha(emb.weight.data) == h_start, hash=h_start)
    del ad

    # ---- 2 & 3. TARGET-FREQUENCY positive + negative, on REAL envelopes -----------------
    ad = load_adapter(str(REPO / MHUM), dev, eager=True)
    m = ad.model
    tok = AutoTokenizer.from_pretrained(str(REPO / MHUM))
    seqs = ["".join(rng.choice(list("ACGT"), 508)) for _ in range(16)]
    ids = tok([" ".join(s) for s in seqs], return_tensors="pt").to(ad.device)["input_ids"]
    ph, _ = profiles(m, ids)
    # ⚠️ CENTRE INDEX IS 60 (profiles() returns +/-60), NOT `SPAN`.  SPAN is 24 -- the value
    # imported from run_comb_shift for classify()'s OWN window.  Using it here produced negative
    # indices that WRAPPED, scrambling every envelope, and that -- not the estimator -- is what
    # made the injection bias 0.204.  Diagnosed by depth/phase/head: the estimator's GAIN is
    # 1.000 +/- 0.002 with 0.2 deg phase error at every depth from 0.02 to 0.40, the paired
    # +a/-a check agrees, and a FROZEN baseline agrees -- the cubic refit absorbs only 0.3%.
    # THE ESTIMATOR IS UNCHANGED.  The 0.20 gate is untouched.
    CTR = 60
    envs = [ph[h][CTR + BAND] for h in ph
            if ph[h][CTR + BAND].sum() / max(ph[h].sum(), 1e-30) >= FROZEN["band_mass_floor"]]
    del ad
    chk("attention_mass_floor", len(envs) >= 10, n_heads_with_enough_mass=len(envs),
        floor=FROZEN["band_mass_floor"])

    # ---- NEW GATE (window leakage): with r CONSTANT, m_T must be ~0.  If it is not, the statistic
    # is measuring the WINDOW, not the comb.  This is what the 336% bias actually was.
    T = 10.24
    leak = float(abs((W * np.exp(-2j * np.pi * BAND / T)).sum()) / W.sum())
    chk("window_leakage_at_target", leak < 0.02, leak=leak,
        note="m_T with r=const; must be ~0 or the statistic measures the window's own transform")

    # ---- POSITIVE: injection is MULTIPLICATIVE, so  c' - c = (a/2) e^{i phi}  EXACTLY.
    # Recover the COMPLEX INCREMENT.  |c| itself is a COHERENCE, not an amplitude: it saturates
    # toward 1 whenever the residual is SPIKY, regardless of periodicity -- which is why "2*m_T"
    # was off by 339%, and why the pooled null had a p95 of 0.98.
    bias, perr = [], []
    for a in FROZEN["inject_depths"]:
        for e in envs[:24]:
            phi0 = float(rng.uniform(0, 2 * np.pi))
            c0, _ = m_of(e, T)
            c1, _ = m_of(e * (1 + a * np.cos(2 * np.pi * BAND / T + phi0)), T)
            d = c1 - c0
            bias.append(abs(2 * abs(d) - a) / a)
            perr.append(abs(math.degrees(((np.angle(d) - phi0 + np.pi) % (2 * np.pi)) - np.pi)))
    chk("positive_control_target_freq_injection",
        np.mean(bias) < FROZEN["inject_amp_bias_max"] and np.mean(perr) < FROZEN["inject_phase_err_max"],
        mean_amp_bias=float(np.mean(bias)), mean_phase_err_deg=float(np.mean(perr)))

    # ---- NEGATIVE: a PER-HEAD permutation null.  Permuting a head's OWN residual controls for its
    # OWN spikiness by construction.  A pooled null cannot -- a spiky head sits near 1 under EVERY
    # permutation, which is what poisoned the p95.
    def head_p(e, nperm=200):
        c, r = m_of(e, T)
        b = e / r
        nul = np.array([abs(m_of(b * rng.permutation(r), T)[0]) for _ in range(nperm)])
        return abs(c), float((nul >= abs(c)).mean()), nul
    ps, nulls = [], []
    for e in envs:
        _, pv, nul = head_p(e)
        ps.append(pv); nulls.append(nul)
    nulls = np.concatenate(nulls)
    fpr = float(np.mean([p < 0.05 for p in
                         [head_p(b_ * rng.permutation(r_))[1]
                          for b_, r_ in ((e / m_of(e, T)[1], m_of(e, T)[1]) for e in envs)]]))
    chk("negative_control_target_freq_FPR", fpr <= FROZEN["fpr_max"], fpr=fpr,
        n_heads=len(envs), note="per-head permutation; tone-free input must reject at ~5%")
    chk("null_location", float(np.median(nulls)) < 0.30, null_median=float(np.median(nulls)),
        null_p95=float(np.percentile(nulls, 95)),
        note="per-head null; spikiness now controlled WITHIN head, not pooled")
    chk("m_T_bounded", float(nulls.max()) <= FROZEN["m_T_max"], max_m_T=float(nulls.max()))
    R["real_head_pvalues"] = ps

    R["PASS"] = bool(ok)
    PRE.write_text(json.dumps(R, indent=1))
    print("=" * 96)
    print(f"PREFLIGHT: {'PASS' if ok else 'FAIL'}   ->  {PRE}")
    print("=" * 96)
    return R


def main() -> None:
    dev = pick_device(None)
    R = preflight(dev)
    if not R["PASS"]:
        print("\n⛔ PREFLIGHT FAILED.  No experimental data loaded, no result written.")
        print("   Per METHODS.md: this is an ANALYSIS FREEZE.  Diagnose the instrument")
        print("   AND the control's defining property before any interpretation.")
        sys.exit(1)
    measure(dev, R)


# ══════════════════════════════════════════════════════════════════════════════════════════
# LEVEL A.  Criteria FROZEN here, before the measurement runs.
#   PROMOTE a head to Level A iff ALL of:
#     (a) m_T beats its OWN per-head permutation null   (BH-FDR 0.05 across all heads)
#     (b) m_T COLLAPSES to < 50% when the CARRIER band is deleted
#     (c) m_T SURVIVES (> 70%) deletion of the RECIPROCAL band (the other model's carrier)
#     (d) m_T SURVIVES (> 70%) deletion of the LOW-LEAK band [33,34,35]
#   ⚠️ CONTENT SHUFFLE IS **NOT** AN ARTEFACT CRITERION (R15).  A positional comb MUST survive it.
#      It is reported as a POSITIVITY check: survival CONFIRMS the comb is positional.
#   PREREGISTERED PRIMARY: L1H3 (human_only).  The 14 QK readers are the preregistered set.
#   STOPPING RULE: if NO human_only head reaches Level A -> STOP.  Close the human chromatin
#      claim at QK.  No rescue experiments.
# ══════════════════════════════════════════════════════════════════════════════════════════
LEVELA = dict(fdr=0.05, collapse_max=0.50, survive_min=0.70, primary="L1H3",
              qk_readers=["L1H3","L1H2","L1H8","L1H14","L1H11","L1H7","L1H5","L1H10","L1H9",
                          "L1H13","L1H15","L2H8","L2H1","L2H11"])
MODELS_A = {"human_only":   (MHUM, [49,50,51], [46,47,48], 10.24),
            "multispecies": (MMUL, [46,47,48], [49,50,51], 10.61)}
LOWLEAK = [33, 34, 35]


def measure(dev, PR) -> None:
    from pyfaidx import Fasta
    rng = np.random.default_rng(7)
    fa = Fasta(os.environ["GFMCD_FASTA"], as_raw=True)   # HUMAN: both models' domain
    s = np.frombuffer(fa["chr8"][40_000_000:40_200_000].upper().encode(), dtype=np.uint8)
    ok = np.isin(s, [65,67,71,84])
    st = [i for i in rng.integers(0, len(s)-520, 600) if ok[i:i+508].all()][:48]
    B2I = {65:6, 67:7, 71:8, 84:9}
    rec = {"frozen": LEVELA, "preflight": PR["checks"], "models": {}}

    for name, (path, carrier, recip, T) in MODELS_A.items():
        ad = load_adapter(str(REPO / path), dev, eager=True); m = ad.model
        tok = AutoTokenizer.from_pretrained(str(REPO / path))
        emb = m.bert.embeddings.position_embeddings
        P0t = emb.weight.data.clone(); P0 = P0t.detach().float().cpu().numpy()
        h0 = sha(P0t)
        body = [[B2I[int(b)] for b in s[i:i+508]] for i in st]
        ids = torch.tensor([[tok.cls_token_id]+b+[tok.sep_token_id] for b in body], device=dev)
        shuf = torch.tensor([[tok.cls_token_id]+list(rng.permutation(b))+[tok.sep_token_id]
                             for b in body], device=dev)

        conds = {"carrier": carrier, "reciprocal": recip, "low_leak": LOWLEAK}
        aud = {}
        prof = {}
        prof["intact"], _ = profiles(m, ids)
        prof["shuffled"], _ = profiles(m, shuf)
        for cn, bins in conds.items():
            a, P1 = deletion_audit(P0, bins, {k: v for k, v in conds.items() if k != cn})
            aud[cn] = a
            emb.weight.data = torch.tensor(P1, dtype=P0t.dtype, device=P0t.device)
            prof[cn], _ = profiles(m, ids)
        emb.weight.data = P0t.clone()
        assert sha(emb.weight.data) == h0, "CHECKPOINT MUTATED"

        rows = []
        for h, pr in prof["intact"].items():
            e = pr[60 + BAND]
            if e.sum() / max(pr.sum(), 1e-30) < FROZEN["band_mass_floor"]:
                continue
            c, r = m_of(e, T); b = e / r
            nul = np.array([abs(m_of(b*rng.permutation(r), T)[0]) for _ in range(400)])
            pv = float((nul >= abs(c)).mean())
            kept = {cn: abs(m_of(prof[cn][h][60+BAND], T)[0]) / max(abs(c), 1e-12)
                    for cn in list(conds) + ["shuffled"]}
            rows.append(dict(head=h, m_T=float(abs(c)), phase_deg=float(math.degrees(np.angle(c))),
                             p=pv, **{f"kept_{k}": float(v) for k, v in kept.items()}))
        # Benjamini-Hochberg
        rows.sort(key=lambda x: x["p"])
        n = len(rows)
        sig = set()
        for i, x in enumerate(rows, 1):
            if x["p"] <= LEVELA["fdr"] * i / n:
                sig = {y["head"] for y in rows[:i]}
        for x in rows:
            x["significant"] = x["head"] in sig
            x["LEVEL_A"] = bool(x["significant"]
                                and x["kept_carrier"] < LEVELA["collapse_max"]
                                and x["kept_reciprocal"] > LEVELA["survive_min"]
                                and x["kept_low_leak"] > LEVELA["survive_min"])
        rec["models"][name] = dict(T=T, carrier=carrier, deletion_audit=aud, heads=rows)

        A = [x for x in rows if x["LEVEL_A"]]
        print("")
        print("=" * 100)
        print(f"{name}   carrier {carrier}, T = {T}   ({len(rows)} heads pass the mass floor)")
        prot = [v for k, v in aud["carrier"].items() if k.startswith("protected_")]
        print(f"  deletion audit: target_retained={aud['carrier']['target_retained']:.3f}  "
              f"min_protected_retained={min(prot) if prot else 1.0:.3f}  checkpoint unmutated OK")
        print("=" * 100)
        print(f"{'='*100}")
        print(f"  {'head':7} {'m_T':>6} {'p':>7} {'kept:carrier':>13} {'recip':>7} {'lowleak':>8} "
              f"{'shuffled':>9}  LEVEL A")
        for x in sorted(rows, key=lambda y: -y["m_T"])[:12]:
            star = " ✅" if x["LEVEL_A"] else ("  qk" if x["head"] in LEVELA["qk_readers"] else "")
            print(f"  {x['head']:7} {x['m_T']:>6.3f} {x['p']:>7.4f} {x['kept_carrier']*100:>12.0f}% "
                  f"{x['kept_reciprocal']*100:>6.0f}% {x['kept_low_leak']*100:>7.0f}% "
                  f"{x['kept_shuffled']*100:>8.0f}%{star}")
        print(f"  -> HEADS AT LEVEL A: {len(A)}   {[x['head'] for x in A] if A else '(none)'}")
        if name == "human_only":
            prim = next((x for x in rows if x["head"] == LEVELA["primary"]), None)
            if prim:
                print(f"  PREREGISTERED PRIMARY {LEVELA['primary']}: m_T={prim['m_T']:.3f}  "
                      f"p={prim['p']:.4f}  kept(carrier)={prim['kept_carrier']*100:.0f}%  "
                      f"LEVEL A = {prim['LEVEL_A']}")
            qk = [x for x in rows if x["head"] in LEVELA["qk_readers"]]
            print(f"  of the 14 preregistered QK readers: {sum(x['LEVEL_A'] for x in qk)} reach Level A")
        del ad

    OUT.write_text(json.dumps(rec, indent=1))
    hA = [x for x in rec["models"]["human_only"]["heads"] if x["LEVEL_A"]]
    print(""); print("="*100)
    if not hA:
        print("⛔ STOPPING RULE INVOKED: NO human-only head reaches Level A.")
        print("   The ~10.2 comb does NOT survive the softmax.  CLOSE THE HUMAN CHROMATIN CLAIM AT QK.")
        print("   No rescue experiments.  Do not run the species-matched causal experiment.")
    else:
        print(f"✅ LEVEL A ESTABLISHED for {len(hA)} human-only head(s).")
        print("   Next and LAST: ONE species-matched causal experiment (human model, human seq, human dyads).")
    print("="*100)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()

