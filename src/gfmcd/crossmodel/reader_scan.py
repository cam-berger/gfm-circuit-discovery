"""Post-softmax attention reader scan + P-band causal surgery, model-agnostic port.

Ported from scripts/splicebert/run_postsoftmax.py.  The SpliceBERT script hard-codes the
biological preflight (D21 period-3 purity 0.794, the six comb heads, etc.) -- that biology
does NOT transfer.  What transfers is the INSTRUMENT MECHANICS, kept verbatim in spirit:

  * ``profiles`` -- the local post-softmax attention profile A(k) per head, the diagonal
    band mean over queries and batch.  This is the object every "reader" claim is about:
    attention, AFTER the softmax, on real (random) sequence -- not a QK logit.
  * ``m_of`` -- the BOUNDED coherence statistic.  m_T = |SUM w r e^{-2pi i k/T}| / SUM w r
    with r = A/baseline >= 0, bounded in [0,1] by the triangle inequality, no sinusoidal
    model assumed.  The band is SPLIT (self-spike excluded) and EACH HALF gets its own Hann
    window, because one window across the split put its peak on the discontinuity and made
    the statistic measure the window's own leakage (the 336% bug).  Baseline = exp(cubic in
    log A), which fits the steep attention decay with negligible bandwidth at carrier freq.
  * ``strip_fft`` / ``deletion_audit`` -- the P-band surgery + its mandatory audit (target
    band removed, every protected band retained, tensor actually changed).

CLAIM FIREWALL (do not cross without the evidence):
    P         a feature exists in the position table          <- carrier_spectrum
    QK        a logit circuit reads it
    A         it survives the softmax into attention          <- profiles + m_of, THIS module
    OV        the head's output carries relevant content
    BEHAVIOUR model performance causally depends on it        <- surgery on masked-LM readout
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def band_and_window(span_inner: int = 6, span_outer: int = 60):
    """The split band |k| in [span_inner, span_outer], each contiguous half Hann-windowed."""
    kl = np.arange(-span_outer, -span_inner + 1)
    kr = np.arange(span_inner, span_outer + 1)
    band = np.concatenate([kl, kr])
    w = np.concatenate([np.hanning(len(kl)), np.hanning(len(kr))])
    return band, w


def m_of(A: np.ndarray, T: float, band: np.ndarray, w: np.ndarray, deg: int = 3):
    """Bounded coherence coefficient c at period T.  |c| = m_T in [0,1].

    Returns (c: complex, r: residual-ratio array).  Baseline = exp(cubic fit to log A).
    """
    A = np.asarray(A, float)
    if A.min() <= 0:
        A = A - A.min() + 1e-12
    base = np.exp(np.polyval(np.polyfit(band, np.log(A), deg), band))
    r = A / base
    num = (w * r * np.exp(-2j * np.pi * band / T)).sum()
    den = (w * r).sum()
    return complex(num / den), r


def profiles(model, input_ids, *, span: int = 60, attention_mask=None):
    """Local post-softmax attention profile A(k) per head, averaged over queries and batch.

    Returns ({head_label: profile[2*span+1]}, max_row_sum_error).  Uses the HF
    ``output_attentions`` path; requires an eager attention implementation (SDPA silently
    returns zero attentions).  Head labels are ``L{l}H{h}``.
    """
    kw = {"input_ids": input_ids, "output_attentions": True}
    if attention_mask is not None:
        kw["attention_mask"] = attention_mask
    with torch.no_grad():
        out = model(**kw, return_dict=True)
    att = out.attentions
    S = input_ids.shape[1]
    lo, hi = span + 10, S - span - 10
    if hi <= lo:
        raise ValueError(f"sequence too short ({S}) for span {span}")
    prof, rowmax = {}, 0.0
    q = np.arange(lo, hi)
    for l, a in enumerate(att):
        a = a.float().cpu().numpy()
        rowmax = max(rowmax, float(np.abs(a.sum(-1) - 1.0).max()))
        for h in range(a.shape[1]):
            Mh = a[:, h]
            prof[f"L{l}H{h}"] = np.array([Mh[:, q, q + k].mean() for k in range(-span, span + 1)])
    return prof, rowmax


def band_mass_fraction(profile: np.ndarray, band: np.ndarray, span: int = 60) -> float:
    """Fraction of a head's total local attention that sits in the analysis band."""
    ctr = span
    tot = profile.sum()
    return float(profile[ctr + band].sum() / max(tot, 1e-30))


def strip_fft(P: np.ndarray, bins) -> np.ndarray:
    """Zero the given rfft bins of P (dim-wise), return the reconstructed table."""
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    for b in bins:
        F[b] = 0
    return np.fft.irfft(F, n=P.shape[0], axis=0) + mu


def attenuate_fft(P: np.ndarray, bins, factor: float) -> np.ndarray:
    """Scale the given rfft bins of P by ``factor`` (0=delete, 1=intact)."""
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    for b in bins:
        F[b] = F[b] * factor
    return np.fft.irfft(F, n=P.shape[0], axis=0) + mu


def _sha(x) -> str:
    import hashlib
    a = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def deletion_audit(P0: np.ndarray, bins, protect: dict,
                   *, target_retained_max: float = 0.05,
                   protected_retained_min: float = 0.95) -> tuple[dict, np.ndarray]:
    """Prove a band deletion did what it claims and only what it claims."""
    P1 = strip_fft(P0, bins)

    def pw(P):
        F = np.fft.rfft(P - P.mean(0, keepdims=True), axis=0)
        return (np.abs(F) ** 2).sum(1)

    p0, p1 = pw(P0), pw(P1)
    nb = [b for b in range(min(bins) - 3, max(bins) + 4) if b not in bins and 0 < b < len(p0)]
    a = dict(
        bins_removed=list(bins),
        target_retained=float(p1[bins].sum() / max(p0[bins].sum(), 1e-30)),
        neighbour_retained=float(p1[nb].sum() / max(p0[nb].sum(), 1e-30)) if nb else None,
        frobenius_removed=float(np.linalg.norm(P1 - P0) / np.linalg.norm(P0)),
        energy_removed_frac=float(1 - p1.sum() / p0.sum()),
        hash_before=_sha(P0), hash_after=_sha(P1),
        tensors_differ=bool(_sha(P0) != _sha(P1)),
    )
    for name, pb in protect.items():
        pb = [b for b in pb if 0 < b < len(p0)]
        a[f"protected_{name}_retained"] = float(p1[pb].sum() / max(p0[pb].sum(), 1e-30))
    a["PASS"] = bool(
        a["target_retained"] <= target_retained_max
        and a["tensors_differ"]
        and all(v >= protected_retained_min for k, v in a.items()
                if k.startswith("protected_") and v is not None)
    )
    return a, P1


def head_permutation_fpr(profile_band: np.ndarray, T: float, band: np.ndarray,
                         w: np.ndarray, rng, nperm: int = 200):
    """Per-head permutation null: permute the head's OWN residual ratio, controls for its
    own spikiness.  Returns (|c|, p_value, null_array)."""
    c, r = m_of(profile_band, T, band, w)
    base = profile_band / r
    nul = np.array([abs(m_of(base * rng.permutation(r), T, band, w)[0]) for _ in range(nperm)])
    return abs(c), float((nul >= abs(c)).mean()), nul
