"""Acceptor-address transplantation: phase-matched decoy pairs, preference readouts.

This module is the reusable kernel for donor-graph experiment 3
(``preregistrations/PREREGISTRATION_ACCEPTOR_ADDRESS.md``).  It builds a partner-identity-controlled
candidate pair for each eligible example -- the model's real intron-mate acceptor ``a+``
versus a decoy AG ``a-`` matched on the AG motif, the donor->candidate distance, AND the
distance-mod-3 reading phase -- and provides pure tensor helpers to read L3H1's donor-query
preference for one acceptor window over another at the QK-logit and post-softmax levels.

Design rules inherited from the study (do not relax without a prereg amendment):

* The decoy differs from the partner ONLY in partner identity once matched (§0/§2 of the
  prereg).  ``phase_matched_decoy`` enforces the mod-3 match that ``circuit.matched_decoy_
  acceptor`` does not.
* Positions here are 0-based NUCLEOTIDE positions in the 1024nt window, following the
  ``CouplingExample`` convention: an acceptor AG sits at ``[pos-2, pos-1]`` and ``pos`` is
  the first exonic base (same convention as ``next_acceptor_pos``).  The token index is
  ``pos + adapter.prefix_tokens`` -- the runner adds the offset, the helpers here take token
  indices already offset.
* The analytic QK logit ``q . k / sqrt(d_head)`` is a candidate generator (QK level), never
  evidence on its own; the decisive test is the transplant (prereg §4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gfmcd.splicing.coupling import CouplingExample


# --------------------------------------------------------------------------------------
# Candidate pairs: the model's partner a+ vs a nucleotide/distance/PHASE-matched decoy a-.
# --------------------------------------------------------------------------------------
def phase_matched_decoy(x: CouplingExample, tol: int = 40) -> int | None:
    """A non-partner AG matched to ``a+`` on donor-distance AND distance-mod-3 (a GENERIC site).

    Mirrors ``circuit.matched_decoy_acceptor`` (motif = AG, distance within ``tol``) and ADDS the
    reading-phase constraint ``(a- - donor) % 3 == (a+ - donor) % 3``; returns the closest-distance
    candidate or ``None``.  This is the "partner vs generic matched site" decoy: an intronic AG is
    a much WEAKER 3' splice site than a real acceptor, so its motif is NOT matched (reported by
    ``match_audit`` as ``motif_gap_sd``).  The salience-controlled partner-IDENTITY comparison
    uses ``real_acceptor_decoy`` instead, where splice-site strength is matched by construction.
    """
    d, pt = x.donor_pos, x.next_acceptor_pos
    if pt is None:
        return None
    dist = pt - d
    phase = dist % 3
    cands = [
        p for p in x.other_ag
        if p > d + 20
        and abs((p - d) - dist) <= tol
        and abs(p - pt) > 12
        and (p - d) % 3 == phase
        and p + 1 < len(x.sequence)
        and p - 3 > 0
    ]
    return min(cands, key=lambda p: abs((p - d) - dist)) if cands else None


def real_acceptor_decoy(x: CouplingExample, tol: int = 60) -> int | None:
    """A non-partner REAL annotated acceptor at matched distance-mod-3 (strength-controlled).

    Uses ``x.real_acc`` (other real annotated acceptors in-window): because both the partner
    and this decoy are genuine 3' splice sites, splice-site STRENGTH is controlled by
    construction, so a preference here is the cleanest partner-IDENTITY test.  Matched on
    reading phase; distance is matched only within a looser ``tol`` (real acceptors are sparse)
    and reported.  Returns ``None`` when no phase-matched real non-partner acceptor exists.
    """
    d, pt = x.donor_pos, x.next_acceptor_pos
    if pt is None or not getattr(x, "real_acc", None):
        return None
    dist = pt - d
    phase = dist % 3
    cands = [
        p for p in x.real_acc
        if p > d + 20
        and abs(p - pt) > 12
        and (p - d) % 3 == phase
        and abs((p - d) - dist) <= tol
        and p + 1 < len(x.sequence)
        and p - 3 > 0
    ]
    return min(cands, key=lambda p: abs((p - d) - dist)) if cands else None


def acceptor_motif_score(seq: str, pos: int, upstream: int = 20) -> float:
    """Coarse 3' splice-site strength at an acceptor whose AG is ``[pos-2, pos-1]``.

    Half AG presence + half polypyrimidine fraction over ``[pos-2-upstream, pos-2]``.  A
    reported covariate for the matching audit, NOT a fitted PWM (the repo has no validated
    acceptor PWM for ``CouplingExample``; see the data-map risk note).
    """
    lo = max(0, pos - 2 - upstream)
    tract = seq[lo:pos - 2]
    py = (sum(c in "CT" for c in tract) / len(tract)) if tract else 0.0
    has_ag = 1.0 if seq[pos - 2:pos] == "AG" else 0.0
    return 0.5 * has_ag + 0.5 * py


def gc_content(seq: str, lo: int, hi: int) -> float:
    s = seq[max(0, lo):hi]
    return (sum(c in "GC" for c in s) / len(s)) if s else 0.0


@dataclass(frozen=True)
class AddressPair:
    """An eligible example annotated with its partner and a phase-matched decoy."""
    example: CouplingExample
    a_plus: int           # partner intron-mate acceptor (next_acceptor_pos), first exon base
    a_minus: int          # phase-matched decoy AG, same coordinate convention
    dist_plus: int        # a_plus - donor
    dist_minus: int       # a_minus - donor
    phase: int            # (a_plus - donor) % 3 == (a_minus - donor) % 3 by construction
    motif_plus: float
    motif_minus: float
    gc_plus: float
    gc_minus: float
    coding: bool          # example_phase is not None (internal exon is a clean CDS)
    a_real: int | None = None    # non-partner REAL acceptor decoy (strength-controlled), if any
    dist_real: int | None = None

    @property
    def gene_id(self) -> str:
        return self.example.gene_id

    @property
    def strand(self) -> str:
        return self.example.strand


def build_pairs(
    examples,
    *,
    tol: int = 40,
    real_tol: int = 60,
    cds_lookup: dict | None = None,
) -> list[AddressPair]:
    """Annotate each example that has a phase-matched decoy with its covariate pair.

    ``tol`` bounds the generic-AG decoy's donor-distance match; ``real_tol`` bounds the (sparser)
    real-acceptor decoy's — the two arms are independently parameterised and both tolerances are
    recorded in provenance.  ``cds_lookup`` (from ``phase.load_cds_frames``) supplies the
    coding-state covariate; when ``None`` every pair is marked non-coding.
    """
    from gfmcd.splicing.phase import example_phase

    pairs: list[AddressPair] = []
    for x in examples:
        am = phase_matched_decoy(x, tol=tol)
        if am is None:
            continue
        ap = x.next_acceptor_pos
        d = x.donor_pos
        ar = real_acceptor_decoy(x, tol=real_tol)
        coding = (cds_lookup is not None) and (example_phase(x, cds_lookup) is not None)
        pairs.append(AddressPair(
            example=x,
            a_plus=ap, a_minus=am,
            dist_plus=ap - d, dist_minus=am - d,
            phase=(ap - d) % 3,
            motif_plus=acceptor_motif_score(x.sequence, ap),
            motif_minus=acceptor_motif_score(x.sequence, am),
            gc_plus=gc_content(x.sequence, ap - 20, ap),
            gc_minus=gc_content(x.sequence, am - 20, am),
            coding=coding,
            a_real=ar, dist_real=(ar - d) if ar is not None else None,
        ))
    return pairs


def match_audit(pairs: list[AddressPair]) -> dict:
    """The G-Match correctness gate (prereg §2a): the decoy must differ from the partner
    only in partner identity.  Returns metrics and a PASS bool."""
    if not pairs:
        return {"PASS": False, "reason": "no pairs"}
    mp = np.array([p.motif_plus for p in pairs])
    mm = np.array([p.motif_minus for p in pairs])
    dp = np.array([p.dist_plus for p in pairs])
    dm = np.array([p.dist_minus for p in pairs])
    phase_mismatch = int(sum((p.a_plus - p.example.donor_pos) % 3
                             != (p.a_minus - p.example.donor_pos) % 3 for p in pairs))
    pooled_sd = float(np.std(np.concatenate([mp, mm]))) or 1.0
    motif_gap = float(abs(mp.mean() - mm.mean())) / pooled_sd
    # SYSTEMATIC distance bias (|Δ of means|) is what would bias the aggregate-mean preference by
    # attention distance-decay; it is GATED.  The per-pair SPREAD (mean|Δ|) is large and
    # unavoidable (the closest phase-matched intronic AG is typically ~20-30nt off), so it is
    # REPORTED, and the identity claim is adjudicated on the real-acceptor arm with distance
    # stratification (prereg §7.2), not on tight per-pair matching that AG sparsity cannot provide.
    dist_bias = float(abs(dp.mean() - dm.mean()))
    dist_spread = float(np.mean(np.abs(dp - dm)))
    coding_frac = float(np.mean([p.coding for p in pairs]))

    # Real-acceptor arm: report the SIGNED distance gap (dist_real - dist_plus).  Non-partner real
    # acceptors are generically FARTHER from the donor than the intron-mate partner (the nearest
    # downstream acceptor), so a positive S_A_real could be attention distance-decay rather than
    # partner identity.  This distribution is what adjudicates that rival (prereg §7.2).
    real = [(p.dist_real - p.dist_plus) for p in pairs if p.a_real is not None]
    if real:
        rs = np.array(real, dtype=float)
        real_report = {"n": len(real), "signed_gap_mean": float(rs.mean()),
                       "signed_gap_median": float(np.median(rs)),
                       "abs_gap_mean": float(np.abs(rs).mean()),
                       "gap_min": float(rs.min()), "gap_max": float(rs.max())}
    else:
        real_report = {"n": 0}

    checks = {
        "motif_gap_sd": motif_gap,               # reported, NOT gated (intronic AG != real 3'SS)
        "dist_bias_nt": dist_bias,               # |mean(a+)-mean(a-)| systematic bias, GATED
        "dist_spread_nt": dist_spread,           # per-pair mean|Δ|, reported (AG sparsity)
        "phase_mismatch_rate": phase_mismatch,   # gated (exact 0)
        "coding_frac": coding_frac,              # shared by construction; reported
        "noncoding_frac": 1.0 - coding_frac,
        "real_acceptor_distance": real_report,   # the salience-arm distance confound, reported
    }
    passed = (dist_bias <= 8.0 and phase_mismatch == 0)
    return {"PASS": bool(passed), **checks}


# --------------------------------------------------------------------------------------
# Pure tensor readouts.  These take TOKEN indices (nucleotide pos + prefix_tokens) already.
# Kept dependency-light and side-effect-free so they are unit-testable without a model.
# --------------------------------------------------------------------------------------
def _head(x: torch.Tensor, head: int, d_head: int) -> torch.Tensor:
    """Contiguous head slice ``[..., head*d_head:(head+1)*d_head]`` of a projection [.,.,H*Dh]."""
    return x[..., head * d_head:(head + 1) * d_head]


def qk_logit_to_window(
    query: torch.Tensor, key: torch.Tensor, head: int, d_head: int,
    q_tok: torch.Tensor, win: torch.Tensor,
) -> torch.Tensor:
    """Mean pre-softmax attention logit from one query token to a key window, for one head.

    ``query``/``key``: projection outputs ``[B, S, H*Dh]``.  ``q_tok``: ``[B]`` donor token
    index.  ``win``: ``[B, K]`` key token indices.  Returns ``[B]`` =
    ``mean_{j in win} (q_{q_tok} . k_j) / sqrt(d_head)`` -- the exact scaling BERT uses
    (``query.size(-1) ** -0.5``, ``circuit._rolled_attention_forward``).
    """
    B = query.shape[0]
    b = torch.arange(B, device=query.device)
    q = _head(query[b, q_tok], head, d_head)              # [B, Dh]
    k = _head(key, head, d_head)                          # [B, S, Dh]
    kk = k[b[:, None], win]                               # [B, K, Dh]
    logits = (kk * q[:, None, :]).sum(-1) / (d_head ** 0.5)
    return logits.mean(-1)                                # [B]


def attn_to_window(
    attn_head: torch.Tensor, q_tok: torch.Tensor, win: torch.Tensor,
) -> torch.Tensor:
    """Post-softmax attention mass from one query token to a key window, for one head.

    ``attn_head``: ``[B, S, S]`` (already selected layer+head, e.g. ``attentions[3][:, 1]``).
    Returns ``[B]`` = ``sum_{j in win} A[q_tok, j]``.
    """
    B = attn_head.shape[0]
    b = torch.arange(B, device=attn_head.device)
    row = attn_head[b, q_tok]                             # [B, S]
    return row[b[:, None], win].sum(-1)                  # [B]


def windows(positions, off: int, device, offsets=(-3, -2, -1, 0)) -> torch.Tensor:
    """[B, K] token indices for the ACC window around each nucleotide ``pos`` (pos+off+offset)."""
    return torch.tensor([[p + off + k for k in offsets] for p in positions], device=device)
