"""MaxEnt strength scoring + matched-decoy selection for the competition cohort (R22-safe control).

A matched decoy is a REAL, same-type, non-competitor motif at matched target-distance and matched MaxEnt
strength — the control that makes `|M_comp|` vs `|M_decoy|` a competition-SPECIFICITY test rather than a
generic-perturbation test. Extracted so both the specificity pilot and the all-head screen share it.
"""
from __future__ import annotations

import math

from gfmcd.splicing.competition_seq import WindowedCompetition

DONOR_MOTIFS = ("GT", "GC")


def _ok(mat: str) -> bool:
    return set(mat) <= set("ACGT")


def don_score(seq: str, d: int, m5) -> float:
    """MaxEnt 5' (donor) score at donor position d (GT at [d,d+2)); 9-mer = 3 exon + GT + 4 intron."""
    from maxentpy import maxent
    s = seq[d - 3:d + 6]
    return maxent.score5(s, matrix=m5) if len(s) == 9 and _ok(s) else float("nan")


def acc_score(seq: str, a: int, m3) -> float:
    """MaxEnt 3' (acceptor) score at acceptor anchor a (AG at [a-2,a)); 23-mer = 20 intron + 3 exon."""
    from maxentpy import maxent
    s = seq[a - 20:a + 3]
    return maxent.score3(s, matrix=m3) if len(s) == 23 and _ok(s) else float("nan")


def find_matched_decoy(w: WindowedCompetition, m5, m3, *, dist_tol: int = 40, strength_caliper: float = 2.0):
    """A real same-type non-competitor motif at matched target-distance and matched MaxEnt strength.

    Returns (decoy_position | None, competitor_strength | None).
    """
    seq = w.seq
    comp_is_donor = w.competitor_is_donor
    comp_strength = don_score(seq, w.competitor_win, m5) if comp_is_donor else acc_score(seq, w.competitor_win, m3)
    if math.isnan(comp_strength):
        return None, None
    d_tc = abs(w.target_win - w.competitor_win)
    forbid = {w.target_win, w.partner_win, w.competitor_win}
    best, best_gap = None, strength_caliper
    N = len(seq)
    for q in range(24, N - 8):
        if any(abs(q - f) < 4 for f in forbid):
            continue
        if abs(abs(q - w.target_win) - d_tc) > dist_tol:
            continue
        if comp_is_donor:
            if seq[q:q + 2] not in DONOR_MOTIFS:
                continue
            st = don_score(seq, q, m5)
        else:
            if seq[q - 2:q] != "AG":
                continue
            st = acc_score(seq, q, m3)
        if math.isnan(st):
            continue
        gap = abs(st - comp_strength)
        if gap < best_gap:
            best, best_gap = q, gap
    return best, (comp_strength if best is not None else None)
