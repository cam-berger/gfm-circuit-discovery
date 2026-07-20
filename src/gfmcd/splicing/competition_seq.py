"""Sequence layer for the intron-mate competition cohort: build a fixed-length real-genome window
holding the target, its shared partner, AND a competitor, with each site's in-window position + motif.

Kept separate from ``competition.py`` (which is annotation-only) because this touches the genome.
Geometry mirrors ``build_coupling_dataset``: constant window length (no length channel), a
deterministic acceptor-style offset drawn from a range independent of the intron length, real-genome
filler, strand-correct transcript orientation, never PAD/[MASK].
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from gfmcd.splicing.competition import CompetitionConfig, CompetitionExample
from gfmcd.splicing.dataset import GenomeAccessor, reverse_complement

DONOR_MOTIFS = ("GT", "GC")
ACCEPTOR_MOTIFS = ("AG",)


def _stable_offset(example_id: str, lo: int, hi: int, seed: int = 0) -> int:
    """Deterministic per-example offset in [lo, hi] (reproducible, not constant; seeded on the id)."""
    h = hashlib.sha256(f"{seed}:{example_id}".encode()).digest()
    return lo + (int.from_bytes(h[:8], "big") % (hi - lo + 1))


@dataclass(frozen=True)
class WindowedCompetition:
    example_id: str
    gene_id: str
    chrom: str
    strand: str
    split: str
    site_type: str
    seq: str
    target_win: int
    partner_win: int
    competitor_win: int
    target_is_donor: bool
    competitor_is_donor: bool
    partner_is_donor: bool
    cluster_size: int
    n_usable_competitors: int
    target_partner_distance: int
    target_competitor_distance: int


def _motif_ok(seq: str, pos: int, is_donor: bool) -> bool:
    if is_donor:
        return 0 <= pos <= len(seq) - 2 and seq[pos:pos + 2] in DONOR_MOTIFS
    return 2 <= pos <= len(seq) and seq[pos - 2:pos] in ACCEPTOR_MOTIFS


def build_window(ex: CompetitionExample, genome: GenomeAccessor, cfg: CompetitionConfig,
                 *, offset_seed: int = 0) -> WindowedCompetition | None:
    """Construct the fixed-length transcript-oriented window; None if geometry/motifs fail.

    site_type semantics:
      alt5ss_donor    -> target & competitor are DONORS, shared partner is an ACCEPTOR
      alt3ss_acceptor -> target & competitor are ACCEPTORS, shared partner is a DONOR
    """
    if not ex.usable:
        return None
    target_is_donor = ex.site_type == "alt5ss_donor"
    competitor_is_donor = target_is_donor
    partner_is_donor = not target_is_donor

    g_min = min(ex.target, ex.partner, ex.competitor)
    g_max = max(ex.target, ex.partner, ex.competitor)
    span = g_max - g_min
    N = cfg.window_length
    hi = min(cfg.acc_max, N - cfg.min_context - span)
    lo = cfg.acc_min
    if hi < lo:
        return None
    o = _stable_offset(ex.example_id, lo, hi, offset_seed)

    if ex.strand == "+":
        ws = g_min - o
        we = ws + N
    else:
        we = g_max + 1 + o
        ws = we - N
    if ws < 0 or we > genome.length(ex.chrom):
        return None
    seq = genome.fetch(ex.chrom, ws, we)
    if ex.strand == "-":
        seq = reverse_complement(seq)
    seq = seq.upper().replace("U", "T")
    if len(seq) != N or set(seq) - set("ACGT"):
        return None

    def tw(g: int) -> int:
        return (g - ws) if ex.strand == "+" else (we - 1 - g)

    t_win, p_win, c_win = tw(ex.target), tw(ex.partner), tw(ex.competitor)
    if not (4 <= t_win < N - 4 and 4 <= p_win < N - 4 and 4 <= c_win < N - 4):
        return None
    if not (_motif_ok(seq, t_win, target_is_donor)
            and _motif_ok(seq, c_win, competitor_is_donor)
            and _motif_ok(seq, p_win, partner_is_donor)):
        return None
    # destroy windows must not overlap the other two sites' motif footprints
    if abs(c_win - t_win) < 3 or abs(c_win - p_win) < 3:
        return None

    return WindowedCompetition(
        example_id=ex.example_id, gene_id=ex.gene_id, chrom=ex.chrom, strand=ex.strand,
        split=ex.split, site_type=ex.site_type, seq=seq,
        target_win=t_win, partner_win=p_win, competitor_win=c_win,
        target_is_donor=target_is_donor, competitor_is_donor=competitor_is_donor,
        partner_is_donor=partner_is_donor,
        cluster_size=ex.cluster_size, n_usable_competitors=ex.n_usable_competitors,
        target_partner_distance=ex.target_partner_distance,
        target_competitor_distance=ex.target_competitor_distance,
    )
