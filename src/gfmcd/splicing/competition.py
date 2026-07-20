"""Intron-mate splice-competition cohort for the Head-Context Atlas (estimand C).

WHY THIS MODULE EXISTS (read before touching the definitions)
-------------------------------------------------------------
The Head-Context Atlas asks which components are recruited because a splice site sits in a
COMPETITIVE choice set. The design red-team (2026-07-18) killed the first draft's cohort because
it reused ``partner_data._build_gene_index.attested``, which maps an acceptor (exon START) to the
donors at the SAME exon's END -- i.e. EXON-MATE alternative-5' sites -- while the coupling behavior
``donor_margin`` and the whole C1-C35 circuit operate on the INTRON-MATE partner (this exon's donor
to the NEXT exon's acceptor, across the intron). Building the cohort on ``attested`` would validate
one behavior and license claims about another.

This module builds the INTRON-MATE competition structure directly from transcript order:

  * a JUNCTION (intron) links consecutive transcript-order exons (e_k, e_{k+1}):
      donor    = _donor_anchor(e_k)        (first intron base after e_k, transcript orientation)
      acceptor = _acceptor_anchor(e_{k+1}) (first base of e_{k+1})
  * alt-5'SS DONOR competition  : two+ distinct donors splice to the SAME downstream acceptor
  * alt-3'SS ACCEPTOR competition: two+ distinct acceptors receive the SAME upstream donor

Competitors that share a partner necessarily come from DIFFERENT transcripts (one partner has one
mate per transcript), so they are mutually exclusive by construction.

This module is GEOMETRY/ANNOTATION ONLY -- no genome sequence, no model. It reports how many usable
competition examples exist per arm/split so the projected MDE gate (STAGE 0) can decide whether a
paid confirmatory run is even licensable BEFORE any GPU spend. Sequence-level tags (MaxEnt strength,
GC, canonicality, CDS phase) are added by a separate enrichment pass, not here.

Determinism: every set is iterated via ``sorted(...)``; no PYTHONHASHSEED-dependent ordering.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator

from gfmcd.splicing.gtf import TranscriptRecord
from gfmcd.splicing.partner_data import _acceptor_anchor, _donor_anchor

# Discovery vs sealed splits (identical convention to CouplingConfig / PartnerConfig).
VAL_CHROMOSOMES = ("chr11", "chr16", "chr19")
TEST_CHROMOSOMES = ("chr8", "chr9")  # SEALED — never opened before the confirmatory run


def _norm_chrom(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def split_for(chrom: str) -> str:
    c = _norm_chrom(chrom)
    if c in VAL_CHROMOSOMES:
        return "val"
    if c in TEST_CHROMOSOMES:
        return "test"  # sealed
    return "train"  # discovery pool = autosomes/other ∉ {chr8, chr9, chr11, chr16, chr19}


@dataclass(frozen=True)
class Junction:
    """One annotated intron: a (donor, acceptor) pair in transcript orientation."""

    gene_id: str
    chrom: str
    strand: str
    donor: int      # genomic coord of first intron base after the upstream exon (transcript-oriented)
    acceptor: int   # genomic coord of first base of the downstream exon (transcript-oriented)
    transcript_id: str
    transcript_type: str
    is_mane: bool


def iter_junctions(transcripts: Iterable[TranscriptRecord]) -> Iterator[Junction]:
    """Every intron-mate junction from consecutive transcript-order exon pairs."""
    for tx in transcripts:
        exons = tx.ordered_exons()  # strand-correct TRANSCRIPT order
        if len(exons) < 2:
            continue
        is_mane = ("MANE_Select" in tx.tags)
        for k in range(len(exons) - 1):
            up, dn = exons[k], exons[k + 1]
            yield Junction(
                gene_id=tx.gene_id,
                chrom=tx.chrom,
                strand=tx.strand,
                donor=_donor_anchor(up),        # 5'SS of the intron
                acceptor=_acceptor_anchor(dn),  # 3'SS of the intron
                transcript_id=tx.transcript_id,
                transcript_type=tx.transcript_type,
                is_mane=is_mane,
            )


@dataclass
class CompetitionIndex:
    """Intron-mate competition structure keyed by the SHARED partner site.

    donors_by_acceptor[(gene,chrom,strand,acceptor)] -> {donor: n_transcripts}
        alt-5'SS: donors competing for one downstream acceptor
    acceptors_by_donor[(gene,chrom,strand,donor)]    -> {acceptor: n_transcripts}
        alt-3'SS: acceptors competing for one upstream donor
    junction_tx[(gene,donor,acceptor)] -> n_transcripts   (n_annotated_transcripts_for_junction)
    site_tx_don[(gene,donor)] / site_tx_acc[(gene,acceptor)] -> n_transcripts (per-site support)
    """

    donors_by_acceptor: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    acceptors_by_donor: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    junction_tx: dict = field(default_factory=lambda: defaultdict(int))
    site_tx_don: dict = field(default_factory=lambda: defaultdict(int))
    site_tx_acc: dict = field(default_factory=lambda: defaultdict(int))
    mane_junctions: set = field(default_factory=set)


def build_competition_index(transcripts: Iterable[TranscriptRecord]) -> CompetitionIndex:
    idx = CompetitionIndex()
    for j in iter_junctions(transcripts):
        akey = (j.gene_id, j.chrom, j.strand, j.acceptor)
        dkey = (j.gene_id, j.chrom, j.strand, j.donor)
        idx.donors_by_acceptor[akey][j.donor] += 1
        idx.acceptors_by_donor[dkey][j.acceptor] += 1
        idx.junction_tx[(j.gene_id, j.donor, j.acceptor)] += 1
        idx.site_tx_don[(j.gene_id, j.donor)] += 1
        idx.site_tx_acc[(j.gene_id, j.acceptor)] += 1
        if j.is_mane:
            idx.mane_junctions.add((j.gene_id, j.donor, j.acceptor))
    return idx


@dataclass(frozen=True)
class CompetitionConfig:
    window_length: int = 1024
    # geometry: reuse the CouplingConfig anchoring so the atlas cohort is comparable.
    acc_min: int = 100
    acc_max: int = 300
    min_context: int = 100
    # a target-anchored window must hold target, partner AND competitor with flanks:
    #   span(target,partner,competitor) <= window_length - acc_min - min_context
    # (conservative; the far site still leaves >= min_context of real filler).
    min_separation: int = 30      # target<->competitor min bp; excludes NAGNAG (~3nt) / tandem wobble
    max_separation: int | None = None
    require_mutually_exclusive: bool = True  # competitors from different transcripts (true by construction)

    @property
    def max_span(self) -> int:
        return self.window_length - self.acc_min - self.min_context


@dataclass(frozen=True)
class CompetitionExample:
    """One (target, shared partner, competitor) triple — annotation only."""

    example_id: str
    gene_id: str
    chrom: str
    strand: str
    split: str
    site_type: str            # 'alt5ss_donor' | 'alt3ss_acceptor'
    target: int               # genomic anchor of the target site
    partner: int              # genomic anchor of the SHARED partner (intron-mate of both)
    competitor: int           # genomic anchor of the PRIMARY competing same-type site (nearest usable)
    cluster_size: int         # number of competing same-type sites for this shared partner
    n_competitors: int        # competitors other than the target (cluster_size - 1)
    n_usable_competitors: int # competitors that are in-window AND >= min_separation from the target
    target_partner_distance: int
    competitor_partner_distance: int
    target_competitor_distance: int
    nearest_competitor_distance: int   # to nearest competitor of ANY separation (tandem-aware)
    span: int
    in_window: bool
    min_sep_ok: bool
    is_tandem: bool           # nearest competitor < 30 nt
    n_tx_target_junction: int
    n_tx_competitor_junction: int
    n_tx_target_site: int
    target_is_mane: bool

    @property
    def usable(self) -> bool:
        """Geometrically constructible with a clean destroy: primary competitor in-window, separated."""
        return self.in_window and self.min_sep_ok and self.n_usable_competitors >= 1


def _emit_group(gene, chrom, strand, partner, competitors_tx, site_type, cfg):
    """One example per TARGET (not per ordered pair — avoids O(k^2) blow-up and matches the
    estimand: one competitor per target example, K draws vary the choice). The PRIMARY competitor
    is the nearest usable one (in-window AND >= min_separation); ties broken by lower coordinate.
    If none is usable, the nearest competitor of any separation is recorded and usable=False."""
    comps = sorted(competitors_tx)
    cluster_size = len(comps)
    if cluster_size < 2:
        return
    for target in comps:
        others = [c for c in comps if c != target]
        if not others:
            continue
        tp = abs(target - partner)

        def _geom(c):
            tc = abs(target - c)
            span = max(target, partner, c) - min(target, partner, c)
            in_win = span <= cfg.max_span
            sep_ok = tc >= cfg.min_separation and (cfg.max_separation is None or tc <= cfg.max_separation)
            return tc, span, in_win, sep_ok

        usable = [c for c in others if all(_geom(c)[2:4])]
        n_usable = len(usable)
        nearest_any = min(others, key=lambda c: (abs(target - c), c))
        # primary = nearest usable if any, else nearest overall (records why it fails)
        primary = min(usable, key=lambda c: (abs(target - c), c)) if usable else nearest_any
        tc, span, in_win, sep_ok = _geom(primary)
        nearest_dist = abs(target - nearest_any)
        eid = f"{chrom}:{strand}:{site_type}:t{target}:p{partner}:{gene}"
        yield CompetitionExample(
            example_id=eid, gene_id=gene, chrom=chrom, strand=strand,
            split=split_for(chrom), site_type=site_type,
            target=target, partner=partner, competitor=primary,
            cluster_size=cluster_size, n_competitors=len(others), n_usable_competitors=n_usable,
            target_partner_distance=tp, competitor_partner_distance=abs(primary - partner),
            target_competitor_distance=tc, nearest_competitor_distance=nearest_dist, span=span,
            in_window=in_win, min_sep_ok=sep_ok, is_tandem=nearest_dist < 30,
            n_tx_target_junction=0, n_tx_competitor_junction=0, n_tx_target_site=0,
            target_is_mane=False,
        )


def enumerate_competition_examples(
    idx: CompetitionIndex, cfg: CompetitionConfig
) -> list[CompetitionExample]:
    """All (target, partner, competitor) triples for both competition types, annotation only."""
    out: list[CompetitionExample] = []
    # alt-5'SS donor competition: donors sharing a downstream acceptor
    for akey in sorted(idx.donors_by_acceptor):
        gene, chrom, strand, acceptor = akey
        donors_tx = idx.donors_by_acceptor[akey]
        for ex in _emit_group(gene, chrom, strand, acceptor, donors_tx, "alt5ss_donor", cfg):
            out.append(_attach_support(ex, idx))
    # alt-3'SS acceptor competition: acceptors sharing an upstream donor
    for dkey in sorted(idx.acceptors_by_donor):
        gene, chrom, strand, donor = dkey
        acc_tx = idx.acceptors_by_donor[dkey]
        for ex in _emit_group(gene, chrom, strand, donor, acc_tx, "alt3ss_acceptor", cfg):
            out.append(_attach_support(ex, idx))
    return out


def _attach_support(ex: CompetitionExample, idx: CompetitionIndex) -> CompetitionExample:
    """Fill transcript-support counts + MANE flag (n_annotated_transcripts_for_*)."""
    if ex.site_type == "alt5ss_donor":
        n_t_junc = idx.junction_tx.get((ex.gene_id, ex.target, ex.partner), 0)
        n_c_junc = idx.junction_tx.get((ex.gene_id, ex.competitor, ex.partner), 0)
        n_t_site = idx.site_tx_don.get((ex.gene_id, ex.target), 0)
        is_mane = (ex.gene_id, ex.target, ex.partner) in idx.mane_junctions
    else:
        n_t_junc = idx.junction_tx.get((ex.gene_id, ex.partner, ex.target), 0)
        n_c_junc = idx.junction_tx.get((ex.gene_id, ex.partner, ex.competitor), 0)
        n_t_site = idx.site_tx_acc.get((ex.gene_id, ex.target), 0)
        is_mane = (ex.gene_id, ex.partner, ex.target) in idx.mane_junctions
    return CompetitionExample(
        **{**ex.__dict__,
           "n_tx_target_junction": n_t_junc,
           "n_tx_competitor_junction": n_c_junc,
           "n_tx_target_site": n_t_site,
           "target_is_mane": is_mane}
    )
