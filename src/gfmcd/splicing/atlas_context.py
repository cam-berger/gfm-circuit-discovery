"""Annotation-only multi-label packet builder for the SpliceBERT Head-Context Atlas.

The competition study implemented one context.  This module implements the broader
atlas substrate without reading genome sequence or model outcomes.  It treats every
unique annotated intron junction as a graph edge and emits two possible MLM readout
packets per edge (donor target and acceptor target) when the pair fits in 1024 nt.

Labels intentionally overlap.  For example, a cassette-exon edge can also belong to
an acceptor cluster, sit in a packet containing independent pairs, and be part of a
complex local graph.  Annotation tags and model-derived quantities remain separate.

Definitions
-----------
``junction_one_to_one``
    Target edge's donor has one annotated acceptor and its acceptor has one donor.
``donor_cluster``
    Two or more annotated donors share the target edge's acceptor (alternative 5'SS).
``acceptor_cluster``
    Two or more annotated acceptors share the target edge's donor (alternative 3'SS).
``single_pair``
    Exactly one fully visible annotated junction occurs in the 1024-nt packet.
``multiple_independent_pairs``
    The packet contains at least two visible junctions with distinct donor and
    acceptor endpoints; unlike a cluster, that pair of edges shares no site.
``cassette_exon``
    An included path d_up->a_exon and d_exon->a_down coexists with the skipping edge
    d_up->a_down in the annotation.
``mutually_exclusive_exon``
    Two or more distinct internal exons connect the same upstream donor and downstream
    acceptor through separate included paths.
``complex_event``
    The target belongs to a cassette/mutually-exclusive pattern or its local union of
    incident edges contains at least three edges.

All iteration is sorted.  Window placement uses a SHA256-derived offset independent of
model outcomes, transcript support, and target role.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Iterable, Iterator

from gfmcd.splicing.competition import split_for
from gfmcd.splicing.gtf import TranscriptRecord
from gfmcd.splicing.partner_data import _acceptor_anchor, _donor_anchor, _to_window

JunctionKey = tuple[str, str, str, int, int]  # gene, chrom, strand, donor, acceptor


@dataclass(frozen=True)
class AtlasConfig:
    window_length: int = 1024
    min_offset: int = 100
    max_offset: int = 300
    min_context: int = 100
    seed: int = 0

    @property
    def max_span(self) -> int:
        return self.window_length - self.min_offset - self.min_context

    def validate(self) -> None:
        if not (4 <= self.min_offset <= self.max_offset):
            raise ValueError("need 4 <= min_offset <= max_offset")
        if self.min_offset == self.max_offset:
            raise ValueError("constant offset would create a positional shortcut")
        if self.window_length <= self.max_offset + self.min_context:
            raise ValueError("window is too short for the requested offset/context")


@dataclass
class AtlasIndex:
    junction_support: dict[JunctionKey, int] = field(default_factory=lambda: defaultdict(int))
    junction_transcripts: dict[JunctionKey, set[str]] = field(default_factory=lambda: defaultdict(set))
    donors_by_acceptor: dict[tuple[str, str, str, int], set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    acceptors_by_donor: dict[tuple[str, str, str, int], set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )
    donor_support: dict[tuple[str, int], set[str]] = field(default_factory=lambda: defaultdict(set))
    acceptor_support: dict[tuple[str, int], set[str]] = field(default_factory=lambda: defaultdict(set))
    gene_junctions: dict[tuple[str, str, str], set[JunctionKey]] = field(
        default_factory=lambda: defaultdict(set)
    )
    mane_junctions: set[JunctionKey] = field(default_factory=set)
    cassette_edges: set[JunctionKey] = field(default_factory=set)
    mutually_exclusive_edges: set[JunctionKey] = field(default_factory=set)


def _edge(gene: str, chrom: str, strand: str, donor: int, acceptor: int) -> JunctionKey:
    return gene, chrom, strand, donor, acceptor


def build_atlas_index(transcripts: Iterable[TranscriptRecord]) -> AtlasIndex:
    """Build the unique splice graph plus cassette/MXE event membership."""
    idx = AtlasIndex()
    internal_paths: list[tuple[str, str, str, int, int, int, int]] = []
    for tx in transcripts:
        exons = tx.ordered_exons()
        if len(exons) < 2:
            continue
        mane = "MANE_Select" in tx.tags
        tx_edges: list[JunctionKey] = []
        for upstream, downstream in zip(exons[:-1], exons[1:]):
            donor = _donor_anchor(upstream)
            acceptor = _acceptor_anchor(downstream)
            key = _edge(tx.gene_id, tx.chrom, tx.strand, donor, acceptor)
            tx_edges.append(key)
            idx.junction_support[key] += 1
            idx.junction_transcripts[key].add(tx.transcript_id)
            idx.donors_by_acceptor[(tx.gene_id, tx.chrom, tx.strand, acceptor)].add(donor)
            idx.acceptors_by_donor[(tx.gene_id, tx.chrom, tx.strand, donor)].add(acceptor)
            idx.donor_support[(tx.gene_id, donor)].add(tx.transcript_id)
            idx.acceptor_support[(tx.gene_id, acceptor)].add(tx.transcript_id)
            idx.gene_junctions[(tx.gene_id, tx.chrom, tx.strand)].add(key)
            if mane:
                idx.mane_junctions.add(key)
        for pos in range(1, len(exons) - 1):
            upstream, exon, downstream = exons[pos - 1], exons[pos], exons[pos + 1]
            internal_paths.append((
                tx.gene_id,
                tx.chrom,
                tx.strand,
                _donor_anchor(upstream),
                _acceptor_anchor(exon),
                _donor_anchor(exon),
                _acceptor_anchor(downstream),
            ))

    # An internal exon is cassette-supported when its two inclusion edges and the
    # direct skipping edge all appear in the annotation.
    flanked_exons: dict[tuple[str, str, str, int, int], set[tuple[int, int]]] = defaultdict(set)
    for gene, chrom, strand, d_up, a_exon, d_exon, a_down in sorted(set(internal_paths)):
        left = _edge(gene, chrom, strand, d_up, a_exon)
        right = _edge(gene, chrom, strand, d_exon, a_down)
        skip = _edge(gene, chrom, strand, d_up, a_down)
        if skip in idx.junction_support:
            idx.cassette_edges.update((left, right, skip))
        flanked_exons[(gene, chrom, strand, d_up, a_down)].add((a_exon, d_exon))

    # Multiple distinct internal exons connecting the same outer sites constitute an
    # annotation-supported mutually-exclusive-exon candidate. Expression exclusivity
    # is not claimed.
    for (gene, chrom, strand, d_up, a_down), middle_exons in sorted(flanked_exons.items()):
        if len(middle_exons) < 2:
            continue
        for a_exon, d_exon in sorted(middle_exons):
            idx.mutually_exclusive_edges.add(_edge(gene, chrom, strand, d_up, a_exon))
            idx.mutually_exclusive_edges.add(_edge(gene, chrom, strand, d_exon, a_down))
    return idx


@dataclass(frozen=True)
class AtlasPacket:
    packet_id: str
    junction_id: str
    gene_id: str
    chrom: str
    strand: str
    split: str
    target_role: str  # donor | acceptor
    donor: int
    acceptor: int
    window_start: int
    window_end: int
    donor_win: int
    acceptor_win: int
    target_anchor_win: int
    masked_base_win: int
    partner_anchor_win: int
    target_partner_distance: int
    n_annotated_transcripts_for_junction: int
    n_annotated_transcripts_for_target_site: int
    donor_degree: int
    acceptor_degree: int
    local_edge_count: int
    visible_donor_count: int
    visible_acceptor_count: int
    visible_pair_count: int
    target_is_mane: bool
    tags: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def _stable_offset(junction_id: str, lo: int, hi: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{junction_id}".encode()).digest()
    return lo + int.from_bytes(digest[:8], "big") % (hi - lo + 1)


def _junction_id(key: JunctionKey) -> str:
    gene, chrom, strand, donor, acceptor = key
    return f"{chrom}:{strand}:d{donor}:a{acceptor}:{gene}"


def _window_for(key: JunctionKey, cfg: AtlasConfig) -> tuple[int, int] | None:
    _gene, _chrom, strand, donor, acceptor = key
    lo_site, hi_site = min(donor, acceptor), max(donor, acceptor)
    span = hi_site - lo_site
    hi_offset = min(cfg.max_offset, cfg.window_length - cfg.min_context - span)
    if hi_offset < cfg.min_offset:
        return None
    offset = _stable_offset(_junction_id(key), cfg.min_offset, hi_offset, cfg.seed)
    if strand == "+":
        start = lo_site - offset
        end = start + cfg.window_length
    else:
        end = hi_site + 1 + offset
        start = end - cfg.window_length
    if start < 0:
        return None
    return start, end


def _independent_pair_exists(edges: list[JunctionKey]) -> bool:
    for i, left in enumerate(edges):
        for right in edges[i + 1:]:
            if left[3] != right[3] and left[4] != right[4]:
                return True
    return False


def _packet_tags(key: JunctionKey, idx: AtlasIndex, visible: list[JunctionKey], role: str) -> tuple[str, ...]:
    gene, chrom, strand, donor, acceptor = key
    donor_degree = len(idx.acceptors_by_donor[(gene, chrom, strand, donor)])
    acceptor_degree = len(idx.donors_by_acceptor[(gene, chrom, strand, acceptor)])
    incident = {
        edge for edge in idx.gene_junctions[(gene, chrom, strand)]
        if edge[3] == donor or edge[4] == acceptor
    }
    tags = {"annotated_site", "annotated_junction", f"target_{role}"}
    if donor_degree == 1 and acceptor_degree == 1:
        tags.add("junction_one_to_one")
    if acceptor_degree >= 2:
        tags.update(("donor_cluster", "alternative_donor"))
    if donor_degree >= 2:
        tags.update(("acceptor_cluster", "alternative_acceptor"))
    if key in idx.cassette_edges:
        tags.add("cassette_exon")
    if key in idx.mutually_exclusive_edges:
        tags.add("mutually_exclusive_exon")
    if len(incident) >= 3 or key in idx.cassette_edges or key in idx.mutually_exclusive_edges:
        tags.add("complex_event")
    if len(visible) == 1:
        tags.add("single_pair")
    else:
        tags.add("multi_site")
    if _independent_pair_exists(visible):
        tags.add("multiple_independent_pairs")
    if donor_degree == 1 and acceptor_degree == 1 and key not in idx.cassette_edges \
            and key not in idx.mutually_exclusive_edges:
        tags.add("isolated_pair_control_candidate")
    return tuple(sorted(tags))


def enumerate_atlas_packets(idx: AtlasIndex, cfg: AtlasConfig) -> Iterator[AtlasPacket]:
    """Yield every constructible donor/acceptor target packet in deterministic order."""
    cfg.validate()
    for key in sorted(idx.junction_support):
        window = _window_for(key, cfg)
        if window is None:
            continue
        gene, chrom, strand, donor, acceptor = key
        start, end = window
        donor_win = _to_window(donor, strand, start, end)
        acceptor_win = _to_window(acceptor, strand, start, end)
        if not (4 <= donor_win < cfg.window_length - 4 and 4 <= acceptor_win < cfg.window_length - 4):
            continue
        visible = sorted(
            edge for edge in idx.gene_junctions[(gene, chrom, strand)]
            if 4 <= _to_window(edge[3], strand, start, end) < cfg.window_length - 4
            and 4 <= _to_window(edge[4], strand, start, end) < cfg.window_length - 4
        )
        if key not in visible:
            raise RuntimeError("target junction missing from its own packet")
        donor_degree = len(idx.acceptors_by_donor[(gene, chrom, strand, donor)])
        acceptor_degree = len(idx.donors_by_acceptor[(gene, chrom, strand, acceptor)])
        local_edges = {
            edge for edge in idx.gene_junctions[(gene, chrom, strand)]
            if edge[3] == donor or edge[4] == acceptor
        }
        for role in ("donor", "acceptor"):
            target_anchor = donor_win if role == "donor" else acceptor_win
            masked_base = target_anchor if role == "donor" else target_anchor - 1
            partner_anchor = acceptor_win if role == "donor" else donor_win
            junction_id = _junction_id(key)
            packet_id = f"{junction_id}:target_{role}"
            target_support = (
                len(idx.donor_support[(gene, donor)])
                if role == "donor" else len(idx.acceptor_support[(gene, acceptor)])
            )
            yield AtlasPacket(
                packet_id=packet_id,
                junction_id=junction_id,
                gene_id=gene,
                chrom=chrom,
                strand=strand,
                split=split_for(chrom),
                target_role=role,
                donor=donor,
                acceptor=acceptor,
                window_start=start,
                window_end=end,
                donor_win=donor_win,
                acceptor_win=acceptor_win,
                target_anchor_win=target_anchor,
                masked_base_win=masked_base,
                partner_anchor_win=partner_anchor,
                target_partner_distance=abs(donor - acceptor),
                n_annotated_transcripts_for_junction=idx.junction_support[key],
                n_annotated_transcripts_for_target_site=target_support,
                donor_degree=donor_degree,
                acceptor_degree=acceptor_degree,
                local_edge_count=len(local_edges),
                visible_donor_count=len({edge[3] for edge in visible}),
                visible_acceptor_count=len({edge[4] for edge in visible}),
                visible_pair_count=len(visible),
                target_is_mane=key in idx.mane_junctions,
                tags=_packet_tags(key, idx, visible, role),
            )


def match_stratum(packet: AtlasPacket) -> tuple:
    """Coarse annotation-only common-support stratum.

    Sequence composition, MaxEnt strength, and masked-MLM health are intentionally
    absent; those are required in the subsequent enrichment/matching stage.
    """
    distance = packet.target_partner_distance
    if distance < 100:
        distance_bin = "000_099"
    elif distance < 200:
        distance_bin = "100_199"
    elif distance < 400:
        distance_bin = "200_399"
    else:
        distance_bin = "400_824"
    support = packet.n_annotated_transcripts_for_target_site
    support_bin = "1" if support == 1 else "2" if support == 2 else "3_4" if support <= 4 else "5_plus"
    pairs = packet.visible_pair_count
    pair_bin = "1" if pairs == 1 else "2" if pairs == 2 else "3_plus"
    anchor_bin = f"{(packet.target_anchor_win // 100) * 100:03d}"
    return packet.split, packet.target_role, distance_bin, support_bin, pair_bin, anchor_bin


def atlas_analysis_split(gene_id: str, chromosome_split: str, *, discovery_percent: int = 80) -> str:
    """Outcome-independent gene-hash split within the untouched chromosome-train pool.

    Historical chr11/16/19 validation examples were touched by the competition screen,
    so the wider atlas creates a fresh internal validation split from train chromosomes.
    Every packet from a gene receives the same assignment.
    """
    if chromosome_split == "test":
        return "sealed_test"
    if chromosome_split == "val":
        return "touched_val_excluded"
    if not 1 <= discovery_percent <= 99:
        raise ValueError("discovery_percent must be in [1, 99]")
    bucket = int.from_bytes(hashlib.sha256(gene_id.encode()).digest()[:8], "big") % 100
    return "atlas_discovery" if bucket < discovery_percent else "atlas_internal_validation"
