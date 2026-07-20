"""Fast annotation-only tests for the multi-context atlas substrate."""
from __future__ import annotations

from gfmcd.splicing.atlas_context import (
    AtlasConfig,
    atlas_analysis_split,
    build_atlas_index,
    enumerate_atlas_packets,
    match_stratum,
)
from gfmcd.splicing.gtf import ExonRecord, TranscriptRecord


def make_tx(tid: str, gene: str, ranges, *, strand="+", tags=()):
    exons = tuple(
        ExonRecord(
            chrom="chr1", start=start, end=end, strand=strand, gene_id=gene,
            transcript_id=tid, exon_number=i + 1, gene_type="protein_coding",
            transcript_type="protein_coding", tags=tuple(tags),
        )
        for i, (start, end) in enumerate(ranges)
    )
    return TranscriptRecord(
        transcript_id=tid, gene_id=gene, chrom="chr1", strand=strand,
        transcript_type="protein_coding", tags=tuple(tags), exons=exons,
    )


def synthetic_transcripts():
    return [
        # isolated one-to-one pair
        make_tx("single", "g_single", [(4000, 4050), (4200, 4250)], tags=("MANE_Select",)),
        # two independent junctions visible in one packet
        make_tx("independent", "g_ind", [(3000, 3050), (3150, 3200), (3300, 3350)]),
        # alternative donors sharing acceptor 1200
        make_tx("d1", "g_dcluster", [(1000, 1050), (1200, 1250)]),
        make_tx("d2", "g_dcluster", [(1000, 1080), (1200, 1250)]),
        # alternative acceptors sharing donor 1550
        make_tx("a1", "g_acluster", [(1500, 1550), (1700, 1750)]),
        make_tx("a2", "g_acluster", [(1500, 1550), (1720, 1770)]),
        # cassette exon: included two-edge path plus skip edge 150->400
        make_tx("cass_in", "g_cass", [(100, 150), (250, 300), (400, 450)]),
        make_tx("cass_skip", "g_cass", [(100, 150), (400, 450)]),
        # annotation-supported mutually exclusive internal exons
        make_tx("mxe1", "g_mxe", [(2000, 2050), (2100, 2130), (2300, 2350)]),
        make_tx("mxe2", "g_mxe", [(2000, 2050), (2160, 2190), (2300, 2350)]),
        # strand-coordinate contract
        make_tx("minus", "g_minus", [(5000, 5050), (5200, 5250)], strand="-"),
    ]


def packets_by_gene(transcripts=None):
    txs = synthetic_transcripts() if transcripts is None else transcripts
    idx = build_atlas_index(txs)
    packets = list(enumerate_atlas_packets(idx, AtlasConfig()))
    out = {}
    for packet in packets:
        out.setdefault(packet.gene_id, []).append(packet)
    return idx, packets, out


def test_multi_label_event_definitions():
    _idx, _packets, by_gene = packets_by_gene()
    assert all("single_pair" in p.tags for p in by_gene["g_single"])
    assert all("junction_one_to_one" in p.tags for p in by_gene["g_single"])
    assert all("isolated_pair_control_candidate" in p.tags for p in by_gene["g_single"])

    assert any("multiple_independent_pairs" in p.tags for p in by_gene["g_ind"])
    assert all("donor_cluster" in p.tags for p in by_gene["g_dcluster"])
    assert all("acceptor_cluster" in p.tags for p in by_gene["g_acluster"])
    assert all("cassette_exon" in p.tags for p in by_gene["g_cass"])
    assert all("complex_event" in p.tags for p in by_gene["g_cass"])
    assert all("mutually_exclusive_exon" in p.tags for p in by_gene["g_mxe"])
    assert all("complex_event" in p.tags for p in by_gene["g_mxe"])


def test_target_coordinate_and_packet_geometry_contract():
    _idx, packets, by_gene = packets_by_gene()
    for packet in packets:
        assert packet.window_end - packet.window_start == 1024
        assert 4 <= packet.donor_win < 1020
        assert 4 <= packet.acceptor_win < 1020
        if packet.target_role == "donor":
            assert packet.target_anchor_win == packet.donor_win
            assert packet.masked_base_win == packet.donor_win
        else:
            assert packet.target_anchor_win == packet.acceptor_win
            assert packet.masked_base_win == packet.acceptor_win - 1
    minus = by_gene["g_minus"]
    assert {p.target_role for p in minus} == {"donor", "acceptor"}
    assert all(p.strand == "-" for p in minus)


def test_packet_generation_is_order_independent_and_gene_disjoint_ready():
    txs = synthetic_transcripts()
    _idx1, packets1, _ = packets_by_gene(txs)
    _idx2, packets2, _ = packets_by_gene(list(reversed(txs)))
    sig1 = [(p.packet_id, p.tags, match_stratum(p)) for p in packets1]
    sig2 = [(p.packet_id, p.tags, match_stratum(p)) for p in packets2]
    assert sig1 == sig2
    assert len({p.packet_id for p in packets1}) == len(packets1)


def test_cassette_edges_include_skip_and_both_inclusion_edges():
    idx, _packets, _ = packets_by_gene()
    cass = {edge for edge in idx.cassette_edges if edge[0] == "g_cass"}
    assert {(edge[3], edge[4]) for edge in cass} == {(150, 250), (300, 400), (150, 400)}


def test_gene_hash_analysis_split_is_stable_and_preserves_seals():
    assert atlas_analysis_split("gene1", "train") == atlas_analysis_split("gene1", "train")
    assert atlas_analysis_split("gene1", "test") == "sealed_test"
    assert atlas_analysis_split("gene1", "val") == "touched_val_excluded"
    assert atlas_analysis_split("gene1", "train") in {
        "atlas_discovery", "atlas_internal_validation"
    }
