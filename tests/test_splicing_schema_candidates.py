from gfmcd.splicing.candidates import CandidateConfig, enumerate_candidate_pairs
from gfmcd.splicing.schema import PairCandidate, SiteCandidate, SplicePairExample


def make_example():
    seq = "C" * 8 + "AG" + "A" * 20 + "GT" + "C" * 12 + "AG" + "A" * 15 + "GT" + "C" * 8
    cfg = CandidateConfig(min_exon_length=10, max_exon_length=60, edge_buffer=2, max_pairs=None)
    acceptor = 10
    donor = 30
    acc, don, pairs, true = enumerate_candidate_pairs(
        seq, cfg, true_acceptor=acceptor, true_donor=donor
    )
    ex = SplicePairExample(
        example_id="ex1",
        sequence=seq,
        acceptors=acc,
        donors=don,
        pairs=pairs,
        true_pair_index=true,
        split="val",
        chrom="chr1",
        strand="+",
        gene_id="g1",
        transcript_id="t1",
        exon_number=2,
        genomic_exon_start=100,
        genomic_exon_end=120,
    )
    ex.validate()
    return ex


def test_candidate_enumeration_retains_true_and_decoys():
    ex = make_example()
    assert ex.true_pair.acceptor_pos == 10
    assert ex.true_pair.donor_pos == 30
    assert len(ex.pairs) >= 2


def test_roundtrip():
    ex = make_example()
    assert SplicePairExample.from_json(ex.to_json()) == ex
