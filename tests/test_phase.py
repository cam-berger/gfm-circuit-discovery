from types import SimpleNamespace

from gfmcd.splicing.phase import (
    codon_slot, example_phase, intron_phase, load_cds_frames, window_codon_slot)

GTF = (
    # a two-CDS transcript: first CDS frame 0 (len 517) -> donor phase 1; next CDS frame 2
    'chr1\tHAVANA\tCDS\t924432\t924948\t.\t+\t0\tgene_id "G1"; transcript_id "T1";\n'
    'chr1\tHAVANA\tCDS\t925922\t926013\t.\t+\t2\tgene_id "G1"; transcript_id "T1";\n'
    'chr1\tHAVANA\texon\t924432\t924948\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
    # an ambiguous exon: same coords, disagreeing frames across transcripts -> dropped
    'chr2\tX\tCDS\t100\t200\t.\t+\t0\tgene_id "G2"; transcript_id "T2";\n'
    'chr2\tX\tCDS\t100\t200\t.\t+\t1\tgene_id "G2"; transcript_id "T3";\n'
)


def test_intron_phase_known_values():
    assert intron_phase(924431, 924948, 0) == 1  # validated on GENCODE
    assert intron_phase(0, 3, 0) == 0
    assert intron_phase(0, 4, 0) == 1
    assert intron_phase(0, 4, 1) == 0
    assert intron_phase(0, 4, 2) == 2


def test_load_cds_frames_and_ambiguity(tmp_path):
    p = tmp_path / "mini.gtf"
    p.write_text(GTF)
    lookup = load_cds_frames(p)
    assert lookup[("G1", 924431, 924948)] == 1               # first CDS -> donor phase 1
    assert lookup[("G1", 925921, 926013)] == 0               # intron_phase(925921,926013,2)=0
    assert ("G2", 99, 200) not in lookup                     # ambiguous -> dropped


def test_example_phase_matches_by_exon_coords(tmp_path):
    p = tmp_path / "mini.gtf"
    p.write_text(GTF)
    lookup = load_cds_frames(p)
    coding = SimpleNamespace(
        gene_id="G1", metadata={"genomic_exon_start": 924431, "genomic_exon_end": 924948})
    utr = SimpleNamespace(
        gene_id="G1", metadata={"genomic_exon_start": 111, "genomic_exon_end": 222})
    assert example_phase(coding, lookup) == 1
    assert example_phase(utr, lookup) is None                # no matching CDS -> noncoding boundary


def test_codon_slot_plus_and_minus_and_consistency():
    assert codon_slot(100, 100, 200, "+", 0) == 0
    assert codon_slot(101, 100, 200, "+", 0) == 1
    assert codon_slot(103, 100, 200, "+", 0) == 0
    assert codon_slot(100, 100, 200, "+", 1) == 2           # offset 0, frame 1 -> tail of prev codon
    assert codon_slot(101, 100, 200, "+", 1) == 0
    assert codon_slot(100, 100, 200, "+", 2) == 1
    # minus strand: offset measured from the genomic 3' end (translation 5')
    assert codon_slot(199, 100, 200, "-", 0) == 0
    assert codon_slot(198, 100, 200, "-", 0) == 1
    # slot at the position just past the CDS end equals the donor intron phase
    for f in (0, 1, 2):
        assert codon_slot(200, 100, 200, "+", f) == intron_phase(100, 200, f)


def test_window_codon_slot_plus_and_minus():
    ivals = {"G1": [(1100, 1200, "+", 0)], "G2": [(1100, 1200, "-", 0)]}
    plus = SimpleNamespace(gene_id="G1", strand="+",
                           metadata={"genomic_window_start": 1000, "genomic_window_end": 2000})
    assert window_codon_slot(plus, 150, ivals) == 2         # g=1150, offset 50, 50%3=2
    assert window_codon_slot(plus, 50, ivals) is None       # g=1050, outside the CDS
    minus = SimpleNamespace(gene_id="G2", strand="-",
                            metadata={"genomic_window_start": 1000, "genomic_window_end": 2000})
    assert window_codon_slot(minus, 850, ivals) == 2        # g=1149, offset 50, 50%3=2
