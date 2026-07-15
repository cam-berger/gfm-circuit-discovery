from types import SimpleNamespace

from gfmcd.splicing.phase import example_phase, intron_phase, load_cds_frames

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
