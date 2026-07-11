from pathlib import Path

from gfmcd.splicing.candidates import CandidateConfig
from gfmcd.splicing.dataset import DatasetBuildConfig, build_exon_pair_examples, reverse_complement


class MemoryGenome:
    def __init__(self, seqs):
        self.seqs = seqs
    def fetch(self, chrom, start, end):
        return self.seqs[chrom][start:end]
    def length(self, chrom):
        return len(self.seqs[chrom])


def test_reverse_complement():
    assert reverse_complement("AAGTCN") == "NGACTT"


def test_build_one_internal_exon(tmp_path: Path):
    # Internal exon [100,130); AG immediately upstream, GT immediately downstream.
    seq = list("C" * 300)
    seq[98:100] = "AG"
    seq[130:132] = "GT"
    # Decoy sites inside the 70-160 window.
    seq[78:80] = "AG"
    seq[150:152] = "GT"
    genome = MemoryGenome({"chr1": "".join(seq)})
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text(
        "\n".join([
            'chr1\ttest\texon\t51\t70\t.\t+\t.\tgene_id "g"; transcript_id "t"; transcript_type "protein_coding"; exon_number "1";',
            'chr1\ttest\texon\t101\t130\t.\t+\t.\tgene_id "g"; transcript_id "t"; transcript_type "protein_coding"; exon_number "2";',
            'chr1\ttest\texon\t181\t200\t.\t+\t.\tgene_id "g"; transcript_id "t"; transcript_type "protein_coding"; exon_number "3";',
        ]) + "\n"
    )
    cfg = DatasetBuildConfig(
        left_flank=30,
        right_flank=30,
        max_sequence_length=100,
        min_exon_length=10,
        max_exon_length=60,
        candidate=CandidateConfig(
            min_exon_length=10,
            max_exon_length=60,
            edge_buffer=2,
            max_pairs=None,
        ),
        val_chromosomes=("chr1",),
        test_chromosomes=(),
        exclude_chromosomes=(),
    )
    examples, skips = build_exon_pair_examples(gtf, genome, cfg)
    assert len(examples) == 1, skips
    ex = examples[0]
    assert ex.split == "val"
    assert ex.true_pair.acceptor_pos == 30
    assert ex.true_pair.donor_pos == 60
    assert len(ex.pairs) >= 2
