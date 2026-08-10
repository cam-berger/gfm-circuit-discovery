from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from gfmcd.splicing import boundary_grammar as bg
from gfmcd.splicing.coupling import CouplingConfig, build_coupling_dataset
from gfmcd.splicing.dataset import PyfaidxGenome


def _gtf_exon(chrom: str, start: int, end: int, gene: str, transcript: str, number: int) -> str:
    attributes = (
        f'gene_id "{gene}"; transcript_id "{transcript}"; '
        f'gene_type "protein_coding"; transcript_type "protein_coding"; '
        f'exon_number "{number}"; tag "basic";'
    )
    return "\t".join(
        (chrom, "test", "exon", str(start + 1), str(end), ".", "+", ".", attributes)
    )


def _plant(sequence: list[str], start: int, text: str) -> None:
    sequence[start:start + len(text)] = text


def test_streaming_boundary_loader_matches_stock_builder_with_complete_examples(tmp_path: Path):
    sequence = list("C" * 5000)
    genes = {
        "gene_a": ((700, 750), (1000, 1050), (1120, 1170)),
        "gene_b": ((2500, 2550), (2800, 2860), (2940, 3000)),
    }
    for exons in genes.values():
        internal, downstream = exons[1], exons[2]
        _plant(sequence, internal[0] - 2, "AG")
        _plant(sequence, internal[0] + 1, "A")  # keeps every target-panel touched register clean
        _plant(sequence, internal[1], "GT")
        _plant(sequence, downstream[0] - 2, "AG")
        _plant(sequence, internal[0] - 70, "AG")  # retained motif-matched nulls
        _plant(sequence, internal[0] - 50, "GT")

    fasta = tmp_path / "tiny.fa"
    fasta.write_text(">chr1\n" + "".join(sequence) + "\n")
    rows = []
    for gene, exons in genes.items():
        # Reverse lexical input order exercises the stock transcript sort.  The duplicate
        # isoform also checks that support counts are computed before exon deduplication.
        transcripts = ("tx_z", "tx_a") if gene == "gene_a" else ("tx_b",)
        for transcript in transcripts:
            rows.extend(
                _gtf_exon("chr1", start, end, gene, transcript, number)
                for number, (start, end) in enumerate(exons, start=1)
            )
    gtf = tmp_path / "tiny.gtf"
    gtf.write_text("\n".join(rows) + "\n")

    stock = build_coupling_dataset(gtf, PyfaidxGenome(fasta), CouplingConfig())
    expected = [example for example in stock if bg.boundary_relevant(example)]
    bg._BOUNDARY_DATASET_CACHE.clear()
    observed = bg.load_boundary_examples(str(gtf), str(fasta))

    assert [asdict(item) for item in observed] == [asdict(item) for item in expected]
    assert len(observed) == 2
    assert {item.n_isoforms_donor for item in observed} == {1, 2}
    assert all(item.other_ag and item.other_gt for item in observed)
