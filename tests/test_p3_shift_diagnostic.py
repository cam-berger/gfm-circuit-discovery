import importlib.util
from pathlib import Path

from gfmcd.splicing.coupling import CouplingExample


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/splicebert/run_p3_shift_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("run_p3_shift_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
attention_metrics = MODULE.attention_metrics
choose_insertion = MODULE.choose_insertion
insertion_variant = MODULE.insertion_variant
motif_counts = MODULE.motif_counts
validate_variant = MODULE.validate_variant


def example():
    seq = list("C" * 1024)
    seq[200:202] = "GT"
    seq[298:300] = "AG"
    return CouplingExample(
        example_id="x", sequence="".join(seq), acceptor_pos=100, donor_pos=200,
        chrom="chr1", strand="+", gene_id="g", split="val", exon_length=100,
        intron_length_down=100, n_isoforms_donor=1, n_isoforms_acceptor=1,
        gene_type="protein_coding", next_acceptor_pos=300,
        metadata={"genomic_window_start": 1000, "genomic_window_end": 2024,
                  "genomic_exon_start": 1100, "genomic_exon_end": 1200},
    )


def test_choose_insertion_never_increases_local_splice_motifs():
    seq = "ACCCCTTTTACCCCTTTT"
    for size in (1, 2, 3):
        ins = choose_insertion(seq, 9, size, radius=8)
        before = motif_counts(seq[1:17])
        after = motif_counts(seq[1:9] + ins + seq[9:17])
        assert all(after[m] <= before[m] for m in before)


def test_intron_and_exon_coordinates_and_contexts():
    x = example()
    intron = insertion_variant(x, "intron_insert", 2)
    assert intron.donor_pos == 200
    assert intron.next_acceptor_pos == 302
    validate_variant(x, intron, "intron_insert", 2)

    exon = insertion_variant(x, "exon_insert", 2)
    assert exon.donor_pos == 202
    assert exon.next_acceptor_pos == 302
    validate_variant(x, exon, "exon_insert", 2)

    sham = insertion_variant(x, "upstream_intron_sham", 2)
    assert sham.acceptor_pos == 102
    assert sham.donor_pos == 202
    assert sham.next_acceptor_pos == 302
    validate_variant(x, sham, "upstream_intron_sham", 2)


def test_zero_shift_is_exact_identity():
    x = example()
    for kind in ("intron_insert", "exon_insert", "upstream_intron_sham"):
        assert insertion_variant(x, kind, 0) is x


def test_attention_peak_and_residue_mass_are_distinct_readouts():
    import numpy as np

    row = np.zeros(1024)
    row[211] = 0.2
    row[214] = 0.3
    row[217] = 0.4
    row[299] = 0.6
    m = attention_metrics(row, donor=200, acceptor=300)
    assert m["peak_abs"] == 299
    assert m["peak_rel_acceptor"] == -1
    assert abs(sum(m["mass_rel_donor_mod3"]) - 1) < 1e-12
    assert m["mass_rel_donor_mod3"][0] > 0  # the other comb teeth remain represented
