from __future__ import annotations

from dataclasses import replace

import pytest

from gfmcd.splicing.acceptor_mechanism_cohort import (
    R5_EXPECTED_CENSUS_IDENTITY,
    R5_EXPECTED_SOURCE_IDENTITY,
    R5_ROWS_PER_BUCKET,
    build_r5_reservations,
    deterministic_adjacent_window,
    evaluate_r5_row,
    r5_gene_bucket,
    r5_gene_priority,
    r5_row_priority,
    validate_expected_r5_source_identity,
    validate_r5_reservation_audit,
)
from gfmcd.splicing.coupling import CouplingExample


def _example(index: int, *, gene: str | None = None, strand: str = "+") -> CouplingExample:
    sequence = list("C" * 120)
    donor = 40
    pseudo = 50
    acceptor = 90
    sequence[donor : donor + 2] = "GT"
    sequence[pseudo : pseudo + 2] = "GT"
    sequence[acceptor - 2 : acceptor] = "AG"
    return CouplingExample(
        example_id=f"example-{index}",
        sequence="".join(sequence),
        acceptor_pos=20,
        donor_pos=donor,
        chrom="chr11",
        strand=strand,
        gene_id=gene or f"gene-{index}",
        split="val",
        exon_length=20,
        intron_length_down=50,
        n_isoforms_donor=1,
        n_isoforms_acceptor=1,
        gene_type="protein_coding",
        other_gt=(pseudo,),
        next_acceptor_pos=acceptor,
    )


def test_adjacent_window_and_three_acceptor_only_sequence_states_are_exact():
    example = _example(0)
    adjacent = deterministic_adjacent_window(
        example_id=example.example_id,
        sequence=example.sequence,
        paired_donor=example.donor_pos,
        pseudo_donor=50,
        acceptor_g=89,
    )
    assert adjacent is not None
    assert len(adjacent) == 4
    assert max(abs(position - example.donor_pos) for position in adjacent) <= 34
    row, reason = evaluate_r5_row(example, 50, excluded_gene_ids=frozenset())
    assert reason is None and row is not None
    assert row.clean_sequence[89] == "G"
    assert row.paired_corrupt_sequence[40:42] == "AC"
    assert row.paired_corrupt_sequence[50:52] == "GT"
    assert row.pseudo_corrupt_sequence[40:42] == "GT"
    assert row.pseudo_corrupt_sequence[50:52] == "AC"
    assert row.identity_record().keys().isdisjoint({"donor_readout", "masked_donor"})


def test_eligibility_fails_closed_on_exclusion_caliper_motif_and_adjacent_control():
    example = _example(1)
    row, reason = evaluate_r5_row(
        example, 50, excluded_gene_ids=frozenset({example.gene_id})
    )
    assert row is None and reason == "excluded_gene"
    row, reason = evaluate_r5_row(example, 100, excluded_gene_ids=frozenset())
    assert row is None and reason == "pseudo_donor_outside_40nt_caliper"
    no_pseudo = replace(example, sequence=example.sequence[:50] + "CC" + example.sequence[52:])
    row, reason = evaluate_r5_row(no_pseudo, 50, excluded_gene_ids=frozenset())
    assert row is None and reason == "pseudo_donor_not_GT"

    blocked = list(example.sequence)
    # Put motifs throughout every padded shell on both sides.
    for start in range(6, 72, 2):
        blocked[start : start + 2] = "AG" if start % 4 else "GT"
    blocked[example.donor_pos : example.donor_pos + 2] = "GT"
    blocked[50:52] = "GT"
    blocked[88:90] = "AG"
    row, reason = evaluate_r5_row(
        replace(example, sequence="".join(blocked)),
        50,
        excluded_gene_ids=frozenset(),
    )
    assert row is None and reason == "missing_admissible_adjacent_window"


def test_reservations_choose_one_row_per_gene_and_bind_exact_constructions():
    examples = []
    pseudo = {}
    for index in range(80):
        for duplicate in range(2):
            item = _example(
                index * 2 + duplicate,
                gene=f"gene-{index}",
                strand="+" if index % 2 == 0 else "-",
            )
            examples.append(item)
            pseudo[item.example_id] = 50
    selected, audit = build_r5_reservations(
        examples,
        pseudo,
        historical_r3_gene_ids=["gene-excluded-old"],
        r3c_gene_ids=["gene-excluded-current"],
        rows_per_bucket=10,
    )
    for bucket in (0, 1):
        assert len(selected[bucket]) == 10
        genes = [row.example.gene_id for row in selected[bucket]]
        assert len(set(genes)) == 10
        assert genes == sorted(genes, key=r5_gene_priority)
        for row in selected[bucket]:
            siblings = [
                example
                for example in examples
                if example.gene_id == row.example.gene_id
            ]
            assert row.example.example_id == min(
                (example.example_id for example in siblings), key=r5_row_priority
            )
            assert r5_gene_bucket(row.example.gene_id) == bucket
    validate_r5_reservation_audit(audit, rows_per_bucket=10)
    assert audit["selection_reads_model_outputs"] is False
    assert audit["sealed_chr8_chr9_opened"] is False
    assert audit["eligible_rows"] == 160
    assert audit["eligible_genes"] == 80


def test_capacity_and_two_strand_gates_cannot_be_self_attested():
    examples = [_example(index, strand="+") for index in range(12)]
    pseudo = {example.example_id: 50 for example in examples}
    with pytest.raises(RuntimeError, match="capacity is insufficient"):
        build_r5_reservations(
            examples,
            pseudo,
            historical_r3_gene_ids=[],
            r3c_gene_ids=[],
            rows_per_bucket=R5_ROWS_PER_BUCKET,
        )
    _, audit = build_r5_reservations(
        examples,
        pseudo,
        historical_r3_gene_ids=[],
        r3c_gene_ids=[],
        rows_per_bucket=2,
        require_capacity=False,
    )
    with pytest.raises(ValueError, match="both strands"):
        validate_r5_reservation_audit(audit, rows_per_bucket=2)


def test_expected_full_census_identity_is_literal_and_capacity_is_not_rounded_up():
    expected = R5_EXPECTED_CENSUS_IDENTITY
    assert expected["input_rows"] == 6064
    assert expected["eligible_rows"] == 366
    assert expected["eligible_genes"] == 272
    assert expected["available_genes_by_bucket"] == {"0": 142, "1": 130}
    assert expected["selected_by_bucket"]["0"]["rows"] == 128
    assert expected["selected_by_bucket"]["1"]["rows"] == 128
    assert expected["selected_by_bucket"]["1"]["gene_ids_sha256"].startswith(
        "e6c30992"
    )


def test_source_identity_pins_comprehensive_annotation_and_rejects_other_bytes(
    tmp_path,
):
    assert R5_EXPECTED_SOURCE_IDENTITY["gtf"] == {
        "label": "GENCODE_v50_comprehensive",
        "size_bytes": 124_527_720,
        "sha256": "83fba3e9b03f0b8c958f3595c6c350adc55f468abf8b0e47b6d5284cfe13a453",
    }
    gtf = tmp_path / "annotation.gtf.gz"
    fasta = tmp_path / "reference.fa"
    gtf.write_bytes(b"not the pinned comprehensive annotation")
    fasta.write_bytes(b">chr11\nACGT\n")
    with pytest.raises(RuntimeError, match="source identity drifted"):
        validate_expected_r5_source_identity(gtf, fasta)
