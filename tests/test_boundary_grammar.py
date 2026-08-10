from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from gfmcd.splicing.boundary_grammar import (
    ALL_CONTRASTS,
    MINUS3_PANEL,
    MINUS3_T_MINUS_A_SINGLE,
    MINUS4_PANEL,
    MINUS4_T_MINUS_A,
    PLUS1_PANEL,
    PLUS1_G_MINUS_A_SINGLE,
    PLUS2_PANEL,
    PLUS2_G_MINUS_A,
    ROLE_WEIGHTS,
    TARGET_CONTRASTS,
    TARGET_PANEL,
    PanelCell,
    PanelSpec,
    add_one_pvalue,
    aggregate_gene_cells,
    biological_to_array_index,
    biological_to_array_offset,
    build_panel,
    build_single_locus_panel,
    build_target_panel,
    cohort_declaration,
    construct_sha256,
    contrast,
    declaration_sha256,
    design_declaration,
    holm_adjust,
    margins_from_logits,
    max_family_add_one_pvalues,
    positioned_motif_inventory,
    select_gene_cohort,
    stable_gene_role,
    target_contrasts,
    validate_panel,
)


ACCEPTOR_POS = 10


def _clean_sequence(*, minus3: str = "T", plus1: str = "A", plus2: str = "A") -> str:
    # index:                 6      7       8:10    10      11    12
    # biological:          -4     -3       core    +1      +2    +3
    return "C" * 6 + "C" + minus3 + "AG" + plus1 + plus2 + "A" + "C" * 7


def _example(
    gene_id: str = "g0",
    example_id: str = "e0",
    *,
    strand: str = "+",
    sequence: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        gene_id=gene_id,
        example_id=example_id,
        chrom="chr1",
        strand=strand,
        sequence=sequence or _clean_sequence(),
        donor_pos=3,
        next_acceptor_pos=ACCEPTOR_POS,
    )


def test_frozen_panels_and_contrast_ids_are_complete():
    assert ROLE_WEIGHTS == (("behavior", 3), ("route", 2), ("reserve", 1))
    assert [cell.cell_id for cell in TARGET_PANEL.cells] == [
        f"m3_{minus3}__p1_{plus1}"
        for minus3 in "ACGT"
        for plus1 in ("A", "G")
    ]
    assert [spec.contrast_id for spec in TARGET_CONTRASTS] == [
        "m3_T_minus_A",
        "p1_G_minus_A",
        "m3_by_p1_interaction",
    ]
    all_ids = [spec.contrast_id for spec in ALL_CONTRASTS]
    assert len(all_ids) == len(set(all_ids))
    assert [cell.cell_id for cell in MINUS3_PANEL.cells] == ["m3_A", "m3_T"]
    assert [cell.cell_id for cell in PLUS1_PANEL.cells] == ["p1_A", "p1_G"]
    assert [cell.cell_id for cell in MINUS4_PANEL.cells] == ["m4_A", "m4_T"]
    assert [cell.cell_id for cell in PLUS2_PANEL.cells] == ["p2_A", "p2_G"]


def test_biological_offsets_and_reverse_strand_use_transcript_oriented_arrays():
    assert biological_to_array_offset(-4) == -4
    assert biological_to_array_offset(-3) == -3
    assert biological_to_array_offset(-1) == -1
    assert biological_to_array_offset(1) == 0
    assert biological_to_array_offset(2) == 1
    with pytest.raises(ValueError, match="does not exist"):
        biological_to_array_offset(0)

    for strand in ("+", "-"):
        assert biological_to_array_index(ACCEPTOR_POS, -3, strand=strand) == 7
        assert biological_to_array_index(ACCEPTOR_POS, 1, strand=strand) == 10
    plus = build_target_panel(_example(strand="+"))
    minus = build_target_panel(_example(strand="-"))
    assert plus == minus


def test_target_panel_is_strict_motif_clean_and_changes_only_declared_loci():
    sequence = _clean_sequence()
    cells = build_panel(sequence, ACCEPTOR_POS)
    result = validate_panel(sequence, cells, acceptor_pos=ACCEPTOR_POS)
    assert result.passed, result.failures
    assert result.metrics["declared_array_indices"] == (7, 10)
    assert result.metrics["touched_register_starts"] == (6, 7, 9, 10)
    assert all(not inventory for inventory in result.metrics["positioned_inventories"].values())
    for cell in TARGET_PANEL.cells:
        variant = cells[cell.cell_id]
        assert variant[8:10] == "AG"
        changed = {index for index, pair in enumerate(zip(sequence, variant)) if pair[0] != pair[1]}
        assert changed.issubset({7, 10})
        assert all(variant[biological_to_array_index(ACCEPTOR_POS, pos)] == base
                   for pos, base in cell.alleles)


def test_target_panel_rejects_flanks_that_make_inventory_cell_dependent():
    # +2=G makes the p1=A cell plant AG at [a,a+2), while p1=G does not.
    unsafe = _clean_sequence(plus2="G")
    raw = build_panel(unsafe, ACCEPTOR_POS, require_valid=False)
    result = validate_panel(unsafe, raw, acceptor_pos=ACCEPTOR_POS)
    assert not result.checks["positioned_inventories_identical"]
    assert not result.checks["no_extra_touched_motifs"]
    with pytest.raises(ValueError, match="boundary panel validation failed"):
        build_panel(unsafe, ACCEPTOR_POS)


def test_target_panel_rejects_destroying_a_native_touched_splice_motif():
    # Native +1=C forms GC with the invariant -1=G. Synthetic +1=A/G removes it.
    unsafe = _clean_sequence(plus1="C")
    raw = build_panel(unsafe, ACCEPTOR_POS, require_valid=False)
    result = validate_panel(unsafe, raw, acceptor_pos=ACCEPTOR_POS)
    assert result.metrics["positioned_inventories"]["native"] == (
        (ACCEPTOR_POS - 1, "GC"),
    )
    assert not result.checks["positioned_inventories_identical"]
    assert not result.checks["no_extra_touched_motifs"]
    with pytest.raises(ValueError, match="boundary panel validation failed"):
        build_panel(unsafe, ACCEPTOR_POS)


def test_plus1_c_and_t_exhaustively_plant_gc_and_gt():
    bad_panel = PanelSpec(
        "bad_p1_ct",
        (1,),
        (PanelCell("p1_C", ((1, "C"),)), PanelCell("p1_T", ((1, "T"),))),
    )
    for minus3, plus2 in product("ACGT", repeat=2):
        sequence = _clean_sequence(minus3=minus3, plus2=plus2)
        cells = build_panel(
            sequence, ACCEPTOR_POS, bad_panel, require_valid=False
        )
        result = validate_panel(
            sequence, cells, bad_panel, acceptor_pos=ACCEPTOR_POS
        )
        inventories = result.metrics["positioned_inventories"]
        assert (ACCEPTOR_POS - 1, "GC") in inventories["p1_C"]
        assert (ACCEPTOR_POS - 1, "GT") in inventories["p1_T"]
        assert not result.checks["positioned_inventories_identical"]
        assert not result.checks["no_extra_touched_motifs"]


def test_positioned_inventory_and_panel_validation_detect_undeclared_edits():
    sequence = _clean_sequence()
    cells = build_panel(sequence, ACCEPTOR_POS)
    contaminated = dict(cells)
    first = next(iter(contaminated))
    contaminated[first] = "A" + contaminated[first][1:]
    result = validate_panel(sequence, contaminated, acceptor_pos=ACCEPTOR_POS)
    assert not result.checks["only_declared_loci_changed"]

    motif = positioned_motif_inventory(
        "C" * 8 + "AGTC" + "C" * 8,
        acceptor_pos=10,
        biological_positions=(1,),
    )
    assert motif == ((9, "GT"),)


def test_single_locus_panels_accept_native_noop_and_use_fixed_directions():
    example = _example(sequence=_clean_sequence(minus3="T", plus1="G", plus2="A"))
    minus3 = build_single_locus_panel(example, -3, ("A", "T"))
    plus1 = build_single_locus_panel(example, 1, ("A", "G"))
    minus4 = build_single_locus_panel(example, -4, ("A", "T"))
    plus2 = build_single_locus_panel(example, 2, ("A", "G"))
    assert minus3["m3_T"] == example.sequence
    assert plus1["p1_G"] == example.sequence
    assert minus4["m4_A"] != minus4["m4_T"]
    assert plus2["p2_A"] == example.sequence
    assert contrast({"m3_A": 1.0, "m3_T": 2.5}, MINUS3_T_MINUS_A_SINGLE) == 1.5
    assert contrast({"p1_A": 1.0, "p1_G": 2.5}, PLUS1_G_MINUS_A_SINGLE) == 1.5
    assert contrast({"m4_A": 1.0, "m4_T": 2.5}, MINUS4_T_MINUS_A) == 1.5
    assert contrast({"p2_A": 1.0, "p2_G": 2.5}, PLUS2_G_MINUS_A) == 1.5


def test_target_contrast_arithmetic_and_constant_zero_response():
    values = {}
    for cell in TARGET_PANEL.cells:
        states = dict(cell.alleles)
        values[cell.cell_id] = (
            2.0 * (states[-3] == "T")
            + 3.0 * (states[1] == "G")
            + 5.0 * (states[-3] == "T" and states[1] == "G")
        )
    got = target_contrasts(values)
    assert got == pytest.approx({
        "m3_T_minus_A": 4.5,
        "p1_G_minus_A": 4.25,
        "m3_by_p1_interaction": 5.0,
    })
    zeros = target_contrasts({cell.cell_id: 7.0 for cell in TARGET_PANEL.cells})
    assert zeros == pytest.approx({name: 0.0 for name in zeros}, abs=1e-12)


def test_gene_roles_and_one_per_gene_selection_are_disjoint_and_order_independent():
    examples = [
        _example(f"g{gene:03d}", f"g{gene:03d}-e{copy}")
        for gene in range(120)
        for copy in range(2)
    ]
    cohorts = {
        role: select_gene_cohort(examples, role=role)
        for role, _weight in ROLE_WEIGHTS
    }
    reverse = {
        role: select_gene_cohort(reversed(examples), role=role)
        for role, _weight in ROLE_WEIGHTS
    }
    assert {
        role: [example.example_id for example in cohort] for role, cohort in cohorts.items()
    } == {
        role: [example.example_id for example in cohort] for role, cohort in reverse.items()
    }
    gene_sets = {role: {example.gene_id for example in cohort} for role, cohort in cohorts.items()}
    assert all(gene_sets[left].isdisjoint(gene_sets[right])
               for left, right in (("behavior", "route"), ("behavior", "reserve"),
                                   ("route", "reserve")))
    assert set().union(*gene_sets.values()) == {f"g{gene:03d}" for gene in range(120)}
    assert all(len(cohort) == len(gene_sets[role]) for role, cohort in cohorts.items())
    assert stable_gene_role("g007") == stable_gene_role("g007")
    declaration = cohort_declaration(cohorts)
    assert set(declaration) == {"ids", "genes", "sha256"}
    assert len(declaration["sha256"]) == 64


def test_margins_use_one_logit_array_for_linear_and_max_reductions():
    logits = np.full((2, 5, 9), -100.0)
    base_ids = {"A": 2, "C": 4, "G": 6, "T": 8}
    donors = np.asarray([1, 2])
    logits[0, 2, [2, 4, 6, 8]] = [1.0, 2.0, 6.0, 3.0]
    logits[1, 3, [2, 4, 6, 8]] = [5.0, 1.0, 2.0, 0.0]
    linear, maximum = margins_from_logits(
        logits, donors, base_token_ids=base_ids, token_offset=1
    )
    assert linear == pytest.approx([4.0, 0.0])
    assert maximum == pytest.approx([3.0, -3.0])


def test_gene_aggregation_averages_repeats_then_requires_complete_panel():
    rows = []
    for gene in ("g2", "g1"):
        for index, cell in enumerate(TARGET_PANEL.cells):
            for bump in (0.0, 2.0):
                rows.append({
                    "gene_id": gene,
                    "cell_id": cell.cell_id,
                    "linear_margin": index + bump,
                    "max_margin": 2 * index + bump,
                })
    aggregated = aggregate_gene_cells(rows)
    assert len(aggregated) == 16
    assert [row["gene_id"] for row in aggregated[:8]] == ["g1"] * 8
    assert aggregated[0]["linear_margin"] == pytest.approx(1.0)
    assert aggregated[0]["n_rows"] == 2
    with pytest.raises(ValueError, match="complete"):
        aggregate_gene_cells(rows[:-2])


def test_exact_add_one_holm_and_max_family_helpers():
    assert add_one_pvalue(4.0, [0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.2)
    assert add_one_pvalue(-4.0, [-1.0, 1.0, 2.0], alternative="two-sided") == 0.25
    assert holm_adjust([0.01, 0.04, 0.03]).tolist() == pytest.approx([0.03, 0.06, 0.06])
    null = np.asarray([[0.1, 0.5], [0.2, 0.4], [0.3, 0.1]])
    assert max_family_add_one_pvalues([0.45, 0.25], null).tolist() == pytest.approx([0.5, 1.0])


def test_declaration_and_construct_hashes_are_canonical():
    declaration = design_declaration()
    assert declaration["sesoi"] == 0.05
    assert set(declaration["panels"]) == {
        "target", "minus3_inner", "minus4_diagnostic", "plus1_inner", "plus2_diagnostic"
    }
    assert len(declaration_sha256()) == 64
    cells = build_panel(_clean_sequence(), ACCEPTOR_POS)
    assert construct_sha256(cells) == construct_sha256(dict(reversed(tuple(cells.items()))))
