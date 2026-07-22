from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from gfmcd.splicing.p3_closure import (
    ContextBlock,
    aggregate_gene_conditions,
    block_candidates,
    deterministic_rotations,
    gene_first_paired_phase_contrasts,
    gene_first_phase_contrasts,
    in_frame_stop_positions,
    local_recovery_contrast,
    rotate_cyclic,
    select_cds_blocks,
    select_spread_blocks,
    select_upstream_intron_blocks,
    site_excluded_phasor,
    stable_gene_bucket,
    stable_gene_role,
    validate_rotation_gates,
)


def test_cyclic_rotation_is_exact_and_composition_preserving():
    sequence = "NNNNACGTACNN"
    block = ContextBlock(4, 10, "cds")
    left = rotate_cyclic(sequence, block, 2, "left")
    right = rotate_cyclic(sequence, block, 2, "right")
    assert left == "NNNNGTACACNN"
    assert right == "NNNNACACGTNN"
    assert len(left) == len(sequence)
    assert Counter(left[block.start:block.end]) == Counter(sequence[block.start:block.end])
    assert left[:block.start] == sequence[:block.start]
    assert left[block.end:] == sequence[block.end:]
    assert rotate_cyclic(sequence, block, block.length, "left") is sequence


def test_deterministic_rotations_canonicalize_sets():
    sequence = "AACCGGTTAACC"
    block = ContextBlock(2, 10, "cds")
    got = deterministic_rotations(sequence, block, {9, 3, 6}, {"right", "left"})
    assert [(x.amount, x.direction) for x in got] == [
        (3, "left"), (3, "right"),
        (6, "left"), (6, "right"),
        (9, "left"), (9, "right"),
    ]
    assert got == deterministic_rotations(sequence, block, (6, 3, 9), ("left", "right"))


@pytest.mark.parametrize("amount", [-1, 1.5, True])
def test_rotation_rejects_ambiguous_amounts(amount):
    with pytest.raises(ValueError):
        rotate_cyclic("ACGT", (0, 4), amount, "left")


def test_context_block_candidates_pin_codon_boundary_and_protection():
    candidates = block_candidates(
        20,
        150,
        block_size=60,
        right_clearance=24,
        end_phase_origin=0,
        forbidden=((20, 25),),
        context="cds",
    )
    assert candidates
    assert all(block.length == 60 for block in candidates)
    assert all(block.end <= 126 and block.end % 3 == 0 for block in candidates)
    assert all(not (block.start < 25 and 20 < block.end) for block in candidates)


def test_context_selectors_choose_two_spread_valid_blocks():
    cds = select_cds_blocks(
        100,
        300,
        codon_boundary_origin=1,
        donor_clearance=24,
        n=2,
    )
    assert len(cds) == 2
    assert cds[0].start < cds[1].start
    assert all((block.end - 1) % 3 == 0 for block in cds)
    assert all(block.end <= 276 for block in cds)

    intron = select_upstream_intron_blocks(0, 190, acceptor_clearance=18, n=2)
    assert len(intron) == 2
    assert intron[0].start == 0
    assert intron[-1].end == 172


def test_spread_selector_refuses_insufficient_unique_blocks():
    with pytest.raises(ValueError, match="at least 2"):
        select_spread_blocks((ContextBlock(0, 60),), n=2)


def _safe_rotation_pair() -> tuple[str, str, ContextBlock]:
    # Repeating triplets prevent the rotation from creating splice motifs or stops.
    original = "C" * 24 + "ACG" * 20 + "C" * 36
    block = ContextBlock(24, 84, "cds")
    variant = rotate_cyclic(original, block, 3, "left")
    return original, variant, block


def test_rotation_gates_accept_clean_length_composition_and_protection():
    original, variant, block = _safe_rotation_pair()
    result = validate_rotation_gates(
        original,
        variant,
        block,
        protected_intervals=((0, 16), (96, 112)),
        expected_length=120,
        cds_scan=(24, 84, 24),
    )
    assert result.passed, result.failures
    assert result.metrics["block_hamming"] == 0  # period-3 content rotated by one codon
    assert result.metrics["native_stop_positions"] == ()
    result.require_passed()


def test_rotation_gates_detect_protected_site_length_and_composition_failures():
    original = "C" * 120
    block = ContextBlock(24, 84, "cds")
    variant = list(original)
    variant[30] = "A"
    variant[100] = "T"
    result = validate_rotation_gates(
        original,
        "".join(variant) + "A",
        block,
        protected_intervals=((96, 112),),
        expected_length=120,
    )
    assert not result.passed
    assert not result.checks["length_equal"]
    assert not result.checks["expected_length"]
    assert not result.checks["only_selected_block_changed"]
    assert not result.checks["block_composition_exact"]
    assert not result.checks["protected_sites_unchanged"]
    with pytest.raises(ValueError, match="rotation construction failed"):
        result.require_passed()


def test_rotation_gates_reject_new_canonical_motif():
    # First establish a composition-preserving, motif-safe edit.
    original = "TTAACCTT"
    variant = "TTCCAATT"
    result = validate_rotation_gates(original, variant, (2, 6), expected_length=8)
    assert result.checks["block_composition_exact"]

    # Rotating AACG right by one preserves composition but creates AG at the left boundary.
    bad_original = "AAACGT"
    bad_variant = rotate_cyclic(bad_original, (1, 5), 1, "right")
    assert bad_variant == "AGAACT"
    bad = validate_rotation_gates(bad_original, bad_variant, (1, 5), expected_length=6)
    assert not bad.checks["canonical_motif_counts_nonincreasing"]


def test_in_frame_stop_gate_uses_declared_frame():
    sequence = "CCCATGAAATAACCC"
    assert in_frame_stop_positions(sequence, 3, 12, frame_origin=3) == (9,)
    assert in_frame_stop_positions(sequence, 3, 12, frame_origin=5) == ()
    variant = sequence[:9] + "CAA" + sequence[12:]
    result = validate_rotation_gates(
        sequence, variant, (9, 12), expected_length=len(sequence), cds_scan=(3, 12, 3)
    )
    assert not result.checks["native_cds_stop_free"]
    assert result.checks["edited_cds_stop_free"]


def test_site_excluded_phasor_uses_distributed_interior_and_signed_class():
    attention = np.zeros(100)
    donor, acceptor = 10, 70
    # All included attention lies in acceptor-relative class 1.
    included = [x for x in range(donor + 7, acceptor - 6) if (x - acceptor) % 3 == 1]
    attention[included] = np.linspace(1.0, 2.0, len(included))
    # Very large site-local peaks must be excluded.
    attention[donor + 1] = 100.0
    attention[acceptor - 1] = 200.0
    result = site_excluded_phasor(attention, donor, acceptor)
    assert result.n_keys == (acceptor - donor - 1) - 12
    assert result.signed_projection == pytest.approx(1.0, abs=1e-12)
    assert result.amplitude == pytest.approx(1.0, abs=1e-12)
    assert result.class_masses == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert result.included_mass == pytest.approx(attention[included].sum())


def test_site_excluded_signed_projection_distinguishes_phase_and_zero_mass():
    attention = np.zeros(60)
    donor, acceptor = 5, 50
    class_two = [x for x in range(12, 44) if (x - acceptor) % 3 == 2]
    attention[class_two] = 1.0
    result = site_excluded_phasor(attention, donor, acceptor, reference_class=1)
    assert result.signed_projection == pytest.approx(-0.5, abs=1e-12)
    with pytest.raises(ValueError, match="mass is zero"):
        site_excluded_phasor(np.zeros(60), donor, acceptor)


def test_local_369_recovery_contrast_and_linear_trend_null():
    linear = {amount: 2.0 * amount + 5.0 for amount in range(2, 11)}
    assert local_recovery_contrast(linear) == pytest.approx(0.0, abs=1e-12)
    recovered = dict(linear)
    recovered[3] += 0.3
    recovered[6] += 0.6
    recovered[9] += 0.9
    assert local_recovery_contrast(recovered) == pytest.approx(0.6)
    with pytest.raises(ValueError, match="missing rotation amounts"):
        local_recovery_contrast({3: 1.0})


def test_gene_roles_are_stable_and_predeclared():
    assert stable_gene_bucket("ENSG000001", namespace="p3-closure-v1") == 9794
    roles = (("development", 2), ("confirmation", 1), ("unused", 7))
    assert stable_gene_role("ENSG000001", namespace="p3-closure-v1", roles=roles) == "unused"
    assert stable_gene_role("ENSG000001", namespace="p3-closure-v1", roles=roles) == \
        stable_gene_role("ENSG000001", namespace="p3-closure-v1", roles=roles)
    with pytest.raises(ValueError, match="unique"):
        stable_gene_role("g", namespace="n", roles=(("same", 1), ("same", 2)))


def _phase_rows(gene: str, arm: str, bump: float, duplicate: bool = False):
    rows = []
    for amount in range(2, 11):
        value = float(amount) + (bump if amount in (3, 6, 9) else 0.0)
        rows.append({"gene_id": gene, "context": arm, "rotation_amount": amount, "value": value})
        if duplicate:
            rows.append({
                "gene_id": gene,
                "context": arm,
                "rotation_amount": amount,
                "value": value,
            })
    return rows


def test_gene_first_aggregation_averages_edits_before_contrast():
    rows = _phase_rows("g2", "cds", 0.4, duplicate=True) + _phase_rows("g1", "cds", 0.2)
    aggregated = aggregate_gene_conditions(
        rows, condition_keys=("context", "rotation_amount"), value_key="value"
    )
    assert [row["gene_id"] for row in aggregated[:9]] == ["g1"] * 9
    assert sum(row["n_edits"] for row in aggregated if row["gene_id"] == "g2") == 18
    contrasts = gene_first_phase_contrasts(rows)
    assert contrasts == pytest.approx({"g1": 0.2, "g2": 0.4})


def test_gene_first_paired_phase_contrast_is_target_minus_control():
    rows = (
        _phase_rows("g1", "cds", 0.7)
        + _phase_rows("g1", "intron", 0.2)
        + _phase_rows("g2", "cds", 0.4)
        + _phase_rows("g2", "intron", 0.1)
    )
    paired = gene_first_paired_phase_contrasts(
        rows, target_arm="cds", control_arm="intron"
    )
    assert paired == pytest.approx({"g1": 0.5, "g2": 0.3})


def test_gene_first_paired_phase_contrast_requires_paired_genes():
    rows = _phase_rows("g1", "cds", 0.5) + _phase_rows("g2", "intron", 0.1)
    with pytest.raises(ValueError, match="unpaired gene arms"):
        gene_first_paired_phase_contrasts(rows, target_arm="cds", control_arm="intron")
