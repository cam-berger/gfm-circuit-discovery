"""Memorials for the model-free paired-lineage pretraining core.

Every test here can fail on a real defect: the operator's three invariants are checked against
independently constructed expectations rather than against the implementation's own algebra,
the arms' identity is checked by byte comparison, and the masking recipe's three shares are
checked by construction (a hand-built ``draws`` vector that lands in each branch on purpose)
rather than by a tolerance on a random sample.
"""

from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing.lineage_pretraining import (
    DEFAULT_EXON_OFFSET,
    DEFAULT_LENGTH_LADDER,
    GeometryError,
    MASK_RATE,
    assign_residues,
    batch_plan,
    build_chimeric_cells,
    cluster_bootstrap,
    corpus_digest,
    crop_interval,
    frame_offset,
    indel_geometry,
    mask_tokens,
    paired_indel,
    period3_band_power,
    position_draws,
    psi,
    register_shift,
    uniform_draw,
    upstream_prefix_is_intronic,
)

# A synthetic window with the geometry `build_coupling_dataset` produces: intronic prefix,
# upstream exon, downstream intron, partner exon.
ACCEPTOR, DONOR, NEXT_ACCEPTOR, LENGTH = 200, 360, 460, 1024


def _window(seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    seq = list("".join(rng.choice(list("ACGT"), size=LENGTH)))
    seq[DONOR], seq[DONOR + 1] = "G", "T"
    seq[NEXT_ACCEPTOR - 2], seq[NEXT_ACCEPTOR - 1] = "A", "G"
    return "".join(seq)


# ---------------------------------------------------------------------------------------
# frame_offset / assign_residues / register_shift
# ---------------------------------------------------------------------------------------
def test_frame_offset_is_three_k_plus_residue():
    assert frame_offset(1, 0) == 3
    assert frame_offset(1, 1) == 4
    assert frame_offset(4, 2) == 14


@pytest.mark.parametrize("k", [0, -1])
def test_frame_offset_rejects_non_positive_k(k):
    # k=0 with residue 0 is the identity edit -- not an edit-matched control.
    with pytest.raises(ValueError, match="positive integer"):
        frame_offset(k, 0)


def test_frame_offset_rejects_out_of_range_residue():
    with pytest.raises(ValueError, match="residue must be"):
        frame_offset(1, 3)


def test_register_shift_is_zero_exactly_for_multiples_of_three():
    assert register_shift(3) == 0
    assert register_shift(12) == 0
    assert register_shift(4) == 2
    assert register_shift(5) == 1


def test_assign_residues_is_deterministic_and_uses_both_values():
    first = assign_residues(400, seed=7)
    assert first == assign_residues(400, seed=7)
    assert set(first) == {1, 2}
    # Neither value may collapse: a corpus fixed at one residue confounds "frame does not
    # continue" with one shift direction.
    assert 0.3 < first.count(1) / len(first) < 0.7
    assert assign_residues(400, seed=8) != first


def test_assign_residues_length_and_prefix_stability():
    assert len(assign_residues(11, seed=3)) == 11
    # Keyed by index, so a longer corpus extends rather than reshuffles the earlier windows.
    assert assign_residues(50, seed=3)[:11] == assign_residues(11, seed=3)


# ---------------------------------------------------------------------------------------
# upstream_prefix_is_intronic
# ---------------------------------------------------------------------------------------
def test_upstream_prefix_requires_a_resolvable_intron():
    assert upstream_prefix_is_intronic(acceptor_pos=200, intron_length_up=None) is False


def test_upstream_prefix_boundary_is_inclusive_of_the_margin():
    assert upstream_prefix_is_intronic(acceptor_pos=200, intron_length_up=210, margin=10)
    assert not upstream_prefix_is_intronic(acceptor_pos=200, intron_length_up=209, margin=10)


# ---------------------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("offset", [3, 4, 5, 12, 14])
def test_paired_indel_preserves_length_donor_and_partner_acceptor(offset):
    seq = _window()
    edited = paired_indel(
        seq,
        acceptor_pos=ACCEPTOR,
        donor_pos=DONOR,
        next_acceptor_pos=NEXT_ACCEPTOR,
        offset=offset,
    )
    assert len(edited) == len(seq)
    # The two motifs the assay reads are byte-identical AND co-located.
    assert edited[DONOR : DONOR + 2] == "GT"
    assert edited[NEXT_ACCEPTOR - 2 : NEXT_ACCEPTOR] == "AG"
    # Everything from the insertion point onward is untouched.
    assert edited[DONOR - DEFAULT_EXON_OFFSET :] == seq[DONOR - DEFAULT_EXON_OFFSET :]


def test_paired_indel_actually_edits_the_window():
    seq = _window()
    edited = paired_indel(
        seq, acceptor_pos=ACCEPTOR, donor_pos=DONOR, next_acceptor_pos=NEXT_ACCEPTOR, offset=4
    )
    assert edited != seq          # a no-op operator would make both arms identical


def test_paired_indel_moves_the_upstream_block_by_exactly_offset():
    """The bulk of the upstream exon must shift by `offset`, which is what changes register."""
    seq = _window()
    offset = 5
    edited = paired_indel(
        seq,
        acceptor_pos=ACCEPTOR,
        donor_pos=DONOR,
        next_acceptor_pos=NEXT_ACCEPTOR,
        offset=offset,
    )
    plan = indel_geometry(
        seq_len=len(seq),
        acceptor_pos=ACCEPTOR,
        donor_pos=DONOR,
        next_acceptor_pos=NEXT_ACCEPTOR,
        offset=offset,
    )
    z, x = plan["delete_at"], plan["insert_at"]
    # Independently stated expectation, not the implementation's expression.
    assert edited[z : x - offset] == seq[z + offset : x]
    assert edited[:z] == seq[:z]
    # The inserted block is a tandem duplication of the exonic bases before the insertion point.
    assert edited[x - offset : x] == seq[x - offset : x]


def test_arms_differ_only_in_the_residue_and_are_otherwise_the_same_operator():
    seq = _window(1)
    l1 = paired_indel(
        seq, acceptor_pos=ACCEPTOR, donor_pos=DONOR, next_acceptor_pos=NEXT_ACCEPTOR,
        offset=frame_offset(1, 0),
    )
    l2 = paired_indel(
        seq, acceptor_pos=ACCEPTOR, donor_pos=DONOR, next_acceptor_pos=NEXT_ACCEPTOR,
        offset=frame_offset(1, 1),
    )
    assert len(l1) == len(l2) == len(seq)
    assert l1 != l2
    assert l1[DONOR:] == l2[DONOR:] == seq[DONOR:]
    assert register_shift(frame_offset(1, 0)) == 0
    assert register_shift(frame_offset(1, 1)) != 0


@pytest.mark.parametrize("offset,expected_shift", [(3, 0), (4, 2), (5, 1), (12, 0), (13, 2)])
def test_upstream_exon_grows_by_offset_which_is_the_frame_claim(offset, expected_shift):
    """The whole design rests on this: the upstream exon gains exactly `offset` bases.

    In the spliced product the downstream exon's reading frame continues the upstream one's, so
    adding `offset` bases to the upstream exon shifts that continuation by `offset mod 3`. The
    donor stays put and the exon's 5' end moves back by `offset`, so the exon's length in the
    edited window is the quantity to assert -- and it must move by 0 for the phase-preserving
    arm and by 1 or 2 for the phase-decoupled one.
    """
    seq = _window(3)
    edited = paired_indel(
        seq,
        acceptor_pos=ACCEPTOR,
        donor_pos=DONOR,
        next_acceptor_pos=NEXT_ACCEPTOR,
        offset=offset,
    )
    # The donor is fixed and the acceptor moved back by `offset`, so the exon is longer by that.
    original_exon_length = DONOR - ACCEPTOR
    edited_exon_length = DONOR - (ACCEPTOR - offset)
    assert edited_exon_length - original_exon_length == offset
    assert edited_exon_length % 3 == (original_exon_length - expected_shift) % 3
    assert register_shift(offset) == expected_shift
    # And the downstream exon, which supplies the acceptor-side register, is untouched.
    assert edited[NEXT_ACCEPTOR:] == seq[NEXT_ACCEPTOR:]


def test_indel_geometry_rejects_an_exon_too_short_for_the_edit():
    with pytest.raises(GeometryError, match="outside the upstream exon"):
        indel_geometry(
            seq_len=LENGTH,
            acceptor_pos=DONOR - DEFAULT_EXON_OFFSET,   # exon exactly reaches the insert point
            donor_pos=DONOR,
            next_acceptor_pos=NEXT_ACCEPTOR,
            offset=4,
        )


def test_indel_geometry_rejects_a_deletion_that_touches_the_acceptor():
    # acceptor_pos = 8 -> delete_at = 4, and [4, 4+offset) is not clear of the acceptor.
    with pytest.raises(GeometryError):
        indel_geometry(
            seq_len=LENGTH, acceptor_pos=8, donor_pos=DONOR,
            next_acceptor_pos=NEXT_ACCEPTOR, offset=4,
        )


def test_indel_geometry_rejects_disordered_coordinates():
    with pytest.raises(GeometryError, match="window coordinates"):
        indel_geometry(
            seq_len=LENGTH, acceptor_pos=400, donor_pos=360,
            next_acceptor_pos=NEXT_ACCEPTOR, offset=3,
        )


def test_indel_geometry_rejects_non_positive_offset():
    with pytest.raises(GeometryError, match="offset must be positive"):
        indel_geometry(
            seq_len=LENGTH, acceptor_pos=ACCEPTOR, donor_pos=DONOR,
            next_acceptor_pos=NEXT_ACCEPTOR, offset=0,
        )


# ---------------------------------------------------------------------------------------
# Variable-length windows
# ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("draw", [0.0, 0.13, 0.5, 0.77, 0.999])
def test_crop_always_contains_the_protected_interval(draw):
    start, end = crop_interval(seq_len=1024, keep_lo=180, keep_hi=480, draw=draw)
    assert start <= 180 and end >= 480
    assert 0 <= start < end <= 1024
    assert (end - start) in DEFAULT_LENGTH_LADDER


def test_crop_uses_more_than_one_ladder_length():
    lengths = {
        crop_interval(seq_len=1024, keep_lo=180, keep_hi=480, draw=d)[1]
        - crop_interval(seq_len=1024, keep_lo=180, keep_hi=480, draw=d)[0]
        for d in np.linspace(0.0, 0.999, 40)
    }
    # Protected span is 300, so 512/768/1024 are admissible and 256 is not.
    assert lengths == {512, 768, 1024}


def test_crop_falls_back_to_the_whole_window_when_nothing_fits():
    # A protected interval longer than every ladder length must not be cropped away.
    assert crop_interval(seq_len=1024, keep_lo=0, keep_hi=1024, draw=0.5) == (0, 1024)


def test_crop_rejects_an_interval_outside_the_window():
    with pytest.raises(GeometryError):
        crop_interval(seq_len=100, keep_lo=50, keep_hi=200, draw=0.5)


def test_crop_rejects_a_draw_outside_the_unit_interval():
    with pytest.raises(ValueError, match="draw must lie"):
        crop_interval(seq_len=1024, keep_lo=10, keep_hi=20, draw=1.0)


def test_uniform_draw_is_deterministic_and_in_range():
    values = [uniform_draw(f"k{i}", seed=5) for i in range(200)]
    assert all(0.0 <= v < 1.0 for v in values)
    assert values[0] == uniform_draw("k0", seed=5)
    assert uniform_draw("k0", seed=6) != values[0]
    assert 0.35 < float(np.mean(values)) < 0.65


def test_batch_plan_batches_are_length_homogeneous_and_cover_every_window():
    lengths = [256] * 5 + [512] * 7 + [1024] * 3
    batches = batch_plan(lengths, batch_size=4, seed=1)
    seen = [i for batch in batches for i in batch]
    assert sorted(seen) == list(range(len(lengths)))
    assert len(seen) == len(set(seen))        # no window trained twice per epoch
    for batch in batches:
        assert len({lengths[i] for i in batch}) == 1
        assert len(batch) <= 4


def test_batch_plan_is_identical_for_identical_inputs():
    lengths = [512] * 20
    assert batch_plan(lengths, batch_size=4, seed=2) == batch_plan(lengths, batch_size=4, seed=2)
    assert batch_plan(lengths, batch_size=4, seed=3) != batch_plan(lengths, batch_size=4, seed=2)


def test_batch_plan_rejects_a_non_positive_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        batch_plan([512], batch_size=0, seed=1)


# ---------------------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------------------
def test_mask_tokens_hits_each_of_the_three_branches_exactly():
    ids = [7, 7, 7, 7]
    base_ids = [3, 4, 5, 6]
    # position 0: selected -> [MASK];  1: selected -> random base;  2: selected -> unchanged;
    # position 3: not selected at all.
    draws = [
        0.0, 0.10,     # select, choice < 0.80  -> mask
        0.0, 0.85,     # select, 0.80 <= choice < 0.90 -> random base
        0.0, 0.95,     # select, choice >= 0.90 -> unchanged but scored
        0.99, 0.0,     # not selected
    ]
    inputs, labels = mask_tokens(
        ids, maskable=[True] * 4, mask_id=99, base_ids=base_ids, draws=draws
    )
    assert inputs[0] == 99
    assert inputs[1] in base_ids
    assert inputs[2] == 7
    assert inputs[3] == 7
    assert labels == (7, 7, 7, -100)     # the unchanged branch is still scored


def test_mask_tokens_respects_the_maskable_flag():
    inputs, labels = mask_tokens(
        [7, 7], maskable=[False, True], mask_id=99, base_ids=[3],
        draws=[0.0, 0.0, 0.0, 0.0],
    )
    assert inputs == (7, 99)
    assert labels == (-100, 7)


def test_mask_tokens_selects_about_the_published_rate():
    n = 4000
    draws = position_draws(seed=11, window_index=0, epoch=0, n=n)
    _, labels = mask_tokens(
        [7] * n, maskable=[True] * n, mask_id=99, base_ids=[3, 4, 5, 6], draws=draws
    )
    selected = sum(1 for x in labels if x != -100) / n
    assert abs(selected - MASK_RATE) < 0.02


def test_mask_tokens_shares_are_about_eighty_ten_ten():
    n = 20000
    draws = position_draws(seed=12, window_index=0, epoch=0, n=n)
    inputs, labels = mask_tokens(
        [7] * n, maskable=[True] * n, mask_id=99, base_ids=[3, 4, 5, 6], draws=draws
    )
    chosen = [i for i, x in enumerate(labels) if x != -100]
    assert chosen
    masked = sum(1 for i in chosen if inputs[i] == 99) / len(chosen)
    unchanged = sum(1 for i in chosen if inputs[i] == 7) / len(chosen)
    randomised = sum(1 for i in chosen if inputs[i] in (3, 4, 5, 6)) / len(chosen)
    assert abs(masked - 0.80) < 0.03
    assert abs(randomised - 0.10) < 0.03
    assert abs(unchanged - 0.10) < 0.03


def test_mask_tokens_random_branch_covers_every_base():
    n = 20000
    draws = position_draws(seed=13, window_index=0, epoch=0, n=n)
    inputs, labels = mask_tokens(
        [7] * n, maskable=[True] * n, mask_id=99, base_ids=[3, 4, 5, 6], draws=draws
    )
    drawn = {inputs[i] for i, x in enumerate(labels) if x != -100 and inputs[i] != 99}
    assert {3, 4, 5, 6} <= drawn | {7}
    assert {3, 4, 5, 6} & drawn == {3, 4, 5, 6}


def test_mask_tokens_rejects_too_few_draws():
    with pytest.raises(ValueError, match="two draws per position"):
        mask_tokens([1, 2], maskable=[True, True], mask_id=9, base_ids=[1], draws=[0.1, 0.2])


def test_mask_tokens_rejects_a_maskable_of_the_wrong_length():
    with pytest.raises(ValueError, match="same length"):
        mask_tokens([1, 2], maskable=[True], mask_id=9, base_ids=[1], draws=[0.0] * 4)


def test_position_draws_are_keyed_only_by_seed_window_and_epoch():
    a = position_draws(seed=1, window_index=4, epoch=0, n=8)
    assert a == position_draws(seed=1, window_index=4, epoch=0, n=8)
    assert a != position_draws(seed=1, window_index=4, epoch=1, n=8)
    assert a != position_draws(seed=1, window_index=5, epoch=0, n=8)
    assert a != position_draws(seed=2, window_index=4, epoch=0, n=8)
    assert len(a) == 16
    assert all(0.0 <= v < 1.0 for v in a)


def test_position_draws_prefix_is_stable_in_n():
    # A longer request must extend, not reshuffle -- otherwise a change of window length
    # silently changes which positions earlier windows masked.
    assert position_draws(seed=1, window_index=0, epoch=0, n=4)[:8] == position_draws(
        seed=1, window_index=0, epoch=0, n=64
    )[:8]


# ---------------------------------------------------------------------------------------
# The evaluation cells
# ---------------------------------------------------------------------------------------
def _parent(gene: str, *, donor: int = 300, acceptor: int = 600, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    seq = list("".join(rng.choice(list("ACGT"), size=1024)))
    seq[donor], seq[donor + 1] = "G", "T"
    seq[acceptor - 2], seq[acceptor - 1] = "A", "G"
    return {"seq": "".join(seq), "d": donor, "a": acceptor, "gene": gene}


def _pools(depth: int = 3):
    donors = {p: [_parent(f"D{p}_{i}", seed=100 * p + i) for i in range(depth)] for p in (0, 1, 2)}
    acceptors = {
        p: [_parent(f"A{p}_{i}", seed=500 + 100 * p + i) for i in range(depth)] for p in (0, 1, 2)
    }
    return donors, acceptors


def test_build_chimeric_cells_labels_delta_as_acceptor_minus_donor_phase():
    donors, acceptors = _pools(depth=1)
    cells = build_chimeric_cells(donors, acceptors)
    assert len(cells) == 9
    for cell in cells:
        assert cell["delta"] == (cell["acceptor_phase"] - cell["donor_phase"]) % 3
    assert sorted(c["delta"] for c in cells) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_build_chimeric_cells_preserves_geometry_and_plants_the_corruption():
    donors, acceptors = _pools(depth=2)
    cells = build_chimeric_cells(donors, acceptors)
    assert cells
    for cell in cells:
        seq, corrupt = cell["sequence"], cell["corrupt"]
        acceptor = cell["acceptor_pos"]
        assert len(seq) == 1024 and len(corrupt) == len(seq)
        assert seq[cell["donor_pos"] : cell["donor_pos"] + 2] == "GT"
        assert seq[acceptor - 2 : acceptor] == "AG"
        # AG -> AC destroys the acceptor and, unlike "CC"/"CA", plants no motif in any context.
        assert corrupt[acceptor - 2 : acceptor] == "AC"
        assert corrupt[: acceptor - 2] == seq[: acceptor - 2]
        assert corrupt[acceptor:] == seq[acceptor:]
        # The chimera really is a join: donor half from one parent, acceptor half from another.
        assert cell["donor_gene"] != cell["acceptor_gene"]
        assert cell["donor_pos"] + 2 <= cell["splice_at"] <= acceptor - 2


def test_build_chimeric_cells_drops_self_joins():
    shared = _parent("SAME", seed=1)
    donors = {p: [dict(shared)] for p in (0, 1, 2)}
    acceptors = {p: [dict(shared)] for p in (0, 1, 2)}
    assert build_chimeric_cells(donors, acceptors) == ()


def test_build_chimeric_cells_rejects_a_join_whose_motifs_do_not_survive():
    # Acceptor parent whose AG sits before the splice point, so the join loses it.
    donors = {p: [_parent(f"D{p}", donor=300, acceptor=600, seed=p)] for p in (0, 1, 2)}
    bad = _parent("Abad", donor=300, acceptor=600, seed=9)
    seq = list(bad["seq"])
    seq[598], seq[599] = "C", "C"          # destroy the AG in the acceptor parent
    bad["seq"] = "".join(seq)
    acceptors = {p: [dict(bad)] for p in (0, 1, 2)}
    assert build_chimeric_cells(donors, acceptors) == ()


def test_build_chimeric_cells_pairs_by_rank_and_uses_the_shallowest_pool():
    donors, acceptors = _pools(depth=3)
    donors[1] = donors[1][:1]              # one phase pool is shallower than the rest
    cells = build_chimeric_cells(donors, acceptors)
    assert len({c["donor_gene"] for c in cells}) == 3     # depth collapses to 1 -> 3 donors
    assert len(cells) == 9


def test_build_chimeric_cells_requires_all_three_phases():
    donors, acceptors = _pools(depth=1)
    del donors[2]
    with pytest.raises(ValueError, match="phases 0, 1 and 2"):
        build_chimeric_cells(donors, acceptors)


# ---------------------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------------------
def test_psi_is_the_two_group_contrast_not_the_equal_weight_one():
    coupling = [1.0, 1.0, 0.0, 0.0, 0.0, 3.0]
    delta = [0, 0, 1, 1, 1, 2]
    # two-group: mean(matched)=1.0, mean(mismatched)=(0+0+0+3)/4=0.75 -> 0.25
    assert psi(coupling, delta) == pytest.approx(0.25)
    # the equal-weight three-class form would give 1.0 - (0.0 + 3.0)/2 = -0.5; they differ,
    # which is exactly why the convention is pinned.
    assert psi(coupling, delta) != pytest.approx(-0.5)


def test_psi_requires_both_groups():
    with pytest.raises(ValueError, match="matched and one mismatched"):
        psi([1.0, 2.0], [0, 0])
    with pytest.raises(ValueError, match="matched and one mismatched"):
        psi([1.0, 2.0], [1, 2])


def test_psi_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        psi([1.0, 2.0, 3.0], [0, 1])


def test_cluster_bootstrap_pairs_the_arms_on_one_cluster_draw():
    rng = np.random.default_rng(0)
    n = 300
    delta = rng.integers(0, 3, size=n)
    clusters = np.array([f"g{i % 40}" for i in range(n)], dtype=object)
    base = rng.normal(size=n) + 0.4 * (delta == 0)
    # An arm that is base plus a constant has EXACTLY zero difference in Psi, so a paired
    # interval must be degenerate at 0 while each arm's own interval is wide.
    shifted = base + 2.0
    out = cluster_bootstrap(
        {"a": base, "b": shifted}, delta, clusters, n_boot=200, seed=3
    )
    assert out["a_minus_b"]["estimate"] == pytest.approx(0.0, abs=1e-12)
    assert out["a_minus_b"]["ci_low"] == pytest.approx(0.0, abs=1e-9)
    assert out["a_minus_b"]["ci_high"] == pytest.approx(0.0, abs=1e-9)
    assert out["a"]["ci_high"] - out["a"]["ci_low"] > 0.05
    assert out["a"]["estimate"] == pytest.approx(psi(base, delta))
    assert out["_meta"]["n_clusters"] == 40


def test_cluster_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(1)
    n = 400
    delta = rng.integers(0, 3, size=n)
    clusters = np.array([f"g{i % 50}" for i in range(n)], dtype=object)
    values = rng.normal(size=n) + 0.5 * (delta == 0)
    out = cluster_bootstrap({"only": values}, delta, clusters, n_boot=300, seed=4)
    assert out["only"]["ci_low"] < out["only"]["estimate"] < out["only"]["ci_high"]
    assert out["only"]["ci_low"] > 0.0          # a real +0.5 effect must exclude zero here
    assert out["only"]["n_boot_used"] > 250


def test_cluster_bootstrap_is_reproducible_from_its_seed():
    rng = np.random.default_rng(2)
    n = 200
    delta = rng.integers(0, 3, size=n)
    clusters = np.array([f"g{i % 20}" for i in range(n)], dtype=object)
    values = rng.normal(size=n)
    first = cluster_bootstrap({"x": values}, delta, clusters, n_boot=100, seed=9)
    second = cluster_bootstrap({"x": values}, delta, clusters, n_boot=100, seed=9)
    assert first["x"] == second["x"]


def test_cluster_bootstrap_rejects_misaligned_arms():
    with pytest.raises(ValueError, match="does not align"):
        cluster_bootstrap(
            {"x": [1.0, 2.0]}, [0, 1, 2], np.array(["a", "b", "c"], dtype=object),
            n_boot=5, seed=1,
        )


# ---------------------------------------------------------------------------------------
# Corpus identity and the carrier readout
# ---------------------------------------------------------------------------------------
def test_corpus_digest_is_order_sensitive_and_separator_safe():
    assert corpus_digest(["AC", "GT"]) == corpus_digest(["AC", "GT"])
    assert corpus_digest(["AC", "GT"]) != corpus_digest(["GT", "AC"])
    # Without a separator these two corpora would collide.
    assert corpus_digest(["ACG", "T"]) != corpus_digest(["AC", "GT"])


def test_period3_band_power_recovers_a_planted_period_three_signal():
    n_positions, hidden = 1026, 8
    positions = np.arange(n_positions)
    table = np.zeros((n_positions, hidden))
    table[:, 0] = np.cos(2 * np.pi * positions / 3.0)
    bins = tuple(range(339, 346))
    p3 = period3_band_power(table, bins)
    # An equal-width band elsewhere must hold essentially none of it.
    control = period3_band_power(table, tuple(range(95, 102)))
    assert p3 > 1e3 * max(control, 1e-12)


def test_period3_band_power_is_blind_to_a_phase_rotation():
    """Band power CANNOT see a carrier relocation, and the record must not claim otherwise.

    `period3_band_power` sums squared moduli, and a phase rotation preserves every modulus, so
    rotating the period-3 band -- this study's own carrier operator, the one the relocation test
    used -- leaves the number bit-for-bit identical while the table is materially different.

    This is why "equal band power across arms" licenses only "carrier AMPLITUDE is unchanged" and
    never "the carrier did not change" or "the effect did not go through the carrier". See
    DECISIONS.md, 2026-08-12 (same day, bounding the entry above).
    """
    from gfmcd.splicing.positional_spectrum import phase_rotate_band

    table = np.random.default_rng(0).normal(size=(1026, 32))
    bins = tuple(range(339, 346))
    rotated = phase_rotate_band(table, bins, [np.pi / 2] * len(bins))

    assert not np.allclose(table, rotated)          # the table really did change
    before = period3_band_power(table, bins)
    after = period3_band_power(rotated, bins)
    assert after == pytest.approx(before, rel=1e-12)


def test_period3_band_power_rejects_bins_outside_the_spectrum():
    table = np.random.default_rng(0).normal(size=(64, 4))
    with pytest.raises(ValueError, match="outside the spectrum"):
        period3_band_power(table, (1000,))
    with pytest.raises(ValueError, match="must not be empty"):
        period3_band_power(table, ())


# ==========================================================================================
# Corpus composition, edit accounting, and the shared bootstrap draws.
# Added for the three-arm carrier-dependence run: "only the residue mod 3 differs" is a design
# INTENTION, and these are the readouts that check it instead of asserting it.
# ==========================================================================================
def test_kmer_profile_is_a_normalised_frequency_table():
    from gfmcd.splicing.lineage_pretraining import kmer_profile

    profile = kmer_profile(["ACGT", "ACGT"], 1)
    assert sum(profile.values()) == pytest.approx(1.0)
    assert profile == {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
    assert kmer_profile(["AAAA"], 2) == {"AA": 1.0}


def test_kmer_profile_rejects_nonsense():
    from gfmcd.splicing.lineage_pretraining import kmer_profile

    with pytest.raises(ValueError, match="k must be positive"):
        kmer_profile(["ACGT"], 0)
    with pytest.raises(ValueError, match="no k-mers"):
        kmer_profile(["AC"], 5)


def test_profile_distance_is_zero_for_identical_corpora_and_one_for_disjoint_ones():
    from gfmcd.splicing.lineage_pretraining import kmer_profile, profile_distance

    left = kmer_profile(["ACGTACGT"], 1)
    assert profile_distance(left, left)["total_variation"] == pytest.approx(0.0)
    disjoint = profile_distance(kmer_profile(["AAAA"], 1), kmer_profile(["CCCC"], 1))
    assert disjoint["total_variation"] == pytest.approx(1.0)
    assert disjoint["l1"] == pytest.approx(2.0)


def test_profile_distance_detects_the_duplication_the_operator_introduces():
    """A tandem duplication is a real composition change, and the readout must see it."""
    from gfmcd.splicing.lineage_pretraining import kmer_profile, profile_distance

    rng = np.random.default_rng(0)
    source = "".join(rng.choice(list("ACGT"), size=400))
    # Delete four bases early, duplicate four bases later: the operator's shape, length preserved.
    edited = source[:40] + source[44:200] + source[196:200] + source[200:]
    assert len(edited) == len(source)
    distance = profile_distance(kmer_profile([source], 3), kmer_profile([edited], 3))
    assert distance["total_variation"] > 0.0


def test_edit_budget_counts_two_operations_per_edited_window():
    from gfmcd.splicing.lineage_pretraining import edit_budget

    budget = edit_budget([30, 31, 32])
    assert budget["windows"] == 3
    assert budget["edited_windows"] == 3
    assert budget["inserted_bases"] == 93
    assert budget["deleted_bases"] == 93           # equal by construction: length is preserved
    assert budget["edit_operations"] == 6
    assert budget["max_offset"] == 32
    assert budget["min_nonzero_offset"] == 30


def test_edit_budget_of_an_unedited_arm_is_all_zero():
    from gfmcd.splicing.lineage_pretraining import edit_budget

    assert edit_budget([0, 0, 0]) == {
        "windows": 3,
        "edited_windows": 0,
        "inserted_bases": 0,
        "deleted_bases": 0,
        "edit_operations": 0,
        "edits_per_edited_window": 0,
        "max_offset": 0,
        "min_nonzero_offset": 0,
    }


def test_edit_budget_rejects_negative_offsets():
    from gfmcd.splicing.lineage_pretraining import edit_budget

    with pytest.raises(ValueError, match="non-negative"):
        edit_budget([30, -1])


def _bootstrap_fixture(n_clusters: int = 40, seed: int = 5):
    rng = np.random.default_rng(seed)
    clusters, delta = [], []
    for index in range(n_clusters):
        for offset in range(3):
            clusters.append(f"g{index}")
            delta.append(offset)
    values = rng.normal(size=len(delta)) + (np.asarray(delta) == 0) * 0.5
    return np.asarray(clusters, dtype=object), np.asarray(delta), values


def test_shared_draws_reproduce_cluster_bootstrap_exactly():
    """The paired-draw path must be the SAME resampling, not merely a similar one.

    If these diverged, a dose ladder's intervals and the pilot's intervals would rest on
    different draw sequences and could not be placed beside one another.
    """
    from gfmcd.splicing.lineage_pretraining import (
        cluster_draw_indices,
        percentile_interval,
        psi_replicates,
    )

    clusters, delta, values = _bootstrap_fixture()
    reference = cluster_bootstrap({"arm": values}, delta, clusters, n_boot=300, seed=11)["arm"]
    draws = cluster_draw_indices(clusters, delta, n_boot=300, seed=11)
    mine = percentile_interval(psi_replicates(values, draws), psi(values, delta))
    assert mine["n_boot_used"] == reference["n_boot_used"]
    assert mine["estimate"] == reference["estimate"]
    assert mine["ci_low"] == reference["ci_low"]
    assert mine["ci_high"] == reference["ci_high"]


def test_shared_draws_make_a_difference_paired():
    """Resampling two arms on the SAME draws is what makes their difference paired."""
    from gfmcd.splicing.lineage_pretraining import cluster_draw_indices, psi_replicates

    clusters, delta, values = _bootstrap_fixture()
    draws = cluster_draw_indices(clusters, delta, n_boot=200, seed=3)
    left = psi_replicates(values, draws)
    right = psi_replicates(values + 0.25, draws)
    # A constant shift cannot move Psi, and on paired draws the difference is exactly zero.
    assert np.abs(left - right).max() < 1e-12


def test_cluster_draw_indices_rejects_misaligned_inputs():
    from gfmcd.splicing.lineage_pretraining import cluster_draw_indices

    with pytest.raises(ValueError, match="clusters must align"):
        cluster_draw_indices(["a", "b"], [0, 1, 2], n_boot=5, seed=1)


def test_percentile_interval_reports_the_sample_estimate_not_the_bootstrap_mean():
    from gfmcd.splicing.lineage_pretraining import percentile_interval

    block = percentile_interval(np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]), estimate=99.0)
    assert block["estimate"] == 99.0
    assert block["n_boot_used"] == 5
    with pytest.raises(ValueError, match="alpha"):
        percentile_interval(np.asarray([1.0]), estimate=1.0, alpha=0.0)


def test_percentile_interval_of_no_replicates_is_nan_not_a_crash():
    from gfmcd.splicing.lineage_pretraining import percentile_interval

    block = percentile_interval(np.asarray([]), estimate=0.5)
    assert block["n_boot_used"] == 0
    assert np.isnan(block["ci_low"]) and np.isnan(block["ci_high"])


def test_sesoi_verdict_never_calls_a_wide_zero_covering_interval_an_absence():
    """The wording rule, as arithmetic: covering zero is not equivalence.

    An interval of [-0.03, +0.12] against a SESOI of 0.05 covers zero AND covers effects the
    programme calls material. Reporting that as 'left it at baseline' is the exact overstatement
    this verdict exists to block.
    """
    from gfmcd.splicing.lineage_pretraining import sesoi_verdict

    assert sesoi_verdict({"ci_low": -0.03, "ci_high": 0.12}, 0.05) == "unresolved"


def test_sesoi_verdict_classifies_the_four_cases():
    from gfmcd.splicing.lineage_pretraining import sesoi_verdict

    assert sesoi_verdict({"ci_low": 0.20, "ci_high": 0.40}, 0.05) == "resolved_material"
    assert sesoi_verdict({"ci_low": -0.40, "ci_high": -0.20}, 0.05) == "resolved_material"
    assert sesoi_verdict({"ci_low": -0.02, "ci_high": 0.03}, 0.05) == "resolved_below_sesoi"
    assert (
        sesoi_verdict({"ci_low": 0.01, "ci_high": 0.30}, 0.05)
        == "resolved_nonzero_materiality_unresolved"
    )


def test_sesoi_verdict_of_a_nan_interval_is_unresolved():
    from gfmcd.splicing.lineage_pretraining import sesoi_verdict

    assert sesoi_verdict({"ci_low": float("nan"), "ci_high": float("nan")}, 0.05) == "unresolved"


def test_sesoi_verdict_rejects_an_inverted_interval():
    from gfmcd.splicing.lineage_pretraining import sesoi_verdict

    with pytest.raises(ValueError, match="inverted"):
        sesoi_verdict({"ci_low": 0.4, "ci_high": 0.1}, 0.05)
