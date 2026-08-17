"""Memorials for the reverse-direction (masked-acceptor) coupling readout.

Each test below can fail, and each guards a defect that actually occurred:

* ``edit_size`` -- E11's frozen primary destroyed the donor with two substitutions and the
  acceptor with one, which moved its verdict from ``RECIPROCAL_SYMMETRIC`` to
  ``RECIPROCAL_ASYMMETRIC``.  Its own test asserted only that a donor edit moved ``<= 2``
  bases, which a half-edit passes.  These assert the count itself.
* ``donor_alt_admissible`` -- R44: an alt tuple constrains what a substitution *writes*, not
  what it *creates*.
* the role-swap identity -- the whole design rests on ``CouplingTask.encode``/``margin``
  containing nothing donor-specific.  If that ever stops being true, the reversal silently
  stops being a mirror.
"""

from __future__ import annotations

import pytest

from gfmcd.splicing.coupling import MOTIF_CLEAN_DONOR_ALT, CouplingExample
from gfmcd.splicing.reverse_coupling import (
    MATCHED_TWO_SUB,
    acceptor_readout,
    destroy_donor,
    donor_alt_admissible,
    edit_size,
    matched_operators,
)


def _example(seq: str, donor_pos: int, next_acceptor_pos: int) -> CouplingExample:
    return CouplingExample(
        example_id="t", sequence=seq, acceptor_pos=4, donor_pos=donor_pos,
        chrom="chr1", strand="+", gene_id="g", split="val",
        exon_length=20, intron_length_down=100,
        n_isoforms_donor=9, n_isoforms_acceptor=9, gene_type="protein_coding",
        next_acceptor_pos=next_acceptor_pos,
    )


# --------------------------------------------------------------------------- edit size

def test_edit_size_counts_bases_actually_changed():
    """GT -> AT rewrites the T with itself: one substitution, a half-destroyed donor."""
    assert edit_size("GT", "AC") == 2
    assert edit_size("GT", "AT") == 1
    assert edit_size("GT", "CT") == 1
    assert edit_size("GT", "TT") == 1
    assert edit_size("GT", "CG") == 2
    assert edit_size("AG", "CC") == 2
    assert edit_size("AG", "AC") == 1
    assert edit_size("AG", "AT") == 1


def test_edit_size_rejects_length_mismatch():
    with pytest.raises(ValueError):
        edit_size("GT", "ACG")


def test_matched_operators_is_the_check_e11_was_missing():
    """GT->AC against AG->CC is matched at two; GT->AC against AG->AC is not."""
    assert matched_operators(*MATCHED_TWO_SUB)
    assert matched_operators("AC", "CC")
    assert matched_operators("AT", "AC")          # one each
    assert not matched_operators("AC", "AC")      # two vs one -- E11's frozen primary
    assert not matched_operators("AT", "CC")      # one vs two


# ------------------------------------------------------------------- readout position

def test_acceptor_readout_lands_on_the_G_of_the_AG():
    seq = "A" * 40 + "GT" + "C" * 50 + "AG" + "T" * 40
    d = 40
    p = 40 + 2 + 50 + 2                      # first exon base after the AG
    x = _example(seq, d, p)
    assert acceptor_readout(x) == p - 1
    assert seq[acceptor_readout(x)] == "G"
    assert seq[x.donor_pos] == "G"           # the mirror: both readouts mask a G


def test_acceptor_readout_requires_a_partner():
    x = _example("A" * 100, 40, None)
    with pytest.raises(ValueError):
        acceptor_readout(x)


# ------------------------------------------------------------------------ R44 hygiene

def test_cc_is_inadmissible_at_a_consensus_donor():
    """seq[d-1] == 'G' at 81% of donors, and CC then plants a GC donor one base upstream."""
    seq = "A" * 39 + "G" + "GT" + "C" * 60
    x = _example(seq, 40, 90)
    assert seq[x.donor_pos - 1] == "G"
    assert not donor_alt_admissible(x, "CC")
    assert donor_alt_admissible(x, MOTIF_CLEAN_DONOR_ALT)


def test_motif_clean_alt_is_admissible_in_every_flanking_context():
    for left in "ACGT":
        for right in "ACGT":
            seq = "A" * 39 + left + "GT" + right + "C" * 59
            x = _example(seq, 40, 90)
            assert donor_alt_admissible(x, MOTIF_CLEAN_DONOR_ALT), (left, right)


def test_ta_plants_a_canonical_donor_at_a_consensus_site():
    """R44's worst case: TA turns a destroyed GT into a live GT one base upstream."""
    seq = "A" * 39 + "G" + "GT" + "C" * 60
    x = _example(seq, 40, 90)
    assert not donor_alt_admissible(x, "TA")


# ------------------------------------------------------------------------ substitution

def test_destroy_donor_preserves_length_and_changes_only_the_dinucleotide():
    seq = "A" * 40 + "GT" + "C" * 58
    x = _example(seq, 40, 90)
    out = destroy_donor(x, "AC")
    assert len(out) == len(seq)
    assert out[40:42] == "AC"
    assert out[:40] == seq[:40] and out[42:] == seq[42:]
    assert sum(1 for a, b in zip(seq, out) if a != b) == edit_size("GT", "AC")


def test_destroy_donor_with_a_one_substitution_alt_leaves_the_T_standing():
    seq = "A" * 40 + "GT" + "C" * 58
    x = _example(seq, 40, 90)
    out = destroy_donor(x, "AT")
    assert out[41] == "T" == seq[41]
    assert sum(1 for a, b in zip(seq, out) if a != b) == 1


# ------------------------------------------------------- the role swap is a pure swap

def test_reverse_health_masks_the_acceptor_where_the_forward_one_masks_the_donor():
    """The reason ``reverse_mlm_health`` has to exist at all.

    ``CouplingTask.mlm_health`` masks ``donor_pos``.  Health-gating a reverse-direction
    ablation with it would score a DIFFERENT forward pass than the one whose coupling is
    being damaged -- the gate would pass while the thing it guards was destroyed.
    """
    import inspect

    from gfmcd.splicing.circuit import CouplingTask
    from gfmcd.splicing.reverse_coupling import reverse_mlm_health

    fwd = inspect.getsource(CouplingTask.mlm_health)
    rev = inspect.getsource(reverse_mlm_health)
    assert "x.donor_pos for x in ch" in fwd, "forward health no longer masks the donor"
    assert "acceptor_readout(x)" in rev, "reverse health no longer masks the acceptor"
    assert "x.donor_pos for x in ch" not in rev, "reverse health regressed to masking the donor"


def test_reverse_health_probes_are_masked_and_exclude_both_sites():
    """R8: an unmasked-probe health score once ROSE under ablation -- it measured copying.

    And a probe landing on either site would score the intervention rather than collateral
    health, so both exclusions are load-bearing. ACR11 additionally requires the reverse
    behavioral acceptor mask to remain present while the non-site probe is masked.
    """
    import inspect

    from gfmcd.splicing.reverse_coupling import reverse_mlm_health

    src = inspect.getsource(reverse_mlm_health)
    assert "task.encode(sub, readout)" in src, "reverse health no longer masks the acceptor"
    assert (
        'enc["input_ids"][row, p + task.ad.prefix_tokens] = task.mask_id' in src
    ), "non-site probes are no longer masked alongside the acceptor"
    assert "abs(p - b) > 3" in src, "probes no longer exclude the masked acceptor"
    assert "abs(p - x.donor_pos) > 3" in src, "probes no longer exclude the corrupted donor"


def test_reverse_pseudo_coupling_corrupts_the_site_it_is_given():
    """The specificity null must perturb the NON-site it is handed, never the real partner."""
    import inspect

    from gfmcd.splicing.reverse_coupling import reverse_pseudo_coupling

    src = inspect.getsource(reverse_pseudo_coupling)
    assert "zip(exs, sites)" in src, "pseudo-coupling no longer corrupts the supplied sites"
    assert "destroy_donor" not in src, "pseudo-coupling regressed to destroying the real partner"


def test_margin_readout_contains_nothing_donor_specific():
    """``CouplingTask.margin`` scores G vs max(A,C,T) at whatever positions it is given.

    The reversal is exactly the substitution of one position list for another.  If this
    source-level property ever breaks, the mirror silently stops being a mirror.
    """
    import inspect

    from gfmcd.splicing.circuit import CouplingTask

    src = inspect.getsource(CouplingTask.margin) + inspect.getsource(CouplingTask.encode)
    assert "donor_pos" not in src, "margin/encode became donor-specific; the swap is no longer pure"
    assert '"G"' in src or "'G'" in src, "margin no longer scores G"
