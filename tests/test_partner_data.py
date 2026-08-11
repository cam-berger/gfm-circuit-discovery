"""Memorial tests for the v2 partner dataset.

These exist because v1 shipped two fatal, silent confounds.  Each test below is a
tripwire for one of them.  A test here failing means the science is invalid, not
that a refactor broke a unit -- treat it that way.

Requires the real GENCODE GTF + hg38 (skipped if absent).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gfmcd.splicing.partner_data import (
    PartnerConfig,
    _build_gene_index,
    build_partner_dataset,
    split_partner_examples,
)

GTF = Path(os.environ.get("GFMCD_GTF", "/nonexistent/gencode.gtf.gz"))
FASTA = Path(os.environ.get("GFMCD_FASTA", "/nonexistent/hg38.fa"))

pytestmark = pytest.mark.skipif(
    not (GTF.exists() and FASTA.exists()), reason="genome/annotation not available"
)


@pytest.fixture(scope="module")
def examples():
    from gfmcd.splicing.dataset import PyfaidxGenome

    return build_partner_dataset(
        GTF, PyfaidxGenome(FASTA), PartnerConfig(), allowed_chromosomes={"chr22"}
    )


@pytest.fixture(scope="module")
def gene_index():
    from gfmcd.splicing.gtf import load_transcripts

    return _build_gene_index(load_transcripts(GTF, allowed_chromosomes={"chr22"}))


def test_dataset_is_non_empty(examples):
    assert len(examples) > 100


def test_acceptor_position_is_not_constant(examples):
    """THE MEMORIAL TEST FOR THE v1 POSITION LEAK.

    v1's dataset.py did ``acceptor = cfg.left_flank`` -- a constant -- so the true
    acceptor sat at position 200 in 100.0% of examples and could be identified by
    reading a position embedding.  A no-neural-net model exploiting this scored
    PMR 0.483 against a published baseline of 0.000.

    This test would have caught it on day one.  Had it existed, v1 would never have run.
    """
    positions = {ex.acceptor_pos for ex in examples}
    assert len(positions) >= 50, (
        f"POSITION LEAK: only {len(positions)} distinct acceptor position(s). "
        "The true acceptor must not sit at a predictable offset -- randomise the flank."
    )
    # no single position may dominate
    from collections import Counter

    share = max(Counter(ex.acceptor_pos for ex in examples).values()) / len(examples)
    assert share < 0.10, f"POSITION LEAK: {share:.1%} of examples share one acceptor position"


def test_fixed_offset_would_fail_the_memorial_test():
    """The tripwire must actually discriminate: with acc_min == acc_max we reconstruct
    the v1 leak (a constant acceptor position), and the config must refuse to build it."""
    with pytest.raises(ValueError, match="positional leak"):
        PartnerConfig(acc_min=200, acc_max=200).validate()


def test_window_length_leak_is_refused():
    """The second leak: if the acceptor range is not feasible for the LONGEST exon, the
    sampling range must shrink for long exons -- and then acceptor_pos leaks exon length."""
    with pytest.raises(ValueError, match="leaks the exon length"):
        PartnerConfig(window_length=512, acc_max=300, max_exon_length=500).validate()


def test_decoys_are_real_annotated_splice_sites(examples, gene_index):
    """Every decoy must be a REAL annotated donor.  If decoys were spurious motifs
    (as in v1) an oracle site detector would solve the task and partner binding
    would never be tested."""
    from gfmcd.splicing.partner_data import _to_window

    for ex in examples[:200]:
        annotated = gene_index.donors[ex.gene_id]
        ws = ex.metadata["genomic_window_start"]
        we = ex.metadata["genomic_window_end"]
        in_window = {_to_window(g, ex.strand, ws, we) for g in annotated}
        for d in ex.real_decoy_donor_pos:
            assert d in in_window, f"{ex.example_id}: decoy {d} is not an annotated donor"


def test_decoys_are_never_attested_partners(examples, gene_index):
    """LOAD-BEARING.  A donor this acceptor pairs with in *some other isoform* is a
    POSITIVE, not a decoy.  Counting those as decoys mislabels the data and inflates
    the yield (it inflated our own first estimate from 25k to 55k)."""
    from gfmcd.splicing.partner_data import _to_window

    for ex in examples[:200]:
        ws = ex.metadata["genomic_window_start"]
        we = ex.metadata["genomic_window_end"]
        # recover the acceptor's genomic anchor from the window mapping
        acc_g = ws + ex.acceptor_pos if ex.strand == "+" else we - 1 - ex.acceptor_pos
        attested = gene_index.attested[(ex.gene_id, acc_g)]
        attested_win = {_to_window(g, ex.strand, ws, we) for g in attested}
        for d in ex.real_decoy_donor_pos:
            assert d not in attested_win, (
                f"{ex.example_id}: decoy {d} IS an attested partner of this acceptor "
                "in another isoform -- it is a positive, not a decoy"
            )


def test_true_donor_is_not_a_decoy(examples):
    for ex in examples:
        assert ex.true_donor_pos not in ex.real_decoy_donor_pos


def test_every_candidate_is_motif_compatible(examples):
    for ex in examples[:300]:
        assert ex.sequence[ex.acceptor_pos - 2 : ex.acceptor_pos] == "AG"
        for d in ex.candidates:
            assert ex.sequence[d : d + 2] in ("GT", "GC")


def test_task_is_a_forced_choice_among_real_sites(examples):
    """Every example must have >= 2 candidates, all real sites -- otherwise there is
    nothing for partner information to discriminate."""
    for ex in examples:
        assert len(ex.candidates) >= 2


def test_splits_are_gene_disjoint(examples):
    s = split_partner_examples(examples)
    genes = {k: {ex.gene_id for ex in v} for k, v in s.items()}
    assert not (genes["train"] & genes["val"])
    assert not (genes["train"] & genes["test"])
    assert not (genes["val"] & genes["test"])
