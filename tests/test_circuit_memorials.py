"""Memorial tests for the four pre-flight blockers (2026-07-12).

Each test below is a tripwire for a defect that produced a *publishable, wrong* number and
was caught only by an adversarial audit, hours before the sealed split was to be opened.
A failure here means the science is invalid, not that a refactor broke a unit.

THREE OF THE FOUR WERE CONTROLS THAT COULD NOT FAIL. That is the pattern to watch for: a
control is only a control if some achievable result would falsify it. Before pre-registering
a gate, state what would break it and check that the breaking result is reachable.

  R7  the (key,value) SWAP is an algebraic identity -- it "confirmed" binding on an
      untrained model
  R8  mlm_health scored the model COPYING UNMASKED INPUT -- it rose when the model was destroyed
  R9  the null p-value was 1/(n_null+1), a function of the flag, not of the data
  R11 the minus strand was silently dropped -- half the genome, never mentioned
  B-3 the "non-partner AG" null pool contained the partner
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

GTF = Path(os.environ.get("GFMCD_GTF", "/nonexistent/gencode.gtf.gz"))
FASTA = Path(os.environ.get("GFMCD_FASTA", "/nonexistent/hg38.fa"))
MODEL = (Path(__file__).resolve().parents[1]
         / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt")

needs_genome = pytest.mark.skipif(
    not (GTF.exists() and FASTA.exists()), reason="genome/annotation not available")
needs_model = pytest.mark.skipif(not MODEL.exists(), reason="SpliceBERT checkpoint not available")


# ---------------------------------------------------------------------------------------
# R7 -- a (key, value) SWAP cannot measure anything.  This one needs no genome and no data:
# it is algebra, and it must be provable on random weights in under a second.
# ---------------------------------------------------------------------------------------
def test_key_value_swap_is_an_identity():
    """THE MEMORIAL TEST FOR R7.

    out_i = sum_j softmax(q_i . k_j) v_j.  Transposing the (k, v) PAIR between two positions
    permutes the summation index j: the multiset of pairs is unchanged, so out' == out
    EXACTLY.  We shipped such a swap as a "competing-acceptor double dissociation" and read
    its 0% margin loss as proof that address and content are separable.  It was proof of
    nothing -- it reproduces on a randomly-initialised network.

    This test asserts the identity, so that anyone who reaches for a swap again is told why
    it cannot work.  Use a ONE-WAY COPY (gfmcd.splicing.circuit.relocate_proj) instead.
    """
    torch.manual_seed(0)
    B, S, H, Dh = 2, 12, 3, 4
    q = torch.randn(B, H, S, Dh)
    k = torch.randn(B, H, S, Dh)
    v = torch.randn(B, H, S, Dh)
    pa, pb = 3, 9

    def attend(k_, v_):
        w = torch.softmax((q @ k_.transpose(-1, -2)) / Dh ** 0.5, dim=-1)
        return w @ v_

    base = attend(k, v)

    k2, v2 = k.clone(), v.clone()
    k2[:, :, [pa, pb]] = k[:, :, [pb, pa]]
    v2[:, :, [pa, pb]] = v[:, :, [pb, pa]]
    both_swap = attend(k2, v2)
    assert torch.allclose(base, both_swap, atol=1e-6), (
        "A (key,value) pair swap changed the head's output. It must not -- if this fails the "
        "identity has been broken, not fixed.")

    # ...and key-only and value-only swaps are the SAME intervention, which is why their
    # margins agreed to 7 significant figures in the published run.
    k3, v3 = k.clone(), v.clone()
    k3[:, :, [pa, pb]] = k[:, :, [pb, pa]]
    key_only = attend(k3, v)
    v4 = v.clone()
    v4[:, :, [pa, pb]] = v[:, :, [pb, pa]]
    value_only = attend(k, v4)
    assert torch.allclose(key_only, value_only, atol=1e-6), (
        "key-swap and value-swap should be provably identical")
    assert not torch.allclose(base, key_only, atol=1e-4), "a single-channel swap must do something"


# ---------------------------------------------------------------------------------------
# R9 -- the null must have as many distinct members as it claims
# ---------------------------------------------------------------------------------------
def test_matched_random_head_sets_are_distinct():
    """THE MEMORIAL TEST FOR R9.

    1000 draws WITH REPLACEMENT from a 95-head universe at k=1 are 95 distinct values
    duplicated ~10.5x.  Mean-ablation is deterministic, so the duplicates carry no
    information -- but the add-one p-value divided by n+1 = 1001 anyway, and reported
    p = 0.000999.  That number was a function of the --n-null flag.  Honest value: 1/96.
    """
    from gfmcd.circuits.hf_bert import matched_random_head_sets

    class FakeAdapter:
        num_layers, num_heads = 6, 16

    sets = matched_random_head_sets(FakeAdapter(), [(3, 1)], n=1000, exclude_selected=True)
    assert len(sets) == len(set(sets)), "matched-random head sets must be DISTINCT"
    # k=1 over 95 candidates is exhaustible -- it must be ENUMERATED, not sampled
    assert len(sets) == 95, (
        f"k=1 over a 95-head universe has exactly 95 members; got {len(sets)}. A null cannot "
        "have more members than the universe it is drawn from.")

    pairs = matched_random_head_sets(FakeAdapter(), [(3, 1), (2, 15)], n=400,
                                     exclude_selected=True)
    assert len(pairs) == len(set(pairs)) == 400
    assert all((3, 1) not in s and (2, 15) not in s for s in pairs), (
        "a null that may delete the circuit is not matched")


# ---------------------------------------------------------------------------------------
# R11 + B-3 -- the dataset itself
# ---------------------------------------------------------------------------------------
@needs_genome
def test_both_strands_are_present():
    """THE MEMORIAL TEST FOR R11.

    `exons` is sorted by GENOMIC start, so on the minus strand the transcript-order next exon
    is exons[i-1], not exons[i+1].  Taking exons[i+1] there yields a negative intron length,
    which was nulled -- so next_acceptor_pos was never assigned and eligible() silently
    dropped EVERY minus-strand exon.  41,153/41,153 on chr8.  The study ran on half the genome
    for weeks without anyone saying so.
    """
    from gfmcd.splicing.circuit import DEFAULT_FASTA, DEFAULT_GTF, _dataset, eligible

    E = eligible(_dataset(DEFAULT_GTF, DEFAULT_FASTA), "val")
    strands = {x.strand for x in E}
    assert strands == {"+", "-"}, f"STRAND LEAK: eligible examples are {strands} only"
    minus = sum(1 for x in E if x.strand == "-")
    frac = minus / len(E)
    assert 0.30 < frac < 0.70, (
        f"STRAND IMBALANCE: {100*frac:.1f}% minus-strand. The genome is roughly balanced; a "
        "large skew means the transcript-order neighbour is being taken from the wrong end.")


@needs_genome
def test_partner_is_not_in_the_nonpartner_null_pool():
    """THE MEMORIAL TEST FOR B-3.

    `other_ag` is the "motif-matched NON-partner AG" pool -- the control for "is this just a
    reaction to destroying any AG?".  It excluded the exon's own acceptor but not
    next_acceptor_pos, which IS the partner under the coupling design.  The partner was a
    member of the pool in 400/400 examples, and the median-position pick landed on it ~2% of
    the time -- at which point the control corruption is BYTE-IDENTICAL to the treatment.
    The control contained the treatment.
    """
    from gfmcd.splicing.circuit import DEFAULT_FASTA, DEFAULT_GTF, _dataset, eligible

    E = eligible(_dataset(DEFAULT_GTF, DEFAULT_FASTA), "val")
    assert E, "no eligible examples"
    hits = [x.example_id for x in E
            if any(abs(p - x.next_acceptor_pos) <= 10 for p in x.other_ag)]
    assert not hits, (
        f"CONTAMINATED NULL: the partner acceptor is inside the 'non-partner AG' pool for "
        f"{len(hits)}/{len(E)} examples. Destroying it IS the treatment, so the control "
        "measures the effect it is supposed to bound.")


# ---------------------------------------------------------------------------------------
# R8 -- the health control must be able to fail
# ---------------------------------------------------------------------------------------
@needs_genome
@needs_model
def test_mlm_health_can_fail():
    """THE MEMORIAL TEST FOR R8.

    The old control took the argmax at probe positions that were NEVER MASKED and compared it
    to the base sitting right there in the input.  It scored the model copying itself: ~0.99
    for everything, and it ROSE to 0.999 with all 96 heads ablated.  It was cited as evidence
    that "ablation did not simply break the model" -- a claim it was structurally incapable of
    contradicting.

    A health metric that cannot go down when you destroy the model is not a health metric.
    """
    from gfmcd.splicing.circuit import (
        DEFAULT_FASTA, DEFAULT_GTF, DEFAULT_MODEL, CouplingTask, ablate, load_adapter,
        load_examples, pick_device,
    )

    ad = load_adapter(DEFAULT_MODEL, pick_device())
    task = CouplingTask(ad, batch=4)
    X, _ = load_examples(DEFAULT_GTF, DEFAULT_FASTA, "val", 8, task=task, tag="memorial",
                         min_effect=None)

    intact = task.mlm_health(X)
    all_heads = [(l, h) for l in range(ad.num_layers) for h in range(ad.num_heads)]
    wrecked = task.mlm_health(X, ablate(ad, all_heads))

    assert intact > 0.30, f"baseline MLM health {intact:.3f} is implausibly low"
    assert intact < 0.95, (
        f"baseline MLM health {intact:.3f} is implausibly HIGH for masked nucleotide "
        "prediction -- this is the signature of scoring UNMASKED positions, i.e. of the "
        "model copying its own input. Are the probe positions actually masked?")
    assert wrecked < intact - 0.05, (
        f"ablating ALL 96 attention heads changed MLM health from {intact:.4f} to "
        f"{wrecked:.4f}. A control that does not move when the model is destroyed cannot "
        "license the claim that an ablation left the model intact.")
