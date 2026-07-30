from types import SimpleNamespace

import numpy as np
import pytest

from gfmcd.splicing.coupling import CouplingExample
from gfmcd.splicing.native_classifier import (
    ACCEPTOR_LABEL,
    DONOR_LABEL,
    NativeClassifierTask,
    linear_cka,
    load_annotation_cohort,
    neutral_dinucleotide_position,
)


def example() -> CouplingExample:
    sequence = list("C" * 160)
    sequence[40:42] = list("GT")
    sequence[98:100] = list("AG")
    sequence[130:132] = list("AG")
    return CouplingExample(
        example_id="e1",
        sequence="".join(sequence),
        acceptor_pos=20,
        donor_pos=40,
        chrom="chr11",
        strand="+",
        gene_id="g1",
        split="val",
        exon_length=20,
        intron_length_down=60,
        n_isoforms_donor=1,
        n_isoforms_acceptor=1,
        gene_type="protein_coding",
        other_ag=(132,),
        next_acceptor_pos=100,
    )


def test_score_is_class_vs_no_site_margin_and_probability():
    task = object.__new__(NativeClassifierTask)
    task.prefix_tokens = 1
    logits = np.zeros((2, 6, 3), dtype=np.float64)
    logits[0, 2] = [1.0, 4.0, -1.0]
    logits[1, 4] = [2.0, -2.0, 3.0]

    donor = task.score(logits[:1], [1], label=DONOR_LABEL)
    acceptor = task.score(logits[1:], [3], label=ACCEPTOR_LABEL)
    assert donor.margin.tolist() == [3.0]
    assert acceptor.margin.tolist() == [1.0]
    assert 0.0 < donor.probability[0] < 1.0
    assert 0.0 < acceptor.probability[0] < 1.0


def test_neutral_control_avoids_sites_and_is_deterministic():
    item = example()
    first = neutral_dinucleotide_position(item)
    second = neutral_dinucleotide_position(item)
    assert first == second
    assert first is not None
    assert abs(first - (item.next_acceptor_pos - 2)) >= 12
    assert item.sequence[first : first + 2] not in {"AG", "GT"}


def test_linear_cka_is_one_for_scaled_copy_and_checks_shapes():
    x = np.arange(24, dtype=np.float64).reshape(6, 4)
    assert linear_cka(x, 3 * x + 7) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        linear_cka(x, np.ones((5, 4)))


def test_annotation_cohort_never_scores_a_model(monkeypatch):
    items = [example()]
    monkeypatch.setattr(
        "gfmcd.splicing.circuit._dataset",
        lambda _gtf, _fasta: items.copy(),
    )
    monkeypatch.setattr(
        "gfmcd.splicing.circuit.eligible",
        lambda values, split: list(values),
    )
    cohort = load_annotation_cohort(
        "annotation.gtf",
        "genome.fa",
        split="val",
        n=1,
        tag="frozen",
    )
    assert cohort == items
