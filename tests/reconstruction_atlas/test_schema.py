from __future__ import annotations

import pytest

from gfmcd.reconstruction_atlas.schema import (
    AnnotatedSequence,
    BiologicalElement,
    ContextRegion,
    Evidence,
    EvidenceType,
    Interval,
)


def example() -> AnnotatedSequence:
    evidence = Evidence("GENCODE", "v50", EvidenceType.DERIVED, "tx1:donor")
    donor = BiologicalElement(
        "tx1:donor",
        "splice.donor",
        Interval(4, 6),
        evidence,
        sequence="GT",
        paired_element_ids=("tx1:acceptor",),
    )
    acceptor = BiologicalElement(
        "tx1:acceptor",
        "splice.acceptor",
        Interval(8, 10),
        Evidence("GENCODE", "v50", EvidenceType.DERIVED, "tx1:acceptor"),
        sequence="AG",
        paired_element_ids=("tx1:donor",),
    )
    region = ContextRegion(
        "donor:downstream",
        "paired_biological_element",
        Interval(8, 10),
        ("tx1:donor",),
    )
    return AnnotatedSequence(
        example_id="example-1",
        sequence="AAAAGTCCAGTT",
        assembly="GRCh38",
        chrom="chr1",
        genomic_start=100,
        genomic_end=112,
        strand="+",
        sequence_orientation="transcription_5_to_3",
        split="development",
        homology_cluster="gene-1",
        elements=(donor, acceptor),
        context_regions=(region,),
    )


def test_roundtrip_preserves_nucleotide_labels() -> None:
    record = example()
    assert AnnotatedSequence.from_dict(record.to_dict()) == record
    assert record.element("tx1:donor").interval == Interval(4, 6)


def test_element_sequence_must_match_window() -> None:
    value = example().to_dict()
    value["elements"][0]["sequence"] = "AG"
    with pytest.raises(ValueError, match="disagrees"):
        AnnotatedSequence.from_dict(value)


def test_context_cannot_reference_an_absent_target() -> None:
    value = example().to_dict()
    value["context_regions"][0]["target_element_ids"] = ["ghost"]
    with pytest.raises(ValueError, match="absent targets"):
        AnnotatedSequence.from_dict(value)


def test_genomic_span_must_equal_sequence_length() -> None:
    value = example().to_dict()
    value["genomic_end"] += 1
    with pytest.raises(ValueError, match="length"):
        AnnotatedSequence.from_dict(value)
