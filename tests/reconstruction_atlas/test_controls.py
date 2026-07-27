from __future__ import annotations

from gfmcd.reconstruction_atlas.controls import (
    MatchingSpec,
    SiteFeatures,
    build_matched_sets,
)
from gfmcd.reconstruction_atlas.schema import (
    ControlFamily,
    Interval,
    SiteReference,
)


def features(
    example: str,
    start: int,
    sequence: str,
    *,
    annotated: bool,
) -> SiteFeatures:
    site = SiteReference(
        example, Interval(start, start + len(sequence)), "promoter.tata"
    )
    return SiteFeatures(
        site=site,
        sequence=sequence,
        gc_fraction=0.0,
        cpg_fraction=0.0,
        dinucleotide_frequencies={"TA": 1.0},
        token_count=2,
        covered_nucleotides=len(sequence),
        geometry_signature=((0, 2), (2, len(sequence))),
        overlaps_annotation=annotated,
    )


def test_control_families_are_constructed_separately() -> None:
    target = features("target", 10, "TATA", annotated=True)
    candidates = {
        ControlFamily.SAME_MOTIF_UNANNOTATED: [
            features("other", 20, "TATA", annotated=False)
        ],
        ControlFamily.COMPOSITION_TOKEN_MATCHED: [
            features("third", 30, "AAAA", annotated=False)
        ],
        ControlFamily.SAME_SEQUENCE_NON_SITE: [
            features("target", 30, "AAAA", annotated=False)
        ],
    }
    matched = build_matched_sets(
        [target],
        candidates,
        MatchingSpec(
            gc_caliper=1,
            cpg_caliper=1,
            dinucleotide_l1_caliper=2,
        ),
    )
    assert len(matched) == 1
    assert {control.family for control in matched[0].controls} == set(candidates)


def test_same_motif_control_cannot_overlap_an_annotation() -> None:
    target = features("target", 10, "TATA", annotated=True)
    candidates = {
        ControlFamily.SAME_MOTIF_UNANNOTATED: [
            features("other", 20, "TATA", annotated=True)
        ],
        ControlFamily.COMPOSITION_TOKEN_MATCHED: [
            features("third", 30, "AAAA", annotated=False)
        ],
        ControlFamily.SAME_SEQUENCE_NON_SITE: [
            features("target", 30, "AAAA", annotated=False)
        ],
    }
    try:
        build_matched_sets(
            [target],
            candidates,
            MatchingSpec(gc_caliper=1, cpg_caliper=1, dinucleotide_l1_caliper=2),
        )
    except ValueError as exc:
        assert "same_motif_unannotated" in str(exc)
    else:
        raise AssertionError("annotated same-motif control was accepted")
