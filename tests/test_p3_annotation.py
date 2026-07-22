from __future__ import annotations

from gfmcd.splicing.p3_annotation import (
    AnnotationFeature,
    AnnotationSafetyIndex,
    validate_annotation_block,
    window_block_to_genomic,
)


def _feature(start, end, *, feature, gene="g1", strand="+", frame=0, transcript="t1"):
    return AnnotationFeature("chr1", start, end, strand, gene, transcript, feature, frame)


def test_window_block_to_genomic_is_strand_aware():
    assert window_block_to_genomic(10, 20, window_start=100, window_end=200, strand="+") == (110, 120)
    assert window_block_to_genomic(10, 20, window_start=100, window_end=200, strand="-") == (180, 190)


def test_clean_cds_block_requires_full_unambiguous_register():
    index = AnnotationSafetyIndex(
        exons={"chr1": [_feature(100, 300, feature="exon", frame=None)]},
        cds={"chr1": [_feature(100, 300, feature="CDS", frame=0)]},
    )
    result = validate_annotation_block(
        index, chrom="chr1", start=140, end=200, strand="+", gene_id="g1", context="cds"
    )
    assert result.passed, result.failures
    assert result.metrics["codon_slots_present"] == (0, 1, 2)


def test_cds_block_rejects_register_disagreement_and_nearby_boundary():
    index = AnnotationSafetyIndex(
        exons={"chr1": [
            _feature(100, 300, feature="exon", frame=None),
            _feature(150, 300, feature="exon", frame=None, transcript="t2"),
        ]},
        cds={"chr1": [
            _feature(100, 300, feature="CDS", frame=0),
            _feature(101, 300, feature="CDS", frame=0, transcript="t2"),
        ]},
    )
    result = validate_annotation_block(
        index, chrom="chr1", start=140, end=200, strand="+", gene_id="g1", context="cds"
    )
    assert not result.passed
    assert not result.checks["cds_register_unambiguous"]
    assert not result.checks["no_annotated_splice_boundary_nearby"]


def test_intron_block_rejects_any_alternative_exon_overlap():
    clean = AnnotationSafetyIndex(exons={"chr1": []}, cds={"chr1": []})
    assert validate_annotation_block(
        clean, chrom="chr1", start=400, end=460, strand="+", gene_id="g1",
        context="upstream_intron",
    ).passed

    alternative = AnnotationSafetyIndex(
        exons={"chr1": [_feature(430, 450, feature="exon", frame=None, transcript="alt")]},
        cds={"chr1": []},
    )
    result = validate_annotation_block(
        alternative, chrom="chr1", start=400, end=460, strand="+", gene_id="g1",
        context="upstream_intron",
    )
    assert not result.checks["no_annotated_exon_overlap_with_clearance"]


def test_non_cds_exon_gate_requires_exon_coverage_and_no_cds():
    index = AnnotationSafetyIndex(
        exons={"chr1": [_feature(100, 200, feature="exon", frame=None)]},
        cds={"chr1": []},
    )
    result = validate_annotation_block(
        index, chrom="chr1", start=120, end=150, strand="+", gene_id="g1",
        context="non_cds_exon", splice_clearance=4,
    )
    assert result.passed
    assert result.metrics["n_matching_exons"] == 1


def test_non_cds_exon_gate_rejects_any_cds_overlap():
    index = AnnotationSafetyIndex(
        exons={"chr1": [_feature(100, 200, feature="exon", frame=None)]},
        cds={"chr1": [_feature(130, 180, feature="CDS", frame=0)]},
    )
    result = validate_annotation_block(
        index, chrom="chr1", start=120, end=150, strand="+", gene_id="g1",
        context="non_cds_exon", splice_clearance=4,
    )
    assert not result.passed
    assert "no_annotated_cds_overlap" in result.failures
