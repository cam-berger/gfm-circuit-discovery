from __future__ import annotations

from collections import Counter

import pytest

from gfmcd.splicing.coupling import CouplingExample
from gfmcd.splicing.p3_annotation import AnnotationFeature, AnnotationSafetyIndex
from gfmcd.splicing.p3_closure import ContextBlock, rotate_cyclic
from gfmcd.splicing.p3_cohort import (
    P3CohortConfig,
    annotation_safe_non_cds_candidate_pools,
    normalized_hamming,
    normalized_trimer_divergence,
)


def test_rotation_distance_metrics_are_normalized_and_exact():
    sequence = "A" * 10 + "ACGTAC" + "C" * 10
    block = ContextBlock(10, 16)
    variant = rotate_cyclic(sequence, block, 1, "left")
    assert normalized_hamming(sequence, variant, block) == pytest.approx(1.0)
    assert 0 <= normalized_trimer_divergence(sequence, variant, block) <= 1
    assert Counter(sequence[10:16]) == Counter(variant[10:16])


def test_cohort_config_refuses_design_drift():
    P3CohortConfig().validate()
    with pytest.raises(ValueError, match="exactly 2..10"):
        P3CohortConfig(rotation_amounts=(2, 3, 4)).validate()
    with pytest.raises(ValueError, match="multiple of three"):
        P3CohortConfig(block_size=59).validate()


def test_non_cds_pool_is_exon_local_and_cds_free():
    example = CouplingExample(
        example_id="e", sequence="A" * 300, acceptor_pos=80, donor_pos=200,
        chrom="chr1", strand="+", gene_id="g", split="train", exon_length=120,
        intron_length_down=60, n_isoforms_donor=1, n_isoforms_acceptor=1,
        gene_type="protein_coding", next_acceptor_pos=260,
        metadata={
            "genomic_window_start": 1000,
            "genomic_window_end": 1300,
            "genomic_exon_start": 1080,
            "genomic_exon_end": 1200,
        },
    )
    safety = AnnotationSafetyIndex(
        exons={"chr1": [AnnotationFeature(
            "chr1", 1080, 1200, "+", "g", "t", "exon", None
        )]},
        cds={"chr1": []},
    )
    pools, audit = annotation_safe_non_cds_candidate_pools(
        example, safety=safety, config=P3CohortConfig(block_size=36, n_blocks=2)
    )
    assert pools["non_cds_exon"]
    assert pools["upstream_intron"]
    assert audit["n_non_cds_exon_annotation_safe"] > 0
