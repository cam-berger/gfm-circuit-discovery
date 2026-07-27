from __future__ import annotations

import torch

from gfmcd.adapters.dnabert2.adapter import Dnabert2Adapter
from gfmcd.reconstruction_atlas.assembly import (
    GenomicFeature,
    assemble_annotated_windows,
    gencode_annotation_intervals_for_windows,
    gencode_features_from_gtf,
    read_normalized_bed,
    reverse_complement,
    sample_gencode_features,
    select_gencode_gene_ids,
    split_for_cluster,
)
from gfmcd.reconstruction_atlas.schema import (
    AnnotatedSequence,
    BiologicalElement,
    Evidence,
    EvidenceType,
    Interval,
)


def test_reverse_complement_is_transcription_oriented() -> None:
    assert reverse_complement("AAGTCN") == "NGACTT"


def test_mosaicbert_unpadding_map_is_explicit() -> None:
    attention = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert Dnabert2Adapter.unpadded_flat_indices(attention).tolist() == [0, 1, 3, 4, 5]
    assert Dnabert2Adapter.unpadded_row(attention, 1, 1) == 3
    try:
        Dnabert2Adapter.unpadded_row(attention, 0, 2)
    except ValueError as exc:
        assert "padding tokens" in str(exc)
    else:
        raise AssertionError("padding token received an activation row")


def test_gencode_derivation_respects_plus_and_minus_transcript_direction(
    tmp_path,
) -> None:
    gtf = tmp_path / "annotation.gtf"
    gtf.write_text(
        'chr1\tX\texon\t101\t110\t.\t+\t.\tgene_id "g1"; transcript_id "tx1";\n'
        'chr1\tX\texon\t201\t210\t.\t+\t.\tgene_id "g1"; transcript_id "tx1";\n'
        'chr1\tX\tstart_codon\t103\t105\t.\t+\t0\tgene_id "g1"; transcript_id "tx1";\n'
        'chr1\tX\tstop_codon\t205\t207\t.\t+\t0\tgene_id "g1"; transcript_id "tx1";\n'
        'chr1\tX\texon\t301\t310\t.\t-\t.\tgene_id "g2"; transcript_id "tx2";\n'
        'chr1\tX\texon\t401\t410\t.\t-\t.\tgene_id "g2"; transcript_id "tx2";\n'
        'chr1\tX\tstart_codon\t406\t408\t.\t-\t0\tgene_id "g2"; transcript_id "tx2";\n'
        'chr1\tX\tstop_codon\t303\t305\t.\t-\t0\tgene_id "g2"; transcript_id "tx2";\n'
    )
    features = gencode_features_from_gtf(gtf, source_version="v-test")
    intervals = {
        feature.feature_id: (feature.start, feature.end) for feature in features
    }
    assert intervals["tx1:intron1:donor"] == (110, 112)
    assert intervals["tx1:intron1:acceptor"] == (198, 200)
    assert intervals["tx2:intron1:donor"] == (398, 400)
    assert intervals["tx2:intron1:acceptor"] == (310, 312)
    assert intervals["tx1:kozak"] == (96, 106)
    assert intervals["tx2:kozak"] == (404, 414)


def test_assembly_labels_a_present_paired_element_as_context() -> None:
    evidence = Evidence("test", "v1", EvidenceType.EXPERIMENTAL, "record")
    donor = GenomicFeature(
        "donor",
        "chr1",
        10,
        12,
        "+",
        "splice.donor",
        evidence,
        "gene",
        {"paired_element_id": "acceptor"},
    )
    acceptor = GenomicFeature(
        "acceptor",
        "chr1",
        18,
        20,
        "+",
        "splice.acceptor",
        evidence,
        "gene",
        {"paired_element_id": "donor"},
    )
    fasta = {"chr1": "A" * 100}
    records = assemble_annotated_windows(
        (donor, acceptor),
        fasta,
        assembly="test",
        upstream_nt=10,
        downstream_nt=10,
        split_seed=7,
    )
    donor_record = next(
        record for record in records if record.metadata["target_element_id"] == "donor"
    )
    paired_regions = [
        region
        for region in donor_record.context_regions
        if region.region_type == "paired_biological_element"
    ]
    assert len(paired_regions) == 1
    assert paired_regions[0].attributes["paired_element_id"] == "acceptor"


def test_assembly_uses_full_annotation_set_for_control_exclusion() -> None:
    evidence = Evidence("test", "v1", EvidenceType.CURATED, "record")
    target = GenomicFeature(
        "target",
        "chr1",
        20,
        22,
        "+",
        "splice.donor",
        evidence,
        "gene",
    )
    other = GenomicFeature(
        "other",
        "chr1",
        30,
        32,
        "+",
        "splice.donor",
        evidence,
        "gene",
    )
    record = assemble_annotated_windows(
        (target,),
        {"chr1": "A" * 100},
        annotation_features=(target, other),
        assembly="test",
        upstream_nt=20,
        downstream_nt=20,
        split_seed=7,
    )[0]
    assert {element.element_id for element in record.elements} == {"target", "other"}


def test_normalized_bed_preserves_source_evidence_and_attributes(tmp_path) -> None:
    bed = tmp_path / "sites.bed"
    bed.write_text("chr1\t10\t16\ttata-1\t.\t+\tpromoter.tata\tgene-1\tassay=CAGE\n")
    features = read_normalized_bed(
        bed,
        source_id="promoter-source",
        source_version="release-1",
        evidence_type=EvidenceType.EXPERIMENTAL,
    )
    assert len(features) == 1
    assert features[0].site_class == "promoter.tata"
    assert features[0].evidence.source_version == "release-1"
    assert features[0].attributes["assay"] == "CAGE"


def test_three_way_cluster_split_is_deterministic_and_disjoint() -> None:
    observed = {
        split_for_cluster(
            f"gene-{index}",
            seed=7,
            development_percent=30,
            baseline_training_percent=20,
        )
        for index in range(100)
    }
    assert observed == {"development", "baseline_training", "confirmation"}
    assert split_for_cluster(
        "gene-1",
        seed=7,
        development_percent=30,
        baseline_training_percent=20,
    ) == split_for_cluster(
        "gene-1",
        seed=7,
        development_percent=30,
        baseline_training_percent=20,
    )


def test_gencode_sampling_happens_at_gene_cluster_before_feature_expansion(
    tmp_path,
) -> None:
    gtf = tmp_path / "annotation.gtf"
    rows = []
    for index in range(30):
        gene = f"g{index}"
        transcript = f"tx{index}"
        rows.extend(
            (
                (
                    f'chr1\tX\tgene\t1\t500\t.\t+\t.\tgene_id "{gene}"; '
                    f'gene_type "protein_coding";\n'
                ),
                (
                    f'chr1\tX\texon\t101\t110\t.\t+\t.\tgene_id "{gene}"; '
                    f'transcript_id "{transcript}";\n'
                ),
                (
                    f'chr1\tX\texon\t201\t210\t.\t+\t.\tgene_id "{gene}"; '
                    f'transcript_id "{transcript}";\n'
                ),
            )
        )
    gtf.write_text("".join(rows))
    selected = select_gencode_gene_ids(
        gtf,
        seed=11,
        maximum_per_split=3,
        development_percent=30,
        baseline_training_percent=20,
        chromosomes={"chr1"},
    )
    assert all(len(gene_ids) <= 3 for gene_ids in selected.values())
    included = frozenset(gene for genes in selected.values() for gene in genes)
    features = gencode_features_from_gtf(
        gtf,
        source_version="v-test",
        include_gene_ids=included,
    )
    sampled = sample_gencode_features(
        features,
        seed=11,
        maximum_clusters_per_site_class=2,
        development_percent=30,
        baseline_training_percent=20,
    )
    keys = [
        (
            split_for_cluster(
                item.homology_cluster,
                seed=11,
                development_percent=30,
                baseline_training_percent=20,
            ),
            item.site_class,
            item.homology_cluster,
        )
        for item in sampled
    ]
    assert len(keys) == len(set(keys))


def test_gencode_annotation_exclusions_include_unselected_transcripts(
    tmp_path,
) -> None:
    gtf = tmp_path / "annotation.gtf"
    gtf.write_text(
        'chr1\tX\texon\t21\t30\t.\t+\t.\tgene_id "other"; transcript_id "other-tx";\n'
    )
    evidence = Evidence("test", "v1", EvidenceType.CURATED, "target")
    target = BiologicalElement(
        "target",
        "splice.donor",
        interval=Interval(10, 12),
        evidence=evidence,
        sequence="AA",
    )
    record = AnnotatedSequence(
        example_id="example",
        sequence="A" * 50,
        assembly="test",
        chrom="chr1",
        genomic_start=0,
        genomic_end=50,
        strand="+",
        sequence_orientation="transcription_5_to_3",
        split="development",
        homology_cluster="gene",
        elements=(target,),
        metadata={"target_element_id": "target"},
    )
    intervals = gencode_annotation_intervals_for_windows(gtf, [record])
    assert any(interval.start == 18 for interval in intervals["example"])
