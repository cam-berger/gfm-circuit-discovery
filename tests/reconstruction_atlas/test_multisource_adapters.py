from __future__ import annotations

import gzip

from gfmcd.reconstruction_atlas.assembly import (
    GenomicFeature,
    assemble_annotated_windows,
)
from gfmcd.reconstruction_atlas.schema import Evidence, EvidenceType
from gfmcd.reconstruction_atlas.sources import (
    annotation_intervals_from_bed_for_windows,
    derive_polypyrimidine_tracts,
    iter_epdnew_tss_cores,
    iter_polyasite_elements,
    sample_locus_blocked_features,
)


def test_epdnew_adapter_centers_both_strands_on_reported_tss(tmp_path) -> None:
    bed = tmp_path / "epd.bed"
    bed.write_text(
        "chr1 100 160 GENE1_1 900 + 149 160\n" "chr1 300 360 GENE2_1 900 - 300 311\n"
    )
    features = list(iter_epdnew_tss_cores(bed))
    assert [(item.start, item.end, item.strand) for item in features] == [
        (146, 153, "+"),
        (307, 314, "-"),
    ]
    assert all(item.site_class == "promoter.epd_tss_core" for item in features)


def test_polyasite_adapter_maps_signal_coordinates_on_both_strands(tmp_path) -> None:
    path = tmp_path / "polyasite.tsv.gz"
    header = [
        "chrom",
        "chromStart",
        "chromEnd",
        "name",
        "score",
        "strand",
        "rep",
        "frac_samples",
        "nr_prots",
        "annotation",
        "gene_name",
        "gene_id",
        "repSite_signals",
    ]
    rows = [
        [
            "1",
            "190",
            "200",
            "1:200:+",
            "1",
            "+",
            "200",
            "0.2",
            "3",
            "TE",
            "A",
            "ENSG1",
            "AATAAA@-24@177",
        ],
        [
            "1",
            "390",
            "400",
            "1:391:-",
            "1",
            "-",
            "391",
            "0.2",
            "3",
            "TE",
            "B",
            "ENSG2",
            "ATTAAA@-21@412",
        ],
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")
    features = list(iter_polyasite_elements(path))
    intervals = {
        (item.site_class, item.strand): (item.start, item.end) for item in features
    }
    assert intervals[("polya.signal", "+")] == (176, 182)
    assert intervals[("polya.signal", "-")] == (406, 412)
    assert intervals[("polya.cleavage_region", "+")] == (199, 200)
    assert intervals[("polya.cleavage_region", "-")] == (390, 391)


def test_ppt_derivation_chooses_nearest_all_pyrimidine_tract() -> None:
    evidence = Evidence("GENCODE", "test", EvidenceType.DERIVED, "acceptor")
    acceptor = GenomicFeature(
        "acceptor",
        "chr1",
        50,
        52,
        "+",
        "splice.acceptor",
        evidence,
        "gene",
    )
    sequence = list("A" * 100)
    sequence[32:39] = "CTTCTCC"
    sequence[40:47] = "TTTCTCT"
    tracts = derive_polypyrimidine_tracts(
        [acceptor],
        {"chr1": "".join(sequence)},
        tract_width=7,
        search_upstream_nt=40,
        acceptor_buffer_nt=3,
    )
    assert len(tracts) == 1
    assert (tracts[0].start, tracts[0].end) == (40, 47)
    assert tracts[0].attributes["paired_element_id"] == "acceptor"


def test_locus_blocking_is_shared_across_classes_and_excludes_boundaries() -> None:
    evidence = Evidence("source", "test", EvidenceType.EXPERIMENTAL, "record")
    features = [
        GenomicFeature(
            "promoter",
            "chr1",
            10_000,
            10_007,
            "+",
            "promoter.epd_tss_core",
            evidence,
            "source-cluster-1",
        ),
        GenomicFeature(
            "enhancer",
            "chr1",
            20_000,
            20_007,
            "+",
            "enhancer.encode_dels_core",
            evidence,
            "source-cluster-2",
        ),
        GenomicFeature(
            "boundary",
            "chr1",
            999_800,
            999_807,
            "+",
            "enhancer.encode_dels_core",
            evidence,
            "source-cluster-3",
        ),
    ]
    selected, audit = sample_locus_blocked_features(
        features,
        seed=7,
        maximum_clusters_per_site_class=2,
        development_percent=30,
        baseline_training_percent=20,
        window_flank_nt=512,
    )
    assert len({item.homology_cluster for item in selected}) == 1
    assert audit["excluded_cross_block_window"] == 1


def test_genomic_forward_regulatory_windows_and_bed_exclusions(tmp_path) -> None:
    evidence = Evidence("ENCODE", "test", EvidenceType.EXPERIMENTAL, "record")
    target = GenomicFeature(
        "target",
        "chr1",
        20,
        27,
        "+",
        "enhancer.encode_dels_core",
        evidence,
        "locus",
        {"sequence_orientation": "genomic_forward"},
    )
    record = assemble_annotated_windows(
        [target],
        {"chr1": "ACGT" * 30},
        assembly="test",
        upstream_nt=10,
        downstream_nt=10,
        split_seed=7,
    )[0]
    assert record.sequence_orientation == "genomic_forward"
    bed = tmp_path / "annotations.bed.gz"
    with gzip.open(bed, "wt", encoding="utf-8") as handle:
        handle.write("chr1\t30\t35\tother\tenhancer.encode_dels_core\n")
    intervals = annotation_intervals_from_bed_for_windows(bed, [record])
    assert [(item.start, item.end) for item in intervals[record.example_id]] == [
        (20, 25)
    ]
