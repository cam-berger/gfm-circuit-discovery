from __future__ import annotations

import torch

from gfmcd.adapters.dnabert2.adapter import Dnabert2Adapter
from gfmcd.reconstruction_atlas.assembly import (
    GenomicFeature,
    assemble_annotated_windows,
    gencode_features_from_gtf,
    read_normalized_bed,
    reverse_complement,
)
from gfmcd.reconstruction_atlas.schema import Evidence, EvidenceType


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
