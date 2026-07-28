from __future__ import annotations

import csv
import gzip
from pathlib import Path

import pytest

from gfmcd.reconstruction_atlas.promoter_assay import (
    binary_classification_metrics,
    context_indices,
    has_nearby_tss,
    iter_gencode_mane_tss,
    matched_context_shuffles,
    oriented_tss_window,
    read_gue_sequences,
)


def test_iter_gencode_mane_tss_filters_and_converts_coordinates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny.gtf.gz"
    rows = [
        (
            'chr1\tX\ttranscript\t11\t30\t.\t+\t.\tgene_id "g1"; '
            'transcript_id "t1"; gene_type "protein_coding"; gene_name "A"; '
            'tag "MANE_Select";\n'
        ),
        (
            'chr1\tX\ttranscript\t41\t70\t.\t-\t.\tgene_id "g2"; '
            'transcript_id "t2"; gene_type "protein_coding"; gene_name "B"; '
            'tag "MANE_Select";\n'
        ),
        (
            'chr1\tX\ttranscript\t80\t90\t.\t+\t.\tgene_id "g3"; '
            'transcript_id "t3"; gene_type "lncRNA"; tag "MANE_Select";\n'
        ),
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.writelines(rows)
    features = list(iter_gencode_mane_tss(path))
    assert [(feature.transcript_id, feature.tss) for feature in features] == [
        ("t1", 10),
        ("t2", 69),
    ]


def test_oriented_tss_window_and_nearby_tss() -> None:
    fasta = {"chr1": "ACGT" * 30}
    forward = oriented_tss_window(
        fasta,
        chrom="chr1",
        tss=20,
        strand="+",
        upstream_nt=4,
        downstream_nt=3,
    )
    reverse = oriented_tss_window(
        fasta,
        chrom="chr1",
        tss=20,
        strand="-",
        upstream_nt=4,
        downstream_nt=3,
    )
    assert forward == "ACGTACGT"
    assert reverse == "TACGTACG"
    assert has_nearby_tss({"chr1": [10, 50, 90]}, "chr1", 55, radius_nt=5)
    assert not has_nearby_tss({"chr1": [10, 50, 90]}, "chr1", 70, radius_nt=5)


def test_read_gue_sequences_indexes_reverse_complements(tmp_path: Path) -> None:
    path = tmp_path / "test.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sequence", "label"))
        writer.writeheader()
        writer.writerow({"sequence": "AACG", "label": 1})
    index = read_gue_sequences([path])
    assert index["AACG"] == [("test", 1)]
    assert index["CGTT"] == [("test_reverse", 1)]


def test_matched_context_shuffles_preserve_anchor_composition_and_dose() -> None:
    target = "ACGT" * 75
    control = "TGCA" * 75
    indices, anchor = context_indices(
        tss_index=249,
        target_width_nt=5,
        upstream_context_nt=50,
        downstream_context_nt=10,
    )
    result = matched_context_shuffles(
        target, control, indices=indices, seed=7, candidates=512
    )
    assert result is not None
    target_shuffle, control_shuffle, changed = result
    assert target[anchor[0] : anchor[1]] == target_shuffle[anchor[0] : anchor[1]]
    assert control[anchor[0] : anchor[1]] == control_shuffle[anchor[0] : anchor[1]]
    assert sorted(target[index] for index in indices) == sorted(
        target_shuffle[index] for index in indices
    )
    assert sorted(control[index] for index in indices) == sorted(
        control_shuffle[index] for index in indices
    )
    assert sum(a != b for a, b in zip(target, target_shuffle)) == changed
    assert sum(a != b for a, b in zip(control, control_shuffle)) == changed


def test_binary_metrics_recognize_perfect_and_invalid_inputs() -> None:
    metrics = binary_classification_metrics([0, 0, 1, 1], [-2.0, -1.0, 1.0, 2.0])
    assert metrics["accuracy"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["auroc"] == 1.0
    with pytest.raises(ValueError):
        binary_classification_metrics([0, 2], [-1.0, 1.0])
