"""Cohort serialisation -- the thing that makes "no GTF, no FASTA" true.

The R4 Colab package asserted a genome-free run before this existed, and the claim was false:
the runner still parsed GENCODE and indexed hg38 through pyfaidx, which fails on a remote
runtime. These tests pin the round-trip and the seal refusal.
"""

from __future__ import annotations

import json

import pytest

from gfmcd.splicing.cohort_io import (
    example_from_dict, example_to_dict, read_cohort, write_cohort,
)
from gfmcd.splicing.coupling import CouplingExample


def example(example_id="e1", chrom="chr11", gene_id="g1"):
    return CouplingExample(
        example_id=example_id, sequence="ACGT" * 32, acceptor_pos=20, donor_pos=60,
        chrom=chrom, strand="+", gene_id=gene_id, split="val", exon_length=40,
        intron_length_down=120, n_isoforms_donor=9, n_isoforms_acceptor=7, gene_type="pc",
        intron_length_up=80, other_ag=(10, 30), other_gt=(50,), real_acc=(12,), real_don=(62,),
        next_acceptor_pos=140, metadata={"note": "x"},
    )


def test_round_trip_preserves_every_field(tmp_path):
    original = example()
    restored = example_from_dict(json.loads(json.dumps(example_to_dict(original))))
    assert restored == original


def test_tuple_fields_survive_json_as_tuples(tmp_path):
    restored = example_from_dict(json.loads(json.dumps(example_to_dict(example()))))
    for name in ("other_ag", "other_gt", "real_acc", "real_don"):
        assert isinstance(getattr(restored, name), tuple)


def test_file_round_trip_preserves_order_and_content(tmp_path):
    cohort = [example(f"e{i}") for i in range(5)]
    path = tmp_path / "cohort.jsonl"
    write_cohort(path, cohort)
    assert read_cohort(path) == cohort


def test_limit_takes_a_prefix_not_a_sample(tmp_path):
    cohort = [example(f"e{i}") for i in range(10)]
    path = tmp_path / "cohort.jsonl"
    write_cohort(path, cohort)
    assert read_cohort(path, limit=3) == cohort[:3]


def test_summary_reports_what_a_manifest_needs(tmp_path):
    cohort = [example("a", gene_id="g1"), example("b", gene_id="g2")]
    summary = write_cohort(tmp_path / "c.jsonl", cohort)
    assert summary["n"] == 2 and summary["n_genes"] == 2
    assert summary["chroms"] == ["chr11"] and summary["splits"] == ["val"]
    assert len(summary["sha256"]) == 64


def test_sealed_chromosome_is_refused_not_filtered(tmp_path):
    """A sealed chromosome reaching a cohort file is a protocol failure, not a filter miss."""
    cohort = [example("a"), example("b", chrom="chr8")]
    with pytest.raises(ValueError, match="sealed chromosomes"):
        write_cohort(tmp_path / "c.jsonl", cohort, refuse_chroms=("chr8", "chr9"))


def test_refusal_leaves_no_partial_file_behind(tmp_path):
    path = tmp_path / "c.jsonl"
    with pytest.raises(ValueError):
        write_cohort(path, [example("a", chrom="chr9")], refuse_chroms=("chr8", "chr9"))
    assert not path.exists()


def test_empty_cohort_file_is_rejected(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        read_cohort(path)
