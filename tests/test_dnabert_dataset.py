from gfmcd.dnabert.dataset import (
    build_records,
    exact_duplicate_report,
    gc_fraction,
    records_sha256,
    reverse_complement,
    reverse_complement_overlap_report,
    split_summary,
)


def test_reverse_complement():
    assert reverse_complement("ACGT") == "ACGT"
    assert reverse_complement("AAGC") == "GCTT"


def test_gc_fraction():
    assert gc_fraction("GGCC") == 1.0
    assert gc_fraction("AATT") == 0.0
    assert gc_fraction("ACGT") == 0.5


def test_build_records_and_summary():
    records = build_records(
        ["ACGT", "GGGG"],
        [0, 1],
        split="train",
    )

    assert len(records) == 2
    assert records[0].sequence == "ACGT"
    assert records[1].label == 1

    summary = split_summary(records)

    assert summary["rows"] == 2
    assert summary["label_counts"] == {"0": 1, "1": 1}
    assert summary["unique_lengths"] == [4]


def test_exact_duplicate_report():
    records = {
        "train": build_records(
            ["AAAA", "CCCC"],
            [0, 1],
            split="train",
        ),
        "validation": build_records(
            ["AAAA"],
            [0],
            split="validation",
        ),
    }

    report = exact_duplicate_report(records)

    assert report["duplicate_group_count"] == 1
    assert report["cross_split_duplicate_group_count"] == 1
    assert report["conflicting_label_duplicate_group_count"] == 0


def test_conflicting_duplicate_labels():
    records = {
        "train": build_records(
            ["AAAA"],
            [0],
            split="train",
        ),
        "test": build_records(
            ["AAAA"],
            [1],
            split="test",
        ),
    }

    report = exact_duplicate_report(records)

    assert report["conflicting_label_duplicate_group_count"] == 1


def test_reverse_complement_overlap():
    records = {
        "train": build_records(
            ["AAGC"],
            [0],
            split="train",
        ),
        "test": build_records(
            ["GCTT"],
            [0],
            split="test",
        ),
    }

    report = reverse_complement_overlap_report(records)

    assert report["overlap_pair_count"] == 1
    assert report["cross_split_overlap_pair_count"] == 1


def test_records_hash_is_deterministic():
    records = build_records(
        ["ACGT", "GGGG"],
        [0, 1],
        split="train",
    )

    assert records_sha256(records) == records_sha256(records)
