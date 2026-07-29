from pathlib import Path

from gfmcd.splicing.biological_cohort import (
    _complete_starts_from_slots,
    collect_prior_gene_exclusion,
)


def test_prior_gene_exclusion_is_conservative_and_hashes_sources(tmp_path: Path):
    (tmp_path / "a.json").write_text(
        '{"gene_id": "ENSG000001.2", "other": "ENSG000002"}\n'
    )
    (tmp_path / "b.json").write_text('{"nothing": true}\n')
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.json").write_text('{"gene_ids": ["ENSG000003.4"]}\n')
    first = collect_prior_gene_exclusion(tmp_path)
    second = collect_prior_gene_exclusion(tmp_path)
    assert first.gene_ids == ("ENSG000001.2", "ENSG000002", "ENSG000003.4")
    assert len(first.source_files) == 2
    assert first.aggregate_sha256 == second.aggregate_sha256


def test_complete_codon_starts_retains_only_full_012_runs():
    starts = _complete_starts_from_slots(
        10,
        22,
        (2, 0, 1, 2, 0, 1, 2, None, 1, 2, 0, 1),
    )
    assert starts == (11, 14)
