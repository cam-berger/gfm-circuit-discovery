from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path

import pytest

from gfmcd.reconstruction_atlas.promoter_source_localization import (
    BinnedIntervalIndex,
    EpdMotif,
    GenomicInterval,
    composition_preserving_paired_disruption,
    fixed_dose_tile,
    iter_encode_pls,
    iter_fantom5_dominant_ctss,
    iter_ucsc_cpg_islands,
    oriented_interval_to_genome,
    paired_fixed_dose_tile,
    parse_epd_motif,
    scan_epd_motif,
)


def test_binned_interval_index_deduplicates_and_honors_strand() -> None:
    intervals = [
        GenomicInterval("chr1", 9, 21, "plus", "+"),
        GenomicInterval("chr1", 15, 17, "minus", "-"),
    ]
    index = BinnedIntervalIndex(intervals, bin_size=10)
    assert index.overlapping("chr1", 10, 20, strand="+") == (intervals[0],)
    assert set(index.overlapping("chr1", 16, 17)) == set(intervals)
    assert index.overlapping("chr1", 30, 31) == ()


def test_external_interval_parsers(tmp_path: Path) -> None:
    cage = tmp_path / "cage.bed.gz"
    with gzip.open(cage, "wt") as handle:
        handle.write("chr1\t5\t20\tpeak\t9\t+\t11\t12\t255,0,0\n")
    assert list(iter_fantom5_dominant_ctss(cage)) == [
        GenomicInterval("chr1", 11, 12, "peak", "+")
    ]

    ccre = tmp_path / "ccre.bed.gz"
    with gzip.open(ccre, "wt") as handle:
        handle.write("chr1\t1\t2\ta\t0\t.\t1\t2\t0,0,0\tdELS\n")
        handle.write("chr1\t3\t7\tb\t0\t.\t3\t7\t0,0,0\tPLS\n")
    assert list(iter_encode_pls(ccre)) == [GenomicInterval("chr1", 3, 7, "b")]

    cpg = tmp_path / "cpg.txt.gz"
    with gzip.open(cpg, "wt") as handle:
        handle.write("1\tchr2\t9\t19\tCpG: 1\t10\n")
    assert list(iter_ucsc_cpg_islands(cpg)) == [
        GenomicInterval("chr2", 9, 19, "CpG: 1")
    ]


def test_parse_and_scan_epd_matrix(tmp_path: Path) -> None:
    source = tmp_path / "motif.html"
    source.write_text("""
        <table>
        <tr><td>Position</td><td>-1</td><td>0</td><td>1</td></tr>
        <tr><td>W(A)</td><td>0</td><td>-2</td><td>-2</td></tr>
        <tr><td>W(C)</td><td>-2</td><td>0</td><td>-2</td></tr>
        <tr><td>W(G)</td><td>-2</td><td>-2</td><td>0</td></tr>
        <tr><td>W(T)</td><td>-2</td><td>-2</td><td>-2</td></tr>
        <tr><td>Preferred region: Position 0 between -1 and +1 relative to
        transcription start site.</td></tr>
        <tr><td>Optimized cut-off value: -1.0 (99%)</td></tr>
        </table>
        """)
    motif = parse_epd_motif(source, name="ACG", family="core")
    calls = scan_epd_motif("TTACGTT", motif, tss_index=3)
    assert len(calls) == 1
    assert calls[0].sequence == "ACG"
    assert calls[0].position0_relative_tss == 0


def test_oriented_interval_mapping_is_strand_symmetric() -> None:
    assert oriented_interval_to_genome(
        tss=1000, strand="+", start=249, end=250, tss_index=249
    ) == (1000, 1001)
    assert oriented_interval_to_genome(
        tss=1000, strand="-", start=249, end=250, tss_index=249
    ) == (1000, 1001)
    assert oriented_interval_to_genome(
        tss=1000, strand="-", start=239, end=249, tss_index=249
    ) == (1001, 1011)


def test_fixed_dose_tile_maximizes_overlap_and_pair_gate() -> None:
    offsets = [(0, 0), (0, 4), (4, 10), (10, 16), (0, 0)]
    tile = fixed_dose_tile(offsets, nominal_start=5, nominal_end=15, token_dose=2)
    assert tile is not None
    assert tile.token_indices == (2, 3)
    assert (tile.cover_start, tile.cover_end, tile.overlap_nt) == (4, 16, 10)
    assert (
        paired_fixed_dose_tile(
            offsets,
            [(0, 0), (0, 5), (5, 11), (11, 15), (0, 0)],
            nominal_start=5,
            nominal_end=15,
            token_dose=2,
            maximum_cover_nt=12,
            maximum_cover_length_difference_nt=2,
        )
        is not None
    )
    assert (
        paired_fixed_dose_tile(
            offsets,
            [(0, 0), (0, 2), (2, 4), (4, 20), (0, 0)],
            nominal_start=5,
            nominal_end=15,
            token_dose=2,
            maximum_cover_nt=12,
            maximum_cover_length_difference_nt=2,
        )
        is None
    )


def test_epd_motif_validates_geometry() -> None:
    with pytest.raises(ValueError):
        EpdMotif(
            name="bad",
            family="core",
            position_offsets=(-1, 1),
            weights={base: (0.0, 0.0) for base in "ACGT"},
            preferred_position0_min=0,
            preferred_position0_max=0,
            cutoff=0,
            source_sha256="x",
        )
    assert Counter("AAG") == Counter("GAA")


def test_paired_motif_disruption_preserves_composition_and_equal_dose() -> None:
    motif = EpdMotif(
        name="ACG",
        family="core",
        position_offsets=(-1, 0, 1),
        weights={
            "A": (0.0, -5.0, -5.0),
            "C": (-5.0, 0.0, -5.0),
            "G": (-5.0, -5.0, 0.0),
            "T": (-5.0, -5.0, -5.0),
        },
        preferred_position0_min=0,
        preferred_position0_max=0,
        cutoff=-1.0,
        source_sha256="test",
    )
    target = "TTACGTT"
    control = "TTAGCTT"
    disruption = composition_preserving_paired_disruption(
        target,
        control,
        start=2,
        end=5,
        motif=motif,
        tss_index=3,
        seed=7,
        geometry_qualified_batch=lambda values: [True] * len(values),
        candidates=128,
    )
    assert disruption is not None
    assert disruption["composition_preserved"]
    assert disruption["changed_nucleotides_each"] >= 2
    assert Counter(disruption["target_original_subsequence"]) == Counter(
        disruption["target_disrupted_subsequence"]
    )
    assert Counter(disruption["control_original_subsequence"]) == Counter(
        disruption["control_disrupted_subsequence"]
    )
