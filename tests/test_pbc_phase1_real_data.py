from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from gfmcd.splicing import pbc_freshness
from gfmcd.splicing.pbc_biology import (
    GtfFeature,
    IndexedFasta,
    MaxEntTables,
    _FeatureIntervalIndex,
    _overlaps,
)
from gfmcd.splicing.positional_boundary_coherence import (
    DEFAULT_BRIDGE_ACCEPTOR_POS,
    _source_strata_complete,
    sha256_payload,
)


REPO = Path(__file__).resolve().parents[1]

# `maxentpy` is the optional `maxent` extra (pinned to kepbod/maxentpy b5e69aad in pyproject.toml,
# not the PyPI package). Phase-I candidate scoring genuinely needs it, so these tests cannot be
# rewritten to run without it -- but a missing *optional* extra must skip, not fail, which is the
# convention `tests/test_splice_landscape.py` already follows for the same dependency. Without this
# the documented `uv run --extra splicebert --extra biology --extra dev pytest` goes red for reasons
# that have nothing to do with the code under test. Install with `uv sync --extra maxent`.
#
# What stops being checked when this skips: the MaxEnt pin memorial
# (`test_explicit_maxent_tables_match_installed_pinned_implementation`, which asserts the study's
# scorer reproduces the pinned upstream implementation exactly), real-FASTA/GTF census
# reconstruction on both strands, the recorded-exclusion paths, and the candidate-audit tamper
# memorials. Do not read a skipped run as a passing one.
pytest.importorskip(
    "maxentpy",
    reason="optional maxentpy unavailable; install the pinned `maxent` extra to run Phase-I "
    "real-data and MaxEnt-pin memorials",
)


def _load(name: str, relative: str):
    path = REPO / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD_CENSUS = _load(
    "build_pbc_candidate_census",
    "scripts/splicebert_positional_boundary_coherence/build_candidate_census.py",
)
AUDIT_CENSUS = _load(
    "audit_pbc_candidate_census",
    "scripts/splicebert_positional_boundary_coherence/audit_candidate_census.py",
)
HELPERS = _load("pbc_phase1_test_helpers", "tests/test_positional_boundary_coherence.py")


def _write_fasta(path: Path, sequence: str, width: int = 60) -> Path:
    lines = [sequence[index : index + width] for index in range(0, len(sequence), width)]
    path.write_text(">chr1\n" + "\n".join(lines) + "\n")
    fai = path.with_suffix(path.suffix + ".fai")
    fai.write_text(f"chr1\t{len(sequence)}\t6\t{width}\t{width + 1}\n")
    return fai


def _attrs(gene: str, transcript: str) -> str:
    return (
        f'gene_id "{gene}"; transcript_id "{transcript}"; '
        'gene_type "protein_coding"; transcript_type "protein_coding";'
    )


def _edge_gtf_rows(
    *, gene: str, transcript: str, strand: str, phase: int, offset: int
) -> list[str]:
    if strand == "+":
        first_start, first_end = offset + 1, offset + 300 + phase
        donor = first_end
        second_start, second_end = donor + 201, donor + 500
        up_frame, down_frame = 0, (3 - phase) % 3
        exons = ((first_start, first_end), (second_start, second_end))
        cds = ((first_start, first_end, up_frame), (second_start, second_end, down_frame))
    else:
        # Transcript-upstream exon is the higher-coordinate interval.
        low_start, low_end = offset + 1, offset + 300
        high_start, high_end = low_end + 201, low_end + 500 + phase
        up_frame, down_frame = 0, (3 - phase) % 3
        exons = ((low_start, low_end), (high_start, high_end))
        cds = ((low_start, low_end, down_frame), (high_start, high_end, up_frame))
    attrs = _attrs(gene, transcript)
    rows = [
        f"chr1\ttest\texon\t{start}\t{end}\t.\t{strand}\t.\t{attrs}"
        for start, end in exons
    ]
    rows.extend(
        f"chr1\ttest\tCDS\t{start}\t{end}\t.\t{strand}\t{frame}\t{attrs}"
        for start, end, frame in cds
    )
    return rows


def _install_motifs(sequence: list[str], rows: list[str]) -> None:
    parsed = [row.split("\t") for row in rows if "\texon\t" in row]
    strand = parsed[0][6]
    intervals = [(int(row[3]) - 1, int(row[4])) for row in parsed]
    if strand == "+":
        upstream, downstream = sorted(intervals)
        donor, acceptor = upstream[1], downstream[0]
        sequence[donor : donor + 2] = list("GT")
        sequence[acceptor - 2 : acceptor] = list("AG")
    else:
        downstream, upstream = sorted(intervals)
        donor, acceptor = upstream[0], downstream[1]
        sequence[donor - 2 : donor] = list("AC")  # reverse complement -> GT
        sequence[acceptor : acceptor + 2] = list("CT")  # reverse complement -> AG


def _biology_fixture(
    tmp_path: Path,
    *,
    conflict: bool = False,
    bad_plus_motif: bool = False,
    alternative_acceptor: bool = False,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sequence = list("A" * 12000)
    rows = []
    designs = (
        ("ENSG00000000001.5", "ENST_PLUS0.2", "+", 0, 1200),
        ("ENSG00000000002.3", "ENST_PLUS1.1", "+", 1, 3200),
        ("ENSG00000000003.7", "ENST_MINUS2.4", "-", 2, 7000),
    )
    rows_by_design = []
    for gene, transcript, strand, phase, offset in designs:
        edge_rows = _edge_gtf_rows(
            gene=gene, transcript=transcript, strand=strand, phase=phase, offset=offset
        )
        _install_motifs(sequence, edge_rows)
        rows.extend(edge_rows)
        rows_by_design.append(edge_rows)
    if alternative_acceptor:
        alternative_rows = []
        for line in rows_by_design[0]:
            columns = line.split("\t")
            columns[8] = columns[8].replace(
                "ENST_PLUS0.2", "ENST_PLUS0_ALTERNATIVE.1"
            )
            if int(columns[3]) > 1500:
                columns[3] = str(int(columns[3]) + 99)
                columns[4] = str(int(columns[4]) + 99)
            alternative_rows.append("\t".join(columns))
        _install_motifs(sequence, alternative_rows)
        rows.extend(alternative_rows)
    if bad_plus_motif:
        plus_exons = [row.split("\t") for row in rows_by_design[0] if "\texon\t" in row]
        donor = min((int(row[4]) for row in plus_exons))
        sequence[donor : donor + 2] = list("AA")
    comprehensive_rows = list(rows)
    if conflict:
        # Opposite-strand coding/exon feature overlaps the first donor window.
        attrs = _attrs("ENSG00000999999.1", "ENST_CONFLICT.1")
        comprehensive_rows.extend(
            [
                f"chr1\ttest\texon\t1450\t1550\t.\t-\t.\t{attrs}",
                f"chr1\ttest\tCDS\t1450\t1550\t.\t-\t0\t{attrs}",
            ]
        )
    fasta = tmp_path / "tiny.fa"
    fai = _write_fasta(fasta, "".join(sequence))
    basic = tmp_path / "basic.gtf"
    comprehensive = tmp_path / "comprehensive.gtf"
    basic.write_text("\n".join(rows) + "\n")
    comprehensive.write_text("\n".join(comprehensive_rows) + "\n")
    from maxentpy import maxent

    data = Path(maxent.__file__).resolve().parent / "data"
    return fasta, fai, basic, comprehensive, data / "score5_matrix.txt", data / "score3_matrix.txt"


def _freshness(gene: str = "ENSG00000888888") -> tuple[dict, dict]:
    genes = [gene]
    registry = {
        "schema": pbc_freshness.REGISTRY_SCHEMA,
        "stage": "construction",
        "model_outcomes_read": False,
        "gene_ids": genes,
        "gene_ids_sha256": pbc_freshness.sha256_payload(genes),
        "normalized_count": 1,
    }
    registry["content_sha256"] = pbc_freshness.sha256_payload(registry)
    audit = {
        "schema": pbc_freshness.AUDIT_SCHEMA,
        "stage": "construction",
        "status": "AUDIT_PASS",
        "passed": True,
        "registry_content_sha256": registry["content_sha256"],
    }
    return registry, audit


def _feature(start: int, end: int, label: str) -> GtfFeature:
    return GtfFeature(
        chrom="chr1",
        feature="CDS",
        start=start,
        end=end,
        strand="+",
        frame=0,
        gene_id=f"GENE_{label}",
        transcript_id=label,
        gene_type="protein_coding",
        transcript_type="protein_coding",
    )


def test_feature_interval_index_matches_naive_overlap_and_source_order():
    features = [
        _feature(90, 110, "late"),
        _feature(0, 1000, "spanning"),
        _feature(20, 30, "right_boundary"),
        _feature(15, 18, "inside"),
        _feature(5, 10, "left_boundary"),
        _feature(400, 405, "remote"),
    ]
    index = _FeatureIntervalIndex(features)
    interval_families = [
        ((10, 20),),
        ((100, 110),),
        ((10, 20), (400, 401)),
        ((1001, 1010),),
    ]
    for intervals in interval_families:
        expected = tuple(
            feature
            for feature in features
            if any(_overlaps(feature, start, end) for start, end in intervals)
        )
        assert index.overlaps(intervals) == expected

    assert [row.transcript_id for row in index.overlaps(((10, 20),))] == [
        "spanning",
        "inside",
    ]
    with pytest.raises(ValueError, match="non-empty"):
        index.overlaps(((4, 4),))


def test_auditor_interval_index_matches_naive_overlap_and_source_order():
    rows = [
        {"start": 90, "end": 110, "label": "late"},
        {"start": 0, "end": 1000, "label": "spanning"},
        {"start": 20, "end": 30, "label": "right_boundary"},
        {"start": 15, "end": 18, "label": "inside"},
        {"start": 5, "end": 10, "label": "left_boundary"},
        {"start": 400, "end": 405, "label": "remote"},
    ]
    index = AUDIT_CENSUS._AuditFeatureIntervalIndex(rows)
    intervals = ((10, 20), (400, 401))
    expected = tuple(
        row
        for row in rows
        if any(
            int(row["start"]) < end and int(row["end"]) > start
            for start, end in intervals
        )
    )
    assert index.overlaps(intervals) == expected
    assert [row["label"] for row in index.overlaps(((10, 20),))] == [
        "spanning",
        "inside",
    ]
    with pytest.raises(ValueError, match="non-empty"):
        index.overlaps(((4, 4),))


def _candidate_package(tmp_path: Path, **fixture_kwargs):
    fasta, fai, basic, comprehensive, donor_table, acceptor_table = _biology_fixture(
        tmp_path, **fixture_kwargs
    )
    registry, freshness_audit = _freshness()
    package = BUILD_CENSUS.construct_candidate_package(
        fasta_path=fasta,
        fai_path=fai,
        basic_gtf_path=basic,
        comprehensive_gtf_path=comprehensive,
        donor_table_path=donor_table,
        acceptor_table_path=acceptor_table,
        freshness_registry=registry,
        freshness_audit=freshness_audit,
    )
    return package, registry, freshness_audit, (
        fasta,
        fai,
        basic,
        comprehensive,
        donor_table,
        acceptor_table,
    )


def test_indexed_fasta_and_real_census_reconstruct_both_strands(tmp_path):
    package, _registry, _audit, paths = _candidate_package(tmp_path)
    fasta, fai, *_ = paths
    genome = IndexedFasta(fasta, fai)
    assert genome.fetch("chr1", 0, 12) == "A" * 12
    modules = [row for row in package["candidates"] if row["role"] != "scaffold"]
    assert {row["strand"] for row in modules} == {"+", "-"}
    assert {row["phase"] for row in modules} == {0, 1, 2}
    for row in modules:
        boundary = int(row["boundary_index"])
        motif = (
            row["sequence"][boundary : boundary + 2]
            if row["role"] == "donor"
            else row["sequence"][boundary - 2 : boundary]
        )
        assert motif == ("GT" if row["role"] == "donor" else "AG")
        assert row["annotation"]["upstream_cds_end_phase"] == row["phase"]
        assert row["annotation"]["downstream_cds_start_phase"] == row["phase"]
    scaffolds = [row for row in package["candidates"] if row["role"] == "scaffold"]
    assert len(scaffolds) == 3  # no exact-120-intron bottleneck
    assert all(row["sequence"][435:437] == "AG" for row in scaffolds)
    assert all(row["bridge_acceptor_pos"] == DEFAULT_BRIDGE_ACCEPTOR_POS for row in scaffolds)
    assert all(row["donor_anchor_covariates"] for row in scaffolds)
    assert all(row["acceptor_anchor_covariates"] for row in scaffolds)


def test_explicit_maxent_tables_match_installed_pinned_implementation(tmp_path):
    *_prefix, donor_table, acceptor_table = _biology_fixture(tmp_path)
    scorer = MaxEntTables(donor_table, acceptor_table)
    from maxentpy import maxent
    from maxentpy.maxent import load_matrix3, load_matrix5

    donor = "CAGGTAAGT"
    acceptor = "TTCCAAACGAACTTTTGTAGGGA"
    assert scorer.score5(donor) == pytest.approx(maxent.score5(donor, matrix=load_matrix5()))
    assert scorer.score3(acceptor) == pytest.approx(maxent.score3(acceptor, matrix=load_matrix3()))


def test_comprehensive_conflict_and_reference_motif_are_recorded_exclusions(tmp_path):
    conflict_package, *_ = _candidate_package(tmp_path / "conflict", conflict=True)
    assert any(
        "comprehensive_local_strand_conflict" in row["reasons"]
        for row in conflict_package["exclusions"]
    )
    bad_package, *_ = _candidate_package(tmp_path / "bad", bad_plus_motif=True)
    assert any("module:ValueError:module lacks canonical GT" in row["reasons"] for row in bad_package["exclusions"])


def test_coordinate_identical_basic_frame_conflict_is_excluded(tmp_path):
    fasta, fai, basic, comprehensive, donor_table, acceptor_table = _biology_fixture(tmp_path)
    rows = [line.split("\t") for line in basic.read_text().splitlines()]
    first_gene_rows = [row for row in rows if 'gene_id "ENSG00000000001.5"' in row[8]]
    conflicting = []
    for row in first_gene_rows:
        changed = list(row)
        changed[8] = changed[8].replace("ENST_PLUS0.2", "ENST_PLUS0_CONFLICT.1")
        if changed[2] == "CDS":
            changed[7] = "2"  # same edge, independently derived phase 1 instead of 0
        conflicting.append("\t".join(changed))
    basic.write_text(basic.read_text() + "\n".join(conflicting) + "\n")
    registry, freshness_audit = _freshness()
    package = BUILD_CENSUS.construct_candidate_package(
        fasta_path=fasta,
        fai_path=fai,
        basic_gtf_path=basic,
        comprehensive_gtf_path=comprehensive,
        donor_table_path=donor_table,
        acceptor_table_path=acceptor_table,
        freshness_registry=registry,
        freshness_audit=freshness_audit,
    )
    assert any(
        "basic_coordinate_transcript_phase_conflict" in row["reasons"]
        for row in package["exclusions"]
    )


def test_alternative_partner_duplicate_source_loci_are_all_excluded(tmp_path):
    package, registry, freshness_audit, paths = _candidate_package(
        tmp_path, alternative_acceptor=True
    )
    gene = "ENSG00000000001"
    gene_rows = [
        row for row in package["candidates"] if row["gene_id"].split(".", 1)[0] == gene
    ]
    assert not [row for row in gene_rows if row["role"] == "donor"]
    assert len([row for row in gene_rows if row["role"] == "acceptor"]) == 2
    assert len([row for row in gene_rows if row["role"] == "scaffold"]) == 2
    duplicate_exclusions = [
        row
        for row in package["exclusions"]
        if row["gene_id"] == gene
        and row["reasons"] == ["duplicate_source_locus:donor"]
    ]
    assert len(duplicate_exclusions) == 2
    loci = [
        (
            row["role"],
            row["chrom"],
            row["strand"],
            row["source_start"],
            row["source_end"],
        )
        for row in package["candidates"]
    ]
    assert len(loci) == len(set(loci))
    fasta, fai, basic, comprehensive, donor_table, acceptor_table = paths
    audited = AUDIT_CENSUS.audit_package(
        package,
        fasta_path=fasta,
        fai_path=fai,
        basic_gtf_path=basic,
        comprehensive_gtf_path=comprehensive,
        donor_table_path=donor_table,
        acceptor_table_path=acceptor_table,
        freshness_registry=registry,
        freshness_audit=freshness_audit,
    )
    assert audited["passed"] is True


def test_freshness_blacklist_removes_gene_before_candidate_materialization(tmp_path):
    fasta, fai, basic, comprehensive, donor_table, acceptor_table = _biology_fixture(tmp_path)
    registry, freshness_audit = _freshness("ENSG00000000002")
    package = BUILD_CENSUS.construct_candidate_package(
        fasta_path=fasta,
        fai_path=fai,
        basic_gtf_path=basic,
        comprehensive_gtf_path=comprehensive,
        donor_table_path=donor_table,
        acceptor_table_path=acceptor_table,
        freshness_registry=registry,
        freshness_audit=freshness_audit,
    )
    assert all(row["gene_id"].split(".", 1)[0] != "ENSG00000000002" for row in package["candidates"])
    assert any(
        row["gene_id"] == "ENSG00000000002"
        and row["reasons"] == ["freshness_registry_gene"]
        for row in package["exclusions"]
    )


def test_candidate_audit_catches_caller_phase_score_and_maxent_pin_tamper(tmp_path):
    package, registry, freshness_audit, paths = _candidate_package(tmp_path)
    fasta, fai, basic, comprehensive, donor_table, acceptor_table = paths
    kwargs = dict(
        fasta_path=fasta,
        fai_path=fai,
        basic_gtf_path=basic,
        comprehensive_gtf_path=comprehensive,
        donor_table_path=donor_table,
        acceptor_table_path=acceptor_table,
        freshness_registry=registry,
        freshness_audit=freshness_audit,
    )
    passed = AUDIT_CENSUS.audit_package(package, **kwargs)
    assert passed["passed"] is True
    assert package["stage"] == passed["stage"] == "construction"

    phase_tamper = copy.deepcopy(package)
    donor = next(row for row in phase_tamper["candidates"] if row["role"] == "donor")
    donor["phase"] = (int(donor["phase"]) + 1) % 3
    phase_tamper["content_sha256"] = pbc_freshness.sha256_payload(
        {key: value for key, value in phase_tamper.items() if key != "content_sha256"}
    )
    assert AUDIT_CENSUS.audit_package(phase_tamper, **kwargs)["passed"] is False

    score_tamper = copy.deepcopy(package)
    next(row for row in score_tamper["candidates"] if row["role"] == "acceptor")["scores"]["maxent"] += 1
    score_tamper["content_sha256"] = pbc_freshness.sha256_payload(
        {key: value for key, value in score_tamper.items() if key != "content_sha256"}
    )
    assert AUDIT_CENSUS.audit_package(score_tamper, **kwargs)["passed"] is False

    tampered_table = tmp_path / "score5_tampered.txt"
    lines = Path(donor_table).read_text().splitlines()
    key, value = lines[0].split()
    lines[0] = f"{key} {float(value) * 1.01}"
    tampered_table.write_text("\n".join(lines) + "\n")
    tampered_kwargs = {**kwargs, "donor_table_path": tampered_table}
    assert AUDIT_CENSUS.audit_package(package, **tampered_kwargs)["passed"] is False


def test_candidate_auditor_does_not_import_producer_biology_reconstructors(tmp_path):
    package, registry, freshness_audit, paths = _candidate_package(tmp_path)
    fasta, fai, basic, comprehensive, donor_table, acceptor_table = paths
    producer_reconstructors = {
        "_FeatureIntervalIndex",
        "enumerate_edge_evidence",
        "module_record",
        "scaffold_record",
    }
    assert producer_reconstructors.isdisjoint(AUDIT_CENSUS.__dict__)
    assert AUDIT_CENSUS.audit_package(
        package,
        fasta_path=fasta,
        fai_path=fai,
        basic_gtf_path=basic,
        comprehensive_gtf_path=comprehensive,
        donor_table_path=donor_table,
        acceptor_table_path=acceptor_table,
        freshness_registry=registry,
        freshness_audit=freshness_audit,
    )["passed"] is True


def test_freshness_requires_every_category_and_audit_parser_is_independent(tmp_path, monkeypatch):
    sources = []
    for index, category in enumerate(pbc_freshness.REQUIRED_CATEGORIES, 1):
        source = tmp_path / f"{category}.json"
        source.write_text(json.dumps({"genes": [f"ENSG{index:011d}.9"]}) + "\n")
        sources.append(
            {
                "source_id": category,
                "category": category,
                "revision": "WORKTREE",
                "path": str(source),
                "expected_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "expected_normalized_count": 1,
                "extraction": {"format": "json", "pointers": ["genes"]},
            }
        )
    spec = {"schema": pbc_freshness.SOURCE_SPEC_SCHEMA, "sources": sources}
    registry = pbc_freshness.derive_registry(spec, repo=REPO)
    assert registry["stage"] == "construction"
    monkeypatch.setattr(
        pbc_freshness,
        "extract_gene_ids",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("producer parser called")),
    )
    registry_audit = pbc_freshness.audit_registry(registry, spec, repo=REPO)
    assert registry_audit["passed"] is True
    assert registry_audit["stage"] == "construction"

    missing = copy.deepcopy(spec)
    missing["sources"] = missing["sources"][:-1]
    with pytest.raises(ValueError, match="missing required categories"):
        pbc_freshness.derive_registry(missing, repo=REPO)


def test_source_stratum_gate_materializes_and_rejects_missing_role_phase_stratum():
    chroms = {"chr1", "chr2"}
    counts = {}
    for role in ("donor", "acceptor"):
        for phase in (0, 1, 2):
            counts[f"{role}.phase{phase}"] = 4
            for chrom in chroms:
                for strand in ("+", "-"):
                    counts[f"{role}.phase{phase}.chrom.{chrom}.strand.{strand}"] = 1
    assert _source_strata_complete(
        counts,
        expected_blocks=4,
        allowed_chromosomes=chroms,
        minimum_per_source_stratum=1,
    )
    counts["donor.phase2.chrom.chr2.strand.-"] = 0
    assert not _source_strata_complete(
        counts,
        expected_blocks=4,
        allowed_chromosomes=chroms,
        minimum_per_source_stratum=1,
    )


def test_scaffold_centered_matching_rejects_missing_production_anchor():
    # A production census cannot silently fall back to phase-0-centered matching.
    from gfmcd.splicing.positional_boundary_coherence import validate_candidate_package

    package = HELPERS.synthetic_candidate_package(n_per_pool=1)
    package["candidate_mode"] = "production"
    package["stage"] = "construction"
    next(row for row in package["candidates"] if row["role"] == "scaffold").pop(
        "donor_anchor_covariates"
    )
    with pytest.raises(ValueError, match="anchor vectors are required"):
        validate_candidate_package(package)


def test_triplet_selection_is_centered_on_scaffold_anchor_not_phase_zero():
    from gfmcd.splicing import positional_boundary_coherence as pbc

    raw_package = HELPERS.synthetic_candidate_package(n_per_pool=2)
    # Make index 1 a distinct composition in all phases, then point the
    # scaffold anchor exactly at that composition.  A phase-0-centred matcher
    # could choose index 0; the frozen scaffold-centred matcher must choose 1.
    for row in raw_package["candidates"]:
        if row["role"] == "donor" and row["candidate_id"].endswith("0001"):
            sequence = list("C" * 126)
            sequence[90:92] = list("GT")
            row["sequence"] = "".join(sequence)
    candidates = [pbc.validate_candidate(row) for row in raw_package["candidates"]]
    donors = [row for row in candidates if row["role"] == "donor"]
    pools = {
        phase: [row for row in donors if row["phase"] == phase]
        for phase in (0, 1, 2)
    }
    cache = {}
    means, scales = pbc._feature_scale(donors, cache)
    anchor_source = next(row for row in donors if row["candidate_id"] == "donor-p0-0001")
    selected = pbc._select_phase_triplet(
        pools,
        used_genes=set(),
        means=means,
        scales=scales,
        search_width=2,
        neighbor_width=2,
        namespace="test-scaffold-anchor",
        anchor_features=pbc.extract_module_covariates(anchor_source),
        feature_cache=cache,
    )
    assert selected is not None
    triplet, score = selected
    assert all(row["candidate_id"].endswith("0001") for row in triplet.values())
    assert score == pytest.approx(0.0)
