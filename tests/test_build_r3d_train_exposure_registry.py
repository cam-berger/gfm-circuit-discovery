from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from gfmcd.provenance import sha256_file


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts/splicebert_acceptor_coupling/build_r3d_train_exposure_registry.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_r3d_train_exposure_registry", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RESULT = (
    REPO
    / "results/splicebert_acceptor_coupling/r3d_train_exposure_registry_v1.json"
)
MANIFEST = RESULT.with_suffix(".manifest.json")


@dataclass(frozen=True)
class Example:
    example_id: str
    gene_id: str
    sequence: str
    donor_pos: int
    next_acceptor_pos: int
    n_isoforms_donor: int
    chrom: str = "chr1"
    split: str = "train"
    other_gt: tuple[int, ...] = ()


def _example(
    name: str,
    *,
    gene: str,
    motif: str,
    gap: int,
    support: int,
    flank: tuple[str, str, str] = ("A", "A", "A"),
) -> Example:
    donor = 20
    bases = list("A" * 220)
    bases[donor - 1] = flank[0]
    bases[donor : donor + 2] = motif
    bases[donor + 2] = flank[1]
    bases[donor + 4] = flank[2]
    return Example(
        example_id=name,
        gene_id=gene,
        sequence="".join(bases),
        donor_pos=donor,
        next_acceptor_pos=donor + gap,
        n_isoforms_donor=support,
    )


def test_matcher_is_deterministic_consumes_comparators_and_support_stratifies():
    treatment = [
        _example("t-low", gene="gt1", motif="GC", gap=61, support=1),
        _example("t-high", gene="gt2", motif="GC", gap=72, support=20),
    ]
    pool = [
        _example("c-low", gene="gc1", motif="GT", gap=80, support=1),
        _example("c-high", gene="gc2", motif="GT", gap=62, support=20),
        _example("c-low-2", gene="gc3", motif="GT", gap=100, support=2),
        _example("c-high-2", gene="gc4", motif="GT", gap=90, support=15),
    ]

    flank_only = MODULE.match_comparators(treatment, list(reversed(pool)), k=1)
    support_aware = MODULE.match_comparators(
        list(reversed(treatment)),
        pool,
        k=1,
        key_fn=MODULE.flank_and_support_key,
    )

    assert [row.example_id for row in flank_only.comparators] == [
        "c-low",
        "c-high",
    ]
    assert [row.example_id for row in support_aware.comparators] == [
        "c-high",
        "c-low",
    ]
    assert len({row.example_id for row in support_aware.comparators}) == 2
    assert support_aware.unmatched_treatment == ()


def test_head_shards_are_sorted_hashed_and_acceptor_filtered(tmp_path):
    rows = [
        ("shard_z.json", "acceptor-z", "gene-z", False, "chr2"),
        ("shard_a.json", "donor-a", "gene-a", True, "chr8"),
        ("shard_m.json", "acceptor-m", "gene-m", False, "chr1"),
    ]
    for name, example_id, gene_id, is_donor, chrom in rows:
        (tmp_path / name).write_text(
            json.dumps(
                {
                    "conditions": {},
                    "example": {
                        "example_id": example_id,
                        "gene_id": gene_id,
                        "target_is_donor": is_donor,
                        "chrom": chrom,
                    },
                }
            )
        )

    observed = MODULE.scan_head_screen_shards(tmp_path)

    assert observed["shards"] == 3
    assert observed["rows"] == observed["genes"] == 3
    assert observed["row_ids"] == ["acceptor-m", "acceptor-z", "donor-a"]
    assert observed["gene_ids"] == ["gene-a", "gene-m", "gene-z"]
    assert observed["endpoint_strata"]["masked_acceptor"]["rows"] == 2
    assert observed["endpoint_strata"]["masked_donor"]["rows"] == 1
    assert observed["all_shard_chr8_chr9_row_ids"] == ["donor-a"]
    assert observed["row_ids_sha256"] == MODULE.canonical_sha256(
        observed["row_ids"]
    )


def test_train_dataset_restriction_is_passed_before_loader_and_leaks_fail():
    calls = []

    def loader(gtf, fasta, *, allowed_chromosomes):
        calls.append((gtf, fasta, allowed_chromosomes))
        return [
            _example("ok", gene="g", motif="GT", gap=60, support=1)
        ]

    observed = MODULE.load_train_dataset("annotation", "genome", loader=loader)
    assert len(observed) == 1
    assert calls == [("annotation", "genome", MODULE.TRAIN_CHROMOSOMES)]
    assert not set(MODULE.TRAIN_CHROMOSOMES) & MODULE.FORBIDDEN_CHROMOSOMES

    def leaking_loader(_gtf, _fasta, *, allowed_chromosomes):
        assert allowed_chromosomes == MODULE.TRAIN_CHROMOSOMES
        example = _example("leak", gene="g", motif="GT", gap=60, support=1)
        return [Example(**{**example.__dict__, "chrom": "chr8"})]

    with pytest.raises(RuntimeError, match="forbidden chromosomes"):
        MODULE.load_train_dataset("annotation", "genome", loader=leaking_loader)


def test_capacity_is_one_row_per_gene_and_partitions_raw_rows():
    g1a = _example("g1-a", gene="g1", motif="GT", gap=60, support=1)
    g1b = _example("g1-b", gene="g1", motif="GT", gap=61, support=1)
    g2 = _example("g2", gene="g2", motif="GT", gap=62, support=1)
    observed = MODULE.capacity_record(
        [(g1a, 10), (g1b, 11), (g2, 12)],
        ["g1"],
        {"outside_caliper": 4},
    )

    assert observed["eligible_rows"] == 3
    assert observed["eligible_genes_one_row_each"] == 2
    assert observed["exposed_development"]["rows_available"] == 2
    assert observed["exposed_development"]["genes_one_row_each"] == 1
    assert observed["fresh_target"]["rows_available"] == 1
    assert observed["fresh_target"]["genes_one_row_each"] == 1
    assert observed["exclusions"] == {"outside_caliper": 4}


def test_immutable_writer_never_replaces_existing_bytes(tmp_path):
    path = tmp_path / "registry.json"
    MODULE.write_immutable_json(path, {"first": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="refuse to overwrite"):
        MODULE.write_immutable_json(path, {"second": True})
    assert path.read_bytes() == before


def test_clean_source_state_accepts_clean_repo_and_rejects_tracked_or_unrelated_dirt(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "r3d-test@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "R3d Test"], cwd=repo, check=True
    )
    tracked = repo / "tracked.txt"
    tracked.write_text("clean\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)

    clean = MODULE.require_clean_repository_state(repo)
    assert clean["git_dirty"] is False
    assert len(clean["git_revision"]) == 40

    tracked.write_text("dirty\n")
    with pytest.raises(RuntimeError, match="clean source worktree"):
        MODULE.require_clean_repository_state(repo)

    tracked.write_text("clean\n")
    (repo / "unrelated.txt").write_text("untracked\n")
    with pytest.raises(RuntimeError, match="clean source worktree"):
        MODULE.require_clean_repository_state(repo)


def test_manifest_uses_prepublication_clean_source_state(monkeypatch, tmp_path):
    record = {
        "generator": {
            "script": "scripts/splicebert_acceptor_coupling/build_r3d_train_exposure_registry.py",
            "script_sha256": "a" * 64,
        },
        "input_bindings": {
            "annotation": {"size_bytes": 1, "sha256": "b" * 64},
            "genome": {"size_bytes": 2, "sha256": "c" * 64},
            "genome_index": {"size_bytes": 3, "sha256": "d" * 64},
        },
        "sources": {
            "A_claude_inverse_acceptor": {
                "artifact_bindings": {},
                "transcript_binding": {"size_bytes": 4, "sha256": "e" * 64},
                "result_manifest_semantic_bindings": {},
                "reconstruction_git_blobs": {},
            },
            "B_head_screen_clean_alt_candidate_selection": {
                "shards": 0,
                "shard_inventory_sha256": "inventory",
                "comparison_artifact": {"size_bytes": 5, "sha256": "f" * 64},
                "comparison_manifest": {"size_bytes": 6, "sha256": "1" * 64},
            },
        },
    }
    monkeypatch.setattr(
        MODULE,
        "file_binding",
        lambda _path: {
            "logical_name": "registry.json",
            "size_bytes": 7,
            "sha256": "2" * 64,
        },
    )
    monkeypatch.setattr(
        MODULE,
        "scan_head_screen_shards",
        lambda _path: {
            "shards": 0,
            "shard_bindings": [],
            "shard_inventory_sha256": "inventory",
        },
    )
    state = {"git_revision": "3" * 40, "git_dirty": False}

    manifest = MODULE.build_manifest(
        record=record,
        output_path="results/splicebert_acceptor_coupling/registry.json",
        head_shard_dir="unused",
        source_state=state,
    )
    assert manifest["code"]["git_revision"] == "3" * 40
    assert manifest["code"]["git_dirty"] is False

    external_output = (tmp_path / "durable-results" / "registry.json").resolve()
    external = MODULE.build_manifest(
        record=record,
        output_path=external_output,
        head_shard_dir="unused",
        source_state=state,
    )
    assert external["outputs"][0]["path"] == str(external_output)
    assert external["command"][-1] == str(external_output)
    assert external["code"]["git_dirty"] is False

    with pytest.raises(RuntimeError, match="captured clean Git"):
        MODULE.build_manifest(
            record=record,
            output_path="results/splicebert_acceptor_coupling/registry.json",
            head_shard_dir="unused",
            source_state={**state, "git_dirty": True},
        )


def test_registered_compact_result_replays_builder_and_scope_contract():
    record = json.loads(RESULT.read_text())
    MODULE.validate_production_identity(record)

    assert record["schema"] == MODULE.RESULT_SCHEMA
    assert record["scope_id"] == (
        "claude_inverse_acceptor_and_clean_alt_head_screen_direct_exposures_v1"
    )
    assert record["gene_ids"] == sorted(set(record["gene_ids"]))
    assert len(record["gene_ids"]) == 1112
    assert record["gene_ids_sha256"] == MODULE.canonical_sha256(
        record["gene_ids"]
    )
    assert record["counts"]["source_classes"] == 2
    assert record["scope"]["global_freshness_claimed"] is False
    assert record["scope"]["freshness_label"] == (
        "disjoint_from_two_registered_direct_exposure_classes"
    )
    assert "broader Atlas" in record["scope"]["atlas_disclosure"]
    assert "donor-circuit" in record["scope"]["not_included"]
    assert "earlier R3" in record["scope"]["not_included"]
    assert record["sources"]["A_claude_inverse_acceptor"][
        "all_head_cohort"
    ]["adds_new_exposure_ids"] is False
    source_b = record["sources"]["B_head_screen_clean_alt_candidate_selection"]
    assert source_b["rows"] == 400
    assert source_b["genes"] == 360
    assert source_b["endpoint_strata"]["masked_acceptor"]["rows"] == 222
    assert source_b["endpoint_strata"]["masked_donor"]["rows"] == 178
    assert source_b["exact_masked_acceptor_endpoint_is_stratum_only"] is True
    source_a = record["sources"]["A_claude_inverse_acceptor"]
    assert len(source_a["result_manifest_semantic_bindings"]) == 8
    assert all(
        binding["semantic_binding_pass"] is True
        for binding in source_a["result_manifest_semantic_bindings"].values()
    )
    assert len(source_a["reconstruction_git_blobs"]) == 7
    assert all(
        len(binding["sha256"]) == 64
        for binding in source_a["reconstruction_git_blobs"].values()
    )
    assert record["known_artifact_chr8_chr9_model_scored_check"][
        "rows_found"
    ] == 0
    assert record["generator"]["model_loaded_or_run"] is False
    assert record["input_bindings"]["genome_index"]["logical_name"] == "hg38.fa.fai"
    assert record["input_bindings"]["genome_index"]["sha256"] == (
        "3b425de206296a5c8053023fa5ca61da43cfe78c1737c12e58c83367c7e83c21"
    )
    manifest = json.loads(MANIFEST.read_text())
    assert record["generator"]["script_sha256"] == manifest["code"][
        "builder_sha256"
    ]
    rendered = RESULT.read_text()
    assert "/private/tmp" not in rendered
    assert "/Users/" not in rendered


def test_registered_manifest_binds_output_sources_command_and_dirty_revision():
    record = json.loads(RESULT.read_text())
    manifest = json.loads(MANIFEST.read_text())

    assert manifest["schema"] == "gfmcd.r3d.exposed_gene_registry_manifest/1"
    assert manifest["outputs"] == [{
        "logical_name": RESULT.name,
        "path": str(RESULT.relative_to(REPO)),
        "role": "registry",
        "size_bytes": RESULT.stat().st_size,
        "sha256": sha256_file(RESULT),
    }]
    assert manifest["parameters"]["scope_id"] == record["scope_id"]
    assert manifest["parameters"][
        "source_b_candidate_selection_uses_both_endpoint_strata"
    ] is True
    assert manifest["seeds"]["coupling_dataset_seed"] == 0
    assert manifest["code"]["builder_sha256"] == record["generator"][
        "script_sha256"
    ]
    assert isinstance(manifest["code"]["git_dirty"], bool)
    assert manifest["inputs"]["head_screen_shards"]["files_count"] == 400
    assert len(manifest["inputs"]["head_screen_shards"]["files"]) == 400
    assert len(manifest["inputs"]["claude_reconstruction_git_blobs"]) == 7
    assert len(manifest["inputs"]["claude_result_manifest_semantic_bindings"]) == 8
    assert manifest["inputs"]["head_screen_shards"]["inventory_sha256"] == (
        record["sources"]["B_head_screen_clean_alt_candidate_selection"]
        ["shard_inventory_sha256"]
    )
    rendered = MANIFEST.read_text()
    assert "/private/tmp" not in rendered
    assert "/Users/" not in rendered
