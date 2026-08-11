"""Task 1 of the PBC v2 model-free capacity analysis: census streaming -> slot gene sets.

The load-bearing test in this file (`test_slot_gene_sets_reproduce_committed_certificate_hashes`)
is the memorial that makes the whole downstream capacity analysis (demand model, exact max-flow
feasibility solver, power projection -- Tasks 2-5, layered onto the same module) trustworthy: it
streams the real, hash-pinned, 2.34 GB candidate census and asserts that `slot_gene_sets` reproduces
the seven gene-set hashes already committed in `composition_balanced_cohort_v1.json`. It is skipped
(never silently "passed") when the census file is not present on this machine.

Everything else in this file is a fast, always-run unit test of the same three public functions
against small synthetic fixtures, so the tricky parts (the streaming state machine, the dead
blacklist filter, the scaffold geometry filter, hash verification) stay covered even when the real
census is absent.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from gfmcd.splicing import pbc_v2_capacity
from gfmcd.splicing.pbc_v2_capacity import slot_gene_sets, stream_candidates, verify_slot_hashes
from gfmcd.splicing.positional_boundary_coherence import (
    DEFAULT_ACCEPTOR_WINDOW,
    DEFAULT_DONOR_WINDOW,
    gene_root,
    normalize_chromosome,
    sha256_payload,
)


# These are INPUTS, not part of this repo: the census is a 2.34 GB file the PBC v2 successor
# project produced (see the task-1 brief). Never write to them.
CENSUS = Path(
    "/Users/cberg/Research/projects/gfm-circuit-discovery-pbc-successor/results/"
    "splicebert_positional_boundary_coherence/development/candidate_census_v1.json"
)
COHORT_ARTIFACT = CENSUS.with_name("composition_balanced_cohort_v1.json")

# Applied to the two tests that actually open the 2.34 GB census, NOT to the whole module: a
# module-level skipif would silently take the synthetic unit tests below with it on any machine
# lacking the census, so the streaming state machine, the dead blacklist filter, the scaffold
# geometry filter and hash verification would all go uncovered while the file still read as
# "skipped, fine".
requires_census = pytest.mark.skipif(
    not (CENSUS.exists() and COHORT_ARTIFACT.exists()),
    reason=(
        f"PBC v2 capacity census reproduction inputs not found on this machine: "
        f"{CENSUS} and/or {COHORT_ARTIFACT} are absent"
    ),
)

# candidate_census_v1.json's own top-level "counts" block (also duplicated verbatim into
# candidate_census_v1.audit.json) records donor=6861, acceptor=7827, scaffold=50773 -- the
# manifest (candidate_census_v1.manifest.json) carries no candidate count. 6861 + 7827 + 50773 =
# 65461, cross-checked directly against the file by counting "    {" lines strictly inside the
# "candidates" array's own open/close markers (`awk 'NR==27{ok=1} NR==54973552{ok=0} ok && NR>27 &&
# $0=="    {"{c++} END{print c}' candidate_census_v1.json`, run once against this exact,
# sha256-pinned file). Pinned here as a literal per the task-1 brief so a truncated or mis-parsed
# stream fails loudly instead of silently under-counting.
_EXPECTED_CANDIDATE_COUNT = 65461


def _filters_from_artifact() -> dict:
    """Build `slot_gene_sets` kwargs from the committed cohort artifact + module constants.

    `construct_length`/`donor_pos`/`acceptor_pos` and `allowed_chromosomes` are read from the
    artifact at test time (never hard-coded), per the task-1 brief. `donor_window`/
    `acceptor_window` are not stored in the artifact; they come from
    `positional_boundary_coherence`'s own `DEFAULT_DONOR_WINDOW`/`DEFAULT_ACCEPTOR_WINDOW`.
    """

    cohort = json.loads(COHORT_ARTIFACT.read_text())
    config = cohort["config"]
    allowed_chroms = {normalize_chromosome(chrom) for chrom in config["allowed_chromosomes"]}
    return dict(
        # The artifact records only the blacklist's sha256 (629c050b...) and its member count
        # (5998), not the member IDs themselves, so the real list cannot be rebuilt from what is
        # committed. That is fine: the filter is a dead one by construction (see
        # `slot_gene_sets`'s docstring) -- unversioned blacklist entries never match a versioned
        # `gene_id`, so an empty set reproduces the exact same filtered pool as the real blacklist
        # would.
        blacklist=set(),
        allowed_chroms=allowed_chroms,
        # Empty for this reproduction: the census's own `allowed_chromosomes` (21 entries, chr1..
        # chrX, no chr8/chr9/chrY) was already applied when the census itself was built, so no
        # candidate row can carry chr8/chr9/chrM/chrMT/chrY regardless of what this artifact's
        # `config.forbidden_chromosomes` lists -- a non-empty forbidden set here would be
        # redundant, not corrective, over this input.
        forbidden_chroms=set(),
        construct_length=int(config["construct_length"]),
        donor_pos=int(config["donor_pos"]),
        acceptor_pos=int(config["acceptor_pos"]),
        donor_window=DEFAULT_DONOR_WINDOW,
        acceptor_window=DEFAULT_ACCEPTOR_WINDOW,
    )


@requires_census
def test_slot_gene_sets_reproduce_committed_certificate_hashes():
    sets = slot_gene_sets(stream_candidates(CENSUS), **_filters_from_artifact())
    cert = json.loads(COHORT_ARTIFACT.read_text())["capacity"]["slots"]

    assert set(sets) == set(cert), "slot_gene_sets must return exactly the certified slot names"
    for slot, entry in cert.items():
        assert len(sets[slot]) == entry["available_unique_genes"], slot

    expected_hashes = {slot: entry["gene_ids_sha256"] for slot, entry in cert.items()}
    verified = verify_slot_hashes(sets, expected_hashes)
    assert verified == {slot: True for slot in expected_hashes}


@requires_census
def test_stream_candidates_count_matches_census_own_counts_block():
    count = sum(1 for _ in stream_candidates(CENSUS))
    assert count > 0
    assert count == _EXPECTED_CANDIDATE_COUNT


# --------------------------------------------------------------------------------------------
# Fast, always-run unit tests against small synthetic fixtures (no dependency on the 2.34 GB file).
# --------------------------------------------------------------------------------------------


def _fixture_census(tmp_path: Path, candidates: list[dict], exclusions: list[dict] | None = None) -> Path:
    """A tiny file shaped like the real census: same key order risk, same trailing sections.

    Critically, `exclusions` is a second list-of-dicts top-level key that sits *after*
    `candidates` and whose elements land at the exact same 4-space nesting depth as candidate
    objects once `json.dumps(..., indent=2)` renders it -- exactly like the real census's
    ~327k-row `exclusions` array. Any test that streams a fixture built with a non-empty
    `exclusions` list is implicitly checking that the streamer does not leak into it.
    """

    payload = {
        "algorithm_version": "test-v1",
        "allowed_chromosomes": ["chr1", "chr2"],
        "candidate_mode": "production",
        "candidates": candidates,
        "exclusions": exclusions or [],
    }
    path = tmp_path / "synthetic_census.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def _synthetic_candidate(
    role: str,
    phase: int,
    gene_id: str,
    *,
    chrom: str = "chr1",
    sequence: str = "A" * 10,
    donor_pos: int = 100,
    acceptor_pos: int = 200,
    donor_window: tuple = (90, 106),
    acceptor_window: tuple = (190, 206),
) -> dict:
    return {
        "role": role,
        "phase": phase,
        "gene_id": gene_id,
        "chrom": chrom,
        "sequence": sequence,
        "donor_pos": donor_pos,
        "acceptor_pos": acceptor_pos,
        "donor_window": list(donor_window),
        "acceptor_window": list(acceptor_window),
    }


def test_iter_candidate_objects_parses_all_rows_and_ignores_trailing_exclusions(tmp_path):
    candidates = [
        _synthetic_candidate("donor", 0, "ENSG1.1"),
        _synthetic_candidate("acceptor", 1, "ENSG2.2"),
    ]
    exclusions = [{"edge_id": "chr1:1>2:+:ENSG3", "gene_id": "ENSG3", "reasons": ["r"]}]
    path = _fixture_census(tmp_path, candidates, exclusions)

    streamed = list(pbc_v2_capacity._iter_candidate_objects(path))

    assert streamed == candidates


def test_iter_candidate_objects_raises_when_candidates_array_marker_absent(tmp_path):
    path = tmp_path / "no_candidates.json"
    path.write_text(json.dumps({"foo": "bar"}, indent=2))

    with pytest.raises(ValueError, match="candidates"):
        list(pbc_v2_capacity._iter_candidate_objects(path))


def test_iter_candidate_objects_raises_on_mid_object_truncation(tmp_path):
    path = _fixture_census(tmp_path, [_synthetic_candidate("donor", 0, "ENSG1.1")])
    text = path.read_text()
    cut_at = text.index("    {\n") + len("    {\n")
    path.write_text(text[:cut_at])

    with pytest.raises(ValueError, match="truncated"):
        list(pbc_v2_capacity._iter_candidate_objects(path))


def test_iter_candidate_objects_raises_when_array_close_missing(tmp_path):
    path = _fixture_census(tmp_path, [_synthetic_candidate("donor", 0, "ENSG1.1")])
    text = path.read_text()
    open_idx = text.index('"candidates": [')
    close_idx = text.index("\n  ]", open_idx)
    path.write_text(text[:close_idx] + "\n")

    with pytest.raises(ValueError, match="truncat"):
        list(pbc_v2_capacity._iter_candidate_objects(path))


def test_slot_gene_sets_buckets_by_role_and_phase_and_normalizes_gene_root():
    candidates = [
        _synthetic_candidate("donor", 0, "ENSG1.3"),
        _synthetic_candidate("donor", 1, "ENSG2.1"),
        _synthetic_candidate("acceptor", 2, "ENSG3.9"),
        _synthetic_candidate("scaffold", 0, "ENSG4.2", sequence="A" * 20),
    ]
    sets = slot_gene_sets(
        candidates,
        blacklist=set(),
        allowed_chroms={"chr1"},
        forbidden_chroms=set(),
        construct_length=20,
        donor_pos=100,
        acceptor_pos=200,
        donor_window=(90, 106),
        acceptor_window=(190, 206),
    )

    assert set(sets) == {
        "donor_phase_0",
        "donor_phase_1",
        "donor_phase_2",
        "acceptor_phase_0",
        "acceptor_phase_1",
        "acceptor_phase_2",
        "scaffold",
    }
    assert sets["donor_phase_0"] == {"ENSG1"}
    assert sets["donor_phase_1"] == {"ENSG2"}
    assert sets["donor_phase_2"] == set()
    assert sets["acceptor_phase_2"] == {"ENSG3"}
    assert sets["scaffold"] == {"ENSG4"}


def test_slot_gene_sets_blacklist_is_inert_against_versioned_gene_id():
    """The audited quirk: an unversioned blacklist entry never matches a versioned gene_id."""

    candidate = _synthetic_candidate("donor", 0, "ENSG5.4")
    sets = slot_gene_sets(
        [candidate],
        blacklist={gene_root(candidate["gene_id"])},  # "ENSG5" -- never equals "ENSG5.4"
        allowed_chroms={"chr1"},
        forbidden_chroms=set(),
        construct_length=20,
        donor_pos=100,
        acceptor_pos=200,
        donor_window=(90, 106),
        acceptor_window=(190, 206),
    )

    assert sets["donor_phase_0"] == {"ENSG5"}


def test_slot_gene_sets_applies_allowed_and_forbidden_chrom_filters():
    allowed = _synthetic_candidate("donor", 0, "ENSG6.1", chrom="chr1")
    not_allowed = _synthetic_candidate("donor", 0, "ENSG7.1", chrom="chr3")
    forbidden = _synthetic_candidate("donor", 0, "ENSG8.1", chrom="chr2")

    sets = slot_gene_sets(
        [allowed, not_allowed, forbidden],
        blacklist=set(),
        allowed_chroms={"chr1", "chr2"},
        forbidden_chroms={"chr2"},
        construct_length=20,
        donor_pos=100,
        acceptor_pos=200,
        donor_window=(90, 106),
        acceptor_window=(190, 206),
    )

    assert sets["donor_phase_0"] == {"ENSG6"}


def test_slot_gene_sets_scaffold_requires_exact_geometry_match():
    good = _synthetic_candidate("scaffold", 0, "ENSG9.1", sequence="A" * 20)
    wrong_length = _synthetic_candidate("scaffold", 0, "ENSG10.1", sequence="A" * 19)
    wrong_donor_pos = _synthetic_candidate(
        "scaffold", 0, "ENSG11.1", sequence="A" * 20, donor_pos=101
    )
    wrong_window = _synthetic_candidate(
        "scaffold", 0, "ENSG12.1", sequence="A" * 20, donor_window=(0, 1)
    )

    sets = slot_gene_sets(
        [good, wrong_length, wrong_donor_pos, wrong_window],
        blacklist=set(),
        allowed_chroms={"chr1"},
        forbidden_chroms=set(),
        construct_length=20,
        donor_pos=100,
        acceptor_pos=200,
        donor_window=(90, 106),
        acceptor_window=(190, 206),
    )

    assert sets["scaffold"] == {"ENSG9"}


def test_verify_slot_hashes_reports_per_slot_match():
    slot_sets = {"a": {"G1", "G2"}, "b": {"G3"}}
    expected = {
        "a": sha256_payload(sorted(slot_sets["a"])),
        "b": "0" * 64,  # deliberately wrong
    }

    assert verify_slot_hashes(slot_sets, expected) == {"a": True, "b": False}


def test_module_does_not_import_torch_or_model_adapters():
    """Stage `construction`: this module must stay importable without the `splicebert` extra."""

    source = Path(pbc_v2_capacity.__file__).read_text()
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert "torch" not in imported
    assert not any("adapter" in name.lower() for name in imported)
