"""Task 5 of the PBC v2 model-free capacity analysis: the ladder runner.

Covers ``scripts/splicebert_positional_clock/run_pbc_v2_capacity.py`` without needing the 2.34 GB
census: the ladder logic (``build_ladder``) and the artifact assembly (``assemble_artifact``) are
pure functions taking injected slot sets, and the CLI is exercised only through ``--smoke``, which
never opens ``--census``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts" / "splicebert_positional_clock"
sys.path.insert(0, str(SCRIPTS))

from gfmcd.artifact_guard import ArtifactExistsError  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = _load("run_pbc_v2_capacity_test", "run_pbc_v2_capacity.py")

_GENE_ID_PATTERN = re.compile(r"^ENSG\d")


def _assert_no_gene_ids(node) -> None:
    """Recursively assert no string in `node` matches `^ENSG\\d` -- the artifact's no-gene-list rule."""

    if isinstance(node, dict):
        for value in node.values():
            _assert_no_gene_ids(value)
    elif isinstance(node, list):
        for value in node:
            _assert_no_gene_ids(value)
    elif isinstance(node, str):
        assert not _GENE_ID_PATTERN.match(node), f"gene id leaked into artifact: {node!r}"


# -------------------------------------------------------------------------------------------
# --smoke never opens --census
# -------------------------------------------------------------------------------------------


def test_smoke_returns_without_opening_census_path(tmp_path):
    out = tmp_path / "art.json"
    missing_census = tmp_path / "does_not_exist.json"
    assert not missing_census.exists()

    RUNNER.main(["--smoke", "--out", str(out), "--census", str(missing_census)])

    assert out.exists()
    assert not missing_census.exists(), "smoke mode must never create/touch the census path"
    payload = json.loads(out.read_text())
    assert payload["mode"] == "smoke"
    assert payload["model_execution"] is False
    assert payload["checkpoint_loaded"] is None
    assert payload["cohort_selection_attempted"] is False
    assert payload["stage"] == "construction"
    _assert_no_gene_ids(payload)


def test_smoke_manifest_is_written_alongside_the_artifact(tmp_path):
    out = tmp_path / "art.json"
    RUNNER.main(["--smoke", "--out", str(out)])

    manifest_path = tmp_path / "art.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "SMOKE_ONLY_NON_AUTHORIZING"
    assert any(Path(o["path"]).name == "art.json" for o in manifest["outputs"])


# -------------------------------------------------------------------------------------------
# overwrite guard
# -------------------------------------------------------------------------------------------


def test_runner_refuses_to_overwrite_existing_output(tmp_path):
    out = tmp_path / "art.json"
    RUNNER.main(["--smoke", "--out", str(out)])
    assert out.exists()

    with pytest.raises(ArtifactExistsError):
        RUNNER.main(["--smoke", "--out", str(out)])


# -------------------------------------------------------------------------------------------
# build_ladder: pure function, injected synthetic slot sets, no filesystem access
# -------------------------------------------------------------------------------------------


def _disjoint_capacity_slot_sets() -> dict[str, set[str]]:
    """Pairwise-disjoint synthetic pools sized to clear a 0.25 margin through N=780.

    Disjointness means per-slot counting and exact `feasibility` always agree here (there is no
    Hall obstruction to find), which isolates this test to the selection-rule logic itself rather
    than to the max-flow layer (that is `test_pbc_v2_capacity_design.py`'s job).
    """

    def genes(prefix: str, n: int) -> set[str]:
        return {f"{prefix}_{i}" for i in range(n)}

    return {
        "scaffold": genes("scaffold", 1300),
        "donor_phase_0": genes("donor0", 900),
        "donor_phase_1": genes("donor1", 900),
        "donor_phase_2": genes("donor2", 900),
        "acceptor_phase_0": genes("acceptor0", 900),
        "acceptor_phase_1": genes("acceptor1", 900),
        "acceptor_phase_2": genes("acceptor2", 900),
    }


def test_build_ladder_produces_one_row_per_rung_with_residual_holdout_column():
    slot_sets = _disjoint_capacity_slot_sets()
    ladder = (600, 729, 780)

    result = RUNNER.build_ladder(slot_sets, ladder=ladder)

    assert [row["n_blocks"] for row in result["rungs"]] == list(ladder)
    for row in result["rungs"]:
        entries = row["necessary_counts"]["entries"]
        # Seven real slots plus the three union families.
        assert set(entries) == {
            "scaffold",
            "donor_phase_0",
            "donor_phase_1",
            "donor_phase_2",
            "acceptor_phase_0",
            "acceptor_phase_1",
            "acceptor_phase_2",
            "donor_all_phases",
            "acceptor_all_phases",
            "all_roles",
        }
        for name, entry in entries.items():
            assert "residual_holdout_unique_genes" in entry, name
            assert entry["residual_holdout_unique_genes"] == (
                entry["available_unique_genes"] - entry["demand_unique_genes"]
            )

    # scaffold at N=600: 1300 available, 600 demanded.
    scaffold_600 = result["rungs"][0]["necessary_counts"]["entries"]["scaffold"]
    assert scaffold_600["available_unique_genes"] == 1300
    assert scaffold_600["demand_unique_genes"] == 600
    assert scaffold_600["residual_holdout_unique_genes"] == 700


def test_build_ladder_selection_rule_picks_the_largest_qualifying_rung():
    slot_sets = _disjoint_capacity_slot_sets()
    # N=600 is below the psi power floor (conditional_on_rho, not powered_for_every_rho); both
    # 729 and 780 clear it, so a correct "largest N" rule must return 780, not just the first
    # rung (729) that happens to qualify.
    ladder = (600, 729, 780)

    result = RUNNER.build_ladder(slot_sets, ladder=ladder)

    by_n = {row["n_blocks"]: row for row in result["rungs"]}
    assert by_n[600]["psi_verdict"] == "conditional_on_rho"
    assert by_n[600]["selection_candidate"] is False
    assert by_n[729]["psi_verdict"] == "powered_for_every_rho"
    assert by_n[729]["selection_candidate"] is True
    assert by_n[780]["psi_verdict"] == "powered_for_every_rho"
    assert by_n[780]["selection_candidate"] is True

    assert result["selection_rule"] == RUNNER.SELECTION_RULE
    assert result["selected_rung"] == 780
    assert result["no_feasible_rung_reason"] is None


def test_build_ladder_records_a_reason_when_no_rung_qualifies():
    # Pools far too small to ever clear a 0.25 margin, on any rung.
    slot_sets = {
        "scaffold": {f"s_{i}" for i in range(10)},
        "donor_phase_0": {f"d0_{i}" for i in range(5)},
        "donor_phase_1": {f"d1_{i}" for i in range(5)},
        "donor_phase_2": {f"d2_{i}" for i in range(5)},
        "acceptor_phase_0": {f"a0_{i}" for i in range(5)},
        "acceptor_phase_1": {f"a1_{i}" for i in range(5)},
        "acceptor_phase_2": {f"a2_{i}" for i in range(5)},
    }

    result = RUNNER.build_ladder(slot_sets, ladder=(6, 9))

    assert result["selected_rung"] is None
    assert isinstance(result["no_feasible_rung_reason"], str)
    assert result["no_feasible_rung_reason"]


# -------------------------------------------------------------------------------------------
# assemble_artifact: honesty fields and the no-gene-list rule
# -------------------------------------------------------------------------------------------


def test_assemble_artifact_records_model_execution_false_and_leaks_no_gene_ids():
    slot_sets = {
        "scaffold": {"ENSG00000000001", "ENSG00000000002"},
        "donor_phase_0": {"ENSG00000000003"},
        "donor_phase_1": {"ENSG00000000004"},
        "donor_phase_2": {"ENSG00000000005"},
        "acceptor_phase_0": {"ENSG00000000006"},
        "acceptor_phase_1": {"ENSG00000000007"},
        "acceptor_phase_2": {"ENSG00000000008"},
    }
    verified = {slot: True for slot in slot_sets}

    artifact = RUNNER.assemble_artifact(
        mode="real",
        slot_sets=slot_sets,
        slot_hash_verification=RUNNER._hash_verification_view(verified),
        census_receipt={"mode": "file", "path": "fake/census.json", "sha256": "0" * 64, "size_bytes": 1},
        certificate_receipt={"mode": "file", "path": "fake/cert.json", "sha256": "1" * 64, "size_bytes": 2},
        filters_view=RUNNER._public_filters_view(
            dict(
                blacklist=set(),
                allowed_chroms={"chr1"},
                forbidden_chroms=set(),
                construct_length=1022,
                donor_pos=392,
                acceptor_pos=512,
                donor_window=(302, 428),
                acceptor_window=(462, 602),
            )
        ),
        ladder=(6, 9),
    )

    assert artifact["model_execution"] is False
    assert artifact["checkpoint_loaded"] is None
    assert artifact["cohort_selection_attempted"] is False
    assert artifact["stage"] == "construction"
    assert artifact["slot_hash_verification"]["all_verified"] is True

    # json.dumps must succeed (no stray `set` objects) and no gene ID may appear anywhere.
    json.dumps(artifact)
    _assert_no_gene_ids(artifact)


def test_hash_verification_view_reports_smoke_mode_as_not_applicable():
    view = RUNNER._hash_verification_view(None)
    assert view["applicable"] is False
