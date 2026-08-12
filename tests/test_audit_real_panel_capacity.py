"""Tests for the independent audit of real_panel_capacity_v1.json.

None of these tests require the real 2.34 GB census or the real 54 MB C43 cells file: every test
either exercises a pure function directly, or builds a small self-consistent synthetic census /
cells / certificate / analysis / artifact fixture set and points the auditor at that instead (with
the module's sha256/total pins monkeypatched to match the synthetic fixture, since those pins are
audited literals for the *real* files and would never match a synthetic stand-in).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert_positional_clock/audit_real_panel_capacity.py"
SPEC = importlib.util.spec_from_file_location("audit_real_panel_capacity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


# =================================================================================================
# Hashing / string primitives
# =================================================================================================


def test_gene_root_strips_version_suffix():
    assert MODULE._gene_root("ENSG00000099194.8") == "ENSG00000099194"
    assert MODULE._gene_root("ENSG00000099194") == "ENSG00000099194"


def test_gene_root_rejects_empty():
    with pytest.raises(ValueError):
        MODULE._gene_root("")


def test_normalize_chromosome_adds_chr_prefix_only_when_missing():
    assert MODULE._normalize_chromosome("chr1") == "chr1"
    assert MODULE._normalize_chromosome("1") == "chr1"


def test_sha256_payload_is_order_independent_via_sort_keys():
    a = MODULE._sha256_payload(sorted(["b", "a", "c"]))
    b = MODULE._sha256_payload(sorted(["c", "b", "a"]))
    assert a == b
    assert a == hashlib.sha256(b'["a","b","c"]').hexdigest()


def test_sha256_file_matches_hashlib_reference(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"some content 123")
    assert MODULE._sha256_file(path) == hashlib.sha256(b"some content 123").hexdigest()


def test_count_key_occurrences_recurses_dicts_and_lists():
    node = {
        "a": {"upstream_downstream_phase_equal": True, "nested": {"upstream_downstream_phase_equal": False}},
        "b": [{"upstream_downstream_phase_equal": True}, {"other": 1}],
        "upstream_downstream_phase_equal": True,
    }
    assert MODULE._count_key_occurrences(node, "upstream_downstream_phase_equal") == 4


# =================================================================================================
# Census streaming
# =================================================================================================


def _write_census(tmp_path: Path, candidates: list[dict], exclusions: list[dict]) -> Path:
    doc = {"candidates": candidates, "exclusions": exclusions}
    path = tmp_path / "census.json"
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return path


def test_stream_census_sections_yields_candidates_then_exclusions(tmp_path):
    candidates = [{"role": "donor", "chrom": "chr1", "gene_id": "ENSG1.1", "phase": 0}]
    exclusions = [{"edge_id": "e1", "gene_id": "ENSGZ", "reasons": ["r1"]}]
    path = _write_census(tmp_path, candidates, exclusions)

    items = list(MODULE.stream_census_sections(path))
    assert [kind for kind, _ in items] == ["candidate", "exclusion"]
    assert items[0][1]["gene_id"] == "ENSG1.1"
    assert items[1][1]["reasons"] == ["r1"]


def test_stream_census_sections_raises_on_truncated_file(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{\n  "candidates": [\n    {\n      "role": "donor"\n')
    with pytest.raises(ValueError):
        list(MODULE.stream_census_sections(path))


def test_stream_census_sections_raises_when_candidates_array_missing(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nope": []}, indent=2, sort_keys=True))
    with pytest.raises(ValueError):
        list(MODULE.stream_census_sections(path))


# =================================================================================================
# Slot bucketing + exclusion tally (independent_recompute_from_census)
# =================================================================================================


def _scaffold_candidate(gene_id: str, *, chrom: str = "chr1", ok: bool = True) -> dict:
    return {
        "role": "scaffold",
        "chrom": chrom,
        "gene_id": gene_id,
        "sequence": "A" * (10 if ok else 9),
        "donor_pos": 4,
        "acceptor_pos": 6,
        "donor_window": list(MODULE.DEFAULT_DONOR_WINDOW) if ok else [1, 2],
        "acceptor_window": list(MODULE.DEFAULT_ACCEPTOR_WINDOW),
    }


def test_independent_recompute_buckets_donor_acceptor_and_scaffold(tmp_path):
    candidates = [
        {"role": "donor", "chrom": "chr1", "gene_id": "ENSGD0.3", "phase": 0},
        {"role": "donor", "chrom": "chr1", "gene_id": "ENSGD1.1", "phase": 1},
        {"role": "acceptor", "chrom": "chr1", "gene_id": "ENSGA0.2", "phase": 0},
        # excluded by chromosome allowlist
        {"role": "donor", "chrom": "chr9", "gene_id": "ENSGOUT.1", "phase": 0},
        _scaffold_candidate("ENSGS0.5", ok=True),
        # excluded by scaffold geometry mismatch (wrong donor_window)
        _scaffold_candidate("ENSGS1.1", ok=False),
    ]
    exclusions = [
        {"edge_id": "e1", "gene_id": "Z1", "reasons": ["reasonA"]},
        {"edge_id": "e2", "gene_id": "Z2", "reasons": ["reasonA", "reasonB"]},
    ]
    path = _write_census(tmp_path, candidates, exclusions)

    result = MODULE.independent_recompute_from_census(
        path,
        allowed_chroms={"chr1"},
        construct_length=10,
        donor_pos=4,
        acceptor_pos=6,
        donor_window=MODULE.DEFAULT_DONOR_WINDOW,
        acceptor_window=MODULE.DEFAULT_ACCEPTOR_WINDOW,
    )
    slots = result["slot_genes"]
    assert slots["donor_phase_0"] == {"ENSGD0"}
    assert slots["donor_phase_1"] == {"ENSGD1"}
    assert slots["acceptor_phase_0"] == {"ENSGA0"}
    assert slots["acceptor_phase_2"] == set()
    assert slots["scaffold"] == {"ENSGS0"}  # ENSGS1 excluded: geometry mismatch
    assert result["n_candidates"] == 6
    assert result["n_exclusions"] == 2
    assert result["sole_reason_counts"] == {"reasonA": 1}
    assert result["raw_reason_counts"] == {"reasonA": 2, "reasonB": 1}
    assert result["multi_reason_exclusions"] == 1


def test_independent_recompute_gene_root_dedupes_versions(tmp_path):
    candidates = [
        {"role": "donor", "chrom": "chr1", "gene_id": "ENSGD0.1", "phase": 0},
        {"role": "donor", "chrom": "chr1", "gene_id": "ENSGD0.2", "phase": 0},  # same gene, different version
    ]
    # a single placeholder exclusion: json.dumps collapses a truly empty list to one line ("[]"),
    # which the marker-based streamer (matching the real census's own format) cannot open.
    path = _write_census(tmp_path, candidates, [{"edge_id": "e0", "gene_id": "Z0", "reasons": ["r0"]}])
    result = MODULE.independent_recompute_from_census(
        path,
        allowed_chroms={"chr1"},
        construct_length=10,
        donor_pos=4,
        acceptor_pos=6,
        donor_window=MODULE.DEFAULT_DONOR_WINDOW,
        acceptor_window=MODULE.DEFAULT_ACCEPTOR_WINDOW,
    )
    assert result["slot_genes"]["donor_phase_0"] == {"ENSGD0"}


# =================================================================================================
# Max-flow feasibility
# =================================================================================================


def test_feasibility_true_when_a_perfect_assignment_exists():
    slot_sets = {"a": {"g1", "g2"}, "b": {"g3", "g4"}}
    result = MODULE.feasibility(slot_sets, {"a": 2, "b": 2})
    assert result == {"feasible": True, "max_flow": 4, "total_demand": 4}


def test_feasibility_false_on_a_hall_violator():
    # Both slots draw from the SAME two genes; global disjointness means at most 2 units of flow
    # total can ever reach the sink, but demand is 4 -- infeasible by construction.
    slot_sets = {"a": {"g1", "g2"}, "b": {"g1", "g2"}}
    result = MODULE.feasibility(slot_sets, {"a": 2, "b": 2})
    assert result["feasible"] is False
    assert result["max_flow"] == 2
    assert result["total_demand"] == 4


def test_feasibility_zero_demand_is_trivially_feasible():
    result = MODULE.feasibility({"a": set(), "b": set()}, {"a": 0, "b": 0})
    assert result["feasible"] is True


def test_feasibility_raises_on_unmodelled_slot():
    with pytest.raises(ValueError):
        MODULE.feasibility({"a": {"g1"}}, {"missing_slot": 1})


# =================================================================================================
# Ceiling search machinery
# =================================================================================================


def test_largest_n_satisfying_finds_the_boundary():
    assert MODULE._largest_n_satisfying(lambda n: n <= 5, 100) == 5
    assert MODULE._largest_n_satisfying(lambda n: False, 100) == 0
    assert MODULE._largest_n_satisfying(lambda n: True if n < 100 else False, 100) == 99


def test_plain_lattice_demands_shape():
    demands = MODULE.plain_lattice_demands(7)
    assert demands == {"donor_phase_0": 7, "donor_phase_1": 7, "acceptor_phase_0": 7, "acceptor_phase_1": 7, "scaffold": 7}


def test_allocate_blocks_largest_remainder_sums_to_n():
    for n in range(0, 13):
        allocation = MODULE._allocate_blocks_largest_remainder(n)
        assert sum(allocation.values()) == n
        assert set(allocation) == set(MODULE._PAIR_TYPES)


def test_rotated_demands_scaffold_equals_n_and_every_phase_appears():
    demands = MODULE.rotated_demands(9)
    assert demands["scaffold"] == 9
    assert set(demands) == set(MODULE.ALL_SLOTS)
    # each phase belongs to exactly two of the three pairs, so with equal weights and n divisible
    # by 3 every donor/acceptor phase slot gets exactly 2 * (n // 3) demand
    assert demands["donor_phase_0"] == demands["donor_phase_1"] == demands["donor_phase_2"] == 6


def test_exact_plain_ceiling_matches_hand_derived_bound():
    # donor_phase_0/acceptor_phase_0 overlap on one gene (union size 3): the pair alone caps N at 1
    # (2*N <= 3). donor_phase_1/acceptor_phase_1 are disjoint (union size 4, cap N <= 2). Scaffold
    # has 5 genes (cap N <= 5). The binding constraint is the phase-0 pair: ceiling == 1.
    slot_sets = {
        "donor_phase_0": {"d0a", "d0b"},
        "acceptor_phase_0": {"d0a", "a0b"},
        "donor_phase_1": {"d1a", "d1b"},
        "acceptor_phase_1": {"a1a", "a1b"},
        "acceptor_phase_2": set(),
        "donor_phase_2": set(),
        "scaffold": {"s1", "s2", "s3", "s4", "s5"},
    }
    assert MODULE.exact_plain_ceiling(slot_sets) == 1


# =================================================================================================
# Sign-free variance decomposition
# =================================================================================================


def _cell_row(block_id: str, compatible: bool, c_ac: float) -> dict:
    return {"block_id": block_id, "compatible": compatible, "effects": {"c_ac": c_ac}}


def test_compute_variance_components_matches_hand_derived_values():
    rows = []
    rows += [_cell_row("b0", True, v) for v in (1.0, 2.0, 3.0)]
    rows += [_cell_row("b0", False, v) for v in (0.0,) * 6]
    rows += [_cell_row("b1", True, v) for v in (4.0, 5.0, 6.0)]
    rows += [_cell_row("b1", False, v) for v in (1.0,) * 6]

    result = MODULE.compute_variance_components(rows)

    assert result["n_cell_rows"] == 18
    assert result["n_blocks"] == 2
    assert result["pooled_within_block_within_group_df"] == 14
    assert result["pooled_within_block_within_group_variance"] == pytest.approx(4.0 / 14.0)
    assert result["across_block_variance_psi_b"] == pytest.approx(2.0)
    assert result["block_sd"] == pytest.approx(math.sqrt(2.0))
    expected_sigma_u2 = 2.0 - (4.0 / 14.0) * 0.5
    assert result["sigma_u2"] == pytest.approx(expected_sigma_u2)
    assert result["rho"] == pytest.approx(expected_sigma_u2 / 2.0)
    # sign-free: no mean, no per-block Psi value, and no sign anywhere in the return value
    assert set(result) == {
        "n_cell_rows",
        "n_blocks",
        "pooled_within_block_within_group_variance",
        "pooled_within_block_within_group_df",
        "across_block_variance_psi_b",
        "block_sd",
        "sigma_u2",
        "rho",
    }


def test_compute_variance_components_rejects_wrong_lattice_shape():
    rows = [_cell_row("b0", True, 1.0), _cell_row("b0", False, 0.0)]  # 1 matched / 1 mismatched
    with pytest.raises(ValueError):
        MODULE.compute_variance_components(rows)


# =================================================================================================
# Power formulas and the cross-lineage required-correlation formula, at the REAL audited literals
# =================================================================================================

# From the committed results/splicebert_positional_clock/real_panel_capacity_v1.json (B1 output).
_REAL_RHO = 0.368264006833713
_REAL_V9 = 0.14012168638599368


def test_power_formulas_reproduce_the_real_reported_block_counts():
    z = MODULE._z_critical()
    assert MODULE.blocks_for_sesoi(MODULE.var_psi(2, 2, _REAL_RHO, _REAL_V9), z=z) == 718
    assert MODULE.blocks_for_sesoi(MODULE.var_direction(1, 2, _REAL_RHO, _REAL_V9), z=z) == 996
    assert MODULE.blocks_for_sesoi(MODULE.var_psi(3, 6, _REAL_RHO, _REAL_V9), z=z) == 440
    assert MODULE.blocks_for_sesoi(MODULE.var_direction(3, 3, _REAL_RHO, _REAL_V9), z=z) == 533


def test_var_psi_at_3x3_equals_raw_v9_for_every_rho():
    # documented identity: (1/3 + 1/6) = 1/2, so var_psi(3, 6, rho, v9) == v9 regardless of rho.
    for rho in (0.0, 0.25, _REAL_RHO, 0.9):
        assert MODULE.var_psi(3, 6, rho, _REAL_V9) == pytest.approx(_REAL_V9)


def test_required_cross_lineage_correlation_matches_the_documented_correct_and_wrong_figures():
    z = MODULE._z_critical()
    correct_var = (2.0 - _REAL_RHO) * _REAL_V9
    wrong_var = _REAL_V9  # the raw-3x3-V9 bug

    r544_correct = MODULE.required_cross_lineage_correlation(544, correct_var, z=z)
    r726_correct = MODULE.required_cross_lineage_correlation(726, correct_var, z=z)
    assert r544_correct == pytest.approx(0.621, abs=1e-3)
    assert r726_correct == pytest.approx(0.494, abs=1e-3)

    r544_wrong = MODULE.required_cross_lineage_correlation(544, wrong_var, z=z)
    r726_wrong = MODULE.required_cross_lineage_correlation(726, wrong_var, z=z)
    assert r544_wrong == pytest.approx(0.382, abs=1e-3)
    assert r726_wrong == pytest.approx(0.175, abs=1e-3)

    # the two formulas must NOT be interchangeable at this precision
    assert abs(r544_correct - r544_wrong) > 0.2
    assert abs(r726_correct - r726_wrong) > 0.2


# =================================================================================================
# End-to-end: a small, fully self-consistent synthetic scenario
# =================================================================================================


def _with_phase_equal_marks(candidate: dict) -> dict:
    """Every real candidate carries `upstream_downstream_phase_equal` exactly twice (once under
    `annotation.checks`, once under `eligibility_checks`) -- mirrored here so the synthetic
    scenario's phase-equal occurrence count is genuinely 2x its candidate count, like the real
    census.
    """
    candidate = dict(candidate)
    candidate["annotation"] = {"checks": {"upstream_downstream_phase_equal": True}}
    candidate["eligibility_checks"] = {"upstream_downstream_phase_equal": True}
    return candidate


def _build_synthetic_scenario(tmp_path: Path) -> dict:
    """A tiny census/cells/certificate/analysis/artifact/manifest set that is internally
    consistent by construction: every reported figure is computed by calling the module's own
    (independently-tested-elsewhere) functions, so `run_audit` should reach AUDIT_PASS.
    """
    candidates = [
        _with_phase_equal_marks(c)
        for c in (
            {"role": "donor", "chrom": "chr1", "gene_id": "ENSGD0A.1", "phase": 0},
            {"role": "donor", "chrom": "chr1", "gene_id": "ENSGD0B.1", "phase": 0},
            {"role": "donor", "chrom": "chr1", "gene_id": "ENSGD1A.1", "phase": 1},
            {"role": "donor", "chrom": "chr1", "gene_id": "ENSGD1B.1", "phase": 1},
            {"role": "acceptor", "chrom": "chr1", "gene_id": "ENSGD0A.1", "phase": 0},  # shared w/ donor
            {"role": "acceptor", "chrom": "chr1", "gene_id": "ENSGA0B.1", "phase": 0},
            {"role": "acceptor", "chrom": "chr1", "gene_id": "ENSGA1A.1", "phase": 1},
            {"role": "acceptor", "chrom": "chr1", "gene_id": "ENSGA1B.1", "phase": 1},
            # off-allowlist chromosome -- must be excluded from every slot
            {"role": "donor", "chrom": "chr9", "gene_id": "ENSGOUT.1", "phase": 0},
        )
    ]
    for i in range(5):
        candidates.append(_with_phase_equal_marks(_scaffold_candidate(f"ENSGS{i}.1", ok=True)))
    exclusions = [
        {"edge_id": "e1", "gene_id": "ENSGZ1", "reasons": ["reasonA"]},
        {"edge_id": "e2", "gene_id": "ENSGZ2", "reasons": ["reasonA"]},
        {"edge_id": "e3", "gene_id": "ENSGZ3", "reasons": ["reasonB", "reasonA"]},
    ]
    census_path = _write_census(tmp_path, candidates, exclusions)
    census_bytes = census_path.read_bytes()
    census_sha256 = hashlib.sha256(census_bytes).hexdigest()
    census_size = len(census_bytes)

    allowed_chroms = {"chr1"}
    combined = MODULE.independent_recompute_from_census(
        census_path,
        allowed_chroms=allowed_chroms,
        construct_length=10,
        donor_pos=4,
        acceptor_pos=6,
        donor_window=MODULE.DEFAULT_DONOR_WINDOW,
        acceptor_window=MODULE.DEFAULT_ACCEPTOR_WINDOW,
    )
    slot_genes = combined["slot_genes"]

    cert_slots = {slot: {"gene_ids_sha256": MODULE._sha256_payload(sorted(genes))} for slot, genes in slot_genes.items()}
    certificate = {
        "config": {
            "allowed_chromosomes": ["chr1"],
            "construct_length": 10,
            "donor_pos": 4,
            "acceptor_pos": 6,
        },
        "capacity": {"slots": cert_slots},
    }
    certificate_path = tmp_path / "certificate.json"
    certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True))
    certificate_sha256 = MODULE._sha256_file(certificate_path)

    union4 = len(set().union(*(slot_genes[s] for s in MODULE._PLAIN_PHASE_SLOTS)))
    exact_ceiling = MODULE.exact_plain_ceiling(slot_genes)
    slack_ceiling = MODULE.ceiling_with_assignment_slack(slot_genes, union_size=union4)
    rotated_ceil = MODULE.rotated_ceiling(slot_genes)
    intersections = {p: len(slot_genes[f"donor_phase_{p}"] & slot_genes[f"acceptor_phase_{p}"]) for p in (0, 1, 2)}

    rows = []
    rows += [_cell_row("b0", True, v) for v in (1.0, 2.0, 3.0)]
    rows += [_cell_row("b0", False, v) for v in (0.0,) * 6]
    rows += [_cell_row("b1", True, v) for v in (4.0, 5.0, 6.0)]
    rows += [_cell_row("b1", False, v) for v in (1.0,) * 6]
    cells_doc = {"cell_rows": rows}
    cells_path = tmp_path / "cells.json"
    cells_path.write_text(json.dumps(cells_doc, indent=2, sort_keys=True))
    cells_sha256 = MODULE._sha256_file(cells_path)

    analysis_path = tmp_path / "analysis.json"
    analysis_path.write_text(json.dumps({"cells_result_sha256": cells_sha256}, indent=2, sort_keys=True))
    analysis_sha256 = MODULE._sha256_file(analysis_path)

    b1 = MODULE.compute_variance_components(rows)
    rho, v9 = b1["rho"], b1["across_block_variance_psi_b"]
    z = MODULE._z_critical()
    power = {
        "psi_plain_2x2": {"blocks_needed": MODULE.blocks_for_sesoi(MODULE.var_psi(2, 2, rho, v9), z=z)},
        "m_pm_plain_2x2": {"blocks_needed": MODULE.blocks_for_sesoi(MODULE.var_direction(1, 2, rho, v9), z=z)},
        "psi_full_3x3": {"blocks_needed": MODULE.blocks_for_sesoi(MODULE.var_psi(3, 6, rho, v9), z=z)},
        "m_pm_full_3x3": {"blocks_needed": MODULE.blocks_for_sesoi(MODULE.var_direction(3, 3, rho, v9), z=z)},
    }
    var_psi_plain = (2.0 - rho) * v9
    required_corr = {
        str(slack_ceiling): {
            "required_same_block_correlation_min": MODULE.required_cross_lineage_correlation(slack_ceiling, var_psi_plain, z=z)
        },
        str(exact_ceiling): {
            "required_same_block_correlation_min": MODULE.required_cross_lineage_correlation(exact_ceiling, var_psi_plain, z=z)
        },
    }

    artifact = {
        "schema": MODULE.AUDITED_ARTIFACT_SCHEMA,
        "census_receipt": {"sha256": census_sha256},
        "certificate_receipt": {"sha256": certificate_sha256},
        "cells_receipt": {"sha256": cells_sha256},
        "analysis_receipt": {"sha256": analysis_sha256},
        "slot_hash_verification": {
            "all_verified": True,
            "per_slot": {slot: True for slot in MODULE.ALL_SLOTS},
        },
        "b1_measured_variance_components": b1,
        "b2_real_panel_ceilings": {
            "union_of_four_phase01_module_slots": union4,
            "exact_ceiling": exact_ceiling,
            "ceiling_with_25pct_assignment_slack": slack_ceiling,
            "rotated_lattice_ceiling_for_comparison": rotated_ceil,
            "donor_acceptor_phase_intersections": {str(k): v for k, v in intersections.items()},
        },
        "b3_power_at_measured_components": power,
        "b4_cross_lineage_reframe": {
            "var_psi_per_block_used": var_psi_plain,
            "required_same_block_correlation": required_corr,
        },
        "b5_filter_relaxation_closed": {
            "census_exclusions_total": combined["n_exclusions"],
            "sole_reason_counts": combined["sole_reason_counts"],
            "multi_reason_exclusions": combined["multi_reason_exclusions"],
            "upstream_downstream_phase_equal": {
                "occurrences": combined["phase_equal_occurrences"],
                "candidates": combined["phase_equal_occurrences"] // 2,
            },
        },
    }
    artifact_path = tmp_path / "artifact.json"
    artifact_text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    artifact_path.write_text(artifact_text)
    artifact_sha256 = hashlib.sha256(artifact_text.encode("utf-8")).hexdigest()

    artifact_manifest = {
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "outputs": [{"sha256": artifact_sha256}],
    }
    artifact_manifest_path = tmp_path / "artifact.manifest.json"
    artifact_manifest_path.write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True))

    return {
        "artifact_path": artifact_path,
        "artifact_manifest_path": artifact_manifest_path,
        "census_path": census_path,
        "certificate_path": certificate_path,
        "cells_path": cells_path,
        "analysis_path": analysis_path,
        "census_sha256": census_sha256,
        "census_size": census_size,
        "cells_sha256": cells_sha256,
        "artifact": artifact,
        "artifact_text": artifact_text,
    }


def _patch_pins(monkeypatch, scenario: dict, *, exclusions_total: int) -> None:
    monkeypatch.setattr(MODULE, "_CENSUS_SHA256_PIN", scenario["census_sha256"])
    monkeypatch.setattr(MODULE, "_CENSUS_SIZE_PIN", scenario["census_size"])
    monkeypatch.setattr(MODULE, "_CELLS_SHA256_PIN", scenario["cells_sha256"])
    monkeypatch.setattr(MODULE, "_EXPECTED_EXCLUSIONS_TOTAL", exclusions_total)


def test_end_to_end_audit_pass_on_a_self_consistent_synthetic_scenario(tmp_path, monkeypatch):
    scenario = _build_synthetic_scenario(tmp_path)
    _patch_pins(monkeypatch, scenario, exclusions_total=3)

    out_path = tmp_path / "out" / "audit.json"
    out, manifest_path, audit = MODULE.run_audit(
        artifact_path=scenario["artifact_path"],
        artifact_manifest_path=scenario["artifact_manifest_path"],
        census_path=scenario["census_path"],
        certificate_path=scenario["certificate_path"],
        cells_path=scenario["cells_path"],
        analysis_path=scenario["analysis_path"],
        out_path=out_path,
    )

    failed = [c for c in audit["checks"] if not c["passed"]]
    assert failed == [], f"unexpected failing checks: {failed}"
    assert audit["status"] == "AUDIT_PASS"
    assert audit["passed"] is True
    assert out.exists() and manifest_path.exists()
    # CORRECTED SEMANTICS. This previously asserted True, mirroring a producer bug: the field
    # was computed as "do all inputs exist on this machine", which is a DIFFERENT question and
    # is in fact the condition under which the audit is NOT clean-checkout reproducible. The
    # 2.34 GB census and the 54 MB C43 cells file are untracked and live outside the repository,
    # so a clean clone can never re-run this audit. The artifact must say so, and the note beside
    # the field already did -- the boolean contradicted its own note.
    assert audit["clean_checkout_reexecutable"] is False
    assert audit["clean_checkout_blocking_inputs"], "the blocking inputs must be named, not implied"
    assert "NOT re-executable from a clean checkout" in audit["clean_checkout_note"]
    assert audit["independence"]["producer_verdict_functions_imported"] == []
    assert "networkx" in audit["independence"]["shared_primitives_imported"][0]


def test_end_to_end_audit_fail_when_a_reported_figure_is_wrong(tmp_path, monkeypatch):
    scenario = _build_synthetic_scenario(tmp_path)
    _patch_pins(monkeypatch, scenario, exclusions_total=3)

    # Corrupt one reported figure: the artifact now claims a rho it did not measure.
    artifact = json.loads(scenario["artifact_path"].read_text())
    artifact["b1_measured_variance_components"]["rho"] = 0.999
    corrupted_text = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    scenario["artifact_path"].write_text(corrupted_text)
    corrupted_sha256 = hashlib.sha256(corrupted_text.encode("utf-8")).hexdigest()
    manifest = json.loads(scenario["artifact_manifest_path"].read_text())
    manifest["outputs"][0]["sha256"] = corrupted_sha256
    scenario["artifact_manifest_path"].write_text(json.dumps(manifest, indent=2, sort_keys=True))

    out_path = tmp_path / "out" / "audit.json"
    out, manifest_path, audit = MODULE.run_audit(
        artifact_path=scenario["artifact_path"],
        artifact_manifest_path=scenario["artifact_manifest_path"],
        census_path=scenario["census_path"],
        certificate_path=scenario["certificate_path"],
        cells_path=scenario["cells_path"],
        analysis_path=scenario["analysis_path"],
        out_path=out_path,
    )

    assert audit["status"] == "AUDIT_FAIL"
    assert audit["passed"] is False
    by_id = {c["id"]: c for c in audit["checks"]}
    assert by_id["variance.rho_matches_artifact"]["passed"] is False
    # unrelated checks must still pass -- one bad figure does not collapse the whole table
    assert by_id["exclusions.total_matches_artifact"]["passed"] is True
    assert by_id["ceilings.exact_ceiling"]["passed"] is True


def test_audit_fails_closed_when_a_raw_input_is_missing(tmp_path, monkeypatch):
    scenario = _build_synthetic_scenario(tmp_path)
    _patch_pins(monkeypatch, scenario, exclusions_total=3)

    out_path = tmp_path / "out" / "audit.json"
    out, manifest_path, audit = MODULE.run_audit(
        artifact_path=scenario["artifact_path"],
        artifact_manifest_path=scenario["artifact_manifest_path"],
        census_path=tmp_path / "does_not_exist.json",
        certificate_path=scenario["certificate_path"],
        cells_path=scenario["cells_path"],
        analysis_path=scenario["analysis_path"],
        out_path=out_path,
    )
    assert audit["status"] == "AUDIT_FAIL"
    assert audit["passed"] is False
    by_id = {c["id"]: c for c in audit["checks"]}
    assert by_id["inputs.all_four_raw_inputs_present"]["passed"] is False


def test_no_gene_id_appears_anywhere_in_the_serialized_audit_output(tmp_path, monkeypatch):
    """Mirrors the family's own hygiene invariant: no string in the payload matches ^ENSG\\d."""
    scenario = _build_synthetic_scenario(tmp_path)
    _patch_pins(monkeypatch, scenario, exclusions_total=3)

    out_path = tmp_path / "out" / "audit.json"
    MODULE.run_audit(
        artifact_path=scenario["artifact_path"],
        artifact_manifest_path=scenario["artifact_manifest_path"],
        census_path=scenario["census_path"],
        certificate_path=scenario["certificate_path"],
        cells_path=scenario["cells_path"],
        analysis_path=scenario["analysis_path"],
        out_path=out_path,
    )
    serialized = out_path.read_text()
    assert "ENSG" not in serialized


def test_rerun_without_overwrite_flag_refuses_to_clobber(tmp_path, monkeypatch):
    scenario = _build_synthetic_scenario(tmp_path)
    _patch_pins(monkeypatch, scenario, exclusions_total=3)

    out_path = tmp_path / "out" / "audit.json"
    MODULE.run_audit(
        artifact_path=scenario["artifact_path"],
        artifact_manifest_path=scenario["artifact_manifest_path"],
        census_path=scenario["census_path"],
        certificate_path=scenario["certificate_path"],
        cells_path=scenario["cells_path"],
        analysis_path=scenario["analysis_path"],
        out_path=out_path,
    )
    from gfmcd.artifact_guard import ArtifactExistsError

    with pytest.raises(ArtifactExistsError):
        MODULE.run_audit(
            artifact_path=scenario["artifact_path"],
            artifact_manifest_path=scenario["artifact_manifest_path"],
            census_path=scenario["census_path"],
            certificate_path=scenario["certificate_path"],
            cells_path=scenario["cells_path"],
            analysis_path=scenario["analysis_path"],
            out_path=out_path,
        )


# =================================================================================================
# CLI surface
# =================================================================================================


def test_out_argument_is_required_with_no_default():
    with pytest.raises(SystemExit):
        MODULE.parse_args([])


def test_out_argument_has_no_default_value():
    ap_source = SCRIPT.read_text()
    assert '"--out", required=True' in ap_source
