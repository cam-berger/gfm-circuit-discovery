"""Memorials for C43's re-executable crossed-register endpoint (CORRECTIONS.md R49).

The reproduction is only evidence if each check can fail on the measurement path, so the
planted cases below perturb the raw cell rows, the native-source rows, the pseudo-register
family and the floor stratification, and require the comparison against the controlling
audit to break where, and only where, the perturbation reaches.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO / "scripts/splicebert_exon_edge_compatibility/"
    "reproduce_crossed_direct_edge_replication.py"
)
NATIVE_FIELDS = {
    "instrument_endpoints.native_source_c_ac",
    "gates.source_edge_positive_control.passed",
    "independent_checks.raw_native_rows_and_block_arithmetic",
    "independent_checks.native_floor_calibration",
    "counts.native_rows",
    "native_result.native_block_rows",
    "native_result.floor_calibration",
    "cells_header.native_floor_calibration",
}


def _load():
    spec = importlib.util.spec_from_file_location("c43_reproduction", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


repro = _load()


@pytest.fixture(scope="module")
def audit():
    return json.loads(repro.AUDIT.read_text())


@pytest.fixture(scope="module")
def cells():
    rows, _, _ = repro.load_archive(repro.CELLS_ARCHIVE)
    return rows


@pytest.fixture(scope="module")
def native():
    rows, _, _ = repro.load_archive(repro.NATIVE_ARCHIVE)
    return rows


def _failing_fields(audit, cells, native):
    result = repro.recompute(audit, cells, native)
    return {row["field"] for row in result["comparisons"] if not row["within_audit_tolerance"]}


def _shift_native(native, field, delta, *, rows=None):
    """Move one raw margin and its coupling together, so each row stays self-consistent."""

    planted = copy.deepcopy(native)
    for row in planted["native_rows"] if rows is None else rows(planted):
        row[field] += delta
        # c_ac = donor_margin_intact - donor_margin_true_acceptor_ac
        if field == "donor_margin_intact":
            row["c_ac"] += delta
        elif field == "donor_margin_true_acceptor_ac":
            row["c_ac"] -= delta
    return planted


def test_archives_decompress_to_the_audited_raw_rows(audit):
    for archive, field in (
        (repro.CELLS_ARCHIVE, "cells_result_sha256"),
        (repro.NATIVE_ARCHIVE, "native_result_sha256"),
    ):
        raw = gzip.decompress(archive.read_bytes())
        assert hashlib.sha256(raw).hexdigest() == audit[field], archive.name


def test_executed_and_transcribed_code_is_identified_by_the_audit_manifest():
    manifest = json.loads(repro.AUDIT_MANIFEST.read_text())
    for relative in repro.PINNED_CODE:
        record = repro._record_for(manifest, "inputs", relative)
        assert repro._matches_record(REPO / relative, record), relative
    for relative, sha256 in repro.TRANSCRIBED_FROM.items():
        assert repro._record_for(manifest, "inputs", relative)["sha256"] == sha256, relative
    # The runner is pinned only by the uncarried freeze package, never by the audit manifest.
    for relative in repro.TRANSCRIBED_FROM_FREEZE_PACKAGE:
        with pytest.raises(ValueError, match="no unique inputs record"):
            repro._record_for(manifest, "inputs", relative)


def test_reproduction_matches_the_audit_bit_for_bit(tmp_path):
    out = tmp_path / "reproduction.json"
    assert repro.main(["--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["status"] == "REPRODUCED_BITWISE"
    assert result["comparison"]["bitwise_identical_fields"] == result["comparison"]["fields"]
    assert NATIVE_FIELDS <= {row["field"] for row in result["comparison"]["rows"]}
    headline = result["headline"]
    assert headline["pseudo_register_controls"] == 99
    assert headline["exact_add_one_one_sided_p"] == 0.01
    assert headline["classification"] == "MATERIAL_PREFERENTIAL_POSITIVE"
    assert round(headline["psi"]["mean"], 4) == 0.3901
    assert result["recomputed_gates"]["source_edge_positive_control"] is True
    assert round(headline["native_source_c_ac"]["ci95"][0], 4) == 0.2800
    assert headline["native_floors"] == {"donor": 0.0, "acceptor": 0.0}
    assert set(result["recomputed_native_audit_checks"].values()) == {True}
    assert "source_edge_positive_control" not in result["attested_gates_taken_from_audit"]
    manifest = json.loads(out.with_suffix(".manifest.json").read_text())
    assert manifest["outputs"][0]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert any(record["path"] == str(repro.NATIVE_ARCHIVE) for record in manifest["inputs"])


def test_reproduction_refuses_to_overwrite(tmp_path):
    out = tmp_path / "reproduction.json"
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        repro.main(["--out", str(out)])


def test_planted_cell_shift_breaks_the_target_comparison(audit, cells, native):
    planted = copy.deepcopy(cells)
    row = next(r for r in planted["cell_rows"] if (r["p"], r["q"]) == (0, 1))
    row["donor_margins"]["true_acceptor_ac"] -= 0.05
    row["effects"]["c_ac"] += 0.05
    failing = _failing_fields(audit, planted, native)
    assert {"target.primary_psi", "target.pseudo_phase", "target.block_endpoints"} <= failing


def test_inconsistent_raw_effect_is_rejected_before_inference(cells):
    planted = copy.deepcopy(cells)
    planted["cell_rows"][7]["effects"]["c_ac"] += 1e-6
    with pytest.raises(ValueError, match="raw-margin differences"):
        repro.reduce_cells(planted)


def test_wrong_pseudo_register_family_breaks_the_controls(audit, cells, monkeypatch):
    # Keep the recorded seed list; only the assignments drawn from it change.
    original = repro.balanced_pseudo_phase_assignments
    monkeypatch.setattr(
        repro,
        "balanced_pseudo_phase_assignments",
        lambda block_ids, *, seed: original(block_ids, seed=seed + 1),
    )
    target = repro.target_family(repro.reduce_cells(cells))
    recorded = audit["recomputed"]["target"]["pseudo_phase"]
    assert target["pseudo_phase"]["seeds"] == recorded["seeds"]
    assert target["pseudo_phase"]["control_means"] != recorded["control_means"]


def test_planted_native_shift_breaks_gate_2_and_the_native_record(audit, cells, native):
    # +0.05 coupling on one native source row of one block.
    planted = _shift_native(
        native, "donor_margin_true_acceptor_ac", -0.05,
        rows=lambda payload: payload["native_rows"][:1],
    )
    failing = _failing_fields(audit, cells, planted)
    assert {
        "instrument_endpoints.native_source_c_ac",
        "independent_checks.raw_native_rows_and_block_arithmetic",
        "independent_checks.native_floor_calibration",
        "native_result.native_block_rows",
        "native_result.floor_calibration",
        "cells_header.native_floor_calibration",
    } <= failing
    # The cell rows are untouched, so nothing in the target family may move.
    assert not {field for field in failing if field.startswith("target.")}


def test_weak_native_coupling_fails_gate_2_and_the_classification(audit, cells, native):
    # Native C_AC lower bound is ~0.28; removing 0.3 everywhere puts it below the +0.05 gate.
    planted = _shift_native(native, "donor_margin_true_acceptor_ac", 0.3)
    result = repro.recompute(audit, cells, planted)
    assert result["gates"]["source_edge_positive_control"] is False
    assert result["classification"] == "INSTRUMENT_UNRESOLVED"
    failing = {
        row["field"] for row in result["comparisons"] if not row["within_audit_tolerance"]
    }
    assert {
        "gates.source_edge_positive_control.passed",
        "final_replication_classification",
    } <= failing


def test_negative_native_floor_reaches_the_gate_8_record(audit, cells, native):
    # Lower every native donor margin by 4 logit units (coupling unchanged): the native lower bound,
    # ~3.58, turns negative, so the floor min(0, lower) is no longer 0.
    planted = _shift_native(native, "donor_margin_intact", -4.0)
    planted = _shift_native(planted, "donor_margin_true_acceptor_ac", -4.0)
    result = repro.recompute(audit, cells, planted)
    assert result["floors"]["donor_margin_intact"]["floor"] < 0
    failing = {
        row["field"] for row in result["comparisons"] if not row["within_audit_tolerance"]
    }
    assert "instrument_endpoints.phase_blind_site_health" in failing
    assert "instrument_endpoints.native_source_c_ac" not in failing
    # Forced by construction, not a test of the data: a floor is at most 0 and the crossed
    # lower bounds here are ~3.40 and ~2.22, so no native floor can fail gate 8's pass flag.
    assert result["gates"]["phase_blind_site_health"] is True
    assert "gates.phase_blind_site_health.passed" not in failing


def test_floor_stratum_encoding_is_load_bearing(audit, cells, native, monkeypatch):
    # The floors stratify on `chrom NUL strand`; `chrom:strand` sorts the strata differently.
    monkeypatch.setattr(repro, "NATIVE_FLOOR_STRATUM_SEPARATOR", ":")
    failing = _failing_fields(audit, cells, native)
    assert {
        "independent_checks.native_floor_calibration",
        "native_result.floor_calibration",
        "cells_header.native_floor_calibration",
    } <= failing
    # Gate 2 resamples with the cell blocks' `chrom:strand` indices and does not move.
    assert "instrument_endpoints.native_source_c_ac" not in failing


def test_native_source_identity_swap_is_rejected(cells, native):
    planted = copy.deepcopy(native)
    block = planted["native_rows"][0]["block_id"]
    donors = [
        row for row in planted["native_rows"]
        if row["block_id"] == block and row["source_role"] == "donor"
    ]
    donors[0]["source_id"], donors[1]["source_id"] = donors[1]["source_id"], donors[0]["source_id"]
    with pytest.raises(ValueError, match="native source identities"):
        repro.reduce_native(planted, cells)


def test_inconsistent_native_coupling_is_rejected(cells, native):
    planted = copy.deepcopy(native)
    planted["native_rows"][11]["c_ac"] += 1e-6
    with pytest.raises(ValueError, match="raw-margin difference"):
        repro.reduce_native(planted, cells)


def test_native_rows_with_other_bytes_are_refused(tmp_path, native):
    # Same content, different serialization: the audit binds bytes, not parsed values.
    other = tmp_path / "native.json"
    other.write_text(json.dumps(native))
    with pytest.raises(ValueError, match="native_rows_equal_audited_native_result"):
        repro.reproduce(
            cells_path=repro.CELLS_ARCHIVE,
            native_path=other,
            audit_path=repro.AUDIT,
            audit_manifest_path=repro.AUDIT_MANIFEST,
        )


def test_reproduction_restores_no_archived_module():
    """The archived arc stays archived (AGENTS.md; the successor study's frozen protocol).

    The reproduction transcribes the arithmetic it needs, so it may import nothing from
    `gfmcd.splicing`, and the archived modules it transcribes must not reappear in the tree.
    """

    source = SCRIPT.read_text()
    assert not re.search(r"^\s*(from|import)\s+gfmcd\.splicing", source, re.MULTILINE)
    for relative in (*repro.TRANSCRIBED_FROM, *repro.TRANSCRIBED_FROM_FREEZE_PACKAGE):
        if relative.startswith("src/"):
            assert not (REPO / relative).exists(), relative
