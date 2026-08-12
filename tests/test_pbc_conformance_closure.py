from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO / "scripts/splicebert_positional_boundary_coherence"
sys.path.insert(0, str(SCRIPT_DIR))

import audit_protocol_code_conformance as conformance_runner  # noqa: E402
import freeze_carrier_factorial_target as freeze_runner  # noqa: E402
import run_carrier_factorial_target as target_runner  # noqa: E402
from gfmcd.provenance import file_record  # noqa: E402
from gfmcd.splicing import pbc_frozen_inputs as frozen_inputs  # noqa: E402


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_official_checkpoint_identity_is_hash_based_not_path_based(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    model = tmp_path / "external-model-directory"
    model.mkdir()
    weight = model / "pytorch_model.bin"
    weight.write_bytes(b"frozen-official-fixture")
    monkeypatch.setattr(
        frozen_inputs, "OFFICIAL_SPLICEBERT_WEIGHT_SHA256", _sha(weight)
    )
    assert frozen_inputs.validate_official_checkpoint(model) == weight.resolve()
    (model / "model.safetensors").write_bytes(b"alternate")
    with pytest.raises(RuntimeError, match="alternate model.safetensors"):
        frozen_inputs.validate_official_checkpoint(model)
    (model / "model.safetensors").unlink()
    weight.write_bytes(b"substituted")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        frozen_inputs.validate_official_checkpoint(model)


def test_candidate_external_input_records_are_execution_gates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    paths = {
        name: tmp_path / f"{name}.fixture"
        for name in frozen_inputs.FROZEN_BIOLOGY_SHA256
    }
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"fixture-{index}".encode())
    expected = {name: _sha(path) for name, path in paths.items()}
    monkeypatch.setattr(frozen_inputs, "FROZEN_BIOLOGY_SHA256", expected)
    package = {
        "input_pins": {
            "fasta": file_record(paths["fasta"]),
            "fai": file_record(paths["fai"]),
            "basic_gtf": {
                **file_record(paths["basic_gtf"]),
                "release": "GENCODE_v50",
                "coordinate_convention": (
                    "GTF_1_based_closed_parsed_as_0_based_half_open"
                ),
                "phase_convention": "GTF_frame_to_conventional_intron_phase",
            },
            "comprehensive_gtf": {
                **file_record(paths["comprehensive_gtf"]),
                "release": "GENCODE_v50",
                "role": "local_coding_strand_and_phase_safety_audit",
            },
            "maxent": {
                "donor_table": file_record(paths["maxent_donor_table"]),
                "acceptor_table": file_record(paths["maxent_acceptor_table"]),
                "equations": "Yeo_Burge_MaxEntScan_9mer_23mer_local_v1",
            },
        }
    }
    assert frozen_inputs.validate_candidate_input_pins(package) == {
        name: path.resolve() for name, path in paths.items()
    }
    paths["maxent_donor_table"].write_bytes(b"substituted")
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        frozen_inputs.validate_candidate_input_pins(package)


def test_conformance_chain_binds_passing_artifact_manifest_and_inputs(
    tmp_path: Path,
):
    revision = "f" * 40
    required = tmp_path / "final-code.py"
    required.write_text("# frozen\n")
    records = [file_record(required)]
    audit: dict[str, object] = {
        "schema": conformance_runner.CONFORMANCE_SCHEMA,
        "stage": "development",
        "status": "PASS",
        "passed": True,
        "target_outcomes_read": False,
        "model_loaded": False,
        "tracked_tree_clean": True,
        "git_revision": revision,
        "failures": [],
        "coverage_matrix": [{"requirement_id": "fixture", "passed": True}],
        "audited_inputs": records,
    }
    audit["content_sha256"] = freeze_runner._canonical_json_sha256(audit)
    audit_path = tmp_path / "conformance.json"
    audit_path.write_text(json.dumps(audit))
    manifest = {
        "experiment_id": conformance_runner.CONFORMANCE_EXPERIMENT_ID,
        "status": "PASS",
        "git_commit": revision,
        "config": {
            "stage": "development",
            "target_outcomes_read": False,
            "git_revision": revision,
        },
        "inputs": records,
        "outputs": [file_record(audit_path)],
    }
    assert freeze_runner.validate_conformance_chain(
        audit,
        manifest,
        audit_path=audit_path,
        required_inputs=(required,),
        frozen_revision=revision,
    ) == (required.resolve(),)
    failing = {**audit, "status": "FAIL", "passed": False}
    failing_body = dict(failing)
    failing_body.pop("content_sha256")
    failing["content_sha256"] = freeze_runner._canonical_json_sha256(failing_body)
    with pytest.raises(RuntimeError, match="conformance producer chain"):
        freeze_runner.validate_conformance_chain(
            failing,
            manifest,
            audit_path=audit_path,
            required_inputs=(required,),
            frozen_revision=revision,
        )


def test_target_runtime_must_match_seal(monkeypatch: pytest.MonkeyPatch):
    seal = {"runtime": {"device": "cpu", "batch_size": 4, "dtype": "float32"}}
    monkeypatch.setattr(target_runner, "pick_device", lambda _requested: "cpu")
    assert target_runner._require_sealed_runtime(
        seal,
        requested_device=None,
        batch_size=4,
        observed_dtype="float32",
    ) == seal["runtime"]
    with pytest.raises(RuntimeError, match="batch size"):
        target_runner._require_sealed_runtime(
            seal, requested_device=None, batch_size=8
        )
    with pytest.raises(RuntimeError, match="dtype"):
        target_runner._require_sealed_runtime(
            seal,
            requested_device=None,
            batch_size=4,
            observed_dtype="float16",
        )


def test_conformance_inventory_covers_full_stage_graph():
    files = {path.name for path in conformance_runner.conformance_code_files(REPO)}
    assert {
        "audit_protocol_code_conformance.py",
        "run_carrier_factorial_preflight.py",
        "freeze_carrier_factorial_target.py",
        "run_carrier_factorial_target.py",
        "audit_carrier_factorial.py",
        "freeze_p3_register_candidates.py",
        "freeze_p3_register_target.py",
        "run_p3_register_target.py",
        "run_p3_register_mediation_target.py",
        "test_pbc_conformance_closure.py",
    } <= files
