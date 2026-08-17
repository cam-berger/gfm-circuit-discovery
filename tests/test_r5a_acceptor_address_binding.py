from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert_acceptor_coupling/run_r5a_acceptor_address_binding.py"
)
SPEC = importlib.util.spec_from_file_location("run_r5a_acceptor_address_binding", SCRIPT)
R5A = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = R5A
SPEC.loader.exec_module(R5A)


def _zeros(n: int) -> np.ndarray:
    return np.zeros(n, dtype=float)


def _cells(n: int, values=(0.0, 0.0, 0.0, 0.0)):
    return {
        name: np.full(n, value, dtype=float)
        for name, value in zip(R5A.AV_CELL_ORDER, values)
    }


def _measurements(n: int = 40, *, material_content: bool = False):
    position = {
        head: {
            comparison: {
                measure: {
                    cell: _zeros(n) for cell in R5A.SHIFT_CELLS
                }
                for measure in R5A.SHIFT_MEASURES
            }
            for comparison in R5A.SHIFT_COMPARISONS
        }
        for head in R5A.SHIFT_HEADS
    }
    # full-position_shift and word_shift-joint_shift are both +0.40.  When
    # material_content=True the content simple effects are also +0.40, so the basic
    # position-sensitive path remains true while its nested dominance path must fail.
    primary = (0.8, 0.4, 0.4, 0.0) if material_content else (0.4, 0.4, 0.0, 0.0)
    for measure in R5A.SHIFT_MEASURES:
        position["L3H1"]["paired_adjacent"][measure] = {
            cell: np.full(n, value, dtype=float)
            for cell, value in zip(R5A.SHIFT_CELLS, primary)
        }

    qk = {
        head: {
            window: {"partner": _zeros(n), "pseudo": _zeros(n)}
            for h, window in R5A.QK_ARMS
            if h == head
        }
        for head, _ in R5A.QK_ARMS
    }
    qk["L3H1"]["paired"]["partner"] = np.full(n, 0.4)

    attention_value = {
        window: {
            "partner": _cells(n),
            "pseudo": _cells(n),
        }
        for window in R5A.WINDOWS
    }
    attention_value["paired"]["partner"] = _cells(n, (0.0, 0.0, 0.4, 0.4))

    l2_role = {
        builder: {
            "partner": _cells(n),
            "pseudo": _cells(n),
        }
        for builder in R5A.BUILDERS
    }
    l2_role["L2H15"]["partner"] = _cells(n, (0.0, 0.4, 0.0, 0.4))
    return {
        "position_shift": position,
        "qk_surgery": qk,
        "attention_value": attention_value,
        "l2_role": l2_role,
    }


def test_exact_four_families_and_closed_path_registry():
    rows = R5A.family_row_names()
    assert tuple(rows) == R5A.FAMILY_ORDER
    assert {name: len(value) for name, value in rows.items()} == {
        "position_shift": 60,
        "qk_surgery": 12,
        "attention_value": 90,
        "l2_role": 60,
    }
    assert all(len(value) == len(set(value)) for value in rows.values())
    paths = R5A.named_path_specs()
    assert tuple(paths) == R5A.PATH_ORDER
    assert paths["P_POSITION_SENSITIVE_L3_READER"]["family"] == "position_shift"
    assert paths["P_POSITION_DOMINANT_L3_READER"]["family"] == "position_shift"
    assert R5A.FAMILY_ALPHA == pytest.approx(0.0125)


def test_position_sensitive_is_nested_below_position_dominant():
    paths = R5A.named_path_specs()
    sensitive = paths["P_POSITION_SENSITIVE_L3_READER"]["requirements"]
    dominant = paths["P_POSITION_DOMINANT_L3_READER"]["requirements"]
    assert set(sensitive.items()) < set(dominant.items())
    assert not any("content_effect" in row for row in sensitive)
    assert any("content_effect" in row for row in dominant)
    for measure in R5A.SHIFT_MEASURES:
        for quantity in R5A.SHIFT_QUANTITIES[:2]:
            row = R5A.position_row_name(
                "L3H1", "paired_adjacent", measure, quantity
            )
            assert sensitive[row] == "MATERIAL_POSITIVE"


def test_all_derived_rows_replay_from_raw_cells_and_preserve_algebra():
    matrices = R5A.build_family_matrices(_measurements(n=7))
    names = R5A.family_row_names()
    qk = dict(zip(names["qk_surgery"], matrices["qk_surgery"]))
    np.testing.assert_allclose(
        qk[R5A.qk_row_name("L3H1", "paired", "specific")],
        qk[R5A.qk_row_name("L3H1", "paired", "partner")]
        - qk[R5A.qk_row_name("L3H1", "paired", "pseudo")],
    )
    av = dict(zip(names["attention_value"], matrices["attention_value"]))
    np.testing.assert_allclose(
        av[R5A.av_row_name("paired", "joint_lift", "partner")], 0.4
    )
    np.testing.assert_allclose(
        av[
            R5A.av_row_name(
                "paired", "value_effect_at_address_plus", "partner"
            )
        ],
        0.4,
    )
    l2 = dict(zip(names["l2_role"], matrices["l2_role"]))
    np.testing.assert_allclose(
        l2[
            R5A.l2_row_name(
                "L2H15", "address_effect_at_value_plus", "partner"
            )
        ],
        0.4,
    )
    np.testing.assert_allclose(
        l2[
            R5A.l2_row_name(
                "L2H15", "value_effect_at_address_plus", "partner"
            )
        ],
        0.0,
    )


def test_boundary_tost_iut_holm_classifier_supports_basic_position_with_content_effect():
    n = 40
    matrices = R5A.build_family_matrices(
        _measurements(n=n, material_content=True)
    )
    groups = [f"gene-{index:03d}" for index in range(n)]
    result = R5A.infer_all_families(
        matrices,
        groups,
        n_draws=399,
        seed=0,
        namespace="r5a-unit",
        min_genes=30,
    )
    assert (
        result["paths"]["P_POSITION_SENSITIVE_L3_READER"]["classification"]
        == "SUPPORTED"
    )
    assert (
        result["paths"]["P_POSITION_DOMINANT_L3_READER"]["classification"]
        == "NOT_SUPPORTED"
    )
    assert result["decisions"]["reader"] == "L3H1_DONOR_LOCAL_READER_SUPPORTED"
    assert result["decisions"]["opens_r5b"] is True
    assert result["decisions"]["l2_role"] == "L2H15_ADDRESS_REUSE_SUPPORTED"
    for family in R5A.FAMILY_ORDER:
        audit = result["families"][family]["boundary_randomisation_audit"]
        assert audit["all_decision_edges_share_one_sign_matrix"] is True
        assert audit["B"] == 399


def test_resolved_active_reader_control_is_not_collapsed_into_generic_null():
    n = 40
    measurements = _measurements(n=n)
    measurements["qk_surgery"]["L3H1"]["pseudo"]["partner"] = np.full(
        n, 0.40
    )
    result = R5A.infer_all_families(
        R5A.build_family_matrices(measurements),
        [f"gene-{index:03d}" for index in range(n)],
        n_draws=399,
        seed=0,
        namespace="r5a-active-control-unit",
        min_genes=30,
    )
    assert result["paths"]["P_QK_L3_READER"]["failure_mode"] == (
        "NONSELECTIVE_OR_CONTROL_ACTIVE"
    )
    assert result["decisions"]["reader"] == "NONSELECTIVE_OR_CONTROL_ACTIVE"


def test_shift_swaps_exact_component_windows_and_is_involutive():
    captured = torch.arange(2 * 12 * 3, dtype=torch.float32).reshape(2, 12, 3)
    left = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    right = torch.tensor([[7, 8, 9, 10], [7, 8, 9, 10]])
    shifted = R5A.swap_component_tensor(captured, left, right)
    rows = torch.arange(2)[:, None]
    torch.testing.assert_close(shifted[rows, left], captured[rows, right])
    torch.testing.assert_close(shifted[rows, right], captured[rows, left])
    restored = R5A.swap_component_tensor(shifted, left, right)
    torch.testing.assert_close(restored, captured)
    with pytest.raises(ValueError, match="disjoint"):
        R5A.swap_component_tensor(captured, left, left)


def test_qk_row_and_exact_four_logit_surgery_are_numerically_explicit():
    query = torch.zeros(2, 8, 4)
    key = torch.zeros_like(query)
    query[:, 5, 2:] = torch.tensor([1.0, 2.0])
    key[:, :, 2:] = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1).expand(-1, -1, 2)
    logits, attention = R5A.qk_attention_row(
        query,
        key,
        head=1,
        query_positions=torch.tensor([5, 5]),
        attention_mask=torch.ones(2, 8),
        num_heads=2,
    )
    torch.testing.assert_close(attention.sum(dim=1), torch.ones(2))
    source = logits + 10.0
    positions = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    edited = R5A.replace_selected_logits(logits, source, positions)
    rows = torch.arange(2)[:, None]
    torch.testing.assert_close(edited[rows, positions], source[rows, positions])
    mask = torch.zeros_like(edited, dtype=torch.bool)
    mask[rows, positions] = True
    torch.testing.assert_close(edited[~mask], logits[~mask])


class _TinyMLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cls = torch.nn.Linear(3, 7)

    def forward(self, input_ids, attention_mask=None):
        hidden = torch.nn.functional.one_hot(input_ids % 3, num_classes=3).float()
        return SimpleNamespace(logits=self.cls(hidden))


def test_full_mlm_replay_uses_complete_cls_and_acceptor_margin_only():
    model = _TinyMLM()
    adapter = SimpleNamespace(model=model)

    class Task:
        bid = {"A": 0, "C": 1, "G": 2, "T": 3}

        @staticmethod
        def margin(logits, positions):
            rows = torch.arange(len(positions))
            pos = torch.tensor(positions)
            values = logits[rows, pos]
            alt = values[:, [0, 1, 3]].max(dim=1).values
            return values[:, 2] - alt

    encoded = {
        "input_ids": torch.tensor([[0, 1, 2], [2, 1, 0]]),
        "attention_mask": torch.ones(2, 3),
    }
    result = R5A.full_mlm_replay_capability(adapter, Task(), encoded, [1, 1])
    assert result["ok"] is True
    assert result["uses_complete_model_cls"] is True
    assert result["donor_readout_constructed"] is False
    assert result["full_logits_max_abs_error"] == 0.0
    assert result["acceptor_G_margin_max_abs_error"] == 0.0


def test_target_stops_before_nonexistent_user_paths_are_opened():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "model",
            "--stage",
            "target",
            "--gtf",
            "/definitely/not/r5a.gtf",
            "--fasta",
            "/definitely/not/r5a.fa",
            "--model",
            "/definitely/not/r5a-model",
        ],
        cwd=SCRIPT.parents[2],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "R5a target authorization failed" in combined
    assert "remains DRAFT" in combined
    assert "No such file" not in combined


def test_protocol_and_binding_freeze_position_assignment_and_adjacent_bank():
    text = R5A.PROTOCOL.read_text()
    assert "paired↔adjacent" in text or "paired-to-adjacent" in text
    assert "position-sensitive" in text.lower()
    assert "fixed" in text.lower() and "V" in text
    payload = R5A.instrument_binding_payload()
    policy = payload["constants"]["adjacent_window_policy"]
    assert payload["constants"]["cohort_schema"] == "gfmcd.r5.acceptor_mechanism_cohort/2"
    assert policy["salt"] == "r5-adjacent-bank-v2"
    assert policy["shells"] == 8
    assert payload["constants"]["behavioral_endpoint"] == "masked_acceptor_G_margin_only"
    assert payload["constants"]["donor_readout_constructed"] is False


def test_calibration_rebuilds_27_boundary_signatures_and_all_path_plants():
    calibration = R5A.classifier_calibration(
        n_genes=R5A.PRODUCTION_N,
        n_draws=399,
        n_trials=1,
        seed=R5A.PRODUCTION_SEED,
    )
    assert calibration["production_classifier_used"] is True
    assert calibration["primitive_cells_rebuilt_before_every_classifier"] is True
    assert calibration["derived_rows_directly_shifted"] is False
    assert len(calibration["surfaces"]) == 27
    assert all(
        record["false_positive_trials"] == 0
        for record in calibration["surfaces"].values()
    )
    assert set(calibration["plants"]) == set(R5A.PATH_ORDER)
    assert all(record["successes"] == 1 for record in calibration["plants"].values())
    assert R5A.canonical_sha256(calibration["boundary_signature_registry"]) == (
        calibration["boundary_signature_registry_sha256"]
    )


def test_health_probe_positions_are_deterministic_and_exclude_guarded_sites():
    sequence = "ACGT" * 40
    protected = [40, 41, 42, 43, 80, 81, 82, 83]
    first = R5A.health_probe_positions(
        sequence,
        example_id="example-1",
        readout_position=60,
        protected_positions=protected,
    )
    second = R5A.health_probe_positions(
        sequence,
        example_id="example-1",
        readout_position=60,
        protected_positions=protected,
    )
    assert first == second
    assert len(first) == R5A.HEALTH_PROBES_PER_EXAMPLE
    guarded = set(range(57, 64))
    for position in protected:
        guarded.update(range(position - 4, position + 5))
    assert not set(first) & guarded


def test_checkpoint_aggregate_matches_r3c_ordered_bundle_schema(tmp_path):
    model = tmp_path / "model"
    nested = model / "nested"
    nested.mkdir(parents=True)
    (model / "config.json").write_text("{}")
    (nested / "weights.bin").write_bytes(b"weights")
    identity = R5A._checkpoint_identity(str(model))
    assert [row["relative_path"] for row in identity["ordered_bundle"]] == [
        "config.json",
        "nested/weights.bin",
    ]
    assert identity["aggregate_sha256"] == R5A.canonical_sha256(
        identity["ordered_bundle"]
    )


def test_model_main_calls_executable_stage_and_writes_artifact(monkeypatch, tmp_path):
    r3c = tmp_path / "r3c.json"
    r3c.write_text("{}")
    r3c_manifest = r3c.with_suffix(".manifest.json")
    r3c_manifest.write_text("{}")
    calls = []
    writes = []

    monkeypatch.setattr(
        R5A,
        "authorize_model_scoring",
        lambda stage: {
            "binding": {"stage": stage},
            "fingerprint": "f" * 64,
            "r5_stack_binding": {
                "schema": "test",
                "local_python_closure": {
                    "files": {str(R5A.RUNNER.relative_to(R5A.REPO)): {}}
                },
                "protocols": {str(R5A.PROTOCOL.relative_to(R5A.REPO)): {}},
            },
            "r5_stack_fingerprint_sha256": "s" * 64,
        },
    )

    def fake_stage(args, *, authorization, output):
        calls.append((args.stage, authorization["fingerprint"], output))
        return (
            {
                "status": "PREFLIGHT_PASS",
                "bucket": 0,
                "construction_sha256": "c" * 64,
                "source_identity": {
                    "gtf": {"size_bytes": 1, "sha256": "a" * 64},
                    "fasta": {"size_bytes": 1, "sha256": "b" * 64},
                },
                    "r3c_entry": {
                        "path": str(r3c),
                        "sha256": "d" * 64,
                        "manifest": {
                            "path": str(r3c_manifest),
                            "size_bytes": r3c_manifest.stat().st_size,
                            "sha256": "m" * 64,
                        },
                    },
                "checkpoint": {
                    "root": str(tmp_path),
                    "files": {},
                    "aggregate_sha256": "e" * 64,
                },
                    "bound_preflight": None,
                    "r5_exclusion_registry": {
                        "identity": {
                            "path": str(
                                output.with_suffix(".exclusion-registry.json")
                            ),
                            "size_bytes": len(
                                (json.dumps({"record_sha256": "r" * 64}, indent=2, sort_keys=True) + "\n").encode()
                            ),
                            "sha256": R5A.hashlib.sha256(
                                (json.dumps({"record_sha256": "r" * 64}, indent=2, sort_keys=True) + "\n").encode()
                            ).hexdigest(),
                            "record_sha256": "r" * 64,
                        },
                        "record": {"record_sha256": "r" * 64},
                    },
            },
            {"commit": "1" * 40, "branch": "test", "dirty": False},
        )

    class DummyManifest:
        def __init__(self):
            self.git_commit = None
            self.git_dirty = None
            self.inputs = []
            self.outputs = []

        def to_dict(self):
            return {
                "git_commit": self.git_commit,
                "git_dirty": self.git_dirty,
                "inputs": self.inputs,
                "outputs": self.outputs,
            }

    monkeypatch.setattr(R5A, "run_model_stage", fake_stage)
    monkeypatch.setattr(
        R5A.ExperimentManifest, "create", lambda *args, **kwargs: DummyManifest()
    )
    monkeypatch.setattr(
        R5A, "atomic_write_new", lambda path, data: writes.append((path, data))
    )
    output = tmp_path / "r5a.json"
    R5A.main(
        [
            "--mode",
            "model",
            "--stage",
            "preflight",
            "--r3c-artifact",
            str(r3c),
            "--out",
            str(output),
        ]
    )
    assert calls == [("preflight", "f" * 64, output.resolve())]
    assert [path for path, _data in writes] == [
        output.resolve(),
        output.with_suffix(".exclusion-registry.json").resolve(),
        output.with_suffix(".manifest.json").resolve(),
    ]
    manifest = json.loads(writes[-1][1])
    manifest_inputs = {Path(item["path"]).resolve() for item in manifest["inputs"]}
    assert R5A.RUNNER in manifest_inputs
    assert R5A.PROTOCOL in manifest_inputs


def test_exclusion_registry_is_derived_from_authenticated_r3c_only(monkeypatch):
    historical = ["historical-a", "historical-b"]
    by_bucket = {"0": ["r3c-a"], "1": ["r3c-b", "r3c-c"]}
    r3c_union = sorted(by_bucket["0"] + by_bucket["1"])
    monkeypatch.setattr(
        R5A,
        "R5_EXPECTED_CENSUS_IDENTITY",
        {
            "historical_r3_gene_count": len(historical),
            "historical_r3_gene_ids_sha256": R5A.canonical_sha256(historical),
            "r3c_gene_count": len(r3c_union),
            "r3c_gene_ids_sha256": R5A.canonical_sha256(r3c_union),
        },
    )
    record = {
        "cohort_audit": {
            "historical_reconstruction": {"excluded_gene_ids": historical},
            "current_census": {
                "fixed_selection_gene_ids_by_bucket": {
                    bucket: {
                        "gene_ids": genes,
                        "gene_ids_sha256": R5A.canonical_sha256(genes),
                    }
                    for bucket, genes in by_bucket.items()
                }
            },
        }
    }
    registry = R5A.derive_r5_exclusion_registry(
        record,
        r3c_artifact_identity={
            "path": "/r3c.json",
            "size_bytes": 10,
            "sha256": "a" * 64,
        },
        r3c_manifest_identity={
            "path": "/r3c.manifest.json",
            "size_bytes": 10,
            "sha256": "b" * 64,
        },
    )
    assert registry["r3c_gene_ids"] == r3c_union
    assert registry["derivation"].endswith("no free gene-list input")
    payload = {key: value for key, value in registry.items() if key != "record_sha256"}
    assert registry["record_sha256"] == R5A.canonical_sha256(payload)
    changed = json.loads(json.dumps(record))
    changed["cohort_audit"]["current_census"][
        "fixed_selection_gene_ids_by_bucket"
    ]["1"]["gene_ids"].append("unreviewed")
    with pytest.raises(RuntimeError):
        R5A.derive_r5_exclusion_registry(
            changed,
            r3c_artifact_identity={
                "path": "/r3c.json",
                "size_bytes": 10,
                "sha256": "a" * 64,
            },
            r3c_manifest_identity={
                "path": "/r3c.manifest.json",
                "size_bytes": 10,
                "sha256": "b" * 64,
            },
        )
