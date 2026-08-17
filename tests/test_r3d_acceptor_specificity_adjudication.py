from __future__ import annotations

import ast
from copy import deepcopy
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/splicebert_acceptor_coupling/run_r3d_acceptor_specificity_adjudication.py"
)
REGISTERED_REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "results/splicebert_acceptor_coupling/r3d_train_exposure_registry_v1.json"
)
SPEC = importlib.util.spec_from_file_location(
    "run_r3d_acceptor_specificity_adjudication", SCRIPT
)
R3D = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = R3D
SPEC.loader.exec_module(R3D)


def _measurements(n: int = 5, *, fill: float = 0.0):
    output = {R3D.MEAN_MODE: {}, R3D.NOISING_MODE: {}}
    output[R3D.MEAN_MODE]["L3H1"] = R3D.endpoint_vectors(
        np.full(n, fill), np.zeros(n)
    )
    for component in R3D.NOISING_COMPONENTS:
        output[R3D.NOISING_MODE][R3D.component_name(component)] = (
            R3D.endpoint_vectors(np.full(n, fill), np.zeros(n))
        )
    return output


def _health(ok: bool = True):
    return {
        path: {"ok": ok, "participants": [], "failed_participants": []}
        for path in R3D.PATH_NAMES
    }


def test_exact_14_primitive_23_row_family_and_frozen_paths_have_no_h7():
    primitive, primitive_names = R3D.build_primitive_matrix(_measurements())
    family, row_names = R3D.derive_family_from_primitives(primitive)

    assert primitive.shape == (14, 5)
    assert len(primitive_names) == len(set(primitive_names)) == 14
    assert family.shape == (23, 5)
    assert row_names == R3D.family_row_names()
    assert len(row_names) == len(set(row_names)) == 23
    specs = R3D.named_path_specs()
    assert tuple(specs) == R3D.PATH_NAMES
    assert len(specs[R3D.PATH_A]["requirements"]) == 6
    assert len(specs[R3D.PATH_B]["requirements"]) == 17
    assert all("L4H7" not in row for row in row_names)
    assert all("L4H7" not in row for row in specs[R3D.PATH_B]["requirements"])


def test_all_specifics_and_two_delivery_gaps_are_derived_after_primitive_values():
    measurements = _measurements(n=3)
    measurements[R3D.NOISING_MODE]["L4H11"] = R3D.endpoint_vectors(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([0.1, 0.2, 0.3])
    )
    measurements[R3D.NOISING_MODE]["L4H9"] = R3D.endpoint_vectors(
        np.asarray([3.0, 4.0, 5.0]), np.asarray([0.3, 0.4, 0.5])
    )
    measurements[R3D.NOISING_MODE]["L4H0"] = R3D.endpoint_vectors(
        np.asarray([0.2, 0.4, 0.6]), np.asarray([0.0, 0.1, 0.2])
    )
    measurements[R3D.NOISING_MODE]["L4H2"] = R3D.endpoint_vectors(
        np.asarray([0.4, 0.6, 0.8]), np.asarray([0.1, 0.1, 0.1])
    )
    measurements[R3D.NOISING_MODE]["L5H11"] = R3D.endpoint_vectors(
        np.asarray([0.6, 0.8, 1.0]), np.asarray([0.2, 0.2, 0.2])
    )
    primitive, _ = R3D.build_primitive_matrix(measurements)
    family, names = R3D.derive_family_from_primitives(primitive)
    by_name = {name: family[index] for index, name in enumerate(names)}

    for mode, head in [(R3D.MEAN_MODE, "L3H1")] + [
        (R3D.NOISING_MODE, R3D.component_name(component))
        for component in R3D.NOISING_COMPONENTS
    ]:
        np.testing.assert_allclose(
            by_name[R3D.family_row_name(mode, head, "specific")],
            by_name[R3D.family_row_name(mode, head, "partner")]
            - by_name[R3D.family_row_name(mode, head, "pseudo")],
        )
    expected_gap = np.mean(
        [
            by_name[R3D.family_row_name(R3D.NOISING_MODE, head, "partner")]
            for head in R3D.DELIVERY_CANDIDATES
        ],
        axis=0,
    ) - np.mean(
        [
            by_name[R3D.family_row_name(R3D.NOISING_MODE, head, "partner")]
            for head in R3D.DONOR_ROUTE_COMPARATORS
        ],
        axis=0,
    )
    np.testing.assert_allclose(
        by_name[R3D.delivery_gap_row_name("partner")], expected_gap
    )


def test_boundary_suite_has_36_unpacked_faces_and_each_is_algebraically_valid():
    surfaces = R3D.boundary_face_surface_specs()
    assert len(surfaces) == 36
    assert sum(spec["path"] == R3D.PATH_A for spec in surfaces.values()) == 8
    assert sum(spec["path"] == R3D.PATH_B for spec in surfaces.values()) == 28
    assert len({(spec["path"], spec["target_row"], spec["edge"]) for spec in surfaces.values()}) == 36
    assert R3D.boundary_face_suite_binding()["surface_count"] == 36
    for name, spec in surfaces.items():
        R3D._validate_surface(name, spec)


def test_test_inverted_bounds_are_exactly_dual_to_strict_reference_pvalues():
    n = 40
    groups = [f"g{index:02d}" for index in range(n)]
    primitive = np.repeat(R3D.coherent_path_offset()[:, None], n, axis=1)
    family, names = R3D.derive_family_from_primitives(primitive)
    result = R3D.infer_one_path(
        family,
        groups,
        names,
        R3D.PATH_A,
        n_draws=399,
        seed=0,
        namespace="r3d-test-inversion-dual",
        health={"ok": True},
    )

    assert result["classification"] == "SUPPORTED"
    assert result["all_edges_reject"] is True
    assert result["test_interval_coherence"] is True
    assert result["randomisation_audit"]["one_shared_sign_matrix"] is True
    for row in result["rows"].values():
        assert row["classification"] == row["required"]
        assert row["strict_test_interval_coherence"] is True
        for edge in row["edges"].values():
            reference = edge["reference_test"]
            inversion = edge["inversion"]
            assert edge["strict_reject"] is (reference["p_value"] < R3D.PATH_ALPHA)
            assert inversion["transition_test"]["p_value"] >= R3D.PATH_ALPHA
            assert inversion["rejecting_adjacent_test"]["p_value"] < R3D.PATH_ALPHA
            assert inversion["nonrejecting_adjacent_test"]["p_value"] >= R3D.PATH_ALPHA


def test_fixed_sequence_never_tests_path_b_when_path_a_is_not_supported(monkeypatch):
    calls = []

    def fake_infer(_family, _groups, _names, path, **_kwargs):
        calls.append(path)
        return {
            "path": path,
            "classification": "NOT_SUPPORTED",
            "health": {"ok": True},
            "rows": {},
        }

    monkeypatch.setattr(R3D, "infer_one_path", fake_infer)
    result = R3D.fixed_sequence_inference(
        np.zeros((23, 30)),
        [f"g{i}" for i in range(30)],
        R3D.family_row_names(),
        n_draws=39,
        seed=0,
        namespace="r3d-test-gate",
        path_health=_health(),
    )
    assert calls == [R3D.PATH_A]
    assert result["paths"][R3D.PATH_B]["classification"] == (
        "NOT_TESTED_GATE_CLOSED"
    )


def test_fixed_sequence_supports_b_only_after_a_and_controls_studywise_alpha():
    n = 40
    groups = [f"g{index:02d}" for index in range(n)]
    primitive = np.repeat(R3D.coherent_path_offset()[:, None], n, axis=1)
    family, names = R3D.derive_family_from_primitives(primitive)
    result = R3D.fixed_sequence_inference(
        family,
        groups,
        names,
        n_draws=399,
        seed=0,
        namespace="r3d-test-fixed-sequence-supported",
        path_health=_health(),
    )
    assert result["studywise_alpha"] == 0.05
    assert result["path_order"] == list(R3D.PATH_NAMES)
    assert result["paths"][R3D.PATH_A]["classification"] == "SUPPORTED"
    assert result["paths"][R3D.PATH_B]["classification"] == "SUPPORTED"


def test_three_way_health_can_only_shrink_and_never_revive():
    current = {
        path: {"ok": True, "participants": [path], "failed_participants": []}
        for path in R3D.PATH_NAMES
    }
    preflight = {
        path: {"health": {"ok": True}} for path in R3D.PATH_NAMES
    }
    result = R3D.three_way_monotone_health(
        r3c_health={R3D.PATH_A: True, R3D.PATH_B: False},
        current_health=current,
        r3d_preflight_paths=preflight,
    )
    assert result[R3D.PATH_A]["ok"] is True
    assert result[R3D.PATH_B]["ok"] is False

    suppressed = deepcopy(preflight)
    suppressed[R3D.PATH_A]["health"]["ok"] = False
    result = R3D.three_way_monotone_health(
        r3c_health={path: True for path in R3D.PATH_NAMES},
        current_health=current,
        r3d_preflight_paths=suppressed,
    )
    assert result[R3D.PATH_A]["ok"] is False
    assert result[R3D.PATH_A]["can_only_shrink"] is True


def test_path_b_health_failure_does_not_erase_a_supported_result_or_fail_global_gates():
    n = 40
    groups = [f"g{index:02d}" for index in range(n)]
    primitive = np.repeat(R3D.coherent_path_offset()[:, None], n, axis=1)
    family, names = R3D.derive_family_from_primitives(primitive)
    health = _health()
    health[R3D.PATH_B]["ok"] = False
    inference = R3D.fixed_sequence_inference(
        family,
        groups,
        names,
        n_draws=399,
        seed=0,
        namespace="r3d-test-local-health",
        path_health=health,
    )
    assert inference["paths"][R3D.PATH_A]["classification"] == "SUPPORTED"
    assert inference["paths"][R3D.PATH_B]["classification"] == "HEALTH_FAILED"

    measurements = {R3D.MEAN_MODE: {}, R3D.NOISING_MODE: {}}
    for mode, head in {
        participant
        for participants in R3D._path_health_participants().values()
        for participant in participants
    }:
        measurements[mode][head] = {"health": {"total": 1}}
    monotone = {
        path: {
            "ok": path == R3D.PATH_A,
            "can_only_shrink": True,
            "participants": [],
            "failed_participants": [] if path == R3D.PATH_A else ["x"],
        }
        for path in R3D.PATH_NAMES
    }
    gates = R3D.build_gates(
        n_genes=40,
        stage="smoke",
        native={"partner": np.full(40, 0.3), "pseudo": np.zeros(40)},
        construction={
            "all_single_acceptor_G": True,
            "all_donor_targets_GT": True,
            "all_motif_clean": True,
            "policy": {"donor_readout_constructed": False},
        },
        cohort_audit={
            "preconstruction_allowed_chromosomes": list(R3D.TRAIN_CHROMOSOMES),
            "provenance_hash_reads_full_fasta_bytes": True,
            "forbidden_chromosomes_materialized_as_examples": False,
            "forbidden_chromosomes_model_scored": False,
            "one_row_per_gene": True,
            "frozen_registry_capacity_replayed": True,
            "selected_rows": 40,
            "registry_gene_intersection_count": 40,
        },
        measurements=measurements,
        calibration={
            "n_boundary_faces": 36,
            "all_plants_meet_stage_threshold": True,
        },
        health=monotone,
        r3c_health_ok={R3D.PATH_A: True, R3D.PATH_B: False},
    )
    assert gates["three_way_monotone_health_registry_valid"] is True
    assert all(gates.values())


def test_preflight_cannot_authorize_target_when_path_a_health_is_suppressed():
    measurements = {R3D.MEAN_MODE: {}, R3D.NOISING_MODE: {}}
    for mode, head in {
        participant
        for participants in R3D._path_health_participants().values()
        for participant in participants
    }:
        measurements[mode][head] = {"health": {"total": 1}}
    health = {
        path: {
            "ok": path != R3D.PATH_A,
            "can_only_shrink": True,
            "participants": [],
            "failed_participants": ["x"] if path == R3D.PATH_A else [],
        }
        for path in R3D.PATH_NAMES
    }
    gates = R3D.build_gates(
        n_genes=256,
        stage="preflight",
        native={"partner": np.full(256, 0.3), "pseudo": np.zeros(256)},
        construction={
            "all_single_acceptor_G": True,
            "all_donor_targets_GT": True,
            "all_motif_clean": True,
            "policy": {"donor_readout_constructed": False},
        },
        cohort_audit={
            "preconstruction_allowed_chromosomes": list(R3D.TRAIN_CHROMOSOMES),
            "provenance_hash_reads_full_fasta_bytes": True,
            "forbidden_chromosomes_materialized_as_examples": False,
            "forbidden_chromosomes_model_scored": False,
            "one_row_per_gene": True,
            "frozen_registry_capacity_replayed": True,
            "selected_rows": 256,
            "registry_gene_intersection_count": 256,
        },
        measurements=measurements,
        calibration={
            "all_boundary_face_engineering_fpr_ok": True,
            "all_plants_at_least_462_of_500": True,
        },
        health=health,
        r3c_health_ok={path: True for path in R3D.PATH_NAMES},
    )
    assert gates["three_way_monotone_health_registry_valid"] is True
    assert gates["preflight_path_a_health_authorizes_target_entry"] is False
    assert not all(gates.values())


def test_one_row_per_gene_is_deterministic_and_independent_of_input_order():
    def example(gene, example_id):
        return SimpleNamespace(gene_id=gene, example_id=example_id)

    rows = [
        (example("g1", "e1"), 10),
        (example("g1", "e2"), 20),
        (example("g2", "e3"), 30),
        (example("g3", "e4"), 40),
        (example("g3", "e5"), 50),
    ]
    forward = R3D.one_hash_ranked_row_per_gene(rows)
    reverse = R3D.one_hash_ranked_row_per_gene(list(reversed(rows)))
    assert [(row[0].example_id, row[1]) for row in forward] == [
        (row[0].example_id, row[1]) for row in reverse
    ]
    assert len(forward) == len({row[0].gene_id for row in rows}) == 3


def test_cohort_restricts_before_load_and_target_takes_every_registry_disjoint_gene(monkeypatch):
    calls = []

    def example(gene, chrom="chr1"):
        return SimpleNamespace(
            gene_id=gene,
            example_id=f"{chrom}:{gene}",
            chrom=chrom,
            donor_pos=10,
            sequence="A" * 10 + "GT" + "A" * 88,
            real_don=(),
        )

    examples = [example(f"g{i}") for i in range(R3D.TARGET_MIN_GENES + 3)]

    def fake_load(gtf, fasta, split, n, *, tag, allowed_chromosomes):
        calls.append((gtf, fasta, split, n, tag, tuple(allowed_chromosomes)))
        return examples

    monkeypatch.setattr(R3D, "load_examples_unscored", fake_load)
    monkeypatch.setattr(R3D, "matched_decoy_donor", lambda _example: 10)
    # The equality exclusion would remove the rows, so provide a distinct clean pseudo.
    for item in examples:
        item.sequence = "A" * 10 + "GT" + "A" * 8 + "GT" + "A" * 78
    monkeypatch.setattr(R3D, "matched_decoy_donor", lambda _example: 20)
    monkeypatch.setattr(R3D, "donor_alt_admissible", lambda *_args: True)
    monkeypatch.setattr(R3D, "planted_donor_motifs", lambda *_args: ())
    args = SimpleNamespace(
        gtf="g.gtf", fasta="g.fa", stage="target", n_genes=None
    )
    registry = {
        "schema": R3D.REGISTRY_SCHEMA,
        "scope_id": R3D.REGISTRY_SCOPE,
        "scope": {"global_freshness_claimed": False},
        "gene_ids": ["g0", "g1"],
        "gene_ids_sha256": R3D.canonical_sha256(["g0", "g1"]),
    }
    registry["train_capacity"] = R3D.train_capacity_record(
        [(item, 20) for item in examples], registry["gene_ids"], {}
    )
    selected, decoys, audit = R3D.load_cohort(args, registry)

    assert calls == [
        (
            "g.gtf",
            "g.fa",
            "train",
            None,
            R3D.CURRENT_TAG,
            R3D.TRAIN_CHROMOSOMES,
        )
    ]
    assert set(R3D.TRAIN_CHROMOSOMES).isdisjoint(R3D.FORBIDDEN_CHROMOSOMES)
    assert len(selected) == R3D.TARGET_MIN_GENES + 1
    assert len(selected) == len(decoys) == audit["selected_genes"]
    assert audit["target_uses_every_eligible_registry_disjoint_gene"] is True
    assert audit["registry_gene_intersection_count"] == 0
    assert audit["provenance_hash_reads_full_fasta_bytes"] is True
    assert audit["forbidden_chromosomes_materialized_as_examples"] is False
    assert audit["forbidden_chromosomes_model_scored"] is False


def test_registry_is_content_audited_and_requires_sorted_unique_ids(tmp_path):
    genes = ["g1", "g2"]
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema": R3D.REGISTRY_SCHEMA,
                "scope_id": R3D.REGISTRY_SCOPE,
                "scope": {"global_freshness_claimed": False},
                "gene_ids": genes,
                "gene_ids_sha256": R3D.canonical_sha256(genes),
            }
        )
    )
    assert R3D.read_exposed_gene_registry(path)["gene_ids"] == genes
    malformed = json.loads(path.read_text())
    malformed["gene_ids"] = ["g2", "g1"]
    path.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="sorted unique"):
        R3D.read_exposed_gene_registry(path)


def test_registered_builder_artifact_is_directly_consumable_by_runner():
    record = R3D.read_exposed_gene_registry(REGISTERED_REGISTRY)
    assert record["scope_id"] == R3D.REGISTRY_SCOPE
    assert record["scope"]["global_freshness_claimed"] is False
    assert len(record["gene_ids"]) == 1112
    assert record["train_capacity"]["eligible_genes_one_row_each"] == 4843
    assert record["train_capacity"]["exposed_development"][
        "genes_one_row_each"
    ] == 934
    assert record["train_capacity"]["fresh_target"]["genes_one_row_each"] == 3909
    assert record["input_bindings"]["genome_index"]["sha256"] == (
        "3b425de206296a5c8053023fa5ca61da43cfe78c1737c12e58c83367c7e83c21"
    )


def test_target_seal_opens_only_after_reviewed_preflight_is_pinned():
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--stage",
            "target",
            "--gtf",
            "/definitely/missing.gtf",
            "--fasta",
            "/definitely/missing.fa",
            "--model",
            "/definitely/missing-model",
            "--exposed-gene-registry",
            "/definitely/missing-registry.json",
            "--r3c-preflight-artifact",
            "/definitely/missing-r3c.json",
            "--preflight-artifact",
            "/definitely/missing-r3d.json",
            "--out",
            "/definitely/missing-output.json",
        ],
        cwd=SCRIPT.parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    output = completed.stdout + completed.stderr
    assert "hard-blocked" not in output
    assert "checkpoint path is missing: /definitely/missing-model" in output


def test_bindable_preflight_requires_exact_current_hashes_and_fingerprint():
    hashes = {
        "protocol_sha256": "a" * 64,
        "exposed_gene_registry_sha256": "b" * 64,
        "exposed_gene_registry_manifest_sha256": "f" * 64,
        "r3c_v2_1_runner_sha256": "c" * 64,
        "r3c_v2_1_preflight_sha256": "d" * 64,
    }
    kwargs = {
        "frozen_protocol_sha256": "a" * 64,
        "frozen_fingerprint_sha256": "e" * 64,
        "frozen_registry_sha256": "b" * 64,
        "frozen_registry_manifest_sha256": "f" * 64,
        "frozen_r3c_runner_sha256": "c" * 64,
        "frozen_r3c_preflight_sha256": "d" * 64,
    }
    assert R3D.authorize_bindable_preflight(
        hashes=hashes,
        fingerprint="e" * 64,
        protocol_text="> **Status: FROZEN.**\n",
        **kwargs,
    ) is True
    drifted = dict(hashes, exposed_gene_registry_sha256="f" * 64)
    with pytest.raises(R3D.TargetSealError, match="input hashes differ"):
        R3D.authorize_bindable_preflight(
            hashes=drifted,
            fingerprint="e" * 64,
            protocol_text="> **Status: FROZEN.**\n",
            **kwargs,
        )
    with pytest.raises(R3D.TargetSealError, match="fingerprint differs"):
        R3D.authorize_bindable_preflight(
            hashes=hashes,
            fingerprint="f" * 64,
            protocol_text="> **Status: FROZEN.**\n",
            **kwargs,
        )


def test_bound_execution_source_drift_changes_the_instrument_fingerprint(monkeypatch):
    monkeypatch.setattr(R3D.r3c, "runtime_identity", lambda _device: {"runtime": "fixed"})
    common = {
        "batch": R3D.PRODUCTION_BATCH,
        "device": "cpu",
        "seed": R3D.PRODUCTION_SEED,
        "n_randomisation": R3D.PRODUCTION_RANDOMISATION,
        "n_trials": R3D.PRODUCTION_TRIALS,
    }
    first = R3D.instrument_binding_payload(
        hashes={
            "bound_code_sources": {
                "src/gfmcd/circuits/hf_bert.py": {"sha256": "a" * 64}
            }
        },
        **common,
    )
    second = R3D.instrument_binding_payload(
        hashes={
            "bound_code_sources": {
                "src/gfmcd/circuits/hf_bert.py": {"sha256": "b" * 64}
            }
        },
        **common,
    )
    assert R3D.canonical_sha256(first) != R3D.canonical_sha256(second)


def test_target_binding_check_precedes_cohort_construction_and_model_loading():
    source = inspect.getsource(R3D.main)
    frozen_check = source.index(
        "fingerprint != FROZEN_INSTRUMENT_FINGERPRINT_SHA256"
    )
    assert frozen_check < source.index("load_cohort(args, registry)")
    assert frozen_check < source.index("load_adapter(args.model")


def test_artifact_and_sibling_manifest_can_be_pinned_separately(tmp_path):
    artifact = tmp_path / "preflight.json"
    artifact.write_text('{"schema":"x","status":"DEVELOPMENT"}\n')
    artifact_digest = R3D.sha256_file(artifact)
    manifest = artifact.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "command": ["runner", "--stage", "preflight"],
                "status": "DEVELOPMENT",
                "outputs": [
                    {
                        "path": str(artifact.resolve()),
                        "size_bytes": artifact.stat().st_size,
                        "sha256": artifact_digest,
                    }
                ]
            }
        )
    )
    manifest_digest = R3D.sha256_file(manifest)
    record, package = R3D._load_artifact_package(
        artifact, artifact_digest, manifest_digest
    )
    assert record == {"schema": "x", "status": "DEVELOPMENT"}
    assert package["manifest"]["sha256"] == manifest_digest
    with pytest.raises(ValueError, match="sibling manifest differs"):
        R3D._load_artifact_package(artifact, artifact_digest, "0" * 64)


def test_clean_target_artifact_package_replays_target_not_preflight_stage(tmp_path):
    artifact = tmp_path / "target.json"
    identity = {"commit": "a" * 40, "dirty": False}
    fingerprint = "b" * 64
    record = {
        "schema": R3D.RESULT_SCHEMA,
        "status": "PROSPECTIVE_VERIFICATION",
        "stage": "target",
        "target_bindable_preflight": True,
        "git_authorization_identity": identity,
        "input_snapshot": {},
        "input_snapshot_sha256": R3D.canonical_sha256({}),
        "instrument_fingerprint_sha256": fingerprint,
        "command": ["runner", "--stage", "target"],
    }
    artifact.write_text(json.dumps(record))
    digest = R3D.sha256_file(artifact)
    manifest = artifact.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "command": record["command"],
                "status": record["status"],
                "git_dirty": False,
                "git_commit": identity["commit"],
                "inputs": [],
                "config": {
                    "stage": "target",
                    "input_snapshot_sha256": R3D.canonical_sha256({}),
                    "instrument_fingerprint_sha256": fingerprint,
                },
                "outputs": [
                    {
                        "path": str(artifact),
                        "size_bytes": artifact.stat().st_size,
                        "sha256": digest,
                    }
                ],
            }
        )
    )
    loaded, _ = R3D._load_artifact_package(artifact, digest, None)
    assert loaded["stage"] == "target"
    malformed = json.loads(manifest.read_text())
    malformed["config"]["stage"] = "preflight"
    manifest.write_text(json.dumps(malformed))
    with pytest.raises(ValueError, match="clean git/input/instrument"):
        R3D._load_artifact_package(artifact, digest, None)


def test_protocol_and_runner_bind_reviewed_preflight_package():
    frozen_preflight_inputs = {
        "protocol": R3D.FROZEN_PROTOCOL_SHA256,
        "fingerprint": R3D.FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
        "registry": R3D.FROZEN_EXPOSED_GENE_REGISTRY_SHA256,
        "registry_manifest": R3D.FROZEN_EXPOSED_GENE_REGISTRY_MANIFEST_SHA256,
        "r3c_runner": R3D.FROZEN_R3C_V2_1_RUNNER_SHA256,
        "r3c_preflight": R3D.FROZEN_R3C_V2_1_PREFLIGHT_SHA256,
    }
    assert frozen_preflight_inputs == {
        "protocol": "e319694f3f0b53695e787a67b5bcb6f1d016ac5e148e36e3050e090284f66592",
        "fingerprint": "0f35295643238e9360d22bfe550e981e7788fa0975a3a9fdfe96c5f5f94fa5de",
        "registry": "17ad316d5f9c478e37b91f34d87bc1d0c6a714f411947419e9dbcced2443598c",
        "registry_manifest": "745d7a31837c3575edcd9816464f82fc7f23b440b00b06fde4f6185a5f50afde",
        "r3c_runner": "25db5d95e7c5a99697f69d2153ec0d770571e3063106aa13c7f00f74e5065e31",
        "r3c_preflight": "c0c9306fe894ce73bfd981a14c85e2aec6660da33f01d9716cee4327b3ff2d3e",
    }
    assert R3D.FROZEN_PREFLIGHT_ARTIFACT_SHA256 == (
        "7c89765e3cf4505b848788972c3a3ab6c3033dd3953d7d3fd4b6e1a527da9d31"
    )
    assert R3D.FROZEN_PREFLIGHT_MANIFEST_SHA256 == (
        "38b65671bd2bf3b4ea23aa6082a5e1c2841dda02386e6e582c664ee54ec00172"
    )
    assert R3D.PROTOCOL.name == "DRAFT_R3D_ACCEPTOR_SPECIFICITY_ADJUDICATION.md"
    protocol = R3D.PROTOCOL.read_text()
    assert "FROZEN_FOR_BINDABLE_PREFLIGHT" in protocol
    assert R3D.sha256_file(R3D.PROTOCOL) == R3D.FROZEN_PROTOCOL_SHA256
    assert R3D.sha256_file(R3D.RUNNER) != R3D.normalized_runner_sha256()
    assert "14 primitive rows" in protocol
    assert "23 derived family rows" in protocol
    assert "Path B contains 28 decision edges" in protocol
    assert "`L4H7` is not" in protocol
    assert R3D.REGISTRY_SCOPE in protocol
    assert "pointwise one-sided decision bounds" in protocol
    assert "fixed sequence" in protocol
    assert "all eight literal pins" in protocol
    assert "at most\n20 of 200" in protocol
    assert "at least 462" in protocol
    assert "zero-residual coherent classifier/algebra calibration" in protocol
    assert "observed-centered-residual plant outcome" in protocol
    assert "explicit non-gating diagnostic" in protocol
    assert "candidate-selection-disjoint" in protocol
    assert "α_A = 0.040" not in protocol


def test_compact_calibration_executes_all_faces_and_fixed_sequence_plant():
    n = 30
    groups = [f"g{index:02d}" for index in range(n)]
    primitive = np.zeros((14, n), dtype=float)
    result = R3D.calibration_preflight(
        primitive,
        groups,
        n_draws=39,
        n_trials=1,
        seed=0,
        namespace="r3d-test-compact-calibration",
    )
    assert result["n_boundary_faces"] == 36
    assert len(result["boundary_faces"]) == 36
    assert result["trials_per_surface"] == 1
    assert result["all_plants_meet_stage_threshold"] is True
    assert all(
        record["successes"] == 1 for record in result["plant_successes"].values()
    )
    assert result["all_plants_at_least_462_of_500"] is False


def test_smoke_calibration_gates_on_zero_residual_mechanics_and_reports_observed_power():
    n = R3D.SMOKE_N_GENES
    groups = [f"g{index:02d}" for index in range(n)]
    wave = np.linspace(-2.0, 2.0, n)
    high_variance = np.vstack(
        [np.roll(wave, index) for index in range(len(R3D.primitive_row_names()))]
    )
    namespace = "r3d-test-smoke-calibration-split"

    previous_smoke_behavior = R3D.calibration_preflight(
        high_variance,
        groups,
        n_draws=R3D.SMOKE_RANDOMISATION,
        n_trials=R3D.SMOKE_TRIALS,
        n_plant_trials=R3D.SMOKE_TRIALS,
        seed=0,
        namespace=namespace,
    )
    assert previous_smoke_behavior["all_plants_meet_stage_threshold"] is False

    staged = R3D.calibration_for_stage(
        high_variance,
        groups,
        stage="smoke",
        n_draws=R3D.SMOKE_RANDOMISATION,
        n_trials=R3D.SMOKE_TRIALS,
        seed=0,
        namespace=namespace,
    )

    assert staged["gating_basis"] == "ZERO_RESIDUAL_SMOKE_MECHANICS"
    assert staged["gating_calibration"]["all_plants_meet_stage_threshold"] is True
    diagnostic = staged["observed_centered_residual_non_gating_diagnostic"]
    assert diagnostic == previous_smoke_behavior
    assert diagnostic["plant_support_patterns"] == {
        R3D.PATH_A: [0, 0],
        R3D.PATH_B: [0, 0],
    }


def test_smoke_calibration_report_labels_the_observed_residual_outcome_non_gating():
    n = R3D.SMOKE_N_GENES
    groups = [f"g{index:02d}" for index in range(n)]
    wave = np.linspace(-2.0, 2.0, n)
    high_variance = np.vstack(
        [np.roll(wave, index) for index in range(len(R3D.primitive_row_names()))]
    )
    staged = R3D.calibration_for_stage(
        high_variance,
        groups,
        stage="smoke",
        n_draws=R3D.SMOKE_RANDOMISATION,
        n_trials=R3D.SMOKE_TRIALS,
        seed=0,
        namespace="r3d-test-smoke-calibration-report",
    )

    report = R3D.smoke_calibration_report(staged)

    assert report["gating_basis"] == "ZERO_RESIDUAL_SMOKE_MECHANICS"
    assert report["zero_residual_mechanics"]["plant_support_patterns"] == {
        R3D.PATH_A: [1, 1],
        R3D.PATH_B: [1, 1],
    }
    diagnostic = report["observed_centered_residual_non_gating_diagnostic"]
    assert diagnostic["plant_support_patterns"] == {
        R3D.PATH_A: [0, 0],
        R3D.PATH_B: [0, 0],
    }
    assert diagnostic["gating"] is False
    assert diagnostic["can_authorize_target"] is False


def test_production_calibration_remains_the_exact_observed_residual_256_by_500_path():
    n = R3D.PREFLIGHT_N_GENES
    groups = [f"g{index:03d}" for index in range(n)]
    wave = np.linspace(-8.0, 8.0, n)
    high_variance = np.vstack(
        [np.roll(wave, index) for index in range(len(R3D.primitive_row_names()))]
    )
    kwargs = {
        "n_draws": 39,
        "n_trials": 1,
        "seed": 0,
        "namespace": "r3d-test-production-observed-residuals",
    }
    previous_production_behavior = R3D.calibration_preflight(
        high_variance,
        groups,
        n_plant_trials=R3D.PRODUCTION_PLANT_TRIALS,
        **kwargs,
    )
    assert previous_production_behavior["all_plants_at_least_462_of_500"] is False

    staged = R3D.calibration_for_stage(
        high_variance,
        groups,
        stage="preflight",
        **kwargs,
    )

    assert staged["gating_basis"] == "OBSERVED_CENTERED_RESIDUALS_PRODUCTION"
    assert staged["gating_calibration"] == previous_production_behavior
    assert staged["observed_centered_residual_non_gating_diagnostic"] is None


def test_runner_declares_train_verification_not_sealed_confirmation():
    source = SCRIPT.read_text()
    assert "not_sealed_or_independent_confirmation" in source
    assert '"tier": "PROSPECTIVE_ENDPOINT_SPECIFIC_TRAIN_VERIFICATION"' in source
    assert '"donor_readout_constructed": False' in source
    assert "No masked-donor endpoint is constructed" in source


def test_registry_package_replays_dedicated_manifest_and_all_registered_source_bindings():
    record, package = R3D._load_registry_package(
        REGISTERED_REGISTRY,
        None,
        None,
        require_clean_generator=False,
    )
    assert record["scope_id"] == R3D.REGISTRY_SCOPE
    assert package["artifact"]["sha256"] == R3D.sha256_file(REGISTERED_REGISTRY)
    assert package["manifest"]["sha256"] == R3D.sha256_file(
        REGISTERED_REGISTRY.with_suffix(".manifest.json")
    )


def _copy_registry_package(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / REGISTERED_REGISTRY.name
    artifact.write_bytes(REGISTERED_REGISTRY.read_bytes())
    manifest = json.loads(
        REGISTERED_REGISTRY.with_suffix(".manifest.json").read_text()
    )
    manifest["outputs"][0]["path"] = str(artifact)
    manifest_path = artifact.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest))
    return artifact, manifest, manifest_path


def test_registry_package_rejects_empty_or_forged_400_shard_inventory(tmp_path):
    artifact, manifest, manifest_path = _copy_registry_package(tmp_path)
    head = manifest["inputs"]["head_screen_shards"]
    head["files"] = []
    head["files_count"] = 0
    head["inventory_sha256"] = R3D.canonical_sha256([])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="semantic binding"):
        R3D._load_registry_package(
            artifact, None, None, require_clean_generator=False
        )


def test_registry_package_rejects_source_a_binding_and_committed_builder_tampering(
    tmp_path, monkeypatch
):
    artifact, manifest, manifest_path = _copy_registry_package(tmp_path)
    manifest["inputs"]["claude_transcript"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="semantic binding"):
        R3D._load_registry_package(
            artifact, None, None, require_clean_generator=False
        )

    artifact, manifest, manifest_path = _copy_registry_package(tmp_path / "builder")
    manifest["code"]["git_dirty"] = False
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(R3D, "git_blob_sha256", lambda *_args, **_kwargs: "0" * 64)
    with pytest.raises(ValueError, match="semantic binding"):
        R3D._load_registry_package(
            artifact, None, None, require_clean_generator=True
        )


def test_registry_package_requires_the_fixed_registry_builder_path(tmp_path):
    artifact, manifest, manifest_path = _copy_registry_package(tmp_path)
    record = json.loads(artifact.read_text())
    record["generator"]["script"] = "README.md"
    artifact.write_text(json.dumps(record))
    manifest["code"]["builder"] = "README.md"
    manifest["outputs"][0]["size_bytes"] = artifact.stat().st_size
    manifest["outputs"][0]["sha256"] = R3D.sha256_file(artifact)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="semantic binding"):
        R3D._load_registry_package(
            artifact, None, None, require_clean_generator=False
        )


def test_calibration_operating_characteristic_thresholds_are_exact():
    assert R3D.face_max_count_pass(20, 200) is True
    assert R3D.face_max_count_pass(21, 200) is False
    assert R3D.plant_count_pass(462, 500) is True
    assert R3D.plant_count_pass(461, 500) is False
    _, face_upper = R3D.one_sided_binomial_bounds(20, 200, alpha=0.05)
    plant_lower, _ = R3D.one_sided_binomial_bounds(462, 500, alpha=0.05)
    assert face_upper == pytest.approx(R3D.FACE_CP_UPPER_AT_MAX_COUNT)
    assert plant_lower == pytest.approx(R3D.PLANT_CP_LOWER_AT_THRESHOLD)
    assert R3D.FACE_SUITE_PASS_LOWER_AT_POINT05 > 0.95
    assert R3D.PLANT_PASS_PROBABILITY_AT_POINT95 > 0.99


def _compact_production_calibration_fixture():
    namespace = "r3d:production-boundary-face-preflight"
    faces = {}
    for index, (name, spec) in enumerate(R3D.boundary_face_surface_specs().items()):
        pattern = [0] * R3D.PRODUCTION_TRIALS
        _, upper = R3D.one_sided_binomial_bounds(
            0, R3D.PRODUCTION_TRIALS, alpha=R3D.PATH_ALPHA
        )
        faces[name] = {
            "path": spec["path"],
            "target_row": spec["target_row"],
            "edge": spec["edge"],
            "trials": R3D.PRODUCTION_TRIALS,
            "false_positive_trials": 0,
            "false_positive_rate": 0.0,
            "one_sided_95pct_clopper_pearson_upper": upper,
            "one_sided_cp_upper_is_descriptive_only": True,
            "maximum_false_positive_trials": R3D.MAX_FACE_FALSE_POSITIVES,
            "engineering_fpr_ok": True,
            "outer_namespace_template": f"{namespace}:face:{name}:outer:<trial>",
            "classifier_namespace_template": (
                f"{namespace}:face:{name}:classifier:<trial>"
            ),
            "support_pattern": pattern,
            "pattern_sha256": R3D.canonical_sha256(pattern),
            "outer_null_digest_sha256": f"{index:064x}",
            "classifier_audit_digest_sha256": f"{index + 100:064x}",
        }
    plant_patterns = {
        path: [1] * R3D.PRODUCTION_PLANT_TRIALS for path in R3D.PATH_NAMES
    }
    plants = {
        path: {
            "successes": R3D.PRODUCTION_PLANT_TRIALS,
            "trials": R3D.PRODUCTION_PLANT_TRIALS,
            "power": 1.0,
            "minimum_successes": R3D.MIN_PLANT_SUCCESSES,
            "powered": True,
        }
        for path in R3D.PATH_NAMES
    }
    return {
        "schema": "gfmcd.r3d.boundary_face_calibration/1",
        "randomisation_unit": "whole gene",
        "translation_invariant_centering": True,
        "n_primitive_rows": 14,
        "n_family_rows": 23,
        "n_boundary_faces": 36,
        "trials_per_surface": R3D.PRODUCTION_TRIALS,
        "plant_trials": R3D.PRODUCTION_PLANT_TRIALS,
        "n_calibration_draws": R3D.PRODUCTION_RANDOMISATION,
        "seed": R3D.PRODUCTION_SEED,
        "namespace": namespace,
        "boundary_face_suite": R3D.boundary_face_suite_binding(),
        "boundary_faces": faces,
        "all_boundary_face_engineering_fpr_ok": True,
        "face_familywise_max_count_operating_characteristics": {
            "point_null_per_face_probability": R3D.PATH_ALPHA,
            "maximum_false_positive_trials": R3D.MAX_FACE_FALSE_POSITIVES,
            "single_face_fail_probability": R3D.FACE_FAIL_PROBABILITY_AT_POINT05,
            "union_bound_any_of_36_faces_fail": (
                R3D.FACE_SUITE_UNION_BOUND_AT_POINT05
            ),
            "suite_pass_probability_lower_bound": (
                R3D.FACE_SUITE_PASS_LOWER_AT_POINT05
            ),
            "cp_upper_at_maximum_count_descriptive_only": (
                R3D.FACE_CP_UPPER_AT_MAX_COUNT
            ),
            "does_not_assume_cross_face_independence": True,
        },
        "plant_delta": R3D.PLANT_DELTA,
        "coherent_primitive_plant_sha256": R3D._array_sha256(
            R3D.coherent_path_offset()
        ),
        "plant_successes": plants,
        "plant_operating_characteristics": {
            "trials": R3D.PRODUCTION_PLANT_TRIALS,
            "minimum_successes": R3D.MIN_PLANT_SUCCESSES,
            "one_sided_95pct_cp_lower_at_threshold": (
                R3D.PLANT_CP_LOWER_AT_THRESHOLD
            ),
            "pass_probability_at_true_joint_success_point95": (
                R3D.PLANT_PASS_PROBABILITY_AT_POINT95
            ),
            "path_b_successes_must_not_exceed_path_a_successes": True,
        },
        "plant_outer_namespace_template": f"{namespace}:plant:outer:<trial>",
        "plant_classifier_namespace_template": (
            f"{namespace}:plant:classifier:<trial>"
        ),
        "plant_support_patterns": plant_patterns,
        "plant_support_pattern_sha256": {
            path: R3D.canonical_sha256(pattern)
            for path, pattern in plant_patterns.items()
        },
        "plant_classifier_audit_digest_sha256": "a" * 64,
        "plant_outer_null_digest_sha256": "b" * 64,
        "all_plants_at_least_462_of_500": True,
        "all_plants_meet_stage_threshold": True,
        "engineering_not_inferential": True,
    }


def test_semantic_calibration_validation_replays_production_and_rejects_tampered_pass(
    monkeypatch,
):
    calibration = _compact_production_calibration_fixture()
    calls = []

    def replay(values, groups, **kwargs):
        calls.append((np.asarray(values).shape, list(groups), kwargs))
        return calibration

    monkeypatch.setattr(R3D, "calibration_preflight", replay)
    primitive = np.zeros((14, 30))
    groups = [f"g{i}" for i in range(30)]
    R3D._validate_r3d_calibration(
        calibration,
        primitive_values=primitive,
        primitive_groups=groups,
    )
    assert calls[0][2]["n_draws"] == R3D.PRODUCTION_RANDOMISATION
    assert calls[0][2]["n_trials"] == R3D.PRODUCTION_TRIALS
    assert calls[0][2]["n_plant_trials"] == R3D.PRODUCTION_PLANT_TRIALS

    tampered = deepcopy(calibration)
    name = next(iter(tampered["boundary_faces"]))
    face = tampered["boundary_faces"][name]
    pattern = [1] * 21 + [0] * (R3D.PRODUCTION_TRIALS - 21)
    _, upper = R3D.one_sided_binomial_bounds(21, 200, alpha=0.05)
    face.update(
        {
            "support_pattern": pattern,
            "pattern_sha256": R3D.canonical_sha256(pattern),
            "false_positive_trials": 21,
            "false_positive_rate": 21 / 200,
            "one_sided_95pct_clopper_pearson_upper": upper,
            "engineering_fpr_ok": True,
        }
    )
    with pytest.raises(ValueError, match="CP gate differ"):
        R3D._validate_r3d_calibration(
            tampered,
            primitive_values=primitive,
            primitive_groups=groups,
        )


def test_target_attempt_key_is_semantic_create_only_and_precedes_any_target_access(
    tmp_path, monkeypatch
):
    registry = R3D.read_exposed_gene_registry(REGISTERED_REGISTRY)
    hashes = {
        "gtf_sha256": "1" * 64,
        "fasta_sha256": "2" * 64,
        "fasta_index_sha256": "3" * 64,
        "checkpoint_aggregate_sha256": "4" * 64,
        "exposed_gene_registry_sha256": "5" * 64,
        "exposed_gene_registry_manifest_sha256": "6" * 64,
    }
    first = R3D.target_identity_from_registry(registry, hashes)
    repackaged = dict(hashes)
    repackaged["exposed_gene_registry_sha256"] = "7" * 64
    repackaged["exposed_gene_registry_manifest_sha256"] = "8" * 64
    second = R3D.target_identity_from_registry(registry, repackaged)
    assert first == second
    monkeypatch.setattr(R3D, "target_attempt_ledger", lambda _repo=R3D.REPO: tmp_path)
    path = R3D.canonical_target_attempt_path(first)
    assert path == R3D.canonical_target_attempt_path(second)
    receipt = R3D.write_target_attempt(
        path,
        target_identity=first,
        invocation=["runner", "--stage", "target"],
        authorized_git_identity={"commit": "a" * 40, "dirty": False},
        instrument_fingerprint="b" * 64,
        pins={"protocol_sha256": "c" * 64},
        input_snapshot={},
    )
    assert Path(receipt["path"]).is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        R3D.write_target_attempt(
            path,
            target_identity=first,
            invocation=["runner", "--stage", "target"],
            authorized_git_identity={"commit": "a" * 40, "dirty": False},
            instrument_fingerprint="b" * 64,
            pins={"protocol_sha256": "c" * 64},
            input_snapshot={},
        )
    source = inspect.getsource(R3D.main)
    reservation = source.index("write_target_attempt(")
    assert reservation < source.index("load_cohort(args, registry)")
    assert reservation < source.index("load_adapter(args.model")


def test_global_gate_failure_suppresses_official_paths_and_positive_license():
    diagnostic = {
        path: {"classification": "SUPPORTED", "rows": {}}
        for path in R3D.PATH_NAMES
    }
    official = R3D.official_paths_after_global_gates(
        diagnostic, {"assay": True, "provenance": False}
    )
    assert all(
        record["classification"] == "INCONCLUSIVE_GLOBAL_GATE_FAILED"
        for record in official.values()
    )
    assert all(
        record["diagnostic_classification"] == "SUPPORTED"
        for record in official.values()
    )
    assert all(record["global_gate_failures"] == ["provenance"] for record in official.values())
    assert R3D.scientific_license(
        stage="target", all_gates_pass=False, official_paths=official
    ).startswith("no positive")


def test_target_license_tracks_supported_official_paths_not_merely_global_gates():
    neither = {
        R3D.PATH_A: {"classification": "NOT_SUPPORTED"},
        R3D.PATH_B: {"classification": "NOT_TESTED_GATE_CLOSED"},
    }
    assert R3D.scientific_license(
        stage="target", all_gates_pass=True, official_paths=neither
    ) == "no positive scientific disposition"
    a_only = deepcopy(neither)
    a_only[R3D.PATH_A]["classification"] = "SUPPORTED"
    assert R3D.scientific_license(
        stage="target", all_gates_pass=True, official_paths=a_only
    ).endswith("L3H1 reuse only")
    both = deepcopy(a_only)
    both[R3D.PATH_B]["classification"] = "SUPPORTED"
    assert "distinct-route" in R3D.scientific_license(
        stage="target", all_gates_pass=True, official_paths=both
    )


def test_clean_margin_baseline_is_replayed_independently_not_as_partner_minus_pseudo():
    native = {
        "partner": np.asarray([1.0, 2.0, 3.0]),
        "pseudo": np.asarray([0.5, 0.25, 0.75]),
        "clean_margin": np.asarray([8.0, 9.0, 10.0]),
    }
    stats = R3D.baseline_sufficient_statistics(native, ["g1", "g2", "g3"])
    values, groups = R3D.r3c._reconstruct_cluster_array(stats)
    assert groups == ["g1", "g2", "g3"]
    assert np.mean(values[2]) == pytest.approx(9.0)
    assert np.mean(values[2]) != pytest.approx(
        np.mean(values[0]) - np.mean(values[1])
    )


def test_result_manifest_output_binding_is_relocation_safe(tmp_path, monkeypatch):
    new_repo = tmp_path / "new-worktree"
    artifact = new_repo / "results/splicebert_acceptor_coupling/preflight.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text('{"schema":"x","status":"DEVELOPMENT"}\n')
    digest = R3D.sha256_file(artifact)
    manifest = artifact.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "command": ["runner"],
                "status": "DEVELOPMENT",
                "outputs": [
                    {
                        "path": "/old/worktree/results/splicebert_acceptor_coupling/preflight.json",
                        "size_bytes": artifact.stat().st_size,
                        "sha256": digest,
                    }
                ],
            }
        )
    )
    monkeypatch.setattr(R3D, "REPO", new_repo)
    record, _ = R3D._load_artifact_package(artifact, digest, None)
    assert record["schema"] == "x"
    recorded = {
        role: {
            "path": f"/old/worktree/results/study/{role}.json",
            "size_bytes": 10 + index,
            "sha256": str(index + 1) * 64,
        }
        for index, role in enumerate(("artifact", "manifest"))
    }
    relocated = deepcopy(recorded)
    for item in relocated.values():
        item["path"] = item["path"].replace("/old/worktree", "/new/worktree")
    assert R3D._packages_semantically_equal(recorded, relocated)
    relocated["manifest"]["sha256"] = "f" * 64
    assert not R3D._packages_semantically_equal(recorded, relocated)


def _synthetic_package(tmp_path: Path, name: str):
    artifact = tmp_path / f"{name}.json"
    manifest = tmp_path / f"{name}.manifest.json"
    artifact.write_text(f'{{"name":"{name}"}}\n')
    manifest.write_text(f'{{"name":"{name}-manifest"}}\n')

    def item(path: Path):
        return {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": R3D.sha256_file(path),
        }

    return {"artifact": item(artifact), "manifest": item(manifest)}


def _synthetic_target_for_semantic_replay(tmp_path: Path):
    n = R3D.TARGET_MIN_GENES
    target_gene_ids = [f"target-gene-{index:04d}" for index in range(n)]
    selected_ids = sorted(
        target_gene_ids,
        key=lambda gene: (R3D._hash_rank(R3D.GENE_RANK_SALT, gene), gene),
    )
    assert selected_ids != sorted(selected_ids)
    selected_rows = [
        {
            "example_id": f"example-{gene}",
            "gene_id": gene,
            "chrom": "chr1",
            "decoy_donor": 100 + index,
        }
        for index, gene in enumerate(selected_ids)
    ]
    exposed = ["direct-exposure-gene"]
    capacity = {
        "eligible_genes_one_row_each": n + len(exposed),
        "exposed_development": {"genes_one_row_each": len(exposed)},
        "fresh_target": {
            "genes_one_row_each": n,
            "gene_ids_sha256": R3D.canonical_sha256(sorted(selected_ids)),
            "ordered_one_row_per_gene_sha256": R3D.canonical_sha256(
                selected_rows
            ),
        },
    }
    registry = {
        "schema": R3D.REGISTRY_SCHEMA,
        "scope_id": R3D.REGISTRY_SCOPE,
        "scope": {"global_freshness_claimed": False},
        "gene_ids": exposed,
        "gene_ids_sha256": R3D.canonical_sha256(exposed),
        "train_capacity": capacity,
    }
    registry_package = _synthetic_package(tmp_path, "registry")
    r3c_package = _synthetic_package(tmp_path, "r3c-preflight")
    r3d_package = _synthetic_package(tmp_path, "r3d-preflight")
    packages = (registry_package, r3c_package, r3d_package)
    pre_attempt_snapshot = {
        str(Path(item["path"]).resolve()): {
            "size_bytes": item["size_bytes"],
            "sha256": item["sha256"],
        }
        for package in packages
        for item in package.values()
    }
    hashes = {
        "protocol_sha256": "0" * 64,
        "runner_normalized_sha256": "1" * 64,
        "r3c_v2_1_runner_sha256": "2" * 64,
        "r3c_v2_1_preflight_sha256": r3c_package["artifact"]["sha256"],
        "r3c_v2_1_preflight_manifest_sha256": r3c_package["manifest"]["sha256"],
        "exposed_gene_registry_sha256": registry_package["artifact"]["sha256"],
        "exposed_gene_registry_manifest_sha256": registry_package["manifest"]["sha256"],
        "stat_core_sha256": "3" * 64,
        "bound_code_sources": {},
        "gtf_sha256": "4" * 64,
        "fasta_sha256": "5" * 64,
        "fasta_index_sha256": "6" * 64,
        "checkpoint_aggregate_sha256": "7" * 64,
    }
    binding = {"fixture": "target-semantic-replay", "hashes": hashes}
    fingerprint = R3D.canonical_sha256(binding)
    identity = {"commit": "a" * 40, "dirty": False}
    command = ["runner", "--stage", "target"]
    target_identity = R3D.target_identity_from_registry(registry, hashes)
    pins = {
        "protocol_sha256": R3D.FROZEN_PROTOCOL_SHA256,
        "instrument_fingerprint_sha256": R3D.FROZEN_INSTRUMENT_FINGERPRINT_SHA256,
        "preflight_artifact_sha256": R3D.FROZEN_PREFLIGHT_ARTIFACT_SHA256,
        "preflight_manifest_sha256": R3D.FROZEN_PREFLIGHT_MANIFEST_SHA256,
        "registry_artifact_sha256": R3D.FROZEN_EXPOSED_GENE_REGISTRY_SHA256,
        "registry_manifest_sha256": R3D.FROZEN_EXPOSED_GENE_REGISTRY_MANIFEST_SHA256,
        "r3c_runner_sha256": R3D.FROZEN_R3C_V2_1_RUNNER_SHA256,
        "r3c_preflight_sha256": R3D.FROZEN_R3C_V2_1_PREFLIGHT_SHA256,
    }
    attempt = R3D.write_target_attempt(
        tmp_path / "target-attempt.json",
        target_identity=target_identity,
        invocation=command,
        authorized_git_identity=identity,
        instrument_fingerprint=fingerprint,
        pins=pins,
        input_snapshot=pre_attempt_snapshot,
    )
    snapshot = dict(pre_attempt_snapshot)
    snapshot[attempt["path"]] = {
        "size_bytes": attempt["size_bytes"],
        "sha256": attempt["sha256"],
    }
    snapshot = dict(sorted(snapshot.items()))

    primitive = np.repeat(R3D.coherent_path_offset()[:, None], n, axis=1)
    primitive[0] += np.arange(n, dtype=float) / 10_000_000.0
    family, row_names = R3D.derive_family_from_primitives(primitive)
    component_health = {"ok": True, "total": 1, "failures": []}
    measurements = {R3D.MEAN_MODE: {}, R3D.NOISING_MODE: {}}
    for mode, head in [(R3D.MEAN_MODE, "L3H1")] + [
        (R3D.NOISING_MODE, R3D.component_name(component))
        for component in R3D.NOISING_COMPONENTS
    ]:
        measurements[mode][head] = {"health": deepcopy(component_health)}
    current_health = R3D.observed_path_health(measurements)
    r3c_health = {path: True for path in R3D.PATH_NAMES}
    r3d_preflight = {
        "family": {
            "production_preflight": {
                "n_boundary_faces": 36,
                "all_boundary_face_engineering_fpr_ok": True,
                "all_plants_at_least_462_of_500": True,
                "all_plants_meet_stage_threshold": True,
            }
        },
        "paths": {
            path: {"health": {"ok": True}} for path in R3D.PATH_NAMES
        },
    }
    health = R3D.three_way_monotone_health(
        r3c_health=r3c_health,
        current_health=current_health,
        r3d_preflight_paths=r3d_preflight["paths"],
    )
    replay_paths = {}
    replay_rows = {}
    for path, spec in R3D.named_path_specs().items():
        rows = {
            row: {"row": row, "required": required, "fixture": True}
            for row, required in spec["requirements"].items()
        }
        replay_rows.update(rows)
        replay_paths[path] = {
            "path": path,
            "classification": "SUPPORTED",
            "health": health[path],
            "rows": rows,
        }
    replay = {
        "schema": "gfmcd.r3d.fixed_sequence_inference/1",
        "studywise_alpha": R3D.PATH_ALPHA,
        "path_order": list(R3D.PATH_NAMES),
        "paths": replay_paths,
    }
    family_lookup = {name: family[index] for index, name in enumerate(row_names)}
    components = {}
    expected_modes = {
        "L3H1": [R3D.MEAN_MODE, R3D.NOISING_MODE],
        **{
            R3D.component_name(component): [R3D.NOISING_MODE]
            for component in R3D.ROUTE_COMPONENTS
        },
    }
    for head, modes in expected_modes.items():
        components[head] = {"kind": "head", "modes": {}}
        for mode in modes:
            components[head]["modes"][mode] = {
                "health": deepcopy(component_health),
                "endpoints": {
                    endpoint: {
                        "raw_mean": float(
                            np.mean(
                                family_lookup[
                                    R3D.family_row_name(mode, head, endpoint)
                                ]
                            )
                        ),
                        "decision": replay_rows[
                            R3D.family_row_name(mode, head, endpoint)
                        ],
                    }
                    for endpoint in R3D.ENDPOINTS
                },
            }
    partner = 0.30 + np.arange(n, dtype=float) / 100_000_000.0
    pseudo = np.zeros(n)
    clean_margin = 1.0 + np.arange(n, dtype=float) / 100_000_000.0
    baselines = {
        "partner": float(np.mean(partner)),
        "pseudo": float(np.mean(pseudo)),
        "clean_margin": float(np.mean(clean_margin)),
        "gene_sufficient_statistics": R3D.baseline_sufficient_statistics(
            {
                "partner": partner,
                "pseudo": pseudo,
                "clean_margin": clean_margin,
            },
            selected_ids,
        ),
    }
    row_sha256 = [
        {"example_id": row["example_id"], "sha256": f"{index:064x}"}
        for index, row in enumerate(selected_rows)
    ]
    construction_policy = {
        "readout": "mask exactly one G at next_acceptor_pos - 1",
        "paired_operator": "GT -> AC at donor_pos",
        "pseudo_operator": "GT -> AC at matched_decoy_donor",
        "pseudo_distance_caliper": R3D.DECOY_TOL,
        "donor_readout_constructed": False,
    }
    construction = {
        "schema": "gfmcd.r3c.acceptor_only_construction/1",
        "rows": n,
        "policy": construction_policy,
        "policy_sha256": R3D.canonical_sha256(construction_policy),
        "all_single_acceptor_G": True,
        "all_donor_targets_GT": True,
        "all_motif_clean": True,
        "ordered_rows_sha256": R3D.canonical_sha256(row_sha256),
        "row_sha256": row_sha256,
    }
    cohort = {
        "schema": "gfmcd.r3d.train_cohort_audit/1",
        "selection_reads_model_outputs": False,
        "split": "train",
        "preconstruction_allowed_chromosomes": list(R3D.TRAIN_CHROMOSOMES),
        "forbidden_chromosomes": sorted(R3D.FORBIDDEN_CHROMOSOMES),
        "provenance_hash_reads_full_fasta_bytes": True,
        "forbidden_chromosomes_materialized_as_examples": False,
        "forbidden_chromosomes_model_scored": False,
        "sealed_or_independent_claim": False,
        "selected_rows": n,
        "selected_genes": n,
        "one_row_per_gene": True,
        "registry_gene_intersection_count": 0,
        "target_uses_every_eligible_registry_disjoint_gene": True,
        "target_minimum_genes": R3D.TARGET_MIN_GENES,
        "row_rank_salt": R3D.ROW_RANK_SALT,
        "gene_rank_salt": R3D.GENE_RANK_SALT,
        "frozen_registry_capacity_replayed": True,
        "frozen_registry_capacity_sha256": R3D.canonical_sha256(capacity),
        "eligible_genes_one_row_each": capacity["eligible_genes_one_row_each"],
        "eligible_exposed_genes": capacity["exposed_development"][
            "genes_one_row_each"
        ],
        "eligible_registry_disjoint_genes": n,
        "selected_gene_ids": selected_ids,
        "selected_row_records": selected_rows,
        "ordered_rows_sha256": R3D.canonical_sha256(selected_rows),
        "ordered_gene_ids_sha256": R3D.canonical_sha256(selected_ids),
    }
    gates = R3D.build_gates(
        n_genes=n,
        stage="target",
        native={"partner": partner, "pseudo": pseudo},
        construction=construction,
        cohort_audit=cohort,
        measurements=measurements,
        calibration=r3d_preflight["family"]["production_preflight"],
        health=health,
        r3c_health_ok=r3c_health,
    )
    assert all(gates.values())
    official = R3D.official_paths_after_global_gates(replay_paths, gates)
    record = {
        "schema": R3D.RESULT_SCHEMA,
        "experiment": "R3d_acceptor_specificity_adjudication",
        "instrument_revision": R3D.INSTRUMENT_REVISION,
        "stage": "target",
        "status": "PROSPECTIVE_VERIFICATION",
        "tier": "PROSPECTIVE_ENDPOINT_SPECIFIC_TRAIN_VERIFICATION",
        "not_sealed_or_independent_confirmation": True,
        "target_bindable_preflight": True,
        "split": "train",
        "n": n,
        "n_genes": n,
        "primary_estimand": "raw masked-acceptor-G logit-margin component drop",
        "component_names": [
            R3D.component_name(component) for component in R3D.COMPONENTS
        ],
        "command": command,
        "hashes": hashes,
        "instrument_binding_payload": binding,
        "instrument_fingerprint_sha256": fingerprint,
        "git_authorization_identity": identity,
        "r3c_v2_1_preflight_package": r3c_package,
        "r3d_preflight_package": r3d_package,
        "input_snapshot": snapshot,
        "input_snapshot_sha256": R3D.canonical_sha256(snapshot),
        "registry": {
            "schema": registry["schema"],
            "scope_id": registry["scope_id"],
            "scope": registry["scope"],
            "global_gene_freshness_claimed": False,
            "gene_count": len(exposed),
            "gene_ids_sha256": registry["gene_ids_sha256"],
            "file_sha256": registry_package["artifact"]["sha256"],
            "package": registry_package,
        },
        "cohort_audit": cohort,
        "target_identity": target_identity,
        "target_attempt": attempt,
        "construction": construction,
        "baselines": baselines,
        "family": {
            "definition": "14 primitive partner/pseudo rows derive exactly 23 decision rows",
            "primitive_row_names": R3D.primitive_row_names(),
            "primitive_row_order_sha256": R3D.canonical_sha256(
                R3D.primitive_row_names()
            ),
            "primitive_gene_sufficient_statistics": R3D.r3c.gene_sufficient_statistics(
                primitive, selected_ids, R3D.primitive_row_names()
            ),
            "row_names": row_names,
            "row_order_sha256": R3D.canonical_sha256(row_names),
            "n_primitive_rows": 14,
            "n_family_rows": 23,
            "gene_sufficient_statistics": R3D.r3c.gene_sufficient_statistics(
                family, selected_ids, row_names
            ),
            "fixed_sequence_inference": replay,
            "production_preflight": r3d_preflight["family"][
                "production_preflight"
            ],
            "production_preflight_origin": "INHERITED_FROM_BOUND_R3D_PREFLIGHT",
        },
        "components": components,
        "three_way_monotone_health": health,
        "diagnostic_paths": replay_paths,
        "paths": official,
        "path_family": {
            "path_order": list(R3D.PATH_NAMES),
            "studywise_alpha": R3D.PATH_ALPHA,
            "strong_fwer_control": "fixed-sequence gatekeeping",
            "path_b_tested_only_after_path_a_supported": True,
        },
        "gates": gates,
        "all_gates_pass": True,
        "official_scientific_disposition_gates_pass": True,
        "components_are_diagnostic_when_global_gate_fails": False,
        "verdict": "PROSPECTIVE_TRAIN_ENDPOINT_VERIFICATION_COMPLETE",
        "licenses": R3D.scientific_license(
            stage="target", all_gates_pass=True, official_paths=official
        ),
    }
    context = {
        "registry": registry,
        "registry_package": registry_package,
        "fingerprint": fingerprint,
        "binding_payload": binding,
        "r3c_health_ok": r3c_health,
        "r3c_preflight_package": r3c_package,
        "r3d_preflight": r3d_preflight,
        "r3d_preflight_package": r3d_package,
    }
    return record, context, family, selected_ids, replay


def test_target_semantic_replay_restores_hash_ranked_gene_order_and_rejects_tamper(
    tmp_path, monkeypatch
):
    record, context, expected_family, selected_ids, replay = (
        _synthetic_target_for_semantic_replay(tmp_path)
    )
    monkeypatch.setattr(R3D, "validate_r3d_preflight", lambda *_args, **_kwargs: None)
    calls = []

    def replay_inference(values, groups, names, **kwargs):
        calls.append((np.asarray(values).copy(), list(groups), list(names), kwargs))
        return replay

    monkeypatch.setattr(R3D, "fixed_sequence_inference", replay_inference)
    R3D.validate_r3d_target(record, **context)
    np.testing.assert_array_equal(calls[0][0], expected_family)
    assert calls[0][1] == selected_ids
    assert calls[0][2] == R3D.family_row_names()

    relocated_context = deepcopy(context)
    for key in (
        "registry_package",
        "r3c_preflight_package",
        "r3d_preflight_package",
    ):
        for item in relocated_context[key].values():
            item["path"] = str(Path("/relocated-worktree") / Path(item["path"]).name)
    R3D.validate_r3d_target(record, **relocated_context)

    inconclusive = deepcopy(record)
    inconclusive["construction"]["all_motif_clean"] = False
    inconclusive["gates"]["zero_planted_donor_motifs"] = False
    inconclusive["all_gates_pass"] = False
    inconclusive["official_scientific_disposition_gates_pass"] = False
    inconclusive["components_are_diagnostic_when_global_gate_fails"] = True
    inconclusive["paths"] = R3D.official_paths_after_global_gates(
        inconclusive["diagnostic_paths"], inconclusive["gates"]
    )
    inconclusive["verdict"] = "INCONCLUSIVE"
    inconclusive["licenses"] = R3D.scientific_license(
        stage="target",
        all_gates_pass=False,
        official_paths=inconclusive["paths"],
    )
    R3D.validate_r3d_target(inconclusive, **context)
    assert all(
        path["classification"] == "INCONCLUSIVE_GLOBAL_GATE_FAILED"
        for path in inconclusive["paths"].values()
    )

    health_inconclusive = deepcopy(record)
    health_inconclusive["components"]["L3H1"]["modes"][R3D.MEAN_MODE][
        "health"
    ]["total"] = 0
    health_inconclusive["gates"][
        "bounded_identical_policy_health_executed"
    ] = False
    health_inconclusive["all_gates_pass"] = False
    health_inconclusive["official_scientific_disposition_gates_pass"] = False
    health_inconclusive["components_are_diagnostic_when_global_gate_fails"] = True
    health_inconclusive["paths"] = R3D.official_paths_after_global_gates(
        health_inconclusive["diagnostic_paths"], health_inconclusive["gates"]
    )
    health_inconclusive["verdict"] = "INCONCLUSIVE"
    health_inconclusive["licenses"] = R3D.scientific_license(
        stage="target",
        all_gates_pass=False,
        official_paths=health_inconclusive["paths"],
    )
    R3D.validate_r3d_target(health_inconclusive, **context)

    tampered = deepcopy(record)
    tampered["licenses"] = "unsupported positive claim"
    with pytest.raises(ValueError, match="license do not replay"):
        R3D.validate_r3d_target(tampered, **context)

    tampered_component = deepcopy(record)
    tampered_component["components"]["L3H1"]["modes"][R3D.MEAN_MODE][
        "endpoints"
    ]["partner"]["raw_mean"] += 1.0
    with pytest.raises(ValueError, match="component summary differs"):
        R3D.validate_r3d_target(tampered_component, **context)


def test_target_semantic_validation_precedes_result_publication():
    source = inspect.getsource(R3D.main)
    validation = source.index("validate_r3d_target(")
    assert validation < source.index("result_bytes =")
    assert validation < source.index("_atomic_write(output")


def test_every_preflight_semantic_replay_call_binds_the_predecessor_package():
    tree = ast.parse(SCRIPT.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_r3d_preflight"
    ]
    assert len(calls) == 3
    assert all(
        "r3c_preflight_package" in {keyword.arg for keyword in call.keywords}
        for call in calls
    )
    signature = inspect.signature(R3D.validate_r3d_preflight)
    signature.bind(
        {},
        registry={},
        registry_package={},
        fingerprint="0" * 64,
        binding_payload={},
        r3c_health_ok={},
        r3c_preflight_package={},
    )


def test_semantic_replay_order_matches_nonlexical_hash_ranked_observed_digests():
    genes = [f"g{index:02d}" for index in range(30)]
    ordered = sorted(
        genes,
        key=lambda gene: (R3D._hash_rank(R3D.GENE_RANK_SALT, gene), gene),
    )
    assert ordered != sorted(ordered)
    rng = np.random.default_rng(9127)
    primitive = np.repeat(R3D.coherent_path_offset()[:, None], len(ordered), axis=1)
    primitive += rng.normal(0.0, 0.01, size=primitive.shape)
    expected_family, names = R3D.derive_family_from_primitives(primitive)
    stats = R3D.r3c.gene_sufficient_statistics(
        primitive, ordered, R3D.primitive_row_names()
    )
    stored_values, stored_groups = R3D.r3c._reconstruct_cluster_array(stats)
    assert stored_groups == sorted(ordered)
    restored, restored_groups = R3D._reorder_one_row_per_gene_statistics(
        stored_values,
        stored_groups,
        ordered,
        context="test replay",
    )
    restored_family, restored_names = R3D.derive_family_from_primitives(restored)
    health = _health()
    expected = R3D.fixed_sequence_inference(
        expected_family,
        ordered,
        names,
        n_draws=99,
        seed=0,
        namespace="r3d:order-regression",
        path_health=health,
    )
    replayed = R3D.fixed_sequence_inference(
        restored_family,
        restored_groups,
        restored_names,
        n_draws=99,
        seed=0,
        namespace="r3d:order-regression",
        path_health=health,
    )
    sorted_family, sorted_names = R3D.derive_family_from_primitives(stored_values)
    wrong_order = R3D.fixed_sequence_inference(
        sorted_family,
        stored_groups,
        sorted_names,
        n_draws=99,
        seed=0,
        namespace="r3d:order-regression",
        path_health=health,
    )
    assert replayed == expected
    assert wrong_order != expected
    for validator in (R3D.validate_r3d_preflight, R3D.validate_r3d_target):
        source = inspect.getsource(validator)
        assert source.index("_reorder_one_row_per_gene_statistics(") < source.index(
            "fixed_sequence_inference("
        )
