import ast
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from gfmcd.circuits.multiview_transcoder import (
    LABEL_FREE_SEQUENCE_PACKAGE_SCHEMA,
    StructuredMultiViewSparseTranscoder,
    assert_label_free_payload,
    label_free_package_binding,
    validate_label_free_state_rows,
)


REPO = Path(__file__).resolve().parents[1]
NAMESPACE = "gtex-native-biological-features-v1"


def load_script(name: str):
    path = REPO / "scripts/splicebert_gtex_finetune_shift" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


TRAINING = load_script("run_gtex_sparse_transcoder_training.py")
BEHAVIOR = load_script("run_gtex_sparse_transcoder_behavior_preflight.py")


def sequence_package():
    clean_sequence = "ACGTACGT"
    changed_sequence = "AGCTACGT"
    clean = {
        "state_id": "example-1:donor:clean",
        "example_id": "example-1",
        "gene_id": "gene-1",
        "endpoint": "donor",
        "variant_id": "clean",
        "variant_kind": "clean",
        "sequence": clean_sequence,
        "sequence_sha256": hashlib.sha256(clean_sequence.encode()).hexdigest(),
        "target_position": 3,
        "reference_state_id": None,
        "changed_positions": [],
        "hamming": 0,
    }
    changed = {
        **clean,
        "state_id": "example-1:donor:shuffle",
        "variant_id": "shuffle",
        "variant_kind": "target_shuffle",
        "sequence": changed_sequence,
        "sequence_sha256": hashlib.sha256(changed_sequence.encode()).hexdigest(),
        "reference_state_id": clean["state_id"],
        "changed_positions": [1, 2],
        "hamming": 2,
    }
    sources = {
        role: {
            "path": f"/frozen/{role}",
            "sha256": "a" * 64,
            "size_bytes": 1,
            "role": role,
        }
        for role in (
            "preflight",
            "discovery",
            "protocol",
            "addendum",
            "basic_gtf",
            "comprehensive_gtf",
            "fasta",
            "producer",
        )
    }
    authorization = {
        "status": "PRECONDITIONS_VERIFIED",
        "preflight_status": "PASS",
        "discovery_status": "COMPLETE",
        "producer": (
            "scripts/splicebert_gtex_finetune_shift/"
            "export_gtex_sparse_transcoder_sequence_package.py"
        ),
    }
    authorization["binding_sha256"] = label_free_package_binding(
        authorization,
        sources,
    )
    return {
        "schema": LABEL_FREE_SEQUENCE_PACKAGE_SCHEMA,
        "cohort_namespace": NAMESPACE,
        "partition": "discovery",
        "label_free": True,
        "sparse_training_authorized": True,
        "authorization": authorization,
        "sources": sources,
        "states": [clean, changed],
    }


def test_label_free_state_schema_accepts_sequences_and_rejects_outcomes():
    rows = validate_label_free_state_rows(
        sequence_package(),
        namespace=NAMESPACE,
    )
    assert [row["variant_kind"] for row in rows] == ["clean", "target_shuffle"]
    contaminated = sequence_package()
    contaminated["states"][1]["margin"] = 2.3
    with pytest.raises(ValueError, match="label-bearing"):
        validate_label_free_state_rows(contaminated, namespace=NAMESPACE)
    with pytest.raises(ValueError, match="label-bearing"):
        assert_label_free_payload({"nested": {"logits": [1.0, 2.0]}})


def test_label_free_state_schema_rejects_cross_example_or_false_edit_audits():
    crossed = sequence_package()
    crossed["states"][1]["gene_id"] = "other-gene"
    with pytest.raises(ValueError, match="crosses reference identity"):
        validate_label_free_state_rows(crossed, namespace=NAMESPACE)

    false_positions = sequence_package()
    false_positions["states"][1]["changed_positions"] = [1]
    false_positions["states"][1]["hamming"] = 1
    with pytest.raises(ValueError, match="changed-position audit"):
        validate_label_free_state_rows(false_positions, namespace=NAMESPACE)

    aliased_outcome = sequence_package()
    aliased_outcome["states"][1]["margin_damage"] = 1.0
    with pytest.raises(ValueError, match="non-allowlisted"):
        validate_label_free_state_rows(aliased_outcome, namespace=NAMESPACE)


def test_dictionary_gene_split_is_deterministic_and_disjoint():
    genes = [f"gene-{index}" for index in range(100)]
    first_train, first_validation = TRAINING.gene_train_validation_split(genes)
    second_train, second_validation = TRAINING.gene_train_validation_split(genes)
    assert np.array_equal(first_train, second_train)
    assert np.array_equal(first_validation, second_validation)
    assert not np.any(first_train & first_validation)
    assert np.all(first_train | first_validation)
    assert set(np.asarray(genes)[first_train]).isdisjoint(
        set(np.asarray(genes)[first_validation])
    )


def test_pair_construction_never_crosses_dictionary_split():
    states = []
    for gene_index in range(40):
        for kind in ("clean", "target_shuffle"):
            states.append(
                {
                    "state_id": f"g{gene_index}:{kind}",
                    "gene_id": f"g{gene_index}",
                    "endpoint": "donor",
                    "variant_kind": kind,
                    "reference_state_id": (
                        None if kind == "clean" else f"g{gene_index}:clean"
                    ),
                }
            )
    genes = [row["gene_id"] for row in states]
    train, validation = TRAINING.gene_train_validation_split(genes)
    train_pairs = TRAINING.pair_indices(states, train)
    validation_pairs = TRAINING.pair_indices(states, validation)
    assert all(train[left] and train[right] for left, right in train_pairs)
    assert all(
        validation[left] and validation[right]
        for left, right in validation_pairs
    )


def test_behavioral_pair_contract_and_exact_reconstruction_gate():
    package = sequence_package()
    acceptor_clean = {
        **package["states"][0],
        "state_id": "example-2:acceptor:clean",
        "example_id": "example-2",
        "gene_id": "gene-2",
        "endpoint": "acceptor",
    }
    acceptor_changed = {
        **acceptor_clean,
        "state_id": "example-2:acceptor:shuffle",
        "variant_kind": "target_shuffle",
        "reference_state_id": acceptor_clean["state_id"],
    }
    donor_clean_2 = {
        **package["states"][0],
        "state_id": "example-3:donor:clean",
        "example_id": "example-3",
        "gene_id": "gene-3",
    }
    donor_changed_2 = {
        **package["states"][1],
        "state_id": "example-3:donor:shuffle",
        "example_id": "example-3",
        "gene_id": "gene-3",
        "reference_state_id": donor_clean_2["state_id"],
    }
    acceptor_clean_2 = {
        **acceptor_clean,
        "state_id": "example-4:acceptor:clean",
        "example_id": "example-4",
        "gene_id": "gene-4",
    }
    acceptor_changed_2 = {
        **acceptor_changed,
        "state_id": "example-4:acceptor:shuffle",
        "example_id": "example-4",
        "gene_id": "gene-4",
        "reference_state_id": acceptor_clean_2["state_id"],
    }
    states = [
        *package["states"],
        acceptor_clean,
        acceptor_changed,
        donor_clean_2,
        donor_changed_2,
        acceptor_clean_2,
        acceptor_changed_2,
    ]
    pairs = BEHAVIOR.target_shuffle_pairs(states)
    assert pairs["donor"].tolist() == [[0, 1], [4, 5]]
    assert pairs["acceptor"].tolist() == [[2, 3], [6, 7]]

    width = 4
    residual = np.zeros((2, width))
    write = np.asarray([[2.0, -2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    parameters = {
        "layer_norm_weight": np.ones(width),
        "layer_norm_bias": np.zeros(width),
        "layer_norm_eps": 1e-12,
        "classifier_direction": np.asarray([1.0, -1.0, 0.0, 0.0]),
        "classifier_bias": 0.0,
    }
    metrics = BEHAVIOR.endpoint_metrics(
        residual,
        write,
        write.copy(),
        np.asarray([[0, 1]]),
        parameters,
    )
    assert metrics["margin_mae"] == pytest.approx(0.0)
    assert metrics["effect_recovery_ratio"] == pytest.approx(1.0)
    assert metrics["PASS"]


def test_planted_sparse_feature_preflight_recovers_detector_and_writer():
    result = TRAINING.planted_feature_preflight(torch.device("cpu"))
    assert result["PASS"]
    assert result["minimum_activation_correlation"] >= 0.80
    assert result["minimum_writer_cosine"] >= 0.70


def test_stability_census_covers_every_tuned_view_and_fit_pair():
    models = [
        StructuredMultiViewSparseTranscoder(TRAINING.model_config(128, 8))
        for _ in TRAINING.INITIALIZATIONS
    ]
    generator = np.random.default_rng(17)
    feature_sets = [
        {
            view: generator.random((12, 128), dtype=np.float32)
            for view in TRAINING.VIEWS
        }
        for _ in TRAINING.INITIALIZATIONS
    ]
    result = TRAINING.stability_score(models, feature_sets)
    observed = {
        (row["view"], tuple(row["initialization_pair"]))
        for row in result["all_initialization_pairs"]
    }
    expected = {
        (view, pair)
        for view in TRAINING.TUNED_VIEWS
        for pair in ((0, 1), (0, 2), (1, 2))
    }
    assert observed == expected


def test_label_free_trainer_has_no_model_or_classifier_import():
    path = (
        REPO
        / "scripts/splicebert_gtex_finetune_shift/run_gtex_sparse_transcoder_training.py"
    )
    tree = ast.parse(path.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name.startswith("transformers") for name in imported)
    assert "gfmcd.splicing.gtex_finetune" not in imported
    assert "gfmcd.splicing.mlp_units" not in imported


def test_sparse_model_payload_is_compatible_with_weights_only_load(tmp_path):
    config = TRAINING.model_config(128, 8)
    model = StructuredMultiViewSparseTranscoder(config)
    path = tmp_path / "dictionary.pt"
    torch.save(
        {
            "schema_version": 1,
            "views": list(TRAINING.VIEWS),
            "tuned_views": list(TRAINING.TUNED_VIEWS),
            "normalization": {
                view: {
                    "mean": torch.zeros(512),
                    "scale": torch.ones(512),
                }
                for view in TRAINING.VIEWS
            },
            "models": {
                "w128_k8_r0": {
                    "config": TRAINING.asdict(config),
                    "state_dict": TRAINING.clone_cpu_state(model),
                }
            },
        },
        path,
    )
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    restored_config = BEHAVIOR.config_from_dict(
        loaded["models"]["w128_k8_r0"]["config"]
    )
    restored = StructuredMultiViewSparseTranscoder(restored_config)
    restored.load_state_dict(
        loaded["models"]["w128_k8_r0"]["state_dict"],
        strict=True,
    )
