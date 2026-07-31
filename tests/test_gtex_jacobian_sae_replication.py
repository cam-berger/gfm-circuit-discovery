import importlib.util
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/splicebert_gtex_finetune_shift/run_gtex_jacobian_sae_replication.py"
SPEC = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def test_initialization_seeds_are_unique_and_frozen():
    seeds = {
        RUNNER.initialization_seed(checkpoint, initialization)
        for checkpoint in RUNNER.EXPECTED_SEEDS
        for initialization in RUNNER.INITIALIZATIONS
    }
    assert len(seeds) == 15
    assert RUNNER.initialization_seed(2020, 0) == 22_200_000


def test_permuted_features_reciprocally_match():
    generator = np.random.default_rng(7)
    left = np.maximum(generator.normal(size=(500, 12)), 0)
    permutation = generator.permutation(12)
    right = left[:, permutation]
    left_decoder = generator.normal(size=(5, 12))
    right_decoder = left_decoder[:, permutation]
    match = RUNNER.feature_match(
        left,
        right,
        left_decoder,
        right_decoder,
    )
    assert match["PASS"]
    assert match["reciprocal_match_fraction"] == 1.0
    assert match["reciprocal_score_mean"] > 0.99


def test_representation_gate_requires_matched_graph_gain():
    metric = {
        "input_fve": 0.90,
        "output_write_fve": 0.98,
        "sandwich_output_fve": 0.95,
        "active_edge_mass_count": {"E90": {"median": 10.0}},
    }
    baseline = {
        **metric,
        "active_edge_mass_count": {"E90": {"median": 20.0}},
    }
    train = {
        "input_dead_feature_fraction": 0.2,
        "output_dead_feature_fraction": 0.3,
    }
    passed = RUNNER.representation_gate(
        train,
        metric,
        baseline,
        mechanical_pass=True,
    )
    assert passed["PASS"]
    failed = RUNNER.representation_gate(
        train,
        {**metric, "input_fve": 0.5},
        baseline,
        mechanical_pass=True,
    )
    assert not failed["PASS"]


def test_graph_accumulator_uses_float32_for_mps_compatibility():
    class FakeSAE:
        num_features = 3

        def encode(self, values, return_indices=False):
            acts = torch.ones((len(values), 3))
            indices = torch.tensor([[0, 1]]).repeat(len(values), 1)
            return (acts, indices) if return_indices else acts

    class FakeModel:
        input_center = torch.zeros(2)
        input_sae = FakeSAE()
        output_sae = FakeSAE()

        def eval(self):
            return self

        def normalize_input(self, values):
            return values

        def normalize_output(self, values):
            return values

        def active_jacobian(self, values, input_indices, output_indices):
            return torch.ones((len(values), 2, 2), dtype=torch.float32)

    _, _, graph = RUNNER.fingerprints_and_graph(
        FakeModel(),
        torch.zeros((4, 2)),
        torch.zeros((4, 2)),
        np.arange(4),
        batch_size=2,
    )
    assert graph.dtype == np.float32
