import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO
    / "scripts/splicebert_gtex_finetune_shift/run_gtex_jacobian_sae_instrument.py"
)
SPEC = importlib.util.spec_from_file_location(SCRIPT.stem, SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


def test_gene_split_is_deterministic_disjoint_and_matches_frozen_namespace():
    genes = [f"gene-{index // 3}" for index in range(300)]
    train_a, validation_a = RUNNER.gene_train_validation_split(genes)
    train_b, validation_b = RUNNER.gene_train_validation_split(genes)
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(validation_a, validation_b)
    assert np.all(train_a ^ validation_a)
    assert set(np.asarray(genes)[train_a]).isdisjoint(
        set(np.asarray(genes)[validation_a])
    )


def test_gene_balanced_epoch_equalizes_expected_gene_weight():
    genes = ["large"] * 20 + ["small"] * 2 + ["medium"] * 8
    indices = np.arange(len(genes))
    selected = RUNNER.gene_balanced_epoch(indices, genes, seed=1)
    observed = [genes[index] for index in selected]
    counts = {gene: observed.count(gene) for gene in set(genes)}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_candidate_selection_requires_fidelity_and_graph_improvement():
    def row(e90, input_fve=0.90, output_fve=0.97, sandwich_fve=0.94):
        return {
            "train": {
                "input_dead_feature_fraction": 0.1,
                "output_dead_feature_fraction": 0.1,
            },
            "validation": {
                "input_fve": input_fve,
                "output_write_fve": output_fve,
                "sandwich_output_fve": sandwich_fve,
                "active_edge_mass_count": {"E90": {"median": e90}},
            }
        }

    candidates = {
        "0": row(40),
        "0.015625": row(37),
        "0.0625": row(32),
        "0.25": row(30, output_fve=0.80),
    }
    selected = RUNNER.select_candidate(candidates)
    assert selected["PASS"]
    assert selected["selected_coefficient"] == 0.0625
    assert not candidates["0.25"]["representation_PASS"]
