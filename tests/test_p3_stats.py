from __future__ import annotations

import numpy as np
import pytest

from gfmcd.splicing.p3_stats import (
    bootstrap_studentized_max_t,
    gene_paired_phase_table,
    interval_classification,
    mde80,
)


def _rows(gene, context, bump, metric="coupling"):
    return [
        {
            "gene_id": gene,
            "chrom": "chr1",
            "context": context,
            "amount": amount,
            metric: amount + (bump if amount in (3, 6, 9) else 0),
        }
        for amount in range(2, 11)
    ]


def test_gene_table_applies_exact_paired_phase_contrast():
    rows = (
        _rows("g1", "cds", 0.7) + _rows("g1", "upstream_intron", 0.2)
        + _rows("g2", "cds", 0.4) + _rows("g2", "upstream_intron", 0.1)
    )
    table = gene_paired_phase_table(rows, metrics=("coupling",))
    assert [row["psi_coupling"] for row in table] == pytest.approx([0.5, 0.3])


def test_gene_table_supports_one_cycle_specificity_family():
    rows = [
        {"gene_id": "g", "chrom": "chr1", "context": context, "amount": amount,
         "coupling": (1.0 if amount == 3 else 0.0) * scale}
        for context, scale in (("cds", 0.8), ("upstream_intron", 0.2))
        for amount in (2, 3, 4)
    ]
    table = gene_paired_phase_table(rows, metrics=("coupling",), centers=(3,))
    assert table[0]["psi_coupling"] == pytest.approx(0.6)


def test_bootstrap_max_t_is_deterministic_and_simultaneous():
    matrix = np.array([[0.1 * i, -0.05 * i + (i % 2) * 0.01] for i in range(1, 21)])
    first = bootstrap_studentized_max_t(matrix, names=("a", "b"), seed=7, n_bootstrap=499)
    second = bootstrap_studentized_max_t(matrix, names=("a", "b"), seed=7, n_bootstrap=499)
    assert first == second
    assert first.critical_value > 1.64
    assert all(lo < mean < hi for lo, mean, hi in zip(first.lower, first.means, first.upper))
    assert mde80(matrix[:, 0], critical_value=first.critical_value) > 0


def test_interval_classifier_uses_whole_interval():
    assert interval_classification(0.06, 0.09, sesoi=0.05) == "material"
    assert interval_classification(0.01, 0.09, sesoi=0.05).startswith("detectable")
    assert interval_classification(-0.03, 0.04, sesoi=0.05) == "bounded_null"
    assert interval_classification(-0.02, 0.08, sesoi=0.05) == "inconclusive"
