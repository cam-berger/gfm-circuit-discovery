"""Memorial test for RRR finding 4: top-1 accuracy == positive-margin rate.

The pair score is the true-vs-best-decoy margin, so (margin > 0) <=> (true pair is
rank 1). `top1_accuracy` and `positive_margin_rate` are therefore the SAME statistic,
not two independent metrics. This test both proves the identity from the metric code
and guards the summary artifact, so a future reader is never misled into reading them
as corroborating evidence.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from gfmcd.splicing.metrics import evaluate_pair_logits
from gfmcd.splicing.readout import PairIndexBatch

REPO = Path(__file__).resolve().parents[1]


def test_identity_from_metric_code():
    # 2 examples x 3 candidate pairs; both true labels = index 0.
    # ex0: true is argmax (margin +1); ex1: true is not argmax (margin -4).
    # -> top1 = 0.5 and PMR = 0.5, and they must be EQUAL by construction.
    logits = torch.tensor([[2.0, 1.0, 0.5], [-1.0, 0.0, 3.0]])
    pairs = PairIndexBatch(
        positions=torch.zeros(2, 3, 2, dtype=torch.long),
        lengths=torch.zeros(2, 3, dtype=torch.long),
        mask=torch.ones(2, 3, dtype=torch.bool),
        labels=torch.tensor([0, 0]),
    )
    m = evaluate_pair_logits(logits, pairs)
    assert m.top1_accuracy == m.positive_margin_rate == 0.5


@pytest.mark.skipif(not (REPO / "colab_results/summary.csv").exists(),
                    reason="matrix summary artifact not present")
def test_identity_holds_in_summary_artifact():
    # top1 (argmax, tie-broken) and PMR (strict margin>0) coincide EXCEPT on exact
    # rank-1 ties, which are measure-zero for continuous logits and rare even for the
    # degenerate random-frozen control (~1-2 tied examples / 4000). So: equal within a
    # small tie tolerance, never two independent metrics.
    rows = [r for r in csv.DictReader(open(REPO / "colab_results/summary.csv"))
            if r.get("status") == "ok"]
    assert rows, "no complete rows"
    for r in rows:
        assert abs(float(r["val_top1"]) - float(r["val_pmr"])) < 5e-3, (
            f"{r['regime']} seed{r['seed']}: top1 and PMR differ beyond tie tolerance -- "
            "the metric changed; they are no longer the same statistic")
