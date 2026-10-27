"""gfmcd.readout: parity with every pinned margin copy, and the measurement seam end to end.

A stub adapter (character tokenizer with a CLS prefix, and a tiny model whose masked-position
logits depend on the whole unpadded sequence) stands in for SpliceBERT, so CI exercises masking,
offsets, corruption, batching and interventions without weights. A real-model parity test runs
where the checkpoint exists.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gfmcd.readout import (Batch, ClassifierMargin, MaskedBaseMargin, Paired, frozen_cc, measure,
                           recovery, recovery_interval)
from gfmcd.splicing import boundary_grammar, circuit, coupling, register_operator_measurement, unembed

REPO = Path(__file__).resolve().parents[1]
BASE_IDS = {"A": 6, "C": 7, "G": 8, "T": 9}
MODEL = REPO / "data/models/splicebert_official_archive/extracted/models/SpliceBERT.1024nt"


# -- stub adapter ---------------------------------------------------------------------------------

class _Encoding(dict):
    def to(self, device):
        return self


class StubTokenizer:
    mask_token_id, pad_token_id, cls_token_id = 1, 0, 2

    def convert_tokens_to_ids(self, token: str) -> int:
        return BASE_IDS[token]

    def __call__(self, texts, return_tensors="pt", padding=True):
        rows = [[self.cls_token_id] + [BASE_IDS[c] for c in t.split(" ")] for t in texts]
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        for i, row in enumerate(rows):
            ids[i, :len(row)] = torch.tensor(row)
        return _Encoding(input_ids=ids, attention_mask=(ids != self.pad_token_id).long())


class StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.local = torch.nn.Parameter(torch.randn(10, 10, generator=g) * 3)
        self.context = torch.nn.Parameter(torch.randn(10, 10, generator=g) * 3)

    def forward(self, input_ids, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        context = (self.context[input_ids] * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
        return SimpleNamespace(logits=self.local[input_ids] + context)


@pytest.fixture()
def adapter():
    return SimpleNamespace(model=StubModel().eval(), tokenizer=StubTokenizer(), device="cpu",
                           prefix_tokens=1)


def _examples(n: int, length: int = 60, seed: int = 0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        seq = list(rng.choice("ACGT") for _ in range(length))
        seq[10:12], seq[38:40] = "GT", "AG"
        out.append(SimpleNamespace(sequence="".join(seq), donor_pos=10, next_acceptor_pos=40,
                                   gene_id=f"G{i % 4}"))
    return out


# -- golden logits: parity with every pinned copy ---------------------------------------------------

def _golden_rows() -> np.ndarray:
    rng = np.random.default_rng(5)
    rows = rng.normal(0, 4, size=(64, 10)).astype(np.float32)
    rows[0, [6, 7, 8, 9]] = [1.0, 1.0, 1.0, 1.0]            # four-way tie
    rows[1, [6, 7, 8, 9]] = [2.0, -3.0, 2.0, 0.5]           # G ties the max other
    rows[2, [6, 7, 8, 9]] = [20.0, 19.999998, 20.000002, -20.0]  # float32 near-tie
    rows[3, [6, 7, 8, 9]] = [-1e3, 5e2, 7e2, 1e-3]           # wide range
    return rows


def test_masked_margin_is_exact_against_the_float64_copies() -> None:
    rows = _golden_rows()
    linear, maximum = boundary_grammar.margins_from_logits(rows, np.zeros(len(rows), dtype=int),
                                                           base_token_ids=BASE_IDS)
    np.testing.assert_array_equal(MaskedBaseMargin(reduce="linear").from_logits(rows, BASE_IDS), linear)
    np.testing.assert_array_equal(MaskedBaseMargin(reduce="max").from_logits(rows, BASE_IDS), maximum)
    hard = unembed.margin_hard(torch.from_numpy(rows.astype(np.float64)), BASE_IDS).numpy()
    np.testing.assert_array_equal(MaskedBaseMargin().from_logits(rows, BASE_IDS), hard)


def test_masked_margin_agrees_with_the_float32_copies_to_float32_resolution() -> None:
    rows = _golden_rows()
    ours_max = MaskedBaseMargin().from_logits(rows, BASE_IDS)
    ours_lin = MaskedBaseMargin(reduce="linear").from_logits(rows, BASE_IDS)
    task = SimpleNamespace(off=0, bid=BASE_IDS)
    pinned_max = circuit.CouplingTask.margin(task, torch.from_numpy(rows)[:, None, :], [0] * len(rows))
    four = rows[:, [6, 7, 8, 9]]
    register = register_operator_measurement.nucleotide_margins(four, target=2)
    tolerance = dict(rtol=2e-6, atol=2e-6 * float(np.abs(rows).max()))
    np.testing.assert_allclose(ours_max, pinned_max.numpy(), **tolerance)
    np.testing.assert_allclose(ours_max, register["max"], **tolerance)
    np.testing.assert_allclose(ours_lin, register["linear"], **tolerance)


def test_other_bases_and_classifier_margin() -> None:
    rows = _golden_rows().astype(np.float64)
    a = MaskedBaseMargin(base="A").from_logits(rows, BASE_IDS)
    np.testing.assert_array_equal(a, rows[:, 6] - np.max(rows[:, [7, 8, 9]], axis=1))
    np.testing.assert_array_equal(ClassifierMargin(label=3).from_logits(rows), rows[:, 3] - rows[:, 0])
    with pytest.raises(ValueError):
        MaskedBaseMargin(base="N")


# -- the measurement seam -------------------------------------------------------------------------

def test_measure_matches_the_pinned_scorer_exactly_and_coupling_task_to_float32(adapter) -> None:
    examples = _examples(11)
    paired = measure(adapter, examples, MaskedBaseMargin(), batch_size=4)
    linear = measure(adapter, examples, MaskedBaseMargin(reduce="linear"), batch_size=4)
    scorer = coupling.CouplingScorer(adapter.model, adapter.tokenizer, batch_size=16)
    seqs, donors = [x.sequence for x in examples], [x.donor_pos for x in examples]
    np.testing.assert_array_equal(paired.clean, scorer._margin(seqs, donors, "G"))
    np.testing.assert_array_equal(linear.clean, scorer._margin(seqs, donors, "G", reduce="mean"))
    np.testing.assert_array_equal(paired.corrupt, scorer._margin([frozen_cc(x) for x in examples], donors, "G"))
    task = circuit.CouplingTask(adapter, batch=4)
    np.testing.assert_allclose(paired.coupling, task.coupling(examples), rtol=0, atol=1e-5)
    assert list(paired.groups) == [x.gene_id for x in examples]


def test_intervention_sees_each_arm_masked_and_corrupted(adapter) -> None:
    examples = _examples(5)
    seen: list[Batch] = []

    @contextmanager
    def record(batch: Batch):
        seen.append(batch)
        yield

    measure(adapter, examples, MaskedBaseMargin(), intervention=record, batch_size=2)
    assert [b.arm for b in seen] == ["clean", "corrupt"] * 3
    assert [len(b.examples) for b in seen] == [2, 2, 2, 2, 1, 1]
    for batch in seen:
        assert batch.positions == tuple(x.donor_pos + 1 for x in batch.examples)
        ids = batch.encoding["input_ids"]
        assert all(ids[i, p] == StubTokenizer.mask_token_id for i, p in enumerate(batch.positions))
        acceptor = ids[:, 1 + 38:1 + 40].tolist()
        expected = [BASE_IDS["C"]] * 2 if batch.arm == "corrupt" else [BASE_IDS["A"], BASE_IDS["G"]]
        assert all(row == expected for row in acceptor)


def test_intervention_changes_only_what_it_patches(adapter) -> None:
    examples = _examples(6)

    @contextmanager
    def zero_context_on_corrupt(batch: Batch):
        saved = adapter.model.context.data.clone()
        if batch.arm == "corrupt":
            adapter.model.context.data.zero_()
        try:
            yield
        finally:
            adapter.model.context.data.copy_(saved)

    base = measure(adapter, examples, MaskedBaseMargin())
    patched = measure(adapter, examples, MaskedBaseMargin(), intervention=zero_context_on_corrupt)
    np.testing.assert_array_equal(patched.clean, base.clean)
    assert not np.array_equal(patched.corrupt, base.corrupt)


def test_measure_refuses_a_classifier_readout(adapter) -> None:
    with pytest.raises(TypeError, match="logits-only"):
        measure(adapter, _examples(2), ClassifierMargin(label=1))


# -- recovery --------------------------------------------------------------------------------------

def test_recovery_is_a_ratio_of_cohort_means_with_a_grouped_interval() -> None:
    rng = np.random.default_rng(2)
    clean = rng.normal(2.0, 0.3, 80)
    corrupt = clean - rng.normal(0.9, 0.2, 80)
    baseline = Paired(clean, corrupt, np.repeat([f"g{i}" for i in range(20)], 4).astype(object))
    intervened = corrupt + 0.5 * (clean - corrupt) + rng.normal(0, 0.01, 80)
    r = recovery(baseline, intervened)
    assert r == pytest.approx(np.mean(intervened - corrupt) / np.mean(clean - corrupt))
    interval = recovery_interval(baseline, intervened, n_boot=500, seed=0)
    assert interval.low <= r <= interval.high and interval.n_groups == 20
    assert interval.estimate == pytest.approx(r)


def test_recovery_interval_refuses_a_denominator_that_changes_sign() -> None:
    rng = np.random.default_rng(3)
    clean = rng.normal(0, 1, 60)
    baseline = Paired(clean, clean - rng.normal(0.02, 1.0, 60), np.repeat(np.arange(15), 4).astype(object))
    with pytest.raises(ValueError, match="change sign"):
        recovery_interval(baseline, baseline.corrupt + 0.1, n_boot=500, seed=0)


# -- real model, where the checkpoint exists --------------------------------------------------------

@pytest.mark.skipif(not (MODEL / "pytorch_model.bin").exists(), reason="SpliceBERT weights not present")
def test_measure_matches_coupling_task_on_the_real_model() -> None:
    adapter = circuit.load_adapter(str(MODEL), "cpu")
    examples = _examples(8, length=200, seed=11)
    ours = measure(adapter, examples, MaskedBaseMargin(), batch_size=4)
    pinned = circuit.CouplingTask(adapter, batch=4).coupling(examples)
    np.testing.assert_allclose(ours.coupling, pinned, rtol=0, atol=1e-4)
