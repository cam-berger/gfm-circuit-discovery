"""Falsifiable memorials on actual tokenization, logits, corruption and resampling paths."""
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from gfmcd.splicing.register_operator_construction import Request, edit_acceptor, exact_stop_omega
from gfmcd.splicing.register_operator_measurement import score_requests, nucleotide_margins, token_digest
from gfmcd.splicing.register_operator_statistics import make_inference_context, balanced_assignments
from gfmcd.splicing.register_operator_io import guard_paths, write_artifact, verify_artifact_manifest, require_manifest_input


class FakeAdapter:
    def __init__(self, offset=1):
        self.offset = offset
        self.ids = {"A": 6, "C": 7, "G": 8, "T": 9}
        self.tokenizer = SimpleNamespace(mask_token_id=4, convert_tokens_to_ids=self.ids.__getitem__)

    def nucleotide_to_token(self, pos):
        return pos + self.offset

    def tokenize(self, sequences):
        ids = torch.tensor([[2] + [self.ids[b] for b in sequence] + [3] for sequence in sequences])
        return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))

    def model(self, *, input_ids, attention_mask, return_dict):
        # The special N token wins the full vocabulary; max margin must ignore it.
        logits = torch.tensor([0., 0., 0., 0., 0., 100., 0., 1., 4., 3.])
        return SimpleNamespace(logits=logits.broadcast_to((*input_ids.shape, 10)))


def test_actual_mask_digest_margin_competitors_and_full_vocabulary_health():
    request = Request("real-path", "G" * 1022, 392, "G")
    scores = score_requests(FakeAdapter(), [request])
    row = scores[0]
    expected = [2] + [8] * 1022 + [3]
    expected[393] = 4
    assert row["input_ids_sha256"] == token_digest(expected)
    assert row["max_margin"] == 1.0
    assert row["linear_margin"] == float(torch.tensor(4.) - (torch.tensor([0., 1., 4., 3.]).sum() - 4.) / 3.)
    assert row["true_base_log_probability"] < -90
    assert row["correct"] is False
    assert row["attention_mask_sha256"] == token_digest([1] * 1024)
    assert row["base_logits"] == [0., 1., 4., 3.]


def test_plus_one_mask_fault_and_wrong_true_base_fail_on_path():
    with pytest.raises(ValueError, match="mapping"):
        score_requests(FakeAdapter(offset=2), [Request("bad", "G" * 1022, 392, "G")])
    with pytest.raises(ValueError, match="request"):
        Request("bad-base", "G" * 1022, 392, "A")


def test_all_flanks_cc_gate_and_no_hidden_unmasked_edit():
    for left in "ACGT":
        for right in "ACGT":
            sequence = "T" * 20 + left + "AG" + right + "A" * 20
            if left == "G":
                with pytest.raises(ValueError, match="plants"):
                    edit_acceptor(sequence, 21, "CC")
            else:
                edited = edit_acceptor(sequence, 21, "CC")
                assert edited[21:23] == "CC"
                assert edited[:21] == sequence[:21] and edited[23:] == sequence[23:]


def test_linear_order_and_max_readout_use_same_actual_float32_logits():
    x = np.array([[1000.25, -4.9, .01, -300.123], [0., -5., 3., -1.]])
    metrics = nucleotide_margins(x)
    tx = torch.tensor(x, dtype=torch.float32)
    expected = tx[:, 2] - (tx.sum(dim=1) - tx[:, 2]) / 3
    np.testing.assert_array_equal(metrics["linear"], expected.numpy())
    assert metrics["max"][1] == 3
    assert metrics["linear"][1] == 5
    assert not np.array_equal(metrics["max"], metrics["linear"])


def test_cached_counts_match_frozen_stratified_indexed_means():
    ids = [f"b{i}" for i in range(12)]
    strata = ["chr1:+"] * 4 + ["chr2:-"] * 8
    context = make_inference_context(ids, strata, draws=64, pseudo_seeds=[2026080401])
    x = np.arange(48).reshape(12, 4) ** 2 / 17
    np.testing.assert_allclose(context.weights @ x, x[context.indices].mean(axis=1), rtol=0, atol=1e-12)
    assert context.pseudo_seeds == (2026080401,)
    for indices in context.indices:
        assert sum(i < 4 for i in indices) == 4
    assignment = balanced_assignments(ids, 2026080401)
    # A nonsymmetric matrix distinguishes original-label mapping from inverse mapping.
    matrix = np.arange(9).reshape(3, 3) ** 2
    for i, permutation in enumerate(assignment):
        diagonal = [matrix[p, q] for p in range(3) for q in range(3) if permutation[q] == p]
        off = [matrix[p, q] for p in range(3) for q in range(3) if permutation[q] != p]
        assert np.sum(context.coefficients[0, i] * matrix) == pytest.approx(np.mean(diagonal) - np.mean(off))


def test_immutable_outputs_and_manifest_binds_current_bytes(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("frozen source")
    out = tmp_path / "result.json"
    write_artifact(out, {"status": "PASS"}, inputs=[source], experiment="test")
    verify_artifact_manifest(out)
    with pytest.raises(FileExistsError):
        write_artifact(out, {}, inputs=[source], experiment="test")
    source.write_text("changed")
    with pytest.raises(ValueError, match="input has changed"):
        verify_artifact_manifest(out)
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "not-yet-created")
    with pytest.raises(FileExistsError):
        guard_paths(dangling)


def test_passing_audit_for_different_input_cannot_authorize_this_package(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("same chosen IDs, different package")
    second.write_text("the actual package")
    out = tmp_path / "audit.json"
    write_artifact(out, {"status": "PASS"}, inputs=[first], experiment="test")
    manifest = verify_artifact_manifest(out)
    require_manifest_input(manifest, first)
    with pytest.raises(ValueError, match="does not bind"):
        require_manifest_input(manifest, second)


def test_stop_diagnostic_canonical_integer_contrast_preserves_ties():
    low_diagonal = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 0]])
    high_diagonal = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert exact_stop_omega(low_diagonal) == exact_stop_omega(high_diagonal) == 2 / 3
    assert not (exact_stop_omega(high_diagonal) > 2 / 3)
    with pytest.raises(ValueError, match="integer"):
        exact_stop_omega(np.ones((3, 3)) * .5)
