from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from gfmcd.splicing.pbc_register_runner import (
    RegisterCell,
    carrier_edit_delta,
    embedding_delta_context,
    evaluate_register_lattice,
)


class _Batch:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids
        self.attention_mask = torch.ones_like(input_ids)


class _Tokenizer:
    mask_token_id = 5
    _ids = {"A": 1, "C": 2, "G": 3, "T": 4}

    def convert_tokens_to_ids(self, base: str) -> int:
        return self._ids[base]


class _FakeModel(torch.nn.Module):
    def __init__(self, positions: int = 18, hidden: int = 4):
        super().__init__()
        self.bert = SimpleNamespace(
            embeddings=SimpleNamespace(
                position_embeddings=torch.nn.Embedding(positions, hidden)
            )
        )
        with torch.no_grad():
            position = torch.arange(positions, dtype=torch.float32)
            table = torch.stack(
                (
                    torch.cos(2 * torch.pi * position / 3),
                    torch.sin(2 * torch.pi * position / 3),
                    position / positions,
                    torch.ones_like(position),
                ),
                dim=1,
            )
            self.bert.embeddings.position_embeddings.weight.copy_(table)

    def forward(self, *, input_ids, attention_mask, return_dict=True):
        del attention_mask, return_dict
        length = input_ids.shape[1]
        position_ids = torch.arange(length, device=input_ids.device).unsqueeze(0)
        positions = self.bert.embeddings.position_embeddings(position_ids)
        if positions.shape[0] == 1 and input_ids.shape[0] > 1:
            positions = positions.expand(input_ids.shape[0], -1, -1)
        # The synthetic G logit depends jointly on the position carrier at the masked
        # donor and nucleotide content elsewhere; this gives the paired corruption a
        # deterministic, intervention-sensitive coupling value.
        content = input_ids.float().sum(dim=1, keepdim=True)
        signal = positions[..., 0] * content
        logits = torch.zeros((*input_ids.shape, 8), dtype=torch.float32)
        logits[..., 3] = signal
        return SimpleNamespace(logits=logits)


class _Adapter:
    prefix_tokens = 1

    def __init__(self):
        self.model = _FakeModel()
        self.tokenizer = _Tokenizer()

    def nucleotide_to_token(self, position: int) -> int:
        return position + 1

    def tokenize(self, sequences):
        rows = []
        for sequence in sequences:
            rows.append([0, *[self.tokenizer._ids[base] for base in sequence], 0])
        return _Batch(torch.tensor(rows, dtype=torch.long))


def _cells() -> list[RegisterCell]:
    cells = []
    for donor_phase in range(3):
        for acceptor_phase in range(3):
            intact = "ACGTGTACAGACGTAC"
            corrupt = intact[:8] + "AC" + intact[10:]
            cells.append(
                RegisterCell(
                    block_id="b0",
                    chrom="chr1",
                    strand="+",
                    donor_phase=donor_phase,
                    acceptor_phase=acceptor_phase,
                    intact_sequence=intact,
                    acceptor_ac_sequence=corrupt,
                    donor_position=4,
                    acceptor_position=10,
                )
            )
    return cells


def test_register_cell_rejects_any_nonoperator_sequence_change():
    with pytest.raises(ValueError, match="differ only"):
        RegisterCell(
            block_id="b",
            chrom="chr1",
            strand="+",
            donor_phase=0,
            acceptor_phase=0,
            intact_sequence="ACGTGTACAGACGTAC",
            acceptor_ac_sequence="TCGTGTACACACGTAC",
            donor_position=4,
            acceptor_position=10,
        )


def test_combined_delta_is_additive_and_register_zero_is_replay():
    position = torch.arange(15, dtype=torch.float32)
    p3 = torch.stack(
        (
            torch.cos(2 * torch.pi * position / 3),
            torch.sin(2 * torch.pi * position / 3),
            torch.cos(2 * torch.pi * position / 3),
            torch.sin(2 * torch.pi * position / 3),
        ),
        dim=1,
    )
    p10 = torch.ones_like(p3)
    replay = carrier_edit_delta(p3, p3_bins=(5,), register=0, start=3, end=10)
    assert torch.equal(replay, torch.zeros_like(p3))
    combined = carrier_edit_delta(
        p3,
        p3_bins=(5,),
        register=1,
        start=3,
        end=10,
        p10_component=p10,
        p10_dose=0.5,
    )
    assert torch.allclose(combined[:3], -0.5 * p10[:3])
    # The donor endpoint is fixed while the acceptor endpoint receives one full
    # period-3 register shift; intervening rows follow the analytic phase ramp.
    assert torch.allclose(combined[3], -0.5 * p10[3], atol=1e-6)
    assert torch.allclose(combined[9], p3[8] - p3[9] - 0.5, atol=1e-5)
    assert not torch.allclose(combined[4], combined[3])


def test_embedding_context_is_batch_specific_and_restores_hook():
    embedding = torch.nn.Embedding(8, 2)
    ids = torch.arange(5).unsqueeze(0)
    native = embedding(ids)
    delta_a = torch.zeros_like(embedding.weight)
    delta_b = torch.zeros_like(embedding.weight)
    delta_a[2] = 1
    delta_b[3] = 2
    with embedding_delta_context(embedding, (delta_a, delta_b)):
        edited = embedding(ids)
    assert edited.shape == (2, 5, 2)
    assert torch.allclose(edited[0, 2], native[0, 2] + 1)
    assert torch.allclose(edited[1, 3], native[0, 3] + 2)
    assert torch.equal(embedding(ids), native)


def test_through_model_lattice_emits_complete_linear_c43_rows():
    adapter = _Adapter()
    rows, mechanics = evaluate_register_lattice(
        adapter,
        _cells(),
        p3_bins=(6,),
        span_start_token=5,
        span_end_token=11,
        decoy_span_token=(11, 17),
        control_seeds=(7, 11),
        batch_size=4,
    )
    assert len(rows) == 9 * 3 * 5
    assert {row["component"] for row in rows} == {
        "p3",
        "p3_signed_00",
        "p3_signed_01",
        "p3_common_mode",
        "p3_decoy_span",
    }
    assert {row["register"] for row in rows} == {-1, 0, 1}
    assert {row["p10_state"] for row in rows} == {"intact"}
    assert all(isinstance(row["coupling"], float) for row in rows)
    assert mechanics["span_token"] == [5, 11]
    assert mechanics["decoy_span_token"] == [11, 17]
    assert mechanics["control_seeds"] == [7, 11]
    assert mechanics["register_zero_no_hook_max_abs"] == 0.0
    assert len(mechanics["no_hook_rows"]) == 9
    for label in ("p3_signed_00", "p3_signed_01", "p3_common_mode", "p3_decoy_span"):
        assert mechanics["component_specs"][label]["rms_match"]
        for register in ("-1", "1"):
            assert mechanics["delta_rms"][label]["intact"][register] == pytest.approx(
                mechanics["delta_rms"]["p3"]["intact"][register], rel=1e-6
            )
            assert mechanics["carrier_delta_rms"][label][register] == pytest.approx(
                mechanics["carrier_delta_rms"]["p3"][register], rel=1e-6
            )
