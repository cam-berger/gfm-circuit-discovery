"""Through-model kernel for the conditional P3 register-relocation experiment.

The kernel applies additive edits to the position-embedding lookup while keeping nucleotide
IDs fixed.  It scores the exact C43 linear masked-donor endpoint in intact and AG->AC states
under the same edit.  Artifact sealing, authorization, and adjudication belong to the study
runners; this module is intentionally small enough to test with an in-memory model.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator, Mapping, Sequence

import torch

from gfmcd.splicing.p3_phase_surgery import (
    fourier_band_component,
    signed_permutation_component,
)
from gfmcd.splicing.pbc_register_relocation import (
    P10_ATTENUATED,
    P10_INTACT,
    REGISTERS,
    phase_delta,
)


@dataclass(frozen=True)
class RegisterCell:
    """One fixed crossed-boundary cell, including its paired corruption."""

    block_id: str
    chrom: str
    strand: str
    donor_phase: int
    acceptor_phase: int
    intact_sequence: str
    acceptor_ac_sequence: str
    donor_position: int
    acceptor_position: int

    def __post_init__(self) -> None:
        intact = self.intact_sequence.upper().replace("U", "T")
        corrupt = self.acceptor_ac_sequence.upper().replace("U", "T")
        if not self.block_id or not self.chrom or self.strand not in {"+", "-"}:
            raise ValueError("register cell needs block, chromosome, and strand identity")
        if len(intact) != len(corrupt) or not intact or set(intact + corrupt) - set("ACGT"):
            raise ValueError("register-cell sequences must be equal-length, nonempty ACGT strings")
        donor = int(self.donor_position)
        acceptor = int(self.acceptor_position)
        if not (0 <= donor < acceptor <= len(intact)):
            raise ValueError("invalid donor/acceptor coordinates")
        if intact[donor : donor + 2] != "GT":
            raise ValueError("intact cell lacks donor GT")
        if intact[acceptor - 2 : acceptor] != "AG":
            raise ValueError("intact cell lacks acceptor AG")
        if corrupt[acceptor - 2 : acceptor] != "AC":
            raise ValueError("corruption is not acceptor AG->AC")
        changed = [index for index, (left, right) in enumerate(zip(intact, corrupt)) if left != right]
        if changed != [acceptor - 1]:
            raise ValueError("AG->AC state must differ only at the acceptor G")
        # Validate phase values and convention now, before any model access.
        phase_delta(self.donor_phase, self.acceptor_phase)
        object.__setattr__(self, "intact_sequence", intact)
        object.__setattr__(self, "acceptor_ac_sequence", corrupt)
        object.__setattr__(self, "donor_position", donor)
        object.__setattr__(self, "acceptor_position", acceptor)


def register_cell_from_mapping(row: Mapping[str, object]) -> RegisterCell:
    """Parse the stable Phase-I cell fields used by the Phase-II/III runners."""

    return RegisterCell(
        block_id=str(row["block_id"]),
        chrom=str(row["chrom"]),
        strand=str(row["strand"]),
        donor_phase=int(row["donor_phase"]),
        acceptor_phase=int(row["acceptor_phase"]),
        intact_sequence=str(row["sequence_intact"]),
        acceptor_ac_sequence=str(row["sequence_acceptor_ac"]),
        donor_position=int(row["donor_pos"]),
        acceptor_position=int(row["acceptor_pos"]),
    )


def position_embedding_module(model) -> torch.nn.Embedding:
    """Resolve the BERT position table and fail on architecture drift."""

    candidates = (
        "bert.embeddings.position_embeddings",
        "base_model.bert.embeddings.position_embeddings",
        "base_model.embeddings.position_embeddings",
    )
    for path in candidates:
        value = model
        for part in path.split("."):
            if not hasattr(value, part):
                value = None
                break
            value = getattr(value, part)
        if isinstance(value, torch.nn.Embedding):
            return value
    raise AttributeError("model has no supported BERT position embedding module")


def carrier_edit_delta(
    p3_component: torch.Tensor,
    *,
    p3_bins: Sequence[int],
    register: int,
    start: int,
    end: int,
    direction: str = "right",
    phase_mode: str = "relative",
    match_p3_rms: float | None = None,
    p10_component: torch.Tensor | None = None,
    p10_dose: float = 0.0,
) -> torch.Tensor:
    """Return a relative analytic phase-ramp edit and optional P10 attenuation."""

    if p3_component.ndim != 2 or not torch.isfinite(p3_component).all():
        raise ValueError("P3 component must be a finite [positions, hidden] tensor")
    amount = int(register)
    if amount not in REGISTERS:
        raise ValueError("register must be -1, 0, or +1")
    if direction not in {"left", "right"}:
        raise ValueError("direction must be left or right")
    if phase_mode not in {"relative", "common"}:
        raise ValueError("phase_mode must be relative or common")
    if not (0 <= int(start) < int(end) <= p3_component.shape[0]):
        raise ValueError("carrier-edit span leaves the position table")
    target = torch.arange(int(start), int(end), device=p3_component.device)
    if target.numel() < 2:
        raise ValueError("carrier-edit span needs at least two rows")
    dose = float(p10_dose)
    if not math.isfinite(dose) or not 0 <= dose <= 1:
        raise ValueError("P10 dose must lie in [0, 1]")
    delta = torch.zeros_like(p3_component)
    if amount:
        sign = 1.0 if direction == "left" else -1.0
        fractions = (
            torch.linspace(
                0.0,
                1.0,
                target.numel(),
                device=p3_component.device,
                dtype=torch.float64,
            )
            if phase_mode == "relative"
            else torch.ones(target.numel(), device=p3_component.device, dtype=torch.float64)
        )
        phases = sign * (2.0 * torch.pi * amount / 3.0) * fractions
        analytic = _analytic_band_rows_torch(
            p3_component, p3_bins, target.to(torch.float64)
        )
        shifted = torch.real(analytic * torch.exp(1j * phases[:, None]))
        delta[target] = shifted.to(p3_component.dtype) - p3_component[target]
        if match_p3_rms is not None:
            requested = float(match_p3_rms)
            if not math.isfinite(requested) or requested <= 0:
                raise ValueError("matched P3 RMS must be positive and finite")
            observed = float(torch.sqrt(torch.mean(delta.float().square())))
            if observed <= 0:
                raise ValueError("cannot RMS-match a zero carrier edit")
            delta = delta * (requested / observed)
    elif match_p3_rms is not None and float(match_p3_rms) != 0:
        raise ValueError("register-zero edit can only match zero P3 RMS")
    if dose:
        if p10_component is None or p10_component.shape != p3_component.shape:
            raise ValueError("P10 attenuation needs a shape-matched component")
        delta = delta - dose * p10_component.to(device=delta.device, dtype=delta.dtype)
    return delta


def _analytic_band_rows_torch(
    component: torch.Tensor,
    bins: Sequence[int],
    coordinates: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the positive-frequency analytic selected-band component."""

    selected = tuple(sorted(set(int(value) for value in bins)))
    length = component.shape[0]
    if not selected or selected[0] <= 0 or selected[-1] >= length / 2:
        raise ValueError("analytic carrier evaluation requires non-DC, non-Nyquist bins")
    if coordinates.ndim != 1 or not torch.isfinite(coordinates).all():
        raise ValueError("carrier coordinates must be a finite vector")
    # Float64 mechanics prevent an avoidable endpoint error; the result is cast back to
    # the embedding dtype only after the exact selected-bin evaluation.
    values = component.to(torch.float64)
    spectrum = torch.fft.rfft(values, dim=0)[list(selected)]
    local_bins = torch.tensor(selected, dtype=torch.float64, device=values.device)
    angles = 2 * torch.pi * coordinates.to(values.device)[:, None] * local_bins[None, :] / length
    phasors = torch.exp(1j * angles)
    return (2.0 / length) * (phasors @ spectrum)


@contextmanager
def embedding_delta_context(
    embedding: torch.nn.Embedding,
    deltas: Sequence[torch.Tensor],
) -> Iterator[None]:
    """Apply one frozen additive position-table delta per batch element."""

    frozen = tuple(deltas)
    if not frozen:
        raise ValueError("at least one embedding delta is required")
    for delta in frozen:
        if delta.shape != embedding.weight.shape or not torch.isfinite(delta).all():
            raise ValueError("every delta must be finite and match the position table")

    def hook(_module, _inputs, output):
        if output.ndim != 3 or output.shape[-1] != embedding.weight.shape[1]:
            raise RuntimeError("unexpected position-embedding output shape")
        batch = len(frozen)
        if output.shape[0] not in {1, batch}:
            raise RuntimeError("position-embedding batch does not match edit batch")
        edited = output.expand(batch, -1, -1).clone() if output.shape[0] == 1 else output.clone()
        for index, delta in enumerate(frozen):
            local = delta.to(device=edited.device, dtype=edited.dtype)
            edited[index] += local[: edited.shape[1]]
        return edited

    handle = embedding.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _linear_masked_margin(adapter, sequences: Sequence[str], positions: Sequence[int]) -> torch.Tensor:
    if len(sequences) != len(positions) or not sequences:
        raise ValueError("one masked position is required per sequence")
    encoded = adapter.tokenize(sequences)
    mask_id = int(adapter.tokenizer.mask_token_id)
    token_positions = [adapter.nucleotide_to_token(int(position)) for position in positions]
    for index, token_position in enumerate(token_positions):
        encoded.input_ids[index, token_position] = mask_id
    output = adapter.model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        return_dict=True,
    )
    logits = output.logits.float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("model emitted non-finite logits")
    base_ids = {
        base: int(adapter.tokenizer.convert_tokens_to_ids(base)) for base in "ACGT"
    }
    if len(set(base_ids.values())) != 4 or any(value < 0 for value in base_ids.values()):
        raise RuntimeError("tokenizer lacks four distinct nucleotide IDs")
    batch_index = torch.arange(len(sequences), device=logits.device)
    position_index = torch.tensor(token_positions, device=logits.device)
    selected = logits[batch_index, position_index]
    target = selected[:, base_ids["G"]]
    alternatives = torch.stack([selected[:, base_ids[base]] for base in "ACT"], dim=1)
    return (target - alternatives.mean(dim=1)).detach().cpu()


def score_cell_batch(
    adapter,
    cells: Sequence[RegisterCell],
    *,
    delta: torch.Tensor,
) -> list[dict[str, float]]:
    """Score intact and AG->AC coupling under one identical carrier edit."""

    if not cells:
        return []
    embedding = position_embedding_module(adapter.model)
    deltas = [delta] * len(cells)
    positions = [cell.donor_position for cell in cells]
    with torch.inference_mode(), embedding_delta_context(embedding, deltas):
        intact = _linear_masked_margin(
            adapter, [cell.intact_sequence for cell in cells], positions
        )
        corrupt = _linear_masked_margin(
            adapter, [cell.acceptor_ac_sequence for cell in cells], positions
        )
    return [
        {
            "intact_margin": float(left),
            "corrupt_margin": float(right),
            "coupling": float(left - right),
        }
        for left, right in zip(intact, corrupt)
    ]


def score_cell_batch_native(
    adapter,
    cells: Sequence[RegisterCell],
) -> list[dict[str, float]]:
    """Score the no-hook baseline used to prove the register-zero replay."""

    if not cells:
        return []
    positions = [cell.donor_position for cell in cells]
    with torch.inference_mode():
        intact = _linear_masked_margin(
            adapter, [cell.intact_sequence for cell in cells], positions
        )
        corrupt = _linear_masked_margin(
            adapter, [cell.acceptor_ac_sequence for cell in cells], positions
        )
    return [
        {
            "intact_margin": float(left),
            "corrupt_margin": float(right),
            "coupling": float(left - right),
        }
        for left, right in zip(intact, corrupt)
    ]


def evaluate_register_lattice(
    adapter,
    cells: Sequence[RegisterCell],
    *,
    p3_bins: Sequence[int],
    span_start_token: int,
    span_end_token: int,
    decoy_span_token: tuple[int, int] = (153, 273),
    control_seeds: Sequence[int],
    batch_size: int,
    direction: str = "right",
    joint: bool = False,
    p10_bins: Sequence[int] = (),
    p10_dose: float = 0.0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Evaluate the complete target/control lattice and return raw rows plus mechanics."""

    examples = tuple(cells)
    if not examples or batch_size <= 0:
        raise ValueError("cells and a positive batch size are required")
    if len({cell.block_id for cell in examples}) * 9 != len(examples):
        raise ValueError("each block must contribute exactly nine crossed cells")
    by_block: dict[str, set[tuple[int, int]]] = {}
    for cell in examples:
        by_block.setdefault(cell.block_id, set()).add((cell.donor_phase, cell.acceptor_phase))
    if any(cells_present != {(p, q) for p in range(3) for q in range(3)} for cells_present in by_block.values()):
        raise ValueError("every block needs a complete donor-phase x acceptor-phase lattice")
    seeds = tuple(int(seed) for seed in control_seeds)
    if len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("control seeds must be unique non-negative integers")

    embedding = position_embedding_module(adapter.model)
    weight = embedding.weight.detach().cpu()
    p3 = fourier_band_component(weight, p3_bins)
    components = {"p3": p3}
    components.update(
        {
            f"p3_signed_{index:02d}": signed_permutation_component(p3, seed=seed)
            for index, seed in enumerate(seeds)
        }
    )
    p10 = None
    if joint:
        if not p10_bins or not 0 < float(p10_dose) <= 1:
            raise ValueError("joint mode needs P10 bins and a nonzero frozen dose")
        p10 = fourier_band_component(weight, p10_bins)
    states = (P10_INTACT, P10_ATTENUATED) if joint else (P10_INTACT,)
    decoy_start, decoy_end = (int(value) for value in decoy_span_token)
    if (
        decoy_end - decoy_start != span_end_token - span_start_token
        or not (0 <= decoy_start < decoy_end <= weight.shape[0])
        or max(decoy_start, span_start_token) < min(decoy_end, span_end_token)
    ):
        raise ValueError("decoy span must be valid, disjoint, and length-matched")
    specifications = {
        label: {
            "component": component,
            "span": (span_start_token, span_end_token),
            "phase_mode": "relative",
            "family": "target" if label == "p3" else "signed_hidden_axis",
            "rms_match": label != "p3",
        }
        for label, component in components.items()
    }
    specifications["p3_common_mode"] = {
        "component": p3,
        "span": (span_start_token, span_end_token),
        "phase_mode": "common",
        "family": "common_mode",
        "rms_match": True,
    }
    specifications["p3_decoy_span"] = {
        "component": p3,
        "span": (decoy_start, decoy_end),
        "phase_mode": "relative",
        "family": "decoy_span",
        "rms_match": True,
    }
    rows: list[dict[str, object]] = []
    no_hook_rows: list[dict[str, object]] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        scores = score_cell_batch_native(adapter, chunk)
        for cell, score in zip(chunk, scores):
            no_hook_rows.append(
                {
                    "block_id": cell.block_id,
                    "chrom": cell.chrom,
                    "strand": cell.strand,
                    "donor_phase": cell.donor_phase,
                    "acceptor_phase": cell.acceptor_phase,
                    "phase_delta": phase_delta(cell.donor_phase, cell.acceptor_phase),
                    **score,
                }
            )
    delta_rms: dict[str, dict[str, dict[str, float]]] = {}
    carrier_delta_rms: dict[str, dict[str, float]] = {}
    for component_label, specification in specifications.items():
        component = specification["component"]
        delta_rms[component_label] = {}
        carrier_delta_rms[component_label] = {}
        for state in states:
            dose = 0.0 if state == P10_INTACT else float(p10_dose)
            delta_rms[component_label][state] = {}
            for register in REGISTERS:
                target_delta = carrier_edit_delta(
                    p3,
                    p3_bins=p3_bins,
                    register=register,
                    start=span_start_token,
                    end=span_end_token,
                    direction=direction,
                )
                target_rms = float(torch.sqrt(torch.mean(target_delta.float().square())))
                local_start, local_end = specification["span"]
                carrier_delta = carrier_edit_delta(
                    component,
                    p3_bins=p3_bins,
                    register=register,
                    start=local_start,
                    end=local_end,
                    direction=direction,
                    phase_mode=str(specification["phase_mode"]),
                    match_p3_rms=(
                        target_rms
                        if bool(specification["rms_match"]) and register != 0
                        else None
                    ),
                )
                carrier_delta_rms[component_label][str(register)] = float(
                    torch.sqrt(torch.mean(carrier_delta.float().square()))
                )
                delta = (
                    carrier_delta
                    if not dose
                    else carrier_delta
                    - dose * p10.to(
                        device=carrier_delta.device, dtype=carrier_delta.dtype
                    )
                )
                delta_rms[component_label][state][str(register)] = float(
                    torch.sqrt(torch.mean(delta.float().square()))
                )
                for start in range(0, len(examples), batch_size):
                    chunk = examples[start : start + batch_size]
                    scores = score_cell_batch(adapter, chunk, delta=delta)
                    for cell, score in zip(chunk, scores):
                        rows.append(
                            {
                                "block_id": cell.block_id,
                                "chrom": cell.chrom,
                                "strand": cell.strand,
                                "component": component_label,
                                "control_family": specification["family"],
                                "p10_state": state,
                                "register": register,
                                "donor_phase": cell.donor_phase,
                                "acceptor_phase": cell.acceptor_phase,
                                "phase_delta": phase_delta(
                                    cell.donor_phase, cell.acceptor_phase
                                ),
                                **score,
                            }
                        )
    replay_rows = [
        row
        for row in rows
        if row["component"] == "p3"
        and row["p10_state"] == P10_INTACT
        and int(row["register"]) == 0
    ]
    replay_by_key = {
        (str(row["block_id"]), int(row["donor_phase"]), int(row["acceptor_phase"])): row
        for row in replay_rows
    }
    native_by_key = {
        (str(row["block_id"]), int(row["donor_phase"]), int(row["acceptor_phase"])): row
        for row in no_hook_rows
    }
    if set(replay_by_key) != set(native_by_key):
        raise RuntimeError("register-zero and no-hook cell supports differ")
    no_hook_max_abs = max(
        abs(float(replay_by_key[key][metric]) - float(native_by_key[key][metric]))
        for key in replay_by_key
        for metric in ("intact_margin", "corrupt_margin", "coupling")
    )
    mechanics = {
        "position_table_shape": list(weight.shape),
        "p3_bins": [int(value) for value in p3_bins],
        "p10_bins": [int(value) for value in p10_bins],
        "p10_dose": float(p10_dose),
        "span_token": [int(span_start_token), int(span_end_token)],
        "decoy_span_token": [decoy_start, decoy_end],
        "direction": direction,
        "control_seeds": list(seeds),
        "component_specs": {
            label: {
                "family": str(specification["family"]),
                "span_token": [int(value) for value in specification["span"]],
                "phase_mode": str(specification["phase_mode"]),
                "rms_match": bool(specification["rms_match"]),
            }
            for label, specification in specifications.items()
        },
        "delta_rms": delta_rms,
        "carrier_delta_rms": carrier_delta_rms,
        "no_hook_rows": no_hook_rows,
        "register_zero_no_hook_max_abs": float(no_hook_max_abs),
    }
    return rows, mechanics


__all__ = [
    "RegisterCell",
    "carrier_edit_delta",
    "embedding_delta_context",
    "evaluate_register_lattice",
    "position_embedding_module",
    "register_cell_from_mapping",
    "score_cell_batch",
    "score_cell_batch_native",
]
