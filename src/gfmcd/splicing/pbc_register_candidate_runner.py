"""Through-model development screen for P3 register-mediation candidates.

The official provider hooks all 96 attention-head context outputs and all six MLP
writes.  Every behavior intervention patches the complete token interval [393,513)
from the same sequence and the same AG/AC corruption state.  A provider interface is
kept deliberately small so the stage graph can be tested without a checkpoint.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np
import torch

from gfmcd.circuits.hf_bert import (
    _context_from_output,
    _merge_context,
    _replace_output,
    _split_context,
)
from gfmcd.splicing.p3_phase_surgery import fourier_band_component
from gfmcd.splicing.pbc_register_candidates import (
    CANDIDATE_SCREEN_SCHEMA,
    PATH_NAMES,
    PREFIX_CAP,
    TOKEN_SUPPORT,
    component_universe,
)
from gfmcd.splicing.pbc_register_relocation import REGISTERS, phase_delta, preferred_delta
from gfmcd.splicing.pbc_register_runner import (
    RegisterCell,
    carrier_edit_delta,
    embedding_delta_context,
    position_embedding_module,
)


PATCH_OPERATION_REPLACE = "replace_same_state_activation"
PATCH_OPERATION_INJECT = "inject_same_state_activation_delta"


class CandidateMeasurementProvider(Protocol):
    """Injectable boundary used by fast synthetic stage-graph tests."""

    def measure(
        self, cells: Sequence[RegisterCell], *, batch_size: int
    ) -> Mapping[str, object]: ...


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()


def tensor_mapping_sha256(
    tensors: Mapping[str, torch.Tensor],
    *,
    token_support: tuple[int, int] = TOKEN_SUPPORT,
) -> str:
    """Hash actual patched tensor values in stable component order."""

    start, end = token_support
    digest = sha256()
    for component_id in sorted(tensors):
        value = tensors[component_id]
        digest.update(component_id.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        if value.ndim != 3 or not 0 <= start < end <= value.shape[1]:
            raise ValueError("patch tensor does not cover the frozen full token support")
        digest.update(_tensor_bytes(value[:, start:end]))
    return digest.hexdigest()


def sequence_sha256(sequence: str) -> str:
    return sha256(sequence.encode("ascii")).hexdigest()


def validate_same_state_identity(
    *,
    host_sequences: Sequence[str],
    source_sequences: Sequence[str],
    host_state: str,
    source_state: str,
) -> None:
    if host_state not in {"intact", "acceptor_ac"} or source_state != host_state:
        raise ValueError("activation patches must preserve AG/AC corruption state")
    if tuple(host_sequences) != tuple(source_sequences):
        raise ValueError("activation patches must preserve exact sequence identity")


@contextmanager
def activation_patch_context(
    adapter: Any,
    *,
    components: Sequence[Mapping[str, object]],
    source_tensors: Mapping[str, torch.Tensor],
    host_sequences: Sequence[str],
    source_sequences: Sequence[str],
    host_state: str,
    source_state: str,
    operation: str,
    token_support: tuple[int, int] = TOKEN_SUPPORT,
) -> Iterator[dict[str, object]]:
    """Patch exact head/MLP hook sites over the complete frozen token support."""

    validate_same_state_identity(
        host_sequences=host_sequences,
        source_sequences=source_sequences,
        host_state=host_state,
        source_state=source_state,
    )
    if operation not in {PATCH_OPERATION_REPLACE, PATCH_OPERATION_INJECT}:
        raise ValueError("unknown activation patch operation")
    start, end = (int(value) for value in token_support)
    if (start, end) != TOKEN_SUPPORT:
        raise ValueError("candidate intervention must patch the full [393,513) support")
    selected = tuple(components)
    if not selected:
        raise ValueError("activation patch needs at least one component")
    ids = [str(row.get("component_id", "")) for row in selected]
    if len(ids) != len(set(ids)) or any(value not in source_tensors for value in ids):
        raise ValueError("activation source is missing or duplicates a selected component")
    by_head_layer: dict[int, list[int]] = defaultdict(list)
    mlp_layers: list[int] = []
    for row in selected:
        component_id = str(row["component_id"])
        layer = int(row["layer"])
        kind = str(row["kind"])
        if kind == "head":
            head = int(row["head"])
            if component_id != f"L{layer}H{head}":
                raise ValueError("head component identity disagrees with its hook coordinates")
            by_head_layer[layer].append(head)
        elif kind == "mlp":
            if component_id != f"MLP{layer}":
                raise ValueError("MLP component identity disagrees with its hook coordinates")
            mlp_layers.append(layer)
        else:
            raise ValueError("candidate component kind must be head or mlp")
    handles = []

    def head_hook(layer: int, heads: Sequence[int]):
        def hook(_module, _inputs, output):
            context = _context_from_output(output)
            target = _split_context(context, adapter.num_heads).clone()
            if end > target.shape[2]:
                raise RuntimeError("head hook output is shorter than the frozen token support")
            for head in heads:
                component_id = f"L{layer}H{head}"
                source = source_tensors[component_id].to(
                    device=target.device, dtype=target.dtype
                )
                if source.shape != target[:, head].shape:
                    raise ValueError("head patch source shape does not match the host")
                if operation == PATCH_OPERATION_REPLACE:
                    target[:, head, start:end] = source[:, start:end]
                else:
                    target[:, head, start:end] += source[:, start:end]
            return _replace_output(output, _merge_context(target))

        return hook

    def mlp_hook(layer: int):
        def hook(_module, _inputs, output):
            if not torch.is_tensor(output) or output.ndim != 3 or end > output.shape[1]:
                raise RuntimeError("unexpected MLP write shape at the frozen patch hook")
            component_id = f"MLP{layer}"
            source = source_tensors[component_id].to(
                device=output.device, dtype=output.dtype
            )
            if source.shape != output.shape:
                raise ValueError("MLP patch source shape does not match the host")
            edited = output.clone()
            if operation == PATCH_OPERATION_REPLACE:
                edited[:, start:end] = source[:, start:end]
            else:
                edited[:, start:end] += source[:, start:end]
            return edited

        return hook

    try:
        for layer, heads in sorted(by_head_layer.items()):
            handles.append(
                adapter.attention_self_modules[layer].register_forward_hook(
                    head_hook(layer, tuple(sorted(heads)))
                )
            )
        for layer in sorted(mlp_layers):
            handles.append(
                adapter.encoder_layers[layer].output.dense.register_forward_hook(
                    mlp_hook(layer)
                )
            )
        provenance = {
            "operation": operation,
            "token_support": list(TOKEN_SUPPORT),
            "components": ids,
            "host_state": host_state,
            "source_state": source_state,
            "same_sequence": True,
            "same_corruption_state": True,
            "patch_tensor_sha256": tensor_mapping_sha256(
                {value: source_tensors[value] for value in ids}
            ),
        }
        yield provenance
    finally:
        for handle in handles:
            handle.remove()


def _masked_batch(
    adapter: Any,
    sequences: Sequence[str],
    donor_positions: Sequence[int],
):
    if not sequences or len(sequences) != len(donor_positions):
        raise ValueError("masked batch needs one donor position per sequence")
    encoded = adapter.tokenize(sequences)
    mask_id = int(adapter.tokenizer.mask_token_id)
    for index, position in enumerate(donor_positions):
        encoded.input_ids[index, adapter.nucleotide_to_token(int(position))] = mask_id
    return encoded


def _margin_from_output(adapter: Any, encoded: Any, output: Any, positions: Sequence[int]) -> torch.Tensor:
    logits = output.logits.float()
    if not torch.isfinite(logits).all():
        raise RuntimeError("SpliceBERT emitted non-finite logits during candidate screening")
    base_ids = {
        base: int(adapter.tokenizer.convert_tokens_to_ids(base)) for base in "ACGT"
    }
    if len(set(base_ids.values())) != 4 or any(value < 0 for value in base_ids.values()):
        raise RuntimeError("tokenizer does not expose four distinct nucleotide IDs")
    batch = torch.arange(len(positions), device=logits.device)
    token_positions = torch.tensor(
        [adapter.nucleotide_to_token(int(value)) for value in positions],
        device=logits.device,
    )
    selected = logits[batch, token_positions]
    alternatives = torch.stack([selected[:, base_ids[value]] for value in "ACT"], dim=1)
    return (selected[:, base_ids["G"]] - alternatives.mean(dim=1)).detach().cpu()


@dataclass
class CapturedForward:
    margins: torch.Tensor
    components: dict[str, torch.Tensor]
    qk: dict[str, torch.Tensor]
    attention: dict[str, torch.Tensor]


def _captured_forward(
    adapter: Any,
    *,
    sequences: Sequence[str],
    positions: Sequence[int],
    position_delta: torch.Tensor,
    capture: bool,
    diagnostics: bool,
    patch: Mapping[str, object] | None = None,
) -> CapturedForward:
    encoded = _masked_batch(adapter, sequences, positions)
    component_cache: dict[str, torch.Tensor] = {}
    query_cache: dict[int, torch.Tensor] = {}
    key_cache: dict[int, torch.Tensor] = {}
    handles = []

    if capture:
        for layer, module in enumerate(adapter.attention_self_modules):
            def head_capture(_module, _inputs, output, layer=layer):
                component_cache.update(
                    {
                        f"L{layer}H{head}": tensor.detach().clone()
                        for head, tensor in enumerate(
                            _split_context(_context_from_output(output), adapter.num_heads).unbind(1)
                        )
                    }
                )

            handles.append(module.register_forward_hook(head_capture))
        for layer, encoder_layer in enumerate(adapter.encoder_layers):
            handles.append(
                encoder_layer.output.dense.register_forward_hook(
                    lambda _module, _inputs, output, layer=layer: component_cache.__setitem__(
                        f"MLP{layer}", output.detach().clone()
                    )
                )
            )
    if diagnostics:
        for layer, module in enumerate(adapter.attention_self_modules):
            handles.append(
                module.query.register_forward_hook(
                    lambda _module, _inputs, output, layer=layer: query_cache.__setitem__(
                        layer, output.detach().clone()
                    )
                )
            )
            handles.append(
                module.key.register_forward_hook(
                    lambda _module, _inputs, output, layer=layer: key_cache.__setitem__(
                        layer, output.detach().clone()
                    )
                )
            )
    embedding = position_embedding_module(adapter.model)
    patch_context = (
        activation_patch_context(adapter, **patch)
        if patch is not None
        else _null_context()
    )
    try:
        with torch.inference_mode(), embedding_delta_context(
            embedding, [position_delta] * len(sequences)
        ), patch_context:
            output = adapter.model(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                output_attentions=bool(diagnostics),
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()
    margins = _margin_from_output(adapter, encoded, output, positions)
    qk: dict[str, torch.Tensor] = {}
    attention: dict[str, torch.Tensor] = {}
    if diagnostics:
        attentions = getattr(output, "attentions", None)
        if attentions is None or len(attentions) != adapter.num_layers:
            raise RuntimeError("eager SpliceBERT forward did not return all attention matrices")
        donor_tokens = torch.tensor(
            [adapter.nucleotide_to_token(int(value)) for value in positions],
            device=query_cache[0].device,
        )
        batch_index = torch.arange(len(sequences), device=donor_tokens.device)
        start, end = TOKEN_SUPPORT
        for layer in range(adapter.num_layers):
            queries = _split_context(query_cache[layer], adapter.num_heads)
            keys = _split_context(key_cache[layer], adapter.num_heads)
            donor_query = queries[batch_index, :, donor_tokens, :]
            logits = torch.einsum(
                "bhd,bhsd->bhs", donor_query, keys[:, :, start:end, :]
            ) / math.sqrt(adapter.head_dim)
            probabilities = attentions[layer][
                batch_index[:, None],
                torch.arange(adapter.num_heads, device=donor_tokens.device)[None, :],
                donor_tokens[:, None],
                start:end,
            ]
            for head in range(adapter.num_heads):
                qk[f"L{layer}H{head}"] = logits[:, head].detach().cpu()
                attention[f"L{layer}H{head}"] = probabilities[:, head].detach().cpu()
    return CapturedForward(
        margins=margins,
        components={key: value.detach().cpu() for key, value in component_cache.items()},
        qk=qk,
        attention=attention,
    )


@contextmanager
def _null_context() -> Iterator[None]:
    yield


def _subtract_tensors(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    if set(left) != set(right):
        raise ValueError("activation caches do not cover the same candidate universe")
    return {key: left[key] - right[key] for key in left}


def _coupling_rows(
    cells: Sequence[RegisterCell],
    *,
    register: int,
    intact: torch.Tensor,
    corrupt: torch.Tensor,
) -> list[dict[str, object]]:
    return [
        {
            "block_id": cell.block_id,
            "chrom": cell.chrom,
            "strand": cell.strand,
            "register": int(register),
            "donor_phase": int(cell.donor_phase),
            "acceptor_phase": int(cell.acceptor_phase),
            "phase_delta": phase_delta(cell.donor_phase, cell.acceptor_phase),
            "coupling": float(left - right),
        }
        for cell, left, right in zip(cells, intact, corrupt)
    ]


def block_omega3(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, object]]:
    """Reduce a complete development lattice to one omega3 per block."""

    grouped: dict[str, dict[tuple[int, int], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    metadata: dict[str, tuple[str, str]] = {}
    for row in rows:
        block = str(row["block_id"])
        register = int(row["register"])
        delta = int(row["phase_delta"])
        grouped[block][(register, delta)].append(float(row["coupling"]))
        metadata[block] = (str(row["chrom"]), str(row["strand"]))
    output: dict[str, dict[str, object]] = {}
    for block in sorted(grouped):
        scores = []
        for register in REGISTERS:
            means = {
                delta: float(np.mean(grouped[block][(register, delta)]))
                for delta in range(3)
            }
            preferred = preferred_delta(register)
            scores.append(
                means[preferred]
                - float(np.mean([means[value] for value in range(3) if value != preferred]))
            )
        chrom, strand = metadata[block]
        output[block] = {
            "block_id": block,
            "chrom": chrom,
            "strand": strand,
            "omega3": float(np.mean(scores)),
        }
    return output


class HfCandidateMeasurementProvider:
    """Official SpliceBERT hook provider used by the CLI runner."""

    def __init__(
        self,
        adapter: Any,
        *,
        p3_bins: Sequence[int],
        p3_attenuation_dose: float,
        direction: str = "right",
    ) -> None:
        self.adapter = adapter
        self.p3_bins = tuple(int(value) for value in p3_bins)
        self.p3_dose = float(p3_attenuation_dose)
        self.direction = str(direction)
        if adapter.num_layers != 6 or adapter.num_heads != 16:
            raise ValueError("official candidate census requires six layers and sixteen heads")
        if not 0 < self.p3_dose <= 1:
            raise ValueError("candidate screen needs the frozen nonzero P3 attenuation dose")

    def _delta(self, register: int, *, attenuated: bool) -> torch.Tensor:
        embedding = position_embedding_module(self.adapter.model)
        weight = embedding.weight.detach().cpu()
        p3 = fourier_band_component(weight, self.p3_bins)
        delta = carrier_edit_delta(
            p3,
            p3_bins=self.p3_bins,
            register=register,
            start=TOKEN_SUPPORT[0],
            end=TOKEN_SUPPORT[1],
            direction=self.direction,
        )
        if attenuated:
            # Freeze the order: attenuate the complete P3 component first, then
            # phase-ramp the remaining component on support.  Thus the support is
            # (1-dose)*C_r, not C_r-dose*C.
            ramp_delta = delta.clone()
            delta = delta - self.p3_dose * p3
            delta[TOKEN_SUPPORT[0] : TOKEN_SUPPORT[1]] -= (
                self.p3_dose
                * ramp_delta[TOKEN_SUPPORT[0] : TOKEN_SUPPORT[1]]
            )
        return delta

    def _forward_pair(
        self,
        cells: Sequence[RegisterCell],
        *,
        delta: torch.Tensor,
        capture: bool,
        diagnostics: bool,
        intact_patch: Mapping[str, object] | None = None,
        corrupt_patch: Mapping[str, object] | None = None,
    ) -> tuple[CapturedForward, CapturedForward]:
        positions = [cell.donor_position for cell in cells]
        intact_sequences = [cell.intact_sequence for cell in cells]
        corrupt_sequences = [cell.acceptor_ac_sequence for cell in cells]
        intact = _captured_forward(
            self.adapter,
            sequences=intact_sequences,
            positions=positions,
            position_delta=delta,
            capture=capture,
            diagnostics=diagnostics,
            patch=intact_patch,
        )
        corrupt = _captured_forward(
            self.adapter,
            sequences=corrupt_sequences,
            positions=positions,
            position_delta=delta,
            capture=capture,
            diagnostics=diagnostics,
            patch=corrupt_patch,
        )
        return intact, corrupt

    @staticmethod
    def _patch(
        *,
        components: Sequence[Mapping[str, object]],
        source: Mapping[str, torch.Tensor],
        sequences: Sequence[str],
        state: str,
        operation: str,
    ) -> dict[str, object]:
        return {
            "components": components,
            "source_tensors": source,
            "host_sequences": sequences,
            "source_sequences": sequences,
            "host_state": state,
            "source_state": state,
            "operation": operation,
            "token_support": TOKEN_SUPPORT,
        }

    def measure(
        self, cells: Sequence[RegisterCell], *, batch_size: int
    ) -> Mapping[str, object]:
        examples = tuple(cells)
        if not examples or batch_size <= 0:
            raise ValueError("candidate screen needs cells and a positive batch size")
        universe = list(component_universe())
        ids = [str(row["component_id"]) for row in universe]
        metadata = {str(row["component_id"]): row for row in universe}
        raw_rows: list[dict[str, object]] = []
        attenuated_rows: list[dict[str, object]] = []
        no_ramp_rows: list[dict[str, object]] = []
        scrub_rows: dict[str, list[dict[str, object]]] = {value: [] for value in ids}
        restore_rows: dict[str, list[dict[str, object]]] = {value: [] for value in ids}
        sufficiency_rows: dict[str, list[dict[str, object]]] = {value: [] for value in ids}
        norm_squares = {value: {name: 0.0 for name in PATH_NAMES} for value in ids}
        norm_counts = {value: {name: 0 for name in PATH_NAMES} for value in ids}
        qk_sums = defaultdict(float)
        attn_sums = defaultdict(float)
        diagnostic_counts = defaultdict(int)
        stream = sha256()
        patch_events = 0

        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            positions = [cell.donor_position for cell in chunk]
            intact_sequences = [cell.intact_sequence for cell in chunk]
            corrupt_sequences = [cell.acceptor_ac_sequence for cell in chunk]
            zero = self._delta(0, attenuated=False)
            a0_intact, a0_corrupt = self._forward_pair(
                chunk, delta=zero, capture=True, diagnostics=True
            )
            for register in REGISTERS:
                raw_delta = self._delta(register, attenuated=False)
                attenuated_delta = self._delta(register, attenuated=True)
                ar_intact, ar_corrupt = self._forward_pair(
                    chunk, delta=raw_delta, capture=True, diagnostics=True
                )
                br_intact, br_corrupt = self._forward_pair(
                    chunk, delta=attenuated_delta, capture=True, diagnostics=False
                )
                raw_rows.extend(
                    _coupling_rows(
                        chunk,
                        register=register,
                        intact=ar_intact.margins,
                        corrupt=ar_corrupt.margins,
                    )
                )
                attenuated_rows.extend(
                    _coupling_rows(
                        chunk,
                        register=register,
                        intact=br_intact.margins,
                        corrupt=br_corrupt.margins,
                    )
                )
                no_ramp_rows.extend(
                    _coupling_rows(
                        chunk,
                        register=register,
                        intact=a0_intact.margins,
                        corrupt=a0_corrupt.margins,
                    )
                )
                d_intact = _subtract_tensors(a0_intact.components, ar_intact.components)
                d_corrupt = _subtract_tensors(a0_corrupt.components, ar_corrupt.components)
                r_intact = _subtract_tensors(ar_intact.components, br_intact.components)
                r_corrupt = _subtract_tensors(ar_corrupt.components, br_corrupt.components)
                q_intact = _subtract_tensors(ar_intact.components, a0_intact.components)
                q_corrupt = _subtract_tensors(ar_corrupt.components, a0_corrupt.components)
                path_tensors = {
                    "d_path": (d_intact, d_corrupt),
                    "r_path": (r_intact, r_corrupt),
                    "q_path": (q_intact, q_corrupt),
                }
                for path_name, pair in path_tensors.items():
                    for state, values in zip(("intact", "acceptor_ac"), pair):
                        stream.update(f"{start}:{register}:{path_name}:{state}".encode())
                        stream.update(bytes.fromhex(tensor_mapping_sha256(values)))
                        for component_id, tensor in values.items():
                            local = tensor[:, TOKEN_SUPPORT[0] : TOKEN_SUPPORT[1]].float()
                            norm_squares[component_id][path_name] += float(local.square().sum())
                            norm_counts[component_id][path_name] += int(local.numel())
                if register:
                    for component_id in ids:
                        qk_sums[component_id] += float(
                            torch.mean(
                                torch.abs(
                                    ar_intact.qk.get(component_id, torch.zeros(1))
                                    - a0_intact.qk.get(component_id, torch.zeros(1))
                                )
                            )
                        )
                        attn_sums[component_id] += float(
                            torch.mean(
                                torch.abs(
                                    ar_intact.attention.get(component_id, torch.zeros(1))
                                    - a0_intact.attention.get(component_id, torch.zeros(1))
                                )
                            )
                        )
                        if component_id.startswith("L"):
                            diagnostic_counts[component_id] += 1
                for component_id in ids:
                    component = [metadata[component_id]]
                    scrub_intact_patch = self._patch(
                        components=component,
                        source={component_id: a0_intact.components[component_id]},
                        sequences=intact_sequences,
                        state="intact",
                        operation=PATCH_OPERATION_REPLACE,
                    )
                    scrub_corrupt_patch = self._patch(
                        components=component,
                        source={component_id: a0_corrupt.components[component_id]},
                        sequences=corrupt_sequences,
                        state="acceptor_ac",
                        operation=PATCH_OPERATION_REPLACE,
                    )
                    scrub_intact, scrub_corrupt = self._forward_pair(
                        chunk,
                        delta=raw_delta,
                        capture=False,
                        diagnostics=False,
                        intact_patch=scrub_intact_patch,
                        corrupt_patch=scrub_corrupt_patch,
                    )
                    restore_intact_patch = self._patch(
                        components=component,
                        source={component_id: ar_intact.components[component_id]},
                        sequences=intact_sequences,
                        state="intact",
                        operation=PATCH_OPERATION_REPLACE,
                    )
                    restore_corrupt_patch = self._patch(
                        components=component,
                        source={component_id: ar_corrupt.components[component_id]},
                        sequences=corrupt_sequences,
                        state="acceptor_ac",
                        operation=PATCH_OPERATION_REPLACE,
                    )
                    restored_intact, restored_corrupt = self._forward_pair(
                        chunk,
                        delta=attenuated_delta,
                        capture=False,
                        diagnostics=False,
                        intact_patch=restore_intact_patch,
                        corrupt_patch=restore_corrupt_patch,
                    )
                    sufficiency_intact_patch = self._patch(
                        components=component,
                        source={component_id: q_intact[component_id]},
                        sequences=intact_sequences,
                        state="intact",
                        operation=PATCH_OPERATION_INJECT,
                    )
                    sufficiency_corrupt_patch = self._patch(
                        components=component,
                        source={component_id: q_corrupt[component_id]},
                        sequences=corrupt_sequences,
                        state="acceptor_ac",
                        operation=PATCH_OPERATION_INJECT,
                    )
                    sufficient_intact, sufficient_corrupt = self._forward_pair(
                        chunk,
                        delta=zero,
                        capture=False,
                        diagnostics=False,
                        intact_patch=sufficiency_intact_patch,
                        corrupt_patch=sufficiency_corrupt_patch,
                    )
                    scrub_rows[component_id].extend(
                        _coupling_rows(
                            chunk,
                            register=register,
                            intact=scrub_intact.margins,
                            corrupt=scrub_corrupt.margins,
                        )
                    )
                    restore_rows[component_id].extend(
                        _coupling_rows(
                            chunk,
                            register=register,
                            intact=restored_intact.margins,
                            corrupt=restored_corrupt.margins,
                        )
                    )
                    sufficiency_rows[component_id].extend(
                        _coupling_rows(
                            chunk,
                            register=register,
                            intact=sufficient_intact.margins,
                            corrupt=sufficient_corrupt.margins,
                        )
                    )
                    patch_events += 6

        raw_blocks = block_omega3(raw_rows)
        attenuated_blocks = block_omega3(attenuated_rows)
        no_ramp_blocks = block_omega3(no_ramp_rows)
        singleton_records: list[dict[str, object]] = []
        for component_id in ids:
            scrub_blocks = block_omega3(scrub_rows[component_id])
            restore_blocks = block_omega3(restore_rows[component_id])
            sufficiency_blocks = block_omega3(sufficiency_rows[component_id])
            per_block = [
                {
                    "block_id": block,
                    "chrom": raw_blocks[block]["chrom"],
                    "strand": raw_blocks[block]["strand"],
                    "D_path": float(raw_blocks[block]["omega3"])
                    - float(scrub_blocks[block]["omega3"]),
                    "R_path": float(restore_blocks[block]["omega3"])
                    - float(attenuated_blocks[block]["omega3"]),
                    "Q_path": float(sufficiency_blocks[block]["omega3"])
                    - float(no_ramp_blocks[block]["omega3"]),
                }
                for block in sorted(raw_blocks)
            ]
            path_norms = {
                path_name: math.sqrt(
                    norm_squares[component_id][path_name]
                    / norm_counts[component_id][path_name]
                )
                for path_name in PATH_NAMES
            }
            singleton_records.append(
                {
                    "component_id": component_id,
                    "same_state_scrub_loss_in_omega3": float(
                        np.mean([row["D_path"] for row in per_block])
                    ),
                    "same_state_restore_gain_in_omega3": float(
                        np.mean([row["R_path"] for row in per_block])
                    ),
                    "same_state_sufficiency_gain_in_omega3": float(
                        np.mean([row["Q_path"] for row in per_block])
                    ),
                    "path_output_delta_norms": path_norms,
                    "qk_pre_softmax_register_change": (
                        qk_sums[component_id] / diagnostic_counts[component_id]
                        if diagnostic_counts[component_id]
                        else None
                    ),
                    "attention_post_softmax_register_change": (
                        attn_sums[component_id] / diagnostic_counts[component_id]
                        if diagnostic_counts[component_id]
                        else None
                    ),
                    "per_block": per_block,
                }
            )
        ranking = sorted(
            singleton_records,
            key=lambda row: (
                -float(row["same_state_scrub_loss_in_omega3"]),
                str(row["component_id"]),
            ),
        )
        ranked_ids = [str(row["component_id"]) for row in ranking]

        # Static singleton effects nominate the order only.  Every prefix is then
        # re-run jointly through the model, as frozen in §10.1.
        prefix_scrub = {size: [] for size in range(1, PREFIX_CAP + 1)}
        prefix_restore = {size: [] for size in range(1, PREFIX_CAP + 1)}
        prefix_sufficiency = {size: [] for size in range(1, PREFIX_CAP + 1)}
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            intact_sequences = [cell.intact_sequence for cell in chunk]
            corrupt_sequences = [cell.acceptor_ac_sequence for cell in chunk]
            zero = self._delta(0, attenuated=False)
            a0_intact, a0_corrupt = self._forward_pair(
                chunk, delta=zero, capture=True, diagnostics=False
            )
            for register in REGISTERS:
                raw_delta = self._delta(register, attenuated=False)
                attenuated_delta = self._delta(register, attenuated=True)
                ar_intact, ar_corrupt = self._forward_pair(
                    chunk, delta=raw_delta, capture=True, diagnostics=False
                )
                for size in range(1, PREFIX_CAP + 1):
                    selected_ids = ranked_ids[:size]
                    selected = [metadata[value] for value in selected_ids]
                    scrub_intact, scrub_corrupt = self._forward_pair(
                        chunk,
                        delta=raw_delta,
                        capture=False,
                        diagnostics=False,
                        intact_patch=self._patch(
                            components=selected,
                            source={value: a0_intact.components[value] for value in selected_ids},
                            sequences=intact_sequences,
                            state="intact",
                            operation=PATCH_OPERATION_REPLACE,
                        ),
                        corrupt_patch=self._patch(
                            components=selected,
                            source={value: a0_corrupt.components[value] for value in selected_ids},
                            sequences=corrupt_sequences,
                            state="acceptor_ac",
                            operation=PATCH_OPERATION_REPLACE,
                        ),
                    )
                    restored_intact, restored_corrupt = self._forward_pair(
                        chunk,
                        delta=attenuated_delta,
                        capture=False,
                        diagnostics=False,
                        intact_patch=self._patch(
                            components=selected,
                            source={value: ar_intact.components[value] for value in selected_ids},
                            sequences=intact_sequences,
                            state="intact",
                            operation=PATCH_OPERATION_REPLACE,
                        ),
                        corrupt_patch=self._patch(
                            components=selected,
                            source={value: ar_corrupt.components[value] for value in selected_ids},
                            sequences=corrupt_sequences,
                            state="acceptor_ac",
                            operation=PATCH_OPERATION_REPLACE,
                        ),
                    )
                    q_intact = {
                        value: ar_intact.components[value] - a0_intact.components[value]
                        for value in selected_ids
                    }
                    q_corrupt = {
                        value: ar_corrupt.components[value] - a0_corrupt.components[value]
                        for value in selected_ids
                    }
                    sufficient_intact, sufficient_corrupt = self._forward_pair(
                        chunk,
                        delta=zero,
                        capture=False,
                        diagnostics=False,
                        intact_patch=self._patch(
                            components=selected,
                            source=q_intact,
                            sequences=intact_sequences,
                            state="intact",
                            operation=PATCH_OPERATION_INJECT,
                        ),
                        corrupt_patch=self._patch(
                            components=selected,
                            source=q_corrupt,
                            sequences=corrupt_sequences,
                            state="acceptor_ac",
                            operation=PATCH_OPERATION_INJECT,
                        ),
                    )
                    prefix_scrub[size].extend(
                        _coupling_rows(
                            chunk,
                            register=register,
                            intact=scrub_intact.margins,
                            corrupt=scrub_corrupt.margins,
                        )
                    )
                    prefix_restore[size].extend(
                        _coupling_rows(
                            chunk,
                            register=register,
                            intact=restored_intact.margins,
                            corrupt=restored_corrupt.margins,
                        )
                    )
                    prefix_sufficiency[size].extend(
                        _coupling_rows(
                            chunk,
                            register=register,
                            intact=sufficient_intact.margins,
                            corrupt=sufficient_corrupt.margins,
                        )
                    )
                    patch_events += 6
        prefixes: list[dict[str, object]] = []
        for size in range(1, PREFIX_CAP + 1):
            scrub_blocks = block_omega3(prefix_scrub[size])
            restore_blocks = block_omega3(prefix_restore[size])
            sufficiency_blocks = block_omega3(prefix_sufficiency[size])
            per_block = [
                {
                    "block_id": block,
                    "chrom": raw_blocks[block]["chrom"],
                    "strand": raw_blocks[block]["strand"],
                    "D_path": float(raw_blocks[block]["omega3"])
                    - float(scrub_blocks[block]["omega3"]),
                    "R_path": float(restore_blocks[block]["omega3"])
                    - float(attenuated_blocks[block]["omega3"]),
                    "Q_path": float(sufficiency_blocks[block]["omega3"])
                    - float(no_ramp_blocks[block]["omega3"]),
                }
                for block in sorted(raw_blocks)
            ]
            prefixes.append(
                {
                    "prefix_size": size,
                    "component_ids": ranked_ids[:size],
                    "D_path": float(np.mean([row["D_path"] for row in per_block])),
                    "R_path": float(np.mean([row["R_path"] for row in per_block])),
                    "Q_path": float(np.mean([row["Q_path"] for row in per_block])),
                    "per_block": per_block,
                }
            )
        return {
            "candidate_universe": universe,
            "baseline": {
                "mean_raw_omega3": float(
                    np.mean([row["omega3"] for row in raw_blocks.values()])
                ),
                "mean_attenuated_host_omega3": float(
                    np.mean([row["omega3"] for row in attenuated_blocks.values()])
                ),
                "mean_no_ramp_host_omega3": float(
                    np.mean([row["omega3"] for row in no_ramp_blocks.values()])
                ),
                "per_block": {
                    block: {
                        "raw_omega3": raw_blocks[block]["omega3"],
                        "attenuated_host_omega3": attenuated_blocks[block]["omega3"],
                        "no_ramp_host_omega3": no_ramp_blocks[block]["omega3"],
                    }
                    for block in sorted(raw_blocks)
                },
            },
            "singletons": singleton_records,
            "joint_prefixes": prefixes,
            "tensor_stream": {
                "sha256": stream.hexdigest(),
                "patch_events": patch_events,
                "same_sequence_only": True,
                "same_corruption_state_only": True,
                "full_token_support_only": True,
            },
            "evidence_levels": {
                "qk_pre_softmax": "diagnostic_only",
                "attention_post_softmax": "diagnostic_only",
                "component_output": "measured_output_delta_norm",
                "behavior_same_state_patch": "candidate_rank_and_joint_prefix",
            },
            "model_hook_contract": {
                "heads": "attention.self context before attention.output.dense",
                "mlps": "encoder.layer.output.dense before residual add and LayerNorm",
                "token_support": list(TOKEN_SUPPORT),
            },
        }


def execute_candidate_screen(
    cells: Sequence[RegisterCell],
    *,
    provider: CandidateMeasurementProvider,
    batch_size: int,
    enforce_development_contract: bool = True,
) -> dict[str, object]:
    """Attach immutable stage metadata to one provider's executed measurements."""

    blocks = {cell.block_id for cell in cells}
    if enforce_development_contract and len(blocks) != 120:
        raise ValueError("official candidate screen requires exactly 120 development blocks")
    measured = dict(provider.measure(cells, batch_size=batch_size))
    return {
        "schema": CANDIDATE_SCREEN_SCHEMA,
        "status": "COMPLETE",
        "stage": "development",
        "n_blocks": len(blocks),
        "n_cells": len(cells),
        "token_support": list(TOKEN_SUPPORT),
        "rank_metric": "same_state_scrub_loss_in_omega3",
        "prefix_rule": {
            "threshold": 0.80,
            "cap": PREFIX_CAP,
            "fallback": "top_8_distributed_candidate_set",
        },
        "outcome_firewall": {"phase3_reserve_read": False},
        **measured,
    }


__all__ = [
    "CandidateMeasurementProvider",
    "CapturedForward",
    "HfCandidateMeasurementProvider",
    "PATCH_OPERATION_INJECT",
    "PATCH_OPERATION_REPLACE",
    "activation_patch_context",
    "block_omega3",
    "execute_candidate_screen",
    "sequence_sha256",
    "tensor_mapping_sha256",
    "validate_same_state_identity",
]
