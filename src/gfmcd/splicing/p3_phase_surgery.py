"""Local, nucleotide-invariant surgery on SpliceBERT's learned period-3 carrier.

The intervention leaves token IDs and every nucleotide unchanged.  It decomposes the learned
position table into a frozen Fourier-band component ``C3`` and a residual, then replaces ``C3[x]``
inside a selected block by ``C3[x +/- r]``.  All non-P3 position information and all positions
outside the block are byte-identical.

This isolates present-day carrier phase use.  It does not identify why training created the carrier
or imply that the spliceosome performs the same operation.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator, Sequence

import torch


P3_BAND_1026 = tuple(range(339, 346))


@dataclass(frozen=True)
class PhaseSurgerySpec:
    """One batch element's token-coordinate carrier shift."""

    start: int
    end: int
    amount: int
    direction: str

    def validate(self, table_length: int) -> None:
        if not (0 <= self.start < self.end <= table_length):
            raise ValueError("phase-surgery block lies outside position table")
        if isinstance(self.amount, bool) or not isinstance(self.amount, int) or self.amount < 0:
            raise ValueError("phase-surgery amount must be a non-negative integer")
        if self.direction not in {"left", "right"}:
            raise ValueError("phase-surgery direction must be left or right")
        source_start = self.start + self.amount if self.direction == "left" else self.start - self.amount
        source_end = self.end + self.amount if self.direction == "left" else self.end - self.amount
        if source_start < 0 or source_end > table_length:
            raise ValueError("phase-surgery source positions leave the position table")


def fourier_band_component(weight: torch.Tensor, bins: Sequence[int]) -> torch.Tensor:
    """Return the real position-table component supported only on selected rFFT bins."""

    if weight.ndim != 2 or weight.shape[0] < 3 or not torch.isfinite(weight).all():
        raise ValueError("position weight must be a finite [positions, hidden] tensor")
    unique = tuple(sorted(set(int(index) for index in bins)))
    if not unique:
        raise ValueError("at least one Fourier bin is required")
    maximum = weight.shape[0] // 2
    if unique[0] <= 0 or unique[-1] > maximum:
        raise ValueError(f"Fourier bins must lie in [1, {maximum}]")
    centered = weight - weight.mean(dim=0, keepdim=True)
    spectrum = torch.fft.rfft(centered.float(), dim=0)
    selected = torch.zeros_like(spectrum)
    selected[list(unique)] = spectrum[list(unique)]
    component = torch.fft.irfft(selected, n=weight.shape[0], dim=0)
    return component.to(dtype=weight.dtype, device=weight.device)


def phase_shift_delta(
    component: torch.Tensor,
    spec: PhaseSurgerySpec,
) -> torch.Tensor:
    """Return ``C3[source]-C3[target]`` for a validated local shift."""

    if component.ndim != 2:
        raise ValueError("carrier component must be [positions, hidden]")
    spec.validate(component.shape[0])
    target = torch.arange(spec.start, spec.end, device=component.device)
    source = target + spec.amount if spec.direction == "left" else target - spec.amount
    return component[source] - component[target]


def surgery_rms(component: torch.Tensor, spec: PhaseSurgerySpec) -> float:
    delta = phase_shift_delta(component, spec).float()
    return float(torch.sqrt(torch.mean(delta * delta)))


def signed_permutation_component(component: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Return an exactly norm/geometry-matched hidden-direction control.

    A signed permutation is an orthogonal transform of the hidden axis.  It preserves the complete
    spatial Gram matrix, every row norm, every phase-shift RMS curve, and the Fourier support of the
    learned carrier while scrambling which model features receive that carrier.  This makes it a
    stronger local-surgery null than an unrelated Fourier band with a different wavelength or dose.
    """

    if component.ndim != 2 or component.shape[1] < 2 or not torch.isfinite(component).all():
        raise ValueError("component must be a finite [positions, hidden] tensor")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("signed-permutation seed must be a non-negative integer")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    hidden = component.shape[1]
    permutation = torch.randperm(hidden, generator=generator)
    signs = torch.randint(0, 2, (hidden,), generator=generator, dtype=torch.int64)
    signs = signs.mul(2).sub(1).to(dtype=component.dtype)
    return component[:, permutation.to(component.device)] * signs.to(component.device)


@contextmanager
def local_phase_surgery(
    embedding: torch.nn.Embedding,
    specs: Sequence[PhaseSurgerySpec],
    component: torch.Tensor,
) -> Iterator[None]:
    """Apply batch-specific local P3 shifts to a Hugging Face position embedding lookup.

    BERT normally looks up position IDs with shape ``[1, length]`` and broadcasts the result over
    the token batch.  The hook expands that output to ``len(specs)`` and applies one audited block
    per batch item.  Clean and partner-corrupted forwards must be executed inside the same context.
    """

    if not specs:
        raise ValueError("at least one phase-surgery spec is required")
    if component.shape != embedding.weight.shape:
        raise ValueError("carrier component must match the position embedding table")
    frozen_specs = tuple(specs)
    for spec in frozen_specs:
        spec.validate(component.shape[0])

    def hook(_module, _inputs, output):
        if output.ndim != 3 or output.shape[-1] != component.shape[-1]:
            raise RuntimeError("unexpected position-embedding output shape")
        batch = len(frozen_specs)
        if output.shape[0] not in (1, batch):
            raise RuntimeError(
                f"position lookup batch={output.shape[0]} incompatible with {batch} surgery specs"
            )
        edited = output.expand(batch, -1, -1).clone() if output.shape[0] == 1 else output.clone()
        local_component = component.to(device=edited.device, dtype=edited.dtype)
        for index, spec in enumerate(frozen_specs):
            edited[index, spec.start:spec.end] += phase_shift_delta(local_component, spec)
        return edited

    handle = embedding.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def modulo_recovery_profile(
    component: torch.Tensor,
    *,
    start: int,
    end: int,
    direction: str,
    amounts: Sequence[int] = tuple(range(2, 11)),
) -> dict[int, float]:
    """Mechanical RMS dose curve used to verify 3/6/9 carrier recovery before model inference."""

    return {
        int(amount): surgery_rms(
            component,
            PhaseSurgerySpec(start=start, end=end, amount=int(amount), direction=direction),
        )
        for amount in amounts
    }


__all__ = [
    "P3_BAND_1026",
    "PhaseSurgerySpec",
    "fourier_band_component",
    "local_phase_surgery",
    "modulo_recovery_profile",
    "phase_shift_delta",
    "signed_permutation_component",
    "surgery_rms",
]
