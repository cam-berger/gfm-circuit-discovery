"""Precision policy, batch autotuning and the device floor that sets R4's tau grid.

THE ONE OPTIMISATION THAT IS REFUSED RATHER THAN TUNED (D7.1)
-------------------------------------------------------------
ACDC's decision rule is ``delta(metric) < tau``.  On Ampere and later, TF32 carries a 10-bit
mantissa; bf16 fewer still.  Enabling either raises the numerical reproducibility floor, which
by R4 section 3 raises the minimum admissible ``tau``, which coarsens the recovered circuit.
That is a scientific cost paid for throughput and it would be invisible in the artifact.

So the decision path runs in fp32 with TF32 explicitly disabled -- not by relying on a library
default, which has changed across torch versions, but by setting it and recording what was set.
``measure_device_floor`` then quantifies what the refused options would have cost, so the trade
is reported rather than assumed.

WHY A FLOOR IS NEEDED AT ALL
----------------------------
A ``tau`` below the hardware's reproducibility floor yields a **device-dependent circuit**: the
same code and data give different answers on different accelerators and nothing says so.  This
is not hypothetical here -- ``colab_atlas_package/CHECKLIST.md`` gates its cross-device smoke at
absolute tolerance **2e-3**, and ``MECHANISM_VALIDATION_AMENDMENT_A3.md`` accepts a cross-batch
discrepancy in ``(1e-6, 1e-4]``.  An a-priori grid starting at 3e-4 -- the natural first guess --
would have sat *below* the Atlas floor.  The grid is therefore derived from measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import torch

#: tau must clear the measured floor by this factor (R4 section 3).
TAU_FLOOR_MULTIPLE: float = 3.0

#: Five points, log-spaced over two decades from the floor-derived minimum.
TAU_GRID_POINTS: int = 5
TAU_GRID_DECADES: float = 2.0

#: Fraction of free VRAM the caches may claim.  The rest absorbs activations and fragmentation.
DEFAULT_SAFETY: float = 0.6

#: Batch used when no CUDA device is present (MPS/CPU reference runs).
CPU_REFERENCE_BATCH: int = 4


@dataclass(frozen=True)
class PrecisionPolicy:
    """What was actually set, for the manifest.  Not what was intended."""

    matmul_tf32: bool
    cudnn_tf32: bool
    matmul_precision: str
    dtype: str
    deterministic_algorithms: bool

    def to_dict(self) -> dict:
        return asdict(self)


def set_precision_policy(*, deterministic: bool = True) -> PrecisionPolicy:
    """Disable TF32 on both backends and record the resulting state.

    Returns the state read back from torch, not the values requested -- a policy that silently
    failed to apply would otherwise be recorded as applied.
    """
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return PrecisionPolicy(
        matmul_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
        cudnn_tf32=bool(torch.backends.cudnn.allow_tf32),
        matmul_precision=str(torch.get_float32_matmul_precision()),
        dtype="float32",
        deterministic_algorithms=bool(deterministic),
    )


def cache_bytes_per_example(n_nodes: int, seq_len: int, hidden: int, *, dtype_bytes: int = 4) -> int:
    """Corrupt write caches plus a same-order live transient, per example.

    ``n_nodes`` writes of ``[seq_len, hidden]`` are held for the whole sweep, and the
    per-destination reconstruction needs a transient of comparable size.
    """
    return 2 * n_nodes * seq_len * hidden * dtype_bytes


def autotune_batch(
    n_nodes: int,
    seq_len: int,
    hidden: int,
    *,
    device: str | torch.device | None = None,
    safety: float = DEFAULT_SAFETY,
    maximum: int = 64,
) -> int:
    """Largest batch whose caches fit in ``safety`` of free VRAM.

    D7.5.  The design default of 4 came from a 16 GB assumption and wastes most of a 40 GB
    device.  Returns ``CPU_REFERENCE_BATCH`` when there is no CUDA device to query, so the local
    reference run is reproducible rather than host-dependent.
    """
    if safety <= 0 or safety > 1:
        raise ValueError(f"safety must be in (0, 1], got {safety}")
    resolved = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    if not resolved.startswith("cuda") or not torch.cuda.is_available():
        return CPU_REFERENCE_BATCH
    free, _total = torch.cuda.mem_get_info()
    per_example = cache_bytes_per_example(n_nodes, seq_len, hidden)
    batch = int((free * safety) // max(per_example, 1))
    return max(1, min(batch, maximum))


@dataclass(frozen=True)
class DeviceFloor:
    """Cross-device agreement on a fixed panel, and the tau grid it forces."""

    floor: float
    n_panel: int
    tau_min: float
    tau_grid: tuple[float, ...]
    reference_device: str
    run_device: str

    @property
    def passes(self) -> bool:
        return self.tau_min >= TAU_FLOOR_MULTIPLE * self.floor

    def to_dict(self) -> dict:
        return {**asdict(self), "tau_grid": list(self.tau_grid),
                "tau_floor_multiple": TAU_FLOOR_MULTIPLE, "PASS": self.passes}


def tau_grid_from_floor(floor: float) -> tuple[float, ...]:
    """Five log-spaced points from ``3 * floor`` over two decades."""
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor}")
    tau_min = TAU_FLOOR_MULTIPLE * floor
    if tau_min <= 0:
        raise ValueError(
            "measured device floor is exactly 0, which means the panel did not discriminate; "
            "a tau grid cannot be derived from it"
        )
    step = TAU_GRID_DECADES / (TAU_GRID_POINTS - 1)
    return tuple(tau_min * (10.0 ** (step * i)) for i in range(TAU_GRID_POINTS))


def measure_device_floor(
    metric_fn: Callable[[torch.Tensor], float],
    panel: Sequence[torch.Tensor],
    reference: Sequence[float],
    *,
    reference_device: str,
    run_device: str,
) -> DeviceFloor:
    """Max absolute disagreement between this device and a committed reference panel.

    ``reference`` is the metric value for each panel subgraph as computed on the reference
    device and committed to the repository.  The floor is the largest absolute difference; the
    tau grid is derived from it and gated by ``DeviceFloor.passes``.
    """
    if len(panel) != len(reference):
        raise ValueError(
            f"panel has {len(panel)} subgraphs but reference has {len(reference)} values"
        )
    if not panel:
        raise ValueError("device-floor panel is empty; it cannot bound anything")
    floor = max(abs(float(metric_fn(mask)) - float(ref)) for mask, ref in zip(panel, reference))
    return DeviceFloor(
        floor=floor, n_panel=len(panel), tau_min=TAU_FLOOR_MULTIPLE * floor,
        tau_grid=tau_grid_from_floor(floor), reference_device=reference_device,
        run_device=run_device,
    )


def assert_decision_determinism(
    run_prefix: Callable[[], Sequence[bool]], *, n_repeats: int = 2
) -> dict:
    """Re-run a prefix of the search and require an identical accept/reject sequence.

    ACDC is a sequential greedy: one nondeterministic kernel call early in a sweep changes every
    downstream decision, so a re-run silently produces a different circuit.  This is the gate
    that makes that impossible to miss.
    """
    first = list(run_prefix())
    for _ in range(n_repeats - 1):
        again = list(run_prefix())
        if again != first:
            mismatch = next(
                (i for i, (a, b) in enumerate(zip(first, again)) if a != b), len(first)
            )
            return {"PASS": False, "n_decisions": len(first), "first_mismatch_index": mismatch}
    return {"PASS": True, "n_decisions": len(first), "first_mismatch_index": None}
