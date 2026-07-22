"""Pure construction and analysis mechanics for the P3 closure study.

This module deliberately contains no model-loading or forward-pass code.  It implements the
sequence invariants and gene-first estimands specified by
``PREREGISTRATION_P3_CLOSURE.md`` so that pilots, package builders, and confirmatory runners use
the same audited primitives.

Coordinates are zero-based, half-open, and refer to the nucleotide sequence supplied by the
caller.  Splice-site positions are treated as coordinates in that same sequence; callers remain
responsible for using the project's established donor/acceptor coordinate convention.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


CANONICAL_SPLICE_MOTIFS = ("AG", "GT", "GC")
STOP_CODONS = frozenset(("TAA", "TAG", "TGA"))
ROTATION_DIRECTIONS = ("left", "right")


@dataclass(frozen=True, order=True)
class ContextBlock:
    """A named half-open sequence block eligible for cyclic rotation."""

    start: int
    end: int
    context: str = ""

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class RotationVariant:
    """One deterministic block-rotation variant."""

    amount: int
    direction: str
    block: ContextBlock
    sequence: str


@dataclass(frozen=True)
class RotationGateResult:
    """Outcome-blind sequence-construction gate result.

    ``checks`` keeps every individual gate visible.  A caller must not infer eligibility by
    inspecting only a subset of checks; use ``passed`` or ``require_passed``.
    """

    checks: Mapping[str, bool]
    metrics: Mapping[str, object]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(name for name, ok in self.checks.items() if not ok)

    def require_passed(self) -> None:
        if not self.passed:
            raise ValueError("rotation construction failed: " + ", ".join(self.failures))


@dataclass(frozen=True)
class PhasorResult:
    """Site-excluded period-3 attention phasor and its frozen signed projection."""

    value: complex
    signed_projection: float
    amplitude: float
    phase_angle: float
    included_mass: float
    n_keys: int
    class_masses: tuple[float, float, float]


def _as_block(block: ContextBlock | tuple[int, int], *, context: str = "") -> ContextBlock:
    if isinstance(block, ContextBlock):
        return block
    if len(block) != 2:
        raise ValueError("a block must be a (start, end) pair")
    return ContextBlock(int(block[0]), int(block[1]), context)


def _validate_block(block: ContextBlock, sequence_length: int | None = None) -> None:
    if block.start < 0 or block.end <= block.start:
        raise ValueError(f"invalid half-open block [{block.start}, {block.end})")
    if sequence_length is not None and block.end > sequence_length:
        raise ValueError(f"block [{block.start}, {block.end}) exceeds sequence length")


def intervals_overlap(a: ContextBlock | tuple[int, int], b: ContextBlock | tuple[int, int]) -> bool:
    """Return whether two half-open intervals overlap."""

    aa, bb = _as_block(a), _as_block(b)
    return max(aa.start, bb.start) < min(aa.end, bb.end)


def rotate_cyclic(
    sequence: str,
    block: ContextBlock | tuple[int, int],
    amount: int,
    direction: str,
) -> str:
    """Rotate one sequence block without changing length, composition, or coordinates.

    Rotation amounts are reduced modulo the block length.  Negative amounts are rejected instead
    of silently reversing direction, which keeps package construction explicit and auditable.
    """

    b = _as_block(block)
    _validate_block(b, len(sequence))
    if direction not in ROTATION_DIRECTIONS:
        raise ValueError(f"direction must be one of {ROTATION_DIRECTIONS}")
    if isinstance(amount, bool) or not isinstance(amount, (int, np.integer)) or amount < 0:
        raise ValueError("rotation amount must be a non-negative integer")
    r = int(amount) % b.length
    if r == 0:
        return sequence
    cassette = sequence[b.start:b.end]
    if direction == "left":
        rotated = cassette[r:] + cassette[:r]
    else:
        rotated = cassette[-r:] + cassette[:-r]
    return sequence[:b.start] + rotated + sequence[b.end:]


def deterministic_rotations(
    sequence: str,
    block: ContextBlock | tuple[int, int],
    amounts: Iterable[int],
    directions: Iterable[str] = ROTATION_DIRECTIONS,
) -> tuple[RotationVariant, ...]:
    """Build rotations in a hash-seed-independent, canonical order.

    Amounts are sorted numerically and de-duplicated.  Directions always follow the frozen
    ``left, right`` order, regardless of whether callers pass a list or a set.
    """

    b = _as_block(block)
    _validate_block(b, len(sequence))
    clean_amounts: set[int] = set()
    for amount in amounts:
        if isinstance(amount, bool) or not isinstance(amount, (int, np.integer)) or amount < 0:
            raise ValueError("rotation amounts must be non-negative integers")
        clean_amounts.add(int(amount))
    requested_directions = set(directions)
    invalid = requested_directions.difference(ROTATION_DIRECTIONS)
    if invalid:
        raise ValueError(f"invalid rotation direction(s): {sorted(invalid)}")
    return tuple(
        RotationVariant(amount, direction, b, rotate_cyclic(sequence, b, amount, direction))
        for amount in sorted(clean_amounts)
        for direction in ROTATION_DIRECTIONS
        if direction in requested_directions
    )


def block_candidates(
    span_start: int,
    span_end: int,
    *,
    block_size: int = 60,
    left_clearance: int = 0,
    right_clearance: int = 0,
    end_phase_origin: int | None = None,
    period: int = 3,
    forbidden: Iterable[ContextBlock | tuple[int, int]] = (),
    context: str = "",
) -> tuple[ContextBlock, ...]:
    """Enumerate eligible blocks inside an annotated context span.

    If ``end_phase_origin`` is provided, block ends must satisfy
    ``(end - end_phase_origin) % period == 0``.  For CDS cassettes this pins the downstream edge to
    an annotated codon boundary.  Returned candidates are sorted by coordinate.
    """

    if span_start < 0 or span_end <= span_start:
        raise ValueError("invalid context span")
    if block_size <= 0 or left_clearance < 0 or right_clearance < 0:
        raise ValueError("block size must be positive and clearances non-negative")
    if period <= 0:
        raise ValueError("period must be positive")
    lo, hi = span_start + left_clearance, span_end - right_clearance
    forbidden_blocks = tuple(_as_block(x) for x in forbidden)
    out = []
    for start in range(lo, hi - block_size + 1):
        candidate = ContextBlock(start, start + block_size, context)
        if end_phase_origin is not None and (candidate.end - end_phase_origin) % period:
            continue
        if any(intervals_overlap(candidate, x) for x in forbidden_blocks):
            continue
        out.append(candidate)
    return tuple(out)


def select_spread_blocks(
    candidates: Iterable[ContextBlock],
    *,
    n: int = 2,
) -> tuple[ContextBlock, ...]:
    """Select ``n`` coordinate-spread candidates with deterministic tie-breaking.

    The endpoints are selected when ``n=2``.  More generally, integer-spaced quantiles cover the
    available span.  Fewer than ``n`` unique candidates is a construction failure.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    ordered = tuple(sorted(set(candidates), key=lambda b: (b.start, b.end, b.context)))
    if len(ordered) < n:
        raise ValueError(f"need at least {n} block candidates, found {len(ordered)}")
    if n == 1:
        return (ordered[(len(ordered) - 1) // 2],)
    denominator = n - 1
    indices = tuple((i * (len(ordered) - 1) + denominator // 2) // denominator for i in range(n))
    selected = tuple(ordered[i] for i in indices)
    if len(set(selected)) != n:  # Defensive: the formula is unique whenever len(ordered) >= n.
        raise AssertionError("spread selection produced duplicate blocks")
    return selected


def select_cds_blocks(
    cds_start: int,
    donor_pos: int,
    *,
    codon_boundary_origin: int,
    block_size: int = 60,
    donor_clearance: int = 24,
    n: int = 2,
    forbidden: Iterable[ContextBlock | tuple[int, int]] = (),
) -> tuple[ContextBlock, ...]:
    """Select coding cassettes ending on codon boundaries upstream of a donor."""

    candidates = block_candidates(
        cds_start,
        donor_pos,
        block_size=block_size,
        right_clearance=donor_clearance,
        end_phase_origin=codon_boundary_origin,
        forbidden=forbidden,
        context="cds",
    )
    return select_spread_blocks(candidates, n=n)


def select_upstream_intron_blocks(
    intron_start: int,
    acceptor_pos: int,
    *,
    block_size: int = 60,
    acceptor_clearance: int = 18,
    n: int = 2,
    forbidden: Iterable[ContextBlock | tuple[int, int]] = (),
) -> tuple[ContextBlock, ...]:
    """Select sham cassettes wholly upstream of and clear of the current-exon acceptor."""

    candidates = block_candidates(
        intron_start,
        acceptor_pos,
        block_size=block_size,
        right_clearance=acceptor_clearance,
        forbidden=forbidden,
        context="upstream_intron",
    )
    return select_spread_blocks(candidates, n=n)


def motif_counts(sequence: str, motifs: Sequence[str] = CANONICAL_SPLICE_MOTIFS) -> dict[str, int]:
    """Count overlapping motif occurrences in a sequence."""

    return {
        motif: sum(sequence[i:i + len(motif)] == motif for i in range(len(sequence) - len(motif) + 1))
        for motif in motifs
    }


def in_frame_stop_positions(
    sequence: str,
    start: int,
    end: int,
    *,
    frame_origin: int,
) -> tuple[int, ...]:
    """Return in-frame stop starts in a strand-oriented CDS interval."""

    if not (0 <= start <= end <= len(sequence)):
        raise ValueError("invalid CDS scan interval")
    first = start + ((frame_origin - start) % 3)
    return tuple(i for i in range(first, end - 2, 3) if sequence[i:i + 3] in STOP_CODONS)


def validate_rotation_gates(
    original: str,
    variant: str,
    block: ContextBlock | tuple[int, int],
    *,
    protected_intervals: Iterable[ContextBlock | tuple[int, int]] = (),
    expected_length: int | None = 1024,
    motifs: Sequence[str] = CANONICAL_SPLICE_MOTIFS,
    cds_scan: tuple[int, int, int] | None = None,
) -> RotationGateResult:
    """Evaluate the pure sequence gates for a proposed rotation.

    ``cds_scan`` is ``(start, end, frame_origin)`` in the strand-oriented model sequence.  Both the
    native and edited intervals must be stop-free, matching the preregistered primary domain.
    MaxEnt/annotation safety gates require external pinned data and intentionally remain in the
    package builder rather than being approximated here.
    """

    b = _as_block(block)
    _validate_block(b, len(original))
    protected = tuple(_as_block(x) for x in protected_intervals)
    for interval in protected:
        _validate_block(interval, len(original))

    length_equal = len(original) == len(variant)
    expected_length_ok = expected_length is None or (
        len(original) == expected_length and len(variant) == expected_length
    )
    variant_block = variant[b.start:b.end] if b.end <= len(variant) else ""
    original_block = original[b.start:b.end]
    block_composition_equal = Counter(original_block) == Counter(variant_block)
    global_composition_equal = Counter(original) == Counter(variant)
    only_block_changed = length_equal and (
        original[:b.start] == variant[:b.start] and original[b.end:] == variant[b.end:]
    )
    block_clear_of_protected = not any(intervals_overlap(b, x) for x in protected)
    protected_unchanged = length_equal and all(
        original[x.start:x.end] == variant[x.start:x.end] for x in protected
    )

    before_motifs = motif_counts(original, motifs)
    after_motifs = motif_counts(variant, motifs)
    motif_nonincrease = all(after_motifs[m] <= before_motifs[m] for m in motifs)

    checks: dict[str, bool] = {
        "length_equal": length_equal,
        "expected_length": expected_length_ok,
        "only_selected_block_changed": only_block_changed,
        "block_composition_exact": block_composition_equal,
        "global_composition_exact": global_composition_equal,
        "block_clear_of_protected_sites": block_clear_of_protected,
        "protected_sites_unchanged": protected_unchanged,
        "canonical_motif_counts_nonincreasing": motif_nonincrease,
    }
    before_stops: tuple[int, ...] = ()
    after_stops: tuple[int, ...] = ()
    if cds_scan is not None:
        scan_start, scan_end, frame_origin = cds_scan
        before_stops = in_frame_stop_positions(
            original, scan_start, scan_end, frame_origin=frame_origin
        )
        after_stops = in_frame_stop_positions(
            variant, scan_start, scan_end, frame_origin=frame_origin
        )
        checks["native_cds_stop_free"] = not before_stops
        checks["edited_cds_stop_free"] = not after_stops

    metrics: dict[str, object] = {
        "length_before": len(original),
        "length_after": len(variant),
        "block_hamming": sum(a != z for a, z in zip(original_block, variant_block)),
        "motif_counts_before": before_motifs,
        "motif_counts_after": after_motifs,
        "native_stop_positions": before_stops,
        "edited_stop_positions": after_stops,
    }
    return RotationGateResult(checks=checks, metrics=metrics)


def site_excluded_phasor(
    attention: Sequence[float] | np.ndarray,
    donor_pos: int,
    acceptor_pos: int,
    *,
    site_radius: int = 6,
    reference_class: int = 1,
) -> PhasorResult:
    """Compute the signed acceptor-relative period-3 phasor from an L3H1 attention row.

    Only keys strictly inside the donor--acceptor span are used.  Keys whose coordinate lies
    within ``site_radius`` (inclusive) of either endpoint are excluded.  The complex convention is
    ``exp(+2*pi*i*(x-acceptor)/3)``.  ``signed_projection`` is its real projection onto the frozen
    acceptor-relative residue class (class 1 by default), rather than the upward-biased magnitude.
    """

    row = np.asarray(attention, dtype=float)
    if row.ndim != 1:
        raise ValueError("attention must be a one-dimensional key row")
    if not np.isfinite(row).all() or np.any(row < 0):
        raise ValueError("attention must be finite and non-negative")
    if not (0 <= donor_pos < row.size and 0 <= acceptor_pos < row.size):
        raise ValueError("site coordinate outside attention row")
    if donor_pos == acceptor_pos or site_radius < 0:
        raise ValueError("sites must differ and site_radius must be non-negative")
    if reference_class not in (0, 1, 2):
        raise ValueError("reference_class must be 0, 1, or 2")

    lo, hi = sorted((donor_pos, acceptor_pos))
    keys = np.arange(lo + 1, hi, dtype=int)
    keep = (np.abs(keys - donor_pos) > site_radius) & (
        np.abs(keys - acceptor_pos) > site_radius
    )
    keys = keys[keep]
    if keys.size == 0:
        raise ValueError("no site-excluded keys remain in the donor--acceptor interior")
    weights = row[keys]
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("site-excluded attention mass is zero")

    residues = (keys - acceptor_pos) % 3
    phases = np.exp(2j * np.pi * residues / 3.0)
    value = complex(np.sum(weights * phases) / total)
    reference = complex(np.exp(2j * np.pi * reference_class / 3.0))
    signed = float(np.real(value * np.conjugate(reference)))
    class_masses = tuple(float(weights[residues == r].sum() / total) for r in range(3))
    return PhasorResult(
        value=value,
        signed_projection=signed,
        amplitude=float(abs(value)),
        phase_angle=float(np.angle(value)),
        included_mass=total,
        n_keys=int(keys.size),
        class_masses=class_masses,  # type: ignore[arg-type]
    )


def local_recovery_contrast(
    values: Mapping[int, float],
    *,
    centers: Sequence[int] = (3, 6, 9),
) -> float:
    """Return the locally symmetric modulo-3 recovery contrast ``S(Y)``.

    For each center ``c``, this computes ``Y[c] - (Y[c-1] + Y[c+1]) / 2`` and averages
    centers.  The default is the preregistered 3/6/9 contrast and is exactly zero for a linear
    rotation-amount trend.
    """

    if not centers:
        raise ValueError("at least one recovery center is required")
    needed = {x for c in centers for x in (c - 1, c, c + 1)}
    missing = sorted(needed.difference(values))
    if missing:
        raise ValueError(f"missing rotation amounts: {missing}")
    terms = []
    for center in centers:
        triple = (float(values[center - 1]), float(values[center]), float(values[center + 1]))
        if not all(math.isfinite(x) for x in triple):
            raise ValueError("recovery inputs must be finite")
        terms.append(triple[1] - (triple[0] + triple[2]) / 2.0)
    return float(np.mean(terms))


def stable_gene_bucket(gene_id: str, *, namespace: str, n_buckets: int = 10_000) -> int:
    """Map a gene to a stable SHA-256 bucket, independent of ``PYTHONHASHSEED``."""

    if not gene_id or not namespace:
        raise ValueError("gene_id and namespace must be non-empty")
    if n_buckets <= 0:
        raise ValueError("n_buckets must be positive")
    payload = f"gfmcd-gene-role-v1\0{namespace}\0{gene_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % n_buckets


def stable_gene_role(
    gene_id: str,
    *,
    namespace: str,
    roles: Sequence[tuple[str, int]],
) -> str:
    """Assign a gene to a predeclared weighted role using stable hashing.

    ``roles`` is an ordered sequence of ``(name, positive_integer_weight)`` pairs.  Ordering is
    intentionally explicit: changing it changes the assignment and therefore requires a new
    package/namespace.
    """

    if not roles:
        raise ValueError("at least one role is required")
    names = [name for name, _ in roles]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("role names must be non-empty and unique")
    if any(isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0
           for _, weight in roles):
        raise ValueError("role weights must be positive integers")
    total = sum(weight for _, weight in roles)
    bucket = stable_gene_bucket(gene_id, namespace=namespace, n_buckets=total)
    cumulative = 0
    for name, weight in roles:
        cumulative += weight
        if bucket < cumulative:
            return name
    raise AssertionError("unreachable role bucket")


def aggregate_gene_conditions(
    rows: Iterable[Mapping[str, object]],
    *,
    condition_keys: Sequence[str],
    value_key: str,
    gene_key: str = "gene_id",
) -> tuple[dict[str, object], ...]:
    """Average repeated edits within gene and condition in stable key order."""

    grouped: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        gene = str(row[gene_key])
        condition = tuple(row[key] for key in condition_keys)
        value = float(row[value_key])
        if not gene or not math.isfinite(value):
            raise ValueError("gene IDs must be non-empty and values finite")
        grouped[(gene, *condition)].append(value)
    out = []
    for key in sorted(grouped, key=lambda x: tuple(str(v) for v in x)):
        gene, *condition = key
        item: dict[str, object] = {gene_key: gene}
        item.update(zip(condition_keys, condition))
        item[value_key] = float(np.mean(grouped[key]))
        item["n_edits"] = len(grouped[key])
        out.append(item)
    return tuple(out)


def gene_first_phase_contrasts(
    rows: Iterable[Mapping[str, object]],
    *,
    value_key: str = "value",
    amount_key: str = "rotation_amount",
    gene_key: str = "gene_id",
    centers: Sequence[int] = (3, 6, 9),
) -> dict[str, float]:
    """Average edits within gene/amount, then calculate each gene's phase contrast."""

    aggregated = aggregate_gene_conditions(
        rows, condition_keys=(amount_key,), value_key=value_key, gene_key=gene_key
    )
    per_gene: dict[str, dict[int, float]] = defaultdict(dict)
    for row in aggregated:
        per_gene[str(row[gene_key])][int(row[amount_key])] = float(row[value_key])
    return {
        gene: local_recovery_contrast(values, centers=centers)
        for gene, values in sorted(per_gene.items())
    }


def gene_first_paired_phase_contrasts(
    rows: Iterable[Mapping[str, object]],
    *,
    target_arm: str,
    control_arm: str,
    arm_key: str = "context",
    value_key: str = "value",
    amount_key: str = "rotation_amount",
    gene_key: str = "gene_id",
    centers: Sequence[int] = (3, 6, 9),
) -> dict[str, float]:
    """Return gene-level target-minus-control phase contrasts for a paired design."""

    materialized = tuple(rows)
    target = gene_first_phase_contrasts(
        (r for r in materialized if r[arm_key] == target_arm),
        value_key=value_key,
        amount_key=amount_key,
        gene_key=gene_key,
        centers=centers,
    )
    control = gene_first_phase_contrasts(
        (r for r in materialized if r[arm_key] == control_arm),
        value_key=value_key,
        amount_key=amount_key,
        gene_key=gene_key,
        centers=centers,
    )
    if set(target) != set(control):
        target_only = sorted(set(target).difference(control))
        control_only = sorted(set(control).difference(target))
        raise ValueError(
            f"unpaired gene arms; target_only={target_only}, control_only={control_only}"
        )
    return {gene: target[gene] - control[gene] for gene in sorted(target)}


__all__ = [
    "CANONICAL_SPLICE_MOTIFS",
    "ContextBlock",
    "PhasorResult",
    "ROTATION_DIRECTIONS",
    "RotationGateResult",
    "RotationVariant",
    "aggregate_gene_conditions",
    "block_candidates",
    "deterministic_rotations",
    "gene_first_paired_phase_contrasts",
    "gene_first_phase_contrasts",
    "in_frame_stop_positions",
    "intervals_overlap",
    "local_recovery_contrast",
    "motif_counts",
    "rotate_cyclic",
    "select_cds_blocks",
    "select_spread_blocks",
    "select_upstream_intron_blocks",
    "site_excluded_phasor",
    "stable_gene_bucket",
    "stable_gene_role",
    "validate_rotation_gates",
]
