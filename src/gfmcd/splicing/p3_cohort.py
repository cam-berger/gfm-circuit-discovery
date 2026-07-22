"""Outcome-blind cohort construction for the P3 closure experiment."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import itertools
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from gfmcd.splicing.coupling import CouplingExample
from gfmcd.splicing.p3_annotation import (
    AnnotationSafetyIndex,
    validate_annotation_block,
    window_block_to_genomic,
)
from gfmcd.splicing.p3_closure import (
    ContextBlock,
    block_candidates,
    deterministic_rotations,
    local_recovery_contrast,
    select_spread_blocks,
    validate_rotation_gates,
)
from gfmcd.splicing.splice_landscape import ScoreFunction, compare_landscapes


PRIMARY_AMOUNTS = tuple(range(2, 11))
IN_REGISTER = frozenset((3, 6, 9))


@dataclass(frozen=True)
class P3CohortConfig:
    block_size: int = 60
    n_blocks: int = 2
    donor_clearance: int = 24
    acceptor_clearance: int = 18
    site_radius: int = 8
    splice_clearance: int = 8
    rotation_amounts: tuple[int, ...] = PRIMARY_AMOUNTS
    directions: tuple[str, ...] = ("left", "right")
    max_abs_delta_score3: float = 2.0
    max_abs_delta_score5: float = 2.0
    strong_score_threshold: float = 6.0
    max_hamming_arm_imbalance: float = 0.10
    max_trimer_arm_imbalance: float = 0.10
    max_block_candidates_per_context: int = 48
    require_canonical_motif_nonincrease: bool = True
    require_per_variant_maxent_delta: bool = True
    require_no_new_competitive_site: bool = True
    require_strong_site_count_preserved: bool = True
    max_nuisance_score3_contrast: float = 1.0
    max_nuisance_score5_contrast: float = 1.0
    max_nuisance_created_site_contrast: float = 0.10
    max_nuisance_motif_contrast: float = 0.25
    min_block_start_separation: int = 12

    def validate(self) -> None:
        if self.block_size <= 0 or self.block_size % 3:
            raise ValueError("block_size must be a positive multiple of three")
        if self.n_blocks < 2:
            raise ValueError("at least two blocks are required")
        required = set(range(2, 11))
        if set(self.rotation_amounts) != required:
            raise ValueError("primary rotations must be exactly 2..10")
        if set(self.directions) != {"left", "right"}:
            raise ValueError("both rotation directions are required")
        if min(self.max_abs_delta_score3, self.max_abs_delta_score5) < 0:
            raise ValueError("MaxEnt bands must be non-negative")
        if min(self.max_hamming_arm_imbalance, self.max_trimer_arm_imbalance) < 0:
            raise ValueError("balance bands must be non-negative")
        if self.max_block_candidates_per_context < self.n_blocks:
            raise ValueError("block search budget must cover the requested blocks")
        if min(
            self.max_nuisance_score3_contrast,
            self.max_nuisance_score5_contrast,
            self.max_nuisance_created_site_contrast,
            self.max_nuisance_motif_contrast,
        ) < 0:
            raise ValueError("nuisance balance bands must be non-negative")
        if self.min_block_start_separation <= 0:
            raise ValueError("block locations must have positive separation")


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def normalized_hamming(original: str, variant: str, block: ContextBlock) -> float:
    if len(original) != len(variant):
        raise ValueError("Hamming distance requires equal lengths")
    return sum(a != b for a, b in zip(original[block.start:block.end],
                                      variant[block.start:block.end])) / block.length


def normalized_trimer_divergence(original: str, variant: str, block: ContextBlock) -> float:
    """Total-variation distance between overlapping 3-mer spectra in a block."""

    def counts(sequence: str) -> Counter[str]:
        part = sequence[block.start:block.end]
        return Counter(part[i:i + 3] for i in range(len(part) - 2))

    before, after = counts(original), counts(variant)
    total = sum(before.values())
    if total == 0 or total != sum(after.values()):
        raise ValueError("trimer divergence requires blocks of length at least three")
    return sum(abs(before[key] - after[key]) for key in before.keys() | after.keys()) / (2 * total)


def _protected_intervals(example: CouplingExample, radius: int) -> tuple[ContextBlock, ...]:
    sites = (example.acceptor_pos, example.donor_pos, example.next_acceptor_pos)
    return tuple(
        ContextBlock(max(0, int(site) - radius), min(len(example.sequence), int(site) + radius))
        for site in sites if site is not None
    )


def _annotation_safe_candidates(
    example: CouplingExample,
    candidates: Iterable[ContextBlock],
    *,
    safety: AnnotationSafetyIndex,
    context: str,
    splice_clearance: int,
) -> tuple[tuple[ContextBlock, ...], tuple[dict[str, object], ...]]:
    md = example.metadata
    safe = []
    audits = []
    for block in candidates:
        genomic_start, genomic_end = window_block_to_genomic(
            block.start,
            block.end,
            window_start=int(md["genomic_window_start"]),
            window_end=int(md["genomic_window_end"]),
            strand=example.strand,
        )
        gate = validate_annotation_block(
            safety,
            chrom=example.chrom,
            start=genomic_start,
            end=genomic_end,
            strand=example.strand,
            gene_id=example.gene_id,
            context=context,
            splice_clearance=splice_clearance,
        )
        audits.append({
            "block": [block.start, block.end],
            "genomic_block": [genomic_start, genomic_end],
            "checks": dict(gate.checks),
            "metrics": dict(gate.metrics),
        })
        if gate.passed:
            safe.append(block)
    return tuple(safe), tuple(audits)


def annotation_safe_candidate_pools(
    example: CouplingExample,
    *,
    donor_phase: int,
    safety: AnnotationSafetyIndex,
    config: P3CohortConfig,
) -> tuple[dict[str, tuple[ContextBlock, ...]], dict[str, object]]:
    """Enumerate outcome-blind, annotation-safe CDS and intron cassette pools."""

    config.validate()
    if donor_phase not in (0, 1, 2):
        raise ValueError("donor phase must be 0, 1, or 2")
    protected = _protected_intervals(example, config.site_radius)
    codon_origin = example.donor_pos - donor_phase
    cds_candidates = block_candidates(
        example.acceptor_pos,
        example.donor_pos,
        block_size=config.block_size,
        right_clearance=config.donor_clearance,
        end_phase_origin=codon_origin,
        forbidden=protected,
        context="cds",
    )
    intron_candidates = block_candidates(
        0,
        example.acceptor_pos,
        block_size=config.block_size,
        right_clearance=config.acceptor_clearance,
        forbidden=protected,
        context="upstream_intron",
    )
    safe_cds, cds_audit = _annotation_safe_candidates(
        example, cds_candidates, safety=safety, context="cds",
        splice_clearance=config.splice_clearance,
    )
    safe_intron, intron_audit = _annotation_safe_candidates(
        example, intron_candidates, safety=safety, context="upstream_intron",
        splice_clearance=config.splice_clearance,
    )
    pools = {"cds": safe_cds, "upstream_intron": safe_intron}
    return pools, {
        "n_cds_candidates": len(cds_candidates),
        "n_cds_annotation_safe": len(safe_cds),
        "n_intron_candidates": len(intron_candidates),
        "n_intron_annotation_safe": len(safe_intron),
        "cds_candidate_audit": cds_audit,
        "intron_candidate_audit": intron_audit,
    }


def annotation_safe_non_cds_candidate_pools(
    example: CouplingExample,
    *,
    safety: AnnotationSafetyIndex,
    config: P3CohortConfig,
) -> tuple[dict[str, tuple[ContextBlock, ...]], dict[str, object]]:
    """Enumerate exon-local, comprehensive-CDS-free blocks and matched upstream intron blocks."""

    config.validate()
    protected = _protected_intervals(example, config.site_radius)
    exon_candidates = block_candidates(
        example.acceptor_pos,
        example.donor_pos,
        block_size=config.block_size,
        right_clearance=config.donor_clearance,
        forbidden=protected,
        context="non_cds_exon",
    )
    intron_candidates = block_candidates(
        0,
        example.acceptor_pos,
        block_size=config.block_size,
        right_clearance=config.acceptor_clearance,
        forbidden=protected,
        context="upstream_intron",
    )
    safe_exon, exon_audit = _annotation_safe_candidates(
        example,
        exon_candidates,
        safety=safety,
        context="non_cds_exon",
        splice_clearance=config.splice_clearance,
    )
    safe_intron, intron_audit = _annotation_safe_candidates(
        example,
        intron_candidates,
        safety=safety,
        context="upstream_intron",
        splice_clearance=config.splice_clearance,
    )
    pools = {"non_cds_exon": safe_exon, "upstream_intron": safe_intron}
    return pools, {
        "n_non_cds_exon_candidates": len(exon_candidates),
        "n_non_cds_exon_annotation_safe": len(safe_exon),
        "n_intron_candidates": len(intron_candidates),
        "n_intron_annotation_safe": len(safe_intron),
        "non_cds_exon_candidate_audit": exon_audit,
        "intron_candidate_audit": intron_audit,
    }


def choose_example_blocks(
    example: CouplingExample,
    *,
    donor_phase: int,
    safety: AnnotationSafetyIndex,
    config: P3CohortConfig,
) -> tuple[dict[str, tuple[ContextBlock, ...]], dict[str, object]]:
    """Choose two spread annotation-safe blocks without sequence-variant gates.

    Package construction performs the stronger search in :func:`construct_example_package`, where
    candidate blocks must pass *every* rotation before they can be selected.
    """

    pools, audit = annotation_safe_candidate_pools(
        example, donor_phase=donor_phase, safety=safety, config=config
    )
    blocks = {
        context: select_spread_blocks(candidates, n=config.n_blocks)
        for context, candidates in pools.items()
    }
    return blocks, audit


def _arm_imbalance(rows: Sequence[Mapping[str, object]], metric: str) -> float:
    in_register = [float(row[metric]) for row in rows if int(row["amount"]) in IN_REGISTER]
    off_register = [float(row[metric]) for row in rows if int(row["amount"]) not in IN_REGISTER]
    if not in_register or not off_register:
        raise ValueError("both register arms are required")
    return abs(float(np.mean(in_register)) - float(np.mean(off_register)))


NUISANCE_METRICS = (
    "hamming",
    "trimer_divergence",
    "delta_max_score3",
    "delta_max_score5",
    "created_competitive_score3",
    "created_competitive_score5",
    "canonical_motif_increase",
)


def _block_phase_nuisance(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Apply the exact 3/6/9 statistic to sequence-only nuisance variables."""

    out = {}
    for metric in NUISANCE_METRICS:
        values: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            values[int(row["amount"])].append(float(row[metric]))
        out[metric] = local_recovery_contrast({
            amount: float(np.mean(items)) for amount, items in values.items()
        })
    return out


def _separated(blocks: Sequence[ContextBlock], minimum: int) -> bool:
    return all(
        abs(left.start - right.start) >= minimum
        for left, right in itertools.combinations(blocks, 2)
    )


def _select_nuisance_matched_blocks(
    rows_by_context: Mapping[str, Mapping[ContextBlock, Sequence[Mapping[str, object]]]],
    *,
    config: P3CohortConfig,
) -> tuple[dict[str, tuple[ContextBlock, ...]], dict[str, object]]:
    """Jointly select CDS/intron blocks whose nuisance variables have no modulo-3 pattern."""

    scales = {
        "hamming": config.max_hamming_arm_imbalance,
        "trimer_divergence": config.max_trimer_arm_imbalance,
        "delta_max_score3": config.max_nuisance_score3_contrast,
        "delta_max_score5": config.max_nuisance_score5_contrast,
        "created_competitive_score3": config.max_nuisance_created_site_contrast,
        "created_competitive_score5": config.max_nuisance_created_site_contrast,
        "canonical_motif_increase": config.max_nuisance_motif_contrast,
    }
    if any(value <= 0 for value in scales.values()):
        raise ValueError("nuisance matching scales must be positive")

    candidates = {}
    for context, block_rows in rows_by_context.items():
        entries = []
        block_nuisance = {
            block: _block_phase_nuisance(tuple(rows)) for block, rows in block_rows.items()
        }
        for blocks in itertools.combinations(sorted(block_rows), config.n_blocks):
            if not _separated(blocks, config.min_block_start_separation):
                continue
            vector = {
                metric: float(np.mean([block_nuisance[block][metric] for block in blocks]))
                for metric in NUISANCE_METRICS
            }
            within = max(abs(vector[metric]) / scales[metric] for metric in NUISANCE_METRICS)
            span = max(block.end for block in blocks) - min(block.start for block in blocks)
            entries.append((within, -span, tuple((b.start, b.end) for b in blocks), blocks, vector))
        if not entries:
            raise ValueError(
                f"no {context} block pairs meet start-separation="
                f"{config.min_block_start_separation}"
            )
        # The top 64 within-context pairs bound the cross-product deterministically.
        candidates[context] = sorted(entries)[:64]

    best = None
    for cds in candidates["cds"]:
        for intron in candidates["upstream_intron"]:
            cross = {
                metric: cds[4][metric] - intron[4][metric] for metric in NUISANCE_METRICS
            }
            cross_score = max(abs(cross[metric]) / scales[metric] for metric in NUISANCE_METRICS)
            objective = max(cds[0], intron[0], cross_score)
            key = (objective, cds[0] + intron[0] + cross_score, cds[1], intron[1], cds[2], intron[2])
            if best is None or key < best[0]:
                best = (key, cds, intron, cross)
    assert best is not None
    _key, cds, intron, cross = best
    selected = {"cds": cds[3], "upstream_intron": intron[3]}
    diagnostics = {
        "thresholds": scales,
        "cds_phase_nuisance": cds[4],
        "upstream_intron_phase_nuisance": intron[4],
        "paired_phase_nuisance": cross,
        "normalized_max_imbalance": best[0][0],
        "passed": best[0][0] <= 1.0,
        "n_candidate_pairs": {context: len(items) for context, items in candidates.items()},
    }
    return selected, diagnostics


def construct_example_package(
    example: CouplingExample,
    *,
    donor_phase: int,
    safety: AnnotationSafetyIndex,
    score3: ScoreFunction,
    score5: ScoreFunction,
    config: P3CohortConfig,
) -> dict[str, object]:
    """Construct and audit every primary sequence without reading model outputs."""

    pools, annotation_audit = annotation_safe_candidate_pools(
        example, donor_phase=donor_phase, safety=safety, config=config
    )
    protected = _protected_intervals(example, config.site_radius)
    selected_rows: dict[str, dict[ContextBlock, list[dict[str, object]]]] = {
        "cds": {}, "upstream_intron": {}
    }
    search_audit: dict[str, object] = {}
    for context in ("cds", "upstream_intron"):
        candidates = pools[context]
        if len(candidates) < config.n_blocks:
            raise ValueError(
                f"need at least {config.n_blocks} block candidates, found {len(candidates)}"
            )
        if len(candidates) > config.max_block_candidates_per_context:
            candidates = select_spread_blocks(
                candidates, n=config.max_block_candidates_per_context
            )
        valid_blocks = []
        block_failures: dict[str, int] = defaultdict(int)
        for block in candidates:
            block_rows = []
            block_passed = True
            for item in deterministic_rotations(
                example.sequence, block, config.rotation_amounts, config.directions
            ):
                cds_scan = None
                if context == "cds":
                    cds_scan = (block.start, example.donor_pos, example.donor_pos - donor_phase)
                pure = validate_rotation_gates(
                    example.sequence,
                    item.sequence,
                    block,
                    protected_intervals=protected,
                    expected_length=len(example.sequence),
                    cds_scan=cds_scan,
                )
                landscape3 = compare_landscapes(
                    example.sequence, item.sequence, block.start, block.end,
                    window_length=23, scorer=score3,
                    strong_threshold=config.strong_score_threshold,
                )
                landscape5 = compare_landscapes(
                    example.sequence, item.sequence, block.start, block.end,
                    window_length=9, scorer=score5,
                    strong_threshold=config.strong_score_threshold,
                )
                landscape_checks = {
                    "score3_delta_bounded": abs(landscape3.delta_max) <= config.max_abs_delta_score3,
                    "score5_delta_bounded": abs(landscape5.delta_max) <= config.max_abs_delta_score5,
                    "no_new_competitive_score3": not landscape3.new_competitive_site,
                    "no_new_competitive_score5": not landscape5.new_competitive_site,
                    "strong_score3_count_preserved": landscape3.delta_n_strong == 0,
                    "strong_score5_count_preserved": landscape5.delta_n_strong == 0,
                }
                required_pure = all(
                    ok for name, ok in pure.checks.items()
                    if config.require_canonical_motif_nonincrease
                    or name != "canonical_motif_counts_nonincreasing"
                )
                required_landscape = []
                if config.require_per_variant_maxent_delta:
                    required_landscape.extend((
                        landscape_checks["score3_delta_bounded"],
                        landscape_checks["score5_delta_bounded"],
                    ))
                if config.require_no_new_competitive_site:
                    required_landscape.extend((
                        landscape_checks["no_new_competitive_score3"],
                        landscape_checks["no_new_competitive_score5"],
                    ))
                if config.require_strong_site_count_preserved:
                    required_landscape.extend((
                        landscape_checks["strong_score3_count_preserved"],
                        landscape_checks["strong_score5_count_preserved"],
                    ))
                passed = required_pure and all(required_landscape)
                block_passed &= passed
                motif_before = pure.metrics["motif_counts_before"]
                motif_after = pure.metrics["motif_counts_after"]
                row = {
                    "context": context,
                    "block": [block.start, block.end],
                    "amount": item.amount,
                    "direction": item.direction,
                    "sequence_sha256": sequence_sha256(item.sequence),
                    "hamming": normalized_hamming(example.sequence, item.sequence, block),
                    "trimer_divergence": normalized_trimer_divergence(
                        example.sequence, item.sequence, block
                    ),
                    "pure_checks": dict(pure.checks),
                    "landscape_checks": landscape_checks,
                    "delta_max_score3": landscape3.delta_max,
                    "delta_max_score5": landscape5.delta_max,
                    "max_score3_before": landscape3.before.max_score,
                    "max_score3_after": landscape3.after.max_score,
                    "max_score5_before": landscape5.before.max_score,
                    "max_score5_after": landscape5.after.max_score,
                    "created_competitive_score3": float(landscape3.new_competitive_site),
                    "created_competitive_score5": float(landscape5.new_competitive_site),
                    "canonical_motif_increase": float(sum(
                        max(0, int(motif_after[motif]) - int(motif_before[motif]))
                        for motif in motif_before
                    )),
                    "delta_n_strong_score3": landscape3.delta_n_strong,
                    "delta_n_strong_score5": landscape5.delta_n_strong,
                    "passed": passed,
                }
                block_rows.append(row)
                if not passed:
                    for name, ok in pure.checks.items():
                        if not ok:
                            block_failures[name] += 1
                    for name, ok in landscape_checks.items():
                        if not ok:
                            block_failures[name] += 1
            if block_passed:
                valid_blocks.append(block)
                selected_rows[context][block] = block_rows
        if len(valid_blocks) < config.n_blocks:
            raise ValueError(
                f"need at least {config.n_blocks} all-rotation-safe {context} blocks, "
                f"found {len(valid_blocks)} of {len(candidates)} searched; "
                f"failures={dict(sorted(block_failures.items()))}"
            )
        search_audit[context] = {
            "n_annotation_safe": len(pools[context]),
            "n_searched": len(candidates),
            "n_all_rotation_safe": len(valid_blocks),
            "failed_gate_counts": dict(sorted(block_failures.items())),
        }

    blocks, nuisance_match = _select_nuisance_matched_blocks(selected_rows, config=config)
    selected_rows = {
        context: {block: selected_rows[context][block] for block in chosen}
        for context, chosen in blocks.items()
    }
    variants = []
    for context in ("cds", "upstream_intron"):
        for block_index, block in enumerate(blocks[context]):
            for row in selected_rows[context][block]:
                row["block_index"] = block_index
                variants.append(row)

    balance = {}
    balance_pass = bool(nuisance_match["passed"])
    for context in ("cds", "upstream_intron"):
        arm = [row for row in variants if row["context"] == context]
        hamming_imbalance = _arm_imbalance(arm, "hamming")
        trimer_imbalance = _arm_imbalance(arm, "trimer_divergence")
        checks = {
            "hamming_arm_balanced": hamming_imbalance <= config.max_hamming_arm_imbalance,
            "trimer_arm_balanced": trimer_imbalance <= config.max_trimer_arm_imbalance,
        }
        balance_pass &= all(checks.values())
        balance[context] = {
            "hamming_imbalance": hamming_imbalance,
            "trimer_imbalance": trimer_imbalance,
            "checks": checks,
        }

    return {
        "example_id": example.example_id,
        "gene_id": example.gene_id,
        "chrom": example.chrom,
        "strand": example.strand,
        "donor_phase": donor_phase,
        "native_sequence_sha256": sequence_sha256(example.sequence),
        "acceptor_pos": example.acceptor_pos,
        "donor_pos": example.donor_pos,
        "next_acceptor_pos": example.next_acceptor_pos,
        "exon_length": example.exon_length,
        "donor_partner_distance": int(example.next_acceptor_pos - example.donor_pos),
        "n_isoforms_donor": example.n_isoforms_donor,
        "n_isoforms_acceptor": example.n_isoforms_acceptor,
        "annotation_audit": annotation_audit,
        "block_search_audit": search_audit,
        "nuisance_match": nuisance_match,
        "blocks": {
            context: [[block.start, block.end] for block in selected]
            for context, selected in blocks.items()
        },
        "variants": variants,
        "balance": balance,
        "eligible": bool(balance_pass),
        "config": asdict(config),
    }


def construction_failure_reason(exc: Exception) -> str:
    text = str(exc).lower()
    if "block candidates" in text:
        return "insufficient_annotation_safe_blocks"
    if "maxent" in text or "landscape" in text:
        return "splice_landscape_failure"
    if "phase" in text:
        return "phase_failure"
    return f"construction_error:{type(exc).__name__}"


__all__ = [
    "IN_REGISTER",
    "P3CohortConfig",
    "PRIMARY_AMOUNTS",
    "annotation_safe_candidate_pools",
    "choose_example_blocks",
    "construct_example_package",
    "construction_failure_reason",
    "normalized_hamming",
    "normalized_trimer_divergence",
    "sequence_sha256",
]
