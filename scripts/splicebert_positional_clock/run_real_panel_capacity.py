#!/usr/bin/env python
"""Real-panel authentic-module capacity and power, memorialized against measured components.

This artifact exists so that a later pivot to a different evaluation design cannot silently
supersede the finding recorded here: **the authentic-module design cannot be built at the frozen
SESOI**, using the real gene pool and the real (not borrowed) variance components. If a future
runner evaluates a different lattice, a different SESOI, or a relaxed filter set and reaches a
different verdict, that is a new design decision to be recorded as such -- not a silent
replacement of this one.

Stage: ``construction``. Same hard constraints as the sibling ladder runner
(``run_pbc_v2_capacity.py``): this module never imports ``torch``, never imports a model adapter,
never loads a SpliceBERT checkpoint, performs NO cohort selection, NO block assembly, and NO cell
materialisation. The output carries counts, hashes, verdicts and projections only -- never a gene
ID and never a per-gene-effect value.

Five sections, matching the task-6 brief's B1-B5:

* **B1** -- measured variance components (pooled within-block within-group variance, across-block
  Var(Psi_b), block SD, sigma_u2, rho) recomputed from the 3,456 `C43` replication cell rows.
  **Sign-free**: this runner never reads, computes, prints, or stores a mean, a sign, or a
  direction of any `C43` endpoint -- only variances. rho is recorded as a POINT ESTIMATE WITH NO
  INTERVAL, from one cohort, under an approximate exchangeability assumption -- a design
  diagnostic, not a frozen sizing input.
* **B2** -- real-panel ceilings for the plain {0,1} lattice by exact max-flow (never hand-derived):
  the four phase-0/1 module slots' union, the exact ceiling, and a ceiling verified to also clear a
  25% margin on that union. The withdrawn rotated-lattice ceiling is recorded for comparison only.
* **B3** -- power at the MEASURED components (not the PBC-v1 development-role upper bound):
  blocks needed for Psi and M+/M- on both the plain 2x2 and the full 3x3 lattice.
* **B4** -- the cross-lineage (Delta-Psi) reframe: the same-block L1/L2 correlation a paired design
  would need at two candidate N, marked `requires_pilot` -- never a block count, because the
  cross-lineage covariance is not `C43`'s within-block rho and is not estimable from anything here.
* **B5** -- why no filter relaxation is available: a streamed, sole-reason exclusion tally from the
  census, plus the model-free fact that resolved a separate question about donor/acceptor phase
  equality (definitional, merely un-contradicted here -- never "empirically verified").

Four inputs, all read-only:

* the hash-pinned candidate census (shared with ``run_pbc_v2_capacity.py``);
* the committed PBC-v1 capacity certificate (same as above, for the same seven slot hashes);
* ``crossed_direct_edge_replication_cells.json`` (the `C43` replication's 3,456 per-cell rows) --
  UNTRACKED, lives outside this repo, treated exactly like the census: never copied in, and this
  runner exits cleanly with a stated reason if it is absent rather than crashing;
* ``crossed_direct_edge_replication_analysis.json`` (tracked, in this repo), whose
  ``cells_result_sha256`` field is the provenance binding this runner verifies the cells file
  against before reading a single row from it.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.pbc_v2_capacity import (  # noqa: E402
    SESOI,
    SLOT_ORDER,
    _Z,
    feasibility,
    rotated_demands,
    slot_gene_sets,
    stream_candidates,
    stream_exclusions,
    verify_slot_hashes,
)
from gfmcd.splicing.positional_boundary_coherence import (  # noqa: E402
    DEFAULT_ACCEPTOR_WINDOW,
    DEFAULT_DONOR_WINDOW,
    normalize_chromosome,
)

# ---------------------------------------------------------------------------------------------
# Fixed design constants
# ---------------------------------------------------------------------------------------------

SCHEMA = "gfmcd.splicebert_positional_clock.real_panel_capacity.v1"
EXPERIMENT_ID = "splicebert-positional-clock:real-panel-capacity-v1"

DEFAULT_CENSUS_PATH = Path(
    "/Users/cberg/Research/projects/gfm-circuit-discovery-pbc-successor/results/"
    "splicebert_positional_boundary_coherence/development/candidate_census_v1.json"
)
DEFAULT_CERTIFICATE_PATH = (
    REPO / "results/splicebert_positional_boundary_coherence/development/composition_balanced_cohort_v1.json"
)
# Untracked, outside this repo -- see this module's docstring. The analysis.json that carries the
# provenance-binding sha256 IS tracked, inside this repo.
DEFAULT_CELLS_PATH = Path(
    "/Users/cberg/Research/projects/gfm-circuit-discovery/results/"
    "splicebert_exon_edge_compatibility/crossed_direct_edge_replication_cells.json"
)
DEFAULT_ANALYSIS_PATH = (
    REPO / "results/splicebert_exon_edge_compatibility/crossed_direct_edge_replication_analysis.json"
)
_CELLS_SHA256_PIN = "428ad5d6b357bfba287fcfe3e358b00ab91ddcd748118b83b60b626593c440bf"

MARGIN_THRESHOLD = 0.25

# ---------------------------------------------------------------------------------------------
# B1 -- measured variance components from the C43 replication cell rows. SIGN-FREE: only
# variances are computed; no mean, sign, or direction of any C43 endpoint is read anywhere below.
# ---------------------------------------------------------------------------------------------

_B1_EXPECTED_N_CELL_ROWS = 3456
_B1_EXPECTED_N_BLOCKS = 384
_B1_EXPECTED_DF = 2688
_B1_EXPECTED = {
    "pooled_within_block_within_group_variance": 0.177039825,
    "across_block_variance_psi_b": 0.140121686,
    "block_sd": 0.374328,
    "sigma_u2": 0.051601774,
}
_B1_EXPECTED_RHO_DISPLAY = 0.3683


def variance_components_from_cell_rows(cell_rows: Iterable[Mapping]) -> dict:
    """Sign-free variance decomposition of Psi_b = mean(matched c_ac) - mean(mismatched c_ac).

    ``matched`` cells are ``compatible: true`` (donor_phase == acceptor_phase, 3 per block);
    ``mismatched`` are ``compatible: false`` (6 per block) -- the frozen 3x3 lattice. Returns the
    pooled within-block within-group variance (df = 384 blocks * (2 + 5) = 2688), the across-block
    Var(Psi_b), its square root, sigma_u2 = Var(Psi_b) - sigma_c2*(1/3 + 1/6), and
    rho = sigma_u2 / Var(Psi_b). Never returns a mean, a sign, or an individual Psi_b -- only the
    aggregate variance quantities above.
    """

    by_block: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"matched": [], "mismatched": []})
    for row in cell_rows:
        group = "matched" if row["compatible"] else "mismatched"
        by_block[row["block_id"]][group].append(float(row["effects"]["c_ac"]))

    total_ss = 0.0
    total_df = 0
    psi_b_values: list[float] = []
    n_cell_rows = 0
    for block_id, groups in by_block.items():
        matched, mismatched = groups["matched"], groups["mismatched"]
        n_cell_rows += len(matched) + len(mismatched)
        if len(matched) != 3 or len(mismatched) != 6:
            raise ValueError(
                f"block {block_id!r} has {len(matched)} matched / {len(mismatched)} mismatched "
                "cells; expected exactly 3/6 (the frozen 3x3 lattice) -- the input has changed "
                "shape since this runner was audited against it"
            )
        matched_mean = sum(matched) / len(matched)
        mismatched_mean = sum(mismatched) / len(mismatched)
        total_ss += sum((x - matched_mean) ** 2 for x in matched)
        total_ss += sum((x - mismatched_mean) ** 2 for x in mismatched)
        total_df += (len(matched) - 1) + (len(mismatched) - 1)
        psi_b_values.append(matched_mean - mismatched_mean)  # used only for its VARIANCE, below

    n_blocks = len(psi_b_values)
    sigma_c2 = total_ss / total_df
    psi_grand_mean = sum(psi_b_values) / n_blocks  # local only: feeds the variance sum, never returned
    var_psi_b = sum((p - psi_grand_mean) ** 2 for p in psi_b_values) / (n_blocks - 1)
    block_sd = var_psi_b ** 0.5
    sigma_u2 = var_psi_b - sigma_c2 * (1.0 / 3.0 + 1.0 / 6.0)
    rho = sigma_u2 / var_psi_b

    return {
        "n_cell_rows": n_cell_rows,
        "n_blocks": n_blocks,
        "pooled_within_block_within_group_variance": sigma_c2,
        "pooled_within_block_within_group_df": total_df,
        "across_block_variance_psi_b": var_psi_b,
        "block_sd": block_sd,
        "sigma_u2": sigma_u2,
        "rho": rho,
    }


def _assert_b1_matches_expected(components: dict) -> None:
    """Hard-fail with NEEDS_CONTEXT if any B1 quantity has drifted from the audited literals.

    Per the task-6 brief: a mismatch means the input changed and this run must stop -- never
    adjust the computation to match a target.
    """

    problems: list[str] = []
    if components["n_cell_rows"] != _B1_EXPECTED_N_CELL_ROWS:
        problems.append(
            f"n_cell_rows: expected {_B1_EXPECTED_N_CELL_ROWS}, observed {components['n_cell_rows']}"
        )
    if components["n_blocks"] != _B1_EXPECTED_N_BLOCKS:
        problems.append(f"n_blocks: expected {_B1_EXPECTED_N_BLOCKS}, observed {components['n_blocks']}")
    if components["pooled_within_block_within_group_df"] != _B1_EXPECTED_DF:
        problems.append(
            f"pooled_within_block_within_group_df: expected {_B1_EXPECTED_DF}, observed "
            f"{components['pooled_within_block_within_group_df']}"
        )
    for field, expected in _B1_EXPECTED.items():
        observed = components[field]
        if abs(observed - expected) > 1e-6 * abs(expected):
            problems.append(f"{field}: expected {expected}, observed {observed}")
    rho_observed = components["rho"]
    if abs(rho_observed - _B1_EXPECTED_RHO_DISPLAY) > 5e-4:
        problems.append(f"rho: expected ~{_B1_EXPECTED_RHO_DISPLAY}, observed {rho_observed}")

    if problems:
        raise SystemExit(
            "NEEDS_CONTEXT: B1 measured variance components do not reproduce the audited "
            "literals -- the cells input has changed since this runner was written against it. "
            "Stop; do not adjust the computation to match. Observed vs expected:\n  "
            + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------------------------
# B2 -- real-panel ceilings for the plain {0,1} lattice, by exact max-flow (never hand-derived).
# ---------------------------------------------------------------------------------------------

PLAIN_LATTICE_PHASE_SLOTS: tuple[str, ...] = (
    "donor_phase_0",
    "donor_phase_1",
    "acceptor_phase_0",
    "acceptor_phase_1",
)

_B2_EXPECTED_UNION4 = 2904
_B2_EXPECTED_EXACT_CEILING = 726
_B2_EXPECTED_INTERSECTIONS = {0: 1869, 1: 1370, 2: 800}
_B2_EXPECTED_ROTATED_CEILING = 647


def plain_lattice_demands(n_blocks: int) -> dict[str, int]:
    """Plain {0,1} lattice: one donor module at phase 0, one at phase 1, one acceptor module at
    phase 0, one at phase 1, and one scaffold, every block -- demand N on each of five slots, no
    rotation (contrast `rotated_demands`, which spends only 2/3 of blocks on each phase).
    """

    demands = {slot: int(n_blocks) for slot in PLAIN_LATTICE_PHASE_SLOTS}
    demands["scaffold"] = int(n_blocks)
    return demands


def _largest_n_satisfying(predicate, upper_bound: int) -> int:
    """Largest non-negative N <= `upper_bound` for which `predicate(N)` holds, by binary search.

    Assumes `predicate` is monotonically non-increasing in N -- true for both exact max-flow
    feasibility and a margin fraction here, since demand only grows with N. Returns 0 if
    `predicate(0)` is already False.
    """

    if not predicate(0):
        return 0
    lo, hi = 0, upper_bound
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if predicate(mid):
            lo = mid
        else:
            hi = mid - 1
    if lo == upper_bound:
        raise ValueError(
            f"_largest_n_satisfying: predicate still holds at upper_bound={upper_bound}; the "
            "search range is too small to have found the true ceiling"
        )
    return lo


def exact_plain_ceiling(slot_sets: Mapping[str, set], *, upper_bound: int = 4000) -> int:
    """Largest N for which the plain lattice's five-slot demand is exactly max-flow feasible."""

    return _largest_n_satisfying(
        lambda n: feasibility(slot_sets, plain_lattice_demands(n))["feasible"], upper_bound
    )


def ceiling_with_assignment_slack(
    slot_sets: Mapping[str, set], *, margin_threshold: float = MARGIN_THRESHOLD, upper_bound: int = 4000
) -> int:
    """Largest N for which the plain lattice is BOTH exactly max-flow feasible AND leaves at least
    `margin_threshold` margin on the four phase-0/1 module slots' OWN union.

    This is the pool's own tightest real constraint for the plain lattice: at N=726 (the exact
    ceiling) every individual slot's own margin is still comfortably above 0.25 -- margin only
    binds when measured against the four-slot union together, which per-slot margin cannot see
    (the same joint-vs-per-slot distinction `feasibility()` exists to catch elsewhere in this
    family). Verified by the real max-flow solver at every candidate N, not inferred by scaling
    the exact ceiling by 0.75.
    """

    union_size = len(set().union(*(slot_sets[slot] for slot in PLAIN_LATTICE_PHASE_SLOTS)))

    def predicate(n: int) -> bool:
        if not feasibility(slot_sets, plain_lattice_demands(n))["feasible"]:
            return False
        if union_size == 0:
            return False
        return (union_size - 4 * n) / union_size >= margin_threshold

    return _largest_n_satisfying(predicate, upper_bound)


def rotated_ceiling(slot_sets: Mapping[str, set], *, upper_bound: int = 4000) -> int:
    """Largest N for which the WITHDRAWN rotated 2x2 lattice's seven-slot demand is exactly
    max-flow feasible -- recorded for comparison only (see the withdrawal note in the artifact).
    """

    def predicate(n: int) -> bool:
        demands = rotated_demands(n)
        real_demands = {slot: demands[slot] for slot in SLOT_ORDER}
        return feasibility(slot_sets, real_demands)["feasible"]

    return _largest_n_satisfying(predicate, upper_bound)


def _assert_b2_matches_expected(union4: int, exact_ceiling: int, intersections: dict, rotated: int) -> None:
    problems: list[str] = []
    if union4 != _B2_EXPECTED_UNION4:
        problems.append(f"union of four phase-0/1 slots: expected {_B2_EXPECTED_UNION4}, observed {union4}")
    if exact_ceiling != _B2_EXPECTED_EXACT_CEILING:
        problems.append(f"exact ceiling: expected {_B2_EXPECTED_EXACT_CEILING}, observed {exact_ceiling}")
    for phase, expected in _B2_EXPECTED_INTERSECTIONS.items():
        observed = intersections[phase]
        if observed != expected:
            problems.append(f"donor/acceptor phase {phase} intersection: expected {expected}, observed {observed}")
    if rotated != _B2_EXPECTED_ROTATED_CEILING:
        problems.append(f"rotated-lattice ceiling: expected {_B2_EXPECTED_ROTATED_CEILING}, observed {rotated}")

    if problems:
        raise SystemExit(
            "NEEDS_CONTEXT: B2 real-panel ceilings do not reproduce the audited literals -- the "
            "gene pool has changed since this runner was written against it. Stop; do not adjust "
            "the computation to match. Observed vs expected:\n  " + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------------------------
# B3 -- power at the MEASURED components (B1's rho and Var(Psi_b)), not the PBC-v1 development-
# role upper bound. Same formula shape as `gfmcd.splicing.pbc_v2_capacity.var_psi`/`var_direction`
# (see A2's docstring there for the module-term caveat this inherits), parametrised on the
# measured inputs instead of the frozen legacy constant.
# ---------------------------------------------------------------------------------------------

_CALIBRATED_MIN_DELTA_CELLS = 3


def _cell_variance_at(rho: float, v9: float) -> float:
    return 2.0 * (1.0 - rho) * v9


def _var_psi_at(matched: int, mismatched: int, rho: float, v9: float) -> float:
    return rho * v9 + _cell_variance_at(rho, v9) * (1.0 / matched + 1.0 / mismatched)


def _var_direction_at(delta_cells: int, matched: int, rho: float, v9: float) -> float:
    return rho * v9 + _cell_variance_at(rho, v9) * (1.0 / delta_cells + 1.0 / matched)


def _blocks_for_sesoi_at(variance: float) -> int:
    """Same MDE80<=SESOI closed form as `gfmcd.splicing.pbc_v2_capacity.blocks_for_sesoi`, reused
    verbatim (alpha=0.05 two-sided, power=0.80) but taking `variance` directly so it can be
    evaluated at the measured (not the legacy) inputs.
    """

    return math.ceil(_Z * _Z * variance / (SESOI ** 2))


def _psi_endpoint_block(blocks_needed: int, *, matched: int, mismatched: int) -> dict:
    return {"matched_cells": matched, "mismatched_cells": mismatched, "blocks_needed": blocks_needed}


def _direction_endpoint_block(blocks_needed: int, *, matched: int, delta_cells: int) -> dict:
    """Same estimate_class/lower_bound_reason honesty fields as A2's `power_projection`
    direction block, evaluated here at the measured rho/V9 instead of the legacy band.
    """

    calibrated = delta_cells >= _CALIBRATED_MIN_DELTA_CELLS
    payload = {
        "matched_cells": matched,
        "delta_cells": delta_cells,
        "blocks_needed": blocks_needed,
        "estimate_class": "calibrated" if calibrated else "lower_bound_only",
    }
    if not calibrated:
        payload["lower_bound_reason"] = (
            "this lattice has fewer than 3 cells per delta class, so the M_delta variance "
            "formula omits the uncancelled donor/acceptor module component -- see A2 / "
            "gfmcd.splicing.pbc_v2_capacity.var_direction's docstring; this is a floor, not a "
            "calibrated point"
        )
    return payload


_B3_EXPECTED = {
    "psi_plain_2x2": 718,
    "m_pm_plain_2x2": 996,
    "psi_full_3x3": 440,
    "m_pm_full_3x3": 533,
}


def power_at_measured_components(rho: float, v9: float) -> dict:
    """B3's four blocks-needed figures at the measured rho/V9 -- point values, not a band (unlike
    `power_projection`'s legacy-variance band over rho in [0, 1]): B1's rho is itself a point
    estimate, so a band over an already-point input would misrepresent it as still unidentified.
    """

    psi_plain = _blocks_for_sesoi_at(_var_psi_at(2, 2, rho, v9))
    m_pm_plain = _blocks_for_sesoi_at(_var_direction_at(1, 2, rho, v9))
    psi_full = _blocks_for_sesoi_at(_var_psi_at(3, 6, rho, v9))
    m_pm_full = _blocks_for_sesoi_at(_var_direction_at(3, 3, rho, v9))

    return {
        "rho_used": rho,
        "v9_used": v9,
        "psi_plain_2x2": _psi_endpoint_block(psi_plain, matched=2, mismatched=2),
        "m_pm_plain_2x2": _direction_endpoint_block(m_pm_plain, matched=2, delta_cells=1),
        "psi_full_3x3": _psi_endpoint_block(psi_full, matched=3, mismatched=6),
        "m_pm_full_3x3": _direction_endpoint_block(m_pm_full, matched=3, delta_cells=3),
    }


def _assert_b3_matches_expected(power: dict) -> None:
    observed = {
        "psi_plain_2x2": power["psi_plain_2x2"]["blocks_needed"],
        "m_pm_plain_2x2": power["m_pm_plain_2x2"]["blocks_needed"],
        "psi_full_3x3": power["psi_full_3x3"]["blocks_needed"],
        "m_pm_full_3x3": power["m_pm_full_3x3"]["blocks_needed"],
    }
    problems = [
        f"{key}: expected {expected}, observed {observed[key]}"
        for key, expected in _B3_EXPECTED.items()
        if observed[key] != expected
    ]
    if problems:
        raise SystemExit(
            "NEEDS_CONTEXT: B3 power at measured components does not reproduce the audited "
            "literals. Stop; do not adjust the computation to match. Observed vs expected:\n  "
            + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------------------------
# B4 -- the cross-lineage (Delta-Psi) reframe: required same-block L1/L2 correlation, not a block
# count. Delta-Psi = Psi_L2 - Psi_L1 on the SAME blocks is paired, so the gene-level component
# largely cancels for a high enough same-block correlation r; this asks how high `r` must be.
# ---------------------------------------------------------------------------------------------

# CORRECTED, and the correction matters. An earlier draft used the raw across-block
# Var(Psi_b) = V9 measured on C43. That is the THREE-BY-THREE value: with m=3 matched and u=6
# mismatched cells, (1/m + 1/u) = 1/2, which makes var_psi(3,6,rho) == V9 for every rho. The real
# transfer panel is the PLAIN {0,1} lattice (m=2, u=2), whose per-block variance is
# (2 - rho)*V9 = 0.228637 -- 1.63x larger. Applying the 3x3 variance to a 2x2 panel understated the
# required correlation as 0.382/0.175 when the correct figures are 0.621/0.494. A same-block L1/L2
# correlation above 0.62 is a materially stronger assumption than one above 0.38, so this is not a
# rounding difference: it changes whether the paired design looks adequate.
_B4_EXPECTED = {544: 0.621, 726: 0.494}


def required_cross_lineage_correlation(n_blocks: int, var_psi_block: float) -> float:
    """Minimum same-block L1/L2 correlation r such that Var(Delta-Psi) = 2*Var(Psi_b)*(1-r)
    reaches MDE80 <= SESOI at alpha=0.05, power=0.80, `n_blocks` blocks -- i.e. the r a paired
    design needs for the SAME MDE80/SESOI gate this whole family uses elsewhere, applied to the
    paired difference instead of a single-checkpoint Psi.

    `var_psi_block` must be the per-block Psi variance OF THE PANEL BEING SIZED, not C43's raw V9.
    """

    max_var = (SESOI ** 2) * n_blocks / (_Z ** 2)
    return 1.0 - max_var / (2.0 * var_psi_block)


def cross_lineage_reframe(
    var_psi_b: float, *, rho: float, usable_ceiling: int, absolute_ceiling: int
) -> dict:
    # The real transfer panel is plain {0,1}: two matched cells, two mismatched.
    var_psi_plain = _var_psi_at(2, 2, rho, var_psi_b)
    rows = {}
    for n_blocks in (usable_ceiling, absolute_ceiling):
        r = required_cross_lineage_correlation(n_blocks, var_psi_plain)
        rows[str(n_blocks)] = {
            "n_blocks": n_blocks,
            "required_same_block_correlation_min": r,
            "status": "requires_pilot",
            "reason": (
                "the cross-lineage (same-block L1/L2) covariance must be estimated by a "
                "seed-level pilot; it is NOT C43's within-block rho (B1) and is not estimable "
                "from anything committed to this record"
            ),
        }
    return {
        "primary_endpoint_note": (
            "single-checkpoint Psi is NOT the primary endpoint of the downstream study; "
            "Delta-Psi = Psi_L2 - Psi_L1 on the same blocks is, and it is paired, so the "
            "gene-level component largely cancels for a high enough same-block correlation"
        ),
        "lattice_used_for_variance": "plain_zero_one",
        "var_psi_per_block_used": var_psi_plain,
        "var_psi_note": (
            "(2 - rho) * V9 for the plain {0,1} lattice (m=2 matched, u=2 mismatched). NOT C43's "
            "raw V9, which is the 3x3 value and would understate the required correlation"
        ),
        "replicate_caveat": (
            "this figure sizes the BLOCK axis only. The replicate for a corpus intervention is the "
            "training seed, not the block; a block-clustered interval on Delta-Psi with one L1 and "
            "one L2 seed is an interval for that lineage pair, not for the intervention"
        ),
        "required_same_block_correlation": rows,
        "delta_psi_blocks_needed": (
            "not emitted -- the cross-lineage covariance is unknown (see each row's reason); a "
            "block count here would be invented"
        ),
    }


def _assert_b4_matches_expected(reframe: dict) -> None:
    problems: list[str] = []
    for n_blocks, expected_r in _B4_EXPECTED.items():
        row = reframe["required_same_block_correlation"][str(n_blocks)]
        observed = row["required_same_block_correlation_min"]
        if abs(observed - expected_r) > 5e-4:
            problems.append(f"N={n_blocks}: expected r>={expected_r}, observed {observed}")
    if problems:
        raise SystemExit(
            "NEEDS_CONTEXT: B4 cross-lineage required-correlation figures do not reproduce the "
            "audited literals. Stop; do not adjust the computation to match. Observed vs "
            "expected:\n  " + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------------------------
# B5 -- why no filter relaxation is available: a streamed, SOLE-reason exclusion tally (an edge
# excluded for exactly one reason), which is what "relax filter X to gain N genes" would actually
# buy -- an edge excluded for several reasons at once is not freed by lifting just one of them.
# ---------------------------------------------------------------------------------------------

_B5_EXPECTED_TOTAL = 282057
_B5_EXPECTED_SOLE_REASONS = {
    "freshness_registry_gene": 84998,
    "comprehensive_local_coding_register_conflict": 81958,
    "module:ValueError:native donor lacks the frozen health-window support": 60353,
    "duplicate_source_locus:scaffold": 17946,
    "comprehensive_local_gene_conflict": 4432,
    "duplicate_source_locus:donor": 1522,
    "module:ValueError:module lacks canonical GT": 739,
    "scaffold:ValueError:module lacks canonical GT": 654,
    "duplicate_source_locus:acceptor": 556,
    "scaffold:ValueError:fixed scaffold bridge lacks AG at frozen boundary 437": 104,
    "module:ValueError:native source motifs do not reconstruct": 3,
    "scaffold:ValueError:scaffold lacks a complete unambiguous 1,022-nt hg38 window": 1,
}
_B5_EXPECTED_DISAGREEMENT_RAW = {
    "comprehensive_coordinate_transcript_phase_conflict": 9778,
    "basic_comprehensive_phase_conflict": 6325,
    "basic_coordinate_transcript_phase_conflict": 3529,
}
_B5_LOAD_BEARING_NOTE = (
    "each of the three largest sole-reason exclusions is load-bearing for a panel that reads "
    "annotated register -- freshness_registry_gene (freshness), "
    "module:ValueError:native donor lacks the frozen health-window support (phase-annotation "
    "safety), and comprehensive_local_coding_register_conflict (the geometry that makes C43 "
    "comparable) -- so none is available to lift supply by relaxing it"
)
_B5_PHASE_EQUALITY_KEY = "upstream_downstream_phase_equal"
_B5_EXPECTED_PHASE_EQUALITY_OCCURRENCES = 130922
_B5_EXPECTED_PHASE_EQUALITY_CANDIDATES = 65461


def exclusion_reason_tallies(exclusions: Iterable[Mapping]) -> dict:
    """Sole-reason and raw (co-occurrence-inclusive) exclusion-reason counts, streamed once.

    `sole_reason_counts` tallies only records whose `reasons` array has exactly one entry -- what
    relaxing that one filter would actually free. `raw_reason_counts` tallies every occurrence of
    every reason regardless of co-occurrence (matching the census's own precomputed
    `exclusion_counts`, recomputed independently here rather than trusted from the input).
    """

    total = 0
    sole_reason_counts: Counter[str] = Counter()
    raw_reason_counts: Counter[str] = Counter()
    for exclusion in exclusions:
        total += 1
        reasons = exclusion["reasons"]
        for reason in reasons:
            raw_reason_counts[reason] += 1
        if len(reasons) == 1:
            sole_reason_counts[reasons[0]] += 1

    return {
        "total": total,
        "sole_reason_counts": dict(sole_reason_counts),
        "raw_reason_counts": dict(raw_reason_counts),
        "multi_reason_total": total - sum(sole_reason_counts.values()),
    }


def _assert_b5_matches_expected(tallies: dict, phase_equality_occurrences: int, phase_equality_candidates: int) -> None:
    problems: list[str] = []
    if tallies["total"] != _B5_EXPECTED_TOTAL:
        problems.append(f"total exclusions: expected {_B5_EXPECTED_TOTAL}, observed {tallies['total']}")
    for reason, expected in _B5_EXPECTED_SOLE_REASONS.items():
        observed = tallies["sole_reason_counts"].get(reason, 0)
        if observed != expected:
            problems.append(f"sole-reason {reason!r}: expected {expected}, observed {observed}")
    for reason, expected in _B5_EXPECTED_DISAGREEMENT_RAW.items():
        observed = tallies["raw_reason_counts"].get(reason, 0)
        if observed != expected:
            problems.append(f"raw-reason {reason!r}: expected {expected}, observed {observed}")
    for reason in _B5_EXPECTED_DISAGREEMENT_RAW:
        if tallies["sole_reason_counts"].get(reason, 0) != 0:
            problems.append(
                f"{reason!r} was expected to never appear as a sole reason (always co-occurs), "
                f"but sole count is {tallies['sole_reason_counts'][reason]}"
            )
    if phase_equality_occurrences != _B5_EXPECTED_PHASE_EQUALITY_OCCURRENCES:
        problems.append(
            f"{_B5_PHASE_EQUALITY_KEY!r} occurrences: expected "
            f"{_B5_EXPECTED_PHASE_EQUALITY_OCCURRENCES}, observed {phase_equality_occurrences}"
        )
    if phase_equality_occurrences != 2 * phase_equality_candidates:
        problems.append(
            f"{_B5_PHASE_EQUALITY_KEY!r} occurrences ({phase_equality_occurrences}) is not "
            f"exactly 2x its candidate count ({phase_equality_candidates})"
        )
    if phase_equality_candidates != _B5_EXPECTED_PHASE_EQUALITY_CANDIDATES:
        problems.append(
            f"{_B5_PHASE_EQUALITY_KEY!r} candidate count: expected "
            f"{_B5_EXPECTED_PHASE_EQUALITY_CANDIDATES}, observed {phase_equality_candidates}"
        )
    if tallies["raw_reason_counts"].get(_B5_PHASE_EQUALITY_KEY, 0) != 0:
        problems.append(
            f"{_B5_PHASE_EQUALITY_KEY!r} appeared as an exclusion reason "
            f"({tallies['raw_reason_counts'][_B5_PHASE_EQUALITY_KEY]} times); expected never"
        )

    if problems:
        raise SystemExit(
            "NEEDS_CONTEXT: B5 exclusion tallies do not reproduce the audited literals -- the "
            "census has changed since this runner was written against it. Stop; do not adjust "
            "the computation to match. Observed vs expected:\n  " + "\n  ".join(problems)
        )


def _count_key_occurrences(node, key: str) -> int:
    """Recursively count occurrences of dict key `key` anywhere inside `node` (dicts and lists)."""

    if isinstance(node, dict):
        count = sum(1 for k in node if k == key)
        for value in node.values():
            count += _count_key_occurrences(value, key)
        return count
    if isinstance(node, list):
        return sum(_count_key_occurrences(item, key) for item in node)
    return 0


def _tallying_candidates(candidates: Iterable[dict], tally: dict, key: str):
    """Pass-through generator: yields every candidate unchanged, tallying `key`'s occurrences.

    Lets the B5 `upstream_downstream_phase_equal` count ride along on the SAME single candidates
    pass `slot_gene_sets` already needs for B2, instead of re-streaming the whole census again.
    """

    for candidate in candidates:
        tally[key] = tally.get(key, 0) + _count_key_occurrences(candidate, key)
        yield candidate


# ---------------------------------------------------------------------------------------------
# Certificate -> slot_gene_sets filters (mirrors run_pbc_v2_capacity.py's own
# `_filters_from_certificate` -- duplicated rather than imported cross-script, matching this
# repo's convention that each runner is self-contained against `src/gfmcd` only).
# ---------------------------------------------------------------------------------------------


def _filters_from_certificate(cohort: dict) -> dict:
    config = cohort["config"]
    allowed_chroms = {normalize_chromosome(chrom) for chrom in config["allowed_chromosomes"]}
    return dict(
        blacklist=set(),
        allowed_chroms=allowed_chroms,
        forbidden_chroms=set(),
        construct_length=int(config["construct_length"]),
        donor_pos=int(config["donor_pos"]),
        acceptor_pos=int(config["acceptor_pos"]),
        donor_window=DEFAULT_DONOR_WINDOW,
        acceptor_window=DEFAULT_ACCEPTOR_WINDOW,
    )


# ---------------------------------------------------------------------------------------------
# Artifact assembly -- pure, no filesystem access.
# ---------------------------------------------------------------------------------------------


def assemble_artifact(
    *,
    census_receipt: dict,
    certificate_receipt: dict,
    cells_receipt: dict,
    analysis_receipt: dict,
    slot_hash_verification: dict,
    b1: dict,
    b2: dict,
    b3: dict,
    b4: dict,
    b5: dict,
) -> dict:
    psi_plain_needed = b3["psi_plain_2x2"]["blocks_needed"]
    return {
        "schema": SCHEMA,
        "study_directory": "results/splicebert_positional_clock/",
        "study_directory_provisional": True,
        "study_directory_provisional_note": (
            "the study directory name 'splicebert_positional_clock' is PROVISIONAL and may be "
            "renamed before anything freezes; unlike the ladder family's development/ artifacts, "
            "this one lives at the study root deliberately -- it exists precisely so a later "
            "pivot to a different evaluation design cannot silently supersede this finding"
        ),
        "stage": "construction",
        "purpose": (
            "memorialize the real-panel capacity finding: the authentic-module design cannot be "
            "built at the frozen SESOI, using the real gene pool (not a synthetic one) and "
            "measured variance components (not the PBC-v1 development-role upper bound)"
        ),
        "model_execution": False,
        "checkpoint_loaded": None,
        "cohort_selection_attempted": False,
        "census_receipt": census_receipt,
        "certificate_receipt": certificate_receipt,
        "cells_receipt": cells_receipt,
        "analysis_receipt": analysis_receipt,
        "slot_hash_verification": slot_hash_verification,
        "b1_measured_variance_components": b1,
        "b2_real_panel_ceilings": b2,
        "b3_power_at_measured_components": b3,
        "b3_headline": (
            f"Psi needs {psi_plain_needed} blocks against a usable ceiling of "
            f"~{b2['ceiling_with_25pct_assignment_slack']} and an absolute ceiling of "
            f"{b2['exact_ceiling']}"
        ),
        "real_panel_constructible_at_sesoi": False,
        "b4_cross_lineage_reframe": b4,
        "b5_filter_relaxation_closed": b5,
    }


# ---------------------------------------------------------------------------------------------
# CLI / I/O layer
# ---------------------------------------------------------------------------------------------


def _repo_relative_or_absolute(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--census", default=str(DEFAULT_CENSUS_PATH), help="path to candidate_census_v1.json")
    ap.add_argument(
        "--certificate", default=str(DEFAULT_CERTIFICATE_PATH), help="path to composition_balanced_cohort_v1.json"
    )
    ap.add_argument(
        "--cells",
        default=str(DEFAULT_CELLS_PATH),
        help="path to crossed_direct_edge_replication_cells.json (untracked, outside this repo)",
    )
    ap.add_argument(
        "--analysis",
        default=str(DEFAULT_ANALYSIS_PATH),
        help="path to crossed_direct_edge_replication_analysis.json (carries cells_result_sha256)",
    )
    ap.add_argument("--out", required=True, help="output artifact path")
    ap.add_argument(
        "--overwrite-record",
        action="store_true",
        dest="overwrite_record",
        help="replace an existing artifact of record (results are immutable by default)",
    )
    return ap.parse_args(argv)


def run_real(args: argparse.Namespace) -> tuple[Path, Path] | None:
    census_path = Path(args.census).expanduser().resolve()
    certificate_path = Path(args.certificate).expanduser().resolve()
    cells_path = Path(args.cells).expanduser().resolve()
    analysis_path = Path(args.analysis).expanduser().resolve()

    # Both the census and the cells file are treated as external inputs that may simply not be
    # present on a given machine; skip cleanly with a reason rather than crash, and write nothing.
    if not census_path.exists():
        print(f"SKIPPED: census not found at {census_path}; no artifact written.")
        return None
    if not cells_path.exists():
        print(f"SKIPPED: cells file not found at {cells_path}; no artifact written.")
        return None
    if not certificate_path.exists():
        print(f"SKIPPED: certificate not found at {certificate_path}; no artifact written.")
        return None
    if not analysis_path.exists():
        print(f"SKIPPED: analysis file not found at {analysis_path}; no artifact written.")
        return None

    out = guard_output_path(Path(args.out).expanduser().resolve(), overwrite=args.overwrite_record)
    manifest_path = guard_output_path(
        out.with_name(out.stem + ".manifest.json"), overwrite=args.overwrite_record
    )

    # ---- provenance binding: the cells file's sha256 must match BOTH the pin this runner was
    # written against AND the analysis.json's own cells_result_sha256 field, before a single row
    # is read from it.
    cells_receipt = file_record(cells_path)
    cells_receipt["mode"] = "file"
    cells_sha256 = cells_receipt["sha256"]
    analysis_doc = json.loads(analysis_path.read_text())
    analysis_cells_sha256 = analysis_doc.get("cells_result_sha256")
    if cells_sha256 != _CELLS_SHA256_PIN:
        raise SystemExit(
            f"FATAL: cells file sha256 mismatch: expected {_CELLS_SHA256_PIN}, found "
            f"{cells_sha256} at {cells_path}. Refusing to read an unverified input; no artifact "
            "was written."
        )
    if analysis_cells_sha256 != _CELLS_SHA256_PIN:
        raise SystemExit(
            f"FATAL: {analysis_path} cells_result_sha256={analysis_cells_sha256!r} does not match "
            f"the pinned cells sha256 {_CELLS_SHA256_PIN!r}. The provenance binding between the "
            "cells file and its analysis record has broken; refusing to proceed. No artifact was "
            "written."
        )
    analysis_receipt = file_record(analysis_path)
    analysis_receipt["mode"] = "file"
    analysis_receipt["path"] = _repo_relative_or_absolute(analysis_path, REPO)

    # ---- B1: sign-free variance decomposition from the cells file's 3,456 rows.
    cells_doc = json.loads(cells_path.read_text())
    cell_rows = cells_doc["cell_rows"]
    b1 = variance_components_from_cell_rows(cell_rows)
    _assert_b1_matches_expected(b1)
    b1_view = dict(b1)
    b1_view["sign_free_restriction_note"] = (
        "this runner never reads, computes, prints, or stores a mean, a sign, or a direction of "
        "any C43 endpoint -- only variances, as required by the B1 brief"
    )
    b1_view["rho_status"] = {
        "type": "point_estimate_no_interval",
        "note": (
            "rho is a point estimate from one cohort, under an exchangeability assumption across "
            "cells within a compatibility group that is only approximate (cells differ by phase "
            "pair); it is a design diagnostic, not a frozen sizing input"
        ),
    }

    # ---- census -> the seven role/phase slot gene sets, verified against the certificate, plus
    # the upstream_downstream_phase_equal tally riding along on the same pass (B5).
    cohort = json.loads(certificate_path.read_text())
    filters = _filters_from_certificate(cohort)
    phase_equality_tally: dict[str, int] = {}
    candidates = _tallying_candidates(
        stream_candidates(census_path), phase_equality_tally, _B5_PHASE_EQUALITY_KEY
    )
    slot_sets = slot_gene_sets(candidates, **filters)

    cert_slots = cohort["capacity"]["slots"]
    expected_hashes = {slot: entry["gene_ids_sha256"] for slot, entry in cert_slots.items()}
    verified = verify_slot_hashes(slot_sets, expected_hashes)
    if not all(verified.values()):
        failed = sorted(slot for slot, ok in verified.items() if not ok)
        raise SystemExit(
            f"FATAL: slot hash verification failed against the committed certificate for "
            f"{failed}. Refusing to compute real-panel capacity on an unverified gene pool; no "
            "artifact was written."
        )
    slot_hash_verification = {"per_slot": {slot: bool(ok) for slot, ok in verified.items()}, "all_verified": True}

    # ---- B2: real-panel ceilings by exact max-flow.
    union4 = len(set().union(*(slot_sets[slot] for slot in PLAIN_LATTICE_PHASE_SLOTS)))
    exact_ceiling = exact_plain_ceiling(slot_sets)
    slack_ceiling = ceiling_with_assignment_slack(slot_sets)
    intersections = {
        phase: len(slot_sets[f"donor_phase_{phase}"] & slot_sets[f"acceptor_phase_{phase}"])
        for phase in (0, 1, 2)
    }
    rotated = rotated_ceiling(slot_sets)
    _assert_b2_matches_expected(union4, exact_ceiling, intersections, rotated)
    b2 = {
        "lattice": "plain_zero_one",
        "slots_demanded_per_block": list(PLAIN_LATTICE_PHASE_SLOTS) + ["scaffold"],
        "union_of_four_phase01_module_slots": union4,
        "exact_ceiling": exact_ceiling,
        "ceiling_with_25pct_assignment_slack": slack_ceiling,
        "ceiling_with_25pct_assignment_slack_note": (
            "largest N for which the plain lattice is both exactly max-flow feasible AND leaves "
            ">=25% margin on the four phase-0/1 module slots' own union -- verified by the exact "
            "solver at every candidate N, not derived by scaling the exact ceiling"
        ),
        "donor_acceptor_phase_intersections": {str(k): v for k, v in intersections.items()},
        "rotated_lattice_ceiling_for_comparison": rotated,
        "rotated_design_withdrawn_reason": (
            "the rotated 2x2 lattice purchases M+/M- identification (delta orthogonal to donor "
            "and acceptor phase, see delta_balance_report) at a cost of "
            f"{exact_ceiling - rotated} blocks of capacity ({exact_ceiling} - {rotated} = "
            f"{exact_ceiling - rotated}), while M+/M- is unpowerable on either lattice at the "
            "frozen SESOI regardless (see B3) -- so the rotated design is withdrawn in favour of "
            "the plain lattice for sizing purposes"
        ),
    }

    # ---- B3: power at the measured components.
    b3 = power_at_measured_components(b1["rho"], b1["across_block_variance_psi_b"])
    _assert_b3_matches_expected(b3)

    # ---- B4: cross-lineage reframe.
    b4 = cross_lineage_reframe(
        b1["across_block_variance_psi_b"],
        rho=b1["rho"],
        usable_ceiling=slack_ceiling,
        absolute_ceiling=exact_ceiling,
    )
    _assert_b4_matches_expected(b4)

    # ---- B5: exclusion tallies (independent full pass over the census's exclusions array).
    exclusion_tallies = exclusion_reason_tallies(stream_exclusions(census_path))
    phase_equality_occurrences = phase_equality_tally.get(_B5_PHASE_EQUALITY_KEY, 0)
    phase_equality_candidates = phase_equality_occurrences // 2
    _assert_b5_matches_expected(exclusion_tallies, phase_equality_occurrences, phase_equality_candidates)
    b5 = {
        "census_exclusions_total": exclusion_tallies["total"],
        "sole_reason_counts": exclusion_tallies["sole_reason_counts"],
        "multi_reason_exclusions": exclusion_tallies["multi_reason_total"],
        "load_bearing_note": _B5_LOAD_BEARING_NOTE,
        "upstream_downstream_phase_equal": {
            "occurrences": phase_equality_occurrences,
            "candidates": phase_equality_candidates,
            "never_an_exclusion_reason": exclusion_tallies["raw_reason_counts"].get(_B5_PHASE_EQUALITY_KEY, 0) == 0,
            "note": (
                "no junction was excluded for donor/acceptor phase inequality; this is DEFINITIONAL "
                "(the candidate pool is constructed to require phase equality upstream) and merely "
                "UN-CONTRADICTED here, not empirically verified"
            ),
        },
        "annotation_disagreement_reason_counts": dict(_B5_EXPECTED_DISAGREEMENT_RAW),
        "annotation_disagreement_note": (
            "what DOES appear at scale is annotation disagreement BETWEEN SOURCES "
            "(comprehensive_coordinate_transcript_phase_conflict, basic_comprehensive_phase_conflict, "
            "basic_coordinate_transcript_phase_conflict) -- a GTF-consistency matter, not biology"
        ),
    }

    census_receipt = file_record(census_path)
    census_receipt["mode"] = "file"
    certificate_receipt = file_record(certificate_path)
    certificate_receipt["mode"] = "file"
    certificate_receipt["path"] = _repo_relative_or_absolute(certificate_path, REPO)
    cells_receipt["path"] = _repo_relative_or_absolute(cells_path, REPO)

    artifact = assemble_artifact(
        census_receipt=census_receipt,
        certificate_receipt=certificate_receipt,
        cells_receipt=cells_receipt,
        analysis_receipt=analysis_receipt,
        slot_hash_verification=slot_hash_verification,
        b1=b1_view,
        b2=b2,
        b3=b3,
        b4=b4,
        b5=b5,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    manifest = ExperimentManifest.create(
        EXPERIMENT_ID,
        sys.argv,
        repo=REPO,
        status="BUILT",
        config={
            "stage": "construction",
            "checkpoint_loaded": None,
            "real_panel_constructible_at_sesoi": False,
        },
        seeds={},
        notes=[
            "Model-free construction-stage real-panel capacity/power record: no checkpoint "
            "loaded, no cohort selected, no block assembled, no cell materialised.",
            "Slot gene sets were verified against the seven certificate gene_ids_sha256 hashes "
            "before B2/B5 were evaluated (hard gate, not a log line).",
            "The cells file's sha256 was verified against both a pin and "
            "crossed_direct_edge_replication_analysis.json's cells_result_sha256 before any B1 "
            "row was read (hard gate).",
            "Every B1-B5 quantity was asserted against the task-6 brief's audited literals before "
            "this artifact was written; any mismatch halts the run with NEEDS_CONTEXT.",
        ],
    )
    manifest.inputs.append(census_receipt)
    manifest.inputs.append(certificate_receipt)
    manifest.inputs.append(cells_receipt)
    manifest.inputs.append(analysis_receipt)
    manifest.add_input(Path(__file__).resolve())
    manifest.add_input(REPO / "src/gfmcd/splicing/pbc_v2_capacity.py")
    manifest.add_output(out)
    manifest.write(manifest_path)
    return out, manifest_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_real(args)
    if result is None:
        return
    out, manifest_path = result
    print(
        json.dumps(
            {"status": "BUILT", "output": str(out), "manifest": str(manifest_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
