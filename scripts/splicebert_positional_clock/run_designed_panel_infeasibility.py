#!/usr/bin/env python
"""Designed-panel (register 3x3) infeasibility record -- an instrument/design failure, not biology.

An earlier assessment sized the designed 3x3 register panel in isolation and concluded it "clears
comfortably." That conclusion was wrong. An independent adversarial review found three defects, all
confirmed by recomputation here:

* **D1 -- additive capacity.** The designed panel and the real (plain {0,1}) panel are both
  mandatory and both draw module parent genes from the SAME pool (donor parents UNION acceptor
  parents across all three phases, size 3,227 -- see ``pool_union_size`` below). A designed block
  spends 2 module parents; a real plain-{0,1} block spends 4 module genes. Their demands are
  therefore ADDITIVE, and the earlier assessment sized each panel as if it had the pool to itself.
* **D2 -- the corrected parent-pair variance.** The earlier figures (440 blocks for Psi, 533 for
  M+/M-) transferred `C43`'s measured block variance ``V9`` unchanged to the designed panel. That
  transfer is invalid: `C43`'s 3x3 matched diagonal averages three DISTINCT donor genes and three
  DISTINCT acceptor genes (a factor 1/3 on the block-level true-effect variance), while the designed
  panel's diagonal is supplied by a SINGLE parent pair, so u/v/w are identical across the three
  matched cells and the factor 1/3 disappears -- the block-level term alone is already larger than
  `V9`, before any cell noise is added.
* **D3 -- the raw-periodicity confound.** Structural, not a power problem: the donor-to-acceptor
  span (120 nt) is a multiple of the model's period-3 carrier, so matched registers add
  constructively and mismatched add destructively for nothing more than "how strong is the
  3-periodicity of this window" -- a translation-invariant detector with no positional coordinate
  and no long-range comparison reproduces the matched-over-mismatched ranking the panel was built to
  test for. A sequence-only panel cannot separate the two, and the proposed frozen gate ("match
  period-3 amplitude AND phase across register conditions") is structurally unsatisfiable on the
  joined construct.

This artifact exists so that a later pivot to a different evaluation design cannot silently supersede
this finding, exactly as ``real_panel_capacity_v1.json`` records the real-panel failure. **This
records an instrument/design failure. It is not evidence about biology, and it licenses no
conclusion of any sign about the underlying science.**

Stage: ``construction``. This module never imports ``torch``, never imports a model adapter, never
loads a SpliceBERT checkpoint, performs NO cohort selection, NO block assembly, and NO cell
materialisation -- ``model_execution``, ``checkpoint_loaded`` and ``cohort_selection_attempted`` are
recorded as honesty fields, not merely implied by omission. The output carries counts, ratios,
verdicts and prose only -- never a gene ID.

Two inputs, both committed, small, and read-only:

* ``results/splicebert_positional_clock/development/pbc_v2_capacity_ladder_v2.json`` -- the N=909
  rung's infeasible min-cut certificate supplies ``pool_union_size`` (3,227): at N=909 the min cut's
  violating slots are exactly the six real donor/acceptor phase slots
  (``{donor,acceptor}_phase_{0,1,2}``), so ``min_cut_gene_count`` there IS
  ``|donor parents UNION acceptor parents|`` by construction of the min-cut certificate, not a
  hand-derived figure;
* ``results/splicebert_positional_clock/real_panel_capacity_v1.json`` -- supplies the measured `rho`
  and `V9` (B1), the real panel's usable/absolute ceilings and four-slot union (B2), and the naive
  (invalid-for-the-designed-panel, but already-computed) 3x3 Psi/M+/M- block counts (B3), all
  verified against their audited literals before use.

Every D1/D2 quantity below is asserted against exact recomputation before the artifact is written;
any mismatch halts the run with a ``NEEDS_CONTEXT`` message rather than writing a silently-adjusted
number. One correction was found and is documented at the assertion site: the D2 brief's f=0.5
entry for N_Psi (626) does not reproduce under the exact formula it specifies -- see
``_D2_EXPECTED_BAND``'s inline note. The value used and recorded here (625) is the one the formula
actually produces; the formula, `rho`, `V9`, `Z`, and `SESOI` were not adjusted to reach either
number, and the other seven of eight D2 band literals reproduce exactly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.artifact_guard import guard_output_path  # noqa: E402
from gfmcd.provenance import ExperimentManifest, file_record  # noqa: E402
from gfmcd.splicing.pbc_v2_capacity import SESOI, _Z  # noqa: E402

SCHEMA = "gfmcd.splicebert_positional_clock.designed_panel_infeasibility.v1"
EXPERIMENT_ID = "splicebert-positional-clock:designed-panel-infeasibility-v1"

DEFAULT_LADDER_PATH = (
    REPO / "results/splicebert_positional_clock/development/pbc_v2_capacity_ladder_v2.json"
)
DEFAULT_REAL_PANEL_PATH = REPO / "results/splicebert_positional_clock/real_panel_capacity_v1.json"

MARGIN_THRESHOLD = 0.25
DESIGNED_MODULE_PARENTS_PER_BLOCK = 2
REAL_MODULE_GENES_PER_BLOCK = 4

SCOPE_NOTE = (
    "this records an instrument/design failure. It is not evidence about biology, and it licenses "
    "no conclusion of any sign about the underlying science"
)

# ---------------------------------------------------------------------------------------------
# Input receipts -- both committed, small artifacts already produced by sibling runners in this
# study. Every field pulled from them is verified against its audited literal before use (below);
# a mismatch means the upstream record has changed since this runner was written against it.
# ---------------------------------------------------------------------------------------------

_LADDER_909_RUNG = 909
_LADDER_909_MIN_CUT_SLOTS_EXPECTED = sorted(
    [
        "acceptor_phase_0",
        "acceptor_phase_1",
        "acceptor_phase_2",
        "donor_phase_0",
        "donor_phase_1",
        "donor_phase_2",
    ]
)
_EXPECTED_POOL_UNION_SIZE = 3227

_EXPECTED_RHO = 0.368264006833713
_EXPECTED_V9 = 0.14012168638599368
_EXPECTED_REAL_USABLE_N = 544
_EXPECTED_REAL_ABSOLUTE_N = 726
_EXPECTED_PSI_FULL_3X3 = 440
_EXPECTED_MPM_FULL_3X3 = 533
_EXPECTED_REAL_FOUR_SLOT_UNION = 2904


def _read_pool_union_size(ladder_doc: dict) -> tuple[int, dict]:
    """Pull the 6-slot pool union size from the ladder's N=909 infeasible min-cut certificate.

    At N=909 the reported min cut's violating slots are exactly the six real donor/acceptor phase
    slots (not a proper subset), which is what makes ``min_cut_gene_count`` there equal to
    ``|donor parents UNION acceptor parents|`` rather than some smaller obstructed subset -- this is
    asserted, not assumed.
    """

    rungs = ladder_doc["rungs"]
    matches = [row for row in rungs if row["n_blocks"] == _LADDER_909_RUNG]
    if len(matches) != 1:
        raise SystemExit(
            f"NEEDS_CONTEXT: expected exactly one N={_LADDER_909_RUNG} rung in the ladder artifact, "
            f"found {len(matches)}. Stop; do not adjust the computation to match."
        )
    feas = matches[0]["feasibility"]
    observed_slots = sorted(feas["min_cut_slots"])
    if observed_slots != _LADDER_909_MIN_CUT_SLOTS_EXPECTED or feas["feasible"]:
        raise SystemExit(
            "NEEDS_CONTEXT: the N=909 rung's min-cut certificate no longer violates exactly the six "
            f"real donor/acceptor phase slots (expected {_LADDER_909_MIN_CUT_SLOTS_EXPECTED}, "
            f"observed feasible={feas['feasible']} slots={observed_slots}). Stop; do not adjust the "
            "computation to match."
        )
    return int(feas["min_cut_gene_count"]), feas


def _read_real_panel_inputs(real_panel_doc: dict) -> dict:
    b1 = real_panel_doc["b1_measured_variance_components"]
    b2 = real_panel_doc["b2_real_panel_ceilings"]
    b3 = real_panel_doc["b3_power_at_measured_components"]
    return {
        "rho": float(b1["rho"]),
        "v9": float(b1["across_block_variance_psi_b"]),
        "real_usable_n": int(b2["ceiling_with_25pct_assignment_slack"]),
        "real_absolute_n": int(b2["exact_ceiling"]),
        "real_four_slot_union": int(b2["union_of_four_phase01_module_slots"]),
        "psi_full_3x3": int(b3["psi_full_3x3"]["blocks_needed"]),
        "m_pm_full_3x3": int(b3["m_pm_full_3x3"]["blocks_needed"]),
    }


def _assert_inputs_match_expected(pool_union_size: int, real_inputs: dict) -> None:
    problems: list[str] = []
    if pool_union_size != _EXPECTED_POOL_UNION_SIZE:
        problems.append(f"pool_union_size: expected {_EXPECTED_POOL_UNION_SIZE}, observed {pool_union_size}")
    if abs(real_inputs["rho"] - _EXPECTED_RHO) > 1e-9:
        problems.append(f"rho: expected {_EXPECTED_RHO}, observed {real_inputs['rho']}")
    if abs(real_inputs["v9"] - _EXPECTED_V9) > 1e-9:
        problems.append(f"v9: expected {_EXPECTED_V9}, observed {real_inputs['v9']}")
    if real_inputs["real_usable_n"] != _EXPECTED_REAL_USABLE_N:
        problems.append(f"real_usable_n: expected {_EXPECTED_REAL_USABLE_N}, observed {real_inputs['real_usable_n']}")
    if real_inputs["real_absolute_n"] != _EXPECTED_REAL_ABSOLUTE_N:
        problems.append(
            f"real_absolute_n: expected {_EXPECTED_REAL_ABSOLUTE_N}, observed {real_inputs['real_absolute_n']}"
        )
    if real_inputs["real_four_slot_union"] != _EXPECTED_REAL_FOUR_SLOT_UNION:
        problems.append(
            f"real_four_slot_union: expected {_EXPECTED_REAL_FOUR_SLOT_UNION}, observed "
            f"{real_inputs['real_four_slot_union']}"
        )
    if real_inputs["psi_full_3x3"] != _EXPECTED_PSI_FULL_3X3:
        problems.append(f"psi_full_3x3: expected {_EXPECTED_PSI_FULL_3X3}, observed {real_inputs['psi_full_3x3']}")
    if real_inputs["m_pm_full_3x3"] != _EXPECTED_MPM_FULL_3X3:
        problems.append(f"m_pm_full_3x3: expected {_EXPECTED_MPM_FULL_3X3}, observed {real_inputs['m_pm_full_3x3']}")
    if problems:
        raise SystemExit(
            "NEEDS_CONTEXT: one or more upstream inputs (ladder N=909 certificate / real-panel "
            "capacity record) do not reproduce the audited literals this runner was written "
            "against. Stop; do not adjust the computation to match. Observed vs expected:\n  "
            + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------------------------
# Shared pure primitives -- duplicated (not imported) from run_real_panel_capacity.py /
# gfmcd.splicing.pbc_v2_capacity by this family's own convention: each runner is self-contained
# against src/gfmcd only, so a sibling runner's private helpers never become a load-bearing import.
# ---------------------------------------------------------------------------------------------


def _cell_variance_at(rho: float, v9: float) -> float:
    return 2.0 * (1.0 - rho) * v9


def _var_psi_at(matched: int, mismatched: int, rho: float, v9: float) -> float:
    return rho * v9 + _cell_variance_at(rho, v9) * (1.0 / matched + 1.0 / mismatched)


def _var_dir_at(delta_cells: int, matched: int, rho: float, v9: float) -> float:
    """Same shape as `_var_psi_at`, kept as a separate named function for readability at call
    sites even though the two are algebraically identical: `rho*v9 + cell_variance*(1/x + 1/y)`.
    """

    return rho * v9 + _cell_variance_at(rho, v9) * (1.0 / delta_cells + 1.0 / matched)


def _raw_blocks_for_sesoi(variance: float) -> float:
    """The exact (pre-ceiling) MDE80<=SESOI block count at alpha=0.05 two-sided, power=0.80."""

    return _Z * _Z * variance / (SESOI**2)


def _ceil_blocks_for_sesoi(variance: float) -> int:
    return math.ceil(_raw_blocks_for_sesoi(variance))


# ---------------------------------------------------------------------------------------------
# D1 -- the additive-capacity failure.
#
# `union_spend` uses the RAW (pre-ceiling) designed-side block requirement, not the ceiling'd
# integer displayed as `designed_blocks_needed`: a design that needs 486.02 blocks-worth of target
# variance draws 486.02 blocks-worth of nominal pool capacity even though only a whole 487 blocks
# could actually be run. This is not a free choice -- it is the only computation of the four that
# reproduces all four audited union-spend literals below to within rounding; multiplying the
# CEILING'd designed block count by 2 instead reproduces three of the four rows exactly but misses
# the fourth (487 * 2 + 544 * 4 = 3150, not the audited 3148) by exactly 2, because 487 is 0.98
# blocks above its own raw value while 440/764 sit within 0.1 of theirs.
# ---------------------------------------------------------------------------------------------


def additive_capacity_row(
    *,
    row_label: str,
    designed_source: str,
    designed_raw_blocks: float,
    real_source: str,
    real_blocks: int,
    pool_union_size: int,
    margin_threshold: float = MARGIN_THRESHOLD,
) -> dict:
    designed_display = math.ceil(designed_raw_blocks)
    spend = int(
        round(
            DESIGNED_MODULE_PARENTS_PER_BLOCK * designed_raw_blocks
            + REAL_MODULE_GENES_PER_BLOCK * real_blocks
        )
    )
    if spend > pool_union_size:
        margin: float | None = None
        verdict = "infeasible"
    else:
        margin = (pool_union_size - spend) / pool_union_size
        verdict = "below the 0.25 rule" if margin < margin_threshold else "clears the 0.25 rule"
    return {
        "row_label": row_label,
        "designed_source": designed_source,
        "designed_blocks_needed": designed_display,
        "designed_blocks_needed_raw": designed_raw_blocks,
        "real_source": real_source,
        "real_blocks_needed": int(real_blocks),
        "designed_spend_per_block_module_parents": DESIGNED_MODULE_PARENTS_PER_BLOCK,
        "real_spend_per_block_module_genes": REAL_MODULE_GENES_PER_BLOCK,
        "union_spend": spend,
        "pool_union_size": int(pool_union_size),
        "margin": margin,
        "verdict": verdict,
    }


_D1_EXPECTED = [
    {"designed_blocks_needed": 440, "real_blocks_needed": 544, "union_spend": 3056, "margin": 0.053, "verdict": "below the 0.25 rule"},
    {"designed_blocks_needed": 487, "real_blocks_needed": 544, "union_spend": 3148, "margin": 0.024, "verdict": "below the 0.25 rule"},
    {"designed_blocks_needed": 440, "real_blocks_needed": 726, "union_spend": 3784, "margin": None, "verdict": "infeasible"},
    {"designed_blocks_needed": 764, "real_blocks_needed": 544, "union_spend": 3704, "margin": None, "verdict": "infeasible"},
]


def additive_capacity_table(
    *, rho: float, v9: float, real_usable_n: int, real_absolute_n: int, pool_union_size: int
) -> list[dict]:
    psi_full_3x3_raw = _raw_blocks_for_sesoi(_var_psi_at(3, 6, rho, v9))
    f0_raw = _raw_blocks_for_sesoi(var_psi_designed(0.0, rho, v9))
    f1_psi_raw = _raw_blocks_for_sesoi(var_psi_designed(1.0, rho, v9))

    return [
        additive_capacity_row(
            row_label="naive_psi_full_3x3_vs_usable_ceiling",
            designed_source="psi_full_3x3 (naive V9 transfer, invalid for the designed panel -- B3)",
            designed_raw_blocks=psi_full_3x3_raw,
            real_source="real panel usable ceiling (25% margin)",
            real_blocks=real_usable_n,
            pool_union_size=pool_union_size,
        ),
        additive_capacity_row(
            row_label="corrected_f0_vs_usable_ceiling",
            designed_source="corrected designed-panel capacity at f=0 (D2 band floor)",
            designed_raw_blocks=f0_raw,
            real_source="real panel usable ceiling (25% margin)",
            real_blocks=real_usable_n,
            pool_union_size=pool_union_size,
        ),
        additive_capacity_row(
            row_label="naive_psi_full_3x3_vs_absolute_ceiling",
            designed_source="psi_full_3x3 (naive V9 transfer, invalid for the designed panel -- B3)",
            designed_raw_blocks=psi_full_3x3_raw,
            real_source="real panel absolute exact ceiling",
            real_blocks=real_absolute_n,
            pool_union_size=pool_union_size,
        ),
        additive_capacity_row(
            row_label="corrected_f1_psi_vs_usable_ceiling",
            designed_source="corrected designed-panel Psi capacity at f=1.0 (D2 band ceiling)",
            designed_raw_blocks=f1_psi_raw,
            real_source="real panel usable ceiling (25% margin)",
            real_blocks=real_usable_n,
            pool_union_size=pool_union_size,
        ),
    ]


def _assert_d1_matches_expected(rows: list[dict]) -> None:
    problems: list[str] = []
    if len(rows) != len(_D1_EXPECTED):
        raise SystemExit(
            f"NEEDS_CONTEXT: D1 table has {len(rows)} rows, expected {len(_D1_EXPECTED)}. Stop."
        )
    for i, (row, expected) in enumerate(zip(rows, _D1_EXPECTED)):
        for field in ("designed_blocks_needed", "real_blocks_needed", "union_spend", "verdict"):
            if row[field] != expected[field]:
                problems.append(f"row {i} {field}: expected {expected[field]!r}, observed {row[field]!r}")
        if expected["margin"] is None:
            if row["margin"] is not None:
                problems.append(f"row {i} margin: expected None (infeasible), observed {row['margin']!r}")
        else:
            if row["margin"] is None or abs(row["margin"] - expected["margin"]) > 5e-4:
                problems.append(f"row {i} margin: expected ~{expected['margin']}, observed {row['margin']!r}")
    if problems:
        raise SystemExit(
            "NEEDS_CONTEXT: D1 additive-capacity table does not reproduce the audited literals. "
            "Stop; do not adjust the computation to match. Observed vs expected:\n  "
            + "\n  ".join(problems)
        )


# ---------------------------------------------------------------------------------------------
# D2 -- the corrected parent-pair variance. `f` is the ratio of designed within-block cell
# heterogeneity to C43's -- an UNKNOWN, never a measured quantity; every N below is a BAND over f,
# never a point.
# ---------------------------------------------------------------------------------------------


def sig_c2(rho: float, v9: float) -> float:
    return 2.0 * (1.0 - rho) * v9


def var_psi_designed(f: float, rho: float, v9: float) -> float:
    return 3.0 * rho * v9 + f * sig_c2(rho, v9) * (1.0 / 3.0 + 1.0 / 6.0)


def var_dir_designed(f: float, rho: float, v9: float) -> float:
    return 3.0 * rho * v9 + f * sig_c2(rho, v9) * (1.0 / 3.0 + 1.0 / 3.0)


def designed_variance_band(rho: float, v9: float, f_values: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)) -> list[dict]:
    band = []
    for f in f_values:
        v_psi = var_psi_designed(f, rho, v9)
        v_dir = var_dir_designed(f, rho, v9)
        band.append(
            {
                "f": f,
                "f_status": "unmeasured",
                "var_psi_designed": v_psi,
                "var_dir_designed": v_dir,
                "n_psi_raw": _raw_blocks_for_sesoi(v_psi),
                "n_psi": _ceil_blocks_for_sesoi(v_psi),
                "n_m_raw": _raw_blocks_for_sesoi(v_dir),
                "n_m": _ceil_blocks_for_sesoi(v_dir),
            }
        )
    return band


def implicit_f(target_variance: float, coefficient: float, rho: float, v9: float) -> float:
    """Solve `3*rho*v9 + f*sig_c2*coefficient = target_variance` for `f`."""

    return (target_variance - 3.0 * rho * v9) / (sig_c2(rho, v9) * coefficient)


# CORRECTED at the assertion site, and the correction is recorded rather than silently absorbed.
# The task brief's D2 literal table states "f=0.5 -> 626, 672" for (N_Psi, N_M). Recomputing
# var_psi_designed(0.5, rho, v9) via the EXACT formula the brief specifies and ceiling it gives 625,
# not 626 -- verified three ways (full-precision rho/V9, the brief's own truncated 9-sig-fig V9, and
# a higher-precision Z), all agreeing on 624.9758 raw, ceil 625. The other seven of eight D2 band
# literals (f=0 both entries, f=0.25 both entries, f=0.5's N_M=672, f=1.0 both entries) reproduce
# exactly under the identical formula and code path, which rules out a formula- or precision-level
# explanation specific to this one cell. Per this runner's own rule -- recompute, do not hardcode,
# and do not adjust a computation to match a stated target -- the value recorded here is 625, the
# one the formula actually produces; this comment is the flag, not a silent substitution.
_D2_EXPECTED_BAND = {
    0.0: {"n_psi": 487, "n_m": 487},
    0.25: {"n_psi": 556, "n_m": 579},
    0.5: {"n_psi": 625, "n_m": 672},  # brief stated n_psi=626; see correction note above
    1.0: {"n_psi": 764, "n_m": 857},
}
_D2_EXPECTED_LEGACY_533_IMPLICIT_F = 0.125


def _assert_d2_matches_expected(band: list[dict], legacy_533_implicit_f_value: float) -> None:
    problems: list[str] = []
    observed_fs = {row["f"] for row in band}
    expected_fs = set(_D2_EXPECTED_BAND)
    if observed_fs != expected_fs:
        raise SystemExit(f"NEEDS_CONTEXT: D2 band f-values {observed_fs} != expected {expected_fs}. Stop.")
    for row in band:
        expected = _D2_EXPECTED_BAND[row["f"]]
        if row["n_psi"] != expected["n_psi"]:
            problems.append(f"f={row['f']} n_psi: expected {expected['n_psi']}, observed {row['n_psi']}")
        if row["n_m"] != expected["n_m"]:
            problems.append(f"f={row['f']} n_m: expected {expected['n_m']}, observed {row['n_m']}")
    if abs(legacy_533_implicit_f_value - _D2_EXPECTED_LEGACY_533_IMPLICIT_F) > 1e-3:
        problems.append(
            "legacy_533_implicit_f: expected ~"
            f"{_D2_EXPECTED_LEGACY_533_IMPLICIT_F}, observed {legacy_533_implicit_f_value}"
        )
    if problems:
        raise SystemExit(
            "NEEDS_CONTEXT: D2 corrected-variance band does not reproduce the audited literals. "
            "Stop; do not adjust the computation to match. Observed vs expected:\n  "
            + "\n  ".join(problems)
        )


def d2_corrected_variance_section(rho: float, v9: float) -> dict:
    band = designed_variance_band(rho, v9)
    three_rho_v9 = 3.0 * rho * v9
    sig_c2_value = sig_c2(rho, v9)

    # No f >= 0 reproduces the naive 440 (psi_full_3x3's variance, which equals V9 exactly, since
    # var_psi(3,6,rho) == V9 for every rho -- see run_real_panel_capacity.py's B4 note): the f that
    # WOULD reproduce it is negative, so it is not attainable by any non-negative heterogeneity ratio.
    f_implicit_for_440 = implicit_f(v9, 0.5, rho, v9)
    no_f_makes_440_correct = f_implicit_for_440 < 0.0

    # The earlier 533 (m_pm_full_3x3, the naive V9-transfer M+/M- figure) implicitly assumed the
    # f solving var_dir_designed(f) == that same naive variance -- an assumption never stated.
    naive_mpm_variance = _var_dir_at(3, 3, rho, v9)
    f_implicit_for_533 = implicit_f(naive_mpm_variance, 2.0 / 3.0, rho, v9)

    _assert_d2_matches_expected(band, f_implicit_for_533)

    return {
        "rho": rho,
        "v9": v9,
        "sig_c2": sig_c2_value,
        "three_rho_v9": three_rho_v9,
        "three_rho_v9_over_v9_ratio": three_rho_v9 / v9,
        "invalid_transfer_note": (
            "the earlier figures (440 blocks for Psi, 533 for M+/M-) transferred C43's measured "
            "block variance V9 unchanged to the designed panel; that transfer is invalid"
        ),
        "diagonal_averaging_note": (
            "in C43's 3x3, the matched diagonal averages three DISTINCT donor genes and three "
            "DISTINCT acceptor genes, so the block-level true-effect variance carries a factor 1/3: "
            "(su2+sv2+sw2)/3 = rho*V9, giving su2+sv2+sw2 = 3*rho*V9"
        ),
        "designed_panel_note": (
            "in the designed panel ONE parent pair supplies all three diagonal cells, so u, v, w "
            "are identical across them and the factor 1/3 disappears -- the block-level term is the "
            "full 3*rho*V9, already larger than V9 before any cell noise is added"
        ),
        "f_status": "unmeasured",
        "f_definition": (
            "the ratio of designed within-block cell heterogeneity to C43's -- an UNKNOWN, not a "
            "measured quantity"
        ),
        "band": band,
        "band_note": "N values are a BAND over f, never a point; f is not measured",
        "no_f_makes_440_correct": bool(no_f_makes_440_correct),
        "no_f_makes_440_correct_derivation": (
            f"the f solving var_psi_designed(f) = V9 (440's variance) is {f_implicit_for_440:.6f}, "
            "which is negative; f must be >= 0, so no such f exists"
        ),
        "legacy_533_implicit_f": f_implicit_for_533,
        "legacy_533_implicit_f_note": (
            "the earlier 533 (M+/M-) implicitly assumed f approximately 0.125 -- an assumption "
            "never stated"
        ),
    }


# ---------------------------------------------------------------------------------------------
# D3 -- the raw-periodicity confound. Structural, not a power problem: prose plus the one
# arithmetic fact (the donor-acceptor span is an exact multiple of the period-3 carrier).
# ---------------------------------------------------------------------------------------------

SPAN_NT = 120
PERIOD_NT = 3


def d3_raw_periodicity_confound() -> dict:
    if SPAN_NT % PERIOD_NT != 0:
        raise SystemExit(
            f"NEEDS_CONTEXT: SPAN_NT={SPAN_NT} is no longer a multiple of PERIOD_NT={PERIOD_NT}. Stop."
        )
    return {
        "span_nt": SPAN_NT,
        "period_nt": PERIOD_NT,
        "span_is_multiple_of_period": True,
        "span_cycles": SPAN_NT // PERIOD_NT,
        "structural_not_a_power_problem": True,
        "confound_statement": (
            "the donor-to-acceptor span is 120 nt, and 120 is a multiple of 3. The two module "
            "interiors each carry a period-3 Fourier component, and their relative phase across the "
            "joined window is exactly delta * 120 degrees, with no fractional-cycle residue left by "
            "the intervening span (because the span itself is an exact multiple of the period). "
            "Matched registers therefore add constructively and mismatched add destructively: for "
            "comparable component magnitudes a ~ b, |a+b| = 2a against sqrt(a^2 + b^2 - ab) = a. A "
            "model that computes nothing but 'how strong is the 3-periodicity of this window' -- "
            "translation-invariant, with no positional coordinate and no long-range comparison -- "
            "reproduces the matched-over-mismatched ranking. That is precisely the alternative to "
            "the hypothesis the panel was built to test, and a sequence-only panel cannot separate "
            "the two"
        ),
        "constructive_matched_magnitude": "|a+b| = 2a for comparable components a ~ b (delta = 0, in phase)",
        "destructive_mismatched_magnitude": (
            "sqrt(a^2 + b^2 - ab) = a for comparable components a ~ b (delta = 1 or 2, 120 or 240 "
            "degrees out of phase)"
        ),
        "gate_unsatisfiable_note": (
            "the proposed frozen gate 'match period-3 amplitude AND phase across register "
            "conditions' is structurally unsatisfiable on the joined construct: equalising "
            "joined-window period-3 phase across register conditions IS abolishing delta. Either the "
            "gate is enforced and the treatment vanishes, or it is not enforced and the confound "
            "stands unmeasured"
        ),
    }


# ---------------------------------------------------------------------------------------------
# D4 -- disposition.
# ---------------------------------------------------------------------------------------------


def d4_disposition() -> dict:
    return {
        "designed_panel_viable_as_primary_identifier": False,
        "superseded_conclusion": "clears comfortably",
        "superseded_conclusion_note": (
            "an earlier assessment concluded the designed 3x3 register panel 'clears comfortably'; "
            "that conclusion was wrong -- see D1 (additive capacity), D2 (corrected parent-pair "
            "variance), and D3 (the raw-periodicity confound)"
        ),
        "successor_primary_endpoint_note": (
            "the successor primary endpoint is a carrier-register intervention (a prespecified "
            "cyclic delta x r interaction on identical sequences), not a sequence-only "
            "matched-minus-mismatched contrast"
        ),
        "successor_not_further_described_note": (
            "the successor is not built yet, and this artifact must not imply it works"
        ),
    }


# ---------------------------------------------------------------------------------------------
# Artifact assembly -- pure, no filesystem access.
# ---------------------------------------------------------------------------------------------


def assemble_artifact(
    *,
    ladder_receipt: dict,
    real_panel_receipt: dict,
    pool_union_size: int,
    real_four_slot_union: int,
    d1_rows: list[dict],
    d2: dict,
    d3: dict,
    d4: dict,
) -> dict:
    return {
        "schema": SCHEMA,
        "study_directory": "results/splicebert_positional_clock/",
        "study_directory_provisional": True,
        "study_directory_provisional_note": (
            "the study directory name 'splicebert_positional_clock' is PROVISIONAL and may be "
            "renamed before anything freezes; this artifact deliberately lives at the study root, "
            "not under development/, so a later pivot to a different evaluation design cannot "
            "silently supersede the finding it memorializes -- exactly like its sibling "
            "real_panel_capacity_v1.json"
        ),
        "stage": "construction",
        "scope_note": SCOPE_NOTE,
        "purpose": (
            "memorialize the designed 3x3 register panel's infeasibility: an earlier assessment "
            "concluded it 'clears comfortably'; an independent adversarial review found three "
            "confirmed defects (additive capacity, an invalid variance transfer, and a structural "
            "raw-periodicity confound), recorded here so a later pivot cannot silently supersede it"
        ),
        "model_execution": False,
        "checkpoint_loaded": None,
        "cohort_selection_attempted": False,
        "ladder_receipt": ladder_receipt,
        "real_panel_receipt": real_panel_receipt,
        "designed_panel_viable_as_primary_identifier": False,
        "d1_additive_capacity_failure": {
            "pool_union_size": pool_union_size,
            "pool_union_size_provenance": (
                "results/splicebert_positional_clock/development/pbc_v2_capacity_ladder_v2.json, "
                "N=909 rung, feasibility.min_cut_gene_count -- the min cut there violates exactly "
                "the six real donor/acceptor phase slots, so this figure is "
                "|donor parents UNION acceptor parents| by construction of the min-cut certificate"
            ),
            "designed_spend_per_block_module_parents": DESIGNED_MODULE_PARENTS_PER_BLOCK,
            "real_spend_per_block_module_genes": REAL_MODULE_GENES_PER_BLOCK,
            "demands_are_additive_note": (
                "both panels are mandatory and both draw module parent genes from the SAME pool; "
                "their demands are therefore additive, and the earlier assessment sized each panel "
                "as if it had the pool to itself"
            ),
            "margin_formula": "(pool_union_size - union_spend) / pool_union_size",
            "margin_threshold": MARGIN_THRESHOLD,
            "rows": d1_rows,
            "real_panel_four_slot_union": real_four_slot_union,
            "real_panel_four_slot_union_is_strict_subset_of_pool": bool(
                real_four_slot_union < pool_union_size
            ),
            "real_panel_subset_note": (
                "the real panel actually draws from D0 UNION D1 UNION A0 UNION A1, a strict SUBSET "
                "of the 3,227-gene pool union used above; the true position is therefore at least "
                "this bad and never better"
            ),
        },
        "d2_corrected_parent_pair_variance": d2,
        "d3_raw_periodicity_confound": d3,
        "d4_disposition": d4,
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
    ap.add_argument(
        "--ladder",
        default=str(DEFAULT_LADDER_PATH),
        help="path to pbc_v2_capacity_ladder_v2.json (committed, supplies pool_union_size)",
    )
    ap.add_argument(
        "--real-panel",
        default=str(DEFAULT_REAL_PANEL_PATH),
        dest="real_panel",
        help="path to real_panel_capacity_v1.json (committed, supplies rho/V9/ceilings/naive B3)",
    )
    ap.add_argument("--out", required=True, help="output artifact path")
    ap.add_argument(
        "--overwrite-record",
        action="store_true",
        dest="overwrite_record",
        help="replace an existing artifact of record (results are immutable by default)",
    )
    return ap.parse_args(argv)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    ladder_path = Path(args.ladder).expanduser().resolve()
    real_panel_path = Path(args.real_panel).expanduser().resolve()

    if not ladder_path.exists():
        raise SystemExit(
            f"FATAL: ladder artifact not found at {ladder_path}. This is a committed, tracked "
            "input, not an external one; its absence indicates a broken checkout, not a normal "
            "skip condition. No artifact was written."
        )
    if not real_panel_path.exists():
        raise SystemExit(
            f"FATAL: real-panel capacity artifact not found at {real_panel_path}. This is a "
            "committed, tracked input; its absence indicates a broken checkout. No artifact was "
            "written."
        )

    out = guard_output_path(Path(args.out).expanduser().resolve(), overwrite=args.overwrite_record)
    manifest_path = guard_output_path(
        out.with_name(out.stem + ".manifest.json"), overwrite=args.overwrite_record
    )

    ladder_doc = json.loads(ladder_path.read_text())
    real_panel_doc = json.loads(real_panel_path.read_text())

    pool_union_size, _min_cut = _read_pool_union_size(ladder_doc)
    real_inputs = _read_real_panel_inputs(real_panel_doc)
    _assert_inputs_match_expected(pool_union_size, real_inputs)

    d1_rows = additive_capacity_table(
        rho=real_inputs["rho"],
        v9=real_inputs["v9"],
        real_usable_n=real_inputs["real_usable_n"],
        real_absolute_n=real_inputs["real_absolute_n"],
        pool_union_size=pool_union_size,
    )
    _assert_d1_matches_expected(d1_rows)

    d2 = d2_corrected_variance_section(real_inputs["rho"], real_inputs["v9"])
    d3 = d3_raw_periodicity_confound()
    d4 = d4_disposition()

    ladder_receipt = file_record(ladder_path)
    ladder_receipt["mode"] = "file"
    ladder_receipt["path"] = _repo_relative_or_absolute(ladder_path, REPO)
    real_panel_receipt = file_record(real_panel_path)
    real_panel_receipt["mode"] = "file"
    real_panel_receipt["path"] = _repo_relative_or_absolute(real_panel_path, REPO)

    artifact = assemble_artifact(
        ladder_receipt=ladder_receipt,
        real_panel_receipt=real_panel_receipt,
        pool_union_size=pool_union_size,
        real_four_slot_union=real_inputs["real_four_slot_union"],
        d1_rows=d1_rows,
        d2=d2,
        d3=d3,
        d4=d4,
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
            "designed_panel_viable_as_primary_identifier": False,
        },
        seeds={},
        notes=[
            "Model-free construction-stage designed-panel infeasibility record: no checkpoint "
            "loaded, no cohort selected, no block assembled, no cell materialised.",
            "pool_union_size, rho, V9, real ceilings, and the naive B3 block counts were verified "
            "against their audited literals before D1/D2 were evaluated (hard gate, not a log "
            "line).",
            "Every D1/D2 quantity was asserted against exact recomputation before this artifact "
            "was written; any mismatch halts the run with NEEDS_CONTEXT. One correction was found "
            "and is documented at the assertion site in the runner source: the D2 f=0.5 N_Psi "
            "literal is 625 by exact recomputation, not 626.",
            "This records an instrument/design failure. It is not evidence about biology, and it "
            "licenses no conclusion of any sign about the underlying science.",
        ],
    )
    manifest.inputs.append(ladder_receipt)
    manifest.inputs.append(real_panel_receipt)
    manifest.add_input(Path(__file__).resolve())
    manifest.add_input(REPO / "src/gfmcd/splicing/pbc_v2_capacity.py")
    manifest.add_output(out)
    manifest.write(manifest_path)
    return out, manifest_path


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    out, manifest_path = run(args)
    print(
        json.dumps(
            {"status": "BUILT", "output": str(out), "manifest": str(manifest_path)},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
