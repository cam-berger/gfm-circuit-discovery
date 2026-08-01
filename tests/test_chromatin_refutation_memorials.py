"""TRIPWIRES FOR R17 — the chromatin-source hypothesis is REFUTED, and here is what holds it up.

R17 rests on exactly two load-bearing facts.  If either silently drifts, the refutation collapses
and nobody notices.  So both are pinned here, along with the three nulls that must NEVER be quoted
as evidence and the two claims that must never be over-read.

    D27  the 512-row grid does NOT snap wavelengths to its Fourier bins
    D28  the biological signal is at ~10.30 nt, NOT at SpliceBERT's 10.67

Read `studies/splicebert_partner_selection/CLAIMS.md` before touching anything here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GRID = REPO / "results/phase-lattice/synth_grid.json"
WAVE = REPO / "results/phase-lattice/dyad_wavelength.json"
REAL = REPO / "results/phase-lattice/dyad_real.json"

# Gate per-artifact, not as one conjunction.
#
# This used to be a single module-level skipif over GRID and WAVE and REAL. `synth_grid.json` was
# deleted in af64808 without a correction entry, so from that commit the whole file skipped — all
# twelve tripwires, including the seven that need only WAVE and REAL, which are present. Twelve
# memorials defending a live claim (C14/R17, the powered refutation of the nucleosome reading) read
# as passing coverage in the suite count while asserting nothing. Splitting the gate revives them.
#
# The five D27 grid tripwires still cannot run: their artifact is gone and no runner in the tree
# regenerates it. They are marked so, rather than being hidden inside a broader skip.
_MISSING_GRID = pytest.mark.skipif(
    not GRID.exists(),
    reason="results/phase-lattice/synth_grid.json was deleted in af64808 and has no runner in the "
           "tree; the D27 grid-resolution tripwires cannot be evaluated (see CORRECTIONS.md R41)")

pytestmark = pytest.mark.skipif(
    not (WAVE.exists() and REAL.exists()),
    reason="dyad_wavelength.json / dyad_real.json not present")


@pytest.fixture(scope="module")
def grid():
    return json.loads(GRID.read_text())["runs"]


@pytest.fixture(scope="module")
def wave():
    return json.loads(WAVE.read_text())


@pytest.fixture(scope="module")
def real():
    return json.loads(REAL.read_text())


# ---------------------------------------------------------------- D27: the grid does NOT snap
@_MISSING_GRID
def test_an_OFF_grid_tone_is_represented_FAITHFULLY_not_snapped(grid):
    """THE load-bearing fact of R17's first leg.

    If the table snapped an off-grid tone to one bin, then SpliceBERT's sharp bin 48 could be a
    grid artefact and the 10.30-vs-10.67 mismatch would be explainable.  It does not snap.
    """
    r = grid["abs_tone|10.7789|d0.76|L512|s0"]
    assert abs(r["mode_hz"] - 47.5) < 0.01, "10.7789 must be mode 47.5 -- maximally off-grid"

    # it lands where the EXACT DFT of that tone says it must, not in one snapped bin
    assert abs(r["q_win"] - r["faithful_qwin"]) < 0.05, (
        f"off-grid tone gave Q_win {r['q_win']:.3f} against a faithful prediction of "
        f"{r['faithful_qwin']:.3f} -- if these ever diverge, the no-snap claim is dead")
    assert r["q_win"] < 0.6, "an off-grid tone must NOT come out sharp"
    assert r["p2n"] < 5.0, "peak-to-neighbour must be LOW -- the tone is SPLIT across two bins"


@_MISSING_GRID
def test_on_grid_tones_at_TWO_DIFFERENT_bins_are_both_sharp(grid):
    """The control that makes the split above mean 'faithful', not 'failed to learn'.

    Two on-grid tones, two DIFFERENT bins, both razor-sharp.  So the table has no preferred bin
    and the model is perfectly capable of learning a sharp carrier when the data has one.
    """
    for key, bin_, mode in [("abs_tone|10.6667|d0.76|L512|s0", 48, 48.0),
                            ("abs_tone|10.8936|d0.76|L512|s0", 47, 47.0)]:
        r = grid[key]
        assert abs(r["mode_hz"] - mode) < 0.01
        assert r["win_bin"] == bin_
        assert r["q_win"] > 0.9, f"{key}: an ON-grid tone must be SHARP"
        assert r["p2n"] > 100, f"{key}: peak must tower over its neighbours"
        # and it actually learned the tone -- the split at 47.5 is not a capacity problem
        assert r["final_loss"] < 1.22, "the model must have LEARNED the tone (oracle ~1.2025)"


@_MISSING_GRID
def test_the_readout_reports_the_flat_null_when_there_is_no_carrier(grid):
    """The negative arm.  Same readout, order-destroyed corpus -> nothing.  It is not blind,
    and it does not hallucinate."""
    r = grid["control|d0.76|L512|s0"]
    assert r["q_win"] < 0.15, "a no-structure corpus must give the flat null (~1/11 = 0.091)"
    assert r["p2n"] < 1.5
    assert abs(r["final_loss"] - 1.3700) < 0.005, (
        "the control must converge to H_binary(0.59) + ln2 = 1.3700 -- composition, nothing else")


@_MISSING_GRID
def test_the_VOID_stage_P_control_is_never_treated_as_a_control(grid):
    """make_corpus_abs originally had no shuffle_mod, so this 'control' regenerated the TREATMENT
    corpus and came back BIT-IDENTICAL.  It is marked void.  If anyone un-marks it, fail."""
    r = grid["abs_tone|control|d0.76|L512|s0"]
    assert r.get("void") is True and "VOID" in r, (
        "this run is a duplicate of the treatment, not a control -- it must stay marked VOID")


# ---------------------------------------------------------------- D28: the biology is at 10.30
def test_the_biological_wavelength_is_10_30_NOT_10_67(wave):
    """THE load-bearing fact of R17's second leg."""
    T = wave["fine_scan"]["T_peak"]
    assert 10.15 < T < 10.45, f"the dyad-anchored WW tone must be ~10.30, got {T:.3f}"
    assert abs(T - 10.6667) > 0.25, (
        f"peak at {T:.3f} must be clearly DISTINCT from SpliceBERT's 10.6667 -- "
        "if these ever coincide, R17 is dead and the chromatin hypothesis revives")


def test_the_phase_gives_the_SAME_wavelength_as_the_magnitude(wave):
    """An INDEPENDENT statistic: the phase rotates with anchor offset at exactly 360/T deg/bp.
    Magnitude and phase are different estimators; agreeing is what makes 10.30 believable."""
    Ts, Tp = wave["fine_scan"]["T_peak"], wave["phase_rotation"]["T"]
    assert abs(Ts - Tp) < 0.15, f"scan says {Ts:.3f}, phase says {Tp:.3f} -- they must agree"
    assert wave["phase_rotation"]["rms_resid_deg"] < 10.0, "the rotation must be LINEAR in offset"


def test_restricting_to_TRANSCRIBED_regions_does_not_rescue_10_67(wave):
    """The last escape hatch: SpliceBERT saw pre-mRNA, not whole genome.  It does not help --
    the genic wavelength is the same.  There is no genic sub-population sitting at 10.67."""
    if "genic" not in wave:
        pytest.skip("genic restriction not run")
    g, a = wave["genic"]["T_peak"], wave["fine_scan"]["T_peak"]
    assert abs(g - a) < 0.10, f"genic {g:.3f} vs genome-wide {a:.3f} -- must not move"
    assert abs(g - 10.6667) > 0.25, "the GENIC peak must still be far from SpliceBERT's carrier"


def test_the_instrument_was_NOT_blind__so_R17_is_a_POWERED_refutation(real):
    """THE distinction R16 and Assay 1 turned on, and the reason R17 may be stated at all.

    A null from a blind instrument is worth nothing.  This instrument FOUND the biological signal
    -- overwhelmingly -- and then measured its frequency.  It is not a failure to detect.
    """
    p = real["primary"]
    assert p["z"] > 50, f"z = {p['z']:.0f} -- the instrument must demonstrably SEE the biology"
    assert p["depth"] > 0.02, "and measure a real modulation depth"
    assert abs(p["phase_agreement_deg"]) < 5.0, (
        "the frozen phase from chr1+2 must predict chr3+4 -- this is rho, and it must be ~1")


def test_the_signal_dies_outside_the_nucleosome(real):
    """The control that could have failed.  A composition/mappability/spacing artefact has no
    reason to vanish 500 bp from a dyad."""
    oc = real["offset_control"]
    core, far = oc["0"]["absC"], oc["500"]["absC"]
    assert far < core / 10, (
        f"|C| must COLLAPSE off the nucleosome: core {core:,.0f} -> +500bp {far:,.0f}")


def test_WW_and_SS_are_counterphase(real):
    """Real DNA bending forces it.  An estimator artefact would not."""
    ww = real["primary"]["phi_deg"]
    ss = real["channels"]["SS"]["phi_deg"]
    d = abs(((ss - ww) + 180) % 360 - 180)
    assert d > 150, f"WW and SS must be counterphase, got {d:.1f} deg apart"


# ---------------------------------------------------------------- what must NEVER be quoted
def test_THREE_nulls_that_are_not_evidence():
    """R16 (WW helical-phasing), Assay 1 (unanchored corpus spectrum), and D30 (the unanchored
    WW power spectrum) are ALL underpowered instruments.  None is evidence of absence.

    Every one of them came from a QUADRATIC detector (SNR ~ depth^2 * sqrt(N)) pointed at a
    few-percent signal.  The anchored estimator is LINEAR (SNR ~ depth * sqrt(N)) and finds the
    same signal at z = +249.  That is the entire difference.

    R17 is NOT one of these.  R17 rests on a POSITIVE detection at the wrong frequency.
    """
    claims = (REPO / "studies/splicebert_partner_selection/CLAIMS.md").read_text()
    assert "D30" in claims, "the sensitivity-limit row must exist"
    assert "SENSITIVITY LIMIT, not a null" in claims, (
        "D30 must be recorded as a LIMIT.  If anyone rewrites it as a null, fail.")


@_MISSING_GRID
def test_the_phase_reset_null_is_BOUNDED_to_a_small_model(grid):
    """D29 says a 4-layer MLM cannot learn a relative ruler at 76% depth.  It does NOT say
    transformers cannot.  SpliceBERT is far larger.  The scale-up has NOT been run."""
    claims = (REPO / "studies/splicebert_partner_selection/CLAIMS.md").read_text()
    assert "Do NOT write 'transformers cannot learn relative rulers.'" in claims
    # and the null is real: the table genuinely never moved, in every arm
    for c in ("10.45", "10.6667", "10.85"):
        r = grid[f"{c}|d0.76|L512|s0"]
        assert r["q_win"] < 0.15, f"{c}: the phase-reset table must sit at the flat null"
        # ...but the corpus DID contain the band -- so this is a null, not a void run
        assert r["void"] is False, f"{c}: the generator must have put the band in the sequence"
