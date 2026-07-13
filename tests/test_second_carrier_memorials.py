"""Tripwires for the SECOND positional carrier (period ~10.69, P's bin 96).

Found because the user asked why L0H11 was the ragged comb: "L0H11 seems to have another
frequency embedded, hence why the phase isn't matched."  It does.  It is P's OTHER high-frequency
Fourier mode, and it accounts for L0H11's raggedness exactly (80.8% -> 99.9% variance explained).

Two things about it are traps, and both already have a body count in this project:

  1. THE 'CONTROL BAND' WAS NEVER INERT.  It is bins 94-100, and bin 96 alone is 64% of its
     energy.  Every 'energy-matched control' arm in the period-3 study has been deleting a real,
     strong, widely-read carrier.  This does NOT weaken the period-3 result -- it strengthens it
     (deleting a comparably strong learned carrier still does nothing to splicing).  But calling
     that band 'inert' would be false, and the next person to read CLAIMS.md must not.

  2. A TONE IN THE PRE-SOFTMAX LOGITS IS NOT A TONE IN THE ATTENTION.  L0H5 has the largest
     period-10.69 amplitude of any head (9.07 logits) -- and L0H5 is one of R14's SINGLE-OFFSET
     COPIERS.  Its attention is a spike.  Softmax takes a big smooth oscillation plus one sharp
     peak and returns the peak.  "25 of 96 heads read the carrier" is a statement about QK
     LOGITS, and nothing else.  R14 died of exactly this confusion in a different costume.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "results/phase-lattice/second_carrier.json"
FIG = REPO / "results/phase-lattice/fig_phase_data.json"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run_second_carrier.py has not been run")


@pytest.fixture(scope="module")
def S():
    return json.loads(DATA.read_text())


@pytest.fixture(scope="module")
def D():
    return json.loads(FIG.read_text())


# ------------------------------------------------------------------ 1. THE CONTROL BAND IS ALIVE
def test_the_energy_matched_control_band_IS_the_second_carrier(D):
    """*** Do not describe bins 94-100 as an inert or empty region of P. ***

    They are the second carrier.  The band was picked to be energy-matched to the codon clock,
    and the reason it HAS that much energy is that a real carrier is sitting in it.
    """
    sp = {b["k"]: b for b in D["spectrum"]["bins"]}
    ctrl = D["ctrl_band"]
    assert ctrl == list(range(94, 101))

    bin96 = sp[96]
    assert bin96["period"] == pytest.approx(10.6875, abs=1e-3)
    assert bin96["rel"] > 500, "bin 96 is a major mode, not background"

    total = sum(sp[k]["rel"] for k in ctrl)
    assert bin96["rel"] / total > 0.6, "bin 96 alone is most of the 'control' band's energy"

    # P has exactly TWO high-frequency carriers.  Above 200x the median bin, the ONLY fast mode
    # outside the codon band is bin 96.  Every other strong mode in P is slow global drift
    # (period 22 to 1026) -- the smooth positional ramp any learned position table has.
    strong_fast = [b for b in D["spectrum"]["bins"]
                   if b["rel"] > 200 and b["period"] < 20 and b["k"] not in D["band"]]
    assert {b["k"] for b in strong_fast} == {96}, (
        "P has exactly TWO high-frequency carriers: period 3, and period 10.69"
    )
    # 97 and 98 are its sidebands (105x, 131x) -- the same phase-modulation skirt the codon
    # clock has at 341/343, which is why the band is 7 bins wide and not 1.
    assert 100 < sp[97]["rel"] < 200 and 100 < sp[98]["rel"] < 200


# ------------------------------------------------------------------ 2. THE SOFTMAX TRAP
def test_a_logit_tone_is_NOT_an_attention_tone(S):
    """*** R14's mistake, in a new costume.  Do not report 'N heads read the carrier' as a claim
    about attention. ***

    The single largest period-10.69 amplitude in the model belongs to L0H5 -- which R14 identified
    as a SINGLE-OFFSET COPIER (it reads offset +2; that is why its 'phase purity' was 1.000).  Its
    R(k) genuinely oscillates, hugely, at period 10.69.  Its attention is still a spike, because
    softmax of (a big smooth wave + one sharp peak) is the peak.

    Everything measured in second_carrier.json is PRE-SOFTMAX.  Whether any of these heads USES
    the carrier in its attention is a separate, untested question.
    """
    by = {r["name"]: r for r in S["heads"]}
    l0h5 = by["L0H5"]
    assert l0h5["intact"]["A10"] > 8.0, "L0H5 has the biggest logit tone in the model"

    # ... and its R(k) has a single dominant maximum -- the copier's offset -- towering over it
    y = np.array(l0h5["Rk_intact"])
    k = np.array(S["offsets"])
    assert k[int(y.argmax())] == 2, "L0H5's R(k) peaks at offset +2 -- R14's copier offset"

    # the tone is real (it collapses when its band is deleted) but that is a LOGIT fact
    assert l0h5["no_p10"]["A10"] / l0h5["intact"]["A10"] < 0.05


# ------------------------------------------------------------------ 3. THE DOUBLE DISSOCIATION
def test_the_two_carriers_delete_independently(S):
    """The result itself.  Each carrier survives the other's deletion and dies with its own."""
    by = {r["name"]: r for r in S["heads"]}
    floor10 = float(np.percentile([r["no_p10"]["A10"] for r in S["heads"]], 95))

    l0h11 = by["L0H11"]
    # delete the CODON CLOCK -> the second carrier is untouched
    assert l0h11["no_p3"]["A3"] / l0h11["intact"]["A3"] < 0.05
    assert l0h11["no_p3"]["A10"] / l0h11["intact"]["A10"] > 0.80

    # delete the SECOND CARRIER -> the codon clock is untouched
    assert l0h11["no_p10"]["A10"] / l0h11["intact"]["A10"] < 0.10
    assert l0h11["no_p10"]["A3"] / l0h11["intact"]["A3"] > 0.90

    # L2H15, the addresser, reads the codon clock and NOT the second carrier
    l2h15 = by["L2H15"]
    assert l2h15["intact"]["A3"] > 1.5
    assert l2h15["intact"]["A10"] < floor10, "the addresser does not read the second carrier"


def test_L0H11_is_the_ONLY_comb_that_reads_the_second_carrier(S):
    """Which is exactly why it was the ragged one.  The other five fit one cosine to >=98%."""
    by = {r["name"]: r for r in S["heads"]}
    floor10 = float(np.percentile([r["no_p10"]["A10"] for r in S["heads"]], 95))
    combs = ["L0H0", "L0H11", "L1H10", "L1H12", "L2H2", "L2H15"]
    readers = [c for c in combs if by[c]["intact"]["A10"] > floor10]
    assert readers == ["L0H11"]


def test_the_data_names_the_period_it_was_not_told(S):
    """The frequency was NOT imposed.  A free scan over T in [4,30], with the envelope and the
    period-3 term fitted jointly, lands on P's bin-96 period on its own."""
    by = {r["name"]: r for r in S["heads"]}
    assert S["period_p10_exact"] == pytest.approx(10.6875, abs=1e-3)
    for nm in ("L0H11", "L0H5", "L0H4"):
        assert by[nm]["T_free"] == pytest.approx(10.6875, abs=0.30)

    # and a head with no second carrier produces NO peak -- its scan wanders to the scan edge
    l2h15 = by["L2H15"]
    assert l2h15["A_free"] < 0.10, "the addresser's periodogram is flat, as it must be"


# ------------------------------------------------------------------ 4. TWO CARRIERS, TWO JOBS
def test_only_the_codon_clock_carries_the_splicing_circuit(D):
    """Deleting the second carrier costs a small, UNIFORM amount and leaves splicing alone."""
    d = {(r["band"], r["alpha"]): r for r in D["dose"]}
    base = d[("period-3", 1.0)]

    p3 = d[("period-3", 0.0)]
    exon3, intron3 = base["exon"] - p3["exon"], base["intron"] - p3["intron"]
    assert exon3 - intron3 > 0.10, "the codon clock is CODING-SPECIFIC"
    assert p3["coupling_kept"] < 0.05, "and it carries the splicing circuit"

    p10 = d[("control", 0.0)]
    exon10, intron10 = base["exon"] - p10["exon"], base["intron"] - p10["intron"]
    assert abs(exon10 - intron10) < 0.02, "the second carrier damages exons and introns EQUALLY"
    assert p10["coupling_kept"] > 0.95, "and splicing does not care about it at all"

    # it is not FREE, though -- it costs a real, if small, amount everywhere
    assert exon10 > 0.01 and intron10 > 0.01, "the second carrier is used for something"


def test_the_helical_repeat_reading_is_a_HYPOTHESIS_not_a_result():
    """*** 10.69 is close to 10.5, the helical repeat of B-form DNA.  That is seductive and it is
    NOT evidence. ***

    Nothing in this repo links the period-10.69 carrier to DNA structure.  We know only:
      - it exists in P, and 25 heads carry it in their QK logits
      - it is causally independent of the codon clock
      - deleting it costs ~0.03 nats uniformly and does nothing to splicing
    What it is FOR is open.  If a future claim says 'helical periodicity', this test is the
    objection: show the evidence, not the coincidence.
    """
    helical_bp = 10.5
    measured = 1026 / 96
    assert abs(measured - helical_bp) > 0.15, (
        "the measured period is 10.69, not 10.5 -- do not round it into a story"
    )


# ------------------------------------------------------------------ 5. THE B-DNA CORRECTION
GEO = REPO / "results/phase-lattice/carrier_geometry.json"


@pytest.fixture(scope="module")
def G():
    if not GEO.exists():
        pytest.skip("run_carrier_geometry.py has not been run")
    return json.loads(GEO.read_text())


def test_B_DNA_is_10_5_not_10_2(D):
    """*** The user's correction, pinned. ***

    B-DNA's helical repeat is ~10.5 bp in SOLUTION.  10.2 is the NUCLEOSOMAL value -- a different,
    more specialised quantity.  An earlier summary quoted 10.2 for "B-DNA", put it in the DEAD
    list, and omitted 10.5 entirely -- which made B-DNA look excluded on frequency when it is the
    closest LIVE candidate.  Do not repeat that.
    """
    sp = {b["k"]: b for b in D["spectrum"]["bins"]}
    # 10.5 has NO exact bin (1026/10.5 = 97.71); its power would split across 97 and 98 -- and
    # both are live.  It is NOT dead.
    assert sp[97]["rel"] > 100 and sp[98]["rel"] > 100
    # 10.2 (nucleosomal) maps to bins 100/101, which ARE dead -- but that is a different claim.
    assert sp[100]["rel"] < 50 and sp[101]["rel"] < 50


def test_the_satellite_bins_are_SIDEBANDS_not_a_separate_10_5_carrier(G):
    """So B-DNA's 10.5 is not separately present in P either.

    A rotary carrier whose direction precesses produces sidebands living in the SAME embedding
    directions as their parent.  The codon clock calibrates it: bins 341/343 vs 342 overlap 0.99.
    Bins 97/98 vs 96 overlap just as much -- they are the 10.69 carrier's skirt, not a 10.5 tone.
    """
    calib = G["calib_p3_carrier_vs_sidebands"]
    assert calib > 0.95, "the known sideband pair must show near-total subspace overlap"
    assert G["p10_carrier_vs_97_98"] > 0.9 * calib, (
        "bins 97/98 share bin 96's embedding directions -- SIDEBAND, not a separate ~10.5 carrier"
    )
    # P therefore holds exactly ONE tone in this region, at 10.69: neither 10.5 nor 10.8.
    assert G["periods"]["96"] == pytest.approx(10.6875, abs=1e-3)


def test_the_two_clocks_are_ORTHOGONAL_in_embedding_space(G):
    """The 'bank of oscillators' picture, measured: the codon clock and the period-10.69 carrier
    occupy DISJOINT subspaces of R^512.  That is why each deletes without touching the other."""
    assert G["negctl_p3_vs_p10"] < 0.02, "two distinct carriers, orthogonal directions"
