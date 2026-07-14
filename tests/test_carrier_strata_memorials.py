"""Tripwires for WHAT the period-10.69 carrier is for.  Two structural hypotheses, both tested,
both failed -- and the positive control that makes those failures mean something.

The rule this study lives by: A NULL WITHOUT A POSITIVE CONTROL IS NOT A RESULT.  Both tests here
carry one, and both controls behaved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "results/phase-lattice/carrier_strata.json"
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run_carrier_strata.py has not been run")


@pytest.fixture(scope="module")
def R():
    return json.loads(DATA.read_text())


def test_the_translation_assay_IS_POWERED(R):
    """*** Without this, the alpha-helix null below is worthless. ***

    The codon clock, through the identical machinery, MUST show the coding-specific damage that
    the alpha-helix hypothesis predicts for the carrier.  If it does not, the assay cannot see
    coding-specificity at all and the carrier's null means nothing.
    """
    did = R["translation"]["period3"]["did"]
    assert did["lo"] > 0.10, "the codon clock's coding-specificity must be unmistakable"
    assert did["lo"] > 0, "CI excludes zero"


def test_the_protein_alpha_helix_hypothesis_is_NOT_SUPPORTED(R):
    """An alpha-helical periodicity (3.6 residues = 10.8 nt) exists ONLY in translated, in-frame
    coding sequence.  It is meaningless in an intron.  So the carrier must damage coding >> intron.
    It does not: the CI on the contrast includes zero, in an assay that sees the codon clock's
    coding-specificity 11x over."""
    p3 = R["translation"]["period3"]["did"]
    c10 = R["translation"]["carrier10"]["did"]
    assert c10["lo"] <= 0 <= c10["hi"], "the carrier's coding-specificity CI includes zero"
    assert p3["mean"] > 5 * c10["mean"], "and the codon clock's is an order of magnitude larger"


def test_the_helical_phasing_metric_is_NOT_CONFOUNDED(R):
    """The control that can fail.  The codon clock has no business tracking the sequence's WW/SS
    ~10.7-nt periodicity.  If it did, the phasing metric would be measuring something else and
    the chromatin test would be void."""
    hml = R["phasing"]["period3"]["high_minus_low"]
    assert hml["lo"] <= 0 <= hml["hi"], "the codon clock must NOT track WW/SS phasing"


def test_the_chromatin_hypothesis_is_NOT_SUPPORTED(R):
    """Nucleosome-positioning sequence carries its own ~10.5-bp WW/SS periodicity.  If the carrier
    encoded DNA-helical orientation, it should be worth MORE where the sequence is helically
    phased.  It is not -- and the CI is tight relative to the total damage."""
    hml = R["phasing"]["carrier10"]["high_minus_low"]
    assert hml["lo"] <= 0 <= hml["hi"]
    total = R["phasing"]["carrier10"]["mid"]["damage"]
    assert (hml["hi"] - hml["lo"]) < total, (
        "the CI must be tight enough to exclude a large phasing effect, or this is not a result"
    )


def test_what_is_left_standing_is_a_RESIDUAL_not_a_finding(R):
    """*** 'A generic positional ruler' is the hypothesis LEFT STANDING, not one confirmed. ***

    Two structural readings were tested and failed.  Two were NOT tested: A-form RNA (needs stem
    annotations; disfavoured by frequency, bins 93/94 are dead) and nucleosome occupancy measured
    directly (needs MNase-seq).  Nor has L0H11's OV circuit been examined.  Do not write
    'the second carrier is a generic distance ruler' as though it were shown.
    """
    c10 = R["translation"]["carrier10"]
    # it IS used -- both regions lose real log-prob.  It is not decoration.
    assert c10["exon"]["lo"] > 0 and c10["intron"]["lo"] > 0
    # but it is small, and it is flat
    assert c10["exon"]["damage"] < 0.06 and c10["intron"]["damage"] < 0.06
