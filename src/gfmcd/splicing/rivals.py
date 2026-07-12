"""Rival models = the pre-registered shortcut table, made runnable.

v1 died because no such table existed.  Each rival here is an ADVERSARY that tries to
solve the partner task WITHOUT partner information.  Every one of them must sit at (or
below) chance on the ``adversarial`` stratum, or the experiment is not identified and
we do not get to make the claim.

Measured on v1, for reference -- these are the numbers that killed it:
    geometry oracle (reads ZERO nucleotides) ....... PMR 1.000
    oracle site detector ........................... PMR 1.000
    leak-only (position + PWM donor + length) ...... PMR 0.483
    length-only .................................... PMR 0.495
Measured on v2 (chr22), same adversaries:
    geometry oracle ................................ 0.076   (chance 0.458)
    oracle site detector ........................... 0.458   = chance, exactly
    candidate-index prior .......................... 0.469   ~ chance
and on the ``adversarial`` stratum, length-prior and nearest-downstream score 0.000
by construction.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from gfmcd.splicing.partner_data import LengthPrior, PartnerExample


class Rival(Protocol):
    name: str
    shortcut: str

    def score(self, ex: PartnerExample) -> np.ndarray:
        """One score per candidate in ``ex.candidate_donor_pos``."""
        ...


class PositionOnlyRival:
    """Shortcut: POSITIONAL LEAK.  On v1 the acceptor sat at position 200 in 100% of
    examples.  With a randomised acceptor offset this must fall to chance."""

    name = "position-only"
    shortcut = "positional leak"

    def __init__(self, mean_donor_pos: float):
        self.mu = mean_donor_pos

    def score(self, ex: PartnerExample) -> np.ndarray:
        return -np.abs(np.asarray(ex.candidates, float) - self.mu)


class GeometryOracleRival:
    """Shortcut: CONSTANT-ANCHOR GEOMETRY.  Reads ZERO nucleotides.  On v1 this scored
    PMR 1.000 -- the true donor sat at a constant offset from the window edge."""

    name = "geometry-oracle"
    shortcut = "constant-anchor geometry"

    def __init__(self, mean_dist_from_end: float):
        self.mu = mean_dist_from_end

    def score(self, ex: PartnerExample) -> np.ndarray:
        d_end = len(ex.sequence) - np.asarray(ex.candidates, float)
        return -np.abs(d_end - self.mu)


class LengthPriorRival:
    """Shortcut: EXON-LENGTH PRIOR.  The single strongest natural shortcut: real exon
    lengths are highly stereotyped.  BELOW chance on the adversarial stratum by
    construction (that stratum is defined as 'the length prior prefers a decoy')."""

    name = "length-prior"
    shortcut = "exon-length prior"

    def __init__(self, prior: LengthPrior):
        self.prior = prior

    def score(self, ex: PartnerExample) -> np.ndarray:
        return np.asarray([self.prior(d - ex.acceptor_pos) for d in ex.candidates], float)


class NearestDownstreamRival:
    """Shortcut: NEAREST-DOWNSTREAM DEGENERACY.  In natural pre-mRNA the true donor is
    almost always the nearest plausible downstream donor -- this rival scores 0.895 on
    the full v2 set, which is exactly why the adversarial stratum exists."""

    name = "nearest-downstream"
    shortcut = "nearest-downstream rule"

    def score(self, ex: PartnerExample) -> np.ndarray:
        return -(np.asarray(ex.candidates, float) - ex.acceptor_pos)


class OracleSiteDetectorRival:
    """Shortcut: SITE DETECTION.  A PERFECT splice-site detector: +1 to any candidate
    that is a real annotated splice site.

    THIS IS THE GUARDRAIL THAT DEFINES v2.  On v1 it scored PMR 1.000 -- proving the
    task never required partner information at all.  On v2 every candidate IS a real
    annotated site, so it is at chance BY CONSTRUCTION.  If it ever rises above chance,
    the dataset has been broken: stop and find the leak."""

    name = "oracle-site-detector"
    shortcut = "site detection"

    def score(self, ex: PartnerExample) -> np.ndarray:
        # every candidate is a real annotated donor -> identical score -> chance
        return np.ones(len(ex.candidates), dtype=float)


class PwmDonorRival:
    """Shortcut: LOCAL MOTIF STRENGTH.  Scores the donor PWM only."""

    name = "pwm-donor"
    shortcut = "local motif strength"

    def __init__(self, donor_pwm, window_fn):
        self.pwm, self.window_fn = donor_pwm, window_fn

    def score(self, ex: PartnerExample) -> np.ndarray:
        return np.asarray(
            [self.pwm.score(self.window_fn(ex.sequence, d, "donor")) for d in ex.candidates],
            float,
        )


class CandidateIndexRival:
    """Shortcut: LIST-ORDER LEAK.  If the true donor always sat at index 0 (as it did in
    the first draft of partner_data), the index alone would be a perfect label.
    Candidates are now permuted with a seed derived from the SEQUENCE, so this is at
    chance."""

    name = "candidate-index"
    shortcut = "list-order leak"

    def score(self, ex: PartnerExample) -> np.ndarray:
        s = np.zeros(len(ex.candidates))
        s[0] = 1.0
        return s


def top1(rival: Rival, examples: Sequence[PartnerExample]) -> float:
    if not examples:
        return float("nan")
    hits = 0
    for ex in examples:
        s = rival.score(ex)
        # ties -> credit only the chance share, never a free win
        best = np.flatnonzero(s == s.max())
        hits += (1.0 / len(best)) if ex.true_index in best else 0.0
    return hits / len(examples)


def chance_level(examples: Sequence[PartnerExample]) -> float:
    if not examples:
        return float("nan")
    return float(np.mean([1.0 / len(ex.candidates) for ex in examples]))


def gene_grouped_bootstrap(
    per_example: Sequence[tuple[str, float]], n: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI resampling GENES (not examples) -- exons of one gene are not independent."""
    rng = np.random.default_rng(seed)
    by_gene: dict[str, list[float]] = {}
    for gene, v in per_example:
        by_gene.setdefault(gene, []).append(v)
    genes = list(by_gene)
    means = np.empty(n)
    for i in range(n):
        pick = rng.choice(len(genes), size=len(genes), replace=True)
        vals = [v for j in pick for v in by_gene[genes[j]]]
        means[i] = float(np.mean(vals))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
