"""ED1 primitives -- architecture-conditioned boundary pairing (exon vs intron definition).

THE GEOMETRY.  One internal exon gives a donor with TWO acceptor partners::

    upstream acceptor -- EXON -- donor -- INTRON -- downstream acceptor
         (a)                      (d)                  (next_acc)
         |<---- L_exon ---->|      |<---- L_down ---->|

``C_exon``   = donor-margin loss when the UPSTREAM acceptor ``a`` is destroyed
               (coupling ACROSS THE EXON -- the exon-definition geometry).
``C_intron`` = donor-margin loss when the DOWNSTREAM acceptor ``next_acc`` is destroyed
               (coupling ACROSS THE INTRON -- the intron-definition geometry).

WHY A RAW ``C_exon`` vs ``C_intron`` DIFFERENCE IS NOT A RESULT.  The two are measured at
different separations (``L_exon`` vs ``L_down``), across different sequence types (exonic
vs intronic), on different sides of the readout.  Every one of those is a confound, and
they are *direction-constant*: they cancel in an architecture x direction INTERACTION but
not in the raw difference.  Hence the estimands here are all interactions.

THE ONE DISTANCE TRAP.  ``L_down`` IS the cross-intron separation, so stratifying on
downstream-intron length is definitionally entangled with distance on the ``C_intron``
side: "C_intron falls when the intron is long" is predicted by plain distance decay.  It
is NOT entangled on the ``C_exon`` side -- ``C_exon``'s separation is ``L_exon``, which is
matched across strata.  That asymmetry is the whole identification strategy: the
distance-clean half of the interaction is the ``C_exon`` half, and it is what the
decision rule is allowed to lean on.

Architecture thresholds are frozen from the classical literature, not from these data:
cross-exon recognition operates out to ~300 nt (Robberson 1990) and systematic constructs
place the intron-definition -> exon-definition transition at ~200-250 nt (Amit 2012).  So
SHORT <= 200 (intron-definition regime), LONG >= 300 (exon-definition regime), with the
201-299 transition band excluded as a buffer.
"""
from __future__ import annotations

from dataclasses import dataclass

from gfmcd.splicing.coupling import CouplingExample

# --- frozen architecture thresholds (nt) -----------------------------------------------
ARCH_SHORT_MAX = 200      # <= this = SHORT flank  (intron-definition regime)
ARCH_LONG_MIN = 300       # >= this = LONG flank   (exon-definition regime)
# 201..299 is the classical transition band and is EXCLUDED from the 2x2.


def gc_fraction(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else float("nan")


def eligible(x: CouplingExample) -> bool:
    """Both partners present, canonical GT donor, upstream intron annotated.

    ``next_acceptor_pos`` is only set when the downstream acceptor falls INSIDE the
    1024-nt window, so the cross-intron arm is structurally capped at ~800 nt.  That is a
    real scope limit (most human introns are longer) and is recorded as such -- but it
    still straddles the 200-250 nt transition the study is about.
    """
    return (
        x.next_acceptor_pos is not None
        and x.intron_length_up is not None
        and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
        and x.sequence[x.acceptor_pos - 2:x.acceptor_pos] == "AG"
        and x.sequence[x.next_acceptor_pos - 2:x.next_acceptor_pos] == "AG"
    )


def _band(n: int | None) -> str | None:
    if n is None:
        return None
    if n <= ARCH_SHORT_MAX:
        return "S"
    if n >= ARCH_LONG_MIN:
        return "L"
    return None          # transition buffer


def architecture_cell(x: CouplingExample) -> str | None:
    """(upstream, downstream) flanking-intron architecture: 'SS' | 'SL' | 'LS' | 'LL'.

    First letter = UPSTREAM intron, second = DOWNSTREAM intron.  ``None`` if either flank
    lands in the 201-299 transition buffer.  The upstream axis is the clean one -- it is
    not a separation for either coupling -- while the downstream axis is the one the
    classical transition is stated in.
    """
    u, d = _band(x.intron_length_up), _band(x.intron_length_down)
    return None if (u is None or d is None) else u + d


@dataclass(frozen=True)
class ShamPolicy:
    """Where a distance-matched non-partner AG sham may sit.

    The sham is the specificity control: same edit DOSE (AG -> a non-AG dinucleotide;
    identical dose does not imply identical effect on model state),
    same side of the donor, comparable separation, but not a real partner.  ``min_sep``
    keeps it clear of the true acceptor's polypyrimidine/branchpoint zone and of the
    donor's own 5'ss context, so a sham cannot damage the very site it is controlling for.
    """

    min_sep: int = 25             # nt clearance from donor, own acceptor, and next acceptor
    max_log2_ratio: float = 1.0   # separation within a factor of 2 of the true partner's
    k: int = 1                    # how many best-matched shams to use (frozen at 1)
    # k=1 keeps the state count at 7 per example.  The sham enters only SECONDARY
    # estimands (S_exon/S_intron, Gamma_sham), so its extra precision is second-order,
    # whereas each additional sham costs 4 more forwards per example on the primary run.


def sham_candidates(x: CouplingExample, direction: str,
                    pol: ShamPolicy = ShamPolicy()) -> tuple[int, float]:
    """(number of admissible shams, |log2 distance ratio| of the best one).

    ``direction`` is ``"exon"`` (upstream of the donor, matched to ``L_exon``) or
    ``"intron"`` (downstream, matched to ``L_down``).
    """
    return _sham_summary(sham_positions(x, direction, pol), x, direction)


def sham_positions(x: CouplingExample, direction: str,
                   pol: ShamPolicy = ShamPolicy()) -> list[int]:
    """Admissible sham AG positions, best-distance-matched first (deterministic)."""
    import math

    d, a, na = x.donor_pos, x.acceptor_pos, x.next_acceptor_pos
    assert na is not None
    true_dist = (d - a) if direction == "exon" else (na - d)
    if true_dist <= 0:
        return []
    out = []
    for p in x.other_ag:
        if direction == "exon" and p >= d:
            continue
        if direction == "intron" and p <= d:
            continue
        if min(abs(p - a), abs(p - d), abs(p - na)) < pol.min_sep:
            continue
        dist = (d - p) if direction == "exon" else (p - d)
        if dist <= 0:
            continue
        r = math.log2(dist / true_dist)
        if abs(r) > pol.max_log2_ratio:
            continue
        out.append((abs(r), p))
    # sort by distance match, then position -> fully deterministic, no RNG, no hash order
    out.sort(key=lambda t: (t[0], t[1]))
    return [p for _, p in out[:pol.k]]


# --- outcome-blind matching ------------------------------------------------------------
#
# The downstream-intron strata are NOT exchangeable as they come out of the genome: short
# introns sit preferentially in GC-rich compact genes (val: gc_intron 0.62 vs 0.54, SMD
# +0.74).  Matching is therefore load-bearing, not decoration, and it runs on ANNOTATION
# ONLY -- no model output is consulted, so it cannot be tuned to the outcome.
#
# `clean` (the native donor margin) is deliberately NOT a matching covariate: it is a
# component of every outcome here (C = f(native) - f(perturbed)), so conditioning on it
# would condition on part of the outcome -- the same collider error IC1 removed in its
# Amendment A2.  Donor motif strength `s_don` is matched instead: exogenous, genomic, and
# the main sequence determinant of the native margin.

BASE_COVARIATES = ("L_exon", "gc_exon", "gc_intron", "s_acc", "s_next", "s_don",
                   "log_iso_don", "log_iso_acc", "donor_pos", "acc_pos")

#: In-window competing REAL sites.  These are deliberately NOT in ``BASE_COVARIATES``.
#:
#: The red-team (M4) flagged in-window site density as the strongest rival account -- "the
#: donor leans less on its upstream acceptor when more real sites compete" would reproduce
#: a positive E_switch with no definition switch -- and prescribed adding them to the
#: primary adjustment set.  That prescription is WRONG and is corrected here (M4'): site
#: density is substantially a DESCENDANT of the exposure (a short downstream intron
#: mechanically pulls the next exon into the window; corr(log L_down, n_real_don) = -0.31),
#: so conditioning on it estimates a CONTROLLED DIRECT EFFECT, not the total architecture
#: effect the primary claims -- and risks collider bias on any unmeasured common cause.
#:
#: They are therefore handled as a preregistered SECONDARY: the primary estimates the total
#: effect and reports density imbalance as an exposure property, while the density-matched
#: secondary adjudicates whether the effect is partner-specific or generic dilution.
DENSITY_KEYS = ("n_real_don", "n_real_acc")

#: Exact-matching on site density costs common support in window position, so the D1 spec
#: adds one caliper to keep the adjustment set balanced (val: worst |SMD| 0.140 -> 0.096,
#: 1,245 pairs / 1,241 genes).
DENSITY_EXTRA_CALIPERS = {"donor_pos": 80.0}


def density_spec(axis: str):
    """(covariates, calipers, exact-keys) for the CONTROLLED-DIRECT-EFFECT secondary."""
    cov, cal = axis_spec(axis)
    return cov, {**cal, **DENSITY_EXTRA_CALIPERS}, EXACT_KEYS[axis] + DENSITY_KEYS

#: Hard per-covariate calipers on the covariates that are imbalanced at source.
#: A pair violating any of these is not formed at all, whatever its overall distance.
BASE_CALIPERS = {"L_exon": 10.0, "gc_intron": 0.03, "gc_exon": 0.05}

#: The OPPOSITE flank is a covariate on each axis (and gets a factor-1.65 caliper), never
#: the exposure.  Calipering the exposed flank -- as an early version of this pilot did --
#: is not conservatism, it deletes the contrast: it collapsed the upstream axis from 3,727
#: possible pairs to 102 by demanding that up-LONG and up-SHORT have similar L_up.
OPPOSITE_FLANK = {"down": "log_L_up", "up": "log_L_down"}
OPPOSITE_FLANK_CALIPER = 0.5

#: Keys that must agree EXACTLY within a matched pair, per axis.  The opposite flank's band
#: keeps a contrast on one architecture axis from smuggling in a shift on the other;
#: ``gene_type`` is exact-matched because short introns are not distributed uniformly
#: across biotypes (red-team M5) -- exact matching keeps the whole cohort, unlike
#: restricting to protein-coding.
EXACT_KEYS = {"down": ("up_band", "gene_type"), "up": ("dn_band", "gene_type")}


def axis_spec(axis: str) -> tuple[tuple[str, ...], dict[str, float]]:
    """(covariates, calipers) for ``axis`` in ``{"down", "up"}`` -- the exposed flank is
    excluded from both, the opposite flank is included in both."""
    if axis not in OPPOSITE_FLANK:
        raise ValueError(f"axis must be 'down' or 'up', got {axis!r}")
    other = OPPOSITE_FLANK[axis]
    return BASE_COVARIATES + (other,), {**BASE_CALIPERS, other: OPPOSITE_FLANK_CALIPER}


def covariate_row(r: dict) -> dict:
    """Derived matching covariates for one cohort row (log-scaled counts/lengths)."""
    import math

    return dict(r, log_iso_don=math.log1p(r["n_iso_don"]),
                log_iso_acc=math.log1p(r["n_iso_acc"]),
                log_L_up=math.log1p(r["L_up"] or 0),
                log_L_down=math.log1p(r["L_down"] or 0))


def match_pairs(rows, treated_mask, *, calipers, covariates, exact=()):
    """Deterministic greedy 1:1 nearest-neighbour matching without replacement.

    Returns ``[(treated_index, control_index), ...]``.  Treated units are processed in a
    frozen order (by ``example_id``), so the result depends on nothing but the covariates
    -- no RNG, no dict/set iteration order, reproducible across processes.

    ``exact`` names row keys that must agree exactly (e.g. the opposite flank's band), so
    a contrast on one architecture axis cannot smuggle in a shift on the other.
    """
    import numpy as np

    cal = calipers
    rr = [covariate_row(r) for r in rows]
    Z = np.array([[float(r[c]) for c in covariates] for r in rr])
    sd = Z.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Zs = (Z - Z.mean(axis=0)) / sd

    t_idx = np.flatnonzero(np.asarray(treated_mask, bool))
    c_idx = np.flatnonzero(~np.asarray(treated_mask, bool))
    order = sorted(t_idx, key=lambda i: rows[i]["example_id"])

    cal_cols = [(covariates.index(k), v) for k, v in cal.items() if k in covariates]
    Craw, Cs = Z[c_idx], Zs[c_idx]
    ex_ctrl = {k: np.array([rows[j][k] for j in c_idx], dtype=object) for k in exact}
    used = np.zeros(len(c_idx), bool)
    out = []
    for i in order:
        ok = ~used
        for k in exact:
            ok &= ex_ctrl[k] == rows[i][k]
        for col, width in cal_cols:
            ok &= np.abs(Craw[:, col] - Z[i, col]) <= width
        if not ok.any():
            continue
        d = np.full(len(c_idx), np.inf)
        d[ok] = np.linalg.norm(Cs[ok] - Zs[i], axis=1)
        j = int(np.argmin(d))
        used[j] = True
        out.append((int(i), int(c_idx[j])))
    return out


# --- dyadic inference ------------------------------------------------------------------
#
# A matched pair joins a TREATED unit from gene g_t and a CONTROL unit from gene g_c, and a
# gene appears as treated in some pairs and control in others.  Pair differences therefore
# share clusters on BOTH ends, so a one-way gene bootstrap under-covers -- the same failure
# that R22/A9b hit with interchange pairs (permutation-on-correlation under dyadic reuse).

def dyadic_bootstrap(values, genes_a, genes_b, *, n_boot: int = 2000, seed: int = 0,
                     alpha: float = 0.05, weights=None):
    """Two-way cluster (multiplier) bootstrap percentile CI for a mean over dyads.

    ``values[p]`` is the statistic for pair ``p``, which loads on clusters ``genes_a[p]``
    and ``genes_b[p]``.

    Uses ADDITIVE mean-zero multipliers: draw ``e_g ~ N(0,1)`` iid per gene and form

        T* = mean_p [ (e_{g_a(p)} + e_{g_b(p)}) / sqrt(2) * (v_p - v_bar) ]

    whose variance is ``(1/n^2) * sum_{p,q} (#genes shared by p and q)/2 * dev_p * dev_q``
    -- i.e. exactly the two-way cluster-robust variance, with the diagonal normalized to
    the iid case by the ``1/sqrt(2)``.  The interval is the basic (pivotal) one.

    NOT a Bayesian-bootstrap weighted mean with Exp(1) weights: averaging two Exp(1)s
    gives ``Var(w) = 0.5``, so that scheme produces intervals a factor ~sqrt(2) too narrow
    and over-rejects at **FPR ~0.19 for alpha=0.05, at every n** (verified by simulation
    at n_pairs = 20 ... 3000 before this was used on any confirmatory data).
    """
    import numpy as np

    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    v = v[ok]
    ga = np.asarray(genes_a, object)[ok]
    gb = np.asarray(genes_b, object)[ok]
    w0 = np.ones(len(v)) if weights is None else np.asarray(weights, float)[ok]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")

    keys = sorted(set(ga.tolist()) | set(gb.tolist()))       # sorted -> hashseed-independent
    idx = {g: i for i, g in enumerate(keys)}
    ia = np.array([idx[g] for g in ga])
    ib = np.array([idx[g] for g in gb])

    mean = float(np.sum(w0 * v) / np.sum(w0))
    dev = w0 * (v - mean) / np.mean(w0)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    # A unit that loads on the SAME cluster twice must get multiplier e (variance 1), not
    # 2e/sqrt(2) (variance 2) -- otherwise a one-way clustered mean passed in as ga==gb, or
    # a genuine within-gene matched pair, gets a CI inflated by sqrt(2).
    same = ia == ib
    inv = 1.0 / np.sqrt(2.0)
    for b in range(n_boot):
        e = rng.standard_normal(len(keys))
        mult = np.where(same, e[ia], (e[ia] + e[ib]) * inv)
        draws[b] = float(np.mean(mult * dev))
    lo_q, hi_q = np.percentile(draws, [100 * (1 - alpha / 2), 100 * alpha / 2])
    return mean, float(mean - lo_q), float(mean - hi_q)


def signflip_p(values, *, n_perm: int = 2000, seed: int = 0) -> float:
    """Two-sided p from the exact within-pair sign-flip randomization null.

    Permuting the architecture label inside a matched pair flips the sign of that pair's
    difference, so this is the randomization null the matched design actually implies.
    """
    import numpy as np

    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan")
    obs = abs(v.mean())
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(n_perm, len(v)))
    return float((np.abs((s * v).mean(axis=1)) >= obs).mean())


def dyadic_bootstrap_p(values, genes_a, genes_b, *, n_boot: int = 2000, seed: int = 0,
                       weights=None) -> float:
    """Two-sided wild-bootstrap p for ``mean(values) == 0``, matching `dyadic_bootstrap`.

    Uses the same pivot draws as the CI, so the p-value and the interval agree.  Its point
    is calibration: with a heavy-tailed statistic the nominal alpha=0.05 CI can under-cover
    (measured 0.087 on the ED1 train pilot), and a p-value can be compared against a
    threshold calibrated on the design's own sign-flip null rather than assumed.
    """
    import numpy as np

    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    v = v[ok]
    ga = np.asarray(genes_a, object)[ok]
    gb = np.asarray(genes_b, object)[ok]
    w0 = np.ones(len(v)) if weights is None else np.asarray(weights, float)[ok]
    if len(v) == 0:
        return float("nan")
    keys = sorted(set(ga.tolist()) | set(gb.tolist()))
    idx = {g: i for i, g in enumerate(keys)}
    ia = np.array([idx[g] for g in ga])
    ib = np.array([idx[g] for g in gb])
    mean = float(np.sum(w0 * v) / np.sum(w0))
    dev = w0 * (v - mean) / np.mean(w0)
    same = ia == ib
    inv = 1.0 / np.sqrt(2.0)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_boot):
        e = rng.standard_normal(len(keys))
        mult = np.where(same, e[ia], (e[ia] + e[ib]) * inv)
        hits += abs(float(np.mean(mult * dev))) >= abs(mean)
    return hits / n_boot


def twoway_wild_p(values, genes_a, genes_b, *, n_perm: int = 2000, seed: int = 0) -> float:
    """Two-sided p from a TWO-WAY wild sign-flip: iid sign per gene, pair multiplier =
    the product of its two genes' signs.

    The frozen fallback if the dyadic multiplier bootstrap fails its permuted-label FPR
    gate.  A pair-level sign-flip treats pairs as independent, which they are not (genes
    recur across pairs); flipping at the gene level and multiplying moves co-clustered
    pairs together, so the null carries the dependence the data actually have.
    """
    import numpy as np

    v = np.asarray(values, float)
    ok = ~np.isnan(v)
    v = v[ok]
    ga = np.asarray(genes_a, object)[ok]
    gb = np.asarray(genes_b, object)[ok]
    if v.size == 0:
        return float("nan")
    keys = sorted(set(ga.tolist()) | set(gb.tolist()))
    idx = {g: i for i, g in enumerate(keys)}
    ia = np.array([idx[g] for g in ga])
    ib = np.array([idx[g] for g in gb])
    # The OBSERVED statistic must be the uncentred mean; centring belongs only in the null
    # deviations.  Centring both makes obs identically 0, so p == 1 always (FPR 0).
    obs = abs(float(np.mean(v)))
    v = v - np.mean(v)
    rng = np.random.default_rng(seed)
    same = ia == ib
    inv = 1.0 / np.sqrt(2.0)
    hits = 0
    for _ in range(n_perm):
        sg = rng.choice([-1.0, 1.0], size=len(keys))
        # ADDITIVE, matching dyadic_bootstrap (same self-pair normalization).  A
        # multiplicative s_a*s_b null is far too coarse -- it collapses toward a single
        # global sign and gives FPR ~0 (uselessly conservative) at every n.
        mult = np.where(same, sg[ia], (sg[ia] + sg[ib]) * inv)
        hits += abs(float(np.mean(mult * v))) >= obs
    return hits / n_perm


def _sham_summary(pos: list[int], x: CouplingExample, direction: str) -> tuple[int, float]:
    import math

    if not pos:
        return 0, float("nan")
    d, a, na = x.donor_pos, x.acceptor_pos, x.next_acceptor_pos
    true_dist = (d - a) if direction == "exon" else (na - d)
    best = pos[0]
    dist = (d - best) if direction == "exon" else (best - d)
    return len(pos), math.log2(dist / true_dist)
