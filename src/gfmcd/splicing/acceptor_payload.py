"""M1 payload-identity primitives for the masked-acceptor readout.

The ladder asks one question: does the model treat a ``GC`` donor as a donor?  ``GC`` is
the only single-base substitution that keeps the ``G``, destroys the ``T``, and yields a
real (minor) splice donor.  Its non-motif twins ``GA`` and ``GG`` are identical in edit
size, edited position, and keeps-G/destroys-T structure, so the contrast isolates motif
identity with no composition or edit-size confound.

``GC`` is absent from ``coupling.DONOR_ALTS`` by construction ("none is GT and none is
GC") and ``planted_donor_motifs`` flags it.  That rule protects the pseudo-donor control
from silently becoming a real donor; here the planted donor IS the treatment, so the
carve-out is scoped to the ``written`` register of the treatment arm alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from gfmcd.splicing.coupling import CouplingExample, planted_donor_motifs

REFERENCE_ALT = "AC"
TREATMENT_ARM = "GC"
TWIN_ARMS = ("GA", "GG")
LADDER_ARMS = ("AC", "AT", "GA", "GG", "GC")

#: Minimum DISTINCT genes a gene-clustered bootstrap may run on.
#:
#: Mirrors ``gfmcd.splicing.paired_component.DEFAULT_MIN_GENES = 30``, the repo's existing
#: cluster floor.  It is restated here rather than imported because that module is not on
#: this branch (it lives on ``claude/distracted-volhard-c466c1``, not on ``master``), and
#: importing it would make this module unimportable.  A decision floor must also not
#: silently change value depending on which branches happen to be merged.  If
#: ``paired_component`` ever lands on ``master``, collapse the two into one constant.
DEFAULT_MIN_GENES = 30

_WRITTEN_MOTIF_TREATMENTS = frozenset({TREATMENT_ARM})


def allows_written_motif(alt: str) -> bool:
    """True only for arms whose planted ``written`` motif is the treatment itself."""
    return alt in _WRITTEN_MOTIF_TREATMENTS


def arm_dispositions(x: CouplingExample, alt: str) -> Mapping[str, str]:
    """Splice motifs created by writing ``alt`` over this example's donor ``GT``.

    Keys are register names (``upstream`` / ``written`` / ``downstream``); an empty
    mapping means the construct plants nothing.  Serialize this per row -- it is the
    audit record for the carve-out.
    """
    return planted_donor_motifs(x.sequence, x.donor_pos, alt)


def arm_admissible(
    x: CouplingExample, alt: str, *, allow_written_motif: bool = False
) -> bool:
    """Is this construct admissible for this arm?

    With ``allow_written_motif=False`` this is exactly ``donor_alt_admissible``.  With it
    set, a motif in the ``written`` register is accepted -- and only there; ``upstream``
    and ``downstream`` violations disqualify on every arm, because a planted ``AG``
    acceptor or a planted flanking ``GT`` is a confound, never a treatment.

    The carve-out is enforced here, not merely by convention in the caller: passing
    ``allow_written_motif=True`` for an arm that ``allows_written_motif`` rejects raises,
    so no caller can waive a genuinely planted motif (e.g. a canonical ``GT`` written by
    ``alt="GT"``) by treating it as this experiment's treatment.
    """
    if allow_written_motif and not allows_written_motif(alt):
        raise ValueError(
            f"allow_written_motif=True is not permitted for alt={alt!r}; the carve-out is "
            f"scoped to {sorted(_WRITTEN_MOTIF_TREATMENTS)}, whose written motif IS the "
            "treatment. For any other arm a written motif is a planted confound."
        )
    planted = arm_dispositions(x, alt)
    disqualifying = {
        register: motif
        for register, motif in planted.items()
        if not (allow_written_motif and register == "written")
    }
    return not disqualifying


def common_admissible_rows(
    exs: Sequence[CouplingExample], arms: Sequence[str]
) -> list[int]:
    """Indices admissible under EVERY arm, so all arms share one row set.

    Measuring each arm on its own admissible subset makes the arms incomparable:
    admissibility runs from 0.37 (``GA``) to 0.842 (``GG``) in the recorded val ladder,
    so per-arm subsets differ in both size and sequence composition.  The contrast is
    only interpretable on rows every arm can construct.

    SCOPE RESTRICTION -- report this with any result computed on this row set.  For
    ``arms == LADDER_ARMS`` the intersection is far narrower than it looks.  An
    exhaustive sweep of the 16 ``(-1 exon base, +3 base)`` flank contexts shows a row
    survives IF AND ONLY IF the ``-1`` base is NOT ``A`` and the ``+3`` base IS ``A`` --
    3 contexts of 16.  The whole analysis is therefore conditioned on strong-consensus
    donors (a ``+3 A``), which is not a random subsample of donors and bounds what any
    verdict generalises to.

    What binds is the TWIN arms, not the ``GC`` carve-out.  ``GA`` excludes ``+3 G`` and
    ``GG`` excludes ``+3 C``/``+3 T``, so together they force ``+3 A``; ``-1 A`` is
    excluded by every ``G``-writing arm alike (it plants an upstream ``AG``).  ``GC``
    with its carve-out is admissible on all 12 non-``A`` ``-1`` contexts and removes
    nothing the twins had not already removed.  The surviving row fraction on a real
    cohort is bounded above by ``GA``'s (0.37 in the recorded val ladder), since the
    common set is a subset of ``GA``'s admissible set; the genomic ``+3`` base is
    A-biased, so it is not the uniform 3/16.
    """
    return [
        i
        for i, x in enumerate(exs)
        if all(
            arm_admissible(x, alt, allow_written_motif=allows_written_motif(alt))
            for alt in arms
        )
    ]


def admissible_fractions(
    exs: Sequence[CouplingExample], arms: Sequence[str]
) -> dict[str, float]:
    """Per-arm admissible fraction of the supplied cohort, for the artifact record.

    Reported rather than assumed: a carve-out that quietly changed ``GC``'s
    admissibility relative to its twins would bias the contrast even on a common row set,
    because the common set would be driven by whichever arm is most restrictive.
    """
    total = len(exs)
    out: dict[str, float] = {}
    for alt in arms:
        allow = allows_written_motif(alt)
        n = sum(1 for x in exs if arm_admissible(x, alt, allow_written_motif=allow))
        out[alt] = (n / total) if total else 0.0
    return out


def rho(v: np.ndarray, ref: np.ndarray) -> float:
    """Ratio of means, i.e. ratio of sums on a common row set.

    Never mean-of-ratios: near-zero denominators make the per-example ratio explode.
    """
    return float(np.sum(v) / max(float(np.sum(ref)), 1e-9))


def pooled_twin_effect(
    per_arm: Mapping[str, np.ndarray], twins: Sequence[str]
) -> np.ndarray:
    """Row-wise mean of the non-motif twin arms, on the common row set."""
    if not twins:
        raise ValueError("at least one twin arm is required")
    return np.mean(np.vstack([np.asarray(per_arm[a], dtype=float) for a in twins]), axis=0)


def gboot_rho_difference(
    v_treatment: np.ndarray,
    v_reference_arm: np.ndarray,
    ref: np.ndarray,
    genes: Sequence[str],
    *,
    n_draws: int = 2000,
    seed: int = 0,
    min_genes: int = DEFAULT_MIN_GENES,
) -> tuple[float, float, float]:
    """``(estimate, lo, hi)`` for ``rho(treatment) - rho(reference_arm)``.

    Both ratios are recomputed inside every draw on the same resampled genes, so their
    correlation is preserved; subtracting two independently bootstrapped intervals would
    overstate the width.  The randomisation unit is the gene, never the row.

    Two refusals, because a degenerate cohort here does not fail loudly -- it manufactures
    a positive finding downstream:

    * Fewer than ``min_genes`` DISTINCT genes raises.  With one cluster every draw is the
      same draw, the interval collapses to zero width, and ``classify_difference`` then
      reads that point as ``EQUIVALENT_WITHIN_SESOI`` while ``precision_gate`` reads it as
      ``PROCEED`` -- a certified equivalence from a single gene.  Lower the floor only
      deliberately, for a test that is ABOUT the degenerate case.
    * A cohort reference sum that is not strictly positive raises.  ``rho``'s ``1e-9``
      floor (the repo's ``circuit.ratio_of_means`` idiom) maps a negative denominator to a
      tiny POSITIVE one, sign-flipping the ratio and inflating it by ~1e9; a single such
      bootstrap draw drags the percentile quantiles arbitrarily.
    """
    t = np.asarray(v_treatment, dtype=float)
    r = np.asarray(v_reference_arm, dtype=float)
    d = np.asarray(ref, dtype=float)
    if not (len(t) == len(r) == len(d) == len(genes)):
        raise ValueError("all arrays and the gene list must be the same length")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")

    by: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        by.setdefault(str(g), []).append(i)
    keys = list(by)
    if len(keys) < min_genes:
        raise ValueError(
            f"gene-clustered bootstrap needs at least {min_genes} distinct genes, got "
            f"{len(keys)}; a degenerate cohort yields a zero-width interval that reads as "
            "EQUIVALENT_WITHIN_SESOI and PROCEED"
        )
    if not float(d.sum()) > 0.0:
        raise ValueError(
            f"the cohort reference sum must be strictly positive, got {float(d.sum())!r}; "
            "a non-positive denominator is sign-flipped by the ratio floor"
        )

    estimate = rho(t, d) - rho(r, d)

    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        picked = rng.choice(len(keys), len(keys), replace=True)
        rows = [row for j in picked for row in by[keys[j]]]
        denom = max(float(d[rows].sum()), 1e-9)
        draws[i] = (t[rows].sum() / denom) - (r[rows].sum() / denom)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return estimate, float(lo), float(hi)


def classify_difference(lo: float, hi: float, sesoi: float) -> str:
    """Strict classification of a difference interval.  Touching a boundary is unresolved."""
    if lo > sesoi:
        return "MATERIAL_POSITIVE"
    if hi < -sesoi:
        return "MATERIAL_NEGATIVE"
    if lo > -sesoi and hi < sesoi:
        return "EQUIVALENT_WITHIN_SESOI"
    return "UNRESOLVED"


def half_width(lo: float, hi: float) -> float:
    """Half the width of a confidence interval."""
    return (float(hi) - float(lo)) / 2.0


def precision_gate(lo: float, hi: float, sesoi: float) -> str:
    """Stop only when this interval settles nothing AND could not have.

    A half-width inside the SESOI is what is needed to CERTIFY EQUIVALENCE; it is not a
    precondition for accepting a material verdict.  An interval that already clears the
    SESOI has answered the question at whatever width it achieved -- ``[+0.06, +0.20]``
    at ``sesoi=0.05`` is ``MATERIAL_POSITIVE`` and wide, and vetoing it would discard a
    real finding for being imprecise about its size.

    So the design is declared unpowerable exactly when the interval is ``UNRESOLVED``
    (it straddles a boundary) AND its half-width is at or above the SESOI (no point
    estimate inside it could have resolved either way).  An ``UNRESOLVED`` interval that
    is nonetheless narrower than the SESOI is a near miss on a workable design, not a
    dead design.
    """
    unresolved = classify_difference(lo, hi, sesoi) == "UNRESOLVED"
    if unresolved and half_width(lo, hi) >= sesoi:
        return "NOT_POWERABLE_ON_THIS_DESIGN"
    return "PROCEED"


def planted_positive_check(
    ref: np.ndarray,
    genes: Sequence[str],
    *,
    planted_rho: float,
    twin_rho: float,
    sesoi: float,
    n_draws: int = 400,
    seed: int = 0,
    noise_frac: float = 0.25,
) -> dict:
    """Sensitivity: a synthetic arm at ``planted_rho`` must classify MATERIAL_NEGATIVE.

    Runs the production estimator and the production classifier, so a decision-layer bug
    fails this check rather than surviving into a verdict.  Falsifiable: with
    ``planted_rho == twin_rho`` there is nothing to detect and PASS must be False.

    The arms carry SEEDED PER-ROW NOISE, at SD ``noise_frac * std(ref)`` and drawn
    independently for the two arms, and this is load-bearing.  Noiseless arms
    (``ref * rho`` exactly) are a zero-variance construction: every bootstrap draw
    returns the identical difference, the measured CI width is ~1e-16, and the check
    degenerates into a test of the classifier's sign convention.  It would then report
    PASS for an estimator resampling rows instead of genes, or computing mean-of-ratios
    instead of ratio-of-sums, because neither defect can move a constant.  With noise the
    interval has real width, so the check asks the question it is meant to ask: does THIS
    cohort have the power to detect an effect of this size?
    """
    d = np.asarray(ref, dtype=float)
    rng = np.random.default_rng(seed)
    noise_sd = float(noise_frac) * float(np.std(d))
    treatment = d * planted_rho + rng.normal(0.0, noise_sd, d.shape)
    twin = d * twin_rho + rng.normal(0.0, noise_sd, d.shape)
    est, lo, hi = gboot_rho_difference(
        treatment, twin, d, genes, n_draws=n_draws, seed=seed
    )
    classification = classify_difference(lo, hi, sesoi)
    return {
        "PASS": classification == "MATERIAL_NEGATIVE",
        "planted_rho": float(planted_rho),
        "twin_rho": float(twin_rho),
        "noise_frac": float(noise_frac),
        "estimate": est,
        "ci": [lo, hi],
        "classification": classification,
    }


def permuted_label_fpr(
    v_treatment: np.ndarray,
    v_twin: np.ndarray,
    ref: np.ndarray,
    genes: Sequence[str],
    *,
    sesoi: float,
    n_permutations: int = 200,
    n_draws: int = 400,
    seed: int = 0,
    alpha: float = 0.05,
    tolerance: float = 0.05,
) -> dict:
    """Specificity: swap the arm labels with one exchangeable sign per gene.

    The arms are measured on the same rows, so under the null "GC behaves like a twin"
    the two labels are exchangeable within a gene.  A material verdict on permuted labels
    is a false positive; the rate should sit near ``alpha``.

    PASS has no LOWER bound by design.  An FPR of 0 is the legitimate reading of a
    conservative test, and requiring a floor would manufacture failures on correct
    behaviour.  Do not add one.
    """
    t = np.asarray(v_treatment, dtype=float)
    w = np.asarray(v_twin, dtype=float)
    d = np.asarray(ref, dtype=float)
    if n_permutations < 1:
        raise ValueError("n_permutations must be positive")
    by: dict[str, list[int]] = {}
    for i, g in enumerate(genes):
        by.setdefault(str(g), []).append(i)
    keys = list(by)
    rng = np.random.default_rng(seed)

    materials = 0
    for p in range(n_permutations):
        swap = rng.choice([False, True], size=len(keys))
        a, b = t.copy(), w.copy()
        for j, do_swap in enumerate(swap):
            if do_swap:
                rows = by[keys[j]]
                a[rows], b[rows] = w[rows], t[rows]
        _, lo, hi = gboot_rho_difference(a, b, d, genes, n_draws=n_draws, seed=seed + p)
        if classify_difference(lo, hi, sesoi).startswith("MATERIAL_"):
            materials += 1

    fpr = materials / n_permutations
    return {
        "PASS": fpr <= alpha + tolerance,
        "fpr": fpr,
        "alpha": alpha,
        "tolerance": tolerance,
        "n_permutations": n_permutations,
    }
