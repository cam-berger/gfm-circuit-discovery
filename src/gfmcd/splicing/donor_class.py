"""Donor-class contrast primitives: real ``GC-AG`` treatment against matched ``GT-AG``.

M1's primary estimand asks whether the masked-acceptor readout depends on the partner donor
being a recognisable splice donor, or only on the canonical ``GT`` bases.  It intervenes on
NATIVE donors -- the ``GC`` of a real ``GC-AG`` intron is destroyed, not written -- so there is
no synthetic construct and no motif-hygiene carve-out.

Two facts make matching mandatory rather than cosmetic.  The classes differ sharply in flanking
context: after the ``GG`` operator filter the ``+3`` base is ``A`` in 94% of ``GC-AG`` rows but
only 50% of ``GT-AG`` rows.  And ``GT`` and ``GC`` share the ``G`` at ``d``, so an alt is
edit-size matched across them only when it ends in ``A`` or ``G``; ``GG`` is the operator of
record because its own admissibility filter is nearly orthogonal to ``+3`` (it excludes only the
rare ``+3 T``/``+3 C``), whereas ``AA`` would exclude ``+3 G`` and so select on the very
confounder under test.

Everything here is outcome-blind by construction: nothing in this module accepts, computes, or
inspects a margin.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from gfmcd.splicing.coupling import CouplingExample

TREATMENT_MOTIF = "GC"
COMPARATOR_MOTIF = "GT"
MATCHED_OPERATOR = "GG"

#: Offsets from ``donor_pos`` of the flank bases the two classes differ on.
FLANK_OFFSETS: tuple[tuple[str, int], ...] = (("-1", -1), ("+3", 2), ("+5", 4))


def donor_motif(x: CouplingExample) -> str:
    """The native donor dinucleotide at ``[donor_pos, donor_pos+2)``."""
    return x.sequence[x.donor_pos:x.donor_pos + 2]


def flank_key(x: CouplingExample) -> tuple[str, ...]:
    """The discrete flank stratum this row belongs to, as ``(-1, +3, +5)`` bases."""
    return tuple(x.sequence[x.donor_pos + off] for _, off in FLANK_OFFSETS)


def _gap(x: CouplingExample) -> int:
    if x.next_acceptor_pos is None:
        raise ValueError(f"{x.example_id} has no intron-mate acceptor")
    return int(x.next_acceptor_pos - x.donor_pos)


@dataclass(frozen=True)
class MatchedCohort:
    """A treatment arm and its matched comparators, plus what matching discarded."""

    treatment: list[CouplingExample] = field(default_factory=list)
    comparators: list[CouplingExample] = field(default_factory=list)
    pairs: list[tuple[str, list[str]]] = field(default_factory=list)
    unmatched_treatment: list[str] = field(default_factory=list)
    k: int = 1


def isoform_bin(x: CouplingExample) -> str:
    """Coarsened donor transcript support, a proxy for constitutive-vs-alternative use.

    GC-AG introns are markedly less supported than GT-AG (median 4 donor isoforms against
    12 on the train cohort), and coupling strength rises with support, so leaving this
    uncontrolled confounds donor class with intron quality.  Bins are coarse and fixed in
    advance rather than fitted, so they cannot be tuned to an outcome.
    """
    n = int(x.n_isoforms_donor or 0)
    if n <= 2:
        return "iso<=2"
    if n <= 5:
        return "iso3-5"
    if n <= 11:
        return "iso6-11"
    return "iso12+"


def flank_and_support_key(x: CouplingExample) -> tuple[str, ...]:
    """Flank stratum plus donor transcript-support bin."""
    return flank_key(x) + (isoform_bin(x),)


def match_comparators(
    treatment: Sequence[CouplingExample],
    pool: Sequence[CouplingExample],
    *,
    k: int = 1,
    gap_tol: int | None = None,
    key_fn=None,
) -> MatchedCohort:
    """Match each treatment row to ``k`` comparators sharing its exact flank stratum.

    Within a stratum, comparators are chosen by nearest donor-acceptor gap; ties break on
    ``example_id`` so the result is deterministic without a seed.  A comparator is consumed
    once and never reused -- reuse would duplicate rows into the gene-clustered bootstrap and
    understate variance.  A treatment row with no admissible partner is DROPPED and recorded in
    ``unmatched_treatment``; it is never silently retained on an unmatched footing.

    ``gap_tol`` optionally caps how far a partner's gap may sit from the treatment row's.
    """
    if k < 1:
        raise ValueError("k must be positive")
    if gap_tol is not None and gap_tol < 0:
        raise ValueError("gap_tol must be non-negative when supplied")

    key = key_fn or flank_key
    by_stratum: dict[tuple[str, ...], list[CouplingExample]] = defaultdict(list)
    for candidate in pool:
        by_stratum[key(candidate)].append(candidate)
    for rows in by_stratum.values():
        rows.sort(key=lambda r: r.example_id)

    used: set[str] = set()
    kept_treatment: list[CouplingExample] = []
    kept_comparators: list[CouplingExample] = []
    pairs: list[tuple[str, list[str]]] = []
    unmatched: list[str] = []

    for x in sorted(treatment, key=lambda r: r.example_id):
        target = _gap(x)
        candidates = [
            c for c in by_stratum.get(key(x), ())
            if c.example_id not in used
            and (gap_tol is None or abs(_gap(c) - target) <= gap_tol)
        ]
        if len(candidates) < k:
            unmatched.append(x.example_id)
            continue
        candidates.sort(key=lambda c: (abs(_gap(c) - target), c.example_id))
        chosen = candidates[:k]
        used.update(c.example_id for c in chosen)
        kept_treatment.append(x)
        kept_comparators.extend(chosen)
        pairs.append((x.example_id, [c.example_id for c in chosen]))

    return MatchedCohort(
        treatment=kept_treatment,
        comparators=kept_comparators,
        pairs=pairs,
        unmatched_treatment=unmatched,
        k=k,
    )


def matching_balance(cohort: MatchedCohort) -> dict:
    """Realized flank composition of both arms after matching, for the artifact.

    Reported rather than assumed: exact stratum matching should drive every entry to
    agreement, so any residual imbalance here is a defect in the matcher and must surface.
    """

    def profile(rows: Sequence[CouplingExample]) -> dict:
        out: dict = {}
        total = max(len(rows), 1)
        for label, off in FLANK_OFFSETS:
            counts = {b: 0 for b in "ACGT"}
            for r in rows:
                base = r.sequence[r.donor_pos + off]
                counts[base] = counts.get(base, 0) + 1
            out[label] = {b: counts[b] / total for b in counts}
        return out

    return {
        "n_treatment": len(cohort.treatment),
        "n_comparator": len(cohort.comparators),
        "n_unmatched_treatment": len(cohort.unmatched_treatment),
        "k": cohort.k,
        "treatment": profile(cohort.treatment),
        "comparator": profile(cohort.comparators),
        "matched_on": [label for label, _ in FLANK_OFFSETS] + ["donor_acceptor_gap"],
        "outcome_blind": True,
    }


def gboot_two_sample_ratio(
    v_treatment,
    genes_treatment: Sequence[str],
    v_comparator,
    genes_comparator: Sequence[str],
    *,
    n_draws: int = 2000,
    seed: int = 0,
    min_genes: int = 30,
):
    """``(estimate, lo, hi)`` for ``mean(treatment) / mean(comparator)``, clustered by gene.

    The two arms live on different rows, so the paired estimator in ``acceptor_payload`` does
    not apply.  They are nonetheless correlated through the gene -- most treatment genes also
    carry a comparator intron -- so resampling the arms independently would break that
    correlation and misstate the interval.  Genes are therefore drawn once, with replacement,
    from the UNION of both arms' genes, and each drawn gene contributes its rows to BOTH sums.

    Ratio of sums throughout, never a mean of per-row ratios: near-zero denominators make the
    per-example ratio explode.
    """
    import numpy as _np

    t = _np.asarray(v_treatment, dtype=float)
    c = _np.asarray(v_comparator, dtype=float)
    if len(t) != len(genes_treatment) or len(c) != len(genes_comparator):
        raise ValueError("each arm's values and gene labels must be the same length")
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    if not _np.isfinite(t).all() or not _np.isfinite(c).all():
        raise ValueError("values contain non-finite entries")

    t_rows: dict[str, list[int]] = {}
    for i, g in enumerate(genes_treatment):
        t_rows.setdefault(str(g), []).append(i)
    c_rows: dict[str, list[int]] = {}
    for i, g in enumerate(genes_comparator):
        c_rows.setdefault(str(g), []).append(i)

    keys = sorted(set(t_rows) | set(c_rows))
    if len(keys) < min_genes:
        raise ValueError(
            f"at least {min_genes} independent gene clusters are required; got {len(keys)}"
        )
    denom = float(c.mean())
    if not denom > 0.0:
        raise ValueError("the comparator arm's mean must be strictly positive")
    estimate = float(t.mean()) / denom

    # Ratio of MEANS, not of sums: the comparator arm is k:1, so its total is inflated by k.
    # Sums coincide with means only for equal-size arms, and k>1 would divide rho by k.
    t_sum = _np.array([t[t_rows[k]].sum() if k in t_rows else 0.0 for k in keys])
    c_sum = _np.array([c[c_rows[k]].sum() if k in c_rows else 0.0 for k in keys])
    t_cnt = _np.array([float(len(t_rows[k])) if k in t_rows else 0.0 for k in keys])
    c_cnt = _np.array([float(len(c_rows[k])) if k in c_rows else 0.0 for k in keys])

    rng = _np.random.default_rng(seed)
    draws = _np.empty(n_draws)
    for i in range(n_draws):
        picked = rng.integers(0, len(keys), len(keys))
        nt, nc = t_cnt[picked].sum(), c_cnt[picked].sum()
        if nt <= 0 or nc <= 0:
            draws[i] = _np.nan
            continue
        d = c_sum[picked].sum() / nc
        draws[i] = (t_sum[picked].sum() / nt) / d if d > 0 else _np.nan
    draws = draws[_np.isfinite(draws)]
    if draws.size == 0:
        raise ValueError("every bootstrap draw had a non-positive comparator sum")
    lo, hi = _np.quantile(draws, [0.025, 0.975])
    return estimate, float(lo), float(hi)


#: Chromosomes a train-split donor-class assay may read. Derived once from
#: ``CouplingConfig`` rather than hardcoded, so it cannot drift from the split definition.
def train_chromosomes() -> tuple[str, ...]:
    """Every chromosome outside the validation and SEALED test splits.

    Passing this to ``circuit._dataset(..., allowed_chromosomes=...)`` is what actually
    preserves the seal.  Filtering a completed all-chromosome dataset by split does NOT:
    ``build_coupling_dataset`` has already fetched chr8/chr9 sequence from FASTA by then, so
    a runner that filters afterwards has read the sealed split and any
    ``sealed_chr8_chr9_opened: False`` it writes is false.
    """
    from gfmcd.splicing.coupling import CouplingConfig

    cfg = CouplingConfig()
    every = tuple(f"chr{i}" for i in range(1, 23)) + ("chrX", "chrY")
    reserved = set(cfg.val_chromosomes) | set(cfg.test_chromosomes)
    return tuple(c for c in every if c not in reserved)


def assert_seal_intact(dataset) -> int:
    """Fail loudly if any SEALED-split row was materialised. Returns the row count checked.

    Checks ``chrom`` against the sealed chromosomes on the RAW dataset. A check written
    against the output of ``eligible(D, "train")`` is a tautology -- that helper filters to
    the requested split first, so no sealed row can survive to be tested.
    """
    from gfmcd.splicing.coupling import CouplingConfig

    sealed = set(CouplingConfig().test_chromosomes)
    rows = list(dataset)
    leaked = sorted({x.chrom for x in rows if x.chrom in sealed})
    if leaked:
        raise AssertionError(
            f"SEAL LEAK: sealed chromosomes materialised in the cohort: {leaked}. "
            "Pass allowed_chromosomes=train_chromosomes() to _dataset."
        )
    return len(rows)
