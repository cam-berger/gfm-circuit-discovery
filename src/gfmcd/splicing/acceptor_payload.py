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

from gfmcd.splicing.coupling import CouplingExample, planted_donor_motifs

REFERENCE_ALT = "AC"
TREATMENT_ARM = "GC"
TWIN_ARMS = ("GA", "GG")
LADDER_ARMS = ("AC", "AT", "GA", "GG", "GC")

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
    """
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
