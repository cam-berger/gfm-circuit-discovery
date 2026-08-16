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

from collections.abc import Mapping

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
