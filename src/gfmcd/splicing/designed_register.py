"""Designed-register counterfactuals: a model-free, length-preserving rotation operator.

A "designed register" counterfactual holds an authentic exon module fixed except for the
codon register of its interior relative to the module boundary. Three versions are built —
register 0 (native), +1, and -1 — by circularly rotating the module's *interior* while its
protected flanks (splice core, branchpoint/polypyrimidine region) stay byte-identical.

GOVERNING RULE (absolute, per the study protocol): engineer the intended register
relationship, not variants selected for producing an intended model response. Every
function in this module is a pure, deterministic string transform. None of them accepts a
model, a score, a logit, or a label as input, and none of them contains a conditional
branch, loop-until, or retry driven by an external signal. There is no rejection sampling
and no selection here — ``shift_module`` always returns exactly one output for one input,
and ``register_triplet`` always returns exactly one triplet for one ``k``. If a variant
needs to be discarded because of what it does to a model, that decision belongs to a
different, model-aware module entirely; it must never be folded into this one.

WHY A CIRCULAR ROTATION OF THE INTERIOR. Substituting bases (as the donor/acceptor
corruption operators in ``coupling.py`` and ``reverse_coupling.py`` do) changes composition,
which is exactly what a register-only counterfactual must avoid — a register shift should
change *where* bases sit relative to the boundary, not *which* bases are present at all. A
circular rotation is the simplest length-preserving permutation with that property: it
changes every interior base's position (generically), changes none of their identities, and
is invertible (rotate back by ``len(interior) - shift``). Two further choices follow from
that: shifts are shared across a register triplet only through ``k`` in
``register_triplet``, so the +1/0/-1 arms differ from each other by exactly one codon
position and nothing else (see that function's docstring); and the rotation point is
"circular" rather than a simple slide, because a slide would have to either shrink the
interior or invent bases at one edge, both of which reintroduce the exact composition change
a rotation is meant to avoid.

SEAMS AND R44. ``CORRECTIONS.md`` R44 established, for the donor/acceptor *substitution*
operator, that an edit at a locus can create a splice motif (a ``GT`` or ``AG``) that was
present on neither side of the edit before it — the confound lives in what the edit
CREATES, not in what it is documented to WRITE, and it was found only because someone
checked the coordinate the edit touched against what had been there before. A circular
rotation is not a substitution, but it is not exempt from this either: it manufactures THREE
junctions — the 5' flank/interior join, the interior's own wrap point (where the original
last base and original first base of the interior become neighbors), and the
interior/3' flank join — and any of the three can juxtapose two bases that were never
adjacent before, at a coordinate that may or may not have held a motif previously. The
protected-flank joins are seams exactly as much as the internal wrap point is, and because
they sit right next to sequence nobody edited, they are the ones most likely to be
forgotten. ``seam_census`` checks all three, plus every other coordinate in the module,
because R44's other lesson was that a motif can appear away from the edited site once a
block of sequence is relocated to sit at a new absolute coordinate.
"""

from __future__ import annotations

#: The two splice motifs a junction can newly present: the donor GT, the acceptor AG.
_SPLICE_MOTIFS = ("GT", "AG")


def shift_module(module: str, *, shift: int, protect_5p: int, protect_3p: int) -> str:
    """Circularly rotate the interior of ``module`` by ``shift``, flanks untouched.

    ``module[:protect_5p]`` and ``module[len(module) - protect_3p:]`` are the protected
    flanks and are copied through byte-identically. The interior,
    ``module[protect_5p : len(module) - protect_3p]``, is rotated circularly:
    ``interior[shift:] + interior[:shift]`` (after reducing ``shift`` modulo the interior's
    length, so ``shift == len(interior)`` and ``shift == 0`` both leave it unchanged, and a
    negative ``shift`` rotates the other way rather than raising).

    This function is model-free and deterministic: same arguments, same output, always.
    """
    if protect_5p < 0 or protect_3p < 0:
        raise ValueError(
            f"protect_5p ({protect_5p}) and protect_3p ({protect_3p}) must be non-negative"
        )
    if protect_5p + protect_3p > len(module):
        raise ValueError(
            f"protected flanks (protect_5p={protect_5p} + protect_3p={protect_3p}) exceed "
            f"module length ({len(module)})"
        )

    flank_5p = module[:protect_5p]
    flank_3p = module[len(module) - protect_3p :]
    interior = module[protect_5p : len(module) - protect_3p]

    if interior:
        shift_mod = shift % len(interior)
        interior = interior[shift_mod:] + interior[:shift_mod]

    return flank_5p + interior + flank_3p


def register_triplet(k: int) -> tuple[int, int, int]:
    """The three shifts ``(3k, 3k+1, 3k+2)`` for an edit-matched register-0/+1/-1 triplet.

    ``k`` must be shared across the three versions built for one module. Sharing it is what
    makes the triplet edit-matched: the three shifts differ from each other by at most 2, so
    the wrap point ``shift_module`` introduces and the number of interior bases it displaces
    are (to within that same +/-2) the same for all three arms — the only thing that differs
    is which codon position each interior base lands on relative to the boundary. Pick two
    unrelated ``k`` values for the three arms instead, and the register-0 arm stops being an
    edit-matched control for the other two: it would carry a different wrap location and a
    different displacement magnitude, so any behavioral difference between arms could come
    from the edit's mechanics rather than from the register relationship under study.

    ``k = 0`` is rejected rather than silently accepted: it collapses the register-0 shift to
    0, which is the identity operation (``shift_module`` returns ``module`` unchanged) rather
    than a rotation that merely happens to preserve register. An identity register-0 arm is
    not edit-matched to the +1/-1 arms at all — it carries none of the wrap-point or
    Hamming-distance cost they do — which defeats the entire purpose of using ``3*k`` instead
    of ``0`` for the zero-register shift. Negative ``k`` is rejected for the same reason: it
    does not produce a well-defined "edit cost" relative to a ``k>0`` triplet built for the
    same module.
    """
    if k <= 0:
        raise ValueError(
            f"k must be a positive integer, got {k!r}. k=0 makes the register-0 shift an "
            "identity (no rotation at all), which is not edit-matched to the +1/-1 arms; "
            "k<0 is rejected for the same reason."
        )
    return (3 * k, 3 * k + 1, 3 * k + 2)


def _dinucleotide(seq: str, position: int) -> str | None:
    """The 2-mer straddling ``position`` (bases at ``position - 1`` and ``position``).

    Returns ``None`` when ``position`` is too close to either edge of ``seq`` for both bases
    to exist.
    """
    if position < 1 or position >= len(seq):
        return None
    return seq[position - 1 : position + 1]


def _classify_motif(dinucleotide: str | None) -> str | None:
    """``dinucleotide`` if it is a splice motif (``GT`` or ``AG``), else ``None``."""
    if dinucleotide in _SPLICE_MOTIFS:
        return dinucleotide
    return None


def _junction_report(original: str, shifted: str, position: int) -> dict:
    """Position, local context, and before/after motif status for one junction."""
    lo = max(0, position - 3)
    hi = min(len(shifted), position + 3)

    before = _dinucleotide(original, position)
    after = _dinucleotide(shifted, position)
    after_motif = _classify_motif(after)
    # "Created" means: a splice motif sits at this coordinate now, and this exact
    # coordinate did not carry that same motif before the edit. This is the R44
    # methodology applied to a rotation instead of a substitution -- compare the same
    # absolute coordinate before and after, not whether the two bases were ever adjacent
    # in the original sequence. That distinction matters here: a rotation can relocate an
    # already-adjacent GT/AG pair, unedited relative to each other, to a new coordinate
    # that previously held something else -- R44's "created away from the edit site" case,
    # not merely "created by joining two bases that were never neighbors".
    created = after_motif if (after_motif is not None and before != after_motif) else None

    return {
        "position": position,
        "context_6mer": shifted[lo:hi],
        "dinucleotide_before": before,
        "dinucleotide_after": after,
        "created_motif": created,
    }


def seam_census(
    original: str, shifted: str, *, protect_5p: int, protect_3p: int, shift: int
) -> dict:
    """Report every junction a rotation can introduce, and whether it planted a motif.

    A circular rotation of the interior creates three junctions in the reassembled module:

    * ``flank5_interior`` -- where the untouched 5' flank meets the (now rotated) interior.
    * ``internal_wrap`` -- where the interior's own wrap point falls: the original LAST
      interior base and the original FIRST interior base become neighbors here, for any
      ``shift`` that is not a multiple of the interior's length.
    * ``interior_flank3`` -- where the (rotated) interior meets the untouched 3' flank.

    Each junction is reported as its 0-based position in ``shifted`` (the index of the base
    immediately downstream of the join), a 6-mer of local context centered on the join, the
    dinucleotide straddling it before and after, and which splice motif (if any) the edit
    newly planted there (see ``_junction_report``).

    Beyond the three named junctions, ``created_motif_count`` scans every coordinate in the
    module: R44 established that an edit can create a motif away from the site it touches,
    and a rotation can do the same by relocating an intact block of sequence to a new
    absolute position. ``hamming_distance`` is the total number of positions that changed;
    lengths are equal by construction, so this is always well-defined.
    """
    if len(original) != len(shifted):
        raise ValueError(
            f"original (len {len(original)}) and shifted (len {len(shifted)}) must be the "
            "same length -- shift_module is length-preserving by construction, so a "
            "mismatch here means these two strings do not describe one rotation"
        )

    module_len = len(shifted)
    if protect_5p < 0 or protect_3p < 0 or protect_5p + protect_3p > module_len:
        raise ValueError(
            f"protect_5p ({protect_5p}) and protect_3p ({protect_3p}) are not a valid split "
            f"of a module of length {module_len}"
        )

    interior_len = module_len - protect_5p - protect_3p
    shift_mod = (shift % interior_len) if interior_len else 0

    flank5_interior_pos = protect_5p
    interior_flank3_pos = module_len - protect_3p
    internal_wrap_pos = protect_5p + (interior_len - shift_mod)

    junctions = {
        "flank5_interior": _junction_report(original, shifted, flank5_interior_pos),
        "internal_wrap": _junction_report(original, shifted, internal_wrap_pos),
        "interior_flank3": _junction_report(original, shifted, interior_flank3_pos),
    }

    hamming_distance = sum(1 for a, b in zip(original, shifted) if a != b)

    created_motif_count = 0
    for i in range(module_len - 1):
        after = shifted[i : i + 2]
        if after in _SPLICE_MOTIFS and original[i : i + 2] != after:
            created_motif_count += 1

    return {
        "junctions": junctions,
        "hamming_distance": hamming_distance,
        "created_motif_count": created_motif_count,
    }
