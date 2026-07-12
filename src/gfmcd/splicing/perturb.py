"""Counterfactual sequence edits for partner-aware splice-site assays.

Every function returns a new string and leaves the input untouched.  The study
stores the edit recipe alongside each generated conflict example; an edit is not
accepted merely because it changes model behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Literal

from gfmcd.splicing.schema import SplicePairExample


@dataclass(frozen=True)
class SequenceEdit:
    name: str
    start: int
    end: int
    before: str
    after: str
    note: str = ""

    def apply(self, sequence: str) -> str:
        if sequence[self.start : self.end] != self.before:
            raise ValueError(
                f"edit precondition failed for {self.name}: expected {self.before!r}, "
                f"found {sequence[self.start:self.end]!r}"
            )
        if len(self.before) != len(self.after):
            raise ValueError("study edits must preserve sequence length")
        return sequence[: self.start] + self.after + sequence[self.end :]


def ablate_true_site(example: SplicePairExample, kind: Literal["acceptor", "donor"]) -> tuple[str, SequenceEdit]:
    """Destroy the canonical dinucleotide with a one-base matched edit."""
    pair = example.true_pair
    if kind == "acceptor":
        start = pair.acceptor_pos - 2
        before = example.sequence[start : start + 2]
        after = before[0] + ("C" if before[1] != "C" else "T")
    else:
        start = pair.donor_pos
        before = example.sequence[start : start + 2]
        after = ("C" if before[0] != "C" else "A") + before[1]
    edit = SequenceEdit(
        name=f"ablate_true_{kind}",
        start=start,
        end=start + 2,
        before=before,
        after=after,
        note="one-base disruption of the true canonical splice dinucleotide",
    )
    return edit.apply(example.sequence), edit


def dinucleotide_shuffle(sequence: str, seed: int = 0) -> str:
    """Return an Eulerian shuffle preserving every directed dinucleotide count.

    The original first and last nucleotides are retained.  Very short or
    homopolymer sequences may have only one valid shuffle and are returned
    unchanged.
    """
    if len(sequence) < 3:
        return sequence
    rng = random.Random(seed)
    adjacency: dict[str, list[str]] = {}
    for a, b in zip(sequence, sequence[1:]):
        adjacency.setdefault(a, []).append(b)
    for edges in adjacency.values():
        rng.shuffle(edges)

    stack = [sequence[0]]
    trail: list[str] = []
    while stack:
        node = stack[-1]
        edges = adjacency.get(node)
        if edges:
            stack.append(edges.pop())
        else:
            trail.append(stack.pop())
    shuffled = "".join(reversed(trail))
    if len(shuffled) != len(sequence):
        raise RuntimeError("dinucleotide shuffle failed to consume every edge")
    # A path constructed from the same directed edge multiset has the same
    # degree imbalance and therefore the same endpoints.
    if shuffled[0] != sequence[0] or shuffled[-1] != sequence[-1]:
        raise RuntimeError("dinucleotide shuffle changed path endpoints")
    return shuffled


def shuffle_exon_core(
    example: SplicePairExample,
    *,
    protect: int = 12,
    seed: int = 0,
) -> tuple[str, SequenceEdit]:
    """Shuffle the exon interior while preserving local site neighborhoods."""
    pair = example.true_pair
    start = pair.acceptor_pos + protect
    end = pair.donor_pos - protect
    if end - start < 3:
        raise ValueError("exon core too short after protected site neighborhoods")
    before = example.sequence[start:end]
    after = dinucleotide_shuffle(before, seed=seed)
    edit = SequenceEdit(
        name="shuffle_exon_core_dinucleotide",
        start=start,
        end=end,
        before=before,
        after=after,
        note=f"protects {protect} nt around each splice boundary",
    )
    return edit.apply(example.sequence), edit


def swap_site_neighborhood(
    recipient: SplicePairExample,
    donor_example: SplicePairExample,
    *,
    kind: Literal["acceptor", "donor"],
    flank: int = 12,
) -> tuple[str, SequenceEdit]:
    """Swap an equal-width local site neighborhood between aligned examples.

    This preserves the recipient's sequence length and candidate coordinate.  It
    is intended for matched examples with the same true acceptor/donor positions;
    matching is enforced rather than assumed.
    """
    rp = recipient.true_pair
    dp = donor_example.true_pair
    rpos = rp.acceptor_pos if kind == "acceptor" else rp.donor_pos
    dpos = dp.acceptor_pos if kind == "acceptor" else dp.donor_pos
    if rpos != dpos or len(recipient.sequence) != len(donor_example.sequence):
        raise ValueError("site-swap examples must be length- and anchor-aligned")
    start = rpos - flank
    end = rpos + flank
    if start < 0 or end > len(recipient.sequence):
        raise ValueError("site neighborhood exceeds sequence boundaries")
    before = recipient.sequence[start:end]
    after = donor_example.sequence[start:end]
    edit = SequenceEdit(
        name=f"swap_{kind}_neighborhood",
        start=start,
        end=end,
        before=before,
        after=after,
        note=f"matched {2*flank}-nt boundary swap",
    )
    return edit.apply(recipient.sequence), edit


def pair_swap_quadruplet(
    a: SplicePairExample,
    b: SplicePairExample,
    *,
    flank: int = 12,
) -> dict[str, str]:
    """Construct AA, AB, BA, BB boundary pairings from aligned natural exons.

    ``AB`` keeps A's acceptor neighborhood and swaps in B's donor; ``BA`` does
    the converse.  The natural sequences ``AA`` and ``BB`` are retained as
    controls.  This edit is a secondary assay, not the primary natural conflict
    set, because chimeric sequence joins can introduce their own artifacts.
    """
    if len(a.sequence) != len(b.sequence):
        raise ValueError("pair-swap examples must have equal sequence length")
    if (a.true_pair.acceptor_pos, a.true_pair.donor_pos) != (
        b.true_pair.acceptor_pos,
        b.true_pair.donor_pos,
    ):
        raise ValueError("pair-swap examples must have aligned true boundaries")
    ab, _ = swap_site_neighborhood(a, b, kind="donor", flank=flank)
    ba, _ = swap_site_neighborhood(b, a, kind="donor", flank=flank)
    return {"AA": a.sequence, "AB": ab, "BA": ba, "BB": b.sequence}


def boundary_pair_factorial(
    background: SplicePairExample,
    source_a: SplicePairExample,
    source_b: SplicePairExample,
    *,
    flank: int = 12,
) -> dict[str, str]:
    """Build AA/AB/BA/BB acceptor×donor combinations on one background.

    All three examples must have equal length and aligned true boundaries.  The
    four outputs therefore share token coordinates, making them suitable for a
    non-additivity assay and later activation patching.  These are chimeric
    controls and must be reported separately from natural conflict examples.
    """
    examples = (background, source_a, source_b)
    if len({len(x.sequence) for x in examples}) != 1:
        raise ValueError("factorial examples must have equal sequence length")
    anchors = {
        (x.true_pair.acceptor_pos, x.true_pair.donor_pos)
        for x in examples
    }
    if len(anchors) != 1:
        raise ValueError("factorial examples must have aligned true boundaries")
    acceptor, donor = next(iter(anchors))
    if acceptor + flank > donor - flank:
        raise ValueError("acceptor and donor swap windows overlap")

    def window(example: SplicePairExample, pos: int) -> str:
        start, end = pos - flank, pos + flank
        if start < 0 or end > len(example.sequence):
            raise ValueError("boundary window exceeds sequence")
        return example.sequence[start:end]

    a_acceptor = window(source_a, acceptor)
    b_acceptor = window(source_b, acceptor)
    a_donor = window(source_a, donor)
    b_donor = window(source_b, donor)

    def combine(acc: str, don: str) -> str:
        seq = list(background.sequence)
        seq[acceptor - flank : acceptor + flank] = acc
        seq[donor - flank : donor + flank] = don
        return "".join(seq)

    return {
        "AA": combine(a_acceptor, a_donor),
        "AB": combine(a_acceptor, b_donor),
        "BA": combine(b_acceptor, a_donor),
        "BB": combine(b_acceptor, b_donor),
    }
