"""Reverse-direction partner coupling: mask the ACCEPTOR, corrupt the partner DONOR.

Every measurement in ``studies/splicebert_partner_selection`` masks the donor and corrupts
the intron-mate acceptor.  This module is the mirror -- the same intron, read from the other
end.  It is a role swap, not a new task: the readout is the same masked-G margin
(``CouplingTask.margin`` scores ``logit(G) - max(A, C, T)``) because the acceptor's masked
base is the ``G`` of its ``AG``, exactly as the donor's is the ``G`` of its ``GT``.

    forward (C1)   mask ``donor_pos``            corrupt ``[p-2, p)``   ->  Y_D
    reverse (here) mask ``next_acceptor_pos-1``  corrupt ``[d, d+2)``   ->  Y_A

THE OPERATOR IS NOT FREE TO CHOOSE.  R44 established that ``seq[d-1] == "G"`` at 81.0% of
validation donors (the 5'SS consensus), so a donor substitution can destroy its target and
*create* a motif one base away: ``CC`` plants a ``GC`` donor, ``TA`` plants a canonical
``GT`` (95.5% on the atlas condition-B cohort), and ``AA`` plants an ``AG`` (62.0%). ``AC``
and ``AT`` are universally motif-clean; ``AC`` is the clean two-substitution operator and
``AT`` the clean one-substitution operator (D8). C42's acceptor-side alt-invariance does **not** transfer
to this direction -- AR1's ``DONA`` arm returned ``NOT_INVARIANT``.  ``MOTIF_CLEAN_DONOR_ALT``
is therefore the default, and every construct is re-verified in its own flanking context by
``planted_donor_motifs`` rather than trusted from a constant.

EDIT SIZE IS A FACTOR, NOT A NUISANCE.  ``GT -> AC`` is two substitutions; ``GT -> AT`` is
one, and leaves the ``T`` at ``d+1`` standing.  AR1's ``DONA`` arm (rho ~ 0.62 for every alt
with ``z[1] == "T"``) and E11's reciprocal assay (``Retention_A`` 3% at two substitutions,
22% at one) reached that dose-response independently, from different cohorts.  A symmetry
statement that does not match substitution counts across the two directions is measuring the
operator, not the computation -- ``edit_size`` and ``matched_operators`` exist to make that
mismatch impossible to introduce silently.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from gfmcd.splicing.coupling import (
    MOTIF_CLEAN_DONOR_ALT,
    CouplingExample,
    planted_donor_motifs,
    substitute,
)

#: The forward direction's operator, retained by R44 and validated alt-invariant by C42.
FORWARD_ACCEPTOR_ALT = "CC"

#: Substitution counts that match across the two directions (``donor_alt``, ``acceptor_alt``).
#: ``GT -> AC`` (2) against ``AG -> CC`` (2) is E11's own named successor arm.
MATCHED_TWO_SUB = ("AC", "CC")


def edit_size(motif: str, alt: str) -> int:
    """How many bases a substitution actually changes.

    ``GT -> AT`` reports 1: the ``T`` at ``d+1`` is rewritten with itself, so the donor is
    only half destroyed.  This is the quantity E11's memorial was missing -- its test
    asserted the edit moved ``<= 2`` bases, which a half-edit passes.
    """
    if len(motif) != len(alt):
        raise ValueError(f"motif {motif!r} and alt {alt!r} differ in length")
    return sum(1 for m, a in zip(motif, alt) if m != a)


def matched_operators(donor_alt: str, acceptor_alt: str) -> bool:
    """True when the two directions' operators change the same number of bases."""
    return edit_size("GT", donor_alt) == edit_size("AG", acceptor_alt)


def acceptor_readout(x: CouplingExample) -> int:
    """Position of the ``G`` in the intron-mate acceptor's ``AG``.

    ``next_acceptor_pos`` is the first exon base and the ``AG`` occupies ``[p-2, p)``, so the
    ``G`` sits at ``p-1``.  This mirrors ``donor_pos``, which is the first intron base and
    the ``G`` of the ``GT``.
    """
    if x.next_acceptor_pos is None:
        raise ValueError(f"{x.example_id} has no intron-mate acceptor")
    return x.next_acceptor_pos - 1


def destroy_donor(x: CouplingExample, alt: str = MOTIF_CLEAN_DONOR_ALT) -> str:
    """The sequence with this example's partner donor ``GT`` overwritten by ``alt``."""
    return substitute(x.sequence, x.donor_pos, alt)


def donor_alt_admissible(x: CouplingExample, alt: str) -> bool:
    """Does writing ``alt`` at this donor create a splice motif in its own flanking context?

    Checked per example, not argued from a tuple -- that inference is exactly what R44
    corrected.
    """
    return not planted_donor_motifs(x.sequence, x.donor_pos, alt)


def margins_at(task, positions: Sequence[int], seqs: Sequence[str]) -> np.ndarray:
    """Masked-``G`` margin at arbitrary positions, batched.

    ``CouplingTask.encode``/``margin`` take a position list they call ``donors``; nothing in
    either is donor-specific, so passing acceptor positions is the whole of the role swap.
    """
    if len(positions) != len(seqs):
        raise ValueError("positions and seqs must be the same length")
    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(positions), task.batch):
            sl = slice(i, i + task.batch)
            enc = task.encode(list(seqs[sl]), list(positions[sl]))
            out += task.margin(task.ad.model(**enc).logits.float(), list(positions[sl])).tolist()
    return np.asarray(out)


def reverse_coupling(
    task,
    exs: Sequence[CouplingExample],
    *,
    alt: str = MOTIF_CLEAN_DONOR_ALT,
    require_clean: bool = True,
) -> np.ndarray:
    """``Y_A`` per example: margin at the masked acceptor, clean minus partner-donor-destroyed.

    Raises when any construct would plant a motif, unless ``require_clean`` is cleared --
    a planted ``GT`` one base upstream of the destroyed one is not a destruction at all.
    """
    if require_clean:
        bad = [x.example_id for x in exs if not donor_alt_admissible(x, alt)]
        if bad:
            raise ValueError(
                f"{len(bad)} construct(s) would plant a motif with alt {alt!r}; "
                f"first: {bad[0]}. Filter with donor_alt_admissible or pass require_clean=False."
            )
    pos = [acceptor_readout(x) for x in exs]
    clean = margins_at(task, pos, [x.sequence for x in exs])
    corrupt = margins_at(task, pos, [destroy_donor(x, alt) for x in exs])
    return clean - corrupt


def reverse_pseudo_coupling(
    task,
    exs: Sequence[CouplingExample],
    sites: Sequence[int],
    *,
    alt: str = MOTIF_CLEAN_DONOR_ALT,
) -> np.ndarray:
    """Specificity null: destroy a NON-partner site and read the same masked acceptor.

    Mirrors ``run_ablation_scan.py``'s ``drop_control`` idiom -- an ablation that damages the
    real coupling *and* this pseudo-coupling equally is damaging the model, not the mechanism.
    ``sites`` are corruption positions (e.g. ``matched_decoy_donor`` output), not readouts.
    """
    pos = [acceptor_readout(x) for x in exs]
    clean = margins_at(task, pos, [x.sequence for x in exs])
    corrupt = margins_at(task, pos, [substitute(x.sequence, s, alt) for x, s in zip(exs, sites)])
    return clean - corrupt


def reverse_mlm_health(
    task,
    exs: Sequence[CouplingExample],
    ctx=None,
    *,
    n_probe: int = 40,
    seed: int = 0,
) -> float:
    """Top-1 masked-prediction accuracy at random non-site positions, ACCEPTOR readout masked.

    ``CouplingTask.mlm_health`` masks ``donor_pos``; a reverse-direction ablation must be
    health-gated with the acceptor masked instead, or the gate is scoring a different forward
    pass than the one whose coupling is being damaged.

    THE PROBE POSITIONS ARE MASKED, and that is load-bearing (R8): an earlier version of the
    forward control read the argmax at *unmasked* positions, i.e. scored the model's ability to
    copy its own input, and returned ~0.99 for a model with 94 of 96 heads ablated.
    """
    import random as _random

    ctx = ctx or (lambda: __import__("contextlib").nullcontext())
    rng = _random.Random(seed)
    hits = tot = 0
    with torch.no_grad():
        for i in range(0, len(exs), task.batch):
            ch = exs[i:i + task.batch]
            base = [acceptor_readout(x) for x in ch]
            seqs, probes = [], []
            for x, b in zip(ch, base):
                n = len(x.sequence)
                ps = [rng.randrange(4, n - 4) for _ in range(n_probe)]
                # never probe either site: the acceptor is already masked, the donor is the edit
                ps = [p for p in ps
                      if abs(p - b) > 3 and abs(p - x.donor_pos) > 3]
                probes.append(ps)
                seqs.append(x.sequence)
            for k in range(max((len(p) for p in probes), default=0)):
                sel = [(j, probes[j][k]) for j in range(len(ch)) if k < len(probes[j])]
                if not sel:
                    continue
                js = [j for j, _ in sel]
                pos = [p for _, p in sel]
                sub = [seqs[j] for j in js]
                # Match the reverse behavioral forward pass: keep the acceptor G masked and
                # add one non-site probe mask.  Encoding at ``pos`` alone would test a different
                # single-mask policy and was the defect memorialized in ACR11.
                readout = [base[j] for j in js]
                enc = task.encode(sub, readout)
                for row, p in enumerate(pos):
                    enc["input_ids"][row, p + task.ad.prefix_tokens] = task.mask_id
                with ctx():
                    logits = task.ad.model(**enc).logits.float()
                for row, (j, p) in enumerate(sel):
                    v = logits[row, p + task.ad.prefix_tokens]
                    pred = max("ACGT", key=lambda b_: v[task.bid[b_]].item())
                    hits += int(pred == seqs[j][p])
                    tot += 1
    return hits / max(tot, 1)


__all__ = [
    "FORWARD_ACCEPTOR_ALT",
    "MATCHED_TWO_SUB",
    "acceptor_readout",
    "destroy_donor",
    "donor_alt_admissible",
    "edit_size",
    "margins_at",
    "matched_operators",
    "reverse_coupling",
    "reverse_mlm_health",
    "reverse_pseudo_coupling",
]
