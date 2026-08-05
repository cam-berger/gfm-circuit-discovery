"""Cross-site coupling read from EITHER end of the same intron (A13/A14 ``--site`` axis).

THE MIRROR
----------
``gfmcd.splicing.circuit.CouplingTask`` reads the DONOR end: mask the G of the donor ``GT`` and
corrupt the intron-mate acceptor ``AG``.  This module adds the other end -- mask the G of that
same ``AG`` and corrupt that same donor ``GT`` -- so both arms run on the same intron, the same
pair, the same separation and the same cohort.  Only the readout end and the corruption end swap.

``CouplingTask`` is imported by every frozen donor-arc runner and is NOT modified here.  The
donor branch below reproduces it exactly (same mask position, same base, same ``CORRUPT_ALT``);
``tests/test_site_coupling.py`` asserts the two agree to machine precision, so this module cannot
silently drift from the frozen definition.

THE ARMS ARE NOT SYMMETRIC IN THEIR ALT, AND MUST NOT BE (R44)
--------------------------------------------------------------
R44 measured, model-free, that the last exon base is ``G`` in **81.0%** of val GENCODE donors
(the -1 G of the 5'SS consensus).  Writing the symmetric ``CC`` over a donor therefore plants a
``GC`` -- a real minor-class donor -- one base upstream in four of five examples, and ``TA`` plants
a *canonical* ``GT`` in 95.5%.  Only ``AC`` plants nothing (0.0%) everywhere measured.

R44 exempts the partner-selection study because all of its corruptions are acceptor-side, where
the analogous rate is 0.35-0.36%.  **The acceptor arm corrupts the donor and inherits the full
exposure.**  Using ``CC`` there would convert the donor into a functional ``GC`` site rather than
destroying it, and the arm would measure dependence on donor *class* instead of donor *presence*.

So the two arms deliberately use different letters in order to be comparable:

    donor arm     AG -> "CC"  at the acceptor   (frozen; C42 alt-invariant within +/-25%)
    acceptor arm  GT -> "AC"  at the donor      (MOTIF_CLEAN_DONOR_ALT; 0.0% plant rate)

``planted_motif_rate`` is the gating memorial: model-free, computable from sequence alone, and it
returns a nonzero rate the moment someone "restores symmetry" by switching the acceptor arm to CC.

THE ACCEPTOR BEHAVIOUR IS NOT ESTABLISHED
-----------------------------------------
C1 establishes the donor direction (+0.872 [0.771, 0.969], sealed).  Nothing in any ledger
establishes the acceptor direction.  ACDC searches for the circuit implementing a behaviour, so
the acceptor arm is gated on ``coupling >= 0.20`` plus partner-specificity against the
motif-matched non-partner ``GT`` decoys already carried on ``CouplingExample.other_gt``.  If those
fail the circuit search does not run and the arm reports NO BEHAVIOUR -- a directional asymmetry
in cross-site dependence is a finding, not a failed experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Literal, Sequence

import numpy as np
import torch

from gfmcd.adapters.splicebert import SpliceBertAdapter
from gfmcd.splicing.coupling import (
    ACCEPTOR_ALTS, MOTIF_CLEAN_DONOR_ALT, CouplingExample, planted_acceptor_motifs,
    planted_donor_motifs, substitute,
)

Site = Literal["donor", "acceptor"]
SITES: tuple[Site, ...] = ("donor", "acceptor")

DONOR_CORRUPT_ALT: str = ACCEPTOR_ALTS[0]           # "CC" -- frozen, matches circuit.CORRUPT_ALT
ACCEPTOR_CORRUPT_ALT: str = MOTIF_CLEAN_DONOR_ALT   # "AC" -- the only 0.0%-plant alt at a donor


@dataclass(frozen=True)
class SiteSpec:
    """Where the readout is masked and where the partner is destroyed, for one direction."""

    site: Site
    alt: str
    readout_base: str

    def readout_position(self, example: CouplingExample) -> int:
        """Nucleotide index whose base is masked and scored."""
        if self.site == "donor":
            return example.donor_pos                       # the G of the donor GT
        return example.next_acceptor_pos - 1                # the G of the intron-mate AG

    def corrupt_sequence(self, example: CouplingExample) -> str:
        if self.site == "donor":
            return substitute(example.sequence, example.next_acceptor_pos - 2, self.alt)
        return substitute(example.sequence, example.donor_pos, self.alt)

    def planted_motifs(self, example: CouplingExample) -> dict[str, str]:
        """Splice motifs CREATED by this arm's corruption.  Empty dict == clean."""
        if self.site == "donor":
            return planted_acceptor_motifs(
                example.sequence, example.next_acceptor_pos, self.alt
            )
        return planted_donor_motifs(example.sequence, example.donor_pos, self.alt)

    def decoy_positions(self, example: CouplingExample) -> tuple[int, ...]:
        """Motif-matched non-partner sites, for the partner-specificity control."""
        return example.other_ag if self.site == "donor" else example.other_gt

    def corrupt_decoy(self, example: CouplingExample, position: int) -> str:
        if self.site == "donor":
            return substitute(example.sequence, position - 2, self.alt)
        return substitute(example.sequence, position, self.alt)


SPEC: dict[Site, SiteSpec] = {
    "donor": SiteSpec("donor", DONOR_CORRUPT_ALT, "G"),
    "acceptor": SiteSpec("acceptor", ACCEPTOR_CORRUPT_ALT, "G"),
}


def planted_motif_rate(
    examples: Sequence[CouplingExample], site: Site
) -> tuple[float, dict[str, int]]:
    """Model-free memorial: fraction of the cohort whose corruption plants a splice motif.

    Must be exactly 0.0 for both arms.  Returns ``(rate, {motif: count})`` so a failure names
    what was planted rather than only that something was.
    """
    spec = SPEC[site]
    planted = 0
    by_motif: dict[str, int] = {}
    for example in examples:
        motifs = spec.planted_motifs(example)
        if motifs:
            planted += 1
            for value in motifs.values():
                by_motif[value] = by_motif.get(value, 0) + 1
    return (planted / len(examples) if examples else 0.0), by_motif


def eligible_for_site(examples: Sequence[CouplingExample], site: Site) -> list[CouplingExample]:
    """Both arms require the intron-mate acceptor, so eligibility is shared by construction."""
    return [
        x for x in examples
        if x.next_acceptor_pos is not None
        and x.sequence[x.donor_pos:x.donor_pos + 2] == "GT"
        and x.sequence[x.next_acceptor_pos - 2:x.next_acceptor_pos] == "AG"
    ]


def noop() -> Iterator[None]:
    yield


class SiteCouplingTask:
    """Native masked-LM coupling readout for one direction of the same intron.

    ``coupling = margin(clean) - margin(partner destroyed)``.  No probe, no readout head, no
    labels -- so there is no task to win and nothing to leak, exactly as in ``CouplingTask``.
    """

    def __init__(self, adapter: SpliceBertAdapter, site: Site = "donor", batch: int = 8):
        if site not in SPEC:
            raise ValueError(f"unknown site {site!r}; expected one of {SITES}")
        self.ad = adapter
        self.site = site
        self.spec = SPEC[site]
        self.batch = batch
        self.mask_id = adapter.tokenizer.mask_token_id
        self.bid = {b: adapter.tokenizer.convert_tokens_to_ids(b) for b in "ACGT"}
        self.off = adapter.prefix_tokens

    # ---------------------------------------------------------------------------------
    def encode(self, seqs: Sequence[str], positions: Sequence[int]):
        enc = self.ad.tokenizer(
            [" ".join(s) for s in seqs], return_tensors="pt", padding=True
        ).to(self.ad.device)
        for j, p in enumerate(positions):
            enc["input_ids"][j, p + self.off] = self.mask_id
        return enc

    def margin(self, logits: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
        base = self.spec.readout_base
        others = [b for b in "ACGT" if b != base]
        i = torch.arange(len(positions), device=logits.device)
        p = torch.tensor([q + self.off for q in positions], device=logits.device)
        v = logits[i, p]
        alt = torch.stack([v[:, self.bid[b]] for b in others], -1).max(-1).values
        return v[:, self.bid[base]] - alt

    def pair(self, examples: Sequence[CouplingExample]):
        """``(clean_enc, corrupt_enc, readout_positions)`` for one batch."""
        positions = [self.spec.readout_position(x) for x in examples]
        clean = self.encode([x.sequence for x in examples], positions)
        corrupt = self.encode(
            [self.spec.corrupt_sequence(x) for x in examples], positions
        )
        return clean, corrupt, positions

    def coupling(
        self, examples: Sequence[CouplingExample], ctx: Callable[[], Iterator] | None = None
    ) -> np.ndarray:
        ctx = ctx or noop
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(examples), self.batch):
                clean, corrupt, positions = self.pair(examples[i:i + self.batch])
                with ctx():
                    m_clean = self.margin(self.ad.model(**clean).logits.float(), positions)
                    m_corrupt = self.margin(self.ad.model(**corrupt).logits.float(), positions)
                out += (m_clean - m_corrupt).tolist()
        return np.asarray(out)

    def margins(
        self, examples: Sequence[CouplingExample], ctx: Callable[[], Iterator] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        ctx = ctx or noop
        clean_out: list[float] = []
        corrupt_out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(examples), self.batch):
                clean, corrupt, positions = self.pair(examples[i:i + self.batch])
                with ctx():
                    clean_out += self.margin(
                        self.ad.model(**clean).logits.float(), positions
                    ).tolist()
                    corrupt_out += self.margin(
                        self.ad.model(**corrupt).logits.float(), positions
                    ).tolist()
        return np.asarray(clean_out), np.asarray(corrupt_out)

    def decoy_coupling(self, examples: Sequence[CouplingExample]) -> np.ndarray:
        """Partner-specificity control: destroy a motif-matched NON-partner site instead.

        Returns NaN for examples with no available decoy, so the caller sees the denominator
        rather than a silently smaller sample.
        """
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(examples), self.batch):
                chunk = examples[i:i + self.batch]
                usable = [(x, self.spec.decoy_positions(x)) for x in chunk]
                positions = [self.spec.readout_position(x) for x in chunk]
                clean = self.encode([x.sequence for x in chunk], positions)
                m_clean = self.margin(self.ad.model(**clean).logits.float(), positions)
                decoy_seqs, keep = [], []
                for j, (x, decoys) in enumerate(usable):
                    if decoys:
                        decoy_seqs.append(self.spec.corrupt_decoy(x, decoys[0]))
                        keep.append(j)
                    else:
                        decoy_seqs.append(x.sequence)
                enc = self.encode(decoy_seqs, positions)
                m_decoy = self.margin(self.ad.model(**enc).logits.float(), positions)
                delta = (m_clean - m_decoy).tolist()
                out += [
                    delta[j] if j in set(keep) else float("nan")
                    for j in range(len(chunk))
                ]
        return np.asarray(out)
