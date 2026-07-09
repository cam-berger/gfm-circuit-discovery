"""Registry of substrates, and the firewall that stops us over-claiming on them.

SpliceBERT was pretrained on pre-mRNA from 72 vertebrates including the whole human
genome; DNABERT-2 on the whole genome. There is no held-out chromosome for either. That
is fine for mechanistic claims -- we ask what computation the weights implement, not how
well they generalise -- and fatal for generalisation claims. The distinction is easy to
lose in a results table six weeks from now, so `assert_holdout_valid` raises rather than
warns.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Provenance(Enum):
    PLANTED = "planted"
    TRAINED_BY_US = "trained_by_us"
    PRETRAINED = "pretrained"


@dataclass(frozen=True)
class Substrate:
    name: str
    provenance: Provenance
    hf_id: str | None = None
    single_nt: bool = True
    both_strands: bool = False
    notes: str = ""

    @property
    def holdout_is_meaningful(self) -> bool:
        return self.provenance is not Provenance.PRETRAINED


class ContaminationError(RuntimeError):
    pass


SUBSTRATES: dict[str, Substrate] = {
    "planted-compositional": Substrate("planted-compositional", Provenance.PLANTED),
    "planted-heuristic": Substrate("planted-heuristic", Provenance.PLANTED),
    "gfm-nano-cds": Substrate("gfm-nano-cds", Provenance.TRAINED_BY_US),
    "gfm-nano-dna": Substrate("gfm-nano-dna", Provenance.TRAINED_BY_US, both_strands=True),
    "splicebert": Substrate(
        "splicebert",
        Provenance.PRETRAINED,
        hf_id="yangheng/SpliceBERT-510nt",  # community mirror; official weights are chenkenbio/SpliceBERT — verify at load (Phase 1)
        both_strands=False,
        notes="pre-mRNA, sense strand only: cannot carry Rung 3's cross-strand axis",
    ),
    "dnabert2": Substrate(
        "dnabert2",
        Provenance.PRETRAINED,
        hf_id="zhihan1996/DNABERT-2-117M",
        single_nt=False,
        both_strands=True,
        notes="BPE. Window-level readouts only; per-nucleotide claims are invalid",
    ),
}


def assert_holdout_valid(name: str) -> None:
    sub = SUBSTRATES[name]
    if not sub.holdout_is_meaningful:
        raise ContaminationError(
            f"{name!r} was pretrained on the human genome; no chromosome is held out from it. "
            f"Mechanistic claims are fine. Generalisation claims are not. "
            f"Use a TRAINED_BY_US substrate, or state the claim mechanistically."
        )


def assert_per_nucleotide_readout(name: str) -> None:
    sub = SUBSTRATES[name]
    if not sub.single_nt:
        raise ContaminationError(
            f"{name!r} does not tokenise at single-nucleotide resolution; a per-nucleotide "
            f"readout is not well defined and a frameshift is not a minimal pair."
        )


def assert_cross_strand_capable(name: str) -> None:
    sub = SUBSTRATES[name]
    if not sub.both_strands:
        raise ContaminationError(
            f"{name!r} never saw the reverse strand ({sub.notes}); it cannot represent the "
            f"strand-composition variable that separates a G4 circuit from a G-density counter."
        )
