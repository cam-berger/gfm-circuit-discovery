"""Data contracts for the SpliceBERT partner-selection study.

All positions are **0-based nucleotide coordinates in transcript orientation**
inside ``sequence``.  The sequence is always represented 5'->3' as the primary
RNA transcript would be read, even for genes on the genomic minus strand.

Anchor conventions
------------------
* acceptor position: first exonic nucleotide. The canonical ``AG`` is therefore
  ``sequence[pos-2:pos]``.
* donor position: first intronic nucleotide after the exon. The canonical
  ``GT``/``GC`` is therefore ``sequence[pos:pos+2]``.

The true exon is the half-open interval ``[acceptor_pos, donor_pos)``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Iterable, Literal

SiteKind = Literal["acceptor", "donor"]
_ALLOWED = frozenset("ACGTN")


@dataclass(frozen=True)
class SiteCandidate:
    position: int
    kind: SiteKind
    motif: str
    local_score: float | None = None

    def validate(self, sequence: str) -> None:
        if self.kind == "acceptor":
            if not 2 <= self.position < len(sequence):
                raise ValueError(f"invalid acceptor position {self.position}")
            observed = sequence[self.position - 2 : self.position]
        elif self.kind == "donor":
            if not 0 <= self.position <= len(sequence) - 2:
                raise ValueError(f"invalid donor position {self.position}")
            observed = sequence[self.position : self.position + 2]
        else:  # pragma: no cover - Literal is a static guard; runtime still validates.
            raise ValueError(f"unknown site kind: {self.kind}")
        if observed != self.motif:
            raise ValueError(
                f"{self.kind} motif mismatch at {self.position}: "
                f"recorded={self.motif!r}, observed={observed!r}"
            )


@dataclass(frozen=True)
class PairCandidate:
    acceptor_pos: int
    donor_pos: int
    acceptor_index: int
    donor_index: int

    @property
    def exon_length(self) -> int:
        return self.donor_pos - self.acceptor_pos

    def validate(self, acceptors: tuple[SiteCandidate, ...], donors: tuple[SiteCandidate, ...]) -> None:
        if not 0 <= self.acceptor_index < len(acceptors):
            raise ValueError("acceptor_index out of range")
        if not 0 <= self.donor_index < len(donors):
            raise ValueError("donor_index out of range")
        if acceptors[self.acceptor_index].position != self.acceptor_pos:
            raise ValueError("acceptor position/index disagree")
        if donors[self.donor_index].position != self.donor_pos:
            raise ValueError("donor position/index disagree")
        if self.acceptor_pos >= self.donor_pos:
            raise ValueError("acceptor must precede donor in transcript orientation")


@dataclass(frozen=True)
class SplicePairExample:
    example_id: str
    sequence: str
    acceptors: tuple[SiteCandidate, ...]
    donors: tuple[SiteCandidate, ...]
    pairs: tuple[PairCandidate, ...]
    true_pair_index: int
    split: str
    chrom: str
    strand: Literal["+", "-"]
    gene_id: str
    transcript_id: str
    exon_number: int | None
    genomic_exon_start: int
    genomic_exon_end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def true_pair(self) -> PairCandidate:
        return self.pairs[self.true_pair_index]

    def validate(self) -> None:
        seq = self.sequence.upper().replace("U", "T")
        if seq != self.sequence:
            raise ValueError("sequence must be uppercase DNA alphabet with U normalized to T")
        bad = set(seq) - _ALLOWED
        if bad:
            raise ValueError(f"sequence contains unsupported symbols: {sorted(bad)}")
        if not seq:
            raise ValueError("empty sequence")
        if self.strand not in {"+", "-"}:
            raise ValueError("strand must be '+' or '-'")
        if not self.pairs:
            raise ValueError("example has no candidate pairs")
        if not 0 <= self.true_pair_index < len(self.pairs):
            raise ValueError("true_pair_index out of range")
        for site in self.acceptors + self.donors:
            site.validate(seq)
        for pair in self.pairs:
            pair.validate(self.acceptors, self.donors)
        true = self.true_pair
        if true.acceptor_pos - 2 < 0 or true.donor_pos + 2 > len(seq):
            raise ValueError("true splice motifs fall outside the sequence")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SplicePairExample":
        obj = cls(
            example_id=data["example_id"],
            sequence=data["sequence"],
            acceptors=tuple(SiteCandidate(**x) for x in data["acceptors"]),
            donors=tuple(SiteCandidate(**x) for x in data["donors"]),
            pairs=tuple(PairCandidate(**x) for x in data["pairs"]),
            true_pair_index=int(data["true_pair_index"]),
            split=data["split"],
            chrom=data["chrom"],
            strand=data["strand"],
            gene_id=data["gene_id"],
            transcript_id=data["transcript_id"],
            exon_number=data.get("exon_number"),
            genomic_exon_start=int(data["genomic_exon_start"]),
            genomic_exon_end=int(data["genomic_exon_end"]),
            metadata=dict(data.get("metadata", {})),
        )
        obj.validate()
        return obj

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, line: str) -> "SplicePairExample":
        return cls.from_dict(json.loads(line))


def validate_examples(examples: Iterable[SplicePairExample]) -> None:
    seen: set[str] = set()
    for example in examples:
        example.validate()
        if example.example_id in seen:
            raise ValueError(f"duplicate example_id: {example.example_id}")
        seen.add(example.example_id)
