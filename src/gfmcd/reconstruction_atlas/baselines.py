"""Low-order sequence baselines that never consume model activations."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class NucleotideNGramBaseline:
    """Add-alpha nucleotide model of order 0, 1, or 2."""

    order: int
    alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.order not in {0, 1, 2}:
            raise ValueError("baseline order must be 0, 1, or 2")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        self._counts: dict[str, Counter[str]] = defaultdict(Counter)
        self._fitted = False

    def fit(self, sequences: Iterable[str]) -> NucleotideNGramBaseline:
        seen = 0
        for raw in sequences:
            sequence = raw.upper()
            for index, base in enumerate(sequence):
                if base not in "ACGT":
                    continue
                context = self._context(sequence, index)
                if set(context) - set("^ACGT"):
                    continue
                self._counts[context][base] += 1
                seen += 1
        if seen == 0:
            raise ValueError("baseline training data contains no canonical nucleotides")
        self._fitted = True
        return self

    def log_probability(self, sequence: str) -> float:
        if not self._fitted:
            raise RuntimeError("baseline must be fit before scoring")
        total = 0.0
        for index, base in enumerate(sequence.upper()):
            if base not in "ACGT":
                total += math.log(0.25)
                continue
            context = self._context(sequence.upper(), index)
            counts = self._counts.get(context, Counter())
            denominator = sum(counts.values()) + 4 * self.alpha
            total += math.log((counts.get(base, 0) + self.alpha) / denominator)
        return total

    def log_probability_per_nt(self, sequence: str) -> float:
        if not sequence:
            raise ValueError("cannot score an empty sequence")
        return self.log_probability(sequence) / len(sequence)

    def identity(self) -> dict[str, float | int]:
        if not self._fitted:
            raise RuntimeError("baseline must be fit before recording identity")
        return {
            "order": self.order,
            "alpha": self.alpha,
            "contexts": len(self._counts),
            "observations": sum(
                sum(counts.values()) for counts in self._counts.values()
            ),
        }

    def _context(self, sequence: str, index: int) -> str:
        if self.order == 0:
            return ""
        prefix = "^" * self.order + sequence
        shifted = index + self.order
        return prefix[shifted - self.order : shifted]
