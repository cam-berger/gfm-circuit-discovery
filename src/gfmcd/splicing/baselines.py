"""Registered local/additive baselines for splice-pair selection.

These models are controls, not strawmen.  They are fit only on the training
split, use the same candidate sets as SpliceBERT, and are frozen before the test
set or circuit trace is inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

import numpy as np

from gfmcd.splicing.schema import PairCandidate, SplicePairExample

ALPHABET = "ACGT"
BASE_TO_INDEX = {b: i for i, b in enumerate(ALPHABET)}


def site_window(
    sequence: str,
    position: int,
    kind: Literal["acceptor", "donor"],
    *,
    acceptor_upstream: int = 20,
    acceptor_downstream: int = 3,
    donor_upstream: int = 3,
    donor_downstream: int = 6,
) -> str | None:
    if kind == "acceptor":
        start, end = position - acceptor_upstream, position + acceptor_downstream
    else:
        start, end = position - donor_upstream, position + donor_downstream
    if start < 0 or end > len(sequence):
        return None
    return sequence[start:end]


@dataclass
class PositionWeightModel:
    log_odds: np.ndarray  # [length, 4]
    background: np.ndarray  # [4]
    pseudocount: float = 0.5

    @classmethod
    def fit(
        cls,
        windows: Iterable[str],
        *,
        background: np.ndarray | None = None,
        pseudocount: float = 0.5,
    ) -> "PositionWeightModel":
        items = [w.upper().replace("U", "T") for w in windows]
        if not items:
            raise ValueError("cannot fit PWM with no windows")
        length = len(items[0])
        if any(len(w) != length for w in items):
            raise ValueError("all PWM windows must have equal length")
        counts = np.full((length, 4), pseudocount, dtype=np.float64)
        bg_counts = np.full(4, pseudocount, dtype=np.float64)
        for window in items:
            for pos, base in enumerate(window):
                idx = BASE_TO_INDEX.get(base)
                if idx is not None:
                    counts[pos, idx] += 1
                    bg_counts[idx] += 1
        probs = counts / counts.sum(axis=1, keepdims=True)
        if background is None:
            background = bg_counts / bg_counts.sum()
        else:
            background = np.asarray(background, dtype=np.float64)
            if background.shape != (4,) or np.any(background <= 0):
                raise ValueError("background must be a positive four-element vector")
            background = background / background.sum()
        return cls(np.log(probs) - np.log(background[None, :]), background, pseudocount)

    @property
    def length(self) -> int:
        return int(self.log_odds.shape[0])

    def score(self, window: str | None) -> float:
        if window is None:
            return float("-inf")
        if len(window) != self.length:
            raise ValueError(f"expected length {self.length}, got {len(window)}")
        total = 0.0
        for pos, base in enumerate(window.upper().replace("U", "T")):
            idx = BASE_TO_INDEX.get(base)
            if idx is None:
                # Unknown bases contribute the expectation under background.
                total += float((self.background * self.log_odds[pos]).sum())
            else:
                total += float(self.log_odds[pos, idx])
        return total


@dataclass
class LengthPrior:
    min_length: int
    max_length: int
    log_prob: np.ndarray

    @classmethod
    def fit(
        cls,
        lengths: Iterable[int],
        *,
        min_length: int,
        max_length: int,
        pseudocount: float = 1.0,
    ) -> "LengthPrior":
        counts = np.full(max_length - min_length + 1, pseudocount, dtype=np.float64)
        for length in lengths:
            if min_length <= length <= max_length:
                counts[length - min_length] += 1
        probs = counts / counts.sum()
        return cls(min_length, max_length, np.log(probs))

    def score(self, length: int) -> float:
        if not self.min_length <= length <= self.max_length:
            return float("-inf")
        return float(self.log_prob[length - self.min_length])


@dataclass
class LocalPairBaseline:
    acceptor_pwm: PositionWeightModel
    donor_pwm: PositionWeightModel
    length_prior: LengthPrior
    length_weight: float = 1.0

    @classmethod
    def fit(
        cls,
        examples: Iterable[SplicePairExample],
        *,
        length_weight: float = 1.0,
    ) -> "LocalPairBaseline":
        items = list(examples)
        if not items:
            raise ValueError("no training examples")
        acceptor_windows: list[str] = []
        donor_windows: list[str] = []
        lengths: list[int] = []
        for example in items:
            pair = example.true_pair
            aw = site_window(example.sequence, pair.acceptor_pos, "acceptor")
            dw = site_window(example.sequence, pair.donor_pos, "donor")
            if aw is not None and dw is not None:
                acceptor_windows.append(aw)
                donor_windows.append(dw)
                lengths.append(pair.exon_length)
        if not lengths:
            raise ValueError("no complete true-site windows")
        return cls(
            acceptor_pwm=PositionWeightModel.fit(acceptor_windows),
            donor_pwm=PositionWeightModel.fit(donor_windows),
            length_prior=LengthPrior.fit(
                lengths,
                min_length=min(lengths),
                max_length=max(lengths),
            ),
            length_weight=length_weight,
        )

    def score_pair(self, example: SplicePairExample, pair: PairCandidate) -> float:
        aw = site_window(example.sequence, pair.acceptor_pos, "acceptor")
        dw = site_window(example.sequence, pair.donor_pos, "donor")
        return (
            self.acceptor_pwm.score(aw)
            + self.donor_pwm.score(dw)
            + self.length_weight * self.length_prior.score(pair.exon_length)
        )

    def score_example(self, example: SplicePairExample) -> np.ndarray:
        return np.asarray([self.score_pair(example, pair) for pair in example.pairs], dtype=np.float64)

    def predict(self, example: SplicePairExample) -> int:
        scores = self.score_example(example)
        if not np.isfinite(scores).any():
            raise ValueError(f"all local baseline scores are invalid for {example.example_id}")
        return int(np.nanargmax(scores))

    def true_margin(self, example: SplicePairExample) -> float:
        scores = self.score_example(example)
        true = scores[example.true_pair_index]
        decoys = np.delete(scores, example.true_pair_index)
        return float(true - np.max(decoys)) if len(decoys) else math.inf


def baseline_conflict_examples(
    baseline: LocalPairBaseline,
    examples: Iterable[SplicePairExample],
) -> list[SplicePairExample]:
    """Natural cue conflicts where local/additive evidence picks a decoy pair."""
    return [x for x in examples if baseline.predict(x) != x.true_pair_index]
