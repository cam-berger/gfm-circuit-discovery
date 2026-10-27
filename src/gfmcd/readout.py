"""The coupling readout: margins, paired measurement, and normalized recovery behind one seam.

The masked-base margin was implemented independently at least seven times in the library, in three
numeric paths (torch float32, float64, and a float32 "historical four-base sum order"), and nothing
tested any two against each other. ``CouplingTask.coupling`` took only an argument-free context
factory, so every runner that patched per batch reached past it into ``encode``/``off``/``bid`` and
rewrote the encode -> pair -> forward -> margin loop.

    MaskedBaseMargin(base, reduce)   logit(base) - max | mean(other three), float64 from the logits
    ClassifierMargin(label)          logit[label] - logit[reference]   (logits only, for now)
    measure(adapter, examples, readout, site=, corrupt=, group=, intervention=)  -> Paired
    recovery / recovery_interval     R = (mu_I - mu_-) / (mu_+ - mu_-), a ratio of cohort means

``intervention`` receives each :class:`Batch` (arm, examples, encoding, masked token positions) and
returns a context manager, so a per-batch patch sees its own tensors. Margins are exact against
the float64 copies and agree with the float32 copies to float32 resolution (tests pin both).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Literal, Mapping, Sequence

import numpy as np

from gfmcd.inference import GroupedSample, Interval
from gfmcd.splicing.coupling import ACCEPTOR_ALTS, substitute
from gfmcd.splicing.grouped_stats import ratio_of_means

BASES = "ACGT"
FROZEN_CORRUPTION = ACCEPTOR_ALTS[0]  # "CC": destroys the partner AG and cannot create one


def donor_site(example) -> int:
    return example.donor_pos


def frozen_cc(example) -> str:
    """The frozen partner corruption: the paired acceptor's AG becomes CC."""
    return substitute(example.sequence, example.next_acceptor_pos - 2, FROZEN_CORRUPTION)


def gene_of(example) -> str:
    return example.gene_id


def _float64(logits) -> np.ndarray:
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().double().numpy()
    array = np.asarray(logits, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("logits must be selected rows, shape [batch, vocabulary]")
    return array


@dataclass(frozen=True)
class MaskedBaseMargin:
    """``logit(base) - max(others)`` ("max", the task metric) or ``- mean(others)`` ("linear", the
    functional difference-in-differences estimands need). Others are taken in A, C, G, T order."""

    base: str = "G"
    reduce: Literal["max", "linear"] = "max"

    def __post_init__(self) -> None:
        if self.base not in BASES or self.reduce not in ("max", "linear"):
            raise ValueError("base must be one of A,C,G,T and reduce 'max' or 'linear'")

    def from_logits(self, logits, base_ids: Mapping[str, int]) -> np.ndarray:
        rows = _float64(logits)
        hit = rows[:, base_ids[self.base]]
        a, b, c = (rows[:, base_ids[other]] for other in BASES if other != self.base)
        if self.reduce == "max":
            return hit - np.maximum(np.maximum(a, b), c)
        return hit - ((a + b) + c) / 3.0


@dataclass(frozen=True)
class ClassifierMargin:
    """``logit[label] - logit[reference]`` for a sequence classifier's output logits."""

    label: int
    reference: int = 0

    def from_logits(self, logits, base_ids: Mapping[str, int] | None = None) -> np.ndarray:
        rows = _float64(logits)
        return rows[:, self.label] - rows[:, self.reference]


@dataclass(frozen=True)
class Batch:
    """What an intervention sees: one arm of one batch, already tokenized and masked."""

    arm: Literal["clean", "corrupt"]
    examples: tuple
    encoding: Any
    positions: tuple[int, ...]  # masked token positions, prefix offset included


@dataclass(frozen=True)
class Paired:
    """Per-example readouts of the clean and corrupted arms, with each example's group."""

    clean: np.ndarray
    corrupt: np.ndarray
    groups: np.ndarray

    @property
    def coupling(self) -> np.ndarray:
        return self.clean - self.corrupt


def measure(adapter, examples: Sequence, readout: MaskedBaseMargin, *,
            site: Callable[[Any], int] = donor_site, corrupt: Callable[[Any], str] = frozen_cc,
            group: Callable[[Any], Any] = gene_of,
            intervention: Callable[[Batch], ContextManager] | None = None,
            batch_size: int = 8) -> Paired:
    """Mask ``site`` in each example's clean and corrupted sequence and read the margin there."""
    import torch

    if not isinstance(readout, MaskedBaseMargin):
        raise TypeError("measure() reads a masked site; ClassifierMargin is logits-only for now")
    tokenizer = adapter.tokenizer
    mask_id = tokenizer.mask_token_id
    base_ids = {b: tokenizer.convert_tokens_to_ids(b) for b in BASES}
    offset = adapter.prefix_tokens
    examples = list(examples)
    arms: dict[str, list[float]] = {"clean": [], "corrupt": []}
    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            chunk = examples[start:start + batch_size]
            positions = tuple(site(x) + offset for x in chunk)
            for arm, sequences in (("clean", [x.sequence for x in chunk]),
                                   ("corrupt", [corrupt(x) for x in chunk])):
                encoding = tokenizer([" ".join(s) for s in sequences], return_tensors="pt",
                                     padding=True).to(adapter.device)
                for row, position in enumerate(positions):
                    encoding["input_ids"][row, position] = mask_id
                batch = Batch(arm, tuple(chunk), encoding, positions)
                with intervention(batch) if intervention is not None else nullcontext():
                    logits = adapter.model(**encoding).logits
                rows = logits[torch.arange(len(chunk)), torch.tensor(positions)]
                arms[arm].extend(readout.from_logits(rows, base_ids).tolist())
    return Paired(np.asarray(arms["clean"]), np.asarray(arms["corrupt"]),
                  np.asarray([group(x) for x in examples], dtype=object))


def recovery(baseline: Paired, intervened: Sequence[float] | np.ndarray) -> float:
    """``R = mean(intervened - corrupt) / mean(clean - corrupt)``, from ``baseline``'s arms.

    ``intervened`` holds the margins of the intervened run, aligned with ``baseline``'s examples.
    A ratio of cohort means, never a mean of per-example ratios.
    """
    intervened = np.asarray(intervened, dtype=np.float64)
    return ratio_of_means(intervened - baseline.corrupt, baseline.coupling)


def recovery_interval(baseline: Paired, intervened: Sequence[float] | np.ndarray, *, n_boot: int,
                      seed: int, confidence: float = 0.95, strict_sign: bool = True) -> Interval:
    """Gene-grouped interval for :func:`recovery`, via ``gfmcd.inference`` (sign-strict)."""
    intervened = np.asarray(intervened, dtype=np.float64)
    sample = GroupedSample(intervened - baseline.corrupt, baseline.groups)
    return sample.ratio(baseline.coupling, n_boot=n_boot, seed=seed, confidence=confidence,
                        strict_sign=strict_sign)
