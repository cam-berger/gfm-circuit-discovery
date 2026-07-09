"""hg38 + GENCODE CDS windows, with the one property Rung 1 depends on.

**Window offsets are randomised, and that is not a detail.** Codon phase inside a CDS is
`(g - first_codon_base) % 3`. If windows always began at the CDS start, then phase would equal
`window_index % 3` exactly, and a model with absolute position embeddings would read it straight
off `pos_embed` -- a lookup table, not a circuit. It is the same degeneracy we removed from the
planted model (D4), it would be invisible in the loss curve, and it would silently invalidate
every Rung 1 claim.

Randomising the window start makes `w0 % 3` uniform, so `index % 3` carries no information about
phase across the corpus while remaining perfectly informative *within* a window given the anchor.
`tests/test_genome.py` asserts both halves of that.

Frame labels never enter a backbone's training objective. They exist for probes and the frozen
evaluation head only (D7).
"""

from __future__ import annotations

import gzip
import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

IGNORE = -100
NT2ID = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


class Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


SPLIT_CHROMS: dict[Split, frozenset[str]] = {
    Split.TRAIN: frozenset(f"chr{i}" for i in range(1, 17)),
    Split.VAL: frozenset({"chr17", "chr18", "chr19"}),
    Split.TEST: frozenset({"chr20", "chr21", "chr22", "chrX"}),
}


class SplitLockedError(RuntimeError):
    pass


_TEST_UNLOCKED = False


def unlock_test_split(preregistration_frozen_at: str) -> None:
    """Only callable once `docs/PREREGISTRATION.md` is frozen; pass the commit hash."""
    global _TEST_UNLOCKED
    if len(preregistration_frozen_at) < 7:
        raise ValueError("pass the commit hash at which the pre-registration was frozen")
    _TEST_UNLOCKED = True


def assert_split_allowed(split: Split) -> None:
    if split is Split.TEST and not _TEST_UNLOCKED:
        raise SplitLockedError(
            "the test chromosomes are locked until the pre-registration is frozen; "
            "call unlock_test_split(<commit hash>) deliberately, never in passing"
        )


class FastaLike(Protocol):
    def __getitem__(self, chrom: str) -> object: ...


@dataclass(frozen=True)
class CDS:
    """A coding exon. Coordinates are 0-based inclusive; `phase` is GENCODE's frame field."""

    chrom: str
    start: int
    end: int
    strand: str
    phase: int

    def __len__(self) -> int:
        return self.end - self.start + 1

    @property
    def first_codon_base(self) -> int:
        """Genomic coordinate of the first base of the first complete codon."""
        return self.start + self.phase if self.strand == "+" else self.end - self.phase

    def codon_position(self, g: int) -> int:
        """0, 1 or 2 -- position within the codon, counted along the direction of transcription."""
        delta = g - self.first_codon_base if self.strand == "+" else self.first_codon_base - g
        return delta % 3

    def contains(self, g: int) -> bool:
        return self.start <= g <= self.end


def _open(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


def parse_gencode_cds(path: Path, chroms: frozenset[str] | None = None, min_len: int = 150) -> list[CDS]:
    """CDS features from a GENCODE GTF. One entry per coding exon."""
    out: list[CDS] = []
    with _open(Path(path)) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8 or f[2] != "CDS":
                continue
            chrom, start, end, strand, frame = f[0], int(f[3]), int(f[4]), f[6], f[7]
            if chroms is not None and chrom not in chroms:
                continue
            if frame == "." or end - start + 1 < min_len:
                continue
            out.append(CDS(chrom, start - 1, end - 1, strand, int(frame)))
    return out


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def encode(seq: str) -> np.ndarray:
    return np.fromiter((NT2ID.get(c, 4) for c in seq.upper()), dtype=np.int8, count=len(seq))


@dataclass(frozen=True)
class Window:
    tokens: np.ndarray
    labels: np.ndarray
    chrom: str
    start: int
    strand: str


def sample_window(
    cds: CDS,
    fasta: FastaLike,
    window: int,
    rng: random.Random,
    min_cds_in_window: int = 90,
    chrom_len: int | None = None,
) -> Window | None:
    """One window whose start is uniformly random relative to the CDS.

    That uniformity is the whole point: it decorrelates `index % 3` from codon phase.
    """
    span = min(len(cds), window)
    if span < min_cds_in_window:
        return None

    anchor = rng.randint(cds.start, cds.end - min_cds_in_window + 1)
    offset = rng.randint(0, window - min_cds_in_window)
    w0 = anchor - offset
    if w0 < 0:
        return None
    if chrom_len is not None and w0 + window > chrom_len:
        return None

    seq = str(fasta[cds.chrom][w0 : w0 + window]).upper()
    if len(seq) < window:
        return None

    g = np.arange(w0, w0 + window)
    inside = (g >= cds.start) & (g <= cds.end)
    delta = g - cds.first_codon_base if cds.strand == "+" else cds.first_codon_base - g
    labels = np.where(inside, delta % 3, IGNORE).astype(np.int64)

    if cds.strand == "-":
        seq = revcomp(seq)
        labels = labels[::-1].copy()

    return Window(encode(seq), labels, cds.chrom, w0, cds.strand)


def sample_windows(
    cds_list: list[CDS],
    fasta: FastaLike,
    n: int,
    window: int = 256,
    seed: int = 0,
    min_cds_in_window: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (tokens [n, window] int8, labels [n, window] int64 with IGNORE outside CDS)."""
    rng = random.Random(seed)
    toks: list[np.ndarray] = []
    labs: list[np.ndarray] = []

    tries = 0
    while len(toks) < n and tries < 50 * n:
        tries += 1
        w = sample_window(rng.choice(cds_list), fasta, window, rng, min_cds_in_window)
        if w is None or (w.labels >= 0).sum() < min_cds_in_window:
            continue
        if (w.tokens == 4).mean() > 0.01:
            continue
        toks.append(w.tokens)
        labs.append(w.labels)

    if len(toks) < n:
        raise RuntimeError(f"only sampled {len(toks)}/{n} windows in {tries} tries")
    return np.stack(toks), np.stack(labs)


def iter_splits(cds_list: list[CDS]) -> Iterator[tuple[Split, list[CDS]]]:
    for split, chroms in SPLIT_CHROMS.items():
        yield split, [c for c in cds_list if c.chrom in chroms]


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Anchored vs internal windows.
#
# Rung 1's planted circuit finds a start codon and counts from it. A window drawn at a random
# offset inside a coding exon usually contains no ATG at all -- internal coding exons have none --
# so that computation is not available and a model can only infer phase from codon-usage statistics.
# Measured: a local naive-Bayes predictor over +-15 nt already reaches 0.692.
#
# So we build both regimes explicitly. An *anchored* window contains the translation start codon;
# an *internal* window provably does not. Same architecture, same objective, same model: only the
# availability of an anchor changes. The prediction is that an anchored counter appears where an
# anchor exists and a bag of motifs where none does.
# ---------------------------------------------------------------------------


def first_codon(cds: CDS, fasta: FastaLike) -> str:
    """The three bases of this exon's first complete codon, read along transcription."""
    f = cds.first_codon_base
    if cds.strand == "+":
        return str(fasta[cds.chrom][f : f + 3]).upper()
    return revcomp(str(fasta[cds.chrom][f - 2 : f + 1]).upper())


def split_by_anchor(cds_list: list[CDS], fasta: FastaLike) -> tuple[list[CDS], list[CDS]]:
    """(start-codon exons, internal exons). The first codon of a start exon is ATG."""
    starts, internal = [], []
    for cds in cds_list:
        if cds.phase != 0:
            internal.append(cds)
            continue
        try:
            (starts if first_codon(cds, fasta) == "ATG" else internal).append(cds)
        except (KeyError, ValueError):
            continue
    return starts, internal


def sample_anchored_window(
    cds: CDS,
    fasta: FastaLike,
    window: int,
    rng: random.Random,
    min_downstream: int = 90,
) -> Window | None:
    """A window containing the start codon, placed at a uniformly random offset.

    The offset is still randomised -- `w0 % 3` must stay uniform or phase collapses back into
    `index % 3` and we are training a lookup table again (D21).
    """
    if len(cds) < min_downstream:
        return None

    atg = cds.first_codon_base
    offset = rng.randint(0, window - min_downstream - 3)

    if cds.strand == "+":
        w0 = atg - offset
    else:
        w0 = atg + offset - window + 1
    if w0 < 0:
        return None

    seq = str(fasta[cds.chrom][w0 : w0 + window]).upper()
    if len(seq) < window:
        return None

    g = np.arange(w0, w0 + window)
    inside = (g >= cds.start) & (g <= cds.end)
    delta = g - atg if cds.strand == "+" else atg - g
    labels = np.where(inside & (delta >= 0), delta % 3, IGNORE).astype(np.int64)

    if cds.strand == "-":
        seq = revcomp(seq)
        labels = labels[::-1].copy()

    return Window(encode(seq), labels, cds.chrom, w0, cds.strand)


def sample_anchored_windows(
    cds_list: list[CDS],
    fasta: FastaLike,
    n: int,
    window: int = 256,
    seed: int = 0,
    min_downstream: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    toks: list[np.ndarray] = []
    labs: list[np.ndarray] = []

    tries = 0
    while len(toks) < n and tries < 200 * n:
        tries += 1
        w = sample_anchored_window(rng.choice(cds_list), fasta, window, rng, min_downstream)
        if w is None or (w.labels >= 0).sum() < min_downstream:
            continue
        if (w.tokens == 4).mean() > 0.01:
            continue
        toks.append(w.tokens)
        labs.append(w.labels)

    if len(toks) < n:
        raise RuntimeError(f"only sampled {len(toks)}/{n} anchored windows in {tries} tries")
    return np.stack(toks), np.stack(labs)


def atg_index(tokens: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Index of the start codon's A in each anchored window: the first position with phase 0."""
    first = np.argmax(labels == 0, axis=1)
    return first
