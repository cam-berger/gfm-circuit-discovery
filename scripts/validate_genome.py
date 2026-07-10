"""Validate the CDS pipeline against real hg38 + GENCODE, using biology as ground truth.

The frame labels are checked externally, not just for internal consistency: if the phase and
strand arithmetic is right, in-frame codons must almost never be stop codons, while the two
shifted frames should hit stops at the ~4-6% rate expected from base composition. GENCODE excludes
the terminal stop codon from CDS features, so the in-frame rate should be exactly zero.

    python scripts/validate_genome.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

from gfmcd.data.genome import SPLIT_CHROMS, Split, parse_gencode_cds, sample_windows
from gfmcd.paths import project_paths

DATA = project_paths().references
STOPS = {(3, 0, 0), (3, 0, 2), (3, 2, 0)}  # TAA, TAG, TGA under A=0 C=1 G=2 T=3
WINDOW = 256


def stop_rate(tokens: np.ndarray, labels: np.ndarray, shift: int) -> tuple[float, int]:
    hits = total = 0
    for tok, lab in zip(tokens, labels):
        starts = np.where(lab == 0)[0] + shift
        starts = starts[(starts + 2) < WINDOW]
        starts = starts[lab[starts] >= 0]
        for s in starts:
            hits += (tok[s], tok[s + 1], tok[s + 2]) in STOPS
            total += 1
    return hits / max(total, 1), total


def main() -> None:
    fasta = Fasta(str(DATA / "hg38.fa"), as_raw=True, sequence_always_upper=True)
    cds = parse_gencode_cds(
        DATA / "gencode.v50.basic.annotation.gtf.gz",
        chroms=SPLIT_CHROMS[Split.TRAIN],
        min_len=150,
    )
    print(f"{len(cds):,} train CDS exons >= 150 nt")

    t = time.time()
    tokens, labels = sample_windows(cds, fasta, n=2000, window=WINDOW, seed=0)
    print(f"sampled {tokens.shape} in {time.time() - t:.1f}s\n")

    idx = np.tile(np.arange(WINDOW) % 3, (len(labels), 1))
    mask = labels >= 0
    counts = np.zeros((3, 3), dtype=int)
    np.add.at(counts, (idx[mask], labels[mask]), 1)
    leakage = counts.max(axis=1).sum() / counts.sum()
    print(f"index % 3 -> phase, best fixed map: {leakage:.4f}   (chance 0.3333)")
    assert leakage < 0.40, "windows are not offset-randomised"

    print("\nstop-codon rate by reading frame:")
    for shift, name in [(0, "in-frame (our labels)"), (1, "shifted +1"), (2, "shifted +2")]:
        rate, n = stop_rate(tokens, labels, shift)
        print(f"  {name:24s} {rate:.4f}  (n={n:,})")

    in_frame, _ = stop_rate(tokens, labels, 0)
    off_1, _ = stop_rate(tokens, labels, 1)
    off_2, _ = stop_rate(tokens, labels, 2)
    assert in_frame < 1e-4, f"in-frame stops at {in_frame:.4f}: phase or strand arithmetic is wrong"
    assert min(off_1, off_2) > 0.02, "shifted frames should be littered with stops"
    print("\nOK — biology agrees with the frame labels.")


if __name__ == "__main__":
    main()
