"""Verify the rebuilt pooled oracle against D30's aggregate ceiling before Phase C leans on it.

D30 reported, for real hg38 CDS with no start codon, codon-level naive Bayes:

    ctx  60 nt   0.7542
    ctx 120 nt   0.9216
    ctx 240 nt   0.9509

Those were measured on pure-CDS windows. Cells C and D use *anchored* windows, which carry 5' UTR --
so the oracle is weaker there, and that weaker number is the one T2 must condition on. Both are printed.

    python scripts/pooled_oracle.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from pyfaidx import Fasta

sys.path.insert(0, "scripts")
from train_cell import pool  # noqa: E402

from gfmcd.data.genome import SPLIT_CHROMS, Split, parse_gencode_cds, sample_windows  # noqa: E402
from gfmcd.oracle import oracle_correct  # noqa: E402
from gfmcd.paths import project_paths  # noqa: E402

DATA = project_paths().references
N = 600
FAR = 64


def main() -> None:
    fasta = Fasta(str(DATA / "hg38.fa"), as_raw=True, sequence_always_upper=True)
    cds = parse_gencode_cds(DATA / "gencode.v50.basic.annotation.gtf.gz",
                            chroms=SPLIT_CHROMS[Split.VAL], min_len=150)
    tok, ph = sample_windows(cds, fasta, n=2000, window=256, seed=4242)

    print("coding positions of real CDS windows (D30's setting: 0.7542 / 0.9216 / 0.9509)")
    print(f"{'ctx (nt)':>10}{'coding-masked':>16}{'as the model sees it':>24}")
    for ctx in (30, 60, 120):
        body = ph >= 0
        masked = oracle_correct(tok, ph, ctx=ctx, mask_noncoding=True)[body].mean()
        raw = oracle_correct(tok, ph, ctx=ctx)[body].mean()
        print(f"{2 * ctx:>10}{masked:>16.4f}{raw:>24.4f}")

    print("\nanchored windows, val split (what cells C and D are evaluated on)")
    print(f"{'cell':>6}{'all pos':>12}{'far (>=64)':>14}{'far err':>10}")
    for cell in ("C", "D"):
        t, _, p = pool(cell, N, 256, Split.VAL, 4242)
        ok = oracle_correct(t, p, ctx=120)
        body = p >= 0
        start = np.argmax(p == 0, axis=1)[:, None]
        far = body & (np.arange(t.shape[1])[None, :] - start >= FAR)
        print(f"{cell:>6}{ok[body].mean():>12.4f}{ok[far].mean():>14.4f}{1 - ok[far].mean():>10.4f}")


if __name__ == "__main__":
    main()
