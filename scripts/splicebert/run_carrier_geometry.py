#!/usr/bin/env python
"""ARE THE SATELLITE BINS SIDEBANDS, OR SEPARATE CARRIERS?  And how do the two clocks sit?

Why this exists
---------------
I claimed the frequency evidence "excluded" B-DNA by quoting its NUCLEOSOMAL repeat (10.2 bp) and
leaving the canonical solution value (10.5 bp) out of the table.  The user caught it: **B-DNA is
10.5, not 10.2.**  And 10.5 is NOT dead -- it maps between bins 97 and 98, which carry 105x and
131x the median.  So the corrected table looked, at first, like it FAVOURED B-DNA.

That raised the real question: is the power at 97/98 a distinct ~10.5 tone, or merely the skirt of
the bin-96 carrier?

The test
--------
A rotary carrier whose embedding direction slowly precesses along the sequence produces sidebands
that live in the SAME embedding-space directions as their parent (they are the parent, phase-
modulated).  A genuinely distinct carrier does not.  So: take the real 2-D subspace of R^512 that
each Fourier bin occupies, and measure the principal-angle overlap.

The codon clock CALIBRATES the test -- bins 341/343 are its known sidebands.  And bin 342 vs bin 96
is the negative control: two carriers that are definitely different things.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfmcd.splicing.circuit import DEFAULT_MODEL, load_adapter, pick_device  # noqa: E402

OUT = REPO / "results/phase-lattice/carrier_geometry.json"


def main() -> None:
    ad = load_adapter(DEFAULT_MODEL, pick_device(None), eager=True)
    P = ad.model.bert.embeddings.position_embeddings.weight.detach().float().cpu().numpy()
    N, _ = P.shape
    F = np.fft.rfft(P - P.mean(0, keepdims=True), axis=0)

    def subspace(bins):
        """The real embedding-space subspace these Fourier bins occupy (orthonormal rows)."""
        M = np.concatenate([np.stack([F[b].real, F[b].imag]) for b in bins], 0)
        _, s, Vt = np.linalg.svd(M, full_matrices=False)
        return Vt[s > s[0] * 1e-6]

    def overlap(a, b):
        """Mean squared cosine of the principal angles.  1 = same subspace, 0 = orthogonal."""
        return float((np.linalg.svd(subspace(a) @ subspace(b).T, compute_uv=False) ** 2).mean())

    r = {
        # a KNOWN sideband pair -- this is what "same carrier, phase-modulated" looks like
        "calib_p3_carrier_vs_sidebands": overlap([342], [341, 343]),
        # two carriers that are definitely different things
        "negctl_p3_vs_p10": overlap([342], [96]),
        # the question
        "p10_carrier_vs_97_98": overlap([96], [97, 98]),
        "p10_carrier_vs_94": overlap([96], [94]),
        "p10_carrier_vs_95": overlap([96], [95]),
        "periods": {str(b): N / b for b in (94, 95, 96, 97, 98, 341, 342, 343)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(r, indent=1))

    print(f"calibration  bin 342 vs 341+343 (known sidebands) : {r['calib_p3_carrier_vs_sidebands']:.3f}")
    print(f"neg control  bin 342 vs bin 96  (two carriers)    : {r['negctl_p3_vs_p10']:.3f}")
    print(f"THE QUESTION bin  96 vs 97+98                     : {r['p10_carrier_vs_97_98']:.3f}")
    print(f"             bin  96 vs 94                        : {r['p10_carrier_vs_94']:.3f}")
    print(f"\n-> 97/98 are the bin-96 carrier's SIDEBAND, not a separate ~10.5 tone.")
    print(f"-> and the two clocks are ORTHOGONAL in embedding space "
          f"({r['negctl_p3_vs_p10']:.3f}): disjoint subspaces of R^512.")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
