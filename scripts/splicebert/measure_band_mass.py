#!/usr/bin/env python
"""Recover the ONE quantity Level A used but never wrote down: each head's ATTENTION MASS in
the tested offsets.

run_postsoftmax.py used band mass as a GATE -- a head under the 2% floor was dropped from the
scan -- but it only ever stored the heads that passed, not the value.  Figure 4A wants it as
the point size, and it is not a number I am willing to invent.  So it is measured again here.

This is ONE intact forward pass.  No deletions, no permutations, no training.  The inputs are
bit-reproducible from the original run (np.random.default_rng(7), hg38 chr8:40,000,000-40,200,000,
the first 48 windows of 508 clean nt), so the profiles must come out identical -- and the script
PROVES that by recomputing m_T from them and checking it against the recorded Level-A value.
If m_T reproduces, the band mass beside it is trustworthy.  If it does not, the script stops.

MPS SAFETY: attention for all 6 layers at once would be 6 x 48 x 16 x 510 x 510 x 4 B = 4.8 GB.
It is therefore CHUNKED (8 sequences at a time) and accumulated, which is arithmetically
identical to the original's batch mean.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")

import numpy as np                                                        # noqa: E402
import torch                                                              # noqa: E402
from transformers import AutoTokenizer                                    # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/splicebert"))
from gfmcd.splicing.circuit import load_adapter                           # noqa: E402
from run_postsoftmax import BAND, MODELS_A, m_of, FROZEN                  # noqa: E402

OUT = REPO / "results/phase-lattice/band_mass.json"
SPAN, CHUNK = 60, 8


def main() -> None:
    from pyfaidx import Fasta
    rng = np.random.default_rng(7)                       # <- the original seed, exactly
    fa = Fasta("/Users/cberg/Research/data/genomes/hg38.fa", as_raw=True)
    s = np.frombuffer(fa["chr8"][40_000_000:40_200_000].upper().encode(), dtype=np.uint8)
    ok = np.isin(s, [65, 67, 71, 84])
    st = [i for i in rng.integers(0, len(s) - 520, 600) if ok[i:i + 508].all()][:48]
    B2I = {65: 6, 67: 7, 71: 8, 84: 9}
    print(f"{len(st)} windows of 508 nt from hg38 chr8 (identical to the Level-A run)")

    rec, recorded = {}, json.loads(
        (REPO / "results/phase-lattice/postsoftmax.json").read_text())["models"]

    for name, (path, carrier, recip, T) in MODELS_A.items():
        ad = load_adapter(str(REPO / path), "cpu", eager=True)
        m = ad.model
        m.eval()
        tok = AutoTokenizer.from_pretrained(str(REPO / path))
        body = [[B2I[int(b)] for b in s[i:i + 508]] for i in st]
        ids = torch.tensor([[tok.cls_token_id] + b + [tok.sep_token_id] for b in body])

        S = ids.shape[1]
        lo, hi = SPAN + 10, S - SPAN - 10
        q = np.arange(lo, hi)
        acc, nq = {}, 0
        for i in range(0, len(ids), CHUNK):
            with torch.no_grad():
                att = m.bert(input_ids=ids[i:i + CHUNK], output_attentions=True).attentions
            for l, a in enumerate(att):
                a = a.float().numpy()
                for h in range(a.shape[1]):
                    Mh = a[:, h]
                    v = np.array([Mh[:, q, q + k].sum() for k in range(-SPAN, SPAN + 1)])
                    key = f"L{l}H{h}"
                    acc[key] = acc.get(key, 0.0) + v
            nq += a.shape[0] * len(q)
            del att
        prof = {k: v / nq for k, v in acc.items()}

        rows, checked = {}, []
        for h, pr in prof.items():
            e = pr[60 + BAND]
            bm = float(e.sum() / max(pr.sum(), 1e-30))
            rows[h] = bm
            if bm < FROZEN["band_mass_floor"]:
                continue
            got = float(abs(m_of(e, T)[0]))
            want = next((x["m_T"] for x in recorded[name]["heads"] if x["head"] == h), None)
            if want is not None:
                checked.append((h, got, want))
        bad = [(h, g, w) for h, g, w in checked if abs(g - w) > 2e-3]
        print(f"\n{name}: {len(rows)} heads, {sum(v >= FROZEN['band_mass_floor'] for v in rows.values())} "
              f"clear the {FROZEN['band_mass_floor']:.0%} floor")
        print(f"   m_T reproduced for {len(checked)} scanned heads; "
              f"worst |delta| = {max((abs(g-w) for _, g, w in checked), default=0):.2e}")
        if bad:
            sys.exit(f"   m_T DOES NOT REPRODUCE for {len(bad)} heads, e.g. {bad[:3]}.  STOP -- "
                     "the inputs are not the ones the Level-A run used, so the band mass would "
                     "not be either.")
        print("   -> inputs verified identical to the Level-A run; band mass is trustworthy.")
        rec[name] = rows
        del m, ad

    OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
