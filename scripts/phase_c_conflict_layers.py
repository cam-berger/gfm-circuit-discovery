"""Cue conflict, read at every layer AND at the model's output.

D33's "anchor share under cue conflict" -- the y-axis of the 2x2 and the number that named the relay --
was a linear probe on `L2.resid_post`. D34 then showed that in cells C and D the anchor does not enter
the residual stream until L4. So that axis asked a layer about a variable it does not yet carry.

This is the same fault D34 found on the x-axis, on the other axis. Fix it the same way: sweep depth,
and end at the only readout that cannot be accused of probing the wrong place -- the model's own output.

The conflict: insert one nucleotide at start+3. Downstream content keeps its codon structure but slides
one base, so the anchor and the local content now disagree about the frame by exactly one.

    python scripts/phase_c_conflict_layers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
from phase_c_relay import load, per_position  # noqa: E402
from phase_c_t3 import frame_labels  # noqa: E402
from train_cell import pool  # noqa: E402

from gfmcd.data.genome import IGNORE, Split  # noqa: E402
from gfmcd.signatures import all_residuals, fit_probe, frameshift, start_index  # noqa: E402

OUT = Path("results/reading-frame/fig_data")
N = 800
FAR = 64

# All five cells of the 2x2 plus the LM. The mu=0 arm has a nucleotide head, so "follows the anchor"
# is undefined at its output; it is probe-only, and its verdict rests on the max over depth.
MODELS = [
    ("A · uniform, easy detector", "nano_uniform_step8400.pt", "A", True),
    ("B · uniform, hard detector", "cellB_s1_step24000.pt", "B", True),
    ("C · real CDS, easy detector", "cellC_s0_step8000.pt", "C", True),
    ("D · real CDS, natural", "cellD_s0_step8000.pt", "D", True),
    ("μ=0 · real CDS, LM objective", "nano_ar_internal_step8000.pt", "D", False),
]


def shift_mask(body: np.ndarray, at: np.ndarray) -> np.ndarray:
    """`frameshift` slides everything at/after `at` one base right. Slide the coding mask with it."""
    out = np.zeros_like(body)
    for r, i in enumerate(at):
        out[r, :i] = body[r, :i]
        out[r, i + 1 :] = body[r, i:-1]
    return out


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    for key, ckpt, cell, has_aa_head in MODELS:
        model = load(ckpt, device)
        tok, aa, ph = pool(cell, N, model.cfg.max_len, Split.VAL, 4242)
        start = start_index(ph)
        body = ph >= 0
        ins = start + 3
        sh = frameshift(tok, ins)
        body_sh = shift_mask(body, ins)

        pos = np.arange(tok.shape[1])[None, :]
        far = body_sh & (pos - start[:, None] >= FAR)

        # --- probe at every layer ------------------------------------------------------------
        R0 = all_residuals(model, tok, device)
        R1 = all_residuals(model, sh, device)
        y = torch.from_numpy(ph[body]).long()
        d_atg = pos - start[:, None]

        layers = {}
        for name in R0:
            probe, mu, sd = fit_probe(R0[name][torch.from_numpy(body)], y)
            x = (R1[name][torch.from_numpy(far)] - mu) / sd
            pred = probe(x).argmax(-1).numpy()
            da = d_atg[far]
            layers[name] = dict(anchor=float((pred == da % 3).mean()),
                                content=float((pred == (da - 1) % 3).mean()))

        # --- the model's own output ----------------------------------------------------------
        out_row, decay = None, None
        if has_aa_head:
            aa_anchor = frame_labels(sh, body_sh, start)       # frame the (unmoved) ATG announces
            aa_content = frame_labels(sh, body_sh, start + 1)  # frame the slid codons carry
            c_anchor, _ = per_position(model, sh, aa_anchor, device)
            c_content, _ = per_position(model, sh, aa_content, device)
            scored = body_sh & (aa_anchor != IGNORE) & (aa_content != IGNORE)
            m = far & scored
            out_row = dict(anchor=float(c_anchor[m].mean()), content=float(c_content[m].mean()), n=int(m.sum()))

            # The same readout, resolved by distance -- the honest replacement for the L2-probe
            # decay curve of Phase B.
            d_ins = pos - (ins[:, None] + 1)
            decay = []
            for lo, hi in ((4, 15), (16, 31), (32, 63), (64, 127), (128, 255)):
                s = scored & (d_ins >= lo) & (d_ins <= hi)
                if s.sum() < 200:
                    continue
                decay.append([(lo + hi) / 2, float(c_anchor[s].mean()), float(c_content[s].mean()), int(s.sum())])

        # resid.embed carries no frame, so its probe is at chance and its "anchor share" is noise
        # around 1/3. Peak over the transformer blocks only.
        blocks = [k for k in layers if k != "resid.embed"]
        peak_layer = max(blocks, key=lambda k: layers[k]["anchor"])
        peak = layers[peak_layer]
        results[key] = dict(layers=layers, output=out_row, decay=decay,
                            peak_anchor=peak["anchor"], peak_layer=peak_layer)

        print(f"\n=== {key}   (far field, >= {FAR} nt)")
        print(f"{'readout':>16}{'follows anchor':>17}{'follows content':>18}")
        for name, r in layers.items():
            print(f"{name:>16}{r['anchor']:>17.4f}{r['content']:>18.4f}")
        if out_row:
            print(f"{'MODEL OUTPUT':>16}{out_row['anchor']:>17.4f}{out_row['content']:>18.4f}")
        else:
            print(f"{'MODEL OUTPUT':>16}{'-- (nt head)':>17}")
        print(f"   peak anchor share over depth: {peak['anchor']:.4f} at {peak_layer}")

    (OUT / "phase_c_conflict_layers.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}/phase_c_conflict_layers.json")


if __name__ == "__main__":
    main()
