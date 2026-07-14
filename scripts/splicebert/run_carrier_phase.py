#!/usr/bin/env python
"""THE STRONGEST BINS OF P, AND WHAT "PHASE" EVEN MEANS FOR THEM.

Asked by the user: "can you give me a list of the strongest bins and their corresponding phase?"

The wrinkle: a bin of a [1026 x 512] table does NOT have a phase.  Bin k gives a complex VECTOR
F[k] in C^512 -- a phase per embedding dimension.  Its contribution at position i is

    P_k(i) = (2/N) * [ Re F[k] * cos(2*pi*k*i/N)  -  Im F[k] * sin(2*pi*k*i/N) ]

which traces an ELLIPSE in the 2-D plane spanned by Re F[k] and Im F[k] as i advances.  So:

    circularity = sigma2/sigma1 of [Re F[k]; Im F[k]]
        -> 1.0 : a CIRCLE.  A pure rotation.  EVERY phase in [0, 2pi) is realised in SOME
                 embedding direction, so the bin HAS NO SINGLE PHASE.
        -> 0.0 : a LINE.  A standing wave.  One phase, shared by every dimension.

AND THAT IS THE WHOLE STORY OF THE HEADS' CODON POSITIONS.

Both carriers are near-perfect circles.  So a head does not READ a phase off the table -- it
CHOOSES one, by choosing a direction in the carrier's plane.  Closed form, no fitting to R(k):

    R(k) = DC + Pc*cos(2*pi*k/3) + Qs*sin(2*pi*k/3)
    Pc = (1/2)( W_Q c . W_K c  +  W_Q s . W_K s ) / sqrt(d)     <- symmetric part
    Qs = (1/2)( W_Q c . W_K s  -  W_Q s . W_K c ) / sqrt(d)     <- ROTATION part
    phase = atan2(-Qs, Pc)

The head's codon position IS the rotation angle its QK bilinear form induces on the clock plane.
Predicted vs measured: agrees to 0.8 degrees, worst case, where a whole codon position is 120.

ONE MORE THING FALLS OUT.  The predicted AMPLITUDE comes out 2-3x too small if (c, s) is fitted
over a long window -- because the clock's DIRECTION PRECESSES along the sequence.  That precession
is exactly what the sidebands at 341/343 (and 94/97/98) ARE.  Shrink the fit window and the
amplitude recovers (0.34x -> 0.97x) while the phase does not move: the phase is an angle BETWEEN
W_Q and W_K on the plane, and is invariant under a global rotation of that plane.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from transformers import AutoTokenizer  # noqa: E402

from gfmcd.splicing.circuit import DEFAULT_MODEL, load_adapter, pick_device  # noqa: E402

COMBS = [(0, 0), (0, 11), (1, 10), (1, 12), (2, 2), (2, 15)]
OUT = REPO / "results/phase-lattice/carrier_phase.json"
TH = 2 * np.pi / 3


def main() -> None:
    ad = load_adapter(DEFAULT_MODEL, pick_device(None), eager=True)
    m = ad.model
    P = m.bert.embeddings.position_embeddings.weight.detach().float().cpu().numpy()
    N, _ = P.shape
    F = np.fft.rfft(P - P.mean(0, keepdims=True), axis=0)
    pw = (np.abs(F) ** 2).sum(1)
    pw[0] = 0.0
    med = float(np.median(pw[1:]))

    def geom(k: int):
        M = np.stack([F[k].real, F[k].imag])
        s = np.linalg.svd(M, compute_uv=False)
        amp, ph = np.abs(F[k]), np.angle(F[k])
        conc = float(np.abs((amp * np.exp(2j * ph)).sum()) / amp.sum())   # mod-pi concentration
        return float(s[1] / s[0]), conc

    bins = []
    for k in np.argsort(-pw)[:20]:
        k = int(k)
        circ, conc = geom(k)
        kind = ("codon clock" if 339 <= k <= 345 else
                "second carrier" if 94 <= k <= 100 else
                "slow global drift" if N / k >= 22 else "")
        bins.append(dict(bin=k, period=N / k, power_rel=float(pw[k] / med),
                         circularity=circ, phase_concentration=conc, kind=kind))

    # ---- the heads' phases, predicted in closed form from the clock plane
    tok = AutoTokenizer.from_pretrained(str(REPO / DEFAULT_MODEL))
    rng = np.random.default_rng(0)
    enc = tok([" ".join("".join(rng.choice(list("ACGT"), 1024))) for _ in range(16)],
              return_tensors="pt", padding=True).to(ad.device)
    with torch.no_grad():
        HS = [h.mean(0).float().cpu().numpy()
              for h in m.bert(**enc, output_hidden_states=True).hidden_states]
    nH = m.config.num_attention_heads
    dh = m.config.hidden_size // nH

    def predict(l, h, lo, hi):
        i = np.arange(lo, hi)
        X = np.column_stack([np.ones_like(i), np.cos(TH * i), np.sin(TH * i)]).astype(float)
        B, *_ = np.linalg.lstsq(X, HS[l][i], rcond=None)
        c, s = B[1], B[2]
        sa = m.bert.encoder.layer[l].attention.self
        Wq = sa.query.weight.detach().float().cpu().numpy()[h * dh:(h + 1) * dh]
        Wk = sa.key.weight.detach().float().cpu().numpy()[h * dh:(h + 1) * dh]
        qc, qs, kc, ks = Wq @ c, Wq @ s, Wk @ c, Wk @ s
        Pc = 0.5 * (qc @ kc + qs @ ks) / np.sqrt(dh)
        Qs = 0.5 * (qc @ ks - qs @ kc) / np.sqrt(dh)
        return float(np.hypot(Pc, Qs)), float(np.arctan2(-Qs, Pc))

    W = json.loads((REPO / "results/phase-lattice/fig_phase_data.json").read_text())["wave"]["heads"]
    heads = []
    for (l, h) in COMBS:
        hd = f"L{l}H{h}"
        A_l, ph_l = predict(l, h, 200, 800)          # long window: phase right, amplitude shrunk
        A_s, ph_s = predict(l, h, 490, 511)          # short window: precession has no time to act
        meas = W[hd]["intact"]
        err = float(np.degrees(abs(np.angle(np.exp(1j * (ph_l - meas["phi"]))))))
        heads.append(dict(head=hd, phase_pred=ph_l, phase_meas=meas["phi"], phase_err_deg=err,
                          A_pred_long=A_l, A_pred_short=A_s, A_meas=meas["A"],
                          ratio_long=A_l / meas["A"], ratio_short=A_s / meas["A"],
                          codon_position=meas["codon_position"]))

    rec = dict(bins=bins, heads=heads,
               worst_phase_err_deg=max(h["phase_err_deg"] for h in heads))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))

    print("THE STRONGEST BINS OF P\n")
    print(f"{'bin':>5} {'period':>9} {'power/med':>10} {'circularity':>12} {'phase conc':>11}  kind")
    print("-" * 76)
    for b in bins:
        print(f"{b['bin']:>5} {b['period']:>9.2f} {b['power_rel']:>10.1f} "
              f"{b['circularity']:>12.3f} {b['phase_concentration']:>11.3f}  {b['kind']}")
    print("\nA CIRCLE HAS NO PHASE.  Both carriers are circles -- so each head CHOOSES its phase")
    print("by choosing a direction in the carrier's plane.  Predicted from W_Q, W_K alone:\n")
    print(f"{'head':7s} {'codon':>6s} {'predicted':>11s} {'measured':>11s} {'err':>7s}")
    print("-" * 48)
    for h in heads:
        print(f"{h['head']:7s} {h['codon_position']:>6d} {np.degrees(h['phase_pred']):>+10.1f}° "
              f"{np.degrees(h['phase_meas']):>+10.1f}° {h['phase_err_deg']:>6.1f}°")
    print("-" * 48)
    print(f"  worst: {rec['worst_phase_err_deg']:.1f}°   (a whole codon position is 120°)")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
