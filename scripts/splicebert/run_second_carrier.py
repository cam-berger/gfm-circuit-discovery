#!/usr/bin/env python
"""IS THERE A SECOND POSITIONAL CARRIER IN P, AND WHO READS IT?

Where this came from
--------------------
L0H11 was the ragged comb: one period-3 cosine explained only 80.8% of its readout, against
>=98.4% for the other five.  Raised by the user: "L0H11 seems to have another frequency embedded,
hence why the phase isn't matched."

P's spectrum says what that frequency could be.  Besides the codon clock (bins 339-345, period 3),
there is exactly ONE other high-frequency mode: **bin 96, period 10.69, at 546x the median bin,
rank 7 of 513.**  Everything else in the top ten is slow global drift (periods 22 to 1026).

And the study has been deleting it all along without noticing: the "energy-matched control band"
is bins 94-100, and bin 96 alone is **64% of that band's entire energy**.  The control band IS
the second carrier.  (That does not weaken the period-3 control -- it strengthens it.  Deleting a
comparably strong, genuinely-read carrier still does nothing to splicing.)

What this run does
------------------
1. R(k) over a WIDE window, k = -60..+60 (121 offsets = 11 cycles of period 10.69, vs the 2.3
   cycles the +-12 window gave).  The narrow window could not tell 10.7 from 12, and it could not
   separate a 2.3-cycle oscillation from a polynomial envelope.  This one can.
2. **A FREE-FREQUENCY PERIODOGRAM.**  The period is NOT imposed.  After removing the envelope and
   the period-3 term, scan T in [4, 30] and let R(k) name its own second period.  If the answer
   comes back at P's bin-96 period without being told, that is a real link.
3. ALL 96 HEADS, not just the six combs.  The comb classifier only ever tested congruence mod 3,
   so a head reading period ~10.6 was INVISIBLE to it.  If a second family exists, it was never
   looked for.
4. The double dissociation: delete the period-3 band -> does the 10.6 term survive?  Delete
   bins 94-100 -> does the period-3 term survive?

The empirical noise floor is not assumed: for each frequency it is the amplitude measured in the
condition where that frequency's own band has been deleted.
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

BAND_P3 = list(range(339, 346))          # period 3        -- the codon clock
BAND_P10 = list(range(94, 101))          # period ~10.6    -- the study's "control band"
SPAN = 60                                # k = -60..+60
LO, HI = 260, 760                        # query positions; LO-SPAN >= 0 and HI+SPAN < 1024
NSEQ = 32
OUT = REPO / "results/phase-lattice/second_carrier.json"

W3 = 2 * np.pi / 3
K = np.arange(-SPAN, SPAN + 1, dtype=float)
ENVDEG = 5                               # over 121 points a quintic is a gentle trend, not a wave


def design(w2: float | None, envdeg: int = ENVDEG) -> np.ndarray:
    cols = [K**p for p in range(envdeg + 1)] + [np.cos(W3 * K), np.sin(W3 * K)]
    if w2 is not None:
        cols += [np.cos(w2 * K), np.sin(w2 * K)]
    return np.column_stack(cols)


def amp2(y: np.ndarray, T2: float) -> tuple[float, float, float]:
    """Joint fit: envelope + period-3 + period-T2.  Returns (A3, A_T2, var_explained)."""
    X = design(2 * np.pi / T2)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ b
    denom = (y - np.polyval(b[:ENVDEG + 1][::-1], K)).var()
    return (float(np.hypot(b[ENVDEG + 1], b[ENVDEG + 2])),
            float(np.hypot(b[ENVDEG + 3], b[ENVDEG + 4])),
            float(1 - res.var() / denom) if denom > 0 else 0.0)


def periodogram(y: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Free-frequency scan.  For each candidate T, the amplitude of a period-T term fitted
    JOINTLY with the envelope and the period-3 term -- so it cannot just be re-finding either."""
    return np.array([amp2(y, T)[1] for T in periods])


def main() -> None:
    dev = pick_device(None)
    ad = load_adapter(DEFAULT_MODEL, dev, eager=True)
    m = ad.model
    emb = m.bert.embeddings.position_embeddings
    P0 = emb.weight.detach().clone()
    P = P0.float().cpu().numpy()
    N, _ = P.shape
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)

    def strip(bins):
        G = F.copy()
        for b in bins:
            G[b] = 0
        return torch.tensor(np.fft.irfft(G, n=N, axis=0) + mu, dtype=P0.dtype, device=P0.device)

    tok = AutoTokenizer.from_pretrained(str(REPO / DEFAULT_MODEL))
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 1024)) for _ in range(NSEQ)]
    enc = tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True).to(ad.device)

    nH = m.config.num_attention_heads
    dh = m.config.hidden_size // nH
    nL = m.config.num_hidden_layers

    def hidden_means(Pw, half=None):
        emb.weight.data = Pw.clone()
        with torch.no_grad():
            hs = m.bert(**enc, output_hidden_states=True).hidden_states
        sl = slice(None) if half is None else (slice(0, NSEQ // 2) if half == 0
                                              else slice(NSEQ // 2, NSEQ))
        return [h[sl].mean(0).float().cpu().numpy() for h in hs]

    def Rk(l, h, Hin):
        sa = m.bert.encoder.layer[l].attention.self
        Wq = sa.query.weight.detach().float().cpu().numpy()[h * dh:(h + 1) * dh]
        bq = sa.query.bias.detach().float().cpu().numpy()[h * dh:(h + 1) * dh]
        Wk = sa.key.weight.detach().float().cpu().numpy()[h * dh:(h + 1) * dh]
        bk = sa.key.bias.detach().float().cpu().numpy()[h * dh:(h + 1) * dh]
        Q = Hin @ Wq.T + bq
        Kk = Hin @ Wk.T + bk
        idx = np.arange(LO, HI)
        return np.array([float(np.mean(np.sum(Q[idx] * Kk[idx + int(k)], axis=1)) / np.sqrt(dh))
                         for k in K])

    variants = {"intact": P0, "no_p3": strip(BAND_P3), "no_p10": strip(BAND_P10)}
    H = {name: hidden_means(Pw) for name, Pw in variants.items()}
    H_half = {i: hidden_means(P0, half=i) for i in (0, 1)}       # split-half reliability
    emb.weight.data = P0.clone()

    # ---------------------------------------------------------- 1. LET THE DATA NAME THE PERIOD
    periods = np.arange(4.0, 30.01, 0.05)
    print(f"free-frequency scan over T in [4, 30] on the INTACT model, all {nL*nH} heads")
    print("(the period-3 term and the envelope are fitted jointly, so the scan cannot re-find "
          "either)\n")

    rows = []
    for l in range(nL):
        for h in range(nH):
            y = {c: Rk(l, h, H[c][l]) for c in variants}
            pg = periodogram(y["intact"], periods)
            T_hat = float(periods[int(np.argmax(pg))])
            A_hat = float(pg.max())
            rec = {"layer": l, "head": h, "name": f"L{l}H{h}",
                   "T_free": T_hat, "A_free": A_hat}
            for c in variants:
                a3, a10, ve = amp2(y[c], 1026 / 96)      # the period of P's bin 96, exactly
                rec[c] = {"A3": a3, "A10": a10, "var_explained": ve}
            # split-half reliability, intact only
            hs = [amp2(Rk(l, h, H_half[i][l]), 1026 / 96) for i in (0, 1)]
            rec["split_half"] = {"A3": [hs[0][0], hs[1][0]], "A10": [hs[0][1], hs[1][1]]}
            rec["Rk_intact"] = [float(v) for v in y["intact"]]
            rec["Rk_no_p3"] = [float(v) for v in y["no_p3"]]
            rec["Rk_no_p10"] = [float(v) for v in y["no_p10"]]
            rows.append(rec)
        print(f"  layer {l} done")

    out = {
        "span": SPAN, "offsets": [int(x) for x in K], "lo": LO, "hi": HI, "nseq": NSEQ,
        "envdeg": ENVDEG,
        "band_p3": BAND_P3, "band_p10": BAND_P10,
        "period_p10_exact": 1026 / 96,
        "periods_scanned": [float(x) for x in periods],
        "heads": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out))
    print(f"\nsaved -> {OUT}")

    # ---------------------------------------------------------- quick read-out
    p10 = sorted(rows, key=lambda r: -r["intact"]["A10"])[:10]
    print(f"\nTOP 10 READERS OF THE SECOND CARRIER (period {1026/96:.2f})")
    print(f"{'head':7s} {'A10':>6s} {'A3':>6s} | {'T named by the data':>19s} | "
          f"{'A10 after deleting':>18s}")
    print(f"{'':7s} {'':6s} {'':6s} | {'':19s} | {'its band':>18s}")
    print("-" * 72)
    for r in p10:
        print(f"{r['name']:7s} {r['intact']['A10']:6.3f} {r['intact']['A3']:6.3f} | "
              f"{r['T_free']:19.2f} | {r['no_p10']['A10']:18.3f}")


if __name__ == "__main__":
    main()
