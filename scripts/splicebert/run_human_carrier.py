#!/usr/bin/env python
"""IS THE HUMAN-ONLY CHECKPOINT'S 10.165 PEAK A REAL CARRIER, OR AN INCIDENTAL BUMP?

The human-only model was written off as "no carrier" -- but that was a statement about BIN 48.
Its strongest mode in the 9-12 band is at BIN 50 (lambda ~ 10.165 sub-bin, ~12.7x median), which
is a DIFFERENT object, and it sits close to the NUCLEOSOMAL wavelength (10.298, D28) rather than
to the multispecies carrier (10.630).

    multispecies stage-1 (512):  bin 48   lambda = 10.607   370x median
    human-only   (512):          bin 50   lambda = 10.165    13x median   <- THIS
    nucleosomal dyad signal:              lambda = 10.298   (measured, D28)

THE DECIDING CRITERION (the user's, and it is the right one)
-----------------------------------------------------------
    "A position-table peak without a QK/post-softmax reader is probably incidental.
     A functional reader makes it a genuine second carrier."

So the question is NOT "is there power at bin 50" -- there is.  It is: DOES ANY HEAD READ IT?
A carrier that nothing reads is a fact about the table's noise floor, not about the model.

WHAT IS MEASURED
    1. P's spectrum: peak bin, enrichment, CIRCULARITY (a rank-2 rotary code has none of its own),
       and the sidebands -- a precessing carrier shows them, a spike does not.
    2. Every head's QK readout R(k), with a FREE-FREQUENCY scan: what period does each head
       actually read?  The envelope and the period-3 term are fitted JOINTLY, so the scan cannot
       merely re-find either.
    3. The CAUSAL test: delete bins 49-51 from P and re-measure every head's amplitude.
       A head that READS the band collapses.  A head that does not is untouched.
    4. THE CONTROL THAT CAN FAIL: delete bins 46-48 instead -- where the MULTISPECIES carrier
       lives and where this model has nothing.  If deleting an EMPTY band damages the same heads,
       the "reader" is an artefact of deleting anything.

⚠️ EVERYTHING HERE IS PRE-SOFTMAX.  R(k) is the QK LOGIT as a function of offset.  R14's lesson,
in a new costume: a head whose logits oscillate hugely can still have a spike for attention
(softmax of [big smooth wave + one sharp peak] IS the peak).  "N heads read the carrier" is a
statement about logits until the post-softmax modulation is measured.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/splicebert"))

from gfmcd.splicing.circuit import load_adapter, pick_device      # noqa: E402
from run_second_carrier import amp2, periodogram                  # noqa: E402  the VALIDATED fits
from transformers import AutoTokenizer                            # noqa: E402

HUMAN = "data/models/splicebert_official_archive/extracted/models/SpliceBERT-human.510nt"
MULTI = "data/models/splicebert_official_archive/extracted/models/SpliceBERT.510nt"
OUT = REPO / "results/phase-lattice/human_carrier.json"

NSEQ, SPAN = 32, 60
LO, HI = 120, 390                       # LO-SPAN >= 0 and HI+SPAN < 510
K = np.arange(-SPAN, SPAN + 1, dtype=float)
BAND_50 = [49, 50, 51]                  # the human-only peak
BAND_48 = [46, 47, 48]                  # the MULTISPECIES carrier -- EMPTY here.  The control.


def audit(path: str, name: str, peak_hint: int, T_hint: float, dev) -> dict:
    ad = load_adapter(str(REPO / path), dev)
    m = ad.model
    emb = m.bert.embeddings.position_embeddings
    P0 = emb.weight.data.clone()
    P = P0.detach().float().cpu().numpy()
    L, D = P.shape
    mu = P.mean(0, keepdims=True)
    F = np.fft.rfft(P - mu, axis=0)
    pw = (np.abs(F) ** 2).sum(1)
    pw[0] = 0.0
    med = float(np.median(pw[1:]))

    lo, hi = int(L / 12.0), int(L / 9.0)
    peak = int(lo + np.argmax(pw[lo:hi + 1]))
    sv = np.linalg.svd(np.stack([F[peak].real, F[peak].imag]), compute_uv=False)
    circ = float(sv[1] / sv[0])
    print(f"\n{'='*88}\n{name}  (L = {L})\n{'='*88}")
    print(f"  9-12 band peak: bin {peak}  (lambda = {L/peak:.4f})   {pw[peak]/med:.1f}x median")
    print(f"  CIRCULARITY at the peak = {circ:.3f}   "
          f"({'a CIRCLE -> rank-2 rotary, no phase of its own' if circ > 0.7 else 'a LINE -> standing wave, one phase'})")
    print(f"  sidebands: bin {peak-1} = {pw[peak-1]/med:.1f}x,  bin {peak+1} = {pw[peak+1]/med:.1f}x")

    def strip(bins):
        G = F.copy()
        for b in bins:
            G[b] = 0
        return torch.tensor(np.fft.irfft(G, n=L, axis=0) + mu, dtype=P0.dtype, device=P0.device)

    tok = AutoTokenizer.from_pretrained(str(REPO / path))
    rng = np.random.default_rng(0)
    seqs = ["".join(rng.choice(list("ACGT"), 510)) for _ in range(NSEQ)]
    enc = tok([" ".join(s) for s in seqs], return_tensors="pt", padding=True).to(ad.device)
    nH, nL = m.config.num_attention_heads, m.config.num_hidden_layers
    dh = m.config.hidden_size // nH

    def hidden(Pw):
        emb.weight.data = Pw.clone()
        with torch.no_grad():
            hs = m.bert(**enc, output_hidden_states=True).hidden_states
        return [h.mean(0).float().cpu().numpy() for h in hs]

    def Rk(l, h, Hin):
        sa = m.bert.encoder.layer[l].attention.self
        Wq = sa.query.weight.detach().float().cpu().numpy()[h*dh:(h+1)*dh]
        bq = sa.query.bias.detach().float().cpu().numpy()[h*dh:(h+1)*dh]
        Wk = sa.key.weight.detach().float().cpu().numpy()[h*dh:(h+1)*dh]
        bk = sa.key.bias.detach().float().cpu().numpy()[h*dh:(h+1)*dh]
        Q, Kk = Hin @ Wq.T + bq, Hin @ Wk.T + bk
        idx = np.arange(LO, HI)
        return np.array([float(np.mean(np.sum(Q[idx]*Kk[idx+int(k)], axis=1))/np.sqrt(dh))
                         for k in K])

    Hs = {"intact": hidden(P0), "no_50": hidden(strip(BAND_50)), "no_48": hidden(strip(BAND_48))}
    emb.weight.data = P0.clone()

    periods = np.arange(8.0, 13.01, 0.02)
    rows = []
    for l in range(nL):
        for h in range(nH):
            y = {c: Rk(l, h, Hs[c][l]) for c in Hs}
            pg = periodogram(y["intact"], periods)
            T_hat = float(periods[int(np.argmax(pg))])
            a_int = amp2(y["intact"], T_hint)[1]
            a_d50 = amp2(y["no_50"], T_hint)[1]
            a_d48 = amp2(y["no_48"], T_hint)[1]
            rows.append(dict(head=f"L{l}H{h}", T_free=T_hat, peak_amp=float(pg.max()),
                             amp=float(a_int), kept_del50=float(a_d50/max(a_int, 1e-9)),
                             kept_del48=float(a_d48/max(a_int, 1e-9))))
    rows.sort(key=lambda r: -r["amp"])

    print(f"\n  QK READOUT — every head, amplitude of the {T_hint:.3f} tone in attention LOGITS")
    print(f"  (envelope + period-3 fitted jointly, so the free scan cannot re-find either)\n")
    print(f"  {'head':7} {'amp':>7} {'free T':>7} | {'kept if bins '+str(BAND_50)+' deleted':>28} | "
          f"{'kept if bins '+str(BAND_48)+' deleted':>28}")
    print("  " + "-" * 96)
    for r in rows[:10]:
        reads = r["kept_del50"] < 0.25 and r["kept_del48"] > 0.6
        print(f"  {r['head']:7} {r['amp']:>7.3f} {r['T_free']:>7.2f} | "
              f"{r['kept_del50']*100:>26.0f}% | {r['kept_del48']*100:>26.0f}%"
              f"{'   <-- READS IT' if reads else ''}")

    readers = [r for r in rows if r["kept_del50"] < 0.25 and r["kept_del48"] > 0.6
               and r["amp"] > 0.2]
    print(f"\n  HEADS THAT GENUINELY READ THE BAND (amp > 0.2, collapse on its deletion,")
    print(f"  and SURVIVE deletion of the empty control band): {len(readers)}")
    if readers:
        print("     " + ", ".join(r["head"] for r in readers))
    else:
        print("     NONE.  The peak is in the table but nothing reads it -> INCIDENTAL.")

    del ad
    return dict(L=L, peak_bin=peak, wavelength=L/peak, enrichment=float(pw[peak]/med),
                circularity=circ, sideband_lo=float(pw[peak-1]/med),
                sideband_hi=float(pw[peak+1]/med),
                n_readers=len(readers), readers=[r["head"] for r in readers],
                heads=rows)


def main() -> None:
    dev = pick_device(None)
    rec = {}
    rec["human_only"] = audit(HUMAN, "HUMAN-ONLY 510nt — is bin 50 real?", 50, 10.165, dev)
    rec["multispecies"] = audit(MULTI, "MULTI-SPECIES 510nt — the known carrier, as a POSITIVE CONTROL",
                                48, 10.607, dev)
    OUT.write_text(json.dumps(rec, indent=1))

    h, m2 = rec["human_only"], rec["multispecies"]
    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    print(f"  {'':22} {'human-only':>14} {'multi-species':>16}")
    for lbl, k in (("peak bin", "peak_bin"), ("wavelength", "wavelength"),
                   ("enrichment", "enrichment"), ("circularity", "circularity"),
                   ("HEADS THAT READ IT", "n_readers")):
        a, b = h[k], m2[k]
        fa = f"{a:.3f}" if isinstance(a, float) else str(a)
        fb = f"{b:.3f}" if isinstance(b, float) else str(b)
        print(f"  {lbl:22} {fa:>14} {fb:>16}")
    print()
    if h["n_readers"] == 0:
        print("  ==> The human-only peak is INCIDENTAL. Power in the table, but no reader.")
        print("      It is not a second carrier. Do not build a story on it.")
    else:
        print(f"  ==> The human-only peak IS READ, by {h['n_readers']} head(s). It is a genuine,")
        print("      DIFFERENT carrier -- and it sits near the NUCLEOSOMAL wavelength (10.298).")
    print(f"\n  (the multi-species column is the POSITIVE CONTROL: it must show readers, or the")
    print("   whole reader test is broken and the human-only null means nothing)")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
