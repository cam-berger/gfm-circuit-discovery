#!/usr/bin/env python
"""THE SYNTHETIC ARM: can a LOCALLY PHASE-RESET band be quantized into a sharp bin-48 carrier?

The hypothesis (the user's, now a concrete causal claim):

    "The 512-position grid may have QUANTIZED a weak, broad, nucleosome-scale band into a sharp
     bin-48 positional carrier."

The dyad assay asks whether the biological SOURCE exists.  This asks whether the ARCHITECTURE can
manufacture the sharp tone from a broad, globally-incoherent one.  It needs no external data.

THE DESIGN, AND WHY THE SHIFTED CENTRES ARE THE WHOLE POINT
----------------------------------------------------------
A single band-vs-control experiment CANNOT distinguish "the data's band was compressed onto the
Fourier grid" from "this architecture likes bin 48".  So we move the band and see if the bin
moves with it:

    512/47 = 10.894      512/48 = 10.6667      512/49 = 10.449

    corpus          local signal                                     PREDICTED winning bin
    control         same composition, periodic ORDER destroyed       none (generic spectrum)
    centre 10.67    T ~ U(10.37,10.97), phase reset every 150-300nt  48
    centre 10.45    same bandwidth/coherence, centred 10.45          49
    centre 10.85    same bandwidth/coherence, centred 10.85          47

  * centres -> 49/48/47   => the data's local covariance is QUANTIZED onto the table's grid
  * every periodic corpus -> 48  => the architecture has a privileged bin-48 mode
  * several neighbouring bins   => the band is represented WITHOUT winner-take-all compression
  * seeds pick different bins   => symmetry breaking in a broad band; exact 10.67 means little
  * no difference from control  => local WW/SS covariance alone is insufficient

SIGNAL.  WW and SS are modulated COUNTERPHASE, so overall composition is preserved:
    p_W(x) = p_W0 + depth*cos(2 pi x/T + phi)/2        (and p_S falls by the same amount)
Phase resets every block (150-300 nt), and CROP ORIGINS ARE RANDOMISED relative to block
boundaries, so ABSOLUTE position can never reveal the phase.  The model can only win by learning
a RELATIVE-position ruler at that wavelength -- which is exactly the mechanism in question.

CONTROL.  The modulation values are PERMUTED WITHIN EACH BLOCK: identical base-frequency
distribution and entropy, periodic ORDERING destroyed.  (A composition-matched i.i.d. control
would also differ in entropy; this one does not.)

PAIRED.  Identical init, vocabulary, optimiser, schedule and token budget across corpora.

⛔ DO NOT RUN THIS AS WRITTEN.  THE PILOT FAILED AS AN INSTRUMENT, AND THE REASON IS ANALYTIC.
--------------------------------------------------------------------------------------------
Pilot (2500 steps, depth 10%, centre 10.67 vs control): BOTH models converged to MLM loss
**1.370** -- and 1.370 is EXACTLY H_binary(0.59) + ln2, the entropy of the base composition.
The models learned composition and nothing else.  Both picked winning bin 44.  No effect.

That is not a result.  It is a corpus with no learnable signal in it.  THE ENTROPY GAIN FROM A
WW/SS MODULATION IS QUADRATIC IN DEPTH:

    depth   best achievable loss   headroom      % of the loss
     2%           1.3699            0.0001          0.01%
     5%           1.3694            0.0006          0.05%
    10%           1.3674            0.0026          0.19%     <- the "strong instrument check"
    25%           1.3537            0.0163          1.19%
    50%           1.3027            0.0673          4.91%     <- first depth worth training on
   100%           1.1148            0.2552         18.63%

...and that is the ceiling WITH THE PHASE HANDED TO THE MODEL.  It must instead INFER the phase
from context, which costs more than 0.0026 nats buys.  So no model, at any size or step count,
would be driven to learn a period-10.67 ruler from the 10% corpus.  All 4 corpora x 5 seeds x 3
depths would have returned the same uninformative null, at ~4 min of GPU each.

BEFORE ANY MORE COMPUTE, THE DESIGN NEEDS ONE OF THESE (they are not exclusive):
  1. RUN THE INSTRUMENT CHECK AT depth >= 50%, not 10%.  Establish that the mechanism EXISTS at
     all (does bin 48 sharpen when the signal is unmissable?), THEN walk the dose down.  The
     biologically-plausible 2-5% region may simply be unreachable for a small MLM -- and if so,
     THAT is the finding, and it is worth stating.
  2. RAISE THE SIGNAL-TO-NOISE PER TOKEN.  A WW/SS class modulation wastes most of its depth on
     the A-vs-T and G-vs-C coin flips (the ln2 term), which are pure noise.  Modulating a
     SINGLE base's probability, or reducing the alphabet, puts the same depth into a channel the
     model can actually price.
  3. LENGTHEN THE COHERENCE.  Blocks of 150-300 nt mean the model must ALSO find the block
     boundary before the ruler pays off.  A first check with one phase per crop isolates the
     ruler question from the change-point question.
  4. MEASURE THE HEADROOM FIRST, ALWAYS.  H(composition) is computable on paper in one line.
     Any corpus whose headroom is below ~0.02 nats cannot test anything.

FROZEN OUTCOMES (declared here, before any run)
    PRIMARY   Q_k = power(k) / sum_{j=43..53} power(j)   at the PREDICTED bin, and the WINNING bin
    SECONDARY peak-to-neighbour ratio; carrier circularity; carrier-plane rank;
              QK readout amplitude; post-softmax attention modulation;
              MLM loss on held-out periodic vs control;  loss after DELETING the learned carrier
    The last three separate "the table CONTAINS the frequency" from "the model USES it".
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from gfmcd.splicing.circuit import pick_device  # noqa: E402

from transformers import BertConfig, BertForMaskedLM  # noqa: E402

BASES = "ACGT"
MASK_ID, PAD_ID = 4, 5
VOCAB = 6
BIN_OF = {10.67: 48, 10.45: 49, 10.85: 47}      # 512/48, 512/49, 512/47
BAND = list(range(43, 54))                       # the frozen normalisation band
OUT = REPO / "results/phase-lattice/synth_quantization.json"


# ------------------------------------------------------------------ corpus
def make_corpus(n: int, centre: float | None, depth: float, seed: int,
                shuffle_mod: bool = False, gc: float = 0.41) -> np.ndarray:
    """Synthetic DNA.  If `centre` is None (or shuffle_mod) the periodic ORDER is destroyed."""
    r = np.random.default_rng(seed)
    pW0 = 1.0 - gc
    out = np.empty(n, dtype=np.int8)
    i = 0
    while i < n:
        L = int(r.integers(150, 301))
        L = min(L, n - i)
        if centre is None:
            m = np.zeros(L)
        else:
            T = float(r.uniform(centre - 0.30, centre + 0.30))
            ph = float(r.uniform(0, 2 * math.pi))
            m = depth * np.cos(2 * math.pi * np.arange(L) / T + ph)
            if shuffle_mod:
                m = r.permutation(m)          # same values, same entropy, ORDER destroyed
        pW = np.clip(pW0 + m / 2, 0.03, 0.97)
        isW = r.random(L) < pW
        b = np.where(isW, r.choice([0, 3], L), r.choice([1, 2], L))   # A/T  vs  C/G
        out[i:i + L] = b
        i += L
    return out


# ------------------------------------------------------------------ model
def build(seed: int, npos: int, dev) -> BertForMaskedLM:
    torch.manual_seed(seed)                      # IDENTICAL INIT across corpora
    cfg = BertConfig(vocab_size=VOCAB, hidden_size=128, num_hidden_layers=4,
                     num_attention_heads=4, intermediate_size=512,
                     max_position_embeddings=npos, type_vocab_size=1,
                     pad_token_id=PAD_ID, attn_implementation="eager")
    return BertForMaskedLM(cfg).to(dev)


def train(model, corpus, steps, npos, dev, seed, bs=32, lr=1e-3, mask=0.15):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    g = np.random.default_rng(seed + 7)
    model.train()
    t0 = time.time()
    for s in range(steps):
        # CROP ORIGINS RANDOMISED -- absolute position cannot reveal the phase
        st = g.integers(0, len(corpus) - npos, size=bs)
        x = np.stack([corpus[o:o + npos] for o in st]).astype(np.int64)
        ids = torch.tensor(x, device=dev)
        lab = ids.clone()
        m = torch.rand(ids.shape, device=dev) < mask
        lab[~m] = -100
        ids = ids.masked_fill(m, MASK_ID)
        loss = model(input_ids=ids, labels=lab).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if (s + 1) % max(1, steps // 4) == 0:
            print(f"      step {s+1}/{steps}  loss {loss.item():.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model


# ------------------------------------------------------------------ readout
def spectrum(model):
    P = model.bert.embeddings.position_embeddings.weight.detach().float().cpu().numpy()
    Fp = np.fft.rfft(P - P.mean(0, keepdims=True), axis=0)
    pw = (np.abs(Fp) ** 2).sum(1)
    pw[0] = 0.0
    return P, Fp, pw


def readout(model, npos):
    P, Fp, pw = spectrum(model)
    band = [b for b in BAND if b < len(pw)]
    tot = float(sum(pw[b] for b in band))
    Q = {int(b): float(pw[b] / tot) for b in band}
    win = int(max(band, key=lambda b: pw[b]))
    nb = [b for b in (win - 1, win + 1) if 0 < b < len(pw)]
    p2n = float(pw[win] / np.mean([pw[b] for b in nb])) if nb else float("nan")
    M = np.stack([Fp[win].real, Fp[win].imag])
    sv = np.linalg.svd(M, compute_uv=False)
    return dict(Q=Q, win_bin=win, win_period=npos / win, p2n=p2n,
                circularity=float(sv[1] / sv[0]),
                band_power=tot, med_power=float(np.median(pw[1:])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--depth", type=float, default=0.10)
    ap.add_argument("--npos", type=int, default=512)
    ap.add_argument("--mb", type=float, default=4.0)
    ap.add_argument("--corpora", default="control,10.67")
    args = ap.parse_args()

    dev = pick_device(None)
    n = int(args.mb * 1e6)
    names = args.corpora.split(",")
    print(f"device {dev} | npos {args.npos} | steps {args.steps} | depth {args.depth:.0%} "
          f"| seeds {args.seeds}")
    print(f"bins: " + "  ".join(f"{args.npos}/{b} = {args.npos/b:.3f}" for b in (47, 48, 49)) + "\n")

    rec: dict = {"npos": args.npos, "steps": args.steps, "depth": args.depth,
                 "band": BAND, "runs": []}
    for seed in range(args.seeds):
        for nm in names:
            centre = None if nm == "control" else float(nm)
            pred = BIN_OF.get(centre) if centre else None
            cor = make_corpus(n, centre if centre else 10.67, args.depth, seed * 100 + 1,
                              shuffle_mod=(nm == "control"))
            print(f"[seed {seed}] corpus {nm:8s} (predict bin {pred})", flush=True)
            model = build(seed, args.npos, dev)
            train(model, cor, args.steps, args.npos, dev, seed)
            r = readout(model, args.npos)
            r.update(seed=seed, corpus=nm, predicted_bin=pred)
            rec["runs"].append(r)
            q = r["Q"]
            print(f"      -> winning bin {r['win_bin']} (period {r['win_period']:.3f}), "
                  f"Q = {q[str(r['win_bin'])] if str(r['win_bin']) in q else q[r['win_bin']]:.3f}, "
                  f"peak/neighbour {r['p2n']:.2f}, circularity {r['circularity']:.3f}")
            if pred:
                print(f"         Q at the PREDICTED bin {pred}: {r['Q'][pred]:.3f}")
            del model
            OUT.write_text(json.dumps(rec, indent=1))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
