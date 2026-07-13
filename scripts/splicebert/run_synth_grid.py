#!/usr/bin/env python
"""THE SYNTHETIC ARM, re-anchored.  Supersedes run_synth_quantization.py (which must not be run).

    Q1 (the user's):  "Can local genomic periodicity create the sharp carrier?"
    Decisive result:  "Broad local bands are quantized into the predicted position-table bins."

WHAT CHANGED, AND WHY -- all of it from analyze_synth_design.py, which cost zero GPU
---------------------------------------------------------------------------------
1. DEPTH.  The plan's 10% / 5% / 2% rungs are ALL below the instrument's floor, and not because
   of the entropy ceiling (D26) -- because of PHASE NOISE.  T ~ U(c-0.3, c+0.3) is a BAND, so it
   DECOHERES in ~60 nt (5.7 periods).  A model can pool only ~+/-30 nt, and from that window:

       depth    sigma(phase)   usable signal
        10%        104 deg         0.19        <- the phase is NOISE
        25%         42 deg         0.77        <- marginal
        50%         21 deg         0.94        <- resolved
        76%         14 deg         0.97        <- resolved

   So the ANCHOR runs at depth 0.76 -- the largest depth the generator does not CLIP.
   (pW = clip(0.59 + depth*cos/2, 0.03, 0.97); above 0.76 the cosine's peak is flattened, which
   at depth 1.0 clips 22.5% of it, distorts the tone, and shifts the composition.)
   Then the dose walks DOWN.  If the mechanism dies before 5%, THAT is the finding.

2. THE NULL FOR "SHARP".  The corpus band is 2.70 BINS WIDE and its own peak holds only
   Q_win = 0.25 of the 43-53 band.  So:
       flat / no structure      Q_win ~ 0.09   (1/11; the surviving control run gave 0.101)
       FAITHFUL   (P mirrors the corpus band)  Q_win ~ 0.25
       QUANTIZED  (P collapses it to one bin)  Q_win >> 0.25
       SpliceBERT's real carrier               Q_win = 0.76
   Q_win vs 0.25 IS THE EXPERIMENT.  Before this, "sharp" had no referent.

3. THE TABLE-LENGTH CONTROL, done correctly.  The effective table length is the CROP LENGTH --
   rows past the crop are NEVER TRAINED, so allocating 512 and cropping 496 gives an effective
   L of 496.  Hold lambda = 10.6667 and move the CROP:
       crop 500 -> bin 47      crop 512 -> bin 48      crop 521 -> bin 49
   The bin NUMBER moves while the PHYSICAL wavelength stays.  If the bin instead sticks at 48,
   the carrier is ARCHITECTURAL and the chromatin story dies.  THIS CONTROL CAN FAIL.
   Tokens are held identical across crops by adjusting steps.

4. EVERY RUN CARRIES ITS OWN POSITIVE CONTROL.  The corpus's OWN spectrum is measured and stored
   beside the model's.  If the band is not in the sequence, the run is void -- not a null.
   (The pilot never checked this.  It happens to pass: centres 10.45/10.67/10.85 give corpus
   bins 49/48/47 at ~11x median.)

5. RESULTS ARE MERGED BY KEY, NEVER CLOBBERED.  run_synth_quantization.py rewrote its whole
   output file on every invocation -- which is how the depth-10% pilot that D26 cites was
   DESTROYED by a later depth-1.0 run.  Its numbers now survive only in a console log.

FROZEN OUTCOMES (declared before the first run)
    PRIMARY    win_bin  -- does it track the centre (49/48/47)?
               Q_win    -- does it beat the FAITHFUL null of 0.25, toward SpliceBERT's 0.76?
    CONTROL    crop 500/512/521 at fixed lambda -> bins 47/48/49  (physical, not architectural)
    NEGATIVE   order-destroyed corpus -> no peak (Q_win ~ 0.09)
    VOID-CHECK corpus_win_bin must equal the predicted bin, or the run is discarded

MPS SAFETY -- a training run crashed this machine once; it will not happen twice
    * ONE process, ONE run at a time, strictly sequential.  Nothing is ever queued in parallel.
    * torch.mps.empty_cache() + del between runs (the old script leaked across runs).
    * sdpa attention, not eager -- eager materialises the full B*H*L*L score matrix.
    * peak RSS is measured and printed for every run; --max-rss-gb aborts if it is exceeded.
    * --dry-run prints the whole plan, with predictions, and touches no GPU.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

# Cap MPS allocation.  BOTH must be set: torch requires low <= high, and the LOW default is 1.4,
# so setting HIGH alone to 0.6 raises "invalid low watermark ratio 1.4" before the GPU is touched.
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/splicebert"))

from gfmcd.splicing.circuit import pick_device            # noqa: E402
from run_synth_quantization import make_corpus            # noqa: E402 -- the VALIDATED generator
from transformers import BertConfig, BertForMaskedLM      # noqa: E402

MASK_ID, PAD_ID, VOCAB = 4, 5, 6
BAND = list(range(43, 54))
FAITHFUL_QWIN = 0.25          # frozen: the corpus's own peak share.  See analyze_synth_design.py
FLAT_QWIN = 1 / len(BAND)     # 0.091
TOTAL_TOKENS = 32 * 512 * 3000        # held IDENTICAL across every run (crop changes steps)
OUT = REPO / "results/phase-lattice/synth_grid.json"


def predicted_bin(centre: float, crop: int) -> int:
    return int(round(crop / centre))


# ---------------------------------------------------------------- readouts
def corpus_spectrum(seq: np.ndarray, crop: int) -> dict:
    """The band the SEQUENCE actually contains.  A control that can fail -- run it every time."""
    isW = ((seq == 0) | (seq == 3)).astype(np.float64)
    isW -= isW.mean()
    nwin = len(isW) // crop
    X = isW[: nwin * crop].reshape(nwin, crop)
    pw = (np.abs(np.fft.rfft(X, axis=1)) ** 2).mean(0)
    pw[0] = 0.0
    band = [b for b in BAND if b < len(pw)]
    q = np.array([pw[b] for b in band]) / sum(pw[b] for b in band)
    win = int(band[int(np.argmax(q))])
    return dict(win_bin=win, q_win=float(q.max()),
                peak_over_median=float(pw[win] / np.median(pw[1:])))


def table_spectrum(model, crop: int) -> dict:
    """P's spectrum -- over the TRAINED rows only (rows past the crop were never touched)."""
    P = model.bert.embeddings.position_embeddings.weight.detach().float().cpu().numpy()[:crop]
    Fp = np.fft.rfft(P - P.mean(0, keepdims=True), axis=0)
    pw = (np.abs(Fp) ** 2).sum(1)
    pw[0] = 0.0
    band = [b for b in BAND if b < len(pw)]
    tot = float(sum(pw[b] for b in band))
    q = {int(b): float(pw[b] / tot) for b in band}
    win = int(max(band, key=lambda b: pw[b]))
    nb = [b for b in (win - 1, win + 1) if 0 < b < len(pw)]
    p2n = float(pw[win] / np.mean([pw[b] for b in nb])) if nb else float("nan")
    sv = np.linalg.svd(np.stack([Fp[win].real, Fp[win].imag]), compute_uv=False)
    return dict(win_bin=win, win_period=crop / win, q_win=float(q[win]), q=q, p2n=p2n,
                circularity=float(sv[1] / sv[0]),
                band_power=tot, med_power=float(np.median(pw[1:])),
                band_over_median=tot / float(np.median(pw[1:])) / len(band))


# ---------------------------------------------------------------- one run
def one_run(centre, depth, crop, seed, dev, corpus_mb, max_rss_gb) -> dict:
    steps = int(round(TOTAL_TOKENS / (32 * crop)))       # IDENTICAL TOKENS across crops
    order_destroyed = centre is None
    gen_centre = 10.6667 if order_destroyed else centre
    pred = None if order_destroyed else predicted_bin(centre, crop)

    seq = make_corpus(int(corpus_mb * 1e6), gen_centre, depth, seed * 100 + 1,
                      shuffle_mod=order_destroyed)
    cspec = corpus_spectrum(seq, crop)

    torch.manual_seed(seed)                              # IDENTICAL INIT across corpora
    cfg = BertConfig(vocab_size=VOCAB, hidden_size=128, num_hidden_layers=4,
                     num_attention_heads=4, intermediate_size=512,
                     max_position_embeddings=crop, type_vocab_size=1,
                     pad_token_id=PAD_ID, attn_implementation="sdpa")
    model = BertForMaskedLM(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps, pct_start=0.1)
    g = np.random.default_rng(seed + 7)
    model.train()
    t0, losses = time.time(), []
    for s in range(steps):
        st = g.integers(0, len(seq) - crop, size=32)
        ids = torch.tensor(np.stack([seq[o:o + crop] for o in st]).astype(np.int64), device=dev)
        lab = ids.clone()
        m = torch.rand(ids.shape, device=dev) < 0.15
        lab[~m] = -100
        loss = model(input_ids=ids.masked_fill(m, MASK_ID), labels=lab).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        if s >= steps - 100:
            losses.append(loss.item())
        if (s + 1) % max(1, steps // 3) == 0:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824
            print(f"      {s+1}/{steps}  loss {loss.item():.4f}  {time.time()-t0:.0f}s  "
                  f"rss {rss:.1f}GB", flush=True)
            if rss > max_rss_gb:
                raise MemoryError(f"peak RSS {rss:.1f}GB exceeded --max-rss-gb {max_rss_gb}")

    tspec = table_spectrum(model, crop)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824
    del model, opt, sch
    gc.collect()
    if "mps" in str(dev):          # pick_device returns a STR, not a torch.device
        torch.mps.empty_cache()

    return dict(corpus=("control" if order_destroyed else f"{centre}"), centre=centre,
                depth=depth, crop=crop, seed=seed, steps=steps, predicted_bin=pred,
                final_loss=float(np.mean(losses)),
                corpus_win_bin=cspec["win_bin"], corpus_q_win=cspec["q_win"],
                corpus_peak_over_median=cspec["peak_over_median"],
                void=(None if pred is None else cspec["win_bin"] != pred),
                wall_s=round(time.time() - t0), peak_rss_gb=round(rss, 2), **tspec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A", choices=["A", "B", "C"])
    ap.add_argument("--depth", type=float, default=0.76)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--mb", type=float, default=4.0)
    ap.add_argument("--max-rss-gb", type=float, default=24.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.stage == "A":        # the quantization test: does the bin track the centre?
        cells = [(None, args.depth, 512), (10.45, args.depth, 512),
                 (10.6667, args.depth, 512), (10.85, args.depth, 512)]
    elif args.stage == "B":      # physical vs architectural: hold lambda, move the crop
        cells = [(10.6667, args.depth, 500), (10.6667, args.depth, 521)]
    else:                        # the dose ladder, on the winning centre
        cells = [(10.6667, d, 512) for d in (0.50, 0.25, 0.10, 0.05)]

    print(f"stage {args.stage} | {len(cells)} cells x {args.seeds} seed(s) "
          f"= {len(cells)*args.seeds} runs | tokens/run = {TOTAL_TOKENS/1e6:.1f}M")
    print(f"FROZEN NULLS:  flat {FLAT_QWIN:.3f}   FAITHFUL {FAITHFUL_QWIN:.2f}   "
          f"SpliceBERT 0.76\n")
    print(f"  {'corpus':>10} {'depth':>6} {'crop':>5} {'steps':>6}  predicted bin")
    for c, d, crop in cells:
        pb = "none (control)" if c is None else predicted_bin(c, crop)
        print(f"  {str(c):>10} {d:>6.0%} {crop:>5} {int(round(TOTAL_TOKENS/(32*crop))):>6}  {pb}")
    if args.dry_run:
        print("\n--dry-run: no GPU touched.")
        return

    dev = pick_device(None)
    print(f"\ndevice {dev} | MPS watermark "
          f"{os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO']}/"
          f"{os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']} "
          f"| RSS cap {args.max_rss_gb}GB | SEQUENTIAL, one run at a time\n")

    rec = json.loads(OUT.read_text()) if OUT.exists() else {}
    rec.setdefault("faithful_qwin", FAITHFUL_QWIN)
    rec.setdefault("flat_qwin", FLAT_QWIN)
    rec.setdefault("runs", {})
    for seed in range(args.seeds):
        for centre, depth, crop in cells:
            key = f"{'control' if centre is None else centre}|d{depth}|L{crop}|s{seed}"
            if key in rec["runs"]:
                print(f"[{key}] already done, skipping")
                continue
            print(f"[{key}]", flush=True)
            r = one_run(centre, depth, crop, seed, dev, args.mb, args.max_rss_gb)
            rec["runs"][key] = r
            OUT.write_text(json.dumps(rec, indent=1))     # MERGE by key, never clobber
            flag = ""
            if r["void"] is True:
                flag = "  <-- VOID: the band is not in the corpus"
            elif r["predicted_bin"] and r["win_bin"] == r["predicted_bin"]:
                flag = "  <-- HIT"
            print(f"      corpus bin {r['corpus_win_bin']} (Q {r['corpus_q_win']:.3f}) | "
                  f"TABLE bin {r['win_bin']} (Q_win {r['q_win']:.3f}, p2n {r['p2n']:.1f}) | "
                  f"loss {r['final_loss']:.4f} | {r['wall_s']}s{flag}", flush=True)

    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
