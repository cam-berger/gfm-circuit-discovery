"""Prove the forcing-task model built an anchored counter, using the battery that found none in hg38.

Four tests, each of which the real-DNA models failed:

  1. phase probe      is codon phase linearly decodable, layer by layer, above an untrained control?
  2. ATG ablation     destroy the start codon. With no local shortcut, accuracy must collapse to the
                      alignment-free ceiling (0.3768). In real DNA it fell by 0.01 at long range.
  3. decoy ablation   destroy a downstream in-frame ATG (an ordinary Met codon). It is not the anchor,
                      so nothing should happen.
  4. cue conflict     insert one nucleotide after the ATG. Anchor and content now disagree. A counter
                      follows the anchor at every distance; a bag follows content beyond ~30 nt.

    python scripts/verify_uniform.py --checkpoint checkpoints/nano_uniform_step12000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.genome import IGNORE
from gfmcd.data.uniform_cds import ATG, N_AA, alignment_free_ceiling, make_batch
from gfmcd.models.gfm_nano import GFMNano, NanoConfig

NON_START = np.array([1, 3, 1], dtype=np.int8)  # CTC


def fit_probe(x, y, steps: int = 400):
    torch.manual_seed(0)
    probe = torch.nn.Linear(x.shape[-1], 3)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(probe((x - mu) / sd), y).backward()
        opt.step()
    return probe, mu, sd


@torch.no_grad()
def predict(probe, mu, sd, x):
    return probe((x - mu) / sd).argmax(-1)


@torch.no_grad()
def run(model, tokens: np.ndarray, device: str, layer: str):
    logits, resid = [], []
    t = torch.from_numpy(tokens).long()
    for i in range(0, len(t), 64):
        lg, cache = model.run_with_cache(t[i : i + 64].to(device))
        logits.append(lg.cpu())
        resid.append(cache[layer].cpu())
    return torch.cat(logits), torch.cat(resid)


def aa_accuracy(logits: torch.Tensor, aa: np.ndarray) -> float:
    y = torch.from_numpy(aa)
    m = y != IGNORE
    return float((logits.argmax(-1)[m] == y[m]).float().mean())


def find_downstream_met(tokens: np.ndarray, start: np.ndarray) -> np.ndarray:
    """An in-frame ATG well downstream of the start codon: an ordinary Met, not the anchor."""
    out = np.full(len(tokens), -1, dtype=int)
    for r in range(len(tokens)):
        for p in range(start[r] + 30, tokens.shape[1] - 3, 3):
            if tuple(tokens[r, p : p + 3]) == tuple(ATG):
                out[r] = p
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--seqs", type=int, default=1500)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--layer", default="L2.resid_post")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    ceiling = alignment_free_ceiling()

    tok, aa, phase = make_batch(args.seqs, args.window, seed=777)
    start = np.argmax(phase == 0, axis=1)

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = NanoConfig(**blob["cfg"])
    model = GFMNano(cfg).to(device).eval()
    model.load_state_dict(blob["state"])

    torch.manual_seed(1234)
    untrained = GFMNano(cfg).to(device).eval()

    print(f"{args.checkpoint.name}  step={blob['step']}")
    print(f"alignment-free ceiling {ceiling:.4f}   frame-aware ceiling 1.0000\n")

    # ---- 1. phase probe, layer by layer -----------------------------------
    m = phase >= 0
    y = torch.from_numpy(phase[m]).long()
    n = len(y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n) :]

    print(f"{'layer':<20} {'trained':>9} {'untrained':>11}")
    for name in ["resid.embed"] + [f"L{i}.resid_post" for i in range(cfg.n_layers)]:
        _, rt = run(model, tok, device, name)
        _, ru = run(untrained, tok, device, name)
        xt, xu = rt[torch.from_numpy(m)], ru[torch.from_numpy(m)]
        p, mu, sd = fit_probe(xt[tr], y[tr])
        a = float((predict(p, mu, sd, xt[te]) == y[te]).float().mean())
        p2, mu2, sd2 = fit_probe(xu[tr], y[tr])
        b = float((predict(p2, mu2, sd2, xu[te]) == y[te]).float().mean())
        print(f"{name:<20} {a:>9.4f} {b:>11.4f}")

    # ---- 2 & 3. ablations --------------------------------------------------
    abl_start = tok.copy()
    for r, s in enumerate(start):
        abl_start[r, s : s + 3] = NON_START

    met = find_downstream_met(tok, start)
    have_met = met >= 0
    abl_met = tok.copy()
    for r in np.where(have_met)[0]:
        abl_met[r, met[r] : met[r] + 3] = NON_START

    lg_intact, _ = run(model, tok, device, args.layer)
    lg_no_start, _ = run(model, abl_start, device, args.layer)
    lg_no_met, _ = run(model, abl_met[have_met], device, args.layer)

    print(f"\n{'amino-acid accuracy':<34} {'value':>8}")
    print(f"{'intact':<34} {aa_accuracy(lg_intact, aa):>8.4f}")
    print(f"{'start codon ablated':<34} {aa_accuracy(lg_no_start, aa):>8.4f}")
    print(f"{'downstream Met ablated (control)':<34} {aa_accuracy(lg_no_met, aa[have_met]):>8.4f}   n={int(have_met.sum())}")
    print(f"{'alignment-free ceiling':<34} {ceiling:>8.4f}")

    # ---- 4. cue conflict ---------------------------------------------------
    rng = np.random.default_rng(0)
    shifted = tok.copy()
    for r, s in enumerate(start):
        i = s + 3
        shifted[r, i + 1 :] = tok[r, i:-1]
        shifted[r, i] = rng.integers(0, 4)

    _, r_shift = run(model, shifted, device, args.layer)
    _, r_intact = run(model, tok, device, args.layer)

    x_fit = r_intact[torch.from_numpy(m)]
    probe, mu, sd = fit_probe(x_fit[tr], y[tr])

    pos = np.tile(np.arange(args.window), (len(tok), 1))
    dist = pos - start[:, None]
    live = m & (dist > 3)
    anchor_phase = (dist % 3)[live]
    content_phase = ((dist - 1) % 3)[live]
    pred = predict(probe, mu, sd, r_shift[torch.from_numpy(live)]).numpy()
    d = dist[live]

    print(f"\nafter inserting 1 nt past the ATG, anchor and content disagree by one codon position")
    print(f"{'distance from ATG':<22} {'follows ANCHOR':>16} {'follows CONTENT':>17} {'n':>8}")
    for lo, hi in [(4, 15), (16, 31), (32, 63), (64, 127), (128, 255)]:
        sel = (d >= lo) & (d <= hi)
        if sel.sum() < 200:
            continue
        a = float((pred[sel] == anchor_phase[sel]).mean())
        c = float((pred[sel] == content_phase[sel]).mean())
        print(f"{f'{lo}-{hi} nt':<22} {a:>16.4f} {c:>17.4f} {int(sel.sum()):>8}")

    far = d >= 64
    a_far = float((pred[far] == anchor_phase[far]).mean())
    c_far = float((pred[far] == content_phase[far]).mean())
    print(f"\nfar field (>=64 nt):  anchor {a_far:.4f}   content {c_far:.4f}")
    print("\nVERDICT:", "ANCHORED COUNTER" if a_far > c_far + 0.1 else "BAG OF MOTIFS")


if __name__ == "__main__":
    main()
