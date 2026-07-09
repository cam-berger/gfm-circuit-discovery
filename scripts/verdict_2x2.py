"""Apply the frozen verdict to a 2x2 cell checkpoint.

From docs/PREREGISTRATION.md, frozen before any cell was run:

    far-field cue-conflict anchor share (>=64 nt)   > 0.50
    AND start-codon ablation d(aa accuracy)         > 0.10 beyond the decoy-Met control

Both criteria are causal and were validated on the planted twins (D3, D24). Neither is the training
accuracy, which saturates and has already fooled us once (D27). Continuous values are reported
regardless of which side of the threshold they fall on.

    python scripts/verdict_2x2.py --checkpoint checkpoints/cellD_s0_step8000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.genome import IGNORE
from gfmcd.data.uniform_cds import ATG, N_AA
from gfmcd.models.gfm_nano import GFMNano, NanoConfig

NON_START = np.array([1, 3, 1], dtype=np.int8)  # CTC
ANCHOR_SHARE_MIN = 0.50
ABLATION_MIN = 0.10


def load_pool(cell: str, n: int, window: int, seed: int):
    import sys

    sys.path.insert(0, "scripts")
    from train_cell import pool

    from gfmcd.data.genome import Split

    return pool(cell, n, window, Split.VAL, seed)


@torch.no_grad()
def aa_accuracy(model, tok, aa, device):
    t, y = torch.from_numpy(tok).long(), torch.from_numpy(aa)
    c = n = 0.0
    for i in range(0, len(t), 64):
        lg = model(t[i : i + 64].to(device)).cpu()
        m = y[i : i + 64] != IGNORE
        c += float((lg.argmax(-1)[m] == y[i : i + 64][m]).sum())
        n += float(m.sum())
    return c / n


@torch.no_grad()
def resid(model, tok, device, layer="L2.resid_post"):
    t = torch.from_numpy(tok).long()
    return torch.cat([model.run_with_cache(t[i : i + 64].to(device))[1][layer].cpu() for i in range(0, len(t), 64)])


def fit_probe(x, y, steps=300):
    torch.manual_seed(0)
    p = torch.nn.Linear(x.shape[-1], 3)
    opt = torch.optim.AdamW(p.parameters(), lr=1e-2, weight_decay=1e-4)
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(p((x - mu) / sd), y).backward()
        opt.step()
    return p, mu, sd


def downstream_met(tok, start):
    """An in-frame ATG well past the start: an ordinary Met, not the anchor."""
    out = np.full(len(tok), -1, dtype=int)
    for r in range(len(tok)):
        for p in range(start[r] + 30, tok.shape[1] - 3, 3):
            if tuple(tok[r, p : p + 3]) == tuple(ATG):
                out[r] = p
                break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--seqs", type=int, default=1000)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cell = blob.get("cell") or {"uniform": "A"}.get(blob.get("pool"), blob.get("pool"))
    model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    model.load_state_dict(blob["state"])

    tok, aa, ph = load_pool(cell, args.seqs, model.cfg.max_len, 4242)
    start = np.argmax(ph == 0, axis=1)

    intact = aa_accuracy(model, tok, aa, device)

    abl = tok.copy()
    for r, s in enumerate(start):
        abl[r, s : s + 3] = NON_START
    ablated = aa_accuracy(model, abl, aa, device)

    met = downstream_met(tok, start)
    have = met >= 0
    ctl = tok[have].copy()
    for i, r in enumerate(np.where(have)[0]):
        ctl[i, met[r] : met[r] + 3] = NON_START
    control = aa_accuracy(model, ctl, aa[have], device)
    intact_have = aa_accuracy(model, tok[have], aa[have], device)

    d_start = intact - ablated
    d_ctl = intact_have - control
    ablation_effect = d_start - d_ctl

    rng = np.random.default_rng(0)
    shifted = tok.copy()
    for r, s in enumerate(start):
        i = s + 3
        shifted[r, i + 1 :] = tok[r, i:-1]
        shifted[r, i] = rng.integers(0, 4)

    m = ph >= 0
    x_int = resid(model, tok, device)[torch.from_numpy(m)]
    y = torch.from_numpy(ph[m]).long()
    probe, mu, sd = fit_probe(x_int, y)

    pos = np.tile(np.arange(model.cfg.max_len), (len(tok), 1))
    dist = pos - start[:, None]
    far = m & (dist >= 64)
    x_sh = resid(model, shifted, device)[torch.from_numpy(far)]
    pred = probe((x_sh - mu) / sd).argmax(-1).numpy()
    anchor = float((pred == (dist % 3)[far]).mean())
    content = float((pred == ((dist - 1) % 3)[far]).mean())

    print(f"{args.checkpoint.name}   cell {cell}   step {blob['step']}   seed {blob.get('seed', 0)}\n")
    print(f"{'aa accuracy, intact':<44}{intact:>10.4f}")
    print(f"{'aa accuracy, start codon ablated':<44}{ablated:>10.4f}   d = {d_start:+.4f}")
    print(f"{'aa accuracy, decoy Met ablated (control)':<44}{control:>10.4f}   d = {d_ctl:+.4f}   n={int(have.sum())}")
    print(f"{'ablation effect (start - control)':<44}{ablation_effect:>10.4f}   threshold > {ABLATION_MIN}")
    print()
    print(f"{'cue conflict, far field (>=64 nt)':<44}")
    print(f"{'  follows ANCHOR':<44}{anchor:>10.4f}   threshold > {ANCHOR_SHARE_MIN}")
    print(f"{'  follows CONTENT':<44}{content:>10.4f}")

    formed = anchor > ANCHOR_SHARE_MIN and ablation_effect > ABLATION_MIN
    print(f"\nVERDICT: {'CIRCUIT FORMED' if formed else 'NO CIRCUIT'}")


if __name__ == "__main__":
    main()
