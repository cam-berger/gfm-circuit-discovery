"""Did the lambda arms build an anchored counter, or an anchor-free stop-scan?

aa_acc -> 1.0 is the TRAINING accuracy, and it does not distinguish the two. At lambda >= 1 the codon
distribution is real human usage and GENCODE's CDS excludes the terminal stop, so in-frame stops are
essentially absent: the true frame is the only stop-free frame, recoverable by a pure scan with no
anchor. The cue-conflict test separates them -- a scan follows CONTENT, a counter follows the ANCHOR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.uniform_cds import blended_codon_probs, make_batch
from gfmcd.models.gfm_nano import GFMNano, NanoConfig
from gfmcd.paths import project_paths

NON_START = np.array([1, 3, 1], dtype=np.int8)
CKPT = project_paths().checkpoints


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


@torch.no_grad()
def resid(model, tok, device, layer="L2.resid_post"):
    t = torch.from_numpy(tok).long()
    return torch.cat([model.run_with_cache(t[i : i + 64].to(device))[1][layer].cpu() for i in range(0, len(t), 64)])


@torch.no_grad()
def aa_acc(model, tok, aa, device):
    t, y = torch.from_numpy(tok).long(), torch.from_numpy(aa)
    c = n = 0.0
    for i in range(0, len(t), 64):
        lg = model(t[i : i + 64].to(device)).cpu()
        m = y[i : i + 64] != -100
        c += float((lg.argmax(-1)[m] == y[i : i + 64][m]).sum())
        n += float(m.sum())
    return c / n


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{'arm':<10}{'aa acc':>9}{'ATG-abl dAA':>14}{'anchor':>10}{'content':>10}   mechanism")
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        tag = "uniform" if lam == 0.0 else f"lam{lam:g}"
        cks = sorted(CKPT.glob(f"nano_{tag}_step*.pt"), key=lambda p: int(p.stem.split("step")[1]))
        if not cks:
            continue
        blob = torch.load(cks[-1], map_location=device, weights_only=False)
        model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
        model.load_state_dict(blob["state"])

        probs = None if lam == 0.0 else blended_codon_probs(lam)
        tok, aa, ph = make_batch(700, model.cfg.max_len, seed=4242, codon_probs=probs)
        start = np.argmax(ph == 0, axis=1)

        base = aa_acc(model, tok, aa, device)
        abl = tok.copy()
        for r, s in enumerate(start):
            abl[r, s : s + 3] = NON_START
        d_abl = base - aa_acc(model, abl, aa, device)

        rng = np.random.default_rng(0)
        sh = tok.copy()
        for r, s in enumerate(start):
            i = s + 3
            sh[r, i + 1 :] = tok[r, i:-1]
            sh[r, i] = rng.integers(0, 4)

        m = ph >= 0
        x = resid(model, tok, device)[torch.from_numpy(m)]
        y = torch.from_numpy(ph[m]).long()
        probe, mu, sd = fit_probe(x, y)

        pos = np.tile(np.arange(model.cfg.max_len), (len(tok), 1))
        dist = pos - start[:, None]
        far = m & (dist >= 64)
        pred = probe((resid(model, sh, device)[torch.from_numpy(far)] - mu) / sd).argmax(-1).numpy()
        anchor = float((pred == (dist % 3)[far]).mean())
        content = float((pred == ((dist - 1) % 3)[far]).mean())

        mech = "ANCHORED COUNTER" if anchor > 0.5 else "anchor-free (stop-scan / bag)"
        print(f"{tag:<10}{base:>9.4f}{d_abl:>14.4f}{anchor:>10.4f}{content:>10.4f}   {mech}")


if __name__ == "__main__":
    main()
