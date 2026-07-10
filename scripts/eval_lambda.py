"""Dose-response: how much local shortcut does it take to stop the circuit from forming?

D26 established that a circuit appears when the objective makes it necessary and not when it is merely
useful. That is a claim about a threshold. The lambda dial interpolates codon usage from uniform
(no local phase cue; the anchored counter is the only solution) to real human usage and beyond, while
the task -- predict the amino acid of the containing codon -- never changes.

The y-axis is the circuit, not the accuracy. Two causal readouts, both validated on the planted twins:

  ATG-ablation loss increase    destroy the start codon; a counter collapses, a bag shrugs
  cue-conflict anchor share     insert 1 nt past the ATG; a counter propagates the frameshift at
                                every distance, a bag follows local content beyond ~30 nt

    python scripts/eval_lambda.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.genome import IGNORE
from gfmcd.data.uniform_cds import (
    N_AA,
    blended_codon_probs,
    make_batch,
    nucleotide_phase_information,
)
from gfmcd.models.gfm_nano import GFMNano, NanoConfig

CKPT = Path("checkpoints")
NON_START = np.array([1, 3, 1], dtype=np.int8)
REAL_NATS = 0.0241


def load(path: Path, device: str):
    blob = torch.load(path, map_location=device, weights_only=False)
    model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    model.load_state_dict(blob["state"])
    return model, blob


@torch.no_grad()
def loss_and_acc(model, tok, aa, device):
    t, y = torch.from_numpy(tok).long(), torch.from_numpy(aa)
    tot = corr = n = 0.0
    for i in range(0, len(t), 64):
        lg = model(t[i : i + 64].to(device)).cpu()
        yy = y[i : i + 64]
        tot += float(F.cross_entropy(lg.reshape(-1, N_AA), yy.reshape(-1), ignore_index=IGNORE, reduction="sum"))
        m = yy != IGNORE
        corr += float((lg.argmax(-1)[m] == yy[m]).sum())
        n += float(m.sum())
    return tot / n, corr / n


def fit_probe(x, y, steps: int = 300):
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
def resid(model, tok, device, layer="L2.resid_post"):
    t = torch.from_numpy(tok).long()
    return torch.cat([model.run_with_cache(t[i : i + 64].to(device))[1][layer].cpu() for i in range(0, len(t), 64)])


def transition_step(tag: str) -> int | None:
    p = CKPT / f"nano_{tag}_history.json"
    if not p.exists():
        return None
    for row in json.loads(p.read_text()):
        if row["aa_acc"] > 0.60:
            return row["step"]
    return None


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    rows = []

    for lam in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        tag = "uniform" if lam == 0.0 else f"lam{lam:g}"
        ckpts = sorted(CKPT.glob(f"nano_{tag}_step*.pt"), key=lambda p: int(p.stem.split("step")[1]))
        if not ckpts:
            continue
        model, blob = load(ckpts[-1], device)

        probs = None if lam == 0.0 else blended_codon_probs(lam)
        tok, aa, ph = make_batch(600, model.cfg.max_len, seed=4242, codon_probs=probs)
        start = np.argmax(ph == 0, axis=1)
        shortcut = nucleotide_phase_information(tok, ph)

        base_loss, base_acc = loss_and_acc(model, tok, aa, device)

        ablated = tok.copy()
        for r, s in enumerate(start):
            ablated[r, s : s + 3] = NON_START
        abl_loss, abl_acc = loss_and_acc(model, ablated, aa, device)

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
        live = m & (dist >= 64)
        x_sh = resid(model, shifted, device)[torch.from_numpy(live)]
        pred = probe((x_sh - mu) / sd).argmax(-1).numpy()
        anchor = float((pred == (dist % 3)[live]).mean())
        content = float((pred == ((dist - 1) % 3)[live]).mean())

        rows.append(dict(lam=lam, nats=shortcut, pct=shortcut / REAL_NATS, step=blob["step"],
                         acc=base_acc, dloss=abl_loss - base_loss, abl_acc=abl_acc,
                         anchor=anchor, content=content, trans=transition_step(tag)))

    print(f"{'lam':>5}{'I(nt;phase)':>13}{'% real':>8}{'aa acc':>9}{'ATG-abl Δloss':>15}{'abl acc':>9}"
          f"{'anchor':>9}{'content':>9}{'grok@':>8}")
    for r in rows:
        t = str(r["trans"]) if r["trans"] else "never"
        print(f"{r['lam']:>5.2f}{r['nats']:>13.5f}{r['pct']:>7.0%}{r['acc']:>9.4f}{r['dloss']:>15.4f}"
              f"{r['abl_acc']:>9.4f}{r['anchor']:>9.4f}{r['content']:>9.4f}{t:>8}")

    print("\nanchor/content are far-field (>=64 nt) agreement under cue conflict.")
    print("A counter follows the anchor; a bag follows content. Real hg38 GFM-nano: 0.06 / 0.84")
    Path("results").mkdir(exist_ok=True)
    Path("results/reading-frame/lambda_sweep.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
