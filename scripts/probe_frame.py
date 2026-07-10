"""Rung 1's go/no-go: is codon phase linearly decodable from the residual stream?

If phase is not linearly decodable anywhere in a model we trained on frame-bearing data, there is no
circuit to trace and we learn that for a few hours of compute rather than a few weeks. This probe --
not the behavioural stop-codon proxy, which measured p = 0.356 and proved nothing -- is the
instrument.

Four controls, because "a probe found it" is nearly meaningless on its own:

  untrained     the same architecture at random init. Absolute position embeddings alone could make
                phase decodable without anything being learned. This is the control that matters.
  nucleotide    logistic regression on the one-hot base at that position. Codon usage is biased, so
                a single nucleotide carries real information about phase. Any layer that fails to
                beat this has learned nothing about frame.
  position      phase predicted from `index % 3` alone. Should sit at chance (0.3446 measured), and
                if it does not, `data/genome.py` is broken.
  shuffled      labels permuted across positions. The floor.

    python scripts/probe_frame.py --checkpoint checkpoints/nano_ar_step8000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.genome import SPLIT_CHROMS, Split, parse_gencode_cds, sample_windows
from gfmcd.models.gfm_nano import GFMNano, NanoConfig
from gfmcd.paths import project_paths

DATA = project_paths().references
CHANCE = 1 / 3


def fit_probe(x: torch.Tensor, y: torch.Tensor, xv: torch.Tensor, yv: torch.Tensor, steps: int = 400) -> float:
    """Multinomial logistic regression, fit on `x`, scored on held-out `xv`."""
    torch.manual_seed(0)
    probe = torch.nn.Linear(x.shape[-1], 3).to(x.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)

    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)
    xn, xvn = (x - mu) / sd, (xv - mu) / sd

    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(probe(xn), y).backward()
        opt.step()

    with torch.no_grad():
        return float((probe(xvn).argmax(-1) == yv).float().mean())


@torch.no_grad()
def collect(model: GFMNano, tokens: torch.Tensor, labels: torch.Tensor, device: str) -> tuple[dict, torch.Tensor]:
    """Residual stream at every layer, restricted to labelled (in-CDS) positions."""
    acts: dict[str, list[torch.Tensor]] = {}
    ys: list[torch.Tensor] = []

    for i in range(0, len(tokens), 64):
        tok = tokens[i : i + 64].to(device)
        lab = labels[i : i + 64].to(device)
        _, cache = model.run_with_cache(tok)
        mask = lab >= 0
        ys.append(lab[mask].cpu())

        for name in ["resid.embed"] + [f"L{l}.resid_post" for l in range(model.cfg.n_layers)]:
            acts.setdefault(name, []).append(cache[name][mask].cpu())

    return {k: torch.cat(v) for k, v in acts.items()}, torch.cat(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=1500)
    ap.add_argument("--window", type=int, default=256)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    from pyfaidx import Fasta

    fasta = Fasta(str(DATA / "hg38.fa"), as_raw=True, sequence_always_upper=True)
    cds = parse_gencode_cds(DATA / "gencode.v50.basic.annotation.gtf.gz", chroms=SPLIT_CHROMS[Split.VAL], min_len=150)
    tok_np, lab_np = sample_windows(cds, fasta, n=args.windows, window=args.window, seed=99)
    tokens, labels = torch.from_numpy(tok_np).long(), torch.from_numpy(lab_np).long()

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = NanoConfig(**blob["cfg"])
    trained = GFMNano(cfg).to(device).eval()
    trained.load_state_dict(blob["state"])

    torch.manual_seed(1234)
    untrained = GFMNano(cfg).to(device).eval()

    print(f"{args.checkpoint.name}  causal={cfg.causal}  step={blob['step']}")
    print(f"{tokens.shape[0]} val windows; probing on in-CDS positions only\n")

    acts_t, y = collect(trained, tokens, labels, device)
    acts_u, _ = collect(untrained, tokens, labels, device)

    n = len(y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n) :]

    # Baselines.
    onehot = F.one_hot(tokens[labels >= 0].reshape(-1), num_classes=6).float()
    nt_acc = fit_probe(onehot[tr], y[tr], onehot[te], y[te])

    idx = torch.arange(args.window).repeat(len(tokens), 1)[labels >= 0]
    pos_feat = F.one_hot(idx % 3, num_classes=3).float()
    pos_acc = fit_probe(pos_feat[tr], y[tr], pos_feat[te], y[te])

    last = f"L{cfg.n_layers - 1}.resid_post"
    shuf = y[torch.randperm(n, generator=torch.Generator().manual_seed(7))]
    shuf_acc = fit_probe(acts_t[last][tr], shuf[tr], acts_t[last][te], shuf[te])

    print(f"{'baseline':<22} {'accuracy':>9}")
    print(f"{'chance':<22} {CHANCE:>9.4f}")
    print(f"{'label-shuffled':<22} {shuf_acc:>9.4f}")
    print(f"{'position (index % 3)':<22} {pos_acc:>9.4f}")
    print(f"{'nucleotide identity':<22} {nt_acc:>9.4f}")

    print(f"\n{'layer':<22} {'trained':>9} {'untrained':>11} {'delta':>8}")
    for name in acts_t:
        a = fit_probe(acts_t[name][tr], y[tr], acts_t[name][te], y[te])
        b = fit_probe(acts_u[name][tr], y[tr], acts_u[name][te], y[te])
        print(f"{name:<22} {a:>9.4f} {b:>11.4f} {a-b:>+8.4f}")

    best = max(fit_probe(acts_t[k][tr], y[tr], acts_t[k][te], y[te]) for k in acts_t)
    print(f"\nbest trained layer {best:.4f} vs nucleotide baseline {nt_acc:.4f}")
    if best < nt_acc + 0.05:
        print("VERDICT: no frame representation beyond single-nucleotide bias. Rung 1 has nothing to trace.")
    else:
        print("VERDICT: phase is linearly decodable above every control. Proceed to tracing.")


if __name__ == "__main__":
    main()
