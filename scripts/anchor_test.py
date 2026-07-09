"""Does the model find the start codon and count from it, or is it a bag of motifs?

This is the same assay that separated the planted twins in Rung 0. There, deleting the marker
dropped the compositional twin from 1.000 to 0.383 and left the bag at 0.583 untouched. Here the
marker is a real ATG in real human DNA and the model was trained, not planted.

    anchored windows   contain the translation start codon -- the anchor exists
    internal windows   drawn inside coding exons -- no ATG, so phase can only come from statistics

Predictions, pre-registered before running:

    if the model uses the anchor    accuracy(anchored) > accuracy(internal)
                                    ablating the ATG collapses accuracy(anchored) toward internal
                                    ablating a matched downstream codon does almost nothing
    if it is a bag of motifs        accuracy(anchored) == accuracy(internal)
                                    ablating the ATG changes nothing

The probe is fit ONCE on intact anchored windows and then applied unchanged to ablated ones, so a
drop reflects the model's representation moving, not the probe re-fitting to a new distribution.

    python scripts/anchor_test.py --checkpoint checkpoints/nano_ar_mixed_step8000.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.genome import (
    SPLIT_CHROMS,
    Split,
    atg_index,
    parse_gencode_cds,
    sample_anchored_windows,
    sample_windows,
    split_by_anchor,
)
from gfmcd.models.gfm_nano import GFMNano, NanoConfig

DATA = Path.home() / "Research/data/genomes"
NON_START = [1, 3, 1]  # CTC -- not a start codon, preserves nothing but length
CONTROL_OFFSET = 30  # a matched in-frame codon downstream of the ATG


def fit_probe(x: torch.Tensor, y: torch.Tensor, steps: int = 400) -> tuple[torch.nn.Linear, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    probe = torch.nn.Linear(x.shape[-1], 3).to(x.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp(min=1e-6)

    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(probe((x - mu) / sd), y).backward()
        opt.step()
    return probe, mu, sd


@torch.no_grad()
def score(probe, mu, sd, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((probe((x - mu) / sd).argmax(-1) == y).float().mean())


@torch.no_grad()
def residuals(model: GFMNano, tokens: torch.Tensor, labels: torch.Tensor, layer: str, device: str):
    xs, ys = [], []
    for i in range(0, len(tokens), 64):
        tok, lab = tokens[i : i + 64].to(device), labels[i : i + 64].to(device)
        _, cache = model.run_with_cache(tok)
        mask = lab >= 0
        xs.append(cache[layer][mask].cpu())
        ys.append(lab[mask].cpu())
    return torch.cat(xs), torch.cat(ys)


def ablate(tokens: np.ndarray, at: np.ndarray) -> np.ndarray:
    out = tokens.copy()
    for row, i in enumerate(at):
        out[row, i : i + 3] = NON_START
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=1200)
    ap.add_argument("--window", type=int, default=256)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from pyfaidx import Fasta

    fasta = Fasta(str(DATA / "hg38.fa"), as_raw=True, sequence_always_upper=True)
    cds = parse_gencode_cds(DATA / "gencode.v50.basic.annotation.gtf.gz", chroms=SPLIT_CHROMS[Split.VAL], min_len=150)
    starts, internal = split_by_anchor(cds, fasta)

    a_tok, a_lab = sample_anchored_windows(starts, fasta, n=args.windows, window=args.window, seed=99)
    i_tok, i_lab = sample_windows(internal, fasta, n=args.windows, window=args.window, seed=98)

    atg = atg_index(a_tok, a_lab)
    keep = atg + 3 + CONTROL_OFFSET + 3 < args.window
    a_tok, a_lab, atg = a_tok[keep], a_lab[keep], atg[keep]

    abl_atg = ablate(a_tok, atg)
    abl_ctl = ablate(a_tok, atg + 3 + CONTROL_OFFSET)

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = NanoConfig(**blob["cfg"])
    model = GFMNano(cfg).to(device).eval()
    model.load_state_dict(blob["state"])
    layer = "L0.resid_post"

    to_t = lambda a: torch.from_numpy(a).long()
    x_anchored, y_anchored = residuals(model, to_t(a_tok), to_t(a_lab), layer, device)
    x_internal, y_internal = residuals(model, to_t(i_tok), to_t(i_lab), layer, device)
    x_no_atg, y_no_atg = residuals(model, to_t(abl_atg), to_t(a_lab), layer, device)
    x_ctl, y_ctl = residuals(model, to_t(abl_ctl), to_t(a_lab), layer, device)

    n = len(y_anchored)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n) :]

    probe, mu, sd = fit_probe(x_anchored[tr], y_anchored[tr])

    intact = score(probe, mu, sd, x_anchored[te], y_anchored[te])
    no_atg = score(probe, mu, sd, x_no_atg[te], y_no_atg[te])
    control = score(probe, mu, sd, x_ctl[te], y_ctl[te])

    probe_i, mu_i, sd_i = fit_probe(x_internal[: int(0.7 * len(y_internal))], y_internal[: int(0.7 * len(y_internal))])
    internal_acc = score(probe_i, mu_i, sd_i, x_internal[int(0.7 * len(y_internal)) :], y_internal[int(0.7 * len(y_internal)) :])

    print(f"{args.checkpoint.name}   pool={blob.get('pool','?')}   layer={layer}")
    print(f"anchored windows: {len(a_tok)}   internal windows: {len(i_tok)}\n")
    print(f"{'condition':<40} {'phase accuracy':>15}")
    print(f"{'anchored, intact':<40} {intact:>15.4f}")
    print(f"{'anchored, ATG ablated':<40} {no_atg:>15.4f}")
    print(f"{'anchored, matched downstream codon ablated':<40} {control:>15.4f}")
    print(f"{'internal (no anchor exists)':<40} {internal_acc:>15.4f}")
    print(f"{'chance':<40} {1/3:>15.4f}")

    print(f"\nATG ablation cost:      {intact - no_atg:+.4f}")
    print(f"control ablation cost:  {intact - control:+.4f}")
    print(f"anchored - internal:    {intact - internal_acc:+.4f}")

    uses_anchor = (intact - no_atg) > 0.05 and (intact - no_atg) > 3 * abs(intact - control)
    print("\nVERDICT:", "anchored counter -- the model finds the ATG and counts" if uses_anchor
          else "bag of motifs -- the start codon is not load-bearing")


if __name__ == "__main__":
    main()
