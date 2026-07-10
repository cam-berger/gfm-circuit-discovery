"""Does the model find the start codon and count from it, or is it a bag of motifs?

This is the same assay that separated the planted twins in Rung 0. There, deleting the marker
dropped the compositional twin from 1.000 to 0.383 and left the bag at 0.583 untouched. Here the
marker is a real ATG in real human DNA and the model was trained, not planted.

    anchored windows   contain the translation start codon -- the anchor exists
    internal windows   drawn inside coding exons -- no ATG, so phase can only come from statistics

**The discriminating signal is distance, not magnitude.** Two confounds make the raw ablation cost
uninterpretable: the ATG's own three positions are scored, so deleting them merely removes the input
the probe was reading; and a bag of motifs pools its neighbours, so it is *legitimately* damaged near
any ablation. Both effects are local.

So we exclude the ablated codon's own positions and report the accuracy drop **as a function of
distance downstream of the ablation**:

    anchored counter   every downstream position is counted from the ATG, so the drop persists at
                       long range -- it does not decay
    bag of motifs      the drop decays to zero once you are past the pooling width (~±15-30 nt)

**The control is a decoy ATG, not an arbitrary codon.** Ablating `ATG -> CTC` at the true start and
comparing against ablating some random codon confounds two things: the start codon carries far more
phase information than an average codon, so a bag of motifs will weight it heavily and be hurt by its
removal. Anchored windows contain ~2.5 other ATG trinucleotides, ~0.8 of them in frame. Ablating an
in-frame decoy ATG is the same perturbation, on an equally informative motif, at the wrong location.

    bag of motifs      true ATG and decoy ATG hurt similarly; both decay with distance
    anchored counter   only the true start hurts, and it hurts at long range

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
from gfmcd.paths import project_paths

DATA = project_paths().references
NON_START = [1, 3, 1]  # CTC -- not a start codon
ATG = (0, 3, 2)
MIN_DECOY_GAP = 20  # a decoy must sit well clear of the true start codon
MIN_DOWNSTREAM = 80  # ...and leave enough scored positions behind it


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
def residuals(model: GFMNano, tokens: torch.Tensor, labels: torch.Tensor, layer: str, device: str, anchor=None):
    """Residuals at labelled positions, plus each position's signed distance from `anchor`."""
    xs, ys, ds = [], [], []
    n = tokens.shape[-1]
    for i in range(0, len(tokens), 64):
        tok, lab = tokens[i : i + 64].to(device), labels[i : i + 64].to(device)
        _, cache = model.run_with_cache(tok)
        mask = lab >= 0
        xs.append(cache[layer][mask].cpu())
        ys.append(lab[mask].cpu())
        if anchor is not None:
            pos = torch.arange(n, device=device).expand_as(lab)
            a = torch.as_tensor(anchor[i : i + 64], device=device).unsqueeze(-1)
            ds.append((pos - a)[mask].cpu())
    dist = torch.cat(ds) if anchor is not None else None
    return torch.cat(xs), torch.cat(ys), dist


def ablate(tokens: np.ndarray, at: np.ndarray) -> np.ndarray:
    out = tokens.copy()
    for row, i in enumerate(at):
        if i >= 0:
            out[row, i : i + 3] = NON_START
    return out


def find_decoy_atg(tokens: np.ndarray, labels: np.ndarray, atg: np.ndarray) -> np.ndarray:
    """An in-frame ATG that is not the start codon, with room to score behind it. -1 if none."""
    n = tokens.shape[1]
    out = np.full(len(tokens), -1, dtype=int)
    for r in range(len(tokens)):
        last = int(np.max(np.where(labels[r] >= 0)[0]))
        for p in range(atg[r] + MIN_DECOY_GAP, n - 2):
            if p + MIN_DOWNSTREAM > last:
                break
            if labels[r, p] == 0 and tuple(tokens[r, p : p + 3]) == ATG:
                out[r] = p
                break
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
    decoy = find_decoy_atg(a_tok, a_lab, atg)
    keep = decoy >= 0
    a_tok, a_lab, atg, decoy = a_tok[keep], a_lab[keep], atg[keep], decoy[keep]
    print(f"{keep.sum()}/{len(keep)} anchored windows carry an in-frame decoy ATG")

    abl_atg = ablate(a_tok, atg)
    abl_dec = ablate(a_tok, decoy)

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = NanoConfig(**blob["cfg"])
    model = GFMNano(cfg).to(device).eval()
    model.load_state_dict(blob["state"])
    layer = "L0.resid_post"

    to_t = lambda a: torch.from_numpy(a).long()
    x_intact, y, dist = residuals(model, to_t(a_tok), to_t(a_lab), layer, device, anchor=atg)
    x_no_atg, _, _ = residuals(model, to_t(abl_atg), to_t(a_lab), layer, device, anchor=atg)
    x_no_dec, _, dist_dec = residuals(model, to_t(abl_dec), to_t(a_lab), layer, device, anchor=decoy)
    x_int, y_int, _ = residuals(model, to_t(i_tok), to_t(i_lab), layer, device)

    n = len(y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: int(0.7 * n)], perm[int(0.7 * n) :]
    probe, mu, sd = fit_probe(x_intact[tr], y[tr])

    ni = len(y_int)
    pi, mi, si = fit_probe(x_int[: int(0.7 * ni)], y_int[: int(0.7 * ni)])
    internal_acc = score(pi, mi, si, x_int[int(0.7 * ni) :], y_int[int(0.7 * ni) :])

    print(f"{args.checkpoint.name}   pool={blob.get('pool','?')}   layer={layer}")
    print(f"anchored windows: {len(a_tok)}   internal windows: {len(i_tok)}")
    print(f"internal-window accuracy (no anchor exists): {internal_acc:.4f}   chance {1/3:.4f}\n")

    # Each ablation is scored at distances measured from ITS OWN site, excluding the site itself.
    d_atg, d_dec = dist[te], dist_dec[te]
    bins = [(3, 15), (16, 31), (32, 63), (64, 255)]

    def drop(distance, x_ablated, lo, hi):
        sel = (distance >= lo) & (distance <= hi)
        if int(sel.sum()) < 200:
            return None, None, 0
        ref = score(probe, mu, sd, x_intact[te][sel], y[te][sel])
        abl = score(probe, mu, sd, x_ablated[te][sel], y[te][sel])
        return ref, ref - abl, int(sel.sum())

    print(f"{'distance from ablation':<24} {'true ATG drop':>14} {'n':>7}   {'decoy ATG drop':>15} {'n':>7}")
    for lo, hi in bins:
        _, da, na = drop(d_atg, x_no_atg, lo, hi)
        _, dd, nd = drop(d_dec, x_no_dec, lo, hi)
        if da is None and dd is None:
            continue
        a_s = f"{da:+.4f}" if da is not None else "     --"
        d_s = f"{dd:+.4f}" if dd is not None else "     --"
        print(f"{f'{lo}-{hi} nt':<24} {a_s:>14} {na:>7}   {d_s:>15} {nd:>7}")

    _, far_atg, _ = drop(d_atg, x_no_atg, 64, 255)
    _, far_dec, _ = drop(d_dec, x_no_dec, 64, 255)
    far_atg = far_atg or 0.0
    far_dec = far_dec or 0.0

    print(f"\nfar-field (>=64 nt)   true ATG {far_atg:+.4f}   decoy ATG {far_dec:+.4f}")

    anchored = far_atg > 0.05 and far_atg > 3 * max(abs(far_dec), 1e-6)
    print("\nVERDICT:", "ANCHORED COUNTER -- only the true start codon is load-bearing, at long range" if anchored
          else "BAG OF MOTIFS -- the start codon behaves like any other informative local motif")


if __name__ == "__main__":
    main()
