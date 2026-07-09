"""Which cue does the model follow when the anchor and the local statistics disagree?

`anchor_test.py` ablates the start codon and asks whether accuracy falls. It cannot answer the
question. Deleting the ATG removes one source of phase information and leaves the other -- downstream
codon statistics, worth 0.79 on their own -- entirely intact. A model that genuinely counts from the
ATG can therefore lose nothing when the ATG is destroyed, because it simply falls back on local
evidence. Redundancy makes the ablation assay blind. (Cam's point, and it is correct.)

So do not remove information. Make the two cues **contradict** each other.

Insert one nucleotide immediately after the ATG. Everything downstream shifts right by one:

    original    ...ATG | GCC | ATT | GTA ...        content phase at q = (q - atg) % 3
    shifted     ...ATG X GCC | ATT | GTA ...        content now sits one position later

For a downstream position q in the shifted sequence:

    anchor says    phase = (q - atg) % 3          -- counting codons from the start codon
    content says   phase = (q - 1 - atg) % 3      -- the codon statistics moved with the sequence

They differ by exactly one, everywhere downstream. The probe (fit once on intact anchored windows)
must read out one or the other:

    follows anchor    -> the model counts from the start codon; a frameshift propagates, as in biology
    follows content   -> the model is a bag of motifs reading local codon composition

    python scripts/frameshift_conflict.py --checkpoint checkpoints/nano_ar_mixed_step2800.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.genome import (
    IGNORE,
    SPLIT_CHROMS,
    Split,
    atg_index,
    parse_gencode_cds,
    sample_anchored_windows,
    split_by_anchor,
)
from gfmcd.models.gfm_nano import GFMNano, NanoConfig

DATA = Path.home() / "Research/data/genomes"


def fit_probe(x: torch.Tensor, y: torch.Tensor, steps: int = 400):
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
def predict(probe, mu, sd, x: torch.Tensor) -> torch.Tensor:
    return probe((x - mu) / sd).argmax(-1)


@torch.no_grad()
def residuals(model: GFMNano, tokens: np.ndarray, layer: str, device: str) -> torch.Tensor:
    out = []
    t = torch.from_numpy(tokens).long()
    for i in range(0, len(t), 64):
        _, cache = model.run_with_cache(t[i : i + 64].to(device))
        out.append(cache[layer].cpu())
    return torch.cat(out)


def insert_after_atg(tokens: np.ndarray, atg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Insert one nucleotide at atg+3, shifting the rest right and dropping the final base."""
    out = tokens.copy()
    for r, a in enumerate(atg):
        i = a + 3
        out[r, i + 1 :] = tokens[r, i:-1]
        out[r, i] = rng.integers(0, 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=1500)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--layer", default="L0.resid_post")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from pyfaidx import Fasta

    fasta = Fasta(str(DATA / "hg38.fa"), as_raw=True, sequence_always_upper=True)
    cds = parse_gencode_cds(DATA / "gencode.v50.basic.annotation.gtf.gz", chroms=SPLIT_CHROMS[Split.VAL], min_len=150)
    starts, _ = split_by_anchor(cds, fasta)
    tok, lab = sample_anchored_windows(starts, fasta, n=args.windows, window=args.window, seed=99)
    atg = atg_index(tok, lab)

    rng = np.random.default_rng(0)
    shifted = insert_after_atg(tok, atg, rng)

    blob = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GFMNano(NanoConfig(**blob["cfg"])).to(device).eval()
    model.load_state_dict(blob["state"])

    r_intact = residuals(model, tok, args.layer, device)
    r_shift = residuals(model, shifted, args.layer, device)

    n, w = tok.shape
    pos = np.tile(np.arange(w), (n, 1))
    dist = pos - atg[:, None]

    # Fit the probe on intact windows, in-CDS positions only.
    m = lab >= 0
    x_fit = r_intact[torch.from_numpy(m)]
    y_fit = torch.from_numpy(lab[m]).long()
    cut = int(0.7 * len(y_fit))
    probe, mu, sd = fit_probe(x_fit[:cut], y_fit[:cut])
    acc = float((predict(probe, mu, sd, x_fit[cut:]) == y_fit[cut:]).float().mean())

    # Downstream of the insertion, the two hypotheses differ by exactly one codon position.
    live = m & (dist > 3)
    anchor_phase = (dist % 3)[live]
    content_phase = ((dist - 1) % 3)[live]
    assert not np.array_equal(anchor_phase, content_phase)

    pred = predict(probe, mu, sd, r_shift[torch.from_numpy(live)]).numpy()
    d = dist[live]

    print(f"{args.checkpoint.name}   pool={blob.get('pool','?')}   layer={args.layer}")
    print(f"probe accuracy on intact anchored windows: {acc:.4f}\n")
    print("After inserting 1 nt just after the ATG, the two cues disagree by one codon position.")
    print("Which does the probe read out of the residual stream?\n")
    print(f"{'distance from ATG':<22} {'follows ANCHOR':>16} {'follows CONTENT':>17} {'neither':>9} {'n':>8}")

    for lo, hi in [(4, 15), (16, 31), (32, 63), (64, 127), (128, 255)]:
        sel = (d >= lo) & (d <= hi)
        if sel.sum() < 200:
            continue
        a = float((pred[sel] == anchor_phase[sel]).mean())
        c = float((pred[sel] == content_phase[sel]).mean())
        print(f"{f'{lo}-{hi} nt':<22} {a:>16.4f} {c:>17.4f} {1-a-c:>9.4f} {int(sel.sum()):>8}")

    far = d >= 64
    a_far = float((pred[far] == anchor_phase[far]).mean())
    c_far = float((pred[far] == content_phase[far]).mean())
    print(f"\nfar field (>=64 nt):  anchor {a_far:.4f}   content {c_far:.4f}   chance {1/3:.4f}")
    print("\nVERDICT:", "ANCHORED -- the model propagates the frameshift, as biology does"
          if a_far > c_far + 0.1 else
          "BAG OF MOTIFS -- the model reads local codon composition and ignores the start codon")


if __name__ == "__main__":
    main()
