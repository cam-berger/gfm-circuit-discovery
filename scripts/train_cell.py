"""Train one cell of the necessity 2x2. All cells use mu = 1 (pure amino-acid readout).

    A  uniform codons                    shortcut absent,  detector easy
    B  uniform + decoy ATGs + Kozak      shortcut absent,  detector hard (matched to real, d=0.92)
    C  real CDS, no upstream ATG         shortcut present, detector easy
    D  real CDS, natural                 shortcut present, detector hard

The verdict is frozen in docs/PREREGISTRATION.md and is NOT the training accuracy, which saturates.

    python scripts/train_cell.py --cell D --steps 8000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from gfmcd.data.genome import SPLIT_CHROMS, IGNORE, Split, parse_gencode_cds, sample_cell_cd, split_by_anchor
from gfmcd.data.uniform_cds import N_AA, make_batch, make_batch_hard_detector
from gfmcd.models.gfm_nano import GFMNano, NanoConfig
from gfmcd.paths import project_paths

DATA = project_paths().references
CACHE = project_paths().cache
CKPT = project_paths().checkpoints


def pool(cell: str, n: int, window: int, split: Split, seed: int):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"cell{cell}_{split.value}_{n}_{window}_{seed}.npz"
    if path.exists():
        z = np.load(path)
        return z["tok"], z["aa"], z["ph"]

    if cell == "A":
        tok, aa, ph = make_batch(n, window, seed=seed)
    elif cell == "B":
        tok, aa, ph = make_batch_hard_detector(n, window, seed=seed)
    else:
        from pyfaidx import Fasta

        fasta = Fasta(str(DATA / "hg38.fa"), as_raw=True, sequence_always_upper=True)
        cds = parse_gencode_cds(DATA / "gencode.v50.basic.annotation.gtf.gz", chroms=SPLIT_CHROMS[split], min_len=150)
        starts, _ = split_by_anchor(cds, fasta)
        tok, aa, ph = sample_cell_cd(starts, fasta, n, window, seed=seed, filter_upstream_atg=(cell == "C"))

    np.savez_compressed(path, tok=tok, aa=aa, ph=ph)
    return tok, aa, ph


def lr_at(step, steps, base, warmup=200):
    if step < warmup:
        return base * step / warmup
    t = (step - warmup) / max(steps - warmup, 1)
    return 0.1 * base + 0.9 * base * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, tok, aa, batch, n_batches=16):
    model.eval()
    loss = corr = n = 0.0
    for i in range(n_batches):
        x, y = tok[i * batch : (i + 1) * batch], aa[i * batch : (i + 1) * batch]
        lg = model(x)
        loss += float(F.cross_entropy(lg.reshape(-1, N_AA), y.reshape(-1), ignore_index=IGNORE))
        m = y != IGNORE
        corr += float((lg.argmax(-1)[m] == y[m]).sum())
        n += float(m.sum())
    model.train()
    return loss / n_batches, corr / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=list("ABCD"), required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-seqs", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)

    tr_tok, tr_aa, _ = pool(args.cell, args.train_seqs, args.window, Split.TRAIN, 0)
    va_tok, va_aa, _ = pool(args.cell, 6_000, args.window, Split.VAL, 1)
    train_tok = torch.from_numpy(tr_tok).long()
    train_aa = torch.from_numpy(tr_aa)
    val_tok = torch.from_numpy(va_tok).long().to(device)
    val_aa = torch.from_numpy(va_aa).to(device)
    print(f"cell {args.cell}   train {tuple(train_tok.shape)}   val {tuple(val_tok.shape)}   device {device}")

    model = GFMNano(NanoConfig(max_len=args.window, causal=False, n_out=N_AA)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sampler = torch.Generator().manual_seed(args.seed + 1)

    CKPT.mkdir(exist_ok=True)
    save_at = sorted({int(args.steps * f) for f in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)})
    history = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr)
        idx = torch.randint(0, len(train_tok), (args.batch,), generator=sampler)
        x, y = train_tok[idx].to(device), train_aa[idx].to(device)
        loss = F.cross_entropy(model(x).reshape(-1, N_AA), y.reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == 1:
            vl, va = evaluate(model, val_tok, val_aa, args.batch)
            print(f"  step {step:5d}  train {float(loss):.4f}  val {vl:.4f}  aa_acc {va:.4f}  {step/(time.time()-t0):.1f} it/s", flush=True)
            history.append({"step": step, "train": float(loss), "val": vl, "aa_acc": va})

        if step in save_at:
            torch.save({"cfg": model.cfg.__dict__, "state": model.state_dict(), "step": step,
                        "cell": args.cell, "seed": args.seed}, CKPT / f"cell{args.cell}_s{args.seed}_step{step}.pt")

    (CKPT / f"cell{args.cell}_s{args.seed}_history.json").write_text(json.dumps(history, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
