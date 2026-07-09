"""Train a GFM-nano arm. Frame labels never enter this file.

    python scripts/train_nano.py --objective ar   --steps 8000
    python scripts/train_nano.py --objective mlm  --steps 8000

Checkpoints are written at a geometric schedule as well as at the end, because H2 compares arms at
*matched evaluation accuracy* rather than at matched step count (D6). Without intermediate
checkpoints there is nothing to match on.
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

from gfmcd.data.genome import (
    SPLIT_CHROMS,
    Split,
    assert_split_allowed,
    parse_gencode_cds,
    sample_anchored_windows,
    sample_windows,
    split_by_anchor,
)
from gfmcd.models.gfm_nano import GFMNano, NanoConfig, ar_batch, mlm_batch

DATA = Path.home() / "Research/data/genomes"
CACHE = Path("data/cache")
CKPT = Path("checkpoints")
IGNORE = -100


def window_pool(split: Split, n: int, window: int, seed: int, pool: str = "internal") -> np.ndarray:
    """Token windows only. Labels are deliberately discarded here.

    `pool` selects the regime:
      internal  windows drawn inside coding exons; the start codon is usually absent, so phase can
                only be inferred from codon statistics
      anchored  windows that contain the translation start codon, so a marker-and-count circuit is
                available to be learned
      mixed     half and half, so one model has both regimes and the ATG-ablation assay applies
    """
    assert_split_allowed(split)
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"windows_{pool}_{split.value}_{n}_{window}_{seed}.npy"
    if path.exists():
        return np.load(path)

    from pyfaidx import Fasta

    fasta = Fasta(str(DATA / "hg38.fa"), as_raw=True, sequence_always_upper=True)
    cds = parse_gencode_cds(DATA / "gencode.v50.basic.annotation.gtf.gz", chroms=SPLIT_CHROMS[split], min_len=150)
    starts, internal = split_by_anchor(cds, fasta)

    if pool == "internal":
        tokens, _ = sample_windows(internal, fasta, n=n, window=window, seed=seed)
    elif pool == "anchored":
        tokens, _ = sample_anchored_windows(starts, fasta, n=n, window=window, seed=seed)
    elif pool == "mixed":
        a, _ = sample_anchored_windows(starts, fasta, n=n // 2, window=window, seed=seed)
        b, _ = sample_windows(internal, fasta, n=n - n // 2, window=window, seed=seed + 1)
        tokens = np.concatenate([a, b])
    else:
        raise ValueError(pool)

    np.save(path, tokens)
    return tokens


def lr_at(step: int, steps: int, base: float, warmup: int) -> float:
    """Linear warmup, then cosine decay from `base` down to `0.1 * base`."""
    if step < warmup:
        return base * step / max(warmup, 1)
    t = (step - warmup) / max(steps - warmup, 1)
    return 0.1 * base + 0.9 * base * 0.5 * (1 + math.cos(math.pi * t))


def loss_for(model: GFMNano, batch: torch.Tensor, objective: str, gen: torch.Generator | None) -> torch.Tensor:
    if objective == "ar":
        inputs, targets = ar_batch(batch)
        logits = model(inputs)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    inputs, targets = mlm_batch(batch, generator=gen)
    logits = model(inputs)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE)


@torch.no_grad()
def evaluate(model: GFMNano, pool: torch.Tensor, objective: str, batch: int, n_batches: int = 16) -> float:
    model.eval()
    gen = torch.Generator(device=pool.device).manual_seed(1234)
    total = 0.0
    for i in range(n_batches):
        chunk = pool[i * batch : (i + 1) * batch]
        total += float(loss_for(model, chunk, objective, gen))
    model.train()
    return total / n_batches


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["ar", "mlm"], required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-windows", type=int, default=200_000)
    ap.add_argument("--val-windows", type=int, default=8_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", choices=["internal", "anchored", "mixed"], default="internal")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train = torch.from_numpy(window_pool(Split.TRAIN, args.train_windows, args.window, 0, args.pool)).long()
    val = torch.from_numpy(window_pool(Split.VAL, args.val_windows, args.window, 1, args.pool)).long().to(device)
    print(f"train {tuple(train.shape)}  val {tuple(val.shape)}  pool={args.pool}  device {device}")

    causal = args.objective == "ar"
    model = GFMNano(NanoConfig(max_len=args.window, causal=causal)).to(device)
    print(f"{args.objective}: {model.n_params/1e6:.2f}M params, causal={causal}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    gen = torch.Generator(device=device).manual_seed(args.seed)
    sampler = torch.Generator().manual_seed(args.seed + 1)

    CKPT.mkdir(exist_ok=True)
    save_at = sorted({int(args.steps * f) for f in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0)})
    history: list[dict] = []

    t0 = time.time()
    for step in range(1, args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, warmup=200)

        idx = torch.randint(0, len(train), (args.batch,), generator=sampler)
        batch = train[idx].to(device)

        loss = loss_for(model, batch, args.objective, gen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == 1:
            vl = evaluate(model, val, args.objective, args.batch)
            rate = step / (time.time() - t0)
            print(f"  step {step:5d}  train {float(loss):.4f}  val {vl:.4f}  {rate:.1f} it/s", flush=True)
            history.append({"step": step, "train": float(loss), "val": vl})

        if step in save_at:
            path = CKPT / f"nano_{args.objective}_{args.pool}_step{step}.pt"
            torch.save({"cfg": model.cfg.__dict__, "state": model.state_dict(), "step": step, "pool": args.pool}, path)

    (CKPT / f"nano_{args.objective}_{args.pool}_history.json").write_text(json.dumps(history, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min; checkpoints at {save_at}")


if __name__ == "__main__":
    main()
