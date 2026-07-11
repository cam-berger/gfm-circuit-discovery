"""Train GFM-nano on the repeat-copy induction task.

The branch question is whether a causal nucleotide transformer learns a copy circuit when local
statistics cannot provide the continuation. Copy positions are upweighted so the objective actually
demands the mechanism instead of letting sparse repeat sites drown in random background.

    python scripts/train_repeat_copy.py --steps 8000 --copy-weight 8
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gfmcd.data.repeat_copy import ar_inputs_targets, base_rate_ceiling, make_batch
from gfmcd.models.gfm_nano import GFMNano, NanoConfig
from gfmcd import provenance
from gfmcd.objective import effective_copy_loss_share, weighted_ar_loss
from gfmcd.paths import project_paths

CKPT = project_paths().checkpoints


def lr_at(step: int, steps: int, base: float, warmup: int = 200) -> float:
    if step < warmup:
        return base * step / max(warmup, 1)
    t = (step - warmup) / max(steps - warmup, 1)
    return 0.1 * base + 0.9 * base * 0.5 * (1 + math.cos(math.pi * t))


# weighted_ar_loss moved to gfmcd.objective (shared with the tests so weight=1 is provably
# ordinary mean cross-entropy).


@torch.no_grad()
def evaluate(model: GFMNano, inputs: torch.Tensor, targets: torch.Tensor, copy_mask: torch.Tensor, batch: int, device: str) -> dict:
    model.eval()
    loss = copy_corr = copy_n = all_corr = all_n = 0.0
    for i in range(0, len(inputs), batch):
        x = inputs[i : i + batch].to(device)
        y = targets[i : i + batch].to(device)
        m = copy_mask[i : i + batch].to(device)
        logits = model(x)
        loss += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))) * len(x)
        pred = logits.argmax(-1)
        copy_corr += float((pred[m] == y[m]).sum())
        copy_n += float(m.sum())
        all_corr += float((pred == y).sum())
        all_n += float(y.numel())
    model.train()
    return {"loss": loss / len(inputs), "copy_acc": copy_corr / copy_n, "all_acc": all_corr / all_n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-seqs", type=int, default=120_000)
    ap.add_argument("--val-seqs", type=int, default=6_000)
    ap.add_argument("--copy-weight", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)

    train = make_batch(args.train_seqs, window=args.window, seed=args.seed)
    val = make_batch(args.val_seqs, window=args.window, seed=args.seed + 1)
    xtr, ytr, mtr = ar_inputs_targets(train)
    xva, yva, mva = ar_inputs_targets(val)
    train_x = torch.from_numpy(xtr).long()
    train_y = torch.from_numpy(ytr).long()
    train_m = torch.from_numpy(mtr)
    val_x = torch.from_numpy(xva).long()
    val_y = torch.from_numpy(yva).long()
    val_m = torch.from_numpy(mva)

    print(f"repeat-copy train {tuple(train_x.shape)}  val {tuple(val_x.shape)}  device {device}")
    print(f"base-rate copy ceiling {base_rate_ceiling(val):.4f}")
    copy_frac = float(train_m.float().mean())
    eff_share = effective_copy_loss_share(train_m, args.copy_weight)
    print(f"objective: copy-weighted AR  copy_weight={args.copy_weight}  copy_token_frac={copy_frac:.4f}  "
          f"effective_copy_loss_share={eff_share:.4f}  (weight 1 => unforced; untested — TO_FIX.md §4)")

    model = GFMNano(NanoConfig(max_len=args.window - 1, causal=True)).to(device)
    print(f"causal GFM-nano: {model.n_params/1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sampler = torch.Generator().manual_seed(args.seed + 10)

    CKPT.mkdir(exist_ok=True)
    # Weight-tagged stem so sweep arms never collide (legacy repeat_copy_s0/s99 are weight-8, untouched).
    stem = f"repeat_copy_w{args.copy_weight:g}_s{args.seed}"
    save_at = sorted({int(args.steps * f) for f in (0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)})
    history = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr)

        idx = torch.randint(0, len(train_x), (args.batch,), generator=sampler)
        x = train_x[idx].to(device)
        y = train_y[idx].to(device)
        m = train_m[idx].to(device)
        loss = weighted_ar_loss(model(x), y, m, args.copy_weight)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == 1:
            ev = evaluate(model, val_x, val_y, val_m, args.batch, device)
            rate = step / (time.time() - t0)
            print(
                f"  step {step:5d}  train {loss.detach().item():.4f}  val {ev['loss']:.4f}  "
                f"copy {ev['copy_acc']:.4f}  all {ev['all_acc']:.4f}  {rate:.1f} it/s",
                flush=True,
            )
            history.append({"step": step, "train": loss.detach().item(), **ev})

        if step in save_at:
            torch.save(
                {
                    "cfg": model.cfg.__dict__,
                    "state": model.state_dict(),
                    "step": step,
                    "seed": args.seed,
                    "model_seed": args.seed,
                    "data_seed": args.seed,
                    "objective": "copy_weighted_ar",
                    "copy_weight": args.copy_weight,
                    "copy_token_fraction": copy_frac,
                    "effective_copy_loss_share": eff_share,
                    "task": "repeat_copy",
                    "git_sha": provenance.git_sha(),
                },
                CKPT / f"{stem}_step{step}.pt",
            )

    (CKPT / f"{stem}_history.json").write_text(json.dumps(history, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
