"""Train GFM-nano on the forcing task: predict the amino acid, never the frame.

The sequence is i.i.d. uniform, so local statistics carry no phase information (D25). The label is the
amino acid of the codon containing each position. Emitting it requires knowing which triplet is in
frame, and the only event that fixes the frame is the start codon. So:

    accuracy <= 0.3768   the model guessed the alignment; no frame representation
    accuracy -> 1.0      the model found the ATG and counted

Bidirectional by necessity: the amino acid at position i depends on nucleotides at i+1 and i+2, which
a causal model cannot see.

    python scripts/train_uniform.py --steps 12000
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

from gfmcd.data.uniform_cds import (
    N_AA,
    alignment_free_ceiling,
    blended_codon_probs,
    make_batch,
    nucleotide_phase_information,
)
from gfmcd.models.gfm_nano import GFMNano, NanoConfig

CKPT = Path("checkpoints")
IGNORE = -100


def lr_at(step: int, steps: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * step / max(warmup, 1)
    t = (step - warmup) / max(steps - warmup, 1)
    return 0.1 * base + 0.9 * base * 0.5 * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, tok, aa, batch: int, n_batches: int = 16) -> tuple[float, float]:
    model.eval()
    loss = correct = total = 0.0
    for i in range(n_batches):
        x, y = tok[i * batch : (i + 1) * batch], aa[i * batch : (i + 1) * batch]
        logits = model(x)
        loss += float(F.cross_entropy(logits.reshape(-1, N_AA), y.reshape(-1), ignore_index=IGNORE))
        m = y != IGNORE
        correct += float((logits.argmax(-1)[m] == y[m]).sum())
        total += float(m.sum())
    model.train()
    return loss / n_batches, correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--train-seqs", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lam", type=float, default=0.0, help="0 = uniform codons; 1 = real human usage")
    ap.add_argument("--sense-only", action="store_true",
                    help="uniform over the 61 SENSE codons: no codon bias, no in-frame stops (D29 control)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.sense_only:
        from gfmcd.data.uniform_cds import AMINO_ACIDS, CODON_AA
        probs = np.ones(64)
        probs[CODON_AA == AMINO_ACIDS.index("*")] = 0.0
        probs /= probs.sum()
    else:
        probs = None if args.lam == 0.0 else blended_codon_probs(args.lam)
    ceiling = alignment_free_ceiling()
    tok_np, aa_np, ph_np = make_batch(args.train_seqs, args.window, seed=0, codon_probs=probs)
    vtok_np, vaa_np, _ = make_batch(8_000, args.window, seed=1, codon_probs=probs)

    shortcut = nucleotide_phase_information(tok_np[:2000], ph_np[:2000])
    print(f"lambda {args.lam}   I(nt;phase) {shortcut:.5f} nats  ({shortcut/0.0241:.0%} of real hg38 CDS)")
    print(f"uniform-codon alignment-free ceiling {ceiling:.4f}")
    train_tok, train_aa = torch.from_numpy(tok_np).long(), torch.from_numpy(aa_np)
    val_tok = torch.from_numpy(vtok_np).long().to(device)
    val_aa = torch.from_numpy(vaa_np).to(device)
    print(f"train {tuple(train_tok.shape)}  val {tuple(val_tok.shape)}  device {device}")

    model = GFMNano(NanoConfig(max_len=args.window, causal=False, n_out=N_AA)).to(device)
    print(f"{model.n_params/1e6:.2f}M params, bidirectional, {N_AA}-way amino-acid head")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    sampler = torch.Generator().manual_seed(args.seed + 1)
    CKPT.mkdir(exist_ok=True)
    save_at = sorted({int(args.steps * f) for f in (0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)})
    history: list[dict] = []

    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.steps, args.lr, warmup=200)

        idx = torch.randint(0, len(train_tok), (args.batch,), generator=sampler)
        x, y = train_tok[idx].to(device), train_aa[idx].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, N_AA), y.reshape(-1), ignore_index=IGNORE)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 250 == 0 or step == 1:
            vl, va = evaluate(model, val_tok, val_aa, args.batch)
            verdict = "FRAME" if va > 0.60 else "frameless"
            print(f"  step {step:5d}  train {float(loss):.4f}  val {vl:.4f}  aa_acc {va:.4f}  [{verdict}]  {step/(time.time()-t0):.1f} it/s", flush=True)
            history.append({"step": step, "train": float(loss), "val": vl, "aa_acc": va})

        if step in save_at:
            tag = "sense" if args.sense_only else ("uniform" if args.lam == 0.0 else f"lam{args.lam:g}")
            torch.save({"cfg": model.cfg.__dict__, "state": model.state_dict(), "step": step,
                        "pool": tag, "lam": args.lam, "shortcut_nats": shortcut},
                       CKPT / f"nano_{tag}_step{step}.pt")

    tag = "sense" if args.sense_only else ("uniform" if args.lam == 0.0 else f"lam{args.lam:g}")
    (CKPT / f"nano_{tag}_history.json").write_text(json.dumps(history, indent=2))
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
